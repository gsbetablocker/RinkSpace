"""Stream actual adjacent-image links for an existing bulk ice capture.

Ranges are native [first_pts, stop_pts), with 256-PTS exposure steps. Give each
parallel range its own output directory and overlap by one source exposure;
shot/PTS frame keys and endpoint pair keys can then be reconciled explicitly.
Curve names only locate the original capture inventory, never physical identity.
"""
from pathlib import Path
import argparse
from collections import Counter
import gzip
import hashlib
import json
import time

import cv2
import numpy as np

from rinkspace.path_policy import guard_project_write
from rinkspace.session.material_tracks import source_nodes, same_frame_conflicts, link_fragments
from capture_independent_materials import source_exclusion
from capture_independent_ice import (ice_source_mask, load_navigation_graph,
    navigation_between, timed_capture_frames)


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def read(path):
    data=Path(path).read_bytes()
    return json.loads(gzip.decompress(data) if str(path).endswith('.gz') else data)


def plain(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {str(k):plain(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [plain(v) for v in value]
    return value


def write(path,value):
    data=(json.dumps(plain(value),ensure_ascii=False,separators=(',',':'))+'\n').encode('utf8')
    path=guard_project_write(path)
    path.write_bytes(gzip.compress(data,compresslevel=1,mtime=0) if str(path).endswith('.gz') else data)


def ref(path):return dict(path=str(Path(path).resolve()),sha256=digest(path))


def search_only_adjacent_navigation(graph, previous_pts, current_pts):
    """Navigate one exposure without promoting background-map qualification.

    Unsupported background transfer can still initialize the finite RGB search.
    Only the original adjacent step is available here: no unsupported cumulative
    map, material correspondence, or physical identity is created by this helper.
    """
    if graph is None:return None,'source_graph_unavailable'
    before=graph['index'].get(int(previous_pts));after=graph['index'].get(int(current_pts))
    if (current_pts-previous_pts!=256 or before is None or after!=before+1):
        return None,'not_adjacent_registered_native_exposures'
    h,reason=navigation_between(graph,previous_pts,current_pts)
    if h is not None:return h,reason
    raw=np.asarray(graph['adjacent_h'][before],float)
    if raw.shape!=(3,3) or not np.isfinite(raw).all():return None,'nonfinite_adjacent_navigation'
    try:inverse=np.linalg.inv(raw)
    except np.linalg.LinAlgError:return None,'singular_adjacent_navigation'
    if not np.isfinite(inverse).all():return None,'nonfinite_inverse_adjacent_navigation'
    return raw.copy(),'finite_raw_adjacent_H_search_only_background_support_unqualified'


def actual_candidate_links(edges,old,now):
    """Keep public link decisions, converting node-index pairs to source IDs."""
    result=[]
    for edge in edges:
        pairs=np.asarray(edge['pairs'],np.int64).reshape(-1,2)
        if (np.any(pairs<0) or np.any(pairs[:,0]>=len(old['ids']))
                or np.any(pairs[:,1]>=len(now['ids']))):
            raise ValueError('Public source link pair is outside its original node arrays')
        if (not np.all(old['fragment'][pairs[:,0]]==edge['a'])
                or not np.all(now['fragment'][pairs[:,1]]==edge['b'])):
            raise ValueError('Public link fragments differ from original node membership')
        actual=np.c_[old['ids'][pairs[:,0]],now['ids'][pairs[:,1]]]
        result.append(dict(edge,node_pairs=pairs.tolist(),pairs=actual.tolist(),
            pairs_index_space='previous/current_original_candidate_npz_candidate_id',
            node_pairs_index_space='previous/current_nodes_npz_row',
            fragment_index_space='previous/current_original_capture_fragments',
            fit_eligible=False,physical_identity_confirmed=False))
    return result


def link(args):
    if args.output.resolve()==args.capture.resolve():raise ValueError('Output must differ from source capture')
    index=read(args.root/'index.json');capture_binding=read(args.capture/'binding.json')
    source_sha=index['source']['actual_sha256']
    if capture_binding['source_video_sha256']!=source_sha:raise ValueError('Capture source video differs')
    if digest(index['source']['resolved_path'])!=source_sha:raise ValueError('Original source video changed')
    source=next(s for s in index['shots'] if s['shot_id']==args.shot)
    scope=source['processing'];first,stop=args.first_pts,args.stop_pts
    if (not scope['first_pts']<=first<stop<=scope['stop_pts_exclusive']
            or (first-scope['first_pts'])%256 or (stop-first)%256):
        raise ValueError('Range must be contiguous actual native PTS inside one registered shot')
    graph,graph_note=load_navigation_graph(args.root,source)
    inventory=capture_binding['native_inventory']
    args.output=guard_project_write(args.output);args.output.mkdir(parents=True,exist_ok=True)
    folder=args.output/args.shot;folder.mkdir(exist_ok=True)
    implementations={f.__code__.co_filename for f in (source_nodes,same_frame_conflicts,link_fragments,
        source_exclusion,ice_source_mask,load_navigation_graph,navigation_between,
        search_only_adjacent_navigation,timed_capture_frames)}
    implementations.add(__file__)
    binding=dict(schema='rinkspace.bulk_ice_actual_adjacent_links.v1',shot=args.shot,
        first_pts=first,stop_pts_exclusive=stop,native_step_pts=256,time_base='1/12800',
        source_video_sha256=source_sha,source_index=ref(args.root/'index.json'),
        source_capture_binding=ref(args.capture/'binding.json'),source_navigation=graph_note,
        implementations={str(Path(p).resolve()):digest(p) for p in sorted(implementations)},
        inventory_search_labels=inventory,curve_names_are_physical_identities=False,
        source_nodes_ids_are_original_capture_candidate_ids=True,
        links_pairs_are_original_candidate_ids=True,ambiguities_and_same_frame_conflicts_retained=True,
        thresholds_and_public_link_algorithm_unchanged=True,fit_eligible=False,
        physical_identity_confirmed=False,prepared_modified=False,new_fit_started=False)
    binding_path=args.output/'binding.json'
    if binding_path.exists() and read(binding_path)!=plain(binding):raise ValueError('Output binding differs')
    if not binding_path.exists():write(binding_path,binding)
    cv2.setNumThreads(2);previous=None;expected=first;counts=Counter();timings=Counter();start=time.perf_counter()
    partial=dict(source,processing=dict(scope,first_pts=first,last_pts=stop-256,stop_pts_exclusive=stop))
    for _,(pts,image),decode_s in timed_capture_frames(index,partial,None):
        if pts!=expected:raise ValueError('Decoded original source range has a missing or reordered exposure')
        expected+=256;phase=time.perf_counter();timing=dict(decode_s=decode_s)
        source_json=args.capture/args.shot/f'{pts}.json.gz';record=read(source_json)
        source_npz=args.capture/args.shot/f'{pts}.npz'
        if (record['pts']!=pts or record['source_video_sha256']!=source_sha
                or digest(source_npz)!=record['raw_candidates']['sha256']):
            raise ValueError('Original capture frame or candidate array identity changed')
        rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        if (hashlib.sha256(image.tobytes()).hexdigest()!=record['source_BGR_sha256']
                or hashlib.sha256(rgb.tobytes()).hexdigest()!=record['source_RGB_sha256']):
            raise ValueError('Decoded source RGB differs from exact candidate exposure')
        measured=dict(np.load(source_npz,allow_pickle=False));fragments=record['fragments']
        timing['source_input_and_hash_s']=time.perf_counter()-phase;phase=time.perf_counter()
        mask,mask_path=source_exclusion(args.root,args.shot,pts)
        if digest(mask_path)!=record['mask_sha256']:raise ValueError('Original fine source mask changed')
        mask,floor=ice_source_mask(rgb,mask);mask_sha=hashlib.sha256(mask.tobytes()).hexdigest()
        if mask_sha!=record['source_exclusion_sha256'] or floor!=record['ice_floor_guard']:
            raise ValueError('Actual combined source mask differs from capture')
        timing['source_mask_s']=time.perf_counter()-phase;phase=time.perf_counter()
        nodes=source_nodes(rgb,measured,fragments,inventory)
        ids=nodes['ids'];expected_ids=np.unique([i for f in fragments for i in f['candidate_ids']]).astype(int)
        if (not np.array_equal(ids,expected_ids) or np.any(ids<0)
                or np.any(ids>=len(measured['candidate_uv']))
                or not np.array_equal(nodes['uv'],measured['candidate_uv'][ids])
                or not np.array_equal(nodes['sides'],measured['candidate_sides'][ids])):
            raise ValueError('Source nodes differ from original candidate IDs or UV')
        conflicts=same_frame_conflicts(nodes,fragments)
        for conflict in conflicts:
            conflict['index_space']='original_capture_fragments'
            conflict['a_candidate_ids']=fragments[conflict['a']]['candidate_ids']
            conflict['b_candidate_ids']=fragments[conflict['b']]['candidate_ids']
        timing['source_nodes_and_conflicts_s']=time.perf_counter()-phase;phase=time.perf_counter()
        node_path=folder/f'{pts}.nodes.npz';np.savez_compressed(node_path,**nodes)
        frame=dict(schema='rinkspace.bulk_ice_actual_source_nodes.v1',shot=args.shot,pts=pts,
            source_seconds=pts/12800,time_base='1/12800',source_video_sha256=source_sha,
            source_BGR_sha256=record['source_BGR_sha256'],source_RGB_sha256=record['source_RGB_sha256'],
            source_frame=ref(source_json),source_candidates=ref(source_npz),source_mask=ref(mask_path),
            source_exclusion_sha256=mask_sha,nodes=ref(node_path),nodes_count=len(ids),
            node_index_space='nodes.ids maps each node row to original source NPZ candidate ID',
            original_candidate_count=len(measured['candidate_uv']),original_fragment_count=len(fragments),
            non_node_candidate_count=len(measured['candidate_uv'])-len(ids),
            non_node_candidates_remain_in_original_source=True,same_frame_conflicts=conflicts,
            curve_names_are_physical_identities=False,fit_eligible=False,physical_identity_confirmed=False)
        timing['nodes_write_s']=time.perf_counter()-phase;pair_summary=None
        if previous is not None:
            if pts-previous['pts']!=256:raise ValueError('Only actually adjacent source exposures may link')
            phase=time.perf_counter();qualified_h,qualified_reason=navigation_between(graph,previous['pts'],pts)
            h,reason=search_only_adjacent_navigation(graph,previous['pts'],pts)
            edges=[] if h is None else link_fragments(previous['rgb'],rgb,previous['nodes'],nodes,
                previous['mask'],mask,h)
            links=actual_candidate_links(edges,previous['nodes'],nodes)
            timing['actual_bidirectional_link_s']=time.perf_counter()-phase;phase=time.perf_counter()
            pair_path=folder/f'{previous["pts"]}_{pts}.links.json.gz'
            pair=dict(schema='rinkspace.bulk_ice_actual_adjacent_candidate_links.v1',shot=args.shot,
                previous_pts=previous['pts'],current_pts=pts,time_base='1/12800',source_video_sha256=source_sha,
                previous_frame=previous['frame_ref'],current_frame_path=str((folder/f'{pts}.json.gz').resolve()),
                previous_source_RGB_sha256=previous['rgb_sha'],current_source_RGB_sha256=record['source_RGB_sha256'],
                previous_nodes=previous['nodes_ref'],current_nodes=frame['nodes'],
                source_navigation=dict(status=reason,H_current_from_previous=h,used_only_for_search=True,
                    background_navigation_supported=qualified_h is not None,
                    background_navigation_status=qualified_reason,
                    unsupported_cumulative_navigation_used=False),
                links=links,links_count=len(links),unambiguous_links=sum(bool(e['unambiguous'])for e in links),
                ambiguous_links=sum(not e['unambiguous']for e in links),
                link_decision='unchanged public RGB bidirectional NCC, measured width/normal and fragment competition',
                curve_names_are_physical_identities=False,fit_eligible=False,physical_identity_confirmed=False,
                timing=dict(actual_bidirectional_link_s=timing['actual_bidirectional_link_s']))
            write(pair_path,pair);timing['links_write_s']=time.perf_counter()-phase
            pair_summary={k:pair[k] for k in ('previous_pts','current_pts','links_count','unambiguous_links','ambiguous_links')}
            counts['pairs']+=1;counts['links']+=len(links);counts['ambiguous_links']+=pair['ambiguous_links']
        frame['timing']=timing;frame_path=folder/f'{pts}.json.gz';write(frame_path,frame)
        previous=dict(pts=pts,rgb=rgb,mask=mask,nodes=nodes,frame_ref=ref(frame_path),
            nodes_ref=frame['nodes'],rgb_sha=record['source_RGB_sha256'])
        counts['frames']+=1;counts['nodes']+=len(ids);counts['same_frame_conflicts']+=len(conflicts)
        timings.update(timing)
        print(json.dumps(dict(shot=args.shot,pts=pts,nodes=len(ids),conflicts=len(conflicts),
            pair=pair_summary,timing=timing),ensure_ascii=False),flush=True)
    if expected!=stop:raise ValueError('Original source range ended before its registered stop')
    for path,sha in binding['implementations'].items():
        if digest(path)!=sha:raise ValueError('Public implementation changed during source linking')
    summary=dict(schema='rinkspace.bulk_ice_actual_link_range.v1',shot=args.shot,first_pts=first,
        stop_pts_exclusive=stop,expected_frames=(stop-first)//256,counts=dict(counts),
        total_stage_seconds=dict(timings),elapsed_seconds=time.perf_counter()-start,
        full_requested_range_processed=True,source_hashes_and_exact_node_correspondence_verified=True,
        ambiguities_and_same_frame_conflicts_retained=True,fit_eligible=False,
        physical_identity_confirmed=False,prepared_modified=False,new_fit_started=False)
    write(args.output/f'{args.shot}_{first}_{stop}.summary.json.gz',summary)
    print(json.dumps(plain(summary),ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('root','capture','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--shot',choices=['s07','s08','s09'],required=True)
    p.add_argument('--first-pts',type=int,required=True);p.add_argument('--stop-pts',type=int,required=True)
    link(p.parse_args())
