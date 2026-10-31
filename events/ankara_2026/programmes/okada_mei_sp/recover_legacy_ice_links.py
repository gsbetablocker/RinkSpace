"""Recover only skipped adjacent RGB searches, retaining cached source nodes."""
from pathlib import Path
import argparse
from collections import Counter
import gzip
import hashlib
import json
import time

import cv2
import numpy as np

from rinkspace.session.material_tracks import link_fragments
from capture_independent_ice import (capture_frames, ice_source_mask,
    load_navigation_graph, navigation_between)
from capture_independent_materials import source_exclusion
from link_legacy_ice_sources import (actual_candidate_links, digest, plain,
    read, ref, search_only_adjacent_navigation, write)
from rinkspace.path_policy import guard_project_write


def rows(path):
    with gzip.open(path,'rt',encoding='utf8') as stream:
        yield from (json.loads(line) for line in stream)


def write_rows(path,items):
    data=''.join(json.dumps(plain(r),ensure_ascii=False,separators=(',',':'))+'\n' for r in items)
    path.write_bytes(gzip.compress(data.encode('utf8'),compresslevel=1,mtime=0))


def recover(args):
    started=time.perf_counter();cv2.setNumThreads(2)
    output=guard_project_write(args.output)
    if output.exists() and any(output.iterdir()):raise FileExistsError(output)
    output.mkdir(parents=True,exist_ok=True)
    old=read(args.links/'coverage.json');capture=Path(old['capture'])
    index=read(args.root/'index.json');source_sha=index['source']['actual_sha256']
    if digest(index['source']['resolved_path'])!=source_sha:raise ValueError('Source video changed')
    if digest(capture/'binding.json')!=old['candidate_id_space_capture_binding_sha256']:
        raise ValueError('Original candidate capture changed')
    for key in ('frames_index','pairs_index'):
        if digest(old[key]['path'])!=old[key]['sha256']:raise ValueError('Original unique index changed')
    frame_rows=list(rows(old['frames_index']['path']));pair_rows=list(rows(old['pairs_index']['path']))
    frames={(r['shot'],r['pts']):r for r in frame_rows}
    pair_keys=[(r['shot'],r['previous_pts'],r['current_pts']) for r in pair_rows]
    if len(frames)!=3866 or len(pair_keys)!=3863 or len(set(pair_keys))!=3863:
        raise ValueError('Expected unique complete S07-S09 original source indexes')
    implementations={str(Path(p).resolve()):digest(p) for p in
        {__file__,link_fragments.__code__.co_filename,capture_frames.__code__.co_filename,
         source_exclusion.__code__.co_filename,search_only_adjacent_navigation.__code__.co_filename}}
    replacements={};receipts=[];totals=Counter();timings=Counter()
    for source in index['shots']:
        sid=source['shot_id']
        if sid not in ('s07','s08','s09'):continue
        graph,_=load_navigation_graph(args.root,source);wanted={};unavailable=[]
        for row in pair_rows:
            if row['shot']!=sid:continue
            record=read(row['metadata_path'])
            if record['source_navigation']['H_current_from_previous'] is not None:continue
            if record['links']:raise ValueError('Skipped search unexpectedly contains original links')
            a,b=row['previous_pts'],row['current_pts']
            if navigation_between(graph,a,b)[0] is not None:raise ValueError('Original skip is no longer unsupported')
            h,status=search_only_adjacent_navigation(graph,a,b)
            if h is None:unavailable.append(dict(previous_pts=a,current_pts=b,status=status));continue
            wanted[a,b]=(row,record,h,status)
        selected=sorted({p for pair in wanted for p in pair});folder=output/sid;folder.mkdir()
        previous=None;shot_counts=Counter();t0=time.perf_counter()
        for _,(pts,bgr) in capture_frames(index,source,selected) if selected else ():
            phase=time.perf_counter();item=frames[sid,pts];meta=read(item['metadata_path'])
            record=read(meta['source_frame']['path']);rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
            if (hashlib.sha256(bgr.tobytes()).hexdigest()!=record['source_BGR_sha256'] or
                hashlib.sha256(rgb.tobytes()).hexdigest()!=record['source_RGB_sha256'] or
                record['source_video_sha256']!=source_sha):raise ValueError('Exact original exposure changed')
            for key in ('nodes','source_candidates','source_frame'):
                if digest(meta[key]['path'])!=meta[key]['sha256']:raise ValueError('Cached source input changed')
            nodes=dict(np.load(meta['nodes']['path'],allow_pickle=False))
            with np.load(meta['source_candidates']['path'],allow_pickle=False) as raw:
                if not np.array_equal(nodes['uv'],raw['candidate_uv'][nodes['ids']]):raise ValueError('Cached original candidate UV differs')
            mask,mask_path=source_exclusion(args.root,sid,pts)
            if digest(mask_path)!=record['mask_sha256']:raise ValueError('Fine source mask changed')
            mask,floor=ice_source_mask(rgb,mask)
            if (hashlib.sha256(mask.tobytes()).hexdigest()!=record['source_exclusion_sha256'] or
                floor!=record['ice_floor_guard']):raise ValueError('Exact original exclusion changed')
            timings['source_read_hash_and_mask_s']+=time.perf_counter()-phase
            current=dict(pts=pts,rgb=rgb,nodes=nodes,mask=mask,meta=meta)
            if previous is not None and (previous['pts'],pts) in wanted:
                a=previous['pts'];row,original,h,status=wanted[a,pts];phase=time.perf_counter()
                edges=link_fragments(previous['rgb'],rgb,previous['nodes'],nodes,previous['mask'],mask,h)
                links=actual_candidate_links(edges,previous['nodes'],nodes);elapsed=time.perf_counter()-phase
                timing=dict(actual_bidirectional_link_s=elapsed);timings.update(timing)
                pair=dict(original,links=links,links_count=len(links),
                    unambiguous_links=sum(bool(e['unambiguous'])for e in links),
                    ambiguous_links=sum(not e['unambiguous']for e in links),timing=timing,
                    source_navigation=dict(status=status,H_current_from_previous=h,
                        used_only_for_search=True,background_navigation_supported=False,
                        background_navigation_status=original['source_navigation']['status'],
                        unsupported_cumulative_navigation_used=False),
                    search_recovery=dict(original_skipped_pair=ref(row['metadata_path']),
                        original_nodes_reused=True,original_candidate_ids_and_UV_unchanged=True,
                        actual_RGB_and_original_exclusion_hashes_verified=True,
                        original_successful_pairs_recomputed=False,fit_eligible=False))
                path=folder/f'{a}_{pts}.links.json.gz';write(path,pair)
                replacement=dict(row,metadata_path=str(path.resolve()),metadata_sha256=digest(path),
                    links=len(links),ambiguous_links=pair['ambiguous_links'],
                    original_candidate_pairs=sum(len(e['pairs'])for e in links),
                    search_recovery=True,fit_eligible=False)
                replacements[sid,a,pts]=replacement;shot_counts['recomputed_pairs']+=1
                shot_counts['pairs_with_links']+=bool(links);shot_counts['links']+=len(links)
                shot_counts['ambiguous_links']+=pair['ambiguous_links'];shot_counts['original_candidate_pairs']+=replacement['original_candidate_pairs']
            previous=current;shot_counts['decoded_exposures']+=1
        if shot_counts['recomputed_pairs']!=len(wanted):raise ValueError('Missing recovered pair')
        totals.update(shot_counts);receipt=dict(shot=sid,counts=dict(shot_counts),
            unrecoverable_navigation_pairs=unavailable,elapsed_seconds=time.perf_counter()-t0)
        receipts.append(receipt);print(json.dumps(receipt),flush=True)
    combined=[replacements.get((r['shot'],r['previous_pts'],r['current_pts']),r)for r in pair_rows]
    if {(r['shot'],r['previous_pts'],r['current_pts'])for r in combined}!=set(pair_keys):raise ValueError('Combined pair scope changed')
    pair_index=output/'pairs_index.jsonl.gz';write_rows(pair_index,combined)
    # Preserve the original frame index by reference: cached nodes are not copied.
    shots={sid:dict(record)for sid,record in old['shots'].items()}
    for receipt in receipts:
        sid=receipt['shot'];c=receipt['counts']
        for key in ('links','ambiguous_links','original_candidate_pairs'):shots[sid][key]+=c.get(key,0)
        shots[sid]['search_recovered_pairs']=c.get('recomputed_pairs',0)
    binding=dict(schema='rinkspace.adjacent_search_recovery.v1',original_coverage=ref(args.links/'coverage.json'),
        source_video_sha256=source_sha,source_index=ref(args.root/'index.json'),
        implementations=implementations,only_original_H_None_pairs_recomputed=True,
        original_successful_pairs_recomputed=False,cached_nodes_and_readouts_reused=True,
        unsupported_cumulative_navigation_used=False,RGB_thresholds_and_algorithm_unchanged=True,
        identities_assigned=False,fit_eligible=False,new_fit_started=False)
    for path,sha in implementations.items():
        if digest(path)!=sha:raise ValueError('Implementation changed during recovery')
    coverage=dict(old,shots=shots,pairs_index=ref(pair_index),
        links=old['links']+totals['links'],ambiguous_links=old['ambiguous_links']+totals['ambiguous_links'],
        original_candidate_pairs=old['original_candidate_pairs']+totals['original_candidate_pairs'],
        search_recovery=binding,search_recovery_counts=dict(totals),search_recovery_shots=receipts,
        elapsed_seconds=time.perf_counter()-started,complete=True)
    write(output/'binding.json',binding);write(output/'coverage.json',coverage)
    print(json.dumps(dict(output=str(output/'coverage.json'),counts=dict(totals),elapsed_seconds=coverage['elapsed_seconds'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('root','links','output'):p.add_argument('--'+key,type=Path,required=True)
    recover(p.parse_args())
