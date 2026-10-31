"""Export graph-associated original readouts as unadmitted review source_rows.

No poses, packet construction, new source measurement or fit is performed.
Source UV, sides and quantiles remain exact original candidate-array values.
"""
from pathlib import Path
from collections import Counter, defaultdict
import argparse
import gzip
import json
import time
import numpy as np

from rinkspace.path_policy import guard_project_write
from propagate_ice_readout_identities import read,digest,write,intervals,checked_anchor_document
from readout_exclusions import ReadoutExclusions, retained_runs


def curve_map(inventory):return {p['curve_id']:p for p in inventory}


def spaced_actual_keys(keys,pool,spacing=48.):
    """Select original samples nearest uniform measured-chain arc stations."""
    ordered=[]
    for key in keys:
        if not ordered or key!=ordered[-1]:ordered.append(key)
    if len(ordered)>1 and ordered[0]==ordered[-1]:ordered.pop()
    if not ordered:return []
    uv=np.array([pool[k]['uv']for k in ordered],float)
    arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(uv,axis=0),axis=1))]
    targets=np.arange(0.,arc[-1]+1e-9,spacing)
    chosen=np.unique(np.argmin(abs(arc[:,None]-targets[None]),axis=0))
    return [ordered[int(i)]for i in chosen]


def load_contexts(lc,sid,candidate_identity_root=None,family_identity_root=None):
    candidate_root=Path(candidate_identity_root)if candidate_identity_root is not None else lc/'ice_source_identity'
    family_root=Path(family_identity_root)if family_identity_root is not None else lc/'source_shape_family_identity'
    candidate=read(candidate_root/'full_direct'/f'{sid}.json.gz')
    candidate_arrays_path=Path(candidate['arrays']['path'])
    if digest(candidate_arrays_path)!=candidate['arrays']['sha256']:raise ValueError('Candidate identity arrays changed')
    ca=dict(np.load(candidate_arrays_path,allow_pickle=False));candidate_metadata=candidate_root/'full_direct'/f'{sid}.json.gz'
    candidate_proof=dict(metadata=dict(path=str(candidate_metadata),sha256=digest(candidate_metadata)),arrays=candidate['arrays'])
    contexts={};families={}
    for capture in ('full_direct','luminance_full'):
        graph_root=family_root/capture;gb=read(graph_root/'binding.json');cap=Path(gb['capture_path'])
        binding=read(cap/'binding.json')
        if digest(cap/'binding.json')!=gb['candidate_id_space_capture_binding_sha256']:raise ValueError('Capture candidate ID space changed')
        meta=graph_root/f'{sid}.json.gz';g=read(meta);arrays=dict(np.load(g['arrays']['path'],allow_pickle=False))
        if digest(g['arrays']['path'])!=g['arrays']['sha256']:raise ValueError('Family graph arrays changed')
        by_pts=defaultdict(list)
        for node in g['nodes']:
            ni=node['node'];li=int(arrays['physical_family_index'][ni])
            if li<0 or node['physical_family'] is None or node['identity_status']=='identity_conflict':continue
            if str(arrays['physical_family_values'][li])!=node['physical_family']:raise ValueError('Family graph node/label differs')
            by_pts[node['pts']].append(node)
        contexts[capture]=dict(path=cap,binding=binding,inventory=binding['native_inventory'],by_curve=curve_map(binding['native_inventory']))
        families[capture]=dict(nodes=by_pts,proof=dict(metadata=dict(path=str(meta),sha256=digest(meta)),arrays=g['arrays']),all_nodes=g['nodes'])
    contexts['full_direct']['candidate_frames']={int(f['pts']):i for i,f in enumerate(candidate['frames_index'])}
    return contexts,families,ca,candidate,candidate_proof


def run(args):
    lc=args.legacy;output=guard_project_write(args.output)
    exclusions=ReadoutExclusions(args.readout_exclusions)if getattr(args,'readout_exclusions',None) is not None else None
    if output.exists()and list(output.glob('s??/source_rows.jsonl')):raise FileExistsError(output)
    output.mkdir(parents=True,exist_ok=True);index=read(args.root/'index.json');mapping=read(args.prepared/'mapping.json')
    native_path_ids=set(mapping['path_ids']);prepared_path=args.prepared/'point_observations.npz'
    with np.load(prepared_path,allow_pickle=False)as z:current_pts=z['pts'];current_paths=z['path_indices'];current_uv=z['uv']
    negative=defaultdict(set);negative_bindings={};inputs={};anchor_paths=[lc/'reviewed_source_anchors.json',lc/'reviewed_c0_source_anchors.json']
    for path in anchor_paths:
        document=checked_anchor_document(path);inputs[str(path)]=digest(path)
        for anchor in document['anchors']:
            if not anchor['entity'].startswith('rejected:'):continue
            key=(anchor['capture'],anchor['shot'],anchor['pts']);negative[key].update(anchor['candidate_ids']);negative_bindings[key]=(anchor['measurement_sha256'],anchor['source_BGR_sha256'])
    all_stats={};validation=Counter();group_plan=[];replacement_spans=defaultdict(list);started=time.monotonic()
    for sid in ('s07','s08','s09'):
        shot=next(s for s in index['shots']if s['shot_id']==sid);scope=shot['processing'];pts_list=list(range(scope['first_pts'],scope['stop_pts_exclusive'],256))
        contexts,families,ca,cg,candidate_proof=load_contexts(lc,sid,args.candidate_identity_root,args.family_identity_root)
        if exclusions is not None:
            for capture,context in contexts.items():exclusions.bind_capture(capture,context['path'])
        target_map=contexts['full_direct']['by_curve'];by_path={p['native_path_id']:p for p in contexts['full_direct']['inventory']}
        if not set(by_path)<=native_path_ids:raise ValueError('Current native path mapping is incomplete')
        if ca['pts'].tolist()!=pts_list:raise ValueError('Candidate graph source denominator differs')
        current=defaultdict(set);current_frames=defaultdict(set)
        for i in np.flatnonzero((current_pts>=scope['first_pts'])&(current_pts<=scope['last_pts'])):
            path_id=mapping['path_ids'][int(current_paths[i])]
            if path_id not in by_path:continue
            pt=int(current_pts[i]);current[pt,path_id].add(tuple(map(float,current_uv[i])));current_frames[path_id].add(pt)
        path_stats={p['native_path_id']:dict(label=p['curve_id'],raw_branch_candidate_occurrences=0,raw_unique_readouts=0,raw_unique_path_uv=0,
            thinned_branch_occurrences=0,output_uv=0,exact_new_to_current_uv=0,frames=set(),new_frames=set(),shape_frames=set(),material_pending_rows=0,
            current_input_frames=len(current_frames[p['native_path_id']]),
            current_input_unique_uv=sum(len(current[pt,p['native_path_id']])for pt in current_frames[p['native_path_id']]))for p in by_path.values()}
        stats=Counter();folder=output/sid;folder.mkdir(exist_ok=True);rowfile=folder/'source_rows.jsonl'
        cached_validation={}
        with rowfile.open('w',encoding='utf8',buffering=1)as destination:
            for fi,pts in enumerate(pts_list):
                base=read(contexts['full_direct']['path']/sid/f'{pts}.json.gz');pool={};branches=[];raw_cache={};meta_cache={};deny={};guard_denied={}
                # Candidate-level rejected/conflict states are capture-local.
                start,stop=map(int,ca['offsets'][fi:fi+2]);labels=ca['label_index'][start:stop]
                rejected=np.zeros(len(labels),bool);labeled=labels>=0
                rejected[labeled]=ca['label_is_rejected'][labels[labeled]]
                deny['full_direct']=set(map(int,np.flatnonzero(rejected|(labels==-2))))|negative['full_direct',sid,pts]
                deny['luminance_full']=set(negative['luminance_full',sid,pts])
                for capture in contexts:
                    for node in families[capture]['all_nodes']:
                        if node['pts']==pts and node['identity_status']=='identity_conflict':deny[capture].update(node['candidate_ids'])
                def source(capture):
                    if capture not in raw_cache:
                        context=contexts[capture];meta=read(context['path']/sid/f'{pts}.json.gz');raw=context['path']/sid/f'{pts}.npz';sha=digest(raw)
                        if sha!=meta['raw_candidates']['sha256']or meta['source_BGR_sha256']!=base['source_BGR_sha256']:raise ValueError('Original source measurement binding differs')
                        check=negative_bindings.get((capture,sid,pts))
                        if check and check!=(sha,meta['source_BGR_sha256']):raise ValueError('Negative anchor binding differs')
                        with np.load(raw,allow_pickle=False)as arrays:raw_cache[capture]=dict(arrays)
                        meta_cache[capture]=(meta,dict(path=str(raw),sha256=sha))
                        blocked=(exclusions.for_frame(capture,sid,pts,raw,sha,meta['source_BGR_sha256'],meta['source_RGB_sha256'],len(raw_cache[capture]['candidate_uv']))
                            if exclusions is not None else frozenset())
                        guard_denied[capture]=blocked;deny[capture].update(blocked)
                    return raw_cache[capture],meta_cache[capture][0],meta_cache[capture][1]
                def add_branch(capture,target,ids,branch,evidence,pending=False,relative_role_check=False):
                    z,meta,measurement=source(capture);ids=list(map(int,ids));context=contexts[capture]
                    if any(not 0<=cid<len(z['candidate_uv'])for cid in ids):raise ValueError('Candidate ID outside immutable source')
                    removed=set(ids)&deny[capture]
                    stats['guard_readout_occurrences_excluded']+=len(set(ids)&guard_denied[capture])
                    stats['negative_or_conflict_readouts_excluded']+=len(removed-guard_denied[capture])
                    for ri,run in enumerate(retained_runs(ids,deny[capture])):
                        keys=[];actual_branch=f'{branch}:retained_run:{ri}'if removed else branch
                        for cid in run:
                            qi=int(z['candidate_query'][cid]);original=context['inventory'][int(z['query_curve'][qi])]
                            if original['operator']!=target['operator']:raise ValueError('Source/target material measurement operator differs')
                            if relative_role_check and original['world_curve_role']!=target['world_curve_role']:raise ValueError('Relative material role differs from current native target')
                            if target['operator']=='material_interface'and original['material_transition']!=target['material_transition']:raise ValueError('Source material transition differs')
                            uv=z['candidate_uv'][cid];key=(target['native_path_id'],float(uv[0]),float(uv[1]));keys.append(key)
                            item=pool.setdefault(key,dict(path=target,uv=uv.tolist(),occurrences={}))
                            occurrence=item['occurrences'].get((capture,cid))
                            if occurrence is None:
                                occurrence=dict(capture=capture,candidate_id=cid,raw_measurement=measurement,
                                    source_BGR_sha256=meta['source_BGR_sha256'],actual_source_sides=z['candidate_sides'][cid].tolist(),
                                    actual_source_quantiles=z['candidate_side_quantiles_uv'][cid].tolist(),source_query_index=qi,
                                    actual_normal=z['query_normal'][qi].tolist(),source_flags=int(z['candidate_flags'][cid]),
                                    source_query_curve_navigation_only=original['curve_id'],source_query_role_navigation_only=original['world_curve_role'],
                                    operator=original['operator'],source_material_signal=original['material_signal'],source_material_transition=original['material_transition'],
                                    source_branch_ids=[],graph_evidence=[],material_pending=False)
                                item['occurrences'][capture,cid]=occurrence
                            if actual_branch not in occurrence['source_branch_ids']:occurrence['source_branch_ids'].append(actual_branch)
                            if evidence not in occurrence['graph_evidence']:occurrence['graph_evidence'].append(evidence)
                            occurrence['material_pending']|=bool(pending)
                            path_stats[target['native_path_id']]['raw_branch_candidate_occurrences']+=1
                        if keys:branches.append(dict(path_id=target['native_path_id'],keys=keys,branch=actual_branch))
                # Only positive candidate identity + saved paired straight-line pieces.
                for li in np.unique(labels[labeled&~rejected]):
                    label=str(ca['label_values'][li]);target=target_map.get(label)
                    if target is None:raise ValueError('Positive identity has no exact current native path: '+label)
                    if target['operator']!='paint_center'or target.get('closed') is not False:continue
                    selected=set(map(int,np.flatnonzero(labels==li)));stats['positive_line_graph_readouts_before_fragment_intersection']+=len(selected)
                    z,meta,measurement=source('full_direct')
                    for gi,fragment in enumerate(meta['line_fragments']):
                        pieces=fragment.get('source_visible_candidate_pieces')or[fragment['candidate_ids']]
                        guarded_pieces=[(pi,di,run)for pi,piece in enumerate(pieces)
                            for di,run in enumerate(retained_runs(piece,deny['full_direct']))]
                        for pi,di,piece in guarded_pieces:
                            ids=[int(cid)for cid in piece if int(cid)in selected]
                            # Keep source discontinuities after graph intersection.
                            groups=[]
                            for cid in ids:
                                if not groups or int(z['candidate_query'][cid])-int(z['candidate_query'][groups[-1][-1]])!=1:groups.append([])
                                groups[-1].append(cid)
                            for ri,group in enumerate(groups):
                                if not group:continue
                                if len({int(ca['component'][start+cid])for cid in group})!=1:raise ValueError('Line branch graph component differs')
                                evidence=dict(kind='candidate_identity_component',**candidate_proof,
                                    component=int(ca['component'][start+group[0]]),positive_identity=label,
                                    capture_line_fragment=gi,source_piece=pi,guard_retained_piece=di,graph_intersection_run=ri)
                                add_branch('full_direct',target,group,f'line:{gi}:piece:{pi}:guard_piece:{di}:run:{ri}',evidence)
                # Closed materials use only reviewed physical family labels + exact roles.
                for capture in contexts:
                    for node in families[capture]['nodes'].get(pts,[]):
                        z,meta,measurement=source(capture)
                        if node['measurement']['sha256']!=measurement['sha256']:raise ValueError('Family node measurement changed')
                        role_ids=defaultdict(list)
                        for cid,role in zip(node['candidate_ids'],node['relative_roles']):role_ids[role].append(cid)
                        for role,ids in role_ids.items():
                            label=f'{node["physical_family"]}:{role}';target=target_map.get(label)
                            if target is None or not target['closed']:raise ValueError('Family role lacks exact current native closed path: '+label)
                            chosen=set(map(int,ids));assigned=set();groups=[]
                            for gi,fragment in enumerate(meta['fragments']):
                                for di,piece in enumerate(retained_runs(fragment['candidate_ids'],deny[capture])):
                                    local=[int(cid)for cid in piece if int(cid)in chosen]
                                    if local:groups.append((f'fragment:{gi}:guard_piece:{di}',local));assigned.update(local)
                            remaining=sorted(chosen-assigned,key=lambda cid:int(z['candidate_query'][cid]));runs=[]
                            for cid in remaining:
                                if not runs or int(z['candidate_query'][cid])-int(z['candidate_query'][runs[-1][-1]])!=1:runs.append([])
                                runs[-1].append(cid)
                            groups.extend((f'shape_query_run:{ri}',group)for ri,group in enumerate(runs))
                            evidence=dict(kind='source_shape_family_component',**families[capture]['proof'],node=node['node'],
                                component=node['component'],physical_family=node['physical_family'],relative_role=role,
                                source_shape=node['shape_source'],material_pending=bool(node['material_pending']))
                            for branch,group in groups:add_branch(capture,target,group,f'shape_node:{node["node"]}:role:{role}:{branch}',evidence,node['material_pending'],True)
                chosen_keys=set()
                for branch in branches:
                    selected=spaced_actual_keys(branch['keys'],pool)
                    path_stats[branch['path_id']]['thinned_branch_occurrences']+=len(selected);chosen_keys.update(selected)
                    target=by_path[branch['path_id']]
                    if selected and target['operator']=='paint_center'and target.get('closed')is False:
                        actual_span=[pool[selected[0]]['uv'],pool[selected[-1]]['uv']]
                        group_key=(sid,pts,branch['path_id'])
                        if actual_span not in replacement_spans[group_key]:replacement_spans[group_key].append(actual_span)
                exported=[]
                for key,item in pool.items():
                    p=path_stats[key[0]];p['raw_unique_path_uv']+=1;p['raw_unique_readouts']+=len(item['occurrences'])
                for key in sorted(chosen_keys):
                    item=pool[key];target=item['path'];provenance=[item['occurrences'][k]for k in sorted(item['occurrences'])];primary=provenance[0]
                    pending=any(p['material_pending']for p in provenance)
                    row=dict(path_id=target['native_path_id'],entity=target['curve_id'],relative_role=target['world_curve_role'],uv=item['uv'],
                        capture=primary['capture'],candidate_id=primary['candidate_id'],measurement_sha256=primary['raw_measurement']['sha256'],
                        source_BGR_sha256=primary['source_BGR_sha256'],operator=primary['operator'],
                        measurement_operator='material_interface'if target['operator']=='material_interface'else'paired_material_boundary_midline',
                        actual_source_sides=primary['actual_source_sides'],actual_source_quantiles=primary['actual_source_quantiles'],
                        graph_evidence=primary['graph_evidence'],provenance=provenance,primary_provenance_index=0,
                        material_pending=pending,fit_eligible=False,review_state='material_pending'if pending else'candidate_pending_main_review')
                    exported.append(row);p=path_stats[key[0]];p['output_uv']+=1;p['frames'].add(pts);p['material_pending_rows']+=pending
                    if any(e['kind']=='source_shape_family_component'for v in provenance for e in v['graph_evidence']):p['shape_frames'].add(pts)
                    if tuple(item['uv'])not in current.get((pts,key[0]),set()):p['exact_new_to_current_uv']+=1
                    if pts not in current_frames[key[0]]:p['new_frames'].add(pts)
                # Independent checks use original arrays and graph labels after serialization-compatible mapping.
                for row in exported:
                    target=by_path[row['path_id']]
                    if target['curve_id']!=row['entity']or target['world_curve_role']!=row['relative_role']:raise ValueError('Incorrect target role/path mapping')
                    for provenance in row['provenance']:
                        capture=provenance['capture'];cid=provenance['candidate_id'];z,meta,measurement=source(capture)
                        if cid in deny[capture]or not np.array_equal(np.asarray(row['uv']),z['candidate_uv'][cid]):raise ValueError('Export UV changed or denied ID leaked')
                        if not np.array_equal(np.asarray(provenance['actual_source_sides']),z['candidate_sides'][cid])or not np.array_equal(np.asarray(provenance['actual_source_quantiles']),z['candidate_side_quantiles_uv'][cid]):raise ValueError('Source sides or quantiles changed')
                        validation['UV_sides_quantiles_and_denied_ID_verified_provenances']+=1
                    validation['rows_with_exact_native_path_and_role']+=1
                frame=dict(pts=pts,source_seconds=pts/12800,time_base='1/12800',pixel_domain='native_distorted',
                    source_video_sha256=index['source']['actual_sha256'],source_BGR_sha256=base['source_BGR_sha256'],
                    rows=exported,fit_eligible=False,review_state='candidate_only_for_main_review',source_UV_unchanged=True)
                destination.write(json.dumps(frame,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n');stats['frames']+=1;stats['output_rows']+=len(exported);stats['frames_with_rows']+=bool(exported)
                for path_id,count in Counter(row['path_id']for row in exported).items():
                    straight=by_path[path_id]['operator']=='paint_center'and by_path[path_id].get('closed')is False
                    group_plan.append(dict(shot=sid,pts=pts,path_id=path_id,label=by_path[path_id]['curve_id'],
                        proposed_operation='replace_existing_group'if pts in current_frames[path_id]else'add_new_group',
                        replacement_policy='partial_line_spans'if straight else'whole_group',
                        incoming_candidate_rows=count,current_unique_uv=len(current.get((pts,path_id),set())),
                        activation_performed=False,fit_eligible=False))
                for(s,pt,path_id),spans in replacement_spans.items():
                    if s!=sid or pt!=pts:continue
                    actual={tuple(row['uv'])for row in exported if row['path_id']==path_id}
                    if any(tuple(endpoint)not in actual for span in spans for endpoint in span):raise ValueError('Partial line span endpoint is not an exported actual UV')
                if(fi+1)%500==0:print(json.dumps(dict(shot=sid,frames=fi+1,rows=stats['output_rows'],seconds=time.monotonic()-started)),flush=True)
        per_path={}
        for path_id,p in path_stats.items():
            p['output_frames']=len(p['frames']);p['new_frames_vs_current_input']=len(p['new_frames']);p['output_native_PTS_intervals']=intervals(p.pop('frames'));p['new_native_PTS_intervals']=intervals(p.pop('new_frames'))
            shape_intervals=intervals(p.pop('shape_frames'));p['shape_propagation_intervals']=shape_intervals
            longest=max(shape_intervals,key=lambda interval:interval[2])if shape_intervals else None
            p['longest_shape_interval']=dict(first_pts=longest[0],last_pts=longest[1],frames=longest[2],first_seconds=longest[0]/12800,last_seconds=longest[1]/12800)if longest else None
            p['replacement_groups_vs_current_input']=p['output_frames']-p['new_frames_vs_current_input'];per_path[path_id]=p
        for name in ('raw_branch_candidate_occurrences','raw_unique_readouts','raw_unique_path_uv','thinned_branch_occurrences','material_pending_rows','exact_new_to_current_uv'):
            stats[name]=sum(p[name]for p in per_path.values())
        report=dict(shot=sid,expected_native_frames=scope['native_frame_count'],**dict(stats),paths=per_path,
            source_rows=dict(path=str(rowfile),sha256=digest(rowfile)),fit_eligible=False)
        write(folder/'coverage.json',report);all_stats[sid]=report
        write(folder/'source_replacement.json',dict(policy='partial_line_spans',groups=[dict(pts=pts,path_id=path_id,spans=spans)
            for(s,pts,path_id),spans in sorted(replacement_spans.items())if s==sid]))
        print(json.dumps(dict(stage='shot_export_complete',shot=sid,frames=stats['frames'],rows=stats['output_rows'],raw_unique_readouts=sum(p['raw_unique_readouts']for p in per_path.values()))),flush=True)
    report=dict(schema='rinkspace.unadmitted_graph_source_rows_export.v1',shots=all_stats,validation=dict(validation),
        denominator_native_frames=sum(s['expected_native_frames']for s in all_stats.values()),
        spacing_px=48,spacing_method='nearest original actual candidate to uniform measured-branch arc stations; no interpolation or minimum sample count',
        exact_dedup_key=['shot','native PTS','native_path_id','unchanged actual UV'],all_duplicate_source_provenance_retained=True,
        closed_seam_policy='exact same-path actual UV duplicates merge with all provenance; neighbouring different UV or paths are not merged',
        inputs=dict(negative_anchor_files=inputs,current_prepared_points=dict(path=str(prepared_path),sha256=digest(prepared_path)),
            current_mapping=dict(path=str(args.prepared/'mapping.json'),sha256=digest(args.prepared/'mapping.json')),
            candidate_identity_root=str(args.candidate_identity_root or lc/'ice_source_identity'),
            family_identity_root=str(args.family_identity_root or lc/'source_shape_family_identity')),
        readout_exclusions=exclusions.receipt()if exclusions is not None else None,
        entry_sha256=digest(__file__),original_capture_arrays_modified=False,prepared_modified=False,poses_created=False,
        source_identity_and_pixel_accuracy_accepted=False,fit_eligible=False,new_fit_started=False,elapsed_seconds=time.monotonic()-started)
    if report['denominator_native_frames']!=3866 or sum(s['frames']for s in all_stats.values())!=3866:raise ValueError('Export source denominator differs')
    write(output/'metadata.json',report)
    write(output/'source_replacement.json',dict(policy='partial_line_spans',groups=[dict(pts=pts,path_id=path_id,spans=spans)
        for(s,pts,path_id),spans in sorted(replacement_spans.items())]))
    with(output/'group_update_plan.jsonl').open('w',encoding='utf8')as stream:
        for group in group_plan:stream.write(json.dumps(group,ensure_ascii=False,separators=(',',':'))+'\n')
    overview=dict(schema='rinkspace.registration_candidate_batch_overview.v1',denominator_native_frames=3866,
        groups=dict(add_new=sum(g['proposed_operation']=='add_new_group'for g in group_plan),
                    replace_existing=sum(g['proposed_operation']=='replace_existing_group'for g in group_plan)),
        shots={sid:dict(frames=s['frames'],frames_with_rows=s['frames_with_rows'],raw_unique_readouts=s['raw_unique_readouts'],
            raw_unique_path_uv=s['raw_unique_path_uv'],selected_rows=s['output_rows'],material_pending_rows=s['material_pending_rows'],
            paths={pid:{k:p[k]for k in ('label','output_uv','output_frames','material_pending_rows','new_frames_vs_current_input',
                'replacement_groups_vs_current_input','exact_new_to_current_uv','longest_shape_interval')}for pid,p in s['paths'].items()if p['output_uv']})for sid,s in all_stats.items()},
        fit_eligible=False,activation_performed=False,prepared_modified=False)
    write(output/'overview.json',overview)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('legacy','root','prepared','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--candidate-identity-root',type=Path,help='Root containing capture-separated candidate identity graphs')
    p.add_argument('--family-identity-root',type=Path,help='Root containing capture-separated source-shape family graphs')
    p.add_argument('--readout-exclusions',type=Path,help='Complete all-original-candidate guard coverage JSON')
    run(p.parse_args())


if __name__=='__main__':main()
