"""Connect known-shape source families using unchanged actual RGB pairs only.

Navigation family names never provide physical identity. Reviewed positive
anchors name families only through selected candidate intersections and exact
relative roles. Explicit negative readouts cannot support an edge; their direct
nodes remain material-pending while the component retains the evidence.
"""
from pathlib import Path
from collections import Counter, defaultdict
import argparse
import gzip
import json
import time
import numpy as np

from rinkspace.path_policy import guard_project_write
from rinkspace.session.material_tracks import qualify_fragment_links
from propagate_ice_readout_identities import digest, read, rows, write, intervals, checked_anchor_document


ROLE_VALUES=('inner','middle','outer','center','centreline')


def load_frames(entries,capture,inventory,anchors):
    role_for_curve=np.array([ROLE_VALUES.index(p['world_curve_role']) if p['world_curve_role'] in ROLE_VALUES else -1 for p in inventory],np.int8)
    frames=[];nodes=[];anchor_receipts=[]
    for item in sorted(entries,key=lambda e:int(e['pts'])):
        if digest(item['result_path'])!=item['result_sha256']:raise ValueError('Unique shape result changed')
        shape=read(item['result_path']);pts=int(item['pts']);shot=item['shot'];raw=item['original_candidates']
        source=read(capture/shot/f'{pts}.json.gz')
        sha=digest(raw['path'])
        if (sha!=raw['sha256'] or sha!=shape['measurement_sha256'] or sha!=source['raw_candidates']['sha256'] or
                shape['source_BGR_sha256']!=source['source_BGR_sha256'] or item['source_BGR_sha256']!=source['source_BGR_sha256']):
            raise ValueError('Shape/capture source binding changed')
        if shape['pts']!=pts or shape['shot']!=shot or shape.get('source_UV_unchanged') is not True or shape.get('fit_eligible'):
            raise ValueError('Invalid original shape frame')
        with np.load(raw['path'],allow_pickle=False)as z:
            uv=z['candidate_uv'];candidate_curve=z['query_curve'][z['candidate_query']]
        count=len(uv);membership=np.full(count,-1,np.int32);roles=role_for_curve[candidate_curve]
        frame=dict(pts=pts,shot=shot,first_node=len(nodes),nodes=[],uv=uv,membership=membership,roles=roles,
            negative=np.zeros(count,bool),source_RGB_sha256=source['source_RGB_sha256'],
            source_BGR_sha256=source['source_BGR_sha256'],measurement=raw,shape=dict(path=item['result_path'],sha256=item['result_sha256']))
        for family,result in sorted(shape['shape_families'].items()):
            ids=np.asarray(result['consensus_candidate_ids'],np.int64)
            if not len(ids):continue
            if result.get('identity_from_query_name') is not False or len(np.unique(ids))!=len(ids) or (ids<0).any() or (ids>=count).any():
                raise ValueError('Invalid family consensus candidate IDs')
            if (membership[ids]>=0).any() or (roles[ids]<0).any() or any(inventory[int(ci)]['feature_id']!=family for ci in candidate_curve[ids]):
                raise ValueError('Family consensus leaves its original inventory/role')
            local=len(frame['nodes']);global_id=len(nodes);membership[ids]=local
            node=dict(node=global_id,frame=len(frames),frame_node_index=local,pts=pts,navigation_family=family,
                candidate_ids=ids.tolist(),relative_roles=[ROLE_VALUES[int(r)]for r in roles[ids]],
                positive_anchor_evidence=[],negative_anchor_evidence=[],negative_anchor_candidate_ids=[],
                shape_source=frame['shape'],measurement=raw,identity_from_navigation_name=False,fit_eligible=False)
            nodes.append(node);frame['nodes'].append(global_id)
        for ai,anchor in anchors:
            if anchor['shot']!=shot or anchor['pts']!=pts:continue
            if anchor['measurement_sha256']!=sha or anchor['source_BGR_sha256']!=source['source_BGR_sha256']:
                raise ValueError('Reviewed anchor no longer binds this source measurement')
            if any(cid>=count for cid in anchor['candidate_ids']):raise ValueError('Anchor candidate does not exist')
            rejected=anchor['entity'].startswith('rejected:')
            if rejected:frame['negative'][anchor['candidate_ids']]=True
            by_node=defaultdict(list);outside=[];mismatch=[]
            entity=anchor['entity'][len('rejected:'):] if rejected else anchor['entity']
            physical,separator,relative_role=entity.rpartition(':')
            if not rejected and (not separator or not physical):raise ValueError('Positive family anchor requires explicit family:relative_role')
            for cid in anchor['candidate_ids']:
                local=int(membership[cid])
                if local<0:outside.append(cid);continue
                if not rejected and relative_role!=ROLE_VALUES[int(roles[cid])]:mismatch.append(cid);continue
                by_node[frame['nodes'][local]].append(cid)
            for ni,ids in by_node.items():
                evidence=dict(anchor_record_index=ai,entity=anchor['entity'],candidate_ids=ids,
                    source_BGR_sha256=anchor['source_BGR_sha256'],measurement_sha256=anchor['measurement_sha256'])
                if rejected:
                    nodes[ni]['negative_anchor_evidence'].append(evidence)
                    nodes[ni]['negative_anchor_candidate_ids'].extend(ids)
                else:
                    evidence['physical_family']=physical;evidence['relative_role']=relative_role
                    nodes[ni]['positive_anchor_evidence'].append(evidence)
            anchor_receipts.append(dict(anchor_record_index=ai,entity=anchor['entity'],pts=pts,
                selected_node_intersections={str(k):v for k,v in by_node.items()},
                not_selected_by_shape_candidate_ids=outside,relative_role_mismatch_candidate_ids=mismatch,
                negative_readouts_excluded_from_family_pairs=rejected))
        frames.append(frame)
    return frames,nodes,anchor_receipts


def aggregate_pair(item,previous,current,output):
    path=item['metadata_path'];record=read(path)
    if (record['previous_pts']!=previous['pts'] or record['current_pts']!=current['pts'] or
            record['previous_source_RGB_sha256']!=previous['source_RGB_sha256'] or record['current_source_RGB_sha256']!=current['source_RGB_sha256']):
        raise ValueError('Actual RGB pair exposure binding changed')
    aggregate=defaultdict(lambda:defaultdict(list));rejected=[];stats=Counter();all_original=set()
    for ei,edge in enumerate(record['links']):
        if edge.get('source_only') is not True or edge.get('identity_assumed_from_curve_name') is not False or edge.get('pairs_index_space')!='previous/current_original_candidate_npz_candidate_id':
            raise ValueError('Expected unchanged actual bidirectional RGB original-candidate pairs')
        for pi,(a,b)in enumerate(edge['pairs']):
            if type(a)is not int or type(b)is not int or not 0<=a<len(previous['uv']) or not 0<=b<len(current['uv']):raise ValueError('Invalid actual candidate pair')
            pair=(a,b);all_original.add(pair);stats['original_pair_occurrences']+=1
            if previous['negative'][a] or current['negative'][b]:
                rejected.append(dict(pair=[a,b],original_fragment_edge=ei,original_pair_index=pi,
                    reason='explicit_reviewed_negative_material_readout',previous_negative=bool(previous['negative'][a]),current_negative=bool(current['negative'][b])))
                continue
            left=int(previous['membership'][a]);right=int(current['membership'][b])
            if left<0 or right<0:stats['outside_selected_shape_pair_occurrences']+=1;continue
            if previous['roles'][a]!=current['roles'][b]:stats['relative_role_mismatch_pair_occurrences']+=1;continue
            aggregate[left,right][pair].append([ei,pi]);stats['shape_and_role_supported_pair_occurrences']+=1
    eligible=[];insufficient=[]
    for(a,b),proofs in sorted(aggregate.items()):
        pairs=sorted(proofs);source=np.unique([p[0]for p in pairs]);target=np.unique([p[1]for p in pairs])
        source_span=float(np.linalg.norm(np.ptp(previous['uv'][source].astype(float),axis=0)))
        target_span=float(np.linalg.norm(np.ptp(current['uv'][target].astype(float),axis=0)))
        edge=dict(a=a,b=b,pairs=[list(p)for p in pairs],support=len(target),previous_support=len(source),
            actual_previous_UV_span_px=source_span,actual_current_UV_span_px=target_span,
            pair_provenance=[proofs[p]for p in pairs],pairs_index_space='previous/current_original_candidate_npz_candidate_id',
            family_index_space='previous/current_frame_node_index',source_only=True,identity_assumed_from_curve_name=False,
            includes_original_fragment_ambiguous_alternatives=any(record['links'][ei].get('unambiguous') is not True for refs in proofs.values()for ei,pi in refs))
        if min(len(source),len(target))<3 or min(source_span,target_span)<56.:
            edge['reason']='insufficient_actual_unique_readouts_or_actual_UV_span';insufficient.append(edge);continue
        eligible.append(edge)
    qualify_fragment_links(eligible)
    for edge in eligible:
        if not {tuple(p)for p in edge['pairs']}<=all_original:raise ValueError('Family edge invented an actual readout pair')
    result=dict(schema='rinkspace.actual_source_shape_family_edges.v1',shot=item['shot'],previous_pts=previous['pts'],current_pts=current['pts'],
        original_links=dict(path=path,sha256=digest(path)),links=eligible,insufficient_support_edges=insufficient,
        rejected_negative_material_pairs=rejected,counts=dict(stats),all_family_pairs_are_original_actual_subset=True,
        fragment_ambiguity_requalified_after_family_aggregation=True,predicted_pairs_created=False,fit_eligible=False)
    write(output,result)
    return eligible,dict(path=str(output),sha256=digest(output),previous_pts=previous['pts'],current_pts=current['pts'],
        family_edges=len(eligible),unambiguous_family_edges=sum(e['unambiguous']for e in eligible),
        ambiguous_family_edges=sum(not e['unambiguous']for e in eligible),negative_pair_occurrences_filtered=len(rejected),
        insufficient_support_family_edges=len(insufficient),actual_unique_pairs=sum(len(e['pairs'])for e in eligible),
        includes_fragment_ambiguous_edges=sum(e['includes_original_fragment_ambiguous_alternatives']for e in eligible))


def run_shot(shot,entries,pairs,capture,inventory,anchors,output):
    started=time.monotonic();frames,nodes,anchor_receipts=load_frames(entries,capture,inventory,anchors)
    by_pts={f['pts']:f for f in frames};parent=np.arange(len(nodes),dtype=np.int64);rank=np.zeros(len(nodes),np.int8)
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=int(parent[i])
        return i
    def join(a,b):
        a=find(a);b=find(b)
        if a==b:return
        if rank[a]<rank[b]:a,b=b,a
        parent[b]=a
        if rank[a]==rank[b]:rank[a]+=1
    pair_folder=output/'pairs'/shot;pair_folder.mkdir(parents=True,exist_ok=True);pair_receipts=[]
    for item in sorted(pairs,key=lambda p:int(p['previous_pts'])):
        a=by_pts.get(item['previous_pts']);b=by_pts.get(item['current_pts'])
        if a is None or b is None:continue
        if b['pts']-a['pts']!=256:raise ValueError('Family graph only consumes actual adjacent exposures')
        edges,receipt=aggregate_pair(item,a,b,pair_folder/f'{a["pts"]}_{b["pts"]}.json.gz');pair_receipts.append(receipt)
        for edge in edges:
            if edge['unambiguous']:join(a['nodes'][edge['a']],b['nodes'][edge['b']])
    component=np.array([find(i)for i in range(len(nodes))],np.int64);components={}
    for node,ci in zip(nodes,component):
        group=components.setdefault(int(ci),dict(component=int(ci),nodes=[],positive_anchor_evidence=[],negative_anchor_evidence=[]))
        group['nodes'].append(node['node']);group['positive_anchor_evidence'].extend(node['positive_anchor_evidence']);group['negative_anchor_evidence'].extend(node['negative_anchor_evidence'])
    values=sorted({e['physical_family']for c in components.values()for e in c['positive_anchor_evidence']});lookup={v:i for i,v in enumerate(values)}
    identity=np.full(len(nodes),-1,np.int32);pending=np.zeros(len(nodes),bool)
    for group in components.values():
        names=sorted({e['physical_family']for e in group['positive_anchor_evidence']});group['physical_families']=names
        group['identity_status']='unanchored'if not names else'identity_conflict'if len(names)>1 else'anchored_family_candidate'
        for ni in group['nodes']:
            node=nodes[ni];node['component']=group['component'];node['physical_family']=names[0]if len(names)==1 else None
            node['physical_family_candidates']=names;node['identity_status']=group['identity_status']
            node['material_pending']=bool(node['negative_anchor_evidence']);pending[ni]=node['material_pending']
            node['physical_identity_and_pixel_accuracy_accepted']=False
            identity[ni]=-2 if len(names)>1 else lookup[names[0]]if names else -1
    candidate_offsets=np.r_[0,np.cumsum([len(n['candidate_ids'])for n in nodes])]
    target=output/f'{shot}.npz';np.savez_compressed(target,pts=np.array([f['pts']for f in frames],np.int64),
        frame_node_offsets=np.r_[0,np.cumsum([len(f['nodes'])for f in frames])],component=component,
        physical_family_index=identity,physical_family_values=np.asarray(values,dtype=str),material_pending=pending,
        candidate_offsets=candidate_offsets,candidate_ids=np.array([i for n in nodes for i in n['candidate_ids']],np.int64),
        relative_role_index=np.array([ROLE_VALUES.index(role)for n in nodes for role in n['relative_roles']],np.int8),relative_role_values=np.asarray(ROLE_VALUES))
    coverage=[];allpts=set(by_pts)
    for name,li in lookup.items():
        chosen=[n for n in nodes if identity[n['node']]==li];seen={n['pts']for n in chosen};roles=Counter()
        for n in chosen:roles.update(n['relative_roles'])
        coverage.append(dict(physical_family=name,nodes=len(chosen),frames=len(seen),actual_selected_candidates=sum(len(n['candidate_ids'])for n in chosen),
            relative_role_candidate_counts=dict(roles),direct_material_pending_nodes=sum(n['material_pending']for n in chosen),
            associated_native_PTS_intervals=intervals(seen),unassociated_frames=len(allpts-seen),unassociated_native_PTS_intervals=intervals(allpts-seen),fit_eligible=False))
    totals=Counter()
    for item in pair_receipts:
        for key in ('family_edges','unambiguous_family_edges','ambiguous_family_edges','negative_pair_occurrences_filtered','actual_unique_pairs','includes_fragment_ambiguous_edges'):totals[key]+=item[key]
    report=dict(schema='rinkspace.source_shape_family_identity_graph.v1',shot=shot,frames=len(frames),nodes=nodes,components=list(components.values()),
        source_frames=[dict(pts=f['pts'],node_ids=f['nodes'],shape=f['shape'],measurement=f['measurement'],source_BGR_sha256=f['source_BGR_sha256'])for f in frames],
        anchor_receipts=anchor_receipts,pair_results=pair_receipts,coverage=coverage,edge_counts=dict(totals),
        arrays=dict(path=str(target),sha256=digest(target)),curve_names_used_as_identities=False,source_UV_unchanged=True,
        negative_readouts_excluded_from_edges=True,negative_pending_scope='directly intersected nodes only; other component nodes retain positive identity propagation',
        fit_eligible=False,new_fit_started=False,elapsed_seconds=time.monotonic()-started)
    write(output/f'{shot}.json.gz',report)
    summary={k:report[k]for k in ('shot','frames','coverage','edge_counts','arrays','elapsed_seconds')}
    summary.update(nodes=len(nodes),components=len(components),anchored_nodes=int((identity>=0).sum()),conflicted_nodes=int((identity==-2).sum()),
        unknown_nodes=int((identity==-1).sum()),direct_material_pending_nodes=int(pending.sum()),
        conflict_components=sum(c['identity_status']=='identity_conflict'for c in components.values()),fit_eligible=False)
    print(json.dumps({k:v for k,v in summary.items()if k!='coverage'}),flush=True)
    return summary


def run(args):
    shape=read(args.shape_coverage);links=read(args.links_coverage);capture=Path(shape['capture']);capture_name=capture.name
    if not shape.get('complete')or not shape.get('source_UV_unchanged')or not links.get('complete'):raise ValueError('Complete unique shape/source link indexes required')
    binding=read(capture/'binding.json');sha=digest(capture/'binding.json')
    if Path(links['capture']).resolve()!=capture.resolve()or sha!=shape['capture_binding_sha256']or sha!=links['candidate_id_space_capture_binding_sha256']:
        raise ValueError('Shape and RGB links use different original candidate spaces')
    shape_index=shape['unique_results_index'];pairs_index=links['pairs_index']
    if digest(shape_index['path'])!=shape_index['sha256']or digest(pairs_index['path'])!=pairs_index['sha256']:raise ValueError('Unique source indexes changed')
    entries=list(rows(shape_index['path']));pairs=list(rows(pairs_index['path']))
    if len({(e['shot'],int(e['pts']))for e in entries})!=len(entries):raise ValueError('Duplicate shape exposure in unique index')
    document=checked_anchor_document(args.anchors);anchors=[(i,a)for i,a in enumerate(document['anchors'])if a['capture']in(capture_name,str(capture))]
    output=guard_project_write(args.output)
    if output.exists()and any(output.glob('*.npz')):raise FileExistsError(output)
    output.mkdir(parents=True,exist_ok=True);summaries=[]
    for shot in args.shots:
        chosen=[e for e in entries if e['shot']==shot and(args.pts is None or e['pts']in args.pts)]
        if not chosen:continue
        summaries.append(run_shot(shot,chosen,[p for p in pairs if p['shot']==shot],capture,binding['native_inventory'],anchors,output))
    identity=dict(schema='rinkspace.source_shape_family_identity_run.v1',capture=capture_name,capture_path=str(capture),
        candidate_id_space_capture_binding_sha256=sha,entry_sha256=digest(__file__),qualify_fragment_links_sha256=digest(qualify_fragment_links.__code__.co_filename),
        shape_coverage=dict(path=str(args.shape_coverage),sha256=digest(args.shape_coverage)),shape_index=shape_index,
        original_links_coverage=dict(path=str(args.links_coverage),sha256=digest(args.links_coverage)),pairs_index=pairs_index,
        anchors=dict(path=str(args.anchors),sha256=digest(args.anchors)),shots=args.shots,pts=args.pts,
        all_pairs_original_actual_subset=True,curve_names_used_as_identities=False,source_UV_unchanged=True,
        negative_pending_scope='direct node only',fit_eligible=False,new_fit_started=False)
    write(output/'binding.json',identity);write(output/'coverage.json',dict(binding=identity,shots=summaries,complete=True,fit_eligible=False))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('shape-coverage','links-coverage','anchors','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--shots',nargs='+',choices=['s07','s08','s09'],default=['s07','s08','s09']);p.add_argument('--pts',nargs='+',type=int)
    run(p.parse_args())


if __name__=='__main__':main()
