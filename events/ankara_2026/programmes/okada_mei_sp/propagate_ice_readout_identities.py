"""Propagate reviewed identities over existing original-RGB candidate links.

Anchor JSON schema rinkspace.reviewed_source_anchors.v1 contains an anchors list.
Each record binds capture, shot, integer pts, entity, original candidate_ids,
source_BGR_sha256 and measurement_sha256. Empty anchors are valid. Query names
never assign identity. Outputs are candidate-only and cannot admit a fit row.
"""
from pathlib import Path
from collections import Counter, defaultdict
import argparse
import gzip
import hashlib
import json
import time

import numpy as np

from rinkspace.path_policy import guard_project_write
from rinkspace.session.material_tracks import propagate_readout_identities, qualify_fragment_links
from readout_exclusions import ReadoutExclusions, retained_runs


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def read(path):
    path=Path(path);data=path.read_bytes()
    return json.loads(gzip.decompress(data) if path.suffix=='.gz' else data)


def rows(path):
    with gzip.open(path,'rt',encoding='utf8') as stream:
        for line in stream:
            if line.strip():yield json.loads(line)


def write(path,value):
    path=guard_project_write(path)
    data=(json.dumps(value,ensure_ascii=False,indent=None if path.suffix=='.gz' else 2)+'\n').encode('utf8')
    path.write_bytes(gzip.compress(data,compresslevel=1,mtime=0) if path.suffix=='.gz' else data)


def intervals(values):
    values=sorted(set(map(int,values)))
    if not values:return []
    result=[];first=previous=values[0];count=1
    for pts in values[1:]:
        if pts==previous+256:count+=1
        else:result.append([first,previous,count]);first=pts;count=1
        previous=pts
    return result+[[first,previous,count]]


def verified_line_groups(record,inventory,frame,counts,receipt,excluded=()):
    """Reuse only saved paired whole-line qualifications with exact readout IDs."""
    reports=[]
    for name in ('whole_line_reports','coherent_source_interval_reports'):
        for ri,report in enumerate(record['line_qualification'].get(name,[])):
            reports.append((f'line_qualification/{name}/{ri}',report))
    seen=set()
    for fragment_index,fragment in enumerate(record.get('line_fragments',[])):
        receipt['saved_line_fragments']+=1
        curve=int(fragment['curve']);path=inventory[curve];ids=fragment['candidate_ids']
        if path['operator']!='paint_center' or path.get('closed') is not False:
            receipt['non_straight_material_role_rejected']+=1;continue
        if not ids or any(type(i)is not int or not 0<=i<counts for i in ids):
            raise ValueError('Line fragment references a nonexistent original candidate')
        matched=[]
        for pointer,report in reports:
            sides=report.get('boundary_reports',[])
            if (report.get('curve')==curve and report.get('source_line_support_qualified') is True
                    and report.get('paired_source_boundaries_required') is True
                    and len(sides)==2 and all(s.get('source_line_support_qualified') is True for s in sides)
                    and set(report.get('selected_candidate_ids',[]))==set(ids)):
                matched.append(pointer)
        if not matched:
            receipt['missing_exact_paired_whole_line_qualification']+=1;continue
        key=tuple(sorted(set(ids)))
        if key in seen:receipt['duplicate_qualified_group']+=1;continue
        seen.add(key);removed=sorted(set(ids)&set(excluded));runs=retained_runs(ids,excluded)
        receipt['occluded_qualified_line_readouts_removed']+=len(removed)
        receipt['fully_occluded_qualified_line_groups']+=not bool(runs)
        receipt['qualified_line_groups']+=len(runs)
        for run_index,run in enumerate(runs):
            yield dict(frame=frame,candidate_ids=run,source_line_support_qualified=True,
                source_frame_pts=int(record['pts']),source_fragment_index=fragment_index,
                qualification_pointers=matched,query_name_is_identity=False,
                query_curve=curve,query_curve_name_navigation_only=path['curve_id'],native_path_id=path['native_path_id'],
                original_qualification_not_recomputed=True,retained_subset_run=run_index,
                occluded_original_candidate_ids=removed)


def guarded_candidate_links(record,previous_uv,current_uv,previous_excluded,current_excluded):
    """Requalify only subsets of saved actual bidirectional RGB pairs."""
    eligible=[];insufficient=[];receipt=Counter()
    for ei,original in enumerate(record['links']):
        if original.get('source_only') is not True or original.get('identity_assumed_from_curve_name') is not False:
            raise ValueError('Identity propagation requires actual independent RGB links')
        if original.get('pairs_index_space')!='previous/current_original_candidate_npz_candidate_id':
            raise ValueError('Link pairs are not original candidate IDs')
        pairs=np.asarray(original['pairs'])
        if pairs.ndim!=2 or pairs.shape[1]!=2 or pairs.dtype.kind not in 'iu':raise ValueError('Invalid original candidate pairs')
        if ((pairs<0).any() or (pairs[:,0]>=len(previous_uv)).any() or (pairs[:,1]>=len(current_uv)).any()):
            raise ValueError('Original RGB pair candidate outside immutable measurement')
        proofs=defaultdict(list);removed=[]
        for pi,(a,b) in enumerate(pairs):
            if int(a)in previous_excluded or int(b)in current_excluded:removed.append(pi)
            else:proofs[int(a),int(b)].append(pi)
        kept=sorted(proofs);first=sorted({a for a,b in kept});last=sorted({b for a,b in kept})
        first_span=float(np.linalg.norm(np.ptp(previous_uv[first].astype(float),axis=0)))if first else 0.
        last_span=float(np.linalg.norm(np.ptp(current_uv[last].astype(float),axis=0)))if last else 0.
        edge=dict(a=original['a'],b=original['b'],pairs=[list(p)for p in kept],support=len(last),previous_support=len(first),
            actual_previous_UV_span_px=first_span,actual_current_UV_span_px=last_span,
            original_edge_index=ei,original_pair_indices=[proofs[p]for p in kept],rejected_guard_pair_indices=removed,
            original_quality_summary={k:v for k,v in original.items()if k not in ('pairs','node_pairs')},
            old_quality_summary_recomputed_for_subset=False,source_only=True,identity_assumed_from_curve_name=False,
            pairs_index_space='previous/current_original_candidate_npz_candidate_id',all_pairs_original_actual_subset=True)
        receipt['original_pair_occurrences']+=len(pairs);receipt['occluded_pair_occurrences_removed']+=len(removed)
        if min(len(first),len(last))<3 or last_span<56.:
            edge['reason']='insufficient_actual_unique_readouts_or_actual_UV_span';insufficient.append(edge)
            receipt['insufficient_support_edges']+=1;continue
        eligible.append(edge)
    qualify_fragment_links(eligible)
    receipt['eligible_requalified_edges']=len(eligible)
    receipt['unambiguous_links_used']=sum(e['unambiguous']for e in eligible)
    receipt['ambiguous_links_not_used']=sum(not e['unambiguous']for e in eligible)
    receipt['original_candidate_pairs_used']=sum(len(e['pairs'])for e in eligible if e['unambiguous'])
    return eligible,insufficient,receipt


def checked_anchor_document(path):
    document=read(path)
    if document.get('schema')!='rinkspace.reviewed_source_anchors.v1' or not isinstance(document.get('anchors'),list):
        raise ValueError('Expected rinkspace.reviewed_source_anchors.v1 anchors list')
    required={'capture','shot','pts','entity','candidate_ids','source_BGR_sha256','measurement_sha256'}
    for record in document['anchors']:
        if not required<=record.keys():raise ValueError('Incomplete reviewed source anchor')
        if type(record['pts'])is not int or not isinstance(record['entity'],str) or not record['entity'].strip():
            raise ValueError('Anchor requires integer native PTS and an explicit entity identity')
        ids=record['candidate_ids']
        if not isinstance(ids,list) or not ids or any(type(i)is not int or i<0 for i in ids) or len(set(ids))!=len(ids):
            raise ValueError('Anchor requires distinct nonnegative original candidate IDs')
        rejected=record['entity'].startswith('rejected:')
        if rejected != (record.get('disposition')=='confirmed_not_target_material'):
            raise ValueError('Rejected material anchors require rejected: identity and confirmed_not_target_material disposition')
    return document


def propagate_shot(name,shot,scope,frame_index,pair_index,capture_binding,anchors,label_values,output,exclusions=None):
    started=time.monotonic();inventory=capture_binding['native_inventory'];frames=[];groups=[];group_receipt=Counter()
    by_pts={};navigation_frames=defaultdict(set);source_conflict_refs=[];frame_uv=[];excluded_ids=[]
    for fi,item in enumerate(sorted(frame_index,key=lambda row:int(row['pts']))):
        metadata=read(item['metadata_path']);source=metadata['source_frame'];record=read(source['path'])
        if digest(source['path'])!=source['sha256']:raise ValueError('Capture frame metadata changed')
        raw=metadata['source_candidates'];sha=digest(raw['path'])
        if sha!=raw['sha256'] or sha!=item['source_candidates']['sha256'] or record['raw_candidates']['sha256']!=sha:
            raise ValueError('Original candidate measurement SHA changed')
        with np.load(raw['path'],allow_pickle=False) as arrays:uv=arrays['candidate_uv'];count=len(uv)
        if count!=metadata['original_candidate_count'] or count!=record['summary']['actual_RGB_candidates']:
            raise ValueError('Original candidate count differs')
        if metadata['source_BGR_sha256']!=record['source_BGR_sha256'] or metadata['source_RGB_sha256']!=item['source_RGB_sha256']:
            raise ValueError('Original decoded source identity differs')
        pts=int(item['pts'])
        blocked=(exclusions.for_frame(name,shot,pts,raw['path'],sha,record['source_BGR_sha256'],metadata['source_RGB_sha256'],count)
            if exclusions is not None else frozenset())
        frame_uv.append(uv);excluded_ids.append(blocked)
        if pts in by_pts:raise ValueError('Duplicate canonical frame PTS')
        by_pts[pts]=fi
        frames.append(dict(frame=fi,pts=pts,candidate_count=count,measurement_sha256=sha,
            source_BGR_sha256=record['source_BGR_sha256'],source_RGB_sha256=metadata['source_RGB_sha256'],
            capture_frame=source,source_candidates=raw,source_link_frame=dict(path=item['metadata_path'],sha256=digest(item['metadata_path']))))
        for label,value in record['summary']['by_path'].items():
            if value['queries']:navigation_frames[label].add(pts)
        for group in verified_line_groups(record,inventory,fi,count,group_receipt,blocked):
            group['capture_frame']=source;groups.append(group)
        if metadata.get('same_frame_conflicts'):
            source_conflict_refs.append(dict(frame=fi,pts=pts,source_link_frame=frames[-1]['source_link_frame'],
                same_frame_conflict_count=len(metadata['same_frame_conflicts']),
                conflicts_are_not_identity_links=True))
    expanded=[];selected_anchors=[];skipped_anchors=[]
    for anchor_index,anchor in anchors:
        if anchor['shot']!=shot or anchor['pts']not in by_pts:continue
        fi=by_pts[anchor['pts']];frame=frames[fi]
        if anchor['source_BGR_sha256']!=frame['source_BGR_sha256'] or anchor['measurement_sha256']!=frame['measurement_sha256']:
            raise ValueError('Reviewed anchor source/measurement binding changed')
        if any(i>=frame['candidate_count']for i in anchor['candidate_ids']):raise ValueError('Anchor candidate outside original measurement')
        selected_anchors.append(dict(anchor_record_index=anchor_index,**anchor))
        removed=sorted(set(anchor['candidate_ids'])&excluded_ids[fi])
        if removed:skipped_anchors.append(dict(anchor_record_index=anchor_index,entity=anchor['entity'],pts=anchor['pts'],
            candidate_ids=removed,reason='actual_readout_intersects_new_person_guard',original_anchor_preserved=True))
        for candidate in anchor['candidate_ids']:
            if candidate in excluded_ids[fi]:continue
            expanded.append(dict(frame=fi,candidate=candidate,identity=anchor['entity'],anchor_record_index=anchor_index))
    link_receipt=Counter();pair_sources=[]
    def links():
        for item in pair_index:
            previous=int(item['previous_pts']);current=int(item['current_pts'])
            if previous not in by_pts or current not in by_pts:continue
            if current-previous!=256:raise ValueError('Nonadjacent source PTS link')
            record=read(item['metadata_path']);a=by_pts[previous];b=by_pts[current]
            if (record['previous_source_RGB_sha256']!=frames[a]['source_RGB_sha256'] or
                    record['current_source_RGB_sha256']!=frames[b]['source_RGB_sha256']):
                raise ValueError('Source link exposure binding changed')
            original_ref=dict(path=item['metadata_path'],sha256=digest(item['metadata_path']))
            if exclusions is not None:
                edges,insufficient,receipt=guarded_candidate_links(record,frame_uv[a],frame_uv[b],excluded_ids[a],excluded_ids[b])
                link_receipt.update(receipt)
                target=output/name/'pairs'/shot/f'{previous}_{current}.json.gz';target.parent.mkdir(parents=True,exist_ok=True)
                write(target,dict(original_links=original_ref,previous_pts=previous,current_pts=current,
                    links=edges,insufficient_support_edges=insufficient,receipt=dict(receipt),
                    fragment_competition_requalified=True,NCC_or_normal_thresholds_changed=False,new_pairs_created=False,fit_eligible=False))
                pair_sources.append(dict(previous_pts=previous,current_pts=current,path=str(target),sha256=digest(target),original_links=original_ref))
                for edge in edges:
                    if edge['unambiguous']:yield dict(previous=a,current=b,pairs=np.asarray(edge['pairs'],int),source_only=True,unambiguous=True)
                continue
            pair_sources.append(dict(previous_pts=previous,current_pts=current,**original_ref))
            for edge in record['links']:
                if edge.get('unambiguous') is not True:
                    link_receipt['ambiguous_links_not_used']+=1;continue
                if edge.get('source_only') is not True or edge.get('identity_assumed_from_curve_name') is not False:
                    raise ValueError('Identity propagation requires actual independent RGB links')
                if edge.get('pairs_index_space')!='previous/current_original_candidate_npz_candidate_id':
                    raise ValueError('Link pairs are not original candidate IDs')
                pairs=np.asarray(edge['pairs'])
                if pairs.ndim!=2 or pairs.shape[1]!=2 or pairs.dtype.kind not in 'iu':raise ValueError('Invalid original candidate pairs')
                link_receipt['unambiguous_links_used']+=1;link_receipt['original_candidate_pairs_used']+=len(pairs)
                yield dict(previous=a,current=b,pairs=pairs,source_only=True,unambiguous=True)
    counts=np.array([f['candidate_count']for f in frames],np.int64)
    offsets,component,components=propagate_readout_identities(counts,links(),expanded,groups)
    labels={label:i for i,label in enumerate(label_values)};root_labels=np.full(len(component),-1,np.int32)
    conflicts=[]
    for item in components:
        identities=item['identities']
        root_labels[item['component']]=labels[identities[0]] if len(identities)==1 else -2
        if len(identities)>1:conflicts.append(item)
    label_index=root_labels[component];label_is_rejected=np.array([v.startswith('rejected:')for v in label_values],bool)
    candidate_is_rejected=np.zeros(len(label_index),bool);labeled=label_index>=0
    candidate_is_rejected[labeled]=label_is_rejected[label_index[labeled]]
    for fi,blocked in enumerate(excluded_ids):
        indices=int(offsets[fi])+np.asarray(sorted(blocked),np.int64)
        if len(indices)and((label_index[indices]!=-1).any()or(component[indices]!=indices).any()):
            raise ValueError('Guarded original readout entered an identity component')
    folder=output/name;folder.mkdir(exist_ok=True)
    target=folder/f'{shot}.npz'
    np.savez_compressed(target,pts=np.array([f['pts']for f in frames],np.int64),candidate_counts=counts,
        offsets=offsets,component=component,label_index=label_index,label_values=np.asarray(label_values,dtype=str),
        label_is_rejected=label_is_rejected)
    all_pts=set(by_pts);entities=[];negative=[]
    for label,li in labels.items():
        associated=set();conflict_frames=set();readouts=conflict_readouts=0
        conflict_roots={c['component']for c in conflicts if label in c['identities']}
        for fi,frame in enumerate(frames):
            sl=slice(int(offsets[fi]),int(offsets[fi+1]));n=int(np.sum(label_index[sl]==li))
            if n:associated.add(frame['pts']);readouts+=n
            if conflict_roots:
                nconf=int(np.isin(component[sl],list(conflict_roots)).sum())
                if nconf:conflict_frames.add(frame['pts']);conflict_readouts+=nconf
        coverage=dict(entity=label,uniquely_associated_candidates=readouts,associated_frames=len(associated),
            associated_native_PTS_intervals=intervals(associated),conflicting_candidates=conflict_readouts,
            conflicting_native_PTS_intervals=intervals(conflict_frames),
            unassociated_frames=len(all_pts-associated),unassociated_native_PTS_intervals=intervals(all_pts-associated),
            missing_visibility_not_inferred=True,fit_eligible=False)
        if label.startswith('rejected:'):
            coverage['disposition']='confirmed_not_target_material'
            coverage['negative_material_evidence_only']=True
            coverage['uniquely_rejected_candidates']=coverage.pop('uniquely_associated_candidates')
            coverage['rejected_frames']=coverage.pop('associated_frames')
            coverage['rejected_native_PTS_intervals']=coverage.pop('associated_native_PTS_intervals')
            negative.append(coverage)
        else:entities.append(coverage)
    anchored_by_root={item['component']:item for item in components};negative_groups=[]
    for group in groups:
        first=group['candidate_ids'][0];ci=int(component[int(offsets[group['frame']])+first])
        group['component']=ci;entry=anchored_by_root.get(ci)
        if entry and any(identity.startswith('rejected:')for identity in entry['identities']):
            negative_groups.append(dict(frame=group['frame'],pts=group['source_frame_pts'],component=ci,
                query_curve_name_navigation_only=group['query_curve_name_navigation_only'],native_path_id=group['native_path_id'],
                candidate_ids=group['candidate_ids'],qualification_pointers=group['qualification_pointers'],
                identities=entry['identities'],component_status=entry['status'],capture_frame=group['capture_frame'],
                negative_material_evidence_only=True,fit_eligible=False))
    report=dict(schema='rinkspace.candidate_identity_propagation.v1',capture=name,shot=shot,frames=len(frames),
        original_candidates=int(counts.sum()),arrays=dict(path=str(target),sha256=digest(target)),
        label_index_states={'-1':'no reviewed identity connected','-2':'conflicting reviewed identities; no winner selected','0_and_above':'label_values index; rejected: labels are negative material evidence, never admission'},
        reviewed_anchor_records=selected_anchors,expanded_anchor_readouts=expanded,anchored_components=components,
        skipped_occluded_anchor_readouts=skipped_anchors,readout_exclusions=exclusions.receipt()if exclusions is not None else None,
        conflict_components=conflicts,entities=entities,negative_material_evidence=negative,
        negative_or_conflicted_whole_line_groups=negative_groups,
        line_group_receipt=dict(group_receipt),source_link_receipt=dict(link_receipt),
        qualified_groups=groups,frames_index=frames,used_pair_metadata=pair_sources,
        same_frame_conflict_sources=source_conflict_refs,same_frame_conflicts_not_merged_or_silently_resolved=True,
        navigation_query_coverage={label:dict(frames=len(v),native_PTS_intervals=intervals(v),physical_identity=False)for label,v in navigation_frames.items()},
        unassociated_frame_denominator='all selected actual source frames; source visibility is not inferred from query names',
        curve_names_used_as_identities=False,raw_measurements_modified=False,fit_eligible=False,new_fit_started=False,
        elapsed_seconds=time.monotonic()-started,selected_scope=scope)
    write(folder/f'{shot}.json.gz',report)
    summary={k:report[k]for k in ('capture','shot','frames','original_candidates','arrays','line_group_receipt','source_link_receipt','entities','negative_material_evidence','elapsed_seconds')}
    summary.update(anchored_components=len(components),conflicting_components=len(conflicts),
        associated_candidates=int((labeled&~candidate_is_rejected).sum()),uniquely_labeled_candidates=int(labeled.sum()),
        rejected_material_candidates=int(candidate_is_rejected.sum()),conflicting_candidates=int((label_index==-2).sum()),
        negative_or_conflicted_whole_line_groups=len(negative_groups),
        unassociated_candidates=int((label_index==-1).sum()),
        guard_excluded_candidates=sum(map(len,excluded_ids)),skipped_occluded_anchor_candidates=sum(len(a['candidate_ids'])for a in skipped_anchors),fit_eligible=False)
    print(json.dumps({k:summary[k]for k in ('capture','shot','frames','original_candidates','anchored_components','conflicting_components','associated_candidates','elapsed_seconds')}),flush=True)
    return summary


def run(args):
    document=checked_anchor_document(args.anchors);all_anchors=document['anchors'];labels=sorted({a['entity']for a in all_anchors})
    exclusions=ReadoutExclusions(args.readout_exclusions)if getattr(args,'readout_exclusions',None) is not None else None
    output=guard_project_write(args.output)
    if output.exists() and any(output.glob('*/*.npz')):raise FileExistsError(output)
    output.mkdir(parents=True,exist_ok=True);results=[];inputs={str(args.anchors.resolve()):digest(args.anchors)}
    for name in args.captures:
        folder=args.links_root/name;coverage=read(folder/'coverage.json')
        if coverage.get('complete') is not True or coverage.get('all_original_candidate_SHA_and_node_UV_verified') is not True:
            raise ValueError('Complete verified source links required')
        capture=Path(coverage['capture']);capture_binding=read(capture/'binding.json')
        if exclusions is not None:exclusions.bind_capture(name,capture)
        if digest(capture/'binding.json')!=coverage['candidate_id_space_capture_binding_sha256']:
            raise ValueError('Capture candidate-ID-space binding changed')
        inputs[str((folder/'coverage.json').resolve())]=digest(folder/'coverage.json')
        for key in ('frames_index','pairs_index'):
            if digest(coverage[key]['path'])!=coverage[key]['sha256']:raise ValueError('Source link index changed')
            inputs[coverage[key]['path']]=coverage[key]['sha256']
        frames=list(rows(coverage['frames_index']['path']));pairs=list(rows(coverage['pairs_index']['path']))
        anchors=[(i,a)for i,a in enumerate(all_anchors)if a['capture']in (name,str(capture))]
        for shot in args.shots:
            selected=[f for f in frames if f['shot']==shot and (args.pts is None or f['pts']in args.pts)]
            if not selected:continue
            scope=dict(requested_pts=args.pts,whole_shot=args.pts is None,source_link_coverage=str(folder/'coverage.json'))
            results.append(propagate_shot(name,shot,scope,selected,[p for p in pairs if p['shot']==shot],capture_binding,anchors,labels,output,exclusions))
    binding=dict(schema='rinkspace.candidate_identity_propagation_run.v1',entry_sha256=digest(__file__),
        public_algorithm_sha256=digest(propagate_readout_identities.__code__.co_filename),inputs=inputs,
        anchor_schema=document['schema'],captures=args.captures,shots=args.shots,pts=args.pts,
        candidate_id_spaces_kept_separate=True,curve_names_used_as_identities=False,
        readout_exclusions=exclusions.receipt()if exclusions is not None else None,
        raw_measurements_modified=False,fit_eligible=False,new_fit_started=False)
    write(output/'binding.json',binding);write(output/'coverage.json',dict(binding=binding,shots=results,complete=True,fit_eligible=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--links-root',type=Path,required=True);parser.add_argument('--anchors',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--captures',nargs='+',choices=['full_direct','luminance_full'],default=['full_direct','luminance_full'])
    parser.add_argument('--shots',nargs='+',choices=['s07','s08','s09'],default=['s07','s08','s09'])
    parser.add_argument('--pts',nargs='+',type=int)
    parser.add_argument('--readout-exclusions',type=Path,help='Complete all-original-candidate guard coverage JSON')
    run(parser.parse_args())


if __name__=='__main__':main()
