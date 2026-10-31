"""Apply explicit source-review decisions without changing measured material.

This writes qualified/pending source rows and bounded replacement declarations.
It does not update prepared data, initialize a production camera, or fit pixels.
"""
from pathlib import Path
import argparse
from collections import Counter,defaultdict
import hashlib
import json
import os
import shutil
import time

import numpy as np

from rinkspace.path_policy import guard_project_write
from prepare_native_source_update import (authenticate_replacement_spans,
    in_remeasured_line_spans as _native_span_contains,authenticate_packet)


def in_remeasured_line_spans(uv,spans):
    # JSON declares lists; the numeric contract consumes array endpoints.
    return _native_span_contains(uv,np.asarray(spans,dtype=float))


def digest(path):
    with Path(path).open('rb')as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def ref(path):return dict(path=str(Path(path).resolve()),sha256=digest(path))


def read(path):return json.loads(Path(path).read_text('utf-8-sig'))


def line(value):return json.dumps(value,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n'


def write(path,value):Path(path).write_text(line(value),encoding='utf8')


def row_reason(row,sid,pts,decisions):
    for region in decisions['pending_regions']:
        if region['shot']!=sid or region['pts']!=pts or region['entity']!=row['entity']:continue
        x,y=row['uv'];a,b,c,d=region['box']
        if a<=x<=c and b<=y<=d:return region['reason'],False
    resolved=decisions.get('resolved_material_pending',{})
    allow=(sid==resolved.get('shot') and pts==resolved.get('pts')
           and (row['entity'] in resolved['entities'] if 'entities' in resolved
                else row['entity'].startswith(resolved.get('entities_prefix','\0'))))
    if row.get('material_pending') and not allow:return 'material_pending_without_explicit_source_resolution',False
    return None,bool(row.get('material_pending') and allow)


def bounded_spans(original,frame_rows,admitted):
    """Keep each original span separate and split at pending actual rows."""
    output=[];stats=Counter()
    for group in original['groups']:
        key=group['pts'],group['path_id'];items=frame_rows.get(key,[])
        keep=[(i,r)for i,r in items if i in admitted[key]]
        pending=[r['uv']for i,r in items if i not in admitted[key]]
        if not keep:continue
        spans=[]
        for original_span in group['spans']:
            a,b=np.asarray(original_span,float);direction=b-a
            ordered=[(i,r)for i,r in items if in_remeasured_line_spans(r['uv'],[original_span])]
            ordered.sort(key=lambda item:(float((np.asarray(item[1]['uv'])-a)@direction),item[0]))
            run=[]
            for i,row in ordered:
                if i in admitted[key]:run.append(row['uv'])
                elif run:spans.append([run[0],run[-1]]);run=[]
            if run:spans.append([run[0],run[-1]])
        # Equal projected stations from different source branches cannot grant
        # a continuous interval through a pending pixel. Actual singleton spans
        # preserve only those readouts when projection alone is ambiguous.
        safe=[]
        for span in spans:
            if any(in_remeasured_line_spans(uv,[span])for uv in pending):
                safe.extend([[r['uv'],r['uv']]for _,r in keep if in_remeasured_line_spans(r['uv'],[span])])
                stats['ambiguous_projection_spans_reduced_to_actual_singletons']+=1
            else:safe.append(span)
        unique=[];seen=set()
        for span in safe:
            k=tuple(map(tuple,span))
            if k not in seen:unique.append(span);seen.add(k)
        actual={tuple(r['uv'])for _,r in keep}
        if (not unique or any(tuple(uv)not in actual for span in unique for uv in span)
                or not all(in_remeasured_line_spans(uv,unique)for uv in actual)
                or any(in_remeasured_line_spans(uv,unique)for uv in pending)):
            raise ValueError('Replacement interval changed source endpoints or crossed pending support')
        output.append(dict(pts=key[0],path_id=key[1],spans=unique))
        stats['partial_line_span_groups']+=1;stats['partial_line_spans']+=len(unique)
    return dict(policy='partial_line_spans',groups=output),dict(stats)


def span_points(records):
    paths=list(dict.fromkeys(r['path_id']for _,r in records));lookup={p:i for i,p in enumerate(paths)}
    return dict(path_ids=paths),dict(pts=np.array([p for p,r in records],np.int64),
        path_indices=np.array([lookup[r['path_id']]for p,r in records],np.int32),
        uv=np.asarray([r['uv']for p,r in records],float).reshape(-1,2))


def qualify(args):
    started=time.perf_counter();batch=args.batch.resolve();decisions=read(args.decisions)
    # Both files are producer completion artifacts; do not read a live export.
    metadata=read(batch/'metadata.json');overview=read(batch/'overview.json')
    if (decisions.get('schema')!='rinkspace.registration_batch_main_session_source_review.v1'
            or Path(decisions['source_batch']).resolve()!=batch or not decisions.get('authorization')
            or decisions.get('user_visual_acceptance') or decisions.get('pixel_accuracy_accepted')
            or metadata['denominator_native_frames']!=3866 or overview['denominator_native_frames']!=3866):
        raise ValueError('Completed batch and explicit source-only review decisions required')
    output=guard_project_write(args.output.resolve())
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True);shutil.copyfile(args.decisions,output/'review_decisions.json')
    review=ref(output/'review_decisions.json');protected={str(args.decisions.resolve()):digest(args.decisions),
        str(batch/'metadata.json'):digest(batch/'metadata.json'),str(batch/'overview.json'):digest(batch/'overview.json'),
        str(Path(__file__).resolve()):digest(__file__)}
    summaries={};all_spans=[];resolved_count=0;total_pending=0
    with (output/'pending.jsonl').open('w',encoding='utf8')as pending_all:
        for sid,source in metadata['shots'].items():
            folder=output/sid;folder.mkdir();source_path=batch/sid/'source_rows.jsonl';source_ref=ref(source_path)
            if source_ref!=source['source_rows']:raise ValueError('Completed source rows changed')
            if decisions.get('source_row_sha256',{}).get(sid,source_ref['sha256'])!=source_ref['sha256']:
                raise ValueError('Review source rows SHA differs')
            protected[str(source_path)]=source_ref['sha256']
            poses=args.poses_root/sid/'poses.npz';pose_ref=ref(poses)
            with np.load(poses,allow_pickle=False)as z:expected=z['pts'].copy()
            if len(expected)!=source['expected_native_frames']or not np.all(np.diff(expected)==256):
                raise ValueError('Reused pose clock differs')
            try:os.link(poses,folder/'poses.npz');hardlink=True
            except OSError:shutil.copyfile(poses,folder/'poses.npz');hardlink=False
            protected[str(poses.resolve())]=pose_ref['sha256']
            counts=Counter();all_rows=defaultdict(list);admitted=defaultdict(set);qualified_records=[]
            with source_path.open(encoding='utf8')as stream,(folder/'source_rows.jsonl').open('w',encoding='utf8')as target,(folder/'pending.jsonl').open('w',encoding='utf8')as pending_shot:
                for number,raw in enumerate(stream,1):
                    frame=json.loads(raw);pts=frame['pts']
                    if number>len(expected)or pts!=int(expected[number-1]):raise ValueError('Source rows exposure order differs')
                    accepted=[]
                    for ri,row in enumerate(frame['rows']):
                        key=pts,row['path_id'];all_rows[key].append((ri,row));reason,resolved=row_reason(row,sid,pts,decisions)
                        counts['original_rows']+=1
                        if reason:
                            item=dict(shot=sid,pts=pts,source_BGR_sha256=frame['source_BGR_sha256'],reason=reason,
                                original_source_row=dict(source=source_ref,line=number,row_index=ri),row=row,fit_eligible=False)
                            pending_shot.write(line(item));pending_all.write(line(item));counts['pending_rows']+=1;total_pending+=1
                            continue
                        qualified=dict(row,fit_eligible=True,review_state='main_session_fit_qualified_not_user_accepted',
                            source_review=review)
                        if resolved:
                            qualified.update(material_pending=False,source_material_resolution=dict(review=review,
                                original_material_pending=True,conclusion=decisions['resolved_material_pending']['conclusion']))
                            counts['resolved_material_pending_rows']+=1;resolved_count+=1
                        # All measurement, entity and provenance fields remain original.
                        for name,value in row.items():
                            if name not in ('fit_eligible','review_state','material_pending','source_review')and qualified[name]!=value:
                                raise ValueError('Source measurement or identity changed')
                        accepted.append(qualified);admitted[key].add(ri);qualified_records.append((pts,qualified))
                    result=dict(frame,rows=accepted,fit_eligible=True,source_UV_unchanged=True,
                        review_state='main_session_fit_qualified_not_user_accepted',source_review=review,
                        source_candidate_export=dict(source=source_ref,line=number,original_rows=len(frame['rows'])))
                    target.write(line(result));counts['frames']+=1;counts['qualified_frames_with_rows']+=bool(accepted);counts['qualified_rows']+=len(accepted)
            if counts['frames']!=len(expected)or counts['original_rows']!=source['output_rows']:
                raise ValueError('Completed source scope count differs')
            original_spans_path=batch/sid/'source_replacement.json';protected[str(original_spans_path)]=digest(original_spans_path)
            declaration,span_counts=bounded_spans(read(original_spans_path),all_rows,admitted)
            mapping,points=span_points(qualified_records);mapping['source_replacement']=declaration
            authenticated=authenticate_replacement_spans(mapping,points)
            if len(authenticated)!=len(declaration['groups']):raise ValueError('Partial replacement authentication differs')
            write(folder/'source_replacement.json',declaration);all_spans.extend(declaration['groups'])
            summaries[sid]=dict(counts,**span_counts,source_rows=ref(folder/'source_rows.jsonl'),
                source_replacement=ref(folder/'source_replacement.json'),poses_source=pose_ref,
                poses_reused_by_hardlink=hardlink,source_measurements_and_identity_unchanged=True,
                replacement_endpoints_are_actual_admitted_UV=True,pending_never_inside_replacement_spans=True)
            print(line(dict(shot=sid,counts=dict(counts),span_counts=span_counts)),end='',flush=True)
    expected_resolved=decisions.get('resolved_material_pending',{}).get('expected_rows')
    if expected_resolved is not None and resolved_count!=expected_resolved:
        raise ValueError(f'Explicit resolved-material row count differs: {resolved_count} != {expected_resolved}')
    if sum(s['frames']for s in summaries.values())!=3866:raise ValueError('Qualified batch must retain all3866 exposures')
    for path,sha in protected.items():
        if digest(path)!=sha:raise ValueError('Source/review/pose input changed while qualifying')
    write(output/'source_replacement.json',dict(policy='partial_line_spans',groups=all_spans))
    summary=dict(schema='rinkspace.registration_source_qualification.v1',batch=str(batch),shots=summaries,
        review=review,original_decisions=ref(args.decisions),total_frames=3866,
        qualified_rows=sum(s['qualified_rows']for s in summaries.values()),pending_rows=total_pending,
        all_source_measurement_fields_unchanged=True,only_explicit_pending_measurements_excluded=True,
        closed_groups_keep_whole_group_policy=True,original_batch_modified=False,
        poses_for_packet_initialization_only=True,existing_joint_camera_and_controls_must_be_preserved=True,
        user_visual_acceptance=False,pixel_accuracy_accepted=False,prepared_modified=False,
        new_fit_started=False,elapsed_seconds=time.perf_counter()-started)
    write(output/'qualification_summary.json',summary)
    print(line(dict(output=str(output),qualified_rows=summary['qualified_rows'],pending_rows=total_pending,
        elapsed_seconds=summary['elapsed_seconds'])),end='',flush=True)


def finalize_packets(prepared_root,qualified,packets):
    """Attach authenticated declarations to already built seven-field packets."""
    current,qualified,packets=map(Path,(prepared_root,qualified,packets))
    manifest=read(current/'joint_manifest.json')
    old_mapping=read(current/'mapping.json')
    with np.load(current/'point_observations.npz',allow_pickle=False)as z:old_points=dict(z)
    summary=ref(qualified/'qualification_summary.json');review=ref(qualified/'review_decisions.json')
    receipt=read(packets/'packets_manifest.json');validation=[]
    for item in receipt['shots']:
        sid=item['shot_id'];folder=packets/sid;mapping=read(folder/'mapping.json')
        declaration=read(qualified/sid/'source_replacement.json')
        provenance=dict(schema='rinkspace.registration_packet_qualification.v1',qualification=summary,review=review,
            qualified_source_rows=ref(qualified/sid/'source_rows.jsonl'),source_replacement=ref(qualified/sid/'source_replacement.json'),
            original_measurements_and_identity_unchanged=True,poses_for_packet_construction_only=True,
            production_camera_or_control_update_performed=False,user_visual_acceptance=False,prepared_modified=False,new_fit_started=False)
        write(folder/'qualification_provenance.json',provenance)
        mapping.update(source_replacement=declaration,fit_eligible=True,
            qualification_provenance=ref(folder/'qualification_provenance.json'),source_review=review,
            existing_joint_camera_and_controls_must_be_preserved=True)
        write(folder/'mapping.json',mapping)
        authenticated=authenticate_packet(folder,manifest,old_mapping,old_points)
        with np.load(folder/'point_observations.npz',allow_pickle=False)as z:points=dict(z)
        if not np.array_equal(points['uv'],authenticated['points']['uv']):raise ValueError('Authenticated packet UV differs')
        item.update(mapping=ref(folder/'mapping.json'),qualification_provenance=ref(folder/'qualification_provenance.json'))
        validation.append(dict(shot=sid,rows=authenticated['authenticated_rows'],mapping=item['mapping'],
            authenticated_packet=True,authenticated_partial_span_groups=len(authenticated['replacement_spans']),
            source_UV_exact=True,all_kind1=bool(np.all(points['kind']==1))))
    receipt.update(source_qualification=summary,source_review=review,packet_mapping_and_source_rows_authenticated=True,
        existing_joint_camera_and_controls_must_be_preserved=True,prepare_executed=False,pixel_fit_executed=False)
    write(packets/'packets_manifest.json',receipt)
    result=dict(schema='rinkspace.registration_packet_validation.v1',shots=validation,
        rows=sum(s['rows']for s in validation),qualified_source_rows_and_spans_authenticated=True,
        original_measurement_fields_unchanged=True,user_visual_acceptance=False,prepared_modified=False,new_fit_started=False)
    write(packets/'validation.json',result)
    print(line(dict(packet_output=str(packets),rows=result['rows'],authenticated=True)),end='',flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('batch','decisions','poses-root','output'):p.add_argument('--'+name,type=Path,required=True)
    qualify(p.parse_args())
