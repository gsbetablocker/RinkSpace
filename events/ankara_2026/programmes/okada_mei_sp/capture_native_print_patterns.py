"""Read complete print patterns on all registered exposures from current RGB.

Current native geometry locates searches. Whole-pattern photographic registration
and original signed material profiles establish the actual source coordinates.
Panel detections never suppress glyph searches. No fitting or admission occurs.
"""
from pathlib import Path
from collections import Counter, defaultdict
import argparse
import copy
import hashlib
from itertools import zip_longest
import json
import time
import cv2
import numpy as np

from rinkspace.core.native import NativeCore
from rinkspace.domain.parameters import SceneParameterModel, ParameterState
from rinkspace.domain.multi_shot_motion import ShotMotionTimeline
from rinkspace.review.scene import sample_native_review_scene
from rinkspace.session.print_patterns import PrintPattern
from rinkspace.session.print_pattern_navigation import coarse_print_registration
from rinkspace.path_policy import guard_project_write
from capture_independent_materials import spaced_indices
from programme_source_frames import iter_registration_source_frames


def read(path):
    return json.loads(Path(path).read_text('utf-8-sig'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def plain(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, dict): return {k:plain(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)): return [plain(v) for v in value]
    return value


def write(path, value):
    Path(path).write_text(json.dumps(plain(value), ensure_ascii=False, indent=2, allow_nan=False)+'\n',encoding='utf8')


def bbox_corners(points):
    lo,hi=np.min(points,axis=0),np.max(points,axis=0)
    return np.array([lo,[hi[0],lo[1]],hi,[lo[0],hi[1]]])


def file_ref(path):
    return dict(path=str(Path(path).resolve()),sha256=sha(path))


def pattern_key(note):
    return note['atlas_index'],note['instance']


def needs_retry(note):
    """Whether the original result may need registration recovery after reread."""
    summary=note.get('source_readout',{})
    return (note['registration']['status']!='source_pattern_registered'
        or summary.get('accepted',0)==0 or summary.get('reason','').startswith('no_resolved'))


def validate_retry_capture(root,binding):
    """Authenticate the original run while permitting a new material reader."""
    root=Path(root).resolve(strict=True);path=root/'binding.json';original=read(path)
    if original.get('schema')!=binding['schema']:raise ValueError('Retry capture schema differs')
    for name in ('prepared','state','source_atlases','native_library'):
        if Path(original[name]).resolve()!=Path(binding[name]).resolve():
            raise ValueError('Retry input identity differs: '+name)
    for name in ('state_sha256','source_atlases_sha256','native_library_sha256','source_adapter_sha256',
                 'query_spacing_px','retained_arc_spacing_px'):
        if original[name]!=binding[name]:raise ValueError('Retry input hash/contract differs: '+name)
    if not set(binding['shots'])<=set(original['shots']):raise ValueError('Retry shots absent from original capture')
    prior={pattern_key(p):p for p in original['patterns']}
    if len(prior)!=len(original['patterns']):raise ValueError('Duplicate original pattern identities')
    for item in binding['patterns']:
        if prior.get(pattern_key(item))!=item:raise ValueError('Retry atlas/pattern identity differs')
    for item in original['patterns']:
        if sha(item['reference_image'])!=item['reference_image_sha256']:
            raise ValueError('Original retry reference image changed')
    if 'prepared_inputs' in original and original['prepared_inputs']!=binding['prepared_inputs']:
        raise ValueError('Original retry prepared hashes differ')
    files={sid:{name:file_ref(root/sid/(name+'.jsonl'))for name in ('registration','source_rows')}
           for sid in binding['shots']}
    return dict(root=str(root),binding=file_ref(path),shots=files,
        original_reader_sha256=original['reader_sha256'],
        legacy_prepared_hash_basis='same prepared path and exact original state hash; per-file hashes compared when recorded',
        successful_original_paths_preserved=True,selected_pts_only_limits_retries_not_output_ledger=True)


def iter_retry_records(root,sid,shot):
    """Yield the full original exposure ledger, never a filtered retry subset."""
    expected=range(shot['first_pts'],shot['stop_pts_exclusive'],256);count=0
    with (Path(root)/sid/'source_rows.jsonl').open(encoding='utf8')as rows,\
         (Path(root)/sid/'registration.jsonl').open(encoding='utf8')as registrations:
        for number,pair in enumerate(zip_longest(rows,registrations),1):
            if None in pair:raise ValueError('Original retry exposure ledgers differ in length')
            source,registration=map(json.loads,pair)
            if count>=len(expected)or source['pts']!=expected[count]or registration['pts']!=source['pts']:
                raise ValueError('Original retry ledger has a missing, duplicate or non-native exposure')
            if len({pattern_key(p)for p in registration['patterns']})!=len(registration['patterns']):
                raise ValueError('Original retry frame has duplicate pattern identities')
            if not isinstance(registration.get('all_actual_readouts'),list):
                raise ValueError('Original retry frame lacks unthinned actual readouts')
            yield source,registration,number
            count+=1
    if count!=len(expected)or not count or expected[-1]!=shot['last_pts']:
        raise ValueError('Original retry ledger does not cover the complete shot')


def validate_retry_frame(frame,source):
    for name in ('pts','source_BGR_sha256','source_video_sha256','source_exclusion_sha256','mask_sha256'):
        if frame.get(name)!=source.get(name):raise ValueError('Retry original source identity differs: '+name)


def choose_readouts(hypotheses,original_source=None,original_registration=None):
    """Keep original successful path groups; fill missing paths from best retry."""
    by_path={};retained_by_path=defaultdict(list);provenance={}
    def registered_by(registration):
        # The full gate diagnostics remain in the named per-pattern note (or
        # hashed original capture); do not duplicate them for every path.
        return {key:registration[key]for key in ('status','correlation','H','shared_horizontal_blur_px')if key in registration}
    if original_registration is not None:
        notes={f"{n['atlas_index']}/{n['instance']}":n for n in original_registration['patterns']}
        for row in original_registration['all_actual_readouts']:
            by_path.setdefault(row['path_id'],[]).append(row)
        for row in original_source['rows']:retained_by_path[row['path_id']].append(row)
        if not set(retained_by_path)<=set(by_path):raise ValueError('Original retained rows lack unthinned path support')
        for pid,rows in by_path.items():
            label=rows[0]['source_registration_label']
            if label not in notes:raise ValueError('Original actual row lacks its registration')
            provenance[pid]=dict(origin='original_capture',registration_label=label,
                registration=registered_by(notes[label]['registration']),original_rows_unchanged=True)
    for candidate in sorted(hypotheses,key=lambda h:h['registration'].get('correlation',-1.),reverse=True):
        grouped=defaultdict(list)
        for row in candidate['rows']:grouped[row['path_id']].append(row)
        for pid,rows in grouped.items():
            if pid in by_path:continue
            by_path[pid]=rows
            retained_by_path[pid]=[rows[i]for i in spaced_indices([r['uv']for r in rows])]
            provenance[pid]=dict(origin=candidate['origin'],registration_label=candidate['label'],
                registration=registered_by(candidate['registration']),original_rows_unchanged=False)
    return ([row for pid in by_path for row in retained_by_path[pid]],
            [row for rows in by_path.values()for row in rows],provenance)


def read_pattern_attempt(item,image,exclusion,initial,source_binding,args,origin,original_registration=None,
                         original_had_readouts=False):
    label=f"{item['atlas_index']}/{item['instance']}"
    reuse=False;material_retry=None
    if original_registration is not None and original_registration['status']=='source_pattern_registered':
        registration=copy.deepcopy(original_registration)
        rows,summary=item['pattern'].read_paths(image,exclusion,registration,source_binding,spacing=args.query_spacing)
        reuse=bool(rows)or original_had_readouts;material_retry=summary
    if not reuse:
        if getattr(args,'coarse_navigation',False):
            registration=coarse_print_registration(item['pattern'],image,exclusion,initial)
        else:registration=item['pattern'].register(image,exclusion,initial)
        rows,summary=item['pattern'].read_paths(image,exclusion,registration,source_binding,spacing=args.query_spacing)
    for row in rows:
        suffix=row['path_id'].split('::',1)[1]
        row.update(path_id='print_instance/'+item['instance']+'::'+suffix,entity=item['instance'],
            capture_family='whole_print_original_RGB_glyph_boundary',source_atlas_index=item['atlas_index'],
            source_atlas=str(args.atlases),source_registration_label=label,
            source_identity_basis='Registered whole photographic pattern, native template path topology and registered physical camera/board instance; native search is not a source measurement')
    note=dict(atlas_index=item['atlas_index'],instance=item['instance'],initial_H=initial.tolist(),
        registration=registration,source_readout=summary,readout_origin=origin,
        registration_reused_for_material_retry=reuse,navigation_attempt_performed=not reuse)
    if material_retry is not None:
        note['original_registration_new_material_read']=material_retry
    if origin=='retry':note['actual_readouts']=rows
    return note,dict(label=label,registration=registration,rows=rows,origin=origin)


def run(args):
    output=guard_project_write(args.output.resolve())
    if output.exists(): raise FileExistsError(output)
    manifest=read(args.prepared/'joint_manifest.json')
    state=dict(np.load(args.state,allow_pickle=False))
    retry_root=getattr(args,'retry_capture',None)
    retry_original_binding=read(Path(retry_root)/'binding.json')if retry_root is not None else None
    library=Path(manifest['native_library']);library_sha=sha(library)
    if str(state['native_library_provenance_sha256'])!=library_sha: raise ValueError('Native/state provenance differs')
    scene_path=Path(manifest['scene_path']);scene=read(scene_path)
    if sha(scene_path)!=manifest['scene_file_sha256']: raise ValueError('Scene SHA changed')
    by_cut=None
    if retry_root is None:
        core=NativeCore(library);model=SceneParameterModel(scene,scene_path,optimize_scene=True)
        parameter_state=ParameterState(model.initial_state.layout,state['scene_values'])
        sampled=sample_native_review_scene(scene,core,scene_path,parameter_state=parameter_state,observed_path_ids=set())
        by_cut={c['id']:sampled['xyz'][c['begin']:c['end']] for c in sampled['cuts']}
    atlas_doc=read(args.atlases);patterns=[];all_world=[]
    for atlas_index in args.atlas_ids:
        atlas=dict(atlas_doc['atlases'][atlas_index])
        original_paths=atlas['paths']
        if len({p['path_id']for p in original_paths})!=len(original_paths):
            complete_kinds={'actual_high_resolution_glyph','original_40_mark_native_contour','actual_coloured_bar_source_boundary'}
            atlas['paths']=[p for p in original_paths if p.get('proof',{}).get('kind')in complete_kinds]
            atlas['excluded_partial_reference_paths']=[p['path_id']for p in original_paths if p not in atlas['paths']]
            if not atlas['paths']:raise ValueError('Atlas has only disconnected source fragments: '+str(atlas_index))
        if atlas['ad']==5:atlas['ink_polarity']='dark'
        if atlas.get('partial_source_reference'): raise ValueError('Partial atlas cannot establish whole pattern: '+str(atlas_index))
        if sha(atlas['image'])!=atlas['image_sha256']: raise ValueError('Reference image changed')
        reference=cv2.imread(atlas['image']);transform=atlas.get('photometric_transform')
        if transform=='invert_original_artwork_for_white_print_on_dark_field': reference=255-reference
        elif transform is not None: raise ValueError('Unknown reference photometric transform: '+transform)
        pattern=PrintPattern(atlas,reference)
        template='Ad'+str(atlas['ad'])
        instances=[template,'Ad'+str(atlas['ad']+13)] if args.include_paired else [template]
        for instance in instances:
            if (retry_original_binding is not None and instance in ('Ad15','Ad16')
                    and (atlas_index,instance)not in {pattern_key(p)for p in retry_original_binding['patterns']}):continue
            ids=['print_instance/'+instance+'::'+p['path_id'] for p in atlas['paths']]
            missing=[p for p in ids if by_cut is not None and p not in by_cut]
            if len(missing)==len(ids)and instance in ('Ad15','Ad16'):continue
            if missing: raise ValueError('Atlas paths absent from current native: '+str(missing))
            slices=[];source_boxes=[]
            for p,pid in zip(atlas['paths'],ids):
                if by_cut is not None:
                    begin=sum(len(w) for w in all_world);world=by_cut[pid];all_world.append(world)
                    slices.append((begin,begin+len(world)))
                source_boxes.extend(bbox_corners(np.asarray(p['pixels'])))
            patterns.append(dict(atlas_index=atlas_index,atlas=atlas,pattern=pattern,instance=instance,
                path_ids=ids,slices=slices,source_boxes=np.c_[source_boxes,np.ones(len(source_boxes))]))
    if retry_root is None:
        world=np.vstack(all_world);timeline=ShotMotionTimeline(manifest['shots'],manifest['control_dt'],manifest['first_control_time'])
    cv2.setNumThreads(2);started=time.monotonic()
    binding=dict(schema='rinkspace.current_native_whole_print_RGB_capture.v1',prepared=str(args.prepared),state=str(args.state),
        state_sha256=sha(args.state),native_library=str(library),native_library_sha256=library_sha,
        source_atlases=str(args.atlases),source_atlases_sha256=sha(args.atlases),atlas_ids=args.atlas_ids,
        patterns=[dict(atlas_index=p['atlas_index'],instance=p['instance'],path_ids=p['path_ids'],
            reference_image=p['atlas']['image'],reference_image_sha256=p['atlas']['image_sha256'],
            ink_polarity=p['atlas'].get('ink_polarity','light'),
            excluded_partial_reference_paths=p['atlas'].get('excluded_partial_reference_paths',[]),
            reference_photometric_transform=p['atlas'].get('photometric_transform'))for p in patterns],
        adapter_sha256=sha(__file__),reader_sha256=sha(PrintPattern.read_paths.__code__.co_filename),
        source_adapter_sha256=sha(iter_registration_source_frames.__code__.co_filename),
        query_spacing_px=args.query_spacing,retained_arc_spacing_px=48.,shots=args.shots,selected_pts=args.pts,
        prepared_inputs={name:file_ref(args.prepared/name)for name in ('joint_manifest.json','mapping.json','point_observations.npz')},
        coarse_navigation=getattr(args,'coarse_navigation',False),
        coarse_navigation_helper=file_ref(coarse_print_registration.__code__.co_filename),
        native_coordinates_are_search_only=True,panel_success_does_not_skip_glyphs=True,fit_eligible=False,fit_executed=False)
    if 'point_input_sha256' in state and str(state['point_input_sha256'])!=manifest['point_input_sha256']:
        raise ValueError('Current state/prepared point identity differs')
    if binding['prepared_inputs']['point_observations.npz']['sha256']!=manifest['point_input_sha256']:
        raise ValueError('Current prepared point input hash differs')
    if retry_root is not None:
        binding['retry_capture']=validate_retry_capture(retry_root,binding)
        binding['retry_requested_atlas_ids']=args.atlas_ids
        binding['atlas_ids']=retry_original_binding['atlas_ids']
        binding['patterns']=retry_original_binding['patterns']
    output.mkdir(parents=True)
    write(output/'binding.json',binding);summaries={}
    for sid in args.shots:
        shot=next(s for s in manifest['shots'] if s['id']==sid);folder=output/sid;folder.mkdir();counts=Counter();coverage=[]
        with (folder/'source_rows.jsonl').open('x',encoding='utf8',buffering=1)as rows_out,(folder/'registration.jsonl').open('x',encoding='utf8',buffering=1)as reg_out:
            actual=iter(iter_registration_source_frames(sid,bootstrap_root=args.bootstrap,selected_pts=args.pts))
            if retry_root is None:
                entries=((f,None,None,None)for f in actual)
            else:
                def retry_entries():
                    frame=next(actual,None)
                    for source,registration,number in iter_retry_records(retry_root,sid,shot):
                        if frame is not None and int(frame['pts'])<source['pts']:
                            raise ValueError('Retry actual exposure is absent from original ledger')
                        current=frame if frame is not None and int(frame['pts'])==source['pts'] else None
                        if current is not None:
                            validate_retry_frame(current,source)
                            yield current,source,registration,number
                            frame=next(actual,None)
                        else:
                            if args.pts is None:raise ValueError('Retry current source exposure is missing')
                            yield None,source,registration,number
                    if frame is not None:raise ValueError('Extra current retry source exposure')
                entries=retry_entries()
            for fi,(frame,original_source,original_registration,original_line)in enumerate(entries):
                pt=int(frame['pts']if frame is not None else original_source['pts'])
                hypotheses=[];notes=copy.deepcopy(original_registration['patterns'])if original_registration is not None else []
                previous={pattern_key(n):n for n in notes};positions={pattern_key(n):i for i,n in enumerate(notes)}
                projected=None
                if frame is not None:
                    image=frame['image'];exclusion=frame['source_exclusion_mask'].astype(np.uint8)*255
                else:
                    retained,unthinned,choices=choose_readouts([],original_source,original_registration)
                    record=dict(original_source);record['rows']=retained
                    registration_record=dict(original_registration,path_choices=choices,original_capture_line=original_line)
                    rows_out.write(json.dumps(plain(record),ensure_ascii=False,allow_nan=False)+'\n')
                    reg_out.write(json.dumps(plain(registration_record),ensure_ascii=False,allow_nan=False)+'\n')
                    counts['frames']+=1;counts['source_rows']+=len(retained);counts['frames_with_rows']+=bool(retained)
                    counts['original_exposures_copied_without_retry']+=1
                    coverage.append(dict(pts=pt,rows=len(retained),paths=sorted(choices)))
                    continue
                source_binding=dict(source_sha256=frame['source_video_sha256'],source_pts=pt,time_base='1/12800',shot_id=sid,
                    event_id='ankara_2026',rig_id=str(shot['camera_index']),installation_id=str(shot['camera_index']),pixel_domain='native_distorted',width=1920,height=1080)
                for item in patterns:
                    key=pattern_key(item)
                    if retry_root is not None:
                        prior=previous.get(key)
                        if prior is None:counts['no_original_pattern_initialization']+=1;continue
                        if not needs_retry(prior):counts['successful_original_patterns_reused']+=1
                        initial=np.asarray(prior.get('initial_H'),float)
                        if initial.shape!=(3,3)or not np.isfinite(initial).all():
                            raise ValueError('Original retry pattern initial_H is invalid')
                    else:
                        if projected is None:
                            clock=(pt-shot['reference_pts'])/12800+timeline.offset(shot)
                            projected=core.project_motion(state['rig'][shot['camera_index']],state['controls'],timeline.first,timeline.dt,1080,clock,world)
                        uv=[projected['uv'][a:b] for a,b in item['slices']];status=[projected['status'][a:b]for a,b in item['slices']]
                        all_uv=np.vstack(uv)
                        if any(s.any()for s in status)or not np.isfinite(all_uv).all():counts['outside_or_invalid_native_search']+=1;continue
                        if (all_uv.max(0)<[0,0]).any()or(all_uv.min(0)>[1919,1079]).any():counts['outside_or_invalid_native_search']+=1;continue
                        targets=np.vstack([bbox_corners(q)for q in uv]);initial=np.eye(3)
                        initial[:2]=np.linalg.lstsq(item['source_boxes'],targets,rcond=None)[0].T
                        if not np.isfinite(initial).all()or np.linalg.det(initial[:2,:2])<=0:continue
                    note,candidate=read_pattern_attempt(item,image,exclusion,initial,source_binding,args,
                        'retry'if retry_root is not None else 'current_source',
                        original_registration=prior['registration']if retry_root is not None else None,
                        original_had_readouts=(prior.get('source_readout',{}).get('accepted',0)>0
                            or any(row.get('source_registration_label')==f"{item['atlas_index']}/{item['instance']}"
                                for row in original_registration['all_actual_readouts']))if retry_root is not None else False)
                    if retry_root is not None:
                        note['original_pattern_result']=prior
                        note['original_capture_line']=original_line
                        notes[positions[key]]=note;counts['patterns_retried']+=1
                    else:notes.append(note)
                    counts[note['registration']['status']]+=1
                    if candidate['rows']:hypotheses.append(candidate)
                retained,unthinned,choices=choose_readouts(hypotheses,original_source,original_registration)
                record=dict(pts=pt,source_BGR_sha256=frame['source_BGR_sha256'],source_video_sha256=frame['source_video_sha256'],
                    mask_path=str(frame['mask_path']),mask_sha256=frame['mask_sha256'],mask_sources=frame['mask_sources'],
                    source_exclusion_sha256=frame['source_exclusion_sha256'],rows=retained,fit_eligible=False,
                    review_state='whole_pattern_RGB_candidates_pending_source_review',original_RGB_measurements=True,model_UV_synthesized=False)
                if original_source is not None:record=dict(original_source,rows=retained)
                registration_record=dict(pts=pt,patterns=notes,all_actual_readouts=unthinned,path_choices=choices)
                if original_line is not None:registration_record['original_capture_line']=original_line
                rows_out.write(json.dumps(plain(record),ensure_ascii=False,allow_nan=False)+'\n')
                reg_out.write(json.dumps(plain(registration_record),ensure_ascii=False,allow_nan=False)+'\n')
                counts['frames']+=1;counts['source_rows']+=len(retained);counts['frames_with_rows']+=bool(retained)
                coverage.append(dict(pts=pt,rows=len(retained),paths=sorted(choices)))
                if args.pts or fi%250==0:
                    proof=image.copy()
                    for row in retained:cv2.circle(proof,tuple(np.rint(row['uv']).astype(int)),2,(0,0,255),-1)
                    cv2.imwrite(str(folder/f'{pt}_source.png'),image);cv2.imwrite(str(folder/f'{pt}_glyphs.png'),proof)
                if args.pts or fi%25==0:
                    print(json.dumps(dict(shot=sid,frames=fi+1,pts=pt,current_rows=len(retained),total_rows=counts['source_rows'],seconds=time.monotonic()-started)),flush=True)
        summaries[sid]=dict(counts);write(folder/'coverage.json',dict(counts=dict(counts),frames=coverage))
    if retry_root is not None:
        references=[binding['retry_capture']['binding']]
        references.extend(ref for files in binding['retry_capture']['shots'].values()for ref in files.values())
        if any(sha(ref['path'])!=ref['sha256']for ref in references):raise ValueError('Original retry capture changed during recovery')
    write(output/'summary.json',dict(shots=summaries,seconds=time.monotonic()-started,fit_executed=False,fit_eligible=False))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['prepared','state','bootstrap','atlases','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--atlas-ids',nargs='+',type=int,required=True);p.add_argument('--shots',nargs='+',required=True)
    p.add_argument('--pts',nargs='+',type=int);p.add_argument('--query-spacing',type=float,default=8.)
    p.add_argument('--include-paired',action='store_true')
    p.add_argument('--coarse-navigation',action='store_true',help='Search a bounded source template ROI before the unchanged registration gate.')
    p.add_argument('--retry-capture',type=Path,help='Original capture root containing binding.json and per-shot JSONL ledgers.')
    run(p.parse_args())


if __name__=='__main__':main()
