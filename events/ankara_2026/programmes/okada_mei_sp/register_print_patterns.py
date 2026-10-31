"""Whole-print source registration in the unified programme workflow."""
from pathlib import Path
import collections,json,time
import cv2,numpy as np
import prepare_native_joint_shots as p
from rinkspace.domain.multi_shot_motion import ShotMotionTimeline
from rinkspace.core.gpu import ResidentResidualProblem
from rinkspace.review.scene import sample_scene
from rinkspace.session.print_patterns import PrintPattern


def run(args):
    cv2.setNumThreads(2)
    import run_native_programme as programme
    if args.state is None or args.output is None or args.reference_run is None or args.print_atlas_id is None:
        raise ValueError('State, new output, reference-run and print-atlas-id are required')
    root=args.run_root;out=args.output
    atlas_path=args.reference_run/'atlases.json';atlas=programme.read(atlas_path)['atlases'][args.print_atlas_id]
    if atlas.get('partial_source_reference'):raise ValueError('A fragmented local label cannot define the complete pattern')
    if programme.digest(atlas['image'])!=atlas['image_sha256']:raise ValueError('Reference pixels changed')
    manifest=programme.read(root/'joint_manifest.json');state=p.npz(args.state)
    core=p.NativeCore(manifest['native_library']);raw=programme.read(manifest['scene_path'])
    model=p.SceneParameterModel(raw,Path(manifest['scene_path']),optimize_scene=True)
    parameter_state=p.ParameterState(model.initial_state.layout,state['scene_values'])
    native=p.NativeSceneGPU(manifest['scene_path'],core,parameter_state=parameter_state)
    timeline=ShotMotionTimeline(manifest['shots'],manifest['control_dt'],manifest['first_control_time'])
    sampled=sample_scene(raw,core,Path(manifest['scene_path']),parameter_state=parameter_state)
    by_cut={c['id']:sampled['xyz'][c['begin']:c['end']] for c in sampled['cuts']}
    pattern=PrintPattern(atlas,cv2.imread(atlas['image']));world=[];cuts=[];source_boxes=[]
    for path in atlas['paths']:
        native_id='print_instance/'+path['path_id'].split('.')[0]+'::'+path['path_id']
        v=by_cut[native_id];cuts.append((len(world),len(world)+len(v)));world.extend(v)
        q=np.array(path['pixels']);lo,hi=q.min(0),q.max(0);source_boxes.extend([lo,[hi[0],lo[1]],hi,[lo[0],hi[1]]])
    world=np.array(world);source_boxes=np.c_[source_boxes,np.ones(len(source_boxes))]
    receipt=dict(reference_atlas=atlas_path.as_posix(),reference_atlas_sha256=programme.digest(atlas_path),
        reference_index=args.print_atlas_id,reference_image=atlas['image'],reference_image_sha256=atlas['image_sha256'],
        numeric_search_state=str(args.state),numeric_search_state_sha256=programme.digest(args.state),
        input_sha256=programme.digest(root/'point_observations.npz'),scene_sha256=programme.digest(manifest['scene_path']),
        shared_implementation_sha256=programme.digest(Path(PrintPattern.register.__code__.co_filename)),
        adapter_sha256=programme.digest(Path(__file__)),shots=args.shots,model_used_only_for_search=True,source_UV_from_current_RGB=True)
    out.mkdir(parents=True,exist_ok=True)
    if (out/'receipt.json').exists() and programme.read(out/'receipt.json')!=receipt:raise ValueError('Preserve existing revision; dependencies changed')
    programme.write_json(out/'receipt.json',receipt);programme.write_json(out/'reference.json',atlas)
    started=time.monotonic();totals={}
    with (out/'progress.jsonl').open('a',encoding='utf8',buffering=1) as log:
        for shotid in args.shots:
            shot=next(s for s in manifest['shots'] if s['id']==shotid)
            visibility=programme.DATA/'runs'/f'native_points_{shotid}_20261028/evidence/ad_visibility.json'
            source=programme.read(visibility);frames=source['frames']
            if [f['pts'] for f in frames]!=list(range(shot['first_pts'],shot['last_pts']+1,256)):raise ValueError('Incomplete source sequence')
            folder=out/shotid;folder.mkdir(exist_ok=True);rig=state['rig'][shot['camera_index']];coverage=[];accepted=[];counts=collections.Counter()
            for fi,frame in enumerate(frames):
                pt=frame['pts'];record_path=folder/f'{pt}.json'
                if record_path.exists():record=programme.read(record_path)
                else:
                    t=(pt-shot['reference_pts'])/12800+timeline.offset(shot)
                    with ResidentResidualProblem(core,board=native.scene['board'],layout=native.scene['shared_layout'],options=native.scene['shared_geometry_options'],rig=rig,controls=state['controls'],first_control_time=timeline.first,control_dt=timeline.dt,image_height=1080,codes=np.zeros(len(world),np.int32),stations=np.linspace(0,1,len(world)),times=np.full(len(world),t),observed=np.zeros((len(world),2))) as resident:
                        resident.configure_geometry([dict(kind=4,vertices=world,parameters=[])]);resident.configure_projection_time_support(half_seconds=manifest['max_readout']/2);result=resident.evaluate('cuda')
                    uv=result['values'][:,3:5];target_boxes=[]
                    for begin,end in cuts:
                        q=uv[begin:end];lo,hi=q.min(0),q.max(0);target_boxes.extend([lo,[hi[0],lo[1]],hi,[lo[0],hi[1]]])
                    initial=np.eye(3);initial[:2]=np.linalg.lstsq(source_boxes,target_boxes,rcond=None)[0].T
                    if (uv.max(0)<[0,0]).any() or (uv.min(0)>[1920,1080]).any():
                        reg=dict(status='search_region_outside_image');readouts=[];summary={}
                    elif result['status'].any():
                        reg=dict(status='native_search_projection_failed');readouts=[];summary={}
                    else:
                        image=cv2.imread(frame['image']);mask=cv2.imread(frame['mask_path'],0)
                        if programme.hashlib.sha256(image.tobytes()).hexdigest()!=frame['source_BGR_sha256']:raise ValueError('Current source pixels changed')
                        if programme.digest(frame['mask_path'])!=frame['mask_sha256']:raise ValueError('Current source exclusion changed')
                        reg=pattern.register(image,mask,initial)
                        binding=dict(source_sha256=frame['source_video_sha256'],source_pts=pt,time_base='1/12800',shot_id=shotid,event_id='ankara_2026',rig_id=str(shot['camera_index']),installation_id=str(shot['camera_index']),pixel_domain='native_distorted',width=1920,height=1080)
                        readouts,summary=pattern.read_paths(image,mask,reg,binding)
                        for row in readouts:row.update(pts=pt,identity=row['path_id'],evidence_image=frame['image'],source_BGR_sha256=frame['source_BGR_sha256'],registration_record=str(record_path))
                    record=dict(shot=shotid,pts=pt,source_seconds=pt/12800,source_image=frame['image'],source_BGR_sha256=frame['source_BGR_sha256'],initial_search_H=initial.tolist(),registration=reg,readout_summary=summary,readouts=readouts)
                    programme.write_json(record_path,record)
                rows=record['readouts'];accepted.extend(rows);counts[record['registration']['status']]+=1
                coverage.append(dict(pts=pt,status=record['registration']['status'],source_readouts=len(rows),entities=sorted({r['path_id'] for r in rows}),correlation=record['registration'].get('correlation')))
                if fi%25==0 or fi==len(frames)-1:
                    message=dict(shot=shotid,exposures=fi+1,total=len(frames),source_readouts=len(accepted),seconds=round(time.monotonic()-started,1))
                    log.write(json.dumps(message)+'\n');print(json.dumps(message),flush=True)
            programme.write_json(folder/'coverage.json',dict(shot=shotid,frames=coverage,status_counts=dict(counts)))
            report=dict(shot=shotid,accepted=accepted,frames=len(frames),source_packet=str(out/'reference.json'),source_packet_sha256=programme.digest(out/'reference.json'),source_links=str(folder/'coverage.json'),source_links_sha256=programme.digest(folder/'coverage.json'),native_fit_residual_not_used_for_selection=True,fit_eligible=False,review_state='awaiting_source_review',user_visual_acceptance=False)
            programme.write_json(folder/'source_verified.json',report)
            totals[shotid]=dict(frames=len(frames),registered_frames=counts['source_pattern_registered'],source_readout_frames=sum(f['source_readouts']>0 for f in coverage),source_rows=len(accepted),status_counts=dict(counts))
    programme.write_json(out/'summary.json',dict(shots=totals,fit_input_changed=False,complete_source_sequences=True,seconds=time.monotonic()-started))
    print(json.dumps(totals),flush=True)


def reread(args):
    """Reuse frozen full-image registrations when only source readout changes."""
    import run_native_programme as programme
    import inspect,hashlib
    cv2.setNumThreads(2)
    root=args.reference_directory;out=args.output
    if root is None or out is None or out.exists():raise ValueError('Frozen reference directory and new output required')
    atlas=programme.read(root/'reference.json');pattern=PrintPattern(atlas,cv2.imread(atlas['image']))
    if programme.digest(atlas['image'])!=atlas['image_sha256']:raise ValueError('Reference image changed')
    receipt=dict(parent_registration_receipt=str(root/'receipt.json'),parent_registration_receipt_sha256=programme.digest(root/'receipt.json'),
        source_reader_sha256=hashlib.sha256((inspect.getsource(PrintPattern.read_paths)+inspect.getsource(PrintPattern.resolved_glyphs)).encode()).hexdigest(),
        shared_module_sha256=programme.digest(Path(PrintPattern.read_paths.__code__.co_filename)),registration_recomputed=False,fit_input_changed=False)
    out.mkdir(parents=True);programme.write_json(out/'receipt.json',receipt);total={}
    for shot in args.shots:
        original=programme.read(root/shot/'coverage.json');source=programme.read(programme.DATA/'runs'/f'native_points_{shot}_20261028/evidence/ad_visibility.json');frames={f['pts']:f for f in source['frames']}
        accepted=[];coverage=[];links=[];folder=out/shot;folder.mkdir()
        for item in original['frames']:
            pt=item['pts'];path=root/shot/f'{pt}.json';record=programme.read(path);registration=record['registration'];f=frames[pt];rows=[];summary={}
            if registration['status']=='source_pattern_registered':
                image=cv2.imread(f['image']);mask=cv2.imread(f['mask_path'],0)
                if hashlib.sha256(image.tobytes()).hexdigest()!=f['source_BGR_sha256'] or programme.digest(f['mask_path'])!=f['mask_sha256']:raise ValueError('Source pixels or exclusion changed')
                binding=dict(source_sha256=f['source_video_sha256'],source_pts=pt,time_base='1/12800',shot_id=shot,event_id='ankara_2026',rig_id='2',installation_id='2',pixel_domain='native_distorted',width=1920,height=1080)
                rows,summary=pattern.read_paths(image,mask,registration,binding)
                for row in rows:row.update(pts=pt,identity=row['path_id'],evidence_image=f['image'],source_BGR_sha256=f['source_BGR_sha256'],registration_record=str(path))
            accepted.extend(rows);coverage.append(dict(item,source_readouts=len(rows),entities=sorted({r['path_id'] for r in rows}),source_readout_summary=summary))
            links.append(dict(path=str(path),sha256=programme.digest(path),pts=pt,source_BGR_sha256=f['source_BGR_sha256']))
        programme.write_json(folder/'coverage.json',dict(shot=shot,frames=coverage))
        programme.write_json(folder/'source_links.json',dict(registration_records=links,receipt_sha256=programme.digest(out/'receipt.json')))
        report=dict(shot=shot,accepted=accepted,frames=len(coverage),source_packet=str(root/'reference.json'),source_packet_sha256=programme.digest(root/'reference.json'),
            source_links=str(folder/'source_links.json'),source_links_sha256=programme.digest(folder/'source_links.json'),fit_eligible=False,
            native_fit_residual_not_used_for_selection=True,user_visual_acceptance=False,review_state='awaiting_source_review')
        programme.write_json(folder/'source_verified.json',report)
        total[shot]=dict(exposures=len(coverage),source_readouts=len(accepted),source_readout_frames=sum(x['source_readouts']>0 for x in coverage))
        print(json.dumps(dict(shot=shot,**total[shot])),flush=True)
    programme.write_json(out/'summary.json',dict(shots=total,registration_recomputed=False,fit_input_changed=False))


def prepare_revision(args):
    """Preserve verified old UV and supplement missing current print contours."""
    import os
    from scipy.spatial import cKDTree
    import run_native_programme as programme
    import apply_native_multicamera_review as review
    root=args.run_root;source=args.reference_directory;out=args.output
    if source is None or out is None or args.state is None or out.exists():raise ValueError('Source packet, parent state and new revision output required')
    assessment=programme.read(source/'image_review.json')
    if not assessment.get('source_candidate_eligible'):raise ValueError('Current image review has not admitted this source candidate')
    manifest=programme.read(root/'joint_manifest.json');mapping=programme.read(root/'mapping.json');before=programme.digest(root/'point_observations.npz')
    if before!=manifest['point_input_sha256']:raise ValueError('Current input changed')
    original=p.npz(root/'point_observations.npz');protected=programme.read(root/'source_material_checks/protected_manual_sources.json')
    protected_ids=set(protected['protected_formal_observation_ids'])
    decisions=[];additions=[];retained=[];source_receipts=[];removed_indices=[];summary={}
    for shotid in args.shots:
        packet_path=source/shotid/'source_verified.json';packet=programme.read(packet_path)
        if assessment['source_packets_sha256'][shotid]!=programme.digest(packet_path):raise ValueError('Reviewed source packet changed')
        for key in ('source_packet','source_links'):
            if programme.digest(packet[key])!=packet[key+'_sha256']:raise ValueError('Source dependency changed')
        reference=programme.read(packet['source_packet']);names={'print_instance/'+a['path_id'].split('.')[0]+'::'+a['path_id'] for a in reference['paths']}
        path_indices={mapping['path_ids'].index(x) for x in names};groups=collections.defaultdict(list)
        for row in packet['accepted']:groups[(row['pts'],row['path_id'])].append(row)
        trees={key:cKDTree(np.array([x['uv'] for x in rows])) for key,rows in groups.items()}
        old_kept=collections.defaultdict(list);shot=next(s for s in manifest['shots'] if s['id']==shotid)
        rows=np.arange(shot['row_begin'],shot['row_stop']);rows=rows[np.isin(original['path_indices'][rows],list(path_indices))]
        images={f['pts']:f['image'] for f in programme.read(programme.DATA/'runs'/f'native_points_{shotid}_20261028/evidence/ad_visibility.json')['frames']}
        nretain=0;ngenerated=0;nexclude=0
        for i in rows:
            identity=str(original['observation_ids'][i]);pt=int(original['pts'][i]);name=mapping['path_ids'][original['path_indices'][i]];uv=original['uv'][i];key=(pt,name);qualified=False
            if key in trees:
                distance,j=trees[key].query(uv);candidate=groups[key][int(j)];normal=np.array(candidate['normal']);delta=uv-np.array(candidate['search_uv']);offset=float(delta@normal);along=abs(float(delta@np.array([-normal[1],normal[0]])));quantile=candidate['actual_normal_quantiles']
                qualified=along<=2.5 and quantile[0]-.25<=offset<=quantile[2]+.25
            if qualified:
                old_kept[key].append(uv);nretain+=1;retained.append(dict(observation_id=identity,pts=pt,path_id=name,uv=uv.tolist(),source_candidate=candidate['candidate'],actual_material_transition_normal_interval=quantile))
            else:
                if identity in protected_ids:raise ValueError('An explicit human source point requires preservation: '+identity)
                removed_indices.append(int(i));nexclude+=1
                decisions.append(dict(observation_id=identity,pts=pt,expected_path_id=name,uv=uv.tolist(),decision='exclude_preserving_source',evidence_image=images[pt],reason='Whole-print source reassociation did not verify this original labeled contour in the current exposure. Merged ink, occlusion, wrong source label and unresolved correspondence remain outside fitting; original UV and all earlier evidence are archived. No native residual threshold selected this row.'))
        kept_trees={key:cKDTree(np.array(uv)) for key,uv in old_kept.items()};seen=set()
        for key,values in groups.items():
            for row in values:
                if key in kept_trees and kept_trees[key].query(row['uv'])[0]<3:continue
                pixel=(key,*row['uv'])
                if pixel in seen:continue
                seen.add(pixel);ngenerated+=1
                additions.append(dict(row,observation_id=f'whole_print:{shotid}:{row["pts"]}:{mapping["path_ids"].index(row["path_id"])}:{row["candidate"]}',kind=1,
                    reason='Actual current RGB q50 of a source-registered complete print, with separately visible current ink component and closed-hole evidence where applicable. Projected/native coordinates are search only; original source pixels define every measured UV.'))
        source_receipts.append(dict(path=str(packet_path),sha256=programme.digest(packet_path)))
        summary[shotid]=dict(old_family_rows=len(rows),old_UV_retained=nretain,old_unverified_rows_retired=nexclude,new_source_rows=ngenerated)
    if not decisions and not additions:raise ValueError('No source revision to apply')
    out.mkdir(parents=True);os.link(root/'point_observations.npz',out/'parent_point_observations.npz')
    programme.write_json(out/'parent_joint_manifest.json',manifest);programme.write_json(out/'parent_mapping.json',mapping)
    report=dict(input_sha256=before,source_packets=source_receipts,image_review=str(source/'image_review.json'),image_review_sha256=programme.digest(source/'image_review.json'),decisions=decisions,added_observations=additions,retained_original_material_readouts=retained,summary=summary,source_UV_synthesized=False,native_residual_used_for_selection=False,user_visual_acceptance=False)
    programme.write_json(out/'source_revision.json',report);review.p.RUN=root
    review.apply(out/'source_revision.json',args.state,out/'seed.npz',out/'receipt')
    current=p.npz(root/'point_observations.npz');old_kept=np.ones(len(original['pts']),bool);old_kept[removed_indices]=False;new_kept=~np.char.startswith(current['observation_ids'],'whole_print:')
    checks={key:np.array_equal(original[key][old_kept],current[key][new_kept]) for key in original}
    if not all(checks.values()):raise ValueError('A retained source field changed')
    after=programme.read(root/'joint_manifest.json');old_shots={}
    for shot in manifest['shots']:
        if shot['id'] in args.shots:continue
        new=next(s for s in after['shots'] if s['id']==shot['id']);valid={k:np.array_equal(original[k][shot['row_begin']:shot['row_stop']],current[k][new['row_begin']:new['row_stop']]) for k in original}
        if not all(valid.values()):raise ValueError('Unrelated shot changed')
        old_shots[shot['id']]=valid
    preservation=dict(parent_input_sha256=before,current_input_sha256=programme.digest(root/'point_observations.npz'),summary=summary,current_rows=len(current['pts']),all_retained_fields_identical=checks,unrelated_shots_identical=old_shots,human_scopes_unchanged=True,seed=str(out/'seed.npz'),numerical_validation_pending=True,user_visual_acceptance=False)
    programme.write_json(out/'preservation.json',preservation);print(json.dumps(preservation),flush=True)
