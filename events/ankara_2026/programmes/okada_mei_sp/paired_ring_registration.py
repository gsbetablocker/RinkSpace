"""Programme adapter for whole coloured annuli, retaining measured two-side UV."""
import hashlib
import json
import time
from pathlib import Path
import cv2
import numpy as np
from rinkspace.core.native import NativeCore
from rinkspace.domain.parameters import SceneParameterModel, ParameterState
from rinkspace.domain.multi_shot_motion import ShotMotionTimeline
from rinkspace.compat.d78_entities import sample_entities
from rinkspace.session.material_ring_bands import read_coloured_ring_band


def run(args):
    import run_native_programme as programme
    from rinkspace.session import material_ring_bands, material_rings
    from rinkspace import vision_programme_ice_review
    cv2.setNumThreads(2)
    root=args.run_root
    if args.state is None or args.output is None:raise ValueError('Explicit frozen search state and new output required')
    manifest=programme.read(root/'joint_manifest.json');values=dict(np.load(args.state))
    scene_path=Path(manifest['scene_path']);scene=programme.read(scene_path)
    model=SceneParameterModel(scene,scene_path,optimize_scene=True)
    state=ParameterState(model.initial_state.layout,values['scene_values'])
    core=NativeCore(manifest['native_library']);samples=sample_entities(state.apply_scene(scene),core)
    timeline=ShotMotionTimeline(manifest['shots'],manifest['control_dt'],manifest['first_control_time'])
    house=args.ring_house
    material_signal=getattr(args,'ring_signal','red')
    source_roles=getattr(args,'ring_roles',['middle','outer'])
    refine_source_origin=getattr(args,'refine_source_origin',False)
    if house not in ('F2','R4'):raise ValueError('No current source binding for the requested annulus')
    cuts={c['label'].split(':')[-1]:c for c in samples['cuts'] if c['label'].startswith('ice.'+house+':')}
    middle=samples['xyz'][cuts['middle']['begin']:cuts['middle']['end']]
    outer=samples['xyz'][cuts['outer']['begin']:cuts['outer']['end']]
    if middle.shape!=outer.shape:raise ValueError('Ring seed sampling differs')
    guide=(middle+outer)*.5;center=guide[:-1].mean(0)
    if house=='F2':
        protected=root/'source_material_checks/protected_manual_sources.json'
        human=[r for r in programme.read(protected)['protected_human_ranges']
            if r.get('current_path_correspondence')==cuts['outer']['id']]
        if not human:raise ValueError('No preserved physical ring identity reference')
        scope_ids=[r['scope_id'] for r in human]
        identity_basis='preserved_human_outer_ring_identity_and_actual_current_red_annulus'
    else:
        if args.shots!=['s03']:raise ValueError('Current R4 cross-view identity binding belongs to S03')
        protected=root/'registration/ring_materials_v1/linked_cross_view_scaled/s03/complete_stack_consensus.json'
        linked=programme.read(protected)['qualified']
        if not any(r['identity']==house+':outer' for r in linked):raise ValueError('No current cross-view ring identity')
        scope_ids=[];identity_basis='complete_cross_view_source_stack_with_actual_RGB_correspondence_not_human_acceptance'
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    identity=dict(source_state=str(args.state),source_state_sha256=programme.digest(args.state),
        source_reference=str(protected),source_reference_sha256=programme.digest(protected),
        human_scope_ids=scope_ids,physical_identity_basis=identity_basis,house=house,
        material_family='observed_red_annulus_between_middle_and_outer',
        algorithm_sha256={p:programme.digest(p) for p in [__file__,material_ring_bands.__file__,material_rings.__file__,vision_programme_ice_review.__file__]},
        search_radius_px=160.,source_UV_from_RGB_only=True,model_UV_as_observation=False,
        material_signal=material_signal,source_roles=source_roles,refine_source_origin=refine_source_origin)
    if (out/'identity.json').exists() and programme.read(out/'identity.json')!=identity:raise ValueError('Ring-band source cache changed')
    programme.write_json(out/'identity.json',identity)
    logpath=out/'read.jsonl';print('MAIN LOG: '+str(logpath),flush=True);started=time.monotonic()
    with logpath.open('a',encoding='utf-8',buffering=1) as log:
        for shot_id in args.shots:
            visibility,frames=programme.source_frames(shot_id)
            source_context=dict(source_sha256=visibility['source_video_sha256'],time_base='1/12800',
                authority=[dict(path=str(protected),sha256=identity['source_reference_sha256'])],
                human_scope_ids=identity['human_scope_ids'],prepared_inventory_semantics=identity['material_family'])
            if args.reference_pts is not None:frames=[f for f in frames if f['pts']==args.reference_pts]
            shot=next(s for s in manifest['shots'] if s['id']==shot_id);folder=out/shot_id;folder.mkdir(exist_ok=True);rows=[]
            for fi,frame in enumerate(frames):
                target=folder/f'{frame["pts"]}.json'
                if target.exists():record=programme.read(target)
                else:
                    image=cv2.imread(frame['image'])
                    if hashlib.sha256(image.tobytes()).hexdigest()!=frame['source_BGR_sha256']:raise ValueError('Source pixels changed')
                    masks,_=programme.surfaces(frame,image)
                    clock=(frame['pts']-shot['reference_pts'])/12800+timeline.offset(shot)
                    def project(x):return core.project_motion(values['rig'][shot['camera_index']],values['controls'],timeline.first,timeline.dt,1080,clock,x)
                    projected=project(guide);pc=project(center[None]);projected['valid']=projected['status']==0
                    measured,result=read_coloured_ring_band(cv2.cvtColor(image,cv2.COLOR_BGR2RGB),~masks['ice'],projected,
                        source_sha256=visibility['source_video_sha256'],pts=frame['pts'],center_uv=pc['uv'][0],
                        prepared_source_context=source_context,material_signal=material_signal,
                        refine_source_origin=refine_source_origin)
                    arrays={k:v for k,v in measured.items() if isinstance(v,np.ndarray)}
                    metadata={k:v for k,v in measured.items() if not isinstance(v,np.ndarray)}
                    arrays['metadata_json']=np.array(json.dumps(metadata,default=lambda x:x.tolist()))
                    np.savez_compressed(folder/f'{frame["pts"]}.npz',**arrays)
                    candidates=[];proof=image.copy()
                    for cid in result['selected']:
                        for side,role in enumerate(('middle','outer')):
                            if role not in source_roles:continue
                            quantiles=measured['candidate_side_quantiles_uv'][cid,side]
                            uv=quantiles[1]
                            candidates.append(dict(pts=frame['pts'],identity=f'{house}:{role}',path_id=cuts[role]['id'],candidate=cid,
                                uv=uv.tolist(),actual_source_quantiles=quantiles.tolist(),paired_band_candidate=cid,
                                evidence_image=frame['image'],source_BGR_sha256=frame['source_BGR_sha256'],
                                source_reference=str(protected),measured_band_pair=True,model_UV_synthesized=False))
                            cv2.circle(proof,tuple(np.rint(uv).astype(int)),3,(0,255,0) if side==0 else (0,255,255),-1)
                    record=dict(pts=frame['pts'],evidence_image=frame['image'],source_BGR_sha256=frame['source_BGR_sha256'],
                        source_family=result,candidates=candidates,user_visual_acceptance=False,fit_observations_changed=False)
                    programme.write_json(target,record)
                    if args.reference_pts is not None:cv2.imwrite(str(folder/f'{frame["pts"]}_source_proof.jpg'),np.vstack([image,proof]))
                rows.extend(record['candidates'])
                if fi%50==0 or fi==len(frames)-1:
                    entry=dict(shot=shot_id,frames=fi+1,total=len(frames),source_rows=len(rows),supported_frames=len({r['pts'] for r in rows}),elapsed_s=time.monotonic()-started)
                    line=json.dumps(entry);log.write(line+'\n');print(line,flush=True)
            programme.write_json(folder/'source_candidates.json',dict(candidates=rows,identity=identity,fit_observations_changed=False,user_visual_acceptance=False))


def link(args):
    """Qualify the paired source samples by actual adjacent-exposure RGB."""
    import run_native_programme as programme
    from rinkspace.session import material_tracks
    cv2.setNumThreads(2)
    root=args.run_root;out=args.output
    if out is None or args.state is None:raise ValueError('Explicit source packet and frozen search geometry required')
    manifest=programme.read(root/'joint_manifest.json');values=dict(np.load(args.state))
    core=NativeCore(manifest['native_library'])
    timeline=ShotMotionTimeline(manifest['shots'],manifest['control_dt'],manifest['first_control_time'])
    xx,yy=np.meshgrid(np.linspace(-.52,.52,81),np.linspace(-.27,.27,45))
    world=np.c_[xx.ravel(),yy.ravel(),np.zeros(xx.size)]
    identity=programme.read(out/'identity.json')
    inventory=[dict(operator='paint_center',material_signal=identity.get('material_signal','red'),material_transition=None)]
    for shot_id in args.shots:
        folder=out/shot_id;packet=folder/'source_candidates.json';data=programme.read(packet)
        target=folder/'temporal_source_verified.json'
        if target.exists():raise FileExistsError('Preserve completed paired-source temporal qualification')
        _,frames=programme.source_frames(shot_id)
        shot=next(s for s in manifest['shots'] if s['id']==shot_id)
        rows={f['pts']:[] for f in frames}
        for row in data['candidates']:rows[row['pts']].append(row)
        supported=set();edges=[];previous=None
        for frame in frames:
            pts=frame['pts'];local=rows[pts]
            if not local:previous=None;continue
            measured=dict(np.load(folder/f'{pts}.npz',allow_pickle=False))
            ids=sorted({r['candidate'] for r in local})
            image=cv2.imread(frame['image']);masks,_=programme.surfaces(frame,image)
            if hashlib.sha256(image.tobytes()).hexdigest()!=frame['source_BGR_sha256']:raise ValueError('Source image changed')
            rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            nodes=material_tracks.source_nodes(rgb,measured,[dict(curve=0,candidate_ids=ids)],inventory)
            clock=(pts-shot['reference_pts'])/12800+timeline.offset(shot)
            projection=core.project_motion(values['rig'][shot['camera_index']],values['controls'],timeline.first,timeline.dt,1080,clock,world)
            now=(pts,rgb,nodes,~masks['ice'],projection)
            if previous is not None:
                old_pts,old_rgb,old,old_mask,pa=previous;pb=projection
                valid=(pa['status']==0)&(pb['status']==0)&np.isfinite(pa['uv']).all(1)&np.isfinite(pb['uv']).all(1)
                for projected,points in [(pa,old),(pb,nodes)]:
                    lo=points['uv'].min(0)-120;hi=points['uv'].max(0)+120
                    valid&=(projected['uv']>=lo).all(1)&(projected['uv']<=hi).all(1)
                if valid.sum()>=12:
                    h,_=cv2.findHomography(pa['uv'][valid],pb['uv'][valid],0)
                    linked=material_tracks.link_fragments(old_rgb,rgb,old,nodes,old_mask,~masks['ice'],h)
                    for edge in linked:
                        if not edge['unambiguous']:continue
                        pairs=[]
                        for a,b in edge['pairs']:
                            ca,cb=int(old['ids'][a]),int(nodes['ids'][b]);pairs.append([ca,cb])
                            supported.update([(old_pts,ca),(pts,cb)])
                        edges.append(dict(from_pts=old_pts,to_pts=pts,source_candidate_pairs=pairs,
                            matching=edge,physical_identity_is_assistant_inference=True))
            previous=now
        accepted=[r for r in data['candidates'] if (r['pts'],r['candidate']) in supported]
        graph=folder/'temporal_links.json';programme.write_json(graph,dict(edges=edges,source_packet_sha256=programme.digest(packet),
            matcher_sha256=programme.digest(material_tracks.__file__),state_sha256=programme.digest(args.state)))
        result=dict(shot=shot_id,accepted=accepted,pending_rows=len(data['candidates'])-len(accepted),
            source_packet=str(packet),source_packet_sha256=programme.digest(packet),source_links=str(graph),source_links_sha256=programme.digest(graph),
            qualification='actual_unique_bidirectional_RGB_between_adjacent_exposures_of_the_paired_source_annulus',
            fit_observations_changed=False,user_visual_acceptance=False,source_UV_synthesized=False)
        programme.write_json(target,result)
        print(json.dumps(dict(shot=shot_id,candidate_rows=len(data['candidates']),qualified_rows=len(accepted),
            qualified_frames=len({r['pts'] for r in accepted}),temporal_links=len(edges))),flush=True)


def assimilate(args):
    """Apply one whole-family source revision, keeping every prior input row."""
    import os
    import run_native_programme as programme
    import apply_native_multicamera_review as review
    root=args.run_root
    if args.state is None or args.output is None:raise ValueError('Explicit parent state and revision output required')
    out=args.output
    if out.exists():raise FileExistsError('Preserve the completed or interrupted source revision')
    manifest=programme.read(root/'joint_manifest.json');mapping=programme.read(root/'mapping.json')
    parent_hash=programme.digest(root/'point_observations.npz')
    if parent_hash!=manifest['point_input_sha256']:raise ValueError('Current observation input changed')
    with np.load(root/'point_observations.npz',allow_pickle=False) as z:old=dict(z)
    packets=getattr(args,'source_packets',None)
    superseded=getattr(args,'replace_source_prefixes',None)
    superseded_paths=getattr(args,'replace_source_paths',None)
    append_only=bool(getattr(args,'append_source_only',False))
    prefix=getattr(args,'source_observation_prefix',None)
    if not packets or not prefix or (not superseded and not append_only):
        raise ValueError('Source packets, a new prefix and explicit append or replacement mode are required')
    if append_only and (superseded or superseded_paths):
        raise ValueError('Append-only source admission cannot retire existing observations')
    superseded=superseded or []
    if any(prefix.startswith(p.rstrip(':')) for p in superseded):
        raise ValueError('The new source version must have a distinct observation prefix')
    if superseded_paths and not set(superseded_paths)<=set(mapping['path_ids']):
        raise ValueError('Unknown source paths in the explicit replacement scope')
    additions=[];sources=[];measured_pixels={};duplicate_queries=0
    for packet in packets:
        data=programme.read(packet)
        if data.get('fit_eligible') is False:raise ValueError('Source packet failed its material-role review: '+str(packet))
        if not data['accepted']:raise ValueError('No qualified source samples in '+str(packet))
        for name in ('source_packet','source_links'):
            if programme.digest(Path(data[name]))!=data[name+'_sha256']:
                raise ValueError('Source qualification dependency changed: '+name)
        sources.append(dict(path=str(packet),sha256=programme.digest(packet),rows=len(data['accepted'])))
        for row in data['accepted']:
            key=(row['pts'],row['path_id'],*map(float,row['uv']))
            if key in measured_pixels:
                additions[measured_pixels[key]].setdefault('same_pixel_source_queries',[]).append(row['candidate'])
                duplicate_queries+=1
                continue
            measured_pixels[key]=len(additions)
            additions.append(dict(row,observation_id=f'{prefix}:{data["shot"]}:{row["pts"]}:{mapping["path_ids"].index(row["path_id"])}:{row["candidate"]}',kind=1,
                reason='Original RGB material readout with source identity and bidirectional RGB evidence in the referenced packet. Common geometry validation candidate; not new human pixel acceptance.'))
    images={}
    for shot in {programme.read(packet)['shot'] for packet in packets}:
        _,frames=programme.source_frames(shot);images.update({f['pts']:f['image'] for f in frames})
    decisions=[];retired_indices=[]
    for i,identity in enumerate(old['observation_ids']):
        path=mapping['path_ids'][old['path_indices'][i]]
        if any(str(identity).startswith(p) for p in superseded) and (not superseded_paths or path in superseded_paths):
            retired_indices.append(i)
            decisions.append(dict(observation_id=str(identity),pts=int(old['pts'][i]),expected_path_id=path,uv=old['uv'][i].tolist(),
                decision='exclude_preserving_source',evidence_image=images[int(old['pts'][i])],
                reason='Superseded the explicitly selected source version and material families with fresh source readouts. Original measurements remain in the parent input; no per-point residual threshold used.'))
    if not decisions and not append_only:raise ValueError('No existing observations matched the explicit superseded source prefixes')
    out.mkdir(parents=True)
    os.link(root/'point_observations.npz',out/'parent_point_observations.npz')
    programme.write_json(out/'parent_joint_manifest.json',manifest);programme.write_json(out/'parent_mapping.json',mapping)
    report=dict(input_sha256=parent_hash,decisions=decisions,added_observations=additions,source_packets=sources,
        superseded_observation_prefixes=superseded,new_observation_prefix=prefix,
        append_only=append_only,
        superseded_path_ids=superseded_paths,
        identical_source_pixel_duplicates_collapsed=duplicate_queries,
        source_selection_uses_native_residual=False,source_UV_synthesized=False,old_four_shot_rows_unchanged=True,
        original_human_scopes_preserved=True,user_visual_acceptance=False)
    path=out/'source_revision.json';programme.write_json(path,report)
    review.p.RUN=root;seed=out/'seed.npz';review.apply(path,args.state,seed,out/'receipt')
    current=dict(np.load(root/'point_observations.npz',allow_pickle=False))
    retained=np.ones(len(old['pts']),bool);retained[retired_indices]=False
    current_retained=~np.char.startswith(current['observation_ids'],prefix+':')
    retained_checks={k:np.array_equal(old[k][retained],current[k][current_retained]) for k in old}
    if not all(retained_checks.values()):raise ValueError('An unrelated source observation changed')
    after=programme.read(root/'joint_manifest.json');preservation={}
    for shot in manifest['shots']:
        if shot['id'] not in ('s01','s02','s04','s06'):continue
        new=next(s for s in after['shots'] if s['id']==shot['id'])
        checks={k:np.array_equal(old[k][shot['row_begin']:shot['row_stop']],current[k][new['row_begin']:new['row_stop']]) for k in old}
        if not all(checks.values()):raise ValueError('An existing four-shot observation changed')
        preservation[shot['id']]=checks
    receipt=dict(parent_input_sha256=parent_hash,current_input_sha256=programme.digest(root/'point_observations.npz'),
        retired_family_rows=len(decisions),new_original_RGB_rows=len(additions),current_rows=len(current['pts']),
        all_retained_observations_bit_identical=retained_checks,
        old_four_shots_bit_identical=preservation,seed=str(seed),numerical_validation_pending=True,visual_acceptance=False)
    programme.write_json(out/'preservation.json',receipt);print(json.dumps(receipt),flush=True)
