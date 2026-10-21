"""Qualified successor board points with explicit bounded numerical navigation.

Source ownership/UV qualification is independent of this initializer. Every
actual ray/grid/refinement trial is retained; one material point is shared by
all native exposures. Frozen source issuers are never edited.
"""
from pathlib import Path
from collections import Counter
import copy,json
import numpy as np
from scipy.optimize import least_squares
from .io_utils import read_json,read_jsonl,write_json,write_jsonl,file_sha256,object_sha256
from . import static_board_tracklet_append as source_helpers
from .static_board_tracklet_append import (_source_contract,_ad_block,_ref,
    load_point_qualification,_verified,OPERATOR,BUNDLE_SCHEMA,GATE_SCHEMA,RECEIPT_SCHEMA)


def _bounded_surface_seed(model,scene,camera,times,uv,adid,tid,navigation):
    from .joint_replay import render_visibility
    evidence=next(x for x in navigation['per_track'] if x['track_id']==tid)
    if evidence['source_rows']!=len(uv) or evidence['source_first_uv']!=uv[0].tolist():
        raise ValueError('Bounded board navigation binds different actual source UV')
    ad=np.asarray(model['blocks'][adid]['values'],float)
    def evaluate(x):
        X=scene.board(np.full(len(times),ad[0]+ad[1]*x[0]),np.full(len(times),ad[2]+(ad[3]-ad[2])*x[1]))
        pr=camera.project_rs(X,times)
        vis=render_visibility(camera,scene,X,[dict(begin=0,end=len(X),label='board.texture',board_material_surface=True)],times,pr['uv'])
        if not np.all(pr['valid']&vis):raise ValueError('Current shared point true RS/visibility invalid')
        return pr['uv']
    trials=[];choices=[];seen=set();starts=[]
    for x in evidence['grid_trials']:
        starts.append((x['material_uv'],'full_bounded_material_grid'))
    for x in evidence.get('actual_ray_roots',[]):
        if x['inward_face'] and x['within_finite_board_height']:
            starts.append((np.clip(x['unknown_ad_uv_initialization'],0,1).tolist(),
                'bounded_navigation_of_current_source_ray_not_exact_world_truth'))
    for start,origin in starts:
        token=tuple(np.round(start,13))
        if token in seen:continue
        seen.add(token);rec=dict(initial_material_uv=list(start),origin=origin)
        try:
            before=evaluate(start);rec['initial_RS_RMS_px']=float(np.sqrt(np.mean(np.sum((before-uv)**2,axis=1))))
            rec['initial_inside_image_rows']=int(np.all((before>=[0,0])&(before<=[1920,1080]),axis=1).sum())
            candidates=[(np.asarray(start,float),before,'unrefined_legal_grid_or_bounded_ray',False)]
            try:
                fit=least_squares(lambda x:(evaluate(x)-uv).ravel(),np.asarray(start,float),bounds=([0.,0.],[1.,1.]),method='trf',xtol=1e-12,ftol=1e-12,gtol=1e-9)
                after=evaluate(fit.x);rec.update(refined_material_uv=fit.x.tolist(),optimizer_success=bool(fit.success),optimizer_message=str(fit.message),optimizer_evaluations=int(fit.nfev),refined_RS_RMS_px=float(np.sqrt(np.mean(np.sum((after-uv)**2,axis=1)))),refined_inside_image_rows=int(np.all((after>=[0,0])&(after<=[1920,1080]),axis=1).sum()))
                candidates.append((fit.x,after,'bounded_shared_material_coordinate_refinement',bool(fit.success)))
            except Exception as e:rec['refinement_failure']=str(e)
            for q,pred,kind,success in candidates:
                if np.all((pred>=[0,0])&(pred<=[1920,1080])):
                    choices.append(dict(material_uv=q.tolist(),RMS_px=float(np.sqrt(np.mean(np.sum((pred-uv)**2,axis=1)))),max_px=float(np.linalg.norm(pred-uv,axis=1).max()),kind=kind,optimizer_success=success,all_true_RS_visibility_finite_and_inside_image=True))
        except Exception as e:rec['initial_physical_domain_failure']=str(e)
        trials.append(rec)
    if not choices:raise ValueError('No source-named bounded shared board point is physically valid inside all actual frames')
    choices.sort(key=lambda x:(x['RMS_px'],x['material_uv']))
    selected=choices[0];clusters=[]
    for candidate in choices:
        if not any(np.max(np.abs(np.asarray(candidate['material_uv'])-np.asarray(c['material_uv'])))<1e-7 for c in clusters):clusters.append(candidate)
    report=dict(track_id=tid,source_qualification_independent_of_geometry_initialization=True,
        actual_current_source_ray_roots=evidence.get('actual_ray_roots',[]),all_grid_and_refinement_trials=trials,
        all_physically_legal_inside_image_candidates=choices,distinct_numerical_candidates=clusters,
        selection='minimum actual fixed-camera source reprojection RMS among all retained trueRS/visible/finite/image-valid trials; navigation only, no source UV change',
        selected=selected,ambiguity_retained=True,world_coordinate_truth_or_covariance_claimed=False)
    return np.asarray(selected['material_uv']),report


def prepare_successor_static_board_tracklets(parent_model,generation,qualification,roles,output,initialization_ledger):
    """Build a typed source bundle, never issue or edit a numerical packet."""
    from scipy.optimize import least_squares
    from .joint_model import graph_from_joint,as_event_view
    from .camera_graph import CameraGraph
    from .joint_replay import render_visibility
    from rinkfit_start.native_scene import NativeScene
    out=Path(output).resolve()
    if out.exists():raise FileExistsError('New immutable board texture physical bundle required')
    groups=_source_contract(generation,qualification,roles);q=read_json(qualification);gen=read_json(generation)
    navigation=read_json(initialization_ledger)
    if navigation["model"]["sha256"]!=file_sha256(parent_model) or navigation["qualification"]["sha256"]!=file_sha256(qualification):raise ValueError("Surface navigation has another model or source qualification")
    model=read_json(parent_model);result=copy.deepcopy(model);cg=CameraGraph(graph_from_joint(model));scene=NativeScene(as_event_view(model,'s01'))
    rows=[];receipts=[];latents=[];mapping={};fit_reports=[];projected=[];source_uv=[]
    for k, originals in sorted(groups.items()):
        first=originals[0];sid=first['shot_id'];instance=first['surface_instance_id'];adid=_ad_block(model,instance);ad=np.asarray(model['blocks'][adid]['values'])
        camera=cg.native_camera(sid);pts=np.array([r['native_pts']for r in originals],np.int64);times=cg.local_seconds(sid,pts);uv=np.asarray([r['source_uv']for r in originals])
        seed,hits=_bounded_surface_seed(model,scene,camera,times,uv,adid,k,navigation)
        def evaluate(x):
            X=scene.board(np.full(len(times),ad[0]+ad[1]*x[0]),np.full(len(times),ad[2]+(ad[3]-ad[2])*x[1]));pr=camera.project_rs(X,times)
            records=[dict(begin=0,end=len(X),label='board.texture',board_material_surface=True)]
            visible=render_visibility(camera,scene,X,records,times,pr['uv'])
            if not np.all(pr['valid']&visible):raise ValueError('Finite shared board point has invalid native RS or physical visibility')
            return pr['uv']
        before=evaluate(seed);coord=seed;after=evaluate(coord);mi=len(result['materials']);bi=len(result['blocks']);label='board.texture.'+k.replace('/','.');pathid='static_board_track/'+k
        result['blocks'].append(dict(name='board_texture_shared_uv/'+k,values=[1.,0.,float(coord[0]),0.,1.,float(coord[1])],
            lower=[1.,0.,0.,0.,1.,0.],upper=[1.,0.,1.,0.,1.,1.],scales=[1.]*6,constant=False,fixed_indices=[0,1,3,4]))
        result['materials'].append(dict(label=label,paths=[dict(id=pathid,variant_id='source_static_board_unknown_print_uv',
            kind=2,points=[[0.,0.]],geometry_block=bi,advertisement_block=adid,physical_point=True,closed=False,
            provenance=json.dumps(dict(track_id=k,source_board_instance=instance,static_surface_qualification=_ref(qualification),
                geometry_role='unknown_shared_material_uv_on_existing_continuous_board_ad',entity_station_fixed=0.,
                parent_camera_ray_and_numeric_installation_are_initialization_only=True,known_world_corner_correspondence=False),ensure_ascii=False))]))
        mapping[label]=dict(material_index=mi,geometry_ready=True)
        latents.append(dict(track_id=k,board_instance_id=instance,material_index=mi,path_index=0,geometry_block=bi,advertisement_block=adid,
            native_path_kind=2,points=[[0.,0.]],initial_unknown_material_uv=coord.tolist(),ray_seed_unknown_material_uv=seed.tolist(),
            shared_native_rows=len(originals),free_coordinate_indices=[2,5],fixed_indices=[0,1,3,4],
            world_coordinate_truth_or_covariance_claimed=False,one_shared_latent_per_track=True))
        fit_reports.append(dict(track_id=k,source_rows=len(originals),source_surface_initialization_navigation=hits,
            source_UV_changed=False,bounded_navigation_RS_RMS_px=float(np.sqrt(np.mean(np.sum((before-uv)**2,axis=1)))),
            shared_point_only_RS_RMS_px=float(np.sqrt(np.mean(np.sum((after-uv)**2,axis=1)))),
            camera_scene_existing_ad_blocks_optimized=False,optimizer_success=hits["selected"]["optimizer_success"],
            station_fixed=0.,Query_or_station_sliding=False,source_covariance_claimed=False))
        projected.extend(after);source_uv.extend(uv)
        for original in originals:
            pts0=original['native_pts'];rid='static_board_texture/'+k+'/'+str(pts0)
            row=dict(original,source_row_id=rid,source_family_id=k+'/'+str(pts0),physical_track_id=k,
                target_id=label,joint_material_id=label,joint_material_index=mi,current_uv=original['source_uv'],
                point_correspondence=True,measurement_operator_resolved=OPERATOR,source_geometry_eligible=True,
                statistical_objective_eligible=False,numeric_statistics_ready=False,solver_eligible=False,
                frame_role='development_candidate',normal=[0.,0.],entity_kind=1,entity_station=0.,
                world_position_is_shared_track_latent=True,
                physical_position='one shared unknown bounded Ad material [u,v] per track; not a known world corner')
            receipt=dict(schema=RECEIPT_SCHEMA,source_row_id=rid,track_id=k,native_pts=pts0,shot_id=sid,
                source_uv=original['source_uv'],current_rgb_sha256=original['current_rgb_sha256'],status='accepted_static_board_print',
                target_id=label,existing_material_index=mi,existing_point_path=pathid,point_path_kind=2,
                geometry_block=bi,advertisement_block=adid,source_board_instance_id=instance,
                physical_point=True,entity_kind=1,entity_station_fixed=True,entity_station=0.,new_material_suffix=True,
                new_actual_source_geometry_qualified=True,original_UV_preserved=original['source_uv'],
                current_2D_source_mean_sha256=object_sha256(dict(source_row_id=rid,native_pts=pts0,current_2D_UV=original['source_uv'],estimator=OPERATOR)),
                solver_eligible=False,statistical_objective_eligible=False,numeric_statistics_ready=False,
                original_flags_preserved=dict(manual_reference_preserved=False),static_surface_qualification=_ref(qualification),
                numeric_source_replay=q['numeric_replay'],known_world_corner_correspondence=False,unknown_shared_surface_coordinates=True)
            rows.append(row);receipts.append(receipt)
    out.mkdir(parents=True);result['static_board_successor_shared_uv_append']=dict(parent_model=_ref(parent_model),
        source_generation=_ref(generation),static_surface_qualification=_ref(qualification),tracks=len(groups),source_rows=len(rows),
        unchanged_parent_blocks_materials_rigs_scene_shots=True,unknown_coordinates_per_track=2,
        source_qualified_material_identity_is_not_an_exact_world_or_template_corner_correspondence=True)
    write_json(out/'model.json',result);write_json(out/'latents.json',latents);write_jsonl(out/'source_rows.jsonl',rows)
    write_jsonl(out/'qualification_receipts.jsonl',receipts);write_json(out/'material_map.json',dict(semantic_to_material=mapping))
    write_json(out/'initialization_report.json',dict(schema='rinkfit.source_board_shared_point_initialization.v1',per_track=fit_reports,
        initializer='explicit all-trial bounded shared Ad-material UV navigation and fixed-camera native RS refinement; actual ray roots and source/model discrepancy retained',
        ice_H_or_template_corresponding_station_used=False,unknown_shared_material_coordinates_per_track=2,
        actual_shared_latent_count=len(groups),actual_free_coordinates=2*len(groups),camera_scene_or_parent_values_changed=False,
        optical_camera_ad_geometry_calibrated=False,accepted_full_physical_fit=False))
    np.savez_compressed(out/'source_projection.npz',source_uv=np.asarray(source_uv),prediction_uv=np.asarray(projected))
    gate=dict(schema=GATE_SCHEMA,source_generation=_ref(generation),static_surface_qualification=_ref(qualification),
        numeric_source_replay=q['numeric_replay'],current_RGB_receipt=q['current_RGB_receipt'],roles=_ref(roles),
        input_cumulative_journal=_ref(out/'source_rows.jsonl'),qualification_receipts=_ref(out/'qualification_receipts.jsonl'),
        current_model=_ref(out/'model.json'),track_material_bindings=_ref(out/'latents.json'),gate_code=_ref(__file__),
        qualified_new_measurement_rows=len(rows),old_raw_source_flags_false_are_not_overwritten=True,
        new_measurement_version_geometry_qualification_only=True)
    write_json(out/'qualification_gate.json',gate);counts=dict(Counter(r['shot_id']for r in rows))
    chunk=dict(schema='rinkfit.actual_source_gate_reference_adapter.v1',all_requested_frames_accounted=True,
        source_dataset_tag='static_board_tracklets_physical_v1',source_rows_ref=_ref(out/'source_rows.jsonl'),
        geometry_rows_ref=_ref(out/'source_rows.jsonl'),source_gate_manifest=_ref(out/'qualification_gate.json'),
        actual_source_rows=len(rows),by_shot=counts)
    write_json(out/'chunk_manifest.json',chunk)
    scripts=[Path(qualification).parent/n for n in ('make_source_review.py','numeric_source_replay.py')]
    bundle=dict(schema=BUNDLE_SCHEMA,parent_model=_ref(parent_model),source_generation=_ref(generation),static_surface_qualification=_ref(qualification),
        numeric_source_replay=q['numeric_replay'],current_RGB_receipt=q['current_RGB_receipt'],source_rows_ref=_ref(out/'source_rows.jsonl'),
        qualification_receipts=_ref(out/'qualification_receipts.jsonl'),qualification_gate=_ref(out/'qualification_gate.json'),
        track_material_bindings=_ref(out/'latents.json'),world_latent_bindings=_ref(out/'latents.json'),
        reader_module_refs=[_ref(__file__),_ref(source_helpers.__file__),*gen['reader_modules'],*[_ref(p)for p in scripts]],
        source_ledger_preserved=True,closed_chunks=[_ref(out/'chunk_manifest.json')],typed_entity_qualification=[_ref(out/'qualification_gate.json')],
        material_map=_ref(out/'material_map.json'),model=_ref(out/'model.json'),roles=_ref(roles),source_rows=len(rows),tracks=len(groups),
        free_shared_coordinates=2*len(groups),by_shot=counts,source_noise_statistics_supplied=False,
        absolute_source_UV_only=True,independent_displacement_factors=False,packet_issued=False,initialization_navigation=_ref(initialization_ledger),inherited_append_provenance_preserved=True)
    write_json(out/'manifest.json',bundle)
    load_point_qualification(out/'qualification_gate.json',gate,_ref(out/'qualification_gate.json'),_verified)
    return bundle


def finalize_prepared_successor_bundle(source_folder,parent_model,generation,qualification,roles,initialization_ledger,executed_initializer_module):
    """Publish an unissued partial result after strict readback, without refitting.

    The executed initializer snapshot and current finalization code are separate
    immutable references. A failed write never becomes a successful execution.
    """
    out=Path(source_folder).resolve()
    if(out/'manifest.json').exists():raise FileExistsError('Successor source bundle already issued')
    gen=read_json(generation);q=read_json(qualification);model=read_json(out/'model.json');parent=read_json(parent_model)
    groups=_source_contract(generation,qualification,roles)
    if (model['blocks'][:len(parent['blocks'])]!=parent['blocks']or model['materials'][:len(parent['materials'])]!=parent['materials']
        or any(model.get(k)!=v for k,v in parent.items())):
        # blocks/materials have a strictly appended suffix; every other old field stays exact.
        if any(model.get(k)!=v for k,v in parent.items()if k not in ('blocks','materials'))or model['blocks'][:len(parent['blocks'])]!=parent['blocks']or model['materials'][:len(parent['materials'])]!=parent['materials']:
            raise ValueError('Prepared successor changed inherited model')
    rows=list(read_jsonl(out/'source_rows.jsonl'));latents=read_json(out/'latents.json')
    if len(rows)!=sum(map(len,groups.values()))or len(latents)!=len(groups):raise ValueError('Prepared successor lost actual source denominator')
    report=read_json(out/'initialization_report.json')
    for track in report['per_track']:
        if 'ray_seed_RS_RMS_px'in track:track['bounded_navigation_RS_RMS_px']=track.pop('ray_seed_RS_RMS_px')
    report['executed_initializer_module']=_ref(executed_initializer_module)
    report['finalized_without_new_optimization']=True
    write_json(out/'initialization_report.json',report)
    gate=read_json(out/'qualification_gate.json');gate['gate_code']=_ref(__file__);write_json(out/'qualification_gate.json',gate)
    chunk=read_json(out/'chunk_manifest.json');chunk['source_gate_manifest']=_ref(out/'qualification_gate.json');write_json(out/'chunk_manifest.json',chunk)
    scripts=[Path(qualification).parent/n for n in ('make_source_review.py','numeric_source_replay.py')]
    bundle=dict(schema=BUNDLE_SCHEMA,parent_model=_ref(parent_model),source_generation=_ref(generation),static_surface_qualification=_ref(qualification),
        numeric_source_replay=q['numeric_replay'],current_RGB_receipt=q['current_RGB_receipt'],source_rows_ref=_ref(out/'source_rows.jsonl'),qualification_receipts=_ref(out/'qualification_receipts.jsonl'),qualification_gate=_ref(out/'qualification_gate.json'),track_material_bindings=_ref(out/'latents.json'),world_latent_bindings=_ref(out/'latents.json'),
        reader_module_refs=[_ref(executed_initializer_module),_ref(__file__),_ref(source_helpers.__file__),*gen['reader_modules'],*[_ref(p)for p in scripts]],
        source_ledger_preserved=True,closed_chunks=[_ref(out/'chunk_manifest.json')],typed_entity_qualification=[_ref(out/'qualification_gate.json')],material_map=_ref(out/'material_map.json'),model=_ref(out/'model.json'),roles=_ref(roles),source_rows=len(rows),tracks=len(groups),free_shared_coordinates=2*len(groups),by_shot=dict(Counter(r['shot_id']for r in rows)),source_noise_statistics_supplied=False,absolute_source_UV_only=True,independent_displacement_factors=False,packet_issued=False,initialization_navigation=_ref(initialization_ledger),inherited_append_provenance_preserved=True,executed_initializer_module=_ref(executed_initializer_module),initialization_report=_ref(out/'initialization_report.json'),initial_write_failure_retained='prepare_successor_source.log')
    load_point_qualification(out/'qualification_gate.json',gate,_ref(out/'qualification_gate.json'),_verified)
    write_json(out/'manifest.json',bundle)
    return bundle


def repackage_unissued_successor_bundle(source_folder,output,executed_finalizer_module):
    """Repair unpublished gate/chunk references without copying or changing UV/model."""
    source=Path(source_folder).resolve();out=Path(output).resolve()
    if out.exists():raise FileExistsError('Successor bundle publication exists')
    old=read_json(source/'manifest.json');out.mkdir(parents=True)
    gate=read_json(source/'qualification_gate.json');gate['gate_code']=_ref(__file__);write_json(out/'qualification_gate.json',gate)
    chunk=read_json(source/'chunk_manifest.json');chunk['source_gate_manifest']=_ref(out/'qualification_gate.json');write_json(out/'chunk_manifest.json',chunk)
    b=copy.deepcopy(old);b['qualification_gate']=_ref(out/'qualification_gate.json');b['closed_chunks']=[_ref(out/'chunk_manifest.json')];b['typed_entity_qualification']=[b['qualification_gate']]
    prior=_ref(executed_finalizer_module)
    b['reader_module_refs']=[prior if Path(x['file']).resolve()==Path(__file__).resolve()else x for x in old['reader_module_refs']]+[_ref(__file__)]
    b['unissued_preparation_failure_preserved']=dict(prior_manifest=_ref(source/'manifest.json'),prior_finalizer=prior,reason='gate code refresh had left chunk gate SHA stale; no UV/model or qualification changed')
    load_point_qualification(out/'qualification_gate.json',gate,_ref(out/'qualification_gate.json'),_verified)
    write_json(out/'manifest.json',b)
    return b
