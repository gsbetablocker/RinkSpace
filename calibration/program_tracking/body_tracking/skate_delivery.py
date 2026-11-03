"""Export one existing body state with its attached, versioned skate geometry.

The source packet retains observation/manual/unknown fields. All predictions
are reevaluated on the source clock; this module does not fit or repair them.
"""
from pathlib import Path
import json

import numpy as np

from .footwear_binding import transform
from .rebuild_exposure_body import moving_projection
from .rebuild_prepare import load, digest
from .skate_binding import TEMPLATE_VERSION, BINDING_VERSION
from .skate_template import profile_height
from .skate_physics import fk_state, constraints, ALLOWANCE, SUPPORT_BAND


def export_candidate(root, out, ref, camera, info, evidence):
    """Write ``out/fitted/result.npz`` from the single saved body/FK state.

    The historical solver unit is converted to conditional scene metres once.
    Physics clearance/support heights already use that metric unit, whereas
    its witness position follows the solver-state coordinate unit.
    """
    root,out=Path(root),Path(out)
    source_path=root/'opening_timing_1103/hand_polish/fitted/result.npz'
    source=load(source_path);binding=ref.binding;layout=binding.layout
    pts=np.asarray(ref.p['pts']);times=np.asarray(ref.times);n=len(pts)
    np.testing.assert_array_equal(source['pts'],pts)
    if 'source_pts' in source:np.testing.assert_array_equal(source['source_pts'],pts)
    if times.shape!=(n,) or n<2 or np.any(np.diff(pts)<=0) or np.any(np.diff(times)<=0):
        raise ValueError('Export requires ordered source exposures and matching state times')
    ratio=float(info['length_export_ratio'])
    if not np.isfinite(ratio) or ratio<=0:raise ValueError('Invalid export length ratio')
    np.testing.assert_allclose(ratio,binding.ratio,rtol=0,atol=1e-12)
    np.testing.assert_allclose(ratio,float(info['metres_per_scene_unit'])/float(info['solver_metres_per_scene_unit']),rtol=1e-12)
    packet=dict(source);arrays={}
    def append(key,value):arrays.setdefault(key,[]).append(np.asarray(value))
    template_packet=binding.packet();template=template_packet['template']
    # A fixed material point on the steel web's top surface, not a contact
    # minimum or a point in the hollow below the solid runner centreline.
    p=template_packet['parameters'];reference_x=.4*p.sole_length_m
    reference_B=np.array([[reference_x,0.,float(profile_height(reference_x,p))+p.blade_web_height_m]])
    origin_seconds=pts[0]/12800.
    shutter=float(camera.p['rig'][37])
    for first in range(0,n,32):
        frames=np.arange(first,min(first+32,n));t=times[frames]
        mid_reference=ref.at(t);mid=ref.world(mid_reference)
        node_times=(t[:,None]+np.linspace(-.5,.5,5)*shutter).ravel()
        nodes=ref.world(ref.at(node_times));count=len(frames)
        projection=moving_projection(camera,nodes['world'].reshape(count,5,layout.point_count,3),frames)
        uv=projection['uv']
        for target,key in [('P_world_mid','P'),('native22_world_mid','native'),('COCO_world_mid','coco'),('gear_world_mid','gear')]:
            append(target,mid[key]*ratio)
        append('Q_world_mid',mid['Q']);append('local_R_mid',mid['local_R'])
        append('COCO_uv',uv[:,layout.body_landmarks])
        append('equipment_uv',uv[:,:layout.equipment_count].reshape(count,2,layout.per_foot,2))
        append('P_uv_dynamic',moving_projection(camera,nodes['P'].reshape(count,5,1,3),frames)['uv'][:,0])
        append('world_at_row',projection['world']*ratio)
        append('inside_exposure_rows',projection['inside_exposure_rows'])
        append('source_row_time',pts[frames,None]/12800.+shutter*(np.clip(uv[...,1],0,1079)/1079-.5))

        gear=mid['footwear'];R,O=gear['boot_R'],gear['boot_O']
        lower=transform(R,O,template['vertices'])*ratio
        upper=transform(R,O,gear['upper_local'])*ratio
        cavity=transform(R,O,gear['inner_local'])*ratio
        # The footbed display is a boundary loop: each section's two floor
        # corners, forward along -y and back along +y, without crossing it.
        sections=len(template_packet['xs'])
        rings=gear['inner_local'][:,:,:sections*6].reshape(count,2,sections,6,3)
        footbed=np.concatenate((rings[:,:,:,0],rings[:,:,::-1,1]),axis=2)
        append('skate_lower_world_mid',lower);append('skate_upper_world_mid',upper)
        append('skate_cavity_world_mid',cavity)
        append('skate_footbed_world_mid',transform(R,O,footbed)*ratio)
        edge=template_packet['edges']
        append('skate_edge_world_mid',np.stack([transform(R,O,edge[k])*ratio for k in range(2)],axis=2))
        append('skate_R_world_mid',R);append('skate_origin_world_mid',O*ratio)
        append('skate_reference_world_mid',transform(R,O,reference_B)[:,:,0]*ratio)

        physics=fk_state(ref,mid_reference,np.zeros((count,69)))
        np.testing.assert_allclose(physics['R'],R,rtol=0,atol=1e-10)
        np.testing.assert_allclose(physics['O'],O,rtol=0,atol=1e-10)
        contact=constraints(binding,physics,t,origin_seconds,evidence)
        append('skate_witness_world_mid',contact['witness']*ratio)
        append('skate_clearance_m_mid',contact['clearance'])
        append('skate_support_gap_m_mid',contact['support_height'])

        local=np.einsum('nji,nkj->nki',mid['Q'],mid['native']-mid['P'][:,None])
        metrics=binding.foot_metrics(local,mid['frames'])
        violation=np.stack([np.max(value,axis=(1,2)) for value in metrics['violations']],axis=1)*ratio
        append('foot_cavity_max_violation_m_mid',violation)
        append('cavity_penetration_mid',violation)
        anchor_error=np.linalg.norm(metrics['anchors']-metrics['targets'][None,None],axis=-1)*ratio
        append('heel_drift_mid',anchor_error[:,:,0]);append('forefoot_error_mid',anchor_error[:,:,1])
        anchors=np.concatenate((metrics['anchors'],np.broadcast_to(metrics['targets'],metrics['anchors'].shape)),axis=2)
        append('anchors_world_mid',transform(R,O,anchors)*ratio)
        append('upper_deformation_mid',gear['upper_deformation']*ratio)
        append('effective_ankle_relative_rotation_mid',np.stack([
            np.swapaxes(mid['frames'][:,k],-1,-2)@mid['frames'][:,a] for k,a in ((4,7),(5,8))],axis=1))
        append('blade_tilt_rad_mid',np.arcsin(np.clip(R[:,:,2,0],-1,1)))
        append('blade_pose_world_mid',np.concatenate((R,(O*ratio)[...,None]),axis=3))
        axes=[]
        shank_R=np.stack([mid['Q']@mid['frames'][:,k] for k in (4,5)],axis=1)
        shank_O=mid['native'][:,[7,8]]
        for rotation,origin in ((shank_R,shank_O),(R,O),(R,O)):
            for axis in range(3):
                axes.append(np.stack((origin,origin+rotation[:,:,:,axis]*(.08/ratio)),axis=2)*ratio)
        append('axes_world_mid',np.stack(axes,axis=2))

    packet.update({key:np.concatenate(values,axis=0) for key,values in arrays.items()})
    for key,value in arrays.items():
        if key=='skate_support_gap_m_mid':continue
        if not np.isfinite(packet[key]).all():raise ValueError('Nonfinite exported prediction: '+key)
    P=packet['P_world_mid'];G=P.copy();G[:,2]=0
    gear=packet['gear_world_mid']
    mask=np.zeros(n,bool)
    for state,_ in ref.new_refinements:mask|=np.any(state.basis(times)!=0.,axis=1)
    record=binding.record();layout_record=layout.record()
    dimensions={name:(float(value)*ratio if name.endswith('_m') else value)
        for name,value in template['derived_dimensions'].items()}
    neutral=binding.foot_metrics(binding.neutral_native,binding.neutral_frames,binding.neutral_samples)
    dimensions['generic_skin_extents_in_registered_B_m']=[(np.ptp(v[0],axis=0)*ratio).tolist() for v in neutral['feet']]
    dimensions['foot_extent_status']='generic body surface extent; not athlete foot length or shoe size'
    layout_record.update(render_part_ranges=template['part_ranges'],upper_region_labels=['negative_y_side','positive_y_side','bridge','tongue'],
        fixed_blade_reference_B_m=(reference_B[0]*ratio).tolist())
    for key in ('solver_config_json','active_blocks_json','stopping_reason_json','costs_json'):
        if key in source:packet['parent_'+key]=source[key]
    packet.update(Q_world=packet['Q_world_mid'],boot_world_mid=gear[:,:,:14],blade_world_mid=gear[:,:,14:],G_world_mid=G,
        metres_per_scene_unit=np.array(info['metres_per_scene_unit']),solver_metres_per_scene_unit=np.array(info['solver_metres_per_scene_unit']),
        body_length_ratio=np.array(ratio),camera_binding_json=np.array(json.dumps(info,sort_keys=True)),
        template_version=np.array(TEMPLATE_VERSION),binding_version=np.array(BINDING_VERSION),
        parameter_provenance_json=np.array(json.dumps(record['parameter_provenance'],sort_keys=True)),
        layout_json=np.array(json.dumps(layout_record,sort_keys=True)),footwear_binding_json=np.array(json.dumps(record,sort_keys=True)),
        skate_dimensions_json=np.array(json.dumps(dimensions,sort_keys=True)),
        footwear_contact_evidence_json=np.array(json.dumps(evidence,sort_keys=True)),
        skate_contact_engineering_json=np.array(json.dumps(dict(allowed_entry_m=ALLOWANCE,support_band_m=SUPPORT_BAND,
            support_height='eligible minimum raw height, only evidence exposure; NaN unknown',
            clearance='minimum height plus semantic allowance',witness='moving lower-skate minimum, not a material-point velocity',
            engineering_tolerances_not_measured_ice=True),sort_keys=True)),
        skate_lower_faces=np.asarray(template['faces'],np.int32),skate_lower_face_parts=np.asarray(template['face_parts'],str),
        skate_upper_faces=binding.faces,skate_upper_region_ids=binding.region_ids,
        skate_cavity_faces=binding.faces,skate_collar_boundary=template_packet['collar'],
        skate_contact_mode_mid=np.full((n,2),'unknown'),skate_contact_confidence_mid=np.zeros((n,2)),
        footwear_binding_applied_mask=np.ones(n,bool),body_refinement_window_mask=mask,window_mask=mask,
        parent_window_mask=source['window_mask'],
        boot_surface_world_mid=gear[:,:,:14],boot_surface_faces=binding.adapter_faces,
        boot_surface_region_ids=np.r_[np.zeros(5,int),np.ones(5,int),np.full(12,2)],collar_boundary=np.array([4,5,12,11]),
        footbed_world_mid=packet['skate_footbed_world_mid'],cavity_world_mid=packet['skate_cavity_world_mid'],cavity_faces=binding.faces,
        anchor_labels=np.array(['foot_heel_patch','foot_ball_patch','boot_heel_pocket','boot_ball_support']),
        axis_labels=np.array([name+'_'+str(axis) for name in ('shank','lower_boot','blade') for axis in range(3)]),
        footwear_integration=np.array('shared static T_FB and shape; same body/FK, exact lower-skate contact; source unknown labels retained'),
        skate_export_provenance_json=np.array(json.dumps(dict(parent_result=str(source_path),parent_sha256=digest(source_path),
            reference_hashes=ref.hashes,implementation_sha256=digest(__file__),exposure_nodes=5,
            source_time_base=[1,12800],new_body_refinement_intervals_seconds=[list(ws.seconds) for ws,_ in ref.new_refinements],
            shared_footwear_scope='all saved source exposures',length_conversion='solver geometry times ratio once; physics height already metric',
            no_fit_performed_by_export=True),sort_keys=True)),
        solver_config_json=np.array(json.dumps(dict(export_only=True,body_state='ref.at(t)',shared_parameters=17,exposure_nodes=5))),
        active_blocks_json=np.array(json.dumps(dict(shared_binding=17,body_columns=[ws.active.tolist() for ws,_ in ref.new_refinements]))),
        stopping_reason_json=np.array(json.dumps('export of supplied state; optimizer status belongs to caller')),costs_json=np.array('{}'))
    destination=out/'fitted';destination.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(destination/'result.npz',**packet)
    return packet


def _byte_equal(left,right):
    left,right=np.asarray(left),np.asarray(right)
    return left.shape==right.shape and left.dtype==right.dtype and left.tobytes()==right.tobytes()


def _parameter_replay_checks(replay,saved,shared_x):
    """Check every checkpoint field, including optional underlying body layers."""
    fields={key:key in replay and _byte_equal(replay[key],value) for key,value in saved.items()}
    shared='shared_x' in replay and _byte_equal(replay['shared_x'],shared_x)
    if not all(fields.values()) or not shared:
        failed=[key for key,equal in fields.items() if not equal]
        if not shared:failed.append('shared_x')
        raise ValueError('OneDrive parameter replay changed fields: '+', '.join(failed))
    return dict(controls_byte_equal=True,saved_fields_byte_equal=fields,
                saved_field_count=len(fields),shared_x_byte_equal=shared)


def _delivery_identity(programme,delivery_name,variant_id,pts):
    """Resolve independent parameter names before writing any delivery assets."""
    if not delivery_name or any(c in delivery_name for c in '/\\:'):
        raise ValueError('delivery_name must be a filename stem')
    if not variant_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in variant_id):
        raise ValueError('Invalid viewer variant id')
    paths=tuple(programme/(delivery_name+suffix) for suffix in ('.npz','.json'))
    for path in paths:
        if len(str(path.resolve()).encode('utf-16-le'))//2>=260:raise ValueError('OneDrive path too long')
    historical=delivery_name=='skate_contact_1103' and variant_id=='skate1103'
    return dict(parameter_path=paths[0],record_path=paths[1],
        label='鞋刀联合拟合候选（待人工审核）' if historical else '全 S01 连续鞋刀候选（待人工审核）',
        initial_source_pts=450560 if historical else int(np.asarray(pts)[0]),
        comparisons=['full1102','hand1103',variant_id] if historical else list(dict.fromkeys(['full1102','hand1103','skate1103',variant_id])))


def deliver(project,root,out,ref,camera,info,evidence,problem,report,catalog=None,*,
            delivery_name='skate_contact_1103',variant_id='skate1103'):
    """Persist replayable small parameters, full-S01 impact and player binding."""
    from dataclasses import asdict
    from .fullbody_round import write_json
    from .skate_review import review_candidate
    from rinkspace.viewer3d.data import ResultCatalog
    from rinkspace.viewer3d.prepare import prepare_variant
    project,root,out=map(Path,(project,root,out))
    programme=project/'events/ankara_2026/programmes/okada_mei_sp'
    identity=_delivery_identity(programme,delivery_name,variant_id,ref.p['pts'])
    packet=export_candidate(root,out,ref,camera,info,evidence)
    checkpoint=Path(report.get('candidate_checkpoint',out/'fit/result.npz'))
    fit_report='fit/candidate.json' if report.get('candidate_checkpoint') else 'fit/result.json'
    packet.update(stopping_reason_json=np.array(json.dumps(report.get('stop'))),
        solver_config_json=np.array(json.dumps(dict(solver=fit_report,inputs='inputs.json',
            source_snapshot='fit/kernel_manifest.json',physical_feasibility='unweighted maximum, including source support bands and foot-cavity bounds'))),
        costs_json=np.array(json.dumps(report.get('parts',{}))))
    np.savez_compressed(out/'fitted/result.npz',**packet)
    validation=review_candidate(root,out,ref,camera,info,packet,evidence)
    from .skate_rocker_review import compare_rockers
    validation['main_rocker_candidates']=compare_rockers(root,out,ref,camera,info,packet,evidence)
    from .skate_collar_review import review_collar_gap
    validation['collar_section']=review_collar_gap(root,out,ref,info,packet)
    validation['acceptance_status']='pending human visual review; debugging bands are not calibrated'
    source=load(root/'opening_timing_1103/hand_polish/fitted/result.npz')
    # Replay the actual saved parameters through the same state owner before
    # exposing a new viewer variant. No interpolation of exported poses occurs.
    seconds=ref.p['pts']/12800.;mask=packet['body_refinement_window_mask']
    checks={k:float(np.max(abs(packet[k][~mask]-source[k][~mask]),initial=0.)) for k in
            ('P_world_mid','Q_world_mid','local_R_mid')}
    if max(checks.values())>1e-12:raise ValueError('Body changed outside declared refinement scope')
    binding=ref.binding;foot=packet['foot_cavity_max_violation_m_mid']
    core=(seconds>=35.2-1e-9)&(seconds<=36.8+1e-9)
    validation['foot_fit']=dict(maximum_cavity_breach_m_all_S01=float(foot.max()),
        maximum_cavity_breach_m_core=float(foot[core].max()),
        maximum_cavity_breach_m_changed_body=float(foot[mask].max()),
        source='shared-beta generic human skin against section-offset legal cavity; not athlete foot measurements')
    validation['physical_optimization']=report
    validation['boundary_continuity']={}
    for ws,x in ref.new_refinements:
        at=np.asarray(ws.bounds_time)
        validation['boundary_continuity'][str(ws.seconds)]=dict(
            maximum_increment=float(np.max(abs(ws.increments(x,at)))),
            maximum_control_velocity=float(np.max(abs(ws.basis(at,1)@x.reshape(ws.m,ws.d)))),
            maximum_control_acceleration=float(np.max(abs(ws.basis(at,2)@x.reshape(ws.m,ws.d)))),
            convention=('one continuous C2 source curve, free at available source/exposure support boundary'
                        if type(ws).__name__=='ContinuousShotWindow' else 'one body state, C2 attachment; no separate shoe trajectory'))
    write_json(out/'validation.json',validation)
    param_path=identity['parameter_path'];record_path=identity['record_path']
    saved=load(checkpoint)
    parameters=dict(saved,shared_x=binding.x,base_skate_parameters_json=np.array(json.dumps(asdict(binding.base_parameters))),
        R_FB_initial=binding.R0,t_FB_initial_solver=binding.t0,
        camera_binding_json=np.array(json.dumps(info,sort_keys=True)),
        parent_hand_parameters_sha256=np.array(digest(programme/'opening_hand_1103.npz')),
        parent_timing_parameters_sha256=np.array(digest(programme/'opening_timing_1103.npz')),
        template_version=np.array(TEMPLATE_VERSION),binding_version=np.array(BINDING_VERSION))
    if param_path.exists():
        prior=load(param_path)
        if set(prior)!=set(parameters) or any(prior[k].tobytes()!=np.asarray(v).tobytes() for k,v in parameters.items()):
            history=out/'history'/('delivered_'+digest(param_path)[:12]);history.mkdir(parents=True,exist_ok=True)
            (history/param_path.name).write_bytes(param_path.read_bytes())
            if record_path.exists():(history/record_path.name).write_bytes(record_path.read_bytes())
    np.savez_compressed(param_path,**parameters)
    # Exact round-trip from the OneDrive control packet.
    replay=load(param_path)
    replay_checks=_parameter_replay_checks(replay,saved,binding.x)
    write_json(out/'parameter_replay.json',dict(onedrive_parameters=str(param_path),sha256=digest(param_path),
        **replay_checks,outside_body_maximum_error=checks,
        same_state_template_replay=validation['same_state_template_replay'],source_preservation=validation['source_preservation']))
    core_feasible=bool(report.get('feasible',False) and report.get('dense_gate',{}).get('feasible',False))
    config_path=programme/'viewer3d_fullbody_1102.json'
    config=json.loads(config_path.read_text(encoding='utf8'))
    definition=dict(id=variant_id,label=identity['label'],
        kind='fullbody',result=str((out/'fitted/result.npz').relative_to(root)).replace('\\','/'),
        shell_topology='saved',body_reference_label='hand1103',
        body_refinement_intervals_seconds=[list(ws.seconds) for ws,_ in ref.new_refinements],
        parameter_binding=str(record_path.relative_to(project)).replace('\\','/'))
    config['variants']=[v for v in config['variants'] if v['id']!=variant_id]+[definition]
    config.update(default_variant=variant_id,initial_source_pts=identity['initial_source_pts'],initial_view='side',
        initial_wear_inspection=True,comparison_variants=identity['comparisons'])
    write_json(config_path,config)
    command=project/'START_3D_COMPARISON.cmd'
    command.write_text('@echo off\ncall "%~dp0START_3D_RESULTS.cmd" -Config "%~dp0events\\ankara_2026\\programmes\\okada_mei_sp\\viewer3d_fullbody_1102.json" -Variant '+variant_id+' -Fresh %*\n',encoding='utf8')
    source_seconds=[float(seconds[0]),float(seconds[-1])]
    continuous_mask=np.zeros(len(seconds),bool)
    for ws,_ in ref.new_refinements:
        if type(ws).__name__=='ContinuousShotWindow':continuous_mask|=np.any(ws.basis(ref.times)!=0.,axis=1)
    continuous_all_source=bool(np.all(continuous_mask))
    record=dict(schema='rinkspace.skate-contact-delivery.v1',template_version=TEMPLATE_VERSION,binding_version=BINDING_VERSION,
        delivery_name=delivery_name,variant_id=variant_id,source_seconds=source_seconds,
        parameters=dict(path=str(param_path),sha256=digest(param_path)),run=str(out),camera=info,
        body_parent=dict(hand_parameters=str(programme/'opening_hand_1103.npz'),timing_parameters=str(programme/'opening_timing_1103.npz')),
        body_refinement_intervals_seconds=[list(ws.seconds) for ws,_ in ref.new_refinements],
        shared_shape_scope=dict(source_seconds=source_seconds,source_exposures=len(seconds),
            convention='one shared shape and registration replayed on all available source exposures'),
        body_scope=dict(source_seconds=source_seconds,optimized_source_exposures=int(mask.sum()),
            continuous_all_source=continuous_all_source,
            state_kinds=[type(ws).__name__ for ws,_ in ref.new_refinements],
            intervals_seconds=[list(ws.seconds) for ws,_ in ref.new_refinements],
            convention=('one continuous C2 curve over all available source/exposure support; no internal temporal freeze'
                        if continuous_all_source else 'saved body refinement intervals; parent body replay elsewhere')),
        engineering_allowance_m=ALLOWANCE,support_band_m=SUPPORT_BAND,
        candidate_core_feasible=core_feasible,user_accepted=False,independent_3d_truth=False,
        acceptance_status='uncalibrated debugging thresholds; pending human visual review',
        parameter_provenance=binding.record()['parameter_provenance'],
        evidence={name:dict(path=str(out/name),sha256=digest(out/name)) for name in
            ('inputs.json','binding.json','validation.json',fit_report,'fit/kernel_manifest.json','source_evidence.json','parameter_replay.json','fitted/result.npz','collar_gap.json')},
        viewer_config=str(config_path),viewer_command=str(command))
    write_json(record_path,record)
    player=ResultCatalog(project=project,data_root=root.parents[1],config=config_path)
    prepare_variant(player,player.variant(variant_id),binding.surface)
    print(json.dumps(dict(delivered=str(record_path),viewer_command=str(command),source_seconds=source_seconds,
        actual_body_intervals=record['body_refinement_intervals_seconds'],core_feasible=core_feasible)),flush=True)
    return packet,validation
