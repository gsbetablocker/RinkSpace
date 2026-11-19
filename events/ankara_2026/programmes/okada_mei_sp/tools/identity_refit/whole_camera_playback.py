from source_camera_rebind import checked_entry, write
from calibration.program_tracking.body_tracking.skate_delivery import export_candidate
from calibration.program_tracking.body_tracking.actual_observation_ledger import update_source_observation_export
from calibration.program_tracking.body_tracking.continuous_physics import ContinuousMassGeometry
from calibration.program_tracking.body_tracking.programme_com import export_com_diagnostics
from calibration.program_tracking.body_tracking.skate_physics import fk_state
from rinkspace.domain.parameters import ParameterState,SceneParameterModel
from rinkspace.session.gpu_geometry import NativeSceneGPU
from rinkspace.review.scene import sample_native_review_scene
import shutil
def export(prepared, output, registration, config_base, ledger):
    import json,numpy as np
    from compare import bind
    from source_review import digest
    from pathlib import Path
    current=output/'candidate.npz'
    ref,cam,info,doc,packet=bind(prepared,output/'source.json',current)
    pts=ref.p['pts']; ratio=ref.binding.ratio; n=len(pts)
    print('Exporting midpoint states and source projections',flush=True)
    dest=output/'result.npz'
    if not dest.exists():
        result=export_candidate(prepared,output,ref,cam,info,doc.get('supports',[]),write=False)
        mass=ContinuousMassGeometry(ref)
        state=fk_state(ref,ref.at(ref.times),np.zeros((n,69)))
        centres,rotations=mass.from_pose(state)
        result.update(export_com_diagnostics(mass,ref.times,centres,rotations))
        ground=result['com_world_mid'].copy();ground[:,2]=0.
        result.update(com_projection_world_mid=ground,native_body_local_R_mid=state['local_R'],
            native_body_FK_world_R_mid=state['Q'][:,None]@state['frames'],
            P_velocity_world_mps=ref.root_derivative(ref.times,1)*ratio,
            shared_betas=ref.p['shared_betas'],shared_body_scale=ref.p['body_scale'],
            ice_plane_known=np.array(ref.ice_plane_known),
            sequence_contact_modes=packet['sequence_contact_modes'],sequence_contact_status=packet['sequence_contact_status'],
            sequence_contact_provenance=np.array('Parent mode hypotheses, no new measured contacts'),
            source_action_ledger_sha256=np.array(digest(output/'source.json')),
            transfer_parameter_sha256=np.array(digest(current)))
        with np.load(ledger) as z: observations={k:z[k] for k in z}
        joint_refit='joint_source_fit_json' in packet
        used=np.arange(n,dtype=int) if joint_refit else np.array([],dtype=int)
        update_source_observation_export(result,observations,ledger,used)
        if joint_refit:
            from opening_support import verify
            rule=json.loads(str(packet['programme_double_support_json']))
            verify(ref,ref.absolute_curve,rule)
            result['programme_double_support_json']=packet['programme_double_support_json']
            result['joint_source_fit_json']=packet['joint_source_fit_json']
        np.testing.assert_allclose(result['native22_world_mid'],state['native_world']*ratio,rtol=0,atol=1e-9)
        np.testing.assert_allclose(result['skate_origin_world_mid'],state['O']*ratio,rtol=0,atol=1e-9)
        np.testing.assert_array_equal(result['source_pts'],pts)
        pending=dest.with_suffix('.pending.npz')
        np.savez_compressed(pending,**result)
        pending.replace(dest)
    else:
        with np.load(dest) as z:
            assert str(z['transfer_parameter_sha256'])==digest(current)
    scene=info['scene_binding']; field=Path(scene['structure']['path'])
    structure=json.loads(field.read_text(encoding='utf8'))
    model=SceneParameterModel(structure,field,optimize_scene=True)
    with np.load(scene['path']) as z: state=ParameterState(model.initial_state.layout,z['scene_values'])
    native=NativeSceneGPU(field,cam.core,parameter_state=state)
    samples=sample_native_review_scene(structure,cam.core,field,parameter_state=state,native=native)
    assert set(native.paths)=={cut['id'] for cut in samples['cuts']}
    folder=output/'scene';folder.mkdir(exist_ok=True)
    np.savez_compressed(folder/'review_scene.npz',xyz=samples['xyz'])
    write(folder/'review_scene.json',dict(cuts=samples['cuts'],inventory=samples['inventory'],
        unlocalized=samples['missing_entities'],camera_state_sha256=info['state_sha256'],scene_values_sha256=info['scene_values_sha256']))
    np.savez(folder/'camera_snapshot.npz',board=native.scene['board'],scene_values_sha256=info['scene_values_sha256'])
    write(folder/'metric_binding.json',dict(metres_per_scene_unit=info['metres_per_scene_unit'],uniform_length_ratio=ratio,
        snapshot=scene['path'],source_span_xy_scene_units=(2*np.asarray(native.scene['board'][:2])).tolist(),
        policy='equal_weight_squared_dimension_error',verified_metre_scale=False))
    config=json.loads(config_base.read_text(encoding='utf8')); variant='okada_body_s14_1118'
    joint_refit='joint_source_fit_json' in packet
    if joint_refit:variant='okada_body_joint_1119'
    world_refit=joint_refit and json.loads(str(packet['joint_source_fit_json'])).get('stage')=='world_dynamics_refit'
    if world_refit:variant='okada_body_world_1119'
    fit_info=json.loads(str(packet['joint_source_fit_json'])) if joint_refit else {}
    stature=fit_info if fit_info.get('stage')=='shared_stature_experiment' else None
    limb_review=fit_info.get('stage')=='source_support_and_limbs'
    if limb_review:variant='okada_support_limbs_1119'
    sitspin=fit_info.get('stage')=='sitspin_refit'
    if sitspin:variant='okada_sitspin_1119'
    if sitspin and fit_info.get('native_slip_scale'):variant='okada_sitspin_source_1119'
    spin_pair=fit_info.get('stage')=='paired_spin_refit'
    if spin_pair:variant='okada_spin_pair_1119'
    if stature:variant='okada_height_1119_'+stature['mode']
    shots=list(dict.fromkeys(cam.shot_ids.tolist()))
    ranges=[[int(pts[cam.shot_ids==s][0]),int(pts[cam.shot_ids==s][-1])] for s in shots]
    reg=json.loads(registration.read_text(encoding='utf8'))
    v=dict(id=variant,candidate_id=variant,kind='fullbody',shell_topology='saved',label='冈田芽衣 · 11 月 18 日标定',
        result=str(dest),result_sha256=digest(dest),parameter=str(current),parameter_sha256=digest(current),
        user_accepted=False,accuracy_accepted=False,required_saved_diagnostics=['com_world_mid'],
        body_reference_label='11 月 17 日人体',body_refinement_intervals_pts=ranges,
        body_refinement_intervals_seconds=(np.asarray(ranges)/12800.).tolist(),
        camera_binding=dict(record=checked_entry(registration),state=reg['camera'],scene=scene,
            shot_state_sha256={s:info['state_sha256'] for s in shots},frozen_camera_sha256=info['state_sha256']))
    if joint_refit:
        v.update(label='冈田芽衣 · 全节目人体与鞋刀联合重拟合',body_reference_label='11 月 18 日相机迁移版')
        config.update(review_status='开场双刀持续支撑由用户确认；其余遮挡处为单机位三维估计。',
                      source_review_model_trajectory_default=False)
        config['initial_visible'].update(trajectory=False,com_projection=False,com=False)
    if world_refit:
        v.update(label='冈田芽衣 · 全节目世界坐标运动精修',body_reference_label='11 月 19 日人体与鞋刀联合版')
        config['initial_visible'].update(com=True,com_projection=True)
    if stature:
        mode_label={'free':'全节目共享身高','fixed':'固定身高复算'}[stature['mode']]
        v.update(label=f"冈田芽衣 · {mode_label} {stature['nominal_cm']:.2f} cm",
                 body_reference_label='固定 153 cm 三维运动版')
        config['initial_visible'].update(com=True,com_projection=True)
    if limb_review:
        v.update(label='冈田芽衣 · 支撑刀、膝与手臂精修',body_reference_label='固定 153 cm 全节目复算版')
        config['initial_visible'].update(com=True,com_projection=True)
    if sitspin:
        v.update(label='冈田芽衣 · 末段蹲转左刀精修',body_reference_label='固定 153 cm 支撑刀与四肢精修版')
        config['initial_visible'].update(com=True,com_projection=True)
    if spin_pair:
        v.update(label='冈田芽衣 · 躬身转与蹲转支撑左刀',body_reference_label='固定 153 cm 末段蹲转左刀精修版')
        config['initial_visible'].update(com=True,com_projection=True)
    config.update(variants=[v],comparison_variants=[variant],session_id=variant,default_variant=variant,latest_review_variant=variant,
        shot=variant,initial_source_pts=int(pts[0]),scene_root=str(folder),metric_binding=str(folder/'metric_binding.json'),
        camera_snapshot=str(folder/'camera_snapshot.npz'),scene_label='安卡拉 · lane4_side_1118',
        source_video=reg['source']['path'],body_scope_label='S01–S09 全节目 · s14_boards_1118',
        source_body_observations_default=False,initial_view='side')
    config['initial_visible'].update(boards=True,rink_lines=True)
    write(output/'viewer3d.json',config)
    shutil.copyfile(config_base.parent/'START_REFITTED_3D.cmd',output/'START_REFITTED_3D.cmd')
    print('Saved',dest,output/'viewer3d.json',flush=True)

if __name__=='__main__':
    import argparse
    from pathlib import Path
    parser=argparse.ArgumentParser()
    for field in ('prepared','output','registration','config_base','ledger'):parser.add_argument(field,type=lambda p:Path(p).resolve())
    export(**vars(parser.parse_args()))
