"""Evaluate the landing continuation against M with identical observations."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from calibration.program_tracking.body_tracking.phase_motion_quality import joint_domain_changes
from calibration.program_tracking.body_tracking.programme_surface import CollisionSurface
from calibration.program_tracking.body_tracking.scoped_motion_domains import local_surface_measurements
from calibration.program_tracking.body_tracking.scoped_motion_step import skate_projection
from calibration.program_tracking.body_tracking.sequence_skate_state import SequenceState
from calibration.program_tracking.body_tracking.skate_contact import trapezoid_time_weights
from compare import bind, changed_support, distribution, load, ranges, rotation_step, stage_distributions, write
from motion_compare import MOTION, SHARED, SOURCE, read_fields
from phase_compare import phase_flight, saved_surface_comparison, shoe_measurements
from phase_playback import bound_viewer, local_video
from source_review import digest, read


NEW = 'okada_landing_motion_1115'
IDS = ('okada_phase_motion_1115', NEW)
WINDOWS = dict(action=(80.88,82.70),flight=(81.36,81.48),landing=(81.50,81.60),
               landing_and_exit=(81.48,81.70),initial_spin=(81.62,82.46),knee_and_ankle=(81.14,81.78))
CHECKPOINT_FIELDS = ('base_skate_parameters_json','R_FB_initial','t_FB_initial_solver',
    'fixed_boot_geometry_json','relative_skate_rotation_controls','absolute_knots','whole_skate_match_sha256',
    'prepared_input_sha256','prepared_observations_sha256','source_evidence_sha256','objective_policy_sha256')


def compare(prepared, source, stages, study, targets, *, fit_only=False):
    saved = {k:read_fields(v/'fitted/result.npz',(*SOURCE,*SHARED,*MOTION,'com_world_mid'))
             for k,v in stages.items()}
    m,new = saved.values()
    for name in (*SOURCE,*SHARED):
        np.testing.assert_array_equal(m[name],new[name],err_msg=name)
    pts = m['source_pts']; shot = m['source_shot_ids']; times = pts/12800.
    assert len(pts)==8329 and len(np.unique(shot))==9 and np.all(np.diff(pts)==256)
    use=m['source_observation_objective_use']
    masks={k:(times>=lo)&(times<=hi) for k,(lo,hi) in WINDOWS.items()}
    masks.update({str(k):shot==k for k in np.unique(shot)})
    study = json.loads(read(study))
    if fit_only:
        study['samples'] = [r for r in study['samples'] if r['role']=='fit']
    out=dict(exposures=len(pts),observation_rows=int(use.sum()),source_time_base=m['time_base'].tolist(),
        observation_sha256=str(m['source_observation_ledger_sha256']),source_definition_sha256=digest(source),
        observations_and_shared_geometry_equal=True,windows=WINDOWS,fit_evidence_only=fit_only,
        states={},accuracy_accepted=False,user_accepted=False)
    anatomy={}; packets={}
    for k,path in stages.items():
        arr=saved[k]
        ref,camera,info,document,packet=bind(prepared,source,path/'solve/candidate.npz')
        packets[k]=packet
        a=read_fields(path/'anatomical_frames.npz',('joint_names','normalized_violation','source_pts','anatomical_basis','rest_R','reference_twist'))
        np.testing.assert_array_equal(pts,a['source_pts'])
        anatomy[k]=a
        pixels=arr['COCO_uv'][np.arange(len(pts))[:,None],m['source_body_model_joint_ids']]
        error=np.linalg.norm(pixels-m['source_uv_native'],axis=-1)
        physical,grid=phase_flight(ref,packet['physics_times'],packet,arr,WINDOWS)
        quad=read_fields(path/'com_physics_quadrature.npz',('evaluation_time_s','acceleration_source_pts','acceleration_world_mps2'))
        np.testing.assert_array_equal(grid,quad['evaluation_time_s'])
        transitions=quad['acceleration_source_pts']/12800.
        region=(transitions>=81.48-1.e-10)&(transitions<=81.70+1.e-10)
        phase=quad['acceleration_world_mps2'][region]
        clock=quad['acceleration_source_pts'][region]
        native_phase=arr['com_acceleration_world_mps2'][masks['landing_and_exit']]
        dense_weight=trapezoid_time_weights(clock/12800.)
        native_steps=np.linalg.norm(np.diff(arr['skate_origin_world_mid'],axis=0),axis=-1)
        steps=masks['landing_and_exit'][1:]
        start=np.searchsorted(pts,1041664)
        poses=dict(windows={name:dict(exposures=int(take.sum()),
                body_reprojection_px=distribution(error[take],pts[take],use[take]),
                structure=stage_distributions(arr,a,pts,take)) for name,take in masks.items()},
            physics=physical, dense_landing_trace=dict(pts=clock.tolist(),acceleration_mps2=phase.tolist(),
                time_weighted_rms_mps2=float(np.sqrt(np.average((phase**2).sum(axis=1),weights=dense_weight))),
                native_acceleration_change_rms_mps2=float(np.sqrt(np.mean((np.diff(native_phase,axis=0)**2).sum(axis=1))))),
            landing_skate_steps=dict(end_pts=pts[1:][steps].tolist(),left_right_m=native_steps[steps].tolist()),
            left_knee_step_at_81_38_deg=float(rotation_step(arr['native_body_local_R_mid'][start-1:start+1])[0,3]),
            shoe_source=shoe_measurements(ref,camera,packet,study),knee_source=[],shoe_protection=[])
        state=SequenceState(ref,packet['seconds'])
        context=SimpleNamespace(reference=ref,camera=camera,binding=ref.binding,body=SimpleNamespace(state=state))
        shoe_rows=targets['protected_skate_regions']
        actual,_=skate_projection(context,np.zeros(17+state.size),shoe_rows)
        for row,uv in zip(shoe_rows,actual,strict=True):
            lo,hi=np.asarray(row['bounds_uv']); outside=np.maximum(0.,np.maximum(lo-uv,uv-hi))
            poses['shoe_protection'].append(dict(**row,uv=uv.tolist(),outside_px=float(np.linalg.norm(outside))))
        for row in targets['image_regions']:
            lo,hi=np.asarray(row['bounds_uv']); uv=arr['COCO_uv'][np.searchsorted(pts,row['pts']),row['joint']]
            poses['knee_source'].append(dict(pts=row['pts'],joint=row['joint'],uv=uv.tolist(),
                outside_px=float(np.linalg.norm(np.maximum(0.,np.maximum(lo-uv,uv-hi))))))
        out['states'][k]=poses
        if k=='M':
            out.update(camera_sha256=info['state_sha256'],scene_sha256=info['scene_values_sha256'])
            geometry=CollisionSurface(ref,document)
        else:
            assert out['camera_sha256']==info['state_sha256'] and out['scene_sha256']==info['scene_values_sha256']
    for key in CHECKPOINT_FIELDS:
        np.testing.assert_array_equal(packets['M'][key],packets['L'][key],err_msg=key)
    np.testing.assert_array_equal(packets['M']['x'][:17],packets['L']['x'][:17])
    a,b=anatomy.values()
    for key in ('joint_names','anatomical_basis','rest_R','reference_twist'):
        np.testing.assert_array_equal(a[key],b[key],err_msg=key)
    native=changed_support(packets['M'],packets['L'],ref.times)
    rolling=ref.times[:,None]+camera.readout_times(np.arange(len(pts)))[:,None]*np.linspace(-.5,.5,5)
    exposure=changed_support(packets['M'],packets['L'],rolling.ravel()).reshape(-1,5).any(axis=1)
    midpoint=changed_support(packets['M'],packets['L'],(ref.times[:-1]+ref.times[1:])/2)
    joint_dist=1000*np.linalg.norm(new['native22_world_mid']-m['native22_world_mid'],axis=-1)
    shoe_dist=1000*np.linalg.norm(new['skate_origin_world_mid']-m['skate_origin_world_mid'],axis=-1)
    for actual,active in ((new['COCO_uv']-m['COCO_uv'],exposure),(joint_dist,native),(shoe_dist,native)):
        np.testing.assert_allclose(actual[~active],0.,rtol=0.,atol=2.e-7)
    cuts=np.flatnonzero(shot[1:]!=shot[:-1])+1
    edges=np.flatnonzero(np.diff(native.astype(int)))+1
    ratings={k:dict(rotation=rotation_step(z['native_body_local_R_mid']),
                    shoe=np.linalg.norm(np.diff(z['skate_origin_world_mid'],axis=0),axis=-1),
                    shoe_rotation=rotation_step(z['skate_R_world_mid'])) for k,z in saved.items()}
    out.update(native_support=ranges(native,pts),image_support=ranges(exposure,pts),
        changed_midpoints=int(midpoint.sum()),maximum_joint_motion_mm=float(joint_dist.max()),
        maximum_shoe_motion_mm=float(shoe_dist.max()),
        cut_review=[dict(pts=pts[i-1:i+1].tolist(),joint_change_mm=float(joint_dist[i-1:i+1].max()),
                         shoe_change_mm=float(shoe_dist[i-1:i+1].max())) for i in cuts],
        boundary_review=[dict(pts=pts[i-1:i+1].tolist(),states={k:dict(
            peak_local_rotation_deg=float(v['rotation'][i-1].max()),shoe_step_mm=(1000*v['shoe'][i-1]).tolist(),
            shoe_rotation_deg=v['shoe_rotation'][i-1].tolist()) for k,v in ratings.items()}) for i in edges],
        joint_changes=joint_domain_changes(pts,a['joint_names'],a['normalized_violation'],b['normalized_violation']))
    out['surface']=saved_surface_comparison(stages,geometry,baseline='M')
    for key in stages:
        archive=out['surface'][key]['refreshed_guard']['surface_path']
        cuff=[dict(parts=[side+'_shank',side+'_boot'],source_seconds=t,max_penetration_m=0.)
              for side,stamps in (('right',(81.36,81.38,81.42)),('left',(81.59,81.60,81.61,81.62))) for t in stamps]
        out['surface'][key]['specific_cuff_checks']=local_surface_measurements(load(archive),geometry,cuff)
    inherited=targets.get('phase_quality',{}).get('joint_reference')
    if inherited is not None:
        assert digest(inherited['path'])==inherited['sha256']
        history=read_fields(inherited['path'],('source_pts','joint_names','normalized_violation'))
        np.testing.assert_array_equal(pts,history['source_pts'])
        np.testing.assert_array_equal(a['joint_names'],history['joint_names'])
        mask=masks['knee_and_ankle']
        out['inherited_joint_abnormalities']={k:joint_domain_changes(pts[mask],a['joint_names'],
            history['normalized_violation'][mask],state['normalized_violation'][mask]) for k,state in anatomy.items()}
    return out


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('data','prepared','source','M','L','shoe-study','targets','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--fit-only',action='store_true')
    parser.add_argument('--playback',type=Path)
    parser.add_argument('--video',action='store_true')
    args=parser.parse_args()
    stages=dict(M=args.M,L=args.L)
    targets=json.loads(read(args.targets))
    result=compare(args.prepared,args.source,stages,args.shoe_study,targets,fit_only=args.fit_only)
    write(args.output,result)
    if args.playback:
        scope,original=bound_viewer(args.data,stages,args.playback,result['native_support'],1041664,
            ids=IDS,labels=('M · 相位重拟合基线','L · 落冰续算'),scope_label='S05 入旋与落冰 · M / L')
        if args.video:
            media=local_video(args.prepared,args.source,stages,scope,original,args.playback,(80.88,82.70),
                'landing_M_L',labels=('Original exposure','M  Phase baseline','L  Landing refinement'))
            write(args.playback/'media.json',dict(source_sha256=scope['source']['sha256'],videos=[media],
                parameter_sha256={code:digest(run/'solve/candidate.npz') for code,run in stages.items()},
                common_native_camera=True,display_alignment=False))


if __name__=='__main__':
    main()
