"""Reproduce the same-source, zero-update versus fitted motion comparison."""
import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from compare import (bind, changed_support, distribution, load, load_surface, ranges,
                     rotation_step, skate_reprojection, stage_distributions, write)
from source_review import PROGRAMME, PROJECT, baseline, digest, read
from calibration.program_tracking.body_tracking.scoped_motion_step import skate_projection
from calibration.program_tracking.body_tracking.scoped_motion_domains import local_surface_measurements
from calibration.program_tracking.body_tracking.programme_surface import CollisionSurface
from calibration.program_tracking.body_tracking.sequence_skate_state import SequenceState


PARENT = PROGRAMME/'fit_v11/body_identity_1114.json'
NEW = 'okada_trusted_motion_1115'
REPLAY = 'okada_trusted_motion_1115_replay'
SHARED = ('shared_betas', 'shared_body_scale', 'shared_geometry', 'metres_per_scene_unit',
          'camera_binding_json', 'footwear_binding_json', 'skate_dimensions_json', 'skate_relative_R_mid')
SOURCE = ('source_pts', 'time_base', 'source_shot_ids', 'source_uv_native', 'source_sigma_px',
          'source_body_model_joint_ids', 'source_observation_objective_use', 'source_observation_ledger_sha256',
          'source_support_mode_mid', 'sequence_contact_modes', 'sequence_contact_status')
MOTION = ('COCO_uv', 'native22_world_mid', 'P_world_mid', 'native_body_local_R_mid', 'Q_world_mid',
          'skate_R_world_mid', 'skate_origin_world_mid', 'foot_cavity_max_violation_m_mid',
          'skate_clearance_m_mid', 'heel_drift_mid', 'forefoot_error_mid', 'com_acceleration_world_mps2')


def read_fields(path, keys):
    with np.load(path, allow_pickle=False) as z:
        return {k:z[k] for k in keys}


def states(data, replay, fit):
    scope, paths, _ = baseline(data, PARENT)
    rpath, npath = [stage/'fitted/result.npz' for stage in (replay, fit)]
    r, n = [read_fields(path, (*SOURCE, *SHARED, *MOTION)) for path in (rpath, npath)]
    for key in (*SOURCE, *SHARED):
        np.testing.assert_array_equal(r[key], n[key], err_msg=key)
    a = read_fields(paths['result'], (*SOURCE, 'COCO_uv', 'native22_world_mid', 'skate_origin_world_mid'))
    for key in ('COCO_uv','native22_world_mid','skate_origin_world_mid'):
        np.testing.assert_allclose(a[key], r[key], atol=2.e-8, rtol=0., err_msg=key)
    return scope, paths, a, r, n


def comparison(data, prepared, inputs, replay, fit, output, regions):
    scope, paths, a, r, n = states(data, replay, fit)
    r_control, n_control = [load(stage/'solve/candidate.npz') for stage in (replay,fit)]
    parent = load(paths['solve_state'])
    np.testing.assert_array_equal(parent['x'], r_control['x'])
    np.testing.assert_array_equal(r_control['x'][:17], n_control['x'][:17])
    np.testing.assert_array_equal(r_control['relative_skate_rotation_controls'], n_control['relative_skate_rotation_controls'])
    for z in (r_control,n_control):
        if str(z['prepared_observations_sha256']) != digest(inputs/'observations.npz'):
            raise ValueError('Saved fit has stale source observations')
    pts, shots = r['source_pts'], r['source_shot_ids']
    if len(pts) != 8329 or np.any(np.diff(pts) != 256):
        raise ValueError('Expected the 8,329 registered exposures on the original clock')
    ref, camera, camera_info, document, controls = bind(prepared, inputs/'source.json', fit/'solve/candidate.npz')
    before_skate = read_fields(paths['solve_state'].parent.parent/'skate_source_replay.npz',
                               ('source_pts','outline_median_distance_px_C','boot_chord_distance_px_C'))
    np.testing.assert_array_equal(before_skate['source_pts'], pts)
    next_skate = skate_reprojection(ref,camera,document,controls)
    old_skate = {k:before_skate[k+'_C'] for k in next_skate}
    native_support = changed_support(r_control,n_control,ref.times)
    shutter = ref.times[:,None]+camera.readout_times(np.arange(len(pts)))[:,None]*np.linspace(-.5,.5,5)
    image_support = changed_support(r_control,n_control,shutter.ravel()).reshape(-1,5).any(axis=1)
    midpoint = (ref.times[:-1]+ref.times[1:])/2
    mid_support = changed_support(r_control,n_control,midpoint)
    joint_delta = np.linalg.norm(n['native22_world_mid']-r['native22_world_mid'],axis=-1)*1000
    shoe_delta = np.linalg.norm(n['skate_origin_world_mid']-r['skate_origin_world_mid'],axis=-1)*1000
    body_px_delta = np.linalg.norm(n['COCO_uv']-r['COCO_uv'],axis=-1)
    for value, support in ((joint_delta,native_support),(shoe_delta,native_support),(body_px_delta,image_support)):
        np.testing.assert_allclose(value[~support],0,atol=2.e-7,rtol=0.)

    use = r['source_observation_objective_use']
    model = r['source_body_model_joint_ids']
    common = use & a['source_observation_objective_use'] & (model==a['source_body_model_joint_ids'])
    frame = np.arange(len(pts))[:,None]
    error = {k:np.linalg.norm(z['COCO_uv'][frame,model]-r['source_uv_native'], axis=-1)
             for k,z in (('R',r),('N',n))}
    anatomy = {k:read_fields(stage/'anatomical_frames.npz',('joint_names','normalized_violation'))
               for k,stage in (('R',replay),('N',fit))}
    surf_receipt, surf = {}, {}
    parent_stage = paths['solve_state'].parent.parent
    # The production zero-update export reuses the identical archive at the
    # registered parent vector. There is no new surface trial in that run.
    for k,stage in (('R',parent_stage),('N',fit)):
        surf_receipt[k],surf[k] = load_surface(stage)
        surf[k].update(read_fields(surf_receipt[k]['surface_path'],('pair_counts','parts')))
    np.testing.assert_array_equal(surf['R']['source_seconds'],surf['N']['source_seconds'])
    np.testing.assert_array_equal(surf['R']['parts'],surf['N']['parts'])
    surface_support=np.empty(2*len(pts)-1,dtype=bool)
    surface_support[::2],surface_support[1::2]=native_support,mid_support
    affected=np.flatnonzero(surface_support|(surf['R']['crossing_counts']!=surf['N']['crossing_counts']))
    def pair_counts(stage):
        rows=stage['pair_counts']
        take=np.isin(rows[:,0],affected)
        return {tuple(map(int,r[:3])):int(r[3]) for r in rows[take]}
    old_pairs,new_pairs=(pair_counts(surf[k]) for k in ('R','N'))
    surface_pairs=[dict(source_seconds=float(surf['R']['source_seconds'][i]),
                        parts=surf['R']['parts'][[a,b]].tolist(),crossing_pairs=new_pairs[(i,a,b)])
                   for i,a,b in sorted(new_pairs.keys()-old_pairs.keys())]
    local = {k:rotation_step(z['native_body_local_R_mid']) for k,z in (('R',r),('N',n))}
    shoe_step = {k:np.linalg.norm(np.diff(z['skate_origin_world_mid'],axis=0),axis=-1)
                 for k,z in (('R',r),('N',n))}

    def summary(take):
        asec = surf['R']['source_seconds']
        surface = (asec >= pts[take][0]/12800) & (asec <= pts[take][-1]/12800)
        return dict(exposures=int(take.sum()), pts=[int(pts[take][0]),int(pts[take][-1])],
            joint_motion_mm=distribution(joint_delta[take],pts[take]), shoe_motion_mm=distribution(shoe_delta[take],pts[take]),
            image_motion_px=distribution(body_px_delta[take],pts[take]),
            reprojection={k:distribution(v[take],pts[take],use[take]) for k,v in error.items()},
            knee_reprojection={k:distribution(v[take,13:15],pts[take],use[take,13:15]) for k,v in error.items()},
            shoe_source={k:{name:distribution(v[take],pts[take]) for name,v in s.items()}
                         for k,s in (('R',old_skate),('N',next_skate))},
            structure={k:stage_distributions(z,anatomy[k],pts,take) for k,z in (('R',r),('N',n))},
            surface={k:{f:int(s[f][surface].sum()) for f in ('crossing_counts','grip_crossing_counts','topology_uncertain_candidates')}
                     for k,s in surf.items()},
            new_surface_crossing_samples=int(np.sum(surf['N']['crossing_counts'][surface]>surf['R']['crossing_counts'][surface])))

    core = (pts>=1038592)&(pts<=1044736)
    cuts = []
    for i in np.flatnonzero(shots[1:]!=shots[:-1])+1:
        cuts.append(dict(pts=pts[i-1:i+1].tolist(), shots=shots[i-1:i+1].tolist(),
            pose_change_mm=float(joint_delta[i-1:i+1].max()), shoe_change_mm=float(shoe_delta[i-1:i+1].max()),
            same_source_shutter_support=bool(not image_support[i-1:i+1].any()),
            body_cross_cut_step_m={k:float(np.max(np.linalg.norm(z['native22_world_mid'][i]-z['native22_world_mid'][i-1],axis=-1)))
                                   for k,z in (('R',r),('N',n))}))
    assert len(cuts)==8
    transitions = np.flatnonzero(np.diff(native_support.astype(int)))+1
    edges = [dict(pts=int(pts[i]), previous_pts=int(pts[i-1]),
                  endpoint_joint_motion_mm=joint_delta[i-1:i+1].max(1).tolist(),
                  local_step_deg={k:float(x[i-1].max()) for k,x in local.items()},
                  shoe_step_mm={k:(1000*x[i-1]).tolist() for k,x in shoe_step.items()}) for i in transitions]
    regions = json.loads(read(regions))
    def region(rgn):
        i = np.searchsorted(pts, rgn['pts']); j=rgn['joint']
        lo, hi = np.asarray(rgn['bounds_uv'])
        table = {}
        for k,z in (('R',r),('N',n)):
            point = z['COCO_uv'][i,j]
            table[k] = dict(uv=point.tolist(), excess_px=float(np.linalg.norm(np.maximum(np.maximum(lo-point,point-hi),0.))))
        return dict(**rgn,**table)
    revised = a['source_observation_objective_use'] & ~use
    rebound = (model!=a['source_body_model_joint_ids']) & use
    error_a = np.linalg.norm(a['COCO_uv'][frame,a['source_body_model_joint_ids']]-r['source_uv_native'],axis=-1)
    report = dict(source_exposures=len(pts), source_time_base=r['time_base'].tolist(),
        same_R_N_source_identities=True, R_parameters_equal_parent=True, camera_sha256=camera_info['state_sha256'],
        observations_sha256=str(r['source_observation_ledger_sha256']),
        exact_support=dict(native=ranges(native_support,pts), shutter=ranges(image_support,pts),
                           midpoint_samples=int(mid_support.sum())),
        retired_rows=dict(count=int(revised.sum()), original_pose_error=distribution(error_a,pts,revised)),
        remapped_rows=dict(count=int(rebound.sum()), original_pose_original_identity=distribution(error_a,pts,rebound),
                           original_pose_revised_identity=distribution(error['R'],pts,rebound)),
        same_point_population={k:distribution(v,pts,common) for k,v in error.items()},
        programme=summary(np.ones(len(pts),bool)), motion_scope=summary(core),
        shots={s['id']:summary(shots==s['id']) for s in scope['shots']}, eight_cuts=cuts, support_edges=edges,
        projected_source_regions=[region(row) for row in regions['image_regions']],
        left_knee_steps={k:{str(int(pts[i])):float(w[i-1,3]) for i in np.flatnonzero(core)} for k,w in local.items()},
        root_step_deg={k:distribution(rotation_step(z['Q_world_mid'])[core[1:]&core[:-1]],pts[1:][core[1:]&core[:-1]])
                       for k,z in (('R',r),('N',n))}, surface=surf_receipt, new_surface_pairs=surface_pairs,
        user_accepted=False, accuracy_accepted=False)
    protected=regions.get('protected_skate_regions',[])
    if protected:
        old_ref,old_camera,_,_,old_controls=bind(prepared,inputs/'source.json',replay/'solve/candidate.npz')

        def outsole(ref, camera, control):
            state=SequenceState(ref,control['seconds'])
            p=SimpleNamespace(reference=ref,camera=camera,binding=ref.binding,body=SimpleNamespace(state=state))
            return skate_projection(p,np.r_[ref.binding.x,np.zeros(state.size)],protected)[0]

        before,after=outsole(old_ref,old_camera,old_controls),outsole(ref,camera,controls)
        def projection_stats(row, point):
            lo,hi=row['bounds_uv']
            return dict(projected_uv=point.tolist(),excess_px=float(np.linalg.norm(
                np.maximum(np.maximum(lo-point,point-hi),0.))))
        report['protected_skate_projections']=[dict(**row,R=projection_stats(row,p0),N=projection_stats(row,p1))
            for row,p0,p1 in zip(protected,before,after,strict=True)]
    collar_limits=regions.get('local_surface_caps',[])
    if collar_limits:
        geometry=CollisionSurface(ref,document)
        scans={k:read_fields(receipt['surface_path'],('face_a','face_b','source_seconds','sample','initial_gap'))
               for k,receipt in surf_receipt.items()}
        report['collar_surface_samples']={k:local_surface_measurements(
            scan,geometry,collar_limits) for k,scan in scans.items()}
        limits=[dict(parts=row['parts'],source_seconds=row['source_seconds'],max_penetration_m=0.)
                for row in surface_pairs]
        for row,measured in zip(surface_pairs,local_surface_measurements(scans['N'],geometry,limits),strict=True):
            row['directed_witness_mm']=1000*measured['measured_maximum_m']
    output.mkdir(parents=True,exist_ok=False)
    write(output/'comparison.json',report)
    with (output/'exposures.csv').open('x',encoding='utf-8',newline='') as out:
        writer=csv.writer(out)
        writer.writerow(('pts','shot','source_seconds','active_points','support','body_change_mm','shoe_change_mm',
                         'R_p90_px','N_p90_px','R_left_knee_step_deg','N_left_knee_step_deg',
                         'R_left_gap_m','R_right_gap_m','N_left_gap_m','N_right_gap_m'))
        for i,stamp in enumerate(pts):
            vals=[float(np.percentile(error[k][i,use[i]],90)) if use[i].any() else '' for k in ('R','N')]
            steps=[float(local[k][i-1,3]) if i else '' for k in ('R','N')]
            writer.writerow([stamp,shots[i],stamp/12800,int(use[i].sum()),int(native_support[i]),
                joint_delta[i].max(),shoe_delta[i].max(),*vals,*steps,*r['skate_clearance_m_mid'][i],*n['skate_clearance_m_mid'][i]])
    np.savez_compressed(output/'skate_projection.npz',source_pts=pts,
                        **{k+'_R':v for k,v in old_skate.items()}, **{k+'_N':v for k,v in next_skate.items()})
    return report


def viewer(data, replay, fit, output, report):
    _, paths, _ = baseline(data,PARENT)
    config=json.loads(read(paths['viewer_config']))
    previous=config['variants'][0]
    variants=[]
    for stage, key, label in ((replay,REPLAY,'R · 旧动作零更新'),(fit,NEW,'N · 约束内重拟合')):
        result, parameter = stage/'fitted/result.npz',stage/'solve/candidate.npz'
        variant=dict(previous,id=key,candidate_id=key,label=label,result=str(result),result_sha256=digest(result),
            parameter=str(parameter),parameter_sha256=digest(parameter),blade_review_available=False,
            body_reference_label=REPLAY,candidate_scope_label='全节目，同源 PTS，未验收')
        spans=report['exact_support']['native'] if key==NEW else []
        variant['body_refinement_intervals_seconds']=[r['seconds'] for r in spans]
        variant['body_refinement_intervals_pts']=[[r['first_pts'],r['last_pts']] for r in spans]
        variants.append(variant)
    config.pop('blade_review_asset',None)
    config.update(variants=variants,session_id=NEW,comparison_variants=[REPLAY,NEW],default_variant=NEW,
                  latest_review_variant=NEW,initial_source_pts=1039104,source_body_observations_default=True,
                  source_review_model_body_default=False,source_review_model_runner_default=False,
                  body_scope_label='S05 入旋同源对照 · R / N')
    nav=json.loads(read(config['source_navigation']))
    nav['events'].append(dict(kind='reviewed_motion_scope',start_pts=1038592,end_pts=1044736,worst_pts=1039104,
                              label='81.14–81.62 秒 · 进旋、腾空和落冰'))
    nav['events'].sort(key=lambda event:event['start_pts'])
    write(output/'navigation.json',nav)
    config['source_navigation']=str(output/'navigation.json')
    write(output/'viewer3d.json',config)
    with (output/'START_TRUSTED_MOTION_3D.cmd').open('xb') as stream:
        stream.write(paths['viewer'].read_bytes())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for k in ('data','prepared','inputs','replay','fit','output','regions'):
        parser.add_argument(k,type=lambda value:Path(value).resolve())
    a=parser.parse_args()
    report=comparison(a.data,a.prepared,a.inputs,a.replay,a.fit,a.output,a.regions)
    viewer(a.data,a.replay,a.fit,a.output,report)


if __name__=='__main__':
    main()
