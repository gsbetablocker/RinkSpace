"""Read-only causal value probe of the frozen secondary programme objective.

Uses existing owned source preparation and exact value-only factor interfaces.
No optimizer, Jacobian, global normal, worker pool, training or renderer is run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

PROJECT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(PROJECT), str(PROJECT / 'src')]
sys.dont_write_bytecode = True


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--output', type=Path, required=True)
    cli.add_argument('--export-conflict-only', action='store_true')
    opt = cli.parse_args()
    out = opt.output.resolve()
    allowed = Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/diagnostics/external_expert_20261007').resolve()
    if not out.is_relative_to(allowed):
        raise ValueError('Output must stay in the explicitly authorized diagnostic directory')
    out.mkdir(parents=True, exist_ok=True)
    for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[key] = '1'
    os.environ.update(RINKSPACE_BENCHMARK_DIAGNOSTIC_DIR=str(out),
                      RINKSPACE_BENCHMARK_TIMINGS_PATH=str(out/'preparation_timings.json'),
                      RINKSPACE_PATTERN_WORKERS='0', RINKSPACE_MOTION_THREADS='1',
                      RINKSPACE_SCATTER_THREADS='1')
    from calibration.solver_runtime import install
    install()
    import numpy as np
    import cv2
    from run_shared_print_pairs_and_mount_full_source_candidate import prepare, DEFAULT_SEED
    from advertisement_source_binding_candidate.corrected_advertisement_source_geometry import install_corrected_advertisement_source_geometry
    from shared_ice_hockey_geometry_candidate.hockey_graph import install_shared_ice_hockey_geometry
    from shared_unmeasured_advertisement_geometry_candidate.paired_locations_shared_sizes_geometry import PairedLocationsSharedSizesGeometry
    from simple_wide_perspective_reset_candidate.audit_structured_current_axis_reference import make_structured_geometry
    from structured_programme_source_candidate.paired_programme_blocks import install_structured_programme_blocks
    from structured_programme_source_candidate.paired_programme_chart import CORE_COUNT, WORLD_INDICES
    from user_corrected_programme_candidate.initialize_latest9a8c_world66 import apply_latest_geometry_auxiliary
    from structured_programme_source_candidate.course_station_cache import install_exact_course_station_cache
    from structured_programme_source_candidate.current_ice_material_scene import install_source_ice_material_view
    from z190_joint_source_solver import make_profile_update, _frozen_profile_token, source_cost
    from structured_programme_source_candidate.original_ice_pixel_correction import install_original_ice_pixel_correction
    from structured_programme_source_candidate.native_ice_curve_source import NativeIceCurveFactor
    from structured_programme_source_candidate.native_board_edge_source import NativeBoardEdgeFactor
    from structured_programme_source_candidate.native_board_circle_source import NativeBoardCircleFactor
    from z190_motion_native import metadata as motion_metadata

    runs = Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/calibration/runs/solver_acceleration_20261003')
    run = runs/'source_corrected_RF_complete_joint_20261007'
    manifest_path = run/'input_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf8'))
    paths = {
        'seed_3311': Path(manifest['camera_seed']['path']),
        'initial_9bd': run/'initial_states/9bd499b683371e10ea0dc4b7d7c3e1807ac15a0bfde2fd03ccc21f8e4c0f68bd/complete_state_UNACCEPTED.npz',
        'accepted_a8d': run/'accepted_states/a8d2a76a97862dd2278afbd0af0477892cad167a22df8f5ae370b5cc9e1f2b96/complete_state_UNACCEPTED.npz',
    }
    frozen_path = run/'actual_initial_frozen_profile_arrays.npz'
    saved = {}
    for label, path in paths.items():
        with np.load(path, allow_pickle=False) as z:
            saved[label] = {k: z[k].copy() for k in z.files if k in ('q', 'quadratic', 'scales') or k.startswith('profile__')}
        saved[label]['path'] = str(path)
        saved[label]['file_sha256'] = sha(path)
        saved[label]['q_sha256'] = hashlib.sha256(saved[label]['q'].tobytes()).hexdigest()
    with np.load(frozen_path, allow_pickle=False) as z:
        frozen = {k:z[k].copy() for k in z.files}
    if not all(np.array_equal(saved['initial_9bd'][k], s[k]) for s in saved.values() for k in ('quadratic', 'scales')):
        raise ValueError('The three complete source contracts differ')
    print('preparation_begin', flush=True)
    started = time.perf_counter()
    state, args, template, provenance = prepare(DEFAULT_SEED, out/'read_only_source_binding')
    args.source_processes = False
    args.source_workers = 1
    install_corrected_advertisement_source_geometry(state)
    install_shared_ice_hockey_geometry(state)
    g62, q62, *_ = make_structured_geometry()
    geometry = PairedLocationsSharedSizesGeometry(g62.parent, q62[:g62.parent_parameter_count])
    world_path = runs/'user_corrected_programme_candidate/latest9a8c_world66_initialization_ONLY.json'
    world_packet = json.loads(world_path.read_text(encoding='utf8'))
    apply_latest_geometry_auxiliary(geometry, world_packet)
    install_exact_course_station_cache(geometry.print_course_binding(geometry.initial)[0])
    oldfull = np.r_[state.q, template['q'][-3:], np.zeros(25)]
    token = _frozen_profile_token(make_profile_update(state.blocks))
    install_structured_programme_blocks(state, geometry, oldfull, inherited_profile_token=token)
    install_exact_course_station_cache(state.geometry.source_course.course)
    install_source_ice_material_view(state.geometry, max_cache_bytes=32*1024**2)
    correction = manifest.get('original_ice_pixel_correction')
    if correction:
        install_original_ice_pixel_correction(state, correction['packet_path'])
    # Restore CPU single-thread settings after historical import defaults.
    os.environ.update(RINKSPACE_MOTION_BACKEND='native', RINKSPACE_MOTION_THREADS='1',
                      RINKSPACE_PROJECTION_BACKEND='native', RINKSPACE_PROJECTION_VALUE_BACKEND='opencv',
                      RINKSPACE_PATTERN_WORKERS='0', RINKSPACE_SCATTER_THREADS='1')
    cv2.setNumThreads(1)
    block = next(b for b in state.blocks if b.name == 'secondary')
    index = state.blocks.index(block)
    block_start = sum(len(b.quadratic) for b in state.blocks[:index])
    total = len(block.quadratic)
    mo = block.bundle.motion_rows
    roll_count = len(block.bundle.stamps)
    absolute_stop = total-roll_count
    mask = saved['initial_9bd']['quadratic'][block_start:block_start+total]
    scales = saved['initial_9bd']['scales'][block_start:block_start+total]
    if mo != 79358 or total != 89123 or roll_count != 1061:
        raise ValueError('Unexpected frozen secondary row layout')
    retired_global = manifest['original_source_role_retirement']['global_scalar_slots']
    retired = np.asarray(retired_global, int)-block_start
    if not np.all((retired>=mo)&(retired<absolute_stop)):
        raise ValueError('Retired source slots do not belong to the absolute segment')
    global_bias_slot = CORE_COUNT+1
    print('prepared', time.perf_counter()-started, 'secondary_motion_pairs',mo//2, flush=True)

    if opt.export_conflict_only:
        stamp=np.asarray(block.bundle.stamps,np.int64)
        pts_a,pts_b=stamp[block.bundle.ia],stamp[block.bundle.ib]
        selected=np.flatnonzero((pts_a==230507264)&(pts_b==230507776))
        if len(selected)!=36:raise ValueError('Exact largest-conflict motion timepair inventory changed')
        pair_report=dict(schema='rinkspace.external-largest-motion-conflict-original36.v1',
                         source_pts=230507264,target_pts=230507776,time_base='1/15360',
                         source_video_sha256='f685d27882e78f2446c734ceee4321dcb275a5ad1ebf81ea90bca68c2fcec891',
                         source_manifest=dict(path=str(manifest_path),sha256=sha(manifest_path)),
                         pair_count=36,actual_source_role='unknown-depth original motion correspondence',
                         pair_rows=[dict(original_bundle_pair_index=int(i),source_uv=block.bundle.a[i].tolist(),
                                         target_uv=block.bundle.b[i].tolist(),finite=bool(block.bundle.finite_mask[i]),
                                         sigma_components=block.bundle.motion_sigma[2*i:2*i+2].tolist()) for i in selected])
        cameras={
            'seed_3311':runs/'mainspin_all_tripods_outside_user_domain_v2_actual_fit_nodump_20261006/complete_physical_camera_model_UNACCEPTED_09900e4e4b41d05b.npz',
            'initial_9bd':run/'complete_physical_camera_model_UNACCEPTED_7b0edfe83f8c8c6e.npz',
            'accepted_a8d':run/'complete_physical_camera_model_UNACCEPTED_07f5492daacd7c0a.npz'}
        pair_report['exact_camera_exports']={}
        for name,p in cameras.items():
            with np.load(p,allow_pickle=False) as z:
                ii=[int(np.flatnonzero(z['pts']==t)[0]) for t in (230507264,230507776)]
                vv=z['view_state4'][ii];cc=z['camera25'][ii]
                R=[cv2.Rodrigues(v[:3])[0] for v in vv]
                pair_report['exact_camera_exports'][name]=dict(path=str(p),sha256=sha(p),
                    actual_pts=z['exposure_representative_pts'][ii].tolist(),view_state4=vv.tolist(),camera25=cc.tolist(),
                    rotation_step_degrees=float(np.linalg.norm(cv2.Rodrigues(R[1]@R[0].T)[0])*180/np.pi),
                    logf_step=float(vv[1,3]-vv[0,3]),focal_ratio=float(np.exp(vv[1,3]-vv[0,3])))
        initializer_path=Path(manifest['absolute_source_camera_initializer']['initializer_packet']['path'])
        initializer=json.loads(initializer_path.read_text(encoding='utf8'))
        partial_path=Path(initializer['initializer_report']['path'])
        partial=json.loads(partial_path.read_text(encoding='utf8'))
        solution=next(s for s in partial['solutions'] if s['scope']=='all_FIT_RF_fixed_Pz')
        frames=[f for f in solution['source_frames'] if f['actual_pts'] in (230507264,230507776)]
        pair_report['initializer_provenance']=dict(packet_path=str(initializer_path),packet_sha256=sha(initializer_path),
            partial_report_path=str(partial_path),partial_report_sha256=sha(partial_path),
            both_timepair_endpoints_are_direct_optimized_anchors=all(t in initializer['anchor_actual_pts'] for t in (230507264,230507776)),
            export_interpolation_not_needed_at_this_timepair=True,frames=frames,
            complete_motion_not_in_partial_initializer=solution['complete_motion_not_in_partial_initializer'])
        exported=pair_report['exact_camera_exports']['initial_9bd']['view_state4']
        pair_report['initializer_provenance']['optimized_frame_vs_export_RF_maximum_difference']=max(
            float(np.max(np.abs(np.asarray(f['candidate_ViewState_RF4'])-np.asarray(exported[j])))) for j,f in enumerate(frames))
        dest=out/'largest_motion_conflict_original36_and_initialization.json'
        dest.write_text(json.dumps(pair_report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
        print('OUTPUT',dest,sha(dest),flush=True)
        return

    factors = {
        'additional_ice': NativeIceCurveFactor(state, manifest['additional_real_native_ice_curves']['packet_path'], 34296),
        'additional_board': NativeBoardEdgeFactor(state, manifest['additional_real_native_board_edges']['packet_path'], 34296),
        'additional_circle': NativeBoardCircleFactor(state, manifest['additional_real_native_board_circles']['packet_path'], 34296),
    }

    def restore(profile):
        for i, b in enumerate(state.blocks):
            b.bundle.profile_weights = profile[f'weights_{i}'].copy() if bool(profile[f'present_{i}']) else None
            b.bundle.robust_phase = bool(profile[f'robust_{i}'])

    def compute(q, profile, include_groups=False):
        restore(profile)
        residual = np.asarray(block.evaluate(q[:CORE_COUNT], False),float).copy()
        residual[absolute_stop:] -= q[global_bias_slot]/np.deg2rad(.5)
        residual[retired] = 0.
        robust_absolute = (~mask)&(np.arange(total)>=mo)&(np.arange(total)<absolute_stop)
        quadratic_absolute = mask&(np.arange(total)>=mo)&(np.arange(total)<absolute_stop)
        parts = {
            'motion': source_cost(residual[:mo], mask[:mo]),
            'absolute_robust': source_cost(residual[robust_absolute], mask[robust_absolute]),
            'absolute_quadratic_extents_or_supports': source_cost(residual[quadratic_absolute], mask[quadratic_absolute]),
            'roll_after_installation_bias': source_cost(residual[absolute_stop:], mask[absolute_stop:]),
            'installation_bias_prior': .5*(q[global_bias_slot]/np.deg2rad(.5))**2,
        }
        local = block.local(q[:CORE_COUNT])
        cal = block.bundle.current_calibration(local)
        camera = block.bundle.camera(local,block.bundle.reference_pts)
        from z190_continuous_source_bundle import SCENE_UNIT_M
        parts['reference_optical_height_prior'] = .5*((camera[2]*SCENE_UNIT_M-1.6)/.18)**2
        additional = []
        for name, factor in factors.items():
            fq=q[:factor.parameter_count]
            fb=factor.block(fq,False)
            cursor=0
            subtotal=0.
            for row in factor.rows:
                count=2*len(row['uv'])
                if row['actual_pts'] in state.bypts and state.bypts[row['actual_pts']].name=='secondary':
                    raw=fb.r[cursor:cursor+count]
                    value=source_cost(raw,fb.quadratic[cursor:cursor+count])
                    subtotal+=value
                    additional.append(dict(family=name,id=row['id'],pts=row['actual_pts'],cost=value,rows=count))
                cursor+=count
            parts[name+'_secondary']=subtotal
        output=dict(parts=parts,programme_cost=source_cost(residual,mask),secondary_local_total=sum(parts.values()),
                    raw_absolute_rms_native_px=float(np.sqrt(np.mean((residual[mo:absolute_stop]*scales[mo:absolute_stop])**2))),
                    residual_sha256=hashlib.sha256(residual.tobytes()).hexdigest(),additional_rows=additional)
        if include_groups:
            raw_motion = residual[:mo].reshape(-1,2)
            motion_cost = np.where(np.abs(raw_motion)<=2., .5*raw_motion**2, 2.*np.abs(raw_motion)-2.).sum(axis=1)
            output['_motion_cost_per_pair'] = motion_cost
            output['_motion_normalized_residual'] = raw_motion
            output['motion_physical_surface_split'] = [dict(
                material='explicit_finite_plane' if finite else 'untyped_positive_inverse_depth_profile',
                pairs=int(np.sum(block.bundle.finite_mask==finite)),
                cost=float(motion_cost[block.bundle.finite_mask==finite].sum())) for finite in (False,True)]
            groups=[];cursor=mo
            for pts,rows in block.groups:
                for row in rows:
                    n=2*len(row['uv']);r=residual[cursor:cursor+n];m=mask[cursor:cursor+n]
                    groups.append(dict(pts=int(pts),id=row['id'],split=row.get('split'),points=len(row['uv']),
                                       cost=source_cost(r,m),quadratic_rows=int(m.sum()),
                                       sigma_min=float(np.min(row['sigma'])),sigma_max=float(np.max(row['sigma'])),
                                       retired=bool(np.intersect1d(np.arange(cursor,cursor+n),retired).size)))
                    cursor+=n
            if cursor!=absolute_stop:raise ValueError('Absolute source group segmentation mismatch')
            output['groups']=groups
        return output

    result=dict(schema='rinkspace.external-secondary-frozen-objective-causal-probe.v1',
                source_manifest=dict(path=str(manifest_path),sha256=sha(manifest_path)),
                source_script=dict(path=str(Path(__file__).resolve()),sha256=sha(__file__)),
                frozen_profile=dict(path=str(frozen_path),sha256=sha(frozen_path)),
                state_receipts=[{k:s[k] for k in ('path','file_sha256','q_sha256')} for s in saved.values()],
                backend=motion_metadata(),rows=dict(total=total,motion=mo,absolute=absolute_stop-mo,roll=roll_count),
                frozen_states={},checkpoint_profile_states={},direction_samples={},source_accuracy_accepted=False,
                no_training_optimizer_J_normal_workers_or_video_run=True)
    for label,s in saved.items():
        print('evaluate',label,'common_profile',flush=True)
        result['frozen_states'][label]=compute(s['q'],frozen,True)
        own={k[9:]:v for k,v in s.items() if k.startswith('profile__')}
        print('evaluate',label,'checkpoint_profile',flush=True)
        result['checkpoint_profile_states'][label]=compute(s['q'],own,False)
    result['one_update_profile_states']={}
    for label,s in saved.items():
        restore(frozen)
        local=block.local(s['q'][:CORE_COUNT])
        preliminary=block.bundle.profile_motion_residual(local)
        new_profile={k:v.copy() for k,v in frozen.items()}
        new_profile[f'weights_{index}']=np.minimum(1.,2./np.maximum(np.abs(preliminary),1e-12)).reshape(-1,2)
        result['one_update_profile_states'][label]=compute(s['q'],new_profile,False)
        result['one_update_profile_states'][label]['changed_component_weights']=int(np.sum(new_profile[f'weights_{index}']!=frozen[f'weights_{index}']))
    q0=saved['initial_9bd']['q']
    direction=saved['accepted_a8d']['q']-q0
    for fraction in (-.001,.001,.5):
        print('direction',fraction,flush=True)
        result['direction_samples'][str(fraction)]=compute(q0+fraction*direction,frozen,True)
    result['direction_derivative_at_initial']={
        k:(result['direction_samples']['0.001']['parts'][k]-result['direction_samples']['-0.001']['parts'][k])/.002
        for k in result['frozen_states']['initial_9bd']['parts']}
    initial_groups=result['frozen_states']['initial_9bd']['groups']
    final_groups=result['frozen_states']['accepted_a8d']['groups']
    minus_groups=result['direction_samples']['-0.001']['groups']
    plus_groups=result['direction_samples']['0.001']['groups']
    comparisons=[]
    for a,b,m,p in zip(initial_groups,final_groups,minus_groups,plus_groups):
        if any(a[k]!=x[k] for x in (b,m,p) for k in ('pts','id','points')):raise ValueError('Group identities moved')
        comparisons.append(dict(a,initial_cost=a['cost'],accepted_cost=b['cost'],change=b['cost']-a['cost'],
                                derivative=(p['cost']-m['cost'])/.002))
    result['absolute_groups_cost_and_direction']=comparisons
    stamp=np.asarray(block.bundle.stamps,np.int64)
    pts_a,pts_b=stamp[block.bundle.ia],stamp[block.bundle.ib]
    endpoint_pairs=np.column_stack((pts_a,pts_b))
    unique, inverse=np.unique(endpoint_pairs,axis=0,return_inverse=True)
    group_rows=[]
    initial=result['frozen_states']['initial_9bd']
    accepted=result['frozen_states']['accepted_a8d']
    for i,ps in enumerate(unique):
        chosen=inverse==i
        group_rows.append(dict(source_pts=int(ps[0]),target_pts=int(ps[1]),dt_seconds=float((ps[1]-ps[0])/15360.),
                               pairs=int(chosen.sum()),initial_cost=float(initial['_motion_cost_per_pair'][chosen].sum()),
                               accepted_cost=float(accepted['_motion_cost_per_pair'][chosen].sum())))
    for r in group_rows:r['cost_drop']=r['initial_cost']-r['accepted_cost']
    result['motion_timepair_groups']=sorted(group_rows,key=lambda x:x['cost_drop'],reverse=True)
    drop=initial['_motion_cost_per_pair']-accepted['_motion_cost_per_pair']
    result['largest_motion_pair_cost_drops']=[dict(
        original_bundle_pair_index=int(i),source_pts=int(pts_a[i]),target_pts=int(pts_b[i]),
        source_uv=block.bundle.a[i].tolist(),target_uv=block.bundle.b[i].tolist(),
        finite=bool(block.bundle.finite_mask[i]),sigma_components=block.bundle.motion_sigma[2*i:2*i+2].tolist(),
        initial_normalized_residual=initial['_motion_normalized_residual'][i].tolist(),
        accepted_normalized_residual=accepted['_motion_normalized_residual'][i].tolist(),
        initial_cost=float(initial['_motion_cost_per_pair'][i]),accepted_cost=float(accepted['_motion_cost_per_pair'][i]),cost_drop=float(drop[i]))
        for i in np.argsort(drop)[-60:][::-1]]
    a,b=block.bundle.a,block.bundle.b
    regions={'top_left_scoreboard_RGB_review_candidate_rectangle':(32,28,552,252),
             'bottom_right_timer_RGB_review_candidate_rectangle':(1690,942,1860,1005)}
    result['motion_UI_region_candidates_not_identity_certifications']=[]
    for name,(x0,y0,x1,y1) in regions.items():
        chosen=((a[:,0]>=x0)&(a[:,0]<x1)&(a[:,1]>=y0)&(a[:,1]<y1))|((b[:,0]>=x0)&(b[:,0]<x1)&(b[:,1]>=y0)&(b[:,1]<y1))
        result['motion_UI_region_candidates_not_identity_certifications'].append(dict(
            region=name,rectangle=[x0,y0,x1,y1],pairs=int(chosen.sum()),fraction=float(chosen.mean()),
            initial_cost=float(initial['_motion_cost_per_pair'][chosen].sum()),accepted_cost=float(accepted['_motion_cost_per_pair'][chosen].sum()),
            identity='spatial candidate only; original RGB needs inspection'))
    result['motion_profile_contract']=dict(
        explicit_finite_plane_pairs=int(block.bundle.finite_mask.sum()),all_pairs=len(block.bundle.a),
        minimum_source_depth_scene_units=float(block.bundle.motion_kernel.minimum_source_depth_scene_units),
        finite_records=[{k:v for k,v in r.items() if k not in ('source_qualification','record')} for r in block.bundle.finite_records],
        unknown_depth_is_independently_profiled_for_each_two_view_pair=True,
        cross_pair_shared_landmark_track_3D_identity_not_established_by_this_factor=True)
    for collection in ('frozen_states','direction_samples'):
        for item in result[collection].values():
            item.pop('_motion_cost_per_pair',None);item.pop('_motion_normalized_residual',None)
    # Actual source normalization algebra: use existing reported row sigmas,
    # same measured residual, and an identical duplicate. These are replicas.
    sample=next(row for _,rr in block.groups for row in rr if len(row['uv'])>4)
    sigma=np.asarray(sample['sigma']);err=np.ones(2*len(sigma))*5.;s=np.repeat(sigma,2)
    old=source_cost(err/s,np.zeros(len(err),bool))
    duplicate=source_cost(np.tile(err/s,2),np.zeros(2*len(err),bool))
    result['identical_absolute_sampling_counterfactual']=dict(
        source_id=sample['id'],source_points=len(sigma),same_measurement_native_error_px=5.,
        original_cost=old,duplicate_same_measurements_cost=duplicate,ratio=duplicate/old,
        replicas_are_not_new_measurements=True,unweighted_sum_huber_not_correlated_group_likelihood=True)
    result['motion_normalization_algebra']=dict(
        rule='sigma=max(2,sqrt(pair_point_count)/3)',
        examples=[dict(points=n,sigma=max(2.,np.sqrt(n)/3.),quadratic_cost_for_same_1px_error_two_axes=n/max(2.,np.sqrt(n)/3.)**2)
                  for n in (10,36,100,400,1000)],
        repeated_pair_between_different_times_adds_another_cost_block=True,
        actual_shared_endpoint_covariance_not_whitened_by_this_scalar_rule=True)
    result['runtime_seconds']=time.perf_counter()-started
    destination=out/'external_secondary_objective_probe.json'
    destination.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False,
                                     default=lambda value:value.tolist() if isinstance(value,np.ndarray) else value.item() if isinstance(value,np.generic) else str(value))+'\n',encoding='utf8')
    print('OUTPUT',destination,sha(destination),flush=True)
    print('STATES',json.dumps({k:v['parts'] for k,v in result['frozen_states'].items()}),flush=True)
    print('DERIVATIVES',json.dumps(result['direction_derivative_at_initial']),flush=True)
    print('TOP_DIRECTIONAL_CONFLICTS',json.dumps(sorted(comparisons,key=lambda x:x['derivative'],reverse=True)[:12]),flush=True)


if __name__=='__main__':
    main()
