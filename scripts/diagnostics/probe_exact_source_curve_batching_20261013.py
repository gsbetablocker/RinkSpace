"""Preserve all actual source pixels while testing exact primitive reuse."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'

import numpy as np
from scipy.sparse import csr_matrix

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
from calibration.solver_runtime import install
install()
from calibration.program_tracking.strict_absolute_source import StrictAbsoluteSourcePack
from calibration.program_tracking.strict_model_scope import file_sha, authenticate_source_reference
from calibration.program_tracking.compact_source_header_inventory import read_minimal_compact_source_headers
from calibration.program_tracking.strict_source_representatives import _declared_paths
from calibration.program_tracking.exact_source_curve_batching import (
    prepare_exact_source_curve_batches, evaluate_exact_source_curve_batches)
from z190_source_camera_diagnostic import absolute_value_jac15
from z190_streaming_source_normal import SourceRowBlock, accumulate_source_normal
from z190_physical_camera import SharedCalibration, ViewState


def reference(path):
    return dict(path=str(Path(path).resolve()), sha256=file_sha(path))


def frozen_model():
    path = Path(r'D:/CodexData/RinkSpace_Data/events/ankara_2026/calibration/runs/solver_acceleration_20261003/okada_q32_reviewed_source_pixels_current_points_ready_20261012/accepted_states/8a28afed8f9f896112e5b2f64be515b2bc02019e10e09c7e0eda2598579d4891/state_identity.json')
    identity = json.loads(path.read_text(encoding='utf8'))
    scene_path = path.with_name('scene_UNACCEPTED.json')
    authenticate_source_reference(dict(path=str(scene_path), sha256=identity['scene_sha256']))
    authenticate_source_reference(dict(path=identity['camera_path'], sha256=identity['camera_sha256']))
    scene = json.loads(scene_path.read_text(encoding='utf8'))
    with np.load(identity['camera_path'], allow_pickle=False) as z:
        values = {k:z[k].copy() for k in ('pts', 'camera25', 'view_state4', 'installation_id', 'effective_pp_offset_px')}
        metadata = json.loads(str(z['metadata_json'].item()))
    return identity, reference(path), scene, values, metadata


def world_for(scene, target):
    if target['kind']=='current_scene_ice_curve' and target['feature_id'].startswith('lane'):
        feature = next(x for x in scene['ice_features'] if x['id']==target['feature_id'])
        segment = np.asarray(feature['paint_segments_xyz'][int(target['world_curve_role'][4:])],float)
        return np.linspace(segment[0], segment[-1], 1025)
    if target['kind']=='current_scene_board_print_contour':
        patterns = scene['source_conditioned_board_prints']['patterns']
        pattern = next(x for x in patterns if x['advertisement_id']==target['advertisement_id'])
        contour = next(x for x in pattern['external_contours'] if x['id']==target['contour_id'])
        return np.asarray(contour['xyz_scene_units'],float)
    raise ValueError('Diagnostic target has no supported exact frozen finite primitive')


def entries_from_rows(source_ref, identity_ref, headers, offsets, uv, sigma, row_indices, scene):
    entries=[]
    for ri in row_indices:
        h=headers[ri]
        lo,hi=map(int,offsets[ri:ri+2])
        context=dict(source_pack=source_ref,source_model_identity=identity_ref,
            actual_pts=h['actual_pts'],native_pts=h['native_pts'],decoded_BGR_sha256=h['decoded_BGR_sha256'],
            scope=h['scope'],physical_material_id=h['physical_material_id'],model_target=h['model_target'],
            observation_semantics=h['observation_semantics'],split=h['split'])
        paths,basis=_declared_paths(dict(h,uv=uv[lo:hi]))
        entries.append(dict(source_context=context,
            source_locator=dict(source_pack=source_ref,original_source_row_index=ri,
                original_global_point_range=[lo,hi],source_proof_and_side_arrays_remain_in_original_pack=True),
            source_visible_paths=dict(basis=basis,independent_original_row_point_paths=paths),
            evaluation_row=dict(world=world_for(scene,h['model_target']),uv=uv[lo:hi],sigma=sigma[lo:hi],
                point_correspondence=False,curve_observation_kind='measured_transition_location',
                observation_semantics=h['observation_semantics'])))
    return entries


def same_normal(r,j,sigma,quadratic):
    block=SourceRowBlock(r,csr_matrix(j),np.full(len(r),quadratic,bool),np.repeat(sigma,2),
        dict(family='lossless-actual-source-primitive-reuse-readonly'))
    return accumulate_source_normal(lambda:iter([block]),j.shape[1],expected_row_count=len(r),
        frozen_profile_token='same-8a28-original-sigma-loss-unweighted')


def compare_group(label,entries,values,metadata,payload):
    prepared=prepare_exact_source_curve_batches(entries)
    if prepared['source_operator_calls_after']!=1:
        raise ValueError('One exact exposure/material/operator group required for this production probe')
    context=entries[0]['source_context'];pts=context['actual_pts']
    at=np.flatnonzero(values['pts']==context['native_pts'])
    if len(at)!=1:raise ValueError('Unique frozen native physical camera required')
    ci=int(at[0]);cal=SharedCalibration.from_manifest(metadata['calibrations'][str(values['installation_id'][ci])])
    initial=values['view_state4'][ci];offset=values['effective_pp_offset_px'][ci]
    scale=np.diag([.01]*3+[1.]*5+[.001]*3+[1.]*4)
    rows=[e['evaluation_row'] for e in entries]
    sigma=np.concatenate([r['sigma'] for r in rows])
    uv=np.concatenate([r['uv'] for r in rows])
    def evaluate(state):
        camera,_,mapping=cal.camera25_and_jacobians(ViewState.from_state4(state,effective_pp_offset_px=offset))
        r,j=absolute_value_jac15(camera,rows,True)
        combined_r,combined_j=evaluate_exact_source_curve_batches(prepared,{pts:camera},absolute_value_jac15)
        if not np.array_equal(r,combined_r) or not np.array_equal(j,combined_j):
            raise ValueError('Batching changed original normalized source r/J')
        value=absolute_value_jac15(camera,rows,False)
        combined_value=evaluate_exact_source_curve_batches(prepared,{pts:camera},absolute_value_jac15,jacobian=False)
        if not np.array_equal(value,combined_value):
            raise ValueError('Batching changed original nonlinear source value search')
        if not np.array_equal(value,r):
            # Original value-only and analytic branches may have their own
            # floating-point differences; batching must preserve each branch.
            original_value_analytic_max=float(np.max(abs(value-r)))
        else:original_value_analytic_max=0.
        normal_checks={}
        for name,map_ in [('camera15',np.eye(15)),('view4',scale@mapping[:15])]:
            left_j,right_j=j@map_,combined_j@map_
            for mode,quadratic in [('Huber2',False),('quadratic',True)]:
                left=same_normal(r,left_j,sigma,quadratic)
                right=same_normal(combined_r,right_j,sigma,quadratic)
                exact=(left.cost==right.cost and np.array_equal(left.gradient,right.gradient)
                    and np.array_equal(left.normal.toarray(),right.normal.toarray()))
                if not exact:raise ValueError('Unchanged production cost/g/H differ after original order restoration')
                normal_checks[name+'_'+mode]=dict(original_cost=left.cost,cost_gradient_normal_bit_exact=True)
        return camera,r,j,j@(scale@mapping[:15]),dict(normal_checks=normal_checks,
            normalized_r_camera15_J_bit_exact=True,value_only_original_search_bit_exact=True,
            original_value_vs_analytic_branch_max_difference=original_value_analytic_max)
    camera,r,j,j4,nominal=evaluate(initial)
    if not np.allclose(camera,values['camera25'][ci],rtol=0,atol=1e-12):
        raise ValueError('Nominal physical camera differs from frozen state')
    trials=[]
    native_j=j4.reshape(len(uv),2,4)*sigma[:,None,None]
    for col in range(4):
        sensitivity=float(np.max(np.linalg.norm(native_j[:,:,col],axis=1)))
        if not sensitivity:raise ValueError('Selected source camera trial axis unobservable')
        step=.5/sensitivity
        for sign in (-1,1):
            state=initial.copy();state[col]+=sign*step
            _,tr,_,_,check=evaluate(state)
            trials.append(dict(axis=col,sign=sign,view_state4_delta=(state-initial).tolist(),
                first_order_max_source_displacement_px=.5,
                actual_native_residual_displacement_px=float(np.max(np.linalg.norm((tr-r).reshape(-1,2)*sigma[:,None],axis=1))),**check))
    # Warm both production paths, then compare the same retained input/order.
    sequential_times=[];combined_times=[]
    for _ in range(7):
        begin=time.perf_counter();absolute_value_jac15(camera,rows,True);sequential_times.append(time.perf_counter()-begin)
        begin=time.perf_counter();evaluate_exact_source_curve_batches(prepared,{pts:camera},absolute_value_jac15);combined_times.append(time.perf_counter()-begin)
    s,c=float(np.median(sequential_times)),float(np.median(combined_times))
    payload.update({label+'_uv_original_order':uv,label+'_sigma_original_order':sigma,
        label+'_r_original_order':r,label+'_camera15_J_original_order':j,
        label+'_source_row_indices':np.asarray([e['source_locator']['original_source_row_index'] for e in entries],np.int64)})
    return dict(label=label,context=context,points=len(uv),source_rows_before=len(entries),operator_calls_after=1,
        source_locators=[e['source_locator'] for e in entries],
        independent_visible_paths=[e['source_visible_paths'] for e in entries],
        original_point_ranges=[g['original_scalar_ranges'] for g in prepared['groups']],
        nominal=nominal,eight_actual_physical_camera_trials=trials,
        production_RJ_timing=dict(original_row_calls_seconds=sequential_times,
            batched_curve_seconds=combined_times,original_median_seconds=s,batched_median_seconds=c,
            measured_local_ratio=s/c,complete_iteration_speed_not_inferred=True),
        all_original_UV_sigma_source_point_order_and_proof_locators_retained=True,
        independent_visible_paths_not_connected_by_new_chords=True,source_points_removed=0,
        loss_weights_or_operator_dead_zone_changed=False)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source-work',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();base=args.source_work
    if args.output.exists() or args.output.with_suffix('.npz').exists():raise FileExistsError('Preserve prior real production batching diagnostics')
    identity,identity_ref,scene,values,metadata=frozen_model();payload={};records=[]
    small=base/'proof_locator_final_precision_smoke_038_feedback_minimal_20261012.npz'
    small_ref=reference(small);pack=StrictAbsoluteSourcePack(small,expected_sha256=small_ref['sha256'])
    small_entries=entries_from_rows(small_ref,identity_ref,pack.rows,pack.values['row_offsets'],pack.values['uv'],pack.values['sigma_px'],[10,13],scene)
    records.append(compare_group('ice_smoke_same_target_two_rows',small_entries,values,metadata,payload))
    del pack
    big=base/'source_full_precision_spill_retry3_v1/223403520_R0402_038.npz'
    big_ref=dict(path=str(big),sha256='a206919294c81eb6f5fb719af969c655051a418c6d79ab85ba2561f3721fff1b')
    authenticate_source_reference(big_ref)
    headers,offsets=read_minimal_compact_source_headers(big)
    distribution_path=base/'source038_exact_primitive_row_distribution_readonly_20261013_v1.json'
    distribution=json.loads(distribution_path.read_text(encoding='utf8'))
    if distribution['source_reference']!=big_ref:raise ValueError('Full source row distribution binding differs')
    with np.load(big,allow_pickle=False)as z:uv=z['uv'].copy();sigma=z['sigma_px'].copy()
    selected=[]
    for label,select in [('board_actual_print_fragments',lambda g:g['model_target']['kind']=='current_scene_board_print_contour'),
            ('ice_actual_lane_fragments',lambda g:g['scope']=='ice' and g['model_target'].get('feature_id','').startswith('lane'))]:
        candidates=[g for g in distribution['records'] if select(g) and g['source_rows']>1]
        group=next((g for g in candidates if all(np.isfinite(sigma[int(offsets[i]):int(offsets[i+1])]).all()
            and np.all(sigma[int(offsets[i]):int(offsets[i+1])]>0) for i in g['source_row_indices'])),None)
        if group is None:raise ValueError('No actual finite-source-support fragmented group available')
        entries=entries_from_rows(big_ref,identity_ref,headers,offsets,uv,sigma,group['source_row_indices'],scene)
        records.append(compare_group(label,entries,values,metadata,payload));selected.append(group)
    if file_sha(big)!=big_ref['sha256']:raise ValueError('Immutable full source pack changed during probe')
    report=dict(schema='rinkspace.actual-production-lossless-exact-curve-source-batching-probe.v1',
        q_sha256=identity['q_sha256'],frozen_identity_reference=identity_ref,
        full_source_row_distribution_reference=reference(distribution_path),
        full_source_row_distribution=dict(source_rows=distribution['rows'],source_points=distribution['points'],
            exact_context_groups=distribution['groups'],potential_same_primitive_call_ratio=distribution['rows']/distribution['groups'],
            distribution_is_not_full_model_operator_qualification=True),
        actual_checked_groups=records,selected_full_pack_groups=selected,
        original_source_packs_proofs_roles_time_HELD_and_loss_changed=False,
        complete_world100_producer_and_J_verified=False,shared_optics_mount_verified=False,
        entire_iteration_speed_verified=False,solver_adapter_installed=False,fit_started=False,
        implementation_reference=reference(Path(__file__)),
        batching_implementation_reference=reference(ROOT/'calibration/program_tracking/exact_source_curve_batching.py'),
        header_inventory_implementation_reference=reference(ROOT/'calibration/program_tracking/compact_source_header_inventory.py'),
        source_points_removed=0,new_loss_weights_added=False)
    with args.output.open('x',encoding='utf8')as stream:json.dump(report,stream,ensure_ascii=False,indent=2)
    np.savez_compressed(args.output.with_suffix('.npz'),**payload)
    print(json.dumps(dict(report=str(args.output),checks=[dict(label=r['label'],points=r['points'],calls_before=r['source_rows_before'],calls_after=1,
        same_q_and_eight_trial_r_J_cost_g_H_bit_exact=True,local_timing_ratio=r['production_RJ_timing']['measured_local_ratio']) for r in records],points_removed=0,adapter_installed=False)),flush=True)


if __name__=='__main__':main()
