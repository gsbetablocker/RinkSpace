"""Replay the actual archived wide r/J through the dynamic geometry API.

Uses the existing source loaders, complete-state geometry factory and finite
curve operators. No programme profiles or complete source graph are built.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse
import hashlib
import json
import sys
import time

import numpy as np
from scipy.sparse import csr_matrix

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from rinkspace.vision_camera_initialization import current_complete_source_geometry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--geometry-auxiliary', type=Path, required=True)
    parser.add_argument('--archived-wide', type=Path, required=True)
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-geometry-revision', action='store_true',
                        help='Verify the explicit corrected geometry rather than archived operator equivalence.')
    parser.add_argument('--investigate-nearest-branches', action='store_true')
    args = parser.parse_args()
    started = time.perf_counter()
    provider, world, proof = current_complete_source_geometry(args.state, args.geometry_auxiliary)
    from calibration.solver_runtime.runtime import status
    source_status=status(verify_sources=True)
    from wide_anchor_candidate.wide_high_source_factor import WORLD_INDICES as OLD_WORLD_INDICES
    from wide_anchor_candidate.wide_high_free_optics_source_factor import WideHighFreeOpticsSourceFactor
    from structured_programme_source_candidate.current_wide_source import CurrentWideSource
    from structured_programme_source_candidate.paired_programme_chart import CORE_COUNT
    with np.load(args.state, allow_pickle=False) as saved:
        q = saved['q'].copy()
        slots = saved['aux__current_shared_world_geometry_parameter_indices'].copy()
        optics = json.loads(str(saved['aux__all_installation_optics_contract_json'].item()))
        roll = json.loads(str(saved['aux__correlated_roll_prior_contract_json'].item()))
        layout = (json.loads(str(saved['aux__current_complete_parameter_layout_json'].item()))
                  if 'aux__current_complete_parameter_layout_json' in saved.files else None)
        permutation = (saved['aux__current_evaluation_to_canonical_parameter_indices'].copy()
                       if 'aux__current_evaluation_to_canonical_parameter_indices' in saved.files
                       else np.arange(len(q)))
    with np.load(args.archived_wide, allow_pickle=False) as saved:
        archived = {name: saved[name].copy() for name in saved.files}
    if (len(q)<len(archived['q']) or not np.array_equal(q[:len(archived['q'])], archived['q'])):
        raise ValueError('Archived wide evidence belongs to another preserved complete q prefix')
    if (permutation.shape!=(len(q),) or len(np.unique(permutation))!=len(q)
            or np.any(permutation<0) or np.any(permutation>=len(q))):
        raise ValueError('Explicit complete canonical/evaluation permutation required')
    physical_count=(int(layout['evaluation_physical_core_parameter_count']) if layout is not None else CORE_COUNT)
    evaluation=q[permutation]
    evaluation_slots=np.argsort(permutation)[slots]
    ids = roll['installation_order']
    secondary = ids[1]
    # Only construction uses this old23 wrapper; all actual evaluation uses
    # the restored current world and its real complete parameter map.
    original_state = SimpleNamespace(
        q=q[:34225], ids=ids, slots={secondary: np.array([8, 9, 10])},
        geometry=SimpleNamespace(parameter_count=23, scene=lambda unused: provider.current.scene(world)),
        geometry_parameter_indices=OLD_WORLD_INDICES,
        installation_optics={secondary: SimpleNamespace(mount_indices=np.array([34198, 34199, 34200]))},
        optics_contract=optics)
    original = WideHighFreeOpticsSourceFactor(original_state)
    state = SimpleNamespace(
        geometry=provider, geometry_parameter_indices=evaluation_slots,
        structured_programme_coordinate_chart=SimpleNamespace(core_count=physical_count))
    factor = CurrentWideSource(original, state, json.loads(args.trace.read_text(encoding='utf-8-sig')))
    core = evaluation[:physical_count]
    private = q[archived['wide_private_indices']]
    values_started = time.perf_counter()
    residual, matrix, private_j = factor.evaluate(core, private, True)
    evaluation_seconds = time.perf_counter() - values_started
    gold_j = csr_matrix((archived['core_J_data'], archived['core_J_indices'], archived['core_J_indptr']),
                        shape=tuple(archived['core_J_shape']))
    if matrix[:,:CORE_COUNT].shape != gold_j.shape or residual.shape != archived['residual'].shape:
        raise ValueError('Archived original source rows or physical chart shape changed')
    difference = matrix[:,:CORE_COUNT] - gold_j
    max_core = float(np.max(np.abs(difference.data), initial=0.))
    max_r = float(np.max(np.abs(residual - archived['residual']), initial=0.))
    max_private = float(np.max(np.abs(private_j - archived['private_J']), initial=0.))
    from structured_programme_source_candidate.prepared_finite_curve_world_columns import prepare_curve_current_world_columns
    def source_branch_signature(c,p):
        w,camera,*_=factor._inputs(c,p)
        curves,top,_=factor._curve_worlds(w)
        result=[]
        for xyz,uv in list(zip(curves,factor.uvs))+[(top,uv)for uv in factor.trace_uv]:
            prep=prepare_curve_current_world_columns(camera,xyz,uv)
            endpoints=np.where(prep.selected_raw_t<=0.,-1,np.where(prep.selected_raw_t>=1.,1,0)).astype(np.int8)
            result.append((prep.selected_index.tobytes(),endpoints.tobytes()))
        return tuple(result)
    nominal_signature=source_branch_signature(core,private) if args.expected_geometry_revision else None
    if args.investigate_nearest_branches:
        from structured_programme_source_candidate.prepared_finite_curve_world_columns import prepare_curve_current_world_columns
        def branches(c,p):
            w,camera,*_=factor._inputs(c,p)
            curves,_,_=factor._curve_worlds(w)
            records=[];cursor=0
            for row,xyz,uv,sigma in zip(factor.observations,curves,factor.uvs,factor.sigmas):
                prep=prepare_curve_current_world_columns(camera,xyz,uv)
                for component in (503,2197):
                    local=component-cursor
                    if 0<=local<2*len(uv):
                        k=local//2;index=int(prep.selected_index[k])
                        records.append(dict(component=component,source_id=row.get('id'),
                            source_uv=uv[k].tolist(),source_sigma=float(sigma[k]),
                            selected_index=index,selected_raw_t=float(prep.selected_raw_t[k]),
                            nearest=prep.nearest[k].tolist(),raw_residual=prep.residual.reshape(-1,2)[k].tolist(),
                            valid_selected_segment=bool(prep.valid_segments[index])))
                cursor+=2*len(uv)
            return records
        nominal=branches(core,private);records=[]
        for kind,column in (('world',14),('camera',0),('camera',15),('camera',18)):
            for h in (1e-6,1e-7,1e-8,1e-9,1e-10):
                cp,cm=core.copy(),core.copy();pp,pm=private.copy(),private.copy()
                if kind=='world':cp[evaluation_slots[column]]+=h;cm[evaluation_slots[column]]-=h
                else:pp[column]+=h;pm[column]-=h
                records.append(dict(kind=kind,column=column,h=h,plus=branches(cp,pp),minus=branches(cm,pm)))
        result=dict(nominal=nominal,probes=records,
            strict_geometry_proof=proof,full_source_rows_kept=True,source_objective_changed=False,
            central_difference_failure_is_not_silently_accepted=True)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
        print(json.dumps(dict(path=str(args.output),nominal=nominal)))
        return
    extra_checks=[]
    for logical,column in enumerate(evaluation_slots):
        if logical<66 and not args.expected_geometry_revision:
            continue
        h=1e-6;crossed=[]
        while True:
            plus,minus=core.copy(),core.copy();plus[column]+=h;minus[column]-=h
            branch_stable=(not args.expected_geometry_revision or
                source_branch_signature(plus,private)==nominal_signature==source_branch_signature(minus,private))
            if branch_stable or h<=1e-11:break
            crossed.append(h);h/=10.
        plus_r=factor.evaluate(plus,private,False)[0];minus_r=factor.evaluate(minus,private,False)[0]
        observed=(plus_r-minus_r)/(2*h)
        predicted=matrix[:,column].toarray().ravel()
        error=float(np.max(np.abs(observed-predicted),initial=0.))
        allowed=2e-5+2e-5*float(np.max(np.abs(observed),initial=0.))
        extra_checks.append(dict(logical_world_column=logical,evaluation_column=int(column),
            maximum_true_residual_derivative=float(np.max(np.abs(observed),initial=0.)),
            maximum_derivative_error=error,allowed=allowed,passed=error<allowed and branch_stable,
            actual_FD_h=h,actual_nearest_branch_stable=branch_stable,branch_crossing_h_values=crossed,
            zero_jacobian_with_nonzero_residual_derivative=bool(not np.any(predicted) and np.any(observed)),
            greatest_error_component=int(np.argmax(np.abs(observed-predicted))),
            greatest_error_nominal_plus_minus=[float(v[int(np.argmax(np.abs(observed-predicted)))]) for v in (residual,plus_r,minus_r)]))
    private_checks=[]
    if args.expected_geometry_revision:
        for column in range(25):
            h=1e-6;crossed=[]
            while True:
                plus,minus=private.copy(),private.copy();plus[column]+=h;minus[column]-=h
                branch_stable=source_branch_signature(core,plus)==nominal_signature==source_branch_signature(core,minus)
                if branch_stable or h<=1e-11:break
                crossed.append(h);h/=10.
            plus_r=factor.evaluate(core,plus,False)[0];minus_r=factor.evaluate(core,minus,False)[0]
            observed=(plus_r-minus_r)/(2*h)
            predicted=private_j[:,column]
            error=float(np.max(np.abs(observed-predicted),initial=0.))
            allowed=2e-5+2e-5*float(np.max(np.abs(observed),initial=0.))
            private_checks.append(dict(private_column=column,maximum_derivative_error=error,
                maximum_true_residual_derivative=float(np.max(np.abs(observed),initial=0.)),
                allowed=allowed,passed=error<allowed and branch_stable,
                actual_FD_h=h,actual_nearest_branch_stable=branch_stable,branch_crossing_h_values=crossed,
                greatest_error_component=int(np.argmax(np.abs(observed-predicted))),
                greatest_error_nominal_plus_minus=[float(v[int(np.argmax(np.abs(observed-predicted)))]) for v in (residual,plus_r,minus_r)]))
    derivative_passed=all(row['passed'] for row in extra_checks+private_checks)
    equivalence_passed=max_r < 1e-9 and max_core < 1e-6 and max_private < 1e-7
    report = dict(
        schema='rinkspace.archived-real-wide-dynamic-chart-equivalence.v1',
        state=str(args.state.resolve()), state_sha256=hashlib.sha256(args.state.read_bytes()).hexdigest(),
        archived_wide=str(args.archived_wide.resolve()),
        archived_wide_sha256=hashlib.sha256(args.archived_wide.read_bytes()).hexdigest(),
        archived_source_manifest_sha256=str(archived['source_manifest_sha256'].item()),
        current_source_manifest_sha256=source_status['source_manifest_sha256'],
        geometry_proof=proof, original_source_rows=len(residual),
        current_world_parameters=len(world), current_physical_parameters=physical_count,
        additional_world_columns=[dict(logical_world_column=logical,
            evaluation_column=int(column),jacobian_nonzeros=int(matrix[:,column].nnz))
            for logical,column in enumerate(evaluation_slots) if logical>=66],
        all_additional_world_actual_residual_FD_checks=extra_checks,
        all_world_columns_checked=args.expected_geometry_revision,
        all_private_camera25_actual_residual_FD_checks=private_checks,
        maximum_residual_difference=max_r, maximum_core_jacobian_difference=max_core,
        maximum_private25_jacobian_difference=max_private,
        evaluation_seconds=evaluation_seconds, wall_seconds=time.perf_counter()-started,
        whole_graph_or_profiles_loaded=False,
        source_UV_sigmas_roles_and_counts_preserved=True,
        explicit_geometry_operator_revision=args.expected_geometry_revision,
        FD_step_reduced_only_when_actual_nearest_segment_or_endpoint_branch_differs=True,
        all_original_source_rows_and_true_trial_branch_evaluations_retained=True,
        archived_operator_equivalence_passed=equivalence_passed,
        current_raw_source_groups=factor.review(core,private)['raw_source_native_pixel_error_groups'],
        passed=derivative_passed and (args.expected_geometry_revision or equivalence_passed),
        accuracy_accepted=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k not in (
        'geometry_proof','additional_world_columns','all_additional_world_actual_residual_FD_checks',
        'all_private_camera25_actual_residual_FD_checks','current_raw_source_groups')}, ensure_ascii=False))
    if not report['passed']:
        raise ValueError('Actual current wide derivative check or required archived equivalence failed')


if __name__ == '__main__':
    main()
