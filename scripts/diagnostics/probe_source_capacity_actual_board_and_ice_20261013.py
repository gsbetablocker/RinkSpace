"""Production r/J and bounded camera-trial information audit; never solve."""
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
from calibration.solver_runtime.source_loss import SourceLossWeights
from calibration.program_tracking.strict_absolute_source import StrictAbsoluteSourcePack
from calibration.program_tracking.strict_model_scope import file_sha, authenticate_source_reference
from calibration.program_tracking import strict_source_representatives as representatives
from calibration.program_tracking.source_information_moments import (
    production_point_moments, reduce_nonnegative_exact_moments, moment_comparison)
from rinkspace.source_sampling import source_support_masses
from z190_source_camera_diagnostic import absolute_value_jac15
from z190_streaming_source_normal import SourceRowBlock, accumulate_source_normal
from z190_physical_camera import SharedCalibration, ViewState


def reference(path):
    return dict(path=str(Path(path).resolve()), sha256=file_sha(path))


def production_normal_check(r, jac, sigma, mass, features, quadratic):
    block = SourceRowBlock(r, csr_matrix(jac), np.full(len(r), quadratic, bool),
        np.repeat(sigma, 2), dict(family='actual-source-capacity-readonly'),
        SourceLossWeights(np.repeat(mass, 2)))
    normal = accumulate_source_normal(lambda: iter([block]), jac.shape[1],
        expected_row_count=len(r), frozen_profile_token='same-8a28-normalized-source-readonly')
    upper = np.triu_indices(jac.shape[1])
    expected = np.r_[normal.cost, normal.gradient, normal.normal.toarray()[upper]]
    actual = mass @ features
    denominator = np.maximum(np.sum(abs(mass[:, None]*features), axis=0), np.finfo(float).tiny)
    error = abs(actual-expected)/denominator
    bound = 64*np.finfo(float).eps*len(mass)
    if np.any(error > bound):
        raise ValueError('Moment features do not reproduce the original production normal')
    return dict(maximum_component_relative_absolute_sum_error=float(error.max()),
        floating_point_accumulation_bound=float(bound), r_J_already_sigma_normalized=True,
        source_sigma_applied_again=False)


def shape_support(uv, paths, shape_indices):
    protection = np.zeros(len(uv), bool)
    protection[shape_indices] = True
    path_indicators = np.zeros((len(uv), len(paths)))
    for i, part in enumerate(paths):
        path_indicators[part, i] = 1.
        protection[part[0]] = protection[part[-1]] = True
    return protection, path_indicators


def reduction_record(uv, mass, protected, path_indicators, feature_sets):
    moment = np.column_stack([path_indicators, *feature_sets.values()])
    start = time.perf_counter()
    weights, numerical = reduce_nonnegative_exact_moments(moment, mass, protected)
    keep = np.flatnonzero(weights > 0)
    return dict(source_points=len(uv), retained_points=len(keep),
        protected_source_shape_points=int(protected.sum()),
        retained_original_row_point_indices=keep.tolist(),
        positive_retained_weights=bool(np.all(weights[keep] > 0)),
        source_path_masses_before=(mass @ path_indicators).tolist(),
        source_path_masses_after=(weights @ path_indicators).tolist(),
        numerical_reduction=numerical, elapsed_seconds=time.perf_counter()-start,
        checks={key: moment_comparison(value, mass, weights,
            15 if '_camera15_' in key else 4) for key, value in feature_sets.items()}), weights


def corrected_ice(base, payload):
    prior_path = base/'strict_representative_current_q_source_RJ_probe_20261012.json'
    prior = json.loads(prior_path.read_text(encoding='utf8'))
    pack = StrictAbsoluteSourcePack(prior['source_reference']['path'],
        expected_sha256=prior['source_reference']['sha256'])
    records = []
    with np.load(prior_path.with_suffix('.npz'), allow_pickle=False) as arrays:
        for old in prior['source_rows']:
            ri = old['original_source_row_index']
            lo, hi = map(int, pack.values['row_offsets'][ri:ri+2])
            uv, sigma = pack.values['uv'][lo:hi], pack.values['sigma_px'][lo:hi]
            mass = source_support_masses(uv, old['full_source_visible_paths'])
            mass *= old['actual_source_mass_before']/mass.sum()
            protected, paths = shape_support(uv, old['full_source_visible_paths'],
                old['kept_original_row_point_indices'])
            r = arrays[f'row{ri}_full_r']
            fs, checked = {}, {}
            for name, array in [('camera15', 'full_camera15_J'), ('view4', 'full_view4_J')]:
                jac = arrays[f'row{ri}_{array}']
                for mode, quadratic in [('Huber2', False), ('quadratic', True)]:
                    key = 'nominal_'+name+'_'+mode
                    fs[key] = production_point_moments(r, jac, quadratic=quadratic)
                    checked[key] = production_normal_check(r, jac, sigma, mass, fs[key], quadratic)
            result, weights = reduction_record(uv, mass, protected, paths, fs)
            result.update(source_row_index=ri, actual_pts=old['actual_pts'],
                physical_material_id=old['physical_material_id'], production_normal_checks=checked)
            records.append(result)
            payload.update({f'ice_row{ri}_original_weights': mass,
                f'ice_row{ri}_reduced_weights': weights, f'ice_row{ri}_protected': protected})
    return dict(prior_actual_production_RJ_reference=reference(prior_path.with_suffix('.npz')),
        source_reference=prior['source_reference'], records=records,
        invalid_superseded_diagnostic_reference=reference(base/'actual_same_q_information_moment_experiment_20261012.json'),
        invalid_reason='The preserved earlier moment diagnostic divided already sigma-normalized production r/J by sigma again. Its reduction conclusions are invalid. The original production-normal RDP pilot is unaffected.')


def board_probe(base, payload):
    selected = json.loads((base/'validated_board_delta_selected_refs_pre_late_20261012_v1.json').read_text(encoding='utf8'))
    source_ref = selected['records'][1]['absolute_inputs']
    pack = StrictAbsoluteSourcePack(source_ref['path'], expected_sha256=source_ref['sha256'])
    plan = representatives.compile_strict_representatives(pack, source_reference=source_ref)
    by_record = {r['source_row_index']: r for r in plan['records']}
    identity_path = Path(r'D:/CodexData/RinkSpace_Data/events/ankara_2026/calibration/runs/solver_acceleration_20261003/okada_q32_reviewed_source_pixels_current_points_ready_20261012/accepted_states/8a28afed8f9f896112e5b2f64be515b2bc02019e10e09c7e0eda2598579d4891/state_identity.json')
    identity = json.loads(identity_path.read_text(encoding='utf8'))
    scene_path = identity_path.with_name('scene_UNACCEPTED.json')
    authenticate_source_reference(dict(path=str(scene_path), sha256=identity['scene_sha256']))
    authenticate_source_reference(dict(path=identity['camera_path'], sha256=identity['camera_sha256']))
    scene = json.loads(scene_path.read_text(encoding='utf8'))
    with np.load(identity['camera_path'], allow_pickle=False) as z:
        cameras = {k: z[k].copy() for k in ('pts', 'camera25', 'view_state4', 'installation_id', 'effective_pp_offset_px')}
        metadata = json.loads(str(z['metadata_json'].item()))
    records = []
    for ri in (0, 1, 10):
        header = pack.rows[ri]
        lo, hi = map(int, pack.values['row_offsets'][ri:ri+2])
        uv, sigma = pack.values['uv'][lo:hi], pack.values['sigma_px'][lo:hi]
        if (not np.isfinite(sigma).all() or np.any(sigma <= 0)
                or header['split'] != 'fit' or header['scope'] != 'board'):
            raise ValueError('Actual finite-working-support board review row required')
        paths, path_basis = representatives._declared_paths(dict(header, uv=uv))
        mass = source_support_masses(uv, paths)
        group_total = sum(r['original_source_support_mass'] for i, r in by_record.items()
            if r['split']=='fit' and pack.rows[i]['actual_pts']==header['actual_pts']
            and pack.rows[i]['physical_material_id']==header['physical_material_id'])
        mass /= group_total
        shape = by_record[ri]['kept_indices']
        protected, path_indicators = shape_support(uv, paths, shape)
        target = header['model_target']
        ad = next(a for a in scene['advertisements'] if a['user_number']==target['advertisement_number'])
        world = np.asarray(ad['fitted_boundary_curves_xyz'][target['edge']+'_xyz'], float)
        matches = np.flatnonzero(cameras['pts']==header['native_pts'])
        if len(matches) != 1:
            raise ValueError('Unique same-source native physical camera required')
        ci = int(matches[0])
        calibration = SharedCalibration.from_manifest(metadata['calibrations'][str(cameras['installation_id'][ci])])
        nominal_state = cameras['view_state4'][ci]
        offset = cameras['effective_pp_offset_px'][ci]
        scale = np.diag([.01]*3+[1.]*5+[.001]*3+[1.]*4)
        source = dict(world=world, uv=uv, sigma=sigma, point_correspondence=False,
            curve_observation_kind='measured_transition_location', observation_semantics=header['observation_semantics'])
        def evaluate(state):
            camera, _, mapping = calibration.camera25_and_jacobians(ViewState.from_state4(state, effective_pp_offset_px=offset))
            r, j15 = absolute_value_jac15(camera, [source], True)
            return camera, r, j15, j15 @ (scale @ mapping[:15])
        camera, r, j15, j4 = evaluate(nominal_state)
        if not np.allclose(camera, cameras['camera25'][ci], rtol=0, atol=1e-12):
            raise ValueError('Frozen same-q physical camera reproduction differs')
        nominal, checked = {}, {}
        for name, jac in [('camera15', j15), ('view4', j4)]:
            for mode, quadratic in [('Huber2', False), ('quadratic', True)]:
                key = 'nominal_'+name+'_'+mode
                nominal[key] = production_point_moments(r, jac, quadratic=quadratic)
                checked[key] = production_normal_check(r, jac, sigma, mass, nominal[key], quadratic)
        local, local_weights = reduction_record(uv, mass, protected, path_indicators, nominal)
        combined = dict(nominal)
        trials = []
        pixel_step = .5*float(header['source_readout_uncertainty']['quantization_step_px'])
        native_j = j4.reshape(len(uv), 2, 4)*sigma[:, None, None]
        for col in range(4):
            sensitivity = float(np.max(np.linalg.norm(native_j[:, :, col], axis=1)))
            if sensitivity == 0:
                raise ValueError('Actual camera trial axis is unobservable')
            step = pixel_step/sensitivity
            for sign in (-1, 1):
                trial_state = nominal_state.copy()
                trial_state[col] += sign*step
                _, tr, tj15, tj4 = evaluate(trial_state)
                label = 'axis'+str(col)+('_negative' if sign<0 else '_positive')
                for name, jac in [('camera15', tj15), ('view4', tj4)]:
                    for mode, quadratic in [('Huber2', False), ('quadratic', True)]:
                        combined[label+'_'+name+'_'+mode] = production_point_moments(tr, jac, quadratic=quadratic)
                # Independent nonlinear production search on the exact source
                # subset proves retained pixels are unchanged at this trial.
                keep = np.flatnonzero(local_weights > 0)
                camera, _, _ = calibration.camera25_and_jacobians(ViewState.from_state4(trial_state, effective_pp_offset_px=offset))
                short_r, short_j = absolute_value_jac15(camera, [dict(source, uv=uv[keep], sigma=sigma[keep])], True)
                scalar = (2*keep[:, None]+np.arange(2)).ravel()
                if not np.array_equal(tr[scalar], short_r) or not np.array_equal(tj15[scalar], short_j):
                    raise ValueError('Retained original source production r/J changed at actual camera trial')
                trials.append(dict(label=label, physical_view_state4_delta=(trial_state-nominal_state).tolist(),
                    desired_first_order_max_source_displacement_px=pixel_step,
                    measured_max_native_residual_displacement_px=float(np.max(np.linalg.norm((tr-r).reshape(-1, 2)*sigma[:, None], axis=1))),
                    actual_nonlinear_production_operator_recomputed=True,
                    retained_source_r_J_bit_exact=True,
                    nominal_only_weights_trial_checks={name: moment_comparison(value, mass, local_weights,
                        15 if '_camera15_' in name else 4) for name, value in combined.items() if name.startswith(label+'_')}))
        joint, joint_weights = reduction_record(uv, mass, protected, path_indicators, combined)
        record = dict(source_row_index=ri, actual_pts=header['actual_pts'], native_pts=header['native_pts'],
            decoded_BGR_sha256=header['decoded_BGR_sha256'], material=header['physical_material_id'],
            model_target=target, source_visible_path_basis=path_basis, source_visible_paths=paths,
            full_source_uv_extent=[uv.min(axis=0).tolist(), uv.max(axis=0).tolist()],
            observed_extent_stations_protected=True, source_shape_quantization_step_px=1.,
            source_shape_max_chord_deviation_px=by_record[ri]['maximum_chord_deviation_px'],
            physical_material_exposure_mass=float(mass.sum()), production_normal_checks=checked,
            same_q_only=local, nominal_and_eight_actual_camera_trials=joint, actual_camera_trials=trials,
            source_original_UV_sigma_sides_quantiles_proofs_changed=False,
            source_native_frame_or_HELD_subsampling=False, normal_interval_dead_zone_added=False)
        records.append(record)
        payload.update({f'board_row{ri}_uv': uv, f'board_row{ri}_sigma': sigma,
            f'board_row{ri}_original_weights': mass, f'board_row{ri}_nominal_weights': local_weights,
            f'board_row{ri}_nominal_and_trial_weights': joint_weights,
            f'board_row{ri}_protected': protected, f'board_row{ri}_r': r,
            f'board_row{ri}_camera15_J': j15, f'board_row{ri}_view4_J': j4})
    groups = {}
    for i, h in enumerate(pack.rows):
        key = (h['shot_id'], h['actual_pts'], h['physical_material_id'])
        if i in by_record and by_record[i]['split']=='fit':
            groups[key] = groups.get(key, 0)+sum(by_record[i]['point_integration_weights'])
    return dict(source_reference=source_ref, frozen_q_sha256=identity['q_sha256'],
        frozen_identity_reference=reference(identity_path), scene_reference=reference(scene_path),
        entire_pack_source_points=len(pack.values['uv']), entire_pack_source_rows=len(pack.rows),
        physical_material_exposure_groups=len(groups),
        prepared_mass_max_absolute_error_from_one=float(max(abs(v-1) for v in groups.values())),
        held_points=plan['held_points'], unsupported_source_points=len(plan['pending_support_global_point_indices']),
        checked_rows=records, production_operator='absolute_value_jac15 and production streaming source normal on frozen exported finite board material; exact physical camera25(view_state4)',
        no_substituted_inverse_projection_or_projected_source_UV=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-work', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix('.npz').exists():
        raise FileExistsError('Preserve prior capacity diagnostics')
    frozen = args.source_work/'source_inventory/source_full_precision_spill_retry3_FROZEN_v1.json'
    fingerprint = reference(frozen)
    new_module = ROOT/'calibration/program_tracking/source_information_moments.py'
    if str(new_module.name) in frozen.read_text(encoding='utf8'):
        raise ValueError('Diagnostic must not modify an active frozen source reference')
    payload = {}
    report = dict(schema='rinkspace.actual-production-source-capacity-information-and-camera-trials.v1',
        corrected_ice_same_q=corrected_ice(args.source_work, payload),
        actual_dense_board=board_probe(args.source_work, payload),
        diagnostic_implementation=reference(Path(__file__)), moment_implementation=reference(new_module),
        active_source_manifest_reference=fingerprint, original_source_packs_changed=False,
        complete_100_world_information_verified=False, installation_shared_optics_mount_information_verified=False,
        entire_iteration_speed_verified=False, whole_shot_visual_accuracy_accepted=False,
        representative_consumer_enabled=False, solver_admission_verified=False, solver_started=False,
        diagnostic_only=True, finite_checked_camera_trials_are_not_global_nonlinear_information_proof=True)
    if reference(frozen)!=fingerprint:
        raise ValueError('Active source manifest changed during independent audit')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    np.savez_compressed(args.output.with_suffix('.npz'), **payload)
    print(json.dumps(dict(report=str(args.output), ice=[(r['source_points'],r['retained_points']) for r in report['corrected_ice_same_q']['records']],
        board=[dict(material=r['material'],source=r['same_q_only']['source_points'],
            nominal=r['same_q_only']['retained_points'],with_trials=r['nominal_and_eight_actual_camera_trials']['retained_points']) for r in report['actual_dense_board']['checked_rows']],
        consumer_enabled=False)), flush=True)


if __name__=='__main__':
    main()
