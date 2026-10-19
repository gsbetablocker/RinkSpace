"""Continue the complete low-end rig with reviewed independent source frames.

The complete programme objective and original prior targets are retained.
Only source-reference nodes and factors are added; no camera is fixed.
This is an independent entry so running long-side solves keep their source.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from types import SimpleNamespace

import fit_continuous_rig_source_calibrated as driver
from continuous_camera_solver import save_states, write_json, complete_graph_error, solve
from continuous_low_end_reference_calibration import prepare_low_end_references
from continuous_reference_calibration import seed_reference_initial, attach_reference_factors
from low_end_reference_source_loader import load_inputs
import continuous_precise_inverse_optin as precise
import numpy as np


def _continued_reference_code(phase):
    """Do not warm old reference nodes into a silently revised source model."""
    captured = {name.replace('\\', '/'): digest
                for name, digest in phase.get('code_dependencies', {}).items()}
    names = ['low_end_reference_source_loader.py',
             'continuous_low_end_reference_calibration.py',
             'continuous_reference_calibration.py',
             'continuous_precise_inverse_optin.py']
    if phase.get('correlated_user_f1', {}).get('enabled'):
        names.append('continuous_low_end_f1_correlated_optin.py')
    for name in names:
        suffix = '/calibration/program_tracking/' + name
        matches = [digest for path, digest in captured.items()
                   if ('/' + path).endswith(suffix)]
        if (len(matches) != 1
                or matches[0] != driver.base.sha(Path(__file__).with_name(name))):
            raise ValueError('Continued low-end reference implementation changed: '+name)


def initializer_map(path):
    packet = json.loads(path.read_text(encoding='utf8'))
    if packet['role'] != 'initial_values_only_not_observations_or_priors':
        raise ValueError('Camera starts must not be observation targets')
    cameras = {}
    for row in packet['cameras']:
        for source in row['source_files']:
            if driver.base.sha(Path(source['path'])) != source['sha256']:
                raise ValueError('Reference camera initialization source changed')
        pts = int(row['source_pts'])
        if pts in cameras:
            raise ValueError('Duplicate reference initializer')
        camera = np.asarray(row['camera25'], float)
        if camera.shape != (25,) or not np.isfinite(camera).all():
            raise ValueError('Reference initializer is not finite camera25')
        if row['coordinate_frame_status'] != 'same_rink_axes_and_conditional_scene_units':
            raise ValueError('Reference initializer coordinate frame has not been checked')
        cameras[pts] = camera
    if set(cameras) != {14177280, 205992960, 226160640}:
        raise ValueError('Reference initializer must cover exactly the three reviewed exposures')
    return cameras, dict(path=str(path.resolve()), sha256=driver.base.sha(path),
                         packet=packet)


def run(args, resources):
    base = driver.base
    output = args.output.resolve()
    if output.exists() or not output.is_relative_to(base.TMP.resolve()):
        raise ValueError('Use a fresh output inside programme TMP')
    output.mkdir(parents=True)
    started = time.perf_counter()
    previous = json.loads((args.warm_group/'phase_00.json').read_text(encoding='utf8'))
    if previous['camera_id'] != 'low_end':
        raise ValueError('Only the full low-end group may initialize this entry')
    from continuous_adaptive_curve_implicit_optin import install_adaptive_implicit_optin
    from continuous_central_brown4_optin import enable_projection_backend
    from continuous_central_edge_optin import install_edge_optin, registered_edge_adapter
    install_adaptive_implicit_optin()
    central_backend = enable_projection_backend()
    install_edge_optin()
    base.adapt_factor = registered_edge_adapter
    data, pairs, inputs = base.prepare_inputs(output, 'low_end')
    aliases, alias_meta = base.load_exposure_aliases(data)
    clock, clock_meta, valid_pairs = base.build_exposure_clock(data, aliases)
    model, programme_prior, _, init_meta = base.make_rig(data, aliases, clock, 1., True)
    adapted = driver.reuse_prior_initializer(args.prior_state_cache, model, programme_prior, data)
    prior_copy = {k: v.copy() for k, v in programme_prior.items()}
    pairs = base.extended_native_pairs(pairs)
    from continuous_board_first_hit_optin import upgrade_pairs
    pairs, board_backend = upgrade_pairs(pairs)
    # Same complete-class geometry exclusion as the original source-calibrated entry.
    uncertain_ad4 = [s for s in data['retained'] if s['file'] == 'ad_landmarks.py'
                    and s['function'] == 'factor' and 1004 in s['keys']]
    data = {**data, 'retained': [s for s in data['retained'] if not (
        s['file'] == 'ad_landmarks.py' and s['function'] == 'factor' and 1004 in s['keys'])]}
    from historic_development_ads import prepare_development_ads, attach_development_ad_factors
    ads, ad_metadata = prepare_development_ads(data, aliases)
    cameras, camera_provenance = initializer_map(args.initializer_map)
    inventory = load_inputs(data['scene'], camera25_initializer_by_pts=cameras)
    references = prepare_low_end_references(data, model, programme_prior, inventory)
    old_references = previous.get('reference_calibration', {})
    continuing = bool(old_references.get('reference_pts'))
    if continuing:
        _continued_reference_code(previous)
        if bool(previous.get('correlated_user_f1', {}).get('enabled')) != args.include_user_f1:
            raise ValueError('Continued low-end run must preserve F1 fit membership')
        warm_prior, reference_metadata = references.initial, references.metadata
    else:
        empty = driver.prepare_source_references(data, model, programme_prior)
        warm_prior, reference_metadata = programme_prior, empty.metadata
    warm, warm_meta = driver.warm_states(args.warm_group, model, warm_prior, data, aliases,
        state_filename=args.warm_state, reference_metadata=reference_metadata,
        allow_projection_additions=True)
    with np.load(args.warm_group/'prior_initializer_states.npz', allow_pickle=False) as z:
        old_prior = {int(k): v for k, v in zip(z['keys'], z['states'])}
    if old_prior.keys() != warm_prior.keys() or any(
            not np.array_equal(old_prior[k], warm_prior[k]) for k in old_prior):
        raise ValueError('Original saved prior targets were not reproduced exactly')
    initial = {k: v.copy() for k, v in references.initial.items()}
    initial.update(warm)
    if not continuing:
        seed_reference_initial(references.model, initial, references.references)
    native_args = SimpleNamespace(native_source_edges=True, native_affine_priors=True,
                                  native_edge_threads=args.native_edge_threads)
    manager, affine = driver.make_native_backend(native_args, model, references.initial, resources)
    graph, counts = base.build_graph(data, pairs, model, references.initial, clock,
        valid_pairs, .25, native_edge_manager=manager, graph_owner=resources, affine_priors=affine)
    resources['graph'] = graph
    graph._adaptive_curve_worker_count = args.reference_curve_workers
    if counts != previous['factor_counts']:
        raise ValueError('Programme factor-family counts changed while adding references')
    ad_factors = attach_development_ad_factors(graph, data, model, initial, ads)
    if ad_factors != previous['development_ad_factors']:
        raise ValueError('Programme advertisement factor identity changed')
    programme_factor_count = len(graph._pending)
    previous_programme_factors = previous.get('programme_factors_before_new_references',
                                               previous['graph_factors_with_source_calibration'])
    if programme_factor_count != previous_programme_factors:
        raise ValueError('Complete original programme factor count changed')
    if continuing and previous['graph_factors_with_source_calibration'] != (
            previous_programme_factors + previous['new_reference_factor_count']):
        raise ValueError('Continued reference factor count is inconsistent')
    original_preflight = base.preflight(pairs, model, initial)
    precise.install()
    validity = base.preflight(pairs, model, initial)
    write_json(output/'initial_source_geometry.json', validity)
    if validity['invalid_pair_count']:
        raise ValueError('Warm complete programme has invalid original source geometry')
    programme_cost = complete_graph_error(graph, initial, native_edge_manager=manager,
        diagnostics_path=output/'programme_objective_diagnostics.json')
    if len(graph._pending) != programme_factor_count or not np.isfinite(programme_cost):
        raise ValueError('Original programme objective changed size or is nonfinite')
    ref_metadata = attach_reference_factors(graph, data, references, adaptive_curves=True)
    f1_metadata = {'enabled': False}
    if args.include_user_f1:
        from continuous_low_end_f1_correlated_optin import attach_correlated_f1_factors
        f1_inventory = load_inputs(data['scene'], camera25_initializer_by_pts=cameras,
                                   include_user_f1=True)
        f1_metadata = dict(attach_correlated_f1_factors(
            graph, data, references, f1_inventory), enabled=True)
    added_reference_factors = len(graph._pending) - programme_factor_count
    expected_reference_factors = (ref_metadata['source_measurement_factors']
                                  + ref_metadata['static_c_prior_factors']
                                  + ref_metadata['unused_coordinate_factors']
                                  + f1_metadata.get('ring_count', 0))
    if (added_reference_factors != expected_reference_factors
            or (continuing and added_reference_factors != previous['new_reference_factor_count'])):
        raise ValueError('Independent reference factor count changed')
    if any(not np.array_equal(v, references.initial[k]) for k, v in prior_copy.items()):
        raise RuntimeError('New reference preparation changed an original programme prior')
    phase = {**previous, 'schema': 'rinkspace.shared-rig-original-source-calibration.v1',
        'source_inputs': inputs, 'warm_estimate': warm_meta, 'aliases': alias_meta,
        'exposure_clock': clock_meta, 'initializer': init_meta, 'initializer_adaptation': adapted,
        'initial_geometry': validity, 'factor_counts': counts, 'reference_calibration': ref_metadata,
        'reference_initializer_provenance': camera_provenance,
        'new_low_end_source_references': True, 'reference_states_continued': continuing,
        'development_ad_source': ad_metadata, 'development_ad_factors': ad_factors,
        'board_first_hit_acceleration': board_backend, 'central_domain_acceleration': central_backend,
        'native_source_edge_acceleration': manager.metadata(),
        'native_affine_priors': affine.metadata(), 'adaptive_source_curves': True,
        'graph_factors_with_source_calibration': len(graph._pending),
        'graph_nodes_with_references': len(initial),
        'programme_factors_before_new_references': programme_factor_count,
        'new_reference_factor_count': added_reference_factors,
        'correlated_user_f1': f1_metadata,
        'all_reference_source_curve_groups': ref_metadata['source_curve_group_count']
            + f1_metadata.get('ring_count', 0),
        'all_reference_source_curve_pixels': ref_metadata['source_curve_pixel_count']
            + f1_metadata.get('total_fitted_source_uv', 0),
        'programme_cost_at_warm_state_with_precise_inverse': programme_cost,
        'source_uv_changed': False, 'first_camera_fixed': False, 'optical_center_fixed': False,
        'excluded_unvalidated_ad4_absolute_pixel_factors': len(uncertain_ad4),
        'full_programme_accepted': False}
    phase['code_dependencies'] = driver.snapshot_dependencies(output)
    phase['code_sha256'] = {name: base.sha(Path(base.__file__).with_name(name))
                            for name in previous['code_sha256']}
    write_json(output/'phase_00.json', phase)
    save_states(output/'prior_initializer_states.npz', references.initial)
    base.export(model, initial, data, aliases, output/'initial_cameras.npz')
    np.savez_compressed(output/'initial_reference_cameras.npz', pts=np.asarray(
        [r['source_pts'] for r in references.references]), camera25=np.asarray([
        references.model.camera25(r['source_pts'], initial) for r in references.references]))
    initial_cost = complete_graph_error(graph, initial, native_edge_manager=manager,
        diagnostics_path=output/'initial_solver_diagnostics.json')
    if not np.isfinite(initial_cost):
        raise ValueError('New complete source-reference objective is not finite')
    write_json(output/'initial_graph_error.json', dict(cost=initial_cost,
        programme_cost=programme_cost, programme_factors=programme_factor_count,
        full_factors=len(graph._pending), reference_factors_added=added_reference_factors,
        evaluated_all_factors=True, finite_objective=True))
    write_json(output/'precise_inverse_backend.json', dict(backend=precise.metadata(),
        implementation_sha256=base.sha(Path(precise.__file__)),
        previous_low_end_v5_inverse='legacy_cv2_default_five_iteration',
        current_objective_inverse='continuous_precise_inverse_optin.precise_world_rays',
        previous_objective_cost_not_reused=True,
        original_source_preflight=original_preflight, corrected_source_preflight=validity,
        low_end_previous_solver_used_default_inverse=True))
    print(json.dumps(dict(stage='complete_low_end_with_references_ready',
        programme_frames=len(model.frames), reference_frames=len(references.references),
        source_curve_groups=phase['all_reference_source_curve_groups'],
        source_curve_pixels=phase['all_reference_source_curve_pixels'],
        factors=len(graph._pending), initial_cost=initial_cost, programme_cost=programme_cost,
        seconds=time.perf_counter()-started)), flush=True)
    if args.preflight_only:
        save_states(output/'preflight_initial_states.npz', initial)
        return
    try:
        solved, _ = solve(graph, initial, output, max_iterations=args.iterations,
                           native_edge_manager=manager)
    finally:
        write_json(output/'native_source_edge_runtime.json', manager.metadata())
        write_json(output/'precise_inverse_runtime.json', precise.metadata())
        if args.include_user_f1:
            from continuous_low_end_f1_correlated_optin import runtime_counters
            write_json(output/'correlated_user_f1_runtime.json', runtime_counters(graph))
    report = base.export(model, solved, data, aliases, output/'shot_cameras.npz')
    stamps = np.asarray(sorted(references.model.frames), np.int64)
    np.savez_compressed(output/'reference_cameras.npz', pts=stamps,
        camera25=np.asarray([references.model.camera25(int(p), solved) for p in stamps]),
        source_sha256=data['diag']['source_sha256'], model_sha256=data['diag']['model_sha256'])
    report.update(camera_id='low_end', shot_indices=list(data['shot_indices']),
        source_sha256=data['diag']['source_sha256'], model_sha256=data['diag']['model_sha256'],
        phase_sha256=base.sha(output/'phase_00.json'), seconds=time.perf_counter()-started,
        accepted=False, scope='Complete low-end rig plus source-backed independent exposures')
    write_json(output/'summary.json', report)
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--warm-group', type=Path, required=True)
    parser.add_argument('--warm-state', choices=('states.npz', 'accepted_states.npz'), default='states.npz')
    parser.add_argument('--prior-state-cache', type=Path, required=True)
    parser.add_argument('--initializer-map', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--iterations', type=int, default=80)
    parser.add_argument('--native-edge-threads', type=int, default=8)
    parser.add_argument('--reference-curve-workers', type=int, default=8)
    parser.add_argument('--include-user-f1', action='store_true',
                        help='Add the original three user rings with total Step02 weight budget one')
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    if args.native_edge_threads < 1 or args.reference_curve_workers < 0:
        parser.error('Worker counts must be valid')
    resources = {}
    try:
        run(args, resources)
    finally:
        driver.release_native_resources(resources)


if __name__ == '__main__':
    main()
