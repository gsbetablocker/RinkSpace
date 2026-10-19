"""Replay saved non-FM native truth carriers without fit, J or source images."""
from pathlib import Path
import copy
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_saved_native29_curvature_20261011 as common
import numpy as np
from calibration.solver_runtime.finite_rectangle_station_step_constraints import (
    _current_boundary_clipped_primitive, _current_vertical_print_primitive)

ROOT = common.ROOT
RUN = Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/calibration/runs/solver_acceleration_20261003/okada_review86_q276_vertical_print_station_one_cycle_20261011')
CONFIG = ROOT / 'events/ankara_2026/programmes/okada_mei_sp/full_source_review86_q276_vertical_print_station_one_cycle_20261011.json'
PARENT = common.FINAL
CHILD = '8a15a4ed0c82beb490993db9c58294af592096e8159d9652b5bd5c50217c71eb'
OUT = Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/diagnostics/current56_shared_active_set_filter_20261009/systematic_continuation_20261010/saved_other_primitive_frozen_dynamic_carrier_audit_20261011.json')


def main():
    started = time.perf_counter()
    options = common.read(CONFIG)['source_options']
    directory = common.RUN / 'accepted_states' / PARENT
    identity = common.read(directory / 'state_identity.json')
    path = directory / 'complete_state_UNACCEPTED.npz'
    assert common.sha(path) == identity['state_npz_sha256']
    assert common.sha(identity['camera_path']) == identity['camera_sha256']
    keys = ('q', 'aux__canonical_pts', 'aux__block_lengths', 'aux__block_start', 'aux__block_names',
        'aux__installation_xyz_slots', 'aux__current_shared_mount_parameter_indices',
        'aux__current_all_installation_optics_parameter_indices', 'aux__correlated_roll_prior_contract_json',
        'aux__current_complete_parameter_layout_json')
    with np.load(path, allow_pickle=False) as archive:
        small = {k: archive[k].copy() for k in keys}
    provider, world, proof = common.current_complete_source_geometry(path, options['--world-init'])
    with np.load(identity['camera_path'], allow_pickle=False) as archive:
        cameras = {k: archive[k].copy() for k in archive.files}
    state, error = common.state_camera_adapter(small, cameras, provider)
    assert error == 0.
    manifest = common.read(RUN / 'input_manifest.json')
    state.current_user_pivot_outside_constraint_contract = manifest['camera_pivot_exterior_constraints']
    scope = manifest['additional_real_native_board_edges'].get('physical_course_scope_authority')
    if scope is not None:
        state.current_user_physical_course_scope_source = scope['path']
    board = common.NativeBoardEdgeFactor(state, options['--additional-board-source'], len(small['q']))
    ice = common.NativeIceCurveFactor(state, options['--additional-ice-source'], len(small['q']))
    human = common.HumanStrokeTruth(state, options['--direct-human-truth-authority'], board, ice,
                                    parameter_count=len(small['q']))
    human.bind_full_stroke_guard(options['--direct-human-full-stroke-guard'])
    human.bind_stroke_cap_support(options['--direct-human-stroke-cap-support'])
    human.bind_practical_location_envelope(options['--practical-human-location-envelope'])
    files = list((RUN / 'actual_complete_source_directions').glob(PARENT + '_*.npz'))
    assert len(files) == 1
    with np.load(files[0], allow_pickle=False) as archive:
        q = archive['q'].copy()
        step = archive['step'].copy()
    assert common.asha(q) == PARENT and step.shape == q.shape == (34340,)
    eq, ed = state.evaluation_q(q), state.evaluation_q(step)
    child_path = RUN / 'accepted_states' / CHILD / 'complete_state_UNACCEPTED.npz'
    child_identity = common.read(child_path.parent / 'state_identity.json')
    assert common.sha(child_path) == child_identity['state_npz_sha256']
    with np.load(child_path, allow_pickle=False) as archive:
        child = archive['q'].copy()
    accepted, retraction = common.retract(state, eq + .5 * ed)
    np.testing.assert_array_equal(state.canonical_parameter_layout.canonical_q(accepted), child)
    nominal = human.queries(eq, full_stroke=True)
    full = human.queries(eq + ed, full_stroke=True)
    accepted_queries = human.queries(accepted, full_stroke=True)
    inventory, results = [], []
    for group, (record, base_result, trial_result, accepted_result) in enumerate(zip(
            human.full_stroke_records, nominal, full, accepted_queries)):
        base, trial = base_result['query'], trial_result['query']
        vertical = _current_vertical_print_primitive(record, eq)
        boundary = None if vertical is not None else _current_boundary_clipped_primitive(record, eq)
        converted = vertical is not None or boundary is not None
        base_signature = np.asarray(base.get('source_support_branch_signature'))
        trial_signature = np.asarray(trial.get('source_support_branch_signature'))
        signature_available = base_signature.ndim == 2 and trial_signature.shape == base_signature.shape
        changed = np.flatnonzero(np.any(base_signature != trial_signature, axis=1)) if signature_available else np.empty(0, int)
        primitive = record['factor']._world(eq, record['row']['curve'])
        meta = dict(group=group, id=base_result['id'], curve=record['row']['curve'],
            actual_pts=base_result['actual_pts'], source_guard_vertices=len(record['uv']),
            physical_world_samples=len(primitive), exact_world_loop_closure=np.array_equal(primitive[0], primitive[-1]),
            station_FM_converted=converted, source_association=base.get('finite_source_association'),
            full_step_changed_source_branch_vertices=len(changed), branch_signature_available=signature_available)
        inventory.append(meta)
        if converted or not len(changed) or base.get('finite_source_association') != 'original_support_rectangle_Linf':
            continue
        local = copy.copy(human)
        local.records = [human.records[group]]
        local.full_stroke_records = [record]
        local.blocks = {record['row']['actual_pts']: human.blocks[record['row']['actual_pts']]}
        block = local.blocks[record['row']['actual_pts']]

        def screen_delta_signs(value):
            camera = block.bundle.camera(common.native_camera_local(block,
                value[:state.structured_programme_coordinate_chart.core_count]), record['row']['actual_pts'])
            points = common.project_central_valid_full_brown(camera,
                record['factor']._world(value, record['row']['curve']))[0]
            segments = np.asarray(base['nominal_finite_segments'], int)
            delta = points[segments+1] - points[segments]
            normal = record['normals']
            tangent = np.c_[-normal[:, 1], normal[:, 0]]
            return np.c_[np.einsum('ni,ni->n', delta, normal), np.einsum('ni,ni->n', delta, tangent)]

        nominal_slopes = screen_delta_signs(eq)

        def frozen(value):
            camera = block.bundle.camera(common.native_camera_local(block,
                value[:state.structured_programme_coordinate_chart.core_count]), record['row']['actual_pts'])
            return common.finite_stroke_queries(camera, record['factor']._world(value, record['row']['curve']),
                record['uv'], record['normals'], record['bounds'], tangent_bounds=record['tangent_bounds'],
                frozen_segments=base['nominal_finite_segments'],
                frozen_segment_categories=base['nominal_finite_segment_categories'],
                frozen_support_faces=base['nominal_source_support_faces'],
                frozen_station_branches=base['nominal_source_station_branches'])

        base_carrier = common.carrier(base)
        frozen_at_base = common.carrier(frozen(eq))
        if np.max(abs(frozen_at_base - base_carrier)) > 1e-8:
            raise ValueError('Nominal producer and frozen nominal chart differ')
        # Directional-only value differences. No truth.jacobian/constraints or source graph J.
        differences = []
        try:
            for h in (1e-3, 1e-4):
                f = (common.carrier(frozen(eq + h * ed)) - common.carrier(frozen(eq - h * ed))) / (2 * h)
                d = (common.carrier(local.queries(eq + h * ed, full_stroke=True)[0]['query']) -
                     common.carrier(local.queries(eq - h * ed, full_stroke=True)[0]['query'])) / (2 * h)
                differences.append((h, f, d))
        except ValueError as exc:
            results.append(dict(meta, frozen_domain_error=str(exc)))
            continue
        directional = differences[-1][1]
        lower = np.c_[record['bounds'][:, 0], record['tangent_bounds'][:, 0]].ravel()
        upper = np.c_[record['bounds'][:, 1], record['tangent_bounds'][:, 1]].ravel()
        samples = []
        cases = [('saved_full', 1., eq + ed), ('saved_half_before_retraction', .5, eq + .5 * ed),
                 ('saved_quarter', .25, eq + .25 * ed)]
        for label, alpha, value in cases:
            dynamic_query = local.queries(value, full_stroke=True)[0]['query']
            dynamic = common.carrier(dynamic_query)
            try:
                frozen_query = frozen(value)
                fixed = common.carrier(frozen_query)
            except ValueError as exc:
                samples.append(dict(label=label, alpha=alpha, frozen_domain_error=str(exc)))
                continue
            prediction = base_carrier + alpha * directional
            delta = fixed - dynamic
            worst = int(np.argmax(abs(delta)))
            vertex = worst // 2
            sig = np.asarray(dynamic_query['source_support_branch_signature'])
            actual_guard_worst = int(np.argmax(abs(dynamic_query['residual'])))
            guard_vertex = actual_guard_worst // 2
            slopes = screen_delta_signs(value)
            samples.append(dict(label=label, alpha=alpha,
                changed_branch_vertices=int(np.count_nonzero(np.any(sig != base_signature, axis=1))),
                maximum_frozen_vs_dynamic_carrier_difference_px=float(abs(delta).max()),
                frozen_chart_curvature_maximum_px=float(abs(fixed - prediction).max()),
                dynamic_chart_response_minus_nominal_derivative_maximum_px=float(abs(dynamic - prediction).max()),
                actual_original_guard_maximum_violation_px=float(abs(dynamic_query['residual']).max()),
                frozen_chart_maximum_outside_source_rectangle_px=float(abs(fixed - np.clip(fixed, lower, upper)).max()),
                frozen_local_segment_slope_sign_changed_vertices=int(np.count_nonzero(
                    np.any(np.sign(slopes) != np.sign(nominal_slopes), axis=1))),
                actual_guard_worst_vertex=guard_vertex,
                actual_guard_worst_axis='normal' if actual_guard_worst % 2 == 0 else 'tangent',
                actual_guard_worst_nominal_branch=base_signature[guard_vertex],
                actual_guard_worst_dynamic_branch=sig[guard_vertex],
                actual_guard_worst_dynamic_curvature_vector=dynamic[2*guard_vertex:2*guard_vertex+2]-prediction[2*guard_vertex:2*guard_vertex+2],
                actual_guard_worst_frozen_curvature_vector=fixed[2*guard_vertex:2*guard_vertex+2]-prediction[2*guard_vertex:2*guard_vertex+2],
                worst_disagreement_vertex=vertex, worst_disagreement_axis='normal' if worst % 2 == 0 else 'tangent',
                worst_nominal_source_UV=record['uv'][vertex], nominal_branch=base_signature[vertex],
                dynamic_branch=sig[vertex], nominal_carrier=base_carrier[2*vertex:2*vertex+2],
                dynamic_carrier=dynamic[2*vertex:2*vertex+2], frozen_carrier=fixed[2*vertex:2*vertex+2],
                directional_derivative=directional[2*vertex:2*vertex+2],
                nominal_local_segment_normal_tangent_slopes=nominal_slopes[vertex],
                trial_same_segment_normal_tangent_slopes=slopes[vertex],
                source_normal_bounds=record['bounds'][vertex], source_tangent_bounds=record['tangent_bounds'][vertex]))
        results.append(dict(meta, nominal_frozen_query_maximum_difference_px=float(abs(frozen_at_base - base_carrier).max()),
            frozen_directional_FD_1e3_vs_1e4_maximum_difference_px=float(abs(differences[0][1] - differences[1][1]).max()),
            dynamic_vs_frozen_directional_FD=[dict(h=h, maximum_difference_px=float(abs(d-f).max())) for h, f, d in differences],
            accepted_q8a15_actual_guard_maximum_violation_px=float(abs(accepted_result['query']['residual']).max()),
            samples=samples))
        print(common.json.dumps(dict(stage='readonly_other_carrier', group=group, id=meta['id'],
            changed_vertices=len(changed), maximum_disagreement=max((s.get('maximum_frozen_vs_dynamic_carrier_difference_px', 0.) for s in samples), default=0.))), flush=True)
    source_files = ['calibration/solver_runtime/finite_rectangle_station_step_constraints.py',
        'calibration/solver_runtime/legacy/structured_programme_source_candidate/source_step_curvature_tightening.py',
        'calibration/solver_runtime/legacy/structured_programme_source_candidate/human_constraint_dependency_fd.py',
        'calibration/solver_runtime/legacy/structured_programme_source_candidate/human_stroke_truth.py',
        'calibration/solver_runtime/legacy/structured_programme_source_candidate/normal_scanline_human_query_acceleration.py',
        'calibration/solver_runtime/legacy/run_current_named66_full_programme.py']
    report = dict(schema='rinkspace.saved-other-primitive-frozen-dynamic-carrier-audit.v1',
        new_fit=False, source_or_truth_J_generated=False, directional_value_differences_only=True,
        source_images_written=0, original_source_supports_limits_and_sampling_unchanged=True,
        script=common.ref(__file__), config=common.ref(CONFIG), parent_state=common.ref(path),
        parent_identity=common.ref(directory/'state_identity.json'), parent_camera=common.ref(identity['camera_path']),
        child_state=common.ref(child_path), child_identity=common.ref(child_path.parent/'state_identity.json'),
        saved_direction=common.ref(files[0]), accepted_half_reproduced_bit_for_bit=True,
        accepted_pivot_retraction=dict(changed_columns=retraction['changed_columns'],
            before_valid=retraction['before']['valid'], after_valid=retraction['after']['valid']),
        camera_archive_reproduction_maximum=error,
        geometry_proof={k:v for k,v in proof.items() if k != 'explicit_saved_physical_geometry_initializer'},
        source_references=[common.ref(options[k]) for k in ('--additional-board-source','--additional-ice-source',
            '--direct-human-truth-authority','--direct-human-full-stroke-guard',
            '--direct-human-stroke-cap-support','--practical-human-location-envelope')],
        implementation_sources=[common.ref(ROOT/name) for name in source_files],
        dispatch_inventory=inventory, changed_nonFM_primitives=results,
        findings=[
            'Nominal carrier and the four-field frozen-query replay are exactly equal for every replayed group. Small frozen directional differences are stable, but full dynamic witness switches can move carrier coordinates by many pixels while true rectangle violation stays zero.',
            'Replacing SOC responses with frozen replay is sufficient to align that local formula with its J away from sign/domain branch events. It is not sufficient to preserve all admissible steps: the saved full F2/R1/A4 steps can be dynamically feasible while the frozen local cell is outside the original rectangle.',
            'The frozen API retains segment/category/support-face/station-formula identity; it recomputes station t from trial projections and may extend an interior computational cell. It does not freeze the world/source witness point. Its slab formulas also branch on current signed projected slopes.',
            'A consistent repair must separate numerical-chart response from the unchanged dynamic nonlinear guard, and must treat changed witness/chart events explicitly. World straightness alone does not certify a globally straight Brown projection or justify remote caps in one local sampling-cell chart.'
        ],
        limits=['Only native54 full-stroke groups and the saved q276 direction examined; wide/confirmed routes not numerically certified here.',
            'Directional finite differences follow the exact nominal frozen query used by carrier J, but no new full or local source J is formed.',
            'Fixed-chart trial is a local numerical response, not a replacement physical nonlinear guard; curve cell changes may require chart renewal.',
            'Whole straight-line certificates and endpoint caps cannot be extrapolated to circles, arcs or developed curved print borders.'],
        seconds=time.perf_counter()-started)
    OUT.write_text(common.json.dumps(report, default=common.plain, ensure_ascii=False, indent=2), encoding='utf-8')
    print(common.json.dumps(dict(path=str(OUT), sha256=common.sha(OUT), bytes=OUT.stat().st_size,
        examined_native_groups=len(inventory), changed_nonFM_groups=len(results), seconds=report['seconds'])), flush=True)


if __name__ == '__main__':
    main()
