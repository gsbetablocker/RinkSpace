"""Small original-wrapper closure test with one real residual-only actor.

The public programme evaluator is a six-row test factor. Original activation,
row selection/patching, bias/placement wrappers and actor transport are real;
no event graph, full matrix or benchmark is constructed.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext, redirect_stdout
import io
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from calibration.solver_runtime import install
install()

import numpy as np
from scipy.sparse import csr_matrix
from structured_programme_source_candidate.paired_programme_blocks import StructuredProgrammeBlock


def tiny_public_evaluate(owner, q, jacobian=True):
    q = np.asarray(q, float)
    pool = getattr(owner, '_absolute_geometry_fd_pool', None)
    if jacobian and pool is not None:
        groups = ((pts, rows, np.zeros(25)) for pts, rows in owner.groups)
        list(pool.ordered_groups(groups, np.zeros(1), None, 1e-7, batched_projection=True))
    residual = []
    gradient = []
    if owner.bundle.motion_rows:
        positive = max(q[0] - 1.5, 0.)
        weights = owner.bundle.profile_weights.reshape(-1)
        residual.extend([positive ** 2 * weights[0], 0.])
        gradient.extend([[2. * positive * weights[0], 0.], [0., 0.]])
    positive = max(q[0], 0.)
    for unused_pts, records in owner.groups:
        for record in records:
            residual.extend(np.asarray(record['offset']) + np.asarray(record['coefficient']) * positive)
            gradient.extend([[value if q[0] > 0. else 0., 0.] for value in record['coefficient']])
    result = np.asarray(residual)
    return (result, csr_matrix(np.asarray(gradient))) if jacobian else result


# Spawn imports this diagnostic module before unpickling the tiny real-class
# owner. These evaluator substitutions exist only in the test process tree.
StructuredProgrammeBlock.evaluate = tiny_public_evaluate
StructuredProgrammeBlock._evaluate = tiny_public_evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from calibration.solver_runtime.activation_refresh_runtime import install_parent_activated_source_refresh
    from structured_programme_source_candidate import activated_small_source_blocks as activated
    from structured_programme_source_candidate.activated_source_row_subset import replace_csr_rows
    from structured_programme_source_candidate.current_full_source import CurrentProgrammeSource
    from structured_programme_source_candidate.residual_only_source_actors import ResidualOnlySourceActors
    from structured_programme_source_candidate.course_patterns_independent_placements import SourceWithIndependentPatternPlacements, PLACEMENT_BINDING
    from structured_programme_source_candidate.paired_programme_chart import CanonicalCompleteProgrammeSource
    from structured_programme_source_candidate.lagged_jacobian_source import LaggedJacobianSource
    from correlated_roll_prior_candidate.explicit_bias_source import ExplicitBiasSource, SIGMA
    from z190_joint_source_solver import _BlockProfileUpdate, _frozen_profile_token
    from z190_streaming_source_normal import SourceRowBlock, _checked_block, accumulate_source_normal
    from z190_continuous_source_bundle import SOURCE_SHA
    from calibration.solver_runtime.completion_driven_absolute_fd_runtime import install_completion_driven_absolute_fd_runtime
    from calibration.solver_runtime import absolute_geometry_fd_pool_candidate as pool_source

    previous_validation = (json.loads(args.output.read_text(encoding='utf-8'))
                           if args.output.exists() else None)
    owner = StructuredProgrammeBlock.__new__(StructuredProgrammeBlock)
    owner.name = 'main1'
    owner.groups = [(0, [dict(uv=np.zeros((1, 2)), offset=[0., 0.], coefficient=[1., 2.]),
                         dict(uv=np.zeros((1, 2)), offset=[3., 4.], coefficient=[3., 4.])])]
    owner.geometry_parameter_indices = np.arange(50)
    owner.bundle = SimpleNamespace(
        motion_rows=2, source_residual_rows=6, profile_weights=np.array([[.7, .8]]), robust_phase=True,
        stamps=np.empty(0, int), index={0: 0}, calibration=SimpleNamespace(installation_id='test'),
        a=np.zeros((1, 2)), b=np.zeros((1, 2)), ia=np.zeros(1, int), ib=np.zeros(1, int),
        motion_sigma=np.ones(2), finite_planes=np.empty((0, 4)), finite_mask=np.empty(0, bool),
        native_residual_sigma=np.ones(6), scene=dict(ice_boundary={}, board_surface={}))
    owner.actor_output_capacity = dict(rows=6, nnz=12, motion_rows=2)
    programme = CurrentProgrammeSource.__new__(CurrentProgrammeSource)
    programme.state = SimpleNamespace(blocks=(owner,))
    programme.actors = None
    programme.profile_update = _BlockProfileUpdate((owner,))
    programme.quadratic = np.zeros(6, bool)
    programme.scales = np.ones(6)
    bias = ExplicitBiasSource.__new__(ExplicitBiasSource)
    bias.cap = dict(iter_source_blocks=programme.blocks, scales=programme.scales)
    bias.programme = {'main1': owner}
    bias.core_count, bias.count = 2, 5
    bias.slots = np.arange(2, 5)
    bias.order = ('test',)
    bias.sigma = SIGMA
    bias.validated_programme = {'programme:main1'}
    bias.validated_external = False
    placement = SourceWithIndependentPatternPlacements.__new__(SourceWithIndependentPatternPlacements)
    placement.base = bias
    placement.prefix_count = placement.count = 5
    placement.placement_count = 0
    placement.placement_indices = np.empty(0, int)
    placement.pattern_placement_context = lambda q: nullcontext()
    placement._placement_columns = lambda q, family: None
    source = CanonicalCompleteProgrammeSource.__new__(CanonicalCompleteProgrammeSource)
    source.base, source.count = placement, 5
    source.parameter_layout = SimpleNamespace(evaluation_q=lambda q: np.asarray(q).copy(), additional_world_parameter_count=0)
    q0, q1, q2 = (np.array([value, 0., 0., 0., 0.]) for value in (0., 1., 2.))
    ledger = dict(family='programme:main1', original_start=0, original_stop=6,
                  source_sha256=SOURCE_SHA, statistical_roll_bias_inst=0,
                  original_source_order_retained=True, prior_covariance_changed_explicitly=True,
                  independent_pattern_placement_binding=PLACEMENT_BINDING,
                  independent_pattern_placement_parameter_indices=[])
    # Original actor-free helper is the byte-for-byte reference for every
    # retained wrapper and derivative. Avoid the whole source iterator.
    baseline = {value: activated.fresh_activated_small_source_block(
        source, q, SourceRowBlock(tiny_public_evaluate(owner, q[:2], False), None,
                                 programme.quadratic, programme.scales, ledger))
                for value, q in ((0, q0), (1, q1), (2, q2))}
    assert all(block is not None for block in baseline.values())
    manager = ResidualOnlySourceActors((owner,), task_root=ROOT / 'calibration' / 'solver_runtime' / 'legacy',
                                      native_threads=1, q_columns=2)
    programme.actors = manager
    programme.profile_update = manager.profile_update
    callback = activated.make_activated_small_source_block_refresh(source)
    supplied1 = SourceRowBlock(baseline[1].r.copy(), None, baseline[1].quadratic,
                              baseline[1].scales, baseline[1].row_ledger)
    assert callback(q1, supplied1) is None
    assert callback.exact_row_subset_refresh(q1, supplied1, np.array([2])) is None
    log = io.StringIO()
    report = dict(original_blanket_actor_refusal_reproduced=True,
                  event_graph_loaded=False, full_QP_or_benchmark_recomputed=False,
                  test_public_evaluator_has_six_rows=True)
    original_group_producer = pool_source._evaluate_group_on_owner
    def tiny_FD_group(job, geometry, certificates, route):
        assert route == 'parent_actor_busy_activation'
        assert job['batched_projection'] is True and job['step'] == 1e-7
        return dict(geo=np.zeros((2 * sum(len(row['uv']) for row in job['rows']), 1)),
                    ordinal=job['ordinal'], actual_pts=job['actual_pts'],
                    source_group_sha256=job['source_group_sha256'],
                    stats=dict(actual_route=route))
    try:
        manager.start()
        with redirect_stdout(log):
            install_completion_driven_absolute_fd_runtime()
            pool_source._evaluate_group_on_owner = tiny_FD_group
            pool = pool_source.AbsoluteGeometryGroupPool(dict(sha256='six_row_test_material'), workers=4,
                groups_per_job=16, parent_geometry=SimpleNamespace(parameter_count=1))
            def forbid_worker_or_headroom(*args):
                raise AssertionError('Activation FD attempted a worker or fabricated headroom')
            pool._headroom = forbid_worker_or_headroom
            pool._executor = forbid_worker_or_headroom
            owner._absolute_geometry_fd_pool = pool
            report['installation'] = install_parent_activated_source_refresh()
            token = _frozen_profile_token(programme.profile_update)
            anchor = baseline[0]
            accumulated = accumulate_source_normal(lambda: iter((anchor,)), 5, expected_row_count=6,
                frozen_profile_token=token, retain_native_replay=True, source_state=q0)
            model = LaggedJacobianSource(.25)
            model.begin(q0, token)
            model.capture(accumulated.native_replay_cache)
            # Consume a real copied actor residual while its original ordered
            # iterator is still busy. Parent J does not issue a second command.
            iterator = manager.ordered_evaluations(q1[:2], False)
            unused, residual = next(iterator)
            assert manager.delegate._busy is True
            actual1 = SourceRowBlock(residual, None, supplied1.quadratic, supplied1.scales, supplied1.row_ledger)
            assert np.array_equal(actual1.r, supplied1.r)
            patch = callback.exact_row_subset_refresh(q1, actual1, np.array([2]))
            assert np.array_equal(patch.row_indices, [2, 3])
            assert np.array_equal(patch.matrix.data, baseline[1].J[patch.row_indices].data)
            assert np.array_equal(patch.matrix.indices, baseline[1].J[patch.row_indices].indices)
            assert np.array_equal(patch.matrix.indptr, baseline[1].J[patch.row_indices].indptr)
            signature = _checked_block(actual1, 5, False)[-1]
            replaced = replace_csr_rows(anchor.J, patch, np.array([2]), actual1.r, signature.inventory_sha256)
            for row in (0, 1, 4, 5):
                assert np.array_equal(replaced[row].data, anchor.J[row].data)
                assert np.array_equal(replaced[row].indices, anchor.J[row].indices)
            assert model.begin(q1, token) is True
            result1 = next(model.blocks(lambda: iter((actual1,)),
                fresh_activated_rows=lambda supplied, rows: callback.exact_row_subset_refresh(q1, supplied, rows),
                fresh_activated_block=lambda supplied: callback(q1, supplied)))
            assert np.array_equal(result1.J.data, replaced.data)
            assert model.report()['exact_activated_source_row_derivative_refreshes'][0]['activated_local_rows'] == [2, 3]
            assert manager.delegate._busy is True
            assert _frozen_profile_token(programme.profile_update) == token
            assert not hasattr(pool, '_rinkspace_force_parent_activation_FD')
            assert pool._rinkspace_active_geometry_group_iterators == 0
            assert pool.last_completion_scheduling['force_parent_activation'] is True
            iterator.close()
            report.update(actual_residual_actor_busy_exact_record_subset_passed=True,
                          requested_one_scalar_refreshed_original_two_UV_record=True,
                          row_subset_r_J_and_inventory_match_original_actor_free_reference=True,
                          untouched_CSR_rows_byte_identical=True,
                          original_lagged_zero_J_gate_used_exact_row_patch=True)
            # Next current q activates an original motion row. The row helper
            # must still refuse the subset and use the original complete family.
            iterator = manager.ordered_evaluations(q2[:2], False)
            unused, residual = next(iterator)
            supplied2 = SourceRowBlock(residual, None, supplied1.quadratic, supplied1.scales, supplied1.row_ledger)
            assert np.array_equal(supplied2.r, baseline[2].r)
            assert callback.exact_row_subset_refresh(q2, supplied2, np.array([0])) is None
            released = []
            def release_original_parent_FD():
                owner._absolute_geometry_fd_pool.release_idle()
                released.append(True)
            programme.after_programme_jacobian = release_original_parent_FD
            assert model.begin(q2, token) is True
            result2 = next(model.blocks(lambda: iter((supplied2,)),
                fresh_activated_rows=lambda supplied, rows: callback.exact_row_subset_refresh(q2, supplied, rows),
                fresh_activated_block=lambda supplied: callback(q2, supplied)))
            assert len(released) == 1 and owner._absolute_geometry_fd_pool.executor is None
            assert np.array_equal(result2.J.data, baseline[2].J.data)
            assert np.array_equal(result2.J.indices, baseline[2].J.indices)
            assert np.array_equal(result2.J.indptr, baseline[2].J.indptr)
            assert np.array_equal(result2.r, baseline[2].r)
            assert result2.row_ledger == baseline[2].row_ledger
            assert manager.delegate._busy is True
            assert _frozen_profile_token(programme.profile_update) == token
            assert model.report()['exact_activated_family_derivative_refreshes'][0]['activated_row_count'] == 1
            original_weights = owner.bundle.profile_weights.copy()
            def corrupt_frozen_profile():
                owner.bundle.profile_weights[0, 0] *= .5
            programme.after_programme_jacobian = corrupt_frozen_profile
            try:
                callback(q2, supplied2)
            except RuntimeError as error:
                assert 'changed frozen profiles or actor transport state' in str(error)
            else:
                raise AssertionError('Changed frozen parent profile was admitted')
            finally:
                owner.bundle.profile_weights[:] = original_weights
            from calibration.solver_runtime.solver_control_runtime import SourceControlStop
            interruption = SourceControlStop(dict(reason='small diagnostic explicit interruption'))
            def interrupt_original_callback():
                raise interruption
            programme.after_programme_jacobian = interrupt_original_callback
            try:
                callback(q2, supplied2)
            except SourceControlStop as error:
                assert error is interruption
            else:
                raise AssertionError('Original control exception did not escape unchanged')
            programme.after_programme_jacobian = release_original_parent_FD
            pool._rinkspace_active_geometry_group_iterators = 1
            assert callback(q2, supplied2) is None
            assert callback.exact_row_subset_refresh(q2, supplied2, np.array([2])) is None
            pool._rinkspace_active_geometry_group_iterators = 0
            iterator.close()
            report.update(original_motion_activation_keeps_complete_family_fallback=True,
                          complete_family_r_J_CSR_and_ledger_bitexact_to_actor_free_reference=True,
                          original_parent_FD_release_callback_allowed_and_actor_transport_preserved=True,
                          frozen_profile_token_unchanged=True,
                          changed_parent_frozen_profile_rejected=True,
                          original_BaseException_control_propagated_unchanged=True,
                          inherited_row_view_FD_pool_forced_to_original_parent_route=True,
                          activation_FD_preserves_batched_projection_step_and_grouping=True,
                          activation_FD_never_checks_headroom_or_starts_workers=True,
                          activation_FD_force_attribute_restored_and_active_count_zero=True,
                          busy_geometry_FD_manager_retains_original_refusal=True)
            # Unsupported actors retain the original safe full-J fallback.
            programme.actors = SimpleNamespace()
            assert callback(q2, supplied2) is None
            assert callback.exact_row_subset_refresh(q2, supplied2, np.array([2])) is None
            programme.actors = manager
            report['unsupported_actor_arrangement_keeps_original_refusal'] = True
    finally:
        pool_source._evaluate_group_on_owner = original_group_producer
        programme.actors = manager
        manager.close()
    report['all_passed'] = True
    if previous_validation is not None:
        report['previous_validation'] = previous_validation
    report['observed_refresh_events'] = [json.loads(line) for line in log.getvalue().splitlines()
                                         if line.startswith('{')]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({key: value for key, value in report.items()
                      if key not in ('installation', 'observed_refresh_events', 'previous_validation')}))


if __name__ == '__main__':
    main()
