"""The fixed-envelope callback must reach actual ordinary source acceptance."""
import numpy as np
from scipy.sparse import csr_matrix

from calibration.solver_runtime import install
from calibration.solver_runtime.fixed_seed_truth_envelope import (
    make_fixed_seed_envelope, fixed_seed_trial_decision,
)


def test_fixed_envelope_accepts_real_streaming_solver_and_preserves_seed_budget(tmp_path):
    install()
    import z190_joint_source_solver as solver
    from z190_streaming_source_normal import SourceRowBlock

    initial = np.array([2., 1.])
    target = np.array([0., 2.])
    mask = np.ones(2, bool)
    scales = np.ones(2)
    jacobian = csr_matrix(np.eye(2))
    limits, origin = make_fixed_seed_envelope(abs(initial), [6., 4.],
        seed_q_sha256='1'*64, group_ids=['source_boundary_A', 'source_boundary_B'])
    accepted = []
    callbacks = []

    def source_blocks(q, derivative):
        yield SourceRowBlock(q-target, jacobian if derivative else None, mask, scales,
            dict(family='actual_callback_interface_regression', original_start=0, original_stop=2))

    def guard(q):
        assert np.all(abs(q) <= limits+1e-8)
        return True

    def source_trial_acceptance(parent, trial, *, before_cost, after_cost,
                                default_source_cost_accepts, trial_context):
        assert trial_context['physical_and_complete_source_guard_passed']
        answer = fixed_seed_trial_decision(abs(parent), abs(trial), limits,
            before_cost=before_cost, after_cost=after_cost,
            default_source_cost_accepts=default_source_cost_accepts)
        callbacks.append(answer)
        return answer

    def checkpoint(q, event):
        if event['checkpoint_kind'] == 'accepted':
            accepted.append(q.copy())
            if len(accepted) == 2:
                return dict(stop_branch=True, reason='Actual callback and continuation interface verified')

    def unused(*args):
        raise AssertionError('The actual streaming source must supply residuals and Jacobians')

    result = solver.solve_joint_source(unused, initial, mask, scales, tmp_path, guard,
        source_block_factory=source_blocks, cache_native_source_replay=True,
        reuse_source_jacobian=True, global_schur_count=1, schur_backtracking=True,
        source_trial_acceptance=source_trial_acceptance,
        checkpoint_callback=checkpoint, verbose=False)

    assert result.status == 'stopped_by_source_review'
    assert len(accepted) == len(result.history) == 2
    assert all(entry['accepted'] for entry in result.history)
    assert accepted[0][1] > initial[1]  # One source group may grow while complete cost falls.
    for entry in result.history:
        decision = entry['explicit_source_trial_acceptance']
        assert decision['mode'] == 'source_objective'
        assert decision['progress_policy'] == 'fixed_seed_envelope'
        assert entry['next_cost'] < entry['cost']
    continued, provenance = make_fixed_seed_envelope(abs(result.q), [6., 4.],
        seed_q_sha256='2'*64, group_ids=origin['group_ids'], resumed=origin)
    np.testing.assert_array_equal(continued, limits)
    assert provenance['seed_q_sha256'] == origin['seed_q_sha256']
    assert callbacks and any(answer['report']['temporarily_increased_groups'] for answer in callbacks)
