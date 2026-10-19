"""Real public solver regressions for rejected fresh-source LM directions."""
from pathlib import Path
import inspect
import json
from types import SimpleNamespace

import numpy as np
from scipy.sparse import diags

from calibration.solver_runtime import install
install()
import z190_joint_source_solver as solver
from z190_streaming_source_normal import SourceRowBlock
from calibration.program_tracking.first_shot_prediction_refresh import compile_quality_solver, install_prediction_refresh
from calibration.solver_runtime.schur_rejection_retry import rejected_direction_damping


def implementation():
    return compile_quality_solver(solver.solve_joint_source, retry_rejected_schur=True)


def first_accepted(q, event):
    if event['checkpoint_kind'] == 'accepted':
        return dict(stop_branch=True, reason='Capture first verified numerical transition')


def nonlinear_blocks(calls):
    def blocks(q, derivative):
        if derivative:
            calls.append(q.copy())
        yield SourceRowBlock(np.array([q[0]**2 - 1., q[1]]),
            diags([2*q[0], 1.], format='csr') if derivative else None,
            np.array([False, True]), np.ones(2),
            dict(family='degenerate_nonlinear_start', original_start=0, original_stop=2))
    return blocks


def run_nonlinear(function, directory, calls, **extra):
    def unused(*args):
        raise AssertionError('Use the original source blocks')
    options = dict(source_block_factory=nonlinear_blocks(calls), cache_native_source_replay=True,
        global_schur_count=1, schur_backtracking=True,
        adaptive_source_backtracking=True, initial_damping=1e-12,
        checkpoint_identity={'source': 'exact unchanged toy'},
        checkpoint_arrays={'native_pts': np.array([256256], np.int64)},
        checkpoint_callback=first_accepted, verbose=False)
    options.update(extra)
    return function(unused, np.array([5e-5, 0.]), np.array([False, True]), np.ones(2),
        directory, lambda q: True, **options)


def test_rejected_exact_source_retries_damping_without_new_J_or_parameter_change(tmp_path):
    old_calls, calls = [], []
    original_source = inspect.getsource(solver.solve_joint_source)
    stopped = run_nonlinear(solver.solve_joint_source, tmp_path/'old', old_calls)
    assert stopped.status == 'no_accepted_trial'
    assert stopped.history[0]['trial'] == 12
    profile_updates = []
    def profiles(q):
        profile_updates.append(q.copy())
    profiles.stateless = True
    result = run_nonlinear(implementation(), tmp_path/'fixed', calls, update_profiles=profiles)
    assert result.status == 'stopped_by_source_review'
    assert len(calls) == len(old_calls) == len(profile_updates) == 1
    record = result.history[0]
    assert record['accepted'] and record['next_cost'] < record['cost']
    rounds = record['rejected_schur_direction_damping']['rounds']
    assert rounds and all(row['next_damping'] == row['previous_damping']*10 for row in rounds)
    assert record['trial'] > 12
    schedule = record['adaptive_source_backtracking']
    assert schedule['per_direction_trial_schedule_unchanged']
    assert schedule['total_trials_may_increase']
    assert schedule['real_source_cost_and_physical_guards_unchanged']
    assert 'real_source_cost_guard_and_trial_count_unchanged' not in schedule
    assert np.array_equal(calls[0], np.array([5e-5, 0.]))
    assert result.q[1] == 0.
    assert inspect.signature(implementation()) == inspect.signature(solver.solve_joint_source)
    assert inspect.getsource(solver.solve_joint_source) == original_source


def test_control_stop_and_resume_keep_current_lambda_aux_and_canonical_state(tmp_path):
    calls, attempted = [], []
    output = tmp_path/'resume'
    initial = np.array([5e-5, 0.])
    def guard(q):
        if not np.array_equal(q, initial):
            attempted.append(q.copy())
            if len(attempted) == 12:
                (output/'control.json').write_text(json.dumps({'stop_branch': True, 'reason': 'Test user control'}))
        return True
    function = implementation()
    def controlled(evaluate, q, quadratic, scales, folder, unused_guard, **kwargs):
        return function(evaluate, q, quadratic, scales, folder, guard, **kwargs)
    stopped = run_nonlinear(controlled, output, calls, checkpoint_callback=None)
    assert stopped.status == 'stopped_by_control'
    assert stopped.damping == 1e-11
    assert np.array_equal(stopped.q, initial)
    with np.load(stopped.checkpoint, allow_pickle=False) as saved:
        assert float(saved['damping']) == stopped.damping
        assert float(saved['next_line_search_fraction']) == 1.
        assert np.array_equal(saved['q'], initial)
        assert np.array_equal(saved['aux__native_pts'], np.array([256256]))
    (output/'control.json').unlink()
    resumed = run_nonlinear(function, output, calls, resume=True, initial_damping=.5,
        initial_line_search_fraction=.25)
    assert resumed.status == 'stopped_by_source_review'
    assert resumed.history[-1]['next_cost'] < resumed.history[-1]['cost']
    rounds = resumed.history[-1]['rejected_schur_direction_damping']['rounds']
    assert rounds[0]['previous_damping'] == stopped.damping
    with np.load(resumed.checkpoint, allow_pickle=False) as saved:
        assert np.array_equal(saved['aux__native_pts'], np.array([256256]))
        assert float(saved['damping']) == resumed.damping


def test_real_guard_runs_before_true_cost_on_each_new_damped_direction(tmp_path):
    derivatives, cost_states = [], []
    limit = 1e-5
    def evaluate(q, derivative):
        if derivative:
            derivatives.append(q.copy())
        else:
            assert abs(q[0]) <= limit
            cost_states.append(q.copy())
        r = np.array([q[0]-1., q[1]])
        return (r, diags(np.ones(2), format='csr')) if derivative else r
    def guard(q):
        return abs(q[0]) <= limit
    result = implementation()(evaluate, np.zeros(2), np.ones(2, bool), np.ones(2),
        tmp_path, guard, global_schur_count=1, schur_backtracking=True,
        initial_damping=1e-12, checkpoint_callback=first_accepted, verbose=False)
    assert result.status == 'stopped_by_source_review'
    assert len(derivatives) == 1
    assert 0 < result.q[0] <= limit and cost_states
    assert result.history[0]['next_cost'] < result.history[0]['cost']
    assert any(not row['accepted'] and row['reason'] == 'Physical joint parameter domain'
        for row in result.history[0]['global_schur_trial_records'])


def test_guard_impossibility_stops_at_existing_precision_without_claiming_convergence(tmp_path):
    def evaluate(q, derivative):
        r = np.array([q[0]-1., q[1]])
        return (r, diags(np.ones(2), format='csr')) if derivative else r
    result = implementation()(evaluate, np.zeros(2), np.ones(2, bool), np.ones(2),
        tmp_path, lambda q: np.array_equal(q, np.zeros(2)),
        global_schur_count=1, schur_backtracking=True, initial_damping=1e-12, verbose=False)
    assert result.status == 'no_accepted_trial'
    assert np.array_equal(result.q, np.zeros(2))
    record = result.history[0]
    assert not record['accepted']
    assert record['failure_reason'] == 'rejected_source_model_below_original_acceptance_precision'
    last = record['rejected_schur_direction_damping']['rounds'][-1]
    assert last['retry'] is False
    assert last['search_outcome'] == 'numerical_search_exhausted'
    assert last['convergence_established'] is False


def test_rejection_precision_and_finite_lambda_are_numerical_exit_conditions():
    common = dict(before=5., q=np.zeros(1), normal=diags(np.ones(1)), gradient=np.array([-1.]))
    tiny = rejected_direction_damping(damping=1., step=np.array([1e-12]), **common)
    assert tiny['original_source_acceptance_precision'] == 5e-10
    assert not tiny['retry']
    exhausted = rejected_direction_damping(damping=np.finfo(float).max,
        step=np.array([.1]), **common)
    assert not exhausted['retry'] and exhausted['reason'] == 'finite_numerical_LM_damping_exhausted'


def test_nonpositive_or_nonfinite_saved_damping_cannot_repeat_the_same_direction():
    for damping in (0., -1., np.nan, np.inf, -np.inf):
        decision = rejected_direction_damping(damping=damping, before=1.,
            q=np.zeros(1), step=np.array([.1]), normal=diags(np.ones(1)), gradient=-np.ones(1))
        assert decision['retry'] is False
        assert decision['reason'] == 'positive_finite_LM_damping_required'
        assert decision['search_outcome'] == 'numerical_search_exhausted'
        assert decision['convergence_established'] is False


def test_current_runtime_installs_rejection_retry_and_keeps_saved_numerical_seed():
    class Owner:
        source_producers = []
        recovery_references = []
        source_jacobian_reuse_policy = {}
        def checkpoint_identity(self):
            return {}
        def solver_kwargs(self):
            return dict(reuse_source_jacobian=True, adaptive_source_backtracking=True,
                initial_damping=1.96608e-8, initial_line_search_fraction=.125,
                checkpoint_identity=self.checkpoint_identity())
    owner = Owner()
    driver = SimpleNamespace(solve_joint_source=solver.solve_joint_source, solver_runtime=solver)
    receipt = install_prediction_refresh(owner, driver=driver)
    assert receipt['rejected_exact_Schur_direction_retries_existing_LM']
    assert '_rejected_direction_damping' in driver.solve_joint_source.__globals__
    args = owner.solver_kwargs()
    assert args['initial_damping'] == 1.96608e-8
    assert args['initial_line_search_fraction'] == .125
    assert any(ref['path'].endswith('schur_rejection_retry.py') for ref in owner.source_producers)
