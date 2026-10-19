"""Bounded motion moments reach full and independent short-scale camera QPs."""
import numpy as np
import pytest
from scipy.sparse import csc_matrix, csr_matrix

from calibration.solver_runtime import install
from calibration.solver_runtime import optional_soc_runtime as runtime
from calibration.solver_runtime.bounded_motion_schur import BoundedMotionSchurModel
from calibration.solver_runtime.original_halfspace_primal import original_full_qp_kkt_verified
from calibration.solver_runtime.trial_fraction_source_domain import attach_trial_fraction_domain


@pytest.fixture
def installed_joint(monkeypatch):
    install()
    import z190_global_schur_linear_step as qp
    import z190_joint_source_solver as joint
    monkeypatch.setattr(qp, '_linearly_constrained_shared_step', qp._linearly_constrained_shared_step)
    monkeypatch.setattr(joint, '_sequential_measured_shared_source_curvature',
                        joint._sequential_measured_shared_source_curvature)
    runtime.install_optional_SOC_numerical_failure_route()
    yield joint
    runtime._retained_soc_face_model.set(None)


def bounded_example():
    # The original free-depth model loses the x-direction of this motion row.
    # Its finite displacement restores 3*(x-.2) after the original lambda=0
    # endpoint, so the profiled row contributes 4.5*max(x-.2, 0)**2.
    model = BoundedMotionSchurModel([dict(
        projection_factors=csr_matrix([[1., 0., 0.]]),
        cross_moments=csr_matrix((1, 3)), lambda0=np.array([.2]),
        inner_gradient=np.zeros(1), inner_GN=np.array([9.]),
        outer_GN=np.array([9.]), outer_gradient=np.zeros(1))], 3)
    return model


def source_domain():
    from structured_programme_source_candidate import source_step_curvature_tightening as source
    matrix = np.array([[1.], [-1.]])
    original = np.array([1., -10.])
    response = dict(row_start=0, row_stop=2, nominal_carrier=np.zeros(1),
                    retained_face_mask=np.ones(2, bool), dynamic_group_limits=np.array([.2]),
                    dynamic_row_group_index=np.array([0, 0]),
                    dynamic_guard_comparison_tolerance_px=1e-8)
    domain = attach_trial_fraction_domain(
        dict(matrix=matrix, lower=original-.2, curvature_responses={'native': response}),
        original_constraint_lower=original, row_group_index=np.array([0, 0]),
        previous_group_violations=np.ones(1), common_fraction=.2)

    def actual(q, step, selected, current_domain):
        x = float(step[0])
        carrier = np.array([x-.5*x*x])
        return source.measured_outward_step_tightening(
            current_domain['matrix'], current_domain['lower'], step[selected],
            current_domain['curvature_responses'], {'native': carrier},
            actual_group_violations={'native': np.array([max(1.-carrier[0], 0.)])})

    return domain, actual


def test_full_and_short_SOC_return_KKT_for_their_actual_bounded_active_normal(installed_joint, monkeypatch):
    joint = installed_joint
    domain, actual = source_domain()
    model = bounded_example()
    base_h = csc_matrix(np.eye(3))
    base_g = np.array([-2., 0., 0.])
    diagonal = np.ones(3)
    calls = []
    original_solve = joint.schur_trial_linear_step

    def capture(h, g, supplied_diagonal, **options):
        result = original_solve(h, g, supplied_diagonal, **options)
        calls.append((h.toarray().copy(), np.asarray(g).copy(), result))
        return result

    def bounded_solve(h, g, supplied_diagonal, **options):
        return model.solve(h, g, supplied_diagonal, capture, **options)

    options = dict(global_count=1, damping=.001, schur_view_lm_ratio=0.,
                   global_step_lower_bounds=np.array([-1.]),
                   global_step_upper_bounds=np.array([1.]),
                   global_step_constraint_matrix=domain['matrix'],
                   global_step_constraint_lower=domain['lower'].copy())
    initial = bounded_solve(base_h, base_g, diagonal, **options)
    np.testing.assert_allclose(initial.step[0], .8, atol=1e-9)
    assert initial.report['bounded_motion_nuisance']['active_lower_bound_points'] == 1
    assert runtime._previous_full_KKT_verified(initial)
    full = joint._sequential_measured_shared_source_curvature(
        np.zeros(3), initial, np.array([0]), domain, options,
        base_h, base_g, diagonal, actual, bounded_solve)
    assert full is initial
    assert joint._needs_trial_fraction_SOC(full, domain)
    assert full.report['sequential_measured_shared_source_curvature_tightening']['stop_reason'] == (
        'tightened_QP_failed_keep_last_verified_direction')
    full_final = next(entry for entry in calls if entry[2] is full)
    np.testing.assert_array_equal(full_final[0], np.diag([10., 1., 1.]))
    np.testing.assert_array_equal(full_final[1], [-3.8, 0., 0.])

    short = joint._trial_fraction_source_step(
        np.zeros(3), .5, np.array([0]), domain, options,
        base_h, base_g, diagonal, actual, bounded_solve)
    assert short.step[0] > .5
    assert 1.-short.step[0]+.5*short.step[0]**2 <= .6+1e-8
    assert short.report['bounded_motion_nuisance']['active_lower_bound_points'] == 1
    assert original_full_qp_kkt_verified(short.report['shared_linear_domain_step'], short.step)
    short_final = next(entry for entry in calls if entry[2] is short)
    np.testing.assert_array_equal(short_final[0], np.diag([10., 1., 1.]))
    np.testing.assert_array_equal(short_final[1], [-2.8, 0., 0.])
    assert short.report['bounded_motion_nuisance']['source_model_reduction'] == pytest.approx(
        model.model_reduction(base_h, .5*base_g, short.step))
    np.testing.assert_array_equal(base_h.toarray(), np.eye(3))
    np.testing.assert_array_equal(base_g, [-2., 0., 0.])


def test_joint_short_SOC_prediction_uses_original_bounded_model_not_free_depth_quadratic(
        installed_joint, monkeypatch, tmp_path):
    joint = installed_joint
    domain, actual = source_domain()
    model = bounded_example()
    target = np.array([2., 0., 0.])
    accepted = []
    prepared = []
    calls = []
    original_solve = joint.schur_trial_linear_step

    def capture(h, g, diagonal, **options):
        result = original_solve(h, g, diagonal, **options)
        calls.append((h.toarray().copy(), np.asarray(g).copy(), result))
        return result

    monkeypatch.setattr(joint, 'schur_trial_linear_step', capture)

    def evaluate(q, jacobian):
        residual = np.r_[q-target, 3.*max(float(q[0])-.2, 0.)]
        if not jacobian:
            return residual
        matrix = np.vstack((np.eye(3), [3. if q[0] > .2 else 0., 0., 0.]))
        return residual, csr_matrix(matrix)

    def factory(q):
        prepared.append(q.copy())
        return model

    def guard(q):
        if not -1. <= q[0] <= 1.:
            raise ValueError('Original physical interval')

    def accept(parent, trial, *, before_cost, after_cost, default_source_cost_accepts, trial_context):
        alpha = trial_context['alpha']
        x = float(trial[0])
        decision = 1.-x+.5*x*x <= 1.-alpha*.8+1e-8
        accepted.append(dict(alpha=alpha, decision=decision, step=trial_context['actual_step'].copy(),
                             prediction=trial_context['predicted_source_reduction']))
        return dict(accept=bool(decision), mode='truth_restoration', report={},
                    rejection_reason='Original nonlinear support')

    result = joint.solve_joint_source(
        evaluate, np.zeros(3), np.ones(4, bool), np.ones(4), tmp_path, guard,
        global_schur_count=1, schur_backtracking=True, initial_damping=.001,
        shared_parameter_bounds={0: (-1., 1.)}, shared_linear_constraints=lambda q, selected: domain,
        shared_step_curvature_tightening=actual, source_trial_acceptance=accept,
        bounded_motion_model_factory=factory,
        checkpoint_callback=lambda q, event: ({'stop_branch': True, 'reason': 'Synthetic verification complete'}
            if event['checkpoint_kind'] == 'accepted' else None), verbose=False)
    assert len(prepared) == 1 and np.array_equal(prepared[0], np.zeros(3))
    assert result.history[0]['accepted']
    assert [(entry['alpha'], entry['decision']) for entry in accepted] == [(1., False), (.5, True)]
    np.testing.assert_array_equal(result.q, accepted[-1]['step'])
    assert result.q[0] > .5
    expected = model.model_reduction(csc_matrix(np.eye(3)), -target, result.q)
    old_free_depth_prediction = target@result.q-.5*result.q@result.q
    assert accepted[-1]['prediction'] == pytest.approx(expected)
    assert old_free_depth_prediction-expected > .5
    assert result.history[0]['next_cost'] == pytest.approx(2.-expected)
    corrected_solutions = [entry for entry in calls if entry[0][0, 0] == 10.]
    assert any(np.array_equal(entry[1], [-3.8, 0., 0.]) for entry in corrected_solutions)
    assert any(np.array_equal(entry[1], [-2.8, 0., 0.]) for entry in corrected_solutions)


def test_bounded_streaming_integration_forces_same_state_J_and_eliminates_fixed_moments(
        installed_joint, tmp_path):
    joint = installed_joint
    from z190_streaming_source_normal import SourceRowBlock
    initial = np.array([0., 0., .125])
    target = np.array([1.2, .7, .125])
    jac = csr_matrix(np.eye(3))
    roles = np.ones(3, bool)
    scales = np.ones(3)
    derivative_states = []
    factories = []
    accepted_states = []

    def source_blocks(q, derivative):
        if derivative:
            derivative_states.append(q.copy())
        yield SourceRowBlock(q-target, jac if derivative else None, roles, scales,
                             dict(family='programme:synthetic', original_start=0, original_stop=3))

    def factory(q):
        model = BoundedMotionSchurModel([dict(
            projection_factors=csr_matrix([[1., 0., 10000.]]),
            cross_moments=csr_matrix([[0., 0., 1.]]), lambda0=np.array([10.]),
            inner_gradient=np.zeros(1), inner_GN=np.ones(1),
            outer_GN=np.ones(1), outer_gradient=np.zeros(1))], 3)
        factories.append((q.copy(), model))
        return model

    def checkpoint(q, event):
        if event['checkpoint_kind'] == 'accepted':
            accepted_states.append(q.copy())
            if len(accepted_states) == 2:
                return dict(stop_branch=True, reason='Two synthetic same-state models verified')

    def unused(*args):
        raise AssertionError('Original streaming source must supply the values and J')

    result = joint.solve_joint_source(
        unused, initial, roles, scales, tmp_path, lambda q: True,
        source_block_factory=source_blocks, cache_native_source_replay=True,
        reuse_source_jacobian=True, bounded_motion_model_factory=factory,
        fixed_parameter_indices=[2], global_schur_count=1, schur_backtracking=True,
        checkpoint_callback=checkpoint, verbose=False)
    assert len(result.history) == len(factories) == len(accepted_states) == 2
    assert len(derivative_states) == 2  # Native replay reuses RAM; no second source J call.
    np.testing.assert_array_equal(derivative_states[0], initial)
    np.testing.assert_array_equal(derivative_states[1], accepted_states[0])
    for (factory_q, model), derivative_q in zip(factories, derivative_states):
        np.testing.assert_array_equal(factory_q, derivative_q)
        assert model.chunks[0]['projection_factors'][0, 2] == 0.
        assert model.chunks[0]['cross_moments'][0, 2] == 0.
    assert result.q[2] == initial[2]
    for entry in result.history:
        assert entry['source_jacobian_reuse']['derivative_model'] == 'fresh_source_J'


@pytest.mark.parametrize('options', [dict(global_schur_count=None),
                                  dict(global_schur_count=1, conditional_view_backtracking=True)])
def test_unsupported_bounded_solver_modes_fail_before_any_source_call(installed_joint, tmp_path, options):
    def unused(*args):
        raise AssertionError('Unsupported bounded mode must fail before evaluating original source')

    with pytest.raises(ValueError, match='Bounded motion nuisances require the complete Schur QP'):
        installed_joint.solve_joint_source(
            unused, np.zeros(3), np.ones(3, bool), np.ones(3), tmp_path, unused,
            bounded_motion_model_factory=bounded_example, verbose=False, **options)
