"""One real full-LP fixture, unchanged target and original-IPM fallback."""
from types import SimpleNamespace

import numpy as np
from scipy.optimize import linprog

from calibration.solver_runtime import full_priority_LP_backend_runtime as runtime


def test_full_backend_target_and_uncertified_original_fallback(tmp_path):
    from calibration.solver_runtime import install
    install()
    from structured_programme_source_candidate import feasibility_priority_restoration as source
    original = source.contract_original_support_bounds
    lower, upper = np.array([-1., -1.]), np.array([1., 1.])
    rows = np.array([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]])
    rhs, ids, violations = np.array([.75, .75, .5, .5]), np.array([0, 0, 1, 1]), np.ones(2)
    # The original LP deliberately initializes HiGHS first: the candidate
    # must leave that scheduler and all native parallel settings unchanged.
    expected = original(lower, upper, rows, rhs, ids, violations)
    runtime.install_full_priority_LP_backend_runtime(report_directory=tmp_path)
    try:
        actual = source.contract_original_support_bounds(lower, upper, rows, rhs, ids, violations)
        trace = actual['report']['complete_original_LP_backend']
        assert trace['route'] == 'full_original_default_dual_simplex'
        assert trace['complete_original_problem_KKT_certified']
        assert trace['full_original_rows'] == 8 and trace['original_columns'] == 3
        assert not trace['explicit_parallel_thread_strategy_or_concurrency_options_added']
        assert trace['global_scheduler_not_reset']
        assert not trace['original_IPM_fallback_called']
        assert actual['report']['phase_I_backend'].startswith('scipy HiGHS default serial dual simplex')
        assert abs(actual['report']['phase_I_minimum_common_original_violation_fraction']-.75) <= 1e-9
        assert actual['common_fraction'] == .5*(1.+actual['report']['phase_I_minimum_common_original_violation_fraction'])
        assert abs(actual['common_fraction']-expected['common_fraction']) <= 1e-9
        assert np.array_equal(actual['matrix'], expected['matrix'])
        assert np.max(np.abs(actual['lower']-expected['lower'])) <= 1e-9
    finally:
        source.contract_original_support_bounds = original

    c = np.array([0., 0., 1.])
    A = -np.c_[rows, np.ones(4)]
    kwargs = dict(A_ub=A, b_ub=-rhs, bounds=[(None, None), (None, None), (0., 1.)],
        method='highs-ipm', options=dict(primal_feasibility_tolerance=1e-9, dual_feasibility_tolerance=1e-9))
    calls = []
    def without_backend_dual(objective, **values):
        calls.append(values)
        result = linprog(objective, **values)
        if values['method'] == 'highs-ds':
            result.ineqlin = SimpleNamespace(residual=result.ineqlin.residual)
        return result
    result = runtime.full_original_dual_simplex_linprog(without_backend_dual, c, **kwargs)
    assert result.success
    assert [call['method'] for call in calls] == ['highs-ds', 'highs-ipm']
    assert calls[1] == kwargs
    for call in calls:
        assert call['A_ub'] is A and call['b_ub'] is kwargs['b_ub']
        assert call['bounds'] is kwargs['bounds']
    assert set(calls[0]['options']) == set(kwargs['options']) | {'disp'}
    assert calls[0]['options']['disp'] is True
    fallback = runtime._LATEST.get()
    assert fallback['route'] == 'original_full_HiGHS_IPM'
    assert fallback['original_IPM_fallback_called']
    assert not fallback['complete_original_problem_KKT_certified']
    assert fallback['fallback_reason'] == 'default_dual_simplex_full_backend_original_KKT_not_certified'
    assert runtime._phaseI_backend_label() == 'scipy HiGHS IPM'
