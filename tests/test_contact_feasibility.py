"""Contact feasibility reference checks; no project fit."""
import numpy as np
from scipy.optimize import least_squares
from calibration.program_tracking.body_tracking.contact_feasibility import ContactAL, physical_feasibility, contact_inequalities

def test_al_restores_constraint_despite_conflicting_data():
    al = ContactAL.zeros(1, penalty=10.0)
    x = np.array([-1.0])
    history = []
    for _ in range(8):

        def fun(q):
            r, _ = al.residual_and_jacobian(q)
            return np.r_[q + 1.0, r]

        def jac(q):
            _, j = al.residual_and_jacobian(q, np.eye(1))
            return np.vstack([np.eye(1), j])
        fit = least_squares(fun, x, jac=jac, gtol=1e-12, ftol=1e-12, xtol=1e-12)
        x = fit.x
        history.append(float(x[0]))
        al.update_after_inner_solve(x)
    assert history[0] < -0.09 and abs(history[-1]) < 1e-07

def test_feasibility_gate_catches_one_bad_exposure():
    margins = np.full(1001, 0.02)
    margins[623] = -0.15
    gate = physical_feasibility(margins, tolerance_m=0.0005)
    assert not gate['feasible'] and gate['rows_over_tolerance'] == 1 and (gate['worst_row'] == 623)

def test_weighted_al_directional_derivative():
    check = ContactAL(np.array([0.3, 0.0]), 2.0)
    g = np.array([-0.1, 0.2])
    J = np.array([[2.0, -1.0], [0.5, 3.0]])
    weights = np.array([0.02, 0.04])
    _, analytic = check.residual_and_jacobian(g, J, row_weights=weights)
    direction = np.array([0.4, -0.2])
    h = 1e-06
    plus = check.residual_and_jacobian(g + h * (J @ direction), row_weights=weights)[0]
    minus = check.residual_and_jacobian(g - h * (J @ direction), row_weights=weights)[0]
    error = float(np.max(abs((plus - minus) / (2 * h) - analytic @ direction)))
    assert error < 1e-08

def test_support_lower_and_upper_signs():
    adapted = contact_inequalities(dict(rows=[dict(label='L_nonpenetration', constraint_value_m=-0.01, constraint_lower_m=0.0, constraint_upper_m=None), dict(label='supported_runner', constraint_value_m=0.02, constraint_lower_m=-0.002, constraint_upper_m=0.003)]), normalization_m=0.001, value_jacobian=np.array([[1.0], [2.0]]))
    np.testing.assert_allclose(adapted['margin_m'], [-0.01, 0.022, -0.017])
    np.testing.assert_allclose(adapted['jacobian'].ravel(), [1000.0, 2000.0, -2000.0])

def test_al_integrated_cost_invariant_to_frame_rate():
    costs = []
    for rate in (50, 100):
        w = np.full(rate + 1, 1.0 / rate)
        w[[0, -1]] *= 0.5
        rows, _ = ContactAL.zeros(rate + 1, 2.0).residual_and_jacobian(np.full(rate + 1, -0.25), row_weights=w)
        costs.append(float(rows @ rows))
    np.testing.assert_allclose(costs, [0.125, 0.125], rtol=1e-12)
