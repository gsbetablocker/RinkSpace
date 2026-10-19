import numpy as np
import pytest
from scipy.sparse import csc_matrix

from calibration.solver_runtime.normal_equation_refinement_candidate import (
    SameEquationLU, SameEquationSchur, compensated_original_residual, original_equation_error,
    refine_same_equation, system_identity)


def ill_scaled_equation():
    rng = np.random.default_rng(401)
    basis = rng.normal(size=(24, 24))
    balanced = basis.T @ basis + np.eye(24)
    scale = np.geomspace(1e-8, 1e8, 24)
    matrix = csc_matrix(balanced * scale[:, None] * scale[None, :])
    original_solution = rng.normal(size=24) / scale
    rhs = matrix @ original_solution
    return matrix, rhs


def test_reversible_equilibration_and_refinement_keep_the_exact_supplied_equation():
    matrix, rhs = ill_scaled_equation()
    identity = system_identity(matrix)
    original_data, original_rhs = matrix.data.tobytes(), rhs.tobytes()
    solver = SameEquationLU(matrix, equilibrate=True)
    raw = solver.solve(rhs)
    seeded_error = raw + .002 / solver.coordinate_scale
    refined, report = refine_same_equation(matrix, rhs, seeded_error, solver.solve)
    assert report['accepted_corrections'] > 0
    assert original_equation_error(matrix, rhs, refined)['maximum_componentwise_backward_error'] < 2e-15
    assert report['history'][-1]['original_diagonal_weighted_residual_2_norm'] < report['history'][0]['original_diagonal_weighted_residual_2_norm']
    assert system_identity(matrix) == solver.identity == identity
    assert matrix.data.tobytes() == original_data and rhs.tobytes() == original_rhs
    assert report['added_ridge_prior_or_source_rows'] == 0
    assert not report['native_pixel_consistency_or_accuracy_qualification_claimed']


def test_compensated_residual_preserves_cancelled_float64_product_sum():
    matrix = csc_matrix([[1e16, 1., -1e16], [0., 1., 0.], [0., 0., 1.]])
    step = np.ones(3)
    rhs = np.array([1., 1., 1.])
    assert (rhs-matrix@step)[0] != 0
    assert np.array_equal(compensated_original_residual(matrix, rhs, step), np.zeros(3))


def test_wrong_or_nonfinite_correction_cannot_be_called_a_numerical_fix():
    matrix, rhs = ill_scaled_equation()
    candidate = np.zeros(len(rhs))
    same, report = refine_same_equation(matrix, rhs, candidate, lambda residual: -residual)
    assert np.array_equal(same, candidate) and report['accepted_corrections'] == 0
    with pytest.raises(ValueError, match='same-equation correction'):
        refine_same_equation(matrix, rhs, candidate, lambda residual: np.full(len(rhs), np.nan))


def test_independent_schur_corrections_retain_original_permutation_and_full_equation():
    matrix, rhs = ill_scaled_equation()
    sparse = SameEquationLU(matrix, equilibrate=True)
    schur = SameEquationSchur(matrix, np.array([17, 1, 12, 7]), equilibrate=True)
    assert schur.identity == sparse.identity == system_identity(matrix)
    candidate = schur.solve(rhs)
    polished, report = refine_same_equation(matrix, rhs, candidate, schur.solve)
    error = original_equation_error(matrix, rhs, polished)
    assert error['maximum_componentwise_backward_error'] < 3e-15
    sparse_polished, _ = refine_same_equation(matrix, rhs, sparse.solve(rhs), sparse.solve)
    assert np.allclose(polished, sparse_polished, atol=1e-7, rtol=2e-14)
    assert report['original_equation_and_rhs_unchanged']
