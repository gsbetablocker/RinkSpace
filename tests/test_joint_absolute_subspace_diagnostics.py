"""Closed-form checks against false causal attribution by numerical rank."""
import numpy as np
from scripts.diagnostics.joint_absolute_subspace_diagnostics_20261011 import analyze_subspaces


def run(r, c, g, n, c2=None, g2=None, n2=None, weights=None):
    return analyze_subspaces(r, c, g, n, c if c2 is None else c2,
        g if g2 is None else g2, n if n2 is None else n2,
        np.ones(len(r)) if weights is None else weights)


def test_closed_form_overlapping_geometry_camera_and_nuisance():
    e = np.eye(4)
    answer = run([3., 4., 5., 6.], e[:, [0]], (e[:, 0] + e[:, 1])[:, None], e[:, [3]])
    a = answer['arrays']
    np.testing.assert_allclose(a['first_joint_remaining'], [0., 0., 5., 0.], atol=2e-13)
    np.testing.assert_allclose(a['first_camera_only_remaining'], [0., 4., 5., 0.], atol=2e-13)
    np.testing.assert_allclose(a['first_geometry_only_remaining'], [-.5, .5, 5., 0.], atol=2e-13)
    assert not answer['summary']['parameter_solution_computed']


def test_tiny_noise_only_column_does_not_erase_agreeing_geometry():
    e = np.eye(4)
    first = np.column_stack((e[:, 1], 1e-10 * e[:, 2]))
    second = np.column_stack((e[:, 1], -1e-10 * e[:, 2]))
    answer = run([3., 4., 5., 6.], e[:, [0]], first, np.empty((4, 0)), g2=second)
    s = answer['summary']['spaces']['geometry']
    assert s['conservative_rank_above_global_operator_change_first'] == 0
    assert s['empirical_combination_agreement_rank_first'] == 1
    np.testing.assert_allclose(answer['arrays']['first_joint_remaining'], [0., 0., 5., 6.], atol=2e-13)
    assert answer['summary']['outcomes']['first']['machine_only_joint_remaining_energy_not_trusted'] < 36.000001


def test_fixed_nominal_metric_is_applied_once():
    e = np.eye(4)
    answer = run([3., 4., 5., 6.], e[:, [0]], e[:, [1]], e[:, [3]], weights=[1., 4., .25, 16.])
    np.testing.assert_allclose(answer['arrays']['weighted_residual'], [3., 8., 2.5, 24.])
    np.testing.assert_allclose(answer['arrays']['first_joint_remaining'], [0., 0., 2.5, 0.], atol=2e-13)
    assert abs(answer['summary']['outcomes']['first']['joint_remaining_energy'] - 6.25) < 1e-12


def test_column_units_and_row_order_do_not_change_closed_form_result():
    e = np.eye(5)
    r = np.arange(1., 6.)
    c = e[:, [0, 1]]
    g = np.column_stack((e[:, 1] + e[:, 2], e[:, 0] - e[:, 2]))
    n = e[:, [3]]
    original = run(r, c, g, n)['arrays']['first_joint_remaining']
    permutation = np.array([4, 1, 3, 0, 2])
    altered = run(r[permutation], (c * [1e4, .003])[permutation],
        (g @ np.array([[2., 3.], [-1., 4.]]))[permutation],
        (n * 17.)[permutation])['arrays']['first_joint_remaining']
    np.testing.assert_allclose(altered[np.argsort(permutation)], original, atol=2e-12)


def test_no_columns_or_zero_weight_does_not_invent_information():
    empty = np.empty((3, 0))
    answer = run([2., 3., 4.], empty, empty, empty, weights=[1., 0., 4.])
    np.testing.assert_allclose(answer['arrays']['first_joint_remaining'], [2., 0., 8.])
    assert answer['summary']['spaces']['joint']['empirical_combination_agreement_rank_first'] == 0
