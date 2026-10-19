import numpy as np
import pytest


@pytest.fixture
def factor(monkeypatch):
    monkeypatch.setenv('RINKSPACE_PROJECTION_BACKEND', 'numpy')
    from calibration.solver_runtime import install
    install()
    from structured_programme_source_candidate.native_board_edge_source import NativeBoardEdgeFactor
    value = NativeBoardEdgeFactor.__new__(NativeBoardEdgeFactor)
    value.parameter_count = 1
    world = np.array([[-1., 0., 10.], [0., 0., 10.], [1., .5, 10.]])
    value.rows = [dict(id='finite_computational_knot', actual_pts=100,
        curve=dict(kind='advertisement_boundary'), uv=np.array([[960.00006, 540.]]),
        sigma=np.array([1.]), normal=np.array([[0., 1.]]),
        normal_interval=np.array([[-2., 2.]]), curve_observation_kind='normal_interval')]
    value._world = lambda q, curve: world
    def cameras(q):
        camera = np.zeros(25)
        camera[6:8] = np.log(1000.)
        camera[8:10] = [960. + 5000.*q[0], 540.]
        return {100: camera}
    value._cameras = cameras
    return value


def test_actual_values_stay_dynamic_but_coordinate_J_uses_nominal_segment(factor):
    q = np.zeros(1)
    cameras = factor._cameras(q)
    associations = factor._nominal_associations(q, cameras)
    raw = factor._residual(q, factor.rows, apply_optical_support=False)
    replay = factor._derivative_residual(q, cameras=cameras, associations=associations,
                                         apply_optical_support=False)
    np.testing.assert_array_equal(raw, replay)
    def derivative(evaluate, h):
        return (evaluate(q + h) - evaluate(q - h)) / (2*h)
    actual = lambda p: factor._residual(p, factor.rows, apply_optical_support=False)
    locked = lambda p: factor._derivative_residual(p, cameras=factor._cameras(p),
        associations=associations, apply_optical_support=False)
    large_dynamic = derivative(actual, 1e-6)
    nominal_J = derivative(locked, 1e-6)
    small_dynamic = derivative(actual, 1e-9)
    assert np.max(abs(large_dynamic - nominal_J)) > 100.
    np.testing.assert_allclose(nominal_J, small_dynamic, rtol=0., atol=1e-4)
    np.testing.assert_array_equal(factor._residual(q, factor.rows),
        factor._derivative_residual(q, cameras=cameras, associations=associations))


def test_qualification_detects_knot_crossing_inside_same_normal_interval(factor):
    q = np.zeros(1)
    signature, _, _, _ = factor._support_branches(q)
    plus, _, _, _ = factor._support_branches(q + 1e-6)
    minus, _, _, _ = factor._support_branches(q - 1e-6)
    assert signature[0] % 3 == plus[0] % 3 == minus[0] % 3 == 1
    assert signature[0] != plus[0] or signature[0] != minus[0]
    h, receipt = factor.local_support_directional_step(q, np.ones(1), 1e-6)
    assert receipt['same_support_branch_step_found'] and h < 1e-6
    assert receipt['trials'][0]['changed_support_points'] == 1
    np.testing.assert_array_equal(factor._support_branches(q + h)[0], signature)
    np.testing.assert_array_equal(factor._support_branches(q - h)[0], signature)


def test_special_extent_and_point_operators_are_never_replaced(factor):
    for kind in ('horizontal_extent', 'clipped_horizontal_extent', 'axis_scanline_branch'):
        assert not factor._ordinary_finite_curve(dict(curve_observation_kind=kind))
    assert not factor._ordinary_finite_curve(dict(point_correspondence=True))
    assert not factor._ordinary_finite_curve(dict(must_preserve_upper_branch=True))
