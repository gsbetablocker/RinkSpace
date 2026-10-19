from copy import deepcopy
import numpy as np
import pytest


@pytest.fixture
def vertical_edge(monkeypatch):
    monkeypatch.setenv('RINKSPACE_PROJECTION_BACKEND', 'numpy')
    from calibration.solver_runtime import install
    install()
    from structured_programme_source_candidate.native_board_edge_source import NativeBoardEdgeFactor
    factor = NativeBoardEdgeFactor.__new__(NativeBoardEdgeFactor)
    factor.parameter_count = 1
    factor.rows = [dict(id='actual_print_transition', actual_pts=100,
        curve=dict(kind='advertisement_boundary'), uv=np.array([[865., 540.]]),
        sigma=np.array([8.]), normal=np.array([[1., 0.]]),
        normal_interval=np.array([[-10., 10.]]), curve_observation_kind='normal_interval')]
    factor._world = lambda q, curve: np.array([[-1., -1., 10.], [-1., 1., 10.]])
    def cameras(q):
        camera = np.zeros(25)
        camera[6:8] = np.log(1000.)
        camera[8:10] = [960. + 50.*q[0], 540.]
        return {100: camera}
    factor._cameras = cameras
    return factor


@pytest.mark.parametrize('mode', ['measured_transition_midpoint', 'measured_transition_location'])
def test_measured_edge_keeps_position_signal_inside_blur_footprint(vertical_edge, mode):
    factor = vertical_edge
    before = deepcopy(factor.rows[0])
    q = np.zeros(1)
    # At q=0 the finite vertical edge is x=860, five pixels from the measured
    # x=865. A weak +/-10 interval legitimately has no location preference.
    np.testing.assert_array_equal(factor._residual(q, factor.rows), [0., 0.])
    factor.rows[0]['boundary_location_observation'] = mode
    r = factor._residual(q, factor.rows)
    assert np.linalg.norm(r) == pytest.approx(5./8.)
    # Moving the real projected edge by five pixels reaches the observed edge.
    np.testing.assert_allclose(factor._residual(np.array([.1]), factor.rows), 0., atol=1e-12)
    for name in ('uv', 'normal', 'normal_interval', 'sigma'):
        np.testing.assert_array_equal(before[name], factor.rows[0][name])


@pytest.mark.parametrize('mode', ['measured_transition_midpoint', 'measured_transition_location'])
def test_measured_edge_derivative_does_not_flatten_at_blur_support(vertical_edge, mode):
    factor = vertical_edge
    factor.rows[0]['boundary_location_observation'] = mode
    q = np.zeros(1)
    associations = factor._nominal_associations(q, factor._cameras(q))
    def actual(value):
        return factor._residual(value, factor.rows)
    def derivative(value):
        return factor._derivative_residual(value, cameras=factor._cameras(value),
                                           associations=associations)
    h = 1e-5
    actual_j = (actual(q+h)-actual(q-h))/(2*h)
    replay_j = (derivative(q+h)-derivative(q-h))/(2*h)
    np.testing.assert_allclose(actual_j, replay_j, atol=1e-9, rtol=0.)
    assert np.linalg.norm(actual_j) == pytest.approx(50./8.)
    # Crossing the old interval at x=875 does not introduce a branch in the
    # measured-location objective. The same finite segment remains interior.
    np.testing.assert_array_equal(factor._support_branches(q)[0],
                                  factor._support_branches(np.array([.5]))[0])


def test_unknown_observation_mode_fails_closed(vertical_edge):
    vertical_edge.rows[0]['boundary_location_observation'] = 'mistyped_center'
    with pytest.raises(ValueError, match='Unknown native board'):
        vertical_edge._residual(np.zeros(1), vertical_edge.rows)
