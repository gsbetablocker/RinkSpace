"""Actual source-root offset, sparse derivative layout and time integration."""
from types import SimpleNamespace

import numpy as np
from scipy.interpolate import BSpline

from calibration.program_tracking.body_tracking.entry_motion_factors import EntryMotionFactors


def test_actual_entry_root_prior_has_exact_sparse_jacobian_and_nonuniform_time_measure():
    times = np.array([0., .2, .4, .75, 1.4, 1.8, 2.])
    pts = np.rint((80. + times) * 12800).astype(np.int64)
    # Deliberately permute columns: active coordinates are not column numbers.
    active = np.array([7, 2, 1, 3, 0])
    basis = BSpline(np.r_[[0.]*4, [2.]*4], np.eye(4), 3)
    state = SimpleNamespace(m=4, d=5, size=20, active=active,
                            basis=lambda t, derivative=0: basis(t, derivative))
    def root(t):
        t = np.asarray(t)
        return np.c_[5.+t*t, 6.+2.*t*t, 1.+3.*t*t]
    reference = SimpleNamespace(times=times, p=dict(pts=pts,
        camera_centre=np.tile([1., 2., 3.], (len(times), 1)), metres_per_scene_unit=np.array(2.)),
        at=lambda t: dict(P=root(t)),
        root_derivative=lambda t, order: np.tile([2., 4., 6.], (len(t), 1)))
    problem = SimpleNamespace(body=SimpleNamespace(state=state), reference=reference,
        binding=SimpleNamespace(ratio=.25), camera=SimpleNamespace(p={}), size=37)
    config = dict(source_interval_seconds=[80.4, 81.4], taper_seconds=.4,
                  acceleration_scale_m_s2=4., weight=1.)
    factor = EntryMotionFactors(problem, config)
    zero = factor.evaluate(np.zeros(state.size), jacobian=True)
    t = times[factor.frames]
    ray = (root(t) - np.array([2., 4., 6.]))[:, :2]
    h = ray / np.linalg.norm(ray, axis=1)[:, None]
    expected_a = h @ np.array([2., 4.]) * .25
    np.testing.assert_allclose(zero['metrics']['actual_depth_acceleration_m_s2'], expected_a)
    assert np.linalg.norm(zero['residual']) > 0  # A correction-only prior would be zero.
    dt = np.diff(t)
    expected_dt = np.r_[dt[0]/2, (dt[:-1]+dt[1:])/2, dt[-1]/2]
    np.testing.assert_allclose(factor.dt, expected_dt)
    assert np.ptp(factor.dt) > .1
    np.testing.assert_allclose(factor.phase_weights, [0., .5, 1., 1., 1., 0.], atol=1e-12)
    np.testing.assert_allclose(zero['residual'], expected_a * np.sqrt(expected_dt * factor.phase_weights) / 4.)
    np.testing.assert_allclose(zero['parts'][factor.name], .5*np.sum(expected_dt * factor.phase_weights * expected_a**2)/16.)
    x = np.linspace(-.2, .3, state.size)
    result = factor.evaluate(x, jacobian=True)
    J = result['jac'].toarray()
    assert J.shape == (len(t), problem.size)
    np.testing.assert_array_equal(J[:, :17], 0.)
    for knot in range(state.m):
        for column, coordinate in enumerate(active):
            control = knot * state.d + column
            step = np.zeros(state.size)
            step[control] = 1e-6
            fd = (factor.evaluate(x+step)['residual'] - factor.evaluate(x-step)['residual']) / 2e-6
            np.testing.assert_allclose(J[:, 17+control], fd, rtol=1e-8, atol=1e-10)
            if coordinate not in (0, 1):
                np.testing.assert_array_equal(J[:, 17+control], 0.)
    # Trial changes cannot move the frozen reference depth direction.
    np.testing.assert_allclose(factor.h, h)
    assert result['metrics']['vertical_untouched'] and result['metrics']['finite_only']
