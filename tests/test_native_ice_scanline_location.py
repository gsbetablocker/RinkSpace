"""The source axis must survive the native factor's normalization and FD path."""
import numpy as np
import pytest


@pytest.fixture
def factor(monkeypatch):
    monkeypatch.setenv('RINKSPACE_PROJECTION_BACKEND', 'numpy')
    from calibration.solver_runtime import install
    install()
    from structured_programme_source_candidate.native_ice_curve_source import NativeIceCurveFactor
    f = NativeIceCurveFactor.__new__(NativeIceCurveFactor)
    f.parameter_count = 2
    f.rows = [dict(id='horizontal_RGB_edge', pts=100, actual_pts=100,
        split='fit', feature_id='test_ring', curve=dict(kind='house_ring'),
        uv=np.array([[1150., 610.], [1100., 650.]]),
        sigma=np.array([70., 90.]), curve_observation_kind='axis_scanline_branch',
        scanline_fixed_axis='y', scanline_branch='max_x', _source_optical_support=None)]
    f.held = []
    f.scales = np.repeat(f.rows[0]['sigma'], 2)
    f.source_point_qualification = None
    f.contract = {}
    def cameras(q):
        camera = np.zeros(25)
        camera[6:8] = np.log(1000.)
        camera[8:10] = [960.+50.*q[0], 540.]
        return {100: camera}
    def world(q, curve):
        angle = np.linspace(0., 2.*np.pi, 361)
        return np.c_[(2.+q[1])*np.cos(angle), 1.7*np.sin(angle), np.full(len(angle), 10.)]
    f._cameras, f._world = cameras, world
    return f


def test_native_scanline_keeps_original_horizontal_sigma_and_zero_second_scalar(factor):
    from calibration.solver_runtime.source_scanline_curve import horizontal_scanline_curve_value
    f, q = factor, np.zeros(2)
    row = f.rows[0]
    before = row['sigma'].copy()
    raw = horizontal_scanline_curve_value(f._cameras(q)[100], f._world(q,row['curve']), row['uv'], side='right')
    value = f._residual(q, f.rows)
    np.testing.assert_array_equal(value, raw/np.repeat(before, 2))
    np.testing.assert_array_equal(value[1::2], 0.)
    np.testing.assert_array_equal(row['sigma'], before)
    # Camera x-translation moves the measured x intersection by exactly 5px.
    moved = f._residual(np.array([.1, 0.]), f.rows)
    np.testing.assert_allclose((moved-value)[::2], 5./before, rtol=0., atol=1e-13)


def test_native_scanline_camera_and_world_fd_match_actual_objective(factor):
    from calibration.solver_runtime.source_scanline_curve import horizontal_scanline_curve_value
    f, q = factor, np.zeros(2)
    row = f.rows[0]
    _, association = horizontal_scanline_curve_value(f._cameras(q)[100], f._world(q,row['curve']),
        row['uv'], side='right', with_association=True)
    for column in range(2):
        delta = np.eye(2)[column]*1e-6
        actual = (f._residual(q+delta,f.rows)-f._residual(q-delta,f.rows))/(2e-6)
        chart = (f._derivative_residual(q+delta,cameras=f._cameras(q+delta),associations=[association])
                 - f._derivative_residual(q-delta,cameras=f._cameras(q-delta),associations=[association]))/(2e-6)
        np.testing.assert_allclose(chart, actual, atol=1e-9, rtol=1e-8)
        assert np.linalg.norm(actual)>0.


def test_review_preserves_legacy_distance_and_separately_names_scanline_error(factor):
    from wide_anchor_candidate.full_brown_projection import curve_vector_error_full_brown
    f, q = factor, np.zeros(2)
    row = f.rows[0]
    old = curve_vector_error_full_brown(f._cameras(q)[100],f._world(q,row['curve']),row['uv'],with_jacobian=False)
    distance = np.linalg.norm(np.asarray(old).reshape(-1,2),axis=1)
    review = f.review(q)['rows'][0]
    assert review['p90_native_px'] == pytest.approx(np.percentile(distance,90))
    new = review['axis_scanline_location_objective']
    assert new['legacy_nearest_curve_review_metric_preserved'] is True
    assert new['p90_horizontal_error_px'] != pytest.approx(review['p90_native_px'])
