import numpy as np

from calibration.program_tracking.body_tracking.skate_postfit_plot import (
    reviewed_support_points, strict_collar_values,
)


def test_strict_collar_keeps_unknown_even_when_other_foot_is_known():
    pts = np.array([283136, 283392, 283648])
    records = [dict(pts=int(p), foot=f, status='measured_geometry', maximum_outside_m=.001*(i+f))
               for i, p in enumerate(pts) for f in (0, 1)]
    records[3] = dict(pts=int(pts[1]), foot=1, status='unknown', unknown_reason='open_section')
    report = dict(source_time_base=[1, 12800], source_frames=3,
                  source_first_last_pts=[int(pts[0]), int(pts[-1])], records=records)
    values, info = strict_collar_values(report, pts)
    np.testing.assert_allclose(values[[0, 2]], [.001, .003])
    assert np.isnan(values[1])
    assert info['unknown_or_missing_foot_rows'] == 1


def test_only_saved_finite_reviewed_support_is_plotted_without_filling_gaps():
    pts = np.array([round(t*12800) for t in (26.20, 26.22, 35.20, 39.78, 43.86)])
    packet = dict(pts=pts, skate_support_gap_m_mid=np.array([0., .012, np.nan, -.002, .025]))
    selected, values, info = reviewed_support_points(packet)
    np.testing.assert_array_equal(selected, pts[[1, 3, 4]])
    np.testing.assert_allclose(values, [.012, -.002, .025])
    assert info['reviewed_unknown_exposures'] == 1
    assert info['finite_unreviewed_values_excluded'] == 1
