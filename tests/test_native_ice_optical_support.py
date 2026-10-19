import numpy as np
import pytest

from calibration.solver_runtime import install

install()
from structured_programme_source_candidate.native_ice_curve_source import (
    _source_optical_support, _native_error_with_optical_support,
)


def qualified_row():
    row = dict(uv=[[300., 500.], [400., 500.]],
        normal=[[0., 1.], [0., 1.]], normal_interval_px=[[-3., 4.], [-3., 4.]],
        source_optical_normal_support_authorized=True)
    row['_source_optical_support'] = _source_optical_support(row)
    return row


def test_old_diagnostic_intervals_do_not_change_the_existing_objective():
    row = qualified_row()
    del row['source_optical_normal_support_authorized']
    row['_source_optical_support'] = _source_optical_support(row)
    raw = np.array([7., 2., -8., 9.])
    assert np.array_equal(_native_error_with_optical_support(row, raw), raw)


def test_optical_support_keeps_tangent_and_errors_beyond_visible_edge():
    row = qualified_row()
    assert np.array_equal(_native_error_with_optical_support(row, [7., 2., -8., 9.]),
        [7., 0., -8., 5.])
    assert np.array_equal(row['uv'], [[300., 500.], [400., 500.]])


def test_support_derivative_inside_and_outside_the_optical_band():
    row = qualified_row()
    raw = np.array([7., 2., -8., 9.])
    direction = np.array([2., -5., 3., 6.])
    h = 1e-6
    numerical = (_native_error_with_optical_support(row, raw + h * direction)
        - _native_error_with_optical_support(row, raw - h * direction)) / (2 * h)
    assert np.allclose(numerical, [2., 0., 3., 6.], atol=2e-9)


def test_source_support_rejects_nonunit_or_noncontaining_intervals():
    row = qualified_row()
    row['normal'][0] = [0., 2.]
    with pytest.raises(ValueError, match='unit normals'):
        _source_optical_support(row)
    row = qualified_row()
    row['normal_interval_px'][0] = [2., 5.]
    with pytest.raises(ValueError, match='contain original UV'):
        _source_optical_support(row)
