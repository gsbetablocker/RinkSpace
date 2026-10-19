from types import SimpleNamespace

import numpy as np
import pytest

from calibration.solver_runtime import install
install()
from installation_mount_candidate import mount_graph as mounts


def actual_square_state(monkeypatch):
    # The original current-course and signed-distance algorithms are used;
    # only the unrelated lever-range validation is isolated in this fixture.
    class Course:
        def sample_perimeter(self, value):
            return np.array([[0., 0., 0.], [100., 0., 0.],
                             [100., 100., 0.], [0., 100., 0.]]), None
    class Producer:
        def print_course_binding(self, value):
            return Course(), np.empty(0), np.zeros((1, len(value)))
    state = SimpleNamespace(geometry=SimpleNamespace(current=Producer()),
        current_user_pivot_outside_constraint_contract=dict(
            world_parameter_indices=[5], numerical_world_derivative_step=1e-6,
            by_installation={'LOW': {'pivot_parameter_indices': [0, 1, 2]}}))
    validated = []
    def validate(state, value):
        assert mounts.pivot_outside_domain(state, value)['valid']
        validated.append(value.copy())
    monkeypatch.setattr(mounts, 'validate_mounts', validate)
    return state, validated


@pytest.mark.parametrize('inward', [1e-11, 1e-9, .2])
def test_trial_retracts_to_original_boundary_without_fd_offset(monkeypatch, inward):
    state, validated = actual_square_state(monkeypatch)
    q = np.array([inward, 1., 3., 4., 5., 0.])
    original = q.copy()
    assert not mounts.pivot_outside_domain(state, q)['valid']
    actual, report = mounts.reinitialize_invalid_pivots_outside(
        state, q, require_coordinate_fd_margin=False)
    assert actual[0] == 0.
    np.testing.assert_array_equal(actual[1:], original[1:])
    np.testing.assert_array_equal(q, original)
    assert report['after_domain']['installations'][0]['side'] == 'on'
    assert report['after_domain']['valid']
    assert report['changes'][0]['numerical_initializer_margin_working_m'] == 0.
    assert report['changes'][0]['minimum_original_coordinate_FD_margin_working_m'] is None
    assert report['original_domain_boundary_retraction']
    np.testing.assert_array_equal(validated[0], actual)


def test_original_startup_fd_margin_is_preserved(monkeypatch):
    state, _ = actual_square_state(monkeypatch)
    q = np.array([1e-9, 1., 3., 4., 5., 0.])
    initialized, report = mounts.reinitialize_invalid_pivots_outside(state, q)
    assert initialized[0] == -32e-6
    assert report['changes'][0]['minimum_original_coordinate_FD_margin_working_m'] > 0.
    assert report['original_coordinate_fd_margin_required']
    assert not report['original_domain_boundary_retraction']


def test_legal_parent_is_bitexact_and_trial_does_not_build_fd_constraints(monkeypatch):
    state, _ = actual_square_state(monkeypatch)
    def forbidden(*args):
        raise AssertionError('trial retraction must not request a derivative reserve')
    monkeypatch.setattr(mounts, 'pivot_outside_linear_step_constraints', forbidden)
    q = np.array([0., 1., 3., 4., 5., 0.])
    actual, report = mounts.reinitialize_invalid_pivots_outside(
        state, q, require_coordinate_fd_margin=False)
    np.testing.assert_array_equal(actual, q)
    assert not report['applied']


def test_nonboolean_retraction_mode_is_rejected(monkeypatch):
    state, _ = actual_square_state(monkeypatch)
    with pytest.raises(ValueError, match='boolean'):
        mounts.reinitialize_invalid_pivots_outside(state, np.zeros(6),
                                                  require_coordinate_fd_margin=0)
