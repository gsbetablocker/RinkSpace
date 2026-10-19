"""Independent physical-margin finite differences for the pure-array mount API."""
import itertools

import numpy as np
import pytest
from scipy.optimize import minimize_scalar

from calibration.program_tracking.first_shot_mount_tangent import full_mount_tangent


def _square_bounds(low, high):
    return (0. if low <= 0 <= high else min(low * low, high * high),
            max(low * low, high * high))


def _physical(q):
    value = q[:6]
    tilt, driver, a, b = q[6:8], q[8:10], q[10:13], q[13:16]
    low, high = sorted(tilt)
    bounds = []
    for axis in range(3):
        def component(beta):
            return a[axis] * np.cos(beta) + b[axis] * np.sin(beta)
        candidates = [component(low), component(high)]
        if high > low:
            for sign in (-1., 1.):
                fit = minimize_scalar(lambda beta: sign * component(beta),
                                      bounds=(low, high), method='bounded',
                                      options={'xatol': 1e-14})
                candidates.append(component(fit.x))
        bounds.append([min(candidates), max(candidates)])
    bounds = np.asarray(bounds)
    zlow, zhigh = sorted(driver)
    support = dict(v_min=bounds[:, 0], v_max=bounds[:, 1], intervals=[
        dict(left=0, driver=[zlow, zhigh], v_bounds=bounds)])
    lower, upper = np.array([-.2, -.3, -.1]), np.array([.2, .1, .4])
    matrices, targets = [], []
    for axis in (0, 1):
        for extreme in (0, 1):
            row = np.zeros(6); row[axis] = 1.; row[4] = bounds[axis, extreme]
            matrices.extend((row, -row)); targets.extend((lower[axis], -upper[axis]))
    points = dict.fromkeys(itertools.product((zlow, zhigh), bounds[2],
                                             _square_bounds(zlow, zhigh)))
    points.update(dict.fromkeys((0., rho, 0.) for rho in bounds[2]))
    for z, rho, square in points:
        row = np.array([0., 0., 1., z, rho, square])
        matrices.extend((row, -row)); targets.extend((lower[2], -upper[2]))
    for sign in (1., -1.):
        matrices.extend(sign * np.c_[np.eye(3), np.zeros((3, 3))])
        targets.extend(lower if sign > 0 else -upper)
    return value, np.asarray(matrices), np.asarray(targets) / np.exp(q[16]), support


def _linearize(q):
    value, matrix, lower, support = _physical(q)
    eye = np.eye(len(q))
    return full_mount_tangent(
        value=value, value_jacobian=eye[:6], matrix=matrix,
        absolute_lower=lower, log_scale_gradient=eye[16], support=support,
        tilt=q[6:8], tilt_jacobian=eye[6:8], driver=q[8:10],
        driver_jacobian=eye[8:10], interval_limits=np.array([[0., 1.]]),
        vector_a=q[10:13], vector_b=q[13:16],
        vector_a_jacobian=eye[10:13], vector_b_jacobian=eye[13:16],
        square_interval_bounds=_square_bounds)


def _state():
    return np.array([.03, -.12, .02, .4, .07, -.1, .15, .45, .2, .5,
                     .6, -.3, .4, .2, .7, .1, .2])


def test_full_margin_derivative_includes_pose_driver_axes_scale_and_values():
    q = _state(); result = _linearize(q); ids = result['original_mount_row_ids']
    value, matrix, lower, _ = _physical(q)
    np.testing.assert_array_equal(result['lower'], (lower - matrix @ value)[ids])
    direction = np.random.default_rng(1017).normal(size=len(q))
    direction /= np.linalg.norm(direction)
    eps = 2e-7
    def margins(state):
        value, matrix, lower, _ = _physical(state)
        return matrix @ value - lower
    fd = (margins(q + eps * direction) - margins(q - eps * direction)) / (2 * eps)
    np.testing.assert_allclose(result['matrix'] @ direction, fd[ids], atol=2e-9, rtol=2e-7)


def test_exact_support_ties_retain_each_distinct_branch_and_same_current_margins():
    q = _state(); q[7] = q[6]; q[9] = q[8]
    result = _linearize(q); ids = result['original_mount_row_ids']
    value, matrix, lower, _ = _physical(q)
    assert len(ids) > len(matrix)
    assert set(ids) == set(range(len(matrix)))
    np.testing.assert_array_equal(result['lower'], (lower - matrix @ value)[ids])
    assert result['report']['exact_tie_branches_retained']
    aliases = np.flatnonzero(ids == 0)
    jac = result['matrix'][aliases].toarray()
    assert len(np.unique(jac, axis=0)) > 1


def test_reject_incomplete_common_column_pullback():
    q = _state(); value, matrix, lower, support = _physical(q); eye = np.eye(len(q))
    with pytest.raises(ValueError, match='common-column'):
        full_mount_tangent(value=value, value_jacobian=eye[:5], matrix=matrix,
            absolute_lower=lower, log_scale_gradient=eye[16], support=support,
            tilt=q[6:8], tilt_jacobian=eye[6:8], driver=q[8:10],
            driver_jacobian=eye[8:10], interval_limits=np.array([[0., 1.]]),
            vector_a=q[10:13], vector_b=q[13:16], vector_a_jacobian=eye[10:13],
            vector_b_jacobian=eye[13:16], square_interval_bounds=_square_bounds)
