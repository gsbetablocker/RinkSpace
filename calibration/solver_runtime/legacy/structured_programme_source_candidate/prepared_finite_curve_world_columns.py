"""Prepare the original finite-curve branch once per current evaluation.

The complete world-knot array and its original endpoint/midpoint masks are
retained. Only the geometry derivative changes between ``apply`` calls.
This is evaluation-local preparation, never a cache across camera/world q.
"""
from dataclasses import dataclass

import numpy as np

from acceleration import nearest_segments
from ice_refine import BoardIntersectionError
from wide_anchor_candidate.full_brown_projection import project_central_valid_full_brown


def _owned_readonly(value):
    result = np.array(value, copy=True)
    result.flags.writeable = False
    return result


@dataclass(frozen=True)
class PreparedFiniteCurveWorldColumns:
    camera: np.ndarray
    world: np.ndarray
    observed: np.ndarray
    projected: np.ndarray
    depth: np.ndarray
    point_world_jacobian: np.ndarray
    central: np.ndarray
    midpoint_central: np.ndarray
    valid_segments: np.ndarray
    selected_index: np.ndarray
    selected_raw_t: np.ndarray
    clipped_t: np.ndarray
    nearest: np.ndarray
    residual: np.ndarray
    selected_direction: np.ndarray
    observed_minus_start: np.ndarray
    selected_length2: np.ndarray

    def apply(self, world_jacobian):
        """Return exact original r/J for a new derivative of these same knots.

        The raw station parameter enters the derivative before the original
        endpoint clamp. No knot or observation is removed or connected anew.
        """
        dw = np.asarray(world_jacobian, float)
        if (dw.ndim != 3 or dw.shape[:2] != (len(self.world), 3)
                or not np.isfinite(dw).all()):
            raise ValueError('Actual per-knot scene-unit world derivatives required')
        jac = np.einsum('nij,njk->nik', self.point_world_jacobian, dw)
        ja = jac[self.selected_index]
        jd = jac[self.selected_index + 1] - ja
        d = self.selected_direction
        u = self.observed_minus_start
        numerator_j = (-np.einsum('nij,ni->nj', ja, d)
                       + np.einsum('ni,nij->nj', u, jd))
        length_j = 2. * np.einsum('ni,nij->nj', d, jd)
        t_j = ((numerator_j - self.selected_raw_t[:, None] * length_j)
               / self.selected_length2[:, None])
        t_j[(self.selected_raw_t <= 0.) | (self.selected_raw_t >= 1.)] = 0.
        nearest_j = (ja + self.clipped_t[:, None, None] * jd
                     + d[:, :, None] * t_j[:, None, :])
        return self.residual, nearest_j.reshape(2 * len(self.observed), dw.shape[2])

    def matches(self, camera, world, observed):
        """An optional compatibility guard; callers may use ``apply`` directly."""
        return (np.array_equal(self.camera, np.asarray(camera, float))
                and np.array_equal(self.world, np.asarray(world, float))
                and np.array_equal(self.observed, np.asarray(observed, float).reshape(-1, 2)))


def prepare_curve_current_world_columns(camera, world, observed):
    """Project the entire current curve and choose its original finite branch."""
    xyz = np.asarray(world, float)
    observed = np.asarray(observed, float).reshape(-1, 2)
    projected, depth, flatj, central = project_central_valid_full_brown(
        camera, xyz, with_jacobian=True)
    j11 = np.asarray(flatj).reshape(len(xyz), 2, 11)
    # Camera centre is conditioned C/100. A world displacement has opposite
    # sign and scene-unit size, hence the original negative C columns /100.
    point_world_jacobian = -j11[:, :, :3] / 100.
    start, finish = projected[:-1], projected[1:]
    direction = finish - start
    length2 = np.sum(direction * direction, axis=1)
    mid = project_central_valid_full_brown(
        camera, .5 * (xyz[:-1] + xyz[1:]))[-1]
    valid = (central[:-1] & central[1:] & mid
             & np.isfinite(start).all(axis=1) & np.isfinite(finish).all(axis=1)
             & (depth[:-1] > 0) & (depth[1:] > 0) & (length2 > 1e-4))
    if not valid.any():
        raise BoardIntersectionError('ice_curve_no_central_segment', np.arange(len(observed)))
    index, raw_t, clipped_t, nearest = nearest_segments(
        observed, start, direction, length2, valid)
    return PreparedFiniteCurveWorldColumns(
        *[_owned_readonly(value) for value in (
            camera, xyz, observed, projected, depth, point_world_jacobian,
            central, mid, valid, index, raw_t, clipped_t, nearest,
            (nearest - observed).ravel(), direction[index],
            observed - start[index], length2[index])])


def curve_current_world_columns(camera, world, observed, world_jacobian, *, prepared=None):
    """Compatible r/J API, optionally using preparation of this exact branch."""
    if prepared is None:
        prepared = prepare_curve_current_world_columns(camera, world, observed)
    elif not isinstance(prepared, PreparedFiniteCurveWorldColumns) or not prepared.matches(camera, world, observed):
        raise ValueError('Prepared curve belongs to a different camera/world/UV evaluation')
    return prepared.apply(world_jacobian)
