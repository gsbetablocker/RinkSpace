"""One shared native row readout, without changing source UV or camera knots.

Ordinary FIT factors condition the camera on the observed source row. Explicit
native profile queries and rendering solve their projected row implicitly using
the same trajectory. Native rotations
follow the SO(3) shortest arc; domain log focal follows the native linear
interpolant. This is interpolation, not a temporal penalty or pose averaging.

The first/last sensor exposures use the velocity of their own terminal native
segment for at most half one native interval. No extra camera knot or cut is
introduced. Explicit off-sensor geometry navigation is not a source exposure.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json

import cv2
import numpy as np

from .z190_physical_camera import RowTiming, ViewState


class RowCameraDomainError(ValueError):
    """A row/time leaves this declared continuous native shot."""


def _hat(v):
    x, y, z = np.asarray(v, float)
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def _left_jacobian(v):
    v = np.asarray(v, float); theta2 = float(v @ v); k = _hat(v)
    if theta2 < 1e-10:
        a = .5 - theta2 / 24. + theta2 ** 2 / 720.
        b = 1. / 6. - theta2 / 120. + theta2 ** 2 / 5040.
    else:
        theta = np.sqrt(theta2)
        a = (1. - np.cos(theta)) / theta2
        b = (theta - np.sin(theta)) / (theta2 * theta)
    return np.eye(3) + a * k + b * (k @ k)


def _left_inverse(v):
    v = np.asarray(v, float); theta2 = float(v @ v); k = _hat(v)
    if theta2 < 1e-10:
        coefficient = 1. / 12. + theta2 / 720. + theta2 ** 2 / 30240.
    else:
        theta = np.sqrt(theta2)
        coefficient = (1. - .5 * theta / np.tan(.5 * theta)) / theta2
    return np.eye(3) - .5 * k + coefficient * (k @ k)


@dataclass(frozen=True)
class _Pair:
    left: int
    right: int
    a: np.ndarray
    b: np.ndarray
    ra: np.ndarray
    relative: np.ndarray
    logarithm: np.ndarray
    seconds: float


def _rotation_logs(rotations):
    """Batch principal SO(3) log, with an explicit near-pi fallback."""
    matrices = np.asarray(rotations, float)
    cosine = np.clip((np.trace(matrices, axis1=1, axis2=2) - 1.) * .5, -1., 1.)
    theta = np.arccos(cosine)
    vectors = np.column_stack((matrices[:, 2, 1] - matrices[:, 1, 2],
                               matrices[:, 0, 2] - matrices[:, 2, 0],
                               matrices[:, 1, 0] - matrices[:, 0, 1]))
    small = theta < 1e-5
    factor = np.empty(len(theta))
    factor[small] = .5 + theta[small] ** 2 / 12. + 7. * theta[small] ** 4 / 720.
    ordinary = ~small & (theta < np.pi - 1e-5)
    factor[ordinary] = theta[ordinary] / (2. * np.sin(theta[ordinary]))
    near_pi = ~small & ~ordinary
    factor[near_pi] = 0.
    result = vectors * factor[:, None]
    for index in np.flatnonzero(near_pi):
        result[index] = cv2.Rodrigues(matrices[index])[0].ravel()
    return result


class FrameRowCamera:
    """Immutable same-q trajectory/cache for one native source exposure.

    columns always contain the nine installation columns, unique previous /
    current / next RF4 columns, and one signed-ms readout column. A source row
    uses only two of those view knots; retaining the union prevents shape changes.
    """
    image_height = 1080

    def __init__(self, state, q, programme_index_1based, *, readout_index=None):
        values = np.asarray(state._q(q), float).copy()
        values.flags.writeable = False
        self._q = values
        self.calibration = state.calibration(values)
        self.index = int(programme_index_1based)
        self.frame_count = int(state.frame_count)
        if not 1 <= self.index <= self.frame_count or self.frame_count < 2:
            raise RowCameraDomainError("A row camera needs a supported exposure and two native knots")
        self.pts = np.asarray(state.pts, np.int64).copy()
        if self.pts.shape != (self.frame_count,) or np.any(np.diff(self.pts) <= 0):
            raise ValueError("Explicit strictly increasing native PTS are required")
        self.pts.flags.writeable = False
        self.time_base = float(Fraction(state.source_clock["time_base"]))
        self.native_interval_seconds = float(np.min(np.diff(self.pts)) * self.time_base)
        if not np.allclose(np.diff(self.pts) * self.time_base, .02, rtol=0., atol=1e-15):
            raise ValueError("This first-shot interface requires the actual native 20 ms clock")
        self.readout_index = (int(readout_index) if readout_index is not None
                              else int(state.row_readout_index))
        self.readout_milliseconds = float(values[self.readout_index])
        if abs(self.readout_milliseconds) > 1000. * self.native_interval_seconds + 1e-10:
            raise RowCameraDomainError("Shared row readout exceeds one native exposure interval")
        direction = 1 if self.readout_milliseconds >= 0. else -1
        self.row_timing = RowTiming(abs(self.readout_milliseconds) * .001, direction,
            "native-source-conditioned shared readout candidate; not measured shutter time")
        self._view_columns = tuple(np.asarray(state.view_columns(i + 1), int)
                                   for i in range(self.frame_count))
        self._views = np.asarray([values[columns] for columns in self._view_columns])
        self._views.flags.writeable = False
        neighbours = sorted({max(1, self.index - 1), self.index,
                             min(self.frame_count, self.index + 1)})
        self.columns = np.r_[np.arange(118, 127),
            np.concatenate([self._view_columns[n - 1] for n in neighbours]),
            self.readout_index].astype(int)
        if len(np.unique(self.columns)) != len(self.columns):
            raise ValueError("Readout and installation/view columns must be distinct")
        self.columns.flags.writeable = False
        self._column_locations = {int(column): index for index, column in enumerate(self.columns)}
        self.scene_unit_working_m = float(state.scene_unit_working_m)
        from .first_shot_zoom_camera import zoom_shared_mapping
        self._zoom_columns, self._zoom_selected, self._zoom_scales = zoom_shared_mapping(state, self.calibration)
        if len(self._zoom_columns):
            # Keep readout last for the existing analytic time derivative.
            self.columns = np.r_[self.columns[:-1], self._zoom_columns, self.readout_index]
            if len(np.unique(self.columns)) != len(self.columns):
                raise ValueError("Shared zoom columns overlap the native camera chart")
            self.columns.flags.writeable = False
            self._column_locations = {int(column): index for index, column in enumerate(self.columns)}
        self._pairs = {}
        self._mid_camera = self.calibration.camera25(ViewState.from_state4(self._views[self.index - 1]))
        self.cache_key = ("rinkspace.native-row-camera.v1", self.index,
            int(self.pts[self.index - 1]), hashlib.sha256(values.tobytes()).hexdigest(),
            hashlib.sha256(json.dumps(self.calibration.manifest(), sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest(),
            self.readout_index, self.image_height)

    def _pair(self, left):
        left = int(left)
        if left not in self._pairs:
            a, b = self._views[left], self._views[left + 1]
            ra = cv2.Rodrigues(a[:3])[0]; rb = cv2.Rodrigues(b[:3])[0]
            relative = rb @ ra.T
            # OpenCV's matrix-to-vector branch rounds very small relative
            # rotations to zero. Native stationary/slow-pan neighbours still
            # need their actual short arc and a continuous endpoint derivative.
            logarithm = _rotation_logs(relative[None])[0]
            if np.linalg.norm(logarithm) >= np.pi - 1e-6:
                raise RowCameraDomainError("Adjacent native rotations have an ambiguous shortest arc")
            self._pairs[left] = _Pair(left, left + 1, a, b, ra, relative, logarithm,
                float((self.pts[left + 1] - self.pts[left]) * self.time_base))
        return self._pairs[left]

    def _row_offsets(self, rows, geometry_navigation):
        rows = np.asarray(rows, float)
        if rows.ndim != 1 or not np.isfinite(rows).all():
            raise ValueError("A finite one-dimensional row array is required")
        if not geometry_navigation:
            return self.row_timing.sample_times(0., rows, self.image_height)
        # Off-sensor rows have no measured exposure. Continue the declared row
        # law only for geometry navigation; no pixel/camera parameter is clamped.
        anchors = np.clip(rows, 0., self.image_height - 1.)
        offsets = self.row_timing.sample_times(0., anchors, self.image_height)
        return offsets + self.readout_milliseconds * .001 * (rows - anchors) / (self.image_height - 1.)

    def _brackets(self, offsets, *, geometry_navigation=False):
        target_pts = float(self.pts[self.index - 1]) + offsets / self.time_base
        half_interval = .5 * self.native_interval_seconds / self.time_base
        if np.any(target_pts < self.pts[0] - half_interval - 1e-7) or np.any(target_pts > self.pts[-1] + half_interval + 1e-7):
            raise RowCameraDomainError("Row time exceeds this shot's explicit terminal half-interval support")
        left = np.searchsorted(self.pts, target_pts, side="right") - 1
        left = np.clip(left, 0, self.frame_count - 2)
        # Form alpha from the small relative native interval, not from two
        # large absolute timestamps. This keeps the tau finite-difference and
        # analytic chains consistent down to sub-microsecond row perturbations.
        alpha = (((self.pts[self.index - 1] - self.pts[left]) * self.time_base + offsets) /
                 ((self.pts[left + 1] - self.pts[left]) * self.time_base))
        return left.astype(int), alpha

    def _interpolate(self, pair, alpha, jacobian):
        if alpha == 0.:
            r = pair.a[:3].copy(); rotation = pair.ra
            advance = np.eye(3)
        elif alpha == 1.:
            r = pair.b[:3].copy(); advance = pair.relative
            rotation = advance @ pair.ra
        else:
            advance = cv2.Rodrigues(alpha * pair.logarithm)[0]
            rotation = advance @ pair.ra
            r = _rotation_logs(rotation[None])[0]
        value = np.r_[r, (1. - alpha) * pair.a[3] + alpha * pair.b[3]]
        if not jacobian:
            return ViewState.from_state4(value), None, None, None
        inverse = _left_inverse(r)
        middle = alpha * _left_jacobian(alpha * pair.logarithm) @ _left_inverse(pair.logarithm)
        ja = np.zeros((4, 4)); jb = np.zeros((4, 4))
        ja[:3, :3] = inverse @ (advance - middle @ pair.relative) @ _left_jacobian(pair.a[:3])
        jb[:3, :3] = inverse @ middle @ _left_jacobian(pair.b[:3])
        ja[3, 3] = 1. - alpha; jb[3, 3] = alpha
        derivative_alpha = np.r_[inverse @ pair.logarithm, pair.b[3] - pair.a[3]]
        return ViewState.from_state4(value), ja, jb, derivative_alpha

    def camera15_mapping(self, source_y, *, jacobian=True, geometry_navigation=False):
        rows = np.asarray([source_y], float)
        offsets = self._row_offsets(rows, bool(geometry_navigation))
        left, alpha = self._brackets(offsets, geometry_navigation=bool(geometry_navigation))
        # Select the previous native derivative for upper rows at exactly tau=0.
        if self.readout_milliseconds == 0. and source_y < .5 * (self.image_height - 1.) and self.index > 1:
            left[0] = self.index - 2; alpha[0] = 1.
        pair = self._pair(left[0])
        view, ja, jb, da = self._interpolate(pair, float(alpha[0]), bool(jacobian))
        if not jacobian:
            camera = self._mid_camera.copy() if self.readout_milliseconds == 0. else self.calibration.camera25(view)
            return camera, self.columns, None
        camera, shared, per_view = self.calibration.camera25_and_jacobians(view)
        if self.readout_milliseconds == 0.:
            camera = self._mid_camera.copy()
        names = self.calibration.shared_state_names()
        selected = [names.index("pivot_" + axis + "_scene_div100") for axis in "xyz"]
        selected += [names.index("lever_" + axis + "_scene_div100") for axis in "xyz"]
        selected += [names.index(name + "_degree_0") for name in ("principal_x_fraction", "principal_y_fraction", "k1")]
        install_scale = np.r_[np.full(6, 1. / (100. * self.scene_unit_working_m)),
                              1. / 1920., 1. / 1080., .001]
        full_chain = np.zeros((25, len(self.columns)))
        full_chain[:, :9] = shared[:, selected] * install_scale
        if len(self._zoom_columns):
            locations = [self._column_locations[int(c)] for c in self._zoom_columns]
            full_chain[:, locations] = shared[:, self._zoom_selected] * self._zoom_scales
        for knot, derivative in ((pair.left, ja), (pair.right, jb)):
            locations = [self._column_locations[int(c)] for c in self._view_columns[knot]]
            full_chain[:, locations] += per_view @ derivative
        d_alpha_d_ms = .001 * (float(source_y) / (self.image_height - 1.) - .5) / pair.seconds
        full_chain[:, -1] = per_view @ da * d_alpha_d_ms
        if np.any(abs(full_chain[15:]) > 1e-12):
            raise ValueError("Active rolling camera leaves the original Brown15 residual chart")
        inverse_condition = np.r_[np.full(3, .01), np.ones(5), np.full(3, .001), np.ones(4)]
        return camera, self.columns, full_chain[:15] * inverse_condition[:, None]

    def camera15_row_derivative(self, source_y, *, geometry_navigation=False):
        """Conditioned camera15 velocity per sensor pixel at a fixed state.

        This is the same physical trajectory as camera15_mapping. It supplies
        the row-coordinate derivative needed by a fixed native query's implicit
        sensor-row root; it introduces no state or temporal penalty.
        """
        rows = np.asarray([source_y], float)
        offsets = self._row_offsets(rows, bool(geometry_navigation))
        if self.readout_milliseconds == 0.:
            return np.zeros(15)
        left, alpha = self._brackets(offsets, geometry_navigation=bool(geometry_navigation))
        pair = self._pair(left[0])
        view, _, _, da = self._interpolate(pair, float(alpha[0]), True)
        _, _, per_view = self.calibration.camera25_and_jacobians(view)
        speed = per_view @ da * (.001*self.readout_milliseconds /
                                ((self.image_height-1.)*pair.seconds))
        if np.any(abs(speed[15:]) > 1e-12):
            raise ValueError("Active row velocity leaves the original Brown15 residual chart")
        conditioning = np.r_[np.full(3, .01), np.ones(5), np.full(3, .001), np.ones(4)]
        return speed[:15]*conditioning

    def camera25_at_row(self, row, *, geometry_navigation=False):
        if not geometry_navigation:
            return self.camera15_mapping(row, jacobian=False)[0]
        return self.camera25_at_rows(np.asarray([row]), geometry_navigation=True)[0]

    def camera25_at_rows(self, rows, *, geometry_navigation=False):
        """Vectorised value-only trajectory; no endpoint/installation J is built."""
        rows = np.asarray(rows, float)
        offsets = self._row_offsets(rows, bool(geometry_navigation))
        if self.readout_milliseconds == 0.:
            return np.repeat(self._mid_camera[None], len(rows), axis=0)
        left, alpha = self._brackets(offsets, geometry_navigation=bool(geometry_navigation))
        rotations = np.empty((len(rows), 3, 3)); views = np.empty((len(rows), 4))
        for knot in np.unique(left):
            take = left == knot; pair = self._pair(knot); aa = alpha[take]
            theta = float(np.linalg.norm(pair.logarithm)); k = _hat(pair.logarithm)
            if theta < 1e-8:
                s = aa; c = .5 * aa ** 2
            else:
                s = np.sin(aa * theta) / theta
                c = (1. - np.cos(aa * theta)) / theta ** 2
            advance = np.eye(3)[None] + s[:, None, None] * k + c[:, None, None] * (k @ k)
            rotations[take] = advance @ pair.ra
            views[take, 3] = (1. - aa) * pair.a[3] + aa * pair.b[3]
        views[:, :3] = _rotation_logs(rotations)
        cal = self.calibration; mode = cal.output_mode
        x = views[:, 3] - cal.q_reference
        polynomial = lambda coefficients: np.polynomial.polynomial.polyval(x, coefficients)
        lever = np.tile(np.asarray(cal.lever_xyz), (len(rows), 1))
        lever[:, 2] += polynomial(cal.axial)
        centres = np.asarray(cal.pivot_xyz) + np.einsum("mji,mj->mi", rotations, lever)
        focal = mode.domain_wh[0] * np.exp(views[:, 3]); root = np.sqrt(cal.aspect_fx_over_fy)
        intrinsic = np.zeros((len(rows), 3, 3)); intrinsic[:, 2, 2] = 1.
        intrinsic[:, 0, 0] = focal * root; intrinsic[:, 1, 1] = focal / root
        intrinsic[:, 0, 2] = mode.domain_wh[0] * polynomial(cal.principal_x_fraction)
        intrinsic[:, 1, 2] = mode.domain_wh[1] * polynomial(cal.principal_y_fraction)
        final = mode.A[None] @ intrinsic
        cameras = np.zeros((len(rows), 25))
        cameras[:, :3] = centres; cameras[:, 3:6] = views[:, :3]
        cameras[:, 6] = np.log(final[:, 0, 0]); cameras[:, 7] = np.log(final[:, 1, 1])
        cameras[:, 8:10] = final[:, :2, 2]; cameras[:, 10] = final[:, 0, 1]
        cameras[:, 11] = polynomial(cal.k1); cameras[:, 12] = polynomial(cal.k2)
        cameras[:, 13:15] = cal.tangential_p1_p2
        if not np.isfinite(cameras).all():
            raise RowCameraDomainError("Nonfinite native row camera")
        return cameras

    def provenance(self):
        return dict(schema="rinkspace.native-frame-row-camera.v1", native_pts=int(self.pts[self.index - 1]),
            signed_readout_milliseconds=self.readout_milliseconds,
            source_mid_time_seconds=float(self.pts[self.index - 1] * self.time_base),
            native_interval_seconds=self.native_interval_seconds, interpolation="SO3_shortest_arc_and_linear_domain_logf",
            terminal_policy="same-shot terminal-segment velocity extrapolation, at most half native interval",
            source_row_time_api="z190_physical_camera.RowTiming", per_frame_extra_parameters=0,
            shared_readout_parameter_index=self.readout_index, columns=self.columns.tolist(),
            observed_row_conditioned_FIT=True, renderer_uses_same_trajectory_implicit_row=True,
            off_sensor_navigation_is_measured_exposure=False)


def prepare_frame_row_camera(state, q, programme_index_1based, *, readout_index=None):
    return FrameRowCamera(state, q, programme_index_1based, readout_index=readout_index)


def camera15_mapping_at_row(state, q, programme_index_1based, source_y, *, jacobian=True, readout_index=None):
    return prepare_frame_row_camera(state, q, programme_index_1based,
        readout_index=readout_index).camera15_mapping(source_y, jacobian=jacobian)
