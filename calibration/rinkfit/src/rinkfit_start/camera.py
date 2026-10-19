"""Double-precision physical camera reference for the v1.1 design contract.

Conventions: R maps world to camera; image x is right, y down, camera z
forward; angles are radians, times seconds, coordinates native pixel centers.
Every invalid forward evaluation carries a status.  No failed projection is
replaced by a zero residual, a clamped row/time, or an image-space correction.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import ceil
from typing import Callable, Optional, Sequence

import numpy as np
from scipy.optimize import brentq


class CameraDomainError(ValueError):
    """The requested evaluation is outside the declared physical domain."""


class UnsupportedCameraModel(NotImplementedError):
    """An explicitly requested candidate is not implemented by this reference."""


class SingularRSRoot(CameraDomainError):
    """A row root has no sufficiently conditioned implicit derivative."""


class ProjectionStatus(str, Enum):
    VALID = "valid"
    NON_POSITIVE_DEPTH = "non_positive_depth"
    DISTORTION_DOMAIN_INVALID = "distortion_domain_invalid"
    DISTORTION_UNCERTIFIED = "distortion_uncertified"
    TIME_DOMAIN_INVALID = "time_domain_invalid"
    INVALID_DOMAIN = "invalid_domain"
    OUTSIDE_IMAGE = "outside_image"
    NO_ROOT = "no_root"
    MULTIPLE_ROOTS = "multiple_roots"
    SINGULAR_ROOT = "singular_root"
    UNIQUENESS_UNCERTIFIED = "uniqueness_uncertified"
    ROOT_SEARCH_FAILED = "root_search_failed"


def _array(value, shape, name):
    a = np.array(value, dtype=np.float64, copy=True)
    if a.shape != shape or not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be a finite float64 array of shape {shape}")
    a.setflags(write=False)
    return a


def _finite(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def skew(v: Sequence[float]) -> np.ndarray:
    x, y, z = _array(v, (3,), "rotation vector")
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def axis_rotation(axis: Sequence[float], angle: float) -> np.ndarray:
    """Exp(angle [axis]x) for a *declared unit* axis; no renormalization."""
    axis = _array(axis, (3,), "axis")
    if abs(float(axis @ axis) - 1.0) > 1e-10:
        raise ValueError("rotation axis must already have unit norm")
    angle = _finite(angle, "angle")
    k = skew(axis)
    return np.eye(3) + np.sin(angle) * k + (1.0 - np.cos(angle)) * (k @ k)


@dataclass(frozen=True)
class Pose:
    R: np.ndarray
    C: np.ndarray

    def __post_init__(self):
        object.__setattr__(self, "R", _array(self.R, (3, 3), "R"))
        object.__setattr__(self, "C", _array(self.C, (3,), "C"))
        if not np.allclose(self.R.T @ self.R, np.eye(3), atol=1e-10, rtol=0):
            raise ValueError("R must be a rigid world-to-camera rotation")
        if abs(np.linalg.det(self.R) - 1.0) > 1e-10:
            raise ValueError("R must have determinant +1")


@dataclass(frozen=True)
class FixedHead:
    """One shared fixed mechanism.  P, axes, d and a have no time indices.

    The reference pupil displacement is exactly zero: ell(z)=l1*z+l2*z*z.
    The optional quadratic coefficient is a declared finite optical candidate,
    not a second focus stream.  The default mechanical candidate is orthogonal.
    """

    R0: np.ndarray
    n: np.ndarray
    b: np.ndarray
    P: np.ndarray
    d: np.ndarray
    a: float = 0.0
    pupil_linear: float = 0.0
    pupil_quadratic: float = 0.0
    axis_model: str = "orthogonal"

    def __post_init__(self):
        pose = Pose(self.R0, self.P)
        object.__setattr__(self, "R0", pose.R)
        object.__setattr__(self, "P", pose.C)
        for name in ("n", "b", "d"):
            object.__setattr__(self, name, _array(getattr(self, name), (3,), name))
        for name in ("a", "pupil_linear", "pupil_quadratic"):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        if self.axis_model != "orthogonal":
            raise UnsupportedCameraModel("only one shared orthogonal fixed-head candidate is implemented")
        if abs(self.n @ self.n - 1.0) > 1e-10 or abs(self.b @ self.b - 1.0) > 1e-10:
            raise ValueError("n and b must be unit vectors; use a manifold/chart upstream")
        if abs(self.b @ self.R0 @ self.n) > 1e-10:
            raise ValueError("orthogonal mechanism requires b^T R0 n = 0")

    def pupil(self, z: float) -> float:
        z = _finite(z, "z")
        return self.pupil_linear * z + self.pupil_quadratic * z * z

    def pose(self, alpha: float, beta: float, z: float) -> Pose:
        return self.pose_and_velocity(alpha, beta, z, 0.0, 0.0, 0.0)[0]

    def pose_and_velocity(self, alpha, beta, z, alpha_dot, beta_dot, z_dot):
        for name, val in (("z", z), ("alpha_dot", alpha_dot),
                          ("beta_dot", beta_dot), ("z_dot", z_dot)):
            _finite(val, name)
        R = axis_rotation(self.b, beta) @ self.R0 @ axis_rotation(self.n, alpha)
        R_dot = beta_dot * skew(self.b) @ R + alpha_dot * R @ skew(self.n)
        q = R.T @ self.b
        q_dot = R_dot.T @ self.b
        cross = np.cross(self.n, q)
        norm = float(np.linalg.norm(cross))
        if norm <= 1e-12:
            raise CameraDomainError("fixed-head common-normal direction is singular")
        h = cross / norm
        cross_dot = np.cross(self.n, q_dot)
        h_dot = (cross_dot - h * (h @ cross_dot)) / norm
        lever = self.d + np.array([0.0, 0.0, self.pupil(z)])
        lever_dot = np.array([0.0, 0.0,
                              (self.pupil_linear + 2.0 * self.pupil_quadratic * z) * z_dot])
        C = self.P + R.T @ lever + self.a * h
        C_dot = R_dot.T @ lever + R.T @ lever_dot + self.a * h_dot
        return Pose(R, C), R_dot, C_dot


def _quadratic_minimum(c0, c1, c2, xmax):
    values = [c0, c0 + c1 * xmax + c2 * xmax * xmax]
    if c2 > 0:
        x = -c1 / (2.0 * c2)
        if 0.0 < x < xmax:
            values.append(c0 + c1 * x + c2 * x * x)
    return min(values)


@dataclass(frozen=True)
class Brown:
    k1: float = 0.0
    k2: float = 0.0
    p1: float = 0.0
    p2: float = 0.0
    radius_max: float = 2.0

    def __post_init__(self):
        for name in ("k1", "k2", "p1", "p2", "radius_max"):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        if self.radius_max <= 0:
            raise ValueError("Brown radius_max must be positive and explicitly declared")

    def no_fold_lower_bound(self) -> float:
        """Exact radial minima minus a conservative tangential norm bound.

        Positive certifies the full declared disk at this fixed zoom.  Failure
        is *uncertified*, not proof that the lens is physically invalid.
        """
        r2 = self.radius_max ** 2
        radial = min(_quadratic_minimum(1.0, self.k1, self.k2, r2),
                     _quadratic_minimum(1.0, 3.0 * self.k1, 5.0 * self.k2, r2))
        return radial - 6.0 * np.hypot(self.p1, self.p2) * self.radius_max

    def distort_with_jacobian(self, xy):
        x, y = _array(xy, (2,), "normalized ray")
        r2 = x * x + y * y
        if r2 > self.radius_max * self.radius_max:
            raise CameraDomainError("normalized ray exceeds the declared Brown disk")
        radial = 1.0 + self.k1 * r2 + self.k2 * r2 * r2
        radial_x = 2.0 * x * (self.k1 + 2.0 * self.k2 * r2)
        radial_y = 2.0 * y * (self.k1 + 2.0 * self.k2 * r2)
        xd = x * radial + 2.0 * self.p1 * x * y + self.p2 * (r2 + 2.0 * x * x)
        yd = y * radial + self.p1 * (r2 + 2.0 * y * y) + 2.0 * self.p2 * x * y
        J = np.array([[radial + x * radial_x + 2 * self.p1 * y + 6 * self.p2 * x,
                       x * radial_y + 2 * self.p1 * x + 2 * self.p2 * y],
                      [y * radial_x + 2 * self.p1 * x + 2 * self.p2 * y,
                       radial + y * radial_y + 6 * self.p1 * y + 2 * self.p2 * x]])
        if not np.all(np.isfinite(J)) or not np.isfinite(xd + yd):
            raise CameraDomainError("nonfinite Brown arithmetic")
        return np.array([xd, yd]), J


@dataclass(frozen=True)
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float

    def __post_init__(self):
        for name in ("fx", "fy", "cx", "cy"):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        if self.fx <= 0 or self.fy <= 0:
            raise CameraDomainError("focal lengths must be positive")


@dataclass(frozen=True)
class ZoomOptics:
    """Native K(z), low-order radial zoom dependence and fixed tangential terms.

    f_ref is fixed after independent initialization.  z=log(f/f_ref).
    Principal-point slopes and radial slopes are shared finite candidates.
    There is no arbitrary output affine, OIS or independent focus trajectory.
    """

    f_ref: float
    principal_point: tuple[float, float] = (959.5, 539.5)
    eta: float = 0.0
    principal_linear: tuple[float, float] = (0.0, 0.0)
    k1: tuple[float, float] = (0.0, 0.0)
    k2: tuple[float, float] = (0.0, 0.0)
    p1: float = 0.0
    p2: float = 0.0
    radius_max: float = 2.0
    output_model: str = "native_pinhole_brown"

    def __post_init__(self):
        if self.output_model != "native_pinhole_brown":
            raise UnsupportedCameraModel("output warp, OIS and exposure integration are not implemented")
        for name in ("f_ref", "eta", "p1", "p2", "radius_max"):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        if self.f_ref <= 0 or self.radius_max <= 0:
            raise ValueError("f_ref and radius_max must be positive")
        for name in ("principal_point", "principal_linear", "k1", "k2"):
            a = _array(getattr(self, name), (2,), name)
            object.__setattr__(self, name, tuple(float(v) for v in a))

    def at(self, log_f: float):
        log_f = _finite(log_f, "log_f")
        z = log_f - np.log(self.f_ref)
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            fx = np.exp(log_f + self.eta / 2.0)
            fy = np.exp(log_f - self.eta / 2.0)
        if not np.isfinite(fx + fy) or min(fx, fy) <= 0:
            raise CameraDomainError("focal exponential is outside finite positive arithmetic")
        K = Intrinsics(fx, fy, self.principal_point[0] + self.principal_linear[0] * z,
                       self.principal_point[1] + self.principal_linear[1] * z)
        D = Brown(self.k1[0] + self.k1[1] * z, self.k2[0] + self.k2[1] * z,
                  self.p1, self.p2, self.radius_max)
        return K, D, float(z)


@dataclass(frozen=True)
class ProjectionResult:
    status: ProjectionStatus
    uv: Optional[np.ndarray] = None
    depth: Optional[float] = None
    message: str = ""

    def __post_init__(self):
        if self.uv is not None:
            object.__setattr__(self, "uv", _array(self.uv, (2,), "uv"))

    @property
    def valid(self):
        return self.status is ProjectionStatus.VALID

    def require_uv(self):
        if not self.valid or self.uv is None:
            raise CameraDomainError(f"{self.status.value}: {self.message}")
        return self.uv


def project(world, pose: Pose, intrinsics: Intrinsics, brown: Brown) -> ProjectionResult:
    X = _array(world, (3,), "world point")
    Xc = pose.R @ (X - pose.C)
    if not np.all(np.isfinite(Xc)):
        return ProjectionResult(ProjectionStatus.INVALID_DOMAIN, message="nonfinite camera coordinates")
    depth = float(Xc[2])
    if depth <= 0.0:
        return ProjectionResult(ProjectionStatus.NON_POSITIVE_DEPTH, depth=depth)
    xy = Xc[:2] / depth
    if not np.all(np.isfinite(xy)) or xy @ xy > brown.radius_max ** 2:
        return ProjectionResult(ProjectionStatus.DISTORTION_DOMAIN_INVALID, depth=depth,
                                message="ray outside finite declared distortion disk")
    if brown.no_fold_lower_bound() <= 0:
        return ProjectionResult(ProjectionStatus.DISTORTION_UNCERTIFIED, depth=depth,
                                message="conservative disk certificate failed; this is not an invalidity proof")
    try:
        distorted, _ = brown.distort_with_jacobian(xy)
        uv = distorted * [intrinsics.fx, intrinsics.fy] + [intrinsics.cx, intrinsics.cy]
        if not np.all(np.isfinite(uv)):
            raise CameraDomainError("nonfinite pixel arithmetic")
        return ProjectionResult(ProjectionStatus.VALID, uv=uv, depth=depth)
    except CameraDomainError as exc:
        return ProjectionResult(ProjectionStatus.DISTORTION_DOMAIN_INVALID, depth=depth, message=str(exc))


@dataclass(frozen=True)
class UniformCubicBasis:
    """Cardinal cubic basis with one position per native frame and extensions.

    Control j is located at first_control_time+j*dt.  On cell i the four
    controls i-1..i+2 are active.  The closed evaluation domain is control
    times 1..N-2.  At its right endpoint the last legal cell is evaluated at
    u=1; this is exact endpoint handling, not extrapolation or clamping.
    """

    first_control_time: float
    dt: float
    n_controls: int

    def __post_init__(self):
        _finite(self.first_control_time, "first_control_time")
        _finite(self.dt, "dt")
        if self.dt <= 0 or not isinstance(self.n_controls, int) or self.n_controls < 4:
            raise ValueError("uniform cubic basis needs dt>0 and at least four controls")

    @property
    def domain(self):
        return (self.first_control_time + self.dt,
                self.first_control_time + (self.n_controls - 2) * self.dt)

    @classmethod
    def for_native_frames(cls, n_frames, fps=50.0, first_frame_time=0.0, max_abs_readout=0.02):
        if not isinstance(n_frames, int) or n_frames < 2:
            raise ValueError("at least two uniformly spaced native frames are required")
        fps = _finite(fps, "fps")
        max_abs_readout = _finite(max_abs_readout, "max_abs_readout")
        if fps <= 0 or max_abs_readout < 0:
            raise ValueError("fps>0 and max_abs_readout>=0 are required")
        dt = 1.0 / fps
        extension = int(ceil(max_abs_readout / (2.0 * dt))) + 1
        basis = cls(first_frame_time - extension * dt, dt, n_frames + 2 * extension)
        basis.require_coverage(first_frame_time, first_frame_time + (n_frames - 1) * dt,
                               max_abs_readout)
        return basis

    def require_coverage(self, first_frame_time, last_frame_time, readout):
        left, right = self.domain
        needed_left = first_frame_time - abs(readout) / 2.0
        needed_right = last_frame_time + abs(readout) / 2.0
        if needed_left < left or needed_right > right:
            raise CameraDomainError("spline does not cover the first/last half-readout; add controls")

    def local(self, time: float, derivative=0):
        time = _finite(time, "spline time")
        lo, hi = self.domain
        if time < lo or time > hi:
            raise CameraDomainError(f"time {time} outside closed spline domain [{lo}, {hi}]")
        if derivative not in (0, 1, 2, 3):
            raise ValueError("cubic derivative order must be 0, 1, 2 or 3")
        if time == lo:
            i, u = 1, 0.0
        elif time == hi:
            i, u = self.n_controls - 3, 1.0
        else:
            x = (time - self.first_control_time) / self.dt
            i = int(np.floor(x))
            # Floating division may round an interior time onto the terminal
            # knot.  Selecting the last cell preserves that same time exactly.
            if i == self.n_controls - 2:
                i = self.n_controls - 3
            u = x - i
        if i < 1 or i > self.n_controls - 3:
            raise CameraDomainError("spline cell outside allocated controls")
        if derivative == 0:
            w = np.array([(1-u)**3, 3*u**3-6*u*u+4, -3*u**3+3*u*u+3*u+1, u**3]) / 6.0
        elif derivative == 1:
            w = np.array([-3*(1-u)**2, 9*u*u-12*u, -9*u*u+6*u+3, 3*u*u]) / (6*self.dt)
        elif derivative == 2:
            w = np.array([6*(1-u), 18*u-12, -18*u+6, 6*u]) / (6*self.dt**2)
        else:
            w = np.array([-1.0, 3.0, -3.0, 1.0]) / self.dt**3
        return np.arange(i-1, i+3), w

    def row(self, time, derivative=0):
        indices, weights = self.local(time, derivative)
        result = np.zeros(self.n_controls)
        result[indices] = weights
        return result


@dataclass(frozen=True)
class UniformCubicSpline:
    basis: UniformCubicBasis
    controls: np.ndarray

    def __post_init__(self):
        object.__setattr__(self, "controls", _array(self.controls, (self.basis.n_controls,), "controls"))

    def evaluate(self, time, derivative=0):
        indices, weights = self.basis.local(time, derivative)
        return float(weights @ self.controls[indices])

    __call__ = evaluate


@dataclass(frozen=True)
class GaugeFixedSpline:
    """N-1 coordinates with the exact function-value gauge a(t_ref)=0.

    The coefficient with largest reference basis weight is eliminated.  Its
    value depends on the other reference-support coefficients; no coefficient
    is merely set to zero.  jacobian_free includes that complete dependence.
    """

    basis: UniformCubicBasis
    free_controls: np.ndarray
    reference_time: float
    dependent_index: int = field(init=False)
    free_indices: np.ndarray = field(init=False, repr=False)
    controls: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        free = _array(self.free_controls, (self.basis.n_controls-1,), "free_controls")
        row = self.basis.row(self.reference_time)
        dep = int(np.argmax(np.abs(row)))
        inds = np.delete(np.arange(self.basis.n_controls), dep)
        inds.setflags(write=False)
        controls = np.empty(self.basis.n_controls)
        controls[inds] = free
        controls[dep] = -(row[inds] @ free) / row[dep]
        controls.setflags(write=False)
        object.__setattr__(self, "free_controls", free)
        object.__setattr__(self, "dependent_index", dep)
        object.__setattr__(self, "free_indices", inds)
        object.__setattr__(self, "controls", controls)

    def evaluate(self, time, derivative=0):
        indices, weights = self.basis.local(time, derivative)
        return float(weights @ self.controls[indices])

    __call__ = evaluate

    def jacobian_free(self, time, derivative=0):
        b = self.basis.row(time, derivative)
        bref = self.basis.row(self.reference_time)
        d, f = self.dependent_index, self.free_indices
        return b[f] - b[d] * bref[f] / bref[d]


@dataclass(frozen=True)
class CameraTrajectory:
    alpha: GaugeFixedSpline
    beta: GaugeFixedSpline
    log_f: UniformCubicSpline

    def __post_init__(self):
        if not isinstance(self.alpha, GaugeFixedSpline) or not isinstance(self.beta, GaugeFixedSpline):
            raise ValueError("both angle streams require exact function-value gauge elimination")
        if self.alpha.basis != self.beta.basis or self.alpha.basis != self.log_f.basis:
            raise ValueError("all streams must share the native uniform cubic basis")
        if self.alpha.reference_time != self.beta.reference_time:
            raise ValueError("pan/tilt angle gauges must share the fixed reference time")

    @property
    def domain(self):
        return self.log_f.basis.domain

    def at(self, time, derivative=0):
        return np.array([s.evaluate(time, derivative) for s in (self.alpha, self.beta, self.log_f)])


@dataclass(frozen=True)
class PhysicalCamera:
    head: FixedHead
    optics: ZoomOptics
    trajectory: CameraTrajectory

    def project(self, world, time) -> ProjectionResult:
        try:
            alpha, beta, log_f = self.trajectory.at(time)
            K, D, z = self.optics.at(log_f)
            return project(world, self.head.pose(alpha, beta, z), K, D)
        except CameraDomainError as exc:
            status = (ProjectionStatus.TIME_DOMAIN_INVALID
                      if not (self.trajectory.domain[0] <= time <= self.trajectory.domain[1])
                      else ProjectionStatus.INVALID_DOMAIN)
            return ProjectionResult(status, message=str(exc))

    def time_derivative(self, world, time):
        """Analytic fixed-world-point d(pixel)/dt through this same full model."""
        X = _array(world, (3,), "world point")
        alpha, beta, log_f = self.trajectory.at(time)
        alpha_dot, beta_dot, z_dot = self.trajectory.at(time, 1)
        K, D, z = self.optics.at(log_f)
        pose, R_dot, C_dot = self.head.pose_and_velocity(alpha, beta, z, alpha_dot, beta_dot, z_dot)
        project(X, pose, K, D).require_uv()
        Xc = pose.R @ (X-pose.C)
        Xc_dot = R_dot @ (X-pose.C) - pose.R @ C_dot
        xy = Xc[:2] / Xc[2]
        xy_dot = (Xc_dot[:2] - xy * Xc_dot[2]) / Xc[2]
        distorted, J = D.distort_with_jacobian(xy)
        r2 = float(xy @ xy)
        dz_distorted = xy * (self.optics.k1[1]*r2 + self.optics.k2[1]*r2*r2)
        distorted_dot = J @ xy_dot + dz_distorted * z_dot
        return (np.array([K.fx, K.fy]) * (distorted_dot + distorted*z_dot)
                + np.array(self.optics.principal_linear)*z_dot)

    def project_rs(self, world, frame_time, readout_time, image_height=1080, image_width=1920,
                   **kwargs):
        try:
            self.trajectory.log_f.basis.require_coverage(frame_time, frame_time, readout_time)
        except CameraDomainError as exc:
            return RSProjectionResult(ProjectionStatus.TIME_DOMAIN_INVALID, message=str(exc))
        return rolling_shutter_project(lambda t: self.project(world, t),
                                       lambda t: self.time_derivative(world, t),
                                       frame_time=frame_time, readout_time=readout_time,
                                       image_height=image_height, image_width=image_width, **kwargs)


def row_time(frame_time, signed_readout_time, row, image_height=1080):
    if not isinstance(image_height, int) or image_height < 2:
        raise ValueError("image_height must be an integer >=2")
    for name, value in (("frame_time", frame_time), ("readout", signed_readout_time), ("row", row)):
        _finite(value, name)
    return frame_time + signed_readout_time * (row/(image_height-1) - 0.5)


def implicit_jacobian(pi_theta, pi_t, signed_readout_time, image_height=1080,
                      explicit_time_theta=None, min_abs_denominator=1e-8, max_condition=1e8):
    """Correct RS derivative; fixed-time partials enter exactly once.

    explicit_time_theta holds dt/dtheta with the output row fixed.  For tau,
    that entry is v/(H-1)-1/2; a curve station has no explicit time term.
    The returned derivative includes the predicted row's time dependence.
    """
    row_time(0.0, signed_readout_time, 0.0, image_height)
    J = np.asarray(pi_theta, dtype=np.float64)
    vector_input = J.ndim == 1
    if vector_input:
        J = J[:, None]
    if J.ndim != 2 or J.shape[0] != 2 or not np.all(np.isfinite(J)):
        raise ValueError("pi_theta must have shape (2,n), or (2,) for one scalar")
    velocity = _array(pi_t, (2,), "pi_t")
    dt = np.zeros(J.shape[1]) if explicit_time_theta is None else np.asarray(explicit_time_theta, dtype=float)
    if dt.ndim == 0 and J.shape[1] == 1:
        dt = dt.reshape(1)
    if dt.shape != (J.shape[1],) or not np.all(np.isfinite(dt)):
        raise ValueError("explicit_time_theta must have one finite entry per parameter")
    w = np.array([0.0, signed_readout_time/(image_height-1)])
    F = np.eye(2) - np.outer(velocity, w)
    denominator = 1.0 - velocity[1]*w[1]
    condition = float(np.linalg.cond(F))
    if abs(denominator) < min_abs_denominator or not np.isfinite(condition) or condition > max_condition:
        raise SingularRSRoot(f"implicit row derivative is singular/ill-conditioned: det={denominator}, cond={condition}")
    result = np.linalg.solve(F, J + np.outer(velocity, dt))
    return result[:, 0] if vector_input else result


@dataclass(frozen=True)
class RSIntervalCertificate:
    """Caller-supplied proof obligation over the *entire* row-time interval.

    This asserts a continuous valid projection and an absolute vertical speed
    upper bound, with named derivation/provenance.  A few sampled derivatives
    do not satisfy this contract.  The reference does not synthesize proofs.
    """

    interval: tuple[float, float]
    vertical_speed_bound: float
    provenance: str
    continuous_projection_domain_valid: bool = True

    def __post_init__(self):
        a = _array(self.interval, (2,), "certificate interval")
        object.__setattr__(self, "interval", tuple(float(v) for v in a))
        if a[0] > a[1] or _finite(self.vertical_speed_bound, "speed bound") < 0:
            raise ValueError("invalid certificate interval or speed bound")
        if not self.provenance.strip():
            raise ValueError("a full-interval certificate needs derivation/provenance")


@dataclass(frozen=True)
class RSRoot:
    row: float
    time: float
    uv: np.ndarray
    equation_error_px: float
    denominator: float
    implicit_condition: float

    def __post_init__(self):
        object.__setattr__(self, "uv", _array(self.uv, (2,), "root uv"))


@dataclass(frozen=True)
class RSProjectionResult:
    status: ProjectionStatus
    uv: Optional[np.ndarray] = None
    roots: tuple[RSRoot, ...] = ()
    uniqueness_certified: bool = False
    search_complete_certified: bool = False
    message: str = ""

    def __post_init__(self):
        if self.uv is not None:
            object.__setattr__(self, "uv", _array(self.uv, (2,), "RS uv"))

    @property
    def valid(self):
        return self.status is ProjectionStatus.VALID

    def require_uv(self):
        if not self.valid or self.uv is None:
            raise CameraDomainError(f"{self.status.value}: {self.message}")
        return self.uv


def rolling_shutter_project(
    project_at_time: Callable[[float], ProjectionResult],
    project_time_derivative: Callable[[float], np.ndarray], *,
    frame_time: float, readout_time: float, image_height: int = 1080,
    image_width: Optional[int] = 1920,
    certificate: Optional[RSIntervalCertificate] = None,
    row_samples: int = 129, root_tolerance_px: float = 1e-9,
    min_abs_denominator: float = 1e-8, max_condition: float = 1e8,
) -> RSProjectionResult:
    """Predicted-row RS with bracketed Brent and explicit numerical statuses.

    Without a full-interval monotonicity certificate, this routine discovers
    roots (also checking derivative sign changes for tangencies) but returns
    UNIQUENESS_UNCERTIFIED for a single regular candidate.  Finite sampling
    cannot prove that additional roots or domain holes do not exist.  The
    candidate pixel remains available for diagnostics, never as accepted
    coverage.  Zero readout is exact and needs no external certificate.
    """
    row_time(frame_time, readout_time, 0.0, image_height)
    if image_width is not None and (not isinstance(image_width, int) or image_width < 1):
        raise ValueError("image_width must be positive or None")
    if not isinstance(row_samples, int) or row_samples < 3 or root_tolerance_px <= 0:
        raise ValueError("at least three row samples and positive root tolerance are required")
    left, right = 0.0, float(image_height-1)
    t0, t1 = sorted((row_time(frame_time, readout_time, left, image_height),
                     row_time(frame_time, readout_time, right, image_height)))
    certified = readout_time == 0.0
    if certificate is not None:
        if certificate.interval[0] > t0 or certificate.interval[1] < t1:
            raise ValueError("RS certificate does not cover the entire required time interval")
        certified = certified or (certificate.continuous_projection_domain_valid and
                                  abs(readout_time)*certificate.vertical_speed_bound/(image_height-1) < 1.0)
    cache = {}

    class _ProjectionFailure(Exception):
        def __init__(self, result):
            self.result = result

    def evaluate(v):
        v = float(v)
        if v not in cache:
            t = row_time(frame_time, readout_time, v, image_height)
            p = project_at_time(t)
            if not isinstance(p, ProjectionResult):
                raise TypeError("project_at_time must return ProjectionResult, preserving domain status")
            if not p.valid:
                raise _ProjectionFailure(p)
            velocity = _array(project_time_derivative(t), (2,), "projection time derivative")
            if certificate is not None and certificate.continuous_projection_domain_valid:
                if abs(velocity[1]) > certificate.vertical_speed_bound * (1.0 + 1e-12) + 1e-12:
                    raise ValueError("observed derivative contradicts the supplied RS certificate")
            w_y = readout_time/(image_height-1)
            denominator = 1.0 - velocity[1]*w_y
            F = np.eye(2)-np.outer(velocity, [0.0, w_y])
            cache[v] = (v-float(p.uv[1]), denominator, p, t, float(np.linalg.cond(F)))
        return cache[v]

    try:
        if readout_time == 0.0:
            p = project_at_time(frame_time)
            if not p.valid:
                return RSProjectionResult(p.status, message=p.message)
            candidate_rows = [float(p.uv[1])] if left <= p.uv[1] <= right else []
        else:
            grid = np.linspace(left, right, 2 if certified else row_samples)
            evaluations = [evaluate(v) for v in grid]
            candidate_rows = []
            for v, ev in zip(grid, evaluations):
                if abs(ev[0]) <= root_tolerance_px:
                    candidate_rows.append(float(v))
            for i in range(len(grid)-1):
                a, b = float(grid[i]), float(grid[i+1])
                fa, fb = evaluations[i][0], evaluations[i+1][0]
                if fa*fb < 0.0:
                    candidate_rows.append(brentq(lambda v: evaluate(v)[0], a, b,
                                                  xtol=root_tolerance_px, rtol=4*np.finfo(float).eps))
                if not certified:
                    da, db = evaluations[i][1], evaluations[i+1][1]
                    if da*db < 0.0:
                        extremum = brentq(lambda v: evaluate(v)[1], a, b,
                                          xtol=root_tolerance_px, rtol=4*np.finfo(float).eps)
                        if abs(evaluate(extremum)[0]) <= root_tolerance_px:
                            candidate_rows.append(extremum)
            candidate_rows.sort()
            merged = []
            for v in candidate_rows:
                if not merged or abs(v-merged[-1]) > max(8*root_tolerance_px, 1e-7):
                    merged.append(v)
            candidate_rows = merged
        if not candidate_rows:
            return RSProjectionResult(ProjectionStatus.NO_ROOT, uniqueness_certified=certified,
                                      search_complete_certified=certified,
                                      message="no in-image row root" if certified else
                                      "no row root discovered; finite search is not a nonexistence proof")
        roots = []
        for v in candidate_rows:
            h, denominator, p, t, condition = evaluate(v)
            if abs(h) > 4.0*root_tolerance_px:
                return RSProjectionResult(ProjectionStatus.ROOT_SEARCH_FAILED,
                                          message="root equation did not meet the frozen arithmetic tolerance")
            roots.append(RSRoot(v, t, p.uv, h, denominator, condition))
        roots = tuple(roots)
        if len(roots) > 1:
            return RSProjectionResult(ProjectionStatus.MULTIPLE_ROOTS, roots=roots,
                                      message="multiple relevant row roots; keep explicit v with equality AL")
        root = roots[0]
        if (abs(root.denominator) < min_abs_denominator or not np.isfinite(root.implicit_condition)
                or root.implicit_condition > max_condition):
            return RSProjectionResult(ProjectionStatus.SINGULAR_ROOT, uv=root.uv, roots=roots,
                                      uniqueness_certified=certified, search_complete_certified=certified,
                                      message="root exists but the implicit derivative is singular/ill-conditioned")
        if image_width is not None and not (0.0 <= root.uv[0] <= image_width-1):
            return RSProjectionResult(ProjectionStatus.OUTSIDE_IMAGE, uv=root.uv, roots=roots,
                                      uniqueness_certified=certified, search_complete_certified=certified,
                                      message="predicted horizontal coordinate lies outside the native image")
        status = ProjectionStatus.VALID if certified else ProjectionStatus.UNIQUENESS_UNCERTIFIED
        return RSProjectionResult(status, uv=root.uv, roots=roots, uniqueness_certified=certified,
                                  search_complete_certified=certified,
                                  message="" if certified else "one regular sampled root; continuous uniqueness is not certified")
    except _ProjectionFailure as exc:
        return RSProjectionResult(exc.result.status,
                                  message="row-time evaluation failed: "+exc.result.message)
    except (CameraDomainError, FloatingPointError) as exc:
        return RSProjectionResult(ProjectionStatus.INVALID_DOMAIN, message=str(exc))
    except RuntimeError as exc:
        return RSProjectionResult(ProjectionStatus.ROOT_SEARCH_FAILED, message=str(exc))


def explicit_rs_equality(predicted_row: float, instantaneous_projection: ProjectionResult) -> float:
    """h_RS=v-pi_y(X,t(v)); a recoverable equality, never a fabricated residual."""
    return _finite(predicted_row, "explicit row") - float(instantaneous_projection.require_uv()[1])


# IMPLEMENTED: exact shared orthogonal head, low-dimensional Brown/zoom,
# uniform native cubic controls, exact angle-function gauges, analytic time
# velocity, strict-domain instantaneous projection, protected row-root search,
# explicit RS equality, certified/uncertified statuses and implicit Jacobians.
# NOT IMPLEMENTED: interval propagation to generate RS certificates, nonorthogonal
# head candidate, OIS/focus/exposure streams, crop/deinterlace output alternatives,
# GPU/vectorized production kernels, Ceres manifolds, or a completed data fit.
# Unsupported candidate flags raise; no automatic fallback changes the model.
