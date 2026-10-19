"""Finite, shared scene geometry and distinct observation semantics.

The rink uses Z=0, a named center origin, +X along its long direction, and
right-handed coordinates.  Use length=1 for the monocular scale gauge unless
an independent measured scale is explicitly declared by the caller.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Mapping, Optional, Protocol

import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import brentq


class GeometryDomainError(ValueError):
    """Illegal material station or physical geometry domain."""


class UnsupportedGeometryModel(NotImplementedError):
    """An explicitly requested geometric candidate is not implemented."""


def _array(value, shape, name):
    a = np.array(value, dtype=np.float64, copy=True)
    if a.shape != shape or not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be a finite array of shape {shape}")
    a.setflags(write=False)
    return a


def _scalar(value, name):
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _station(value, interval, name="material station"):
    value = _scalar(value, name)
    if not (interval[0] <= value <= interval[1]):
        raise GeometryDomainError(f"{name} {value} outside finite interval {interval}")
    return value


def _interval(interval, name):
    result = tuple(_scalar(v, name) for v in interval)
    if len(result) != 2 or result[0] > result[1]:
        raise ValueError(f"{name} must be an ordered pair")
    return result


class FiniteCurve(Protocol):
    @property
    def station_interval(self) -> tuple[float, float]: ...

    def point(self, station: float) -> np.ndarray: ...

    def derivative(self, station: float) -> np.ndarray: ...


@dataclass(frozen=True)
class FiniteLine:
    A: np.ndarray
    B: np.ndarray

    def __post_init__(self):
        object.__setattr__(self, "A", _array(self.A, (3,), "line A"))
        object.__setattr__(self, "B", _array(self.B, (3,), "line B"))
        if np.linalg.norm(self.B-self.A) <= 0:
            raise ValueError("finite line endpoints must differ")

    @property
    def station_interval(self):
        return (0.0, 1.0)

    def point(self, station):
        s = _station(station, self.station_interval)
        return (1.0-s)*self.A + s*self.B

    def derivative(self, station):
        _station(station, self.station_interval)
        return self.B-self.A

    def closest_station(self, world):
        """Exact constrained Euclidean minimizer, including active endpoints.

        Selecting an endpoint is the mathematics of finite distance.  An
        illegal input material station to point() is still rejected, not
        silently moved to an endpoint.  Image-metric S is optimized upstream.
        """
        X = _array(world, (3,), "query point")
        d = self.B-self.A
        unconstrained = float((X-self.A) @ d / (d @ d))
        if unconstrained <= 0:
            return 0.0
        if unconstrained >= 1:
            return 1.0
        return unconstrained


@dataclass(frozen=True)
class CircleArc:
    center_xy: np.ndarray
    radius: float
    angle_start: float = 0.0
    angle_end: float = 2*np.pi

    def __post_init__(self):
        object.__setattr__(self, "center_xy", _array(self.center_xy, (2,), "circle center"))
        for name in ("radius", "angle_start", "angle_end"):
            object.__setattr__(self, name, _scalar(getattr(self, name), name))
        if self.radius <= 0 or not (0 < self.angle_end-self.angle_start <= 2*np.pi+1e-12):
            raise ValueError("circle radius must be positive and finite arc span in (0,2*pi]")

    @property
    def station_interval(self):
        return (self.angle_start, self.angle_end)

    @property
    def closed(self):
        return abs(self.angle_end-self.angle_start-2*np.pi) <= 1e-12

    def point(self, station):
        phi = _station(station, self.station_interval, "unwrapped angle")
        return np.array([self.center_xy[0]+self.radius*np.cos(phi),
                         self.center_xy[1]+self.radius*np.sin(phi), 0.0])

    def derivative(self, station):
        phi = _station(station, self.station_interval, "unwrapped angle")
        return self.radius*np.array([-np.sin(phi), np.cos(phi), 0.0])


@dataclass(frozen=True)
class RoundedRink:
    """One closed convex line/ellipse boundary, counterclockwise from bottom.

    Four corners share rx,ry.  p=2 is the implemented low-complexity candidate;
    p!=2 is rejected until stable endpoint formulas and derivatives exist.
    s in [0,8] is an explicitly *piecewise* station, one unit per segment,
    not arclength/meters.  Unit tangents are continuous at all eight joins;
    the parameter speed can jump, so we claim geometric tangency, not C1 in s.
    """

    length: float = 1.0
    width: float = 0.5
    radius_x: float = 0.12
    radius_y: float = 0.12
    exponent: float = 2.0
    origin_name: str = "rink_center"

    def __post_init__(self):
        for name in ("length", "width", "radius_x", "radius_y", "exponent"):
            object.__setattr__(self, name, _scalar(getattr(self, name), name))
        if self.exponent != 2.0:
            raise UnsupportedGeometryModel("only ellipse corners (p=2) are implemented; no superellipse fallback")
        if self.origin_name != "rink_center":
            raise UnsupportedGeometryModel("this chart fixes the named origin at rink_center")
        if not (self.length > 0 and self.width > 0 and
                0 < self.radius_x < self.length/2 and 0 < self.radius_y < self.width/2):
            raise GeometryDomainError("positive side lengths/radii with nonzero straight segments are required")

    @property
    def station_interval(self):
        return (0.0, 8.0)

    @property
    def closed(self):
        return True

    def _segment(self, station):
        s = _station(station, self.station_interval, "rink station")
        if s == 8.0:
            return 7, 1.0
        i = int(np.floor(s))
        return i, s-i

    def _point_and_derivative(self, station):
        i, u = self._segment(station)
        hx, hy, rx, ry = self.length/2, self.width/2, self.radius_x, self.radius_y
        if i == 0:
            point, tangent = (-hx+rx+u*(2*hx-2*rx), -hy), (2*hx-2*rx, 0.0)
        elif i == 2:
            point, tangent = (hx, -hy+ry+u*(2*hy-2*ry)), (0.0, 2*hy-2*ry)
        elif i == 4:
            point, tangent = (hx-rx-u*(2*hx-2*rx), hy), (-2*hx+2*rx, 0.0)
        elif i == 6:
            point, tangent = (-hx, hy-ry-u*(2*hy-2*ry)), (0.0, -2*hy+2*ry)
        else:
            corner = (i-1)//2
            centers = ((hx-rx, -hy+ry), (hx-rx, hy-ry), (-hx+rx, hy-ry), (-hx+rx, -hy+ry))
            angle = -np.pi/2 + corner*np.pi/2 + u*np.pi/2
            cx, cy = centers[corner]
            point = (cx+rx*np.cos(angle), cy+ry*np.sin(angle))
            tangent = (-rx*np.sin(angle)*np.pi/2, ry*np.cos(angle)*np.pi/2)
        return np.array([*point, 0.0]), np.array([*tangent, 0.0])

    def point(self, station):
        return self._point_and_derivative(station)[0]

    def derivative(self, station):
        return self._point_and_derivative(station)[1]

    def unit_tangent(self, station):
        d = self.derivative(station)
        return d/np.linalg.norm(d)

    def outward_normal(self, station):
        t = self.unit_tangent(station)
        return np.array([t[1], -t[0], 0.0])

    def point_unwrapped(self, station):
        """Explicit periodic chart for one continuous material across the seam."""
        s = _scalar(station, "unwrapped boundary station")
        return self.point(s % 8.0)

    def derivative_unwrapped(self, station):
        return self.derivative(_scalar(station, "unwrapped boundary station") % 8.0)


@dataclass(frozen=True)
class BoardSurface:
    boundary: RoundedRink
    height: float
    height_model: str = "constant_shared"

    def __post_init__(self):
        object.__setattr__(self, "height", _scalar(self.height, "board height"))
        if self.height <= 0:
            raise GeometryDomainError("board height must be positive")
        if self.height_model != "constant_shared":
            raise UnsupportedGeometryModel("panel height/tilt variation is not implemented")

    def point(self, unwrapped_station, height):
        h = _station(height, (0.0, self.height), "surface height")
        return self.boundary.point_unwrapped(unwrapped_station)+np.array([0.0, 0.0, h])


@dataclass(frozen=True)
class BoardSplineMaterial:
    """One static clamped cubic logo boundary directly in board (s,h).

    Controls are shared across all frames; there is no frame argument, H8,
    free artwork transform, or per-frame shape.  panel_station_interval must
    come from physical support evidence, and may be unwrapped across s=8.
    Closed components explicitly repeat their first control at the end.
    Different connected components/holes receive separate shared curve IDs.
    """

    material_id: str
    board: BoardSurface
    controls_sh: np.ndarray
    panel_station_interval: tuple[float, float]
    support_provenance: str
    closed: bool = False
    _spline: BSpline = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        controls = np.array(self.controls_sh, dtype=np.float64, copy=True)
        if controls.ndim != 2 or controls.shape[1] != 2 or controls.shape[0] < 4 or not np.all(np.isfinite(controls)):
            raise ValueError("static board cubic requires at least four finite (s,h) controls")
        interval = _interval(self.panel_station_interval, "physical panel station interval")
        if interval[0] == interval[1]:
            raise GeometryDomainError("physical panel support must have positive extent")
        if not self.material_id.strip() or not self.support_provenance.strip():
            raise ValueError("material ID and physical support provenance are required")
        if np.any(controls[:, 0] < interval[0]) or np.any(controls[:, 0] > interval[1]):
            raise GeometryDomainError("shape controls exceed the confirmed unwrapped panel interval")
        if np.any(controls[:, 1] < 0) or np.any(controls[:, 1] > self.board.height):
            raise GeometryDomainError("shape controls exceed actual board height")
        if self.closed and not np.array_equal(controls[0], controls[-1]):
            raise ValueError("a closed clamped curve must explicitly share its endpoint control")
        n = len(controls)
        knots = np.r_[np.zeros(4), np.arange(1, n-3)/(n-3), np.ones(4)]
        controls.setflags(write=False)
        object.__setattr__(self, "controls_sh", controls)
        object.__setattr__(self, "panel_station_interval", interval)
        object.__setattr__(self, "_spline", BSpline(knots, controls, 3, extrapolate=False))

    @property
    def station_interval(self):
        return (0.0, 1.0)

    def surface_coordinates(self, station):
        u = _station(station, self.station_interval, "logo curve station")
        return np.asarray(self._spline(u), dtype=np.float64)

    def point(self, station):
        s, h = self.surface_coordinates(station)
        return self.board.point(s, h)

    def derivative(self, station):
        u = _station(station, self.station_interval, "logo curve station")
        s, _ = self.surface_coordinates(u)
        ds, dh = self._spline(u, nu=1)
        return self.board.boundary.derivative_unwrapped(s)*ds+np.array([0.0, 0.0, dh])


@dataclass(frozen=True)
class PhysicalPoint:
    identity: str
    world: np.ndarray

    def __post_init__(self):
        if not self.identity.strip():
            raise ValueError("a confirmed physical corner/endpoint needs its stable physical identity")
        object.__setattr__(self, "world", _array(self.world, (3,), "physical world point"))


def _native_prediction(predict, world):
    result = predict(world)
    # Both instantaneous and RS result types expose require_uv.  Keeping this
    # protocol avoids importing camera and preserves every invalid status.
    if hasattr(result, "require_uv"):
        return _array(result.require_uv(), (2,), "native prediction")
    return _array(result, (2,), "native prediction")


def physical_point_residual(predict, point: PhysicalPoint, observed_uv):
    """True physical identity: no free material station exists in this API."""
    return _native_prediction(predict, point.world)-_array(observed_uv, (2,), "observed native UV")


def ordinary_curve_residual(predict, curve: FiniteCurve, station, observed_uv,
                            allowed_interval: Optional[tuple[float, float]] = None):
    """Full native 2D error with an explicit finite S optimized upstream.

    Cropped/occluded run ends remain free stations.  Neither repeated source
    IDs nor array indices create persistent world point correspondences.
    """
    interval = curve.station_interval if allowed_interval is None else _interval(allowed_interval, "allowed interval")
    if interval[0] < curve.station_interval[0] or interval[1] > curve.station_interval[1]:
        raise ValueError("allowed support must lie within the actual finite material")
    s = _station(station, interval)
    return _native_prediction(predict, curve.point(s))-_array(observed_uv, (2,), "observed native UV")


@dataclass(frozen=True)
class ProfileQuery:
    """Actual measured query frame; do not manufacture one from ordinary UV.

    Prediction must equal observed_uv + alpha*normal for alpha in the finite
    query interval.  epsilon_n is a frozen native-pixel budget; sigma_n is a
    separately frozen statistical scale with its own provenance upstream.
    """

    observed_uv: np.ndarray
    tangent: np.ndarray
    normal: np.ndarray
    alpha_interval: tuple[float, float]
    sigma_n: float
    epsilon_n: float
    query_provenance: str

    def __post_init__(self):
        for name in ("observed_uv", "tangent", "normal"):
            object.__setattr__(self, name, _array(getattr(self, name), (2,), name))
        if (abs(self.tangent @ self.tangent-1) > 1e-10 or
                abs(self.normal @ self.normal-1) > 1e-10 or
                abs(self.tangent @ self.normal) > 1e-10):
            raise ValueError("profile tangent/normal must be the measured orthonormal query frame")
        object.__setattr__(self, "alpha_interval", _interval(self.alpha_interval, "query alpha interval"))
        for name in ("sigma_n", "epsilon_n"):
            object.__setattr__(self, name, _scalar(getattr(self, name), name))
            if getattr(self, name) <= 0:
                raise ValueError("profile statistical scale and pixel budget must be positive")
        if not self.query_provenance.strip():
            raise ValueError("real profile metadata provenance is required")


@dataclass(frozen=True)
class ProfileEvaluation:
    equality_xy: np.ndarray
    tangent_equality_px: float
    native_normal_error_px: float
    statistical_residual: float
    budget_q: float
    query_inequalities: np.ndarray
    query_interval_valid: bool


def profile_auxiliary(predict, curve: FiniteCurve, station, alpha, query: ProfileQuery,
                      allowed_interval=None) -> ProfileEvaluation:
    """Explicit (s,alpha) recovery: equality enters AL even without a root.

    alpha outside its finite query interval yields positive support
    inequalities; it is never hidden by budget inflation or zeroed residuals.
    The material station itself remains within its declared curve domain.
    """
    e = ordinary_curve_residual(predict, curve, station, query.observed_uv, allowed_interval)
    alpha = _scalar(alpha, "query alpha")
    en = float(query.normal @ e)
    inequalities = np.array([query.alpha_interval[0]-alpha, alpha-query.alpha_interval[1]])
    return ProfileEvaluation(e-alpha*query.normal, float(query.tangent @ e), en,
                             en/query.sigma_n, (en/query.epsilon_n)**2,
                             inequalities, bool(np.all(inequalities <= 0)))


class ProfileStatus(str, Enum):
    UNIQUE_CANDIDATE_UNCERTIFIED = "unique_candidate_uncertified"
    NO_INTERSECTION = "no_intersection"
    MULTIPLE_INTERSECTIONS = "multiple_intersections"
    INVALID_PROJECTION = "invalid_projection"


@dataclass(frozen=True)
class ProfileIntersection:
    station: float
    alpha: float
    native_normal_error_px: float
    tangent_equality_px: float


@dataclass(frozen=True)
class ProfileIntersectionResult:
    status: ProfileStatus
    roots: tuple[ProfileIntersection, ...] = ()
    search_complete_certified: bool = False
    message: str = ""


def find_profile_intersections(predict, curve: FiniteCurve, query: ProfileQuery,
                               allowed_interval=None, samples=129, tolerance_px=1e-9):
    """Diagnostic finite material/query intersection search; retains branches.

    This sampled reference does not certify all intersections or tangencies.
    A production elimination requires a branch-local nonsingularity check and
    certified relevant root; otherwise retain explicit (s,alpha) and AL.
    """
    interval = curve.station_interval if allowed_interval is None else _interval(allowed_interval, "support")
    if interval[0] < curve.station_interval[0] or interval[1] > curve.station_interval[1]:
        raise ValueError("profile support exceeds actual finite material")
    if not isinstance(samples, int) or samples < 3 or tolerance_px <= 0:
        raise ValueError("invalid profile root-search settings")
    def evaluate(s):
        e = ordinary_curve_residual(predict, curve, s, query.observed_uv, interval)
        return float(query.tangent @ e), float(query.normal @ e)
    try:
        grid = np.linspace(*interval, samples)
        values = [evaluate(s)[0] for s in grid]
        candidates = [float(s) for s, h in zip(grid, values) if abs(h) <= tolerance_px]
        for i in range(len(grid)-1):
            if values[i]*values[i+1] < 0:
                candidates.append(brentq(lambda s: evaluate(s)[0], grid[i], grid[i+1], xtol=1e-13))
        roots = []
        for s in sorted(candidates):
            h, alpha = evaluate(s)
            if not (query.alpha_interval[0] <= alpha <= query.alpha_interval[1]):
                continue
            if not roots or abs(s-roots[-1].station) > 1e-8:
                roots.append(ProfileIntersection(s, alpha, alpha, h))
        if not roots:
            status = ProfileStatus.NO_INTERSECTION
        elif len(roots) > 1:
            status = ProfileStatus.MULTIPLE_INTERSECTIONS
        else:
            status = ProfileStatus.UNIQUE_CANDIDATE_UNCERTIFIED
        return ProfileIntersectionResult(status, tuple(roots), False,
                                         "sampled branch candidates only; use explicit equality until support is restored/certified")
    except ValueError as exc:
        # A projection result's require_uv raises ValueError on domain failure.
        return ProfileIntersectionResult(ProfileStatus.INVALID_PROJECTION, message=str(exc))


def verified_order_inequalities(stations, *, verified: bool, direction=1):
    """Non-strict verified order: g_j=direction*(s_j-s_{j+1})<=0.

    Closed-run stations are already unwrapped by an explicitly chosen seam and
    direction branch.  Equal stations are legal.  Unverified order is rejected.
    """
    if not verified:
        raise ValueError("array order is not trace evidence; verified connectivity/order is required")
    s = np.asarray(stations, dtype=np.float64)
    if s.ndim != 1 or not np.all(np.isfinite(s)) or direction not in (-1, 1):
        raise ValueError("finite station vector and an explicit direction branch are required")
    return direction*(s[:-1]-s[1:])


# IMPLEMENTED: finite line/circle, one closed convex G1 line/ellipse rink,
# continuous unwrapped board surface and shared static cubic logo contours,
# fixed physical point / ordinary finite 2D / actual profile semantics,
# explicit profile equality recovery, diagnostic branch search, verified order.
# NOT IMPLEMENTED: p!=2 superellipse or curvature-transition corners, panel
# tilt/height fields, arbitrary static paint corrections, exact certified
# profile-root isolation, visibility/occlusion inference, inverse ray casting,
# adaptive rendered tessellation, or per-frame/per-logo homographies.
# Unsupported geometric candidate flags raise; no silent model downgrade.
