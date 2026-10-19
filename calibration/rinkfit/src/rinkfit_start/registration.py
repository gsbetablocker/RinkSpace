"""Finite same-material planar registration used only for S1 initialization.

H_atlas_to_native always maps a plane atlas into native integer-centre pixels.
G_i_to_j always maps native pixels in frame i into native pixels in frame j.
No source ID, array position, image crop, or missing metadata creates a physical
point correspondence or a connected curve in this module.  This is a numerical
initializer and local comparator, not the final Brown/rolling-shutter camera.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from numpy.polynomial import polynomial as poly
from scipy.special import logsumexp


class RegistrationError(ValueError):
    """Invalid support, projective chart, correspondence, or numerical domain."""


class SamplingLimitError(RegistrationError):
    """A declared finite sampling budget could not certify the requested fill."""


def _immutable_array(value: Any, dtype: Any = np.float64) -> np.ndarray:
    """Copy into a bytes-backed array: write flags cannot be re-enabled."""
    a = np.ascontiguousarray(value, dtype=dtype)
    return np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _immutable_array(value, value.dtype)
    if isinstance(value, Mapping):
        return MappingProxyType({str(k): _deep_freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(v) for v in value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    raise TypeError(f"Unsupported mutable or nonserializable snapshot value: {type(value)}")


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    return value


def _xy(value: Any, name: str, minimum: int = 1) -> np.ndarray:
    a = np.asarray(value, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 2 or len(a) < minimum or not np.isfinite(a).all():
        raise RegistrationError(f"{name} must be finite N x 2 with N >= {minimum}")
    return a


def normalize_homography(matrix: Any) -> np.ndarray:
    """Scale by Frobenius norm, never force h33=1; reject a singular H."""
    h = np.asarray(matrix, dtype=np.float64)
    if h.shape != (3, 3) or not np.isfinite(h).all():
        raise RegistrationError("A homography must be a finite 3 x 3 matrix")
    norm = np.linalg.norm(h)
    if norm == 0:
        raise RegistrationError("Zero homography")
    h = h / norm
    sv = np.linalg.svd(h, compute_uv=False)
    if sv[-1] <= sv[0] * 1e-14:
        raise RegistrationError("Singular or numerically unusable homography")
    pivot = h.ravel()[int(np.argmax(np.abs(h)))]
    return h if pivot > 0 else -h


def project_points(matrix: Any, points: Any, denominator_rtol: float = 1e-11) -> np.ndarray:
    """Project only the declared support; no clipping or visibility deletion."""
    h = normalize_homography(matrix)
    x = _xy(points, "points")
    xh = np.column_stack((x, np.ones(len(x))))
    p = xh @ h.T
    threshold = denominator_rtol * np.linalg.norm(h[2]) * np.linalg.norm(xh, axis=1)
    if np.any(np.abs(p[:, 2]) <= threshold):
        raise RegistrationError("Projective denominator vanishes on used support")
    out = p[:, :2] / p[:, 2, None]
    if not np.isfinite(out).all():
        raise RegistrationError("Nonfinite native projection")
    return out


def pixel_center_transform(scale: float) -> np.ndarray:
    """T_s implements u_s=s*(u+0.5)-0.5, with unchanged aspect ratio."""
    if not np.isfinite(scale) or scale <= 0:
        raise RegistrationError("Pyramid scale must be positive and finite")
    d = (float(scale) - 1.0) / 2.0
    return np.array([[scale, 0.0, d], [0.0, scale, d], [0.0, 0.0, 1.0]])


def scale_atlas_to_native(H_atlas_to_native: Any, scale: float) -> np.ndarray:
    return normalize_homography(pixel_center_transform(scale) @ H_atlas_to_native)


def scale_frame_i_to_j(G_i_to_j: Any, source_scale: float, target_scale: float) -> np.ndarray:
    return normalize_homography(
        pixel_center_transform(target_scale) @ G_i_to_j @ np.linalg.inv(pixel_center_transform(source_scale))
    )


def _hartley(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = points.mean(axis=0)
    radius = np.linalg.norm(points - center, axis=1).mean()
    if radius <= np.finfo(float).eps * max(1.0, float(np.linalg.norm(center))):
        raise RegistrationError("Collapsed point configuration")
    scale = math.sqrt(2.0) / radius
    t = np.array([[scale, 0, -scale * center[0]], [0, scale, -scale * center[1]], [0, 0, 1.0]])
    return (points - center) * scale, t


def _dlt_chart_or_entities(source: Any, target: Any, weights: Any = None, rank_rtol: float = 1e-10):
    a = _xy(source, "source", 4)
    b = _xy(target, "target", 4)
    if a.shape != b.shape:
        raise RegistrationError("Point arrays differ in size")
    an, ta = _hartley(a)
    bn, tb = _hartley(b)
    x, y = an.T
    u, v = bn.T
    one, zero = np.ones(len(a)), np.zeros(len(a))
    row1 = np.column_stack((-x, -y, -one, zero, zero, zero, u*x, u*y, u))
    row2 = np.column_stack((zero, zero, zero, -x, -y, -one, v*x, v*y, v))
    design = np.empty((2 * len(a), 9))
    design[0::2], design[1::2] = row1, row2
    if weights is not None:
        w = np.asarray(weights, float)
        if w.shape != (len(a),) or np.any(w <= 0) or not np.isfinite(w).all():
            raise RegistrationError("DLT weights must be finite and strictly positive")
        design *= np.repeat(np.sqrt(w / w.mean()), 2)[:, None]
    # Four pairs have an 8x9 design and need the ninth right-null vector;
    # larger designs must not allocate an unnecessary (2N)x(2N) U matrix.
    _, sv, vh = np.linalg.svd(design, full_matrices=design.shape[0] < design.shape[1])
    rank = int(np.count_nonzero(sv > sv[0] * rank_rtol))
    if rank < 8:
        raise RegistrationError(f"Entity/virtual-point DLT is rank deficient ({rank} < 8)")
    h = np.linalg.inv(tb) @ vh[-1].reshape(3, 3) @ ta
    h = normalize_homography(h)
    project_points(h, a)
    return h, rank, sv


@dataclass(frozen=True)
class DLTSeed:
    matrix: np.ndarray
    direction: str
    design_rank: int
    singular_values: np.ndarray
    entity_evidence: tuple[str, ...]

    def __post_init__(self):
        object.__setattr__(self, "matrix", _immutable_array(self.matrix))
        object.__setattr__(self, "singular_values", _immutable_array(self.singular_values))


def normalized_dlt(source_points: Any, target_points: Any, *, entity_evidence: str | Sequence[str],
                   direction: str = "atlas_to_native", weights: Any = None) -> DLTSeed:
    """Seed only from explicitly confirmed entities, never sampling-index pairs.

    ``entity_evidence`` must identify the actual physical correspondence review.
    Repeated source_point_id alone does not satisfy that contract.
    """
    source = _xy(source_points, "source_points", 4)
    evidence = (entity_evidence,) * len(source) if isinstance(entity_evidence, str) else tuple(entity_evidence)
    if len(evidence) != len(source) or not all(isinstance(s, str) and s.strip() for s in evidence):
        raise RegistrationError("Every entity correspondence needs nonempty identity evidence")
    if direction not in ("atlas_to_native", "native_i_to_j"):
        raise RegistrationError("Declare direction as atlas_to_native or native_i_to_j")
    h, rank, sv = _dlt_chart_or_entities(source, target_points, weights)
    return DLTSeed(h, direction, rank, sv, evidence)


@dataclass(frozen=True)
class LocalHomographyChart:
    """Eight native-control displacements; the four controls are not data."""
    atlas_controls: np.ndarray
    native_controls_at_origin: np.ndarray
    parameter_scale_native_px: float

    def __post_init__(self):
        a = _xy(self.atlas_controls, "atlas_controls", 4)
        b = _xy(self.native_controls_at_origin, "native_controls", 4)
        if a.shape != (4, 2) or b.shape != (4, 2):
            raise RegistrationError("A local chart has exactly four virtual controls")
        _dlt_chart_or_entities(a, b)
        if self.parameter_scale_native_px <= 0 or not np.isfinite(self.parameter_scale_native_px):
            raise RegistrationError("Invalid local parameter scale")
        object.__setattr__(self, "atlas_controls", _immutable_array(a))
        object.__setattr__(self, "native_controls_at_origin", _immutable_array(b))

    @classmethod
    def from_homography(cls, H_atlas_to_native: Any, atlas_support: Any,
                        atlas_controls: Any = None, parameter_scale_native_px: float | None = None):
        support = _xy(atlas_support, "atlas_support")
        if atlas_controls is None:
            lo, hi = support.min(axis=0), support.max(axis=0)
            center = (lo + hi) / 2
            # This is a coordinate chart, never an added geometric constraint.
            span = np.maximum(hi - lo, max(1e-6, float(np.max(hi-lo))) * 0.02)
            if not np.any(hi > lo):
                span = np.ones(2)
            lo, hi = center - .5*span, center + .5*span
            atlas_controls = np.array([lo, [hi[0], lo[1]], hi, [lo[0], hi[1]]])
        native = project_points(H_atlas_to_native, atlas_controls)
        if parameter_scale_native_px is None:
            projected = project_points(H_atlas_to_native, support)
            parameter_scale_native_px = max(1.0, float(np.sqrt(np.mean(np.sum((projected-projected.mean(0))**2, axis=1)))))
        return cls(atlas_controls, native, float(parameter_scale_native_px))

    def matrix(self, parameters: Any) -> np.ndarray:
        p = np.asarray(parameters, float)
        if p.shape != (8,) or not np.isfinite(p).all():
            raise RegistrationError("A local H chart has eight finite parameters")
        native = self.native_controls_at_origin + self.parameter_scale_native_px * p.reshape(4, 2)
        return _dlt_chart_or_entities(self.atlas_controls, native)[0]

    def parameters(self, H_atlas_to_native: Any) -> np.ndarray:
        return ((project_points(H_atlas_to_native, self.atlas_controls) - self.native_controls_at_origin)
                / self.parameter_scale_native_px).ravel()


def _bezier_restrict(control: np.ndarray, a: float, b: float) -> np.ndarray:
    def split(p: np.ndarray, t: float):
        p01 = (1-t)*p[:-1] + t*p[1:]
        p02 = (1-t)*p01[:-1] + t*p01[1:]
        p03 = (1-t)*p02[0] + t*p02[1]
        return np.array([p[0], p01[0], p02[0], p03]), np.array([p03, p02[1], p01[2], p[3]])
    if a == 0 and b == 1:
        return control
    left, _ = split(control, b)
    if a == 0:
        return left
    _, middle = split(left, a/b)
    return middle


def _angles_between(theta0: float, theta1: float, phase: float, period: float) -> list[float]:
    lo, hi = sorted((theta0, theta1))
    first = math.ceil((lo - phase) / period - 1e-12)
    last = math.floor((hi - phase) / period + 1e-12)
    return [phase + k * period for k in range(first, last + 1)]


@dataclass(frozen=True)
class MaterialSupport:
    """One finite material branch in one shared atlas, with explicit provenance.

    The station is normalized to [0,1] on the declared primitive.  ``points``
    remains a disconnected finite set; ``polyline`` requires verified order.
    An interval endpoint is a support bound, not a fixed observed world point.
    """
    material_id: str
    kind: str
    coordinates: np.ndarray
    provenance: str
    station_interval: tuple[float, float] = (0.0, 1.0)
    branch_id: str = "declared_branch"
    connectivity_evidence: str | None = None

    def __post_init__(self):
        object.__setattr__(self, "material_id", str(self.material_id))
        if not self.provenance or not self.provenance.strip():
            raise RegistrationError("A material support needs provenance")
        allowed = {"point", "points", "segment", "polyline", "arc", "cubic_bezier"}
        if self.kind not in allowed:
            raise RegistrationError(f"Unsupported material kind {self.kind}")
        c = np.asarray(self.coordinates, float)
        if self.kind == "arc":
            if c.shape != (5,) or not np.isfinite(c).all() or c[2] <= 0 or not (0 < abs(c[4]-c[3]) <= 2*math.pi + 1e-10):
                raise RegistrationError("Arc coordinates are [cx,cy,r,theta0,theta1], with 0 < span <= 2pi")
        else:
            _xy(c, "material coordinates")
            expected = {"point": 1, "segment": 2, "cubic_bezier": 4}.get(self.kind)
            if expected and len(c) != expected:
                raise RegistrationError(f"{self.kind} requires {expected} control points")
            if self.kind in ("segment", "polyline"):
                if len(c) < 2 or np.any(np.linalg.norm(np.diff(c, axis=0), axis=1) <= 1e-14):
                    raise RegistrationError("Zero-length segment in finite support")
            if self.kind == "polyline" and not self.connectivity_evidence:
                raise RegistrationError("Polyline creation requires verified connectivity/order evidence")
        a, b = (float(v) for v in self.station_interval)
        if not 0 <= a < b <= 1:
            raise RegistrationError("Legal finite station interval must satisfy 0 <= a < b <= 1")
        if self.kind in ("point", "points") and (a, b) != (0., 1.):
            raise RegistrationError("A disconnected point support uses its explicit points, not a curve interval")
        object.__setattr__(self, "station_interval", (a, b))
        object.__setattr__(self, "coordinates", _immutable_array(c))

    @classmethod
    def segment(cls, material_id: Any, start: Any, end: Any, *, provenance: str, **kw):
        return cls(str(material_id), "segment", np.array([start, end]), provenance, **kw)

    @classmethod
    def point(cls, material_id: Any, point: Any, *, provenance: str, **kw):
        return cls(str(material_id), "point", np.asarray(point).reshape(1, 2), provenance, **kw)

    @classmethod
    def points(cls, material_id: Any, points: Any, *, provenance: str, **kw):
        return cls(str(material_id), "points", points, provenance, **kw)

    @classmethod
    def polyline(cls, material_id: Any, points: Any, *, provenance: str, connectivity_evidence: str, **kw):
        return cls(str(material_id), "polyline", points, provenance, connectivity_evidence=connectivity_evidence, **kw)

    @classmethod
    def arc(cls, material_id: Any, center: Any, radius: float, theta0: float, theta1: float,
            *, provenance: str, **kw):
        return cls(str(material_id), "arc", [*center, radius, theta0, theta1], provenance, **kw)

    @classmethod
    def cubic_bezier(cls, material_id: Any, controls: Any, *, provenance: str, **kw):
        return cls(str(material_id), "cubic_bezier", controls, provenance, **kw)

    @property
    def discrete(self) -> bool:
        return self.kind in ("point", "points")

    def break_stations(self) -> np.ndarray:
        if self.kind == "polyline":
            d = np.linalg.norm(np.diff(self.coordinates, axis=0), axis=1)
            all_s = np.r_[0., np.cumsum(d) / d.sum()]
        elif self.discrete:
            return np.linspace(0., 1., len(self.coordinates)) if len(self.coordinates) > 1 else np.array([0.])
        else:
            all_s = np.array([0., 1.])
        a, b = self.station_interval
        return np.unique(np.r_[a, all_s[(all_s > a) & (all_s < b)], b])

    def atlas_points(self, stations: Any) -> np.ndarray:
        s = np.atleast_1d(np.asarray(stations, float))
        a, b = self.station_interval
        if not np.isfinite(s).all() or np.any(s < a-1e-12) or np.any(s > b+1e-12):
            raise RegistrationError("Station outside the finite legal material interval")
        if self.discrete:
            if len(self.coordinates) == 1:
                return np.repeat(self.coordinates, len(s), axis=0)
            idx = s * (len(self.coordinates)-1)
            if np.any(np.abs(idx - np.rint(idx)) > 1e-9):
                raise RegistrationError("A point-set station must identify an actual discrete point")
            return self.coordinates[np.rint(idx).astype(int)]
        if self.kind == "segment":
            return self.coordinates[0] + s[:, None] * (self.coordinates[1] - self.coordinates[0])
        if self.kind == "polyline":
            d = np.linalg.norm(np.diff(self.coordinates, axis=0), axis=1)
            knots = np.r_[0., np.cumsum(d) / d.sum()]
            return np.column_stack([np.interp(s, knots, self.coordinates[:, k]) for k in (0, 1)])
        if self.kind == "arc":
            cx, cy, r, t0, t1 = self.coordinates
            theta = t0 + (t1-t0)*s
            return np.column_stack((cx+r*np.cos(theta), cy+r*np.sin(theta)))
        c = self.coordinates
        return ((1-s)**3)[:, None]*c[0] + (3*s*(1-s)**2)[:, None]*c[1] + (3*s*s*(1-s))[:, None]*c[2] + (s**3)[:, None]*c[3]

    def bbox_and_speed(self, a: float, b: float) -> tuple[np.ndarray, np.ndarray, float]:
        """Conservative finite bounding box and max atlas speed dX/ds."""
        if self.kind == "arc":
            cx, cy, r, t0, t1 = self.coordinates
            lo, hi = t0+(t1-t0)*a, t0+(t1-t0)*b
            angles = [lo, hi] + _angles_between(lo, hi, 0., math.pi/2)
            p = np.column_stack((cx+r*np.cos(angles), cy+r*np.sin(angles)))
            speed = r*abs(t1-t0)
        elif self.kind == "cubic_bezier":
            p = _bezier_restrict(self.coordinates, a, b)
            speed = float(3*np.max(np.linalg.norm(np.diff(p, axis=0), axis=1))/(b-a))
        elif self.discrete:
            p, speed = self.coordinates, 0.
        else:
            knots = self.break_stations()
            ss = np.r_[a, knots[(knots > a) & (knots < b)], b]
            p = self.atlas_points(ss)
            speed = float(np.linalg.norm(np.diff(self.coordinates, axis=0), axis=1).sum())
        return p.min(axis=0), p.max(axis=0), speed

    def denominator_range(self, H: Any) -> tuple[float, float]:
        h = normalize_homography(H)
        a, b = self.station_interval
        if self.discrete:
            points = self.coordinates
        elif self.kind == "arc":
            cx, cy, r, t0, t1 = self.coordinates
            lo, hi = t0+(t1-t0)*a, t0+(t1-t0)*b
            phase = math.atan2(h[2, 1]*r, h[2, 0]*r)
            angles = [lo, hi] + _angles_between(lo, hi, phase, math.pi)
            points = np.column_stack((cx+r*np.cos(angles), cy+r*np.sin(angles)))
        elif self.kind == "cubic_bezier":
            power = self.power_coefficients()
            coeff = power @ h[2]
            roots = _real_polynomial_roots(poly.polyder(coeff))
            ss = np.r_[a, roots[(roots > a) & (roots < b)], b]
            points = self.atlas_points(ss)
        else:
            points = self.atlas_points(self.break_stations())
        den = points @ h[2, :2] + h[2, 2]
        return float(den.min()), float(den.max())

    def power_coefficients(self) -> np.ndarray:
        if self.kind != "cubic_bezier":
            raise RegistrationError("Power coefficients only apply to cubic Bezier support")
        c = self.coordinates
        xy = np.array([c[0], 3*(c[1]-c[0]), 3*(c[2]-2*c[1]+c[0]), c[3]-3*c[2]+3*c[1]-c[0]])
        return np.column_stack((xy, [1., 0., 0., 0.]))

    def to_dict(self) -> dict:
        return {"material_id": self.material_id, "kind": self.kind, "coordinates": self.coordinates.tolist(),
                "station_interval": list(self.station_interval), "branch_id": self.branch_id,
                "provenance": self.provenance, "connectivity_evidence": self.connectivity_evidence}


@dataclass(frozen=True)
class RegistrationObservations:
    native_uv: np.ndarray
    material_ids: tuple[str, ...] | Sequence[Any]
    record_keys: tuple[Any, ...] | Sequence[Any]
    fixed_stations: np.ndarray | None = None
    fixed_station_evidence: tuple[str | None, ...] | Sequence[str | None] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        uv = _xy(self.native_uv, "native_uv")
        ids = tuple(str(x) for x in self.material_ids)
        keys = tuple(_deep_freeze(k) for k in self.record_keys)
        if len(ids) != len(uv) or len(keys) != len(uv):
            raise RegistrationError("Observation arrays must have the same length")
        try:
            unique = len(set(keys))
        except TypeError as exc:
            raise RegistrationError("record_keys must be immutable composite keys") from exc
        if unique != len(keys):
            raise RegistrationError("Duplicate observation key; use the full PTS/source composite key")
        fixed = np.full(len(uv), np.nan) if self.fixed_stations is None else np.asarray(self.fixed_stations, float)
        if fixed.shape != (len(uv),) or np.isinf(fixed).any():
            raise RegistrationError("fixed_stations must have N values (NaN means unknown station)")
        evidence = (None,) * len(uv) if self.fixed_station_evidence is None else tuple(self.fixed_station_evidence)
        if len(evidence) != len(uv):
            raise RegistrationError("fixed_station_evidence length mismatch")
        if any(np.isfinite(s) and (not isinstance(e, str) or not e.strip()) for s, e in zip(fixed, evidence)):
            raise RegistrationError("A fixed world station requires actual physical identity evidence")
        object.__setattr__(self, "native_uv", _immutable_array(uv))
        object.__setattr__(self, "material_ids", ids)
        object.__setattr__(self, "record_keys", keys)
        object.__setattr__(self, "fixed_stations", _immutable_array(fixed))
        object.__setattr__(self, "fixed_station_evidence", evidence)
        object.__setattr__(self, "metadata", _deep_freeze(self.metadata))

    @property
    def support_fingerprint(self) -> str:
        h = hashlib.sha256()
        h.update(self.native_uv.tobytes())
        h.update(self.fixed_stations.tobytes())
        h.update(json.dumps(_jsonable((self.record_keys, self.material_ids, self.fixed_station_evidence)), sort_keys=True).encode())
        return h.hexdigest()


def _real_polynomial_roots(coefficients: Any) -> np.ndarray:
    c = np.asarray(coefficients, float).copy()
    scale = float(np.max(np.abs(c))) if len(c) else 0.
    if scale == 0:
        return np.empty(0)
    c /= scale
    while len(c) > 1 and abs(c[-1]) < 1e-12:
        c = c[:-1]
    if len(c) <= 1:
        return np.empty(0)
    roots = poly.polyroots(c)
    roots = roots[np.abs(roots.imag) < 1e-7*(1+np.abs(roots.real))].real
    return roots[np.isfinite(roots)]


def _distance_stationary_polynomial(u: np.ndarray, v: np.ndarray, w: np.ndarray, obs: np.ndarray) -> np.ndarray:
    aa, bb = poly.polysub(u, obs[0]*w), poly.polysub(v, obs[1]*w)
    norm_sq = poly.polyadd(poly.polymul(aa, aa), poly.polymul(bb, bb))
    dot = poly.polyadd(poly.polymul(aa, poly.polyder(aa)), poly.polymul(bb, poly.polyder(bb)))
    return poly.polysub(poly.polymul(dot, w), poly.polymul(norm_sq, poly.polyder(w)))


def validate_material_domain(H_atlas_to_native: Any, materials: Sequence[MaterialSupport],
                             expected_orientation: Mapping[str, int] | None = None) -> dict[str, int]:
    h = normalize_homography(H_atlas_to_native)
    orientation = {}
    determinant_sign = 1 if np.linalg.det(h) > 0 else -1
    for m in materials:
        lo, hi = m.denominator_range(h)
        if m.discrete:
            p = project_points(h, m.coordinates)
            del p
            # Disconnected points do not assert absence of a horizon between them.
            signs = np.sign(m.coordinates @ h[2, :2] + h[2, 2]).astype(int)
            signature = tuple((determinant_sign * signs).tolist())
        else:
            bbox_lo, bbox_hi, _ = m.bbox_and_speed(*m.station_interval)
            magnitude = max(1., float(np.linalg.norm(np.r_[np.maximum(np.abs(bbox_lo), np.abs(bbox_hi)), 1.])))
            threshold = 1e-11 * np.linalg.norm(h[2]) * magnitude
            if lo <= threshold and hi >= -threshold:
                raise RegistrationError(
                    f"Initialization support-domain limitation: a horizon or vanishing denominator crosses "
                    f"the full declared finite branch of material {m.material_id}; this is not an observation-error finding. "
                    "Supply an independently supported legal station_interval or a different initial H.")
            signature = determinant_sign * (1 if lo > 0 else -1)
        orientation[m.material_id] = signature
        if expected_orientation is not None and expected_orientation[m.material_id] != signature:
            raise RegistrationError(f"Orientation/mirror branch changed for material {m.material_id}")
    return orientation


def nearest_on_material(H_atlas_to_native: Any, native_uv: Any, material: MaterialSupport,
                        fixed_stations: Any = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Native continuous finite nearest point (all real stationary roots + ends).

    Lines use exact clamped finite segments; arcs and cubic Beziers enumerate
    the stationary polynomial, with endpoints retained.  The result is numerical
    double-precision continuous geometry, not a raster-distance transform.
    Returns predicted_uv, stations, at_support_boundary, ambiguous_minimum.
    """
    uv = _xy(native_uv, "native_uv")
    h = normalize_homography(H_atlas_to_native)
    validate_material_domain(h, [material])
    n = len(uv)
    fixed = np.full(n, np.nan) if fixed_stations is None else np.asarray(fixed_stations, float)
    if fixed.shape != (n,) or np.isinf(fixed).any():
        raise RegistrationError("Invalid fixed station array")
    predicted, stations = np.empty_like(uv), np.empty(n)
    boundary, ambiguous = np.zeros(n, bool), np.zeros(n, bool)
    is_fixed = np.isfinite(fixed)
    if np.any(is_fixed):
        predicted[is_fixed] = project_points(h, material.atlas_points(fixed[is_fixed]))
        stations[is_fixed] = fixed[is_fixed]
    indices = np.flatnonzero(~is_fixed)
    if not len(indices):
        return predicted, stations, boundary, ambiguous
    obs = uv[indices]
    if material.discrete:
        pp = project_points(h, material.coordinates)
        d2 = ((obs[:, None, :] - pp[None, :, :])**2).sum(axis=2)
        jj = np.argmin(d2, axis=1)
        predicted[indices] = pp[jj]
        stations[indices] = jj/max(1, len(pp)-1)
        if len(pp) > 1:
            best2 = np.partition(d2, 1, axis=1)[:, :2]
            ambiguous[indices] = np.abs(best2[:, 0]-best2[:, 1]) <= 1e-9
    elif material.kind in ("segment", "polyline"):
        ss = material.break_stations()
        atlas = material.atlas_points(ss)
        pp = project_points(h, atlas)
        start, delta = pp[:-1], np.diff(pp, axis=0)
        lengths2 = (delta*delta).sum(1)
        if np.any(lengths2 <= 1e-24):
            raise RegistrationError("Projected segment collapsed")
        fractions = np.clip(np.einsum("nki,ki->nk", obs[:, None]-start[None], delta)/lengths2[None], 0., 1.)
        choices = start[None] + fractions[:, :, None]*delta[None]
        distances2 = ((choices-obs[:, None])**2).sum(2)
        jj = np.argmin(distances2, axis=1)
        rows = np.arange(len(obs))
        native_fraction = fractions[rows, jj]
        w = atlas @ h[2, :2] + h[2, 2]
        local_s = native_fraction*w[jj] / ((1-native_fraction)*w[jj+1] + native_fraction*w[jj])
        stations[indices] = ss[jj] + local_s*(ss[jj+1]-ss[jj])
        predicted[indices] = choices[rows, jj]
        boundary[indices] = ((stations[indices]-ss[0] < 1e-8) | (ss[-1]-stations[indices] < 1e-8))
        if len(delta) > 1:
            best2 = np.partition(distances2, 1, axis=1)[:, :2]
            # Shared polyline vertices are the same station, not distinct branches.
            ambiguous[indices] = (np.abs(best2[:, 0]-best2[:, 1]) < 1e-9) & (~np.any(np.abs(stations[indices, None]-ss[None]) < 1e-8, axis=1))
    else:
        a, b = material.station_interval
        if material.kind == "cubic_bezier":
            coefficients = material.power_coefficients() @ h.T
            u, v, w = coefficients.T
        else:
            cx, cy, radius, theta0, theta1 = material.coordinates
            # Rational quadratic under t=tan(theta/2); include chart infinity.
            atlas_coeff = np.array([[cx+radius, cy, 1.], [0., 2*radius, 0.], [cx-radius, cy, 1.]])
            u, v, w = (atlas_coeff @ h.T).T
        for i, observed in zip(indices, obs):
            roots = _real_polynomial_roots(_distance_stationary_polynomial(u, v, w, observed))
            if material.kind == "cubic_bezier":
                candidates = np.r_[a, roots[(roots > a) & (roots < b)], b]
            else:
                lo, hi = theta0+(theta1-theta0)*a, theta0+(theta1-theta0)*b
                angles = [lo, hi] + _angles_between(lo, hi, math.pi, 2*math.pi)
                for root in roots:
                    angles.extend(_angles_between(lo, hi, 2*math.atan(float(root)), 2*math.pi))
                candidates = np.array([(theta-theta0)/(theta1-theta0) for theta in angles])
            candidates = np.unique(np.clip(candidates, a, b))
            pp = project_points(h, material.atlas_points(candidates))
            d2 = ((pp-observed)**2).sum(1)
            j = int(np.argmin(d2))
            predicted[i], stations[i] = pp[j], candidates[j]
            closed_arc = material.kind == "arc" and abs(abs(material.coordinates[4]-material.coordinates[3])*(b-a)-2*math.pi) < 1e-9
            boundary[i] = not closed_arc and (abs(candidates[j]-a) < 1e-8 or abs(candidates[j]-b) < 1e-8)
            distinct = np.linalg.norm(pp-pp[j], axis=1) > 1e-6
            ambiguous[i] = bool(np.any(distinct & (np.abs(d2-d2[j]) <= 1e-8*max(1., d2[j]))))
    return predicted, stations, boundary, ambiguous


@dataclass(frozen=True)
class NativeReplay:
    predicted_native_uv: np.ndarray
    error_native_uv: np.ndarray
    stations: np.ndarray
    at_support_boundary: np.ndarray
    ambiguous_minimum: np.ndarray
    max_error_native_px: float
    rms_error_native_px: float
    support_valid: bool = True
    scope: str = "declared local support only; no whole-shot t_hat"

    def __post_init__(self):
        for name in ("predicted_native_uv", "error_native_uv", "stations"):
            object.__setattr__(self, name, _immutable_array(getattr(self, name)))
        for name in ("at_support_boundary", "ambiguous_minimum"):
            object.__setattr__(self, name, _immutable_array(getattr(self, name), bool))


def _material_map(materials: Mapping[Any, MaterialSupport] | Sequence[MaterialSupport]) -> dict[str, MaterialSupport]:
    sequence = tuple(materials.values()) if isinstance(materials, Mapping) else tuple(materials)
    if not all(isinstance(m, MaterialSupport) for m in sequence):
        raise RegistrationError("All materials must be explicitly constructed MaterialSupport objects")
    result = {m.material_id: m for m in sequence}
    if len(result) != len(sequence):
        raise RegistrationError("A local comparator has one explicit finite branch per material ID")
    return result


def native_replay(H_atlas_to_native: Any, materials: Mapping[Any, MaterialSupport] | Sequence[MaterialSupport],
                  observations: RegistrationObservations) -> NativeReplay:
    mmap = _material_map(materials)
    absent = set(observations.material_ids) - set(mmap)
    if absent:
        raise RegistrationError(f"No finite template for materials {sorted(absent)}; declare an explicit local subset")
    pred = np.empty_like(observations.native_uv)
    station = np.empty(len(pred))
    boundary, ambiguous = np.zeros(len(pred), bool), np.zeros(len(pred), bool)
    ids = np.asarray(observations.material_ids)
    for mid in sorted(set(observations.material_ids)):
        idx = np.flatnonzero(ids == mid)
        pred[idx], station[idx], boundary[idx], ambiguous[idx] = nearest_on_material(
            H_atlas_to_native, observations.native_uv[idx], mmap[mid], observations.fixed_stations[idx])
    error = pred - observations.native_uv
    errors2 = (error*error).sum(axis=1)
    return NativeReplay(pred, error, station, boundary, ambiguous,
                        float(np.sqrt(errors2.max())), float(np.sqrt(errors2.mean())))


def normalized_positive_softmin(native_observations: Any, projected_samples: Any, weights: Any,
                                 sigma: float, *, chunk_size: int = 256) -> np.ndarray:
    """-2*sigma^2*log(sum a*exp(-d^2/(2*sigma^2))), in supplied pixel units."""
    x, y = _xy(native_observations, "observations"), _xy(projected_samples, "projected_samples")
    w = np.asarray(weights, float)
    if w.shape != (len(y),) or np.any(w <= 0) or not np.isfinite(w).all():
        raise RegistrationError("Soft-min quadrature weights must be strictly positive and finite")
    if not np.isclose(w.sum(), 1., atol=1e-12, rtol=1e-12):
        raise RegistrationError("Soft-min quadrature weights must sum to one")
    if not np.isfinite(sigma) or sigma <= 0 or chunk_size < 1:
        raise RegistrationError("Soft-min sigma and chunk_size must be positive")
    energy = np.empty(len(x))
    for start in range(0, len(x), chunk_size):
        block = x[start:start+chunk_size]
        distance2 = ((block[:, None, :] - y[None, :, :])**2).sum(axis=2)
        energy[start:start+len(block)] = -2*sigma*sigma*logsumexp(np.log(w)[None]-distance2/(2*sigma*sigma), axis=1)
    # Tiny negative roundoff is not a change in the declared objective.
    return np.maximum(energy, 0.)


def _interval_geometry(material: MaterialSupport, stations: np.ndarray):
    lows, highs, speeds = [], [], []
    for a, b in zip(stations[:-1], stations[1:]):
        lo, hi, speed = material.bbox_and_speed(float(a), float(b))
        lows.append(lo); highs.append(hi); speeds.append(speed*(b-a))
    return np.asarray(lows).reshape(-1, 2), np.asarray(highs).reshape(-1, 2), np.asarray(speeds)


def _interval_arc_bounds(H: Any, lows: np.ndarray, highs: np.ndarray, speed_times_ds: np.ndarray) -> np.ndarray:
    """Certified conservative native arc-length upper bounds on each interval."""
    if len(lows) == 0:
        return np.empty(0)
    h = normalize_homography(H)
    corners = np.stack([lows, np.column_stack((highs[:, 0], lows[:, 1])), highs,
                        np.column_stack((lows[:, 0], highs[:, 1]))], axis=1)
    w = corners @ h[2, :2] + h[2, 2]
    low, high = w.min(1), w.max(1)
    valid = (low > 0) | (high < 0)
    minimum_w = np.minimum(np.abs(low), np.abs(high))
    p = corners @ h[:2, :2].T + h[:2, 2]
    numerator = h[None, None, :2, :2]*w[:, :, None, None] - p[:, :, :, None]*h[None, None, 2, :2]
    norm = np.sqrt((numerator*numerator).sum(axis=(2, 3))).max(1)
    result = np.full(len(lows), np.inf)
    result[valid] = norm[valid] / minimum_w[valid]**2 * speed_times_ds[valid]
    return result


@dataclass(frozen=True)
class QuadratureSamples:
    material_id: str
    stations: np.ndarray
    atlas_xy: np.ndarray
    weights: np.ndarray
    interval_bbox_low: np.ndarray
    interval_bbox_high: np.ndarray
    speed_times_ds: np.ndarray
    discrete: bool
    measure: str = "positive trapezoidal normalized legal-station measure"

    def __post_init__(self):
        for name in ("stations", "atlas_xy", "weights", "interval_bbox_low", "interval_bbox_high", "speed_times_ds"):
            object.__setattr__(self, name, _immutable_array(getattr(self, name)))

    def native_fill_bound(self, H_atlas_to_native: Any) -> float | None:
        if self.discrete:
            return None  # Discrete support makes no continuous fill-distance assertion.
        bounds = _interval_arc_bounds(H_atlas_to_native, self.interval_bbox_low,
                                      self.interval_bbox_high, self.speed_times_ds)
        return float(bounds.max()/2)


def build_quadrature(material: MaterialSupport, H_atlas_to_native: Any, sigma_native_px: float,
                     max_samples: int = 8193) -> QuadratureSamples:
    """Build with a 2x margin; trials retain samples/weights and check sigma/4."""
    if sigma_native_px <= 0 or max_samples < 2:
        raise RegistrationError("Invalid sampling settings")
    validate_material_domain(H_atlas_to_native, [material])
    stations = material.break_stations()
    if material.discrete:
        n = len(stations)
        if n > max_samples:
            raise SamplingLimitError("Explicit point support exceeds frozen sample budget")
        return QuadratureSamples(material.material_id, stations, material.coordinates, np.full(n, 1/n),
                                 np.empty((0, 2)), np.empty((0, 2)), np.empty(0), True,
                                 "uniform positive finite-point measure; no connectivity inferred")
    for _ in range(40):
        lows, highs, speeds = _interval_geometry(material, stations)
        bound = _interval_arc_bounds(H_atlas_to_native, lows, highs, speeds)
        split = bound > sigma_native_px/4  # fill <= sigma/8 at base, <= sigma/4 in trials
        if not split.any():
            break
        if len(stations) + int(split.sum()) > max_samples:
            raise SamplingLimitError(f"{material.material_id}: cannot certify native fill within {max_samples} samples")
        stations = np.sort(np.r_[stations, (stations[:-1][split] + stations[1:][split])/2])
    else:
        raise SamplingLimitError("Sampling refinement depth exhausted")
    ds = np.diff(stations)
    weights = np.r_[ds[0]/2, (ds[:-1]+ds[1:])/2, ds[-1]/2]
    weights /= weights.sum()
    return QuadratureSamples(material.material_id, stations, material.atlas_points(stations), weights,
                             lows, highs, speeds, False)


@dataclass(frozen=True)
class RankReport:
    rank: int
    parameter_count: int
    singular_values: np.ndarray
    weak_directions: np.ndarray
    relative_threshold: float
    parameter_scale_native_px: float
    residual_scale_native_px: float
    local_differentiability: str
    interpretation: str = "actual finite-material residual Jacobian; no priors or damping"

    def __post_init__(self):
        object.__setattr__(self, "singular_values", _immutable_array(self.singular_values))
        object.__setattr__(self, "weak_directions", _immutable_array(self.weak_directions))


def finite_difference_jacobian(fun: Callable[[np.ndarray], np.ndarray], x: np.ndarray,
                               step: float = 2e-6) -> np.ndarray:
    base = np.asarray(fun(x), float).ravel()
    jac = np.empty((len(base), len(x)))
    for j in range(len(x)):
        delta = step * max(1., abs(float(x[j])))
        for _ in range(14):
            plus, minus = x.copy(), x.copy()
            plus[j] += delta; minus[j] -= delta
            try:
                fp, fm = np.asarray(fun(plus), float).ravel(), np.asarray(fun(minus), float).ravel()
                if np.isfinite(fp).all() and np.isfinite(fm).all():
                    jac[:, j] = (fp-fm)/(2*delta)
                    break
            except (RegistrationError, np.linalg.LinAlgError, FloatingPointError):
                pass
            delta *= .5
        else:
            raise RegistrationError("Cannot differentiate within the true finite-support domain")
    return jac


def rank_diagnostics(H_atlas_to_native: Any, materials: Mapping[Any, MaterialSupport] | Sequence[MaterialSupport],
                      observations: RegistrationObservations, *, chart: LocalHomographyChart | None = None,
                      relative_threshold: float = 1e-5, residual_scale_native_px: float = 2.) -> RankReport:
    mats = tuple(_material_map(materials).values())
    if residual_scale_native_px <= 0:
        raise RegistrationError("Frozen engineering residual scale must be positive")
    if chart is None:
        support = np.concatenate([m.coordinates if m.discrete else m.atlas_points(np.linspace(*m.station_interval, 9)) for m in mats])
        chart = LocalHomographyChart.from_homography(H_atlas_to_native, support)
    p = chart.parameters(H_atlas_to_native)
    fun = lambda par: native_replay(chart.matrix(par), mats, observations).error_native_uv.ravel()/residual_scale_native_px
    jac = finite_difference_jacobian(fun, p)
    _, sv, vh = np.linalg.svd(jac, full_matrices=jac.shape[0] < jac.shape[1])
    rank = int(np.count_nonzero(sv > (sv[0]*relative_threshold if len(sv) and sv[0] else 0.)))
    replay = native_replay(H_atlas_to_native, mats, observations)
    differentiability = "branch_or_finite_endpoint_near: inspect directional rank" if np.any(replay.ambiguous_minimum | replay.at_support_boundary) else "locally smooth nearest branch"
    return RankReport(rank, 8, sv, vh[rank:], relative_threshold,
                      chart.parameter_scale_native_px, residual_scale_native_px, differentiability)


@dataclass(frozen=True)
class RegistrationOptions:
    pyramid_scales: tuple[float, ...] = (.125, .25, .5, 1.)
    sigma_native_px: tuple[float, ...] = (64., 32., 8., 2.)
    capture_iterations_per_stage: int = 12
    native_iterations: int = 25
    max_samples_per_material: int = 8193
    max_sampling_rebuilds_per_stage: int = 2
    lm_initial_damping: float = 1e-3
    max_parameter_step: float = .15
    scaled_gradient_tolerance: float = 1e-6
    relative_cost_tolerance: float = 1e-9
    rank_relative_threshold: float = 1e-5
    engineering_residual_scale_native_px: float = 2.
    objective_version: str = "finite_same_material_softmin_v1.1/native_continuous_v1"

    def __post_init__(self):
        object.__setattr__(self, "pyramid_scales", tuple(float(s) for s in self.pyramid_scales))
        object.__setattr__(self, "sigma_native_px", tuple(float(s) for s in self.sigma_native_px))
        if len(self.pyramid_scales) != len(self.sigma_native_px) or any(s <= 0 or not np.isfinite(s) for s in self.pyramid_scales+self.sigma_native_px):
            raise RegistrationError("Capture scales and native sigmas must be positive parallel lists")
        if self.pyramid_scales and self.pyramid_scales[-1] != 1.:
            raise RegistrationError("Capture schedule must exit at native scale 1")
        if min(self.capture_iterations_per_stage, self.native_iterations) < 0 or self.max_samples_per_material < 2:
            raise RegistrationError("Invalid finite iteration/sample limits")
        if self.max_parameter_step <= 0 or self.engineering_residual_scale_native_px <= 0:
            raise RegistrationError("Invalid parameter or engineering residual scale")

    def to_dict(self) -> dict:
        return {key: _jsonable(getattr(self, key)) for key in self.__dataclass_fields__}


@dataclass(frozen=True)
class CandidateSnapshot:
    frame_id: int
    plane_id: str
    H_atlas_to_native: np.ndarray
    materials: tuple[MaterialSupport, ...]
    observations: RegistrationObservations
    replay: NativeReplay
    objective_settings: Mapping[str, Any]
    stage: str
    pyramid_scale: float
    sigma_native_px: float | None
    quadrature: tuple[QuadratureSamples, ...]
    capture_energy_native: float | None
    provenance: tuple[str, ...]
    rank: RankReport | None = None
    scope: str = "complete immutable local candidate, not a complete physical shot state"

    def __post_init__(self):
        object.__setattr__(self, "H_atlas_to_native", _immutable_array(normalize_homography(self.H_atlas_to_native)))
        object.__setattr__(self, "materials", tuple(self.materials))
        object.__setattr__(self, "quadrature", tuple(self.quadrature))
        object.__setattr__(self, "provenance", tuple(self.provenance))
        object.__setattr__(self, "objective_settings", _deep_freeze(self.objective_settings))

    @property
    def support_fingerprint(self) -> str:
        return self.observations.support_fingerprint

    @property
    def snapshot_fingerprint(self) -> str:
        h = hashlib.sha256(self.H_atlas_to_native.tobytes())
        h.update(self.support_fingerprint.encode())
        h.update(json.dumps([m.to_dict() for m in self.materials], sort_keys=True).encode())
        h.update(json.dumps(_jsonable(self.objective_settings), sort_keys=True).encode())
        h.update(self.replay.stations.tobytes())
        h.update(self.stage.encode())
        return h.hexdigest()

    def to_dict(self, include_observations: bool = True) -> dict:
        out = {"frame_id": self.frame_id, "plane_id": self.plane_id,
               "H_atlas_to_native": self.H_atlas_to_native.tolist(), "stage": self.stage,
               "scope": self.scope, "support_fingerprint": self.support_fingerprint,
               "snapshot_fingerprint": self.snapshot_fingerprint, "materials": [m.to_dict() for m in self.materials],
               "objective_settings": _jsonable(self.objective_settings), "pyramid_scale": self.pyramid_scale,
               "sigma_native_px": self.sigma_native_px, "capture_energy_native": self.capture_energy_native,
               "max_error_native_px": self.replay.max_error_native_px, "rms_error_native_px": self.replay.rms_error_native_px,
               "stations": self.replay.stations.tolist(), "predicted_native_uv": self.replay.predicted_native_uv.tolist(),
               "error_native_uv": self.replay.error_native_uv.tolist(), "support_valid": self.replay.support_valid,
               "at_support_boundary": self.replay.at_support_boundary.tolist(), "ambiguous_minimum": self.replay.ambiguous_minimum.tolist(),
               "provenance": list(self.provenance),
               "quadrature": [{"material_id": q.material_id, "stations": q.stations.tolist(),
                                "weights": q.weights.tolist(), "measure": q.measure,
                                "native_fill_bound": q.native_fill_bound(self.H_atlas_to_native)} for q in self.quadrature],
               "rank": None if self.rank is None else {"rank": self.rank.rank, "singular_values": self.rank.singular_values.tolist(),
                         "weak_directions": self.rank.weak_directions.tolist(), "relative_threshold": self.rank.relative_threshold,
                         "parameter_scale_native_px": self.rank.parameter_scale_native_px,
                         "residual_scale_native_px": self.rank.residual_scale_native_px,
                         "local_differentiability": self.rank.local_differentiability}}
        if include_observations:
            out["observations"] = {"native_uv": self.observations.native_uv.tolist(),
                                   "material_ids": list(self.observations.material_ids),
                                   "record_keys": _jsonable(self.observations.record_keys),
                                   "fixed_stations": [None if np.isnan(s) else float(s) for s in self.observations.fixed_stations],
                                   "fixed_station_evidence": list(self.observations.fixed_station_evidence),
                                   "metadata": _jsonable(self.observations.metadata)}
        return out


@dataclass(frozen=True)
class RegistrationResult:
    best_native: CandidateSnapshot
    final: CandidateSnapshot
    snapshots: tuple[CandidateSnapshot, ...]
    optimization_log: tuple[Mapping[str, Any], ...]
    status: str
    limitations: tuple[str, ...]
    fit_scope: str = "S1 local homography initialization; physical production fitting not executed"

    def __post_init__(self):
        object.__setattr__(self, "snapshots", tuple(self.snapshots))
        object.__setattr__(self, "optimization_log", tuple(_deep_freeze(e) for e in self.optimization_log))
        object.__setattr__(self, "limitations", tuple(self.limitations))


def _finite_lm(fun: Callable[[np.ndarray], np.ndarray], x: np.ndarray, iterations: int,
               options: RegistrationOptions, accepted: Callable[[np.ndarray, float], None] | None = None):
    """Small 8D damped LM: reject true-domain trials, same frozen objective."""
    x = np.asarray(x, float).copy()
    r = np.asarray(fun(x), float).ravel()
    cost = float(.5*np.dot(r, r))
    damping = options.lm_initial_damping
    history = []
    status = "iteration_limit"
    for iteration in range(iterations):
        try:
            jac = finite_difference_jacobian(fun, x)
        except RegistrationError as exc:
            status = "derivative_domain_limited"
            history.append({"iteration": iteration, "status": status, "reason": str(exc)})
            break
        jtj, gradient = jac.T @ jac, jac.T @ r
        diagonal = np.maximum(np.diag(jtj), 1e-12)
        scaled_gradient = float(np.max(np.abs(gradient)/np.sqrt(diagonal)))
        if scaled_gradient <= options.scaled_gradient_tolerance:
            status = "scaled_gradient_converged"
            break
        accepted_trial = False
        for trial in range(16):
            try:
                step = np.linalg.solve(jtj + damping*np.diag(diagonal), -gradient)
            except np.linalg.LinAlgError:
                damping *= 10
                continue
            norm = float(np.linalg.norm(step))
            if norm > options.max_parameter_step:
                step *= options.max_parameter_step/norm
            prediction = float(-gradient @ step - .5*step @ jtj @ step)
            if prediction <= 0 or not np.isfinite(prediction):
                damping *= 10
                continue
            try:
                trial_r = np.asarray(fun(x+step), float).ravel()
                if not np.isfinite(trial_r).all():
                    raise RegistrationError("Nonfinite residual")
                trial_cost = float(.5*trial_r @ trial_r)
            except (RegistrationError, np.linalg.LinAlgError, FloatingPointError) as exc:
                history.append({"iteration": iteration, "trial": trial, "accepted": False,
                                "reason": "true_domain_or_fixed_sampling: " + str(exc)})
                damping *= 5
                continue
            decrease = cost-trial_cost
            ratio = decrease/prediction
            if decrease > 0 and ratio > 1e-4:
                previous = cost
                x, r, cost = x+step, trial_r, trial_cost
                damping = max(1e-12, damping*(.35 if ratio > .75 else 1. if ratio > .25 else 2.))
                history.append({"iteration": iteration, "accepted": True, "cost": cost,
                                "actual_over_predicted": ratio, "scaled_gradient_before": scaled_gradient})
                if accepted:
                    accepted(x, cost)
                accepted_trial = True
                if decrease <= options.relative_cost_tolerance*max(1., previous):
                    status = "relative_cost_converged"
                break
            damping *= 5
        if not accepted_trial:
            status = "trial_or_domain_limited"
            break
        if status == "relative_cost_converged":
            break
    return x, status, history


def fit_registration(materials: Mapping[Any, MaterialSupport] | Sequence[MaterialSupport],
                     observations: RegistrationObservations, initial_H_atlas_to_native: Any,
                     *, frame_id: int, plane_id: str, options: RegistrationOptions | None = None,
                     initial_provenance: str = "caller supplied initial candidate",
                     atlas_controls: Any = None) -> RegistrationResult:
    """Run finite multi-scale capture, native continuous refinement, and rank.

    Supply only one genuine plane's explicit local observation subset.  The
    caller's whole-shot ledger must retain all unsupported records as
    not_applicable for this comparator, and as obligations of the physical fit.
    """
    options = options or RegistrationOptions()
    mmap = _material_map(materials)
    absent = set(observations.material_ids)-set(mmap)
    if absent:
        raise RegistrationError(f"Missing explicit finite supports: {sorted(absent)}")
    mats = tuple(mmap[mid] for mid in sorted(set(observations.material_ids)))
    h = normalize_homography(initial_H_atlas_to_native)
    orientations = validate_material_domain(h, mats)
    support = np.concatenate([m.coordinates if m.discrete else m.atlas_points(np.linspace(*m.station_interval, 9)) for m in mats])
    chart = LocalHomographyChart.from_homography(h, support, atlas_controls)
    x = chart.parameters(h)
    ids = np.asarray(observations.material_ids)
    groups = {m.material_id: np.flatnonzero(ids == m.material_id) for m in mats}
    settings = options.to_dict()
    settings.update({"parameterization": "four_virtual_control_points_no_added_measurements",
                     "atlas_controls": chart.atlas_controls.tolist(),
                     "parameter_scale_native_px": chart.parameter_scale_native_px,
                     "selection": "lexicographic native finite-support max error, then RMS, identical support",
                     "statistical_weights": "capture per-material count only; native frozen engineering scale; not calibrated W",
                     "orientation_branch": orientations})
    snapshots, logs, limitations = [], [], []

    def make_snapshot(matrix, stage, scale=1., sigma=None, quadrature=(), capture=None):
        replay = native_replay(matrix, mats, observations)
        candidate = CandidateSnapshot(int(frame_id), str(plane_id), matrix, mats, observations, replay,
                                      settings, stage, scale, sigma, tuple(quadrature), capture,
                                      (initial_provenance, "immutable original UV and finite same-material support"))
        snapshots.append(candidate)
        return candidate

    best = make_snapshot(h, "initial")
    score = lambda candidate: (candidate.replay.max_error_native_px, candidate.replay.rms_error_native_px)
    last_status = "not_optimized"
    for stage_index, (scale, sigma) in enumerate(zip(options.pyramid_scales, options.sigma_native_px)):
        for rebuild in range(options.max_sampling_rebuilds_per_stage + 1):
            h = chart.matrix(x)
            try:
                quadrature = tuple(build_quadrature(m, h, sigma, options.max_samples_per_material) for m in mats)
            except SamplingLimitError as exc:
                limitations.append(str(exc))
                logs.append({"stage": stage_index, "status": "sampling_limited", "reason": str(exc)})
                last_status = "sampling_limited"
                break
            qmap = {q.material_id: q for q in quadrature}
            scaled_observations = scale*(observations.native_uv+.5)-.5

            def capture_fun(parameters):
                matrix = chart.matrix(parameters)
                validate_material_domain(matrix, mats, orientations)
                h_scaled = scale_atlas_to_native(matrix, scale)
                residual = np.empty(len(observations.native_uv))
                for m in mats:
                    idx, q = groups[m.material_id], qmap[m.material_id]
                    fill = q.native_fill_bound(matrix)
                    if fill is not None and fill > sigma/4*(1+1e-12):
                        raise RegistrationError("Frozen quadrature fill exceeds sigma_native/4; outer rebuild required")
                    fixed = np.isfinite(observations.fixed_stations[idx])
                    energy = np.empty(len(idx))
                    if np.any(~fixed):
                        projected = project_points(h_scaled, q.atlas_xy)  # reproject every trial
                        energy[~fixed] = normalized_positive_softmin(scaled_observations[idx[~fixed]], projected,
                                                                    q.weights, scale*sigma)
                    if np.any(fixed):
                        projected = project_points(h_scaled, m.atlas_points(observations.fixed_stations[idx[fixed]]))
                        energy[fixed] = ((projected-scaled_observations[idx[fixed]])**2).sum(1)
                    residual[idx] = np.sqrt(2*energy/len(idx))
                return residual

            def accepted(parameters, cost):
                nonlocal best
                candidate = make_snapshot(chart.matrix(parameters), f"capture_{stage_index}_accepted",
                                          scale, sigma, quadrature, cost/(scale*scale))
                if score(candidate) < score(best):
                    best = candidate

            x, last_status, history = _finite_lm(capture_fun, x, options.capture_iterations_per_stage, options, accepted)
            logs.append({"stage": stage_index, "rebuild": rebuild, "scale": scale, "sigma_native_px": sigma,
                         "status": last_status, "iterations": history,
                         "quadrature_counts": {q.material_id: len(q.stations) for q in quadrature}})
            exit_energy = float(.5*np.dot(capture_fun(x), capture_fun(x)))/(scale*scale)
            candidate = make_snapshot(chart.matrix(x), f"capture_{stage_index}_exit", scale, sigma, quadrature, exit_energy)
            if score(candidate) < score(best):
                best = candidate
            if last_status not in ("trial_or_domain_limited", "derivative_domain_limited"):
                break
        # Every scale change discards its local model and rebuilds fresh samples/J.

    def native_fun(parameters):
        matrix = chart.matrix(parameters)
        validate_material_domain(matrix, mats, orientations)
        return native_replay(matrix, mats, observations).error_native_uv.ravel()/options.engineering_residual_scale_native_px

    def accepted_native(parameters, cost):
        nonlocal best
        candidate = make_snapshot(chart.matrix(parameters), "native_continuous_accepted")
        if score(candidate) < score(best):
            best = candidate

    # Keep the capture trajectory and best native candidate both intact.  Refine
    # the best same-support state, never splice parameters from different states.
    x = chart.parameters(best.H_atlas_to_native)
    x, native_status, history = _finite_lm(native_fun, x, options.native_iterations, options, accepted_native)
    logs.append({"stage": "native_continuous", "status": native_status, "iterations": history})
    final = make_snapshot(chart.matrix(x), "native_continuous_exit")
    if score(final) < score(best):
        best = final
    rank = rank_diagnostics(best.H_atlas_to_native, mats, observations, chart=chart,
                            relative_threshold=options.rank_relative_threshold,
                            residual_scale_native_px=options.engineering_residual_scale_native_px)
    best = replace(best, rank=rank)
    snapshots.append(best)
    if final.snapshot_fingerprint == best.snapshot_fingerprint:
        final = best
    else:
        final = replace(final, rank=rank_diagnostics(final.H_atlas_to_native, mats, observations, chart=chart,
                        relative_threshold=options.rank_relative_threshold,
                        residual_scale_native_px=options.engineering_residual_scale_native_px))
        snapshots.append(final)
    status = "underconstrained" if rank.rank < 8 else native_status
    if rank.rank < 8:
        limitations.append(f"Native data Jacobian rank {rank.rank}/8; retain the {8-rank.rank} weak directions; no unique H claimed")
    limitations.extend(("GS/undistorted planar approximation only; not a physical lower bound or full-shot acceptance",
                        "No automatic station persistence, profile-query factors, verified run-order coupling, reverse coverage, or multi-branch association",
                        "Finite polynomial-root nearest-curve arithmetic and finite LM budgets can be numerically limited; inspect logs and rank",
                        "Local snapshots cover this declared comparator; the caller must wrap full scene, optics, trajectory, and all-frame ledger for physical snapshots"))
    return RegistrationResult(best, final, tuple(snapshots), tuple(logs), status, tuple(limitations))
