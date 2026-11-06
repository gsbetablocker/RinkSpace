"""Source-driven contour contracts, not a new shoe model or automatic tracker.

Retains finite pixel uncertainty. Full-extent constraints require explicit
per-contour visibility evidence, never the name of the JSON file. The renderer
must supply the visible model branch, NOT all hidden/internal mesh edges.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


def polyline_measure_weights(points: np.ndarray, *, closed: bool = False) -> np.ndarray:
    """Normalized arc-length measure, independent of how densely a trace is clicked."""
    p = np.asarray(points, dtype=float)
    if p.ndim != 2 or p.shape[1] != 2 or len(p) < 2 or not np.isfinite(p).all():
        raise ValueError("Finite native-pixel polyline required")
    if closed and np.array_equal(p[0], p[-1]):
        w = polyline_measure_weights(p[:-1], closed=True)
        return np.r_[w[0]/2, w[1:], w[0]/2]
    lengths = np.linalg.norm(np.diff(p, axis=0), axis=1)
    if np.any(lengths <= 0):
        raise ValueError("Repeated consecutive samples must be explicitly deduplicated")
    w = np.r_[lengths, 0.] + np.r_[0., lengths]
    if closed:
        closing = float(np.linalg.norm(p[-1]-p[0])); w[[0,-1]] += closing
    return w/w.sum()


@dataclass(frozen=True)
class ExtentEvidence:
    source_id: str
    pts: int
    physical_foot: int | None
    temporal_track_id: str
    feature: str
    visibility: str
    completeness: str
    provenance: str
    endpoint_identity: str = "unknown"

    def checked(self) -> "ExtentEvidence":
        if not self.source_id or not self.temporal_track_id or not self.provenance:
            raise ValueError("Source, temporal identity and provenance cannot be fabricated")
        if isinstance(self.pts, bool) or not isinstance(self.pts, (int, np.integer)):
            raise ValueError("Exact integer source PTS required")
        if self.physical_foot not in (None, 0, 1) or isinstance(self.physical_foot, bool):
            raise ValueError("Anatomical identity is 0, 1, or unassigned")
        if self.visibility not in ("visible", "partially_visible", "occluded", "unknown"):
            raise ValueError("Unknown visibility enum")
        if self.completeness not in ("complete", "partial", "unknown"):
            raise ValueError("Explicit contour completeness required")
        if self.endpoint_identity not in ("front_rear_confirmed", "unknown"):
            raise ValueError("Uncertainty about endpoints is not a material observation")
        return self

    def can_constrain_full_extent(self) -> bool:
        self.checked()
        return self.visibility == "visible" and self.completeness == "complete"


def finite_point_region(predicted_uv, centre_uv, radius_px, *, weight=1.) -> tuple[np.ndarray, np.ndarray]:
    """Outside a native-pixel disk; no attraction to an arbitrary region centre."""
    v = np.asarray(predicted_uv, float)-np.asarray(centre_uv, float)
    if v.shape != (2,) or not np.isfinite(v).all() or not np.isfinite([radius_px, weight]).all() or min(radius_px, weight) <= 0:
        raise ValueError("Finite pixel geometry, positive radius and weight required")
    q = v/radius_px; norm = float(np.linalg.norm(q))
    if norm <= 1.:
        return np.zeros(2), np.zeros((2,2))
    scale = np.sqrt(weight)
    return scale*(1-1/norm)*q, scale*((1-1/norm)*np.eye(2)+np.outer(q,q)/norm**3)/radius_px


def directed_endpoint_terms(predicted_front_rear, source_front_rear, radii_px, *, evidence: ExtentEvidence, total_weight: float):
    """Independent semantic endpoints: a front/rear swap must not score zero.

    Use only explicitly observed endpoints; never derive these from silhouette
    extrema or clipped subsegments. Weight is a share of the SAME source budget.
    """
    evidence.checked()
    if evidence.endpoint_identity != "front_rear_confirmed" or evidence.visibility != "visible":
        raise ValueError("Need actual front/rear source identity, not an unoriented axis")
    p, s, rad = np.asarray(predicted_front_rear, float), np.asarray(source_front_rear, float), np.asarray(radii_px, float)
    if p.shape != (2,2) or s.shape != (2,2) or rad.shape != (2,):
        raise ValueError("Expected [front,rear] x uv and their source radii")
    r = []; J = np.zeros((4,4))
    for k in range(2):
        a, D = finite_point_region(p[k], s[k], rad[k], weight=total_weight/2)
        r.append(a); J[2*k:2*k+2,2*k:2*k+2] = D
    return np.concatenate(r), J


def directed_polyline_distances(points, polyline) -> np.ndarray:
    p = np.asarray(points, float); q = np.asarray(polyline, float)
    if p.ndim != 2 or q.ndim != 2 or p.shape[1] != 2 or q.shape[1] != 2 or len(q) < 2 or not np.isfinite(p).all() or not np.isfinite(q).all():
        raise ValueError("Finite pixel arrays required")
    v = np.diff(q, axis=0); length2 = np.sum(v*v, axis=1)
    if np.any(length2 <= 0):
        raise ValueError("Degenerate polyline segments")
    alpha = np.clip(np.einsum("nsi,si->ns",p[:,None]-q[:-1],v)/length2, 0, 1)
    nearest = q[:-1]+alpha[...,None]*v
    return np.sqrt(np.min(np.sum((p[:,None]-nearest)**2, axis=-1), axis=1))


def extent_report(source_uv, model_visible_uv, *, evidence: ExtentEvidence, radius_px: float) -> dict:
    """Validation independent from objective averaging; no invented acceptance gate."""
    evidence.checked()
    if radius_px <= 0 or not np.isfinite(radius_px):
        raise ValueError("Source-specific positive uncertainty required")
    def stats(x):
        return dict(median_px=float(np.median(x)), p95_px=float(np.quantile(x,.95)), maximum_px=float(np.max(x)),
            outside_source_band_fraction=float(np.mean(x>radius_px)))
    forward = directed_polyline_distances(source_uv, model_visible_uv)
    out = dict(source_to_model=stats(forward), complete_extent_admitted=evidence.can_constrain_full_extent(),
        source_pts=int(evidence.pts), temporal_track_id=evidence.temporal_track_id,
        source_range_is_not_ground_truth=True, physical_foot=evidence.physical_foot)
    if evidence.can_constrain_full_extent():
        out['model_to_source'] = stats(directed_polyline_distances(model_visible_uv, source_uv))
    return out
