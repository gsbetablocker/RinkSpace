"""Coordinate-explicit projective mapping and held-out diagnostics."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import cv2
import numpy as np


def points(value: Any) -> np.ndarray:
    p = np.asarray(value, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 2 or not np.all(np.isfinite(p)):
        raise ValueError("Expected finite Nx2 points")
    return p


def homography(value: Any) -> np.ndarray:
    h = np.asarray(value, dtype=np.float64)
    if h.shape != (3, 3) or not np.all(np.isfinite(h)):
        raise ValueError("Expected finite 3x3 homography")
    norm = np.linalg.norm(h)
    if norm == 0 or abs(np.linalg.det(h / norm)) < 1e-15:
        raise ValueError("Singular or numerically degenerate homography")
    return h / norm


def project(h: Any, xy: Any) -> np.ndarray:
    h = homography(h)
    p = points(xy)
    q = np.c_[p, np.ones(len(p))] @ h.T
    if np.any(np.abs(q[:, 2]) < 1e-10):
        raise ValueError("Projection crosses/approaches the homography horizon")
    return q[:, :2] / q[:, 2:3]


def fit_homography(src: Any, dst: Any, threshold_dst: float = 3.0) -> tuple[np.ndarray, np.ndarray]:
    a, b = points(src), points(dst)
    if len(a) != len(b) or len(a) < 4:
        raise ValueError("At least four paired points required")
    for p in (a, b):
        hull = cv2.convexHull(p.astype(np.float32))
        if cv2.contourArea(hull) <= 1e-5:
            raise ValueError("Collinear control points do not determine a plane map")
    cv2.setRNGSeed(20260919)
    h, inliers = cv2.findHomography(a, b, cv2.RANSAC, threshold_dst, maxIters=5000, confidence=0.999)
    if h is None or inliers is None or np.sum(inliers) < 4:
        raise ValueError("No nondegenerate homography found")
    return homography(h), inliers.ravel().astype(bool)


def source_pixel_errors(h_dst_from_src: Any, src: Any, dst: Any) -> np.ndarray:
    """Errors in SOURCE pixels even when destination units are metres."""
    a, b = points(src), points(dst)
    if len(a) != len(b):
        raise ValueError("Mismatched point counts")
    return np.linalg.norm(project(np.linalg.inv(homography(h_dst_from_src)), b) - a, axis=1)


def heldout_audit(h: Any, train_src: Any, train_ids: list[str], check_src: Any,
                  check_dst: Any, check_ids: list[str], separation_px: float = 8.0) -> dict[str, Any]:
    """Audit only. Passing numeric error checks does not prove coplanarity or metre scale."""
    a, c, d = points(train_src), points(check_src), points(check_dst)
    if len(train_ids) != len(a) or len(check_ids) != len(c) or len(c) != len(d):
        raise ValueError("Landmark ids and coordinate lengths differ")
    if len(set(train_ids)) != len(train_ids) or len(set(check_ids)) != len(check_ids):
        raise ValueError("Duplicate physical landmark ids")
    if set(train_ids) & set(check_ids):
        raise ValueError("Held-out landmarks leaked into fitting")
    if len(c) < 4:
        raise ValueError("At least four separate held-out landmarks required")
    nearest = np.min(np.linalg.norm(c[:, None, :] - a[None, :, :], axis=2), axis=1)
    if np.any(nearest < separation_px):
        raise ValueError("Held-out landmark overlaps a fitting neighborhood")
    err = source_pixel_errors(h, c, d)
    return {"heldout_count": len(c), "median_source_px": float(np.median(err)),
            "p95_source_px": float(np.percentile(err, 95)), "max_source_px": float(np.max(err)),
            "errors_source_px": err.tolist(), "semantic_independence_requires_review": True,
            "metric_scale_verified_by_this_test": False}


@dataclass(frozen=True)
class PlaneMap:
    """Map undistorted source pixels into a named plane coordinate system.

    metric_* levels require externally supplied geometric evidence, not simply
    setting this label. The seed deliberately does not auto-promote calibrations.
    """
    matrix: np.ndarray
    level: str = "projective"
    unit: str = "reference_px"
    support_polygon: np.ndarray | None = None
    map_id: str = "unnamed"

    def __post_init__(self) -> None:
        homography(self.matrix)
        allowed = {"projective", "euclidean_unscaled", "metric_assumed", "metric_verified"}
        if self.level not in allowed:
            raise ValueError(f"Unknown geometry level: {self.level}")
        if self.level.startswith("metric_") and self.unit != "m":
            raise ValueError("Metric maps require unit=m")
        if self.level == "projective" and self.unit != "reference_px":
            raise ValueError("Projective map uses reference_px, not metres")
        if self.level == "euclidean_unscaled" and self.unit != "arbitrary_unit":
            raise ValueError("Unscaled Euclidean map uses arbitrary_unit")
        if self.support_polygon is not None and len(points(self.support_polygon)) < 3:
            raise ValueError("Support polygon needs at least three vertices")

    def map_contacts(self, xy: Any, contact: list[str], allow_extrapolation: bool = False) -> np.ndarray:
        p = points(xy)
        if len(p) != len(contact) or any(s != "on_ice" for s in contact):
            raise ValueError("Only reviewed on-ice points may enter the contact-plane mapping")
        if self.support_polygon is not None and not allow_extrapolation:
            poly = points(self.support_polygon).astype(np.float32)
            if any(cv2.pointPolygonTest(poly, (float(x), float(y)), False) < 0 for x, y in p):
                raise ValueError("Point outside supported region; mark extrapolation explicitly")
        return project(self.matrix, p)

    def require_euclidean(self) -> None:
        if self.level == "projective":
            raise ValueError("Projective stabilization does not establish physical angles, curvature or speed")
