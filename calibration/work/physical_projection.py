"""Central-branch Brown projection for the three-frame refit.

The 25-element camera vector follows v0.24: centre(3), Rodrigues(3), log
fx/fy, cx/cy/skew, OpenCV Brown14. Tilt tauX/tauY must be fixed at zero by
the new solver. Invalid samples remain in place as NaNs; no curve may bridge
an invalid sample or a separate path boundary. This is an optical validity
test, not a test that the projected rink geometry matches the source image.
"""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Sequence

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from rinkspace.calibration_companion.wide_angle_camera import (  # noqa: E402
    _brown_map_jacobian, brown_central_inverse, rational_minimum_denominator,
)


_DEPTH_FLOOR = .05
_DENOM_FLOOR = 1e-8
_EIGEN_FLOOR = 1e-8
_AXIS_STATIONS = np.linspace(0.0, 1.0, 9)


def _camera(v: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    vector = np.asarray(v, dtype=float).reshape(-1)
    if vector.shape != (25,) or not np.isfinite(vector).all():
        raise ValueError("Camera vector must contain 25 finite numbers")
    if np.any(np.abs(vector[23:25]) > 1e-12):
        raise ValueError("Brown tilt tauX/tauY must be fixed to zero")
    if not np.isfinite(np.exp(vector[6:8])).all():
        raise ValueError("Camera focal lengths are not finite")
    return vector, vector[11:23]


def _rays(v: np.ndarray, xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rotation = cv2.Rodrigues(np.ascontiguousarray(v[3:6]))[0]
    camera_xyz = np.ascontiguousarray((rotation @ (xyz - v[:3]).T).T)
    depth = camera_xyz[:, 2]
    normalized = np.full((len(xyz), 2), np.nan)
    forward = np.isfinite(camera_xyz).all(axis=1) & (depth > _DEPTH_FLOOR)
    normalized[forward] = camera_xyz[forward, :2] / depth[forward, None]
    return normalized, depth, forward


def safe_project(v: Sequence[float], xyz: Sequence[Sequence[float]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fast forward projection on the axis-connected positive-Jacobian branch.

    The rational denominator minimum along the entire axis-to-ray interval is
    exact for Brown's cubic denominator. The symmetric Jacobian is sampled at
    nine radial stations; ``verify_camera`` performs the slower independent
    central inverse check before accepting a fitted camera. Out-of-frame rays
    remain valid if the optical map is physically valid. NaN UV marks invalid
    rays, so a matcher using adjacent finite samples cannot reuse a ghost.
    """
    camera, distortion = _camera(v)
    points = np.asarray(xyz, dtype=float).reshape(-1, 3)
    if not np.isfinite(points).all():
        raise ValueError("World coordinates must be finite")
    uv = np.full((len(points), 2), np.nan)
    if not len(points):
        return uv, np.empty(0), np.empty(0, dtype=bool)
    normalized, depth, forward = _rays(camera, points)
    valid = np.zeros(len(points), dtype=bool)
    indices = np.flatnonzero(forward)
    if not len(indices):
        return uv, depth, valid
    rays = normalized[indices]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        denominator = rational_minimum_denominator(rays, distortion)
    domain = np.isfinite(denominator) & (denominator > _DENOM_FLOOR)
    indices, rays = indices[domain], rays[domain]
    if not len(indices):
        return uv, depth, valid
    # A positive endpoint Jacobian alone can admit a disconnected outer
    # branch. Test the path from the optical axis to every projected ray.
    path = (rays[:, None, :] * _AXIS_STATIONS[None, :, None]).reshape(-1, 2)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        _, _, _, _, _, eigen = _brown_map_jacobian(path, distortion)
    axis_eigen = eigen.reshape(len(indices), len(_AXIS_STATIONS)).min(axis=1)
    domain = np.isfinite(axis_eigen) & (axis_eigen > _EIGEN_FLOOR)
    indices, rays = indices[domain], rays[domain]
    if not len(indices):
        return uv, depth, valid
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        mapped = _brown_map_jacobian(rays, distortion)[0]
        focal_x, focal_y = math.exp(camera[6]), math.exp(camera[7])
        projected = np.column_stack([
            focal_x * mapped[:, 0] + camera[10] * mapped[:, 1] + camera[8],
            focal_y * mapped[:, 1] + camera[9],
        ])
    finite = np.isfinite(projected).all(axis=1)
    uv[indices[finite]] = projected[finite]
    valid[indices[finite]] = True
    return uv, depth, valid


def nearest_valid_vectors(
    observed_uv: Sequence[Sequence[float]],
    projected_uv: Sequence[Sequence[float]],
    valid: Sequence[bool],
    *,
    path_offsets: Sequence[int] | None = None,
    max_segment_px: float = 1200.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Nearest residuals and selected segment IDs without crossing bad/gap edges."""
    observed = np.asarray(observed_uv, float).reshape(-1, 2)
    projected = np.asarray(projected_uv, float).reshape(-1, 2)
    good_points = np.asarray(valid, bool).reshape(-1)
    if len(projected) != len(good_points) or not np.isfinite(observed).all():
        raise ValueError("Observed/projected points or validity mask are invalid")
    if not np.isfinite(max_segment_px) or max_segment_px <= 0:
        raise ValueError("max_segment_px must be finite and positive")
    good_edges = good_points[:-1] & good_points[1:]
    if path_offsets is not None:
        offsets = [int(x) for x in path_offsets]
        if not offsets or offsets[0] != 0 or offsets[-1] != len(projected) or any(
                a > b for a, b in zip(offsets, offsets[1:])):
            raise ValueError("Invalid path offsets")
        cuts = np.asarray([x for x in offsets[1:-1] if 0 < x < len(projected)], int)
        good_edges[cuts - 1] = False
    good_edges &= np.isfinite(projected[:-1]).all(axis=1)
    good_edges &= np.isfinite(projected[1:]).all(axis=1)
    lengths = np.linalg.norm(projected[1:] - projected[:-1], axis=1)
    good_edges &= np.isfinite(lengths) & (lengths < max_segment_px)
    indices = np.flatnonzero(good_edges)
    if not len(indices):
        return np.full_like(observed, 1e3), np.full(len(observed), -1, dtype=int)
    starts = projected[indices]
    directions = projected[indices + 1] - starts
    denominator = np.maximum(np.sum(directions * directions, axis=1), 1e-20)
    t = np.einsum("nsi,si->ns", observed[:, None, :] - starts[None, :, :], directions)
    t = np.clip(t / denominator[None, :], 0.0, 1.0)
    closest = starts[None, :, :] + t[:, :, None] * directions[None, :, :]
    differences = observed[:, None, :] - closest
    choice = np.argmin(np.sum(differences * differences, axis=2), axis=1)
    return differences[np.arange(len(observed)), choice], indices[choice]


def verify_camera(
    v: Sequence[float],
    xyz_curves: Sequence[Sequence[Sequence[float]]] | np.ndarray,
    *,
    image_size_wh: tuple[int, int] = (1920, 1080),
    sensor_grid: tuple[int, int] = (17, 11),
    inverse_tolerance_px: float = .05,
) -> dict:
    """Independent forward/inverse check on curves and raw sensor rays.

    Returns counts and worst inverse errors. A candidate passes only if every
    supplied curve sample is on the physical branch and every sensor grid ray
    has a central inverse at the chosen tolerance. Pass .005 with the full
    intended curve domains for the final strict QA used by validate_fit.
    """
    camera, distortion = _camera(v)
    if isinstance(xyz_curves, np.ndarray) and xyz_curves.ndim == 2:
        curves = [xyz_curves]
    else:
        curves = list(xyz_curves)
    width, height = map(int, image_size_wh)
    columns, rows = map(int, sensor_grid)
    if min(width, height) <= 0 or min(columns, rows) < 2 or inverse_tolerance_px <= 0:
        raise ValueError("Invalid image/grid size or inverse tolerance")
    fx, fy = math.exp(camera[6]), math.exp(camera[7])
    intrinsic = np.array([[fx, 0.0, camera[8]], [0.0, fy, camera[9]], [0.0, 0.0, 1.0]])
    inverse_camera = SimpleNamespace(K_px=intrinsic, distortion_opencv=distortion)

    def inverse_result(pixels: np.ndarray, expected_rays: np.ndarray | None) -> dict:
        if not len(pixels):
            return {"count": 0, "inverse_valid_count": 0,
                    "max_reprojection_error_px": None, "max_ray_error_px": None}
        deskewed = pixels.copy()
        deskewed[:, 0] -= camera[10] * (pixels[:, 1] - camera[9]) / fy
        result = brown_central_inverse(inverse_camera, deskewed,
                                       tolerance_px=inverse_tolerance_px)
        good = result["valid"]
        ray_error = (np.linalg.norm((result["normalized_xy"] - expected_rays) *
                                    [fx, fy], axis=1) if expected_rays is not None else None)
        return {"count": len(pixels), "inverse_valid_count": int(good.sum()),
                "max_reprojection_error_px": float(np.max(result["reprojection_error_px"])),
                "max_ray_error_px": (float(np.max(ray_error)) if ray_error is not None else None),
                "all_valid": bool(np.all(good) and (ray_error is None or
                    np.all(ray_error <= inverse_tolerance_px)))}

    curve_rows = []
    for curve in curves:
        points = np.asarray(curve, float).reshape(-1, 3)
        uv, depth, valid = safe_project(camera, points)
        normalized, _, _ = _rays(camera, points)
        good_indices = np.flatnonzero(valid)
        inverse = inverse_result(uv[good_indices], normalized[good_indices])
        curve_rows.append({"count": len(points), "fast_valid_count": int(valid.sum()),
                           "invalid_count": int((~valid).sum()),
                           "min_depth": float(np.min(depth)) if len(depth) else None,
                           "inverse": inverse,
                           "all_valid": bool(len(points) and np.all(valid) and inverse.get("all_valid", True))})

    xx, yy = np.meshgrid(np.linspace(0, width - 1, columns),
                         np.linspace(0, height - 1, rows))
    sensor_pixels = np.column_stack([xx.ravel(), yy.ravel()])
    sensor = inverse_result(sensor_pixels, None)
    return {"curves": curve_rows, "sensor": sensor,
            "all_curves_valid": all(r["all_valid"] for r in curve_rows),
            "sensor_valid": sensor["all_valid"]}


@lru_cache(maxsize=96)
def _sensor_coverage_cached(
    camera_values: tuple[float, ...], image_size_wh: tuple[int, int],
    sensor_grid: tuple[int, int], pixel_scale: float,
) -> tuple[float, ...]:
    """Internal cache mainly saves frozen cameras during finite differences."""
    camera, distortion = _camera(camera_values)
    width, height = image_size_wh
    columns, rows = sensor_grid
    fx, fy = math.exp(camera[6]), math.exp(camera[7])
    xx, yy = np.meshgrid(np.linspace(0, width - 1, columns),
                         np.linspace(0, height - 1, rows))
    raw = np.column_stack([xx.ravel(), yy.ravel()])
    deskewed = raw.copy()
    deskewed[:, 0] -= camera[10] * (raw[:, 1] - camera[9]) / fy
    intrinsic = np.array([[fx, 0.0, camera[8]], [0.0, fy, camera[9]], [0.0, 0.0, 1.0]])
    inverse_camera = SimpleNamespace(K_px=intrinsic, distortion_opencv=distortion)
    inverse = brown_central_inverse(inverse_camera, deskewed,
                                   continuation_steps=8, max_iterations=24,
                                   tolerance_px=.005)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        mapped = _brown_map_jacobian(inverse["normalized_xy"], distortion)[0]
        projected = np.column_stack([fx * mapped[:, 0] + camera[8],
                                     fy * mapped[:, 1] + camera[9]])
        reprojection = (projected - deskewed) / pixel_scale
        # This barrier remains parameter-dependent when Newton has not yet
        # progressed from the axis, instead of returning a constant penalty.
        ideal = np.column_stack([(deskewed[:, 0] - camera[8]) / fx,
                                 (deskewed[:, 1] - camera[9]) / fy])
        path = ideal[:, None, :] * np.linspace(0.0, 1.0, 5)[None, :, None]
        eigen = _brown_map_jacobian(path.reshape(-1, 2), distortion)[-1]
        eigen_min = eigen.reshape(len(raw), 5).min(axis=1)
        denominator = rational_minimum_denominator(ideal, distortion)
        # The ideal undistorted ray can miss a fold of the *recovered* central
        # branch. Guard the actual inverse ray and its axis-connected path with
        # the same .08 Jacobian and .1 denominator margins as sensor_guard's
        # coarse ideal grid, so optimization is penalized before inverse loss.
        recovered_path = inverse["normalized_xy"][:, None, :] * np.linspace(
            0.0, 1.0, 9)[None, :, None]
        recovered_eigen = _brown_map_jacobian(
            recovered_path.reshape(-1, 2), distortion)[-1]
        recovered_eigen_min = recovered_eigen.reshape(len(raw), 9).min(axis=1)
        recovered_denominator = rational_minimum_denominator(
            inverse["normalized_xy"], distortion)
        barrier = (200.0 * (np.maximum(.08 - recovered_eigen_min, 0.0) +
                            np.maximum(.1 - recovered_denominator, 0.0)) +
                   5.0 * (np.maximum(1e-5 - eigen_min, 0.0) +
                          np.maximum(1e-5 - denominator, 0.0)))
    # Numeric overflow stays a large but finite, variable residual. Final
    # acceptance always uses verify_camera rather than these solver residuals.
    reprojection = np.nan_to_num(reprojection, nan=1e3, posinf=1e3, neginf=-1e3)
    # A valid central inverse has already passed the declared Newton tolerance.
    # Its remaining reprojection error is solver termination noise, not a
    # sensor-coverage violation; exclude it from finite-difference objectives.
    reprojection[np.asarray(inverse["valid"], bool)] = 0.0
    barrier = np.nan_to_num(barrier, nan=1e3, posinf=1e3, neginf=1e3)
    return tuple(np.r_[reprojection.ravel(), barrier].tolist())


def sensor_coverage_residual(
    v: Sequence[float], *, image_size_wh: tuple[int, int] = (1920, 1080),
    sensor_grid: tuple[int, int] = (5, 3), pixel_scale: float = 10.0,
) -> np.ndarray:
    """Optimizable raw-sensor central-inverse coverage residual.

    Uses a coarse raw-pixel grid during optimization; output is fixed length
    ``3 * columns * rows``. A failed inverse contributes its actual remaining
    two-dimensional reprojection error, not a constant invalidity flag. Valid
    inverse rays contribute zero reprojection guard. The third component at
    each grid location penalizes a noncentral ideal ray.
    Use ``verify_camera`` with a denser grid for final acceptance.
    """
    camera, _ = _camera(v)
    size = tuple(int(x) for x in image_size_wh)
    grid = tuple(int(x) for x in sensor_grid)
    if len(size) != 2 or len(grid) != 2 or min(size) <= 0 or min(grid) < 2:
        raise ValueError("Invalid sensor size or grid")
    if not math.isfinite(pixel_scale) or pixel_scale <= 0:
        raise ValueError("pixel_scale must be finite and positive")
    return np.asarray(_sensor_coverage_cached(tuple(camera.tolist()), size, grid,
                                              float(pixel_scale)), dtype=float)
