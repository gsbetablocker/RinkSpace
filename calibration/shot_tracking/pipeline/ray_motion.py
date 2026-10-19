"""Vectorized fixed-optical-centre Brown ray transfer for local optimization.

`predict_pairs` maps source raw pixels to target raw pixels through one shared
Brown12 lens and changing K/R. It assumes the optical centre is fixed; it is
not a pivot-offset or arbitrary-translation camera. The inverse is an analytic
Jacobian Newton solve with axis-to-target continuation on cold starts. Pass
the returned `normalized_from` as `inverse_initial` on nearby optimizer calls.
Strict final acceptance must still use `camera.world_rays` /
`physical_projection.verify_camera` and original-video overlays.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
WORK = ROOT / "calibration" / "work"
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
from rinkspace.calibration_companion.wide_angle_camera import (  # noqa: E402
    _brown_map_jacobian, rational_minimum_denominator,
)

_EIGEN_FLOOR = 1e-9
_DENOM_FLOOR = 1e-9
_AXIS_STATIONS = np.linspace(0., 1., 9)


def _pixel_error(mapped: np.ndarray, target: np.ndarray,
                 fx: np.ndarray, fy: np.ndarray, skew: np.ndarray) -> np.ndarray:
    delta = mapped - target
    return np.hypot(fx * delta[:, 0] + skew * delta[:, 1], fy * delta[:, 1])


def _central_domain(xy: np.ndarray, distortion: np.ndarray) -> np.ndarray:
    """Conservative axis-connected positive-Jacobian branch check."""
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        minimum_denominator = rational_minimum_denominator(xy, distortion)
        path = (xy[:, None, :] * _AXIS_STATIONS[None, :, None]).reshape(-1, 2)
        minimum_eigen = _brown_map_jacobian(path, distortion)[-1]
    minimum_eigen = minimum_eigen.reshape(len(xy), len(_AXIS_STATIONS)).min(axis=1)
    return (np.isfinite(minimum_denominator) & (minimum_denominator > _DENOM_FLOOR)
            & np.isfinite(minimum_eigen) & (minimum_eigen > _EIGEN_FLOOR))


def _newton_stage(xy: np.ndarray, goal: np.ndarray, distortion: np.ndarray,
                  fx: np.ndarray, fy: np.ndarray, skew: np.ndarray,
                  alive: np.ndarray, tolerance_px: float, max_iterations: int) -> np.ndarray:
    """Update `xy` in place while preserving a finite residual for failures."""
    converged = ~alive.copy()
    for _ in range(max_iterations):
        indices = np.flatnonzero(alive & ~converged)
        if not len(indices):
            break
        z = xy[indices]
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            mapped, a, b, c, d, eigen = _brown_map_jacobian(z, distortion)
        error = mapped - goal[indices]
        pixel_error = _pixel_error(mapped, goal[indices],
                                   fx[indices], fy[indices], skew[indices])
        done = np.isfinite(pixel_error) & (pixel_error <= tolerance_px)
        converged[indices[done]] = True
        indices, z, error, pixel_error, a, b, c, d, eigen = [
            row[~done] for row in (indices, z, error, pixel_error, a, b, c, d, eigen)]
        if not len(indices):
            continue
        determinant = a*d-b*c
        valid = (np.isfinite(determinant) & (determinant > 1e-16)
                 & np.isfinite(eigen) & (eigen > _EIGEN_FLOOR))
        alive[indices[~valid]] = False
        indices, z, error, pixel_error, a, b, c, d, determinant = [
            row[valid] for row in (indices, z, error, pixel_error, a, b, c, d, determinant)]
        if not len(indices):
            continue
        step = np.column_stack(((d*error[:, 0]-b*error[:, 1])/determinant,
                                (a*error[:, 1]-c*error[:, 0])/determinant))
        step *= np.minimum(1., .18/np.maximum(np.linalg.norm(step, axis=1), 1e-30))[:, None]
        factors = np.ones(len(indices))
        accepted = np.zeros(len(indices), bool)
        for _ in range(10):
            todo = np.flatnonzero(~accepted)
            if not len(todo):
                break
            proposed = z[todo] - factors[todo, None] * step[todo]
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                mapped_new, _, _, _, _, eig_new = _brown_map_jacobian(proposed, distortion)
            candidate_error = _pixel_error(mapped_new, goal[indices[todo]],
                                           fx[indices[todo]], fy[indices[todo]],
                                           skew[indices[todo]])
            improve = (np.isfinite(candidate_error) & (candidate_error < pixel_error[todo])
                       & np.isfinite(eig_new) & (eig_new > _EIGEN_FLOOR))
            if np.any(improve):
                take = todo[improve]
                xy[indices[take]] = proposed[improve]
                accepted[take] = True
            factors[todo[~improve]] *= .5
        alive[indices[~accepted]] = False
    alive &= converged
    return alive


def invert_brown_batch(target_distorted: np.ndarray, distortion: np.ndarray,
                       fx: np.ndarray, fy: np.ndarray, skew: np.ndarray,
                       *, inverse_initial: np.ndarray | None = None,
                       tolerance_px: float = 1e-10,
                       continuation_steps: int = 8,
                       max_iterations: int = 12) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Central Brown inverse for one batch with potentially different K per ray.

    Returns normalized rays, central validity, and distorted-pixel inverse
    error. A caller may warm-start using rays from the prior optimizer call.
    Failed warm-start rays are retried from the axis with homotopy.
    """
    target = np.asarray(target_distorted, float).reshape(-1, 2)
    d = np.asarray(distortion, float).reshape(-1)
    n = len(target)
    fx, fy, skew = [np.broadcast_to(np.asarray(x, float), (n,)) for x in (fx, fy, skew)]
    if d.shape != (12,) or min(continuation_steps, max_iterations) < 1:
        raise ValueError("Brown12 and positive iteration counts required")
    if tolerance_px <= 0 or not np.isfinite(tolerance_px):
        raise ValueError("Positive finite inverse tolerance required")
    if n == 0:
        return np.empty((0, 2)), np.empty(0, bool), np.empty(0)
    alive = (np.isfinite(target).all(axis=1) & np.isfinite(fx+fy+skew)
             & (fx > 0) & (fy > 0))
    xy = np.zeros_like(target)
    if inverse_initial is not None:
        warm = np.asarray(inverse_initial, float).reshape(-1, 2)
        if warm.shape != target.shape:
            raise ValueError("Warm rays must have the same shape as input pixels")
        warm_good = alive & np.isfinite(warm).all(axis=1)
        xy[warm_good] = warm[warm_good]
        warm_alive = warm_good.copy()
        _newton_stage(xy, target, d, fx, fy, skew, warm_alive,
                      tolerance_px, max_iterations)
        warm_alive &= _central_domain(xy, d)
    else:
        warm_alive = np.zeros(n, bool)
    cold = alive & ~warm_alive
    if np.any(cold):
        # Maintain full batch indexes for per-sample K; only cold rows are
        # advanced, and failure at any intermediate target stays invalid.
        xy[cold] = 0.
        for stage in range(1, continuation_steps + 1):
            goal = target * (stage / continuation_steps)
            cold = _newton_stage(xy, goal, d, fx, fy, skew, cold,
                                 tolerance_px, max_iterations)
            if not np.any(cold):
                break
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        mapped = _brown_map_jacobian(xy, d)[0]
    inverse_error = _pixel_error(mapped, target, fx, fy, skew)
    # Warm successes have already passed the full branch check; only newly
    # recovered cold rays need it here.
    domain = warm_alive.copy()
    cold_indices = np.flatnonzero(cold)
    if len(cold_indices):
        domain[cold_indices] = _central_domain(xy[cold_indices], d)
    valid = (alive & (warm_alive | cold) & domain
             & np.isfinite(inverse_error) & (inverse_error <= tolerance_px * 2.))
    return xy, valid, inverse_error


def predict_pairs(camera_vectors: np.ndarray, from_index: np.ndarray,
                  to_index: np.ndarray, xy_from: np.ndarray, *,
                  distortion: np.ndarray | None = None,
                  inverse_initial: np.ndarray | None = None,
                  tolerance_px: float = 1e-10) -> dict:
    """Predict many raw-pixel pair targets through fixed-centre relative rays.

    Camera vectors use `C(3), rvec_world_to_camera(3), log_fx, log_fy,
    cx, cy, skew, Brown12, tauX=0, tauY=0`. All frames must share one
    optical centre and one distortion array; K and rotation may vary.
    Invalid rays return NaN predictions, with a finite 2D inverse residual
    available for optimizer penalties. Output includes `normalized_from` for
    warm-starting the next call with the same pair ordering. The default
    inverse tolerance is tighter than finite-difference pixel changes in a
    sparse least-squares Jacobian; a loose 1e-5 px stop can flatten K columns.
    """
    cameras = np.asarray(camera_vectors, float)
    index_from = np.asarray(from_index, int).reshape(-1)
    index_to = np.asarray(to_index, int).reshape(-1)
    pixels = np.asarray(xy_from, float).reshape(-1, 2)
    n = len(pixels)
    if cameras.ndim != 2 or cameras.shape[1] != 25 or not np.isfinite(cameras).all():
        raise ValueError("Expected finite camera_vectors[T,25]")
    if len(index_from) != n or len(index_to) != n or (n and (
            min(index_from.min(), index_to.min()) < 0 or
            max(index_from.max(), index_to.max()) >= len(cameras))):
        raise ValueError("Pair index arrays differ from pixels or exceed camera range")
    if np.max(np.abs(cameras[:, 23:25])) > 1e-12:
        raise ValueError("Brown sensor tilt is unsupported")
    if not np.allclose(cameras[:, :3], cameras[0, :3], atol=1e-9, rtol=0):
        raise ValueError("Relative-ray helper requires one fixed optical centre")
    if distortion is None:
        d = cameras[0, 11:23]
    else:
        d = np.asarray(distortion, float).reshape(-1)
    if d.shape != (12,) or not np.allclose(cameras[:, 11:23], d, atol=1e-12, rtol=0):
        raise ValueError("All cameras must share the supplied fixed Brown12 lens")
    result_uv = np.full((n, 2), np.nan)
    if n == 0:
        return {"predicted_uv": result_uv, "valid": np.empty(0, bool),
                "inverse_error_px": np.empty(0),
                "normalized_from": np.empty((0, 2)),
                "normalized_to": np.empty((0, 2))}
    first = cameras[index_from]
    second = cameras[index_to]
    fx0, fy0 = np.exp(first[:, 6]), np.exp(first[:, 7])
    fy1, fx1 = np.exp(second[:, 7]), np.exp(second[:, 6])
    y_d = (pixels[:, 1] - first[:, 9]) / fy0
    x_d = (pixels[:, 0] - first[:, 8] - first[:, 10]*y_d) / fx0
    normalized_from, invert_valid, inverse_error = invert_brown_batch(
        np.column_stack((x_d, y_d)), d, fx0, fy0, first[:, 10],
        inverse_initial=inverse_initial, tolerance_px=tolerance_px)
    rotations = np.stack([cv2.Rodrigues(camera[3:6])[0] for camera in cameras])
    # R_to @ R_from.T maps source-camera rays to target-camera rays.
    pair_key = np.column_stack((index_from, index_to))
    unique_pairs, inverse_pair = np.unique(pair_key, axis=0, return_inverse=True)
    relative = np.stack([rotations[j] @ rotations[i].T for i, j in unique_pairs])
    rays = np.column_stack((normalized_from, np.ones(n)))
    transformed = np.einsum("nij,nj->ni", relative[inverse_pair], rays)
    depth_valid = np.isfinite(transformed).all(axis=1) & (transformed[:, 2] > 1e-9)
    normalized_to = np.full((n, 2), np.nan)
    normalized_to[depth_valid] = transformed[depth_valid, :2] / transformed[depth_valid, 2, None]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        mapped_to = _brown_map_jacobian(np.nan_to_num(normalized_to, nan=0.), d)[0]
    result_uv[:, 0] = fx1 * mapped_to[:, 0] + second[:, 10] * mapped_to[:, 1] + second[:, 8]
    result_uv[:, 1] = fy1 * mapped_to[:, 1] + second[:, 9]
    output_domain = _central_domain(np.nan_to_num(normalized_to, nan=0.), d)
    valid = invert_valid & depth_valid & output_domain & np.isfinite(result_uv).all(axis=1)
    result_uv[~valid] = np.nan
    return {"predicted_uv": result_uv, "valid": valid,
            "inverse_error_px": inverse_error,
            "normalized_from": normalized_from,
            "normalized_to": normalized_to}
