"""Fixed-centre raw-pixel ray transfer with a separate Brown12 lens per frame.

The API mirrors ``ray_motion.predict_pairs``. Camera vectors are 25-element
``C, rvec, log_fx, log_fy, cx, cy, skew, Brown12, tauX, tauY`` rows. The
optical centre must be shared, while K, rotation and Brown12 may all change.
Source inversions and target projections are grouped by frame so each Brown
map is evaluated in a vectorized batch. Invalid rays have NaN predictions.
"""

from __future__ import annotations

import cv2
import numpy as np

from ray_motion import (  # noqa: E402
    _brown_map_jacobian, _central_domain, invert_brown_batch,
    predict_pairs as _predict_shared_brown,
)


def _groups(indices: np.ndarray):
    """Yield each frame index and the pair rows that refer to it."""
    order = np.argsort(indices, kind="stable")
    values, starts = np.unique(indices[order], return_index=True)
    ends = np.r_[starts[1:], len(order)]
    for value, start, end in zip(values, starts, ends):
        yield int(value), order[start:end]


def predict_pairs(camera_vectors: np.ndarray, from_index: np.ndarray,
                  to_index: np.ndarray, xy_from: np.ndarray, *,
                  inverse_initial: np.ndarray | None = None,
                  tolerance_px: float = 1e-10) -> dict:
    """Transfer source raw pixels to target raw pixels through shared-centre rays.

    ``inverse_initial`` may be the preceding call's ``normalized_from`` in
    identical pair order. A failed warm start is retried by the central-branch
    inverse. ``inverse_error_px`` reports source pixel round-trip error for
    diagnostics; ``valid`` includes source inversion, positive target depth,
    and the target lens's axis-connected central domain.
    """
    cameras = np.asarray(camera_vectors, float)
    index_from = np.asarray(from_index, int).reshape(-1)
    index_to = np.asarray(to_index, int).reshape(-1)
    pixels = np.asarray(xy_from, float).reshape(-1, 2)
    n = len(pixels)
    if cameras.ndim != 2 or cameras.shape[1] != 25 or not len(cameras) or not np.isfinite(cameras).all():
        raise ValueError("Expected nonempty finite camera_vectors[T,25]")
    if len(index_from) != n or len(index_to) != n or (n and (
            min(index_from.min(), index_to.min()) < 0 or
            max(index_from.max(), index_to.max()) >= len(cameras))):
        raise ValueError("Pair index arrays differ from pixels or exceed camera range")
    if np.max(np.abs(cameras[:, 23:25])) > 1e-12:
        raise ValueError("Brown sensor tilt is unsupported")
    if not np.allclose(cameras[:, :3], cameras[0, :3], atol=1e-9, rtol=0):
        raise ValueError("Relative-ray helper requires one fixed optical centre")
    if not np.isfinite(tolerance_px) or tolerance_px <= 0:
        raise ValueError("Positive finite inverse tolerance required")
    warm = None
    if inverse_initial is not None:
        warm = np.asarray(inverse_initial, float)
        if warm.shape != (n, 2):
            raise ValueError("Warm rays must have the same shape as input pixels")
    # Preserve the faster shared-lens path without treating even a very small
    # genuine per-frame Brown change as constant during finite differencing.
    if np.all(cameras[:, 11:23] == cameras[0, 11:23]):
        return _predict_shared_brown(cameras, index_from, index_to, pixels,
                                     inverse_initial=warm, tolerance_px=tolerance_px)
    predicted = np.full((n, 2), np.nan)
    if n == 0:
        return {"predicted_uv": predicted, "valid": np.empty(0, bool),
                "inverse_error_px": np.empty(0),
                "normalized_from": np.empty((0, 2)),
                "normalized_to": np.empty((0, 2))}

    fx0 = np.exp(cameras[index_from, 6])
    fy0 = np.exp(cameras[index_from, 7])
    skew0 = cameras[index_from, 10]
    y_distorted = (pixels[:, 1] - cameras[index_from, 9]) / fy0
    x_distorted = (pixels[:, 0] - cameras[index_from, 8] - skew0*y_distorted) / fx0
    distorted = np.column_stack((x_distorted, y_distorted))
    normalized_from = np.zeros((n, 2))
    inverse_valid = np.zeros(n, bool)
    inverse_error = np.full(n, np.inf)
    for frame, rows in _groups(index_from):
        initial = None if warm is None else warm[rows]
        normalized_from[rows], inverse_valid[rows], inverse_error[rows] = invert_brown_batch(
            distorted[rows], cameras[frame, 11:23], fx0[rows], fy0[rows],
            skew0[rows], inverse_initial=initial, tolerance_px=tolerance_px)

    rotations = np.stack([cv2.Rodrigues(camera[3:6])[0] for camera in cameras])
    pair_keys = np.column_stack((index_from, index_to))
    unique_pairs, pair_inverse = np.unique(pair_keys, axis=0, return_inverse=True)
    relative = np.stack([rotations[j] @ rotations[i].T for i, j in unique_pairs])
    rays = np.column_stack((normalized_from, np.ones(n)))
    transformed = np.einsum("nij,nj->ni", relative[pair_inverse], rays)
    depth_valid = np.isfinite(transformed).all(axis=1) & (transformed[:, 2] > 1e-9)
    normalized_to = np.full((n, 2), np.nan)
    normalized_to[depth_valid] = transformed[depth_valid, :2] / transformed[depth_valid, 2, None]

    target_domain = np.zeros(n, bool)
    for frame, rows in _groups(index_to):
        normalized = normalized_to[rows]
        finite = np.isfinite(normalized).all(axis=1)
        safe = np.where(finite[:, None], normalized, 0.)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            mapped = _brown_map_jacobian(safe, cameras[frame, 11:23])[0]
            target_domain[rows] = finite & _central_domain(safe, cameras[frame, 11:23])
        fx1, fy1 = np.exp(cameras[frame, 6:8])
        predicted[rows, 0] = fx1*mapped[:, 0] + cameras[frame, 10]*mapped[:, 1] + cameras[frame, 8]
        predicted[rows, 1] = fy1*mapped[:, 1] + cameras[frame, 9]

    valid = inverse_valid & depth_valid & target_domain & np.isfinite(predicted).all(axis=1)
    predicted[~valid] = np.nan
    return {"predicted_uv": predicted, "valid": valid,
            "inverse_error_px": inverse_error,
            "normalized_from": normalized_from,
            "normalized_to": normalized_to}
