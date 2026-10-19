"""Opt-in converged Brown rays, without changing source pixels or camera state.

The legacy OpenCV default inverse can stop before closing an otherwise valid
ray. Keep the exact pinhole path; for Brown optics use a converged OpenCV
inverse, check the existing central forward domain, and use the independently
implemented central continuation inverse for difficult rays. Invalid rays stay
NaN so existing factor-specific domain handling keeps its original row IDs.

Installation affects only a newly launched Python process. Running solves and
the historical implementations on disk are deliberately untouched.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys
from types import SimpleNamespace

from runtime.bootstrap import add_geometry_dependencies
add_geometry_dependencies()
import cv2
import numpy as np

from projection import project_central_valid
from rinkspace.calibration_companion.wide_angle_camera import brown_central_inverse


_COUNTS = Counter()
_INSTALLED = None
_PIXEL_TOLERANCE = 1e-7


def precise_world_rays(camera, pixels, *, _verified_projection=None):
    """Optionally expose this call's exact final probe projection for local reuse.

    The private output list is ephemeral, never a cross-state cache. Pinhole
    and partially nonfinite probes supply no reusable projection. Consumers
    must compare exact probe shape/bytes before reuse.
    """
    if _verified_projection is not None:
        _verified_projection.clear()
    c = np.asarray(camera, np.float64)
    xy = np.ascontiguousarray(np.asarray(pixels, np.float64).reshape(-1, 2))
    if (c.shape != (25,) or not np.isfinite(c).all()
            or np.any(np.abs(c[23:25]) > 1e-12)
            or not np.isfinite(xy).all()):
        raise ValueError('Brown inverse needs finite camera25/pixels and zero sensor tilt')
    if not len(xy):
        return np.empty((0, 3), np.float64)
    fx, fy = np.exp(c[6:8])
    deskewed = xy.copy()
    deskewed[:, 0] -= c[10]*(xy[:, 1]-c[9])/fy
    rotation = cv2.Rodrigues(np.ascontiguousarray(c[3:6]))[0]
    _COUNTS['calls'] += 1
    _COUNTS['rays'] += len(xy)
    if np.max(np.abs(c[11:23])) <= 1e-14:
        normalized = np.column_stack(((deskewed[:, 0]-c[8])/fx,
                                       (deskewed[:, 1]-c[9])/fy))
        _COUNTS['pinhole_rays'] += len(xy)
        return np.ascontiguousarray(np.column_stack((normalized, np.ones(len(xy)))) @ rotation)
    intrinsic = np.array([[fx, 0., c[8]], [0., fy, c[9]], [0., 0., 1.]])
    normalized = cv2.undistortPointsIter(deskewed.reshape(-1, 1, 2), intrinsic,
        np.ascontiguousarray(c[11:25]), None, None,
        (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-12)).reshape(-1, 2)

    def check(n):
        rays = np.column_stack((n, np.ones(len(n)))) @ rotation
        good = np.isfinite(rays).all(axis=1)
        indices = np.flatnonzero(good)
        if len(indices):
            probe = np.ascontiguousarray(c[:3]+100.*rays[indices])
            projected, depth, central = project_central_valid(c, probe)
            if _verified_projection is not None and len(indices) == len(rays):
                _verified_projection[:] = [((probe.shape, probe.tobytes()),
                                            (projected, depth, central))]
            good[indices] = (central & np.isfinite(projected).all(axis=1)
                & (np.linalg.norm(projected-xy[indices], axis=1) <= _PIXEL_TOLERANCE))
        return rays, good

    rays, good = check(normalized)
    fallback = np.flatnonzero(~good)
    if len(fallback):
        _COUNTS['central_continuation_rays'] += len(fallback)
        inverse_camera = SimpleNamespace(K_px=intrinsic, distortion_opencv=c[11:23])
        inverse = brown_central_inverse(inverse_camera, deskewed[fallback],
            continuation_steps=8, max_iterations=48, tolerance_px=1e-9)
        normalized[fallback] = inverse['normalized_xy']
        rays, good = check(normalized)
        good[fallback] &= inverse['valid']
    _COUNTS['invalid_rays'] += int((~good).sum())
    rays[~good] = np.nan
    return np.ascontiguousarray(rays)


def install():
    """Replace both original inverse entry points and already-imported aliases."""
    global _INSTALLED
    if _INSTALLED is not None:
        return metadata()
    import reference_board_pair
    import track_sequence
    originals = (reference_board_pair._brown_world_rays, track_sequence.fast_world_rays)
    root = Path(__file__).resolve().parents[2]
    bindings = []
    for module in tuple(sys.modules.values()):
        path = getattr(module, '__file__', None)
        if path is None or not Path(path).resolve().is_relative_to(root):
            continue
        for name, value in tuple(vars(module).items()):
            if any(value is original for original in originals):
                setattr(module, name, precise_world_rays)
                bindings.append(f'{module.__name__}.{name}')
    if (reference_board_pair._brown_world_rays is not precise_world_rays
            or track_sequence.fast_world_rays is not precise_world_rays):
        raise RuntimeError('Both physical inverse entry points must use the new implementation')
    _INSTALLED = sorted(bindings)
    return metadata()


def metadata():
    return dict(schema='rinkspace.precise-central-brown-inverse.v1',
        installed=_INSTALLED is not None, bindings=_INSTALLED,
        pixel_roundtrip_tolerance=_PIXEL_TOLERANCE,
        opencv_iteration_limit=100, opencv_epsilon=1e-12,
        fallback='existing central continuation inverse; 8 stages, 48 iterations, 1e-9 px',
        source_uv_changed=False, camera_parameterization_changed=False,
        objective_note='Physical inverse corrected; not bitwise-equivalent to the old finite-iteration residual',
        counts=dict(_COUNTS))
