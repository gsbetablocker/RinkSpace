"""Observed static-ice pixel pairs through two exact supplied row cameras.

The source ray is intersected with z=0 afresh at every evaluation. The result is
an eliminated conditional plane point, never an added measured 3D landmark.
Both cameras participate in the derivative. This module selects no source
tracks, weights, temporal basis, camera trend, rig parameter or scene scale.
"""
from __future__ import annotations

from types import SimpleNamespace

import cv2
import numpy as np


def _camera_inventory(cameras, count):
    value = np.asarray(cameras, float)
    if value.shape == (25,):
        value = np.broadcast_to(value, (count, 25))
    if value.shape != (count, 25) or not np.isfinite(value).all():
        raise ValueError("Each measured pixel pair requires its exact finite Brown camera25")
    if np.any(value[:, 15:] != 0.):
        raise ValueError("Current native motion factor requires the original Brown4 camera chart")
    return value


def _plane_point(camera, pixel, tolerance):
    from rinkspace.calibration_companion.wide_angle_camera import brown_central_inverse
    fx, fy = np.exp(camera[6:8])
    unskewed = np.asarray(pixel, float).reshape(1, 2).copy()
    unskewed[:, 0] -= camera[10]*(unskewed[:, 1]-camera[9])/fy
    intrinsic = np.array([[fx, 0., camera[8]], [0., fy, camera[9]], [0., 0., 1.]])
    inverse = brown_central_inverse(SimpleNamespace(K_px=intrinsic,
        distortion_opencv=camera[11:23]), unskewed, tolerance_px=tolerance)
    if not inverse["valid"][0]:
        raise ValueError("Measured source ice pixel has no central Brown inverse")
    ray = np.r_[inverse["normalized_xy"][0], 1.] @ cv2.Rodrigues(camera[3:6])[0]
    if ray[2] >= 0.:
        raise ValueError("Measured source ice ray has no finite forward z=0 intersection")
    distance = -camera[2]/ray[2]
    point = camera[:3]+distance*ray
    point[2] = 0.
    if distance <= 0. or not np.isfinite(point).all():
        raise ValueError("Measured source ice intersection is behind the camera")
    return point


def evaluate_ice_motion(camera_a, camera_b, source_uv, target_uv, *, jacobian=True,
                        inverse_tolerance_px=1e-9):
    """Return r, d(r)/d(camera_a15), d(r)/d(camera_b15), conditional XYZ.

    Caller supplies cameras evaluated at the actual source and tracked target
    observed rows. Camera conditioning is C/100, Rodrigues3, logfx/logfy,
    cx/cy/skew divided by1000, then Brown k1/k2/p1/p2, matching the public raw
    source kernel. There is no unknown target-row iteration or fixed-ray cache.
    Input arrays and source point/clock identities remain caller-owned.
    """
    from .first_shot_source_kernel import _interfaces
    a, b = np.asarray(source_uv, float), np.asarray(target_uv, float)
    if a.ndim != 2 or a.shape[1:] != (2,) or a.shape != b.shape or not len(a):
        raise ValueError("Equal nonempty N-by-2 measured source/target pixels required")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Measured motion pixels must be finite")
    if not isinstance(jacobian, (bool, np.bool_)) or not np.isfinite(inverse_tolerance_px) or inverse_tolerance_px <= 0.:
        raise ValueError("Explicit Jacobian mode and positive inverse precision required")
    ca, cb = _camera_inventory(camera_a, len(a)), _camera_inventory(camera_b, len(a))
    project, brown, _, _ = _interfaces()
    values = []; source_j = []; target_j = []; worlds = []
    for i, (left, right, ua, ub) in enumerate(zip(ca, cb, a, b, strict=True)):
        try:
            world = _plane_point(left, ua, float(inverse_tolerance_px))[None]
            pa = project(left, world, with_jacobian=bool(jacobian))
            pb = project(right, world, with_jacobian=bool(jacobian))
            if not pa[-1][0] or not pb[-1][0]:
                raise ValueError("Static ice pair leaves the original central forward optical domain")
            if np.linalg.norm(pa[0][0]-ua) > max(16.*inverse_tolerance_px, 1e-7):
                raise ValueError("Source ice ray/projection round trip failed")
            values.append((pb[0][0]-ub)); worlds.append(world[0])
            if jacobian:
                ja = np.concatenate((np.asarray(pa[2]).reshape(1, 2, 11), brown(left, world)), axis=2)[0]
                jb = np.concatenate((np.asarray(pb[2]).reshape(1, 2, 11), brown(right, world)), axis=2)[0]
                # Projection depends on X-C, so the original conditioned
                # camera-centre Jacobian gives the world derivative exactly.
                plane_a, plane_b = -ja[:, :2]/100., -jb[:, :2]/100.
                source_j.append(-plane_b @ np.linalg.solve(plane_a, ja))
                target_j.append(jb)
        except (ValueError, np.linalg.LinAlgError) as error:
            raise ValueError("Invalid measured static-ice motion pair at index "+str(i)+": "+str(error)) from error
    count = 2*len(a)
    return (np.asarray(values).reshape(-1),
            np.asarray(source_j).reshape(count, 15) if jacobian else np.empty((count, 0)),
            np.asarray(target_j).reshape(count, 15) if jacobian else np.empty((count, 0)),
            np.asarray(worlds))
