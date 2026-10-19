"""A shared-pivot pan/tilt/zoom adapter for the frozen Brown rink camera.

Angles are offsets from the source anchor.  All images and K values are native
pixels.  World units are conditional scene units, never implicitly metres.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'calibration' / 'work'))
from physical_projection import safe_project, verify_camera
from rinkspace.calibration_companion.wide_angle_camera import brown_central_inverse


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def anchor_camera(scene, pts):
    c = next(f['camera'] for f in scene['frame_cameras'] if f['source_pts'] == pts)
    k = np.asarray(c['K_px'], float)
    return np.r_[c['center_xyz'], c['rvec_world_to_camera'],
                 np.log(k[0, 0]), np.log(k[1, 1]), k[0, 2], k[1, 2],
                 k[0, 1], c['distortion_opencv']]


def from_state(anchor, state, offset=None, axis=None, lens_profile=None,
               brown_profile=None):
    """Three temporal variables; a shared pivot-to-optical-centre lever arm.

    axis is the pan axis in world coordinates, nominally world +Z.  Tilt is
    about camera +X. offset is fixed in camera coordinates. The anchor optical
    centre is preserved exactly, with shared pivot P=C0-R0.T@offset.
    """
    a = np.asarray(anchor, float)
    pan, tilt, zoom = np.asarray(state, float)
    d = np.zeros(3) if offset is None else np.asarray(offset, float)
    n = np.array([0., 0., 1.]) if axis is None else np.asarray(axis, float)
    n = n / np.linalg.norm(n)
    r0 = cv2.Rodrigues(a[3:6])[0]
    r = cv2.Rodrigues(np.array([tilt, 0., 0.]))[0] @ r0 @ cv2.Rodrigues(-pan*n)[0]
    v = a.copy()
    v[:3] = a[:3] + (r.T-r0.T) @ d
    v[3:6] = cv2.Rodrigues(r)[0].ravel()
    v[6:8] += zoom
    v[10] *= np.exp(zoom)
    if lens_profile is not None:
        # Shared continuous zoom law, zero at anchor. Coefficients are scaled
        # for numerical conditioning: aspect logs, principal pixels / 1000,
        # and skew pixels / 100. The same law applies to every source frame.
        b=np.asarray(lens_profile,float).reshape(4,2)
        basis=np.array([zoom,zoom*zoom])
        shift=b@basis
        v[7]+=shift[0]
        v[8:10]+=shift[1:3]*1000.
        v[10]+=shift[3]*100.
    if brown_profile is not None:
        profile=np.asarray(brown_profile,float)
        if profile.shape!=(12,2) or not np.isfinite(profile).all():
            raise ValueError('Brown zoom law must have shape (12,2)')
        v[11:23]+=profile@np.array([zoom,zoom*zoom])
    return v


def fast_project(camera, xyz):
    """Brown12 forward map for numerical fitting, checked strictly afterward."""
    v = np.asarray(camera, float)
    r = cv2.Rodrigues(v[3:6])[0]
    k = np.array([[np.exp(v[6]), 0., v[8]], [0., np.exp(v[7]), v[9]], [0., 0., 1.]])
    uv = cv2.projectPoints(np.asarray(xyz, float).reshape(-1, 3),
                           v[3:6], -r @ v[:3], k, v[11:23])[0].reshape(-1, 2)
    uv[:, 0] += v[10] * (uv[:, 1]-v[9]) / np.exp(v[7])
    return uv


def world_rays(camera, pixels):
    v = np.asarray(camera, float)
    p = np.asarray(pixels, float).copy()
    fx, fy = np.exp(v[6:8])
    p[:, 0] -= v[10]*(p[:, 1]-v[9])/fy
    k = np.array([[fx, 0., v[8]], [0., fy, v[9]], [0., 0., 1.]])
    inv = brown_central_inverse(SimpleNamespace(K_px=k, distortion_opencv=v[11:23]),
                                p, tolerance_px=.005)
    rays = np.c_[inv['normalized_xy'], np.ones(len(p))] @ cv2.Rodrigues(v[3:6])[0]
    return rays, np.asarray(inv['valid'], bool)


def attach_to_surface(scene, camera, pixels, surface):
    """Intersect actual anchor image rays with their labelled fixed surface.

    These are conditional anchor attachments, not independent 3D measurements.
    Unknown or nonintersecting points remain NaN and must not enter the solve.
    """
    rays, valid = world_rays(camera, pixels)
    centre = np.asarray(camera[:3], float)
    border = np.asarray(scene['ice_boundary']['points_xyz'], float)
    polygon = border[:, :2].astype(np.float32)
    xyz = np.full((len(pixels), 3), np.nan)
    surface = np.asarray(surface).astype(str)
    for i in np.flatnonzero(valid & (surface == 'ice')):
        if rays[i, 2] >= -1e-8:
            continue
        lam = -centre[2]/rays[i, 2]
        hit = centre+lam*rays[i]
        if cv2.pointPolygonTest(polygon, tuple(hit[:2]), False) >= 0:
            xyz[i] = hit
    starts = border[:, :2]
    edges = np.roll(starts, -1, axis=0)-starts
    rel = starts-centre[:2]
    top = np.asarray(scene['board_surface']['top_edge_xyz'], float)[:, 2]
    for i in np.flatnonzero(valid & (surface == 'board')):
        dx, dy = rays[i, :2]
        denom = dx*edges[:, 1]-dy*edges[:, 0]
        nonzero = np.abs(denom)>1e-10
        lam = np.full(len(edges), np.inf)
        along = np.full(len(edges), np.inf)
        lam[nonzero] = (rel[nonzero, 0]*edges[nonzero, 1]-rel[nonzero, 1]*edges[nonzero, 0])/denom[nonzero]
        along[nonzero] = (rel[nonzero, 0]*dy-rel[nonzero, 1]*dx)/denom[nonzero]
        z = centre[2]+lam*rays[i, 2]
        good = nonzero & (lam>0) & (along>=0) & (along<=1) & (z>=0) & (z<=top.max())
        if good.any():
            j = np.argmin(np.where(good, lam, np.inf))
            xyz[i] = centre+lam[j]*rays[i]
    return xyz, np.isfinite(xyz).all(axis=1)
