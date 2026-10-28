"""Independent central-ray geometry; zero scene/production-solver imports.

All R matrices map a common reference ray into the camera frame. Pixels use
native decoded resolution. Distortion is Brown k1,k2,p1,p2 (no skew, no k3).
"""
from __future__ import annotations
import numpy as np
from scipy.spatial.transform import Rotation


def distort(xy: np.ndarray, d: np.ndarray) -> np.ndarray:
    x, y = xy[..., 0], xy[..., 1]
    k1, k2, p1, p2 = np.moveaxis(np.broadcast_to(d, xy.shape[:-1] + (4,)), -1, 0)
    r2 = x*x + y*y
    radial = 1 + k1*r2 + k2*r2*r2
    return np.stack((x*radial + 2*p1*x*y + p2*(r2 + 2*x*x),
                     y*radial + p1*(r2 + 2*y*y) + 2*p2*x*y), axis=-1)


def undistort(xyd: np.ndarray, d: np.ndarray, iterations=12) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized Newton inversion, including tangential terms and round-trip gate."""
    xy = np.array(xyd, dtype=np.float64, copy=True)
    if np.all(d == 0): return xy, np.ones(xy.shape[:-1], bool)
    k1, k2, p1, p2 = np.moveaxis(np.broadcast_to(d, xy.shape[:-1] + (4,)), -1, 0)
    for _ in range(iterations):
        x, y = xy[..., 0], xy[..., 1]
        r2 = x*x + y*y
        rad = 1 + k1*r2 + k2*r2*r2
        g = k1 + 2*k2*r2
        a = rad + 2*x*x*g + 2*p1*y + 6*p2*x
        b = 2*x*y*g + 2*p1*x + 2*p2*y
        c = rad + 2*y*y*g + 6*p1*y + 2*p2*x
        delta = distort(xy, d) - xyd
        det = a*c - b*b
        det_safe = np.where(np.abs(det) > 1e-14, det, 1e-14)
        step = np.stack(((c*delta[..., 0]-b*delta[..., 1])/det_safe,
                         (a*delta[..., 1]-b*delta[..., 0])/det_safe), axis=-1)
        xy -= np.clip(step, -0.5, 0.5)
    error = np.linalg.norm(distort(xy, d)-xyd, axis=-1)
    good = np.isfinite(xy).all(axis=-1) & (error < 1e-8) & (det > 0)
    return xy, good


def pixel_rays(uv: np.ndarray, optics: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # optics = fx, fy, cx, cy, k1, k2, p1, p2
    xyd = (uv - optics[..., 2:4]) / optics[..., 0:2]
    xy, valid = undistort(xyd, optics[..., 4:8])
    rays = np.concatenate((xy, np.ones(xy.shape[:-1] + (1,))), axis=-1)
    return rays, valid


def project_rays(rays: np.ndarray, optics: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = rays[..., 2]
    valid = z > 1e-8
    xy = rays[..., :2] / np.maximum(z[..., None], 1e-8)
    uv = distort(xy, optics[..., 4:8])*optics[..., :2] + optics[..., 2:4]
    return uv, valid & np.isfinite(uv).all(axis=-1)


def transfer(uv: np.ndarray, oi: np.ndarray, oj: np.ndarray,
             Ri: np.ndarray, Rj: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rays, a = pixel_rays(uv, oi)
    world = np.einsum('...ji,...j->...i', Ri, rays)
    target = np.einsum('...ij,...j->...i', Rj, world)
    pred, b = project_rays(target, oj)
    return pred, a & b


def homography_points(H: np.ndarray, uv: np.ndarray) -> np.ndarray:
    p = np.column_stack((uv, np.ones(len(uv)))) @ H.T
    z = p[:, 2:]
    return p[:, :2] / np.where(np.abs(z) < 1e-12, 1e-12, z)


def nearest_rotation(M: np.ndarray) -> np.ndarray:
    if np.linalg.det(M) < 0: M = -M
    u, _, v = np.linalg.svd(M)
    return u @ np.diag([1., 1., np.linalg.det(u @ v)]) @ v


def K(optics: np.ndarray) -> np.ndarray:
    fx, fy, cx, cy = optics[:4]
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.]])


def symmetric_errors(data: dict, optics: np.ndarray, rotations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    i, j = data['i'], data['j']
    pb, va = transfer(data['a'], optics[i], optics[j], rotations[i], rotations[j])
    pa, vb = transfer(data['b'], optics[j], optics[i], rotations[j], rotations[i])
    residual = np.column_stack((pb - data['b'], pa - data['a']))
    valid = va & vb
    return residual, valid


def error_stats(residual: np.ndarray, valid: np.ndarray | None = None) -> dict:
    r = np.asarray(residual)
    if valid is None: valid = np.isfinite(r).all(axis=1)
    good = r[valid]
    if not len(good): return dict(count=len(r), valid_count=0, invalid_count=len(r), rms_px=None, median_px=None, p95_px=None)
    distances = np.sqrt(np.sum(good*good, axis=1)/2.)
    return dict(count=len(r), valid_count=int(valid.sum()), invalid_count=int((~valid).sum()),
                rms_px=float(np.sqrt(np.mean(distances**2))), median_px=float(np.median(distances)),
                p95_px=float(np.percentile(distances, 95)), max_px=float(np.max(distances)))
