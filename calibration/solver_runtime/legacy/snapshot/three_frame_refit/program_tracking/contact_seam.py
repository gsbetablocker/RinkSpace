"""Source-pixel uncertainty for same-foot contacts at a live camera cut.

Near-horizontal viewing rays amplify a few vertical pixels into a large
ground-depth change. An isotropic ground-distance tolerance hides that fact.
The covariance is frozen at the input calibration, so a trial camera cannot
reduce its penalty by inflating its own uncertainty.
"""
from __future__ import annotations

import numpy as np

from projection import project_fast
from track_sequence import fast_world_rays


def ground(camera, pixel):
    ray = fast_world_rays(camera, np.asarray(pixel).reshape(1, 2))[0]
    if ray[2] >= -1e-8:
        raise ValueError('Support-foot ray crossed the camera horizon')
    return camera[:2] - camera[2]*ray[:2]/ray[2]


def pixel_ground_jacobian(camera, pixel):
    pixel = np.asarray(pixel, float)
    return np.column_stack([(ground(camera, pixel+delta)-ground(camera, pixel-delta))/2.
                            for delta in np.eye(2)])


def ground_whitener(cameras, pixels, sigma_px=(6., 4.), motion_sigma_scene=.15):
    sigma = np.broadcast_to(np.asarray(sigma_px, float), (2, 2))
    if not np.all(np.isfinite(sigma)) or np.any(sigma <= 0) or motion_sigma_scene < 0:
        raise ValueError('Contact uncertainties must be finite and positive')
    covariance = np.eye(2)*float(motion_sigma_scene)**2
    for camera, pixel, uncertainty in zip(cameras, pixels, sigma):
        derivative = pixel_ground_jacobian(camera, pixel)
        scaled = derivative*uncertainty[None, :]
        covariance += scaled@scaled.T
    return np.linalg.inv(np.linalg.cholesky(covariance)), covariance


def shared_ground_diagnostic(cameras, pixels):
    """Minimum two-view pixel correction, not new or adjusted observations.

    This diagnostic assumes zero displacement over the one native frame.
    The optimization factor separately allows a small nonzero displacement.
    """
    pixels = np.asarray(pixels, float)
    xy = np.mean([ground(c, p) for c, p in zip(cameras, pixels)], axis=0)
    def residual(q):
        xyz = np.r_[q, 0.].reshape(1, 3)
        return np.concatenate([project_fast(c, xyz)[0].ravel()-p
                               for c, p in zip(cameras, pixels)])
    for _ in range(20):
        jac = np.column_stack([(residual(xy+d*1e-4)-residual(xy-d*1e-4))/2e-4
                               for d in np.eye(2)])
        delta = np.linalg.lstsq(jac, -residual(xy), rcond=None)[0]
        xy += delta
        if np.linalg.norm(delta) < 1e-8:
            break
    return {'same_point_pixel_corrections': residual(xy).reshape(2, 2).tolist(),
            'semantics': 'diagnostic_zero_displacement_fit; source_pixels_unchanged'}
