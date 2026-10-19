"""Exact polynomial Brown batching for explicitly enabled review processes.

The native row solver, adaptive curves, all nine axis-domain samples and the
complete camera trajectory stay unchanged. Rational batches use the original
implementation, including its exact cubic-denominator extrema checks.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
import zipfile

import numpy as np
from scipy.spatial.transform import Rotation

from . import row_scene_projection as original

_ORIGINAL_BATCH_PROJECT = original._batch_project
_INSTALLED_HANDLER = None


def _brown_map_jacobian_varying(xy, distortion):
    """The public Brown12 expressions, with coefficients broadcast per point.

    Callers supply zero k4..k6. Keep the public operation order, including the
    quotient slope, so scalar grouping and broadcasting are bit comparable.
    """
    coefficients = np.asarray(distortion, float)
    if coefficients.shape[-1:] != (12,):
        raise ValueError('Exactly 12 Brown coefficients per point required')
    k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4 = np.moveaxis(coefficients, -1, 0)
    x, y = np.moveaxis(np.asarray(xy, float), -1, 0)
    r2 = x*x+y*y
    numerator = 1.+k1*r2+k2*r2*r2+k3*r2*r2*r2
    # The caller has proved all rational coefficients are exactly zero.
    denominator = np.ones_like(r2)
    radial = numerator/denominator
    numerator_slope = k1+2.*k2*r2+3.*k3*r2*r2
    denominator_slope = k4+2.*k5*r2+3.*k6*r2*r2
    slope = (numerator_slope*denominator-numerator*denominator_slope)/denominator**2
    predicted = np.stack([x*radial+2.*p1*x*y+p2*(r2+2.*x*x)+s1*r2+s2*r2*r2,
                          y*radial+p1*(r2+2.*y*y)+2.*p2*x*y+s3*r2+s4*r2*r2], axis=-1)
    j11 = radial+2.*x*x*slope+2.*p1*y+6.*p2*x+2.*x*(s1+2.*s2*r2)
    j12 = 2.*x*y*slope+2.*p1*x+2.*p2*y+2.*y*(s1+2.*s2*r2)
    j21 = 2.*x*y*slope+2.*p1*x+2.*p2*y+2.*x*(s3+2.*s4*r2)
    j22 = radial+2.*y*y*slope+6.*p1*y+2.*p2*x+2.*y*(s3+2.*s4*r2)
    eigen = .5*(j11+j22-np.sqrt((j11-j22)**2+(j12+j21)**2))
    return predicted, j11, j12, j21, j22, eigen


def batch_project(cameras, points):
    """Same outputs and guards as the public native-row batch projection."""
    cameras = np.asarray(cameras, float)
    points = np.asarray(points, float)
    if cameras.shape != (len(points), 25) or not np.isfinite(cameras).all():
        raise ValueError('Finite camera25 per world point required')
    if np.any(abs(cameras[:, 23:25]) > 1e-12):
        raise ValueError('Brown tilt is fixed zero')
    if not len(points) or np.any(cameras[:, 16:19] != 0.):
        return _ORIGINAL_BATCH_PROJECT(cameras, points)
    rotations = Rotation.from_rotvec(cameras[:, 3:6]).as_matrix()
    local = np.einsum('nij,nj->ni', rotations, points-cameras[:, :3])
    depth = local[:, 2]
    uv = np.full((len(points), 2), np.nan)
    valid = np.zeros(len(points), bool)
    selected = np.flatnonzero(depth > .05)
    if not len(selected):
        return uv, depth, valid
    rays = local[selected, :2]/depth[selected, None]
    distortion = cameras[selected, 11:23]
    with np.errstate(all='ignore'):
        # For finite polynomial rays the exact axis denominator is one.
        # Preserve invalid overflowing/nonfinite rays from the original guard.
        denominator_finite = np.isfinite(np.sum(rays*rays, axis=1))
        axis = rays[:, None, :]*original._AXIS_STATIONS[None, :, None]
        eigen = _brown_map_jacobian_varying(axis, distortion[:, None, :])[-1].min(axis=1)
        good = denominator_finite & np.isfinite(eigen) & (eigen > original._EIGEN_FLOOR)
        selected = selected[good]
        mapped = _brown_map_jacobian_varying(rays[good], distortion[good])[0]
        cam = cameras[selected]
        projected = np.c_[np.exp(cam[:, 6])*mapped[:, 0]+cam[:, 10]*mapped[:, 1]+cam[:, 8],
                          np.exp(cam[:, 7])*mapped[:, 1]+cam[:, 9]]
    finite = np.isfinite(projected).all(axis=1)
    uv[selected[finite]] = projected[finite]
    valid[selected[finite]] = True
    return uv, depth, valid


def comparison(expected, actual):
    uv0, depth0, valid0 = expected
    uv1, depth1, valid1 = actual
    finite0 = np.isfinite(uv0)
    finite1 = np.isfinite(uv1)
    common = finite0 & finite1
    error = float(np.max(np.abs(uv0[common]-uv1[common]), initial=0.))
    return dict(valid_identical=bool(np.array_equal(valid0, valid1)),
                depth_identical=bool(np.array_equal(np.asarray(depth0).view(np.uint64), np.asarray(depth1).view(np.uint64))),
                UV_finite_mask_identical=bool(np.array_equal(finite0, finite1)),
                UV_bit_identical=bool(np.array_equal(np.asarray(uv0).view(np.uint64), np.asarray(uv1).view(np.uint64))),
                maximum_UV_difference_px=error, points=len(depth0))


def _checked(result):
    if (not result['valid_identical'] or not result['depth_identical']
            or not result['UV_finite_mask_identical'] or result['maximum_UV_difference_px'] > 1e-9):
        raise ValueError('Fast projection differs from the public physical projection: ' + json.dumps(result))


def install(audit=None):
    """Explicitly patch this review process only; importing has no side effect."""
    global _INSTALLED_HANDLER
    handler = batch_project if audit is None else audit
    if original._batch_project not in (_ORIGINAL_BATCH_PROJECT, batch_project, _INSTALLED_HANDLER):
        raise RuntimeError('An unknown batch projection override is already installed')
    original._batch_project = handler
    _INSTALLED_HANDLER = handler
    return dict(implementation=__file__, polynomial_Brown_per_point_broadcast=True,
                axis_domain_stations=original._AXIS_STATIONS.tolist(),
                rational_nonzero_batches='complete original implementation',
                source_kernel_or_solver_modified=False, audit_enabled=audit is not None)


def uninstall():
    global _INSTALLED_HANDLER
    if original._batch_project not in (_ORIGINAL_BATCH_PROJECT, batch_project, _INSTALLED_HANDLER):
        raise RuntimeError('Cannot replace an unknown batch projection override')
    original._batch_project = _ORIGINAL_BATCH_PROJECT
    _INSTALLED_HANDLER = None


class ProjectionAudit:
    """Record every real input once and compare old/new per native draw call.

    This handler returns the original result for the reference PNG. A second
    draw with install() produces the fast PNG for a full pixel comparison.
    """
    def __init__(self, archive):
        self.archive = Path(archive).resolve()
        data = Path('D:/CodexData/RinkSpace_Data').resolve()
        if not self.archive.is_relative_to(data):
            raise ValueError('Real projection inputs must stay in Data')
        self.file = zipfile.ZipFile(self.archive, 'x', compression=zipfile.ZIP_DEFLATED,
                                    compresslevel=1, allowZip64=True)
        self.calls = 0
        self.points = 0
        self.old_seconds = 0.
        self.fast_seconds = 0.
        self.maximum_error = 0.
        self.bit_identical_calls = 0
        self.rational_fallback_calls = 0
        self.ledger = []
        self.valid_depth_finite_identical = True

    def __call__(self, cameras, points):
        index = self.calls
        # Streaming .npy members avoid retaining a full frame's arrays in RAM.
        for name, value in (('cameras', cameras), ('points', points)):
            with self.file.open(f'call_{index:06d}_{name}.npy', 'w', force_zip64=True) as member:
                np.lib.format.write_array(member, np.asarray(value, float), allow_pickle=False)
        started = time.perf_counter()
        expected = _ORIGINAL_BATCH_PROJECT(cameras, points)
        old_seconds = time.perf_counter()-started
        started = time.perf_counter()
        actual = batch_project(cameras, points)
        fast_seconds = time.perf_counter()-started
        result = comparison(expected, actual)
        result.update(call=index, old_seconds=old_seconds, fast_seconds=fast_seconds,
                      rational_fallback=bool(np.any(np.asarray(cameras)[:, 16:19] != 0.)))
        self.calls += 1
        self.points += len(points)
        self.old_seconds += old_seconds
        self.fast_seconds += fast_seconds
        self.maximum_error = max(self.maximum_error, result['maximum_UV_difference_px'])
        self.bit_identical_calls += int(result['UV_bit_identical'])
        self.rational_fallback_calls += int(result['rational_fallback'])
        self.valid_depth_finite_identical &= (result['valid_identical'] and result['depth_identical'] and result['UV_finite_mask_identical'])
        self.ledger.append(result)
        _checked(result)
        return expected

    def close(self):
        if self.file is not None:
            self.file.writestr('comparison_calls.json', json.dumps(self.ledger, allow_nan=False))
            self.file.close()
            self.file = None
        return dict(inputs=str(self.archive), calls=self.calls, points=self.points,
                    maximum_UV_difference_px=self.maximum_error,
                    UV_bit_identical_calls=self.bit_identical_calls,
                    all_UV_bit_identical=self.bit_identical_calls == self.calls,
                    rational_fallback_calls=self.rational_fallback_calls,
                    original_batch_project_seconds=self.old_seconds,
                    fast_batch_project_seconds=self.fast_seconds,
                    all_valid_depth_and_finite_masks_identical=bool(self.valid_depth_finite_identical),
                    original_result_returned_for_reference_render=True)
