"""Process-local exact solver acceleration; set the mode before importing factors.

baseline preserves the original computations, numpy uses fewer temporary arrays,
auto additionally uses the optional strict-double C kernel, and compiled requires
that verified kernel. No observations, robust groups or solver settings change.
The DLL is built explicitly by build_acceleration.py, never during a solve.
"""
from __future__ import annotations
from calibration.solver_runtime.config import native_build_directory

import ctypes
import hashlib
import json
import os
from pathlib import Path

import numpy as np

MODE = os.environ.get('RINKSPACE_SOLVER_ACCELERATION', 'auto').lower()
if MODE not in ('baseline', 'numpy', 'auto', 'compiled'):
    raise ValueError(f'Unknown RINKSPACE_SOLVER_ACCELERATION: {MODE}')
SOURCE = Path(__file__).with_name('native')/'nearest_segment.c'
NATIVE_DIR = native_build_directory('nearest_segment', SOURCE)
_library = None
_kernel = None
_load_error = None
_fallbacks = 0
_build_manifest = None


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


if MODE in ('auto', 'compiled'):
    try:
        manifest = json.loads((NATIVE_DIR/'build.json').read_text(encoding='utf8'))
        binary = NATIVE_DIR/'nearest_segment.dll'
        if manifest['source_sha256'] != _sha(SOURCE) or manifest['dll_sha256'] != _sha(binary):
            raise ValueError('Native kernel source/binary hash differs from build manifest')
        _library = ctypes.CDLL(str(binary))
        _kernel = _library.nearest_segments_strict
        _kernel.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p,
                           ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                           ctypes.c_int64, ctypes.c_void_p, ctypes.c_void_p,
                           ctypes.c_void_p, ctypes.c_void_p]
        _kernel.restype = ctypes.c_int
        _build_manifest = manifest
    except (OSError, ValueError, KeyError, AttributeError) as exc:
        _load_error = str(exc)
        if MODE == 'compiled':
            raise RuntimeError('Build the matching strict C kernel with build_acceleration.py') from exc


def metadata():
    return {'mode': MODE, 'curve_search': 'strict_c' if _kernel else (
                'numpy_original' if MODE == 'baseline' else 'numpy_2d'),
            'board_search': 'numpy_original' if MODE == 'baseline' else 'numpy_2d',
            'fixed_whitening_batched': MODE != 'baseline',
            'huber_cache': MODE != 'baseline' and os.environ.get('RINKSPACE_HUBER_CACHE', '1') != '0',
            'native_load_error': _load_error, 'native_exception_fallbacks': _fallbacks,
            'native_build': _build_manifest}


def _nearest_original(observed, start, direction, length2, valid):
    offset = observed[:, None, :] - start[None, :, :]
    raw_t = (np.einsum('nmi,mi->nm', offset, direction)
             / np.maximum(length2[None, :], 1e-9))
    t = np.clip(raw_t, 0., 1.)
    nearest = start[None, :, :] + t[:, :, None]*direction[None, :, :]
    squared = np.sum((nearest-observed[:, None, :])**2, axis=2)
    squared[:, ~valid] = np.inf
    index = np.argmin(squared, axis=1)
    row = np.arange(len(observed))
    return index, raw_t[row, index], t[row, index], nearest[row, index]


def _nearest_numpy(observed, start, direction, length2, valid):
    dx = observed[:, None, 0] - start[None, :, 0]
    dy = observed[:, None, 1] - start[None, :, 1]
    raw_t = dx*direction[None, :, 0]
    raw_t += dy*direction[None, :, 1]
    raw_t /= np.maximum(length2[None, :], 1e-9)
    t = np.clip(raw_t, 0., 1.)
    del dx, dy
    nx = start[None, :, 0] + t*direction[None, :, 0]
    ny = start[None, :, 1] + t*direction[None, :, 1]
    squared = (nx-observed[:, None, 0])**2
    squared += (ny-observed[:, None, 1])**2
    squared[:, ~valid] = np.inf
    index = np.argmin(squared, axis=1)
    row = np.arange(len(observed))
    return (index, raw_t[row, index], t[row, index],
            np.column_stack((nx[row, index], ny[row, index])))


def nearest_segments(observed, start, direction, length2, valid, *, compiled=True):
    """Preserve finite endpoints and first argmin; caller handles all-invalid.

    C uses contiguous float64 buffers. Nonfinite inputs or overflow must use the
    original NumPy semantics (including NaN argmin), not silently skip a point.
    All scratch storage belongs to this call, so the kernel is reentrant.
    """
    global _fallbacks
    if MODE == 'baseline':
        return _nearest_original(observed, start, direction, length2, valid)
    if compiled and _kernel is not None:
        # Invalid segments are masked by the original algorithm. NaN/Inf among
        # valid segments or observations must not change its argmin behavior.
        if (not np.isfinite(observed).all() or not np.isfinite(length2[valid]).all()
                or not np.isfinite(direction[valid]).all()):
            _fallbacks += 1
            return _nearest_original(observed, start, direction, length2, valid)
        arrays = [np.ascontiguousarray(a, dtype=np.float64)
                  for a in (observed, start, direction, length2)]
        mask = np.ascontiguousarray(valid, dtype=np.uint8)
        index = np.empty(len(observed), np.int64)
        raw_t = np.empty(len(observed), np.float64)
        t = np.empty(len(observed), np.float64)
        nearest = np.empty((len(observed), 2), np.float64)
        code = _kernel(arrays[0].ctypes.data, len(observed), arrays[1].ctypes.data,
                       arrays[2].ctypes.data, arrays[3].ctypes.data, mask.ctypes.data,
                       len(direction), index.ctypes.data, raw_t.ctypes.data,
                       t.ctypes.data, nearest.ctypes.data)
        if not code:
            return index, raw_t, t, nearest
        _fallbacks += 1
        return _nearest_original(observed, start, direction, length2, valid)
    return _nearest_numpy(observed, start, direction, length2, valid)


class BatchedWhitening:
    """Batch frozen covariance blocks by size without changing dot ordering."""

    def __init__(self, blocks):
        groups = {}
        for indices, weight in blocks:
            groups.setdefault(len(indices), []).append((indices, weight))
        self.groups = [(np.stack([row[0] for row in rows]),
                        np.stack([row[1] for row in rows])) for rows in groups.values()]

    def whiten(self, value):
        result = np.empty_like(value)
        for indices, weights in self.groups:
            selected = value[indices]
            if value.ndim == 1:
                result[indices] = np.matmul(weights, selected[..., None])[..., 0]
            else:
                result[indices] = np.matmul(weights, selected)
        return result
