"""Strict-double row-owned native mapping for the complete fixed-XYZ CSR.

Uses the same output pointer, padding removal, geometry insertion, scaling,
zero removal and sort semantics as scatter_fixed_xyz_source_csr. The native
sorted flag is calculated over every output row, never guessed from layout.
"""
from __future__ import annotations
from calibration.solver_runtime.config import native_build_directory
import ctypes
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'native' / 'fixed_xyz_csr_scatter.c'
DEST = native_build_directory('csr_scatter', SOURCE)
_library = None
_kernels = {}
_load_error = None
_manifest = None


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


try:
    _manifest = json.loads((DEST / 'build.json').read_text(encoding='utf8'))
    binary = DEST / _manifest.get('binary', _manifest.get('binary_name', 'fixed_xyz_csr_scatter.dll'))
    if _manifest['source_sha256'] != _sha(SOURCE) or _manifest['dll_sha256'] != _sha(binary):
        raise ValueError('Native CSR scatter source/binary hash mismatch')
    _library = ctypes.CDLL(str(binary))
    for name, dtype in (('scatter_i32', np.dtype(np.int32)), ('scatter_i64', np.dtype(np.int64))):
        kernel = getattr(_library, name)
        kernel.argtypes = ([ctypes.c_int64] * 4 + [ctypes.c_void_p] * 8 +
                           [ctypes.c_int64, ctypes.c_double, ctypes.c_int] + [ctypes.c_void_p] * 4)
        kernel.restype = ctypes.c_int
        _kernels[dtype] = kernel
except (OSError, ValueError, KeyError, AttributeError) as exc:
    _load_error = str(exc)


def metadata():
    return dict(available=bool(_kernels), load_error=_load_error, native_build=_manifest,
                implementation='strict_double_openmp_CSR_row_scatter',
                sorted_state_checked_on_every_output_row=True,
                threads=int(os.environ.get('RINKSPACE_SCATTER_THREADS',
                    os.environ.get('RINKSPACE_MOTION_THREADS', os.cpu_count() or 1))))


def native_scatter(localj, source_rows, geom, geometry_row_start, geometry_indices,
                   xyz_slots, view_start, total_columns, scale):
    if not _kernels:
        raise RuntimeError('Build matching isolated native CSR scatter with build_csr_scatter_native.py: ' + str(_load_error))
    dtype = localj.indices.dtype
    if dtype not in _kernels or localj.indptr.dtype != dtype or localj.data.dtype != np.dtype(np.float64):
        raise TypeError('Native scatter requires original int32/int64 CSR indices and float64 data')
    if geom.data.dtype != np.dtype(np.float64):
        raise TypeError('Native scatter requires float64 geometry values')
    gp = np.ascontiguousarray(geom.indptr, dtype=dtype)
    gi = np.ascontiguousarray(geom.indices, dtype=dtype)
    geometry_indices = np.ascontiguousarray(geometry_indices, dtype=np.int64)
    xyz_slots = np.ascontiguousarray(xyz_slots, dtype=np.int64)
    nnz = localj.nnz - int(source_rows) + geom.nnz
    data = np.empty(nnz, np.float64)
    columns = np.empty(nnz, dtype)
    pointer = np.empty(localj.shape[0] + 1, dtype)
    sorted_output = ctypes.c_int()
    threads = metadata()['threads']
    if threads <= 0:
        raise ValueError('Positive native CSR scatter thread count required')
    lp = np.ascontiguousarray(localj.indptr)
    li = np.ascontiguousarray(localj.indices)
    lv = np.ascontiguousarray(localj.data)
    gv = np.ascontiguousarray(geom.data)
    code = _kernels[dtype](localj.shape[0], int(source_rows), int(geometry_row_start), geom.shape[0],
        *[a.ctypes.data for a in (lp, li, lv, gp, gi, gv, geometry_indices, xyz_slots)],
        int(view_start), float(scale), threads, pointer.ctypes.data, columns.ctypes.data,
        data.ctypes.data, ctypes.byref(sorted_output))
    if code:
        raise RuntimeError('Native CSR scatter failed with code ' + str(code))
    if int(pointer[-1]) != nnz:
        raise RuntimeError('Native CSR scatter output inventory changed')
    matrix = csr_matrix((data, columns, pointer), shape=(localj.shape[0], total_columns))
    if sorted_output.value:
        matrix.has_sorted_indices = True
    if len(geometry_indices) > 5:
        matrix.eliminate_zeros()
    matrix.sort_indices()
    return matrix
