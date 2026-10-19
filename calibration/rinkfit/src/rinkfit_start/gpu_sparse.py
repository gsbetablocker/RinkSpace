"""Explicit opt-in FP64 CSR matrix-vector products using existing cuSPARSE.

This module does not solve or factor a system, change coordinates, choose rows,
modify W, or enable a fitting backend. The caller owns source/state provenance.
Importing the module performs no CUDA initialization or compilation.
"""
from __future__ import annotations

import ctypes
import os
from pathlib import Path
import threading
import numpy as np
from scipy import sparse


class GpuSparseError(RuntimeError):
    """Requested GPU operation could not be completed; no implicit fallback."""


class GpuCsr:
    """Immutable, resident copy of a complete canonical float64/CSR32 matrix.

    Materialize a transpose as CSR explicitly before construction. Sparse upper
    triangles retain their exact storage meaning; symmetry is never inferred.
    Every matvec transfers the full input and full output, in original units.
    Objects can be used as context managers and must be closed after use.
    """

    def __init__(self, matrix, *, library_path=None, cuda_bin=None):
        self._lock = threading.RLock()
        self._handle = ctypes.c_void_p()
        self._dll_dirs = []
        if os.name != "nt":
            raise GpuSparseError("The current native GPU sparse build supports Windows only")
        if not sparse.issparse(matrix) or matrix.format != "csr":
            raise TypeError("Supply an explicit CSR matrix; no hidden sparse conversion")
        if matrix.dtype != np.dtype("float64"):
            raise TypeError("GPU sparse requires original float64 matrix values")
        if matrix.indices.dtype != np.dtype("int32") or matrix.indptr.dtype != np.dtype("int32"):
            raise TypeError("GPU sparse requires int32 CSR indices; no implicit narrowing")
        if not matrix.has_canonical_format or not matrix.has_sorted_indices:
            raise ValueError("GPU sparse requires sorted, duplicate-free canonical CSR")
        if (matrix.indptr.ndim != 1 or matrix.indices.ndim != 1 or matrix.data.ndim != 1
                or matrix.indptr.size != matrix.shape[0] + 1
                or matrix.indptr[0] != 0 or matrix.indptr[-1] != matrix.data.size
                or matrix.indices.size != matrix.data.size
                or np.any(np.diff(matrix.indptr.astype(np.int64)) < 0)
                or np.any(matrix.indices < 0) or np.any(matrix.indices >= matrix.shape[1])):
            raise ValueError("Invalid CSR storage; input matrix is never modified")
        if not np.isfinite(matrix.data).all():
            raise ValueError("Matrix contains nonfinite values")
        for a in (matrix.data, matrix.indices, matrix.indptr):
            if not a.flags.c_contiguous:
                raise ValueError("CSR storage must be contiguous")
        if min(matrix.shape) <= 0 or max(*matrix.shape, matrix.nnz) > np.iinfo(np.int32).max:
            raise ValueError("Matrix shape/nnz exceeds the supported CSR32 contract")
        self.shape = tuple(matrix.shape)
        self.nnz = int(matrix.nnz)
        chosen = library_path or os.environ.get("RINKFIT_GPU_SPARSE_DLL")
        if not chosen:
            raise GpuSparseError("Set library_path or RINKFIT_GPU_SPARSE_DLL explicitly")
        library = Path(chosen).expanduser()
        if not library.is_absolute() or not library.is_file():
            raise GpuSparseError("GPU sparse DLL must be an existing absolute file path")
        if cuda_bin is not None:
            cuda_dir = Path(cuda_bin)
            if not cuda_dir.is_absolute() or not cuda_dir.is_dir():
                raise GpuSparseError("cuda_bin must be an existing absolute directory")
            self._dll_dirs.append(os.add_dll_directory(str(cuda_dir)))
        try:
            self._lib = ctypes.CDLL(str(library))
            self._lib.rinkfit_gpu_sparse_abi.restype = ctypes.c_int
            if self._lib.rinkfit_gpu_sparse_abi() != 1:
                raise GpuSparseError("Unsupported GPU sparse ABI")
            self._lib.rinkfit_gpu_sparse_error.restype = ctypes.c_char_p
            self._lib.rinkfit_gpu_sparse_create.argtypes = [ctypes.c_int64]*3 + [ctypes.c_void_p]*3 + [ctypes.POINTER(ctypes.c_void_p)]
            self._lib.rinkfit_gpu_sparse_create.restype = ctypes.c_int
            self._lib.rinkfit_gpu_sparse_matvec.argtypes = [ctypes.c_void_p]*3
            self._lib.rinkfit_gpu_sparse_matvec.restype = ctypes.c_int
            self._lib.rinkfit_gpu_sparse_destroy.argtypes = [ctypes.c_void_p]
            self._lib.rinkfit_gpu_sparse_destroy.restype = None
            self._check(self._lib.rinkfit_gpu_sparse_create(*self.shape, self.nnz, self._ptr(matrix.indptr), self._ptr(matrix.indices), self._ptr(matrix.data), ctypes.byref(self._handle)))
        except (OSError, AttributeError) as exc:
            self.close()
            raise GpuSparseError(f"Cannot load requested GPU sparse library: {exc}") from exc
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _ptr(a):
        return ctypes.c_void_p(a.ctypes.data)

    def _check(self, status):
        if status:
            message = self._lib.rinkfit_gpu_sparse_error()
            raise GpuSparseError(message.decode("utf-8", errors="replace") if message else "GPU sparse operation failed")

    def matvec(self, x, *, out=None):
        """Return full A @ x in FP64, including host/device transfers."""
        if not isinstance(x, np.ndarray) or x.dtype != np.dtype("float64") or x.shape != (self.shape[1],) or not x.flags.c_contiguous:
            raise TypeError("x must be a contiguous float64 vector of matrix column count")
        if not np.isfinite(x).all():
            raise ValueError("x contains nonfinite values")
        if out is None:
            out = np.empty(self.shape[0], dtype=np.float64)
        if not isinstance(out, np.ndarray) or out.dtype != np.dtype("float64") or out.shape != (self.shape[0],) or not out.flags.c_contiguous or not out.flags.writeable:
            raise TypeError("out must be a writable contiguous float64 vector of matrix row count")
        with self._lock:
            if not self._handle.value:
                raise GpuSparseError("GPU sparse matrix is closed")
            self._check(self._lib.rinkfit_gpu_sparse_matvec(self._handle, self._ptr(x), self._ptr(out)))
        return out

    def close(self):
        with self._lock:
            if self._handle.value:
                self._lib.rinkfit_gpu_sparse_destroy(self._handle)
                self._handle = ctypes.c_void_p()
            for item in self._dll_dirs:
                item.close()
            self._dll_dirs.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
