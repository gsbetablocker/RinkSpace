"""Explicit CUDA shadow adapter for the unchanged native source filter.

The original NumPy RGB noise/channel arithmetic and CPU continuous estimator
remain authoritative. This adapter does not install itself or change a mean.
"""
from pathlib import Path
import ctypes,os
import numpy as np
from scipy.ndimage._filters import _gaussian_kernel1d

class SourceFilterCUDA:
    def __init__(self, library, height, width, *, batch=1,
                 cuda_bin="D:/Program Files/CUDA/bin"):
        self.library=Path(library).resolve()
        self._dll_dir=os.add_dll_directory(str(cuda_bin))
        self._lib=ctypes.WinDLL(str(self.library))
        self._lib.rf_source_filter_error.restype=ctypes.c_char_p
        self._lib.rf_source_filter_create.argtypes=[ctypes.c_int]*3
        self._lib.rf_source_filter_create.restype=ctypes.c_void_p
        self._lib.rf_source_filter_apply.argtypes=[ctypes.c_void_p]*4+[ctypes.c_int,ctypes.c_void_p]
        self._lib.rf_source_filter_apply.restype=ctypes.c_int
        self._lib.rf_source_filter_destroy.argtypes=[ctypes.c_void_p]
        self.h,self.w,self.batch=int(height),int(width),int(batch)
        self.kernel=np.asarray(_gaussian_kernel1d(.8,0,3)[::-1],dtype=np.float64)
        self._ctx=self._lib.rf_source_filter_create(self.h,self.w,self.batch)
        if not self._ctx:raise RuntimeError(self._lib.rf_source_filter_error().decode())
    def __call__(self, channel):
        a=np.ascontiguousarray(channel,dtype=np.float32)
        if a.ndim==2:a=a[None]
        if a.ndim!=3 or a.shape[1:]!=(self.h,self.w) or not 1<=len(a)<=self.batch:
            raise ValueError("CUDA filter shape does not match explicit context")
        g=np.empty_like(a);c=np.empty(a.shape,dtype=np.float64)
        rc=self._lib.rf_source_filter_apply(self._ctx,a.ctypes.data,g.ctypes.data,c.ctypes.data,len(a),self.kernel.ctypes.data)
        if rc:raise RuntimeError(self._lib.rf_source_filter_error().decode())
        return (g[0],c[0]) if channel.ndim==2 else (g,c)
    def close(self):
        if self._ctx:self._lib.rf_source_filter_destroy(self._ctx);self._ctx=None
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
