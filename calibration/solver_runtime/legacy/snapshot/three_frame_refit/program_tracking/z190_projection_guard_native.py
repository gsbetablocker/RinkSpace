"""Fused strict-double Brown4 domain guard; all source UV/J remain OpenCV."""
from __future__ import annotations
from calibration.solver_runtime.config import native_build_directory
import ctypes
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

HERE=Path(__file__).resolve().parent
SOURCE=HERE/'native'/'central_projection_guard.c'
DEST = native_build_directory('projection_guard', SOURCE)
_kernel=None
_library=None
_manifest=None
_error=None


def _sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


try:
    _manifest=json.loads((DEST/'build.json').read_text(encoding='utf8'))
    binary=DEST/_manifest.get('binary_name','central_projection_guard.dll')
    if _manifest['source_sha256']!=_sha(SOURCE)or _manifest['dll_sha256']!=_sha(binary):
        raise ValueError('Native projection guard source/binary hash mismatch')
    _library=ctypes.CDLL(str(binary))
    _kernel=_library.central_brown4_guard
    _kernel.argtypes=[ctypes.c_void_p]*4+[ctypes.c_int64,ctypes.c_void_p]
    _kernel.restype=ctypes.c_int
except(OSError,ValueError,KeyError,AttributeError)as exc:_error=str(exc)


def metadata():return dict(available=_kernel is not None,load_error=_error,native_build=_manifest,
    implementation='strict_double_fused_Brown4_central_guard',source_UV_and_J_remain_original_OpenCV=True)


def central_guard(camera,points,uv,depth,*,rotation=None):
    if _kernel is None:raise RuntimeError('Build isolated projection guard with build_projection_guard_native.py: '+str(_error))
    rotation=np.ascontiguousarray(cv2.Rodrigues(camera[3:6])[0]if rotation is None else rotation)
    status=np.empty(len(points),np.uint8)
    # Match physical_projection._rays arithmetic exactly, including its BLAS
    # association. At huge off-screen pixels, even ray roundoff changes the
    # original absolute 1e-5 px agreement test; no independent C ray is used.
    camera_xyz=np.ascontiguousarray((rotation@(points-camera[:3]).T).T)
    c=np.ascontiguousarray(camera,dtype=np.float64)
    uv=np.ascontiguousarray(uv,dtype=np.float64)
    depth=np.ascontiguousarray(depth,dtype=np.float64)
    result=_kernel(c.ctypes.data,camera_xyz.ctypes.data,uv.ctypes.data,
        depth.ctypes.data,len(points),status.ctypes.data)
    if result:return None
    return status
