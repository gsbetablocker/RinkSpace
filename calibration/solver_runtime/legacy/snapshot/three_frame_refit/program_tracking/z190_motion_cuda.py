"""Optional strict-double CUDA rows; copies are included in call timings.

CPU Brown inversion and NumPy geometric contractions preserve the exact
source profile admission semantics. Device memory is reused by the DLL.
"""
from __future__ import annotations
from calibration.solver_runtime.config import native_build_directory
import ctypes,hashlib,json,os
from pathlib import Path
import numpy as np
from z190_motion_native import _inverse_brown_native

HERE=Path(__file__).resolve().parent
SOURCE=HERE/'native/moving_center_motion_cuda.cu'
DEST = native_build_directory('motion_cuda', SOURCE)
_library=None;_kernel=None;_load_error=None;_manifest=None;_device=None
_dll_dirs=[]

try:
    for folder in('D:/Program Files/CUDA/bin','D:/Program Files/CUDA/bin/x64'):
        if Path(folder).is_dir():_dll_dirs.append(os.add_dll_directory(folder))
    _manifest=json.loads((DEST/'build.json').read_text(encoding='utf8'))
    binary=DEST/_manifest['binary'];sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    if _manifest['source_sha256']!=sha(SOURCE)or _manifest['dll_sha256']!=sha(binary):raise ValueError('CUDA source/binary hash mismatch')
    _library=ctypes.CDLL(str(binary));_kernel=_library.motion_cuda
    _kernel.argtypes=([ctypes.c_int64]+[ctypes.c_void_p]*21+[ctypes.c_double]+[ctypes.c_void_p]*3+
        [ctypes.c_double]+[ctypes.c_int]*3+[ctypes.c_void_p]*6)
    _kernel.restype=ctypes.c_int
    _library.motion_cuda_error.argtypes=[ctypes.c_int];_library.motion_cuda_error.restype=ctypes.c_char_p
    _library.motion_cuda_device.argtypes=[ctypes.c_void_p,ctypes.c_int];_library.motion_cuda_device.restype=ctypes.c_int
    name=ctypes.create_string_buffer(256);code=_library.motion_cuda_device(name,256)
    if code:raise RuntimeError(_library.motion_cuda_error(code).decode())
    _device=name.value.decode()
except(OSError,ValueError,KeyError,AttributeError,RuntimeError)as exc:
    _kernel=None;_load_error=str(exc)

def metadata():return dict(available=_kernel is not None,implementation='strict_double_cuda_row_fused',device=_device,
    native_build=_manifest,load_error=_load_error,allocations='process-local reused device buffers',copies_included_in_call=True)

def cuda_motion(*,a,b,fa,fb,ra,rb,da,db,ca,cb,dca_shared,dcb_shared,dca_view,dcb_view,
                pp,k1,planes,jacobian,sigma,profile_weight=None,minimum_source_depth_scene_units=0.,
                schur=True,return_trace=False):
    if _kernel is None:raise RuntimeError('Build isolated CUDA kernel with build_motion_cuda.py: '+str(_load_error))
    a=np.asarray(a,float);n=len(a)
    if a.shape!=(n,2)or np.shape(b)!=(n,2):raise ValueError('Original a/b must be Nx2')
    floor=float(minimum_source_depth_scene_units)
    if floor<0 or not np.isfinite(floor):raise ValueError('Finite nonnegative conditional minimum source depth required')
    def arr(x,shape):return np.ascontiguousarray(np.broadcast_to(np.asarray(x,float),shape))
    arrays=[arr(x,shape)for x,shape in((a,(n,2)),(b,(n,2)),(fa,(n,)),(fb,(n,)),(ra,(n,3,3)),(rb,(n,3,3)))]
    arrays+=([None,None]if not jacobian else[arr(da,(n,3,3,3)),arr(db,(n,3,3,3))])
    arrays+=[arr(ca,(n,3)),arr(cb,(n,3))]
    arrays+=([None]*4 if not jacobian else[arr(x,(n,3,4))for x in(dca_shared,dcb_shared,dca_view,dcb_view)])
    principal=arr(pp,(2,));planes=arr(planes,(n,4));sig=np.asarray(sigma,float)
    if sig.ndim==1:sig=sig.reshape(-1,1)
    sig=arr(sig,(n,2));weight=np.ones((n,2))if profile_weight is None else arr(profile_weight,(n,2))
    if np.any(arrays[2]<=0)or np.any(arrays[3]<=0):raise ValueError('Explicit fixed PP and positive focal required')
    if np.any(sig<=0)or np.any(weight<0)or not np.isfinite(weight).all():raise ValueError('Positive original sigma/frozen nonnegative weights required')
    threads=int(os.environ.get('RINKSPACE_MOTION_THREADS',os.cpu_count()or 1))
    if threads<1:raise ValueError('Positive RINKSPACE_MOTION_THREADS required')
    inverse_a,inverse_af=_inverse_brown_native(arrays[0],arrays[2],principal,float(k1),threads)
    inverse_b,_=_inverse_brown_native(arrays[1],arrays[3],principal,float(k1),threads)
    inverses=[np.ascontiguousarray(x)for x in(inverse_a,inverse_af,inverse_b)]
    local=np.c_[inverse_a,np.ones(n)];source_direction=np.einsum('ni,nij->nj',local,arrays[4])
    target_direction=np.einsum('nij,nj->ni',arrays[5],source_direction)
    target_translation=np.einsum('nij,nj->ni',arrays[5],arrays[8]-arrays[9])
    geometry=[np.ascontiguousarray(x)for x in(source_direction,target_direction,target_translation)]
    residual=np.empty(2*n);blocks=[np.empty((2*n,4))for _ in range(3)]if jacobian else[None]*3
    trace=np.empty((n,16));errors=np.zeros(n,np.int32)
    def ptr(x):return None if x is None else x.ctypes.data
    code=_kernel(n,*(ptr(x)for x in arrays),*(ptr(x)for x in inverses),*(ptr(x)for x in geometry),ptr(principal),float(k1),
        ptr(planes),ptr(sig),ptr(weight),floor,int(jacobian),int(schur),threads,ptr(residual),*(ptr(x)for x in blocks),ptr(trace),ptr(errors))
    if code:raise RuntimeError('CUDA motion failed: '+_library.motion_cuda_error(code).decode())
    bad=np.flatnonzero(errors)
    if len(bad):
        from z190_moving_center_motion import MovingDepthDomainError
        names={1:'noninvertible_source_Brown',2:'source_Brown_root_unconverged',3:'target_cheirality',
            4:'no_positive_target_depth_interval',5:'reviewed_plane_positive_depth_invalid',6:'source_ray_infinity_behind_target_line_chart'}
        error=errors[bad[0]];raise MovingDepthDomainError(names[int(error)],np.flatnonzero(errors==error))
    checks=dict(rows=n,unknown_interior_schur_rows=int(np.sum(trace[:,9]))if schur else 0,finite_rows=int(np.sum(trace[:,12])),
        endpoint_or_unidentified_rows=int(np.sum((trace[:,9]==0)&(trace[:,12]==0))),
        maximum_native_weighted_gradient_delta=float(np.max(trace[:,13],initial=0.)),maximum_tangent_projection_native=float(np.max(trace[:,14],initial=0.)))
    result=(residual,*blocks)if jacobian else residual
    if return_trace:return result,checks,trace
    return result,checks
