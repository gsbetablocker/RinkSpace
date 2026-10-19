"""Verified strict-double OpenMP backend for the moving-centre source rows.

Build products are local to this isolated source snapshot. Loading requires a
matching source and DLL hash; no compilation or dependency install occurs in
a solve. The original NumPy implementation remains the comparison backend.
"""
from __future__ import annotations
from calibration.solver_runtime.config import native_build_directory
import ctypes
import hashlib
import json
import os
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
SOURCE=HERE/'native'/'moving_center_motion.c'
DEST = native_build_directory('motion', SOURCE)
_library=None
_kernel=None
_nuisance_kernel=None
_inverse_kernel=None
_load_error=None
_manifest=None

def _sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

try:
    _manifest=json.loads((DEST/'build.json').read_text(encoding='utf8'))
    binary=DEST/_manifest.get('binary','moving_center_motion.dll')
    if _manifest['source_sha256']!=_sha(SOURCE) or _manifest['dll_sha256']!=_sha(binary):
        raise ValueError('Moving-centre native source/binary hash mismatch')
    _library=ctypes.CDLL(str(binary))
    _kernel=_library.motion_strict
    _kernel.argtypes=([ctypes.c_int64]+[ctypes.c_void_p]*21+[ctypes.c_double]+
                     [ctypes.c_void_p]*3+[ctypes.c_double]+[ctypes.c_int]*3+
                     [ctypes.c_void_p]*6)
    _kernel.restype=ctypes.c_int
    _nuisance_kernel=_library.motion_strict_nuisance
    _nuisance_kernel.argtypes=_kernel.argtypes+[ctypes.c_void_p]
    _nuisance_kernel.restype=ctypes.c_int
    _inverse_kernel=_library.inverse_brown_strict
    _inverse_kernel.argtypes=[ctypes.c_int64,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,
        ctypes.c_double,ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p]
    _inverse_kernel.restype=ctypes.c_int
except (OSError,ValueError,KeyError,AttributeError)as exc:
    _kernel=None;_nuisance_kernel=None;_inverse_kernel=None
    _load_error=str(exc)

def metadata():
    return dict(available=_kernel is not None,implementation='strict_double_openmp_row_fused',
        threads=int(os.environ.get('RINKSPACE_MOTION_THREADS',os.cpu_count()or 1)),
        native_build=_manifest,load_error=_load_error)

def _inverse_brown_native(uv,focal,principal,k1,threads):
    n=len(uv);result=np.empty((n,2));derivative=np.empty((n,2));errors=np.zeros(n,np.int32)
    code=_inverse_kernel(n,uv.ctypes.data,focal.ctypes.data,principal.ctypes.data,k1,threads,
        result.ctypes.data,derivative.ctypes.data,errors.ctypes.data)
    if code:raise RuntimeError('Native Brown inverse workspace allocation failed')
    if np.any(errors):
        from z190_moving_center_motion import MovingDepthDomainError
        if np.any(errors==1):raise MovingDepthDomainError('noninvertible_source_Brown',np.flatnonzero(errors==1))
        raise MovingDepthDomainError('source_Brown_root_unconverged',np.arange(n))
    return result,derivative

def native_motion(*,a,b,fa,fb,ra,rb,da,db,ca,cb,dca_shared,dcb_shared,dca_view,dcb_view,
                  pp,k1,planes,jacobian,sigma,profile_weight=None,minimum_source_depth_scene_units=0.,
                  schur=True,return_trace=False,return_nuisance=False):
    if _kernel is None:raise RuntimeError('Build matching isolated motion kernel with build_motion_native.py: '+str(_load_error))
    a=np.asarray(a,float);n=len(a)
    if a.shape!=(n,2) or np.shape(b)!=(n,2):raise ValueError('Original a/b must be Nx2')
    floor=float(minimum_source_depth_scene_units)
    if floor<0 or not np.isfinite(floor):raise ValueError('Finite nonnegative conditional minimum source depth required')
    if return_nuisance and (not jacobian or not schur):
        raise ValueError('Eliminated-depth moments require the original Schur Jacobian call')
    def arr(x,shape):return np.ascontiguousarray(np.broadcast_to(np.asarray(x,float),shape))
    arrays=[arr(x,shape)for x,shape in (
        (a,(n,2)),(b,(n,2)),(fa,(n,)),(fb,(n,)),(ra,(n,3,3)),(rb,(n,3,3)))]
    # Residual-only calls never inspect derivatives; valid null pointers avoid
    # materializing the six large derivative arrays in trial evaluations.
    derivatives=[None,None]if not jacobian else[arr(da,(n,3,3,3)),arr(db,(n,3,3,3))]
    arrays+=derivatives+[arr(ca,(n,3)),arr(cb,(n,3))]
    arrays+=([None]*4 if not jacobian else[arr(x,(n,3,4))for x in (dca_shared,dcb_shared,dca_view,dcb_view)])
    principal=arr(pp,(2,));planes=arr(planes,(n,4))
    sig=np.asarray(sigma,float)
    if sig.ndim==1:sig=sig.reshape(-1,1)
    sig=arr(sig,(n,2));weight=np.ones((n,2))if profile_weight is None else arr(profile_weight,(n,2))
    if np.any(arrays[2]<=0)or np.any(arrays[3]<=0):raise ValueError('Explicit fixed PP and positive focal required')
    if np.any(sig<=0)or np.any(weight<0)or not np.isfinite(weight).all():raise ValueError('Positive original sigma/frozen nonnegative weights required')
    # Preserve the batch-wide Newton stop rule and Brown root validation;
    # profile and all J/Schur work are row-fused in the native backend.
    threads=int(os.environ.get('RINKSPACE_MOTION_THREADS',os.cpu_count()or 1))
    if threads<1:raise ValueError('Positive RINKSPACE_MOTION_THREADS required')
    inverse_a,inverse_af=_inverse_brown_native(arrays[0],arrays[2],principal,float(k1),threads)
    inverse_b,_=_inverse_brown_native(arrays[1],arrays[3],principal,float(k1),threads)
    inverses=[np.ascontiguousarray(x)for x in(inverse_a,inverse_af,inverse_b)]
    local=np.c_[inverse_a,np.ones(n)]
    source_direction=np.einsum('ni,nij->nj',local,arrays[4])
    target_direction=np.einsum('nij,nj->ni',arrays[5],source_direction)
    target_translation=np.einsum('nij,nj->ni',arrays[5],arrays[8]-arrays[9])
    geometry=[np.ascontiguousarray(x)for x in(source_direction,target_direction,target_translation)]
    residual=np.empty(2*n);blocks=[np.empty((2*n,4))for _ in range(3)]if jacobian else[None]*3
    trace=np.empty((n,16));errors=np.zeros(n,np.int32)
    nuisance=np.empty((n,14))if return_nuisance else None
    def ptr(x):return None if x is None else x.ctypes.data
    arguments=(n,*(ptr(x)for x in arrays),*(ptr(x)for x in inverses),*(ptr(x)for x in geometry),ptr(principal),float(k1),ptr(planes),ptr(sig),ptr(weight),
        floor,int(jacobian),int(schur),threads,ptr(residual),*(ptr(x)for x in blocks),ptr(trace),ptr(errors))
    code=(_nuisance_kernel(*arguments,ptr(nuisance))if return_nuisance else _kernel(*arguments))
    if code:raise RuntimeError(f'Moving-centre native kernel returned {code}')
    bad=np.flatnonzero(errors)
    if len(bad):
        from z190_moving_center_motion import MovingDepthDomainError
        names={1:'noninvertible_source_Brown',2:'source_Brown_root_unconverged',3:'target_cheirality',
            4:'no_positive_target_depth_interval',5:'reviewed_plane_positive_depth_invalid',6:'source_ray_infinity_behind_target_line_chart'}
        error=errors[bad[0]]
        raise MovingDepthDomainError(names[int(error)],np.flatnonzero(errors==error))
    checks=dict(rows=n,unknown_interior_schur_rows=int(np.sum(trace[:,9]))if schur else 0,
        finite_rows=int(np.sum(trace[:,12])),endpoint_or_unidentified_rows=int(np.sum((trace[:,9]==0)&(trace[:,12]==0))),
        maximum_native_weighted_gradient_delta=float(np.max(trace[:,13],initial=0.)),
        maximum_tangent_projection_native=float(np.max(trace[:,14],initial=0.)))
    result=(residual,*blocks)if jacobian else residual
    if return_nuisance:
        return result,checks,trace,dict(
            profile_nuisance_tangent_native=nuisance[:,:2],
            fixed_lambda_projection_factors=nuisance[:,2:].reshape(n,3,4),
            local_factor_order=('view_a','view_b','shared'),
            original_kernel_outputs_unchanged=True)
    if return_trace:return result,checks,trace
    return result,checks
