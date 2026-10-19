"""Isolated analytic course columns for explicitly declared finite-board rows.

The original motion residual, old camera/lever/optics Jacobians, source roles,
finite extent guards, and frozen profile weights are untouched. Known planes
have an exact ray intersection and no inverse-depth nuisance or Schur projector.
Only selected board rows are materialized; no Nx6 dense source matrix is made.
Plane coefficients follow the native convention n.X + d = 0 in scene units.
"""
from __future__ import annotations
from dataclasses import dataclass
import os
import numpy as np
from z190_motion_native import _inverse_brown_native
from z190_moving_center_motion import MovingDepthDomainError


def _select(value,shape,indices):
    """Broadcast as a view then copy only explicit selected source rows."""
    return np.ascontiguousarray(np.broadcast_to(np.asarray(value,float),shape)[indices])


@dataclass(frozen=True)
class FinitePlaneMotionCache:
    source_indices: np.ndarray
    source_direction: np.ndarray
    target_direction: np.ndarray
    target_translation: np.ndarray
    source_centres: np.ndarray
    target_focal: np.ndarray
    principal: np.ndarray
    k1: float
    sigma: np.ndarray
    profile_weight: np.ndarray
    original_batch_count: int
    original_batch_source_direction_reused: bool


def prepare_finite_plane_motion_cache(native_inputs,declared_board_mask,*,source_direction_selected=None):
    """Build a compact current-camera cache, optionally reusing exact native rays.

    source_direction_selected must be the selected rows of the existing native
    source Brown/einsum cache. If unavailable, Brown is inverted on this compact
    batch; the returned audit explicitly does not claim full-batch ray bitexact.
    In either mode the original source residual and original Jacobian are never
    called or replaced by this helper.
    """
    k=native_inputs;n=len(k['a']);mask=np.asarray(declared_board_mask)
    if mask.dtype!=np.dtype(bool) or mask.shape!=(n,):
        raise ValueError('Explicit original-length boolean finite-board declaration required')
    indices=np.flatnonzero(mask);m=len(indices)
    planes=_select(k['planes'],(n,4),indices)
    if not np.isfinite(planes).all() or np.any(np.sum(planes[:,:3]**2,axis=1)<=0):
        raise ValueError('Every declared board source row must retain an actual finite plane')
    ra=_select(k['ra'],(n,3,3),indices);rb=_select(k['rb'],(n,3,3),indices)
    ca=_select(k['ca'],(n,3),indices);cb=_select(k['cb'],(n,3),indices)
    fb=_select(k['fb'],(n,),indices)
    pp=np.asarray(k['pp'],float);k1=float(k['k1'])
    if pp.shape!=(2,)or not np.isfinite(pp).all()or not np.isfinite(k1):
        raise ValueError('Original explicit constant native principal point and k1 required')
    reused=source_direction_selected is not None
    if reused:
        # Do not change the caller's array or even its writeability flag.
        direction=np.array(source_direction_selected,dtype=float,order='C',copy=True)
        if direction.shape!=(m,3):raise ValueError('Exact selected original native source directions required')
    else:
        aa=_select(k['a'],(n,2),indices);fa=_select(k['fa'],(n,),indices)
        if np.any(fa<=0):raise ValueError('Original finite positive source focal required')
        if m:
            threads=int(os.environ.get('RINKSPACE_MOTION_THREADS','1'))
            if threads<1:raise ValueError('Positive motion thread count required')
            na,_=_inverse_brown_native(aa,fa,np.ascontiguousarray(pp),k1,threads)
            direction=np.einsum('ni,nij->nj',np.c_[na,np.ones(m)],ra)
        else:direction=np.empty((0,3))
    target=np.einsum('nij,nj->ni',rb,direction)
    translation=np.einsum('nij,nj->ni',rb,ca-cb)
    sig=np.asarray(k['sigma'],float)
    if sig.ndim==1:sig=sig.reshape(-1,1)
    sigma=_select(sig,(n,2),indices)
    weights=np.ones((m,2))if k.get('profile_weight')is None else _select(k['profile_weight'],(n,2),indices)
    for array in (direction,target,translation,ca,fb,sigma,weights):
        if not np.isfinite(array).all():raise ValueError('Original selected finite motion inputs must be finite')
    if np.any(fb<=0)or np.any(sigma<=0)or np.any(weights<0):
        raise ValueError('Original positive focal/sigma and frozen nonnegative weights required')
    values=(indices,direction,target,translation,ca,fb,pp.copy(),sigma,weights)
    for value in values:value.flags.writeable=False
    return FinitePlaneMotionCache(*values[:7],k1,*values[7:],n,reused)


def finite_plane_motion_columns(cache,planes_selected,plane_jacobians,*,sigma_normalized=False):
    """Return original native residual rows and their compact 2M xK plane J.

    plane_jacobians is Mx4xK for the declared current physical coordinates;
    it includes both normal and offset derivatives. Original cameras/lenses/
    levers stay fixed for these columns. Rows are never robust-weighted here:
    sigma_normalized divides once by the original source sigma, while source
    cost and H/g must still use the original loss/scales and solver weights.
    """
    m=len(cache.source_indices);plane=np.asarray(planes_selected,float);jp=np.asarray(plane_jacobians,float)
    if (plane.shape!=(m,4)or jp.ndim!=3 or jp.shape[:2]!=(m,4)or jp.shape[2]<1
            or not np.isfinite(plane).all()or not np.isfinite(jp).all()):
        raise ValueError('Exactly the declared M finite planes and Mx4xK current physical derivatives required')
    if np.any(np.sum(plane[:,:3]**2,axis=1)<=0):
        raise ValueError('Course derivative may not turn a declared finite source into unknown depth')
    n=plane[:,:3];distance=-plane[:,3]-np.sum(n*cache.source_centres,axis=1)
    numerator=np.sum(n*cache.source_direction,axis=1)
    with np.errstate(divide='ignore',invalid='ignore',over='ignore'):
        rho=numerator/distance
    q=cache.target_direction;t=cache.target_translation
    invalid=(rho<=0)|~np.isfinite(rho)|(q[:,2]+rho*t[:,2]<=0)
    if np.any(invalid):
        raise MovingDepthDomainError('reviewed_plane_positive_depth_invalid',cache.source_indices[invalid])
    if np.any(q[:,2]<=0):
        raise MovingDepthDomainError('source_ray_infinity_behind_target_line_chart',cache.source_indices[q[:,2]<=0])
    tau=rho/(1.+rho)
    # This is exactly d(rho)/(dq) for n.X+d=0 and a fixed moving camera.
    drho=(np.einsum('nik,ni->nk',jp[:,:3],cache.source_direction)+rho[:,None]*(
        jp[:,3]+np.einsum('nik,ni->nk',jp[:,:3],cache.source_centres)))/distance[:,None]
    dtau=drho*(1.-tau[:,None])**2
    tangent=t-q;Y=q+tau[:,None]*tangent
    if np.any(Y[:,2]<=0)or not np.isfinite(Y).all():
        raise MovingDepthDomainError('target_cheirality',cache.source_indices[(Y[:,2]<=0)|~np.isfinite(Y).all(axis=1)])
    normalized=Y[:,:2]/Y[:,2,None];x=normalized[:,0];y=normalized[:,1];k1=cache.k1;f=cache.target_focal
    radial=1.+k1*(x*x+y*y)
    D=np.empty((m,2,2));D[:,0,0]=radial+2.*k1*(x*x);D[:,0,1]=D[:,1,0]=2.*k1*x*y
    D[:,1,1]=radial+2.*k1*(y*y)
    FY=np.empty((m,2,3));FY[:,:,:2]=f[:,None,None]*D/Y[:,None,2,None]
    FY[:,:,2]=-f[:,None]*(D[:,:,0]*x[:,None]+D[:,:,1]*y[:,None])/Y[:,2,None]
    ftan=np.einsum('nij,nj->ni',FY,tangent)
    columns=ftan[:,:,None]*dtau[:,None,:]
    if sigma_normalized:columns=columns/cache.sigma[:,:,None]
    if not np.isfinite(columns).all():raise ValueError('Finite complete course motion Jacobian required')
    rows=np.column_stack((2*cache.source_indices,2*cache.source_indices+1)).reshape(-1)
    return rows,columns.reshape(2*m,jp.shape[2]),dict(selected_finite_source_pairs=m,
        current_physical_plane_derivative_columns=int(jp.shape[2]),
        original_batch_pairs=cache.original_batch_count,returned_native_residual_rows=2*m,
        original_batch_source_direction_reused=cache.original_batch_source_direction_reused,
        known_depth_not_profiled=True,unknown_depth_schur_not_applied=True,
        native_kernel_or_original_r_J_modified=False,source_or_sigma_roles_changed=False,
        sigma_divisions=1 if sigma_normalized else 0,robust_weight_multiplications=0,
        frozen_profile_weight_values_untouched=True,unknown_and_ice_source_rows_not_materialized=True,
        plane_convention='n.X+d=0; all spatial quantities use original scene units',
        positive_known_depth_domain_checked=True,finite_source_extent_guard_must_still_run_in_original_source=True)
