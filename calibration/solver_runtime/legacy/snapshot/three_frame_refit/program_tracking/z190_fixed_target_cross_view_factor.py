"""Physical cross-view projection of a fixed camera pan-head target.

World positions and the fixed observer lever are explicit metres. A target
pivot is P_j, never its rotating lens centre or the centre of a visible body
box. Body-only evidence needs an explicit offset nuisance and declared bounds;
the caller must retain that nuisance, rather than silently treating it as zero.
"""
from __future__ import annotations
from types import SimpleNamespace
import cv2,numpy as np


def _xyz(value,name):
    result=np.asarray(value,float)
    if result.shape!=(3,)or not np.isfinite(result).all():raise ValueError(name+' must be three finite metre coordinates')
    return result


def fixed_target_projection(observer_pivot_xyz_m,target_pivot_xyz_m,observer_rvec,
                            log_native_focal,observer_lever_xyz_m,principal_px,k1,
                            *,target_offset_world_xyz_m=None):
    """Native pixels and exact derivatives with respect to P_i/P_j/R_i/logf.

    Y=R_i(P_j-P_i)-lever_i. If a separately declared body offset is supplied,
    Y=R_i(P_j+offset-P_i)-lever_i. Target rotation does not enter a fixed-pivot
    projection. The returned observer XYZ columns are per metre, not /100.
    """
    observer=_xyz(observer_pivot_xyz_m,'Observer pivot');target=_xyz(target_pivot_xyz_m,'Target pivot')
    lever=_xyz(observer_lever_xyz_m,'Fixed observer camera-local lever')
    rvec=_xyz(observer_rvec,'Observer rotation vector');pp=np.asarray(principal_px,float)
    if pp.shape!=(2,)or not np.isfinite(pp).all():raise ValueError('Explicit fixed native principal point required')
    offset=np.zeros(3)if target_offset_world_xyz_m is None else _xyz(target_offset_world_xyz_m,'Explicit target world offset')
    f=float(np.exp(float(log_native_focal)));k=float(k1)
    if not np.isfinite(f)or f<=0 or not np.isfinite(k):raise ValueError('Positive native focal and fixed finite k1 required')
    rotation,raw=cv2.Rodrigues(rvec);derivative=raw.reshape(3,3,3)
    delta=target+offset-observer;camera_xyz=rotation@delta-lever;z=float(camera_xyz[2])
    if not np.isfinite(camera_xyz).all()or z<=0:raise ValueError('Fixed target must have positive observer optical depth')
    n=camera_xyz[:2]/z;r2=float(n@n);radial=1.+k*r2
    if radial<=0 or 1.+3.*k*r2<=0:raise ValueError('Target outside central monotonic Brown k1 domain')
    uv=pp+f*n*radial
    brown=radial*np.eye(2)+2.*k*np.outer(n,n)
    normalized=np.c_[np.eye(2),-n]/z
    pixel_camera=f*brown@normalized
    observer_xyz=-pixel_camera@rotation;target_xyz=-observer_xyz
    rotation_tangent=np.column_stack([d@delta for d in derivative])
    view=np.c_[pixel_camera@rotation_tangent,uv-pp]
    return SimpleNamespace(uv_native_px=uv,optical_depth_m=z,observer_optical_center_xyz_m=observer+rotation.T@lever,target_world_xyz_m=target+offset,observer_xyz_m_jacobian=observer_xyz,target_xyz_m_jacobian=target_xyz,observer_view_R3_logf_jacobian=view,target_world_offset_m_jacobian=target_xyz.copy(),fixed_lever_camera_m_jacobian=-pixel_camera,target_rotation_jacobian=np.zeros((2,3)))


def native_bbox_support(projection,bounds_xy_px,sigma_native_px,*,allow_unconstrained_axes=()):
    """Outside a frozen native support rectangle; zero inside its real extent.

    The rectangle is a source interval, not a pixel centre. This primitive
    makes no claim that a camera body box is a pivot support box.
    """
    incoming=np.asarray(bounds_xy_px,object)
    if incoming.shape!=(2,2):raise ValueError('Native X/Y support intervals required')
    bounds=np.array([[(-np.inf if side==0 else np.inf)if value is None else float(value)for side,value in enumerate(axis)]for axis in incoming],float)
    sigma=np.asarray(sigma_native_px,float)
    if np.isnan(bounds).any()or np.any(bounds[:,0]>bounds[:,1])or np.isposinf(bounds[:,0]).any()or np.isneginf(bounds[:,1]).any():raise ValueError('Native support bounds contain NaN or invalid one-sided order')
    allowed=set(allow_unconstrained_axes)
    if not allowed.issubset({0,1}):raise ValueError('Explicit source-qualified unknown support axes must be X=0/Y=1')
    unbounded=~np.isfinite(bounds).any(axis=1)
    if any(unbounded[i]and i not in allowed for i in (0,1)):
        raise ValueError('Each native axis needs a finite bound or explicit source-qualified unknown component')
    if sigma.shape!=(2,)or not np.isfinite(sigma).all()or np.any(sigma<=0):raise ValueError('Explicit engineering native sigma2 required')
    uv=projection.uv_native_px;outside=uv-np.clip(uv,bounds[:,0],bounds[:,1])
    active=(uv<bounds[:,0])|(uv>bounds[:,1]);chain=active.astype(float)/sigma
    return SimpleNamespace(residual=outside/sigma,outside_support_native_px=outside,inside_support=~active,qualified_unknown_support_axes=sorted(allowed),support_bounds_xy_px=bounds,
        observer_xyz_m_jacobian=chain[:,None]*projection.observer_xyz_m_jacobian,
        target_xyz_m_jacobian=chain[:,None]*projection.target_xyz_m_jacobian,
        observer_view_R3_logf_jacobian=chain[:,None]*projection.observer_view_R3_logf_jacobian,
        target_world_offset_m_jacobian=chain[:,None]*projection.target_world_offset_m_jacobian,
        target_rotation_jacobian=np.zeros((2,3)),projection=projection)


def native_bbox_inequalities(projection,bounds_xy_px,sigma_native_px,*,allow_unconstrained_axes=()):
    """Actual finite support faces as scaled constraints g(q) <= 0.

    The lower face is (lower-uv)/sigma; the upper face is
    (uv-upper)/sigma. Unknown one-sided faces produce no fabricated row.
    These signed derivatives remain available inside the source support, unlike
    the active-only hinge Jacobian. Squared positive excess is the same interval
    objective as 0.5*||native_bbox_support.residual||^2. A hard constrained caller
    must handle an infeasible initial state rather than reject all its steps.
    """
    support=native_bbox_support(projection,bounds_xy_px,sigma_native_px,
        allow_unconstrained_axes=allow_unconstrained_axes)
    sigma=np.asarray(sigma_native_px,float);values=[];axes=[];sides=[];boundaries=[];chains=[]
    for axis in (0,1):
        for side,sign in ((0,-1.),(1,1.)):
            boundary=support.support_bounds_xy_px[axis,side]
            if not np.isfinite(boundary):continue
            values.append(sign*(projection.uv_native_px[axis]-boundary)/sigma[axis])
            axes.append(axis);sides.append('lower'if side==0 else'upper');boundaries.append(float(boundary))
            chain=np.zeros(2);chain[axis]=sign/sigma[axis];chains.append(chain)
    chain=np.asarray(chains,float).reshape(-1,2);g=np.asarray(values,float)
    result=SimpleNamespace(constraint_values_scaled=g,constraint_sense='<=0',
        constraint_axis_indices=np.asarray(axes,int),constraint_sides=tuple(sides),
        constraint_native_boundaries_px=np.asarray(boundaries,float),
        positive_excess_residual=np.maximum(g,0.),
        quadratic_excess_cost=.5*float(np.maximum(g,0.)@np.maximum(g,0.)),
        feasible=bool(np.all(g<=0.)),support=support,projection=projection)
    for name in ('observer_xyz_m_jacobian','target_xyz_m_jacobian',
                 'observer_view_R3_logf_jacobian','target_world_offset_m_jacobian',
                 'target_rotation_jacobian'):
        setattr(result,name,chain@getattr(projection,name))
    return result


def fixed_target_source_factor(source,observer_pivot_xyz_m,target_pivot_xyz_m,
                               observer_rvec,log_native_focal,observer_lever_xyz_m,
                               principal_px,k1,*,target_offset_world_xyz_m=None):
    """Qualified fixed-pivot support, or body evidence with an explicit nuisance.

    A body-only box without a declared pivot-to-body offset domain is refused.
    Supplying an offset does not make it a measured target pivot: the caller
    must optimize/profile its returned nuisance columns with the source bounds.
    """
    if source.get('source_visible')is not True or source.get('source_support_manually_qualified')is not True:
        raise ValueError('Cross-view source must be visibly and manually qualified')
    if source.get('model_projection_used_to_generate_or_select_uv')is not False:
        raise ValueError('Cross-view source support cannot be selected by model projection')
    if not source.get('physical_target_authority')or not source.get('sigma_semantics'):
        raise ValueError('Physical target/support and engineering sigma authority required')
    kind=source.get('physical_target_kind');nuisance=False
    if kind=='fixed_pan_head_pivot':
        if source.get('fixed_pivot_source_qualified')is not True or target_offset_world_xyz_m is not None:
            raise ValueError('Fixed pan-head pivot support must have explicit source identity without body offset')
    elif kind=='body_with_explicit_world_offset':
        if target_offset_world_xyz_m is None:raise ValueError('Rotating camera body cannot silently become the fixed target pivot')
        bounds=np.asarray(source['body_pivot_offset_world_bounds_m'],float);offset=_xyz(target_offset_world_xyz_m,'Body offset nuisance')
        if bounds.shape!=(3,2)or np.any(bounds[:,0]>bounds[:,1])or not np.isfinite(bounds).all()or np.any(offset<bounds[:,0])or np.any(offset>bounds[:,1]):
            raise ValueError('Body offset nuisance outside explicitly declared source/engineering domain')
        if not source.get('body_offset_bounds_authority'):raise ValueError('Body offset bound is not a measured pivot and needs its own authority')
        nuisance=True
    else:raise ValueError('A visible body centre/operator is not a qualified fixed pan-head target')
    projection=fixed_target_projection(observer_pivot_xyz_m,target_pivot_xyz_m,observer_rvec,log_native_focal,observer_lever_xyz_m,principal_px,k1,target_offset_world_xyz_m=target_offset_world_xyz_m)
    result=native_bbox_support(projection,source['native_support_bounds_xy_px'],source['sigma_native_px'],allow_unconstrained_axes=source.get('qualified_unknown_support_axes',()))
    result.requires_offset_nuisance_in_solver=nuisance;result.physical_target_kind=kind
    result.body_support_is_not_precise_pivot_measurement=nuisance
    return result


def fixed_target_source_inequalities(source,observer_pivot_xyz_m,target_pivot_xyz_m,
                                   observer_rvec,log_native_focal,observer_lever_xyz_m,
                                   principal_px,k1,*,target_offset_world_xyz_m=None):
    """Strictly qualified physical source, with actual support-face derivatives.

    No support centre or new UV is introduced. Body evidence keeps the same
    explicit offset nuisance requirement as fixed_target_source_factor.
    """
    qualified=fixed_target_source_factor(source,observer_pivot_xyz_m,target_pivot_xyz_m,
        observer_rvec,log_native_focal,observer_lever_xyz_m,principal_px,k1,
        target_offset_world_xyz_m=target_offset_world_xyz_m)
    result=native_bbox_inequalities(qualified.projection,source['native_support_bounds_xy_px'],
        source['sigma_native_px'],allow_unconstrained_axes=source.get('qualified_unknown_support_axes',()))
    result.requires_offset_nuisance_in_solver=qualified.requires_offset_nuisance_in_solver
    result.physical_target_kind=qualified.physical_target_kind
    result.body_support_is_not_precise_pivot_measurement=qualified.body_support_is_not_precise_pivot_measurement
    return result
