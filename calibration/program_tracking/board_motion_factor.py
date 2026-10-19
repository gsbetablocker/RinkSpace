"""Measured board pixel pairs on the existing first-visible segmented wall.

The eliminated source intersection is recomputed from each exact row camera.
This module adds no material landmark, no ad-plane approximation and no image
correspondence. Its caller owns source qualification and verified wall extent.
"""
from __future__ import annotations
from types import SimpleNamespace
import cv2
import numpy as np
from .ice_motion_factor import _camera_inventory


def _world_ray(camera, pixel, tolerance):
    from rinkspace.calibration_companion.wide_angle_camera import brown_central_inverse
    fx, fy = np.exp(camera[6:8]); uv=np.asarray(pixel,float).reshape(1,2).copy()
    uv[:,0]-=camera[10]*(uv[:,1]-camera[9])/fy
    K=np.array([[fx,0.,camera[8]],[0.,fy,camera[9]],[0.,0.,1.]])
    inverse=brown_central_inverse(SimpleNamespace(K_px=K,distortion_opencv=camera[11:23]),uv,tolerance_px=tolerance)
    if not inverse['valid'][0]:raise ValueError('Measured board pixel has no central Brown inverse')
    return np.r_[inverse['normalized_xy'][0],1.] @ cv2.Rodrigues(camera[3:6])[0]


def _world_rays(cameras,pixels,tolerance):
    """Broadcast independent original inverses; retain scalar failure ordering."""
    from .brown_inverse_rows import brown_central_inverse_rows,implementation_identity
    implementation_identity()  # A changed original producer is never a fallback.
    matrices=[];deskewed=[]
    for camera,pixel in zip(cameras,pixels,strict=True):
        fx,fy=np.exp(camera[6:8]);uv=np.asarray(pixel,float).reshape(1,2).copy()
        uv[:,0]-=camera[10]*(uv[:,1]-camera[9])/fy
        matrices.append(np.array([[fx,0.,camera[8]],[0.,fy,camera[9]],[0.,0.,1.]]))
        deskewed.append(uv[0])
    try:
        inverse=brown_central_inverse_rows(np.asarray(matrices),cameras[:,11:23],
            np.asarray(deskewed),tolerance_px=tolerance)
    except (ValueError,ZeroDivisionError,OverflowError,np.linalg.LinAlgError):
        # An exceptional row must not preempt an earlier original visibility
        # failure. Re-run the unchanged scalar loop in its original order.
        return None,None
    rays=np.empty((len(pixels),3))
    for i,camera in enumerate(cameras):
        if inverse['valid'][i]:
            rays[i]=np.r_[inverse['normalized_xy'][i],1.]@cv2.Rodrigues(camera[3:6])[0]
    return rays,inverse['valid']


def evaluate_board_motion(surface, camera_a, camera_b, source_uv, target_uv, *,
                          jacobian=True, inverse_tolerance_px=1e-9):
    """Return r, source/target conditioned camera15 Jacobians and hit XYZ.

    surface is the existing ReferenceBoardPair geometry/visibility context
    with an explicit allowed-panels inventory; its saved XY/cameras are unused.
    Both measured rows must already be applied by the caller's native clock.
    A different first visible wall or an occluded target is a domain rejection.
    """
    from .first_shot_source_kernel import _interfaces
    a,b=np.asarray(source_uv,float),np.asarray(target_uv,float)
    if a.ndim!=2 or a.shape[1:]!=(2,) or a.shape!=b.shape or not len(a) or not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('Equal finite nonempty N-by-2 measured board pairs required')
    if not isinstance(jacobian,(bool,np.bool_)) or not np.isfinite(inverse_tolerance_px) or inverse_tolerance_px<=0:raise ValueError('Explicit Jacobian mode and positive inverse precision required')
    ca,cb=_camera_inventory(camera_a,len(a)),_camera_inventory(camera_b,len(a));project,brown,_,_=_interfaces();res=[];left_j=[];right_j=[];worlds=[]
    rays,inverse_valid=_world_rays(ca,a,float(inverse_tolerance_px))
    for i,(left,right,ua,ub)in enumerate(zip(ca,cb,a,b,strict=True)):
        try:
            if rays is None:
                ray=_world_ray(left,ua,float(inverse_tolerance_px))
            else:
                if not inverse_valid[i]:raise ValueError('Measured board pixel has no central Brown inverse')
                ray=rays[i]
            hit=surface._hits_from_rays(left[:3],ray[None]);surface._require(hit);world=hit.xyz
            target_hit=surface._hits_from_rays(right[:3],world-right[:3]);surface._require(target_hit)
            if np.linalg.norm(target_hit.xyz-world)>1e-4:raise ValueError('Measured board pair becomes target-wall occluded')
            pa=project(left,world,with_jacobian=bool(jacobian));pb=project(right,world,with_jacobian=bool(jacobian))
            if not pa[-1][0] or not pb[-1][0]:raise ValueError('Measured board pair leaves original central forward optical domain')
            if np.linalg.norm(pa[0][0]-ua)>max(16*inverse_tolerance_px,1e-7):raise ValueError('Source board ray/projection round trip failed')
            res.append(pb[0][0]-ub);worlds.append(world[0])
            if jacobian:
                ja=np.concatenate((np.asarray(pa[2]).reshape(1,2,11),brown(left,world)),axis=2)[0];jb=np.concatenate((np.asarray(pb[2]).reshape(1,2,11),brown(right,world)),axis=2)[0]
                edge=surface._edges[int(hit.panel[0])];length=np.linalg.norm(edge)
                if length<=0:raise ValueError('Degenerate existing board panel')
                tangent=np.array([[edge[0]/length,0.],[edge[1]/length,0.],[0.,1.]])
                # dPi/dX=-dPi/dC, respecting the original C/100 chart.
                A=(-ja[:,:3]/100.)@tangent;B=(-jb[:,:3]/100.)@tangent
                left_j.append(-B@np.linalg.solve(A,ja));right_j.append(jb)
        except (ValueError,np.linalg.LinAlgError) as error:
            raise ValueError('Invalid measured board motion pair at index '+str(i)+': '+str(error))from error
    size=2*len(a)
    return(np.asarray(res).reshape(-1),np.asarray(left_j).reshape(size,15)if jacobian else np.empty((size,0)),np.asarray(right_j).reshape(size,15)if jacobian else np.empty((size,0)),np.asarray(worlds))
