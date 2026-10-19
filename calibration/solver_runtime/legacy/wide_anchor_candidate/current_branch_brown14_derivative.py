"""Analytic current-chart Brown14 derivative of UNCHANGED finite source math.

All original UV/domain/nearest-segment evaluations are recomputed for every
camera.  Differentiation uses the selected branch at that current value, as
the original eleven-column provider already does; trials never freeze it.
"""
from __future__ import annotations
import cv2
import numpy as np
from acceleration import nearest_segments
from ice_refine import BoardIntersectionError
from .full_brown_projection import project_central_valid_full_brown


def projection_brown14(camera,world):
    """Original projected UV/depth/validity plus raw Brown14 pixel Jacobian."""
    c=np.asarray(camera,float);xyz=np.ascontiguousarray(np.asarray(world,float).reshape(-1,3))
    uv,depth,central=project_central_valid_full_brown(c,xyz)
    rotation=cv2.Rodrigues(c[3:6])[0];translation=-rotation@c[:3]
    fx,fy=np.exp(c[6:8]);k=np.array([[fx,0.,c[8]],[0.,fy,c[9]],[0.,0.,1.]])
    projected,jac=cv2.projectPoints(xyz,c[3:6],translation,k,c[11:25])
    native=projected.reshape(-1,2);yf=(native[:,1]-c[9])/fy
    native[:,0]+=c[10]*yf
    if not np.allclose(uv,native,rtol=0.,atol=2e-10,equal_nan=True):
        raise ValueError('Original full Brown14 UV and direct OpenCV UV disagree')
    raw=jac[:,10:24].copy()
    raw[0::2]+=c[10]/fy*raw[1::2]
    if not np.isfinite(raw).all():raise ValueError('Current full Brown14 derivative is not finite')
    return uv,depth,raw.reshape(len(xyz),2,14),central


def curve_current_brown14(camera,world,observed):
    """Original normalized-free native residual and analytic Brown14 J.

    Copies original finite nearest-foot algebra exactly, including clipped
    endpoint t'=0. It does not freeze observed UV, a station or a segment
    across evaluations. Domain gates and nearest ties are the original ones.
    """
    observed=np.asarray(observed,np.float64).reshape(-1,2)
    projected,depth,jac,central=projection_brown14(camera,world)
    start,finish=projected[:-1],projected[1:]
    direction=finish-start;length2=np.sum(direction*direction,axis=1)
    midpoint_central=project_central_valid_full_brown(camera,.5*(world[:-1]+world[1:]))[-1]
    valid=(central[:-1]&central[1:]&midpoint_central&np.isfinite(start).all(axis=1)&np.isfinite(finish).all(axis=1)
        &(depth[:-1]>0)&(depth[1:]>0)&(length2>1e-4))
    if not valid.any():raise BoardIntersectionError('ice_curve_no_central_segment',np.arange(len(observed)))
    index,selected_t,clipped_t,nearest=nearest_segments(observed,start,direction,length2,valid)
    residual=(nearest-observed).ravel()
    a=start[index];d=direction[index];u=observed-a;length2_selected=length2[index]
    ja=jac[index];jd=jac[index+1]-ja
    numerator_j=(-np.einsum('nij,ni->nj',ja,d)+np.einsum('ni,nij->nj',u,jd))
    length2_j=2.*np.einsum('ni,nij->nj',d,jd)
    t_j=(numerator_j-selected_t[:,None]*length2_j)/length2_selected[:,None]
    t_j[(selected_t<=0.)|(selected_t>=1.)]=0.
    nearest_j=ja+clipped_t[:,None,None]*jd+d[:,:,None]*t_j[:,None,:]
    return residual,nearest_j.reshape(-1,14)
