"""Independent full Brown14 forward/central chart with active sensor tilt.

Zero tilt delegates to the original source projector byte for byte. Nonzero
tilt preserves its Brown12 central branch, and certifies a positive OpenCV
homogeneous tilt denominator along the same axis-to-ray path. An unresolved
machine-precision domain boundary is invalid, never silently projected.
"""
from __future__ import annotations
import math
import types
import cv2
import numpy as np

from projection import project_central_valid as original_project_central_valid, project_fast
from ice_refine import _curve_vector_error as original_curve_vector_error


def tilt_homography(tau_x,tau_y):
    sx,cx=math.sin(tau_x),math.cos(tau_x)
    sy,cy=math.sin(tau_y),math.cos(tau_y)
    rx=np.array([[1.,0.,0.],[0.,cx,sx],[0.,-sx,cx]])
    ry=np.array([[cy,0.,-sy],[0.,1.,0.],[sy,0.,cy]])
    rotation=ry@rx
    project_z=np.array([[rotation[2,2],0.,-rotation[0,2]],
                        [0.,rotation[2,2],-rotation[1,2]],[0.,0.,1.]])
    return project_z@rotation


_POWER_TO_BERNSTEIN=np.array([[math.comb(i,k)/math.comb(10,k)if k<=i else 0.
                              for k in range(11)]for i in range(11)])


def _split_bernstein(values):
    work=values.copy();left=np.empty(11);right=np.empty(11)
    left[0]=work[0];right[-1]=work[-1]
    for i in range(1,11):
        work=.5*(work[:-1]+work[1:]);left[i]=work[0];right[-i-1]=work[-1]
    return left,right


def _tilt_positive_axis_domain(camera,normalized_xy):
    """Degree≤10 rational numerator; Bernstein convex-hull certificates.

    The original Brown12 guard certifies the rational denominator. Multiplying
    by it does not change the tilt denominator's sign. All polynomial powers
    are explicit; endpoint tests alone do not qualify the connecting domain.
    """
    xy=np.asarray(normalized_xy,float);h=tilt_homography(camera[23],camera[24])
    if not np.isfinite(h).all()or h[2,2]<=0.:return np.zeros(len(xy),bool)
    x,y=xy.T;r2=x*x+y*y;r4=r2*r2;r6=r4*r2
    k1,k2,p1,p2,k3,k4,k5,k6,s1,s2,s3,s4=camera[11:23]
    c=np.zeros((len(xy),11))
    c[:,0]=h[2,2];c[:,2]=h[2,2]*k4*r2;c[:,4]=h[2,2]*k5*r4;c[:,6]=h[2,2]*k6*r6
    linear=h[2,0]*x+h[2,1]*y
    c[:,1]+=linear;c[:,3]+=linear*k1*r2;c[:,5]+=linear*k2*r4;c[:,7]+=linear*k3*r6
    quadratic=(h[2,0]*(2*p1*x*y+p2*(r2+2*x*x)+s1*r2)+
               h[2,1]*(p1*(r2+2*y*y)+2*p2*x*y+s3*r2))
    c[:,2]+=quadratic;c[:,4]+=quadratic*k4*r2;c[:,6]+=quadratic*k5*r4;c[:,8]+=quadratic*k6*r6
    fourth=(h[2,0]*s2+h[2,1]*s4)*r4
    c[:,4]+=fourth;c[:,6]+=fourth*k4*r2;c[:,8]+=fourth*k5*r4;c[:,10]+=fourth*k6*r6
    bernstein=c@_POWER_TO_BERNSTEIN.T
    padding=2048.*np.finfo(float).eps*np.maximum(1.,np.sum(abs(c),axis=1))
    valid=np.isfinite(bernstein).all(axis=1)&(np.min(bernstein,axis=1)>padding)
    uncertain=np.flatnonzero(~valid&np.isfinite(bernstein).all(axis=1))
    for i in uncertain:
        pending=[(bernstein[i],0)];accepted=True
        while pending:
            values,depth=pending.pop()
            if np.min(values)>padding[i]:continue
            # Actual endpoint witness or machine-unresolved branch boundary.
            if values[0]<=padding[i]or values[-1]<=padding[i]or depth>=24:
                accepted=False;break
            left,right=_split_bernstein(values)
            pending.extend(((left,depth+1),(right,depth+1)))
        valid[i]=accepted
    return valid


def project_central_valid_full_brown(camera,xyz,*,with_jacobian=False):
    c=np.asarray(camera,float);points=np.ascontiguousarray(np.asarray(xyz,float).reshape(-1,3))
    if c.shape!=(25,)or not np.isfinite(c).all()or not np.isfinite(points).all():
        raise ValueError('Finite camera25 and finite original world points required')
    if np.all(c[23:25]==0.):
        return original_project_central_valid(c,points,with_jacobian=with_jacobian)
    base=c.copy();base[23:25]=0.
    _,_,central=original_project_central_valid(base,points,with_jacobian=False)
    try:
        projected=project_fast(c,points,with_jacobian=with_jacobian)
    except cv2.error:
        uv=np.full((len(points),2),np.nan);depth=np.full(len(points),np.nan)
        if with_jacobian:return uv,depth,np.zeros((2*len(points),11)),np.zeros(len(points),bool)
        return uv,depth,np.zeros(len(points),bool)
    uv,depth=projected[:2]
    valid=central&np.isfinite(uv).all(axis=1)&np.isfinite(depth)&(depth>.05)
    indices=np.flatnonzero(valid)
    if len(indices):
        rotation=cv2.Rodrigues(c[3:6])[0]
        local=(points[indices]-c[:3])@rotation.T
        valid[indices]=_tilt_positive_axis_domain(c,local[:,:2]/local[:,2,None])
    if with_jacobian:
        valid&=np.isfinite(projected[2].reshape(-1,2,11)).all(axis=(1,2))
        return uv,depth,projected[2],valid
    return uv,depth,valid


def point_projection_full_brown(camera,world):
    uv,depth,jac,central=project_central_valid_full_brown(camera,world,with_jacobian=True)
    if not np.all(central)or np.any(depth<=0.)or not np.isfinite(uv).all()or not np.isfinite(jac).all():
        raise ValueError('Actual full-Brown14 source point leaves positive central chart')
    return uv,np.asarray(jac).reshape(len(uv),2,11),depth


# Preserve the original finite-segment association and derivative function.
# Only this private clone's projector changes; original module bytes/globals
# and every programme camera path remain untouched.
_curve_globals=dict(original_curve_vector_error.__globals__)
_curve_globals['project_central_valid']=project_central_valid_full_brown
curve_vector_error_full_brown=types.FunctionType(original_curve_vector_error.__code__,
    _curve_globals,'curve_vector_error_full_brown',original_curve_vector_error.__defaults__,
    original_curve_vector_error.__closure__)
