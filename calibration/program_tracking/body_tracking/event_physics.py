"""Reference factors on ONE saved/optimized programme state. NumPy/SciPy only.
No inferred source labels, ice tolerance, speed ceiling, or fixed spin centre.
Caller supplies verified flight interiors, units, uncertainty scales and FK chain.
"""
from __future__ import annotations
import numpy as np
from scipy.spatial.transform import Rotation


def time_weights(seconds, *, average=True):
    t=np.asarray(seconds,float)
    if t.ndim!=1 or len(t)<2 or not np.isfinite(t).all() or np.any(np.diff(t)<=0):
        raise ValueError('At least two ordered, finite physical times required')
    h=np.diff(t);w=np.r_[h[0]/2,(h[:-1]+h[1:])/2,h[-1]/2]
    return w/w.sum() if average else w


def ballistic_position_operator(seconds, gravity, sigma_position_m):
    """Eliminate free initial COM position/velocity; not pelvis or foot motion.

    Time-average weighting makes the objective stable under sampling refinement.
    This is a finite residual, NOT a precision/acceptance threshold.
    Return A,b such that residual=(A @ COM + b).ravel(). d r/d COM=kron(A,I3).
    Do not concatenate different flights: call separately, with unique event IDs.
    """
    t=np.asarray(seconds,float);g=np.asarray(gravity,float)
    if len(t)<3 or g.shape!=(3,) or not np.isfinite(g).all() or not np.isfinite(sigma_position_m) or sigma_position_m<=0:
        raise ValueError('Three samples, gravity vector and positive scale required')
    w=time_weights(t);tau=t-t[0];W=np.sqrt(w)
    Q=np.linalg.qr(W[:,None]*np.c_[np.ones(len(t)),tau],mode='reduced')[0]
    A=(np.eye(len(t))-Q@Q.T)*W[None,:]/sigma_position_m
    return A,-A@(.5*tau[:,None]**2*g)


def ballistic_residual(com_world_m,seconds,gravity,*,sigma_position_m):
    C=np.asarray(com_world_m,float)
    if C.shape!=(len(seconds),3) or not np.isfinite(C).all():raise ValueError('Finite Nx3 COM required')
    A,b=ballistic_position_operator(seconds,gravity,sigma_position_m)
    r=A@C+b
    return dict(residual=r.ravel(),operator=A,offset=b,cost=float(np.sum(r*r)/2))


def flight_momentum_residual(H_world,seconds,*,sigma_specific_H,phase):
    """Whole-flight constant unknown H, NOT H=0 and NOT constant root speed.

    H includes orbital + each segment's intrinsic angular momentum about COM.
    Grounded spins must NOT use this: external ice force/torque can change H.
    The mean is eliminated analytically; no per-frame momentum target is fitted.
    """
    H=np.asarray(H_world,float)
    if phase!='flight':raise ValueError('Momentum constancy is flight-only')
    if H.shape!=(len(seconds),3) or not np.isfinite(H).all() or sigma_specific_H<=0 or not np.isfinite(sigma_specific_H):
        raise ValueError('Finite Nx3 momentum and positive explicit scale required')
    w=time_weights(seconds);L=np.sqrt(w)[:,None]*(np.eye(len(w))-w[None,:])/sigma_specific_H
    r=L@H
    return dict(residual=r.ravel(),operator=L,mean_H=w@H,cost=float(np.sum(r*r)/2))


def angular_motion(rotations,seconds):
    """World/parent-frame SO(3) increments. Does not infer missed revolutions."""
    R=np.asarray(rotations,float);t=np.asarray(seconds,float);time_weights(t)
    if R.shape!=(len(t),3,3) or not np.isfinite(R).all():raise ValueError('Nx3x3 rotations required')
    phi=Rotation.from_matrix(R[1:]@R[:-1].swapaxes(-1,-2)).as_rotvec()
    return dict(interval_start=t[:-1],interval_stop=t[1:],omega=phi/np.diff(t)[:,None],
                shortest_increment_rad=np.linalg.norm(phi,axis=-1),source_path_may_be_aliased=True)


def hinge_normal_fan(normal_in_skate, rear_tangent, front_tangent, *, scale, weight=1.):
    """Finite runner endpoint-tangent condition, no saturation of argmin u.

    A global longitudinal endpoint is a minimum if its directional derivative
    has the wrong sign. Unlike a clipped argmin coordinate, tangent terms retain
    a useful orientation gradient when the witness is pinned at an endpoint.
    This does not fix side/inside edge, exact point, pressure or world height.
    """
    n=np.asarray(normal_in_skate,float);a=np.asarray(rear_tangent,float);b=np.asarray(front_tangent,float)
    if n.shape[-1]!=3 or a.shape!=(3,) or b.shape!=(3,) or scale<=0 or weight<0:raise ValueError('Invalid normal fan input')
    raw=np.stack([n@a,-n@b],-1);w=np.sqrt(weight)/scale
    r=np.maximum(raw,0)*w;D=(raw>0)[...,None]*np.stack([a,-b])*w
    return dict(residual=r,jacobian_normal=D,raw=raw)
