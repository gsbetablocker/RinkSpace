"""Explicit finite material/plane and paint-to-course intersection helpers."""
from __future__ import annotations
from copy import deepcopy
import numpy as np


def certified_finite_course_binding(course,world,*,identity,authority,z_policy='contact',z_values=None):
    """Certify a declared WORLD course, then retain its original segment t.

    Caller must already supply the physical semantic identity. This routine
    uses no image, camera, source residual, or nearest projection from UV.
    Projection is only a membership test against the original finite world.
    An off-course sampled chord fails; it is not changed into a new curve.
    """
    if not identity or not authority:raise ValueError('Original physical course identity and authority required')
    world=np.asarray(world,float)
    if world.ndim!=2 or world.shape[1]!=3 or not np.isfinite(world).all():
        raise ValueError('Original finite world required')
    base=course.sample_perimeter()[0]
    edge=np.roll(base,-1,axis=0)-base
    length2=np.sum(edge[:,:2]**2,axis=1)
    delta=world[:,None,:2]-base[None,:,:2]
    t=np.clip(np.sum(delta*edge[None,:,:2],axis=2)/length2[None],0.,1.)
    near=base[None,:,:2]+t[:,:,None]*edge[None,:,:2]
    d2=np.sum((near-world[:,None,:2])**2,axis=2)
    i=np.argmin(d2,axis=1);row=np.arange(len(world));tt=t[row,i]
    # Match original accepted-source finite-world membership precision.
    if np.any(np.sqrt(d2[row,i])>1e-7):
        raise ValueError('Declared original world is not on its finite board course; no automatic snapping')
    cert=course.bind_finite_segments(world,i,tt,z_policy=z_policy,z_values=z_values,
        authority=f'{authority}; physical identity={identity}')
    return cert


def vertical_plane_value_jacobian(start,end,jstart,jend):
    """Plane n.X+d=0, n=[dy,-dx,0]/|edge|; derivative has matching orientation."""
    start,end=np.asarray(start,float),np.asarray(end,float)
    ja,jb=np.asarray(jstart,float),np.asarray(jend,float)
    if start.shape!=(3,)or end.shape!=(3,)or ja.shape!=jb.shape or ja.ndim!=2 or ja.shape[0]!=3:
        raise ValueError('Two finite world endpoints and equal 3xK derivatives required')
    edge=end-start;jed=jb-ja
    raw=np.array([edge[1],-edge[0],0.]);jraw=np.vstack((jed[1],-jed[0],np.zeros(ja.shape[1])))
    length=float(np.linalg.norm(raw))
    if not length>0:raise ValueError('Nondegenerate original finite vertical plane extent required')
    n=raw/length;jn=(jraw-n[:,None]*(n@jraw)[None])/length
    d=-n@start;jd=-start@jn-n@ja
    return np.r_[n,d],np.vstack((jn,jd))


def bound_vertical_plane(binding,q=None,*,original_plane=None):
    world,j=binding.evaluate(q,with_jacobian=True)
    if len(world)!=2:raise ValueError('Two declared finite physical extent endpoints required')
    plane,jplane=vertical_plane_value_jacobian(world[0],world[1],j[0],j[1])
    if original_plane is not None:
        old=np.asarray(original_plane,float)
        expected=vertical_plane_value_jacobian(*binding.evaluate(binding.geometry.initial),
            np.zeros_like(j[0]),np.zeros_like(j[1]))[0]
        if not np.allclose(old,expected,rtol=0,atol=1e-9):raise ValueError('Original plane orientation/extent certificate differs')
        if q is None or np.array_equal(np.asarray(q),binding.geometry.initial):plane=old.copy()
    return dict(plane=plane,plane_jacobian=jplane,extent_world=world,extent_world_jacobian=j,
        z_range_scene=np.array([0.,binding.geometry._scene_q(binding.geometry.initial if q is None else np.asarray(q))[5]]),
        source_support_domain_changed=False,unknown_depth_row_relabelled=False)


def _implicit(course,xy,q):
    """Current analytic primitive signed domain; value/XY gradient/six J."""
    a,b,rx,ry,p,_=course._scene_q(q);x,y=xy
    ax,ay=abs(x),abs(y);sx=np.sign(x);sy=np.sign(y);scale=1/course.scene_unit_m
    jq=np.zeros(6)
    if ax>a-rx and ay>b-ry:
        xx=(ax-a+rx)/rx;yy=(ay-b+ry)/ry
        xp,yp=xx**p,yy**p;dx=p*xx**(p-1);dy=p*yy**(p-1)
        f=xp+yp-1;grad=np.array([sx*dx/rx,sy*dy/ry])
        jq[0]=-dx/rx*scale;jq[1]=-dy/ry*scale
        jq[2]=dx*(1-xx)/rx*scale;jq[3]=dy*(1-yy)/ry*scale
        jq[4]=xp*np.log(xx)+yp*np.log(yy)
    elif ax>a-rx or ax-a>=ay-b:
        f=ax-a;grad=np.array([sx,0.]);jq[0]=-scale
    else:
        f=ay-b;grad=np.array([0.,sy]);jq[1]=-scale
    return f,grad,jq


def transverse_intersections(course,endpoints,q=None,*,endpoint_jacobian=None):
    """Clip the declared infinite painted line to the actual smooth course.

    The original line's direction is preserved, including its small tilt.
    These are world-boundary intersections, never inferred observed endpoints.
    Return original endpoint order, analytic combined world J, and branches.
    Only an originally authenticated wall-terminating line is admissible.
    """
    q=course.initial if q is None else course.validate_parameters(q)
    old=np.asarray(endpoints,float)
    if old.shape!=(2,3)or not np.isfinite(old).all()or np.any(old[:,2]!=0):
        raise ValueError('Two finite original painted ice-line endpoints required')
    ja=np.zeros((2,3,0))if endpoint_jacobian is None else np.asarray(endpoint_jacobian,float)
    if ja.ndim!=3 or ja.shape[:2]!=(2,3):raise ValueError('Original endpoint Jacobian2x3xK required')
    k=ja.shape[-1];a=old[0,:2];d=old[1,:2]-a
    if not np.linalg.norm(d)>0:raise ValueError('Original line direction must be nondegenerate')
    # Validate base endpoints against the original physical analytic primitive.
    if max(abs(_implicit(course,e[:2],course.initial)[0])for e in old)>2e-10:
        raise ValueError('This line lacks authenticated original boundary endpoints; no extrapolation')
    middle=.5
    if _implicit(course,a+middle*d,q)[0]>=0:
        raise ValueError('Original painted line no longer crosses current ice interior')
    low,high=-1.,2.
    while _implicit(course,a+low*d,q)[0]<0:low=2*low-1
    while _implicit(course,a+high*d,q)[0]<0:high=2*high+1
    roots=[]
    for left,right,increasing in ((low,middle,False),(middle,high,True)):
        for _ in range(90):
            m=.5*(left+right);f=_implicit(course,a+m*d,q)[0]
            if (f>0)==increasing:right=m
            else:left=m
        roots.append(.5*(left+right))
    world=np.zeros((2,3));jac=np.zeros((2,3,k+6));report=[]
    for i,t in enumerate(roots):
        xy=a+t*d;_,normal,jq=_implicit(course,xy,q)
        basej=ja[0,:2]+t*(ja[1,:2]-ja[0,:2])
        den=float(normal@d)
        if den==0:raise ValueError('Declared line tangent to current course has no unique intersection derivative')
        tj=-np.r_[normal@basej,jq]/den
        jac[i,:2]=np.c_[basej,np.zeros((2,6))]+d[:,None]*tj
        world[i,:2]=xy
        report.append(dict(line_t=t,normal_xy=normal.tolist(),transversality=den,
            derivative_is_branch_specific=True,physical_endpoint_observed=False))
    if np.array_equal(q,course.initial):world=old.copy()
    return world,jac,report
