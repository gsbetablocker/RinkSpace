"""Isolated row-wise extension of the existing central Brown inverse.

No solver/factor imports or installation hooks. The homotopy, Newton bound,
backtracking, branch guards and precision are the original algorithm. Only
independent polynomial-lens rows are broadcast; nonzero rational denominator
rows retain the original scalar inverse, including its cubic-extrema checks.
"""
from functools import lru_cache
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from rinkspace.calibration_companion import wide_angle_camera as original

ORIGINAL_SHA256='571434ddb6c5fcc3b04197d04331b90ac7a60da62a11a76ff2ba2565caa4297d'


@lru_cache(maxsize=1)
def _original_identity():
    path=Path(original.__file__).resolve()
    actual=hashlib.sha256(path.read_bytes()).hexdigest()
    if actual!=ORIGINAL_SHA256:raise ValueError('Original central Brown inverse producer changed')
    return dict(path=str(path),sha256=actual)


def _map_rows(xy, distortion):
    # Same expressions and operation order as original._brown_map_jacobian.
    k1,k2,p1,p2,k3,k4,k5,k6,s1,s2,s3,s4=np.asarray(distortion,float).T
    x,y=np.asarray(xy,float).T
    r2=x*x+y*y; numerator=1.+k1*r2+k2*r2*r2+k3*r2*r2*r2
    denominator=1.+k4*r2+k5*r2*r2+k6*r2*r2*r2
    safe_denominator=np.where(np.abs(denominator)>=1e-12,denominator,
        np.where(denominator<0,-1e-12,1e-12))
    radial=numerator/safe_denominator
    numerator_slope=k1+2.*k2*r2+3.*k3*r2*r2
    denominator_slope=k4+2.*k5*r2+3.*k6*r2*r2
    slope=(numerator_slope*safe_denominator-numerator*denominator_slope)/safe_denominator**2
    predicted=np.column_stack([x*radial+2.*p1*x*y+p2*(r2+2.*x*x)+s1*r2+s2*r2*r2,
        y*radial+p1*(r2+2.*y*y)+2.*p2*x*y+s3*r2+s4*r2*r2])
    j11=radial+2.*x*x*slope+2.*p1*y+6.*p2*x+2.*x*(s1+2.*s2*r2)
    j12=2.*x*y*slope+2.*p1*x+2.*p2*y+2.*y*(s1+2.*s2*r2)
    j21=2.*x*y*slope+2.*p1*x+2.*p2*y+2.*x*(s3+2.*s4*r2)
    j22=radial+2.*y*y*slope+6.*p1*y+2.*p2*x+2.*y*(s3+2.*s4*r2)
    eigen=.5*(j11+j22-np.sqrt((j11-j22)**2+(j12+j21)**2))
    return predicted,j11,j12,j21,j22,eigen


def _polynomial_denominator(xy, distortion):
    # Keep original zero-times-overflow/NaN behavior; never replace by ones.
    r2=np.sum(xy*xy,axis=1);d=np.asarray(distortion,float)
    return np.minimum(1.,1.+r2*(d[:,5]+r2*(d[:,6]+r2*d[:,7])))


def _polynomial_inverse(K,distortion,pixels,*,continuation_steps,max_iterations,tolerance_px):
    focal=np.diagonal(K,axis1=1,axis2=2)[:,:2]
    target=(pixels-K[:,:2,2])/focal
    n=len(target);xy=np.zeros_like(target)
    alive=np.isfinite(target).all(axis=1)
    tolerance=tolerance_px/np.max(focal,axis=1)
    for stage in range(1,continuation_steps+1):
        goal=target*(stage/continuation_steps)
        converged=~alive
        for _ in range(max_iterations):
            indices=np.flatnonzero(alive&~converged)
            if not len(indices):break
            z=xy[indices]
            predicted,a,b,c,d,eigen=_map_rows(z,distortion[indices])
            error=predicted-goal[indices];err2=np.sum(error*error,axis=1)
            done=err2<=tolerance[indices]*tolerance[indices]
            converged[indices[done]]=True
            indices,z,error,err2,a,b,c,d,eigen=[v[~done]for v in(indices,z,error,err2,a,b,c,d,eigen)]
            if not len(indices):continue
            determinant=a*d-b*c
            valid=(eigen>1e-9)&(determinant>1e-16)&(_polynomial_denominator(z,distortion[indices])>1e-9)
            alive[indices[~valid]]=False
            indices,z,error,err2,a,b,c,d,determinant=[v[valid]for v in(indices,z,error,err2,a,b,c,d,determinant)]
            if not len(indices):continue
            step=np.column_stack([(d*error[:,0]-b*error[:,1])/determinant,
                (a*error[:,1]-c*error[:,0])/determinant])
            step*=np.minimum(1.,.15/np.maximum(np.linalg.norm(step,axis=1),1e-30))[:,None]
            factor=np.ones(len(indices));accepted=np.zeros(len(indices),bool)
            for _ in range(18):
                todo=np.flatnonzero(~accepted)
                if not len(todo):break
                proposal=z[todo]-factor[todo,None]*step[todo]
                pred,_,_,_,_,eig=_map_rows(proposal,distortion[indices[todo]])
                _,_,_,_,_,mid_eig=_map_rows((z[todo]+proposal)/2.,distortion[indices[todo]])
                improves=np.sum((pred-goal[indices[todo]])**2,axis=1)<err2[todo]
                good=improves&(eig>1e-9)&(mid_eig>1e-9)&(_polynomial_denominator(proposal,distortion[indices[todo]])>1e-9)
                take=todo[good]
                xy[indices[take]]=proposal[good];accepted[take]=True
                factor[todo[~good]]*=.5
            alive[indices[~accepted]]=False
        alive&=converged
    prediction,_,_,_,_,eigen=_map_rows(xy,distortion)
    error=np.linalg.norm((prediction-target)*focal,axis=1)
    denominator=_polynomial_denominator(xy,distortion)
    valid=alive&(eigen>1e-9)&(denominator>1e-9)&(error<=tolerance_px*2.)
    return dict(normalized_xy=xy,valid=valid,reprojection_error_px=error,
        minimum_eigenvalue=eigen,minimum_denominator_on_axis_path=denominator)


def brown_central_inverse_rows(K_px,distortion_opencv,pixels_uv,*,continuation_steps=8,
        max_iterations=24,tolerance_px=1e-5):
    """Independent original inverses for K[N,3,3], Brown12[N,12], UV[N,2].

    The caller applies its existing skew removal and exact row camera first.
    This API does not assume or modify any camera centre, rotation or clock.
    Nonzero rational rows use the original single-row implementation.
    """
    _original_identity()
    pixels=np.asarray(pixels_uv,float).reshape(-1,2)
    K=np.asarray(K_px,float);distortion=np.asarray(distortion_opencv,float);n=len(pixels)
    if K.shape!=(n,3,3)or distortion.shape!=(n,12):
        raise ValueError('One exact K and Brown12 per original pixel required')
    result=dict(normalized_xy=np.empty((n,2)),valid=np.empty(n,bool),
        reprojection_error_px=np.empty(n),minimum_eigenvalue=np.empty(n),
        minimum_denominator_on_axis_path=np.empty(n))
    # Original Python scalar division by max(fx,fy)==0 can raise. Keep
    # that exception behavior by dispatching those rows to the same scalar.
    focal_max=np.max(np.diagonal(K,axis1=1,axis2=2)[:,:2],axis=1)
    scalar_rows=np.any(distortion[:,5:8]!=0.,axis=1)|(focal_max==0.)
    polynomial=np.flatnonzero(~scalar_rows)
    if len(polynomial):
        values=_polynomial_inverse(K[polynomial],distortion[polynomial],pixels[polynomial],
            continuation_steps=continuation_steps,max_iterations=max_iterations,tolerance_px=tolerance_px)
        for key in result:result[key][polynomial]=values[key]
    for index in np.flatnonzero(scalar_rows):
        values=original.brown_central_inverse(SimpleNamespace(K_px=K[index],distortion_opencv=distortion[index]),
            pixels[index:index+1],continuation_steps=continuation_steps,
            max_iterations=max_iterations,tolerance_px=tolerance_px)
        for key in result:result[key][index]=values[key][0]
    return result


def implementation_identity():
    return dict(schema='rinkspace.isolated-row-wise-original-brown-inverse.v1',
        original_producer=dict(_original_identity()),producer=dict(path=str(Path(__file__).resolve()),
            sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),
        same_homotopy_Newton_bound_backtracking_branch_guards_and_precision=True,
        per_ray_focal_normalized_tolerance=True,rational_cubic_extrema_scalar_original=True,
        source_pixels_camera_clock_or_factor_model_modified=False,installed_into_solver=False)
