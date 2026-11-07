"""Read-only trajectory diagnostics. No fitting, alignment or acceptance limits.

Reference-derived (oracle) diagnostics are deliberately separated from primary
unaligned metrics. They MUST NOT be used as training/inference inputs.
"""
from __future__ import annotations
import numpy as np


def summary(x):
    x = np.asarray(x, dtype=float).ravel()
    good = np.isfinite(x)
    a = x[good]
    return dict(total=int(x.size), finite=int(good.sum()),
                median=float(np.median(a)) if a.size else None,
                p95=float(np.quantile(a, .95)) if a.size else None,
                maximum=float(a.max()) if a.size else None)


def root_diagnostics(predicted, reference, camera_centre):
    """Keep native-world errors primary; fit ONE oracle scale for diagnosis only."""
    p, r = np.asarray(predicted, float), np.asarray(reference, float)
    c = np.asarray(camera_centre, float)
    if p.ndim != 2 or p.shape[1:] != (3,) or r.shape != p.shape:
        raise ValueError('Expected matching Nx3 roots')
    if c.shape != (3,) or not all(np.isfinite(a).all() for a in (p, r, c)):
        raise ValueError('Finite values and one fixed camera centre required')
    ray = r-c
    distance = np.linalg.norm(ray, axis=1)
    if np.any(distance <= 0):
        raise ValueError('Reference root coincides with camera')
    ray = ray / distance[:, None]
    e = p-r
    radial = np.sum(e*ray, axis=1)
    transverse = np.linalg.norm(e-radial[:, None]*ray, axis=1)
    total_sse = float(np.sum(e*e))
    u, v = p-c, r-c
    denom = float(np.sum(u*u))
    if denom <= 0:
        raise ValueError('Degenerate prediction about camera')
    scale = float(np.sum(u*v)/denom)
    corrected = c+scale*u
    return dict(unaligned_world_position_m=summary(np.linalg.norm(e, axis=1)),
                ray_transverse_m=summary(transverse),
                ray_radial_squared_energy_fraction=(float(np.sum(radial**2)/total_sse)
                    if total_sse > 0 else 0.),
                predicted_to_reference_camera_distance_ratio=summary(np.linalg.norm(u,axis=1)/distance),
                oracle_scale_diagnostic=dict(scale_about_input_camera=scale,
                    residual_position_m=summary(np.linalg.norm(corrected-r, axis=1)),
                    fraction_squared_error_explained=(1-float(np.sum((corrected-r)**2))/total_sse
                        if total_sse > 0 else 0.),
                    reference_used=True, inference_allowed=False,
                    accuracy_improvement_claim=False,
                    caveat='Diagnostic nuisance fit, not a valid correction or benchmark score'))


def segment_lengths(predicted, reference, reference_valid):
    p,r=np.asarray(predicted,float),np.asarray(reference,float)
    valid=np.asarray(reference_valid,bool)
    if p.shape!=r.shape or p.ndim!=3 or p.shape[1:]!=(2,3) or valid.shape!=p.shape[:2]:
        raise ValueError('Expected Nx2x3 endpoints and Nx2 validity')
    ok=valid.all(1)&np.isfinite(r).all((1,2))&np.isfinite(p).all((1,2))
    a=np.linalg.norm(p[:,1]-p[:,0],axis=1)[ok]
    b=np.linalg.norm(r[:,1]-r[:,0],axis=1)[ok]
    if not len(a) or np.any(b<=0):
        raise ValueError('No nondegenerate comparison segments')
    return dict(predicted_m=summary(a),reference_m=summary(b),
                median_length_ratio=float(np.median(a)/np.median(b)),
                semantics='Model/reference landmark definitions may differ; not measured bone-length truth')


def motion_rates(points, pts, time_base):
    p=np.asarray(points,float); t=np.asarray(pts)
    if t.ndim!=1 or t.dtype.kind not in 'iu' or len(t)<2 or np.any(np.diff(t)<=0):
        raise ValueError('Strictly increasing original integer PTS required')
    if p.shape[0]!=len(t) or p.shape[-1]!=3 or not np.isfinite(p).all():
        raise ValueError('Finite points on original clock required')
    if len(time_base)!=2 or min(time_base)<=0:
        raise ValueError('Positive rational time base required')
    dt=np.diff(t)*time_base[0]/time_base[1]
    shape=(len(dt),)+(1,)*(p.ndim-2)
    return np.linalg.norm(np.diff(p,axis=0),axis=-1)/dt.reshape(shape)


def native_line_observation_rank(predicted, a, b, radius):
    """Exact local unweighted 2D dead-zone residual/J; no camera assumptions.

    Demonstrates that inside a source band the observation supplies no local
    gradient; outside, a segment-interior match constrains one image direction.
    """
    p,a,b=map(lambda x:np.asarray(x,float),(predicted,a,b))
    if any(x.shape!=(2,) or not np.isfinite(x).all() for x in (p,a,b)) or radius<=0:
        raise ValueError('Finite 2D points and positive source radius required')
    v=b-a; vv=float(v@v)
    if vv<=0:raise ValueError('Degenerate source line')
    alpha=float(np.clip((p-a)@v/vv,0,1)); nearest=a+alpha*v
    N=np.eye(2)-np.outer(v,v)/vv if 0<alpha<1 else np.eye(2)
    z=(p-nearest)/radius; r=float(np.linalg.norm(z))
    if r<=1:return np.zeros(2),np.zeros((2,2))
    J=((1-1/r)*np.eye(2)+np.outer(z,z)/r**3)@N/radius
    return (1-1/r)*z,J
