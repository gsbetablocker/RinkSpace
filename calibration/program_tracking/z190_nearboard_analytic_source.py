"""Opt-in analytic derivative of the existing finite near-cap support value.

The source operator, central-domain checks, subdivision and scan supports are
unchanged. Differentiate its selected segment/interval branch locally rather
than perturbing camera C/100 by a width that crosses a support hinge. Neither
this module nor its caller changes the default absolute evaluator.
"""
import numpy as np
from projection import project_central_valid
from z190_source_camera_diagnostic import absolute_value_jac15,brown4_world_jac
from z190_secondary_confirmed_nearboard_source import NEAR_KIND
from secondary_nearboard_top_source import nearboard_interval_residual


def _interval_branch_derivative(a,b,ja,jb,low,high,y):
    direction=b-a
    length2=np.maximum(np.sum(direction*direction,axis=1),1e-12)
    with np.errstate(divide='ignore',invalid='ignore'):
        crossing_t=(y-a[:,1])/direction[:,1]
    crossing_x=a[:,0]+crossing_t*direction[:,0]
    crossing=(np.isfinite(crossing_t)&(crossing_t>=0.)&(crossing_t<=1.)&
              (crossing_x>=low)&(crossing_x<=high))
    if np.any(crossing):return np.zeros((2,15))
    candidates=[]
    for endpoint in (a,b):
        q=np.c_[np.clip(endpoint[:,0],low,high),np.full(len(endpoint),y)]
        candidates.append(endpoint-q)
    unclipped=[]
    for qx in (low,high):
        point=np.array([qx,y]);t=np.sum((point-a)*direction,axis=1)/length2
        unclipped.append(t)
        candidates.append(a+np.clip(t,0.,1.)[:,None]*direction-point)
    candidates=np.stack(candidates)
    kind,i=np.unravel_index(np.argmin(np.sum(candidates*candidates,axis=2)),candidates.shape[:2])
    if kind<2:
        endpoint=(a,b)[kind][i];result=(ja,jb)[kind][i].copy()
        # The nearest point on the horizontal source interval follows X only
        # inside its support. Y remains a genuine residual component.
        if low<endpoint[0]<high:result[0]=0.
        return result
    point=np.array([(low,high)[kind-2],y]);d=direction[i];jd=jb[i]-ja[i]
    t=float(unclipped[kind-2][i]);clipped=float(np.clip(t,0.,1.))
    if 0.<t<1.:
        numerator=-np.einsum('ij,i->j',ja[i],d)+np.einsum('i,ij->j',point-a[i],jd)
        denominator=2.*np.einsum('i,ij->j',d,jd)
        dt=(numerator-t*denominator)/length2[i]
    else:dt=np.zeros(15)
    return ja[i]+clipped*jd+d[:,None]*dt[None,:]


def analytic_nearboard_absolute_value_jac15(camera,rows,jacobian=True):
    values=[];blocks=[]
    for row in rows:
        if row.get('curve_observation_kind')!=NEAR_KIND:
            result=absolute_value_jac15(camera,[row],jacobian)
            if jacobian:value,matrix=result;values.append(value);blocks.append(matrix)
            else:values.append(result)
            continue
        world=np.asarray(row['world'],float);measurements=row['source_horizontal_intervals']
        if len(measurements)!=len(row['uv']):raise ValueError('Typed near source row count changed')
        # Always replay the original value and all its full central chord
        # domain checks. The derivative cannot admit an unvalidated value.
        raw=nearboard_interval_residual(camera,world,measurements,
            source_keys=tuple(map(int,row['source_native_members'])))
        scale=np.repeat(np.asarray(row['sigma'],float),2);values.append(raw/scale)
        if not jacobian:continue
        uv,_,j11,valid=project_central_valid(camera,world,with_jacobian=True)
        midpoint=project_central_valid(camera,.5*(world[:-1]+world[1:]))[-1]
        segments=np.flatnonzero(valid[:-1]&midpoint&valid[1:])
        j=np.concatenate([j11.reshape(-1,2,11),brown4_world_jac(camera,world)],axis=2)
        a,b=uv[segments],uv[segments+1];ja,jb=j[segments],j[segments+1]
        matrix=np.stack([_interval_branch_derivative(a,b,ja,jb,
            *map(float,m['x_interval_px']),float(m['y']))for m in measurements]).reshape(-1,15)
        blocks.append(matrix/scale[:,None])
    value=np.concatenate(values)if values else np.empty(0)
    if not jacobian:return value
    return value,np.vstack(blocks)if blocks else np.empty((0,15))
