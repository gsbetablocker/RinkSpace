"""Bound temporary evaluation memory without partitioning the fitted state."""
import numpy as np


def slice_reference(value,indices,total):
    return {key:(item[indices] if isinstance(item,np.ndarray) and item.ndim and len(item)==total else item)
            for key,item in value.items()}


def slice_delta(delta,indices):
    if hasattr(delta,'owner'):
        from .absolute_motion_state import MotionDelta
        result=MotionDelta(np.asarray(delta)[indices],delta.owner,delta.parameters,delta.sample_times[indices])
        if getattr(delta,'skate_relative_R',None) is not None:result.skate_relative_R=delta.skate_relative_R[indices]
        return result
    return np.asarray(delta)[indices]


def world_batches(reference,base,delta,*,samples):
    """Only image/registration-consumed fields survive each FK batch."""
    n=len(delta);arrays={}
    keys=('P','Q','local_R','native','coco','gear','frames','world','hand_local_R','skate_relative_R')
    for first in range(0,n,samples):
        from .memory_guard import check_memory_budget
        check_memory_budget()
        ids=np.arange(first,min(first+samples,n))
        state=reference.world(slice_reference(base,ids,n),slice_delta(delta,ids))
        for key in keys:
            if state.get(key) is not None:arrays.setdefault(key,[]).append(state[key])
    return {key:np.concatenate(value,axis=0) for key,value in arrays.items()}


def contact_batches(problem,x,times,jacobian,*,samples=512):
    """Exact global quadrature and witnesses; temporary FK/J scratch is tiled."""
    from scipy.sparse import vstack
    from .skate_contact import trapezoid_time_weights
    from .body_clock import sample_period
    from .skate_physics import fk_state,constraints,body_constraint_jacobian
    ref=problem.reference;weights=trapezoid_time_weights(times,singleton_duration_s=sample_period(ref.p));pieces=[];matrices=[]
    for first in range(0,len(times),samples):
        from .memory_guard import check_memory_budget
        check_memory_budget()
        last=min(first+samples,len(times));ts=times[first:last]
        base=ref.at(ts);delta=problem.body.state.increments(x[17:],ts);state=fk_state(ref,base,delta)
        part=constraints(problem.binding,state,ts,problem.origin,problem.evidence,time_weights=weights[first:last])
        if jacobian:matrices.append(body_constraint_jacobian(ref,state,delta,problem.body.state.active,problem.body.state.basis(ts),part))
        part['rows']=[(row+first,foot,q,sign)for row,foot,q,sign in part['rows']]
        for row in part['support_groups']:row['frame_index']+=first
        pieces.append(part)
    out={}
    for key in ('margin_m','g','weights','clearance','support_height','witness','values','times'):
        out[key]=np.concatenate([p[key]for p in pieces])
    for key in ('rows','labels','row_upper_vertex','support_groups'):out[key]=sum((p[key]for p in pieces),[])
    out['features']={key:(np.concatenate([p['features'][key]for p in pieces])if isinstance(value,np.ndarray)else value)for key,value in pieces[0]['features'].items()}
    if jacobian:out['body_jac']=vstack(matrices,format='csr')
    return out
