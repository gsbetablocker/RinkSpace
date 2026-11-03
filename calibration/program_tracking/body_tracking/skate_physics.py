"""Batched exact lower-skate witnesses and analytic existing-FK chain.

This is the vectorized counterpart of skate_contact.frame_residuals. Stable
rows are (physical time id, foot nonpenetration or source support group),
irrespective of which exact arc/solid supplies the active witness.
"""
import numpy as np
from scipy.sparse import coo_matrix
from .fullbody_state import matrices
from .fullbody_fit import right_jacobian
from .skate_contact import trapezoid_time_weights

ALLOWANCE={'runner':.002,'toe_pick':.005,'solid_guard':0.}
NORMALIZATION=.005
FEASIBILITY_TOLERANCE=.001
SUPPORT_BAND=(-.002,.003)
# A legacy source support row represents one native 50 Hz observation in
# this pipeline. Its objective weight is fixed, unlike collision quadrature.
# This is NOT shutter/readout duration or an asserted duration of contact.
LEGACY_SOURCE_SUPPORT_WEIGHT_S=256/12800


def source_support_weight_s(record):
    """Fixed measure of a discrete source observation, independent of grid.

    Old S01 evidence has no weight field and keeps a 0.02 s native-cadence
    measure, matching the existing image-block convention. New evidence
    should record source_support_weight_s explicitly. Confidence metadata
    is not silently treated as a calibrated statistical weight.
    """
    value=record.get('source_support_weight_s',LEGACY_SOURCE_SUPPORT_WEIGHT_S)
    if isinstance(value,(bool,np.bool_)):
        raise ValueError('source_support_weight_s must be finite and positive')
    value=float(value)
    if not np.isfinite(value) or value<=0:
        raise ValueError('source_support_weight_s must be finite and positive')
    return value


def fk_state(reference,ref,delta):
    """The same native22 FK, without recomputing COCO skinning for physics."""
    P=ref['P']+delta[:,:3];Q=ref['Q']@matrices(delta[:,3:6])
    local=ref['local_R']@matrices(delta[:,6:].reshape(-1,21,3))
    body=reference.body;n=len(P);rot=np.broadcast_to(np.eye(3),(n,22,3,3)).copy()
    pos=np.broadcast_to(body.J[:22],(n,22,3)).copy()
    for j in range(1,22):
        a=body.parents[j];rot[:,j]=rot[:,a]@local[:,j-1]
        pos[:,j]=pos[:,a]+np.einsum('nij,j->ni',rot[:,a],body.J[j]-body.J[a])
    native=(pos-body.J[0])*body.scale
    R,O=reference.binding.transforms(P,Q,native,rot)
    nativeW=P[:,None]+np.einsum('nij,nkj->nki',Q,native)
    return dict(P=P,Q=Q,local_R=local,frames=rot,native=native,native_world=nativeW,R=R,O=O)


def batch_features(binding,R,O,state=None):
    """Exact continuous arcs; exact vertex minima of supplied solid meshes."""
    p=binding.packet();template=p['template'];ratio=binding.ratio
    normals=R[...,2,:];heights=[];points=[];ids=[];semantics=[];eligible=[]
    for arc in template['contact_arcs']:
        c=np.asarray(arc['center']);a=np.asarray(arc['axis_cos']);b=np.asarray(arc['axis_sin'])
        A=normals@a;B=normals@b
        lo,hi=arc['theta_start_rad'],arc['theta_end_rad']
        interior=np.arctan2(-B,-A)
        interior+=2*np.pi*np.round(((lo+hi)/2-interior)/(2*np.pi))
        ts=np.stack([np.full_like(A,lo),np.full_like(A,hi),np.clip(interior,lo,hi)],-1)
        h=A[...,None]*np.cos(ts)+B[...,None]*np.sin(ts)
        t=np.take_along_axis(ts,np.argmin(h,axis=-1)[...,None],axis=-1)[...,0]
        q=c+arc['radius_m']*(np.cos(t)[...,None]*a+np.sin(t)[...,None]*b)
        heights.append((O[...,2]+np.sum(normals*q,axis=-1))*ratio);points.append(q)
        ids.append(arc['id']);semantics.append(arc['semantic']);eligible.append(True)
    for group in template['contact_point_groups']:
        v=np.asarray(group['points']);z=np.einsum('nfj,vj->nfv',normals,v)
        which=np.argmin(z,axis=-1);q=v[which]
        heights.append((O[...,2]+np.sum(normals*q,axis=-1))*ratio);points.append(q)
        ids.append(group['id']);semantics.append(group['semantic']);eligible.append(group.get('support_eligible',False))
    upper_index=None
    if state is not None and 'native' in state:
        alpha=binding.deformation(state['native'],state['frames'])
        upper=binding.deform(p['upper'],alpha)
        z=np.einsum('nfj,nfvj->nfv',normals,upper)
        upper_index=np.argmin(z,axis=-1)
        q=upper[np.arange(len(R))[:,None],np.arange(2)[None],upper_index]
        heights.append((O[...,2]+np.sum(normals*q,axis=-1))*ratio);points.append(q)
        ids.append('upper_shell_guard');semantics.append('solid_guard');eligible.append(False)
    return dict(height=np.stack(heights,-1),local_point=np.stack(points,-2),
        ids=ids,semantics=semantics,eligible=eligible,upper_index=upper_index)


def constraints(binding,state,times,source_origin,evidence):
    features=batch_features(binding,state['R'],state['O'],state);h=features['height']
    allow=np.array([ALLOWANCE[s] for s in features['semantics']]);clear=h+allow
    k=np.argmin(clear,axis=-1);n=len(times);ni=np.arange(n)[:,None];feet=np.arange(2)[None]
    clearance=clear[ni,feet,k];q=features['local_point'][ni,feet,k]
    witness=state['O']+np.einsum('nfij,nfj->nfi',state['R'],q)
    margins=[];rows=[];labels=[];weights=[];values=[];upper_rows=[];gaps=np.full(n,np.nan)
    support_groups=[]
    tw=trapezoid_time_weights(times,singleton_duration_s=.02)
    for i,t in enumerate(times):
        time_id=int(round((t+source_origin)*1e9))
        for f in range(2):
            margins.append(clearance[i,f]);rows.append((i,f,q[i,f],1.))
            upper_rows.append(int(features['upper_index'][i,f]) if features['ids'][k[i,f]]=='upper_shell_guard' else -1)
            labels.append((time_id,('left','right')[f]+'_nonpenetration'));weights.append(tw[i]);values.append(clearance[i,f])
        # A source image constrains only its own exposure midpoint, not a
        # fabricated dense run of contact labels between reviewed images.
        matches=[r for r in evidence if abs(t+source_origin-r['pts']/12800.)<1e-8]
        for r in matches:
            members=r.get('feet',[0,1]);kind=r.get('mode','glide')
            # Known flight and unknown contact receive only the universal
            # nonpenetration rows above. They are not support observations.
            if kind in ('airborne','unknown'):continue
            weight=source_support_weight_s(r)
            allowed_ids=binding.packet()['template']['support_feature_ids'][kind]
            choices=[(h[i,f,j],f,j) for f in members for j,name in enumerate(features['ids']) if name in allowed_ids]
            height,f,j=min(choices);point=features['local_point'][i,f,j]
            # Retain the historical minimum-height diagnostic; individual
            # groups are also exported so double support is not collapsed.
            gaps[i]=height if np.isnan(gaps[i]) else min(gaps[i],height)
            support_groups.append(dict(frame_index=i,id=r['id'],height_m=float(height),
                witness_foot=int(f),witness_feature=features['ids'][j],weight_s=weight))
            for side,bound,sign in [('lower',SUPPORT_BAND[0],1.),('upper',SUPPORT_BAND[1],-1.)]:
                margins.append(sign*(height-bound));rows.append((i,f,point,sign))
                upper_rows.append(-1)
                labels.append((time_id,r['id']+':'+side));weights.append(weight);values.append(height)
    return dict(margin_m=np.asarray(margins),g=np.asarray(margins)/NORMALIZATION,
        rows=rows,labels=labels,weights=np.asarray(weights),clearance=clearance,row_upper_vertex=upper_rows,
        support_height=gaps,support_groups=support_groups,witness=witness,features=features,
        values=np.asarray(values),times=np.asarray(times))


def body_constraint_jacobian(reference,state,delta,active,B,evaluation):
    """Envelope theorem at each active witness, followed by sparse spline chain."""
    n=len(delta);d=len(active);local=np.zeros((len(evaluation['rows']),d))
    root_axes=state['Q']@right_jacobian(delta[:,3:6]);ratio=reference.binding.ratio
    joint_axes={}
    for j in (1,2,4,5,7,8):
        joint_axes[j]=state['Q']@state['frames'][:,j]@right_jacobian(delta[:,6+(j-1)*3:6+j*3])
    ancestors=({1,4,7},{2,5,8})
    upper_partials={}
    if any(i>=0 for i in evaluation.get('row_upper_vertex',[])):
        # Upper-cuff deformation depends only on each ankle relative to its
        # shank. Six local driver columns are checked directly; rigid lower
        # contacts and the rest of the body keep the analytic FK chain.
        h=1e-5
        for joint in (7,8):
            for axis in range(3):
                a=6+(joint-1)*3+axis
                if a not in active:continue
                plus=delta.copy();minus=delta.copy();plus[:,a]+=h;minus[:,a]-=h
                sp=fk_state(reference,reference.at(evaluation['times']),plus)
                sm=fk_state(reference,reference.at(evaluation['times']),minus)
                vp=reference.binding.deform(reference.binding.packet()['upper'],reference.binding.deformation(sp['native'],sp['frames']))
                vm=reference.binding.deform(reference.binding.packet()['upper'],reference.binding.deformation(sm['native'],sm['frames']))
                upper_partials[a]=(vp-vm)/(2*h)
    for r,(i,f,q,sign) in enumerate(evaluation['rows']):
        point=state['O'][i,f]+state['R'][i,f]@q
        for c,a in enumerate(active):
            if a<3:derivative=float(a==2)
            elif a<6:derivative=np.cross(root_axes[i,:,a-3],point-state['P'][i])[2]
            else:
                j=(a-6)//3+1;axis=(a-6)%3
                derivative=(np.cross(joint_axes[j][i,:,axis],point-state['native_world'][i,j])[2]
                    if j in ancestors[f] else 0.)
            upper_vertex=evaluation.get('row_upper_vertex',[-1]*len(evaluation['rows']))[r]
            if upper_vertex>=0 and a in upper_partials:
                derivative+=state['R'][i,f,2]@upper_partials[a][i,f,upper_vertex]
            local[r,c]=sign*ratio*derivative/NORMALIZATION
    fr=np.array([r[0] for r in evaluation['rows']]);rr=[];cc=[];vv=[]
    for row in range(len(fr)):
        knots=np.flatnonzero(B[fr[row]])
        for k in knots:
            rr.extend([row]*d);cc.extend(k*d+np.arange(d));vv.extend(B[fr[row],k]*local[row])
    J=coo_matrix((vv,(rr,cc)),shape=(len(fr),B.shape[1]*d)).tocsr();J.eliminate_zeros()
    return J
