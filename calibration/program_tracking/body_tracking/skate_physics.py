"""Batched exact lower-skate witnesses and analytic existing-FK chain.

This is the vectorized counterpart of skate_contact.frame_residuals. Stable
rows are (physical time id, foot nonpenetration or source support group),
irrespective of which exact arc/solid supplies the active witness.
"""
import numpy as np
from scipy.sparse import coo_matrix,csr_matrix
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
    from .sequence_skate_state import relative_rotation
    relative=relative_rotation(reference,ref,delta)
    R,O=reference.binding.transforms(P,Q,native,rot,skate_relative_R=relative)
    nativeW=P[:,None]+np.einsum('nij,nkj->nki',Q,native)
    return dict(skate_relative_R=relative,P=P,Q=Q,local_R=local,frames=rot,native=native,native_world=nativeW,R=R,O=O,**({} if 'hand_local_R' not in ref else {'hand_local_R':ref['hand_local_R']}))


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
        alpha=binding.deformation(state['native'],state['frames'],skate_relative_R=state.get('skate_relative_R'))
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
    """Envelope theorem at each active witness, followed by sparse spline chain.

    All rows keep their current witness, sign and active-column ordering.  The
    scalar cross products are evaluated in arrays, rather than a Python loop
    over every constraint/column pair.  In particular the optional upper-row
    default is allocated once, never once per matrix element.
    """
    active=np.asarray(active,dtype=int);d=len(active);rows=evaluation['rows']
    count=len(rows);local=np.zeros((count,d));ratio=reference.binding.ratio
    if not count or not d:return csr_matrix((count,B.shape[1]*d))
    fr=np.fromiter((r[0] for r in rows),dtype=int,count=count)
    foot=np.fromiter((r[1] for r in rows),dtype=int,count=count)
    q=np.asarray([r[2] for r in rows]);sign=np.fromiter((r[3] for r in rows),dtype=float,count=count)
    upper_vertex=np.asarray(evaluation.get('row_upper_vertex',np.full(count,-1)),dtype=int)
    rotation=state['R'][fr,foot]
    point=state['O'][fr,foot]+np.einsum('nij,nj->ni',rotation,q)
    local[:,active==2]=1.
    root_columns=np.flatnonzero((active>=3)&(active<6))
    if len(root_columns):
        axes=(state['Q']@right_jacobian(delta[:,3:6]))[fr]
        arm=point-state['P'][fr]
        cross_z=axes[:,0,:]*arm[:,1,None]-axes[:,1,:]*arm[:,0,None]
        local[:,root_columns]=cross_z[:,active[root_columns]-3]
    for j,f in ((1,0),(4,0),(7,0),(2,1),(5,1),(8,1)):
        columns=np.flatnonzero((active>=6+(j-1)*3)&(active<6+j*3))
        selected=np.flatnonzero(foot==f)
        if not len(columns) or not len(selected):continue
        frames=fr[selected]
        axes=(state['Q']@state['frames'][:,j]@
            right_jacobian(delta[:,6+(j-1)*3:6+j*3]))[frames]
        arm=point[selected]-state['native_world'][frames,j]
        cross_z=axes[:,0,:]*arm[:,1,None]-axes[:,1,:]*arm[:,0,None]
        local[np.ix_(selected,columns)]=cross_z[:,active[columns]-(6+(j-1)*3)]
    selected=np.flatnonzero(upper_vertex>=0)
    if len(selected):
        # Upper-cuff deformation depends only on each ankle relative to its
        # shank. Six local driver columns are checked directly; rigid lower
        # contacts and the rest of the body keep the analytic FK chain.
        h=1e-5;ref=reference.at(evaluation['times'])
        for joint in (7,8):
            for axis in range(3):
                a=6+(joint-1)*3+axis
                if a not in active:continue
                plus=delta.copy();minus=delta.copy();plus[:,a]+=h;minus[:,a]-=h
                sp=fk_state(reference,ref,plus)
                sm=fk_state(reference,ref,minus)
                vp=reference.binding.deform(reference.binding.packet()['upper'],reference.binding.deformation(sp['native'],sp['frames'],skate_relative_R=sp.get('skate_relative_R')))
                vm=reference.binding.deform(reference.binding.packet()['upper'],reference.binding.deformation(sm['native'],sm['frames'],skate_relative_R=sm.get('skate_relative_R')))
                pick=(fr[selected],foot[selected],upper_vertex[selected])
                partial=(vp[pick]-vm[pick])/(2*h)
                value=np.einsum('nj,nj->n',rotation[selected,2],partial)
                local[np.ix_(selected,np.flatnonzero(active==a))]+=value[:,None]
    local*=sign[:,None]*ratio/NORMALIZATION
    if hasattr(delta,'chain'):
        J=delta.chain(local[:,None,:],sample_indices=fr)
        if hasattr(delta.owner,'shoe_chain'):
            RFB,tFB,_=reference.binding.parameters();ankle=-np.einsum('fi,fij->fj',tFB,RFB)
            tangent=np.cross(q-ankle[foot],rotation[:,2,:])
            shoe=np.zeros((count,1,6))
            for f in (0,1):
                keep=foot==f;shoe[keep,0,f*3:f*3+3]=tangent[keep]
            if len(selected):
                from .sequence_skate_state import perturb_relative
                packet=reference.binding.packet()
                for f in (0,1):
                    for axis in range(3):
                        h=1e-5
                        a=fk_state(reference,reference.at(evaluation['times']),perturb_relative(delta,f,axis,h))
                        b=fk_state(reference,reference.at(evaluation['times']),perturb_relative(delta,f,axis,-h))
                        ap=reference.binding.deform(packet['upper'],reference.binding.deformation(a['native'],a['frames'],skate_relative_R=a['skate_relative_R']))
                        bp=reference.binding.deform(packet['upper'],reference.binding.deformation(b['native'],b['frames'],skate_relative_R=b['skate_relative_R']))
                        pick=(fr[selected],foot[selected],upper_vertex[selected])
                        shoe[selected,0,f*3+axis]+=np.einsum('nj,nj->n',rotation[selected,2],(ap[pick]-bp[pick])/(2*h))
            shoe*=sign[:,None,None]*ratio/NORMALIZATION
            J+=delta.owner.shoe_chain(shoe,delta.parameters,delta.sample_times,sample_indices=fr)
        return J
    # Select the original sparse spline support for each physical row. Only
    # values that are nonzero enter COO; large explicit zero blocks never
    # reach the downstream sparse products/LSMR.
    support=csr_matrix(B)[fr].tocoo();support.eliminate_zeros()
    values=local[support.row]*support.data[:,None];keep=values!=0
    rr=np.broadcast_to(support.row[:,None],values.shape)[keep]
    cc=(support.col[:,None]*d+np.arange(d))[keep]
    J=coo_matrix((values[keep],(rr,cc)),shape=(count,B.shape[1]*d)).tocsr();J.eliminate_zeros()
    return J
