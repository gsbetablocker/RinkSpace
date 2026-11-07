"""Absolute body controls, analytic cumulative-SO(3) sparse chain.

The seed controls define a smooth initialization only. There is no old dense
pose curve added back. Each MotionDelta carries its own immutable parameter
and clock context, including during parallel local FK replay.
"""
from dataclasses import dataclass
import copy
import numpy as np
from scipy.sparse import coo_matrix,csr_matrix,issparse,vstack
from .absolute_motion_curve import AbsoluteMotionCurve,_exp,_log
from .fullbody_fit import right_jacobian

SCHEMA='rinkspace.absolute_motion_v1'
LIFTED_SCHEMA='rinkspace.absolute_motion_v2'
CURVE_SCHEMAS=(SCHEMA,LIFTED_SCHEMA)


def curve_metadata(curve):
    lifted=curve.edge_reference_quaternion is not None
    result=dict(state_schema_version=np.array(LIFTED_SCHEMA if lifted else SCHEMA))
    if lifted:
        result.update(absolute_rotation_log_chart=np.array('anchored_su2_v1'),
            absolute_rotation_edge_reference_quaternion=curve.edge_reference_quaternion.copy())
    return result


def curve_from_packet(saved,seed=False):
    version=str(saved.get('state_schema_version',''))
    if version not in CURVE_SCHEMAS:raise ValueError('Unsupported absolute motion schema: '+version)
    key='absolute_rotation_edge_reference_quaternion';reference=None
    if version==LIFTED_SCHEMA:
        if key not in saved or str(saved.get('absolute_rotation_log_chart',''))!='anchored_su2_v1':
            raise ValueError('Lifted state requires its exact archived SU(2) edge chart')
        reference=saved[key]
    elif key in saved:raise ValueError('V1 checkpoint cannot contain an unversioned lifted chart')
    prefix='absolute_seed_' if seed else 'absolute_'
    return AbsoluteMotionCurve(saved['absolute_knots'],saved[prefix+'position_controls'],
        saved[prefix+'rotation_controls'],reference)


def with_warm_rotation_chart(seed_curve,saved,lift=False):
    warm=curve_from_packet(saved)
    reference=warm.edge_reference_quaternion
    if reference is None and not lift:return seed_curve
    if reference is None:
        from scipy.spatial.transform import Rotation
        r=warm.rotation_controls;relative=r[:-1].swapaxes(-1,-2)@r[1:]
        reference=Rotation.from_matrix(relative.reshape(-1,3,3)).as_quat().reshape(*relative.shape[:-2],4)
        reference=np.where((reference[...,3]<0)[...,None],-reference,reference)
    return AbsoluteMotionCurve(seed_curve.knots,seed_curve.position_controls,
        seed_curve.rotation_controls,reference)


def inverse_right_jacobian(v):
    v=np.asarray(v,float);a=np.linalg.norm(v,axis=-1);h=np.maximum(a,1e-7)
    from .fullbody_fit import skew
    K=skew(v);c=np.where(a<1e-4,1/12+a*a/720,1/h**2-(1+np.cos(h))/(2*h*np.sin(h)))
    return np.eye(3)+.5*K+c[...,None,None]*(K@K)


def evaluate_curve(curve,t,*,control_increments=None,with_blocks=False,basis=None):
    t=np.atleast_1d(t).astype(float);B=curve.basis_sparse(t) if basis is None else basis
    m=len(curve.position_controls);idx=np.clip(np.searchsorted(curve.knots,t,side='right')-4,0,m-4)[:,None]+np.arange(4)
    b=B[np.arange(len(t))[:,None],idx] if issparse(B) else np.take_along_axis(B,idx,axis=1)
    b=b.toarray() if issparse(b) else np.asarray(b)
    r=curve.rotation_controls[idx]
    D=r[:,:-1].swapaxes(-1,-2)@r[:,1:];phi=curve._check_log_chart()[idx[:,:-1]]
    beta=np.cumsum(b[:,::-1],axis=1)[:,::-1][:,1:]
    E=_exp(beta[:,:,None,None]*phi)
    tails=[E[:,1]@E[:,2],E[:,2],np.broadcast_to(np.eye(3),E[:,0].shape)]
    rotation=r[:,0]@E[:,0]@tails[0]
    result=dict(P=B@curve.position_controls,Q=rotation[:,0],local_R=rotation[:,1:],rotations=rotation)
    if with_blocks:
        W=right_jacobian(beta[:,:,None,None]*phi)*beta[:,:,None,None,None]
        W=W@inverse_right_jacobian(phi)
        C=[tails[j].swapaxes(-1,-2)@W[:,j] for j in range(3)]
        first=(E[:,0]@tails[0]).swapaxes(-1,-2)-C[0]@D[:,0].swapaxes(-1,-2)
        blocks=np.stack((first,C[0]-C[1]@D[:,1].swapaxes(-1,-2),C[1]-C[2]@D[:,2].swapaxes(-1,-2),C[2]),axis=1)
        if control_increments is not None:blocks=blocks@right_jacobian(control_increments[idx])
        result.update(control_indices=idx,position_weights=b,rotation_right_blocks=blocks)
    return result


class MotionDelta(np.ndarray):
    def __new__(cls,value,owner,parameters,times):
        obj=np.asarray(value).view(cls);obj.owner=owner;obj.parameters=np.asarray(parameters).copy();obj.sample_times=np.asarray(times).copy();return obj
    def __array_finalize__(self,parent):
        if parent is not None:
            self.owner=getattr(parent,'owner',None);self.parameters=getattr(parent,'parameters',None);self.sample_times=getattr(parent,'sample_times',None)
            self.skate_relative_R=getattr(parent,'skate_relative_R',None)
    def chain(self,local,sample_indices=None):
        return self.owner.chain(local,self.parameters,self.sample_times,sample_indices=sample_indices)
    def rotation_chain(self,blocks,sample_indices=None):
        return self.owner.rotation_chain(blocks,self.parameters,self.sample_times,sample_indices=sample_indices)


class AbsoluteMotionState:
    state_kind='absolute_motion_v1';use_pose_cost_guard=False;is_absolute=True
    def __init__(self,reference,seconds,mode='full'):
        self.reference=reference;self.curve=reference.absolute_curve;self.knots=self.curve.knots.copy()
        self.m=len(self.curve.position_controls);self.active=np.arange(69);self.d=69;self.size=self.m*69
        self.seconds=tuple(seconds);self.mode=mode;self.taper=0.;self.bounds_time=self.knots[[3,-4]].copy()
        self._last_parameters=None;self._evaluations={};self._basis={}
        self._fixed_values={};self._curve_parameters=None;self._current_curve=None
    def basis(self,t,derivative=0):
        t=np.atleast_1d(t).astype(float);key=(t.tobytes(),derivative)
        if key not in self._basis:
            if len(self._basis)>=24:self._basis.clear()
            self._basis[key]=self.curve.basis_sparse(t,derivative)
        return self._basis[key]
    def current_curve(self,x):
        x=np.asarray(x)
        if self._curve_parameters is None or not np.array_equal(x,self._curve_parameters):
            a=x.reshape(self.m,69)
            self._current_curve=self.curve.retracted(a[:,:3],a[:,3:].reshape(self.m,22,3))
            self._curve_parameters=x.copy()
        return self._current_curve
    def _evaluation(self,x,t,*,with_blocks):
        x=np.asarray(x);t=np.atleast_1d(t).astype(float)
        if self._last_parameters is None or not np.array_equal(x,self._last_parameters):
            self._last_parameters=x.copy();self._evaluations={}
        key=t.tobytes();cached=self._evaluations.get(key)
        if cached is None or (with_blocks and cached[2] is None):
            B=self.basis(t);cur=self.current_curve(x)
            s=evaluate_curve(cur,t,control_increments=x.reshape(self.m,69)[:,3:].reshape(self.m,22,3),with_blocks=with_blocks,basis=B)
            if key not in self._fixed_values:
                if len(self._fixed_values)>=4:self._fixed_values.clear()
                self._fixed_values[key]=evaluate_curve(self.curve,t,basis=B)
            base=self._fixed_values[key];rot=_log(base['rotations'].swapaxes(-1,-2)@s['rotations'])
            d=np.c_[s['P']-base['P'],rot.reshape(len(t),66)];mapping=None
            if with_blocks:
                mapping=np.empty((len(t),4,23,3,3));mapping[:,:,0]=s['position_weights'][:,:,None,None]*np.eye(3)
                mapping[:,:,1:]=inverse_right_jacobian(rot)[:,None]@s['rotation_right_blocks']
            cached=(d,s,mapping)
            # Large physics clocks are transient. Retaining every shutter,
            # midpoint and factor clock would defeat bounded derivative batches.
            if len(t)<=10000:
                if len(self._evaluations)>=4:self._evaluations.clear()
                self._evaluations[key]=cached
            return cached
        return cached
    def linearization(self,x,t):return self._evaluation(x,t,with_blocks=True)
    def increments(self,x,t):
        d,_,_=self._evaluation(x,t,with_blocks=False);return MotionDelta(d,self,x,np.atleast_1d(t))
    def state_at(self,x,t,reference=None):
        d=self.increments(x,t);return self.reference.world(self.reference.at(t) if reference is None else reference,d)
    def chain(self,local,x,t,*,sample_indices=None):
        local=np.asarray(local,float);n,k,d=local.shape
        if d!=69:raise ValueError('Absolute local tangent requires all69 columns')
        if n>512:
            pick=np.arange(len(t)) if sample_indices is None else np.asarray(sample_indices,int)
            return vstack([self.chain(local[a:a+512],x,t,sample_indices=pick[a:a+512]) for a in range(0,n,512)],format='csr')
        _,s,mapping=self.linearization(x,t);pick=np.arange(len(t)) if sample_indices is None else np.asarray(sample_indices,int)
        if len(pick)!=n:raise ValueError('Local rows and sampled clock differ')
        # General cavity/physics factors are sparse in the local joint axes.
        # Keep the lower-memory per-support assembly here; motion and exposure
        # factors use their dedicated direct-CSR paths below.
        rr=[];cc=[];vv=[];groups=local.reshape(n,k,23,3)
        rows=np.broadcast_to(np.arange(n*k).reshape(n,k,1),(n,k,69))
        for j in range(4):
            val=np.einsum('nkgc,ngcd->nkgd',groups,mapping[pick,j]).reshape(n,k,69)
            col=s['control_indices'][pick,j,None,None]*69+np.arange(69)[None,None,:]
            mask=val!=0.;vv.append(val[mask]);rr.append(rows[mask]);cc.append(np.broadcast_to(col,val.shape)[mask])
        out=coo_matrix((np.concatenate(vv),(np.concatenate(rr),np.concatenate(cc))),shape=(n*k,self.size)).tocsr();out.eliminate_zeros();return out
    def rotation_chain(self,blocks,x,t,*,sample_indices=None):
        """Chain joint-diagonal 3x3 factors without expanding zero 69-columns."""
        blocks=np.asarray(blocks,float);n,j=blocks.shape[:2]
        if blocks.shape!=(n,22,3,3):raise ValueError('Expected22 joint-diagonal rotation blocks')
        _,s,mapping=self.linearization(x,t)
        pick=np.arange(len(t)) if sample_indices is None else np.asarray(sample_indices,int)
        if len(pick)!=n:raise ValueError('Rotation rows and sampled clock differ')
        values=np.einsum('njac,nsjcd->njasd',blocks,mapping[pick,:,1:]).reshape(n*j*3,12)
        row,column=np.nonzero(values)
        controls=(s['control_indices'][pick,None,:,None]*69
            +3+np.arange(j)[None,:,None,None]*3+np.arange(3)[None,None,None,:]).reshape(n,j,12)
        indices=controls[row//(j*3),(row//3)%j,column].astype(np.int32)
        indptr=np.r_[0,np.cumsum(np.bincount(row,minlength=n*j*3))].astype(np.int32)
        return csr_matrix((values[row,column],indices,indptr),shape=(n*j*3,self.size))

    def packet(self,x):
        final=self.current_curve(x)
        return dict(**curve_metadata(final),state_kind=np.array(self.state_kind),
            absolute_knots=self.knots.copy(),absolute_position_controls=final.position_controls.copy(),absolute_rotation_controls=final.rotation_controls.copy(),
            absolute_seed_position_controls=self.curve.position_controls.copy(),absolute_seed_rotation_controls=self.curve.rotation_controls.copy(),
            absolute_parameter_coordinates=np.array('P additive controls; per-control right Exp coordinates about archived seed; final absolute controls are authoritative'),
            no_inherited_high_frequency_curve=np.array(True))


def chain_exposure(local,delta,sample_indices=None):
    local=np.asarray(local);n,k,nodes,d=local.shape
    if nodes!=5 or d!=69:raise ValueError('Native projection requires five69-column shutter nodes')
    indices=np.arange(n*5).reshape(n,5) if sample_indices is None else np.asarray(sample_indices)
    if n>2048:
        return vstack([chain_exposure(local[a:a+2048],delta,sample_indices=indices[a:a+2048]) for a in range(0,n,2048)],format='csr')
    _,state,mapping=delta.owner.linearization(delta.parameters,delta.sample_times)
    controls=state['control_indices'][indices];first=controls.min(axis=(1,2))
    width=int((controls.max(axis=(1,2))-first).max())+1
    if width>nodes*4:
        # Arbitrary, widely separated diagnostic clocks need no dense gap buffer.
        result=csr_matrix((n*k,delta.owner.size))
        for j in range(nodes):result=result+delta.chain(local[:,:,j],sample_indices=indices[:,j])
        result.eliminate_zeros();return result
    # The five exposure nodes mostly share controls. Sum their small blocks
    # before CSR construction, instead of building/adding five large matrices.
    values=np.zeros((n,k,width,69));row=np.arange(n)
    for node in range(nodes):
        groups=local[:,:,node].reshape(n,k,23,3)
        for support in range(4):
            contribution=np.einsum('nkgc,ngcd->nkgd',groups,mapping[indices[:,node],support]).reshape(n,k,69)
            slot=controls[:,node,support]-first
            values[row,:,slot,:]+=contribution
    values=values.reshape(n*k,width*69);rr,cc=np.nonzero(values)
    col=(first[rr//k]*69+cc).astype(np.int32)
    ptr=np.r_[0,np.cumsum(np.bincount(rr,minlength=n*k))].astype(np.int32)
    return csr_matrix((values[rr,cc],col,ptr),shape=(n*k,delta.owner.size))


def absolute_reference(shared,curve,*,root=None):
    """Clone fixed geometry/hands while replacing all P/Q/local motion curves."""
    from .action_integration import ClipReference
    class AbsoluteReference(ClipReference):
        def at(self,t):
            t=np.atleast_1d(t).astype(float);s=evaluate_curve(self.absolute_curve,t)
            result={k:s[k] for k in ('P','Q','local_R')}
            shoe=getattr(self,'relative_skate_curve',None)
            if shoe is not None:result['skate_relative_R']=evaluate_curve(shoe,t)['rotations']
            if self.hand_curves is not None:
                from .hand_state import evaluate_hands
                result['hand_local_R']=evaluate_hands(self.hand_curves,t)
            return result
        def root_derivative(self,t,order):return self.absolute_curve.position_derivative(t,order)
    ref=object.__new__(AbsoluteReference);ref.__dict__.update(shared.__dict__)
    ref.absolute_curve=curve;ref.new_refinements=[];ref.refinements=[]
    if root is not None:ref.root=root
    return ref
