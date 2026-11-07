"""One body curve and two low-frequency, ankle-pivoted skate SO(3) curves.

All controls belong to one state. Missing pixels do not remove the estimated
trajectory; evidence status is stored separately by the sequence ledger.
"""
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, vstack
from .absolute_motion_curve import AbsoluteMotionCurve, clamped_cubic_knots, _exp
from .absolute_motion_state import AbsoluteMotionState, MotionDelta, evaluate_curve, _sampled_clock

SHOE_SCHEMA='rinkspace.relative_skate_so3_v1'


def relative_rotation(reference, ref, delta=None):
    value=getattr(delta,'skate_relative_R',None) if delta is not None else None
    if value is None:value=ref.get('skate_relative_R')
    return value


def shoe_curve_from_packet(packet,seed=False):
    if 'relative_skate_schema' not in packet:return None
    if str(packet['relative_skate_schema'])!=SHOE_SCHEMA:raise ValueError('Unsupported relative skate schema')
    key='relative_skate_seed_rotation_controls' if seed else 'relative_skate_rotation_controls'
    r=packet[key];k=packet['relative_skate_knots']
    return AbsoluteMotionCurve(k,np.zeros((len(r),3)),r)


class SequenceState(AbsoluteMotionState):
    state_kind='absolute_motion_v1'
    def __init__(self,reference,seconds,mode='full'):
        super().__init__(reference,seconds,mode)
        self.body_size=self.size
        curve=getattr(reference,'relative_skate_curve',None)
        if curve is None:
            lo,hi=self.curve.knots[[3,-4]]
            breaks=np.linspace(lo,hi,max(1,int(np.ceil((hi-lo)/.5)))+1)
            k=clamped_cubic_knots(breaks);m=len(k)-4
            curve=AbsoluteMotionCurve(k,np.zeros((m,3)),np.broadcast_to(np.eye(3),(m,2,3,3)).copy())
        self.shoe_curve=curve;self.shoe_m=len(curve.position_controls)
        self.shoe_size=self.shoe_m*6;self.size=self.body_size+self.shoe_size
        self._shoe_x=None;self._shoe_current=None;self._shoe_evaluations={}

    def current_curve(self,x):return super().current_curve(np.asarray(x)[:self.body_size])
    def _evaluation(self,x,t,*,with_blocks):
        return super()._evaluation(np.asarray(x)[:self.body_size],t,with_blocks=with_blocks)
    def shoe_current_curve(self,x):
        a=np.asarray(x)[self.body_size:].reshape(self.shoe_m,2,3)
        if self._shoe_x is None or not np.array_equal(a,self._shoe_x):
            self._shoe_x=a.copy();self._shoe_evaluations={}
            self._shoe_current=self.shoe_curve.retracted(np.zeros((self.shoe_m,3)),a)
        return self._shoe_current
    def shoe_evaluation(self,x,t,with_blocks=False):
        curve=self.shoe_current_curve(x);t=np.atleast_1d(t).astype(float);key=(t.tobytes(),with_blocks)
        cached=self._shoe_evaluations.get(key)
        if cached is None:
            a=np.asarray(x)[self.body_size:].reshape(self.shoe_m,2,3)
            cached=evaluate_curve(curve,t,control_increments=a,with_blocks=with_blocks)
            # Mirror the body-state cache: physics batches must not accumulate
            # until an unrelated shoe control happens to change.
            if len(t)<=10000:
                if len(self._shoe_evaluations)>=4:self._shoe_evaluations.clear()
                self._shoe_evaluations[key]=cached
        return cached
    def increments(self,x,t):
        d,_,_=self._evaluation(x,t,with_blocks=False)
        out=MotionDelta(d,self,x,np.atleast_1d(t))
        out.skate_relative_R=self.shoe_evaluation(x,t)['rotations']
        return out
    def shoe_chain(self,local,x,t,*,sample_indices=None):
        """Local rows d residual / d right skate tangent (two feet x three)."""
        local=np.asarray(local,float);n,k,d=local.shape
        if d!=6:raise ValueError('Expected two right skate tangents')
        pick=np.arange(len(t)) if sample_indices is None else np.asarray(sample_indices,int)
        if pick.shape!=(n,):raise ValueError('Shoe clock and rows differ')
        if not n:return csr_matrix((0,self.size))
        if n>512:
            return vstack([self.shoe_chain(local[a:a+512],x,t,sample_indices=pick[a:a+512]) for a in range(0,n,512)],format='csr')
        clock,pick=_sampled_clock(t,pick)
        e=self.shoe_evaluation(x,clock,with_blocks=True)
        blocks=e['rotation_right_blocks'][pick]
        values=np.einsum('nkfa,nsfab->nksfb',local.reshape(n,k,2,3),blocks).reshape(n,k,24)
        cols=(e['control_indices'][pick,:,None]*6+np.arange(6)[None,None,:]+self.body_size).reshape(n,24)
        rr=np.broadcast_to(np.arange(n*k).reshape(n,k,1),values.shape)
        cc=np.broadcast_to(cols[:,None],values.shape);mask=values!=0
        return coo_matrix((values[mask],(rr[mask],cc[mask])),shape=(n*k,self.size)).tocsr()
    def shoe_exposure_chain(self,local,x,t,sample_indices=None):
        local=np.asarray(local);n,k,nodes,d=local.shape
        if (nodes,d)!=(5,6):raise ValueError('Five native shoe nodes required')
        out=csr_matrix((n*k,self.size));indices=np.arange(n*5).reshape(n,5) if sample_indices is None else np.asarray(sample_indices)
        for j in range(5):out+=self.shoe_chain(local[:,:,j],x,t,sample_indices=indices[:,j])
        return out
    def packet(self,x):
        result=super().packet(x);final=self.shoe_current_curve(x)
        result.update(relative_skate_schema=np.array(SHOE_SCHEMA),
            relative_skate_knots=self.shoe_curve.knots.copy(),
            relative_skate_seed_rotation_controls=self.shoe_curve.rotation_controls.copy(),
            relative_skate_rotation_controls=final.rotation_controls.copy(),
            relative_skate_parameter_offset=np.array(17+self.body_size),
            relative_skate_parameter_count=np.array(self.shoe_size),
            body_parameter_count=np.array(self.body_size),
            relative_skate_pivot=np.array('registered ankle in fixed boot coordinates; no translation DOF'),
            relative_skate_coordinates=np.array('right SO3 controls about archived seed'),
            sequence_state_scope_seconds=np.asarray(self.seconds))
        return result


def perturb_relative(delta,foot,axis,step):
    out=delta.copy();r=delta.skate_relative_R.copy();v=np.zeros(3);v[axis]=step
    r[:,foot]=r[:,foot]@_exp(v);out.skate_relative_R=r
    return out


def add_shoe_columns(J,delta,local,sample_indices=None):
    if not hasattr(delta.owner,'shoe_chain'):return J
    return J+delta.owner.shoe_chain(local,delta.parameters,delta.sample_times,sample_indices=sample_indices)
