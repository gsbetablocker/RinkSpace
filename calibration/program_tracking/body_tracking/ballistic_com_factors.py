"""Gravity-constrained flight of a declared whole-body mass proxy.

The mass centres follow the same current FK and attached shoes. Pelvis is not
substituted for CoM. Segment convex volumes supply a uniform-density mass
prior, with explicit equipment-mass and distribution sensitivity diagnostics.
Absolute athlete mass is neither needed for gravity nor inferred here.
"""
import numpy as np
from scipy.sparse import csr_matrix,hstack,kron,eye
from .segment_mass_proxy import prepare_segment_mass_proxy,tetraVolumeCentroid
from .skating_action_factors import SourceTimeline,MassWeightedCOM,airborne_ballistic_residual
from .skate_physics import fk_state
from .spin_grasp_factors import spline_chain
from .skate_contact import trapezoid_time_weights


def ballistic_projection(times,gravity,weights,sigma):
    t=np.asarray(times,float);t=t-t[0]
    gravity=np.asarray(gravity,float);w=np.asarray(weights,float)
    if len(t)<3 or gravity.shape!=(3,) or w.shape!=t.shape or np.any(w<=0) or sigma<=0:
        raise ValueError('Ballistic projection needs three positive-weight source samples')
    A=w[:,None]*np.c_[np.ones(len(t)),t]
    Q=np.linalg.qr(A,mode='reduced')[0]
    H=np.eye(len(t))-Q@Q.T
    D=H*w[None,:]/sigma
    gravity_offset=-.5*t[:,None]**2*gravity
    return D,gravity_offset


class BallisticCOMFactors:
    name='flight_com_ballistic'
    def __init__(self,problem,config):
        self.problem=problem;self.body=problem.body;self.reference=problem.reference;self.binding=problem.binding
        self.config=dict(config);self.gravity=np.asarray(config['gravity_world_mps2'],float)
        lo,hi=map(float,config['source_interval_seconds']);sec=self.reference.p['pts']/12800.
        self.frames=np.flatnonzero((sec>=lo-1e-10)&(sec<=hi+1e-10))
        if len(self.frames)<3:raise ValueError('Confirmed flight interior needs at least three source samples')
        self.times=self.reference.times[self.frames];self.base=self.reference.at(self.times);self.B=self.body.state.basis(self.times)
        self.clock=SourceTimeline(self.reference.p['pts'][self.frames],(1,12800),config['source_id'])
        self.model=prepare_segment_mass_proxy(self.binding.surface)
        self.joints=np.asarray(self.model['native_joint_ids'],int)
        self.offsets=(np.asarray(self.model['centres_rest'])-self.reference.body.J[self.joints])*self.reference.body.scale
        self.body_mass=np.asarray(self.model['mass_fractions'],float)
        per_shoe=float(config['relative_mass_per_skate']);sigma=float(config['position_scale_m'])
        confidence=float(config['confidence']);objective_weight=float(config['weight'])
        if per_shoe<=0 or not 0<confidence<=1 or objective_weight<=0:raise ValueError('Invalid declared proxy-mass/flight configuration')
        self.masses=np.r_[self.body_mass,per_shoe,per_shoe]
        packet=self.binding.packet()
        _,self.shoe_centre=tetraVolumeCentroid(np.r_[packet['template']['vertices'],packet['upper']])
        self.names=tuple(self.model['names'])+('left_skate_proxy','right_skate_proxy')
        self.model_id=self.model['id']+';rigid-skate-relative-mass-assumption'
        dt=trapezoid_time_weights(self.times)
        self.weights=np.sqrt(objective_weight*confidence*dt)
        self.D,self.gravity_offset=ballistic_projection(self.times,self.gravity,self.weights,sigma)
        self.residual_form=config.get('residual_form','position')
        if self.residual_form=='acceleration':
            d=np.diff(self.times);D2=np.zeros((len(d)-1,len(self.times)));k=np.arange(len(d)-1)
            D2[k,k]=2/(d[:-1]*(d[:-1]+d[1:]));D2[k,k+1]=-2/(d[:-1]*d[1:]);D2[k,k+2]=2/(d[1:]*(d[:-1]+d[1:]))
            measure=(d[:-1]+d[1:])/2
            self.acceleration_weights=np.sqrt(objective_weight*confidence*measure)/float(config['acceleration_scale_m_s2'])
            self.D2=D2
            self.operator=kron(csr_matrix(self.acceleration_weights[:,None]*D2),eye(3),format='csr')
        elif self.residual_form=='position':self.operator=kron(csr_matrix(self.D),eye(3),format='csr')
        else:raise ValueError('Unknown ballistic residual form')
        self.last=None

    def geometry_at(self,delta):
        s=fk_state(self.reference,self.base,delta)
        frames=s['Q'][:,None]@s['frames'][:,self.joints]
        centres=s['native_world'][:,self.joints]+np.einsum('nsij,sj->nsi',frames,self.offsets)
        shoes=s['O']+np.einsum('nfij,j->nfi',s['R'],self.shoe_centre)
        centres=np.concatenate((centres,shoes),axis=1)*self.binding.ratio
        com=MassWeightedCOM(self.clock,centres,self.masses,self.names,self.model_id)
        return com,centres

    def evaluate(self,x,jacobian=False):
        delta=self.body.state.increments(x,self.times);com,centres=self.geometry_at(delta)
        r=((self.acceleration_weights[:,None]*(self.D2@com.position_world_m-self.gravity)).ravel() if self.residual_form=='acceleration' else (self.D@(com.position_world_m+self.gravity_offset)).ravel())
        J=None
        if jacobian:
            derivatives=[];step=1e-6
            for active in self.body.state.active:
                if active<3:
                    d=np.zeros_like(com.position_world_m);d[:,active]=self.binding.ratio
                else:
                    a=delta.copy();b=delta.copy();a[:,active]+=step;b[:,active]-=step
                    d=(self.geometry_at(a)[0].position_world_m-self.geometry_at(b)[0].position_world_m)/(2*step)
                derivatives.append(d)
            pointJ=spline_chain(np.stack(derivatives,-1),self.B)
            J=hstack((csr_matrix((len(r),17)),self.operator@pointJ),format='csr')
        physical=airborne_ballistic_residual(self.clock,com,phases='flight',confidence=self.config['confidence'],
            gravity_world_mps2=self.gravity,sigma_mps2=self.config['acceleration_diagnostic_scale_m_s2'])
        sensitivity=[]
        legs=np.isin(self.joints,[1,2,4,5,7,8,10,11])
        for lower_scale,skate_mass in [(.8,0.),(1.2,.03)]:
            body=self.body_mass*np.where(legs,lower_scale,1.);body/=body.sum();mass=np.r_[body,skate_mass,skate_mass]
            alternative=np.einsum('nsc,s->nc',centres,mass/mass.sum())
            sensitivity.append(dict(lower_body_mass_multiplier=lower_scale,relative_mass_per_skate=skate_mass,
                maximum_com_change_m=float(np.linalg.norm(alternative-com.position_world_m,axis=1).max())))
        self.last=dict(residual=r,jac=J,parts={self.name:float(r@r/2)},block_slices={self.name:slice(0,len(r))},
            metrics=dict(residual_form=self.residual_form,source_pts=self.clock.pts.tolist(),source_interval_seconds=self.config['source_interval_seconds'],
                com_world_m=com.position_world_m.tolist(),gravity_world_mps2=self.gravity.tolist(),
                acceleration_diagnostics={k:(v.tolist() if hasattr(v,'tolist') else v) for k,v in physical.diagnostics.items()},
                mass_model_id=self.model_id,segment_names=list(self.names),normalized_model_masses=self.masses.tolist(),
                mass_is_measured=False,skate_mass_is_assumed=True,distribution_sensitivity=sensitivity,
                pelvis_used_as_com=False,source_boundaries_excluded=True,finite_soft_factor=True,
                model_limitations='uniform-density segment convex-volume/rigid-segment and unmeasured skate-mass proxy; no exact athlete anthropometry claim'))
        return self.last
