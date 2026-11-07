"""Same-state COM, intrinsic+orbital momentum and finite confirmed-flight factors."""
import json
import numpy as np
from scipy.sparse import csr_matrix,hstack,kron,eye,vstack
from .segment_mass_proxy import prepare_segment_mass_proxy,prepare_body_skate_mass_proxy
from .skate_physics import fk_state
from .centroidal_physics_diagnostics import discrete_segment_rates,centroidal_momentum,flight_position_operator,angular_impulse_operator,ground_force_torque_diagnostics

class ContinuousMassGeometry:
    def __init__(self,reference,relative_skate_mass=.02):
        self.ref=reference;self.binding=reference.binding;body=prepare_segment_mass_proxy(self.binding.surface);packet=self.binding.packet()
        self.proxy=prepare_body_skate_mass_proxy(body,reference.body.J[:22],np.r_[packet['template']['vertices'],packet['upper']],
            metres_per_model_unit=reference.body.scale*self.binding.ratio,metres_per_skate_unit=self.binding.ratio,relative_mass_per_skate=relative_skate_mass,mass_variant_id='continuous-uniform-volume-body-rigid-skates-v1')
        self.count=int(self.proxy['body_segment_count']);self.ids=self.proxy['segment_native_joint_ids'][:self.count]
        self.offsets=self.proxy['segment_com_rest_local_m'];self.mass=self.proxy['segment_mass_fractions'];self.inertia=self.proxy['segment_unit_mass_inertia_local_m2']
    def evaluate(self,base,delta):
        s=fk_state(self.ref,base,delta);R=s['Q'][:,None]@s['frames'][:,self.ids]
        C=s['native_world'][:,self.ids]*self.binding.ratio+np.einsum('nsij,sj->nsi',R,self.offsets[:self.count])
        shoe=s['O']*self.binding.ratio+np.einsum('nfij,fj->nfi',s['R'],self.offsets[self.count:])
        return np.concatenate((C,shoe),axis=1),np.concatenate((R,s['R']),axis=1)
    def com(self,C):return np.einsum('s,nsc->nc',self.mass,C)
    def H(self,t,C,R):
        rates=discrete_segment_rates(t,C,R,phase_ids=np.zeros(len(t),int),confirmed_phase=np.ones(len(t),bool))
        idx=rates['source_indices'];out=centroidal_momentum(centres_world_m=C[idx],velocity_world_mps=rates['segment_velocity_world_mps'],rotation_world_from_segment=R[idx],omega_world_radps=rates['segment_omega_world_radps'],segment_mass_fractions=self.mass,unit_mass_inertia_segment_m2=self.inertia)
        return out['specific_angular_momentum_world_m2ps'],rates,out

class ConfirmedFlightFactors:
    def __init__(self,problem,mass_geometry,*,episode=None,event_average=False):
        self.problem=problem;self.body=problem.body;self.ref=problem.reference;self.mass=mass_geometry
        from .body_clock import source_seconds,time_base
        source=source_seconds(self.ref.p);ledger=getattr(self.ref,'source_document',{}).get('source_episode_review')
        if ledger:
            intervals=[e for e in ledger['episodes']if e['kind']=='flight_visible_candidate']if episode is None else[episode];mask=np.zeros(len(source),bool)
            if len(intervals)!=1:raise ValueError('Current flight factor requires one explicitly reviewed flight core')
            episode=intervals[0];mask=(self.ref.p['pts']>=episode['start_pts'])&(self.ref.p['pts']<=episode['end_pts_inclusive'])
            interval=source_seconds(self.ref.p,[episode['start_pts'],episode['end_pts_inclusive']]).tolist()
        else:mask=(source>=81.36-1e-9)&(source<=81.46+1e-9);interval=[81.36,81.46]
        self.frames=np.flatnonzero(mask);self.times=self.ref.times[self.frames];self.base=self.ref.at(self.times)
        # One source-supported interior: exact onset and landing are excluded.
        self.gravity=np.array([0.,0.,-9.81]);self.operator=flight_position_operator(self.times,self.gravity)
        self.position_scale=.03;self.impulse_scale=.2
        self.momentum_scale=.2
        self.event_average=event_average
        if event_average:
            from .event_physics import ballistic_position_operator, flight_momentum_residual
            position, offset = ballistic_position_operator(self.times, self.gravity, self.position_scale)
            h_times = self.times[1:-1]
            momentum = flight_momentum_residual(np.zeros((len(h_times), 3)), h_times,
                                               sigma_specific_H=self.momentum_scale, phase='flight')
            self.position_operator, self.position_offset = position, offset
            self.momentum_operator = momentum['operator']
        self.momentum_name = 'flight_specific_momentum' if event_average else 'flight_specific_angular_impulse'
        self.config=dict(confirmed_source_interval_seconds=interval,source_boundary_unknown=True,position_scale_m=self.position_scale,specific_impulse_scale_m2_s=self.impulse_scale,finite_weight=1.,physics_model_is_proxy=True,source_episode_ledger_used=bool(ledger),event_average=event_average,momentum_scale_specific_m2ps=self.momentum_scale)
    def values(self,C,R):
        H,_,_=self.mass.H(self.times,C,R);com=self.mass.com(C)
        if self.event_average:
            return com, H, (self.position_operator @ com + self.position_offset).ravel(), (self.momentum_operator @ H).ravel()
        op=self.operator['operator'];r=(op@(com-self.operator['gravity_position_m']))/self.position_scale
        dh=np.diff(H,axis=0)/self.impulse_scale
        return com,H,r.ravel(),dh.ravel()
    def mass_sensitivity(self, centres, rotations, com, momentum):
        # Reweight the same segment kinematics without aligning world states.
        original = self.mass.mass
        records = []
        for scale in (0.5, 1.5):
            adjusted = original.copy()
            adjusted[-2:] *= scale
            adjusted /= adjusted.sum()
            rates = discrete_segment_rates(self.times, centres, rotations,
                phase_ids=np.zeros(len(centres), int), confirmed_phase=np.ones(len(centres), bool))
            ix = rates['source_indices']
            alternate = centroidal_momentum(centres_world_m=centres[ix],
                velocity_world_mps=rates['segment_velocity_world_mps'],rotation_world_from_segment=rotations[ix],
                omega_world_radps=rates['segment_omega_world_radps'],segment_mass_fractions=adjusted,
                unit_mass_inertia_segment_m2=self.mass.inertia)['specific_angular_momentum_world_m2ps']
            records.append(dict(skate_proxy_mass_multiplier=scale,
                max_com_shift_m=float(np.max(np.linalg.norm(np.einsum('s,nsc->nc',adjusted,centres) - com, axis=-1))),
                max_specific_H_shift_m2ps=float(np.max(np.linalg.norm(alternate-momentum, axis=-1)))))
        return records

    def evaluate(self,x,jacobian=False):
        d=self.body.state.increments(x,self.times);C,R=self.mass.evaluate(self.base,d);com,H,rp,rh=self.values(C,R);J=None
        if jacobian:
            h=1e-6;dc=[];dH=[[],[],[]];N=len(d)
            for a in range(69):
                plus=d.copy();minus=d.copy();plus[:,a]+=h;minus[:,a]-=h
                Cp,Rp=self.mass.evaluate(self.base,plus);Cm,Rm=self.mass.evaluate(self.base,minus)
                dc.append((self.mass.com(Cp)-self.mass.com(Cm))/(2*h))
                # Three-color local time stencil gives each H output's left/current/right dependence.
                for color in range(3):
                    mask=np.arange(N)%3==color;CA=C.copy();CB=C.copy();RA=R.copy();RB=R.copy()
                    CA[mask]=Cp[mask];CB[mask]=Cm[mask];RA[mask]=Rp[mask];RB[mask]=Rm[mask]
                    dH[color].append((self.mass.H(self.times,CA,RA)[0]-self.mass.H(self.times,CB,RB)[0])/(2*h))
            Jcom=d.chain(np.stack(dc,-1));shoe_colorvals=None
            if callable(getattr(getattr(d,'owner',None),'shoe_chain',None)):
                from .sequence_skate_state import perturb_relative
                shoe_dc=[];shoe_dH=[[],[],[]]
                for foot in range(2):
                    for axis in range(3):
                        plus=perturb_relative(d,foot,axis,h);minus=perturb_relative(d,foot,axis,-h)
                        Cp,Rp=self.mass.evaluate(self.base,plus);Cm,Rm=self.mass.evaluate(self.base,minus)
                        shoe_dc.append((self.mass.com(Cp)-self.mass.com(Cm))/(2*h))
                        for color in range(3):
                            mask=np.arange(N)%3==color;CA=C.copy();CB=C.copy();RA=R.copy();RB=R.copy()
                            CA[mask]=Cp[mask];CB[mask]=Cm[mask];RA[mask]=Rp[mask];RB[mask]=Rm[mask]
                            shoe_dH[color].append((self.mass.H(self.times,CA,RA)[0]-self.mass.H(self.times,CB,RB)[0])/(2*h))
                Jcom=Jcom+d.owner.shoe_chain(np.stack(shoe_dc,-1),d.parameters,d.sample_times)
                shoe_colorvals=[np.stack(values,-1) for values in shoe_dH]
            A = self.position_operator if self.event_average else self.operator['operator']/self.position_scale
            Jp=kron(csr_matrix(A),eye(3),format='csr')@Jcom
            Jh=csr_matrix((H.size,self.body.size));i=np.arange(1,N-1)
            colorvals=[np.stack(values,-1) for values in dH]
            for offset in [-1,0,1]:
                samples=i+offset;local=np.stack([colorvals[int(c)][k] for k,c in enumerate(samples%3)])
                Jh=Jh+d.chain(local,sample_indices=samples)
                if shoe_colorvals is not None:
                    shoe_local=np.stack([shoe_colorvals[int(c)][k] for k,c in enumerate(samples%3)])
                    Jh=Jh+d.owner.shoe_chain(shoe_local,d.parameters,d.sample_times,sample_indices=samples)
            D=np.zeros((len(H)-1,len(H)));rows=np.arange(len(H)-1);D[rows,rows]=-1;D[rows,rows+1]=1
            L = self.momentum_operator if self.event_average else D/self.impulse_scale
            Jh=kron(csr_matrix(L),eye(3),format='csr')@Jh
            J=hstack((csr_matrix((len(rp)+len(rh),17)),vstack((Jp,Jh),format='csr')),format='csr')
        from .weighted_blocks import block
        named=[block('flight_integrated_com',rp,None if not jacobian else J[:len(rp)]),block(self.momentum_name,rh,None if not jacobian else J[len(rp):])]
        r=np.r_[rp,rh];return dict(residual=r,jac=J,parts={'flight_integrated_com':float(rp@rp/2),self.momentum_name:float(rh@rh/2)},weighted_blocks=named,metrics=dict(self.config,com_world_m=com.tolist(),specific_H_world_m2ps=H.tolist(),full_orbital_and_intrinsic=True,local_FK_derivatives_only=True,
            source_pts=self.ref.p['pts'][self.frames].tolist(),
            momentum_pts=self.ref.p['pts'][self.frames[1:-1]].tolist(),
            com_residual_marginal_norm=np.linalg.norm(rp.reshape(-1,3),axis=1).tolist(),
            momentum_residual_norm=np.linalg.norm(rh.reshape(-1,3),axis=1).tolist(),
            mass_proxy_sensitivity=self.mass_sensitivity(C,R,com,H) if self.event_average else []))
