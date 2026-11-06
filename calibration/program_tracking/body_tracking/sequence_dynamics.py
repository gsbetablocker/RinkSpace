"""Finite sequence priors on the actual shoes, with duration-aware inferred modes."""
import numpy as np
from scipy.sparse import csr_matrix,hstack,vstack,kron,eye,block_diag
from .absolute_motion_curve import _log
from .absolute_motion_state import inverse_right_jacobian
from .fullbody_fit import right_jacobian
from .skate_physics import fk_state,batch_features
from .skate_contact import trapezoid_time_weights
from .runner_interior import runner_longitudinal_coordinate,runner_interior_band,longitudinal_barrier
from .runner_support_factors import runner_tangent_limits,runner_mode_terms
from .continuous_motion_factors import robust_vector


def _full(J):return hstack((csr_matrix((J.shape[0],17)),J),format='csr')


def rotation_chain(delta,state,reference):
    """Right tangent of both actual world skate rotations; body plus shoe."""
    n=len(delta);local=np.zeros((n,6,69));R=state['R'];Q=state['Q'];frames=state['frames']
    root=Q@right_jacobian(np.asarray(delta)[:,3:6])
    for foot,ancestors in [(0,(1,4,7)),(1,(2,5,8))]:
        inv=R[:,foot].swapaxes(-1,-2);local[:,foot*3:foot*3+3,3:6]=inv@root
        for j in ancestors:
            a=6+(j-1)*3;axis=Q@frames[:,j]@right_jacobian(np.asarray(delta)[:,a:a+3])
            local[:,foot*3:foot*3+3,a:a+3]=inv@axis
    J=delta.chain(local)
    identity=np.broadcast_to(np.eye(6),(n,6,6))
    return J+delta.owner.shoe_chain(identity,delta.parameters,delta.sample_times)


class SequenceShoeDynamics:
    def __init__(self,problem):
        self.problem=problem;self.body=problem.body;self.ref=problem.reference
        self.times=self.ref.times;self.base=self.ref.at(self.times);self.weights=trapezoid_time_weights(self.times)
        # Finite engineering priors, not physical accuracy gates.
        self.config=dict(relative_rotation_sigma_rad=.25,relative_angular_velocity_scale_rad_s=1.5,
            actual_shoe_angular_acceleration_scale_rad_s2=60.,world_rotation_no_zero_velocity_target=True,
            ankle_pivot_no_translation=True,shoe_knot_spacing_target_s=.5)
    def evaluate(self,x,jacobian=False):
        d=self.body.state.increments(x,self.times);s=fk_state(self.ref,self.base,d)
        rel=d.skate_relative_R;error=_log(rel);w=np.sqrt(self.weights)/self.config['relative_rotation_sigma_rad']
        rp=error*w[:,None,None];rows=[rp.ravel()];mats=[];parts={'finite_relative_skate_wear_prior':float(np.sum(rp*rp)/2)}
        if jacobian:
            blocks=inverse_right_jacobian(error)*w[:,None,None,None]
            local=np.zeros((len(d),6,6))
            for f in (0,1):local[:,f*3:f*3+3,f*3:f*3+3]=blocks[:,f]
            mats.append(d.owner.shoe_chain(local,d.parameters,d.sample_times))
        # Relative SO3 interval velocity limits oscillation, while actual-world
        # acceleration allows the large smooth rotation of a real spin.
        h=np.diff(self.times);phi=_log(rel[1:]@rel[:-1].swapaxes(-1,-2))
        weight=np.sqrt(h)/self.config['relative_angular_velocity_scale_rad_s']
        rv=phi*weight[:,None,None]/h[:,None,None];rows.append(rv.ravel())
        parts['relative_skate_angular_velocity']=float(np.sum(rv*rv)/2)
        if jacobian:
            K=inverse_right_jacobian(phi)@rel[:-1];J=csr_matrix((rv.size,self.body.size))
            for offset,sign in [(0,-1.),(1,1.)]:
                local=np.zeros((len(h),6,6));C=sign*K*(weight/h)[:,None,None,None]
                for f in (0,1):local[:,f*3:f*3+3,f*3:f*3+3]=C[:,f]
                J+=d.owner.shoe_chain(local,d.parameters,d.sample_times,sample_indices=np.arange(len(h))+offset)
            mats.append(J)
        R=s['R'];phi=_log(R[1:]@R[:-1].swapaxes(-1,-2));omega=phi/h[:,None,None]
        a=2/(h[:-1]+h[1:]);alpha=a[:,None,None]*np.diff(omega,axis=0)
        source=self.ref.p['pts'][1:-1]/12800.;phase=np.ones(len(a));phase[((source>=80.8)&(source<=82.4))|((source>=92.2)&(source<=93.2))]=.1
        weight=np.sqrt((h[:-1]+h[1:])/2*phase)/self.config['actual_shoe_angular_acceleration_scale_rad_s2']
        z=alpha*weight[:,None,None];ra,D=robust_vector(z,delta=3.);rows.append(ra.ravel())
        parts['actual_skate_so3_angular_acceleration']=float(np.sum(ra*ra)/2)
        if jacobian:
            JR=rotation_chain(d,s,self.ref);K=inverse_right_jacobian(phi)@R[:-1]
            coeff=[a[:,None,None,None]*K[:-1]/h[:-1,None,None,None],
                -a[:,None,None,None]*(K[:-1]/h[:-1,None,None,None]+K[1:]/h[1:,None,None,None]),
                 a[:,None,None,None]*K[1:]/h[1:,None,None,None]]
            J=csr_matrix((ra.size,self.body.size))
            for offset,C in enumerate(coeff):
                C=D@(weight[:,None,None,None]*C);blocks=block_diag(list(C.reshape(-1,3,3)),format='csr')
                ids=(np.arange(len(a))[:,None]*6+np.arange(6)[None]+offset*6).ravel()
                J+=blocks@JR[ids]
            mats.append(J)
        from .weighted_blocks import block,full_columns
        named=[full_columns(block(name,value,mats[i]if jacobian else None,dict(source_pts=self.ref.p['pts'])))for i,(name,value)in enumerate(zip(parts,rows))]
        r=np.concatenate(rows)
        return dict(residual=r,jac=_full(vstack(mats,format='csr')) if jacobian else None,parts=parts,weighted_blocks=named,
            metrics=dict(relative_skate_degrees=np.rad2deg(np.linalg.norm(error,axis=-1)).tolist(),config=self.config))


MODE_NAMES=np.array(['runner_interior','runner_front','runner_rear','toe_pick','heel','off_ice','unknown'])

def duration_path(cost,switch_cost=1.5,minimum_samples=4):
    """Expanded-state Viterbi: finite switch cost and native-sample duration."""
    n,k=cost.shape;D=minimum_samples;v=np.full((k,D),np.inf);v[:,0]=cost[0]
    back=np.zeros((n,k,D),np.int16)
    for i in range(1,n):
        out=np.full_like(v,np.inf)
        for mode in range(k):
            old=v[mode];out[mode,1:]=old[:-1]+cost[i,mode]
            back[i,mode,1:]=mode*D+np.arange(D-1)
            if old[-1]<=old[-2]:out[mode,-1]=old[-1]+cost[i,mode];back[i,mode,-1]=mode*D+D-1
            other=v[:,-1]+switch_cost;other=other.copy();other[mode]=np.inf;j=int(np.argmin(other))
            out[mode,0]=other[j]+cost[i,mode];back[i,mode,0]=j*D+D-1
        v=out
    flat=int(np.argmin(v));path=np.empty(n,int)
    for i in range(n-1,-1,-1):path[i]=flat//D;flat=int(back[i].ravel()[flat])
    return path


class TemporalContactFactors:
    """A sequence association, separate from true-contact labels and argmin."""
    def __init__(self,problem,x):
        self.problem=problem;self.body=problem.body;self.ref=problem.reference;self.binding=problem.binding
        self.times=self.ref.times;self.pts=self.ref.p['pts'];self.base=self.ref.at(self.times)
        self.weights=trapezoid_time_weights(self.times)*20.
        self.p=self.binding.packet()['parameters'];self.band,self.uscale=runner_interior_band(self.p)
        self.rear,self.front,self.nscale=runner_tangent_limits(self.p);self.gscale=self.p.blade_web_height_m*self.binding.ratio
        self.associate(x)
    def geometry(self,x):
        d=self.body.state.increments(x,self.times);s=fk_state(self.ref,self.base,d)
        f=batch_features(self.binding,s['R'],s['O']);hh=f['height'];q=f['local_point'];ix=np.arange(len(d))[:,None];feet=np.arange(2)[None]
        runner=np.array([i for i,v in enumerate(f['semantics']) if v=='runner']);pick=np.array([i for i,v in enumerate(f['semantics']) if v=='toe_pick'])
        rb=runner[np.argmin(hh[:,:,runner],axis=-1)];pb=pick[np.argmin(hh[:,:,pick],axis=-1)]
        qr=q[ix,feet,rb];qp=q[ix,feet,pb];n=s['R'][:,:,2,:]
        flat=n.reshape(-1,3);valid=(flat[:,2]>0)&(np.sum(flat[:,[0,2]]**2,axis=1)>1e-14)
        x0=-self.p.runner_rear_extension_m;x1=self.p.sole_length_m-self.p.runner_front_inset_m
        raw_u=(qr[:,:,0].ravel()-x0)/(x1-x0)
        co=dict(effective_u=raw_u.copy(),actual_u=raw_u.copy(),derivative_normal=np.zeros_like(flat))
        if valid.any():
            from .runner_interior import canonical_runner_witness
            canonical,good=canonical_runner_witness(flat[valid],self.p,qr.reshape(-1,3)[valid])
            qr=qr.copy();qr.reshape(-1,3)[valid]=canonical
            for key in co:co[key][valid]=good[key]
        co['stationary_valid']=valid
        u=co['effective_u'].reshape(-1,2);gap=hh[ix,feet,rb]-hh[ix,feet,pb]
        runner_height=(s['O'][:,:,2]+np.sum(n*qr,axis=-1))*self.binding.ratio
        gap=runner_height-hh[ix,feet,pb]
        return d,s,f,qr,qp,n,co,u,gap,runner_height,hh[ix,feet,pb]
    def associate(self,x):
        values=self.geometry(x);u=values[7];gap=values[8];rh=values[9];ph=values[10];t=self.pts/12800.;N=len(t)
        modes=np.full((N,2),6,int);status=np.full((N,2),'unknown',dtype='<U16');costs=np.zeros((N,2,7))
        for foot in (0,1):
            cost=np.zeros((N,7));supported=(t>=81.62)&(t<=92.46)&(foot==1)
            lifted=((t>=86.2)&(t<=92.4)&(foot==0))|((t>=81.36)&(t<=81.46))
            ledger=getattr(self.ref,'source_document',{}).get('source_episode_review')
            if ledger:
                supported=np.zeros(N,bool);lifted=np.zeros(N,bool)
                for episode in ledger['episodes']:
                    mask=(self.pts>=episode['start_pts'])&(self.pts<=episode['end_pts_inclusive'])
                    if episode['kind']=='support_candidate'and episode.get('foot')==foot:supported|=mask
                    if(episode['kind']=='free_foot_off_ice_candidate'and episode.get('foot')==foot)or episode['kind']=='flight_visible_candidate':lifted|=mask
                if np.any(supported&lifted):raise ValueError('Conflicting source support/flight episode')
            cost[supported,6]=6.;cost[supported,5]=5.
            cost[supported,:5]=np.c_[np.maximum(self.band[0]-u[supported,foot],0)**2+np.maximum(u[supported,foot]-self.band[1],0)**2,
                .7+np.maximum(.75-u[supported,foot],0)**2,
                .7+np.maximum(u[supported,foot]-.25,0)**2,
                2.+np.maximum(-gap[supported,foot]/self.gscale,0)**2,
                2.+np.maximum(u[supported,foot]-.15,0)**2]
            cost[supported,:3]+=np.maximum(gap[supported,foot,None]/self.gscale,0)**2
            cost[lifted,:5]+=8.;cost[lifted,6]+=2.
            free=~(supported|lifted);cost[free,:6]+=.25
            path=duration_path(cost)
            modes[:,foot]=path;status[supported|lifted,foot]='inferred';costs[:,foot]=cost
        self.modes=modes;self.status=status;self.association_costs=costs
        self.record=dict(method='duration-expanded Viterbi',mode_names=MODE_NAMES.tolist(),switch_cost=1.5,minimum_duration_samples=4,
            source_phase_from='source temporal review; right support and left held foot',
            emission='source phase plus finite current runner/pick geometry; no argmin-as-contact-truth',
            observed_contact_material='unknown',inference_is_not_new_source_observation=True)
        if ledger:self.record.update(source_phase_from='explicit source_episode_review ledger',source_episode_ids=[e['episode_id']for e in ledger['episodes']],toe_pick_heel_office_physics='phase association only; no new toe/heel contact residual')
        from .weighted_blocks import signature
        self.association_signature=signature(self.modes,self.status)
    def evaluate(self,x,jacobian=False):
        d,s,f,qr,qp,n,co,u,gap,rh,ph=self.geometry(x);N=len(d);feet=np.arange(2)[None]
        # Mode association is fixed during this continuous solve. It may be
        # recomputed at an explicit whole-sequence stage boundary.
        interior=self.modes==0;front=self.modes==1;rear=self.modes==2;runner=interior|front|rear
        w=self.weights[:,None]*interior;legacy,D,raw=runner_mode_terms(n.reshape(-1,3),gap.ravel(),self.rear,self.front,self.nscale,self.gscale,np.broadcast_to(w,(N,2)).ravel())
        wr=self.weights[:,None]*runner
        low=np.where(front,.75,np.where(rear,-.2,self.band[0]));high=np.where(front,1.2,np.where(rear,.25,self.band[1]))
        derivative_u=np.sqrt(wr)/self.uscale*((u<low)*-1.+(u>high))
        ur=np.sqrt(wr)/self.uscale*(np.maximum(low-u,0)+np.maximum(u-high,0))
        er=np.zeros((N,2));r=np.concatenate((legacy.reshape(N,2,3),ur[:,:,None],er[:,:,None]),axis=-1)
        J=None
        if jacobian:
            from .sequence_skate_state import perturb_relative
            def terms(sa,sb,h):
                dn=(sa['R'][:,:,2,:]-sb['R'][:,:,2,:])/(2*h)
                dr=np.stack((dn@self.rear,-dn@self.front,self.binding.ratio*np.sum(dn*(qr-qp),axis=-1)),axis=-1)
                du=np.sum(co['derivative_normal'].reshape(N,2,3)*dn,axis=-1)
                return np.concatenate((D.reshape(N,2,3)*dr,(derivative_u*du)[:,:,None],np.zeros((N,2,1))),axis=-1).reshape(N,10)
            body=[];h=1e-6
            for a in self.body.state.active:
                plus=d.copy();minus=d.copy();plus[:,a]+=h;minus[:,a]-=h
                body.append(terms(fk_state(self.ref,self.base,plus),fk_state(self.ref,self.base,minus),h))
            local=np.stack(body,-1);BJ=d.chain(local)
            shoe=[]
            for foot in (0,1):
                for a in range(3):shoe.append(terms(fk_state(self.ref,self.base,perturb_relative(d,foot,a,h)),fk_state(self.ref,self.base,perturb_relative(d,foot,a,-h)),h))
            BJ+=d.owner.shoe_chain(np.stack(shoe,-1),d.parameters,d.sample_times);J=_full(BJ)
        r=r.ravel()
        return dict(residual=r,jac=J,parts={'temporal_contact_mode':float(r@r/2)},metrics=dict(sequence_contact_modes=MODE_NAMES[self.modes].tolist(),sequence_contact_status=self.status.tolist(),mode_association=self.record,stationary_effective_u=u.tolist(),geometric_minimum_is_diagnostic=True))
