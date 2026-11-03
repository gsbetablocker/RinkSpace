"""Evidence-gated material blade velocity, with explicit temporal Jacobians."""
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.sparse import coo_matrix,csr_matrix,hstack
from .skate_physics import batch_features,fk_state
from .skating_action_factors import validate_source_timeline,glide_lateral_residual


def feature_scatter(local,basis,state):
    """Scatter separable per-time derivatives before any temporal operator."""
    fr,kn=np.nonzero(basis);width=local.shape[1]
    v=local[fr]*basis[fr,kn,None,None]
    rr=np.broadcast_to(fr[:,None,None]*width+np.arange(width)[None,:,None],v.shape)
    cc=np.broadcast_to(kn[:,None,None]*state.d+np.arange(state.d),v.shape)
    return coo_matrix((v.ravel(),(rr.ravel(),cc.ravel())),shape=(len(basis)*width,state.size)).tocsr()


class GlideActionFactors:
    """Fixed external phase/foot evidence; no pose-height self-labelling.

    Each residual retains all three time slots. The two-foot state features
    are O, R, exact rocker q*, and its tangent. q* changes with posture,
    but its time derivative is deliberately absent from material velocity.
    """
    def __init__(self,problem,evidence,*,sigma_mps,enabled):
        self.problem=problem;self.body=problem.body;self.reference=problem.reference
        self.binding=problem.binding;self.times=self.body.mid_times
        self.ref=self.reference.at(self.times);pts=self.reference.p['pts'][self.body.frames]
        self.clock=validate_source_timeline(pts[1:-1],(1,12800),source_id=evidence['source_id'])
        self.dt=(pts[2:]-pts[:-2])/12800.
        self.phase=np.full((len(pts)-2,2),'unknown',dtype='<U32');self.confidence=np.zeros(self.phase.shape)
        self.sigma=float(sigma_mps);self.enabled=bool(enabled)
        self.evidence=evidence
        for interval in evidence['intervals']:
            if interval.get('phase')!='stable_glide':continue
            if interval.get('origin') not in ('video_observation','human_review'):continue
            if interval.get('physical_foot') not in (0,1):continue
            foot=interval['physical_foot'];lo,hi=interval['start_pts'],interval['end_pts']
            boundary=interval.get('boundary_uncertainty_pts',[0,0])
            # The entire derivative stencil must be within reliable phase.
            use=(pts[:-2]>=lo+boundary[0])&(pts[2:]<=hi-boundary[1])
            self.phase[use,foot]='stable_glide';self.confidence[use,foot]=interval['confidence']

    def features(self,s):
        R=s.get('boot_R',s.get('R'));O=s.get('boot_O',s.get('O'))
        f=batch_features(self.binding,R,O)
        # Only continuous runner arcs define a blade tangent. The steel web
        # mesh guard shares runner semantics for collision, not for motion.
        arcs=self.binding.packet()['template']['contact_arcs']
        ids=np.array([j for j,arc in enumerate(arcs) if arc['semantic']=='runner'],dtype=int)
        heights=f['height'][...,ids];which=ids[np.argmin(heights,axis=-1)]
        n=len(R);q=f['local_point'][np.arange(n)[:,None],np.arange(2)[None],which]
        tangent=np.zeros_like(q)
        for j,arc in enumerate(arcs):
            mask=which==j
            if not mask.any():continue
            a=np.asarray(arc['axis_cos']);b=np.asarray(arc['axis_sin'])
            rel=(q-np.asarray(arc['center']))/arc['radius_m']
            cosine=rel@a;sine=rel@b
            tangent[mask]=(-sine[...,None]*a+cosine[...,None]*b)[mask]
        return np.concatenate((O*self.binding.ratio,R.reshape(n,2,9),q*self.binding.ratio,tangent),axis=-1).reshape(n,36)

    def rows(self,stencil):
        result=[];details=[]
        for foot in range(2):
            f=stencil[:,:,foot];rot=Rotation.from_matrix(f[:,:,3:12].reshape(-1,3,3)).as_matrix().reshape(-1,3,3,3)
            omega=Rotation.from_matrix(rot[:,2]@rot[:,0].transpose(0,2,1)).as_rotvec()/self.dt[:,None]
            rows=glide_lateral_residual(self.clock,rotation_world=rot[:,1],
                origin_velocity_world_mps=(f[:,2,:3]-f[:,0,:3])/self.dt[:,None],
                angular_velocity_world_radps=omega,contact_point_local_m=f[:,1,12:15],
                blade_tangent_local=f[:,1,15:18],ice_normal_world=[0,0,1],
                phases=self.phase[:,foot],confidence=self.confidence[:,foot],sigma_mps=self.sigma)
            result.append(rows.residual[:,0]);details.append(rows)
        return np.stack(result,axis=1),details

    def evaluate(self,x,jacobian=False):
        state=self.body.state;delta=state.increments(x,self.times)
        physics=fk_state(self.reference,self.ref,delta);f=self.features(physics)
        stencil=np.stack((f[:-2],f[1:-1],f[2:]),axis=1).reshape(-1,3,2,18)
        residual,details=self.rows(stencil);r=residual.ravel()
        J=csr_matrix((len(r),self.problem.size)) if jacobian else None
        if jacobian and self.enabled and np.any(self.confidence>0):
            local=self.body.execution.state_differences(self.reference,self.ref,delta,state.active,self.features,step=1e-5)
            Jf=feature_scatter(np.stack(local,-1),state.basis(self.times),state)
            rr=[];cc=[];vv=[];n=len(stencil);frame=np.arange(n)[:,None];foot=np.arange(2)[None]
            for slot in range(3):
                for column in range(18):
                    h=1e-5;plus=stencil.copy();minus=stencil.copy()
                    plus[:,slot,:,column]+=h;minus[:,slot,:,column]-=h
                    v=(self.rows(plus)[0]-self.rows(minus)[0])/(2*h)
                    rr.extend((frame*2+foot).ravel());cc.extend(((frame+slot)*36+foot*18+column).ravel());vv.extend(v.ravel())
            A=coo_matrix((vv,(rr,cc)),shape=(len(r),f.size)).tocsr()
            body_J=A@Jf
            shared_J=np.zeros((len(r),17))
            if getattr(self.problem,'optimize_shared',True):
                saved=self.binding.x.copy()
                try:
                    for c in range(17):
                        pair=[];h=1e-5
                        for sign in (1.,-1.):
                            self.binding.x=saved.copy();self.binding.x[c]+=sign*h
                            fs=self.features(fk_state(self.reference,self.ref,delta))
                            st=np.stack((fs[:-2],fs[1:-1],fs[2:]),axis=1).reshape(-1,3,2,18)
                            pair.append(self.rows(st)[0].ravel())
                        shared_J[:,c]=(pair[0]-pair[1])/(2*h)
                finally:self.binding.x=saved
            J=hstack((csr_matrix(shared_J),body_J),format='csr')
        active=np.stack([d.enabled for d in details],axis=1)
        speeds=np.stack([d.diagnostics['lateral_velocity_mps']for d in details],axis=1)
        metrics=dict(enabled_rows=int(active.sum()),disabled_rows=int(active.size-active.sum()),
            disabled_reason_counts={reason:sum(d.reasons.count(reason)for d in details)for reason in sorted(set(sum([list(d.reasons)for d in details],[])))},
            sigma_mps=self.sigma,time_measure='one frozen source observation per centre; no extra sqrt(.02)',
            active_lateral_speed_abs_p95_mps=float(np.quantile(abs(speeds[active]),.95)) if active.any() else None)
        if not self.enabled:
            r=np.empty(0);J=csr_matrix((0,self.problem.size)) if jacobian else None
        return dict(residual=r,jac=J,parts={'glide_lateral_material_velocity':float(r@r/2)},metrics=metrics)
