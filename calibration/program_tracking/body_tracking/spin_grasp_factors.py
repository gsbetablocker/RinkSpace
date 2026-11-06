"""Source-supported closed-chain factors on the current body/skate state."""
import numpy as np
from scipy.sparse import csr_matrix,coo_matrix,hstack,vstack,kron,eye,block_diag,diags
from .skate_physics import fk_state,batch_features,body_constraint_jacobian,NORMALIZATION
from .skate_contact import trapezoid_time_weights
from .support_anchor_factors import lowest_runner_witness
from .rebuild_exposure_body import moving_projection
from .grasp_temporal_factors import GripInterval,grip_to_segment_factor,actual_local_acceleration_factor


def vee(A):return .5*np.stack((A[...,2,1]-A[...,1,2],A[...,0,2]-A[...,2,0],A[...,1,0]-A[...,0,1]),-1)

def prepare_geometry(ref,model_sha256):
    """Choose central palmar skin by verified joint/skin semantics and flexion side."""
    surface=ref.binding.surface;body=surface.body;V=surface.shaped.astype(float);J=body.J
    patches={}
    for side,wrist,first,bases,ends in [('left',20,25,[25,28,34],[27,30,36]),('right',21,40,[40,43,49],[42,45,51])]:
        centre=.45*J[wrist]+.55*J[bases].mean(0)
        normal=np.cross(J[bases[0]]-J[wrist],J[bases[-1]]-J[wrist]);normal/=np.linalg.norm(normal)
        # The model's documented non-flat default curls fingers toward the palm.
        # Use it only to select the palmar side, never add it to network rotations.
        local=np.tile(np.eye(3),(55,1,1));local[22:]=body.default;R=local.copy();pos=J.copy()
        for j in range(1,55):
            a=body.parents[j];R[j]=R[a]@local[j];pos[j]=pos[a]+R[a]@(J[j]-J[a])
        curl=(pos[ends]-J[ends]).mean(0)
        if curl@normal<0:normal=-normal
        hand_weight=surface.weights[:,[wrist]+list(range(first,first+15))].sum(1)
        candidates=np.flatnonzero((hand_weight>.8)&((V-centre)@normal>0))
        ids=candidates[np.argsort(np.linalg.norm(V[candidates]-(centre+.007*normal),axis=1))[:12]]
        if len(ids)!=12:raise ValueError('Cannot define a semantic palmar skin patch')
        patches[side]=dict(vertex_ids=ids.tolist(),weights=np.full(len(ids),1/len(ids)).tolist(),
            wrist_native_joint=wrist,finger_root_native_joints=bases,rest_centre_model_m=centre.tolist(),
            palmar_normal_model=normal.tolist(),rest_patch_extent_m=np.ptp(V[ids],axis=0).tolist(),
            provenance='SMPL-X55 joint names/parents and >0.8 hand skin weights; central wrist-to-finger-root region; palmar side selected from default finger flexion, not nearest to shoe')
    packet=ref.binding.packet();p=packet['parameters'];L=p.sole_length_m
    from .skate_template import profile_height
    xs=np.array([.02,.24])*L;zs=profile_height(xs,p)+p.blade_web_height_m
    return dict(schema='rinkspace.spin-grasp-geometry.v1',model_sha256=model_sha256,model_vertices=len(V),
        hand_mean='network flat_hand_mean=True; default used for semantic selection only',palm_patches=patches,
        grip_foot=0,grip_hand='left',grip_region_id='left_rear_sole_blade_web_finite_segment',
        segment_local_solver=np.c_[xs,np.zeros(2),zs].tolist(),length_export_ratio=ref.binding.ratio,
        region_provenance='source 86.8/87.64/89.56/91.98/92.46 supports a finite rear-heel/under-sole region, exact material ambiguous; template rear steel web is its declared positional proxy',
        limitations=['palmar-side semantic selection needs visual inspection','position only: not palm orientation, finger wrapping, forces or certified nonintersection'])


def palm_local(ref,state,spec):
    ids=np.asarray(spec['vertex_ids'],int);kw={} if 'hand_local_R' not in state else {'hand_local_R':state['hand_local_R']}
    vertices=ref.binding.surface.sample_vertices(state['native'],state['frames'],ids,double_precision=True,**kw)
    return np.einsum('v,nvc->nc',np.asarray(spec['weights']),vertices)


def export_grasp(ref,mid):
    spec=ref.grasp_geometry;ratio=ref.binding.ratio
    local=np.einsum('nji,nkj->nki',mid['Q'],mid['native']-mid['P'][:,None])
    state=dict(native=local,frames=mid['frames'],**({} if 'hand_local_R' not in mid else {'hand_local_R':mid['hand_local_R']}))
    H=palm_local(ref,state,spec['palm_patches'][spec['grip_hand']]);H=mid['P']+np.einsum('nij,nj->ni',mid['Q'],H)
    R=mid['boot_R'][:,spec['grip_foot']];O=mid['boot_O'][:,spec['grip_foot']];seg=np.asarray(spec['segment_local_solver'])
    endpoints=(O[:,None]+np.einsum('nij,kj->nki',R,seg))*ratio;H*=ratio
    a=endpoints[:,0];v=endpoints[:,1]-a;u=np.clip(np.sum((H-a)*v,axis=1)/np.sum(v*v,axis=1),0,1)
    distance=np.linalg.norm(H-a-u[:,None]*v,axis=1)
    return dict(grasp_palm_world_mid=H,grasp_segment_world_mid=endpoints,palm_to_segment_m_mid=distance)


def reassociate_support(factor,x):
    delta=factor.body.state.increments(x,factor.times);state=fk_state(factor.reference,factor.ref,delta);changes=[]
    for i,(row,o,_) in enumerate(factor.rows):
        if o.kind!='support_contact_region':continue
        R=state['R'][i*5+2,o.foot];O=state['O'][i*5+2,o.foot]
        candidates=[lowest_runner_witness([a],R,O,plane_normal_world=[0,0,1]) for a in factor.arcs.values()]
        candidates.sort(key=lambda a:a['world_point'][2]);new=candidates[0]['arc_id'];old=factor.arc_ids[row['id']]
        changes.append(dict(id=row['id'],pts=o.pts,old_arc=old,new_arc=new,
            competitors=[dict(arc=c['arc_id'],height_m=float(c['world_point'][2]*factor.binding.ratio)) for c in candidates],
            selected_by='current geometric minimum along the fixed ice normal; never image residual'))
        factor.arc_ids[row['id']]=new
    return changes


def spline_chain(local,B):
    """N x vector-width x active-columns to the same cubic body controls."""
    n,k,d=local.shape;m=B.shape[1];fr,kn=np.nonzero(B)
    values=local[fr]*B[fr,kn,None,None]
    rr=np.broadcast_to(fr[:,None,None]*k+np.arange(k)[None,:,None],values.shape)
    cc=np.broadcast_to(kn[:,None,None]*d+np.arange(d),values.shape)
    J=coo_matrix((values.ravel(),(rr.ravel(),cc.ravel())),shape=(n*k,m*d)).tocsr();J.eliminate_zeros();return J


def region_terms(uv,row,delta=2.):
    W=np.linalg.solve(np.linalg.cholesky(row['region_shape_px2']),np.eye(2));z=W@(uv-row['centre_uv']);r=np.linalg.norm(z)
    e=np.zeros(2) if r<=1 else (1-1/r)*z
    De=np.zeros((2,2)) if r<=1 else ((1-1/r)*np.eye(2)+np.outer(z,z)/r**3)@W
    u=np.sqrt(1+(e@e)/delta**2);k=np.sqrt(2/(u+1));dk=-k/(4*delta**2*u*(u+1));w=np.sqrt(row['objective_weight']*row['reliability'])
    return w*k*e,w*(k*np.eye(2)+2*dk*np.outer(e,e))@De,float(np.linalg.norm(uv-row['centre_uv'])*max(0.,1-1/max(r,1e-30)))


def source_sample_indices(pts,seconds,phase_boundaries_seconds):
    """Keep native source samples and phase boundaries inside this context."""
    pts=np.asarray(pts);source=pts/12800.
    selected=(source>=seconds[0])&(source<=seconds[1])
    boundaries=[round(v*12800) for v in phase_boundaries_seconds]
    return np.unique(np.r_[np.flatnonzero(selected)[::2],
        np.flatnonzero(selected&np.isin(pts,boundaries))])


class SpinGraspFactors:
    def __init__(self,problem,relations):
        self.problem=problem;self.body=problem.body;self.ref=problem.reference;self.binding=problem.binding
        self.relations=relations;self.geometry=self.ref.grasp_geometry;self.config=relations['factor_parameters']
        pts=self.ref.p['pts']
        chosen=source_sample_indices(pts,self.body.state.seconds,relations['phase_boundaries_seconds']);self.pts=pts[chosen]
        self.times=(self.pts-pts[0])/12800.;self.base=self.ref.at(self.times);self.B=self.body.state.basis(self.times)
        held=relations['grip_interval'];interval=GripInterval(source_id=relations['source_id'],shot_id=relations['shot_id'],time_base=tuple(relations['time_base']),**{k:tuple(v) if k=='evidence_pts' else v for k,v in held.items() if k in ('start_pts','end_pts','hand','foot','state','role','evidence_pts','provenance')})
        self.held=interval.admitted_pts(self.pts,source_id=relations['source_id'],shot_id=relations['shot_id'],time_base=relations['time_base'])
        self.grip_weights=trapezoid_time_weights(self.times[self.held],singleton_duration_s=.04) if self.held.any() else np.empty(0)
        self.joints=np.array([1,2,4,5,7,8,16,17,18,19,20,21]);self.last=None
        self.support_mask=np.zeros(len(self.pts),bool);self.support_phase_weight=np.zeros(len(self.pts))
        for interval in relations['support_intervals']:
            mask=(self.pts>=interval['start_pts'])&(self.pts<=interval['end_pts'])
            self.support_mask|=mask;self.support_phase_weight[mask]=interval['weight']
        self.support_times=self.times[self.support_mask];self.support_B=self.B[self.support_mask]
        self.support_base={k:v[self.support_mask] for k,v in self.base.items() if isinstance(v,np.ndarray) and v.shape[0]==len(self.times)}
        self.support_dt=trapezoid_time_weights(self.support_times,singleton_duration_s=.04)
        self.image_rows=[r for r in relations['body_regions'] if r['role']=='fit' and r['visibility'] in ('visible','partially_visible')]
        lookup={int(v):i for i,v in enumerate(pts)};self.image_frames=np.array([lookup[r['pts']] for r in self.image_rows],int)
        self.image_times=(self.ref.times[self.image_frames,None]+np.linspace(-.5,.5,5)*self.body.tau).ravel()
        self.image_base=self.ref.at(self.image_times);self.image_B=self.body.state.basis(self.image_times).reshape(-1,5,self.body.state.m)

    def geometry_at(self,delta):
        # Equivalent Q^T(X-P), with root algebraically eliminated on BOTH chains.
        n=len(delta);base=dict(self.base,P=np.zeros((n,3)),Q=np.tile(np.eye(3),(n,1,1)))
        d=delta.copy();d[:,:6]=0;s=fk_state(self.ref,base,d)
        H=palm_local(self.ref,s,self.geometry['palm_patches'][self.geometry['grip_hand']]);f=self.geometry['grip_foot']
        X=np.concatenate((s['native'][:,self.joints],H[:,None]),axis=1)*self.binding.ratio
        return X,H*self.binding.ratio,s['R'][:,f],s['O'][:,f]*self.binding.ratio

    def image(self,x,jacobian):
        if not len(self.image_rows):return np.empty(0),csr_matrix((0,self.problem.size)),[]
        d=self.body.state.increments(x,self.image_times);s=fk_state(self.ref,self.image_base,d)
        ids=np.repeat([r['native_joint'] for r in self.image_rows],5);pick=np.arange(len(ids))
        nodes=s['native_world'][pick,ids].reshape(-1,5,1,3);pr=moving_projection(self.problem.camera,nodes,self.image_frames,jacobian=jacobian)
        rows=[];Ds=[];metrics=[]
        for i,row in enumerate(self.image_rows):
            r,D,error=region_terms(pr['uv'][i,0],row);rows.append(r);Ds.append(D);metrics.append(dict(id=row['id'],pts=row['pts'],outside_region_px=error))
        r=np.asarray(rows).ravel();J=None
        if jacobian:
            locals=[]
            for active in self.body.state.active:
                h=1e-6;a=d.copy();b=d.copy();a[:,active]+=h;b[:,active]-=h
                sa=fk_state(self.ref,self.image_base,a);sb=fk_state(self.ref,self.image_base,b)
                dX=((sa['native_world'][pick,ids]-sb['native_world'][pick,ids])/(2*h)).reshape(-1,5,3)
                locals.append(np.einsum('nplc,nlc->npl',pr['pixel_jac_nodes'][:,0],dX))
            local=np.stack(locals,-1);local=np.einsum('nab,nbld->nald',np.asarray(Ds),local)
            if hasattr(d,'chain'):
                from .absolute_motion_state import chain_exposure
                J=chain_exposure(local,d)
            else:J=csr_matrix(np.einsum('nald,nlm->namd',local,self.image_B).reshape(len(r),self.body.size))
            J=hstack((csr_matrix((len(r),17)),J),format='csr')
        return r,J,metrics

    def continuous_support(self,x,jacobian):
        d=self.body.state.increments(x,self.support_times);s=fk_state(self.ref,self.support_base,d)
        features=batch_features(self.binding,s['R'],s['O'],s);f=1
        eligible=self.binding.packet()['template']['support_feature_ids']['glide']
        candidates=[j for j,k in enumerate(features['ids']) if k in eligible];h=features['height'][:,f,candidates]
        best=np.asarray(candidates)[np.argmin(h,axis=1)];ix=np.arange(len(h));height=features['height'][ix,f,best];q=features['local_point'][ix,f,best]
        cfg=self.config['continuous_support'];w=np.sqrt(cfg['weight']*self.support_dt*self.support_phase_weight[self.support_mask])/cfg['scale_m']
        excess=np.maximum(height-cfg['deadzone_m'],0);r=w*excess;J=None
        if jacobian:
            e=dict(rows=[(i,f,q[i],-1.) for i in ix],row_upper_vertex=[-1]*len(ix),times=self.support_times)
            derivative=-NORMALIZATION*body_constraint_jacobian(self.ref,s,d,self.body.state.active,self.support_B,e)
            J=hstack((csr_matrix((len(r),17)),diags(w*(excess>0))@derivative),format='csr')
        return r,J,dict(source_inferred_interval_samples=len(r),raw_height_m=height.tolist(),pts=self.pts[self.support_mask].tolist(),feature_ids=[features['ids'][j] for j in best])

    def evaluate(self,x,jacobian=False):
        d=self.body.state.increments(x,self.times);X,H,R,O=self.geometry_at(d);cfg=self.config['grip'];seg=np.asarray(self.geometry['segment_local_solver'])*self.binding.ratio
        grip=[grip_to_segment_factor(H[i],R[i],O[i],seg[0],seg[1],quadrature_seconds=float(w),**cfg) for i,w in zip(np.flatnonzero(self.held),self.grip_weights)]
        rg=np.concatenate([v['residual'] for v in grip]) if grip else np.empty(0);motion=self.config['local_motion']
        temporal=actual_local_acceleration_factor(X,self.times,scale_m_s2=motion['scale_m_s2'],objective_weight=motion['weight'],fixed_point_weights=np.r_[np.ones(len(self.joints)),.2],robust_delta=motion['robust_delta'])
        sec=self.pts[1:-1]/12800.;phase=np.where((sec>=86.8)&(sec<=92.46),1.,.18);scale=np.repeat(np.sqrt(phase),X.shape[1]*3)
        rt=temporal['residual']*scale;Jg=Jt=None
        if jacobian:
            dx=[];dg=[];hstep=1e-6
            for active in self.body.state.active:
                if active<6:dx.append(np.zeros_like(X));dg.append(np.zeros_like(rg).reshape(-1,3));continue
                a=d.copy();b=d.copy();a[:,active]+=hstep;b[:,active]-=hstep
                Xa,Ha,Ra,Oa=self.geometry_at(a);Xb,Hb,Rb,Ob=self.geometry_at(b)
                dx.append((Xa-Xb)/(2*hstep));eta=vee(R.swapaxes(1,2)@((Ra-Rb)/(2*hstep)))
                values=[]
                for i,v in zip(np.flatnonzero(self.held),grip):values.append(v['jacobian_palm_world']@((Ha[i]-Hb[i])/(2*hstep))+v['jacobian_shoe_origin']@((Oa[i]-Ob[i])/(2*hstep))+v['jacobian_shoe_right_rotation']@eta[i])
                dg.append(np.asarray(values).reshape(-1,3))
            Jg=(d.chain(np.stack(dg,-1),sample_indices=np.flatnonzero(self.held)) if hasattr(d,'chain') else spline_chain(np.stack(dg,-1),self.B[self.held])) if grip else csr_matrix((0,self.body.size))
            if grip and callable(getattr(getattr(d,'owner',None),'shoe_chain',None)):
                from .sequence_skate_state import perturb_relative
                shoe_columns=[]
                for foot in range(2):
                    for axis in range(3):
                        a=perturb_relative(d,foot,axis,hstep);b=perturb_relative(d,foot,axis,-hstep)
                        _,Ha,Ra,Oa=self.geometry_at(a);_,Hb,Rb,Ob=self.geometry_at(b)
                        eta=vee(R.swapaxes(1,2)@((Ra-Rb)/(2*hstep)));values=[]
                        for i,v in zip(np.flatnonzero(self.held),grip):
                            values.append(v['jacobian_palm_world']@((Ha[i]-Hb[i])/(2*hstep))+v['jacobian_shoe_origin']@((Oa[i]-Ob[i])/(2*hstep))+v['jacobian_shoe_right_rotation']@eta[i])
                        shoe_columns.append(np.asarray(values).reshape(-1,3))
                Jg=Jg+d.owner.shoe_chain(np.stack(shoe_columns,-1),d.parameters,d.sample_times,sample_indices=np.flatnonzero(self.held))
            local=np.stack(dx,-1).reshape(len(X),-1,self.body.state.d)
            pointJ=d.chain(local) if hasattr(d,'chain') else spline_chain(local,self.B)
            D2=csr_matrix(temporal['second_difference_operator']);A=block_diag(list(temporal['jacobian_per_acceleration'].reshape(-1,3,3)),format='csr')
            Jt=diags(scale)@A@kron(D2,eye(X.shape[1]*3),format='csr')@pointJ
            Jg=hstack((csr_matrix((len(rg),17)),Jg),format='csr');Jt=hstack((csr_matrix((len(rt),17)),Jt),format='csr')
        rs,Js,support=self.continuous_support(x,jacobian);ri,Ji,im=self.image(x,jacobian)
        groups=[('grip',rg,Jg),('actual_local_acceleration',rt,Jt),('continuous_support',rs,Js),('body_source_regions',ri,Ji)]
        residual=np.concatenate([r for _,r,_ in groups]);J=vstack([j for _,_,j in groups],format='csr') if jacobian else None
        offset=0;slices={}
        for name,r,j in groups:slices[name]=slice(offset,offset+len(r));offset+=len(r)
        self.last=dict(residual=residual,jac=J,parts={k:float(r@r/2) for k,r,_ in groups},block_slices=slices,image_cost=float(ri@ri/2),metrics=dict(grip_samples=len(grip),grip_pts=self.pts[self.held].tolist(),palm_to_segment_m=[v['palm_to_segment_m'] for v in grip],temporal_samples=len(X),support=support,body_regions=im,root_columns_of_relative_factors_exactly_zero=True))
        return self.last
