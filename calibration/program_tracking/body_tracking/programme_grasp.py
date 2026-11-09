"""Source-supported S09 palm/held-shoe proximity on the existing full state.

The material point is unobserved: minimize over the actual finite shoe surface.
This is a finite positional expectation, not a finger-contact or force model.
"""
import json
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix,vstack
from .programme_surface import ProgrammeSurfaceFactor,CollisionSurface
from .selected_surface_linearization import SelectedSurfaceLinearization
from .skate_physics import fk_state
from .sequence_skate_state import perturb_relative
from .skate_contact import trapezoid_time_weights
from .weighted_blocks import block,full_columns,signature


def closest_triangle(point,triangles):
    """Exact closest point on each finite triangle, including degenerate edges."""
    p=np.asarray(point,dtype=float);T=np.asarray(triangles,dtype=float);a,b,c=np.moveaxis(T,1,0)
    u=b-a;v=c-a;d=p-a;uu=np.einsum('ni,ni->n',u,u);uv=np.einsum('ni,ni->n',u,v);vv=np.einsum('ni,ni->n',v,v)
    du=np.einsum('ni,ni->n',d,u);dv=np.einsum('ni,ni->n',d,v);den=uu*vv-uv*uv
    good=den>np.maximum(uu*vv,1e-30)*1e-12
    beta=np.divide(vv*du-uv*dv,den,out=np.zeros_like(den),where=good);gamma=np.divide(uu*dv-uv*du,den,out=np.zeros_like(den),where=good)
    bary=np.c_[1-beta-gamma,beta,gamma];inside=good&np.all(bary>=0,axis=1)
    candidates=[np.einsum('ni,nij->nj',bary,T)];weights=[bary];scores=[np.where(inside,np.sum((candidates[0]-p)**2,axis=1),np.inf)]
    for i,j in ((0,1),(1,2),(2,0)):
        edge=T[:,j]-T[:,i];length=np.einsum('ni,ni->n',edge,edge)
        fraction=np.clip(np.divide(np.einsum('ni,ni->n',p-T[:,i],edge),length,out=np.zeros_like(length),where=length>0),0,1)
        q=T[:,i]+fraction[:,None]*edge;w=np.zeros((len(T),3));w[:,i]=1-fraction;w[:,j]=fraction
        candidates.append(q);weights.append(w);scores.append(np.sum((q-p)**2,axis=1))
    scores=np.stack(scores,axis=1);choice=np.argmin(scores,axis=1);ii=np.arange(len(T))
    return np.stack(candidates,axis=1)[ii,choice],np.stack(weights,axis=1)[ii,choice],scores[ii,choice]


class ProgrammeGraspFactors(ProgrammeSurfaceFactor):
    def __init__(self,problem,document):
        self.problem=problem;self.ref=problem.reference;self.state=problem.body.state
        cfg=json.loads(Path(document['programme_structure']['policy']).read_text(encoding='utf8'))
        self.episodes=cfg['source_grasp_review'];self.config=dict(self.ref.grasp_relations['factor_parameters']['grip'])
        self.reliability=float(document['programme_structure']['source_grasp_reliability'])
        if not np.isfinite(self.reliability) or self.reliability<=0:
            raise ValueError('Source grasp reliability must be finite and positive')
        if not self.episodes:
            raise ValueError('Source grasp factors require at least one source window')
        self.geometry=CollisionSurface(self.ref,document)
        self.rows=[];self.dt=[];native=self.ref.times+self.geometry.origin
        for k,e in enumerate(self.episodes):
            chosen=np.flatnonzero((native>=e['source_seconds'][0]-1e-9)&(native<=e['source_seconds'][1]+1e-9))[::2]
            if not len(chosen):
                raise ValueError(f"Source grasp window has no native exposures: {e['source_seconds']}")
            self.rows.extend((int(i),k) for i in chosen)
            self.dt.extend(trapezoid_time_weights(self.ref.times[chosen],singleton_duration_s=.04))
        self.rows=np.asarray(self.rows,int).reshape(-1,2);self.dt=np.asarray(self.dt)
        self.times=self.ref.times[self.rows[:,0]];self.association=signature(self.rows,self.episodes,self.config,self.reliability)
        self.palms=self.ref.grasp_geometry['palm_patches']
        self.gear_faces=[np.flatnonzero(np.all((self.geometry.faces>=self.geometry.skin_count+f*self.geometry.gear_count)&(self.geometry.faces<self.geometry.skin_count+(f+1)*self.geometry.gear_count),axis=1)) for f in (0,1)]

    def evaluate(self,x,jacobian=False):
        from .fullbody_fit import right_jacobian
        n=len(self.rows);residual=np.zeros(n);distance=np.zeros(n);target=np.zeros(n,int);jac=[]
        for first in range(0,n,24):
            rows=np.arange(first,min(first+24,n));ts=self.times[rows];base=self.ref.at(ts);d=self.state.increments(x,ts);posed=fk_state(self.ref,base,d)
            full=self.geometry.vertices(posed);palm_ids=[];palm_weights=[];faces=[];barys=[]
            for k,row in enumerate(rows):
                e=self.episodes[self.rows[row,1]];patch=self.palms[e['hand']]
                pi=np.asarray(patch['vertex_ids'],int);pw=np.asarray(patch['weights']);H=pw@full[k,pi]
                fi=self.gear_faces[e['foot']];C,B,S=closest_triangle(H,full[k,self.geometry.faces[fi]])
                chosen=int(np.argmin(S));face=int(fi[chosen]);palm_ids.append(pi);palm_weights.append(pw);faces.append(self.geometry.faces[face]);barys.append(B[chosen]);target[row]=face
            palm_ids=np.asarray(palm_ids);palm_weights=np.asarray(palm_weights);faces=np.asarray(faces);bary=np.asarray(barys)
            ids=np.unique(np.r_[palm_ids.ravel(),faces.ravel()]);pa=np.searchsorted(ids,palm_ids);pb=np.searchsorted(ids,faces)
            skin_mask=ids<self.geometry.skin_count;skin_ids=ids[skin_mask];linear=None
            if jacobian:linear=SelectedSurfaceLinearization(self.geometry.surface,posed['native'],posed['frames'],skin_ids,hand_local_R=posed.get('hand_local_R'))
            V=self._selected(posed,ids,None if linear is None else linear.samples);ii=np.arange(len(rows))[:,None]
            def vector(V):return np.einsum('ni,nij->nj',palm_weights,V[ii,pa])-np.einsum('ni,nij->nj',bary,V[ii,pb])
            delta=vector(V);length=np.linalg.norm(delta,axis=1);distance[rows]=length
            excess=np.maximum(length-self.config['contact_radius_m'],0)/self.config['residual_scale_m'];u=np.sqrt(1+(excess/self.config['robust_delta'])**2);robust=np.sqrt(2/(u+1))
            weight=np.sqrt(self.dt[rows]*self.config['objective_weight']*self.reliability);residual[rows]=weight*robust*excess
            if not jacobian:continue
            gain=weight*robust*(u+1)/(2*u)/self.config['residual_scale_m']*(excess>0)
            direction=np.divide(delta,length[:,None],out=np.zeros_like(delta),where=length[:,None]>1e-15)
            def contract(V):return gain*np.einsum('ni,ni->n',direction,vector(V))
            local=np.zeros((len(rows),1,69));shoe=np.zeros((len(rows),1,6))
            for joint in range(1,22):
                columns=6+3*(joint-1)+np.arange(3);axes=right_jacobian(np.asarray(d[:,columns]))
                skin_derivatives=linear.derivatives(joint,axes.swapaxes(1,2))*self.geometry.ratio
                for axis,column in enumerate(columns):
                    dv=np.zeros_like(V);dv[:,skin_mask]=skin_derivatives[:,axis]
                    if joint in self.geometry.gear_pose_joints:
                        plus,minus=d.copy(),d.copy();plus[:,column]+=1e-6;minus[:,column]-=1e-6
                        dv[:,~skin_mask]=(self._selected(fk_state(self.ref,base,plus),ids[~skin_mask])-self._selected(fk_state(self.ref,base,minus),ids[~skin_mask]))/(2e-6)
                    local[:,0,column]=contract(dv)
            for foot in (0,1):
                for axis in range(3):
                    dv=np.zeros_like(V);plus=perturb_relative(d,foot,axis,1e-6);minus=perturb_relative(d,foot,axis,-1e-6)
                    dv[:,~skin_mask]=(self._selected(fk_state(self.ref,base,plus),ids[~skin_mask])-self._selected(fk_state(self.ref,base,minus),ids[~skin_mask]))/(2e-6)
                    shoe[:,0,3*foot+axis]=contract(dv)
            J=d.chain(local)+d.owner.shoe_chain(shoe,d.parameters,d.sample_times);jac.append(J)
        J=vstack(jac,format='csr') if jacobian and jac else csr_matrix((n,self.state.size)) if jacobian else None
        b=full_columns(block('source_supported_grasp',residual,J,dict(source_pts=self.ref.p['pts'][self.rows[:,0]],episodes=self.episodes,
            config=self.config,reliability=self.reliability,geometry='source-selected palmar skin patch to nearest actual finite held-shoe triangle',
            material_point_observed=False,exact_nearest_point_recomputed=True,root_rigid_columns_cancel=True),self.association))
        report=[]
        for k,e in enumerate(self.episodes):
            take=self.rows[:,1]==k;report.append(dict(hand=e['hand'],foot=e['foot'],source_seconds=e['source_seconds'],samples=int(take.sum()),
                source_pts=self.ref.p['pts'][self.rows[take,0]].tolist(),distance_m=distance[take].tolist(),target_faces=target[take].tolist(),
                median_distance_m=float(np.median(distance[take])),p95_distance_m=float(np.percentile(distance[take],95)),max_distance_m=float(distance[take].max())))
        return dict(residual=residual,jac=b.jacobian,parts={b.name:float(residual@residual/2)},weighted_blocks=[b],
            metrics={b.name:dict(episodes=report,position_expectation_only=True,contact_or_force_not_certified=True)})
