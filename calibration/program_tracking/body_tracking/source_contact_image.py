"""Native image support witness, without treating all lower steel as Z=0.

Support identity comes from the source episode. The exact moving geometric
minimum must lie in a candidate image band; no material contact location is
invented. Finite ice remains a separate existing physical term.
"""
import numpy as np
from scipy.sparse import csr_matrix,hstack
from .source_support_locations import SourceSupportLocationFactors,_runner_witness
from .fullbody_fit import right_jacobian,skew
from .skate_physics import fk_state
from .rebuild_exposure_body import moving_projection
from .absolute_motion_state import chain_exposure
from .spin_grasp_factors import region_terms
from .weighted_blocks import signature

def polyline_nearest(points,source):
    A=source[:-1];V=np.diff(source,axis=0);l2=np.maximum(np.sum(V*V,axis=1),1e-20)
    alpha=np.clip(np.sum((points[:,None]-A)*V,axis=-1)/l2,0,1);candidate=A+alpha[...,None]*V
    choice=np.argmin(np.sum((points[:,None]-candidate)**2,axis=-1),axis=1);ii=np.arange(len(points));f=alpha[ii,choice];nearest=candidate[ii,choice]
    D=np.broadcast_to(np.eye(2),(len(points),2,2)).copy();inside=(f>0)&(f<1);t=V[choice]
    D[inside]-=t[inside,:,None]*t[inside,None,:]/l2[choice[inside],None,None]
    return nearest,D,choice,inside

def witness_local_derivatives(binding,d,state,witness,rotation,feet):
    n=len(feet);point=witness['world'];local=np.zeros((n,3,69));local[:,:,:3]=np.eye(3)
    dq=np.einsum('nij,njk->nik',rotation,witness['derivative_normal'])
    def apply(axes,pivot):
        dn=np.einsum('nji,njk->nik',rotation,skew(np.array([0.,0.,1.]))@axes)
        return -skew(point-pivot)@axes+dq@dn
    local[:,:,3:6]=apply(state['Q']@right_jacobian(np.asarray(d)[:,3:6]),state['P'])
    for foot,joints in [(0,(1,4,7)),(1,(2,5,8))]:
        selected=feet==foot
        for joint in joints:
            a=6+(joint-1)*3;axes=state['Q']@state['frames'][:,joint]@right_jacobian(np.asarray(d)[:,a:a+3]);value=apply(axes,state['native_world'][:,joint]);local[selected,:,a:a+3]=value[selected]
    RFB,tFB,_=binding.parameters();ankle=-np.einsum('fi,fij->fj',tFB,RFB);right=-rotation@skew(witness['local']-ankle[feet])+dq@skew(rotation[:,2,:]);shoe=np.zeros((n,3,6))
    for foot in (0,1):shoe[feet==foot,:,3*foot:3*foot+3]=right[feet==foot]
    return local,shoe

class NativeSupportContactFactors(SourceSupportLocationFactors):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.frames=np.asarray([r['frame']for r in self.rows]);self.times=(self.ref.times[self.frames,None]+np.linspace(-.5,.5,5)*self.body.tau).ravel();self.base=self.ref.at(self.times)
        self.legacy=SourceSupportLocationFactors(*args,**kwargs)

    def evaluate(self,x,jacobian=False):
        x=np.asarray(x)
        if x.shape==(self.problem.size,):x=x[17:]
        n=len(self.rows)
        if not n:raise ValueError('Native support requires admitted source image candidates')
        d=self.body.state.increments(x,self.times);s=fk_state(self.ref,self.base,d);pick=np.arange(n*5);feet=np.repeat(self.feet,5);R=s['R'][pick,feet];O=s['O'][pick,feet];w=_runner_witness(self.binding,R,O)
        pr=moving_projection(self.problem.camera,w['world'].reshape(n,5,1,3),self.frames,jacobian=jacobian);uv=pr['uv'][:,0];res=[];DD=[];metrics=[];associations=[]
        for i,row in enumerate(self.rows):
            original=row['feature']['source_annotation'];source=np.asarray(original['points_uv'],float);nearest,N,choice,inside=polyline_nearest(uv[i:i+1],source);radius=float(np.median(row['feature']['mapping']['pixel_uncertainty_radius_px']))
            factor=dict(centre_uv=nearest[0],region_shape_px2=np.eye(2)*radius**2,objective_weight=row['objective_weight'],reliability=1.)
            r,D,outside=region_terms(uv[i],factor);res.append(r);DD.append(D@N[0]);associations.append((int(choice[0]),bool(inside[0]),bool(outside>0)))
            metrics.append(dict(pts=row['pts'],foot=row['foot'],predicted_native_uv=uv[i].tolist(),source_candidate_polyline_uv=source.tolist(),source_radius_px=radius,outside_candidate_band_px=float(outside),geometric_minimum_world_conditional_m=(w['world'][5*i+2]*self.ratio).tolist(),source_contact_material='unknown',support_identity='source episode',plane_projection_of_entire_source_line_used=False))
        residual=np.asarray(res).ravel();J=None
        if jacobian:
            local,shoe=witness_local_derivatives(self.binding,d,s,w,R,feet)
            A=np.einsum('nplc,nlck->nplk',pr['pixel_jac_nodes'][:,0],local.reshape(n,5,3,69));S=np.einsum('nplc,nlck->nplk',pr['pixel_jac_nodes'][:,0],shoe.reshape(n,5,3,6));D=np.asarray(DD)
            A=np.einsum('nij,njlk->nilk',D,A);S=np.einsum('nij,njlk->nilk',D,S);sample=np.arange(n*5).reshape(n,5)
            BJ=chain_exposure(A,d,sample_indices=sample)+d.owner.shoe_exposure_chain(S,d.parameters,d.sample_times,sample_indices=sample)
            J=hstack([csr_matrix((len(residual),17)),BJ],format='csr')
        # Preserve the old camera-conditional XY diagnostic, but do not claim
        # that its convexified Z=0 footprint is a measured contact observation.
        legacy=self.legacy.evaluate(x,False)['metrics'];legacy.update(native_source_support_contacts=metrics,source_support_location_optimization_semantics='native support candidate band plus existing finite ice; old planar whole-line footprint diagnostic only')
        self.association_signature=signature(w['feature_index'],w['interior'],str(associations));self.last=dict(residual=residual,jac=J,parts={'native_source_support_contact':float(residual@residual/2)},image_cost=float(residual@residual/2),metrics=legacy)
        return self.last
