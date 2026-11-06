"""Reverse complete visible assembly/tail ranges on the existing skate mesh."""
import numpy as np
from scipy.sparse import csr_matrix,hstack
from .source_visible_model import structure_geometry,camera_centres,visible_segments
from .source_contact_image import polyline_nearest
from .skate_physics import fk_state
from .rebuild_exposure_body import moving_projection
from .absolute_motion_state import chain_exposure
from .sequence_skate_state import perturb_relative
from .spin_grasp_factors import region_terms
from .weighted_blocks import signature

class CompleteStructureExtentFactors:
    def __init__(self,problem,document):
        self.problem=problem;self.body=problem.body;self.ref=problem.reference;self.binding=problem.binding;self.rows=document['rows'];self.count=int(document.get('samples_per_complete_feature',64));self.geometry=structure_geometry(self.binding)
        lookup={int(pt):i for i,pt in enumerate(self.ref.p['pts'])};self.frames=np.asarray([lookup[r['pts']]for r in self.rows]);self.feet=np.asarray([r['foot']for r in self.rows]);self.times=(self.ref.times[self.frames,None]+np.linspace(-.5,.5,5)*self.body.tau).ravel();self.base=self.ref.at(self.times);self.q=None;self.visibility_record=[];self.last=None

    def evaluate(self,x,jacobian=False):
        x=np.asarray(x)
        if x.shape==(self.problem.size,):x=x[17:]
        n=len(self.rows);d=self.body.state.increments(x,self.times);s=fk_state(self.ref,self.base,d);pick=np.arange(n*5);feet=np.repeat(self.feet,5);R=s['R'][pick,feet].reshape(n,5,3,3);O=s['O'][pick,feet].reshape(n,5,3)
        if self.q is None:
            C=camera_centres(self.problem.camera,self.frames,[np.median(np.asarray(r['polyline_uv'])[:,1])for r in self.rows]);local=np.einsum('nji,nj->ni',R[:,2],C-O[:,2]);points=[]
            for i,row in enumerate(self.rows):
                ids,meta=visible_segments(self.geometry,local[i],row['structure_feature'])
                if not len(ids):raise ValueError('No visible complete exterior '+str(row['id']))
                segments=self.geometry['segments'][ids];length=np.linalg.norm(segments[:,1]-segments[:,0],axis=1);cdf=np.r_[0,np.cumsum(length)];at=(np.arange(self.count)+.5)/self.count*cdf[-1];which=np.minimum(np.searchsorted(cdf,at,side='right')-1,len(ids)-1);fraction=(at-cdf[which])/length[which];q=segments[which,0]+fraction[:,None]*(segments[which,1]-segments[which,0]);points.append(q);self.visibility_record.append(dict(id=row['id'],pts=row['pts'],foot=row['foot'],feature=row['structure_feature'],segments=ids.tolist(),quadrature_segment=which.tolist(),quadrature_fraction=fraction.tolist(),**meta))
            self.q=np.asarray(points)
        nodes=O[:,:,None]+np.einsum('nlij,nkj->nlki',R,self.q);pr=moving_projection(self.problem.camera,nodes,self.frames,jacobian=jacobian);uv=pr['uv'];Dall=[];res=[];metrics=[];assoc=[]
        for i,row in enumerate(self.rows):
            source=np.asarray(row['polyline_uv']);nearest,N,choice,inside=polyline_nearest(uv[i],source);radius=float(row['pixel_uncertainty_radius_px']);budget=row['contour_frame_budget']*row['feature_budget_fraction'];Ds=[];dist=np.linalg.norm(uv[i]-nearest,axis=1)
            for k in range(self.count):
                r,D,_=region_terms(uv[i,k],dict(centre_uv=nearest[k],region_shape_px2=np.eye(2)*radius**2,objective_weight=budget/self.count,reliability=row.get('reliability',1.)));res.append(r);Ds.append(D@N[k])
            Dall.extend(Ds);assoc.extend(zip(choice.tolist(),inside.tolist(),(dist>radius).tolist()));metrics.append(dict(id=row['id'],pts=row['pts'],foot=row['foot'],structure_feature=row['structure_feature'],source_polyline_uv=source.tolist(),model_visible_sample_uv=uv[i].tolist(),distance_px=dist.tolist(),outside_complete_source_band_px=np.maximum(dist-radius,0).tolist(),pixel_uncertainty_radius_px=radius,source_complete_contract=True,source_material_endpoints_known=False,source_budget_shared_with_forward=True))
        residual=np.asarray(res).ravel();D=np.asarray(Dall);J=None
        if jacobian:
            def derivative(sa,sb,h):
                ra=sa['R'][pick,feet].reshape(n,5,3,3);rb=sb['R'][pick,feet].reshape(n,5,3,3);oa=sa['O'][pick,feet].reshape(n,5,3);ob=sb['O'][pick,feet].reshape(n,5,3);world=(oa-ob)[:,:,None]/(2*h)+np.einsum('nlij,nkj->nlki',(ra-rb)/(2*h),self.q);pixel=np.einsum('nkplc,nlkc->nkpl',pr['pixel_jac_nodes'],world).reshape(-1,2,5);return np.einsum('nij,njl->nil',D,pixel)
            h=1e-6;axes=np.asarray(getattr(self.problem,'factor_derivative_columns',self.body.state.active));values=[]
            for axis in axes:
                a=d.copy();b=d.copy();a[:,axis]+=h;b[:,axis]-=h;values.append(derivative(fk_state(self.ref,self.base,a),fk_state(self.ref,self.base,b),h))
            compact=np.stack(values,-1);local=np.zeros(compact.shape[:-1]+(69,));local[...,axes]=compact;sample=np.repeat(np.arange(n*5).reshape(n,5),self.count,axis=0);BJ=chain_exposure(local,d,sample_indices=sample);shoe=[]
            for foot in (0,1):
                for axis in range(3):shoe.append(derivative(fk_state(self.ref,self.base,perturb_relative(d,foot,axis,h)),fk_state(self.ref,self.base,perturb_relative(d,foot,axis,-h)),h))
            BJ+=d.owner.shoe_exposure_chain(np.stack(shoe,-1),d.parameters,d.sample_times,sample_indices=sample);J=hstack([csr_matrix((len(residual),17)),BJ],format='csr')
        cost=float(residual@residual/2);self.association_signature=signature(str(assoc));self.last=dict(residual=residual,jac=J,parts={'complete_visible_structure_extent':cost},image_cost=cost,metrics=dict(complete_visible_structure_extents=metrics));return self.last
