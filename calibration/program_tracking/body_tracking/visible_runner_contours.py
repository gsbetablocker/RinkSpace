"""Visible steel subcontours on the actual native-RS runner projection.

A source contour is not a contact point, template endpoint, or a known material
point. Correspondence to either visible runner side is a latent image match;
body, ankle-pivoted skate curves and temporal factors keep physical identity.
"""
import numpy as np
from scipy.sparse import csr_matrix,hstack
from .skate_physics import fk_state
from .rebuild_exposure_body import moving_projection
from .absolute_motion_state import chain_exposure
from .spin_grasp_factors import region_terms
from .sequence_skate_state import perturb_relative


class VisibleRunnerContourFactors:
    def __init__(self,problem,document,*,include_review=False,segment_geometry=None):
        self.problem=problem;self.body=problem.body;self.ref=problem.reference;self.binding=problem.binding
        self.rows=[r for r in document['rows'] if r.get('evidence_status')=='observed' and (r.get('role')=='fit' or include_review) and r.get('optimization_enabled',True)]
        lookup={int(v):i for i,v in enumerate(self.ref.p['pts'])}
        groups=list(dict.fromkeys((int(r['pts']),r['foot']) for r in self.rows));group_lookup={key:i for i,key in enumerate(groups)}
        self.row_groups=np.array([group_lookup[(int(r['pts']),r['foot'])] for r in self.rows],int)
        self.frames=np.array([lookup[pts] for pts,foot in groups],int);self.feet=np.array([foot for pts,foot in groups],int)
        self.times=(self.ref.times[self.frames,None]+np.linspace(-.5,.5,5)*self.body.tau).ravel();self.base=self.ref.at(self.times)
        self.curves=self.binding.packet()['edges'];self.count=self.curves.shape[1];self.q=self.curves.reshape(-1,3)
        self.starts=np.r_[np.arange(self.count-1),self.count+np.arange(self.count-1)];self.ends=self.starts+1
        self.segment_geometry=segment_geometry
        if segment_geometry is not None:
            self.q=np.asarray(segment_geometry['points'],float);self.starts=np.asarray(segment_geometry['starts'],int);self.ends=np.asarray(segment_geometry['ends'],int)
        owner=[];uv=[];factor_rows=[];point_rows=[]
        multiplier=self.ref.grasp_relations['factor_parameters']['shoe_region_weight_multiplier']
        for i,row in enumerate(self.rows):
            points=np.asarray(row['polyline_uv'],float);radius=float(row['pixel_uncertainty_radius_px'])
            if points.ndim!=2 or points.shape[1]!=2 or len(points)<(3 if segment_geometry is None else 2) or radius<=0 or row['foot'] not in (0,1):raise ValueError('Invalid source steel contour')
            if np.any(np.linalg.norm(np.diff(points,axis=0),axis=-1)<1.):raise ValueError('Steel trace samples must be distinct native pixels')
            for point in points:
                owner.append(self.row_groups[i]);point_rows.append(i);uv.append(point)
                factor_rows.append(dict(centre_uv=point,region_shape_px2=np.eye(2)*radius**2,
                    objective_weight=float(row.get('contour_frame_budget',.02*multiplier))*float(row.get('feature_budget_fraction',1.))/len(points),reliability=float(row.get('reliability',1.))))
        self.owner=np.asarray(owner,int);self.point_rows=np.asarray(point_rows,int);self.source_uv=np.asarray(uv,float);self.factor_rows=factor_rows;self.last=None
    def evaluate(self,x,jacobian=False):
        if not self.rows:return dict(residual=np.empty(0),jac=csr_matrix((0,self.problem.size)) if jacobian else None,parts={'visible_runner_contour':0.},metrics={'visible_runner_contours':[]},image_cost=0.)
        d=self.body.state.increments(x,self.times);s=fk_state(self.ref,self.base,d);n=len(self.frames);pick=np.arange(n*5);foot=np.repeat(self.feet,5)
        R=s['R'][pick,foot].reshape(n,5,3,3);O=s['O'][pick,foot].reshape(n,5,3)
        nodes=O[:,:,None]+np.einsum('nlij,kj->nlki',R,self.q)
        projection=moving_projection(self.problem.camera,nodes,self.frames,jacobian=jacobian);uv=projection['uv']
        # Closest *image* segment of the two true curved steel edges. No
        # frame-wise lowest-height witness is used as a source observation.
        starts=self.starts;ends=self.ends
        A=uv[self.owner[:,None],starts[None]];B=uv[self.owner[:,None],ends[None]];V=B-A
        fraction=np.clip(np.sum((self.source_uv[:,None]-A)*V,axis=-1)/np.maximum(np.sum(V*V,axis=-1),1e-20),0,1)
        closest=A+fraction[:,:,None]*V;distance=np.sum((closest-self.source_uv[:,None])**2,axis=-1)
        if self.segment_geometry is not None:
            for i,row in enumerate(self.rows):
                allowed=np.zeros(len(starts),bool);allowed[self.segment_geometry['feature_segments'][row['structure_feature']]]=True
                distance[np.ix_(self.point_rows==i,~allowed)]=np.inf
        choice=np.argmin(distance,axis=1)
        each=np.arange(len(choice));a=starts[choice];b=ends[choice];alpha=fraction[each,choice];prediction=closest[each,choice]
        tangent=V[each,choice];interior=(alpha>0)&(alpha<1);normal_projector=np.broadcast_to(np.eye(2),(len(choice),2,2)).copy()
        normal_projector[interior]-=tangent[interior,:,None]*tangent[interior,None,:]/np.sum(tangent[interior]**2,axis=-1)[:,None,None]
        terms=[region_terms(p,row)for p,row in zip(prediction,self.factor_rows)];residual=np.concatenate([v[0]for v in terms]);D=np.stack([v[1]for v in terms]);J=None
        if jacobian:
            rawA=projection['pixel_jac_nodes'][self.owner,a];rawB=projection['pixel_jac_nodes'][self.owner,b]
            sample=self.owner[:,None]*5+np.arange(5);selected_foot=self.feet[self.owner]
            qa=self.q[a];qb=self.q[b];baseR=R[self.owner];baseO=O[self.owner]
            def derivative(sa,sb,h):
                ra=sa['R'][pick,foot].reshape(n,5,3,3)[self.owner];rb=sb['R'][pick,foot].reshape(n,5,3,3)[self.owner]
                oa=sa['O'][pick,foot].reshape(n,5,3)[self.owner];ob=sb['O'][pick,foot].reshape(n,5,3)[self.owner]
                dr=(ra-rb)/(2*h);do=(oa-ob)/(2*h)
                xa=do+np.einsum('nlij,nj->nli',dr,qa);xb=do+np.einsum('nlij,nj->nli',dr,qb)
                pa=np.einsum('nplc,nlc->npl',rawA,xa);pb=np.einsum('nplc,nlc->npl',rawB,xb)
                pixel=(1-alpha)[:,None,None]*pa+alpha[:,None,None]*pb
                projected=np.einsum('nij,njl->nil',normal_projector,pixel)
                tangent_change=np.einsum('nc,ncl->nl',self.source_uv-prediction,pb-pa)/np.maximum(np.sum(tangent*tangent,axis=-1),1e-20)[:,None]
                projected+=interior[:,None,None]*tangent[:,:,None]*tangent_change[:,None,:]
                return np.einsum('nij,njl->nil',D,projected)
            h=1e-6;values=[]
            for axis in self.body.state.active:
                plus=d.copy();minus=d.copy();plus[:,axis]+=h;minus[:,axis]-=h
                values.append(derivative(fk_state(self.ref,self.base,plus),fk_state(self.ref,self.base,minus),h))
            local=np.stack(values,-1);BJ=chain_exposure(local,d,sample_indices=sample)
            shoe=[]
            for f in (0,1):
                for axis in range(3):shoe.append(derivative(fk_state(self.ref,self.base,perturb_relative(d,f,axis,h)),fk_state(self.ref,self.base,perturb_relative(d,f,axis,-h)),h))
            BJ+=d.owner.shoe_exposure_chain(np.stack(shoe,-1),d.parameters,d.sample_times,sample_indices=sample)
            J=hstack((csr_matrix((len(residual),17)),BJ),format='csr')
        metrics=[]
        for i,row in enumerate(self.rows):
            m=self.point_rows==i;radius=row['pixel_uncertainty_radius_px'];dist=np.sqrt(distance[each,choice])[m]
            metrics.append(dict(id=row.get('id',str(row['pts'])),pts=int(row['pts']),foot=int(row['foot']),role=row['role'],
                source_polyline_uv=row['polyline_uv'],model_closest_uv=prediction[m].tolist(),distance_px=dist.tolist(),
                outside_visible_band_px=np.maximum(dist-radius,0).tolist(),pixel_uncertainty_radius_px=radius,
                latent_runner_side=(a[m]//self.count).tolist(),latent_runner_u=((a[m]%self.count+alpha[m])/(self.count-1)).tolist(),
                structure_feature=row.get('structure_feature','runner_lower_edge'),source_status=row.get('source_status',row['evidence_status']),
                contour_is_observed=True,material_correspondence='inferred_image_match_not_fixed_material',contact_truth='unmeasured'))
        cost=float(residual@residual/2);self.last=dict(residual=residual,jac=J,parts={'visible_runner_contour':cost},image_cost=cost,
            metrics={'visible_runner_contours':metrics,'visible_runner_points':len(self.source_uv),'visible_runner_source_frames':len(self.rows)})
        return self.last
