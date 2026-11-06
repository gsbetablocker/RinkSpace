"""Reverse finite-range matching for explicitly complete visible source runners.

Callers admit only complete, visible runner_lower_edge source polylines. Partial
or occluded outlines must stay in the forward-only structure factors. This
factor spends the supplied share of the same lower-edge feature budget; it is
neither another independent observation nor a fixed material correspondence.
"""
import numpy as np
from scipy.sparse import csr_matrix, hstack

from .absolute_motion_state import chain_exposure
from .rebuild_exposure_body import moving_projection
from .sequence_skate_state import perturb_relative
from .skate_physics import fk_state
from .spin_grasp_factors import region_terms


class CompleteRunnerExtentFactors:
    """Model-to-source extent of the two actual finite curved steel edges.

    Each admitted row explicitly represents the complete visible source lower
    runner. contour_frame_budget * feature_budget_fraction is its allocated
    share, already split with the forward factor by the caller. It is divided
    by all 162 model samples, regardless of source sample density. No shoe
    shape/attachment or per-frame endpoint/pose freedom is introduced.
    """
    def __init__(self, problem, document):
        self.problem=problem; self.body=problem.body
        self.ref=problem.reference; self.binding=problem.binding
        if getattr(problem,'optimize_shared',False):
            raise ValueError('Complete runner extent requires frozen shared shoe shape/attachment')
        if problem.size!=17+self.body.size or not hasattr(self.body.state,'shoe_exposure_chain'):
            raise ValueError('Expected the complete continuous body/relative-shoe state')
        self.rows=[r for r in document['rows'] if r.get('evidence_status')=='observed'
            and r.get('role')=='fit' and r.get('optimization_enabled',True)]
        lookup={int(v):i for i,v in enumerate(self.ref.p['pts'])}
        self.frames=np.array([lookup[int(r['pts'])]for r in self.rows],int)
        self.feet=np.array([r['foot']for r in self.rows],int)
        self.curves=np.asarray(self.binding.packet()['edges'],float)
        if self.curves.shape!=(2,81,3) or not np.isfinite(self.curves).all():
            raise ValueError('Expected both actual finite 81-sample steel edges')
        self.q=self.curves.reshape(-1,3); self.count=len(self.q)
        self.times=(self.ref.times[self.frames,None]+np.linspace(-.5,.5,5)*self.body.tau).ravel()
        self.base=self.ref.at(self.times) if self.rows else None
        self.source=[]; self.factor_rows=[]
        self.association_policy=document.get('association_policy',{});self.visible_mask=None
        keys=set()
        for row in self.rows:
            key=(int(row['pts']),row['foot'])
            if key in keys or row['foot']not in (0,1):
                raise ValueError('One complete source runner per native PTS and physical foot is required')
            keys.add(key)
            if row.get('structure_feature','runner_lower_edge')!='runner_lower_edge':
                raise ValueError('Complete extent only accepts source lower-runner polylines')
            if row.get('complete_visible_runner')is not True or row.get('visibility')in ('partial','occluded','unknown'):
                raise ValueError('Partial or occluded source runners cannot constrain full model extent')
            uv=np.asarray(row['polyline_uv'],float)
            radius=float(row['pixel_uncertainty_radius_px'])
            budget=float(row['contour_frame_budget']); fraction=float(row['feature_budget_fraction'])
            if (uv.ndim!=2 or uv.shape[1]!=2 or len(uv)<2 or not np.isfinite(uv).all()
                    or not np.isfinite([radius,budget,fraction]).all()
                    or min(radius,budget,fraction)<=0 or fraction>1
                    or np.any(np.linalg.norm(np.diff(uv,axis=0),axis=1)<=1e-12)):
                raise ValueError('Finite native source polyline, pixel band and allocated budget required')
            self.source.append(uv)
            self.factor_rows.extend([dict(region_shape_px2=np.eye(2)*radius**2,
                objective_weight=budget*fraction/self.count,
                reliability=float(row.get('reliability',1.)))for _ in range(self.count)])
        self.last=None

    def evaluate(self,x,jacobian=False):
        x=np.asarray(x,float)
        if x.shape==(self.problem.size,):
            if not np.array_equal(x[:17],self.binding.x):
                raise ValueError('Shared shoe shape/attachment is frozen')
            x=x[17:]
        if x.shape!=(self.body.size,):raise ValueError('Expected the full continuous body/shoe parameter vector')
        if not self.rows:
            return dict(residual=np.empty(0),jac=csr_matrix((0,self.problem.size))if jacobian else None,
                parts={'complete_runner_extent':0.},image_cost=0.,metrics={'complete_runner_extents':[]})
        d=self.body.state.increments(x,self.times);s=fk_state(self.ref,self.base,d)
        n=len(self.rows);pick=np.arange(n*5);feet=np.repeat(self.feet,5)
        R=s['R'][pick,feet].reshape(n,5,3,3);O=s['O'][pick,feet].reshape(n,5,3)
        nodes=O[:,:,None]+np.einsum('nlij,kj->nlki',R,self.q)
        projection=moving_projection(self.problem.camera,nodes,self.frames,jacobian=jacobian)
        if self.association_policy.get('model_visibility')=='near_visible_runner_branch'and self.visible_mask is None:
            from .source_visible_model import camera_centres
            centres=camera_centres(self.problem.camera,self.frames,[np.median(v[:,1])for v in self.source])
            local_camera=np.einsum('nji,nj->ni',R[:,2],centres-O[:,2]);sides=(local_camera[:,1]>=0).astype(int)
            self.visible_mask=np.zeros((n,self.count),bool)
            from .source_contour_contract import polyline_measure_weights
            for i,side in enumerate(sides):
                ids=side*81+np.arange(81);self.visible_mask[i,ids]=True
                measure=polyline_measure_weights(self.curves[side][:,[0,2]])
                budget=self.rows[i]['contour_frame_budget']*self.rows[i]['feature_budget_fraction']
                for k in range(self.count):self.factor_rows[i*self.count+k]['objective_weight']=float(budget*measure[k%81])if self.visible_mask[i,k]else 0.
            self.visibility_record=dict(model_side=sides.tolist(),semantics='camera-near visible lower branch; no anatomical edge identity',fixed_inner=True)
        uv=projection['uv'];closest=[];projectors=[];segments=[];alphas=[]
        for i,source in enumerate(self.source):
            A=source[:-1];V=np.diff(source,axis=0);length2=np.sum(V*V,axis=1)
            alpha=np.clip(np.sum((uv[i,:,None]-A)*V,axis=-1)/length2,0.,1.)
            candidates=A+alpha[:,:,None]*V
            choice=np.argmin(np.sum((uv[i,:,None]-candidates)**2,axis=-1),axis=1)
            each=np.arange(self.count);fraction=alpha[each,choice]
            point=candidates[each,choice];tangent=V[choice]
            D=np.broadcast_to(np.eye(2),(self.count,2,2)).copy()
            interior=(fraction>0)&(fraction<1)
            D[interior]-=(tangent[interior,:,None]*tangent[interior,None,:]
                /length2[choice[interior],None,None])
            closest.append(point);projectors.append(D);segments.append(choice);alphas.append(fraction)
        closest=np.asarray(closest);normal_projector=np.asarray(projectors).reshape(-1,2,2)
        flat=uv.reshape(-1,2);centres=closest.reshape(-1,2)
        terms=[region_terms(point,dict(row,centre_uv=centre))
            for point,centre,row in zip(flat,centres,self.factor_rows)]
        residual=np.concatenate([v[0]for v in terms]);D=np.stack([v[1]for v in terms])
        from .weighted_blocks import signature
        aa=np.asarray(alphas)
        self.association_signature=signature(np.asarray(segments),(aa>0)&(aa<1),np.linalg.norm(flat-centres,axis=-1)>np.repeat([r['pixel_uncertainty_radius_px']for r in self.rows],self.count))
        J=None
        if jacobian:
            def derivative(sa,sb,h):
                ra=sa['R'][pick,feet].reshape(n,5,3,3);rb=sb['R'][pick,feet].reshape(n,5,3,3)
                oa=sa['O'][pick,feet].reshape(n,5,3);ob=sb['O'][pick,feet].reshape(n,5,3)
                dr=(ra-rb)/(2*h);do=(oa-ob)/(2*h)
                world=do[:,:,None]+np.einsum('nlij,kj->nlki',dr,self.q)
                pixel=np.einsum('nkplc,nlkc->nkpl',projection['pixel_jac_nodes'],world).reshape(-1,2,5)
                return np.einsum('nij,njk,nkl->nil',D,normal_projector,pixel)
            h=1e-6;values=[];axes=np.asarray(getattr(self.problem,'factor_derivative_columns',self.body.state.active))
            for axis in axes:
                plus=d.copy();minus=d.copy();plus[:,axis]+=h;minus[:,axis]-=h
                values.append(derivative(fk_state(self.ref,self.base,plus),fk_state(self.ref,self.base,minus),h))
            sample=np.repeat(np.arange(n*5).reshape(n,5),self.count,axis=0)
            compact=np.stack(values,-1);local=np.zeros(compact.shape[:-1]+(69,));local[...,axes]=compact
            BJ=chain_exposure(local,d,sample_indices=sample)
            shoe=[]
            for foot in (0,1):
                for axis in range(3):
                    shoe.append(derivative(fk_state(self.ref,self.base,perturb_relative(d,foot,axis,h)),
                        fk_state(self.ref,self.base,perturb_relative(d,foot,axis,-h)),h))
            BJ+=d.owner.shoe_exposure_chain(np.stack(shoe,-1),d.parameters,d.sample_times,sample_indices=sample)
            J=hstack((csr_matrix((len(residual),17)),BJ),format='csr')
        distance=np.linalg.norm(uv-closest,axis=-1);metrics=[]
        for i,row in enumerate(self.rows):
            metrics.append(dict(id=row.get('id'),pts=int(row['pts']),foot=int(row['foot']),role=row['role'],
                source_status=row.get('source_status',row['evidence_status']),complete_visible_runner_required=True,
                source_polyline_uv=row['polyline_uv'],pixel_uncertainty_radius_px=row['pixel_uncertainty_radius_px'],
                actual_model_runner_uv=uv[i].reshape(2,81,2).tolist(),source_closest_uv=closest[i].reshape(2,81,2).tolist(),
                model_to_source_distance_px=distance[i].reshape(2,81).tolist(),
                outside_complete_source_band_px=np.maximum(distance[i]-float(row['pixel_uncertainty_radius_px']),0).reshape(2,81).tolist(),
                source_segment=np.asarray(segments[i]).reshape(2,81).tolist(),source_segment_fraction=np.asarray(alphas[i]).reshape(2,81).tolist(),
                model_samples=self.count,allocated_feature_weight=float(row['contour_frame_budget'])*float(row['feature_budget_fraction']),
                per_model_sample_weight=self.factor_rows[i*self.count]['objective_weight'],
                material_correspondence='latent_nearest_complete_source_polyline_not_fixed_material',
                same_source_as_forward_lower_edge=True,independent_observation=False,contact_truth='unmeasured'))
        cost=float(residual@residual/2)
        self.last=dict(residual=residual,jac=J,parts={'complete_runner_extent':cost},image_cost=cost,
            metrics=dict(complete_runner_extents=metrics,complete_runner_extent_source_rows=n,
                complete_runner_extent_model_samples=n*self.count,
                complete_runner_extent_budget='Caller-provided share of existing forward lower-edge budget, normalized by 162 actual model samples; no added frame weight',
                complete_runner_extent_scope='Only explicitly complete visible source lower runners; never apply to partial/occluded outlines'))
        if self.visible_mask is not None:
            self.last['metrics']['complete_runner_visible_branch']=self.visibility_record
            for i,row in enumerate(metrics):row['model_visible_mask']=self.visible_mask[i].reshape(2,81).tolist()
        return self.last
