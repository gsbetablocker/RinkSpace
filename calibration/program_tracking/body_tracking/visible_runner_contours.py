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
        self.association_policy=document.get('association_policy',{});self.fixed_choice=None;self.visibility_record=[]
        owner=[];uv=[];factor_rows=[];point_rows=[];profiles=[]
        multiplier=self.ref.grasp_relations['factor_parameters']['shoe_region_weight_multiplier']
        for i,row in enumerate(self.rows):
            points=np.asarray(row['polyline_uv'],float);radius=float(row['pixel_uncertainty_radius_px'])
            if points.ndim!=2 or points.shape[1]!=2 or len(points)<(3 if segment_geometry is None else 2) or radius<=0 or row['foot'] not in (0,1):raise ValueError('Invalid source steel contour')
            if np.any(np.linalg.norm(np.diff(points,axis=0),axis=-1)<1.):raise ValueError('Steel trace samples must be distinct native pixels')
            if self.association_policy.get('arc_length_measure'):
                from .source_contour_contract import polyline_measure_weights
                weights=polyline_measure_weights(points,closed=bool(row.get('source_annotation',{}).get('closed',False)))
            else:weights=np.full(len(points),1/len(points))
            source_profiles=row.get('source_pixel_edge_profiles',[None]*len(points))
            if len(source_profiles)!=len(points):raise ValueError('Pixel profiles must preserve source point count')
            for point,measure,profile in zip(points,weights,source_profiles):
                owner.append(self.row_groups[i]);point_rows.append(i);uv.append(point)
                factor_rows.append(dict(centre_uv=point,region_shape_px2=np.eye(2)*radius**2,
                    objective_weight=float(row.get('contour_frame_budget',.02*multiplier))*float(row.get('feature_budget_fraction',1.))*measure,reliability=float(row.get('reliability',1.))))
                if profile is not None and profile['maximum_gradient']>0 and profile['reliability']>0:
                    from scipy.interpolate import PchipInterpolator
                    share=float(row.get('source_pixel_edge_budget_share',0.));weight=factor_rows[-1]['objective_weight'];factor_rows[-1]['objective_weight']*=1-share
                    profiles.append(dict(profile=profile,curve=PchipInterpolator(profile['offset_px'],profile['positive_native_image_gradient']),weight=weight*share*profile['reliability']))
                else:profiles.append(None)
        self.owner=np.asarray(owner,int);self.point_rows=np.asarray(point_rows,int);self.source_uv=np.asarray(uv,float);self.factor_rows=factor_rows;self.pixel_profiles=profiles;self.has_pixel_profiles=any(v is not None for v in profiles);self.last=None
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
        if self.association_policy.get('freeze_inner') and self.fixed_choice is None:
            from .source_visible_model import camera_centres,visible_segments,monotone_match
            centres=camera_centres(self.problem.camera,self.frames,[np.median(self.source_uv[self.owner==g,1])for g in range(n)])
            local_camera=np.einsum('nji,nj->ni',R[:,2],centres-O[:,2]);fixed=np.empty(len(self.source_uv),int)
            for i,row in enumerate(self.rows):
                g=self.row_groups[i];ii=np.flatnonzero(self.point_rows==i)
                allowed,record=visible_segments(self.segment_geometry,local_camera[g],row['structure_feature'])
                if not len(allowed):
                    raise ValueError('No visible model boundary for source feature '+str((row['pts'],row['foot'],row['structure_feature'])))
                if row['structure_feature']in('runner_lower_edge','runner_upper_edge'):
                    cc,reverse=monotone_match(distance[np.ix_(ii,allowed)]);fixed[ii]=allowed[cc];record['source_reversed']=bool(reverse)
                else:fixed[ii]=allowed[np.argmin(distance[np.ix_(ii,allowed)],axis=1)]
                record.update(pts=int(row['pts']),foot=int(row['foot']),feature=row['structure_feature'],source_track_id=row.get('source_track_id'),selected_segments=fixed[ii].tolist())
                self.visibility_record.append(record)
            self.fixed_choice=fixed
        choice=np.argmin(distance,axis=1)if self.fixed_choice is None else self.fixed_choice
        each=np.arange(len(choice));a=starts[choice];b=ends[choice];alpha=fraction[each,choice];prediction=closest[each,choice]
        tangent=V[each,choice];interior=(alpha>0)&(alpha<1);normal_projector=np.broadcast_to(np.eye(2),(len(choice),2,2)).copy()
        normal_projector[interior]-=tangent[interior,:,None]*tangent[interior,None,:]/np.sum(tangent[interior]**2,axis=-1)[:,None,None]
        terms=[region_terms(p,row)for p,row in zip(prediction,self.factor_rows)];residual=np.concatenate([v[0]for v in terms]);D=np.stack([v[1]for v in terms]);J=None
        finite_count=len(residual);pixel_values=None;pixel_offsets=None
        if self.has_pixel_profiles:
            pixel_values=np.zeros(len(prediction));pixel_D=np.zeros((len(prediction),2));pixel_offsets=np.full(len(prediction),np.nan)
            for i,item in enumerate(self.pixel_profiles):
                if item is None:continue
                profile=item['profile'];normal=np.asarray(profile['normal']);offset=float(normal@(prediction[i]-self.source_uv[i]));pixel_offsets[i]=offset;lo,hi=profile['offset_px'][0],profile['offset_px'][-1];s=float(np.clip(offset,lo,hi));g=float(item['curve'](s));gp=float(item['curve'](s,1))if lo<offset<hi else 0.;floor=profile['noise_floor'];scale=np.sqrt(item['weight']);pixel_values[i]=scale*np.log((floor+profile['maximum_gradient'])/(floor+max(g,0.)));pixel_D[i]=-scale*gp/(floor+max(g,0.))*normal
            residual=np.c_[residual.reshape(-1,2),pixel_values].ravel();D=np.concatenate([D,pixel_D[:,None,:]],axis=1)
        from .weighted_blocks import signature
        photo_branch=None
        if self.has_pixel_profiles:
            photo_branch=[None if item is None else item['profile']['offset_px'][0]<offset<item['profile']['offset_px'][-1]for item,offset in zip(self.pixel_profiles,pixel_offsets)]
        self.association_signature=signature(choice,interior,np.linalg.norm(prediction-self.source_uv,axis=1)>np.array([np.sqrt(v['region_shape_px2'][0,0])for v in self.factor_rows]),str(photo_branch))
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
            h=1e-6;values=[];axes=np.asarray(getattr(self.problem,'factor_derivative_columns',self.body.state.active))
            for axis in axes:
                plus=d.copy();minus=d.copy();plus[:,axis]+=h;minus[:,axis]-=h
                values.append(derivative(fk_state(self.ref,self.base,plus),fk_state(self.ref,self.base,minus),h))
            compact=np.stack(values,-1);local=np.zeros(compact.shape[:-1]+(69,));local[...,axes]=compact;BJ=chain_exposure(local,d,sample_indices=sample)
            shoe=[]
            for f in (0,1):
                for axis in range(3):shoe.append(derivative(fk_state(self.ref,self.base,perturb_relative(d,f,axis,h)),fk_state(self.ref,self.base,perturb_relative(d,f,axis,-h)),h))
            BJ+=d.owner.shoe_exposure_chain(np.stack(shoe,-1),d.parameters,d.sample_times,sample_indices=sample)
            J=hstack((csr_matrix((len(residual),17)),BJ),format='csr')
        if self.has_pixel_profiles:
            order=np.r_[np.c_[np.arange(len(prediction))*3,np.arange(len(prediction))*3+1].ravel(),np.arange(len(prediction))*3+2];residual=residual[order]
            if J is not None:J=J[order].tocsr()
        metrics=[]
        for i,row in enumerate(self.rows):
            m=self.point_rows==i;radius=row['pixel_uncertainty_radius_px'];dist=np.sqrt(distance[each,choice])[m]
            metrics.append(dict(id=row.get('id',str(row['pts'])),pts=int(row['pts']),foot=int(row['foot']),role=row['role'],
                source_polyline_uv=row['polyline_uv'],model_closest_uv=prediction[m].tolist(),distance_px=dist.tolist(),
                outside_visible_band_px=np.maximum(dist-radius,0).tolist(),pixel_uncertainty_radius_px=radius,
                latent_runner_side=(a[m]//self.count).tolist(),latent_runner_u=((a[m]%self.count+alpha[m])/(self.count-1)).tolist(),
                structure_feature=row.get('structure_feature','runner_lower_edge'),source_status=row.get('source_status',row['evidence_status']),
                source_structure_feature=row.get('source_structure_feature',row.get('structure_feature','runner_lower_edge')),source_component_budget_share=row.get('source_component_budget_share'),
                contour_is_observed=True,material_correspondence='inferred_image_match_not_fixed_material',contact_truth='unmeasured'))
        cost=float(residual@residual/2);self.last=dict(residual=residual,jac=J,parts={'visible_runner_contour':cost},image_cost=cost,
            metrics={'visible_runner_contours':metrics,'visible_runner_points':len(self.source_uv),'visible_runner_source_frames':len(self.rows)})
        if self.has_pixel_profiles:
            from .weighted_blocks import block
            photo_cost=float(residual[finite_count:]@residual[finite_count:]/2);self.last['parts']=dict(visible_runner_contour=cost-photo_cost,source_pixel_edge_profile=photo_cost)
            self.last['weighted_blocks']=[block('full_source_skate_structure',residual[:finite_count],None if J is None else J[:finite_count],association=self.association_signature),block('source_pixel_edge_profile',residual[finite_count:],None if J is None else J[finite_count:],association=self.association_signature)]
            self.last['metrics']['source_pixel_edge_profiles']=[dict(id=row['id'],pts=row['pts'],foot=row['foot'],source_offsets_px=pixel_offsets[self.point_rows==i].tolist(),weighted_scalar_cost=float(np.sum(pixel_values[self.point_rows==i]**2)/2),source_profile_provenance=row.get('source_pixel_profile_provenance'),source_material_identity='unknown',uncertainty_band_unchanged=True)for i,row in enumerate(self.rows)if row.get('source_pixel_edge_profiles')]
        return self.last
