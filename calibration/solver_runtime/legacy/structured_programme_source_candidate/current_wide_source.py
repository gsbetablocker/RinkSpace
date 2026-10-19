"""All original wide sources and two user-drawn board top contours."""
from collections import OrderedDict
from copy import deepcopy
import numpy as np
from scipy.sparse import csr_matrix
from .prepared_finite_curve_world_columns import prepare_curve_current_world_columns
from .current_ice_material_scene import CurrentIceMaterialView
from .batched_absolute_geometry_fd import _feasible_world_stencil
from wide_anchor_candidate.wide_high_free_optics_source_factor import WideHighFreeOpticsSourceFactor
from wide_anchor_candidate.full_brown_projection import point_projection_full_brown
from wide_anchor_candidate.current_branch_brown14_derivative import curve_current_brown14,projection_brown14
from .paired_programme_chart import CORE_COUNT,WORLD_INDICES
from z190_continuous_source_bundle import SCENE_UNIT_M

class CurrentWideSource(WideHighFreeOpticsSourceFactor):
    def __init__(self,original_factor,state,trace_packet):
        self.__dict__.update(vars(original_factor));self._cache=OrderedDict()
        self.geometry=state.geometry.current
        chart=getattr(state,'structured_programme_coordinate_chart',getattr(state,'chart',None))
        self.physical_parameter_count=int(getattr(chart,'core_count',CORE_COUNT))
        self.qualified_scene=state.geometry.qualified_scene
        self.geometry_parameter_indices=np.asarray(getattr(state,'geometry_parameter_indices',WORLD_INDICES),int).copy()
        self.world_parameter_count=int(self.geometry.parameter_count)
        if (self.geometry_parameter_indices.shape!=(self.world_parameter_count,)
                or len(np.unique(self.geometry_parameter_indices))!=self.world_parameter_count
                or np.any(self.geometry_parameter_indices<0)
                or np.any(self.geometry_parameter_indices>=self.physical_parameter_count)):
            raise ValueError('Complete current world parameter map required')
        self.core_active_indices=np.r_[self.geometry_parameter_indices,self.secondary_xyz_indices,self.secondary_mount_indices]
        self.trace_packet=trace_packet
        self.trace_uv=[np.asarray(r['observed_centres_source_px'],float)for r in trace_packet['contours']]
        self.trace_sigma=2.
        # The complete stroke is one correlated observation.  Its average
        # squared distance contributes once; pixels are not independent data.
        self.trace_scales=np.concatenate([np.full(2*len(uv),self.trace_sigma*np.sqrt(len(uv)))for uv in self.trace_uv])
        self.quadratic=np.r_[original_factor.quadratic,np.ones(len(self.trace_scales),bool)]
        self.scales=np.r_[original_factor.scales,self.trace_scales]
        self.cost_weights=np.r_[original_factor.cost_weights,np.ones(len(self.trace_scales))]
        self.contract=dict(original_factor.contract,schema='rinkspace.current-named66-original-wide-and-user-top-edges.v1',
            world_parameter_count=self.world_parameter_count,physical_parameter_count=self.physical_parameter_count,
            world_parameter_indices=self.geometry_parameter_indices.tolist(),
            source_pts=56218880,user_contour_ids=[r['id']for r in trace_packet['contours']],
            user_contour_source_uv_counts=[len(uv)for uv in self.trace_uv],
            user_contour_role='visible board top edge; original native pixels',
            contour_objective='per stroke mean squared distance divided by explicit engineering sigma2px squared',
            user_sigma_is_engineering_weight_not_measured_uncertainty=True,stroke_pixels_not_independent_measurements=True,
            original45_26_rows_UV_sigma_roles_weights_unchanged=True)
    def _inputs(self,core,private):
        q=np.asarray(core,float)
        if q.shape!=(self.physical_parameter_count,):raise ValueError('Complete current physical chart required')
        private=np.asarray(private,float);world=q[self.geometry_parameter_indices];camera=self.camera(private)
        self.geometry.validate_parameters(world)
        target=q[self.secondary_xyz_indices]/SCENE_UNIT_M;lever=self.secondary_base_lever_scene+q[self.secondary_mount_indices]/SCENE_UNIT_M
        return world,camera,target,lever,(world.tobytes(),private.tobytes(),target.tobytes(),lever.tobytes())
    def _curve_worlds(self,world):
        # Wide source curves use current ice and course geometry, while the
        # 26 named corners already use the exact print material interface.
            # Rebuilding every unrelated advertisement for each world-FD
        # states is unnecessary. Preserve the original producers and source
        # qualification; no observation, numerical curve or guard changes.
        self.geometry.validate_parameters(world)
        view=self.__dict__.get('_wide_ice_material_view')
        if view is None:
            view=CurrentIceMaterialView(self.geometry)
            self._wide_ice_material_view=view
        course,cq,_=self.geometry.print_course_binding(world)
        scene=course.scene(cq,rebuild_grid=False)
        scene['ice_features']=list(view.features(world).values())
        extent_state=getattr(self.geometry,'ice_extent_state',None)
        if callable(extent_state):scene.update(extent_state(world))
        scene=self.qualified_scene(scene)
        getter=getattr(self.geometry,'one_scene_source_world_bundle',None)
        curves=getter(world,scene,self.observations)if callable(getter)else[self.geometry.semantic_world(r['target'],world)for r in self.observations]
        top=np.asarray(scene['board_surface']['top_edge_xyz'],float)
        if not np.array_equal(top[0],top[-1]):top=np.vstack((top,top[0]))
        ads=np.asarray([self.geometry.corner(r['user_ad_number'],r['role'],world)for r in self.ad_packet['rows']])
        return tuple(np.asarray(w,float)for w in curves),top,ads
    def evaluate(self,core,private,jacobian=True):
        world,camera,target,lever,key=self._inputs(core,private)
        cached=self._cache.get(key)
        if cached is not None and (not jacobian or cached[1]is not None):return cached
        curves,top,ads=self._curve_worlds(world);observed=np.asarray([r['source_frame_uv']for r in self.ad_packet['rows']])
        curve_r,curveJ=self._curve_residual(camera,curves,True)if jacobian else(self._curve_residual(camera,curves,False),None)
        aduv,adJ,_=point_projection_full_brown(camera,ads);entity=self._entity(camera,target,lever,jacobian)
        trace=[]
        for uv in self.trace_uv:trace.append(curve_current_brown14(camera,top,uv)[0])
        trace_r=np.concatenate(trace)/self.trace_scales
        residual=np.r_[curve_r,((aduv-observed)/self.advertisement_sigma_px).ravel(),entity['residual_native_px']/self.entity_support_sigma_px,trace_r]
        n=len(residual);coreJ=privateJ=None
        if jacobian:
            privateJ=np.zeros((n,25));privateJ[:4288,:11]=curveJ;privateJ[4288:4340,:11]=adJ.reshape(52,11)/self.advertisement_sigma_px
            cursor=0
            for w,uv,sigma in zip(curves,self.uvs,self.sigmas):
                raw,J14=curve_current_brown14(camera,w,uv);count=len(raw)
                privateJ[cursor:cursor+count,11:]=J14/np.repeat(sigma,2)[:,None];cursor+=count
            _,_,ad14,_=projection_brown14(camera,ads);privateJ[4288:4340,11:]=ad14.reshape(52,14)/self.advertisement_sigma_px
            privateJ[4340:4342]=entity['camera_jacobian']/self.entity_support_sigma_px
            # Current nearest-curve branch supplies all camera derivatives;
            # use the same source value with actual per-knot world derivatives.
            from wide_anchor_candidate.full_brown_projection import curve_vector_error_full_brown
            cursor=4342
            for uv in self.trace_uv:
                raw,J11=curve_vector_error_full_brown(camera,top,uv,with_jacobian=True)
                _,J14=curve_current_brown14(camera,top,uv);count=len(raw);sc=self.trace_scales[cursor-4342:cursor-4342+count]
                privateJ[cursor:cursor+count,:11]=J11/sc[:,None];privateJ[cursor:cursor+count,11:]=J14/sc[:,None];cursor+=count
            ng=self.world_parameter_count
            compact=np.zeros((n,ng+6));h=1e-6
            preparations=[prepare_curve_current_world_columns(camera,w,uv)for w,uv in zip(curves,self.uvs)]
            trace_preparations=[prepare_curve_current_world_columns(camera,top,uv)for uv in self.trace_uv]
            # Scene values are differentiated in one batch per physical
            # column.  The projection/nearest-foot derivative stays analytic.
            domain_checks=[]
            for k in range(ng):
                points,coefficients,report=_feasible_world_stencil(self.geometry,world,k,h)
                values=[self._curve_worlds(point)for point in points]
                if report['method']=='central':
                    (cp,tp,ap),(cm,tm,am)=values;step=report['step']
                    curve_dw=[(wp-wm)/(2*step)for wp,wm in zip(cp,cm)]
                    top_dw=(tp-tm)/(2*step);ad_dw=(ap-am)/(2*step)
                else:
                    (c0,t0,a0),(c1,t1,a1),(c2,t2,a2)=values
                    # Differences from the nominal value keep inactive world
                    # columns exactly zero at a legal physical boundary.
                    curve_dw=[coefficients[1]*(w1-w0)+coefficients[2]*(w2-w0)
                              for w0,w1,w2 in zip(c0,c1,c2)]
                    top_dw=coefficients[1]*(t1-t0)+coefficients[2]*(t2-t0)
                    ad_dw=coefficients[1]*(a1-a0)+coefficients[2]*(a2-a0)
                if report['method']!='central' or report['step']!=h:domain_checks.append(report)
                cursor=0
                for prepared,sigma,dw in zip(preparations,self.sigmas,curve_dw):
                    raw,jw=prepared.apply(dw[:,:,None]);count=len(raw)
                    compact[cursor:cursor+count,k]=jw[:,0]/np.repeat(sigma,2);cursor+=count
                compact[4288:4340,k]=(np.einsum('nij,nj->ni',-adJ[:,:,:3]/100.,ad_dw)/self.advertisement_sigma_px).ravel()
                cursor=4342
                for prepared in trace_preparations:
                    raw,jw=prepared.apply(top_dw[:,:,None]);count=len(raw)
                    compact[cursor:cursor+count,k]=jw[:,0]/self.trace_scales[cursor-4342:cursor-4342+count];cursor+=count
            self._last_world_fd_physical_domain_checks=domain_checks
            compact[4340:4342,ng:ng+3]=entity['target_jacobian']/self.entity_support_sigma_px
            compact[4340:4342,ng+3:ng+6]=entity['lever_jacobian']/self.entity_support_sigma_px
            rr,cc=np.nonzero(compact);coreJ=csr_matrix((compact[rr,cc],(rr,self.core_active_indices[cc])),shape=(n,self.physical_parameter_count))
        if not np.isfinite(residual).all():raise ValueError('Finite complete current wide source required')
        result=(residual,coreJ,privateJ);self._cache[key]=result
        while len(self._cache)>2:self._cache.popitem(last=False)
        return result
    def review(self,core,private):
        r,_,_=self.evaluate(core,private,False);world,camera,*_=self._inputs(core,private)
        def stats(raw):
            distances=np.linalg.norm(np.asarray(raw,float).reshape(-1,2),axis=1)
            return dict(native_UV_count=len(distances),mean_native_px=float(distances.mean()),
                median_native_px=float(np.median(distances)),rms_native_px=float(np.sqrt(np.mean(distances**2))),
                p95_native_px=float(np.quantile(distances,.95)),max_native_px=float(distances.max()))
        groups=[];cursor=0
        for row,uv,sigma in zip(self.observations,self.uvs,self.sigmas):
            count=2*len(uv);raw=r[cursor:cursor+count]*np.repeat(sigma,2)
            groups.append(dict(id=row.get('id',str(row.get('target'))),source_kind='original_wide_curve',**stats(raw)));cursor+=count
        groups.append(dict(id='original_26_advertisement_points',source_kind='original_wide_advertisements',
            **stats(r[4288:4340]*self.advertisement_sigma_px)))
        groups.append(dict(id='same_installation_entity_support',source_kind='original_wide_entity_support',
            **stats(r[4340:4342]*self.entity_support_sigma_px)))
        cursor=4342
        for row,uv in zip(self.trace_packet['contours'],self.trace_uv):
            count=2*len(uv);raw=r[cursor:cursor+count]*self.trace_scales[cursor-4342:cursor-4342+count]
            groups.append(dict(id=row['id'],source_kind='user_board_top_correlated_stroke',**stats(raw)));cursor+=count
        return dict(schema=self.contract['schema'],source_contract=self.contract,camera25=camera.tolist(),
            current_world66=world[:66].tolist(),current_world_parameters=world.tolist(),
            world_parameter_count=self.world_parameter_count,source_normalized_residual=r.tolist(),
            raw_original_wide_curve_native_px=(r[:4288]*self.scales[:4288]).tolist(),
            raw_source_native_pixel_error_groups=groups,
            user_contours=[dict(id=record['id'],uv_count=len(uv))for record,uv in zip(self.trace_packet['contours'],self.trace_uv)],accuracy_accepted=False)
