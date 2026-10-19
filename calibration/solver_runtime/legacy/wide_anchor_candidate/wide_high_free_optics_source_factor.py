"""Original wide ice/ad sources with all 25 high-camera coordinates active.

The private chart is original conditioned camera11 followed by literal raw
Brown14. No coefficient is a measurement or prior. The final two rows use
the separately validated lever-reachable optical-centre support provider.
No original runtime, UV, sigma, role, or integration budget is changed.
"""
from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
import numpy as np
from scipy.sparse import csr_matrix

from projection import from_state
from z190_continuous_source_bundle import SCENE_UNIT_M
from .wide_high_source_factor import (WideHighSourceFactor, PHYSICAL_COUNT,
    WORLD_INDICES, CAMERA_NAMES, _readonly, _stats, _array_hash)
from .wide_user_ad_and_secondary_source import bind_wide_ad_world
from .full_brown_projection import (point_projection_full_brown as _point_projection,
                                   curve_vector_error_full_brown)

BROWN_NAMES=('k1','k2','p1','p2','k3','k4','k5','k6','s1','s2','s3','s4','tauX','tauY')
PRIVATE_NAMES=CAMERA_NAMES+BROWN_NAMES
SCHEMA='rinkspace.original-wide-high56218880-free-camera25-source.v1'


class WideHighFreeOpticsSourceFactor(WideHighSourceFactor):
    """Original4340 ice/ad rows plus2 real optical-centre support rows.

    entity_provider(camera25, target_pivot_m, lever_camera_m,
        support, sigma_px=4, jacobian=bool) returns LeverSupportResult with
    residual_native(2), camera_state25_jacobian(2,25),
    pivot_xyz_m_jacobian(2,3), lever_xyz_m_jacobian(2,3), report. Camera J
    first11 is conditioned to_state11; the last14 is raw Brown14. This class
    normalizes the entire native entity r/J exactly once.
    """
    private_camera_parameter_count=25
    brown_fd_step=1e-7
    source_residual_rows=4342

    def __init__(self,state,*,entity_provider=None):
        super().__init__(state)
        installation=self.secondary_installation_id
        chart=getattr(state,'installation_optics',{}).get(installation)
        if chart is None or not hasattr(chart,'mount_indices'):
            raise ValueError('Authenticated target installation mount chart is required')
        self.secondary_mount_indices=_readonly(chart.mount_indices,int)
        if not np.array_equal(self.secondary_mount_indices,np.arange(34198,34201)):
            raise ValueError('Original secondary camera-local metre mount3 slots must remain unchanged')
        contract=getattr(state,'optics_contract',{})
        if installation not in contract.get('installations',{}):
            raise ValueError('Actual target base calibration contract is required')
        base_calibration=contract['installations'][installation]['base_calibration']
        self.secondary_base_lever_scene=_readonly(base_calibration['lever_xyz'])
        if self.secondary_base_lever_scene.shape!=(3,):
            raise ValueError('Complete original target camera-local lever3 required')
        self.entity_provider=entity_provider
        self.initial_camera_state25=_readonly(np.r_[self.initial_camera_state11,self.template_camera25[11:]])
        self.private_camera_parameter_names=PRIVATE_NAMES
        self.core_active_indices=_readonly(np.r_[WORLD_INDICES,self.secondary_xyz_indices,self.secondary_mount_indices],int)
        if len(np.unique(self.core_active_indices))!=29:
            raise ValueError('Shared world23/targetXYZ3/targetlever3 must be distinct')
        self._cache=OrderedDict()
        contract=deepcopy(self.sourcecontract)
        contract.update(schema=SCHEMA,private_camera_parameter_count=25,
            private_camera_parameter_names=list(PRIVATE_NAMES),
            private_camera_chart='original conditioned to_state11 + current raw Brown14',
            initial_camera_state25=self.initial_camera_state25.tolist(),
            Brown14_order=list(BROWN_NAMES),Brown14_template_role='numerical initial estimate only; every coefficient is active',
            original_Brown14_fixed=False,all_private_camera25_parameters_active=True,
            additional_camera_or_distortion_prior_rows=0,
            current_Brown14_used_for_all_world_and_camera_derivatives=True,
            camera11_derivative='original current-Brown finite-curve analytic association and conditioned point projection',
            Brown14_derivative='central FD1e-7 of original native source residual at current Brown; original associations recomputed',
            raw_Brown14_FD_step=self.brown_fd_step,
            sensor_tilt_is_active_and_original_zero_tilt_source_path_is_exact=True,
            optical_domain='original Brown12 central chart plus continuous positive OpenCV tilt homogeneous denominator on axis-to-ray path; independent Bernstein polynomial certificate',
            cache_contract='at most2 exact world23/private25/targetXYZ/currentlever byte keys; immutable numeric arrays',
            secondary_entity_support_replaces_old_pivot_approximation=True,
            target_mount_parameter_indices=self.secondary_mount_indices.tolist(),
            target_base_lever_scene_units=self.secondary_base_lever_scene.tolist(),
            actual_target_optical_centres='P + Q.T L, all SO3 Q; world sphere radius norm(actual current camera-local lever)',
            target_pose_nuisance_is_profiled_not_a_fixed_optical_centre=True,
            target_XYZ_or_lever_used_as_hardware_measurement=False,
            source_UV_sigma_roles_cost_weights_inherited_unchanged=True,
            must_pass_fresh_complete_source_gate_before_solver=True,accuracy_accepted=False)
        contract['secondary_visible_entity']=dict(contract['secondary_visible_entity'],
            actual_fixed_xyz_indices=self.secondary_xyz_indices.tolist(),
            actual_mount_indices=self.secondary_mount_indices.tolist(),
            modelling_approximation='lever-reachable optical centre in visible body support; target SO3 is profiled',
            original_fixed_pivot_as_camera_body_location_approximation_removed=True,
            sigma_engineering_px=self.entity_support_sigma_px)
        self.sourcecontract=contract;self.source_contract=contract;self.contract=contract

    def camera(self,private25):
        cameraq=np.asarray(private25,float)
        if cameraq.shape!=(25,)or not np.isfinite(cameraq).all():
            raise ValueError('All conditioned camera11 and raw Brown14 private coordinates required')
        camera=np.asarray(from_state(cameraq[:11],self.template_camera25),float).copy()
        camera[11:]=cameraq[11:]
        if not np.isfinite(camera).all()or not np.array_equal(camera[11:],cameraq[11:]):
            raise ValueError('Actual current Brown14 must be finite and exact; no frozen tail')
        return camera

    def _inputs(self,coreq,private25):
        q=np.asarray(coreq,float);cq=np.asarray(private25,float)
        if q.shape!=(PHYSICAL_COUNT,)or not np.isfinite(q).all():
            raise ValueError('Full original physical34225 state required')
        camera=self.camera(cq);qw=q[WORLD_INDICES]
        self.geometry.validate_parameters(qw)
        target=q[self.secondary_xyz_indices]/SCENE_UNIT_M
        lever=self.secondary_base_lever_scene+q[self.secondary_mount_indices]/SCENE_UNIT_M
        key=(qw.dtype.str,qw.tobytes(),cq.dtype.str,cq.tobytes(),target.tobytes(),lever.tobytes())
        return qw,camera,target,lever,key

    def _ice_ad_residual(self,camera,worlds,ad_world):
        curve=self._curve_residual(camera,worlds,False)
        uv,_,_=_point_projection(camera,ad_world)
        observed=np.asarray([r['source_frame_uv']for r in self.ad_packet['rows']],float)
        return np.r_[curve,((uv-observed)/self.advertisement_sigma_px).ravel()]

    def _curve_residual(self,camera,worlds,with_jacobian=False):
        values=[];blocks=[]
        for world,uv,sigma in zip(worlds,self.uvs,self.sigmas):
            result=curve_vector_error_full_brown(camera,world,uv,with_jacobian)
            if with_jacobian:
                r,j=result;blocks.append(j/np.repeat(sigma,2)[:,None])
            else:r=result
            values.append(r/np.repeat(sigma,2))
        r=np.concatenate(values)
        return (r,np.vstack(blocks))if with_jacobian else r

    def _entity(self,camera,target,lever,jacobian):
        provider=self.entity_provider
        if provider is None:
            from .secondary_entity_lever_support import evaluate_secondary_entity_lever_support
            provider=evaluate_secondary_entity_lever_support
        supplied=provider(camera,target*SCENE_UNIT_M,lever*SCENE_UNIT_M,self.entity_support,
                          sigma_px=self.entity_support_sigma_px,jacobian=jacobian)
        result=dict(residual_native_px=supplied.residual_native,
            camera_jacobian=supplied.camera_state25_jacobian if jacobian else None,
            target_jacobian=supplied.pivot_xyz_m_jacobian if jacobian else None,
            lever_jacobian=supplied.lever_xyz_m_jacobian if jacobian else None,audit=supplied.report)
        raw=np.asarray(result['residual_native_px'],float)
        if raw.shape!=(2,)or not np.isfinite(raw).all():
            raise ValueError('Complete finite native-pixel entity residual2 required')
        if jacobian:
            for name,shape in [('camera_jacobian',(2,25)),('target_jacobian',(2,3)),('lever_jacobian',(2,3))]:
                a=np.asarray(result[name],float)
                if a.shape!=shape or not np.isfinite(a).all():
                    raise ValueError('Entity current-camera/target/lever derivative contract changed: '+name)
        return result

    def evaluate(self,coreq,private25,jacobian=True):
        qw,camera,target,lever,key=self._inputs(coreq,private25)
        entry=self._cache.get(key)
        if entry is not None and (not jacobian or entry['core_jacobian']is not None):
            self._cache.move_to_end(key)
            return entry['residual'],entry['core_jacobian']if jacobian else None,entry['private_jacobian']if jacobian else None
        worlds=self._curve_worlds(qw)
        ad=bind_wide_ad_world(self.ad_packet,self.geometry,qw)
        observed=np.asarray([r['source_frame_uv']for r in self.ad_packet['rows']],float)
        if jacobian:
            curve_r,curve_j11=self._curve_residual(camera,worlds,True)
            ad_uv,ad_j11,ad_depth=_point_projection(camera,ad['world'])
        else:
            curve_r=self._curve_residual(camera,worlds,False)
            ad_uv,_,ad_depth=_point_projection(camera,ad['world'])
        ad_r=((ad_uv-observed)/self.advertisement_sigma_px).ravel()
        entity=self._entity(camera,target,lever,jacobian)
        residual=_readonly(np.r_[curve_r,ad_r,np.asarray(entity['residual_native_px'])/self.entity_support_sigma_px])
        if residual.shape!=(4342,)or not np.isfinite(residual).all():
            raise ValueError('All original4342 source components must remain finite and ordered')
        core=private=None
        derivative_audit=[]
        if jacobian:
            private=np.empty((4342,25))
            private[:4340,:11]=np.vstack((curve_j11,ad_j11.reshape(52,11)/self.advertisement_sigma_px))
            h=self.brown_fd_step
            for j in range(14):
                cp,cm=camera.copy(),camera.copy();cp[11+j]+=h;cm[11+j]-=h
                rp=self._ice_ad_residual(cp,worlds,ad['world']);rm=self._ice_ad_residual(cm,worlds,ad['world'])
                private[:4340,11+j]=(rp-rm)/(2*h)
                derivative_audit.append(dict(coordinate=11+j,name=BROWN_NAMES[j],FD_step=h,
                    association_recomputed=True,maximum_normalized_residual_plus_minus_difference=float(np.max(abs(rp-rm)))))
            private[-2:]=np.asarray(entity['camera_jacobian'])/self.entity_support_sigma_px
            compact=np.zeros((4342,29))
            h=self.geometry_fd_step
            for j in range(7):
                qp,qm=qw.copy(),qw.copy();qp[j]+=h;qm[j]-=h
                compact[:4288,j]=(self._curve_residual(camera,self._curve_worlds(qp),False)-
                                  self._curve_residual(camera,self._curve_worlds(qm),False))/(2*h)
            world_projection_j=-ad_j11[:,:,:3]/100.
            compact[4288:4340,:23]=(np.einsum('nij,njk->nik',world_projection_j,ad['world_jacobian'])/
                                      self.advertisement_sigma_px).reshape(52,23)
            compact[-2:,23:26]=np.asarray(entity['target_jacobian'])/self.entity_support_sigma_px
            compact[-2:,26:29]=np.asarray(entity['lever_jacobian'])/self.entity_support_sigma_px
            rr,cc=np.nonzero(compact)
            core=csr_matrix((compact[rr,cc],(rr,self.core_active_indices[cc])),shape=(4342,PHYSICAL_COUNT))
            private=_readonly(private)
            if not np.isfinite(core.data).all()or not np.isfinite(private).all():
                raise ValueError('All actual current-Brown derivatives must remain finite')
            for a in (core.data,core.indices,core.indptr):a.flags.writeable=False
        self._cache[key]=dict(residual=residual,core_jacobian=core,private_jacobian=private,worlds=worlds,
            ad=ad,ad_uv=_readonly(ad_uv),ad_depth=_readonly(ad_depth),camera=_readonly(camera),
            entity=deepcopy(entity),Brown14_derivative_audit=derivative_audit)
        self._cache.move_to_end(key)
        while len(self._cache)>2:self._cache.popitem(last=False)
        return residual,core,private

    def guard(self,coreq,private25):
        self.evaluate(coreq,private25,False)
        return None

    def review(self,coreq,private25):
        qw,camera,target,lever,key=self._inputs(coreq,private25)
        r,_,_=self.evaluate(coreq,private25,False);entry=self._cache[key]
        groups=[];cursor=0
        for row,uv,sigma in zip(self.observations,self.uvs,self.sigmas):
            n=2*len(uv);raw=r[cursor:cursor+n]*np.repeat(sigma,2);cursor+=n
            groups.append(dict(id=row['id'],target=deepcopy(row['target']),source_role='fit',
                original_sigma_px=row['sigma_px'],literal_cost_weight=float(row.get('weight',1.)),
                **_stats(np.linalg.norm(raw.reshape(-1,2),axis=1))))
        ad_raw=r[4288:4340].reshape(26,2)*self.advertisement_sigma_px
        ads=[dict(ledger,projected_uv=projected.tolist(),raw_residual_native_px=raw.tolist(),
                  radial_error_px=float(np.linalg.norm(raw)),original_null_sigma=source['original_uncertainty'],
                  explicit_engineering_sigma_px=self.advertisement_sigma_px)
             for ledger,projected,raw,source in zip(entry['ad']['ledger'],entry['ad_uv'],ad_raw,self.ad_packet['rows'])]
        a=abs(r);cost=float(self.cost_weights@np.where(a<=2.,.5*r*r,2.*a-2.))
        return dict(schema=SCHEMA,sourcecontract=self.sourcecontract,
            whole_core_q_sha256=_array_hash(np.asarray(coreq,float)),world23_sha256=_array_hash(qw),
            private_camera25_sha256=_array_hash(np.asarray(private25,float)),
            private_camera_parameter_count=25,private_camera_state25=np.asarray(private25).tolist(),
            actual_current_Brown14=camera[11:].tolist(),initial_Brown14_is_not_a_measurement_or_prior=True,
            original_Brown14_fixed=False,camera25=camera.tolist(),
            original45_curve_groups=groups,original26_advertisement_corners=ads,
            source_group_debug=[g for g in groups if g['target'].get('kind')=='polygon_supporting_line'],
            secondary_entity_support=dict(current_fixed_xyz_m=(target*SCENE_UNIT_M).tolist(),
                current_camera_local_lever_m=(lever*SCENE_UNIT_M).tolist(),
                native_support_box_ltrb=self.entity_support['native_support_box_ltrb'],
                raw_interval_violation_px=(r[-2:]*self.entity_support_sigma_px).tolist(),
                sphere_profile=deepcopy(entry['entity'].get('audit')),
                pivot_is_not_identified_with_observed_optical_centre=True),
            residual_rows=len(r),single_replica_source_cost=cost,four_replica_source_cost=4.*cost,
            raw_sigma_normalized_residual=r.tolist(),quadratic_mask=self.quadratic.tolist(),
            scales=self.scales.tolist(),cost_weights=self.cost_weights.tolist(),
            Brown14_derivative_audit=deepcopy(entry['Brown14_derivative_audit']),
            original_UV_role_sigma_source_metadata_unchanged=True,accuracy_claim=False)

    def __getstate__(self):
        state=self.__dict__.copy();state['_cache']=OrderedDict();return state
