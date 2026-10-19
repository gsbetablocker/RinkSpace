"""Independent original wide56218880 factor on the explicit physical34225.

Returns original-sigma normalized residuals, never a robust transform or a
family multiplier. The outer source owns component Huber2 and four explicit
objective replicas. Full Brown14 is the inherited estimated camera template.
"""
from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
import hashlib
import numpy as np
from scipy.sparse import csr_matrix

from projection import vector_from_scene, to_state, from_state, project_central_valid
from ice_refine import _curve_vector_error
from z190_continuous_source_bundle import SCENE_UNIT_M
from .native_wide_source import load_native_wide_source_packet, world_curve
from .wide_user_ad_and_secondary_source import (load_wide_user_ad_source,
    bind_wide_ad_world, load_secondary_entity_support, secondary_support_residual)

PHYSICAL_COUNT=34225
WORLD_INDICES=np.r_[np.array([0,1,2,3,4,34189,34190,34191,34192,34193,34194]),np.arange(34213,34225)]
CAMERA_NAMES=('C_x_div100','C_y_div100','C_z_div100','R_x','R_y','R_z',
    'log_fx','log_fy','PPx_div1000','PPy_div1000','skew_div1000')
SCHEMA='rinkspace.original-wide-high56218880-current-paired-world-factor.v1'


def _readonly(value,dtype=float):
    a=np.asarray(value,dtype=dtype).copy();a.flags.writeable=False;return a


def _array_hash(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _stats(value):
    a=np.asarray(value,float)
    return dict(count=len(a),rms_px=float(np.sqrt(np.mean(a*a))),
        median_px=float(np.median(a)),p95_px=float(np.percentile(a,95)),max_px=float(np.max(a)))


def _point_projection(camera,world):
    uv,depth,jac,central=project_central_valid(camera,np.asarray(world,float),with_jacobian=True)
    if (not np.all(central) or np.any(depth<=0) or not np.isfinite(uv).all()
            or not np.isfinite(jac).all()):
        raise ValueError('Original wide source point leaves positive-depth central Brown domain')
    return uv,np.asarray(jac).reshape(len(uv),2,11),depth


class WideHighSourceFactor:
    """45 original curves, 26 user print corners, one visible-body interval.

    The support constrains the existing secondary fixed XYZ through a declared
    engineering approximation. It neither observes an exact optical centre
    nor changes any installation's nonzero moving-centre lever.
    """
    physical_parameter_count=PHYSICAL_COUNT
    geometry_fd_step=1e-6
    curve_groups=45
    curve_source_points=2144
    curve_residual_rows=4288
    source_residual_rows=4342
    advertisement_sigma_px=4.
    entity_support_sigma_px=4.

    def __init__(self,state):
        if (np.asarray(state.q).shape!=(PHYSICAL_COUNT,) or
            getattr(state.geometry,'parameter_count',None)!=23 or
            not np.array_equal(np.asarray(state.geometry_parameter_indices),WORLD_INDICES)):
            raise ValueError('Authenticated whole physical34225 and explicit world23 graph required')
        self.geometry=state.geometry
        self.geometry_parameter_indices=_readonly(WORLD_INDICES,int)
        self.secondary_installation_id=state.ids[1]
        self.secondary_xyz_indices=_readonly(state.slots[self.secondary_installation_id],int)
        if (self.secondary_xyz_indices.shape!=(3,) or
            np.any(self.secondary_xyz_indices<0) or np.any(self.secondary_xyz_indices>=PHYSICAL_COUNT) or
            len(np.unique(np.r_[WORLD_INDICES,self.secondary_xyz_indices]))!=26):
            raise ValueError('Separate explicit original secondary fixed-XYZ3 indices required')
        self.packet=load_native_wide_source_packet()
        self.ad_packet=load_wide_user_ad_source()
        self.entity_support=load_secondary_entity_support()
        if self.entity_support['installation_id']!=self.secondary_installation_id:
            raise ValueError('Original visible entity does not identify this secondary installation')
        source_scene=self.geometry.scene(np.asarray(state.q)[WORLD_INDICES])
        frames=[r for r in source_scene['frame_cameras'] if r.get('source_pts')==56218880
                and r.get('source_video_sha256',r.get('source_sha256',self.packet.contract['source_video_sha256']))==self.packet.contract['source_video_sha256']]
        if len(frames)!=1:
            raise ValueError('Exactly one original wide56218880 observer camera template required')
        self.templateCamera25=_readonly(vector_from_scene(frames[0]))
        self.template_camera25=self.templateCamera25
        self.initial_camera_state11=_readonly(to_state(self.templateCamera25))
        self.observations=tuple(deepcopy(r) for r in self.packet.observations)
        if len(self.observations)!=45 or sum(len(r['pixels_uv'])for r in self.observations)!=2144:
            raise ValueError('Original native45/2144 curve inventory changed')
        self.uvs=tuple(_readonly(r['pixels_uv'])for r in self.observations)
        self.sigmas=tuple(_readonly(np.broadcast_to(np.asarray(r['sigma_px'],float),(len(uv),)))
            for r,uv in zip(self.observations,self.uvs))
        if any(np.any(s<=0)or not np.isfinite(s).all()for s in self.sigmas):
            raise ValueError('Every original curve sigma must remain literal and positive')
        literal_weights=[float(r.get('weight',1.))for r in self.observations]
        if any(not np.isfinite(w)or w<=0 for w in literal_weights):
            raise ValueError('Invalid literal original curve objective weight')
        experiment=self.packet.contract['original_v5_near_right_weight_experiment']
        weighted=[(r['id'],w)for r,w in zip(self.observations,literal_weights)if w!=1.]
        if weighted!=[(experiment['observation_id'],experiment['weight_multiplier'])] or weighted[0][1]!=4.:
            raise ValueError('Original sole near-right literal weight4 source contract changed')
        ad_weights=[]
        for r in self.ad_packet['rows']:
            sqrt=r['original_sqrt_weights']
            integration=np.asarray(r['original_group_weight']['point_integration_weights'],float)
            if integration.shape!=(1,) or not np.isfinite(integration).all()or integration[0]<=0:
                raise ValueError('Exact original one-corner physical group integration required')
            if sqrt is not None:
                sw=np.asarray(sqrt,float)
                if sw.size!=1 or not np.array_equal(sw.ravel()**2,integration):
                    raise ValueError('Stored sqrt weight and original corner integration disagree')
            ad_weights.append(float(integration[0]))
        by_ad={}
        for row,w in zip(self.ad_packet['rows'],ad_weights):
            by_ad[row['user_ad_number']]=by_ad.get(row['user_ad_number'],0.)+w
        if by_ad!={i:1. for i in range(1,8)}:
            raise ValueError('Keep the exact original budget1 for each of seven advertisements')
        self.cost_weights=_readonly(np.r_[np.concatenate([np.full(2*len(uv),w)
            for uv,w in zip(self.uvs,literal_weights)]),np.repeat(ad_weights,2),np.ones(2)])
        # SourceRowBlock.scales restores native residual units for guards and
        # inventory. It does NOT weight the solver cost, gradient, or normal.
        self.scales=_readonly(np.r_[np.concatenate([np.repeat(s,2)for s in self.sigmas]),
            np.full(52,self.advertisement_sigma_px),np.full(2,self.entity_support_sigma_px)])
        self.quadratic_mask=_readonly(np.zeros(self.source_residual_rows),bool)
        self.quadratic=self.quadratic_mask
        self.core_active_indices=_readonly(np.r_[WORLD_INDICES,self.secondary_xyz_indices],int)
        self._cache=OrderedDict()
        self.sourcecontract=dict(schema=SCHEMA,source_packet=self.packet.contract,
            original_source_curves=dict(groups=45,unique_uv=2144,residual_rows=4288,sampling='original_v5',
                historical_step02_resampled_pixels_not_duplicated=True,original_sigma_unchanged=True,
                literal_curve_cost_weights=[dict(id=i,weight=w)for i,w in weighted]),
            advertisements=dict(packet_schema=self.ad_packet['schema'],source_files=self.ad_packet['source_files'],
                groups=26,unique_uv=26,residual_rows=52,original_null_sigma_unchanged_in_provenance=True,
                explicit_engineering_sigma_px=self.advertisement_sigma_px,
                original_stored_sqrt_weights=[r['original_sqrt_weights']for r in self.ad_packet['rows']],
                original_point_integration_cost_scales=ad_weights,
                weight_policy='exact original integration budget as explicit cost_weights outside component Huber2; no invented stored sqrt_weights',
                Ad1_two_left_corners_only=True,Ad5_internal_rectangle_not_ad_extent=True),
            secondary_visible_entity=dict(source=self.entity_support,residual_rows=2,
                actual_fixed_xyz_indices=self.secondary_xyz_indices.tolist(),
                explicit_engineering_sigma_px=self.entity_support_sigma_px,
                modelling_approximation='existing fixed physical pivot XYZ projects within visible camera-body support; not exact optical-centre/origin evidence',
                nonzero_installation_lever_or_current_camera_function_changed=False),
            residual_rows=self.source_residual_rows,physical_parameter_count=PHYSICAL_COUNT,
            world_parameter_count=23,world_parameter_indices=WORLD_INDICES.tolist(),
            actual_curve_geometry_active_local_columns=list(range(7)),
            curve_geometry_derivative='original residual operator central FD1e-6 on current whole world23; association and station recomputed',
            advertisement_geometry_derivative='actual geometry.rebind/world_jacobian23 then general Brown14 world projection chain',
            private_camera11_names=list(CAMERA_NAMES),private_camera11_initial=self.initial_camera_state11.tolist(),
            initial_full_Brown14=self.templateCamera25[11:].tolist(),
            Brown14_template_role='inherited estimated source camera, not hardware measurement or hard prior',
            SCENE_UNIT_M=SCENE_UNIT_M,scene_scale_role='existing conditional gauge, not new measured metres',
            raw_residual_sigma_normalized=True,loss_owner='outer source component Huber2 delta2',
            family_multiplier_applied_here=False,requested_family_multiplier=4.,
            family_multiplier_policy='outer exact row replication counts=4*cost_weights; ordinary4/near16/Ad1corner2/other-ad-corner1/entity4',
            cost_weights=self.cost_weights.tolist(),
            native_row_scales=self.scales.tolist(),
            SourceRowBlock_scales_role='native residual restoration only; solver does not use scales as cost weights',
            historical_original_loss_not_claimed_identical='literal near weight and ad budgets retained as external Huber2 cost scales; engineering ad/entity sigma explicitly new',
            old_world_estimate_or_camera_used_as_numerical_prior=False,
            all_geometry_and_private_camera_parameters_remain_active=True,
            cache_contract='at most2 exact whole-world23+private11+targetXYZ byte keys; numeric immutable values only',
            independent_accuracy_claim=False)
        self.source_contract=self.sourcecontract
        self.contract=self.sourcecontract

    def camera(self,privateCamera11):
        """Exact current general-Brown14 camera for renderer/export consumers."""
        cameraq=np.asarray(privateCamera11,float)
        if cameraq.shape!=(11,)or not np.isfinite(cameraq).all():
            raise ValueError('Complete conditioned private high camera11 required')
        camera=from_state(cameraq,self.templateCamera25)
        if not np.isfinite(camera).all()or not np.array_equal(camera[11:],self.templateCamera25[11:]):
            raise ValueError('Inherited Brown14 template must remain exact and finite')
        return camera

    def _inputs(self,coreq,privateCamera11):
        q=np.asarray(coreq,float);cameraq=np.asarray(privateCamera11,float)
        if q.shape!=(PHYSICAL_COUNT,)or not np.isfinite(q).all():
            raise ValueError('Full explicit physical34225 required; never an arbitrary prefix')
        if cameraq.shape!=(11,)or not np.isfinite(cameraq).all():
            raise ValueError('Full conditioned high camera11 required')
        qw=q[WORLD_INDICES];self.geometry.validate_parameters(qw)
        target=q[self.secondary_xyz_indices]/SCENE_UNIT_M
        camera=self.camera(cameraq)
        if not np.isfinite(camera).all()or not np.array_equal(camera[11:],self.templateCamera25[11:]):
            raise ValueError('General inherited Brown14 must remain exact and finite')
        key=(qw.dtype.str,qw.tobytes(),cameraq.dtype.str,cameraq.tobytes(),target.tobytes())
        return qw,camera,target,key

    def _curve_worlds(self,qw):
        scene=self.geometry.scene(qw)
        return tuple(_readonly(world_curve(scene,row['target'],samples=256,scene_unit_m=SCENE_UNIT_M))
            for row in self.observations)

    def _curve_residual(self,camera,worlds,with_jacobian=False):
        values=[];jacobians=[]
        for world,uv,sigma in zip(worlds,self.uvs,self.sigmas):
            result=_curve_vector_error(camera,world,uv,with_jacobian)
            if with_jacobian:
                r,j=result;jacobians.append(j/np.repeat(sigma,2)[:,None])
            else:r=result
            values.append(r/np.repeat(sigma,2))
        r=np.concatenate(values)
        return (r,np.vstack(jacobians))if with_jacobian else r

    def evaluate(self,coreq,privateCamera11,jacobian=True):
        qw,camera,target,key=self._inputs(coreq,privateCamera11)
        entry=self._cache.get(key)
        if entry is not None and (not jacobian or entry['core_jacobian']is not None):
            self._cache.move_to_end(key)
            return entry['residual'],entry['core_jacobian']if jacobian else None,entry['private_jacobian']if jacobian else None
        worlds=self._curve_worlds(qw)
        curve_r,curve_camera_j=self._curve_residual(camera,worlds,True)
        ad=bind_wide_ad_world(self.ad_packet,self.geometry,qw)
        ad_uv,ad_camera_j,ad_depth=_point_projection(camera,ad['world'])
        ad_observed=np.asarray([r['source_frame_uv']for r in self.ad_packet['rows']],float)
        ad_r=((ad_uv-ad_observed)/self.advertisement_sigma_px).ravel()
        target_uv,target_camera_j,target_depth=_point_projection(camera,target[None,:])
        entity_raw,entity_active=secondary_support_residual(target_uv[0],self.entity_support)
        entity_r=entity_raw/self.entity_support_sigma_px
        residual=_readonly(np.r_[curve_r,ad_r,entity_r])
        if residual.shape!=(self.source_residual_rows,)or not np.isfinite(residual).all():
            raise ValueError('Complete finite original wide source rows required')
        private=core=None
        if jacobian:
            private=_readonly(np.vstack((curve_camera_j,ad_camera_j.reshape(52,11)/self.advertisement_sigma_px,
                entity_active@target_camera_j[0]/self.entity_support_sigma_px)))
            compact=np.zeros((self.source_residual_rows,26))
            h=self.geometry_fd_step
            for column in range(7):
                plus,minus=qw.copy(),qw.copy();plus[column]+=h;minus[column]-=h
                compact[:self.curve_residual_rows,column]=(self._curve_residual(camera,self._curve_worlds(plus))-
                    self._curve_residual(camera,self._curve_worlds(minus)))/(2*h)
            world_projection_j=-ad_camera_j[:,:,:3]/100.
            compact[self.curve_residual_rows:self.curve_residual_rows+52,:23]=(
                np.einsum('nij,njk->nik',world_projection_j,ad['world_jacobian'])/
                self.advertisement_sigma_px).reshape(52,23)
            compact[-2:,23:]=entity_active@(-target_camera_j[0,:,:3]/100.)/(SCENE_UNIT_M*self.entity_support_sigma_px)
            rr,cc=np.nonzero(compact)
            core=csr_matrix((compact[rr,cc],(rr,self.core_active_indices[cc])),shape=(self.source_residual_rows,PHYSICAL_COUNT))
            if not np.isfinite(core.data).all()or not np.isfinite(private).all():
                raise ValueError('Wide derivatives must remain finite and complete')
            for a in (core.data,core.indices,core.indptr):a.flags.writeable=False
        entry=dict(residual=residual,core_jacobian=core,private_jacobian=private,worlds=worlds,
            ad=ad,ad_uv=_readonly(ad_uv),ad_depth=_readonly(ad_depth),
            target_uv=_readonly(target_uv),target_depth=_readonly(target_depth),camera=_readonly(camera))
        self._cache[key]=entry;self._cache.move_to_end(key)
        while len(self._cache)>2:self._cache.popitem(last=False)
        return residual,core,private

    def guard(self,coreq,privateCamera11):
        """Original central-domain and current geometry checks; solver API None."""
        self.evaluate(coreq,privateCamera11,jacobian=False)
        return None

    def review(self,coreq,privateCamera11):
        qw,camera,target,key=self._inputs(coreq,privateCamera11)
        r,_,_=self.evaluate(coreq,privateCamera11,jacobian=False);entry=self._cache[key]
        groups=[];cursor=0
        for row,uv,sigma in zip(self.observations,self.uvs,self.sigmas):
            count=2*len(uv);raw=r[cursor:cursor+count]*np.repeat(sigma,2);cursor+=count
            groups.append(dict(id=row['id'],target=deepcopy(row['target']),source_role='fit',
                original_sigma_px=row['sigma_px'],literal_cost_weight=float(row.get('weight',1.)),
                **_stats(np.linalg.norm(raw.reshape(-1,2),axis=1))))
        ad_raw=r[cursor:cursor+52].reshape(26,2)*self.advertisement_sigma_px
        ads=[dict(ledger,projected_uv=projected.tolist(),raw_residual_native_px=raw.tolist(),
            radial_error_px=float(np.linalg.norm(raw)),original_null_sigma=source['original_uncertainty'],
            explicit_engineering_sigma_px=self.advertisement_sigma_px)
            for ledger,projected,raw,source in zip(entry['ad']['ledger'],entry['ad_uv'],ad_raw,self.ad_packet['rows'])]
        norm=np.abs(r);component_loss=np.where(norm<=2.,.5*r*r,2.*norm-2.)
        cost=float(self.cost_weights@component_loss)
        debug=[g for g in groups if g['target'].get('kind')=='polygon_supporting_line']
        return dict(schema=SCHEMA,sourcecontract=self.sourcecontract,
            whole_core_q_sha256=_array_hash(np.asarray(coreq,float)),world23_sha256=_array_hash(qw),
            private_camera11_sha256=_array_hash(np.asarray(privateCamera11,float)),camera25=camera.tolist(),
            original45_curve_groups=groups,original26_advertisement_corners=ads,
            source_group_debug=debug,secondary_entity_support=dict(projected_fixed_pivot_uv=entry['target_uv'][0].tolist(),
                current_fixed_xyz_m=(target*SCENE_UNIT_M).tolist(),native_support_box_ltrb=self.entity_support['native_support_box_ltrb'],
                raw_interval_violation_px=(r[-2:]*self.entity_support_sigma_px).tolist(),
                fixed_pivot_as_visible_entity_support_is_explicit_engineering_approximation=True),
            residual_rows=len(r),single_replica_source_cost=cost,four_replica_source_cost=4.*cost,
            raw_sigma_normalized_residual=r.tolist(),quadratic_mask=self.quadratic_mask.tolist(),scales=self.scales.tolist(),cost_weights=self.cost_weights.tolist(),
            original_UV_role_sigma_source_metadata_unchanged=True,accuracy_claim=False)

    def __getstate__(self):
        state=self.__dict__.copy();state['_cache']=OrderedDict();return state
