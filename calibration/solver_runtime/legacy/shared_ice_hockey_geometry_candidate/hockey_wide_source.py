"""Current common hockey geometry in all original wide4342 source rows.

Keeps the authenticated wide packets, literal weights, sigma and private25
camera; no source pixels, observed labels or budget are introduced. The
twenty new world coordinates differentiate the actual finite source world
through its current nearest-foot branch. Every trial recomputes the original
domain/association. This is not a frozen station or a new source operator.
"""
from __future__ import annotations

from collections import OrderedDict
import copy
from copy import deepcopy
import numpy as np
from scipy.sparse import csr_matrix
from acceleration import nearest_segments
from ice_refine import BoardIntersectionError
from wide_anchor_candidate.wide_high_analytic_brown_source_factor import WideHighAnalyticBrownSourceFactor
from wide_anchor_candidate.wide_high_free_optics_source_factor import BROWN_NAMES
from wide_anchor_candidate.wide_high_source_factor import _readonly,_array_hash
from wide_anchor_candidate.native_wide_source import world_curve
from wide_anchor_candidate.wide_user_ad_and_secondary_source import bind_wide_ad_world
from wide_anchor_candidate.full_brown_projection import point_projection_full_brown,project_central_valid_full_brown
from wide_anchor_candidate.current_branch_brown14_derivative import curve_current_brown14,projection_brown14
from wide_anchor_candidate.complete_wide_free_optics_source import CompleteWideSource
from z190_continuous_source_bundle import SCENE_UNIT_M
from .shared_ice_hockey_numeric_geometry import (SharedIceHockeyNumericGeometry,
    CIRCLE_NAMES,LINE_NAMES,CREASE_NAMES)
from .hockey_source_chart import HockeyCoordinateChart,OLD_WORLD_INDICES


def curve_current_world_columns(camera,world,observed,world_jacobian):
    """Raw native current finite-foot r/J for explicit shared world columns."""
    xyz=np.asarray(world,float);dw=np.asarray(world_jacobian,float)
    if dw.ndim!=3 or dw.shape[:2]!=(len(xyz),3) or not np.isfinite(dw).all():
        raise ValueError('Actual per-knot scene-unit world derivatives required')
    observed=np.asarray(observed,float).reshape(-1,2)
    projected,j11,depth=point_projection_full_brown(camera,xyz)
    central=project_central_valid_full_brown(camera,xyz)[-1]
    jac=np.einsum('nij,njk->nik',-j11[:,:,:3]/100.,dw)
    start,finish=projected[:-1],projected[1:]
    direction=finish-start;length2=np.sum(direction*direction,axis=1)
    mid=project_central_valid_full_brown(camera,.5*(xyz[:-1]+xyz[1:]))[-1]
    valid=(central[:-1]&central[1:]&mid&np.isfinite(start).all(axis=1)&np.isfinite(finish).all(axis=1)
        &(depth[:-1]>0)&(depth[1:]>0)&(length2>1e-4))
    if not valid.any():raise BoardIntersectionError('ice_curve_no_central_segment',np.arange(len(observed)))
    index,selected_t,clipped_t,nearest=nearest_segments(observed,start,direction,length2,valid)
    a=start[index];d=direction[index];u=observed-a;denominator=length2[index]
    ja=jac[index];jd=jac[index+1]-ja
    numerator_j=(-np.einsum('nij,ni->nj',ja,d)+np.einsum('ni,nij->nj',u,jd))
    length_j=2.*np.einsum('ni,nij->nj',d,jd)
    t_j=(numerator_j-selected_t[:,None]*length_j)/denominator[:,None]
    t_j[(selected_t<=0.)|(selected_t>=1.)]=0.
    nearest_j=ja+clipped_t[:,None,None]*jd+d[:,:,None]*t_j[:,None,:]
    return (nearest-observed).ravel(),nearest_j.reshape(2*len(observed),dw.shape[2])


class HockeyWideHighSourceFactor(WideHighAnalyticBrownSourceFactor):
    """Original wide4342 factor on declared world43/core34245/private25."""
    def __init__(self,state,*,entity_provider=None):
        if type(state.geometry) is not SharedIceHockeyNumericGeometry:
            raise ValueError('Actual installed common hockey world required')
        chart=HockeyCoordinateChart(state.geometry.new_parameter_count)
        chart.core_q(state.q)
        if (not np.array_equal(state.geometry_parameter_indices,chart.world_indices) or
            not hasattr(state,'shared_ice_hockey_contract')):
            raise ValueError('Installed explicit world43/46 and complete physical core required')
        # Reuse original immutable packet construction on the explicitly
        # recoverable old chart, then install the expanded actual provider.
        # This is initialization only, never a current evaluation truncation.
        original=copy.copy(state)
        original.q=np.asarray(state.q)[:34225].copy()
        original.geometry=state.geometry.original
        original.geometry_parameter_indices=np.asarray(OLD_WORLD_INDICES,int)
        super().__init__(original,entity_provider=entity_provider)
        self.geometry=state.geometry;self.coordinate_chart=chart
        self.physical_parameter_count=chart.core_count
        self.world_parameter_count=self.geometry.parameter_count
        self.geometry_parameter_indices=_readonly(chart.world_indices,int)
        self.core_active_indices=_readonly(np.r_[self.geometry_parameter_indices,
            self.secondary_xyz_indices,self.secondary_mount_indices],int)
        if len(np.unique(self.core_active_indices))!=self.world_parameter_count+6:
            raise ValueError('World/target XYZ/lever indices overlap')
        self._base_hockey_world_rows=[]
        for observation in self.observations:
            target=observation['target'];name=target.get('feature_id');row=None
            if name in (*CIRCLE_NAMES,*LINE_NAMES,*CREASE_NAMES):
                before=world_curve(self.geometry.base,target,samples=256,scene_unit_m=SCENE_UNIT_M)
                row=dict(id=name,world=before,original_wide_target_numeric_binding=True)
                self.geometry.world_jacobian(row,self.geometry.initial)
            elif self.geometry.include_fr_ratio_deltas and target.get('kind')=='circle_outline' and target.get('ring_role') in ('middle','inner','center'):
                before=world_curve(self.geometry.base,target,samples=256,scene_unit_m=SCENE_UNIT_M)
                row=dict(id=name+'_'+target['ring_role'],world=before,original_wide_target_numeric_binding=True)
                self.geometry.world_jacobian(row,self.geometry.initial)
            self._base_hockey_world_rows.append(row)
        self._cache=OrderedDict()
        contract=deepcopy(self.contract)
        contract.update(schema='rinkspace.original-wide-common-hockey-world-free25-source.v1',
            physical_parameter_count=chart.core_count,world_parameter_count=self.world_parameter_count,
            world_parameter_indices=list(chart.world_indices),
            actual_curve_geometry_active_local_columns=list(range(7))+list(range(23,self.world_parameter_count)),
            inherited_curve_geometry_derivative='original first7 central source FD at current complete world',
            new_hockey_geometry_derivative='analytic current world/material plus selected original finite nearest-foot derivative; trial association is recomputed',
            advertisement_geometry_derivative='actual original print geometry rebind/world_jacobian43/46 then current full Brown14 chain',
            explicit_common_geometry_coordinate_contract=chart.manifest(),
            current_curve_target_binding_count=sum(r is not None for r in self._base_hockey_world_rows),
            original_wide_source_pixels_sigma_roles_weights_domain_and_sampling_unchanged=True,
            new_source_or_nominal_measurement_prior_rows=0,
            rink_60x30_boundary_reference_or_transport_installed=False,
            cache_contract='at most2 exact complete-world/private25/targetXYZ/lever byte keys; immutable numeric arrays',
            must_pass_fresh_complete_source_gate_before_solver=True)
        self.sourcecontract=contract;self.source_contract=contract;self.contract=contract

    def _inputs(self,coreq,private25):
        q=self.coordinate_chart.core_q(coreq);cq=np.asarray(private25,float)
        camera=self.camera(cq);qw=q[self.geometry_parameter_indices]
        self.geometry.validate_parameters(qw)
        target=q[self.secondary_xyz_indices]/SCENE_UNIT_M
        lever=self.secondary_base_lever_scene+q[self.secondary_mount_indices]/SCENE_UNIT_M
        key=(qw.dtype.str,qw.tobytes(),cq.dtype.str,cq.tobytes(),target.tobytes(),lever.tobytes())
        return qw,camera,target,lever,key

    def _curve_worlds(self,qw):
        scene=self.geometry.source_scene(qw)
        return tuple(_readonly(world_curve(scene,row['target'],samples=256,scene_unit_m=SCENE_UNIT_M))
                     for row in self.observations)

    def evaluate(self,coreq,private25,jacobian=True):
        qw,camera,target,lever,key=self._inputs(coreq,private25)
        entry=self._cache.get(key)
        if entry is not None and (not jacobian or entry['core_jacobian'] is not None):
            self._cache.move_to_end(key)
            return entry['residual'],entry['core_jacobian'] if jacobian else None,entry['private_jacobian'] if jacobian else None
        worlds=self._curve_worlds(qw);ad=bind_wide_ad_world(self.ad_packet,self.geometry,qw)
        observed=np.asarray([r['source_frame_uv'] for r in self.ad_packet['rows']],float)
        if jacobian:
            curve_r,curve_j11=self._curve_residual(camera,worlds,True)
            ad_uv,ad_j11,ad_depth=point_projection_full_brown(camera,ad['world'])
        else:
            curve_r=self._curve_residual(camera,worlds,False)
            ad_uv,_,ad_depth=point_projection_full_brown(camera,ad['world'])
        entity=self._entity(camera,target,lever,jacobian)
        residual=_readonly(np.r_[curve_r,((ad_uv-observed)/self.advertisement_sigma_px).ravel(),
            np.asarray(entity['residual_native_px'])/self.entity_support_sigma_px])
        if residual.shape!=(4342,) or not np.isfinite(residual).all():
            raise ValueError('All original4342 source components must remain finite and ordered')
        core=private=None;derivative_audit=[]
        if jacobian:
            private=np.empty((4342,25));private[:4340,:11]=np.vstack((curve_j11,ad_j11.reshape(52,11)/self.advertisement_sigma_px))
            compact=np.zeros((4342,self.world_parameter_count+6));cursor=0
            for ordinal,(world,uv,sigma) in enumerate(zip(worlds,self.uvs,self.sigmas)):
                raw,j14=curve_current_brown14(camera,world,uv);count=len(raw);scales=np.repeat(sigma,2)
                if not np.array_equal(raw/scales,curve_r[cursor:cursor+count]):
                    raise ValueError('Current analytic Brown provider changed original curve residual')
                private[cursor:cursor+count,11:]=j14/scales[:,None]
                source=self._base_hockey_world_rows[ordinal]
                if source is not None:
                    dw=self.geometry.world_jacobian(source,qw)[:,:,23:]
                    if dw.shape[:2]!=(len(world),3):raise ValueError('Original finite curve sampling changed')
                    checked,jworld=curve_current_world_columns(camera,world,uv,dw)
                    if not np.array_equal(checked,raw):raise ValueError('New world derivative changed original source residual')
                    compact[cursor:cursor+count,23:self.world_parameter_count]=jworld/scales[:,None]
                cursor+=count
            analytic_uv,_,j14,_=projection_brown14(camera,ad['world'])
            if not np.array_equal(analytic_uv,ad_uv):raise ValueError('Current analytic Brown provider changed original print residual')
            private[4288:4340,11:]=j14.reshape(52,14)/self.advertisement_sigma_px
            private[-2:]=np.asarray(entity['camera_jacobian'])/self.entity_support_sigma_px
            h=self.geometry_fd_step
            for col in range(7):
                p,m=qw.copy(),qw.copy();p[col]+=h;m[col]-=h
                compact[:4288,col]=(self._curve_residual(camera,self._curve_worlds(p),False)-
                                   self._curve_residual(camera,self._curve_worlds(m),False))/(2*h)
            compact[4288:4340,:self.world_parameter_count]=(np.einsum('nij,njk->nik',
                -ad_j11[:,:,:3]/100.,ad['world_jacobian'])/self.advertisement_sigma_px).reshape(52,self.world_parameter_count)
            n=self.world_parameter_count
            compact[-2:,n:n+3]=np.asarray(entity['target_jacobian'])/self.entity_support_sigma_px
            compact[-2:,n+3:n+6]=np.asarray(entity['lever_jacobian'])/self.entity_support_sigma_px
            rr,cc=np.nonzero(compact)
            core=csr_matrix((compact[rr,cc],(rr,self.core_active_indices[cc])),shape=(4342,self.physical_parameter_count))
            private=_readonly(private)
            if not np.isfinite(core.data).all() or not np.isfinite(private).all():
                raise ValueError('All actual complete world/private derivatives must remain finite')
            for a in (core.data,core.indices,core.indptr):a.flags.writeable=False
            derivative_audit=[dict(coordinate=11+j,name=name,method='analytic current-branch Brown14',FD_step=None,
                association_recomputed=True,maximum_normalized_derivative=float(np.max(abs(private[:4340,11+j]))))
                for j,name in enumerate(BROWN_NAMES)]
        self._cache[key]=dict(residual=residual,core_jacobian=core,private_jacobian=private,worlds=worlds,
            ad=ad,ad_uv=_readonly(ad_uv),ad_depth=_readonly(ad_depth),camera=_readonly(camera),
            entity=deepcopy(entity),Brown14_derivative_audit=derivative_audit)
        self._cache.move_to_end(key)
        while len(self._cache)>2:self._cache.popitem(last=False)
        return residual,core,private

    def review(self,coreq,private25):
        result=super().review(coreq,private25)
        result.pop('world23_sha256',None)
        result.update(world_parameter_count=self.world_parameter_count,
            actual_complete_world_sha256=_array_hash(np.asarray(coreq,float)[self.geometry_parameter_indices]),
            world_parameter_indices=self.geometry_parameter_indices.tolist(),
            all_original_world_targets_are_current_common_geometry=True)
        return result


class CompleteHockeyWideSource(CompleteWideSource):
    """Original exact fourfold source budget on the declared expanded shape."""
    def __init__(self,base,factor,multiplier=4):
        if not isinstance(factor,HockeyWideHighSourceFactor):
            raise ValueError('Actual common-hockey wide factor required')
        chart=factor.coordinate_chart
        if base.core_count!=chart.core_count or base.count!=chart.base_count:
            raise ValueError('Original three statistical bias rows must follow the actual complete hockey core')
        super().__init__(base,factor,multiplier)
        if self.count!=chart.full_count or not np.array_equal(self.camera_indices,chart.high_indices):
            raise ValueError('Complete declared hockey/bias/high25 column ownership changed')
        self.contract=dict(self.contract,schema='rinkspace.complete-programme-common-hockey-wide-free25-source.v1',
            coordinate_chart=chart.manifest(),world_parameter_count=factor.world_parameter_count,
            original_source_rows_counts_roles_profiles_loss_and_budget_preserved=True,
            new_shared_geometry_from_original_finite_material_not_new_UV=True,
            nominal_60x30_boundary_or_standard_reference_not_installed=True)
