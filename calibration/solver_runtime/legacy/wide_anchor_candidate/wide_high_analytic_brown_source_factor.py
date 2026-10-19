"""Independent free25 wide source: exact old values, analytic Brown14 J.

Only numerical central Brown-tail differentiation is replaced. Old first11,
world/core/entity derivatives, all source pixels/scales/roles/weights and the
actual nearest finite-curve objective remain unchanged.
"""
from __future__ import annotations
from copy import deepcopy
import numpy as np
from scipy.sparse import csr_matrix
from .wide_high_free_optics_source_factor import WideHighFreeOpticsSourceFactor,BROWN_NAMES
from .wide_high_source_factor import _readonly,PHYSICAL_COUNT
from .wide_user_ad_and_secondary_source import bind_wide_ad_world
from .full_brown_projection import point_projection_full_brown
from .current_branch_brown14_derivative import curve_current_brown14,projection_brown14


class WideHighAnalyticBrownSourceFactor(WideHighFreeOpticsSourceFactor):
    def __init__(self,state,*,entity_provider=None):
        super().__init__(state,entity_provider=entity_provider)
        contract=deepcopy(self.contract)
        contract.update(schema='rinkspace.original-wide-high56218880-free-camera25-analytic-Brown14-source.v1',
            Brown14_derivative='analytic OpenCV current Brown14 + original skew + current finite nearest-foot derivative',
            raw_Brown14_FD_step=None,
            derivative_branch='selected original current central-valid segment; every trial recomputes original validity and nearest association',
            original_h1e7_BrownFD_failure_retained=True,
            original_source_objective_UV_sigma_roles_weights_unchanged=True)
        self.sourcecontract=contract;self.source_contract=contract;self.contract=contract

    def evaluate(self,coreq,private25,jacobian=True):
        qw,camera,target,lever,key=self._inputs(coreq,private25)
        entry=self._cache.get(key)
        if entry is not None and (not jacobian or entry['core_jacobian']is not None):
            self._cache.move_to_end(key)
            return entry['residual'],entry['core_jacobian']if jacobian else None,entry['private_jacobian']if jacobian else None
        worlds=self._curve_worlds(qw);ad=bind_wide_ad_world(self.ad_packet,self.geometry,qw)
        observed=np.asarray([r['source_frame_uv']for r in self.ad_packet['rows']],float)
        if jacobian:
            # Original eleven-column algebra and values stay byte for byte.
            curve_r,curve_j11=self._curve_residual(camera,worlds,True)
            ad_uv,ad_j11,ad_depth=point_projection_full_brown(camera,ad['world'])
        else:
            curve_r=self._curve_residual(camera,worlds,False)
            ad_uv,_,ad_depth=point_projection_full_brown(camera,ad['world'])
        ad_r=((ad_uv-observed)/self.advertisement_sigma_px).ravel()
        entity=self._entity(camera,target,lever,jacobian)
        residual=_readonly(np.r_[curve_r,ad_r,np.asarray(entity['residual_native_px'])/self.entity_support_sigma_px])
        if residual.shape!=(4342,)or not np.isfinite(residual).all():raise ValueError('Original complete4342 source rows required')
        core=private=None;derivative_audit=[]
        if jacobian:
            private=np.empty((4342,25));private[:4340,:11]=np.vstack((curve_j11,ad_j11.reshape(52,11)/self.advertisement_sigma_px))
            cursor=0
            for world,uv,sigma in zip(worlds,self.uvs,self.sigmas):
                raw,j14=curve_current_brown14(camera,world,uv);count=len(raw);nscale=np.repeat(sigma,2)
                if not np.array_equal(raw/nscale,curve_r[cursor:cursor+count]):
                    raise ValueError('Analytic provider changed original curve residual')
                private[cursor:cursor+count,11:]=j14/nscale[:,None];cursor+=count
            analytic_ad_uv,_,j14,_=projection_brown14(camera,ad['world'])
            if not np.array_equal(analytic_ad_uv,ad_uv):raise ValueError('Analytic provider changed original ad residual')
            private[4288:4340,11:]=j14.reshape(52,14)/self.advertisement_sigma_px
            private[-2:]=np.asarray(entity['camera_jacobian'])/self.entity_support_sigma_px
            derivative_audit=[dict(coordinate=11+j,name=name,method='analytic current-branch Brown14',FD_step=None,
                association_recomputed=True,maximum_normalized_derivative=float(np.max(abs(private[:4340,11+j]))))for j,name in enumerate(BROWN_NAMES)]
            # The original exact world/core/entity chains are copied unchanged.
            compact=np.zeros((4342,29));h=self.geometry_fd_step
            for j in range(7):
                qp,qm=qw.copy(),qw.copy();qp[j]+=h;qm[j]-=h
                compact[:4288,j]=(self._curve_residual(camera,self._curve_worlds(qp),False)-
                    self._curve_residual(camera,self._curve_worlds(qm),False))/(2*h)
            world_projection_j=-ad_j11[:,:,:3]/100.
            compact[4288:4340,:23]=(np.einsum('nij,njk->nik',world_projection_j,ad['world_jacobian'])/self.advertisement_sigma_px).reshape(52,23)
            compact[-2:,23:26]=np.asarray(entity['target_jacobian'])/self.entity_support_sigma_px
            compact[-2:,26:29]=np.asarray(entity['lever_jacobian'])/self.entity_support_sigma_px
            rr,cc=np.nonzero(compact)
            core=csr_matrix((compact[rr,cc],(rr,self.core_active_indices[cc])),shape=(4342,PHYSICAL_COUNT))
            private=_readonly(private)
            if not np.isfinite(core.data).all()or not np.isfinite(private).all():raise ValueError('All original derivative rows must be finite')
            for a in (core.data,core.indices,core.indptr):a.flags.writeable=False
        self._cache[key]=dict(residual=residual,core_jacobian=core,private_jacobian=private,worlds=worlds,
            ad=ad,ad_uv=_readonly(ad_uv),ad_depth=_readonly(ad_depth),camera=_readonly(camera),
            entity=deepcopy(entity),Brown14_derivative_audit=derivative_audit)
        self._cache.move_to_end(key)
        while len(self._cache)>2:self._cache.popitem(last=False)
        return residual,core,private
