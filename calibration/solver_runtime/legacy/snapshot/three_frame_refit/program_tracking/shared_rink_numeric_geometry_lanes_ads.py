"""Opt-in q11: the unchanged shared Lane q7 plus one Ad1/14 panel q4.

The first seven coordinates are the existing metric rink deltas. The final
four are ABSOLUTE panel station/width/Z-centre/height metres. Every source
world is validated against the untouched base before either family moves.
"""
from __future__ import annotations
from contextlib import contextmanager
import numpy as np
try:
    from .shared_rink_numeric_geometry_lanes import SharedRinkNumericGeometryLanes
    from .shared_ad1_14_numeric_geometry import SharedAd1Ad14NumericGeometry,PARAMETER_NAMES as PANEL_NAMES,ROLES
except ImportError:
    from shared_rink_numeric_geometry_lanes import SharedRinkNumericGeometryLanes
    from shared_ad1_14_numeric_geometry import SharedAd1Ad14NumericGeometry,PARAMETER_NAMES as PANEL_NAMES,ROLES

VERSION='shared_rink_numeric_geometry_lanes_ads_v1'


class SharedRinkNumericGeometryLanesAds:
    parameter_count=11
    parameter_names=(*SharedRinkNumericGeometryLanes.parameter_names,*PANEL_NAMES)

    def __init__(self,base,scene_unit_m=None):
        self.rink=(SharedRinkNumericGeometryLanes(base)if scene_unit_m is None
                   else SharedRinkNumericGeometryLanes(base,scene_unit_m))
        self.base=self.rink.base
        self.scene_unit_m=self.rink.scene_unit_m
        self.panel=SharedAd1Ad14NumericGeometry(self.base,self.scene_unit_m)
        self.initial=np.r_[np.zeros(7),self.panel.initial]

    def __getattr__(self,name):return getattr(self.rink,name)

    def _q(self,q11):
        q=np.asarray(q11,float)
        if q.shape!=(11,)or not np.isfinite(q).all():
            raise ValueError('Explicit seven rink deltas and four absolute panel metres required')
        self.rink._q(q[:7]);self.panel._q(q[7:]);return q

    def validate_parameters(self,q11):
        q=self._q(q11)
        self.rink.validate_parameters(q[:7]);self.panel.validate_parameters(q[7:])
        return q.copy()

    def feature_id(self,row):
        identities=self.panel._identities(row)
        if identities is not None and any(n in(1,14)for n,_ in identities):return 'shared_ad1_14_panel'
        return self.rink.feature_id(row)

    def _world_and_jacobian(self,row,q11):
        q=self._q(q11);identities=self.panel._identities(row)
        if identities is not None and any(n in(1,14)for n,_ in identities):
            world,j=self.panel._world_and_jacobian(row,q[7:])
            return world,np.concatenate((np.zeros((len(world),3,7)),j),axis=2)
        world,j=self.rink._world_and_jacobian(row,q[:7])
        return world,np.concatenate((j,np.zeros((len(world),3,4))),axis=2)

    def rebind(self,row,q11):
        q=self._q(q11);identities=self.panel._identities(row)
        return(self.panel.rebind(row,q[7:])if identities is not None and any(n in(1,14)for n,_ in identities)
               else self.rink.rebind(row,q[:7]))

    def world_jacobian(self,row,q11):return self._world_and_jacobian(row,q11)[1]

    def midpoint_prior(self,q11):
        q=self._q(q11);r,j=self.rink.midpoint_prior(q[:7])
        return r,np.c_[j,np.zeros((len(r),4))]

    def manifest(self,q11=None):
        q=self._q(self.initial if q11 is None else q11)
        value=self.rink.manifest(q[:7]);value.update(version=VERSION,parameter_count=11,
            parameter_names=list(self.parameter_names),parameters_m=q.tolist(),
            panel_parameterization='Last four values are absolute metres, unlike the preceding seven deltas',
            shared_ad1_14_panel=self.panel.manifest(q[7:]),
            additional_prior_rows=0,source_pixels_and_roles_unchanged=True)
        return value

    def scene(self,q11):
        q=self.validate_parameters(q11)
        scene=self.panel.scene(q[7:],scene=self.rink.scene(q[:7]))
        scene[VERSION]=self.manifest(q);return scene


@contextmanager
def bind_external_ad1_14(external,panel,q4):
    """Same shared panel in the real external bundle; restore source objects."""
    original=external.ad14_rows;updated=[]
    for row in original:
        role=row['source_corner_role']
        if role not in ROLES:raise ValueError('External source physical print corner role changed')
        expected=panel.old[14].corner(panel.old[14].initial,role)*panel.scene_unit_m
        if np.max(abs(np.asarray(row['world_xyz_m'],float)-expected))>1e-7:
            raise ValueError('External original world does not match its untouched Ad14 corner')
        value=dict(row);world=panel.corner(14,role,q4)
        value['world_xyz_m']=world*panel.scene_unit_m
        if 'world_xyz_scene_units'in value:value['world_xyz_scene_units']=world.tolist()
        updated.append(value)
    external.ad14_rows=updated
    try:yield updated
    finally:external.ad14_rows=original


def evaluate_external_shared_ad1_14(external,local_q,observer_xyz_m,target_xyz_m,panel,q4,
                                  *,jacobian=True,held=False):
    """Original external rows/order/scales plus exact four shared-panel columns."""
    try:
        from .z190_fixed_target_cross_view_factor import fixed_target_projection,native_bbox_support
    except ImportError:
        from z190_fixed_target_cross_view_factor import fixed_target_projection,native_bbox_support
    with bind_external_ad1_14(external,panel,q4)as rows:
        result=external.evaluate(local_q,observer_xyz_m,target_xyz_m,jacobian=jacobian,held=held)
        if jacobian:
            columns=np.zeros((len(result.residual),4));views,_=external.unpack(local_q)
            by_id={row['id']:row for row in rows}
            for entry in result.source_row_ledger:
                row=by_id.get(entry['id'])
                if row is None:continue
                vi=external.index[row['actual_pts']]
                projection=fixed_target_projection(observer_xyz_m,row['world_xyz_m'],views[vi,:3],views[vi,3],
                    external.lever_m,external.pp,external.k1)
                factor=native_bbox_support(projection,row['native_support_bounds_xy_px'],row['source_sigma_native_px'])
                _,world_j=panel.corner(14,row['source_corner_role'],q4,with_jacobian=True)
                lo,hi=entry['residual_slice']
                if hi-lo!=2:raise ValueError('External corner residual shape changed')
                columns[lo:hi]=factor.target_xyz_m_jacobian@(world_j*panel.scene_unit_m)
            result.shared_panel_jacobian=columns
        else:result.shared_panel_jacobian=None
    return result
