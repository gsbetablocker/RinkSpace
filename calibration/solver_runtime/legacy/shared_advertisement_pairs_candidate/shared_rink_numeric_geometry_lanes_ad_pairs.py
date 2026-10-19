"""Uninstalled q23 world chart: original q11 plus three shared ad pairs."""
from __future__ import annotations
import numpy as np
from shared_rink_numeric_geometry_lanes_ads import SharedRinkNumericGeometryLanesAds
from .shared_advertisement_pairs_numeric_geometry import SharedAdvertisementPairsNumericGeometry,DEFAULT_PAIRS

VERSION='shared_rink_numeric_geometry_lanes_ad_pairs_v1'


def _restore_combined(cls,base,scene_unit_m,pairs):
    # The original wrappers have recursive __getattr__ during bare unpickle.
    # Reconstruct them through their original constructors rather than
    # changing any already verified snapshot class or global pickle policy.
    return cls(base,scene_unit_m,pairs=pairs)


class SharedRinkNumericGeometryLanesAdPairs:
    """Original seven rink deltas/Ad1-14 absolute q4, then pair absolute q12.

    Selected new physical pairs take feature-dispatch precedence, while a
    mixed corner row still receives both its original Ad1/14 and new-pair
    displacements. The source evaluator and camera chain remain external.
    """
    parameter_count=23

    def __init__(self,base,scene_unit_m=None,*,pairs=DEFAULT_PAIRS):
        self.original=SharedRinkNumericGeometryLanesAds(base,scene_unit_m)
        self.rink=self.original.rink
        self.panel=self.original.panel
        self.base=self.original.base
        self.scene_unit_m=self.original.scene_unit_m
        self.pair_panels=SharedAdvertisementPairsNumericGeometry(self.base,self.scene_unit_m,pairs=pairs)
        self.parameter_count=self.original.parameter_count+self.pair_panels.parameter_count
        self.parameter_names=(*self.original.parameter_names,*self.pair_panels.parameter_names)
        self.initial=np.r_[self.original.initial,self.pair_panels.initial]

    def __getattr__(self,name):
        # Safe during ordinary pickle reconstruction, before state is restored.
        old=self.__dict__.get('original')
        if old is None:raise AttributeError(name)
        return getattr(old,name)

    def __reduce__(self):
        return _restore_combined,(type(self),self.base,self.scene_unit_m,self.pair_panels.pairs)

    def _q(self,q):
        q=np.asarray(q,float)
        if q.shape!=(self.parameter_count,)or not np.isfinite(q).all():
            raise ValueError('Original q11 plus full absolute paired-panel chart required')
        self.original._q(q[:11]);self.pair_panels._q(q[11:])
        return q

    def validate_parameters(self,q):
        q=self._q(q)
        self.original.validate_parameters(q[:11]);self.pair_panels.validate_parameters(q[11:])
        return q.copy()

    def feature_id(self,row):
        ids=self.pair_panels.identities(row)
        if ids is not None and any(n in self.pair_panels.slot for n,_ in ids):
            return 'shared_advertisement_pairs_panel'
        return self.original.feature_id(row)

    def _world_and_jacobian(self,row,q):
        q=self._q(q)
        # Validate both physical families against the original source row.
        # Do not let an intermediate moved world requalify source evidence.
        world,j_old=self.original._world_and_jacobian(row,q[:11])
        ids=self.pair_panels.identities(row)
        if ids is None or all(n not in self.pair_panels.slot for n,_ in ids):
            return world,np.concatenate((j_old,np.zeros((len(world),3,self.pair_panels.parameter_count))),axis=2)
        _,j_pairs=self.pair_panels._world_and_jacobian(row,q[11:])
        moved=self.original.rebind(row,q[:11])
        final=self.pair_panels.rebind(moved,q[11:])
        return np.asarray(final['world'],float),np.concatenate((j_old,j_pairs),axis=2)

    def rebind(self,row,q):
        q=self._q(q)
        # Untouched families retain the same original callable's world
        # generation, including its finite Ad1/14 representation.
        moved=self.original.rebind(row,q[:11])
        return self.pair_panels.rebind(moved,q[11:])

    def world_jacobian(self,row,q):
        return self._world_and_jacobian(row,q)[1]

    def project(self,camera,row,q,*,with_jacobian=True):
        from projection import project_central_valid
        world,dw=self._world_and_jacobian(row,q)
        if not with_jacobian:return project_central_valid(camera,world)
        uv,depth,camera_jac,central=project_central_valid(camera,world,with_jacobian=True)
        jc=camera_jac.reshape(len(world),2,-1)
        return uv,depth,np.einsum('nij,njk->nik',-jc[:,:,:3]/100.,dw),central

    def midpoint_prior(self,q):
        q=self._q(q)
        r,j=self.original.midpoint_prior(q[:11])
        return r,np.c_[j,np.zeros((len(r),self.pair_panels.parameter_count))]

    def manifest(self,q=None):
        q=self._q(self.initial if q is None else q)
        value=self.original.manifest(q[:11])
        value.update(version=VERSION,parameter_count=self.parameter_count,
            parameter_names=list(self.parameter_names),parameters_m=q.tolist(),
            original_q11_chart_unchanged=True,
            additional_pair_geometry_parameter_indices=list(range(11,self.parameter_count)),
            shared_advertisement_pairs_panel=self.pair_panels.manifest(q[11:]),
            additional_prior_rows=0,source_pixels_and_roles_unchanged=True,
            source_dispatch='Original q11 then explicit new pairs; both families act on mixed corner rows',
            new_finite_source_polyline_sampling_unchanged=True)
        return value

    def scene(self,q):
        q=self.validate_parameters(q)
        scene=self.pair_panels.scene(q[11:],scene=self.original.scene(q[:11]))
        scene[VERSION]=self.manifest(q)
        return scene
