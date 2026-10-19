"""Opt-in fixed installations with three jointly estimated pivot coordinates.

The existing source graph and four-slot motion kernel stay intact. Shared
slots are [Px/100, Py/100, Pz/100, zero padding], followed by every actual
exposure's Rodrigues3/logf. One physical pivot is shared across its cuts;
the latest nonzero camera-local lever, optics and source roles remain fixed.
Approximate user positions are initial estimates, never per-frame motion.
"""
from __future__ import annotations
from dataclasses import replace
import cv2
import numpy as np
from z190_moving_pivot_source_bundle import MovingPivotSourceBundle


class FixedLeverXYZSourceBundle(MovingPivotSourceBundle):
    @classmethod
    def from_bundle(cls,original):
        result=cls(original,original.calibration,original.motion_kernel,
            reference_pts=original.reference_pts,
            initial_reference_optical_height_scene=original.initial_reference_optical_height_scene,
            finite_planes=original.finite_planes,finite_records=original.finite_records,
            scene=original.scene,absolute_evaluator=original.absolute_evaluator)
        result.robust_phase=original.robust_phase
        result.profile_weights=original.profile_weights
        return result

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.shared_lower=np.array([-np.inf,-np.inf,0.,0.])
        self.shared_upper=np.array([np.inf,np.inf,np.inf,0.])
        self.optimized_local_columns=np.r_[np.arange(3),np.arange(4,4+4*len(self.stamps))]

    def initial_shared(self):
        return np.r_[np.asarray(self.calibration.pivot_xyz)/100.,0.]

    def current_calibration(self,q):
        self._check_shared(q)
        return replace(self.calibration,pivot_xyz=tuple(100.*np.asarray(q[:3])))

    def shared_camera_jacobian(self,shared):
        return np.c_[shared[:15,:3],np.zeros(15)]

    def _check_shared(self,q):
        if not np.isfinite(q[:4]).all()or q[2]<=0 or q[3]!=0.:
            raise ValueError('One finite positive-height fixed pivot and zero padding required')

    def cache(self,q,jacobian):
        self._check_shared(q)
        vv=np.asarray(q[4:]).reshape(-1,4)
        raw=[cv2.Rodrigues(v[:3])for v in vv]
        rr=np.asarray([r[0]for r in raw])
        dr=np.asarray([r[1].reshape(3,3,3)for r in raw])
        lever=np.asarray(self.calibration.lever_xyz)
        centres=100.*q[:3]+np.einsum('vij,i->vj',rr,lever)
        ds=np.zeros((len(vv),3,4));ds[:,:,:3]=100.*np.eye(3)
        dv=np.zeros((len(vv),3,4))
        dv[:,:,:3]=np.einsum('vaij,i->vja',dr,lever)
        return vv,rr,dr,centres,ds,dv
