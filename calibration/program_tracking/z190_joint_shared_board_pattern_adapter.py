"""Shared real advertisement print patterns in the complete source state.

The programme and external observer use one common set of nine material
coordinates on the already modeled zero-thickness perimeter. Actual view
states, fixed installation XYZ and nonzero levers keep their existing units.
"""
from __future__ import annotations
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.sparse import csr_matrix,vstack
from z190_continuous_source_bundle import SCENE_UNIT_M
from z190_physical_camera import ViewState
from z190_shared_board_pattern_source import (
    SharedPerimeterBoardPatternGeometry,SharedBoardPatternSourceFactor,
    admit_pattern_source,fixed_pivot_jacobian)


class JointSharedBoardPatterns:
    def __init__(self,state,source_root,*,initial_q=None,initial_authority=None):
        self.state=state;source_root=Path(source_root)
        self.geometry=SharedPerimeterBoardPatternGeometry(state.geometry.base,SCENE_UNIT_M,
            facet_authority=dict(kind='same_event_source_topology',
                entities='Ad13 TCF outer ring and Ad12 printed shield on the connected left-end board course',
                user_exact_pts=230422016,board_thickness=0,
                position_unknown_not_world_measurement=True))
        self.factor=SharedBoardPatternSourceFactor(self.geometry)
        self.external_factor=None
        self.frames=[];manifests=[]
        for ad,sha in ((13,'214448266965e5ee14e8519c0fef8bac03bba356273c8acdaea17c00f84775e5'),
                       (12,'8bc7571bac78ebb6c3ca6fa2d21f9085ba43b8ac9277021e53883293f3d8e28f')):
            frames,manifest=admit_pattern_source(source_root/f'secondary6_Ad{ad}_native_pattern_QUALIFIED_FROZEN.json',
                state.original.aliases,expected_ad_id=ad,expected_observer='long_side_secondary',expected_sha256=sha)
            for frame in frames:
                p=int(frame['source_actual_pts'])
                if p not in state.bypts or state.bypts[p].name!='secondary':
                    raise ValueError('Shared programme pattern belongs to the actual secondary exposure inventory')
                frame=dict(frame,observer_state_owner='programme')
                self.frames.append(frame)
            manifests.append(manifest)
        if state.external is not None:
            from z190_external_board_pattern_source import admit_external_pattern_frames,ExternalBoardPatternDistanceSourceFactor
            frames,manifest=admit_external_pattern_frames(state.external)
            self.external_factor=ExternalBoardPatternDistanceSourceFactor(self.geometry)
            self.frames.extend(dict(f,observer_state_owner='external')for f in frames)
            manifests.append(manifest)
        self.fit=[f for f in self.frames if f['split']=='fit']
        self.held=[f for f in self.frames if f['split']=='held']
        # Source identities and original rectangles are fixed before any
        # numerical seed. A seed is never a measured world centre or prior.
        if initial_q is None:
            cameras={int(f['source_actual_pts']):self.camera(state.q,f)[0]for f in self.frames}
            q9,self.seed_report=self.geometry.seed(cameras,
                [f for f in self.frames if f['physical_ad_id']==13 and f['observer_state_owner']=='programme'],
                [f for f in self.frames if f['physical_ad_id']==12])
        else:
            q9=np.asarray(initial_q,float).copy()
            if q9.shape!=(9,)or not np.isfinite(q9).all():
                raise ValueError('Complete shared-pattern checkpoint requires nine finite material states')
            self.seed_report=dict(kind='unchanged_shared_material_checkpoint_continuation',
                authority=initial_authority,conditional_source_seed_recomputed=False)
        self.seed_report.update(reference_observer='programme_long_side_secondary',
            external_same_material_source_used_in_joint_residual_not_a_separate_world_center=True)
        self.geometry.guard(q9)
        self.initial_q=q9;self.start=len(state.q)
        residual=self.evaluate(np.r_[state.q,q9],False)
        # External circle retains sixteen original finite normal segments.
        self.scales=np.concatenate([np.full(len(f['source_curve_support'])if f['physical_ad_id']==13 else 6,
            float(f['engineering_sigma_native_px'][0]))for f in self.fit])
        if residual.shape!=self.scales.shape:raise ValueError('Shared pattern native scale inventory changed')
        self.quadratic=np.ones(len(residual),bool)
        self.manifest=dict(source=manifests,shared_parameters=9,source_held_not_fitted=True,
            common_material_across_programme_and_external=True,source_pattern_centres_not_world_anchors=True,
            exact_centrosymmetric_instances=[dict(original=13,mirror=26),dict(original=12,mirror=25)],
            board_geometry_changed=False,initialization=self.seed_report,
            external_factor_operator='distance from the true outward-half projected circle to the original finite normal-support segment',
            external_original_UV_normal_interval_sigma_unchanged=True)

    def camera(self,q,frame,*,held=False):
        state=self.state;p=int(frame['source_actual_pts'])
        if frame['observer_state_owner']=='programme':
            block=state.bypts[p];cal=block.bundle.calibration
            camera=block.bundle.camera(block.local(q),p)
            return camera,block.xyz_slots,block.start+4*block.bundle.index[p],np.asarray(cal.lever_xyz)*SCENE_UNIT_M
        eq=q[state.external_start:state.external_stop]
        if held:eq,_=state.external.infer_held_view_from_neighbours(eq)
        index=int(state.external.index[p]);view=eq[4*index:4*index+4]
        cal=replace(state.original.model.calibrations[state.ids[0]],
            pivot_xyz=tuple(q[state.slots[state.ids[0]]]/SCENE_UNIT_M))
        camera=cal.camera25_and_jacobians(ViewState(tuple(view[:3]),float(view[3]-np.log(1920.))))[0]
        return camera,state.slots[state.ids[0]],state.external_start+4*index,np.asarray(cal.lever_xyz)*SCENE_UNIT_M

    def evaluate(self,q,jacobian=True,*,held=False):
        q9=q[self.start:self.start+9];self.geometry.guard(q9)
        values=[];matrices=[];reports=[]
        for f in (self.held if held else self.fit):
            cam,slots,view,lever=self.camera(q,f,held=held)
            result=self.frame_factor(f).evaluate(q9,cam,f,with_jacobian=jacobian)
            values.append(result.residual)
            reports.append(dict(physical_ad_id=f['physical_ad_id'],source_actual_pts=f['source_actual_pts'],
                source_native_pts=f['source_native_pts'],split=f['split'],observer_state_owner=f['observer_state_owner'],
                source_reports=result.source_reports,source_pixels_roles_sigma_unchanged=True))
            if jacobian:
                jxyz,jview=fixed_pivot_jacobian(result.camera_state11_jacobian,cam,lever,SCENE_UNIT_M)
                matrix=np.zeros((len(result.residual),len(q)))
                matrix[:,slots]=jxyz;matrix[:,view:view+4]=jview
                matrix[:,self.start:self.start+9]=result.shared_q9_jacobian
                matrices.append(csr_matrix(matrix))
        residual=np.concatenate(values)
        if held:return SimpleNamespace(residual=residual,source_reports=reports)
        return (residual,vstack(matrices,format='csr'))if jacobian else residual

    def descriptor(self,q):
        return self.geometry.descriptor(q[self.start:self.start+9])

    def frame_factor(self,frame):
        if frame.get('source_factor_kind')=='external_original_normal_interval':
            if self.external_factor is None:raise ValueError('Qualified external normal-interval factor required')
            return self.external_factor
        return self.factor

    def source_report(self,q):
        fit=[]
        q9=q[self.start:self.start+9]
        for f in self.fit:
            cam,*_=self.camera(q,f)
            result=self.frame_factor(f).evaluate(q9,cam,f,with_jacobian=False)
            fit.append(dict(physical_ad_id=f['physical_ad_id'],actual_pts=f['source_actual_pts'],
                source_reports=result.source_reports))
        held=self.evaluate(q,False,held=True)
        return dict(manifest=self.manifest,shared_geometry=self.descriptor(q),fit=fit,
            held=held.source_reports,held_normalized_residual=held.residual.tolist(),accuracy_accepted=False)
