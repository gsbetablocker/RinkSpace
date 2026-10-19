"""Serial original pattern factors on one currently bound shared course."""
from __future__ import annotations
import copy
from contextlib import contextmanager
from types import SimpleNamespace
import cv2
import numpy as np
from scipy.sparse import csr_matrix,vstack
from installation_mount_candidate.mount_graph import MountOpticsPatterns
from z190_shared_board_pattern_source import fixed_pivot_jacobian
from z190_continuous_source_bundle import SCENE_UNIT_M
from .course_pattern_geometry import rebuild_patterns_geometry
from .course_source_chart import COURSE_INDICES,CAMERA_COUNT


def _factor_geometry(factor,geometry):
    """Private clone preserves exact original factor class/discretization."""
    if factor is None:return None
    clone=copy.copy(factor)
    if hasattr(factor,'factor'):clone.factor=_factor_geometry(factor.factor,geometry)
    elif hasattr(factor,'geometry'):clone.geometry=geometry
    else:raise TypeError('Original typed pattern factor lacks geometry owner')
    return clone


class CourseMountOpticsPatterns(MountOpticsPatterns):
    @classmethod
    def from_owner(cls,owner,world_geometry,*,course_indices=COURSE_INDICES,course_fd_steps=(1e-6,)*6):
        if not isinstance(owner,MountOpticsPatterns):raise TypeError('Original explicit mount/optics pattern owner required')
        out=cls.__new__(cls);out.__dict__.update(vars(owner))
        out._original_course_pattern_geometry=owner.geometry
        out._original_course_pattern_factor=owner.factor
        out._original_course_external_pattern_factor=owner.external_factor
        out.course_world_geometry=world_geometry
        out.course_indices=np.asarray(course_indices,int)
        out.course_fd_steps=np.asarray(course_fd_steps,float)
        if out.course_indices.shape!=(6,)or not np.array_equal(out.course_indices,COURSE_INDICES)or out.course_fd_steps.shape!=(6,)or not np.isfinite(out.course_fd_steps).all()or np.any(out.course_fd_steps<=0):
            raise ValueError('Explicit actual course6 slots and positive physical derivative widths required')
        out.manifest=dict(owner.manifest,current_shared_course=True,source_q9_unchanged=True,
            course_indices=out.course_indices.tolist(),course_fd_steps=out.course_fd_steps.tolist(),
            source_factor_classes_and_source_support_unchanged=True,
            pattern_evaluation='serial original factor; fresh private course geometry, no stale pattern future',
            new_source_or_prior_rows=0)
        return out

    @contextmanager
    def bind_current_geometry(self,q):
        q=np.asarray(q,float)
        if q.shape!=(CAMERA_COUNT,)or not np.isfinite(q).all():raise ValueError('Complete physical34231 pattern evaluation state required')
        old=(self.geometry,self.factor,self.external_factor)
        geometry=rebuild_patterns_geometry(self._original_course_pattern_geometry,self.course_world_geometry.course,q[self.course_indices])
        self.geometry=geometry
        self.factor=_factor_geometry(self._original_course_pattern_factor,geometry)
        self.external_factor=_factor_geometry(self._original_course_external_pattern_factor,geometry)
        try:yield geometry
        finally:self.geometry,self.factor,self.external_factor=old

    def _evaluate_current(self,q,jacobian,held):
        q9=q[self.start:self.start+9];self.geometry.guard(q9)
        values,matrices,reports=[],[],[]
        for frame in self.held if held else self.fit:
            cam,slots,view,lever=self.camera(q,frame,held=held)
            result=self.frame_factor(frame).evaluate(q9,cam,frame,with_jacobian=jacobian)
            values.append(result.residual)
            reports.append(dict(physical_ad_id=frame['physical_ad_id'],
                source_actual_pts=frame['source_actual_pts'],source_native_pts=frame['source_native_pts'],
                split=frame['split'],observer_state_owner=frame['observer_state_owner'],
                source_reports=result.source_reports,source_pixels_roles_sigma_unchanged=True))
            if jacobian:
                jxyz,jview=fixed_pivot_jacobian(result.camera_state11_jacobian,cam,lever,SCENE_UNIT_M)
                matrix=np.zeros((len(result.residual),len(q)))
                matrix[:,slots],matrix[:,view:view+4]=jxyz,jview
                matrix[:,self.start:self.start+9]=result.shared_q9_jacobian
                chart=self.chart_for_frame(frame)
                matrix[:,chart.indices]=result.optics_jacobian
                rotation=cv2.Rodrigues(np.asarray(cam[3:6]))[0]
                matrix[:,chart.mount_indices]=np.asarray(result.camera_state11_jacobian)[:,:3]@rotation.T/(100.*SCENE_UNIT_M)
                matrices.append(csr_matrix(matrix))
        residual=np.concatenate(values)if values else np.empty(0)
        if held:return SimpleNamespace(residual=residual,source_reports=reports)
        return (residual,vstack(matrices,format='csr')if matrices else csr_matrix((0,len(q))))if jacobian else residual

    def _restore_camera_owners(self,q,held=False):
        # Original camera() calls local(q), which can update a private bundle's
        # course plane/guard cache. Restore those explicit current owners too.
        used=set()
        for frame in self.held if held else self.fit:
            if frame['observer_state_owner']!='programme':continue
            block=self.state.bypts[int(frame['source_actual_pts'])]
            if id(block)in used:continue
            used.add(id(block));block.local(q)

    def _course_fd_stencil(self,q,slot,h):
        plus,minus=q.copy(),q.copy();plus[slot]+=h;minus[slot]-=h
        return (plus,minus),dict(method='central',step=float(h),column=int(slot))

    def evaluate(self,q,jacobian=True,*,held=False):
        q=np.asarray(q,float)
        try:
            with self.bind_current_geometry(q):result=self._evaluate_current(q,jacobian,held)
            if not jacobian or held:return result
            residual,matrix=result;columns=np.empty((len(residual),6))
            domain_checks=[]
            for col,(slot,h)in enumerate(zip(self.course_indices,self.course_fd_steps)):
                samples,stencil=self._course_fd_stencil(q,slot,h);values=[]
                for sample in samples:
                    if np.array_equal(sample,q):value=residual
                    else:
                        with self.bind_current_geometry(sample):value=self._evaluate_current(sample,False,False)
                    if value.shape!=residual.shape:raise ValueError('Current pattern course perturbation changed original source row count')
                    values.append(value)
                if stencil['method']=='central':columns[:,col]=(values[0]-values[1])/(2*stencil['step'])
                else:
                    columns[:,col]=stencil['side']*(4*(values[1]-values[0])-(values[2]-values[0]))/(2*stencil['step'])
                    domain_checks.append(stencil)
            self._last_course_pattern_fd_domain_checks=domain_checks
            rr,cc=np.nonzero(columns)
            extra=csr_matrix((columns[rr,cc],(rr,self.course_indices[cc])),shape=matrix.shape)
            return residual,matrix+extra
        finally:self._restore_camera_owners(q,held)

    def descriptor(self,q):
        with self.bind_current_geometry(q):return self.geometry.descriptor(np.asarray(q)[self.start:self.start+9])

    def source_report(self,q):
        q=np.asarray(q,float);fit=[]
        try:
            with self.bind_current_geometry(q):
                q9=q[self.start:self.start+9]
                for frame in self.fit:
                    camera,*_=self.camera(q,frame)
                    result=self.frame_factor(frame).evaluate(q9,camera,frame,with_jacobian=False)
                    fit.append(dict(physical_ad_id=frame['physical_ad_id'],actual_pts=frame['source_actual_pts'],
                        source_reports=result.source_reports))
                descriptor=self.geometry.descriptor(q9)
            held=self.evaluate(q,False,held=True)
            return dict(manifest=self.manifest,shared_geometry=descriptor,fit=fit,
                held=held.source_reports,held_normalized_residual=held.residual.tolist(),accuracy_accepted=False)
        finally:self._restore_camera_owners(q,False)
