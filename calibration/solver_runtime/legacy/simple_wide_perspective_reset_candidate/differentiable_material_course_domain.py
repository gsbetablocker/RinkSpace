"""Unclamped material-point extension for numerical partial derivatives.

Positive width/height, source identities and world equations are unchanged.
Physical print Z admission belongs to actual original-source H-Z/Z>=0
constraints and nonlinear trial guard. It is never replaced by this context.
No released module or shared active object is monkeypatched.
"""
from contextlib import contextmanager
import numpy as np
from shared_board_course_candidate.shared_rink_course_numeric_geometry import SharedRinkCourseNumericGeometry
from shared_board_course_candidate.shared_board_course_numeric_geometry import SharedBoardCourseNumericGeometry


class DifferentiableBoardCourseGeometry(SharedBoardCourseNumericGeometry):
    def _z(self,world,jac,q,policy,values):
        if policy!='absolute_z_m':return super()._z(world,jac,q,policy,values)
        value=np.broadcast_to(np.asarray(values,float),(len(world),))
        if not np.isfinite(value).all():raise ValueError('Finite original absolute material Z required')
        # Same source material world value and exact partials on both sides
        # of an active domain boundary, never clipped to the board height.
        world[:,2]=value/self.scene_unit_m;jac[:,2]=0.


class DifferentiableMaterialCourseDomain(SharedRinkCourseNumericGeometry):
    @contextmanager
    def _current_old_domain(self,q):
        q=np.asarray(q,float)
        panel=[q[7:11],*[q[11+4*k:15+4*k]for k in range(3)]]
        top=max(float(q[28]),*(float(p[2]+p[3]/2)for p in panel))
        bottom=min(0.,*(float(p[2]-p[3]/2)for p in panel))
        with super()._current_old_domain(q):
            one=self.original.panel;oldbottom=one.board_bottom_m;oldtop=one.board_top_m
            arcs=self.original.pair_panels.arc;saved={k:(a._ice_z,a._board_top)for k,a in arcs.items()}
            one.board_bottom_m=bottom;one.board_top_m=top
            for a in arcs.values():a._ice_z=bottom/self.scene_unit_m;a._board_top=top/self.scene_unit_m
            try:yield
            finally:
                one.board_bottom_m=oldbottom;one.board_top_m=oldtop
                for k,a in arcs.items():a._ice_z,a._board_top=saved[k]
