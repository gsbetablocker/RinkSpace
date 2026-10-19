"""Current-course q9 pattern geometry, without installing any source factor."""
from __future__ import annotations
from copy import deepcopy
import numpy as np
from z190_shared_board_pattern_source import SharedPerimeterBoardPatternGeometry,_q9


class CourseBoardPatternGeometry(SharedPerimeterBoardPatternGeometry):
    """Same Ad12/13 material q9 on the complete continuous board course.

    Nine material coordinates and their print identities stay unchanged.
    The old upper-left station origin is retained; positions are periodic,
    without a straight/corner/face placement domain. Board H remains shared.
    Original methods keep returning world plus original q9 Jacobian; separate
    course columns are available for the caller's original factor chain.
    """
    def __init__(self,original,course,q_course=None):
        if not isinstance(original,SharedPerimeterBoardPatternGeometry):
            raise TypeError('Original qualified perimeter-pattern geometry required')
        self.original=original
        self.__dict__.update(deepcopy(original.__dict__))
        self.original=original;self.course=course
        self.q_course=course.initial.copy()if q_course is None else course.validate_parameters(q_course)
        if self.scene_unit_m!=course.scene_unit_m:raise ValueError('Original fixed metric scale must be identical')
        lo,hi=map(int,self.course_source_boundary_indices)
        base=course.sample_perimeter()[0][lo:hi+1,:2]*self.scene_unit_m
        if not np.array_equal(base,original.course_xy_m):raise ValueError('Original pattern course does not use these same finite knots')
        bottom,_,jbottom,_,_=course.sample_perimeter(self.q_course,with_jacobian=True)
        indices=np.r_[np.arange(lo,len(bottom)),np.arange(lo+1)]
        self.course_xy_m=bottom[indices,:2]*self.scene_unit_m
        self._course_xy_jacobian=jbottom[indices,:2]*self.scene_unit_m
        edges=np.diff(self.course_xy_m,axis=0);lengths=np.linalg.norm(edges,axis=1)
        if np.any(lengths<=0):raise ValueError('The actual closed course needs positive segment lengths')
        self.course_station_m=np.r_[0.,np.cumsum(lengths)]
        self.course_tangents=edges/lengths[:,None]
        # Compatibility metadata describes one period, never a placement bound.
        self.station_domain_m=np.array([0.,self.course_station_m[-1]])
        self.z_domain_m=np.array([0.,self.q_course[5]])
        self.x_m=-self.q_course[0]
        self.y_domain_m=np.array([-self.q_course[1]+self.q_course[3],self.q_course[1]-self.q_course[3]])
        self._course_edges=np.diff(self.course_xy_m,axis=0)
        self._course_edges_j=np.diff(self._course_xy_jacobian,axis=0)
        self._course_lengths=np.linalg.norm(self._course_edges,axis=1)
        jl=np.einsum('ni,nik->nk',self._course_edges,self._course_edges_j)/self._course_lengths[:,None]
        self._course_prefix_j=np.vstack((np.zeros((1,6)),np.cumsum(jl,axis=0)))

    def _station_branch(self,station):
        raw=np.asarray(station,float).reshape(-1)
        if not np.isfinite(raw).all():raise ValueError('Finite continuous pattern station required')
        period=self.course_station_m[-1];turns=np.floor(raw/period)
        s=raw%period
        i=np.minimum(np.searchsorted(self.course_station_m,s,side='right')-1,len(self.course_tangents)-1)
        return s,i,turns

    def _arc(self,station):
        s,i,_=self._station_branch(station)
        xy=self.course_xy_m[i]+(s-self.course_station_m[i])[:,None]*self.course_tangents[i]
        return xy,self.course_tangents[i]

    def guard(self,q9):
        q=_q9(q9);_,zc,radius=q[:3];points=q[3:].reshape(3,2)
        if radius<=0:raise ValueError('Shared unfolded printed-circle radius must be positive')
        if zc-radius<self.z_domain_m[0]or zc+radius>self.z_domain_m[1]:
            raise ValueError('Entire printed circle must be inside the existing board height')
        if np.any(points[:,1]<self.z_domain_m[0])or np.any(points[:,1]>self.z_domain_m[1]):
            raise ValueError('Shared printed shield material point outside actual board height')
        area=np.linalg.det(np.stack((points[1]-points[0],points[2]-points[0])))
        if abs(area)<=np.finfo(float).eps:raise ValueError('Three noncollinear printed shield material points required')
        return True

    def _arc_course(self,station):
        xy,tangent=self._arc(station)
        s,i,turns=self._station_branch(station)
        t=(s-self.course_station_m[i])/self._course_lengths[i]
        jl=self._course_prefix_j[i+1]-self._course_prefix_j[i]
        ds=-turns[:,None]*self._course_prefix_j[-1]
        jt=(ds-self._course_prefix_j[i]-t[:,None]*jl)/self._course_lengths[i,None]
        j=self._course_xy_jacobian[i]+t[:,None,None]*self._course_edges_j[i]+self._course_edges[i,:,None]*jt[:,None]
        return xy,tangent,j

    def circle_course_jacobian(self,q9,theta,*,ad_id=13):
        q=_q9(q9);theta=np.asarray(theta,float).reshape(-1)
        if ad_id not in(13,26)or not np.isfinite(theta).all():raise ValueError('Original Ad13/26 finite circle angles required')
        _,_,j=self._arc_course(q[0]+q[2]*np.cos(theta))
        full=np.zeros((len(theta),3,6));full[:,:2]=j/self.scene_unit_m
        if ad_id==26:full[:,:2]*=-1
        return full

    def shield_course_jacobian(self,q9,*,ad_id=12):
        q=_q9(q9)
        if ad_id not in(12,25):raise ValueError('Original Ad12/25 printed material required')
        _,_,j=self._arc_course(q[3:].reshape(3,2)[:,0])
        full=np.zeros((3,3,6));full[:,:2]=j/self.scene_unit_m
        if ad_id==25:full[:,:2]*=-1
        return full

    def descriptor(self,q9,*,engineering_circle_samples=256):
        value=super().descriptor(q9,engineering_circle_samples=engineering_circle_samples)
        value['facet']='complete continuous existing board perimeter; source direction is context'
        value.pop('finite_station_domain_m',None)
        value['station_period_m']=float(self.course_station_m[-1])
        value['material_station_coordinates_are_periodic']=True
        value['computational_segments_do_not_bound_print_placement']=True
        value['shared_board_course_candidate']=dict(q_course=self.q_course.tolist(),
            finite_original_course_indices=list(self.course_source_boundary_indices),
            source_q9_parameter_order_and_role_unchanged=True,station_origin='same authenticated original left upper tangent',
            source_or_prior_added=False,full_source_factor_installed=False)
        return value


def rebuild_patterns_geometry(old_pattern_geometry,course,q_course=None):
    """Return a private current-course world object; original remains untouched."""
    return CourseBoardPatternGeometry(old_pattern_geometry,course,q_course)
