"""Unclamped printed-material numeric extension of the locked V2 binder.

Actual trial qualification still requires every original observed corner's
Z>=0 and current H-Z>=0. This evaluator only permits partial derivatives
to exist on either side of those physical active bounds.
"""
import numpy as np
from shared_unmeasured_advertisement_geometry_candidate.shared_unmeasured_advertisement_geometry_v2 import SharedUnmeasuredAdvertisementGeometryV2
from shared_unmeasured_advertisement_geometry_candidate.shared_unmeasured_advertisement_geometry import current_course_binding,NUMBERS


class DifferentiableUnmeasuredPrintGeometryV2(SharedUnmeasuredAdvertisementGeometryV2):
    def validate_parameters(self,q):
        q=self._q(q);self.parent.validate_parameters(q[:self.parent_parameter_count])
        course,cq,_=current_course_binding(self.parent,q[:self.parent_parameter_count])
        bottom=course.sample_perimeter(cq)[0]
        perimeter=float(np.linalg.norm(np.roll(bottom,-1,axis=0)[:,:2]-bottom[:,:2],axis=1).sum())*self.scene_unit_m
        for i,n in enumerate(NUMBERS):
            _,width,zmin,zmax=q[self.parent_parameter_count+4*i:self.parent_parameter_count+4*i+4]
            if width<=0 or width>=perimeter/2 or zmax<=zmin:raise ValueError(f'Ad{n} needs positive minor arc width and material height')
        return q.copy()
