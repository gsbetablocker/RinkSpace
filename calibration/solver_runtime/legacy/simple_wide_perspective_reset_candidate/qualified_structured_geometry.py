"""Qualified source-domain validator, without unmeasured Lane-left bound.

The released geometry's finite source/render functions remain unchanged.
Only its erroneous proxy-to-painted-right ordering guard is removed. Actual
painted-right points still must lie in the current ice footprint; all native
source support and finite-segment guards remain in the numerical owner.
"""
from copy import deepcopy
import numpy as np
from complete_ice_template_candidate.full_free_boundary_geometry import StructuredSharedIceTemplateFullBoundaryGeometry
from simple_wide_perspective_reset_candidate.differentiable_material_course_domain import (
    DifferentiableMaterialCourseDomain,DifferentiableBoardCourseGeometry)


class QualifiedStructuredFullBoundaryGeometry(StructuredSharedIceTemplateFullBoundaryGeometry):
    def __init__(self,original):
        if type(original)is not StructuredSharedIceTemplateFullBoundaryGeometry:
            raise TypeError('Explicit released structured-full-boundary50 provider required')
        self.__dict__=deepcopy(original.__dict__)
        # The finite material function is mathematically defined at a
        # physical H-Z active bound. Its partials must remain evaluable on
        # both sides; actual admission is checked separately without clamps.
        self.boundary_owner.transporter.__class__=DifferentiableMaterialCourseDomain
        self.boundary_owner.course.__class__=DifferentiableBoardCourseGeometry
        self.boundary_owner.transporter.course.__class__=DifferentiableBoardCourseGeometry
        self.source_scope_guard_contract=dict(schema='qualified-structured-source-domain-without-unmeasured-Lane-left-order.v1',
            removed_unmeasured_open_left_proxy_as_painted_right_lower_bound=True,
            original_material_world_functions_and_source_samples_unchanged=True,
            painted_right_points_inside_actual_ice_still_required=True,
            current_native_source_finite_support_guards_still_required=True,
            unobserved_historical_ad_heights_not_added_as_wide_source_constraints=True)
        self.source_scope_guard_contract.update(numeric_unclamped_material_extension_at_active_Z_bounds=True,
            actual_original_source_Z_and_H_minus_Z_required_by_nonlinear_owner=True)

    def validate_parameters(self,p):
        p=self._q(p);cq=self.boundary_owner.current_course6(self._boundary_q(p))
        if np.any(p[[3,4,5,6,7,8,9,10,11,15,16,17,21,23,24,26,27]]<=0):
            raise ValueError('Structured physical radii/distances/widths must be positive')
        if not 1>p[18]>p[19]>p[20]>0:raise ValueError('Common house rings require strict concentric order')
        if p[11]<=p[10]or p[17]+p[25]<=0:raise ValueError('Outward trapezoid and positive outside back distance required')
        # The two open-left positions are unmeasured rendering/support
        # proxies. They are not a material measurement of a sheet end and
        # cannot constrain the qualified painted-right carrier coordinate.
        owner=self.boundary_owner.transporter;q29=np.r_[self._boundary_q(p)[:23],cq]
        with owner._current_old_domain(q29):
            owner.original.panel.validate_parameters(p[28:32]);owner.original.pair_panels.validate_parameters(p[32:44])
        return p.copy()
