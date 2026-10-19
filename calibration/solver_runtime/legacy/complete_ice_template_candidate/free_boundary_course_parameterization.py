"""Uninstalled minimal free L/W/congruent-corner chart on historical course6.

This is a physical boundary/material primitive, not a whole-source installer.
No pixel, nominal-dimension prior, anisotropic global XYZ transform, source
weight, covariance, role or observed world qualification is introduced.
"""
import numpy as np
from shared_board_course_candidate.shared_board_course_numeric_geometry import SharedBoardCourseNumericGeometry

SCHEMA='rinkspace.uninstalled-free-boundary-three-coordinate-chart.v1'
NAMES=('boundary_half_length_delta_m','boundary_half_width_delta_m','shared_corner_x_extent_delta_m')


class FreeBoundaryCourseParameterization:
    parameter_count=3
    parameter_names=NAMES
    parameter_units=('m',)*3

    def __init__(self,actual_base_scene,scene_unit_m):
        self.course=SharedBoardCourseNumericGeometry(actual_base_scene,scene_unit_m)
        self.initial=np.zeros(3);self.scene_unit_m=self.course.scene_unit_m
        self.base=self.course.base;self.base_course6=self.course.initial.copy()
        self.corner_aspect_y_over_x=float(self.base_course6[3]/self.base_course6[2])
        self.course6_chain=np.zeros((6,3))
        self.course6_chain[0,0]=1.;self.course6_chain[1,1]=1.
        self.course6_chain[2,2]=1.;self.course6_chain[3,2]=self.corner_aspect_y_over_x

    def current_course6(self,q):
        q=np.asarray(q,float)
        if q.shape!=(3,)or not np.isfinite(q).all():raise ValueError('Explicit finite three-dimensional boundary delta chart required')
        out=self.base_course6+self.course6_chain@q
        return self.course.validate_parameters(out)

    def validate_parameters(self,q):
        self.current_course6(q);return np.asarray(q,float).copy()

    def sample_perimeter(self,q,*,with_jacobian=False):
        full=self.current_course6(q)
        if not with_jacobian:return self.course.sample_perimeter(full)
        b,t,jb,jt,ranges=self.course.sample_perimeter(full,with_jacobian=True)
        return b,t,np.einsum('nik,kj->nij',jb,self.course6_chain),np.einsum('nik,kj->nij',jt,self.course6_chain),ranges

    def primitive(self,side,material_t,q,*,z_policy='contact',z_values=None,with_jacobian=False):
        full=self.current_course6(q)
        value=self.course.primitive(side,material_t,full,z_policy=z_policy,z_values=z_values,with_jacobian=with_jacobian)
        if not with_jacobian:return value
        w,j=value;return w,np.einsum('nik,kj->nij',j,self.course6_chain)

    def station(self,station_m,q,*,z_m=0.,with_jacobian=False):
        full=self.current_course6(q)
        value=self.course.station(station_m,full,z_m=z_m,with_jacobian=with_jacobian)
        if not with_jacobian:return value
        world,jcourse,station_tangent,report=value
        return world,np.einsum('nik,kj->nij',jcourse,self.course6_chain),station_tangent,report

    def boundary_payload(self,q):
        """Current boundary/board/guard arrays only; not a rendered whole field.

        Source callers must separately rebind all declared material, finite
        extents, advertisements, pattern, external and current K20 carriers.
        This payload never pretends base static ice_features are current43.
        """
        scene=self.course.scene(self.current_course6(q),rebuild_grid=False)
        return {k:scene[k]for k in ('parameters','ice_boundary','board_surface','boundary_validity_margins')}

    def transport_named_ad_world(self,transporter,old_world23,q,number,world,old_jacobian23,*,physical_curve=False):
        """Reuse the already certified current-station material transport.

        Caller must use its declared named print ID/role and actual old world;
        no UV-to-world, generic snap or new advertisement identity is allowed.
        The returned J has old23 then new boundary3. Full corrected-source
        alias ownership remains the future installer responsibility.
        """
        if not np.array_equal(transporter.course.initial,self.base_course6):
            raise ValueError('Transporter and new chart have different original course authority')
        old=np.asarray(old_world23,float)
        if old.shape!=(23,)or not np.isfinite(old).all():raise ValueError('Actual declared complete original world23 required')
        current=np.r_[old,self.current_course6(q)]
        w,j=transporter._transport_current_ad(world,old_jacobian23,int(number),current,physical_curve=physical_curve)
        return w,np.concatenate((j[:,:,:23],np.einsum('nik,kj->nij',j[:,:,23:],self.course6_chain)),axis=2)

    def plane_and_extent_columns(self,jacobian_course6):
        """Original exact selected finite-plane/extent chain; no rho Schur."""
        value=np.asarray(jacobian_course6,float)
        if value.shape[-1]!=6 or not np.isfinite(value).all():raise ValueError('Actual exact six-column material derivative required')
        return np.einsum('...k,kj->...j',value,self.course6_chain)

    def contract(self,q=None):
        q=self.initial if q is None else self.validate_parameters(q)
        current=self.current_course6(q)
        return dict(schema=SCHEMA,installed_in_source_graph=False,parameter_names=list(NAMES),
            current_delta_m=np.asarray(q).tolist(),current_course6=current.tolist(),
            current_length_width_m=(2*current[:2]).tolist(),
            shared_congruent_corner_extent_xy_m=current[2:4].tolist(),
            original_geometry_authority=self.course.contract,
            no_60x30_target_prior_or_historical_dimension_hard_lock=True,
            conditional_remaining_shape=dict(corner_aspect_y_over_x=self.corner_aspect_y_over_x,
                superellipse_exponent=float(current[4]),board_height_m=float(current[5]),
                not_claimed_hardware_or_survey_measurements=True,
                full_course6_can_release_these_with_explicit_joint_source_chart=True),
            all_house_circles_and_painted_straight_lines_require_their_own_current_physical_generators=True,
            no_anisotropic_global_XYZ_transform=True,
            unknown_motion_planes_stay_unknown_and_ice_z0_planes_unchanged=True,
            no_new_source_pixels_sigma_roles_or_prior_rows=True,
            whole_corrected_AD_K20_source_alias_actor_profile_guard_export_route_not_installed=True)
