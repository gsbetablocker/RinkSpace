"""Opt-in engineering geometry references outside all original source rows.

The nominal values and their working widths are declared modelling choices,
not measurements. This module never changes a source UV/role/sigma, scene,
camera, profile, parameter count or material binding. A new complete objective
gate is required after a caller explicitly wraps its source with these rows.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from copy import deepcopy
import numpy as np
from scipy.sparse import csr_matrix
from z190_streaming_source_normal import SourceRowBlock


VERSION = 'rinkspace.soft-engineering-geometry-relations.v1'


@dataclass(frozen=True)
class EngineeringReferenceConfig:
    """Working engineering widths; none is a surveyed uncertainty."""
    distance_sigma_m: float = .10
    blue_separation_sigma_m: float = .20
    parallel_sigma_rad: float = np.deg2rad(.5)
    goal_end_sigma_m: float = .50
    hockey_radius_m: float = 4.5
    blue_separation_m: float = 15.
    end_circle_to_goal_m: float = 6.
    end_circle_half_separation_m: float = 7.
    design_length_m: float = 60.
    design_width_m: float = 30.
    goal_to_design_end_m: float = 4.
    include_curling_tee_and_outer_radius: bool = False
    curling_tee_separation_m: float = 34.75
    curling_outer_radius_m: float = 1.829
    include_curling_inner_radii: bool = False
    curling_middle_radius_m: float = 1.219
    curling_inner_radius_m: float = .610
    include_button_lower_bound: bool = False
    curling_button_min_radius_m: float = .152
    # Width cannot be inferred from the centre-to-centre pitch. No implicit
    # sheet-width row is enabled by this configuration.

    def __post_init__(self):
        numeric = [v for k,v in asdict(self).items() if not isinstance(v,bool)]
        if not np.isfinite(numeric).all() or min(numeric) <= 0.:
            raise ValueError('Explicit finite positive engineering reference values and widths required')


@dataclass(frozen=True)
class GeometryReferenceEvaluation:
    raw: np.ndarray
    residual: np.ndarray
    jacobian: np.ndarray | None
    quadratic: np.ndarray
    scales: np.ndarray
    rows: tuple[dict, ...]
    report: dict


def _readonly(value, dtype=float):
    a=np.array(value,dtype=dtype,copy=True);a.flags.writeable=False
    return a


# Small forward derivative pairs. Values come from the actual geometry's
# finite-material evaluator; no independent nominal scene is substituted.
def _add(a,b):return a[0]+b[0],a[1]+b[1]
def _sub(a,b):return a[0]-b[0],a[1]-b[1]
def _scale(a,s):return a[0]*s,a[1]*s
def _dot(a,b):return float(a[0]@b[0]),a[1].T@b[0]+b[1].T@a[0]
def _norm(a):
    length=float(np.linalg.norm(a[0]))
    if not np.isfinite(length) or length<=0.:raise ValueError('Nondegenerate current geometry vector required')
    return length,a[0]@a[1]/length
def _unit(a):
    length,dlength=_norm(a)
    return a[0]/length,a[1]/length-np.outer(a[0],dlength)/length**2


class SoftNominalGeometryFactor:
    """Actual world43/46 relations, mapped to an explicit complete chart.

    Each engineering row is a *relation*, not an extra independent image
    observation. Default sixteen rows use current C0/A/C/B/goal geometry.
    FR and ring references are separate explicit opt-ins. Five finite line
    spans and all historical crease samples remain unchanged by this factor.
    """
    def __init__(self, geometry, geometry_indices, parameter_count, *, config=None):
        self.geometry=geometry
        self.config=EngineeringReferenceConfig() if config is None else config
        if not isinstance(self.config,EngineeringReferenceConfig):
            raise TypeError('Explicit EngineeringReferenceConfig required')
        self.world_parameter_count=int(geometry.parameter_count)
        if self.world_parameter_count not in (43,46):
            raise ValueError('Actual shared hockey world43 or explicit optional-ratio world46 required')
        self.geometry_indices=_readonly(geometry_indices,int)
        self.parameter_count=int(parameter_count)
        if (self.geometry_indices.shape!=(self.world_parameter_count,) or
            len(set(self.geometry_indices.tolist()))!=self.world_parameter_count or
            np.any(self.geometry_indices<0) or np.any(self.geometry_indices>=self.parameter_count)):
            raise ValueError('Unique owned complete-world parameter map required')
        if (self.config.include_curling_inner_radii or self.config.include_button_lower_bound) and self.world_parameter_count!=46:
            raise ValueError('Inner/button references require the explicit three active FR ratio coordinates')
        self.unit=float(geometry.scene_unit_m)
        # K20/ratio-only chart leaves the actual finite arena boundary fixed.
        # Use that estimated boundary, never silently substitute +/-30m.
        boundary=np.asarray(geometry.base['ice_boundary']['points_xyz'],float)*self.unit
        if boundary.ndim!=2 or boundary.shape[1]!=3 or not np.isfinite(boundary).all():
            raise ValueError('Actual complete finite ice boundary required')
        self.actual_boundary_bounds_m=np.r_[boundary[:,:2].min(axis=0),boundary[:,:2].max(axis=0)]
        self.actual_boundary_midaxis_y_m=float((self.actual_boundary_bounds_m[1]+self.actual_boundary_bounds_m[3])/2)
        self._primitive_rows={}
        fs={f['id']:f for f in geometry.base['ice_features']}
        for name in ('C0','A1','A2','F1','R1'):
            self._primitive_rows[name]=self._circle_row(fs[name],name)
        for name in ('C','B1','B2','goal_line_left','goal_line_right'):
            self._primitive_rows[name]=dict(id=name,world=np.asarray(fs[name]['segments_xyz'],float).reshape(2,3),
                engineering_relation_primitive_not_image_observation=True)
        if self.world_parameter_count==46:
            rings={r['role']:r for r in fs['F1']['rings']}
            for role in ('middle','inner','center'):
                self._primitive_rows['F1_'+role]=self._circle_row(fs['F1'],'F1_'+role,float(rings[role]['radius']))
        initial=self.evaluate_world(geometry.initial,True)
        self.scales=initial.scales;self.quadratic=initial.quadratic
        self.rows=initial.rows;self.row_count=len(self.scales)
        self.contract=dict(schema=VERSION,original_source_rows_modified=0,new_engineering_reference_rows=self.row_count,
            parameter_count=self.parameter_count,world_parameter_count=self.world_parameter_count,
            world_parameter_indices=self.geometry_indices.tolist(),reference_configuration=asdict(self.config),
            source_role='explicit_engineering_reference_not_measured_source',loss='component quadratic L2',
            widths_are_working_engineering_choices_not_measurement_uncertainties=True,
            official_template_version_at_venue_unknown=True,
            nominal_structure_may_be_overruled_by_original_real_source=True,
            design_dimensions_metres=[self.config.design_length_m,self.config.design_width_m],
            design_dimensions_are_user_authorized_best_reference_not_survey_or_hard_target=True,
            actual_boundary_60x30_transport_installed=False,actual_width30_reference_row_installed=False,
            actual_boundary_bounds_conditional_metres=self.actual_boundary_bounds_m.tolist(),
            actual_boundary_retains_inherited_estimate=True,
            goal_end_reference_uses_current_actual_estimated_boundary_extrema=True,
            goal_to_end_4m_is_low_weight_venue_version_assumption_not_measured=True,
            no_goal_reference_forces_design_X_equal_plus_or_minus_26m=True,
            geometry_world_source_render_never_overwritten=True,
            historical_half_circle_crease_unchanged=True,
            unknown_lane_end_or_missing_observations_added=False,
            sheet_width_reference_installed=False,centre_pitch_not_sheet_width=True,
            blue_reference_is_B2_minus_B1_separation_not_each_distance_from_centre=True,
            blue_B2_and_goal_paint_edge_semantics_not_certified_as_official_dimension_edge=True,
            standard_distances_are_declared_reference_assumptions_not_identity_certification=True,
            original_profile_arrays_roles_UV_sigma_budget_and_aliases_preserved=True,
            existing_parameter_count_unchanged=True,must_pass_fresh_complete_source_gate_before_solver=True,
            accuracy_accepted=False,rows=deepcopy(list(self.rows)))

    @staticmethod
    def _circle_row(feature,name,radius=None):
        centre=np.asarray(feature['center_xyz'],float)
        radius=float(feature['radius']) if radius is None else float(radius)
        offsets=np.array([[1.,0.,0.],[-1.,0.,0.],[0.,1.,0.],[0.,-1.,0.]])
        return dict(id=name+'_outer' if name in ('C0','A1','A2','F1','R1') else name,
            world=centre+radius*offsets,engineering_relation_primitive_not_image_observation=True)

    def _material(self,name,q):
        row=self._primitive_rows[name]
        # Same primitive/world derivative used by source geometry. This
        # private pair accessor avoids duplicate rebinds or building a scene.
        world,j=self.geometry._world_and_jacobian(row,q)
        return np.asarray(world,float)*self.unit,np.asarray(j,float)*self.unit

    def _circle(self,name,q):
        world,j=self._material(name,q)
        centre=(world.mean(axis=0)[:2],j.mean(axis=0)[:2])
        radial=(world[0,:2]-centre[0],j[0,:2]-centre[1])
        return centre,_norm(radial)

    def _line(self,name,q):
        w,j=self._material(name,q)
        centre=(w.mean(axis=0)[:2],j.mean(axis=0)[:2])
        vector=(w[1,:2]-w[0,:2],j[1,:2]-j[0,:2])
        tangent=_unit(vector)
        normal=(np.array([-tangent[0][1],tangent[0][0]]),np.vstack((-tangent[1][1],tangent[1][0])))
        if normal[0][0]<0.:normal=_scale(normal,-1.)
        angle=np.arctan2(vector[0][1],vector[0][0])
        jangle=(vector[0][0]*vector[1][1]-vector[0][1]*vector[1][0])/float(vector[0]@vector[0])
        return dict(centre=centre,normal=normal,angle=(float(angle),jangle))

    def evaluate_world(self,q,jacobian=True):
        q=self.geometry.validate_parameters(q)
        c=self.config;zero=np.zeros(self.world_parameter_count)
        def scalar(v):return float(v),zero
        circles={name:self._circle(name,q) for name in ('C0','A1','A2')}
        lines={name:self._line(name,q) for name in ('C','B1','B2','goal_line_left','goal_line_right')}
        raw=[];jj=[];scales=[];rows=[];near_branch=[]
        def add(name,value,target,sigma,unit,definition,*,semantic='engineering structure reference',branch=None):
            raw.append(float(value[0]-target));jj.append(np.asarray(value[1],float));scales.append(float(sigma))
            rows.append(dict(id=name,role='engineering_reference_not_observation',unit=unit,nominal=float(target),
                working_sigma=float(sigma),working_sigma_is_not_measured=True,definition=definition,
                source_semantic_scope=semantic,loss='quadratic',branch=branch))
        cc,cr=circles['C0'];a1,r1=circles['A1'];a2,r2=circles['A2']
        add('C0_A_common_radius',_sub(cr,_scale(_add(r1,r2),.5)),0.,c.distance_sigma_m,'m','C0 radius minus current A common mean radius')
        add('C0_nominal_radius',cr,c.hockey_radius_m,c.distance_sigma_m,'m','Current C0 circle radius; IIHF nominal engineering reference')
        add('C0_on_current_C',_dot(_sub(cc,lines['C']['centre']),lines['C']['normal']),0.,c.distance_sigma_m,'m','C0 centre signed normal distance from current finite C supporting line')
        add('C0_on_actual_boundary_midaxis',(cc[0][1],cc[1][1]),self.actual_boundary_midaxis_y_m,c.distance_sigma_m,'m','Current C0 centre-Y relative to current estimated boundary mean-Y; boundary remains fixed in this chart')
        am=_scale(_add(a1,a2),.5)
        add('A_pair_midaxis',_sub((am[0][1],am[1][1]),(cc[0][1],cc[1][1])),0.,c.distance_sigma_m,'m','Mean of the two existing right-end A centres minus C0 centre-Y; no left A entities invented')
        ah=_scale(_sub((a2[0][1],a2[1][1]),(a1[0][1],a1[1][1])),.5)
        add('A_pair_half_separation',ah,c.end_circle_half_separation_m,c.distance_sigma_m,'m','Half transverse separation of current existing A1/A2 centres')
        ng=lines['goal_line_right']['normal']
        add('A_pair_to_right_goal',_dot(_sub(lines['goal_line_right']['centre'],am),ng),c.end_circle_to_goal_m,c.distance_sigma_m,'m','Mean current right-end A centre normal distance from current right goal line',semantic='IIHF reference; current goal paint-side definition unspecified')
        add('A_pair_station_symmetry',_dot(_sub(a2,a1),ng),0.,c.distance_sigma_m,'m','Current A2 minus A1 longitudinal displacement measured on current goal normal')
        for name in ('B1','B2','goal_line_left','goal_line_right'):
            angle=_sub(lines[name]['angle'],lines['C']['angle'])
            wrapped=.5*float(np.arctan2(np.sin(2*angle[0]),np.cos(2*angle[0])))
            if abs(abs(wrapped)-np.pi/2)<1e-8:near_branch.append(name+'_parallel_C')
            add(name+'_parallel_C',(wrapped,angle[1]),0.,c.parallel_sigma_rad,'rad','Undirected angle between current line and C; modulo pi',branch='wrap interval [-pi/2,pi/2]; not differentiable at wrap cut')
        bm=_scale(_add(lines['B1']['centre'],lines['B2']['centre']),.5)
        add('blue_pair_symmetry',_dot(_sub(bm,lines['C']['centre']),lines['C']['normal']),0.,c.distance_sigma_m,'m','Mean current B1/B2 station about current C along C normal')
        add('blue_pair_separation',_dot(_sub(lines['B2']['centre'],lines['B1']['centre']),lines['C']['normal']),c.blue_separation_m,c.blue_separation_sigma_m,'m','B2 minus B1 projected separation; 15m between blue lines, not 15m each side of C',semantic='declared design reference; B1 observed thin centre/B2 paint side unspecified')
        left,right=self.actual_boundary_bounds_m[[0,2]]
        add('left_goal_to_actual_estimated_end',(lines['goal_line_left']['centre'][0][0]-left,lines['goal_line_left']['centre'][1][0]),c.goal_to_design_end_m,c.goal_end_sigma_m,'m','Goal-left midpoint X minus actual estimated boundary minimum-X; boundary is not forced to -30m',semantic='low-weight venue-version 4m reference, not a surveyed distance or paint-edge certification')
        add('right_goal_to_actual_estimated_end',(right-lines['goal_line_right']['centre'][0][0],-lines['goal_line_right']['centre'][1][0]),c.goal_to_design_end_m,c.goal_end_sigma_m,'m','Actual estimated boundary maximum-X minus goal-right midpoint X; boundary is not forced to +30m',semantic='low-weight venue-version 4m reference, not a surveyed distance or paint-edge certification')
        if c.include_curling_tee_and_outer_radius:
            fc,fr=self._circle('F1',q);rc,rr=self._circle('R1',q)
            add('FR_common_outer_radius',_scale(_add(fr,rr),.5),c.curling_outer_radius_m,c.distance_sigma_m,'m','One common FR outer-radius engineering reference; not eight duplicated nominal measurements')
            add('FR_tee_separation',_norm(_sub(rc,fc)),c.curling_tee_separation_m,c.distance_sigma_m,'m','Same-row current F1/R1 tee centre distance; no unconfirmed Lane endpoint assumed')
        if c.include_curling_inner_radii:
            for role,target in (('middle',c.curling_middle_radius_m),('inner',c.curling_inner_radius_m)):
                _,radius=self._circle('F1_'+role,q)
                add('FR_common_'+role+'_radius',radius,target,c.distance_sigma_m,'m','Current common FR radial role; explicit optional active ratio chart')
        if c.include_button_lower_bound:
            _,radius=self._circle('F1_center',q)
            v=radius[0]-c.curling_button_min_radius_m
            # Lower-bound hinge has zero derivative on feasible side. At
            # equality the residual derivative is not unique, cost is C1.
            value=radius if v<0. else (c.curling_button_min_radius_m,zero)
            if v==0.:near_branch.append('FR_button_minimum_radius')
            add('FR_button_minimum_radius',value,c.curling_button_min_radius_m,c.distance_sigma_m,'m','Only minimum button radius, never equality to 0.152m',branch='negative-part lower-bound hinge; boundary residual derivative not unique')
        scales=_readonly(scales);raw=_readonly(raw);jraw=np.stack(jj)
        residual=_readonly(raw/scales)
        normalized_j=_readonly(jraw/scales[:,None]) if jacobian else None
        quadratic=_readonly(np.ones(len(raw)),bool)
        return GeometryReferenceEvaluation(raw,residual,normalized_j,quadratic,scales,tuple(rows),
            dict(cost=float(.5*(residual@residual)),derivative_is_unique=not near_branch,
                nondifferentiable_reference_branches=near_branch,original_source_rows_modified=0,
                actual_boundary_60x30_transport_installed=False,sheet_width_not_inferred_from_pitch=True))

    def block(self,q,jacobian=True,*,original_start=0):
        q=np.asarray(q,float)
        if q.shape!=(self.parameter_count,) or not np.isfinite(q).all():
            raise ValueError('Explicit complete state required, including unchanged statistical/high-camera tails')
        result=self.evaluate_world(q[self.geometry_indices],jacobian)
        matrix=None
        if jacobian:
            rr,cc=np.nonzero(result.jacobian)
            matrix=csr_matrix((result.jacobian[rr,cc],(rr,self.geometry_indices[cc])),shape=(self.row_count,self.parameter_count))
        ledger=dict(family='soft_nominal_geometry_engineering_references',schema=VERSION,
            original_start=int(original_start),original_stop=int(original_start)+self.row_count,
            references_are_not_image_observations=True,working_sigma_not_measured=True,
            source_rows=deepcopy(list(self.rows)),complete_parameter_count=self.parameter_count,
            world_parameter_indices=self.geometry_indices.tolist())
        return SourceRowBlock(result.residual,matrix,result.quadratic,result.scales,ledger)


class CompleteSourceWithSoftGeometryReferences:
    """Preserve all original blocks verbatim, append explicit quadratic rows."""
    def __init__(self,base,reference_factor):
        if int(base.count)!=reference_factor.parameter_count:
            raise ValueError('Reference coordinates must match actual complete source count')
        self.base=base;self.reference_factor=reference_factor
        self.count=int(base.count);self.updater=base.updater
        self.quadratic=_readonly(np.r_[base.quadratic,reference_factor.quadratic],bool)
        self.scales=_readonly(np.r_[base.scales,reference_factor.scales])
        self.contract=dict(schema='rinkspace.complete-source-plus-soft-engineering-relations.v1',
            original_source_contract=deepcopy(getattr(base,'contract',None)),
            original_source_rows=len(base.scales),new_engineering_rows=reference_factor.row_count,
            total_rows=len(self.scales),parameter_count=self.count,
            engineering_reference_contract=deepcopy(reference_factor.contract),
            original_blocks_arrays_ledger_order_and_profiles_preserved=True,
            source_objective_changed_explicitly=True,
            must_pass_fresh_complete_source_gate_before_solver=True,accuracy_accepted=False)

    def __getattr__(self,name):
        base=self.__dict__.get('base')
        if base is None:raise AttributeError(name)
        return getattr(base,name)

    def guard(self,q):
        self.base.guard(q)
        self.reference_factor.geometry.validate_parameters(np.asarray(q,float)[self.reference_factor.geometry_indices])

    def blocks(self,q,jacobian=True):
        yield from self.base.blocks(q,jacobian)
        yield self.reference_factor.block(q,jacobian,original_start=len(self.base.scales))

