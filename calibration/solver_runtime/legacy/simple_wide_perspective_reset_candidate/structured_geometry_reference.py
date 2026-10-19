"""Current-axis engineering relations for the direct structured ice family.

This replaces the old global-Y/global-X relations. The structured provider
already enforces common centres/radii/carriers: their zero diagnostic rows
are redundant structural identities, not independent measurements. No row
anchors either hockey centre coordinate to its historical value.
"""
from dataclasses import asdict
from copy import deepcopy
import numpy as np
from shared_nominal_geometry_reference_candidate.soft_nominal_geometry_source import EngineeringReferenceConfig
from complete_ice_template_candidate.complete_ice_numeric_geometry import _boundary_hit

SCHEMA='rinkspace.current-axis-named-structured-engineering-reference18.v2'
REQUIRED_NAMES={3:'all_five_hockey_circle_radius_m',4:'faceoff_goal_inward_distance_m',
    5:'faceoff_half_transverse_separation_m',6:'blue_carrier_gap_m',
    7:'left_goal_to_current_axis_wall_m',8:'right_goal_to_current_axis_wall_m',
    15:'curling_tee_column_separation_m',17:'curling_eight_house_outer_radius_m'}
ACTIVE_ROWS={1:3,5:5,6:4,13:6,14:7,15:8,16:17,17:15}
ROW_NAMES=('C0_A_common_radius_structural','common_hockey_circle_radius',
    'C0_on_current_C_carrier_structural','historical_centre_coordinate_anchor_removed',
    'A_pair_on_current_hockey_transverse_axis_structural','A_pair_half_transverse_separation',
    'faceoff_goal_inward_distance','A_pair_same_longitudinal_station_structural',
    'B1_parallel_current_C_structural','B2_parallel_current_C_structural',
    'left_goal_parallel_current_C_structural','right_goal_parallel_current_C_structural',
    'blue_pair_symmetric_about_current_C_structural','blue_carrier_separation',
    'left_goal_to_actual_current_axis_wall','right_goal_to_actual_current_axis_wall',
    'common_curling_house_outer_radius','curling_tee_column_separation')


def _whole(g,q):
    q=np.asarray(q,float)
    if q.shape!=(int(g.parameter_count),) or not np.isfinite(q).all():
        raise ValueError('Complete current structured geometry chart required')
    for k,name in REQUIRED_NAMES.items():
        if g.parameter_names[k]!=name:raise ValueError('Named structured reference chart mismatch: '+name)
    return q


def structured_reference_values(g,qwhole,current_scene=None):
    """Raw physical values, before nominal subtraction or engineering widths.

    Public spawn-safe getter. Dimensions are named coordinates of the actual
    source provider. Structural zeros follow its enforced current-axis model;
    independent scene reconstruction checks these in the companion audit.
    The optional current_scene is intentionally not rebuilt by this getter.
    """
    q=_whole(g,qwhole);out=np.zeros(18)
    for row,k in ACTIVE_ROWS.items():out[row]=q[k]
    return out


def structured_reference_jacobian(g,qwhole):
    _whole(g,qwhole);J=np.zeros((18,int(g.parameter_count)))
    for row,k in ACTIVE_ROWS.items():J[row,k]=1.
    return J


def structured_reference_scene_values(g,qwhole,current_scene):
    """Independent geometric reconstruction using actual axes and wall hits."""
    q=_whole(g,qwhole);u=float(g.scene_unit_m)
    fs={f['id']:f for f in current_scene['ice_features']}
    a=np.array([np.cos(q[2]),np.sin(q[2]),0.]);v=np.array([-a[1],a[0],0.])
    def circle(name):return np.asarray(fs[name]['center_xyz'],float)*u,float(fs[name]['radius'])*u
    def ends(name):return np.asarray(fs[name]['segments_xyz'],float)[0]*u
    def goal(side):return np.asarray(fs['goal_line_'+side]['shared_physical_goal_frame']['centre_xyz'],float)*u
    def station(name):return float((ends(name)[0]-c)@a)
    def angle(name):
        w=ends(name);d=w[1]-w[0]
        # Undirected carrier angle relative to current transverse axis.
        t=float(np.arctan2(d@a,d@v))
        return .5*np.arctan2(np.sin(2*t),np.cos(2*t))
    c,r=circle('C0');a1,r1=circle('A1');a2,r2=circle('A2');am=(a1+a2)/2
    gl,gr=goal('left'),goal('right')
    poly=np.asarray(current_scene['ice_boundary']['points_xyz'],float)
    wl=_boundary_hit(c/u,-a,poly)*u;wr=_boundary_hit(c/u,a,poly)*u
    values=[r-(r1+r2)/2,r,float((c-ends('C')[0])@a),0.,float((am-c)@v),
        float((a2-a1)@v/2),float((gr-am)@a),float((a2-a1)@a)]
    values.extend(angle(name)-angle('C') for name in ('B1','B2','goal_line_left','goal_line_right'))
    values.extend([(station('B1')+station('B2'))/2,station('B2')-station('B1'),
        float((gl-wl)@a),float((wr-gr)@a)])
    fc,fr=circle('F1');rc,rr=circle('R1')
    ca=np.array([np.cos(q[14]),np.sin(q[14]),0.])
    values.extend([(fr+rr)/2,float((rc-fc)@ca)])
    return np.asarray(values)


def build_structured_reference_contract(g,config=None):
    config=EngineeringReferenceConfig(distance_sigma_m=.15,include_curling_tee_and_outer_radius=True) if config is None else config
    if not isinstance(config,EngineeringReferenceConfig) or not config.include_curling_tee_and_outer_radius:
        raise ValueError('Explicit eighteen-row configuration with curling references required')
    nominal=np.array([0.,config.hockey_radius_m,0.,0.,0.,config.end_circle_half_separation_m,
        config.end_circle_to_goal_m,0.,0.,0.,0.,0.,0.,config.blue_separation_m,
        config.goal_to_design_end_m,config.goal_to_design_end_m,config.curling_outer_radius_m,config.curling_tee_separation_m])
    scales=np.full(18,config.distance_sigma_m);scales[8:12]=config.parallel_sigma_rad
    scales[13]=config.blue_separation_sigma_m;scales[14:16]=config.goal_end_sigma_m
    rows=[]
    for i,name in enumerate(ROW_NAMES):
        active=i in ACTIVE_ROWS
        rows.append(dict(index=i,name=name,nominal=float(nominal[i]),working_sigma=float(scales[i]),
            unit='rad'if 8<=i<12 else'm',quadratic=True,
            role='explicit_soft_engineering_reference'if active else'redundant_structural_diagnostic_zero',
            active_named_parameter=g.parameter_names[ACTIVE_ROWS[i]]if active else None,
            active_world_column=ACTIVE_ROWS.get(i),not_measured_source_uncertainty=True,
            disabled_historical_origin_anchor=(i==3)))
    contract=dict(schema=SCHEMA,reference_configuration=asdict(config),rows=deepcopy(rows),
        active_geometry_count=int(g.parameter_count),source_pixel_rows_modified=0,
        current_hockey_axes_not_global_XY=True,goal_distances_use_actual_axis_course_intersections=True,
        no_historical_hockey_centre_XY_anchor=True,structural_zero_rows_are_not_independent_data=True,
        design_60x30_is_reference_only_not_hard_target=True,current_boundary_not_forced_to_design_dimensions=True,
        working_widths_are_explicit_engineering_choices_not_measured_uncertainties=True,
        actual_venue_standard_version_and_B2_goal_paint_edges_unverified=True,
        crease_and_unknown_Lane_end_observations_not_invented=True,
        source_UV_sigma_role_admission_and_budgets_unchanged=True,
        reference_objective_changed_from_global_axis_v1=True,accuracy_accepted=False)
    return tuple(rows),nominal,scales,contract
