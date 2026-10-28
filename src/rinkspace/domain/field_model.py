"""Current M1--M5 geometry subset with a single native evaluator.

This data adapter never imports cameras, observations or advertisements.
Confirmed relations and unfitted parameter values are explicitly separate.
"""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import numpy as np
from ..path_policy import guard_project_write
from .entity_graph import _code, carrier_boundary_junctions
from .measurement import POLICY_ID, required_measurement, validate_scene_measurements
from ..compat.d78_entities import native_definition


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _write_new(path, data):
    path=guard_project_write(path)
    if path.exists():raise FileExistsError(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return path


def symmetric_goal_end_initial_values(layout, *, reference_end=None):
    """Project an old numeric initializer onto the explicitly requested ties.

    This does not fit observations. Shared dimensions use the explicit reference
    end, or arithmetic means when neither end was selected. Goal distance also
    preserves existing Lane/F left-support feasibility.
    The native evaluator, not this initializer, enforces the relations.
    """
    q = np.asarray(layout, dtype=float).copy()
    if q.shape != (35,) or not np.isfinite(q).all() or min(q[29:35]) <= 0:
        raise ValueError('Goal-end migration requires the explicit layout35 ellipse state')
    if reference_end not in (None, 'left', 'right'):
        raise ValueError('Explicit goal initializer reference must be left or right')
    nominal = (q[24] - q[23]) / 2. if reference_end is None else (-q[23] if reference_end == 'left' else q[24])
    axes = (q[29:31] + q[31:33]) / 2. if reference_end is None else q[29:31] if reference_end == 'left' else q[31:33]
    distance = max(nominal, -min(q[0], *q[10:14]))
    if distance <= q[25] or distance <= 0:
        raise ValueError('Symmetric goal initial distance must enclose faceoff centres')
    q[24] = distance
    q[29:31] = axes
    q[[23, 31, 32]] = 0.  # Explicit unused storage, never independent parameters.
    return q.tolist()


def constrain_symmetric_goal_ends(scene):
    """Declare shared goal-line/crease/trapezoid geometry without changing UV."""
    result = deepcopy(scene)
    if result.get('schema') != 'rinkspace.field_model.v1' or not result.get('shared_geometry_options', 0) & 2:
        raise ValueError('Symmetric goal ends require current rink-aligned board carriers')
    if not result['shared_geometry_options'] & 16:
        result['shared_layout'] = symmetric_goal_end_initial_values(result['shared_layout'])
    result['shared_geometry_options'] |= 16
    result['native_family'] = 'shared_geometry_v5_symmetric_goal_ends'
    blocks = {b['id']: b for b in result['parameter_blocks']}
    blocks['hockey_faceoff_shared'].update(tangent_size=3,
        independent_local_indices=[1, 2, 4], inactive_storage_local_indices=[0, 3, 5],
        positive_local_indices=[1, 2, 4],
        values_meaning=['unused', 'goal_half_longitudinal_separation',
                        'faceoff_half_longitudinal_separation', 'unused',
                        'faceoff_half_transverse_separation', 'unused'])
    blocks['hockey_ends_ellipse'].update(tangent_size=4,
        independent_local_indices=[0, 1, 4, 5], inactive_storage_local_indices=[2, 3],
        positive_local_indices=[0, 1, 4, 5],
        values_meaning=['shared_crease_depth_semiaxis', 'shared_crease_transverse_semiaxis',
                        'unused', 'unused', 'trapezoid_goal_half_span', 'trapezoid_board_half_span'])
    result['constraints']['M3'].update(goal_positions_mirrored=True,
                                      cross_end_inward_distances_shared=True)
    result['constraints']['M4'].update(independent_ax_ay_per_end=False,
        independent_depth_and_transverse_semiaxes=True, cross_end_axes_shared=True,
        contour_arc_only=True, closing_diameter_is_crease_contour=False,
        crease_centre_equals_goal_line_midpoint=True,
        trapezoid_upper_lower_mirrored=True, trapezoid_left_right_mirrored=True)
    result.setdefault('faceoff_parameterization', {})['goal_positions_independent'] = False
    result['goal_end_parameterization'] = dict(
        authority='user_explicit_request_20261028',
        goal_lines='x=-d and x=+d, endpoints on the same continuous board',
        creases='same ax/ay at corresponding goal midpoint; opposite finite half-ellipse arcs',
        trapezoids='shared goal/board half spans, reflected across both rink axes',
        crease_station_domain=[0., 1.], closing_diameter_observable=False,
        remaining_independent_dimensions=5, observation_penalties_added=False)
    for path in result['paths']:
        if path.get('generated_code') in (410, 411):
            path.update(closed=False, contour_domain='half_ellipse_arc_only',
                        centre_parent='corresponding_goal_line_midpoint',
                        closing_diameter_observable=False)
    if 'initial_values' in result:
        result['initial_values'].update(faceoff=result['shared_layout'][23:29],
                                       ends_ellipse=result['shared_layout'][29:35])
    return result


def build_model(source_dir,curling_model_path,out_path=None,ellipse_parameters=None,ellipse_provenance=None):
    source_dir=Path(source_dir).resolve(strict=True)
    curling_model_path=Path(curling_model_path).resolve(strict=True)
    original=_read(source_dir/'model.json');curling=_read(curling_model_path)
    if curling['schema']!='rinkspace.curling_lane.v1' or not curling['constraints'].get('connection_stops_before_lane_right'):
        raise ValueError('M1 must contain the current no-crossing connection support correction')
    q=list(curling['shared_layout'])
    if len(q)!=33 or q[:8]!=curling['curling_parameters'] or q[8:18]!=curling['lane_parameters']:
        raise ValueError('M1/native parameter mapping differs')
    if ellipse_parameters is None:
        raise ValueError('M4 requires four explicit half-ellipse initializer slots before symmetry migration; a circle is not renamed as ellipse')
    axes=list(map(float,ellipse_parameters))
    if len(axes)!=4 or not np.isfinite(axes).all() or min(axes)<=0:
        raise ValueError('M4 half-ellipse semiaxes must be four finite positive values')
    layout=q[:29]+axes+q[31:33]
    layout[18]=0.
    if not curling.get('shared_geometry_options',0)&8:
        old_left=layout[23]+layout[25];old_right=layout[24]-layout[26]
        layout[25]=(old_right-old_left)/2.
        layout[26]=0.;layout[27]=(layout[27]+layout[28])/2.;layout[28]=0.

    paths=deepcopy(curling['paths'])
    for p in paths:
        p.update(layer='ice',stage='M1',finite_support=[0.,1.])
    expected=set(range(400,412))|set(range(420,424))
    seen=set()
    for old in original['scene']['curves']:
        code=_code(old['label'])
        if code not in expected:continue
        if code in seen:raise ValueError(f'duplicate physical generated primitive {code}')
        seen.add(code)
        stage='M2' if code in (400,401,402,405) else 'M3' if code in (403,404,406,407,408,409) else 'M4'
        role='physical_material_centreline' if code in (400,401,402,403,404) else 'physical_material_centreline' if code in (405,406,407,408,409) else 'physical_half_ellipse_boundary' if code in (410,411) else 'physical_material_centreline'
        item={'id':old['id'],'label':old['label'],'entity_id':old['entity_id'],'generated_code':code,
              'layer':'ice','stage':stage,'role':role,'finite_support':[0.,1.],
              'source_observation_ids':[],'creates_source_observation':False,
              'measurement_policy_id':POLICY_ID}
        if code in (400,401,402,403,404):
            item['measurement']='centreline'
            item['endpoint_roles']=['derived_ice_boundary_intersection']*2
            item['endpoint_parent_ids']=[old['entity_id'],'entity/board.continuous']
        elif code in (405,406,407,408,409):
            item['measurement']='centreline'
        elif code in (410,411):
            item['measurement']='contour'
            item['endpoint_roles']=['unmeasured_goal_line_attachment']*2
            item['endpoint_parent_ids']=[old['entity_id'],f'entity/ice.goal_line_{"left" if code==410 else "right"}']
        else:
            side='left' if code<422 else 'right'
            item['measurement']='centreline'
            item['endpoint_roles']=['unmeasured_goal_line_attachment','unmeasured_board_attachment']
            item['endpoint_parent_ids']=[[old['entity_id'],f'entity/ice.goal_line_{side}'],[old['entity_id'],'entity/board.continuous']]
        paths.append(item)
    if seen!=expected or len(paths)!=64:raise ValueError('incomplete M1--M4 geometry inventory')
    # The two rings in M5 are heights of one shared continuous course, not
    # independently fitted top and bottom XY curves.
    for height,name in ((0.,'bottom'),(1.,'top')):
        paths.append({'id':f'board/continuous/{name}','label':f'board.{name}','entity_id':'entity/board.continuous',
                      'native_kind':7,'parameters':[height],'layer':'board','stage':'M5',
                      'role':'physical_board_ice_boundary' if height==0 else 'physical_board_top_boundary',
                      'station_domain':{'kind':'unbounded_periodic','coordinate':'unwrapped_global_course_arc_length','period':'board_perimeter','hard_segment_bounds':False},'source_observation_ids':[],'creates_source_observation':False})
    constraints={
      'M1':deepcopy(curling['constraints']),
      'M2':{'C0_centre_from_scene_origin':True,'C_passes_C0_centre':True,'B1_B2_parallel_to_C':True,
            'B1_B2_symmetric_about_C':True,'shared_paint_frame':'fixed_to_rink_axes','independent_line_rotations':False,
            'carriers_parallel_to_rink_short_axis':True,'carrier_endpoints_on_same_ice_boundary':True,
            'blue_separation_numeric_is_fitted':False},
      'M3':{'same_end_faceoff_pair_symmetric':True,'same_end_inward_distance_shared':True,
            'C0_A1_A4_common_radius':True,'cross_end_inward_distances_shared':False,
            'cross_end_transverse_separations_shared':True,'A1_A4_same_row':True,'A2_A3_same_row':True,'left_right_centres_mirror_about_C':True,'goal_lines_parallel_to_shared_paint_frame':True},
      'M4':{'shape_family':'half_ellipse','independent_ax_ay_per_end':True,'cross_end_axes_shared':False,
            'trapezoid_goal_and_board_half_spans_shared_between_sides':True,
            'trapezoid_goal_endpoint_parent':'actual_goal_line','trapezoid_board_endpoint_parent':'same_continuous_board',
            'ellipse_or_trapezoid_new_endpoint_UV':False},
      'M5':{'same_station_top_bottom_share_XY':True,'continuous_course':True,'top_bottom_height_independent_XY':False}}
    scene={'schema':'rinkspace.field_model.v1','scene_id':'ankara_2026.field','revision':1,
           'status':'unfitted_structure_candidate','native_family':'shared_geometry_v4_rink_axes_board_attached',
           'coordinate_unit':'rink_length_L_equals_1','meters_per_unit':None,
           'dimensions_status':'全部具体尺寸与半轴数值尚未重新拟合；米制未验证',
           'board':deepcopy(curling['board']),'shared_layout':layout,'shared_geometry_options':curling.get('shared_geometry_options',0)|14,
           'curling_layout_parameterization':deepcopy(curling.get('curling_layout_parameterization')),
           'parameter_blocks':[{'id':'curling_shared','offset':0,'size':8},{'id':'lane_shared','offset':8,'size':10},
             {'id':'hockey_shared','offset':18,'size':5},{'id':'hockey_faceoff_shared','offset':23,'size':6},
             {'id':'hockey_ends_ellipse','offset':29,'size':6}],
           'ellipse_initialization':deepcopy(ellipse_provenance),
           'initial_values':{'curling':layout[:8],'lane':layout[8:18],'hockey':layout[18:23],
             'faceoff':layout[23:29],'ends_ellipse':layout[29:35]},
           'paths':paths,'constraints':constraints,'curling_support_revision':deepcopy(curling.get('support_revision')),
           'source_model':str(source_dir/'model.json'),'source_model_sha256':hashlib.sha256((source_dir/'model.json').read_bytes()).hexdigest(),
           'M1_model':str(curling_model_path),'M1_model_revision':curling['revision'],
           'new_source_observations':0,'fit_executed':False,'user_approval':'not_reviewed',
           'scope':'M1--M5 geometry only; no camera, advertisements, solver or new source readings'}
    end_sources=source_dir.parent/'end_sources.json'
    if end_sources.is_file():
        scene['endpoint_source_references']={'path':str(end_sources.resolve()),
            'role':'preserved_source_references_not_additional_measurements_or_constraints'}
    scene['measurement_policy_id']=POLICY_ID
    scene['board_station_policy_id']='continuous_unwrapped_course.v1'
    scene['parameter_blocks'][0].update(tangent_size=7,independent_local_indices=[0,1,3,4,5,6,7],
        derived_local_coordinates=[{'index':2,'expression':'-1.5 * pitch','depends_on':[3]}])
    scene['parameter_blocks'][2].update(tangent_size=2,independent_local_indices=[1,2],
        derived_local_coordinates=[{'index':0,'expression':'0 (rink axes)'}],inactive_storage_local_indices=[3,4])
    scene['parameter_blocks'][3].update(tangent_size=4,independent_local_indices=[0,1,2,4],
        inactive_storage_local_indices=[3,5],values_meaning=['goal_left_x','goal_right_x','faceoff_half_longitudinal_separation','unused','faceoff_half_transverse_separation','unused'])
    scene['faceoff_parameterization']={'kind':'rows_mirror_across_transverse_rink_axis','mirror_axis':'C line, x=0',
        'same_rows':[['A1','A4'],['A2','A3']],'goal_positions_independent':True}
    for path in paths:
        mode=required_measurement(path['label'])
        path['measurement']=mode;path['measurement_policy_id']=POLICY_ID
        path['model_line_width']=0. if mode=='centreline' else None
        if path.get('generated_code') in (405,406,407,408,409):
            path['label']=path['label'].replace(':outer',':centreline')
            path['role']='physical_material_centreline'
        if path.get('generated_code') in (420,421,422,423):path['role']='physical_material_centreline'
    scene=constrain_symmetric_goal_ends(scene)
    if out_path is not None:_write_new(out_path,scene)
    return scene


def _cross2(a,b):return float(a[0]*b[1]-a[1]*b[0])


def sample_model(scene,core,samples=361):
    """Coordinates are generated exclusively by the shared native kernel."""
    if scene['schema']!='rinkspace.field_model.v1':raise ValueError('unsupported field scene schema')
    validate_scene_measurements(scene)
    shapes=[];by_code={};nodes=[]
    for old in scene['paths']:
        kind,parameters=native_definition(scene,old,core)
        count=samples if kind in (1,6) else 1025 if kind in (3,7) else 2
        station=np.linspace(0.,core.board_perimeter(scene['board']) if kind==7 else 1.,count)
        xyz=np.asarray(core.sample_path(scene['board'],kind,parameters,[],station),dtype=np.float64)
        if xyz.shape!=(count,3) or not np.isfinite(xyz).all():raise ValueError('invalid native entity '+old['id'])
        shape={**deepcopy(old),'kind':kind,'parameters':parameters,'xyz':xyz.tolist(),'native_generated':True}
        shapes.append(shape)
        if 'generated_code' in old:by_code[old['generated_code']]=shape
    def node(identity,xyz,parents,role,display,**extra):
        nodes.append({'id':identity,'xyz':list(map(float,xyz)),'parent_ids':parents,'role':role,
                      'display_kind':display,'source_uv':None,'source_observation_ids':[],
                      'new_source_observation':False,'per_frame_station_parameter':False,**extra})
    for code,p in by_code.items():
        if p['kind']==1:
            identity=p['label'].removeprefix('ice.').split(':')[0]
            if not any(n['id']==identity for n in nodes):
                node(identity,p['parameters'][:3],[p['entity_id']],
                     'geometric_circle_centre_not_new_source_UV','centre')
        if p['stage']=='M4':
            ends=np.asarray(p['xyz'])[[0,-1]]
            side='L' if code in (410,420,421) else 'R'
            for i,xyz in enumerate(ends):
                if code in (410,411):parents=p['endpoint_parent_ids'];name=f'K{side}{i+1}'
                else:parents=p['endpoint_parent_ids'][i];name=f'T{side}{"下" if code%2==0 else "上"}{"G" if i==0 else "B"}'
                node(name,xyz,parents,p['endpoint_roles'][i],'unmeasured_attachment',
                     evidence_role='geometric_attachment_with_no_source_endpoint_measurement')
    # These are the same unmeasured physical intersections returned by model
    # graph. The preview's hollow endpoint markers still create no source UV.
    for junction in carrier_boundary_junctions(shapes):
        nodes.append({**junction, "parent_ids": junction["parents"],
                      "display_kind": "derived_boundary_intersection"})
    # M2/M3 checks use native output, not a separately drawn reference model.
    def endpoints(code):return np.asarray(by_code[code]['parameters'],dtype=float).reshape(2,3)
    c=endpoints(400);b1=endpoints(401);b2=endpoints(402)
    direction=c[1]-c[0];length=np.linalg.norm(direction[:2])
    origin=np.asarray(by_code[405]['parameters'][:3])
    parallel=max(abs(_cross2(endpoints(k)[1]-endpoints(k)[0],direction))/length**2 for k in (401,402,403,404))
    short_axis_error=max(abs(endpoints(k)[1,0]-endpoints(k)[0,0]) for k in range(400,405))
    if scene.get('shared_geometry_options',0)&2 and short_axis_error>1e-12:
        raise ValueError('carrier is not parallel to the rink short side')
    c0_distance=abs(_cross2(origin-c[0],direction))/length
    blue_sym=np.linalg.norm((b1+b2)/2-c)
    radii=[by_code[k]['parameters'][3] for k in range(405,410)]
    m3={}
    for side,goal,codes,ellipse in [('left',403,(408,409),410),('right',404,(406,407),411)]:
        a,b=[np.asarray(by_code[k]['parameters'][:3]) for k in codes]
        g=endpoints(goal);d=g[1]-g[0];unit=d/np.linalg.norm(d)
        centre=np.asarray(by_code[ellipse]['parameters'][:3])
        distances=[abs(_cross2(x-g[0],d))/np.linalg.norm(d[:2]) for x in (a,b)]
        transverse=[float(np.dot(x-centre,unit)) for x in (a,b)]
        m3[side]={'inward_distance_difference':abs(distances[0]-distances[1]),
                  'transverse_pair_asymmetry':abs(sum(transverse)),
                  'inward_distance_initial':sum(distances)/2,'pair_transverse_half_separation_initial':abs(transverse[0]-transverse[1])/2}
    centres=[np.asarray(by_code[k]['parameters'][:3]) for k in range(406,410)]
    a1,a2,a3,a4=centres
    row_error=max(abs(a1[1]-a4[1]),abs(a2[1]-a3[1]))
    mirror_error=max(abs(a1[0]+a4[0]),abs(a2[0]+a3[0]))
    if scene.get('shared_geometry_options',0)&8 and max(row_error,mirror_error)>1e-12:
        raise ValueError('faceoff circles do not share rows and mirror about the transverse axis')
    m3['cross_end_rows']={'A1_A4_y_difference':float(abs(a1[1]-a4[1])),
        'A2_A3_y_difference':float(abs(a2[1]-a3[1])),'C_axis_mirror_error':float(mirror_error)}
    for row in range(4):
        line=by_code[200+row]['parameters'];n=by_code[301+row*3]['parameters'][0]
        r=by_code[119+row*4]['parameters'][0]
        if not r<line[3]<n:raise ValueError('M1 connection must extend past R and stop before N')
    goal_left_x=endpoints(403)[0,0]
    lane_left=[by_code[300+i*3]['parameters'][0] for i in range(4)]
    if scene.get('shared_geometry_options',0)&4 and min(lane_left)<goal_left_x-1e-12:
        raise ValueError('Lane drawing support crosses the left goal line')
    centre_rows=[by_code[103+i*4]['parameters'][:3] for i in range(4)]
    lower_gap=scene['board'][1]+centre_rows[0][1];upper_gap=scene['board'][1]-centre_rows[-1][1]
    if scene.get('shared_geometry_options',0)&1 and abs(lower_gap-upper_gap)>1e-12:
        raise ValueError('new centred curling constraint regressed')
    checks={'M2':{'C0_origin_error':float(np.linalg.norm(origin)),'C0_to_C_distance':float(c0_distance),
                   'carrier_parallel_error':float(parallel),'blue_symmetry_error':float(blue_sym),'short_side_direction_error':float(short_axis_error)},
            'M3':{'common_circle_radius_spread':float(np.ptp(radii)),**m3},
            'M1':{'connection_stops_before_N':True,'physical_edge_count':12,'connection_count':4,'ring_boundary_count':32,'lane_left_not_before_left_goal_line':True,'lane_left_support_values':lane_left,'bottom_long_side_centre_gap':lower_gap,'top_long_side_centre_gap':upper_gap,'long_side_gap_difference':abs(lower_gap-upper_gap)},
            'path_count':len(shapes),'new_source_observations':0,'fit_executed':False}
    if scene.get('shared_geometry_options',0)&16:
        left,right=[np.asarray(by_code[k]['parameters']) for k in (410,411)]
        mirror_x=np.array([-1.,1.,1.]);mirror_y=np.array([1.,-1.,1.])
        m4={
            'goal_line_mirror_error':float(np.max(abs(endpoints(403)*mirror_x-endpoints(404)))),
            'crease_axes_difference':float(np.max(abs(left[3:5]-right[3:5]))),
            'crease_centre_to_goal_midpoint_error':float(max(np.linalg.norm(by_code[k]['parameters'][:3]-endpoints(g).mean(0)) for k,g in ((410,403),(411,404)))),
            'crease_mirror_error':float(np.max(abs(np.asarray(by_code[410]['xyz'])*mirror_x-np.asarray(by_code[411]['xyz'])[::-1]))),
            'trapezoid_left_right_mirror_error':float(max(np.max(abs(endpoints(a)*mirror_x-endpoints(b))) for a,b in ((420,422),(421,423)))),
            'trapezoid_upper_lower_mirror_error':float(max(np.max(abs(endpoints(a)*mirror_y-endpoints(b))) for a,b in ((420,421),(422,423)))),
        }
        if max(m4.values())>1e-11:raise ValueError('Native symmetric goal-end relation failed: '+str(m4))
        checks['M4']=dict(**m4,contour_arc_only=True,closing_diameter_observable=False)
    errors=list(checks['M2'].values())+[checks['M3']['common_circle_radius_spread']]+[m3[s][k] for s in ('left','right') for k in ['inward_distance_difference','transverse_pair_asymmetry']]+([row_error,mirror_error] if scene.get('shared_geometry_options',0)&8 else [])
    if max(errors)>1e-11:raise ValueError('native structural relation failed: '+str(checks))
    return {'schema':'rinkspace.field_geometry.v1','scene_id':scene['scene_id'],'shapes':shapes,'nodes':nodes,
            'checks':checks,'dimensions_status':scene['dimensions_status'],'native_generated':True,'fit_executed':False}
