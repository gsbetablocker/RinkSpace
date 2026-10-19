"""Exact user relations under parameter changes and mesh export, using synthetic data.

These checks exercise structural invariants, not video accuracy or fitting.
"""
import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from rinkspace.calibration_companion import Camera, Scene, look_at_rvec
from rinkspace.calibration_companion.joint_scene import JointSceneProblem, SharedGeometry
from rinkspace.calibration_companion.structural_constraints import (
    A_PAIR, EIGHT_CIRCLES, HARD_GEOMETRY_PROFILE, LINE_PAIRS,
    normalize_hard_geometry_constraints, project_scene_to_hard_constraints,
    structural_state, apply_structural_state, hard_geometry_checks, validate_hard_parameter,
)


PROFILE = {'profile': HARD_GEOMETRY_PROFILE}
CORNERS = ('upper_left', 'upper_right', 'lower_left', 'lower_right')
_spec = importlib.util.spec_from_file_location(
    'hard_structure_export', Path(__file__).resolve().parents[1] / 'scripts/export_joint_scene_model.py')
exporter = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(exporter)


def raw_scene():
    features = []
    for index, fid in enumerate(EIGHT_CIRCLES):
        row = int(fid[1])
        center = [(-4. if fid[0] == 'F' else 4.) + .03 * row,
                  -3. + 2. * (row - 1) + (.04 if fid[0] == 'R' else -.04), 0.]
        radius = .60 + .02 * index
        features.append(dict(id=fid, kind='circle', center_xyz=center, radius=radius,
            rings=[dict(role='outer', center_xyz=center.copy(), radius=radius),
                   dict(role='inner', center_xyz=center.copy(), radius=.2)]))
    for fid, x in [('C', .25), ('B1', -2.5), ('B2', 3.1),
                   ('goal_line_left', -9.1), ('goal_line_right', 8.7)]:
        features.append(dict(id=fid, kind='polyline',
                             segments_xyz=[[[x, 6., 0.], [x, -6., 0.]]]))
    features.append(dict(id='C0', kind='circle', center_xyz=[.3, .7, 0.], radius=.9))
    t = np.linspace(-np.pi / 2, np.pi / 2, 25)
    left = np.column_stack([-9.1 + np.cos(t), np.sin(t), np.zeros(len(t))])
    right = left * [-1., 1., 1.] + [.1, .2, 0.]
    features.extend([dict(id='goal_crease_left', kind='polygon', polygon_xyz=left.tolist()),
                     dict(id='goal_crease_right', kind='polygon', polygon_xyz=right.tolist())])
    return dict(schema_version='rinkspace.scene3d.v1', model_id='synthetic_hard_structure',
        revision='source', coordinate_system={'unit': 'synthetic', 'metres_per_unit': None},
        parameters=dict(bounds=[-10., -6., 10., 6.], corner_radius=1.,
            corner_radii=dict(upper_left=3.3, upper_right=1., lower_left=1.3, lower_right=3.8),
            grid_spacing=1., arc_steps_per_quarter=12, board_height=.8, board_thickness=.1),
        ice_features=features, advertisements=[], source={},
        ice_boundary={'points_xyz': [[-10., -6., 0.], [10., -6., 0.], [10., 6., 0.], [-10., 6., 0.]]},
        board_surface={'perimeter_length': 64., 'panels': []},
        meshes=[dict(name='rink_boards', positions=[[-10., -6., 0.], [10., -6., 0.],
                                                   [-10., -6., .8], [10., -6., .8]])])


def load_scene(tmp_path, data, name='scene.json'):
    path = tmp_path / name
    path.write_text(json.dumps(data), encoding='utf8')
    return Scene(path)


def features_by_id(data):
    return {f['id']: f for f in data['ice_features']}


def raw_scene_with_lanes():
    data = raw_scene()
    for i in range(1, 5):
        center, half = -3. + 2. * (i - 1) + .4, .3 + i * .05
        data['ice_features'].append(dict(id=f'lane{i}', kind='polygon', closed=True,
            polygon_xyz=[[-6., center + half, 0.], [6., center + half, 0.],
                         [6., center - half, 0.], [-6., center - half, 0.]]))
    return data


LANE_PROFILE = {**PROFILE, 'lane_centerline_through_FR_rows': True, 'lane_left_open': True}
WIDE_PROFILE = {**LANE_PROFILE, 'equal_corner_radii': True,
                'lane_equal_width': True, 'lane_right_collinear': True}
A_PAIR_PROFILE = {**PROFILE, 'A1_A2_mirror_equal_radius': True}


def add_asymmetric_A_pair(data, *, x1=2.2, x2=2.8, y1=-2.3, y2=2.7,
                          r1=.7, r2=.9):
    data = copy.deepcopy(data)
    for fid, x, y, radius in (('A1', x1, y1, r1), ('A2', x2, y2, r2)):
        center = [x, y, 0.]
        data['ice_features'].append(dict(id=fid, kind='circle', center_xyz=center,
            radius=radius, rings=[dict(role='outer', center_xyz=center.copy(), radius=radius),
                                  dict(role='inner', center_xyz=center.copy(), radius=.25)]))
    return data


@pytest.mark.parametrize('values', [
    dict(x1=2.2, x2=2.8, y1=-2.3, y2=2.7, r1=.7, r2=.9),
    dict(x1=3.1, x2=1.8, y1=-1.7, y2=3.2, r1=1.1, r2=.45),
])
def test_A_pair_projection_is_exact_and_preserves_source(values):
    original = add_asymmetric_A_pair(raw_scene(), **values)
    before = copy.deepcopy(original)
    projected = project_scene_to_hard_constraints(original, A_PAIR_PROFILE)
    assert original == before
    features = features_by_id(projected)
    a1, a2 = (features[fid] for fid in A_PAIR)
    assert a1['center_xyz'][0] == a2['center_xyz'][0] == (values['x1'] + values['x2']) / 2
    assert a1['center_xyz'][1] == -a2['center_xyz'][1]
    assert a1['radius'] == a2['radius'] == (values['r1'] + values['r2']) / 2
    assert all(ring['center_xyz'] == feature['center_xyz'] for feature in (a1, a2)
               for ring in feature['rings'])
    assert all(next(r['radius'] for r in feature['rings'] if r['role'] == 'outer') == feature['radius']
               for feature in (a1, a2))
    assert projected['hard_geometry_projection']['checks']['all_constraints_satisfied']
    assert all(features[fid]['geometry_status'] == 'user_hard_relation_projection_not_video_refit'
               for fid in A_PAIR)
    a2['center_xyz'][1] += .1
    pars = projected['parameters']
    assert not hard_geometry_checks(features, pars, pars['corner_radii'], A_PAIR_PROFILE)['all_constraints_satisfied']


def test_A_pair_flag_requires_both_circles_and_cannot_be_erased():
    with pytest.raises(ValueError, match='missing features'):
        project_scene_to_hard_constraints(raw_scene(), A_PAIR_PROFILE)
    data = project_scene_to_hard_constraints(add_asymmetric_A_pair(raw_scene()), A_PAIR_PROFILE)
    active = normalize_hard_geometry_constraints(data, {'hard_geometry_constraints':
        {**PROFILE, 'A1_A2_mirror_equal_radius': False}})
    assert active['A1_A2_mirror_equal_radius']
    for kind in ('circle_center_x', 'circle_center_y', 'circle_radius'):
        for fid in A_PAIR:
            with pytest.raises(ValueError, match='A1/A2'):
                validate_hard_parameter(dict(kind=kind, feature_id=fid), data['parameters'], active)
    for kind in ('circle_shared_center_x', 'circle_shared_center_y', 'circle_shared_radius'):
        with pytest.raises(ValueError, match='A1/A2'):
            validate_hard_parameter(dict(kind=kind, feature_ids=list(A_PAIR)), data['parameters'], active)
    for kind in ('A_pair_mirror_y_distance', 'A_pair_shared_radius'):
        with pytest.raises(ValueError, match='positive'):
            validate_hard_parameter(dict(kind=kind, feature_ids=list(A_PAIR), bounds=[0., 5.]),
                                    data['parameters'], active)
    with pytest.raises(ValueError, match='feature_ids'):
        validate_hard_parameter(dict(kind='A_pair_shared_center_x', feature_ids=['A1']),
                                data['parameters'], active)
    with pytest.raises(ValueError, match='outer'):
        validate_hard_parameter(dict(kind='A_pair_shared_radius', feature_ids=list(A_PAIR),
                                     ring_role='inner'), data['parameters'], active)


def test_undeclared_lane_relations_do_not_change_legacy_geometry():
    original = raw_scene_with_lanes()
    projected = project_scene_to_hard_constraints(original, PROFILE)
    before, after = features_by_id(original), features_by_id(projected)
    for i in range(1, 5):
        assert before[f'lane{i}'] == after[f'lane{i}']
    assert not any(key.startswith('lane') for key in projected['hard_geometry_projection']['checks']['checks'])


def test_declared_lane_projection_preserves_independent_widths_and_opens_left():
    original = raw_scene_with_lanes(); old = copy.deepcopy(original)
    data = project_scene_to_hard_constraints(original, LANE_PROFILE)
    assert original == old
    features = features_by_id(data)
    for i in range(1, 5):
        f = features[f'lane{i}']; poly = np.array(f['polygon_xyz'])
        assert f['closed'] is False and f['open_edges'] == [3]
        np.testing.assert_allclose((poly[0, 1] + poly[2, 1]) / 2., features[f'F{i}']['center_xyz'][1])
        np.testing.assert_allclose((poly[0, 1] - poly[2, 1]) / 2., .3 + i * .05)
    assert data['hard_geometry_projection']['checks']['all_constraints_satisfied']


def test_lane_row_updates_preserve_current_parameterized_half_width():
    data = project_scene_to_hard_constraints(raw_scene_with_lanes(), LANE_PROFILE)
    features = features_by_id(data); pars = data['parameters']; corners = pars['corner_radii']
    state = structural_state(features, pars, LANE_PROFILE)
    state['row_gaps'][0] += .25
    poly = np.array(features['lane2']['polygon_xyz'])
    center = (poly[0, 1] + poly[2, 1]) / 2.
    poly[[0, 1], 1] = center + .63; poly[[2, 3], 1] = center - .63
    features['lane2']['polygon_xyz'] = poly.tolist()
    apply_structural_state(features, pars, corners, state)
    poly = np.array(features['lane2']['polygon_xyz'])
    np.testing.assert_allclose((poly[0, 1] - poly[2, 1]) / 2., .63)
    np.testing.assert_allclose((poly[0, 1] + poly[2, 1]) / 2., -.75)
    assert hard_geometry_checks(features, pars, corners, LANE_PROFILE)['all_constraints_satisfied']
    features['lane2']['polygon_xyz'][0][1] += .1
    assert not hard_geometry_checks(features, pars, corners, LANE_PROFILE)['all_constraints_satisfied']


def test_eight_lane_left_endpoints_remain_independent_through_projection_and_parameters(tmp_path):
    data = raw_scene_with_lanes()
    expected = {}
    for row in range(1, 5):
        lane = next(f for f in data['ice_features'] if f['id'] == f'lane{row}')
        upper, lower = -8.4 + .15 * row, -7.7 + .11 * row
        lane['polygon_xyz'][0][0] = upper
        lane['polygon_xyz'][3][0] = lower
        lane['left_endpoint_x_intervals_scene_units'] = {
            'upper': [-8.5, -5.], 'lower': [-8.5, -5.]}
        lane['left_endpoint_status'] = 'unresolved'
        lane['left_endpoint_coordinate_roles'] = {
            'upper': 'image_fit_candidate_unvalidated',
            'lower': 'image_fit_candidate_unvalidated'}
        expected[row] = (upper, lower)
    projected = project_scene_to_hard_constraints(data, WIDE_PROFILE)
    assert projected['hard_geometry_projection']['checks']['all_constraints_satisfied']
    features = features_by_id(projected)
    for row in range(1, 5):
        polygon = np.asarray(features[f'lane{row}']['polygon_xyz'])
        np.testing.assert_allclose(polygon[[0, 3], 0], expected[row])
        assert polygon[0, 0] != polygon[3, 0]
    scene = load_scene(tmp_path, projected, 'independent_lane_left_ends.json')
    specs = [dict(id='lane2_upper', kind='lane_left_endpoint_x', feature_id='lane2',
                  edge='upper', bounds=[-8.5, -5.]),
             dict(id='lane2_lower', kind='lane_left_endpoint_x', feature_id='lane2',
                  edge='lower', bounds=[-8.5, -5.])]
    geometry = SharedGeometry(scene, specs, np.array([-8.0, -6.5]))
    lane2 = np.asarray(geometry.features['lane2']['polygon_xyz'])
    np.testing.assert_allclose(lane2[[0, 3], 0], [-8.0, -6.5])
    assert geometry.hard_geometry_checks()['all_constraints_satisfied']
    for row in (1, 3, 4):
        polygon = np.asarray(geometry.features[f'lane{row}']['polygon_xyz'])
        np.testing.assert_allclose(polygon[[0, 3], 0], expected[row])
    outside = SharedGeometry(scene, specs, np.array([-8.0, -4.0]))
    assert not outside.hard_geometry_checks()['all_constraints_satisfied']
    assert not outside.hard_geometry_checks()['checks']['lane2_lower_candidate_inside_interval']


def test_unresolved_lane_proxy_cannot_be_used_as_fitted_finite_endpoint(tmp_path, wide_hard_problem):
    original_scene, original_problem = wide_hard_problem
    scene_data = copy.deepcopy(original_scene.data)
    lane = features_by_id(scene_data)['lane1']
    lane['left_endpoint_status'] = 'unresolved'
    lane['left_endpoint_x_intervals_scene_units'] = {
        'upper': [-8.5, -5.], 'lower': [-8.5, -5.]}
    scene = load_scene(tmp_path, scene_data, 'unresolved_lane_proxy.json')
    problem = copy.deepcopy(original_problem)
    problem['scene_sha256'] = scene.sha256
    with pytest.raises(ValueError, match='unresolved lane1 upper left endpoint'):
        JointSceneProblem(scene, problem)

    # Infinite support lines use only line position, so they remain valid while
    # neither physical left endpoint has been observed.
    problem['observations'] = [o for o in problem['observations'] if o['id'] != 'lane1_outline']
    JointSceneProblem(scene, problem)

    # A finite outline can be used only when both endpoint variables explicitly
    # cite its fit observation as evidence.
    problem['observations'].append(next(o for o in original_problem['observations']
                                        if o['id'] == 'lane1_outline'))
    problem['scene_parameters'].extend([
        dict(id=f'lane1_left_{edge}', kind='lane_left_endpoint_x', feature_id='lane1',
             edge=edge, bounds=[-8.5, -5.], evidence_observation_ids=['lane1_outline'])
        for edge in ('upper', 'lower')])
    JointSceneProblem(scene, problem)


def test_unresolved_lane_intervals_and_core_proxy_follow_current_goal_and_F(tmp_path):
    data = raw_scene_with_lanes()
    for row in range(1, 5):
        lane = next(f for f in data['ice_features'] if f['id'] == f'lane{row}')
        lane['left_endpoint_status'] = 'unresolved'
        lane['left_endpoint_x_intervals_scene_units'] = {
            'upper': [-9.1, -4.6], 'lower': [-9.1, -4.6]}
        lane['left_endpoint_x_interval_bound_sources'] = {
            'lower_x': {'feature_id': 'goal_line_left', 'x_scene_units': -9.1},
            'upper_x': {'feature_id': f'F{row}', 'x_scene_units': -4.6}}
    projected = project_scene_to_hard_constraints(data, WIDE_PROFILE)
    scene = load_scene(tmp_path, projected, 'relative_lane_left_intervals.json')
    initial_interval = scene.features['lane1']['left_endpoint_x_intervals_scene_units']['upper']
    specs = [dict(kind='circle_shared_radius', feature_ids=list(EIGHT_CIRCLES)),
             dict(kind='symmetric_line_distance', feature_ids=['goal_line_left', 'goal_line_right'])]
    geometry = SharedGeometry(scene, specs, np.array([.9, 9.2]))
    goal_x = geometry.features['goal_line_left']['segments_xyz'][0][0][0]
    F_leftmost_x = geometry.features['F1']['center_xyz'][0] - geometry.features['F1']['radius']
    assert [goal_x, F_leftmost_x] != initial_interval
    for row in range(1, 5):
        lane = geometry.features[f'lane{row}']
        for edge, vertex in (('upper', 0), ('lower', 3)):
            np.testing.assert_allclose(lane['left_endpoint_x_intervals_scene_units'][edge],
                                       [goal_x, F_leftmost_x])
            assert lane['left_endpoint_coordinate_roles'][edge] == 'guaranteed_core_proxy'
            np.testing.assert_allclose(lane['polygon_xyz'][vertex][0], F_leftmost_x)
        assert lane['left_endpoint_x_interval_bound_sources']['lower_x']['x_scene_units'] == goal_x
        assert lane['left_endpoint_x_interval_bound_sources']['upper_x']['x_scene_units'] == F_leftmost_x
    assert geometry.hard_geometry_checks()['all_constraints_satisfied']
    stale_features = copy.deepcopy(geometry.features)
    stale_features['lane2']['left_endpoint_x_interval_bound_sources']['upper_x']['x_scene_units'] = initial_interval[1]
    stale_checks = hard_geometry_checks(stale_features, geometry.parameters, geometry.corner_radii,
                                        geometry.hard_geometry_constraints)
    assert not stale_checks['checks']['lane2_left_interval_bound_sources_follow_current_geometry']

    candidate = SharedGeometry(scene, specs + [dict(kind='lane_left_endpoint_x',
        feature_id='lane2', edge='upper')], np.array([.9, 9.2, -7.8]))
    lane2 = candidate.features['lane2']
    assert lane2['left_endpoint_coordinate_roles']['upper'] == 'image_fit_candidate_unvalidated'
    assert lane2['polygon_xyz'][0][0] == -7.8
    assert lane2['polygon_xyz'][3][0] == F_leftmost_x
    assert candidate.hard_geometry_checks()['all_constraints_satisfied']

def test_declared_lane_rules_cannot_be_erased_or_bypassed_by_independent_y_bound():
    data = project_scene_to_hard_constraints(raw_scene_with_lanes(), LANE_PROFILE)
    active = normalize_hard_geometry_constraints(data, {'hard_geometry_constraints':
        {**PROFILE, 'lane_centerline_through_FR_rows': False, 'lane_left_open': False}})
    assert active['lane_centerline_through_FR_rows'] and active['lane_left_open']
    spec = dict(kind='lane_bound_y_max', feature_id='lane2', bounds=[-2., 1.])
    validate_hard_parameter(spec, data['parameters'], PROFILE)  # unchanged legacy meaning
    with pytest.raises(ValueError, match='centerline'):
        validate_hard_parameter(spec, data['parameters'], LANE_PROFILE)
    validate_hard_parameter(dict(kind='lane_half_width', feature_id='lane2', bounds=[.1, 1.]), data['parameters'], LANE_PROFILE)


def test_wide_hard_projection_equalizes_corners_widths_and_right_edges_only():
    original = raw_scene_with_lanes()
    for i in range(1, 5):
        polygon = np.asarray(original['ice_features'][-5+i]['polygon_xyz'], float)
        polygon[[0, 3], 0] -= .12 * i
        polygon[[1, 2], 0] += .2 * i
        original['ice_features'][-5+i]['polygon_xyz'] = polygon.tolist()
    source = copy.deepcopy(original)
    projected = project_scene_to_hard_constraints(original, WIDE_PROFILE)
    assert original == source
    features = features_by_id(projected)
    radii = projected['parameters']['corner_radii']
    assert len(set(radii.values())) == 1
    assert radii[CORNERS[0]] == projected['parameters']['corner_radius']
    widths, right_xs, left_xs = [], [], []
    for i in range(1, 5):
        polygon = np.asarray(features[f'lane{i}']['polygon_xyz'])
        widths.append(polygon[0, 1] - polygon[2, 1])
        right_xs.append(polygon[1, 0]); left_xs.append(polygon[0, 0])
        assert features[f'lane{i}']['open_edges'] == [3]
        assert (polygon[0, 1] + polygon[2, 1]) / 2 == features[f'F{i}']['center_xyz'][1]
    np.testing.assert_allclose(widths, widths[0])
    np.testing.assert_allclose(right_xs, right_xs[0])
    assert len(set(left_xs)) == 4
    assert projected['hard_geometry_projection']['checks']['all_constraints_satisfied']


def test_wide_hard_flags_cannot_be_erased_and_independent_parameters_are_rejected():
    data = project_scene_to_hard_constraints(raw_scene_with_lanes(), WIDE_PROFILE)
    active = normalize_hard_geometry_constraints(data, {'hard_geometry_constraints':
        {**PROFILE, 'equal_corner_radii': False, 'lane_equal_width': False,
         'lane_right_collinear': False}})
    assert all(active[key] for key in ('equal_corner_radii', 'lane_equal_width', 'lane_right_collinear'))
    bad = [('boundary_corner_radius', 'corner', 'upper_right'),
           ('lane_half_width', 'feature_id', 'lane1'),
           ('lane_bound_x_max', 'feature_id', 'lane2')]
    for kind, key, value in bad:
        with pytest.raises(ValueError):
            validate_hard_parameter({'kind': kind, key: value}, data['parameters'], active)
    for spec in ({'kind': 'shared_corner_radius', 'bounds': [.1, 5.]},
                 {'kind': 'shared_lane_half_width', 'feature_ids': list(('lane1', 'lane2', 'lane3', 'lane4')),
                  'bounds': [.1, 3.]},
                 {'kind': 'shared_lane_right_x', 'feature_ids': list(('lane1', 'lane2', 'lane3', 'lane4')),
                  'bounds': [1., 9.]}):
        validate_hard_parameter(spec, data['parameters'], active)


def test_open_lane_export_clips_three_painted_sides_without_filling_or_recentering():
    data = raw_scene_with_lanes()
    lane = next(f for f in data['ice_features'] if f['id'] == 'lane4')
    for p in lane['polygon_xyz']:
        p[0] = -12. if p[0] < 0 else 11.
    semantic = project_scene_to_hard_constraints(data, LANE_PROFILE)
    original = copy.deepcopy(features_by_id(semantic)['lane4']['polygon_xyz'])
    exported, _, checks = exporter.rebuild_scene(semantic)
    lane = features_by_id(exported)['lane4']
    assert lane['polygon_xyz'] == original
    assert lane['closed'] is False and lane['open_edges'] == [3]
    assert lane['paint_segment_source_edge_indices'] == [0, 2]  # right short side is outside ice
    assert len(lane['paint_segments_xyz']) == 2
    assert not any(np.allclose(np.array(s)[:, 0], -12.) for s in lane['paint_segments_xyz'])
    assert exported['hard_geometry_checks']['all_constraints_satisfied']
    assert checks['semantic_checks']['all_ice_features_at_z0_inside_rink']
    reprojected = project_scene_to_hard_constraints(exported, LANE_PROFILE)
    assert features_by_id(reprojected)['lane4']['polygon_xyz'] == original


@pytest.fixture
def hard_problem(tmp_path):
    data = project_scene_to_hard_constraints(raw_scene(), PROFILE)
    scene = load_scene(tmp_path, data)
    truth = SharedGeometry(scene, [], np.array([]))
    center = np.array([22., -25., 18.])
    rvec = look_at_rvec(center, [0., 0., 0.])
    camera = Camera(center, rvec, [[850., 0., 640.], [0., 850., 360.], [0., 0., 1.]], [])
    observations = []

    def observe(oid, target):
        xyz = truth.points(target, 'curve', 24)
        observations.append(dict(id=oid, frame_id='view', type='curve', use='fit',
            target=target, pixels_uv=camera.project(xyz)[0].tolist(), sigma_px=1.))

    for fid in (*EIGHT_CIRCLES, 'C0'):
        observe(fid, dict(kind='circle_outline', feature_id=fid))
    for fid in ('C', 'B1', 'B2', 'goal_line_left', 'goal_line_right'):
        observe(fid, dict(kind='line_segment', feature_id=fid))
    for fid in ('goal_crease_left', 'goal_crease_right'):
        observe(fid, dict(kind='polygon_edge', feature_id=fid, index=0))
    for corner in CORNERS:
        observe(corner, dict(kind='board_corner', corner=corner, edge='top'))
    specs = [
        dict(id='F_x', kind='circle_shared_center_x', feature_ids=[f'F{i}' for i in range(1, 5)], bounds=[-8., -1.]),
        dict(id='R_x', kind='circle_shared_center_x', feature_ids=[f'R{i}' for i in range(1, 5)], bounds=[1., 8.]),
        dict(id='row_origin', kind='circle_row_origin_y', feature_ids=list(EIGHT_CIRCLES), bounds=[-4., -2.]),
        dict(id='radius', kind='circle_shared_radius', feature_ids=list(EIGHT_CIRCLES), bounds=[.2, 1.2]),
        dict(id='C0_radius', kind='circle_radius', feature_id='C0', bounds=[.3, 1.5]),
        dict(id='blue_distance', kind='symmetric_line_distance', feature_ids=['B1', 'B2'], bounds=[1., 4.]),
        dict(id='goal_distance', kind='symmetric_line_distance', feature_ids=['goal_line_left', 'goal_line_right'], bounds=[7., 9.5]),
        dict(id='length', kind='boundary_length', bounds=[19., 26.]),
        dict(id='width', kind='boundary_width', bounds=[11., 16.]),
    ]
    specs += [dict(id=f'gap{row}', kind='circle_row_gap', after_row=row,
                   feature_ids=[f'{col}{i}' for col in 'FR' for i in range(row + 1, 5)],
                   bounds=[.5, 2.5]) for row in (1, 2, 3)]
    specs += [dict(id=corner, kind='boundary_corner_radius', corner=corner, bounds=[.5, 4.5])
              for corner in CORNERS]
    for spec in specs:
        if spec['kind'] in ('boundary_length', 'boundary_width'):
            spec['evidence_observation_ids'] = list(CORNERS)
        elif spec['kind'] == 'boundary_corner_radius':
            spec['evidence_observation_ids'] = [spec['corner']]
        else:
            spec['evidence_observation_ids'] = spec.get('feature_ids', [spec.get('feature_id')])
    problem = dict(synthetic=True, scene_sha256=scene.sha256, geometry_mode='shared_scene_joint',
        scene_parameters=specs, installations=[dict(id='camera', initial_center_xyz=center.tolist(), fit_center=False)],
        frames=[dict(id='view', installation_id='camera', image_size_wh=[1280, 720],
            principal_point_px=[640., 360.], initial_rvec_world_to_camera=rvec.tolist(),
            initial_focal_px=850., fit_pose=False)], observations=observations, curve_samples=64)
    return scene, problem


@pytest.fixture
def wide_hard_problem(tmp_path, hard_problem):
    _, problem = hard_problem
    source = project_scene_to_hard_constraints(raw_scene_with_lanes(), WIDE_PROFILE)
    scene = load_scene(tmp_path, source, 'wide_hard_scene.json')
    problem = copy.deepcopy(problem)
    problem['scene_sha256'] = scene.sha256
    problem['scene_parameters'] = [s for s in problem['scene_parameters']
                                   if s['kind'] != 'boundary_corner_radius']
    problem['scene_parameters'].append(dict(id='corner_shared', kind='shared_corner_radius',
        bounds=[.5, 4.5], evidence_observation_ids=list(CORNERS)))
    problem['scene_parameters'].append(dict(id='lane_width_shared', kind='shared_lane_half_width',
        feature_ids=list(('lane1', 'lane2', 'lane3', 'lane4')), bounds=[.1, 2.],
        evidence_observation_ids=[f'lane{i}_upper' for i in range(1, 5)]))
    problem['scene_parameters'].append(dict(id='lane_right_shared', kind='shared_lane_right_x',
        feature_ids=list(('lane1', 'lane2', 'lane3', 'lane4')), bounds=[1., 9.],
        evidence_observation_ids=[f'lane{i}_right' for i in range(1, 5)]))
    truth = SharedGeometry(scene, [], np.empty(0))
    frame = problem['frames'][0]
    camera = Camera(problem['installations'][0]['initial_center_xyz'],
        frame['initial_rvec_world_to_camera'],
        [[850., 0., 640.], [0., 850., 360.], [0., 0., 1.]], [])
    for i in range(1, 5):
        for name, index in [('upper', 0), ('right', 1), ('outline', None)]:
            target = (dict(kind='polygon_outline', feature_id=f'lane{i}') if index is None else
                      dict(kind='polygon_supporting_line', feature_id=f'lane{i}', index=index))
            xyz = truth.points(target, 'curve', 24)
            problem['observations'].append(dict(id=f'lane{i}_{name}', frame_id='view',
                type='curve', use='fit', target=target, pixels_uv=camera.project(xyz)[0].tolist(),
                sigma_px=1.))
    return scene, problem


@pytest.fixture
def A_pair_problem(tmp_path, hard_problem):
    base_scene, base_problem = hard_problem
    source = project_scene_to_hard_constraints(add_asymmetric_A_pair(base_scene.data), A_PAIR_PROFILE)
    scene = load_scene(tmp_path, source, 'A_pair_scene.json')
    problem = copy.deepcopy(base_problem)
    problem['scene_sha256'] = scene.sha256
    truth = SharedGeometry(scene, [], np.empty(0))
    frame = problem['frames'][0]
    camera = Camera(problem['installations'][0]['initial_center_xyz'],
        frame['initial_rvec_world_to_camera'],
        [[850., 0., 640.], [0., 850., 360.], [0., 0., 1.]], [])
    for fid in A_PAIR:
        target = dict(kind='circle_outline', feature_id=fid)
        xyz = truth.points(target, 'curve', 24)
        problem['observations'].append(dict(id=fid, frame_id='view', type='curve', use='fit',
            target=target, pixels_uv=camera.project(xyz)[0].tolist(), sigma_px=1.))
    for spec in (
        dict(id='A_x', kind='A_pair_shared_center_x', bounds=[1., 5.]),
        dict(id='A_y_distance', kind='A_pair_mirror_y_distance', bounds=[.5, 4.]),
        dict(id='A_radius', kind='A_pair_shared_radius', bounds=[.2, 1.5]),
    ):
        problem['scene_parameters'].append({**spec, 'feature_ids': list(A_PAIR),
            'evidence_observation_ids': list(A_PAIR)})
    return scene, problem


def test_A_pair_shared_parameters_survive_perturbation_and_export(A_pair_problem, tmp_path):
    scene, data = A_pair_problem
    problem = JointSceneProblem(scene, data)
    for name in ('A_x', 'A_y_distance', 'A_radius'):
        assert 'scene:' + name in problem.parameter_names
    x = changed(problem, A_x=3.4, A_y_distance=1.8, A_radius=1.05, gap2=2.2)
    geometry = problem.geometry(x)
    a1, a2 = (geometry.features[fid] for fid in A_PAIR)
    assert geometry.hard_geometry_checks()['all_constraints_satisfied']
    assert a1['center_xyz'] == [3.4, -1.8, 0.]
    assert a2['center_xyz'] == [3.4, 1.8, 0.]
    assert a1['radius'] == a2['radius'] == 1.05
    for feature in (a1, a2):
        assert all(ring['center_xyz'] == feature['center_xyz'] for ring in feature['rings'])
    exported = problem.export_scene_data(x)
    assert exported['hard_geometry_constraints']['A1_A2_mirror_equal_radius']
    assert exported['hard_geometry_checks']['all_constraints_satisfied']
    rebuilt, _, checks = exporter.rebuild_scene(exported)
    assert rebuilt['hard_geometry_checks']['all_constraints_satisfied']
    assert checks['semantic_checks']['all_ice_features_at_z0_inside_rink']
    assert features_by_id(rebuilt)['A1']['center_xyz'] == [3.4, -1.8, 0.]
    assert features_by_id(rebuilt)['A2']['center_xyz'] == [3.4, 1.8, 0.]
    reloaded = load_scene(tmp_path, rebuilt, 'A_pair_rebuilt.json')
    assert SharedGeometry(reloaded, [], np.empty(0)).hard_geometry_checks()['all_constraints_satisfied']


def test_A_pair_sparse_dependencies_and_independent_spec_rejection(A_pair_problem):
    scene, data = A_pair_problem
    problem = JointSceneProblem(scene, data)
    sparse = problem.jac_sparsity.toarray()
    for name in ('A_x', 'A_y_distance', 'A_radius'):
        index = problem.parameter_names.index('scene:' + name)
        plus, minus = problem.x0.copy(), problem.x0.copy()
        plus[index] += 1e-6; minus[index] -= 1e-6
        derivative = (problem.residual(plus) - problem.residual(minus)) / 2e-6
        assert np.any(np.abs(derivative) > 1e-5), name
        assert np.all(sparse[np.abs(derivative) > 1e-5, index]), name
    for kind in ('circle_center_x', 'circle_center_y', 'circle_radius'):
        bad = copy.deepcopy(data)
        bad['scene_parameters'].append(dict(id='independent_A1', kind=kind, feature_id='A1',
            bounds=[.1, 5.], evidence_observation_ids=['A1']))
        with pytest.raises(ValueError, match='A1/A2'):
            JointSceneProblem(scene, bad)


def test_wide_shared_parameters_preserve_constraints_under_perturbation(wide_hard_problem, tmp_path):
    scene, data = wide_hard_problem
    problem = JointSceneProblem(scene, data)
    assert all(f'scene:{name}' in problem.parameter_names for name in
               ('corner_shared', 'lane_width_shared', 'lane_right_shared'))
    x = changed(problem, corner_shared=1.8, lane_width_shared=.6,
                lane_right_shared=6.8, gap2=2.2)
    geometry = problem.geometry(x)
    assert geometry.hard_geometry_checks()['all_constraints_satisfied']
    assert set(geometry.corner_radii.values()) == {1.8}
    assert geometry.parameters['corner_radius'] == 1.8
    left_xs = []
    for i in range(1, 5):
        polygon = np.asarray(geometry.features[f'lane{i}']['polygon_xyz'])
        np.testing.assert_allclose(np.ptp(polygon[:, 1]), 1.2)
        np.testing.assert_allclose(polygon[[1, 2], 0], 6.8)
        left_xs.append(polygon[0, 0])
    assert len(set(left_xs)) == 1  # independent left parameters are tested below
    exported = problem.export_scene_data(x)
    assert exported['parameters']['corner_radii'] == {corner: 1.8 for corner in CORNERS}
    assert exported['hard_geometry_checks']['all_constraints_satisfied']
    rebuilt, _, mesh_checks = exporter.rebuild_scene(exported)
    assert rebuilt['hard_geometry_checks']['all_constraints_satisfied']
    assert mesh_checks['semantic_checks']['all_ice_features_at_z0_inside_rink']
    assert len(set(rebuilt['parameters']['corner_radii'].values())) == 1
    reloaded = load_scene(tmp_path, exported, 'wide_hard_export.json')
    assert SharedGeometry(reloaded, [], np.empty(0)).hard_geometry_checks()['all_constraints_satisfied']


def test_wide_shared_parameter_sparsity_and_free_left_edges(wide_hard_problem):
    scene, data = wide_hard_problem
    data = copy.deepcopy(data)
    data['scene_parameters'].append(dict(id='lane2_left', kind='lane_bound_x_min',
        feature_id='lane2', bounds=[-8., -1.], evidence_observation_ids=['lane2_outline']))
    problem = JointSceneProblem(scene, data)
    x = changed(problem, corner_shared=1.9, lane_width_shared=.7,
                lane_right_shared=6.7, lane2_left=-5.7)
    geometry = problem.geometry(x)
    assert geometry.hard_geometry_checks()['all_constraints_satisfied']
    assert geometry.features['lane2']['polygon_xyz'][0][0] == -5.7
    for i in (1, 3, 4):
        assert geometry.features[f'lane{i}']['polygon_xyz'][0][0] == -6.
    sparse = problem.jac_sparsity.toarray()
    for name in ('corner_shared', 'lane_width_shared', 'lane_right_shared', 'lane2_left'):
        index = problem.parameter_names.index('scene:' + name)
        plus, minus = x.copy(), x.copy()
        plus[index] += 1e-6; minus[index] -= 1e-6
        derivative = (problem.residual(plus) - problem.residual(minus)) / 2e-6
        assert np.any(np.abs(derivative) > 1e-5), name
        assert np.all(sparse[np.abs(derivative) > 1e-5, index]), name


def changed(problem, **values):
    x = problem.x0.copy()
    for name, value in values.items():
        x[problem.parameter_names.index('scene:' + name)] = value
    return x


def assert_structure(geometry):
    checks = geometry.hard_geometry_checks()
    assert checks['all_constraints_satisfied'], checks
    f = geometry.features
    for col in 'FR':
        assert len({f[f'{col}{i}']['center_xyz'][0] for i in range(1, 5)}) == 1
    assert all(f[f'F{i}']['center_xyz'][1] == f[f'R{i}']['center_xyz'][1] for i in range(1, 5))
    assert len({f[fid]['radius'] for fid in EIGHT_CIRCLES}) == 1
    for left, right in LINE_PAIRS:
        a, b = np.array(f[left]['segments_xyz']), np.array(f[right]['segments_xyz'])
        np.testing.assert_allclose(a * [-1., 1., 1.], b, atol=1e-12)


def test_projection_is_explicit_deterministic_and_leaves_source_unchanged(tmp_path):
    original = raw_scene(); before = copy.deepcopy(original)
    projected = project_scene_to_hard_constraints(original, PROFILE)
    assert original == before
    assert projected['hard_geometry_projection']['video_fitting_run'] is False
    assert projected['hard_geometry_projection']['checks']['all_constraints_satisfied']
    assert projected['parameters'] == before['parameters']
    assert_structure(SharedGeometry(load_scene(tmp_path, projected), [], np.array([])))


def test_all_free_structural_parameters_preserve_relations(hard_problem):
    scene, data = hard_problem; problem = JointSceneProblem(scene, data)
    x = changed(problem, F_x=-4.4, R_x=4.8, row_origin=-3.5, gap1=1.7, gap2=2.3, gap3=1.5,
        radius=.9, C0_radius=1.2, blue_distance=3.2, goal_distance=9.2,
        length=21., width=13., upper_left=3.8, upper_right=1.1, lower_left=1.8, lower_right=4.1)
    geometry = problem.geometry(x); assert_structure(geometry)
    np.testing.assert_allclose([geometry.features[f'F{i}']['center_xyz'][1] for i in range(1, 5)], [-3.5, -1.8, .5, 2.])
    assert geometry.features['C0']['center_xyz'] == [0., 0., 0.]
    assert geometry.features['C0']['radius'] == 1.2
    assert len(set(geometry.corner_radii.values())) == 4
    assert problem.export_scene_data(x)['hard_geometry_checks']['all_constraints_satisfied']


@pytest.mark.parametrize('row', [1, 2, 3])
def test_gap_moves_exactly_its_downstream_pairs(hard_problem, row):
    scene, data = hard_problem; problem = JointSceneProblem(scene, data)
    old = problem.geometry(problem.x0)
    new = problem.geometry(changed(problem, **{f'gap{row}': 2.3}))
    for col in 'FR':
        for i in range(1, 5):
            delta = np.array(new.features[f'{col}{i}']['center_xyz']) - old.features[f'{col}{i}']['center_xyz']
            np.testing.assert_allclose(delta, [0., .3 if i > row else 0., 0.], atol=1e-12)


def test_goal_distance_moves_both_attached_creases(hard_problem):
    scene, data = hard_problem; problem = JointSceneProblem(scene, data)
    old = problem.geometry(problem.x0)
    new = problem.geometry(changed(problem, goal_distance=9.2))
    for fid, dx in [('goal_crease_left', -.3), ('goal_crease_right', .3)]:
        a = np.array(new.features[fid]['polygon_xyz']); b = np.array(old.features[fid]['polygon_xyz'])
        np.testing.assert_allclose(a-b, np.broadcast_to([dx, 0., 0.], a.shape), atol=1e-12)
    assert_structure(new)


@pytest.mark.parametrize('override', [None, False, {}, {'enabled': False}])
def test_problem_cannot_disable_scene_profile(hard_problem, override):
    scene, data = hard_problem
    data['hard_geometry_constraints'] = override
    problem = JointSceneProblem(scene, data)
    assert problem.hard_geometry_constraints['enabled']
    assert_structure(problem.geometry(changed(problem, goal_distance=9.3, gap1=.8)))


@pytest.mark.parametrize('spec', [
    dict(kind='circle_center_x', feature_id='F1', bounds=[-8., -1.]),
    dict(kind='circle_center_y', feature_id='R2', bounds=[-5., 5.]),
    dict(kind='circle_radius', feature_id='F4', bounds=[.1, 1.]),
    dict(kind='circle_center_x', feature_id='C0', bounds=[-1., 1.]),
    dict(kind='circle_center_y', feature_id='C0', bounds=[-1., 1.]),
    dict(kind='circle_shared_center_y', feature_ids=['F1', 'R1'], bounds=[-4., -2.]),
    *[dict(kind='line_offset', feature_id=fid, bounds=[-.5, .5])
      for fid in ['C', 'B1', 'B2', 'goal_line_left', 'goal_line_right']],
    dict(kind='circle_shared_center_x', feature_ids=['F1', 'F2'], bounds=[-8., -1.]),
    dict(kind='circle_shared_center_x', feature_ids=[f'F{i}' for i in range(1, 5)], bounds=[-8., 0.]),
    dict(kind='circle_shared_radius', feature_ids=['F1', 'R1'], bounds=[.1, 1.]),
    dict(kind='circle_row_gap', after_row=3, feature_ids=['F4', 'R4'], bounds=[0., 3.]),
    dict(kind='symmetric_line_distance', feature_ids=['B2', 'B1'], bounds=[1., 4.]),
])
def test_destructive_or_incomplete_parameterizations_rejected(hard_problem, spec):
    scene, data = hard_problem
    data['scene_parameters'] = [dict(id='illegal', **spec)]
    with pytest.raises(ValueError, match='hard|independent'):
        JointSceneProblem(scene, data)


@pytest.mark.parametrize('values', [dict(gap1=0.), dict(gap3=-.1), dict(F_x=.5), dict(R_x=-.5), dict(goal_distance=0.)])
def test_direct_parameter_vectors_cannot_bypass_order_side_or_positive_distance(hard_problem, values):
    scene, data = hard_problem; problem = JointSceneProblem(scene, data)
    with pytest.raises(ValueError, match='hard'):
        problem.geometry(changed(problem, **values))


def test_missing_required_features_rejected():
    data = raw_scene()
    data['ice_features'] = [f for f in data['ice_features'] if f['id'] != 'R4']
    with pytest.raises(ValueError, match='missing required.*R4'):
        project_scene_to_hard_constraints(data, PROFILE)


def test_sparse_dependencies_cover_actual_indirect_geometry_derivatives(hard_problem):
    scene, data = hard_problem; problem = JointSceneProblem(scene, data)
    sparse = problem.jac_sparsity.toarray()
    # Goal endpoints move with asymmetric arcs; row gaps affect later rows and
    # paired goal distances move crease observations although their IDs differ.
    for x in (problem.x0, changed(problem, goal_distance=9.3, gap1=1.4, gap2=2.2,
            upper_left=4.2, upper_right=3.4, lower_left=3.9, lower_right=4.4)):
        residual = problem.residual(x)
        assert sparse.shape == (len(residual), len(x))
        for index, name in enumerate(problem.parameter_names):
            plus, minus = x.copy(), x.copy(); plus[index] += 1e-6; minus[index] -= 1e-6
            derivative = (problem.residual(plus) - problem.residual(minus)) / 2e-6
            assert np.isfinite(derivative).all(), name
            assert np.all(sparse[np.abs(derivative) > 1e-5, index]), name


def test_export_reload_preserves_hard_profile_and_geometry(tmp_path, hard_problem):
    scene, data = hard_problem
    destination = tmp_path / 'versioned_hard_scene'
    checks = exporter.export(None, None, destination, semantic=scene.data)
    assert checks['mesh_and_grid_checks_passed']
    loaded = Scene(destination / 'scene.json')
    assert normalize_hard_geometry_constraints(loaded.data, {})['profile'] == HARD_GEOMETRY_PROFILE
    geometry = SharedGeometry(loaded, [], np.array([])); assert_structure(geometry)
    for fid in ('C', 'B1', 'B2', 'goal_line_left', 'goal_line_right'):
        np.testing.assert_allclose(geometry.features[fid]['segments_xyz'], loaded.features[fid]['segments_xyz'], atol=1e-9)
    assert exporter.load(destination / 'scene_geometry.json')['hard_geometry_checks']['all_constraints_satisfied']
    repeated, _, _ = exporter.rebuild_scene(loaded.data)
    for fid in ('goal_crease_left', 'goal_crease_right'):
        np.testing.assert_allclose(features_by_id(repeated)[fid]['polygon_xyz'], loaded.features[fid]['polygon_xyz'], atol=1e-9)


def test_asymmetric_mesh_clipping_preserves_paired_lines_and_large_creases():
    data = raw_scene(); f = features_by_id(data)
    t = np.linspace(-np.pi / 2, np.pi / 2, 61)
    f['goal_crease_left']['polygon_xyz'] = np.column_stack([-9.1 + 5.4*np.cos(t), 5.4*np.sin(t), np.zeros(len(t))]).tolist()
    data = project_scene_to_hard_constraints(data, PROFILE)
    scene, _, checks = exporter.rebuild_scene(data)
    assert checks['semantic_checks']['hard_geometry_checks']['all_constraints_satisfied']
    features = features_by_id(scene); boundary = np.array(scene['ice_boundary']['points_xyz'])[:, :2]
    for left, right in LINE_PAIRS:
        a, b = np.array(features[left]['segments_xyz']), np.array(features[right]['segments_xyz'])
        np.testing.assert_allclose(a * [-1., 1., 1.], b, atol=1e-12)
        assert exporter.inside_convex(a.reshape(-1, 3)[:, :2], boundary).all()
        assert exporter.inside_convex(b.reshape(-1, 3)[:, :2], boundary).all()
    a, b = np.array(features['goal_crease_left']['polygon_xyz']), np.array(features['goal_crease_right']['polygon_xyz'])
    np.testing.assert_allclose(a * [-1., 1., 1.], b, atol=1e-12)
    assert exporter.inside_convex(a[:, :2], boundary).all()
    assert exporter.inside_convex(b[:, :2], boundary).all()
    assert features['goal_crease_left']['joint_export_boundary_clipped']


def test_legacy_without_profile_does_not_gain_new_symmetry():
    data = raw_scene(); scene, _, _ = exporter.rebuild_scene(data)
    features = features_by_id(scene)
    assert 'hard_geometry_constraints' not in scene
    assert features['F1']['center_xyz'] != features['F2']['center_xyz']
    assert not np.allclose(np.array(features['goal_line_left']['segments_xyz']) * [-1., 1., 1.],
                           np.array(features['goal_line_right']['segments_xyz']))


def test_structure_only_export_does_not_claim_new_video_fit():
    data = project_scene_to_hard_constraints(raw_scene(), PROFILE)
    data['hard_structure_update'] = {'video_fitting_run': False, 'source_revision': 'previous_video_candidate'}
    data['joint_refinement'] = {'base_scene_sha256': 'historical_hash',
        'optimizer': {'success': False, 'nfev': 70, 'message': 'historical budget'}}
    data['build_scope'] = {'video_fitting_run': False}
    scene, geometry, _ = exporter.rebuild_scene(data)
    assert scene['build_scope']['video_fitting_run'] is False
    assert scene['parameters']['geometry_source'] == 'user_hard_relation_projection_from_prior_video_candidate'
    assert geometry['parameters']['geometry_source'] == scene['parameters']['geometry_source']
    assert scene['status'] == 'user_hard_relation_projection_not_video_refitted_NOT_physically_validated'
    assert scene['joint_refinement']['optimizer'] == data['joint_refinement']['optimizer']
    # A legacy scene without this profile retains the established export path.
    legacy = raw_scene(); legacy['joint_refinement'] = {'base_scene_sha256': 'historical_hash'}
    old_scene, _, _ = exporter.rebuild_scene(legacy)
    assert old_scene['build_scope']['video_fitting_run'] is True
