"""New wide-view feature freedom must preserve the user's exact structure."""
import copy
import json

import numpy as np
import pytest

from rinkspace.calibration_companion import Camera, Scene
from rinkspace.calibration_companion.joint_scene import JointSceneProblem, SharedGeometry
from rinkspace.calibration_companion.scene_camera import nearest_polyline_vectors
from test_joint_scene_hard_structure import (
    hard_problem, assert_structure, changed, exporter,
)


def test_shared_crease_sizes_preserve_mirrors_and_round_trip(tmp_path, hard_problem):
    scene, data = hard_problem
    for kind in ('shared_crease_depth', 'shared_crease_half_width'):
        data['scene_parameters'].append(dict(id=kind, kind=kind, bounds=[.1, 2.],
            evidence_observation_ids=['goal_crease_left', 'goal_crease_right']))
    problem = JointSceneProblem(scene, data)
    initial = problem.geometry(problem.x0)
    x = changed(problem, shared_crease_depth=1.5, shared_crease_half_width=1.2,
                goal_distance=8.2, length=22., width=14.)
    geometry = problem.geometry(x); assert_structure(geometry)
    for fid in [f'{col}{row}' for col in 'FR' for row in (1, 2, 3, 4)]:
        assert geometry.features[fid] == initial.features[fid]
    left = np.asarray(geometry.features['goal_crease_left']['polygon_xyz'])
    assert np.isclose(np.ptp(left[:, 0]), 1.5)
    assert np.isclose(np.max(np.abs(left[:, 1])), 1.2)
    # Both direct semantic reload and rebuilt mesh reload must keep the sizes.
    exported = problem.export_scene_data(x)
    for feature in exported['ice_features']:
        if feature['id'] in ('goal_crease_left', 'goal_crease_right'):
            assert feature['geometry_status'] == 'joint_video_fit_candidate_not_independent_validation'
    path = tmp_path/'semantic.json'; path.write_text(json.dumps(exported))
    direct = SharedGeometry(Scene(path), [], np.array([])); assert_structure(direct)
    destination = tmp_path/'rebuilt'
    exporter.export(None, None, destination, semantic=exported)
    loaded = SharedGeometry(Scene(destination/'scene.json'), [], np.array([])); assert_structure(loaded)
    for candidate in (direct, loaded):
        for fid in ('goal_crease_left', 'goal_crease_right'):
            np.testing.assert_allclose(candidate.features[fid]['polygon_xyz'], geometry.features[fid]['polygon_xyz'], atol=1e-9)


def test_lane_parameters_preserve_rectangle_and_sample_distorted_edges(tmp_path, hard_problem):
    scene, data = hard_problem
    original = copy.deepcopy(scene.data)
    original['ice_features'].append(dict(id='lane1', kind='polygon',
        polygon_xyz=[[-5., -2., 0.], [5., -2., 0.], [5., 2., 0.], [-5., 2., 0.]]))
    path = tmp_path/'with_lane.json'; path.write_text(json.dumps(original))
    scene = Scene(path); data['scene_sha256'] = scene.sha256
    frame = data['frames'][0]; install = data['installations'][0]
    camera = Camera(install['initial_center_xyz'], frame['initial_rvec_world_to_camera'],
                    [[850., 0., 640.], [0., 850., 360.], [0., 0., 1.]], [])
    target = dict(kind='polygon_outline', feature_id='lane1')
    initial_geometry = SharedGeometry(scene, [], np.array([]))
    data['observations'].append(dict(id='lane', frame_id='view', type='curve', use='fit', sigma_px=1.,
        target=target, pixels_uv=camera.project(initial_geometry.points(target, 'curve', 40))[0].tolist()))
    for kind, bounds in [('lane_bound_x_min', [-8., -1.]), ('lane_bound_x_max', [1., 8.]),
                         ('lane_bound_y_min', [-4., -.5]), ('lane_bound_y_max', [.5, 4.])]:
        data['scene_parameters'].append(dict(id=kind, kind=kind, feature_id='lane1',
            bounds=bounds, evidence_observation_ids=['lane']))
    problem = JointSceneProblem(scene, data)
    x = changed(problem, lane_bound_x_min=-6., lane_bound_x_max=7.,
                lane_bound_y_min=-3., lane_bound_y_max=2.5)
    geometry = problem.geometry(x); assert_structure(geometry)
    np.testing.assert_array_equal(geometry.features['lane1']['polygon_xyz'],
        [[-6., -3., 0.], [7., -3., 0.], [7., 2.5, 0.], [-6., 2.5, 0.]])
    for fid in [f'{col}{row}' for col in 'FR' for row in (1, 2, 3, 4)]:
        assert geometry.features[fid] == initial_geometry.features[fid]
    exported = problem.export_scene_data(x)
    path = tmp_path/'lane_roundtrip.json'; path.write_text(json.dumps(exported))
    reloaded = SharedGeometry(Scene(path), [], np.array([]))
    np.testing.assert_array_equal(reloaded.features['lane1']['polygon_xyz'], geometry.features['lane1']['polygon_xyz'])
    # A visible edge midpoint must be projected on its Brown curve, not the
    # straight chord between the distorted endpoints.
    distorted = Camera(camera.center_xyz, camera.rvec_world_to_camera, camera.K_px, [-.4, .1, .02, .03, 0.])
    xyz = geometry.points(target, 'curve', 128)
    assert len(xyz) >= 129
    np.testing.assert_array_equal(xyz[0], xyz[-1])
    midpoint = (np.asarray(geometry.features['lane1']['polygon_xyz'])[0] +
                np.asarray(geometry.features['lane1']['polygon_xyz'])[1])/2
    observed = distorted.project([midpoint])[0]
    projected = distorted.project(xyz)[0]
    assert np.linalg.norm(nearest_polyline_vectors(observed, projected)) < 1e-8
    vertices = np.asarray(geometry.features['lane1']['polygon_xyz'])
    chord = distorted.project(np.vstack([vertices, vertices[:1]]))[0]
    assert np.linalg.norm(nearest_polyline_vectors(observed, chord)) > .1
    # A supporting-line observation cannot inherit the arbitrary source lane
    # endpoints. Its samples span the rink while leaving the rectangle intact.
    supporting = dict(kind='polygon_supporting_line', feature_id='lane1', index=1)
    support = geometry.points(supporting, 'curve', 129)
    np.testing.assert_allclose(support[:, 0], 7.)
    np.testing.assert_allclose(support[[0, -1], 1], np.asarray(geometry.parameters['bounds'])[[1, 3]])
    changed_endpoints = changed(problem, lane_bound_y_min=-3.9, lane_bound_y_max=3.9,
                                lane_bound_x_min=-6., lane_bound_x_max=7.)
    np.testing.assert_array_equal(problem.geometry(changed_endpoints).points(supporting, 'curve', 129), support)
    for spec in problem.scene_specs:
        if spec.get('feature_id') == 'lane1':
            assert problem._affects(spec, supporting) == (spec['kind'] == 'lane_bound_x_max')


def test_shared_internal_rings_preserve_structure_and_concentric_reload(tmp_path, hard_problem):
    scene, data = hard_problem
    original = copy.deepcopy(scene.data)
    ids = [f'{col}{row}' for col in 'FR' for row in (1, 2, 3, 4)]
    for f in original['ice_features']:
        if f['id'] in ids:
            f['rings'].extend([dict(role='middle', radius=.4, center_xyz=list(f['center_xyz'])),
                               dict(role='center', radius=.08, center_xyz=list(f['center_xyz']))])
    path = tmp_path/'with_rings.json'; path.write_text(json.dumps(original))
    scene = Scene(path); data['scene_sha256'] = scene.sha256
    frame, install = data['frames'][0], data['installations'][0]
    camera = Camera(install['initial_center_xyz'], frame['initial_rvec_world_to_camera'],
                    [[850., 0., 640.], [0., 850., 360.], [0., 0., 1.]], [])
    initial = SharedGeometry(scene, [], np.array([]))
    for role, bounds in [('middle', [.3, .5]), ('inner', [.1, .29]), ('center', [.01, .099])]:
        target = dict(kind='circle_outline', feature_id='F1', ring_role=role)
        oid = f'ring_{role}'
        data['observations'].append(dict(id=oid, frame_id='view', type='curve', use='fit', sigma_px=1.,
            target=target, pixels_uv=camera.project(initial.points(target, 'curve', 32))[0].tolist()))
        data['scene_parameters'].append(dict(id=oid, kind='ring_shared_radius', feature_ids=ids,
            ring_role=role, bounds=bounds, evidence_observation_ids=[oid]))
    problem = JointSceneProblem(scene, data)
    x = changed(problem, ring_middle=.42, ring_inner=.19, ring_center=.09, F_x=-4.5, gap1=1.8)
    geometry = problem.geometry(x); assert_structure(geometry)
    for fid in ids:
        f = geometry.features[fid]
        for role, expected in [('middle', .42), ('inner', .19), ('center', .09)]:
            ring = next(r for r in f['rings'] if r['role'] == role)
            assert ring['radius'] == expected
            np.testing.assert_array_equal(ring['center_xyz'], f['center_xyz'])
        assert f['radius'] == initial.features[fid]['radius']
    # No role may reuse an outer-ring observation as its evidence.
    bad = copy.deepcopy(data); bad['scene_parameters'][-1]['evidence_observation_ids'] = ['F1']
    with pytest.raises(ValueError, match='unrelated'):
        JointSceneProblem(scene, bad)
    bad = copy.deepcopy(data); bad['scene_parameters'][-1]['ring_role'] = 'outer'
    with pytest.raises(ValueError, match='middle, inner or center'):
        JointSceneProblem(scene, bad)
    exported = problem.export_scene_data(x)
    path = tmp_path/'ring_roundtrip.json'; path.write_text(json.dumps(exported))
    loaded = SharedGeometry(Scene(path), [], np.array([])); assert_structure(loaded)
    for fid in ids:
        assert loaded.features[fid]['rings'] == geometry.features[fid]['rings']


def lane_problem(tmp_path, hard_problem):
    scene, data = hard_problem
    original = copy.deepcopy(scene.data)
    for row in range(1, 5):
        half = .4 + .05 * row
        center = scene.features[f'F{row}']['center_xyz'][1] + .2
        original['ice_features'].append(dict(id=f'lane{row}', kind='polygon', closed=True,
            polygon_xyz=[[-6., center+half, 0.], [6., center+half, 0.],
                         [6., center-half, 0.], [-6., center-half, 0.]]))
    path = tmp_path/'open_lane_source.json'; path.write_text(json.dumps(original))
    scene = Scene(path); data['scene_sha256'] = scene.sha256
    data['hard_geometry_constraints'] = {**scene.data['hard_geometry_constraints'],
        'lane_centerline_through_FR_rows': True, 'lane_left_open': True}
    frame, install = data['frames'][0], data['installations'][0]
    camera = Camera(install['initial_center_xyz'], frame['initial_rvec_world_to_camera'],
                    [[850., 0., 640.], [0., 850., 360.], [0., 0., 1.]], [])
    truth = SharedGeometry(scene, [], np.array([]), hard_geometry_constraints=data['hard_geometry_constraints'])
    for row in range(1, 4):
        target = dict(kind='polygon_supporting_line', feature_id=f'lane{row}', index=0)
        data['observations'].append(dict(id=f'lane{row}_top', frame_id='view', type='curve', use='fit', sigma_px=1.,
            target=target, pixels_uv=camera.project(truth.points(target, 'curve', 32))[0].tolist()))
    target = dict(kind='polygon_supporting_line', feature_id='lane1', index=1)
    data['observations'].append(dict(id='lane1_right', frame_id='view', type='curve', use='fit', sigma_px=1.,
        target=target, pixels_uv=camera.project(truth.points(target, 'curve', 32))[0].tolist()))
    return scene, data


def test_independent_open_lane_half_widths_follow_rows_and_preserve_hard_checks(tmp_path, hard_problem):
    scene, data = lane_problem(tmp_path, hard_problem)
    for row in range(1, 5):
        data['scene_parameters'].append(dict(id=f'lane{row}_half', kind='lane_half_width',
            feature_id=f'lane{row}', bounds=[.2, 1.2],
            evidence_observation_ids=[f'lane{row}_top'] if row < 4 else []))
    data['scene_parameters'].append(dict(id='lane1_right', kind='lane_bound_x_max', feature_id='lane1',
        bounds=[5., 7.], evidence_observation_ids=['lane1_right']))
    problem = JointSceneProblem(scene, data)
    x = changed(problem, row_origin=-3.3, gap1=1.8, lane1_half=.7, lane2_half=.65, lane3_half=.8, lane1_right=5.5)
    geometry = problem.geometry(x); assert_structure(geometry)
    for row, half in enumerate((.7, .65, .8, .6), 1):
        feature = geometry.features[f'lane{row}']; polygon = np.asarray(feature['polygon_xyz'])
        np.testing.assert_allclose(np.ptp(polygon[:, 1]) / 2., half)
        np.testing.assert_allclose((polygon[:, 1].min() + polygon[:, 1].max()) / 2., geometry.features[f'F{row}']['center_xyz'][1])
        np.testing.assert_allclose(polygon[:, 0], [-6., 5.5 if row == 1 else 6., 5.5 if row == 1 else 6., -6.])
        assert feature['open_edges'] == [3] and feature['closed'] is False
        outline = geometry.points(dict(kind='polygon_outline', feature_id=f'lane{row}'), 'curve', 64)
        np.testing.assert_array_equal(outline[0], polygon[0]); np.testing.assert_array_equal(outline[-1], polygon[3])
        assert not np.any((np.abs(outline[:, 0]+6.) < 1e-9) &
                          (outline[:, 1] > polygon[:, 1].min()+1e-9) &
                          (outline[:, 1] < polygon[:, 1].max()-1e-9))
    specs = {s['id']: s for s in problem.scene_specs}
    horizontal = dict(kind='polygon_supporting_line', feature_id='lane3', index=0)
    vertical = dict(kind='polygon_supporting_line', feature_id='lane3', index=1)
    assert problem._affects(specs['gap1'], horizontal)
    assert problem._affects(specs['row_origin'], horizontal)
    assert problem._affects(specs['lane3_half'], horizontal)
    assert not problem._affects(specs['gap1'], vertical)
    assert not problem._affects(specs['lane3_half'], vertical)
    assert not problem._affects(specs['gap1'], dict(kind='polygon_supporting_line', feature_id='lane1', index=0))
    assert any(s['id'] == 'lane4_half' for s in problem.retained_parameters)
    with pytest.raises(ValueError, match='open polygon edge'):
        geometry.points(dict(kind='polygon_edge', feature_id='lane1', index=3), 'curve', 16)
    exported = problem.export_scene_data(x)
    path = tmp_path/'open_lanes_roundtrip.json'; path.write_text(json.dumps(exported))
    loaded = SharedGeometry(Scene(path), [], np.array([])); assert_structure(loaded)
    for row in range(1, 5):
        np.testing.assert_allclose(loaded.features[f'lane{row}']['polygon_xyz'], geometry.features[f'lane{row}']['polygon_xyz'])
        assert loaded.features[f'lane{row}']['open_edges'] == [3]
    assert 'candidate_geometry_assumptions' not in exported['joint_refinement']


def test_shared_lane_width_is_explicit_optional_hypothesis(tmp_path, hard_problem):
    scene, data = lane_problem(tmp_path, hard_problem)
    data['scene_parameters'].append(dict(id='lane_shared_half', kind='shared_lane_half_width',
        feature_ids=['lane1', 'lane2', 'lane3', 'lane4'], bounds=[.2, 1.2],
        evidence_observation_ids=['lane1_top', 'lane2_top', 'lane3_top']))
    problem = JointSceneProblem(scene, data)
    x = changed(problem, lane_shared_half=.72, gap2=1.5)
    geometry = problem.geometry(x); assert_structure(geometry)
    for row in range(1, 5):
        p = np.asarray(geometry.features[f'lane{row}']['polygon_xyz'])
        np.testing.assert_allclose(np.ptp(p[:, 1]) / 2., .72)
        np.testing.assert_allclose((p[:, 1].min()+p[:, 1].max()) / 2., geometry.features[f'R{row}']['center_xyz'][1])
    hypothesis = problem.export_scene_data(x)['joint_refinement']['candidate_geometry_assumptions'][0]
    assert hypothesis['user_hard_constraint'] is False
    assert hypothesis['half_width_scene_units'] == .72
    bad = copy.deepcopy(data)
    bad['scene_parameters'].append(dict(id='lane1_independent', kind='lane_half_width', feature_id='lane1',
        bounds=[.2, 1.2], evidence_observation_ids=['lane1_top']))
    with pytest.raises(ValueError, match='duplicate semantic'):
        JointSceneProblem(scene, bad)
