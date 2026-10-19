"""Synthetic regression checks for the requested freedom; not video validation."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from rinkspace.calibration_companion import Camera, Scene, look_at_rvec
from rinkspace.calibration_companion.joint_scene import JointSceneProblem, SharedGeometry


RINGS = tuple(f'{side}{row}' for side in ('F', 'R') for row in range(1, 5))
CORNERS = ('upper_left', 'upper_right', 'lower_left', 'lower_right')


@pytest.fixture
def open_geometry_problem(tmp_path):
    features = []
    for index, fid in enumerate(RINGS):
        center = [-5. if fid[0] == 'F' else 5., -3. + 2. * (int(fid[1]) - 1), 0.]
        radius = .60 + .025 * index
        features.append(dict(id=fid, kind='circle', center_xyz=center, radius=radius,
            rings=[dict(role='outer', center_xyz=center.copy(), radius=radius),
                   dict(role='inner', center_xyz=center.copy(), radius=.25)]))
    features.extend([
        dict(id='C', kind='polyline', segments_xyz=[[[0., 5., 0.], [0., -5., 0.]]]),
        dict(id='unchanged_line', kind='polyline', segments_xyz=[[[-2., 5., 0.], [-2., -5., 0.]]]),
    ])
    data = dict(schema_version='rinkspace.scene3d.v1', revision='synthetic_freedom',
        coordinate_system={'unit': 'synthetic'}, ice_features=features,
        parameters=dict(bounds=[-10., -6., 10., 6.], corner_radius=1., board_height=.8,
                        board_thickness=.1),
        ice_boundary={'points_xyz': [[-10., -6., 0.], [10., -6., 0.], [10., 6., 0.], [-10., 6., 0.]]},
        board_surface={'perimeter_length': 64., 'panels': []},
        meshes=[dict(name='rink_boards', positions=[[-10., -6., 0.], [10., -6., 0.],
                                                  [-10., -6., .8], [10., -6., .8]])],
        advertisements=[])
    path = tmp_path / 'scene.json'
    path.write_text(json.dumps(data), encoding='utf8')
    scene = Scene(path)
    specs = [dict(id='length', kind='boundary_length', bounds=[6., 30.]),
             dict(id='width', kind='boundary_width', bounds=[4., 24.]),
             dict(id='eight_radius', kind='circle_shared_radius', feature_ids=list(RINGS),
                  initial_value=.8, bounds=[.3, 1.5]),
             dict(id='center_line', kind='line_offset', feature_id='C', bounds=[-.5, .5])]
    values = [20., 12., .8, 0.]
    for corner in CORNERS:
        specs.append(dict(id=corner, kind='boundary_corner_radius', corner=corner, bounds=[.2, 4.]))
        values.append(1.)
    for fid in RINGS:
        for axis, kind in enumerate(('circle_center_x', 'circle_center_y')):
            specs.append(dict(id=f'{fid}_{axis}', kind=kind, feature_id=fid, bounds=[-8., 8.]))
            values.append(scene.features[fid]['center_xyz'][axis])
    landmarks = {
        'top_letter': dict(side='upper_long_side', along=1.5, z_fraction=.5),
        'left_letter': dict(side='left_end', along=.5, z_fraction=.6),
        'corner_letter': dict(side='upper_long_side', allow_corner='upper_left', along=-9.5, z_fraction=.4),
    }
    truth = SharedGeometry(scene, specs, np.asarray(values), copy.deepcopy(landmarks))
    center = np.array([25., -28., 21.])
    rvec = look_at_rvec(center, [0., 0., 0.])
    camera = Camera(center, rvec, [[850., 0., 640.], [0., 850., 360.], [0., 0., 1.]], [])
    observations = []

    def observe(oid, target, point=False):
        kind = 'point' if point else 'curve'
        xyz = truth.points(target, kind, 20)
        observations.append(dict(id=oid, frame_id='view', type=kind, use='fit', target=target,
                                 pixels_uv=camera.project(xyz)[0].tolist(), sigma_px=1.))

    for fid in RINGS:
        observe(fid, dict(kind='circle_outline', feature_id=fid))
    observe('C', dict(kind='line_segment', feature_id='C'))
    for side in ('upper_long_side', 'lower_long_side', 'left_end', 'right_end'):
        observe(side, dict(kind='board_side', side=side, edge='top'))
    for corner in CORNERS:
        observe(corner, dict(kind='board_corner', corner=corner, edge='contact'))
    for lid in landmarks:
        observe(lid, dict(kind='board_landmark', landmark_id=lid), point=True)
    for spec in specs:
        if spec['kind'] in ('boundary_length', 'boundary_width'):
            spec['evidence_observation_ids'] = list(CORNERS)
        elif spec['kind'] == 'circle_shared_radius':
            spec['evidence_observation_ids'] = list(RINGS)
        else:
            spec['evidence_observation_ids'] = [spec.get('feature_id', spec.get('corner'))]
    board_landmarks = []
    for lid, item in landmarks.items():
        board_landmarks.append(dict(id=lid, advertisement_instance_id=lid,
            **{key: value for key, value in item.items() if key not in ('along', 'z_fraction')},
            initial_along=item['along'], initial_z_fraction=item['z_fraction'],
            bounds_along=[-9.9, 9.] if 'corner' in lid else [-8., 8.],
            bounds_z_fraction=[.05, .95], fit=True, evidence_observation_ids=[lid]))
    problem = dict(synthetic=True, scene_sha256=scene.sha256, geometry_mode='shared_scene_joint',
        scene_parameters=specs, board_landmarks=board_landmarks,
        installations=[dict(id='camera', initial_center_xyz=center.tolist(), fit_center=False)],
        frames=[dict(id='view', installation_id='camera', image_size_wh=[1280, 720],
                     principal_point_px=[640., 360.], initial_rvec_world_to_camera=rvec.tolist(),
                     initial_focal_px=850., fit_pose=False)], observations=observations, curve_samples=64,
        geometric_constraints={'circle_order': [
            dict(feature_ids=[f'{side}{row}' for row in range(1, 5)], axis=1, min_gap=0.)
            for side in ('F', 'R')] + [
            dict(feature_ids=[f'F{row}', f'R{row}'], axis=0, min_gap=0.)
            for row in range(1, 5)]})
    return scene, problem


def changed(problem, **values):
    x = problem.x0.copy()
    for name, value in values.items():
        x[problem.parameter_names.index('scene:' + name)] = value
    return x


@pytest.mark.parametrize('dimension,value,bounds,landmark', [
    ('length', 24., [-12., -6., 12., 6.], 'left_letter'),
    ('width', 16., [-10., -8., 10., 8.], 'top_letter'),
])
def test_each_dimension_moves_physical_board_projection_without_scaling_ice(
        open_geometry_problem, dimension, value, bounds, landmark):
    scene, data = open_geometry_problem
    problem = JointSceneProblem(scene, data)
    old = problem.geometry(problem.x0)
    x = changed(problem, **{dimension: value})
    new = problem.geometry(x)
    np.testing.assert_array_equal(new.parameters['bounds'], bounds)
    camera = problem.camera(x, 'view')
    old_uv = camera.project([old.board_landmark(landmark)])[0]
    new_uv = camera.project([new.board_landmark(landmark)])[0]
    assert np.linalg.norm(new_uv - old_uv) > 1.
    assert new.features == old.features
    exported = problem.export_scene_data(x)
    assert exported['parameters']['bounds'] == bounds
    assert exported['joint_refinement']['boundary_mesh_rebuild_required']
    assert scene.data['parameters']['bounds'] == [-10., -6., 10., 6.]


def test_shared_radius_updates_all_eight_outer_rings_and_their_export_status(open_geometry_problem):
    scene, data = open_geometry_problem
    problem = JointSceneProblem(scene, data)
    x = changed(problem, eight_radius=1.15)
    geometry = problem.geometry(x)
    exported = {feature['id']: feature for feature in problem.export_scene_data(x)['ice_features']}
    for fid in RINGS:
        assert geometry.features[fid]['radius'] == 1.15
        outer, inner = geometry.features[fid]['rings']
        assert outer['radius'] == 1.15
        assert inner == scene.features[fid]['rings'][1]
        assert geometry.features[fid]['center_xyz'] == scene.features[fid]['center_xyz']
        assert exported[fid]['geometry_status'] == 'joint_video_fit_candidate_not_independent_validation'
    change = next(row for row in problem.parameter_changes(x) if row['id'] == 'eight_radius')
    assert change['original_feature_values'] == {fid: scene.features[fid]['radius'] for fid in RINGS}


@pytest.mark.parametrize('corner', ['lower_left', 'lower_right'])
def test_lower_corners_can_change_independently_and_remain_exact_circles(open_geometry_problem, corner):
    scene, data = open_geometry_problem
    problem = JointSceneProblem(scene, data)
    geometry = problem.geometry(changed(problem, **{corner: 1.7}))
    for name in CORNERS:
        assert geometry.corner_radii[name] == (1.7 if name == corner else 1.)
    points = geometry.points(dict(kind='board_corner', corner=corner), 'curve', 31)
    center = [-10. + 1.7 if corner.endswith('left') else 10. - 1.7, -6. + 1.7]
    np.testing.assert_allclose(np.linalg.norm(points[:, :2] - center, axis=1), 1.7, atol=1e-12)
    np.testing.assert_array_equal(points[:, 2], 0.)


def test_central_line_moves_without_turning_or_moving_other_lines(open_geometry_problem):
    scene, data = open_geometry_problem
    problem = JointSceneProblem(scene, data)
    geometry = problem.geometry(changed(problem, center_line=.3))
    source = np.asarray(scene.features['C']['segments_xyz'])
    shifted = np.asarray(geometry.features['C']['segments_xyz'])
    np.testing.assert_allclose(shifted - source, np.broadcast_to([.3, 0., 0.], source.shape))
    assert geometry.features['unchanged_line'] == scene.features['unchanged_line']


def test_invalid_corner_overlap_is_reported_and_cannot_be_exported(open_geometry_problem):
    scene, data = open_geometry_problem
    problem = JointSceneProblem(scene, data)
    x = changed(problem, width=6., upper_left=3.5, lower_left=3.5)
    summary = problem.summarize(x)
    assert not summary['boundary_geometry_valid']
    assert not summary['fit_geometry_valid']
    assert summary['boundary_invalid_constraints']
    with pytest.raises(ValueError, match='boundary'):
        problem.export_scene_data(x)


@pytest.mark.parametrize('swap', [dict(F1_1=.5), dict(R1_0=-6.)])
def test_relative_order_rejects_swaps_but_allows_nonuniform_spacing(open_geometry_problem, swap):
    scene, data = open_geometry_problem
    problem = JointSceneProblem(scene, data)
    valid = changed(problem, F2_0=-4.5, F2_1=-.3, F3_1=1.2, R2_1=-1.7)
    assert problem.summarize(valid)['circle_order_valid']
    problem.export_scene_data(valid)
    # Identities may shift independently; crossing rows or F/R sides is not allowed.
    invalid = changed(problem, **swap)
    summary = problem.summarize(invalid)
    assert not summary['circle_order_valid']
    assert not summary['fit_geometry_valid']
    assert summary['circle_order_invalid_constraints']
    with pytest.raises(ValueError, match='order'):
        problem.export_scene_data(invalid)


def test_sparse_graph_covers_projection_and_active_shape_domain_guards(open_geometry_problem):
    scene, data = open_geometry_problem
    problem = JointSceneProblem(scene, data)
    sparse = problem.jac_sparsity.toarray()
    # The second state activates boundary, landmark-surface and row-order guards;
    # testing only the valid state would miss sparse dependencies of those guards.
    invalid = changed(problem, length=12., width=6., upper_left=3.5, lower_left=3.5, F1_1=.5)
    for x in (problem.x0, invalid):
        residual = problem.residual(x)
        assert np.isfinite(residual).all()
        assert sparse.shape == (len(residual), len(x))
        for index, name in enumerate(problem.parameter_names):
            plus, minus = x.copy(), x.copy()
            plus[index] += 1e-6
            minus[index] -= 1e-6
            derivative = (problem.residual(plus) - problem.residual(minus)) / 2e-6
            assert np.isfinite(derivative).all(), name
            assert np.all(sparse[np.abs(derivative) > 1e-5, index] != 0), name


@pytest.fixture
def runner_contract(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / 'scripts'
    monkeypatch.syspath_prepend(str(scripts))
    module_spec = importlib.util.spec_from_file_location('synthetic_joint_runner', scripts / 'fit_joint_scene_video.py')
    runner = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(runner)
    ids = (*RINGS, 'C0', 'A1', 'A2')
    features = {fid: dict(id=fid, kind='circle', center_xyz=[0., float(index), 0.], radius=6.)
                for index, fid in enumerate(ids)}
    scene = SimpleNamespace(features=features)
    observations = [dict(id=f'outer_{fid}', use='fit', type='curve', frame_id='view',
                         target=dict(kind='circle_outline', feature_id=fid), pixels_uv=[[50., 60.], [55., 65.]])
                    for fid in ids]
    observations.extend([
        dict(id='center_line', use='fit', type='curve', frame_id='view',
             target=dict(kind='line_segment', feature_id='C'), pixels_uv=[[20., 50.], [30., 60.]]),
        dict(id='check_F1', use='check', type='curve', frame_id='unseen_view',
             target=dict(kind='circle_outline', feature_id='F1'), pixels_uv=[[500., 600.]]),
        dict(id='inner_F1', use='fit', type='curve', frame_id='view',
             target=dict(kind='circle_outline', feature_id='F1', ring_role='inner'), pixels_uv=[[53., 62.]]),
        dict(id='ad_detail', use='fit', type='point', frame_id='view',
             target=dict(kind='board_landmark', landmark_id='letter'), pixels_uv=[[90., 30.]]),
    ])
    board_landmarks = [dict(id='letter', advertisement_instance_id='synthetic_corner_ad',
                           side='upper_long_side', allow_corner='upper_left',
                           initial_along=-90., initial_z_fraction=.4)]
    problem = dict(observations=observations, board_landmarks=board_landmarks,
        installations=[dict(id=iid, initial_center_xyz=[float(i), -70., 15.])
                       for i, iid in enumerate(('main', 'secondary', 'low', 'high'))],
        frames=[dict(id=f'frame_{iid}', installation_id=iid, source_pts=256*i,
                     initial_focal_px=1100.+i, initial_rvec_world_to_camera=[.1, .2, .3])
                for i, iid in enumerate(('main', 'secondary', 'low', 'high'))],
        heldout_shots=['synthetic_unseen_shot'], source_observation_files=['synthetic_only'],
        scene_parameters=[dict(id='old_parameter')], geometric_constraints={'old': True})
    return runner, scene, problem


def test_runner_preserves_only_user_authorized_independent_center_freedoms(runner_contract):
    runner, scene, data = runner_contract
    specs, _ = runner.scene_parameterization(scene, data['observations'], data['board_landmarks'])
    centers = {(s['feature_id'], s['kind']) for s in specs if s['kind'].startswith('circle_center_')}
    assert centers == {(fid, kind) for fid in ('A1', 'A2') for kind in ('circle_center_x', 'circle_center_y')}
    columns = [s for s in specs if s['kind'] == 'circle_shared_center_x']
    assert {tuple(s['feature_ids']) for s in columns} == {
        tuple(f'{side}{n}' for n in range(1, 5)) for side in ('F', 'R')}
    assert sum(s['kind'] == 'circle_row_origin_y' for s in specs) == 1
    gaps = [s for s in specs if s['kind'] == 'circle_row_gap']
    assert {s['after_row'] for s in gaps} == {1, 2, 3}
    assert all(s['bounds'][0] > 0 for s in gaps)
    shared = [s for s in specs if s['kind'] == 'circle_shared_radius']
    assert len(shared) == 1 and set(shared[0]['feature_ids']) == set(RINGS)
    assert {s['feature_id'] for s in specs if s['kind'] == 'circle_radius'} == {'C0', 'A1', 'A2'}
    assert {s['corner'] for s in specs if s['kind'] == 'boundary_corner_radius'} == set(CORNERS)
    assert sum(s['kind'] == 'boundary_length' for s in specs) == 1
    assert sum(s['kind'] == 'boundary_width' for s in specs) == 1
    assert not any(s.get('feature_id') == 'C' for s in specs)
    mirrors = [s for s in specs if s['kind'] == 'symmetric_line_distance']
    assert {tuple(s['feature_ids']) for s in mirrors} == {
        ('B1', 'B2'), ('goal_line_left', 'goal_line_right')}
    assert not any(s['kind'] == 'line_offset' for s in specs)


def test_runner_apply_preserves_all_observed_pixels_checks_and_four_camera_identities(runner_contract):
    runner, scene, data = runner_contract
    before = copy.deepcopy(data)
    source_features = copy.deepcopy(scene.features)
    returned = runner.apply_scene_parameterization(scene, data)
    for key in ('observations', 'board_landmarks', 'installations', 'frames',
                'heldout_shots', 'source_observation_files'):
        assert returned[key] == before[key], key
    assert scene.features == source_features
    assert returned['scene_parameters'] != before['scene_parameters']
    assert returned['geometric_constraints'] != before['geometric_constraints']
    fixed = runner.apply_scene_parameterization(scene, copy.deepcopy(before), fixed=True)
    assert fixed['scene_parameters'] == [] and fixed['geometric_constraints'] == {}
    assert fixed['hard_geometry_constraints'] == returned['hard_geometry_constraints'] == {
        'profile': 'user_20260923_alignment_symmetry'}
    assert fixed['observations'] == before['observations']
    assert fixed['frames'] == before['frames']


def test_runner_routes_shared_positions_to_both_sides_without_equal_spacing(runner_contract):
    runner, scene, data = runner_contract
    specs, constraints = runner.scene_parameterization(scene, data['observations'], data['board_landmarks'])
    by_id = {s['id']: s for s in specs}
    assert 'check_F1' not in {oid for s in specs for oid in s['evidence_observation_ids']}
    assert 'inner_F1' not in by_id['F_R_shared_outer_radius']['evidence_observation_ids']
    for name in ('rink_length', 'rink_width', 'board_height', 'corner_upper_left'):
        assert 'ad_detail' in by_id[name]['evidence_observation_ids'], name
    assert 'ad_detail' not in by_id['corner_lower_left']['evidence_observation_ids']
    # The older order diagnostics remain alongside the actual hard
    # origin/positive-gap parameterization. No equal-spacing prior is added.
    assert set(constraints) == {'circle_order'}
    assert all(set(row) == {'feature_ids', 'axis', 'min_gap'} and row['min_gap'] == 0.
               for row in constraints['circle_order'])
    y_chains = {tuple(row['feature_ids']) for row in constraints['circle_order'] if row['axis'] == 1}
    assert y_chains == {tuple(f'{side}{row}' for row in range(1, 5)) for side in ('F', 'R')}
    x_pairs = {tuple(row['feature_ids']) for row in constraints['circle_order'] if row['axis'] == 0}
    assert x_pairs == {(f'F{i}', f'R{j}') for i in range(1, 5) for j in range(1, 5)}
    assert {'outer_F1', 'inner_F1'} <= set(by_id['F_column_x']['evidence_observation_ids'])
    assert 'outer_R1' not in by_id['F_column_x']['evidence_observation_ids']
    assert 'outer_R4' in by_id['FR_row_gap_3_4']['evidence_observation_ids']
    assert 'outer_F4' in by_id['FR_row_gap_3_4']['evidence_observation_ids']
    assert 'outer_F3' not in by_id['FR_row_gap_3_4']['evidence_observation_ids']
    assert all('prior' not in spec for spec in specs)


def test_shrinking_rink_cannot_export_circles_that_extend_beyond_the_ice(open_geometry_problem):
    scene, data = open_geometry_problem
    problem = JointSceneProblem(scene, data)
    x = changed(problem, width=6.)
    summary = problem.summarize(x)
    assert summary['boundary_geometry_valid']  # The board shape itself is legal.
    assert not summary['circle_geometry_valid']
    assert not summary['fit_geometry_valid']
    assert {'F1', 'F4', 'R1', 'R4'} <= set(summary['circle_invalid_feature_ids'])
    with pytest.raises(ValueError, match='circle'):
        problem.export_scene_data(x)


def test_shared_outer_ring_cannot_shrink_inside_unchanged_inner_ring(open_geometry_problem):
    scene, data = open_geometry_problem
    problem = JointSceneProblem(scene, data)
    # A diagnostic vector may be outside optimizer bounds; validation still has
    # to report why the candidate cannot represent the requested nested rings.
    x = changed(problem, eight_radius=.2)
    summary = problem.summarize(x)
    assert not summary['circle_geometry_valid']
    assert set(summary['circle_invalid_feature_ids']) == set(RINGS)
    checks = {row['feature_id']: row for row in summary['circle_geometry_checks']}
    assert all(checks[fid]['inner_radius_margin_scene_units'] < 0. for fid in RINGS)
    with pytest.raises(ValueError, match='circle'):
        problem.export_scene_data(x)


def test_unobserved_circle_inside_bbox_but_outside_round_corner_is_not_a_fit_constraint(open_geometry_problem):
    scene, data = open_geometry_problem
    old = JointSceneProblem(scene, data)
    before = old.residual(old.x0)
    probe = dict(id='corner_probe', kind='circle', center_xyz=[9.7, 5.7, 0.], radius=.2)
    scene.features[probe['id']] = probe
    scene.data['ice_features'].append(probe)
    problem = JointSceneProblem(scene, data)
    # This schematic, unobserved circle must not create invented pixel evidence
    # or steer the objective, but it makes the exported whole-rink model invalid.
    np.testing.assert_array_equal(problem.residual(problem.x0), before)
    summary = problem.summarize(problem.x0)
    assert summary['boundary_geometry_valid']
    assert not summary['circle_geometry_valid']
    assert summary['circle_invalid_feature_ids'] == ['corner_probe']
    row = next(item for item in summary['circle_geometry_checks'] if item['feature_id'] == 'corner_probe')
    assert row['containment_margin_scene_units'] < 0.
    assert np.all(np.asarray(probe['center_xyz'][:2]) + probe['radius'] < [10., 6.])
    with pytest.raises(ValueError, match='circle'):
        problem.export_scene_data(problem.x0)
