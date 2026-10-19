"""Numerical interface checks only; these fixtures are not video evidence."""
import copy
import json

import numpy as np
import pytest

from rinkspace.calibration_companion import KeyframeProblem, Scene, pose_candidates_from_points


@pytest.fixture
def scene_problem(tmp_path):
    centres = [[-2., -1., 0.], [2., -1., 0.], [1., 2., 0.], [-1., 2., 0.]]
    data = {
        'schema_version': 'rinkspace.scene3d.v1', 'revision': 'unit_fixture',
        'coordinate_system': {'unit': 'unit_fixture'},
        'ice_features': [dict(id=f'F{i}', center_xyz=point, radius=.3)
                         for i, point in enumerate(centres)] +
                        [dict(id='CHECK', center_xyz=[1., 1., -20.], radius=.3)],
        'ice_boundary': {'points_xyz': [[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]},
        'board_surface': {'perimeter_length': 1., 'panels': [
            {'s_range': [0., 1.], 'bottom_vertex_indices': [0, 1]}]},
        'meshes': [{'name': 'rink_boards', 'positions': [[0., 0., 0.], [1., 0., 0.]]}],
        'advertisements': [{'id': 'unknown_height', 'attachment_s': .5,
                            'anchor_xyz': [.5, 0., None]}],
    }
    path = tmp_path / 'scene.json'
    path.write_text(json.dumps(data), encoding='utf8')
    scene = Scene(path)
    frame = {
        'id': 'frame', 'installation_id': 'installation',
        'image_size_wh': [640, 480], 'principal_point_px': [320., 240.],
        'initial_rvec_world_to_camera': [0., 0., 0.],
        'initial_focal_px': 500., 'focal_bounds_px': [250., 1000.],
    }
    observations = [
        dict(id=f'fit{i}', frame_id='frame', type='point',
             target={'kind': 'circle_center', 'feature_id': f'F{i}'},
             pixels_uv=[[320 + 50 * p[0], 240 + 50 * p[1]]], sigma_px=1., use='fit')
        for i, p in enumerate(centres)
    ]
    observations.append(dict(id='check', frame_id='frame', type='point',
                             target={'kind': 'circle_center', 'feature_id': 'CHECK'},
                             pixels_uv=[[120., 150.]], sigma_px=1., use='check'))
    problem = {
        'synthetic': True, 'scene_sha256': scene.sha256,
        'geometry_mode': 'fixed_scene_conditional', 'frames': [frame],
        'installations': [{'id': 'installation', 'initial_center_xyz': [0., 0., -10.],
                           'center_bounds_xyz': [[-5., -5., -15.], [5., 5., -5.]]}],
        'observations': observations,
    }
    return scene, problem


@pytest.mark.parametrize('identity', ['physical_group', 'physical_group_id', 'feature_id'])
def test_rejects_whole_object_fit_check_leakage(scene_problem, identity):
    scene, data = scene_problem
    # Physical-group aliases must have the same meaning.
    data['observations'][0][identity] = 'shared_object'
    check_key = 'physical_group_id' if identity == 'physical_group' else identity
    data['observations'][-1][check_key] = 'shared_object'
    with pytest.raises(ValueError, match='whole-object split violation'):
        KeyframeProblem(scene, data)


def test_different_rings_of_same_feature_are_not_independent(scene_problem):
    scene, data = scene_problem
    for index, role in [(0, 'outer'), (-1, 'inner')]:
        data['observations'][index]['type'] = 'curve'
        data['observations'][index]['target'] = {
            'kind': 'circle_outline', 'feature_id': 'F0', 'ring_role': role}
    scene.features['F0']['rings'] = [
        {'role': role, 'center_xyz': [-2., -1., 0.], 'radius': radius}
        for role, radius in [('outer', .3), ('inner', .1)]
    ]
    with pytest.raises(ValueError, match='whole-object split violation'):
        KeyframeProblem(scene, data)


def test_check_pixels_and_depth_do_not_select_or_change_fit(scene_problem, monkeypatch):
    scene, data = scene_problem
    before = KeyframeProblem(scene, data)
    changed = copy.deepcopy(data)
    changed['observations'][-1]['pixels_uv'] = [[99999., -88888.]]
    after = KeyframeProblem(scene, changed)
    np.testing.assert_array_equal(before.x0, after.x0)
    np.testing.assert_array_equal(before.residual(before.x0), after.residual(after.x0))

    calls = []
    def capture_pnp(world, pixels, *args, **kwargs):
        calls.append((world.copy(), pixels.copy()))
        return 0, [], []
    monkeypatch.setattr('rinkspace.calibration_companion.keyframe_solver.cv2.solvePnPGeneric', capture_pnp)
    for problem in (data, changed):
        pose_candidates_from_points(scene, problem['observations'], problem['frames'][0], [500.])
    assert len(calls) == 2
    assert len(calls[0][0]) == 4
    np.testing.assert_array_equal(calls[0][0], calls[1][0])
    np.testing.assert_array_equal(calls[0][1], calls[1][1])

    # The deliberately behind-camera check is allowed into the diagnostic;
    # it cannot reject the branch selected using fit objects.
    report = before.solve(max_nfev=2)
    check = next(row for row in report['after']['observations'] if row['id'] == 'check')
    assert check['min_depth_scene_units'] < 0
    assert report['after']['any_observed_geometry_behind_camera']


def test_fit_geometry_still_requires_positive_initial_depth(scene_problem):
    scene, data = scene_problem
    scene.features['F0']['center_xyz'][2] = -20.
    with pytest.raises(ValueError, match='initial view puts observed geometry behind camera'):
        KeyframeProblem(scene, data)


def test_unknown_advert_height_is_never_zero(scene_problem):
    scene, _ = scene_problem
    with pytest.raises(ValueError, match='unknown board/ad height is not zero'):
        scene.point({'kind': 'advert_anchor', 'advert_id': 'unknown_height'})


@pytest.mark.parametrize(('missing_field', 'expected_error'), [
    ('attachment_s', 'unknown advert attachment_s'),
    ('anchor_xyz', 'unknown advert anchor_xyz'),
])
def test_unknown_advert_attachment_is_not_projected(scene_problem, missing_field, expected_error):
    scene, _ = scene_problem
    advert = scene.ads['unknown_height']
    advert['anchor_xyz'] = [.5, 0., .4]
    advert[missing_field] = None
    with pytest.raises(ValueError, match=expected_error):
        scene.point({'kind': 'advert_anchor', 'advert_id': 'unknown_height'})
