"""Synthetic numerical checks only; no test fixture is real-video evidence."""
import copy
import json

import numpy as np
import pytest

from rinkspace.calibration_companion import Camera, Scene, look_at_rvec
from rinkspace.calibration_companion.joint_scene import JointSceneProblem, SharedGeometry


@pytest.fixture
def joint_fixture(tmp_path):
    features = [dict(id='F', kind='circle', center_xyz=[-1., .3, 0.], radius=.5),
                dict(id='UNSEEN', kind='circle', center_xyz=[2., -1., 0.], radius=.4),
                dict(id='L', kind='polyline', segments_xyz=[[[1., 2., 0.], [1., -2., 0.]]])]
    data = dict(schema_version='rinkspace.scene3d.v1', revision='synthetic',
        coordinate_system={'unit': 'synthetic'}, ice_features=features,
        parameters=dict(bounds=[-5., -3., 5., 3.], corner_radius=.8, board_height=.6),
        ice_boundary={'points_xyz': [[-5., -3., 0.], [5., -3., 0.], [5., 3., 0.], [-5., 3., 0.]]},
        board_surface={'perimeter_length': 10., 'panels': [
            {'s_range': [0., 10.], 'bottom_vertex_indices': [0, 1]}]},
        meshes=[{'name': 'rink_boards', 'positions': [[-5., -3., 0.], [5., -3., 0.],
                                                       [-5., -3., .6], [5., -3., .6]]}],
        advertisements=[])
    path = tmp_path / 'scene.json'; path.write_text(json.dumps(data), encoding='utf8')
    scene = Scene(path)
    true_specs = [dict(id='Fx', kind='circle_center_x', feature_id='F', bounds=[-2., 0.]),
                  dict(id='Fy', kind='circle_center_y', feature_id='F', bounds=[-.5, 1.]),
                  dict(id='Fr', kind='circle_radius', feature_id='F', bounds=[.3, 1.]),
                  dict(id='L', kind='line_offset', feature_id='L', bounds=[-.5, .5]),
                  dict(id='height', kind='board_height', bounds=[.3, 1.2])]
    truth = np.array([-1.2, .45, .65, -.2, .8])
    geometry = SharedGeometry(scene, true_specs, truth)
    installs, frames, observations = [], [], []
    for group, center in [('main', [0., -9., 5.]), ('high', [6., 6., 8.])]:
        center = np.asarray(center)
        installs.append(dict(id=group, initial_center_xyz=(center + [.06, -.04, .05]).tolist(),
            center_bounds_xyz=[(center - 2).tolist(), (center + 2).tolist()]))
        for index in range(2):
            fid = f'{group}_{index}'; focal = 600. + index * 60
            rvec = look_at_rvec(center, [index * .2, 0., 0.])
            frame = dict(id=fid, installation_id=group, image_size_wh=[800, 600],
                principal_point_px=[400., 300.], initial_rvec_world_to_camera=(rvec + [.002, -.001, .001]).tolist(),
                initial_focal_px=focal * 1.01, focal_bounds_px=[300., 1200.])
            frames.append(frame)
            camera = Camera(center, rvec, [[focal, 0., 400.], [0., focal, 300.], [0., 0., 1.]], [])
            # Distributed synthetic 3D controls give a well-constrained camera,
            # allowing a test of actual shared scene recovery instead of a gauge.
            for j, xyz in enumerate([[-4., -2., 0.], [4., -2., 0.], [4., 2., 0.],
                                     [-4., 2., 0.], [-3., 1., 1.5], [3., -1., 2.]]):
                observations.append(dict(id=f'{fid}_point{j}', frame_id=fid, type='point', use='fit',
                    target=dict(kind='explicit_point', xyz=xyz, provenance='synthetic'),
                    pixels_uv=camera.project([xyz])[0].tolist(), sigma_px=1.))
            for name, target in [('F', dict(kind='circle_outline', feature_id='F')),
                                 ('L', dict(kind='line_segment', feature_id='L')),
                                 ('height', dict(kind='board_side', side='upper_long_side', edge='top'))]:
                xyz = geometry.points(target, 'curve', 24)
                observations.append(dict(id=f'{fid}_{name}', frame_id=fid, type='curve', use='fit',
                    physical_group_id=name, target=target, pixels_uv=camera.project(xyz)[0].tolist(), sigma_px=1.))
            xyz = geometry.points(dict(kind='circle_outline', feature_id='UNSEEN'), 'curve', 24)
            observations.append(dict(id=f'{fid}_check', frame_id=fid, type='curve', use='check',
                target=dict(kind='circle_outline', feature_id='UNSEEN'), pixels_uv=camera.project(xyz)[0].tolist(), sigma_px=1.))
    for spec in true_specs:
        name = spec.get('feature_id', 'height')
        spec['evidence_observation_ids'] = [o['id'] for o in observations if o['id'].endswith('_' + name)]
    problem = dict(synthetic=True, scene_sha256=scene.sha256, geometry_mode='shared_scene_joint',
                   scene_parameters=true_specs, installations=installs, frames=frames,
                   observations=observations, curve_samples=96)
    return scene, problem, truth


def test_recovers_shared_scene_and_centers(joint_fixture):
    scene, data, truth = joint_fixture
    problem = JointSceneProblem(scene, data)
    before = copy.deepcopy(scene.data)
    report = problem.solve(max_nfev=90)
    np.testing.assert_allclose(problem.result.x[:5], truth, atol=.004)
    assert report['after']['overall']['fit']['rmse_px'] < .025
    assert report['after']['overall']['check']['rmse_px'] < .025
    assert report['after']['fit_geometry_valid']
    for group in ('main', 'high'):
        a, b = [row['camera']['center_xyz'] for row in report['frames'] if row['installation_id'] == group]
        np.testing.assert_array_equal(a, b)
    assert scene.data == before
    exported = problem.export_scene_data(problem.result.x)
    assert exported['ice_features'][1] == scene.data['ice_features'][1]
    assert exported['parameters']['board_height'] == pytest.approx(.8, abs=.004)
    assert exported['meshes'][0]['positions'][2][2] == exported['parameters']['board_height']
    assert not exported['uncertainty']['physical_accuracy_verified']


def test_checks_cannot_change_fit_and_unobserved_parameter_retained(joint_fixture):
    scene, data, _ = joint_fixture
    data['scene_parameters'].append(dict(id='unseen_x', kind='circle_center_x',
        feature_id='UNSEEN', bounds=[-3., 3.], prior=dict(mean=0., sigma=.1)))
    first = JointSceneProblem(scene, data)
    changed = copy.deepcopy(data)
    for ob in changed['observations']:
        if ob['use'] == 'check':
            ob['pixels_uv'] = (np.asarray(ob['pixels_uv']) + [10000., -30000.]).tolist()
    second = JointSceneProblem(scene, changed)
    np.testing.assert_array_equal(first.x0, second.x0)
    np.testing.assert_array_equal(first.residual(first.x0), second.residual(second.x0))
    assert first.retained_parameters[0]['value'] == 2.
    assert 'scene:unseen_x' not in first.parameter_names
    assert first.export_scene_data(first.x0)['ice_features'][1]['center_xyz'][0] == 2.


def test_sparse_dependencies_cover_actual_derivatives(joint_fixture):
    scene, data, _ = joint_fixture
    # Lens and individual corner derivatives must be part of the sparse graph.
    data['installations'][1]['radial_distortion'] = dict(initial_k1_image=-.1,
        scale_image=800., fit=True, bounds=[-.3, .1])
    first = JointSceneProblem(scene, data)
    sparse = first.jac_sparsity.toarray()
    for index in range(len(first.x0)):
        a, b = first.x0.copy(), first.x0.copy(); a[index] += 1e-6; b[index] -= 1e-6
        derivative = (first.residual(a) - first.residual(b)) / 2e-6
        assert np.all(sparse[np.abs(derivative) > 1e-6, index] != 0), first.parameter_names[index]


def test_radial_model_is_shared_in_image_units(joint_fixture):
    scene, data, _ = joint_fixture
    data['installations'][1]['radial_distortion'] = dict(initial_k1_image=-.4,
        scale_image=800., fit=True, bounds=[-.6, 0.])
    problem = JointSceneProblem(scene, data)
    for fid in ('high_0', 'high_1'):
        cam = problem.camera(problem.x0, fid)
        assert cam.distortion_opencv[0] == pytest.approx(-.4 * (cam.K_px[0, 0] / 800.)**2)


def test_corner_geometry_is_independent_and_requires_mesh_rebuild(joint_fixture):
    scene, data, _ = joint_fixture
    target = dict(kind='board_corner', corner='upper_left', edge='contact')
    data['observations'].append(dict(id='corner_fit', frame_id='main_0', type='curve',
        target=target, pixels_uv=[[100., 200.], [110., 210.]], sigma_px=2., use='fit'))
    data['scene_parameters'].append(dict(id='corner', kind='boundary_corner_radius', corner='upper_left',
        bounds=[.2, 2.], evidence_observation_ids=['corner_fit']))
    problem = JointSceneProblem(scene, data)
    x = problem.x0.copy(); x[5] = 1.1
    geometry = problem.geometry(x)
    assert geometry.corner_radii['upper_left'] == 1.1
    assert geometry.corner_radii['upper_right'] == .8
    xyz = geometry.points(target, 'curve', 32)
    assert xyz[0, 1] == pytest.approx(3.)
    assert xyz[-1, 0] == pytest.approx(-5.)
    assert problem.export_scene_data(x)['joint_refinement']['boundary_mesh_rebuild_required']


def test_right_end_candidate_domain_allows_both_corners_without_requiring_visibility(joint_fixture):
    scene, data, _ = joint_fixture
    problem = JointSceneProblem(scene, data)
    geometry = problem.geometry(problem.x0)
    for edge in ('contact', 'top'):
        target = dict(side='right_end', edge=edge)
        straight = geometry.board_side(target, 9)
        lower = geometry.board_side(dict(side='lower_right', edge=edge), 9)
        upper = geometry.board_side(dict(side='upper_right', edge=edge), 9)
        candidates = geometry.board_side({**target, 'allow_adjacent_corners': True}, 9)
        np.testing.assert_allclose(candidates, np.vstack([lower[:-1], straight, upper[1:]]))
        assert len(candidates) == 25
        assert np.allclose(candidates[:, 2], 0. if edge == 'contact' else geometry.parameters['board_height'])
    with pytest.raises(ValueError, match='requires right_end'):
        geometry.board_side(dict(side='upper_long_side', allow_adjacent_corners=True), 9)


def test_frame_object_split_does_not_allow_same_frame_ring_leakage(joint_fixture):
    scene, data, _ = joint_fixture
    data['split_mode'] = 'frame_object'
    data['observations'].append(dict(id='badcheck', frame_id='main_0', type='curve',
        use='check', target=dict(kind='circle_outline', feature_id='F'), pixels_uv=[[0., 0.]], sigma_px=1.))
    with pytest.raises(ValueError, match='whole-object split violation'):
        JointSceneProblem(scene, data)


def test_iteration_callback_can_checkpoint_and_warm_resume(joint_fixture):
    scene, data, truth = joint_fixture
    problem = JointSceneProblem(scene, data)
    checkpoints = []

    def checkpoint(intermediate_result):
        checkpoints.append((intermediate_result.x.copy(), float(intermediate_result.cost)))
        raise StopIteration

    stopped = problem.solve(max_nfev=20, callback=checkpoint)
    assert checkpoints and not stopped['optimizer']['success']
    np.testing.assert_array_equal(problem.result.x, checkpoints[-1][0])
    resumed = problem.solve(max_nfev=90, x0=checkpoints[-1][0])
    assert resumed['optimizer']['warm_started']
    np.testing.assert_allclose(problem.result.x[:5], truth, atol=.004)
    with pytest.raises(ValueError, match='vector size'):
        problem.solve(x0=[1.])


def add_board_detail_observations(data):
    """Three distinct latent details; repeated logo names do not tie coordinates."""
    specs = []
    for lid, side, along, zfrac, initial in [('logo_copy_1', 'upper_long_side', -1.4, .5, -1.),
                                           ('logo_copy_2', 'upper_long_side', 1.6, .7, 1.2),
                                           ('letter_corner', 'left_end', .2, .4, 0.)]:
        ids = []
        for frame in data['frames']:
            group = frame['installation_id']; index = int(frame['id'][-1])
            center = np.array([0., -9., 5.] if group == 'main' else [6., 6., 8.])
            focal = 600. + index * 60
            camera = Camera(center, look_at_rvec(center, [index*.2, 0., 0.]),
                            [[focal, 0., 400.], [0., focal, 300.], [0., 0., 1.]], [])
            xyz = [along, 3., zfrac*.8] if side == 'upper_long_side' else [-5., along, zfrac*.8]
            oid = f"{frame['id']}_{lid}"; ids.append(oid)
            data['observations'].append(dict(id=oid, frame_id=frame['id'], type='point', use='fit',
                target=dict(kind='board_landmark', landmark_id=lid), pixels_uv=camera.project([xyz])[0].tolist(), sigma_px=1.))
        specs.append(dict(id=lid, advertisement_instance_id=lid, label='same_visible_logo', side=side,
            initial_along=initial, bounds_along=[-4., 4.], initial_z_fraction=.3,
            bounds_z_fraction=[.05, .95], fit=True, evidence_observation_ids=ids,
            provenance={'status': 'synthetic_correspondence_fixture'}))
    data['board_landmarks'] = specs


def test_board_details_are_joint_latent_surface_points(joint_fixture):
    scene, data, _ = joint_fixture
    add_board_detail_observations(data)
    problem = JointSceneProblem(scene, data)
    report = problem.solve(max_nfev=120)
    estimates = {row['id']: row for row in report['board_landmarks']}
    np.testing.assert_allclose(estimates['logo_copy_1']['estimated_xyz'], [-1.4, 3., .4], atol=.004)
    np.testing.assert_allclose(estimates['logo_copy_2']['estimated_xyz'], [1.6, 3., .56], atol=.004)
    np.testing.assert_allclose(estimates['letter_corner']['estimated_xyz'], [-5., .2, .32], atol=.004)
    assert all(len(row['fit_installation_ids']) == 2 and not row['physical_xyz_verified'] for row in estimates.values())
    assert len(problem.export_scene_data(problem.result.x)['board_landmarks']) == 3
    # Finite differences verify the new latent-point and height sparse columns.
    sparse = problem.jac_sparsity.toarray()
    for index in [4] + [i for sl in problem.landmark_slices.values() for i in range(sl.start, sl.stop)]:
        a, b = problem.x0.copy(), problem.x0.copy(); a[index] += 1e-6; b[index] -= 1e-6
        derivative = (problem.residual(a) - problem.residual(b)) / 2e-6
        assert np.all(sparse[np.abs(derivative) > 1e-6, index] != 0)


def test_board_detail_check_pixels_do_not_change_estimation(joint_fixture):
    scene, data, _ = joint_fixture
    add_board_detail_observations(data)
    data['split_mode'] = 'frame_object'
    frame = copy.deepcopy(data['frames'][0]); frame['id'] = 'unseen_frame'; frame['fit_pose'] = False
    data['frames'].append(frame)
    check = copy.deepcopy(next(o for o in data['observations'] if o['target'].get('landmark_id') == 'logo_copy_1'))
    check.update(id='unseen_check', frame_id='unseen_frame', use='check')
    data['observations'].append(check)
    before = JointSceneProblem(scene, data)
    changed = copy.deepcopy(data); changed['observations'][-1]['pixels_uv'] = [[98765., -43210.]]
    after = JointSceneProblem(scene, changed)
    np.testing.assert_array_equal(before.residual(before.x0), after.residual(after.x0))
    assert before.landmark_estimates(before.x0)[0]['check_observation_ids'] == ['unseen_check']
    data['board_landmarks'][0]['evidence_observation_ids'].append('unseen_check')
    with pytest.raises(ValueError, match='own fit observations only'):
        JointSceneProblem(scene, data)


def test_same_advert_details_cannot_cross_fit_check_within_frame(joint_fixture):
    scene, data, _ = joint_fixture
    add_board_detail_observations(data)
    data['split_mode'] = 'frame_object'
    data['board_landmarks'][1]['advertisement_instance_id'] = 'logo_copy_1'
    next(o for o in data['observations'] if o['id'] == 'main_0_logo_copy_2')['use'] = 'check'
    with pytest.raises(ValueError, match='whole-object split violation'):
        JointSceneProblem(scene, data)


def test_board_detail_can_continue_smoothly_onto_identified_corner(joint_fixture):
    scene, data, _ = joint_fixture
    add_board_detail_observations(data)
    lid = 'logo_copy_2'
    detail = next(s for s in data['board_landmarks'] if s['id'] == lid)
    detail.update(allow_corner='upper_right', initial_along=4.6, bounds_along=[-4., 4.95])
    data['scene_parameters'].append(dict(id='ur', kind='boundary_corner_radius', corner='upper_right',
        bounds=[.2, 2.], evidence_observation_ids=detail['evidence_observation_ids']))
    problem = JointSceneProblem(scene, data)
    geometry = problem.geometry(problem.x0)
    xyz = geometry.board_landmark(lid)
    np.testing.assert_allclose(xyz, [4.6, 2.2 + np.sqrt(.8**2-.4**2), .18])
    assert np.all(geometry.board_landmark_margins(lid) > 0.)
    assert geometry.board_landmark_segment(lid) == 'upper_right_corner'
    assert next(r for r in problem.landmark_estimates(problem.x0) if r['id'] == lid)['surface_segment'] == 'upper_right_corner'
    # Position and tangent match at the straight/arc transition.
    junction = 5. - .8; eps = 1e-5
    geometry.board_landmarks[lid]['along'] = junction-eps; left = geometry.board_landmark(lid)
    geometry.board_landmarks[lid]['along'] = junction; middle = geometry.board_landmark(lid)
    geometry.board_landmarks[lid]['along'] = junction+eps; right = geometry.board_landmark(lid)
    np.testing.assert_allclose((right-middle)/eps, (middle-left)/eps, atol=1e-5)
    # The quarter-circle radius changes the XYZ and therefore the point residual.
    sparse = problem.jac_sparsity.toarray()
    for index in [5, problem.landmark_slices[lid].start]:
        a, b = problem.x0.copy(), problem.x0.copy(); a[index] += 1e-6; b[index] -= 1e-6
        derivative = (problem.residual(a)-problem.residual(b))/2e-6
        assert np.any(np.abs(derivative) > 1e-6)
        assert np.all(sparse[np.abs(derivative)>1e-6, index] != 0)
    detail['bounds_along'][1] = 5.
    with pytest.raises(ValueError, match='strictly below xmax'):
        JointSceneProblem(scene, data)


def test_upper_left_board_detail_surface_and_radius_jacobian(joint_fixture):
    scene, data, _ = joint_fixture
    add_board_detail_observations(data)
    lid = 'logo_copy_1'
    detail = next(s for s in data['board_landmarks'] if s['id'] == lid)
    detail.update(allow_corner='upper_left', initial_along=-4.6, bounds_along=[-4.95, 4.])
    data['scene_parameters'].append(dict(id='ul', kind='boundary_corner_radius', corner='upper_left',
        bounds=[.2, 2.], evidence_observation_ids=detail['evidence_observation_ids']))
    problem = JointSceneProblem(scene, data); geometry = problem.geometry(problem.x0)
    np.testing.assert_allclose(geometry.board_landmark(lid), [-4.6, 2.2+np.sqrt(.8**2-.4**2), .18])
    assert geometry.board_landmark_segment(lid) == 'upper_left_corner'
    assert np.all(geometry.board_landmark_margins(lid) > 0.)
    assert next(r for r in problem.export_scene_data(problem.x0)['board_landmarks'] if r['id'] == lid)['surface_segment'] == 'upper_left_corner'
    # The quarter-circle joins the top board with the same point and tangent.
    junction = -5.+.8; eps = 1e-5
    geometry.board_landmarks[lid]['along'] = junction-eps; left = geometry.board_landmark(lid)
    geometry.board_landmarks[lid]['along'] = junction; middle = geometry.board_landmark(lid)
    geometry.board_landmarks[lid]['along'] = junction+eps; right = geometry.board_landmark(lid)
    np.testing.assert_allclose((right-middle)/eps, (middle-left)/eps, atol=1e-5)
    sparse = problem.jac_sparsity.toarray()
    for index in [5, problem.landmark_slices[lid].start]:
        a, b = problem.x0.copy(), problem.x0.copy(); a[index] += 1e-6; b[index] -= 1e-6
        derivative = (problem.residual(a)-problem.residual(b))/2e-6
        assert np.any(np.abs(derivative) > 1e-6)
        assert np.all(sparse[np.abs(derivative)>1e-6, index] != 0)
    # A left-corner permission does not silently permit the opposite end.
    geometry.board_landmarks[lid]['along'] = 4.6
    assert geometry.board_landmark_margins(lid)[1] < 0.
    detail['bounds_along'][0] = -5.
    with pytest.raises(ValueError, match='strictly above xmin'):
        JointSceneProblem(scene, data)
