"""Synthetic perimeter regressions, never evidence of real-video accuracy."""
import copy
import json

import numpy as np
import pytest

from rinkspace.calibration_companion import Camera, Scene, look_at_rvec
from rinkspace.calibration_companion.joint_scene import (
    JointSceneProblem,
    SharedGeometry,
    landmark_boundary_axes,
    landmark_corner_dependencies,
)


RADII = {'upper_left': .7, 'upper_right': 1.1,
         'lower_left': .9, 'lower_right': 1.3}
JOINS = [
    ('upper_long_side', 'upper_left', 0, -4.3, [-4.3, 3.]),
    ('upper_long_side', 'upper_right', 0, 3.9, [3.9, 3.]),
    ('lower_long_side', 'lower_left', 0, -4.1, [-4.1, -3.]),
    ('lower_long_side', 'lower_right', 0, 3.7, [3.7, -3.]),
    ('left_end', 'lower_left', 1, -2.1, [-5., -2.1]),
    ('left_end', 'upper_left', 1, 2.3, [-5., 2.3]),
    ('right_end', 'lower_right', 1, -1.7, [5., -1.7]),
    ('right_end', 'upper_right', 1, 1.9, [5., 1.9]),
]
CORNER_ENDPOINTS = {
    'upper_right': ([5., 1.9], [3.9, 3.]),
    'upper_left': ([-4.3, 3.], [-5., 2.3]),
    'lower_left': ([-5., -2.1], [-4.1, -3.]),
    'lower_right': ([3.7, -3.], [5., -1.7]),
}


@pytest.fixture
def perimeter_scene(tmp_path):
    data = {
        'schema_version': 'rinkspace.scene3d.v1', 'revision': 'synthetic-perimeter',
        'coordinate_system': {'unit': 'synthetic', 'metres_per_unit': None},
        'parameters': {'bounds': [-5., -3., 5., 3.], 'corner_radius': .8,
                       'corner_radii': RADII.copy(), 'board_height': .6},
        'ice_features': [{'id': 'centre_ring', 'kind': 'circle',
                          'center_xyz': [0., 0., 0.], 'radius': .5}],
        'ice_boundary': {'points_xyz': [[-5., -3., 0.], [5., -3., 0.],
                                         [5., 3., 0.], [-5., 3., 0.]]},
        'board_surface': {'perimeter_length': 10., 'panels': [
            {'s_range': [0., 10.], 'bottom_vertex_indices': [0, 1]}]},
        'meshes': [{'name': 'rink_boards', 'positions': [
            [-5., -3., 0.], [5., -3., 0.], [-5., -3., .6], [5., -3., .6]]}],
        'advertisements': [],
    }
    path = tmp_path / 'scene.json'
    path.write_text(json.dumps(data), encoding='utf-8')
    return Scene(path)


def geometry_with_point(scene, **point):
    return SharedGeometry(scene, [], np.empty(0),
                          {'detail': {'z_fraction': .4, **point}})


@pytest.mark.parametrize('side,corner,axis,along,join', JOINS)
@pytest.mark.parametrize('edge', ['top', 'contact'])
def test_adjacent_board_continuation_has_no_artificial_bridge(perimeter_scene, side, corner, axis, along, join, edge):
    geom = SharedGeometry(perimeter_scene, [], np.empty(0))
    chain = geom.board_side(dict(side=side, continue_corner=corner, edge=edge), 30)
    straight = geom.board_side(dict(side=side, edge=edge), 30)
    arc = geom.board_side(dict(side=corner, edge=edge), 30)
    assert chain.shape == (59, 3)
    np.testing.assert_allclose(chain[29, :2], join, atol=1e-12)
    assert np.max(np.linalg.norm(np.diff(chain, axis=0), axis=1)) < .4
    assert set(map(tuple, np.round(chain, 12))) == set(map(tuple, np.round(np.vstack([straight, arc]), 12)))
    np.testing.assert_allclose(chain[:, 2], .6 if edge == 'top' else 0.)


def test_board_continuation_rejects_nonadjacent_or_ambiguous_domain(perimeter_scene):
    geom = SharedGeometry(perimeter_scene, [], np.empty(0))
    for side, corner in [('right_end', 'upper_left'), ('upper_right', 'lower_right')]:
        with pytest.raises(ValueError, match='adjacent corner'):
            geom.board_side(dict(side=side, continue_corner=corner), 30)


def test_existing_world_coordinate_meaning_is_preserved(perimeter_scene):
    for side, expected in [
        ('upper_long_side', [1.2, 3., .24]),
        ('lower_long_side', [1.2, -3., .24]),
        ('left_end', [-5., 1.2, .24]),
        ('right_end', [5., 1.2, .24]),
    ]:
        geom = geometry_with_point(perimeter_scene, side=side, along=1.2)
        np.testing.assert_allclose(geom.board_landmark('detail'), expected)
        assert geom.board_landmark_segment('detail') == side
    # These two forms predate the full-perimeter extension. Their old X and
    # height-fraction conventions must not be reinterpreted as arc length.
    right = geometry_with_point(perimeter_scene, side='upper_long_side',
                                allow_corner='upper_right', along=4.45)
    left = geometry_with_point(perimeter_scene, side='upper_long_side',
                               allow_corner='upper_left', along=-4.65)
    np.testing.assert_allclose(right.board_landmark('detail'),
                               [4.45, 1.9 + 1.1*np.sqrt(.75), .24])
    np.testing.assert_allclose(left.board_landmark('detail'),
                               [-4.65, 2.3 + .7*np.sqrt(.75), .24])


@pytest.mark.parametrize('side,corner,axis,junction,xy', JOINS)
def test_each_straight_corner_join_is_position_and_tangent_continuous(
        perimeter_scene, side, corner, axis, junction, xy):
    geom = geometry_with_point(perimeter_scene, side=side,
                                allow_corner=corner, along=junction)
    point = geom.board_landmarks['detail']
    step = 1e-6
    samples = []
    for delta in [-step, 0., step]:
        point['along'] = junction + delta
        samples.append(geom.board_landmark('detail'))
    np.testing.assert_allclose(samples[1], [*xy, .24], atol=1e-12)
    tangent = np.zeros(3); tangent[axis] = 1.
    np.testing.assert_allclose((samples[1]-samples[0])/step, tangent, atol=1e-6)
    np.testing.assert_allclose((samples[2]-samples[1])/step, tangent, atol=1e-6)
    # Move into the arc, which must depend only on the named corner radius.
    direction = -1 if ('left' in corner if axis == 0 else 'lower' in corner) else 1
    point['along'] = junction + direction*RADII[corner]/2
    assert geom.board_landmark_segment('detail') == corner + '_corner'
    assert np.all(geom.board_landmark_margins('detail') > 0)
    original = geom.board_landmark('detail')
    for other in RADII:
        if other != corner:
            geom.corner_radii[other] += .11
    np.testing.assert_array_equal(geom.board_landmark('detail'), original)


@pytest.mark.parametrize('corner', list(RADII))
def test_direct_corner_fraction_has_correct_endpoints_orientation_and_height(
        perimeter_scene, corner):
    geom = geometry_with_point(perimeter_scene, side=corner,
                                along_parameterization='corner_fraction', along=0.)
    radius = RADII[corner]
    centre = np.array([5.-radius if 'right' in corner else -5.+radius,
                       3.-radius if 'upper' in corner else -3.+radius])
    points = []
    for fraction in [0., .25, .5, .75, 1.]:
        geom.board_landmarks['detail']['along'] = fraction
        points.append(geom.board_landmark('detail'))
        np.testing.assert_allclose(geom.board_landmark_margins('detail'),
                                   [fraction, 1.-fraction])
    points = np.asarray(points)
    np.testing.assert_allclose(points[[0, -1], :2], CORNER_ENDPOINTS[corner], atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(points[:, :2]-centre, axis=1), radius)
    np.testing.assert_allclose(points[:, 2], .24)
    radial = points[:, :2]-centre
    # Each consecutive pair sweeps counterclockwise, including lower corners.
    assert np.all(radial[:-1, 0]*radial[1:, 1]-radial[:-1, 1]*radial[1:, 0] > 0)
    np.testing.assert_allclose(points[:, :2],
                               geom.board_side({'side': corner}, 5)[:, :2])


def make_problem(scene, *, outside_guards=False, fit_centres=True,
                 variable_scene=True, fit_pose=False):
    """Four cameras see all eight distinct board regions of one synthetic rink."""
    points = []
    for side in ['upper_long_side', 'lower_long_side', 'left_end', 'right_end']:
        axis = 0 if side.endswith('long_side') else 1
        values = [-4.95, 4.95] if axis == 0 else [-2.95, 2.95]
        for end, along in enumerate(values if outside_guards else [0.]):
            points.append({'id': f'{side}_{end}', 'side': side, 'along': along})
    for side, corner, axis, junction, _ in JOINS:
        direction = -1 if ('left' in corner if axis == 0 else 'lower' in corner) else 1
        points.append({'id': f'{side}_{corner}', 'side': side, 'allow_corner': corner,
                       'along': junction + direction*RADII[corner]/2})
    for corner in RADII:
        points.append({'id': corner, 'side': corner,
                       'along_parameterization': 'corner_fraction', 'along': .4})
    geometry = SharedGeometry(scene, [], np.empty(0), {
        p['id']: {**p, 'z_fraction': .4} for p in points})
    installations, frames, observations = [], [], []
    for name, centre in [
        ('main', [0., -12., 7.]), ('secondary', [7., -10., 6.]),
        ('low_end', [-13., -1., 4.]), ('high', [8., 9., 12.]),
    ]:
        centre = np.asarray(centre)
        installations.append({'id': name, 'initial_center_xyz': centre.tolist(),
                              'fit_center': fit_centres,
                              'center_bounds_xyz': [(centre-2).tolist(), (centre+2).tolist()]})
        for n in range(2):
            fid = f'{name}_{n}'; focal = 700.+n*80.
            rvec = look_at_rvec(centre, [.15*n, -.10*n, 0.])
            frames.append({'id': fid, 'installation_id': name, 'image_size_wh': [1280, 720],
                           'principal_point_px': [640., 360.], 'fit_pose': fit_pose,
                           'initial_rvec_world_to_camera': rvec.tolist(),
                           'initial_focal_px': focal, 'focal_bounds_px': [400., 1200.]})
            camera = Camera(centre, rvec,
                            [[focal, 0., 640.], [0., focal, 360.], [0., 0., 1.]], [])
            for point in points:
                observations.append({'id': f'{fid}_{point["id"]}', 'frame_id': fid,
                    'type': 'point', 'use': 'fit', 'sigma_px': 1.,
                    'target': {'kind': 'board_landmark', 'landmark_id': point['id']},
                    'pixels_uv': camera.project([geometry.board_landmark(point['id'])])[0][0].tolist()})
    landmarks = []
    for point in points:
        direct = point['side'] in RADII
        along_bound = [0., 1.] if direct else (
            [-4.999, 4.999] if point['side'].endswith('long_side') else [-2.999, 2.999])
        landmarks.append({**{k:v for k,v in point.items() if k != 'along'},
            'advertisement_instance_id': 'physical_' + point['id'],
            'label': 'repeated_brand_name_does_not_merge_instances',
            'initial_along': point['along'], 'bounds_along': along_bound,
            'initial_z_fraction': .4, 'bounds_z_fraction': [.05, .95], 'fit': True,
            'evidence_observation_ids': [o['id'] for o in observations
                if o['target']['landmark_id'] == point['id']]})
    specs = []
    if variable_scene:
        all_corner_obs = [o['id'] for o in observations if o['target']['landmark_id'] in RADII]
        specs = [
            {'id': 'length', 'kind': 'boundary_length', 'bounds': [8., 12.],
             'evidence_observation_ids': all_corner_obs},
            {'id': 'width', 'kind': 'boundary_width', 'bounds': [4., 8.],
             'evidence_observation_ids': all_corner_obs},
            {'id': 'height', 'kind': 'board_height', 'bounds': [.2, 1.5],
             'evidence_observation_ids': all_corner_obs},
        ]
        specs.extend({'id': corner, 'kind': 'boundary_corner_radius', 'corner': corner,
                       'bounds': [.2, 1.8], 'evidence_observation_ids': [o['id'] for o in observations
                           if o['target']['landmark_id'] == corner]} for corner in RADII)
    return {'synthetic': True, 'scene_sha256': scene.sha256,
            'geometry_mode': 'shared_scene_joint', 'scene_parameters': specs,
            'installations': installations, 'frames': frames,
            'observations': observations, 'board_landmarks': landmarks}


def test_dimension_height_corner_and_active_domain_derivatives_are_in_sparse_graph(perimeter_scene):
    data = make_problem(perimeter_scene, outside_guards=True)
    problem = JointSceneProblem(perimeter_scene, data)
    vector = problem.x0.copy()
    # Exercise the fraction guard too, outside optimizer bounds on purpose:
    # residual must still declare its dependence if a diagnostic evaluates it.
    vector[problem.landmark_slices['lower_right'].start] = 1.1
    sparse = problem.jac_sparsity.toarray()
    numerical_columns = {}
    for index, name in enumerate(problem.parameter_names):
        plus, minus = vector.copy(), vector.copy()
        plus[index] += 1e-6; minus[index] -= 1e-6
        derivative = (problem.residual(plus)-problem.residual(minus))/2e-6
        numerical_columns[name] = derivative
        assert np.all(sparse[np.abs(derivative)>2e-5, index]), name
    guard_start = 4*len(data['observations'])
    for side in ['upper_long_side', 'lower_long_side', 'left_end', 'right_end']:
        for end in (0, 1):
            lid = f'{side}_{end}'
            row = guard_start + 2*problem.active_board_landmark_ids.index(lid) + end
            assert problem.residual(vector)[row] < -100.
            along = problem.landmark_slices[lid].start
            assert sparse[row, along]
            dimension = 'length' if side.endswith('long_side') else 'width'
            assert abs(numerical_columns['scene:'+dimension][row]) > 100.
            assert abs(numerical_columns[problem.parameter_names[along]][row]) > 100.
    # The straight projection uses its normal boundary dimension; corner
    # projection uses both. End-domain guards also depend on the other axis.
    for point in data['board_landmarks']:
        direct_or_arc = point['side'] in RADII or point.get('allow_corner')
        axes = (0, 1) if direct_or_arc else ((1,) if point['side'].endswith('long_side') else (0,))
        assert tuple(landmark_boundary_axes(point)) == axes
        expected_corner = (point['side'],) if point['side'] in RADII else (
            (point['allow_corner'],) if point.get('allow_corner') else ())
        assert tuple(landmark_corner_dependencies(point)) == expected_corner
        observation_index = next(i for i,o in enumerate(data['observations'])
                                 if o['target']['landmark_id'] == point['id'])
        pixel_rows = slice(4*observation_index, 4*observation_index+2)
        # Check the actual numerical effects, not just helper declarations.
        for axis, dimension in [(0, 'length'), (1, 'width')]:
            derivative = numerical_columns['scene:'+dimension][pixel_rows]
            if axis in axes:
                assert np.linalg.norm(derivative) > .01
            else:
                np.testing.assert_allclose(derivative, 0., atol=1e-7)
        assert np.linalg.norm(numerical_columns['scene:height'][pixel_rows]) > .01
        for corner in RADII:
            derivative = numerical_columns['scene:'+corner][pixel_rows]
            if corner in expected_corner:
                assert np.linalg.norm(derivative) > .01
            else:
                np.testing.assert_allclose(derivative, 0., atol=1e-7)


def test_four_installations_have_independent_centres_and_share_one_field(perimeter_scene):
    data = make_problem(perimeter_scene, fit_pose=True)
    problem = JointSceneProblem(perimeter_scene, data)
    assert set(problem.center_slices) == {'main', 'secondary', 'low_end', 'high'}
    assert len({tuple(problem.camera(problem.x0, f'{name}_0').center_xyz)
                for name in problem.center_slices}) == 4
    changed = problem.x0.copy(); changed[problem.center_slices['secondary'].start] += .25
    for name in problem.center_slices:
        for n in (0, 1):
            before = problem.camera(problem.x0, f'{name}_{n}')
            after = problem.camera(changed, f'{name}_{n}')
            np.testing.assert_allclose(after.center_xyz-before.center_xyz,
                                       [.25, 0., 0.] if name == 'secondary' else [0., 0., 0.])
    # Every camera's frame residual responds to the SAME height variable.
    changed = problem.x0.copy(); changed[problem.parameter_names.index('scene:height')] += .05
    difference = problem.residual(changed)-problem.residual(problem.x0)
    for frame in data['frames']:
        offsets = [4*i for i,o in enumerate(data['observations']) if o['frame_id'] == frame['id']]
        assert any(np.linalg.norm(difference[start:start+2]) > .01 for start in offsets)
    assert sum(name == 'scene:height' for name in problem.parameter_names) == 1
    assert len(problem.frame_slices) == 8


def test_advert_pixels_drive_recovery_on_all_sides_and_corners_but_checks_do_not(perimeter_scene):
    data = make_problem(perimeter_scene, fit_centres=False, variable_scene=False)
    for landmark in data['board_landmarks']:
        landmark['initial_along'] += .07
        landmark['initial_z_fraction'] = .3
    check_frame = copy.deepcopy(data['frames'][0]); check_frame['id'] = 'check_frame'
    data['frames'].append(check_frame); data['split_mode'] = 'frame_object'
    check = copy.deepcopy(data['observations'][0])
    check.update(id='unseen_pixels', frame_id='check_frame', use='check')
    data['observations'].append(check)
    first = JointSceneProblem(perimeter_scene, data)
    changed = copy.deepcopy(data)
    changed['observations'][-1]['pixels_uv'] = [99999., -88888.]
    second = JointSceneProblem(perimeter_scene, changed)
    np.testing.assert_array_equal(first.x0, second.x0)
    np.testing.assert_array_equal(first.residual(first.x0), second.residual(second.x0))
    altered_fit = copy.deepcopy(data)
    altered_fit['observations'][0]['pixels_uv'] = (np.asarray(check['pixels_uv'])+[4., -3.]).tolist()
    third = JointSceneProblem(perimeter_scene, altered_fit)
    np.testing.assert_allclose(third.residual(third.x0)[:2]-first.residual(first.x0)[:2], [-4., 3.])
    report = first.solve(max_nfev=80)
    assert report['after']['overall']['fit']['rmse_px'] < 1e-4
    for initial in data['board_landmarks']:
        estimate = first.geometry(first.result.x).board_landmarks[initial['id']]
        assert estimate['along'] == pytest.approx(initial['initial_along']-.07, abs=2e-5)
        assert estimate['z_fraction'] == pytest.approx(.4, abs=2e-5)


@pytest.mark.parametrize('mutation', ['missing_fraction_type', 'world_coordinate_bounds', 'nonadjacent_corner'])
def test_ambiguous_or_wrong_surface_parameterizations_are_rejected(perimeter_scene, mutation):
    data = make_problem(perimeter_scene, fit_centres=False, variable_scene=False)
    if mutation == 'nonadjacent_corner':
        spec = next(p for p in data['board_landmarks'] if p['id'] == 'lower_long_side_0')
        spec['allow_corner'] = 'upper_right'
    else:
        spec = next(p for p in data['board_landmarks'] if p['id'] == 'lower_right')
        if mutation == 'missing_fraction_type':
            del spec['along_parameterization']
        else:
            spec['bounds_along'] = [-4., 4.]
    with pytest.raises(ValueError):
        JointSceneProblem(perimeter_scene, data)
