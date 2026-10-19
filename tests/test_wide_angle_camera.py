"""Optical recovery and full Brown fold checks; not real-video accuracy tests."""
import json
import copy
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from rinkspace.calibration_companion import Camera, Scene, look_at_rvec
from rinkspace.calibration_companion.wide_angle_camera import (
    WideAngleJointProblem, brown_minimum_eigenvalue,
    brown_central_inverse, brown_projection_validity,
    nearest_valid_polyline_vectors,
    _brown_map_jacobian, rational_denominator, rational_minimum_denominator,
)


def test_full_brown_fold_check_matches_opencv_derivative():
    d = np.array([-.15, .05, .003, -.006, -.008])
    points = np.array([[.7, .4], [-.9, -.3], [.1, .8]])
    numerical = []
    for xy in points:
        J = np.zeros((2, 2)); step = 1e-6
        for axis in range(2):
            delta = np.eye(2)[axis] * step
            xyz = np.column_stack([[xy+delta, xy-delta], np.ones(2)])
            uv = cv2.projectPoints(xyz, np.zeros(3), np.zeros(3), np.eye(3), d)[0].reshape(2, 2)
            J[:, axis] = (uv[0]-uv[1])/(2*step)
        numerical.append(np.linalg.eigvalsh(J).min())
    np.testing.assert_allclose(brown_minimum_eigenvalue(points, d), numerical, atol=1e-8)
    # A k1-only check sees identity; k2 alone can still fold the map.
    assert brown_minimum_eigenvalue([[1., 0.]], [0., -1., 0., 0., 0.])[0] < 0.
    assert brown_minimum_eigenvalue([[1., 0.]], [0., 0., 0., 0., -1.])[0] < 0.


@pytest.mark.parametrize('rational', [False, True])
def test_thin_prism_full_nonsymmetric_jacobian_matches_opencv(rational):
    distortion = np.array([-.15, .05, .003, -.006, -.008, 0., 0., 0., .04, -.015, .03, .02])
    if rational:
        distortion[5:8] = [.12, -.03, .006]
    xy = np.array([[.7, .4], [-.9, -.3], [.1, .8]])
    prediction, a, b, c, d, margin = _brown_map_jacobian(xy, distortion)
    expected = cv2.projectPoints(np.column_stack([xy, np.ones(len(xy))]), np.zeros(3),
                                 np.zeros(3), np.eye(3), distortion)[0].reshape(-1, 2)
    np.testing.assert_allclose(prediction, expected, atol=1e-12)
    assert np.max(np.abs(b-c)) > .01
    for i, point in enumerate(xy):
        J = np.zeros((2, 2)); step = 1e-6
        for axis in range(2):
            delta = np.eye(2)[axis]*step
            xyz = np.column_stack([[point+delta, point-delta], np.ones(2)])
            uv = cv2.projectPoints(xyz, np.zeros(3), np.zeros(3), np.eye(3), distortion)[0].reshape(2, 2)
            J[:, axis] = (uv[0]-uv[1])/(2*step)
        np.testing.assert_allclose([[a[i], b[i]], [c[i], d[i]]], J, atol=1e-8)
        np.testing.assert_allclose(margin[i], np.linalg.eigvalsh((J+J.T)/2.).min(), atol=1e-8)
    camera = Camera([0., 0., 0.], [0., 0., 0.],
                    [[1000., 0., 960.], [0., 1000., 540.], [0., 0., 1.]], distortion)
    pixels = camera.project(np.column_stack([xy, np.ones(len(xy))]))[0]
    inverse = brown_central_inverse(camera, pixels)
    assert inverse['valid'].all()
    np.testing.assert_allclose(inverse['normalized_xy'], xy, atol=1e-7)
    # A thin-prism fold alone must be rejected even with zero radial terms.
    folded = np.zeros(12); folded[8] = -1.
    assert brown_minimum_eigenvalue([[1., 0.]], folded)[0] < 0.


def test_rational_denominator_rejects_pole_separated_positive_outer_branch():
    # D(t)=(1-t)(1-t/4). The endpoint beyond both poles is positive,
    # but the central optical domain ends at the first pole.
    distortion = np.zeros(8); distortion[5:7] = [-1.25, .25]
    rays = np.array([[.5, 0.], [3., 0.]])
    assert (rational_denominator(rays, distortion) > 0).all()
    np.testing.assert_allclose(rational_minimum_denominator(rays, distortion), [.703125, -.5625])
    camera = Camera([0., 0., 0.], [0., 0., 0.], np.eye(3), distortion)
    result = brown_projection_validity(camera, np.column_stack([rays, np.ones(2)]))
    assert result['valid'].tolist() == [True, False]
    inverse = brown_central_inverse(camera, result['uv'])
    assert inverse['valid'].all()
    assert (inverse['minimum_denominator_on_axis_path'] > 0).all()
    assert np.linalg.norm(inverse['normalized_xy'][1]-rays[1]) > 2.


def test_central_inverse_rejects_outer_fold_returning_into_image():
    camera = Camera([0., 0., 0.], [0., 0., 0.],
        [[1000., 0., 960.], [0., 1000., 540.], [0., 0., 1.]], [-.2, .03, 0., 0., -.005])
    xyz = np.array([[.5, .3, 1.], [2., 0., 1.]])
    result = brown_projection_validity(camera, xyz)
    assert result['valid'].tolist() == [True, False]
    assert result['minimum_eigenvalue'][1] < 0
    # Both raw projections lie inside a 1920x1080 image. The second is still
    # an invalid outer polynomial branch; clipping by pixels alone misses it.
    assert np.all((result['uv'] >= [0., 0.]) & (result['uv'] < [1920., 1080.]))
    inverse = brown_central_inverse(camera, result['uv'])
    assert inverse['valid'].all()
    np.testing.assert_allclose(inverse['normalized_xy'][0], xyz[0, :2], atol=1e-7)
    assert np.linalg.norm(inverse['normalized_xy'][1]-xyz[1, :2]) > .5
    unreachable = brown_central_inverse(camera, [[2960., 540.]])
    assert not unreachable['valid'][0]


def test_curve_error_excludes_folded_branch_without_penalizing_unmatched_extension():
    camera = Camera([0., 0., 0.], [0., 0., 0.],
        [[1000., 0., 960.], [0., 1000., 540.], [0., 0., 1.]], [-.2, .03, 0., 0., -.005])
    xyz = np.array([[-.2, 0., 1.], [.2, 0., 1.], [2., 0., 1.], [2.1, 0., 1.]])
    projected = camera.project(xyz)[0]
    observed = projected[2:3]
    problem = object.__new__(WideAngleJointProblem)
    problem._initializing_wide = False
    problem.frames = {'wide': {'installation_id': 'camera'}}
    problem.optical_specs = {'camera': {}}
    problem.samples = 4
    geometry = SimpleNamespace(points=lambda *args: xyz)
    observation = dict(frame_id='wide', type='curve', target={})
    delta, depth, fold = problem._error(None, observation, observed, geometry, {'wide': camera})
    # The fake return of the outer branch is an exact raw-pixel match, but is
    # not an admissible correspondence. Its negative Jacobian adds no fold
    # penalty to the valid, actually matched segment either.
    assert np.linalg.norm(delta) > 400.
    assert depth.min() > 0 and fold > 0
    expected, indices = nearest_valid_polyline_vectors(observed, projected, [True, True, False, False])
    np.testing.assert_allclose(delta, expected)
    assert indices.tolist() == [0]


@pytest.mark.parametrize('coverage,prism,aspect,rational', [(False, False, False, False), (True, False, False, False),
                                                 (True, True, False, False), (True, False, True, False),
                                                 (True, True, True, True)])
def test_planted_full_optics_recovered_and_serialized(tmp_path, coverage, prism, aspect, rational):
    data = dict(schema_version='rinkspace.scene3d.v1', parameters=dict(
        bounds=[-12., -9., 12., 9.], corner_radius=1., board_height=1.),
        ice_features=[], advertisements=[], ice_boundary={'points_xyz': [[-12., -9., 0.], [12., 9., 0.]]},
        board_surface={'perimeter_length': 84., 'panels': []},
        meshes=[dict(name='rink_boards', positions=[[-12., -9., 0.], [12., 9., 1.]])])
    path = tmp_path/'scene.json'; path.write_text(json.dumps(data))
    scene = Scene(path)
    center = np.array([2., -16., 18.]); rvec = look_at_rvec(center, [0., 0., 0.])
    focal, scale = 800., 640.; a = focal/scale
    truth = dict(k1_image=-.12, k2_image=.04, k3_image=.003, p1_image=.008,
                 p2_image=-.01, principal_x_px=647., principal_y_px=354.)
    if prism:
        truth.update(s1_image=.012, s2_image=-.007, s3_image=-.009, s4_image=.004)
    if aspect:
        truth['log_fy_over_fx'] = float(np.log(1.06))
    if rational:
        truth.update(k4_image=.12, k5_image=-.04, k6_image=.006)
    distortion = [truth['k1_image']*a*a, truth['k2_image']*a**4,
                  truth['p1_image']*a, truth['p2_image']*a, truth['k3_image']*a**6]
    if prism:
        distortion.extend([0., 0., 0., truth['s1_image']*a, truth['s2_image']*a**3,
                           truth['s3_image']*a, truth['s4_image']*a**3])
    if rational:
        distortion[5:8] = [truth['k4_image']*a**2, truth['k5_image']*a**4, truth['k6_image']*a**6]
    camera = Camera(center, rvec, [[focal, 0., 647.], [0., focal*(1.06 if aspect else 1.), 354.], [0., 0., 1.]], distortion)
    xyz = np.array([[x, y, 0.] for x in np.linspace(-11., 11., 7) for y in np.linspace(-8., 8., 5)])
    uv = camera.project(xyz)[0]
    observations = [dict(id=f'p{i}', frame_id='wide', type='point', use='fit', sigma_px=1.,
                        target=dict(kind='explicit_point', xyz=point.tolist(), provenance='synthetic planted point'),
                        pixels_uv=[pixel.tolist()]) for i, (point, pixel) in enumerate(zip(xyz, uv))]
    problem = dict(synthetic=True, geometry_mode='shared_scene_joint', scene_sha256=scene.sha256,
        installations=[dict(id='camera', initial_center_xyz=center.tolist(), fit_center=False,
                            radial_distortion=dict(initial_k1_image=-.3 if coverage else 0., fit=True, bounds=[-.5, .5], scale_image=scale))],
        frames=[dict(id='wide', installation_id='camera', image_size_wh=[1280, 720], principal_point_px=[640., 360.],
                     initial_rvec_world_to_camera=rvec.tolist(), initial_focal_px=focal, fit_pose=False)],
        observations=observations,
        wide_angle_optics={'active': ['k2_image', 'k3_image', 'p1_image', 'p2_image', 'cx', 'cy']},
        raw_sensor_coverage={'enabled': coverage})
    p = WideAngleJointProblem(scene, problem)
    if prism:
        problem['wide_angle_optics']['active'].extend(['s1_image', 's2_image', 's3_image', 's4_image'])
        p = WideAngleJointProblem(scene, problem)
    if aspect:
        problem['wide_angle_optics']['active'].append('log_fy_over_fx')
        p = WideAngleJointProblem(scene, problem)
    else:
        retained = copy.deepcopy(problem)
        retained['frames'][0]['fy_over_fx'] = 1.4
        fixed_aspect = WideAngleJointProblem(scene, retained)
        K = fixed_aspect.camera(fixed_aspect.x0, 'wide').K_px
        assert np.isclose(K[1, 1]/K[0, 0], 1.4)
        assert not any(name.endswith(':log_fy_over_fx') for name in fixed_aspect.parameter_names)
        assert retained['frames'][0]['fy_over_fx'] == 1.4
    if rational:
        # Recover denominator against known numerator/prism/aspect. Freeing
        # numerator and denominator together can be weakly identifiable; that
        # ambiguity should not turn this numerical regression into a gauge test.
        problem['installations'][0]['radial_distortion'].update(fit=False, initial_k1_image=truth['k1_image'])
        problem['wide_angle_optics']['active'] = ['k4_image','k5_image','k6_image']
        problem['wide_angle_optics']['initial'] = {k:v for k,v in truth.items() if k not in ('k1_image','k4_image','k5_image','k6_image')}
        p = WideAngleJointProblem(scene, problem)
    assert len(p.camera(p.x0, 'wide').distortion_opencv) == (12 if prism else 5)
    if coverage and not rational:
        assert not p.raw_sensor_domain_diagnostics(p.x0)['wide']['all_sampled_pixels_in_central_domain']
    report = p.solve(max_nfev=300 if prism else 150 if coverage else 80, loss='linear', xtol=1e-12, ftol=1e-12, gtol=1e-12)
    estimates = p.optics_report(p.result.x)['installations']['camera']['values']
    for name, value in truth.items():
        np.testing.assert_allclose(estimates[name], value, atol=1e-7)
    assert np.max(np.abs(p.residual(p.result.x))) < 1e-8
    assert report['after']['fit_geometry_valid']
    expected_count = 3+16*coverage if rational else 7+16*coverage+4*prism+aspect
    assert len(report['parameter_names']) == len(report['parameter_values']) == expected_count
    assert len(p.observations) == len(observations)
    if coverage:
        assert p.raw_sensor_domain_diagnostics(p.result.x)['wide']['all_sampled_pixels_in_central_domain']
        domain_report = p.optics_report(p.result.x)['raw_sensor_coverage_constraints']['wide']
        assert max(domain_report['reprojection_error_px']) < 1e-8
        assert min(domain_report['minimum_eigenvalue_along_each_axis_to_ray_path']) > 0.
        assert min(domain_report['minimum_denominator_along_each_axis_to_ray_path']) > 0.
        assert 'not observed rink landmarks' in domain_report['provenance']
    json.dumps(report)
