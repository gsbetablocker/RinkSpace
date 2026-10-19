import numpy as np
import pytest
import json
from scipy.spatial.transform import Rotation

from rinkspace.multisurface_bundle import ImageEdge, GroundLine, bundle_adjust
from rinkspace.multisurface_camera import Camera, project_camera


def camera(angle=0, focal=900):
    return Camera(Rotation.from_euler('y', angle).as_matrix() @ np.diag([1., -1., -1.]),
                  np.array([0., 0., 10.]), focal, np.array([480., 270.]))


def test_real_edges_correct_partition_jump_without_pose_smoothness():
    points = np.array([[x, y, 0.] for x in np.linspace(-3, 3, 5) for y in (-2, 0, 2)])
    boards = points.copy(); boards[:, 2] = 3
    true = [camera(i * .003, 900 + i * 2) for i in range(7)]
    initial = [camera(i * .003 + (.055 if i >= 3 else 0), 900 + i * 2) for i in range(7)]
    edges = []
    for i in range(1, 7):
        for surface, xyz in [('ice_marking', points), ('board_surface', boards)]:
            a, _ = project_camera(true[i - 1], xyz); b, _ = project_camera(true[i], xyz)
            edges.append(ImageEdge(i - 1, i, a, b, surface))
    a, _ = project_camera(true[0], points); b, _ = project_camera(true[-1], points)
    edges.append(ImageEdge(0, 6, a, b, 'ice_marking', 'direct_actual_image_return_to_original_reference'))
    result, report = bundle_adjust(initial, {0}, edges, max_nfev=100)
    assert report['solver_converged']
    assert report['pose_smoothness_prior_used'] is False
    assert report['board_points_used_as_Z0'] is False
    assert max(e['p95_px'] for e in report['edges']) < .02
    assert np.allclose(result[0].rotation, true[0].rotation)
    assert np.allclose(result[6].focal, true[6].focal, atol=.1)
    json.dumps(report, allow_nan=False)


def test_unconnected_frames_never_filled_from_initial_camera():
    points = np.array([[x, y] for x in (200, 400, 600) for y in (200, 400)])
    result, report = bundle_adjust([camera(), camera(), camera()], {0}, [ImageEdge(0, 1, points, points, 'board_surface')], max_nfev=4)
    assert 2 not in result
    assert report['missing_disconnected_indices'] == [2]


def test_board_rays_reject_camera_translation():
    points = np.array([[200, 200], [400, 200], [600, 200], [600, 400]])
    a, b = camera(), camera(); b.centre[0] = 1
    with pytest.raises(ValueError, match='common conditional optical centre'):
        bundle_adjust([a, b], {0}, [ImageEdge(0, 1, points, points, 'board_surface')])


def test_ground_line_is_an_explicit_bounded_z0_residual():
    points = np.array([[200, 200], [400, 200], [600, 200], [600, 400]])
    endpoints = np.array([[-3., 0., 0.], [3., 0., 0.]])
    sample, _ = project_camera(camera(), np.array([[-2., 0., 0.], [0., 0., 0.], [2., 0., 0.]]))
    line = GroundLine(1, endpoints, sample, 'ice_board_contact', 'test_ground_boundary')
    _, report = bundle_adjust([camera(), camera()], {0}, [ImageEdge(0, 1, points, points, 'board_surface')], ground_lines=[line])
    assert report['ground_line_point_count'] == 3
    assert report['ground_line_median_sample_px'] < 1e-7
    assert report['used_ground_lines'][0]['surface'] == 'ice_board_contact'
    assert report['invalid_point_projection_count'] == 0
    json.dumps(report, allow_nan=False)


def test_unsupported_lens_cannot_silently_become_zero_distortion():
    points = np.array([[200, 200], [400, 200], [600, 200], [600, 400]])
    a, b = camera(), camera(); b.radial_ray_k1 = .01
    with pytest.raises(ValueError, match='zero radial distortion only'):
        bundle_adjust([a, b], {0}, [ImageEdge(0, 1, points, points, 'board_surface')])


def test_actual_gap_endpoints_connect_later_observations_without_filling_middle():
    points = np.array([[200, 200], [400, 200], [600, 200], [600, 400]])
    edges = [ImageEdge(0, 1, points, points, 'board_surface'),
             ImageEdge(1, 3, points, points, 'board_surface', 'short_gap_actual_image_pair')]
    result, report = bundle_adjust([camera()] * 4, {0}, edges)
    assert set(result) == {0, 1, 3}
    assert report['missing_disconnected_indices'] == [2]
    assert report['edges'][1]['kind'] == 'short_gap_actual_image_pair'
