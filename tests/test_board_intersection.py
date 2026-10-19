"""Finite full-perimeter ray checks using synthetic geometry, not venue evidence."""
import numpy as np
import pytest

from rinkspace.calibration_companion import Camera, look_at_rvec
from rinkspace.calibration_companion.board_intersection import intersect_board_surface
from rinkspace.calibration_companion.joint_scene import CORNERS, SIDE_CORNERS, STRAIGHT_SIDES


PARAMETERS = {'bounds': [-8., -4., 12., 6.], 'board_height': 2.4, 'corner_radius': 1.5,
              'corner_radii': {'upper_right': 2., 'upper_left': 1.4,
                               'lower_left': 1.2, 'lower_right': 1.8}}
K = np.array([[900., 0., 960.], [0., 900., 540.], [0., 0., 1.]])


def corner_center(corner):
    x0, y0, x1, y1 = PARAMETERS['bounds']
    radius = PARAMETERS['corner_radii'][corner]
    return np.array([x1 - radius if 'right' in corner else x0 + radius,
                     y1 - radius if 'upper' in corner else y0 + radius])


def camera_for(point, center=(2., 1., 1.1), distortion=(-.04, 0., 0., 0., 0.)):
    return Camera(center, look_at_rvec(center, np.asarray(point) + [.3, -.2, .1]), K, distortion)


def project_and_intersect(point, side, allow_corner=None):
    camera = camera_for(point)
    uv = camera.project([point])[0][0]
    return intersect_board_surface(camera, uv, side, PARAMETERS, allow_corner)


@pytest.mark.parametrize('side', STRAIGHT_SIDES)
def test_all_straight_sides_roundtrip_with_offset_origin_and_distortion(side):
    points = {'upper_long_side': [1., 6., 1.3], 'lower_long_side': [1., -4., 1.3],
              'left_end': [-8., .5, 1.3], 'right_end': [12., .5, 1.3]}
    result = project_and_intersect(points[side], side)
    assert result['valid'], result['reasons']
    np.testing.assert_allclose(result['xyz'], points[side], atol=1e-8)
    assert result['surface_segment'] == side
    assert result['inverse_distortion_roundtrip_px'] < 1e-8
    assert result['anchor_projection_not_a_validation_sample']
    assert result['plane_axis'] == (1 if 'long_side' in side else 0)


@pytest.mark.parametrize('corner', CORNERS)
def test_each_direct_corner_roundtrip(corner):
    radius = PARAMETERS['corner_radii'][corner]
    direction = np.array([1. if 'right' in corner else -1., 1. if 'upper' in corner else -1.]) / np.sqrt(2.)
    point = np.r_[corner_center(corner) + radius * direction, 1.2]
    result = project_and_intersect(point, corner)
    assert result['valid'], result['reasons']
    np.testing.assert_allclose(result['xyz'], point, atol=1e-8)
    assert result['surface_segment'] == corner + '_corner'
    assert result['along_parameterization'] == 'corner_fraction'
    assert result['along'] == pytest.approx(.5)


@pytest.mark.parametrize('side,corner', [(side, corner) for side in STRAIGHT_SIDES for corner in SIDE_CORNERS[side]])
def test_adjacent_corner_continuation_and_join(side, corner):
    radius = PARAMETERS['corner_radii'][corner]
    signs = np.array([1. if 'right' in corner else -1., 1. if 'upper' in corner else -1.])
    point = np.r_[corner_center(corner) + radius * signs / np.sqrt(2.), 1.2]
    result = project_and_intersect(point, side, corner)
    assert result['valid'], result['reasons']
    np.testing.assert_allclose(result['xyz'], point, atol=1e-8)
    assert result['surface_segment'] == corner + '_corner'
    axis = 0 if 'long_side' in side else 1
    assert result['along'] == pytest.approx(point[axis])
    assert result['along_parameterization'] == ('world_x' if axis == 0 else 'world_y')
    # Straight-side and cylinder hits at their tangent join are one physical hit.
    normal_axis = 1 - axis
    join = np.r_[corner_center(corner), 1.2]
    join[normal_axis] += radius * signs[normal_axis]
    joined = project_and_intersect(join, side, corner)
    assert joined['valid'], joined['reasons']
    np.testing.assert_allclose(joined['xyz'], join, atol=1e-8)
    assert joined['valid_intersection_count'] == 1
    assert joined['coincident_join_intersections_merged'] >= 1
    direct = project_and_intersect(join, corner)
    assert direct['valid']
    assert direct['along'] == pytest.approx(0., abs=1e-8) or direct['along'] == pytest.approx(1., abs=1e-8)


def test_nearest_positive_corner_intersection_is_selected_when_chord_hits_twice():
    center_xy = corner_center('upper_right')
    # Two intersections in the same quarter: x+y=2.5 relative to a radius-2 arc.
    root = np.sqrt(1.75)
    first = np.r_[center_xy + [(2.5 - root) / 2., (2.5 + root) / 2.], 1.2]
    second = np.r_[center_xy + [(2.5 + root) / 2., (2.5 - root) / 2.], 1.2]
    origin = np.r_[center_xy + [-2., 4.5], 1.2]
    camera = camera_for(second, center=origin)
    result = intersect_board_surface(camera, camera.project([second])[0][0], 'upper_right', PARAMETERS)
    assert result['valid']
    assert result['valid_intersection_count'] == 2
    np.testing.assert_allclose(result['xyz'], first, atol=1e-8)
    assert result['selected_intersection_is_nearest_positive']


@pytest.mark.parametrize('point,reason', [([12., 6., 1.2], 'intersection_outside_frozen_straight_board_segment'),
                                        ([1., 6., 3.], 'intersection_outside_frozen_board_height')])
def test_rejects_infinite_plane_extensions_and_excess_height(point, reason):
    result = project_and_intersect(point, 'upper_long_side')
    assert not result['valid']
    assert reason in result['reasons']


def test_rejects_backwards_and_parallel_rays():
    backwards = Camera([2., 10., 1.], look_at_rvec([2., 10., 1.], [2., 20., 1.]), K, [])
    result = intersect_board_surface(backwards, [960., 540.], 'upper_long_side', PARAMETERS)
    assert not result['valid']
    assert 'board_intersection_behind_anchor_camera' in result['reasons']
    parallel = Camera([2., 1., 1.], look_at_rvec([2., 1., 1.], [12., 1., 1.]), K, [])
    result = intersect_board_surface(parallel, [960., 540.], 'upper_long_side', PARAMETERS)
    assert not result['valid']
    assert 'ray_nearly_parallel_to_board_plane' in result['reasons']


def test_rejects_ambiguous_ray_running_on_cylinder():
    xy = corner_center('upper_right') + np.sqrt(2.)
    camera = Camera(np.r_[xy, 5.], [np.pi, 0., 0.], K, [])
    result = intersect_board_surface(camera, [960., 540.], 'upper_right', PARAMETERS)
    assert not result['valid']
    assert 'ray_coincident_with_board_cylinder' in result['reasons']


def test_rejects_hits_on_wrong_cylinder_quarter_and_above_corner_height():
    camera = Camera([5., 3., 1.2], look_at_rvec([5., 3., 1.2], [12., 3., 1.2]), K, [])
    result = intersect_board_surface(camera, [960., 540.], 'upper_right', PARAMETERS)
    assert not result['valid']
    assert 'intersection_outside_frozen_quarter_circle' in result['reasons']
    high_camera = Camera([5., 5., 3.], look_at_rvec([5., 5., 3.], [12., 5., 3.]), K, [])
    result = intersect_board_surface(high_camera, [960., 540.], 'upper_right', PARAMETERS)
    assert not result['valid']
    assert 'intersection_outside_frozen_board_height' in result['reasons']


def test_surface_identity_is_required_and_corner_must_be_adjacent():
    camera = camera_for([1., 6., 1.])
    assert not intersect_board_surface(camera, [960., 540.], 'unknown', PARAMETERS)['valid']
    result = intersect_board_surface(camera, [960., 540.], 'upper_long_side', PARAMETERS, 'lower_left')
    assert result['reasons'] == ['allow_corner_not_adjacent_to_identified_straight_side']


def test_rejects_wrong_inverse_distortion_and_nonmonotonic_branch(monkeypatch):
    import rinkspace.calibration_companion.board_intersection as module
    monkeypatch.setattr(module.cv2, 'undistortPointsIter', lambda *args: np.array([[[1., 0.]]]))
    camera = camera_for([1., 6., 1.])
    result = intersect_board_surface(camera, [960., 540.], 'upper_long_side', PARAMETERS)
    assert not result['valid']
    assert 'inverse_distortion_roundtrip_exceeds_0.05px' in result['reasons']
    folded = Camera([0., 0., 0.], [0., 0., 0.], K, [-.4, 0., 0., 0., 0.])
    uv = folded.project([[1., 0., 1.]])[0][0]
    result = intersect_board_surface(folded, uv, 'upper_right', PARAMETERS)
    assert not result['valid']
    assert 'nonmonotonic_inverse_radial_branch' in result['reasons']
