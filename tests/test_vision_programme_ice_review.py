import cv2
import numpy as np
import pytest

from rinkspace.vision_programme_ice_review import (
    IceProgrammeSourceReader, _actual_profile_candidates, _batched_actual_profiles,
    ice_curve_inventory)
from rinkspace.vision_source_shapes import _MaterialProfileFrame, read_source_chromatic_stroke_centers


def binding(rgb, pts=100):
    return dict(source_sha256='a'*64, source_pts=pts, time_base='1/90000',
                shot_id='synthetic', native_hw=list(rgb.shape[:2]))


def test_real_two_sides_determine_center_instead_of_navigation_or_darkest_pixel():
    rgb = np.full((180, 220, 3), 220, np.uint8)
    rgb[62:83] = [115, 115, 170]
    rgb[66:71] = [90, 90, 210]  # asymmetric interior paint pigment
    rgb = cv2.GaussianBlur(rgb, (0, 0), 1.)
    frame = _MaterialProfileFrame(rgb, binding(rgb))
    point = np.array([[100., 88.]])
    normal = np.array([[0., 1.]])
    interval = np.array([[-60., 60.]])
    offsets, profile, smoothed, valid = _batched_actual_profiles(frame, point, normal, interval, np.array(['blue']))
    candidates = _actual_profile_candidates(point[0], normal[0], offsets[0, valid[0]],
        profile[0, valid[0]], smoothed[0, valid[0]], operator='paint_center', signal='blue', transition=None)
    assert len(candidates) == 1
    uv, sides, bad = candidates[0]
    assert not bad and np.allclose(uv, sides.mean(0))
    assert 69 < uv[1] < 74
    assert abs(uv[1]-88) > 10 and abs(uv[1]-68) > 2


def test_two_distinct_observed_bands_remain_two_candidates():
    offsets = np.arange(-60., 61.)
    profile = np.zeros(len(offsets))
    profile[(offsets >= -31) & (offsets <= -17)] = 45
    profile[(offsets >= 18) & (offsets <= 34)] = 60
    smooth = cv2.GaussianBlur(profile[None], (5, 1), 1.).ravel()
    candidates = _actual_profile_candidates(np.array([90., 90.]), np.array([0., 1.]),
        offsets, profile, smooth, operator='paint_center', signal='blue', transition=None)
    assert len(candidates) == 2
    assert np.allclose([c[0][1] for c in candidates], [66., 116.], atol=1)


def test_compound_material_cache_cannot_cross_exposure_or_mutable_RGB():
    rgb = np.full((100, 100, 3), 220, np.uint8)
    frame = _MaterialProfileFrame(rgb, binding(rgb))
    common = dict(search_points_uv=[[50., 50.]], normals=[[0., 1.]], search_intervals_px=[[-30., 30.]],
                  _validated_frame=frame)
    with pytest.raises(ValueError, match='exact immutable'):
        read_source_chromatic_stroke_centers(rgb, binding=binding(rgb), **common)
    with pytest.raises(ValueError, match='source_pts'):
        read_source_chromatic_stroke_centers(frame.rgb, binding=binding(rgb, 101), **common)
    with pytest.raises(ValueError, match='original RGB differ'):
        read_source_chromatic_stroke_centers(frame.rgb, binding=binding(rgb), expected_RGB_sha256='b'*64, **common)


def test_complete_inventory_keeps_all_house_layers_and_unresolved_path():
    center = [0., 0., 0.]
    scene = dict(ice_features=[dict(id='R1', kind='circle', rings=[
        dict(role=role, center_xyz=center, radius=radius)
        for role, radius in [('outer', 6), ('middle', 4), ('inner', 2), ('center', .5)]]),
        dict(id='B1', kind='polyline', segments_xyz=[[[-10., 0., 0.], [10., 0., 0.]]]),
        dict(id='house_connection_F1_R1', kind='polyline', segments_xyz=[[[-10., 0., 0.], [10., 0., 0.]]])])
    roles = dict(curve_records=[dict(curve_id='house_connection_F1_R1:segment0',
                                   semantic_definition='physical_paint_raster_role_unresolved')])
    inventory = ice_curve_inventory(scene, roles)
    assert [r['curve_id'] for r in inventory[:4]] == ['R1:'+r for r in ['outer', 'middle', 'inner', 'center']]
    assert inventory[-1]['operator'] == 'unresolved'
    assert all(not r['physical_identity_confirmed_by_navigation'] for r in inventory)
