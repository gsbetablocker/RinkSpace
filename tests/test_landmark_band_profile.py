import cv2
import numpy as np
import pytest

from rinkspace.landmark_band_profile import (evaluate_outer_boundary_cue,
    measure_band_profiles, point_array_sha256)


def _observation(points, colour='gray'):
    return {'observation_id': 'synthetic:0:observed_000', 'points_source_px': points.tolist(),
            'colour_class': colour, 'association_status': 'detected_unidentified'}


def _ring():
    im = np.full((400, 400, 3), 220, np.uint8)
    cv2.circle(im, (200, 200), 95, (125, 125, 125), -1)
    cv2.circle(im, (200, 200), 55, (220, 220, 220), -1)
    theta = np.linspace(0, 2*np.pi, 36, endpoint=False)
    normal = np.c_[np.cos(theta), np.sin(theta)]
    return im, normal


def test_actual_ring_outer_and_inner_brightness_sides_differ():
    image, outward = _ring()
    outer, inner = [200, 200] + 95*outward, [200, 200] + 55*outward
    profiles = measure_band_profiles(image, [_observation(outer), _observation(inner)])
    a, b = [evaluate_outer_boundary_cue(p, outward) for p in profiles]
    assert a['status'] == 'brightness_supports_outer_side_of_this_model_hypothesis'
    assert b['status'] == 'brightness_supports_inner_side_of_this_model_hypothesis'
    assert a['median_outward_alignment'] > .95
    assert b['median_outward_alignment'] < -.95
    assert all(not p['model_projection_or_id_used_in_measurement'] for p in profiles)


def test_gradient_pixel_axes_and_gain_are_explicit():
    gray = np.tile(np.arange(200, dtype=np.uint8)[:, None], (1, 200))
    image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    points = np.array([[60., 70.], [80., 90.], [100., 110.], [120., 130.], [140., 150.]])
    profile = measure_band_profiles(image, [_observation(points)])[0]
    assert np.allclose(profile['gradients_source_xy'], [[0., 1.]]*5, atol=.005)
    assert profile['gradient_reliable'] == [True]*5
    assert evaluate_outer_boundary_cue(profile, [[0, -1]]*5)['median_outward_alignment'] < -.999


def test_colour_flatness_and_foreground_are_unknown_not_false_outer_evidence():
    image, outward = _ring(); points = [200, 200] + 95*outward
    red = measure_band_profiles(image, [_observation(points, 'red')])[0]
    assert red['reliable_point_count'] == 0
    assert evaluate_outer_boundary_cue(red, outward)['status'] == 'unknown'
    masked = _observation(points); masked['association_status'] = 'masked_out'
    p = measure_band_profiles(image, [masked])[0]
    assert p['reliable_point_count'] == 0
    flat = measure_band_profiles(np.full_like(image, 200), [_observation(points)])[0]
    assert flat['reliable_point_count'] == 0
    assert evaluate_outer_boundary_cue(flat, outward)['status'] == 'unknown'


def test_profiles_preserve_original_array_binding_and_reject_wrong_normals():
    image, normal = _ring(); points = [200, 200] + 95*normal
    original = points.copy()
    profile = measure_band_profiles(image, [_observation(points)])[0]
    assert np.array_equal(points, original)
    assert profile['point_array_sha256'] == point_array_sha256(original)
    assert profile['point_array_sha256'] != point_array_sha256(original[::-1])
    with pytest.raises(ValueError):
        evaluate_outer_boundary_cue(profile, normal[:-1])
    with pytest.raises(ValueError):
        evaluate_outer_boundary_cue(profile, np.full_like(normal, np.nan))
