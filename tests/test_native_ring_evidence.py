"""Contracts on the actual15557 native frame, not generated circle drawings.

The source ZIP carries the frozen image and SAM PNG referenced by this case.
A minimal checkout without media evidence reports these tests as skipped.
These are detector-boundary regressions, not independent whole-rink accuracy.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from rinkspace.ring_evidence import native_ring_evidence, qualify_profile_foreground
from rinkspace.landmark_band_profile import measure_band_profiles, BandProfileConfig

ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / 'outputs/event_reuse_v11/failures/men15557'


@pytest.fixture(scope='module')
def source_case():
    image_path = CASE / 'source_15557_native.png'
    mask_path = CASE / 'men600_sam2_center_people_masks/238955520.png'
    if not image_path.exists() or not mask_path.exists():
        pytest.skip('Actual v11 source-frame/SAM evidence is absent; no synthetic substitute')
    intake = json.loads((CASE / 'intake.json').read_text(encoding='utf8'))
    expected = intake['focal_frame']['native_image']['sha256']
    assert hashlib.sha256(image_path.read_bytes()).hexdigest() == expected
    metadata = json.loads((CASE / 'men600_sam2_center_people.json').read_text(encoding='utf8'))
    row = next(r for r in metadata['frames'] if r['pts'] == 238955520)
    assert hashlib.sha256(mask_path.read_bytes()).hexdigest() == row['person_mask_sha256']
    image = cv2.imread(str(image_path))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    assert image.shape == (1080, 1920, 3) and mask.shape == (540, 960)
    frame = intake['focal_frame']['saved_run_row']
    model = {'structure': {'primitives': [
        {'id': 'F0', 'kind': 'circle', 'center': [0., 0.], 'radius': 5.},
        {'id': 'R2', 'kind': 'circle', 'center': [100., 0.], 'radius': 5.},
        {'id': 'red_center', 'kind': 'axis_line', 'axis': 'x', 'value': 0.},
    ]}}
    result = native_ring_evidence(image, frame, model, person_mask_sample=mask, maximum_candidates=120)
    return image, frame, model, mask, result


def test_visible_real_arc_survives_wrong_coarse_ice_mask(source_case):
    _, _, _, _, (observations, profiles, diagnostic) = source_case
    # This region is used only to assert what the full-image detector returned.
    # No region, predicted grid or handwritten circle is passed into detection.
    observed_ring = []
    for observation in observations:
        points = np.asarray(observation['points_source_px'])
        in_actual_ring = ((points[:, 0] > 1000) & (points[:, 0] < 1475)
                          & (points[:, 1] > 495) & (points[:, 1] < 580))
        if (observation['colour_class'] == 'gray' and observation['kind'] == 'arc'
                and in_actual_ring.sum() >= 30 and np.ptp(points[in_actual_ring, 0]) > 200):
            observed_ring.append(observation)
    assert len(observed_ring) >= 2
    assert any(o['original_ice_mask_support_fraction'] == 0 for o in observed_ring)
    assert all(o['association_status'] == 'detected_unidentified' for o in observed_ring)
    assert all(o['fit_ellipse_used_to_generate_observations'] is False for o in observed_ring)
    assert diagnostic['detection_search_uses_camera'] is False
    assert diagnostic['detection_search_uses_reported_ring_roi'] is False
    assert len(profiles) == len(observations)


def test_model_changes_only_identity_candidates_not_measured_pixels(source_case):
    image, frame, model, mask, (original, original_profiles, _) = source_case
    changed_model = deepcopy(model)
    changed_model['structure']['primitives'] = [
        {'id': 'F_changed', 'kind': 'circle', 'center': [-9999., 4444.], 'radius': 900.},
        {'id': 'R_changed', 'kind': 'circle', 'center': [11., 6666.], 'radius': .1},
    ]
    changed, changed_profiles, _ = native_ring_evidence(
        image, frame, changed_model, person_mask_sample=mask, maximum_candidates=120)
    assert len(original) == len(changed) > 0
    assert any(a['candidate_model_ids'] != b['candidate_model_ids'] for a, b in zip(original, changed))
    for a, b in zip(original, changed):
        assert a['observation_id'] == b['observation_id']
        np.testing.assert_array_equal(a['points_source_px'], b['points_source_px'])
        assert a['visible_segment_ranges'] == b['visible_segment_ranges']
        assert a['source_shape'] == b['source_shape']
        assert a['association_status'] == b['association_status']
        assert b['selected_model_id'] is None and b['used_by_solver'] is False
    assert [p['point_array_sha256'] for p in original_profiles] == [p['point_array_sha256'] for p in changed_profiles]


def test_explicit_person_mask_stays_distinct_from_unidentified_edges(source_case):
    image, frame, model, mask, (actual, _, diagnostic) = source_case
    assert {o['association_status'] for o in actual} == {'masked_out', 'detected_unidentified'}
    assert diagnostic['counts']['masked_out'] > 0 and diagnostic['counts']['identity_unresolved'] > 0
    assert all(o['rejection_stage'] == 'foreground_mask' for o in actual if o['association_status'] == 'masked_out')
    # Deliberately occlude every pixel through the same SAM-mask interface.
    # This controlled mask is a test input, not an asserted SAM prediction.
    occluded, _, checks = native_ring_evidence(
        image, frame, model, person_mask_sample=np.full_like(mask, 255), maximum_candidates=120)
    assert len(occluded) > 0  # audit the observed-but-masked edge instead of deleting it
    assert all(o['association_status'] == 'masked_out' for o in occluded)
    assert all(o['selected_model_id'] is None and o['used_by_solver'] is False for o in occluded)
    assert checks['counts']['identity_unresolved'] == 0
    assert checks['counts']['solver_used'] == 0


@pytest.mark.parametrize('sigma,person_pixel,expected_route', [
    (1., (82, 70), 'sample_footprint_touches_foreground'),
    (8., (70, 101), 'gradient_support_touches_foreground'),
])
def test_profile_foreground_checks_samples_and_actual_gaussian_support(sigma, person_pixel, expected_route):
    # A ramp has a stable, reliable gradient. The second person pixel lies
    # perpendicular to the profile, outside a24px guard but inside the actual
    # sigma8 float Gaussian support; endpoint-only checking would miss it.
    gray = np.tile(np.arange(220, dtype=np.uint8), (160, 1))
    image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    observation = {'observation_id': 'source_edge', 'points_source_px': [[70., 70.], [150., 70.]],
                   'source_size': [220, 160], 'colour_class': 'gray', 'association_status': 'detected_unidentified'}
    config = BandProfileConfig(gradient_sigmas_source_px=(sigma,))
    profiles = measure_band_profiles(image, [observation], config)
    before = deepcopy(profiles)
    assert profiles[0]['gradient_reliable'] == [True, True]
    foreground = np.zeros((160, 220), np.uint8)
    foreground[person_pixel[1], person_pixel[0]] = 255
    guarded = qualify_profile_foreground(profiles, [observation], foreground)
    assert guarded[0]['gradient_reliable'] == [False, True]
    assert guarded[0]['unknown_reasons'][0] == ['person_near_profile']
    assert guarded[0]['foreground_profile_qualification'][expected_route] == [True, False]
    assert guarded[0]['reliable_point_count'] == 1
    assert profiles == before  # caller can archive the original unchanged
    for key in ['point_array_sha256', 'profile_luminance_source', 'gradients_source_xy',
                'gradient_unit_source_xy', 'contrast_source_levels', 'median_gradient_source_xy']:
        assert guarded[0][key] == before[0][key]
    assert qualify_profile_foreground(guarded, [observation], foreground) == guarded
    broken = deepcopy(observation); broken['points_source_px'][0][0] += 1
    with pytest.raises(ValueError, match='source-point hash'):
        qualify_profile_foreground(profiles, [broken], foreground)


def test_profile_guard_preserves_other_rejection_reasons():
    image = np.zeros((120, 120, 3), np.uint8)
    observation = {'observation_id': 'weak', 'points_source_px': [[60., 60.]],
                   'colour_class': 'gray', 'association_status': 'detected_unidentified'}
    profiles = measure_band_profiles(image, [observation])
    reasons = profiles[0]['unknown_reasons'][0]
    assert 'weak_luminance_gradient' in reasons
    guarded = qualify_profile_foreground(profiles, [observation], np.full((120, 120), 255, np.uint8))
    assert guarded[0]['gradient_reliable'] == [False]
    assert guarded[0]['unknown_reasons'][0] == reasons + ['person_near_profile']
