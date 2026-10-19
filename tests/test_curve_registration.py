"""Synthetic mask/refinement contracts, not Ankara accuracy evidence."""
from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np
import pytest

from rinkspace.curve_registration import CurveRefinementConfig, refine_model_projection
from rinkspace.geometry import project


def scene():
    image = np.full((240, 320), .82, np.float64)
    cv2.ellipse(image, (105, 119), (70, 47), 12, 0, 340, .36, 3)
    cv2.ellipse(image, (105, 119), (60, 40), 12, 0, 300, .48, 2)
    cv2.line(image, (25, 202), (285, 195), .47, 3)
    cv2.line(image, (246, 34), (239, 210), .43, 3)
    cv2.putText(image, 'ICE', (142, 72), cv2.FONT_HERSHEY_SIMPLEX, 1.2, .31, 3, cv2.LINE_AA)
    cv2.rectangle(image, (177, 142), (217, 171), .38, 3)
    return image


def case():
    reference = scene()
    reference_mask = np.zeros(reference.shape, np.uint8)
    reference_mask[12:-12, 12:-12] = 255
    truth = np.array([[1.02, .015, 5.], [-.012, 1.03, 3.], [2e-5, -3e-5, 1.]])
    current = cv2.warpPerspective(reference, truth, (340, 250), borderValue=.82)
    current_mask = cv2.warpPerspective(reference_mask, truth, (340, 250), flags=cv2.INTER_NEAREST)
    yy, xx = np.mgrid[:250, :340]
    current = np.clip(current + .06 + .025 * xx / 340 + .025 * yy / 250, 0., 1.)
    initial = np.array([[1., 0, -1.8], [0, 1., 1.2], [0, 0, 1.]]) @ truth
    return reference, current, reference_mask, current_mask, truth, initial


def test_refines_near_initial_projective_map_with_lighting_change():
    ref, cur, rm, cm, truth, initial = case()
    result = refine_model_projection(ref, cur, rm, cm, initial)
    assert result['status'] == 'candidate', result['reason']
    probes = np.array([[40, 50], [270, 50], [265, 195], [45, 190]])
    error = np.linalg.norm(project(result['H_current_from_reference'], probes) - project(truth, probes), axis=1)
    assert max(error) < .7
    assert result['final_curve_residual']['mean_px'] < result['comparison_before_curve_residual']['mean_px']
    assert result['matrix_direction'] == 'H_current_from_reference'
    assert not result['independent_validation'] and not result['global_initialization_performed']
    # Affine correction preserves the perspective terms of the initial H.
    np.testing.assert_allclose(np.asarray(result['H_current_from_reference'])[2], initial[2], atol=1e-10)


def test_values_outside_both_masks_cannot_change_any_evidence():
    ref, cur, rm, cm, _, initial = case()
    rm[154:194, 182:227] = 0
    cm[146:178, 42:84] = 0
    baseline = refine_model_projection(ref, cur, rm, cm, initial)
    assert baseline['status'] == 'candidate', baseline['reason']
    changed_ref, changed_cur = ref.copy(), cur.copy()
    changed_ref[rm == 0] = np.nan
    changed_cur[cm == 0] = -9837.
    changed = refine_model_projection(changed_ref, changed_cur, rm, cm, initial)
    assert baseline == changed


def test_white_ice_and_smooth_illumination_cannot_be_a_successful_match():
    yy, xx = np.mgrid[:240, :320]
    ref = .76 + .07 * xx / 320 + .03 * yy / 240
    cur = np.full_like(ref, .9)
    mask = np.full(ref.shape, 255, np.uint8)
    result = refine_model_projection(ref, cur, mask, mask, np.eye(3))
    assert result['status'] == 'unavailable'
    assert result['reason'] == 'insufficient_stable_print_texture'
    assert result['H_current_from_reference'] is None


def test_parallel_line_aperture_problem_is_rejected():
    ref = np.full((240, 320), .8, np.float64)
    for y in range(30, 220, 30):
        cv2.line(ref, (0, y), (319, y), .3, 3)
    mask = np.full(ref.shape, 255, np.uint8)
    result = refine_model_projection(ref, ref, mask, mask, np.eye(3))
    assert result['status'] == 'unavailable'
    assert result['reason'] == 'curve_orientation_aperture_problem'


def test_disjoint_masks_reject_without_returning_initial_h():
    ref = scene()
    left = np.zeros(ref.shape, np.uint8)
    right = left.copy()
    left[:, :120] = 255
    right[:, 200:] = 255
    result = refine_model_projection(ref, ref, left, right, np.eye(3))
    assert result['status'] == 'unavailable'
    assert result['reason'] == 'insufficient_bidirectional_mask_overlap'
    assert result['H_current_from_reference'] is None
    assert result['photometric_audit'] is None


def test_projective_horizon_and_reflection_are_rejected():
    ref = scene()
    mask = np.full(ref.shape, 255, np.uint8)
    for h in [np.array([[1., 0, 0], [0, 1., 0], [-.01, 0, 1.]]),
              np.array([[-1., 0, 319], [0, 1., 0], [0, 0, 1.]])]:
        result = refine_model_projection(ref, ref, mask, mask, h)
        assert result['status'] == 'unavailable'
        assert result['reason'] == 'initial_homography_horizon_reflection_or_empty_support'
        assert result['H_current_from_reference'] is None


def test_too_large_delta_is_rejected_even_if_fit_looks_good():
    ref, cur, rm, cm, _, initial = case()
    result = refine_model_projection(ref, cur, rm, cm, initial,
                                    config=replace(CurveRefinementConfig(), maximum_delta_px=.1))
    assert result['photometric_audit']['status'] == 'candidate'
    assert result['status'] == 'unavailable'
    assert result['reason'] == 'model_projection_correction_out_of_bounds'
    assert result['H_current_from_reference'] is None


def test_no_pattern_is_not_global_relocalization():
    ref, cur, rm, cm, _, initial = case()
    missing = np.full(cur.shape, .84, np.float64)
    result = refine_model_projection(ref, missing, rm, cm, initial)
    assert result['status'] == 'unavailable'
    assert not result['global_initialization_performed']
    assert not result['physical_landmark_identity_verified']


def test_refinement_handles_different_image_sizes_and_invalid_inputs():
    ref, cur, rm, cm, _, initial = case()
    assert ref.shape != cur.shape
    result = refine_model_projection(ref, cur, rm, cm[:-1], initial)
    assert result['reason'] == 'invalid_input'
    result = refine_model_projection(ref, cur, rm, cm, np.zeros((3, 3)))
    assert result['reason'] == 'invalid_initial_homography'


def test_successful_identity_is_still_candidate_and_exposes_residual_units():
    ref = scene()
    mask = np.full(ref.shape, 255, np.uint8)
    result = refine_model_projection(ref, ref, mask, mask, np.eye(3))
    assert result['status'] == 'candidate', result['reason']
    assert result['final_curve_residual']['p90_px'] < 1.
    assert result['photometric_residual_semantics'].endswith('not_geometric_pixels')
    assert result['final_curve_residual']['residual_coordinate_space'] == 'current_input_image_px'
    assert result['physical_landmark_identity_verified'] is False


@pytest.mark.parametrize('motion_model', ['affine', 'similarity'])
def test_overlap_roi_delta_is_conjugated_back_to_far_from_origin_full_pixels(motion_model):
    ref = np.full((850, 1300), .82, np.float64)
    ref[340:580, 610:930] = scene()
    rm = np.zeros(ref.shape, np.uint8)
    rm[352:568, 622:918] = 255
    truth = np.array([[1., .007, -1.2], [-.005, 1., 2.5], [0, 0, 1.]])
    cur = cv2.warpPerspective(ref, truth, (1300, 850), borderValue=.82)
    cm = cv2.warpPerspective(rm, truth, (1300, 850), flags=cv2.INTER_NEAREST)
    initial = np.array([[1., 0, -1.4], [0, 1., 1.], [0, 0, 1.]]) @ truth
    result = refine_model_projection(ref, cur, rm, cm, initial,
                                    config=replace(CurveRefinementConfig(), motion_model=motion_model))
    assert result['status'] == 'candidate', result['reason']
    probes = np.array([[655, 390], [875, 390], [875, 520], [655, 520]])
    error = np.linalg.norm(project(result['H_current_from_reference'], probes) - project(truth, probes), axis=1)
    assert max(error) < .7
    x0, y0 = result['photometric_audit']['roi_origin_full_image_xy']
    assert x0 > 500 and y0 > 250
    assert result['photometric_audit']['support_scope'].startswith('local_observed_overlap')
    assert result['initial_affine_normal_sensitivity']['current']['rank'] == (4 if motion_model == 'similarity' else 6)


def test_similarity_delta_keeps_projective_seed_and_has_only_four_motion_dof():
    ref, cur, rm, cm, truth, _ = case()
    delta = np.array([[1.003, -.004, -1.8], [.004, 1.003, .7], [0, 0, 1.]])
    initial = delta @ truth
    config = replace(CurveRefinementConfig(), motion_model='similarity')
    result = refine_model_projection(ref, cur, rm, cm, initial, config=config)
    assert result['status'] == 'candidate', result['reason']
    probes = np.array([[40, 50], [270, 50], [265, 195], [45, 190]])
    assert max(np.linalg.norm(project(result['H_current_from_reference'], probes)
                              - project(truth, probes), axis=1)) < .7
    correction = np.asarray(result['delta_current_from_initial_projection'])
    np.testing.assert_allclose(correction[0, 0], correction[1, 1], atol=1e-12)
    np.testing.assert_allclose(correction[0, 1], -correction[1, 0], atol=1e-12)
    np.testing.assert_array_equal(correction[2], [0, 0, 1])
    np.testing.assert_allclose(np.asarray(result['H_current_from_reference'])[2], initial[2], atol=1e-12)
    for phase in ('initial_affine_normal_sensitivity', 'final_affine_normal_sensitivity'):
        for side in ('reference', 'current'):
            assert result[phase][side]['rank'] == 4
            assert result[phase][side]['geometric_parameter_count'] == 4
    assert not result['independent_validation']
    assert 'not_a_physical_camera_or_metric_model' in result['model_assumption']


def test_similarity_cannot_fit_a_shear_that_affine_can_explain():
    ref = scene()
    rm = np.zeros(ref.shape, np.uint8)
    rm[12:-12, 12:-12] = 255
    shear = np.array([[1., .06, -7.2], [0, 1., 0], [0, 0, 1.]])
    cur = cv2.warpPerspective(ref, shear, (320, 240), borderValue=.82)
    cm = cv2.warpPerspective(rm, shear, (320, 240), flags=cv2.INTER_NEAREST)
    affine = refine_model_projection(ref, cur, rm, cm, np.eye(3))
    result = refine_model_projection(ref, cur, rm, cm, np.eye(3),
                                     config=replace(CurveRefinementConfig(), motion_model='similarity'))
    assert affine['status'] == 'candidate', affine['reason']
    assert result['status'] == 'unavailable'
    assert result['reason'] == 'photometric_refinement_refused:insufficient_photometric_correlation'
    assert result['H_current_from_reference'] is None
    assert result['thresholds']['maximum_curve_p90_px'] == affine['thresholds']['maximum_curve_p90_px']
    for row in result['photometric_audit']['iterations']:
        h = np.asarray(row['H_reference_from_current'])
        np.testing.assert_allclose(h[0, 1], -h[1, 0], atol=1e-12)
        np.testing.assert_allclose(h[0, 0], h[1, 1], atol=1e-12)


def test_similarity_cannot_read_masked_people_or_recover_translation_along_a_straight_line():
    ref, cur, rm, cm, _, initial = case()
    rm[154:194, 182:227] = 0
    cm[146:178, 42:84] = 0
    config = replace(CurveRefinementConfig(), motion_model='similarity')
    baseline = refine_model_projection(ref, cur, rm, cm, initial, config=config)
    assert baseline['status'] == 'candidate', baseline['reason']
    ref[rm == 0] = np.nan
    cur[cm == 0] = -1.e15
    assert refine_model_projection(ref, cur, rm, cm, initial, config=config) == baseline
    lines = np.full((240, 320), .8, np.float64)
    for y in range(30, 220, 30):
        cv2.line(lines, (0, y), (319, y), .3, 3)
    mask = np.full(lines.shape, 255, np.uint8)
    result = refine_model_projection(lines, lines, mask, mask, np.eye(3), config=config)
    assert result['status'] == 'unavailable'
    assert result['reason'] == 'curve_orientation_aperture_problem'
    assert result['initial_affine_normal_sensitivity']['current']['rank'] < 4
    assert result['photometric_audit'] is None
    assert result['H_current_from_reference'] is None
