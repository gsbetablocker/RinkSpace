"""Independent two-dimensional source readout of a discrete native alias."""
from __future__ import annotations

import cv2
import numpy as np

from .vision_programme_board_review import _sample, _source_orientation_preserved
from .vision_board_material_point_profiles import MATERIAL_ROLES, read_printed_material_corners


def _literal(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _literal(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_literal(v) for v in value]
    return value


def _context(image, exclusion, uv):
    offsets = np.arange(-6., 6.001)
    dx, dy = np.meshgrid(offsets, offsets)
    xy = uv+np.stack([dx, dy], axis=-1)
    pixels = np.rint(xy).astype(int)
    if ((pixels < 0).any() or (pixels >= [1920, 1080]).any()
            or exclusion[pixels[..., 1], pixels[..., 0]].any()):
        return None
    values = _sample(image, xy).astype(float)
    values -= values.mean(axis=(0, 1))
    strength = np.linalg.norm(values)
    return (values.ravel()/strength, strength) if strength > 20. else None


def _navigation(canonical, current, canonical_mask, current_mask, guides):
    """Register actual named print texture; transported corners remain NAV."""
    low = np.maximum(0, np.floor(guides.min(0)-12)).astype(int)
    high = np.minimum([1920, 1080], np.ceil(guides.max(0)+13)).astype(int)
    source_roi = np.zeros((1080, 1920), np.uint8)
    source_roi[low[1]:high[1], low[0]:high[0]] = 255
    target_roi = np.zeros_like(source_roi)
    target_low = np.maximum(0, low-96)
    target_high = np.minimum([1920, 1080], high+96)
    target_roi[target_low[1]:target_high[1], target_low[0]:target_high[0]] = 255
    source_roi[canonical_mask] = 0
    target_roi[current_mask] = 0
    detector = cv2.SIFT_create(nfeatures=800, contrastThreshold=.014)
    first, a = detector.detectAndCompute(cv2.cvtColor(canonical, cv2.COLOR_BGR2GRAY), source_roi)
    second, b = detector.detectAndCompute(cv2.cvtColor(current, cv2.COLOR_BGR2GRAY), target_roi)
    if a is None or b is None or min(len(first), len(second)) < 8:
        return None
    matches = [m for m, n in cv2.BFMatcher().knnMatch(a, b, k=2) if m.distance < .76*n.distance]
    if len(matches) < 8:
        return None
    source = np.asarray([first[m.queryIdx].pt for m in matches], np.float32)
    target = np.asarray([second[m.trainIdx].pt for m in matches], np.float32)
    matrix, inliers = cv2.findHomography(source, target, cv2.RANSAC, 2.)
    if matrix is None or inliers is None:
        return None
    keep = inliers.ravel().astype(bool)
    if (keep.sum() < 8 or keep.mean() < .55
            or len(np.unique(source[keep], axis=0)) < 8
            or np.ptp(source[keep], axis=0).min() < 8.
            or not _source_orientation_preserved(matrix, source[keep])):
        return None
    error = np.linalg.norm(cv2.perspectiveTransform(source[None], matrix)[0]-target, axis=1)
    if error[keep].max() > 1.5:
        return None
    return cv2.perspectiveTransform(guides.astype(np.float32)[None], matrix)[0], dict(
        kind='same_actual_exposure_named_print_actual_RGB_navigation',
        canonical_source_feature_uv=source[keep], native_source_feature_uv=target[keep],
        actual_source_registration_error_px=error[keep],
        original_person_union_exclusion_retained=True,
        registration_and_transported_guides_are_not_pixel_truth=True,
        model_projection_or_fit_residual_used_for_navigation=False)


def review_discrete_board_material_point_alias(*, canonical_raw_bgr, row, raw_bgr,
        canonical_exclusion_mask, exclusion_mask, source_binding, permitted):
    """Use the original three-role guides as NAV, never as observed vertices."""
    role = row.get('material_roles', ())
    same = (int(row.get('actual_pts', -1)) == int(source_binding['actual_pts'])
        and row.get('shot_id') == source_binding.get('shot_id')
        and int(row.get('native_pts', -1)) < int(source_binding.get('native_pts', source_binding.get('source_pts'))))
    if (not permitted or not same or row.get('point_correspondence') is not True
            or len(role) != 1 or role[0] not in MATERIAL_ROLES
            or row.get('source_navigation_material_guides_are_not_pixel_truth') is not True
            or tuple(row.get('source_navigation_material_guide_roles', ())) != MATERIAL_ROLES
            or canonical_exclusion_mask is None):
        return dict(status='pending_discrete_material_point_navigation_or_identity', uv=[], normal=[])
    guides = np.asarray(row.get('source_navigation_material_guides', ()), float)
    canonical_uv = np.asarray(row.get('uv', ()), float)
    if guides.shape != (3, 2) or canonical_uv.shape != (1, 2):
        return dict(status='pending_discrete_material_point_navigation_or_identity', uv=[], normal=[])
    canonical_mask = np.asarray(canonical_exclusion_mask, bool)
    index = MATERIAL_ROLES.index(role[0])
    canonical_context = _context(canonical_raw_bgr, canonical_mask, canonical_uv[0])
    if canonical_context is None:
        return dict(status='pending_discrete_material_corner_person_context', uv=[], normal=[])
    # The verified clock says these are two encodings of the same exposure.
    # Its named guides can navigate a fresh native read without a texture-count
    # gate. The current two borders and material context still own acceptance.
    native_guides = guides
    navigation_evidence = dict(kind='same_verified_actual_exposure_named_material_guides_navigation',
        canonical_source_native_pts=int(row['native_pts']),
        current_source_native_pts=int(source_binding.get('native_pts', source_binding.get('source_pts'))),
        actual_pts=int(source_binding['actual_pts']),
        registration_and_transported_guides_are_not_pixel_truth=True,
        original_person_union_exclusion_retained=True,
        model_projection_or_fit_residual_used_for_navigation=False)
    evidence, correlation, pending_status = None, None, 'pending_missing_or_competing_actual_native_material_corner'
    for attempt in range(2):
        found = read_printed_material_corners(raw_bgr, exclusion_mask, native_guides)
        candidate = found.get(index)
        if candidate is not None:
            native_context = _context(raw_bgr, exclusion_mask, candidate['uv'])
            if native_context is None:
                pending_status = 'pending_discrete_material_corner_person_context'
            else:
                correlation = float(canonical_context[0] @ native_context[0])
                pending_status = 'pending_discrete_material_corner_actual_RGB_context_change'
                if correlation >= .90:
                    evidence = candidate
                    break
        if attempt == 0:
            navigation = _navigation(canonical_raw_bgr, raw_bgr, canonical_mask, exclusion_mask, guides)
            if navigation is None:
                break
            native_guides, navigation_evidence = navigation
    if evidence is None:
        return dict(status=pending_status, uv=[], normal=[])
    native_uv = evidence['uv']
    return _literal(dict(status='actual_native_alias_visible_material_point_read',
        uv=[native_uv.tolist()], normal=[], canonical_point_indices=[0],
        actual_source_RGB_measured=True, canonical_UV_used_as_output=False,
        canonical_UV_not_output=True, new_FIT_observations_created=False,
        material_roles=list(role), point_correspondence=True,
        source_neighbour_adjacency_confirmed=False,
        observation_semantics='board_print_feature', curve=dict(row['curve']),
        model_target=row.get('model_target'),
        source_material_point_readout=evidence,
        actual_native_source_RGB_registration_navigation=navigation_evidence,
        actual_2d_readout_bounds_xy_px=evidence['actual_2d_readout_bounds_xy_px'][None],
        source_sigma_xy_px=None,
        measurement_method=evidence['measurement_method'],
        normal_measurement_origin='not_applicable_discrete_material_point',
        actual_original_source_material_context_correlation=correlation,
        full_current_person_union_exclusion_retained=True,
        source_board_instance_identity_inherited_from_canonical_review=True,
        identity_origin='same_known_physical_material_corner_current_original_native_RGB_readout',
        missing_or_competing_canonical_point_indices=[]))
