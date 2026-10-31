"""Whole-print translation search for the existing photographic registration gate.

This module supplies navigation only. It never reads or creates observations and
never interprets a coarse template score as physical identity or eligibility.
"""
from __future__ import annotations

import cv2
import numpy as np

from rinkspace.session.print_patterns import project


def coarse_print_registration(pattern, image, exclusion, initial_H, search_margin_px=80.):
    """Search a bounded source ROI, then call the unchanged ``pattern.register``.

    A complete projected template must fit inside the actual source search ROI.
    Search canvases never exceed the source image and are never rescaled to make
    a distant or oversized native projection fit. Failure describes navigation,
    not visibility. The original registration result remains the final gate.
    """
    metadata = dict(search_only=True, source_UV_generated=False,
        qualification_thresholds_unchanged=True,
        exclusion_applied_by_original_registration_only=True)

    def unavailable(reason, status='coarse_search_unavailable'):
        return dict(status=status, reason=reason, coarse_search=dict(metadata, status='unavailable'))

    try:
        image_array, mask = np.asarray(image), np.asarray(exclusion)
        H = np.asarray(initial_H, dtype=np.float64)
        points = np.asarray(pattern.points, dtype=np.float64)
        ink = np.asarray(pattern.ink)
        margin = float(search_margin_px)
    except (TypeError, ValueError, AttributeError, OverflowError):
        return unavailable('invalid_input')
    if (image_array.ndim != 3 or image_array.shape[2] != 3 or min(image_array.shape[:2]) < 1
            or image_array.dtype.kind not in 'uif' or not np.isfinite(image_array).all()):
        return unavailable('invalid_source_image')
    if (mask.shape != image_array.shape[:2] or mask.dtype not in
            (np.dtype('uint8'), np.dtype('uint16'), np.dtype('int16'), np.dtype('float32'), np.dtype('float64'))
            or not np.isfinite(mask).all()):
        return unavailable('invalid_exclusion')
    if not np.isfinite(margin) or margin < 0:
        return unavailable('invalid_search_margin')
    if H.shape != (3, 3) or not np.isfinite(H).all():
        return unavailable('invalid_initial_homography')
    sign, logdet = np.linalg.slogdet(H)
    if sign == 0 or not np.isfinite(logdet):
        return unavailable('singular_initial_homography')
    if points.ndim != 2 or points.shape[1:] != (2,) or len(points) < 3 or not np.isfinite(points).all():
        return unavailable('invalid_pattern_points')
    if ink.ndim != 2 or min(ink.shape) < 1 or ink.dtype.kind not in 'uif' or not np.isfinite(ink).all():
        return unavailable('invalid_pattern_template')
    # OpenCV maps points at a projective pole to finite zeros. Check the actual
    # homogeneous denominator before using its projection or casting to ints.
    with np.errstate(over='ignore', invalid='ignore'):
        denominator = points @ H[2, :2] + H[2, 2]
    if (not np.isfinite(denominator).all() or np.any(denominator == 0)
            or not (np.all(denominator > 0) or np.all(denominator < 0))):
        return unavailable('invalid_projected_template')
    q = project(H, points)
    if not np.isfinite(q).all():
        return unavailable('invalid_projected_template')
    lo_float, hi_float = np.floor(q.min(0) - 3), np.ceil(q.max(0) + 4)
    image_size = np.array([image_array.shape[1], image_array.shape[0]])
    metadata.update(initial_H=H.tolist(), search_margin_px=margin,
        projected_template_bbox=dict(min_uv=lo_float.tolist(), max_uv_exclusive=hi_float.tolist()))
    if np.any(hi_float <= -margin) or np.any(lo_float >= image_size + margin):
        return unavailable('projected_pattern_outside_search_range', status='outside_search')
    size_float = hi_float - lo_float
    if not np.isfinite(size_float).all() or np.any(size_float <= 0):
        return unavailable('invalid_template_extent')
    if np.any(size_float > image_size):
        return unavailable('complete_template_larger_than_source_image')
    # Clip floating bounds first: neither a huge translation nor a huge caller
    # margin may overflow an integer cast or request a canvas outside the image.
    with np.errstate(over='ignore', invalid='ignore'):
        target_lo = np.clip(np.floor(lo_float - margin), [0, 0], image_size).astype(np.int64)
        target_hi = np.clip(np.ceil(hi_float + margin), [0, 0], image_size).astype(np.int64)
    size = size_float.astype(np.int64)
    metadata.update(template_size_wh=size.tolist(),
        target_bbox=dict(min_uv=target_lo.tolist(), max_uv_exclusive=target_hi.tolist()))
    if np.any(target_hi - target_lo < size):
        return unavailable('no_complete_template_search_range')
    shift = np.eye(3)
    shift[:2, 2] = -lo_float
    try:
        with np.errstate(over='ignore', invalid='ignore'):
            local_H = shift @ H
            source_ink = ink.astype(np.float32)
        if not np.isfinite(local_H).all() or not np.isfinite(source_ink).all():
            return unavailable('invalid_pattern_template')
        template = cv2.warpPerspective(source_ink, local_H, tuple(map(int, size)))
        if not np.isfinite(template).all() or np.ptp(template) <= 0:
            return unavailable('constant_or_invalid_warped_template')
        target = image_array[target_lo[1]:target_hi[1], target_lo[0]:target_hi[0]]
        signal = np.asarray(pattern.ink_signal(target), dtype=np.float32)
        if signal.shape != target.shape[:2] or not np.isfinite(signal).all():
            return unavailable('invalid_source_ink_signal')
        score = cv2.matchTemplate(signal, template, cv2.TM_CCOEFF_NORMED)
    except cv2.error:
        return unavailable('opencv_coarse_search_failed')
    finite = np.isfinite(score)
    if not finite.any():
        return unavailable('no_finite_template_match')
    clean_score = np.where(finite, score, -np.inf)
    row, column = np.unravel_index(np.argmax(clean_score), score.shape)
    translation = target_lo + np.array([column, row]) - lo_float
    T = np.eye(3)
    T[:2, 2] = translation
    candidate_H = T @ H
    if not np.isfinite(candidate_H).all():
        return unavailable('invalid_search_initialization')
    metadata.update(status='candidate_search_initialization', score=float(score[row, column]),
        match_top_left_uv=(target_lo + [column, row]).tolist(),
        translation_px=translation.tolist(), candidate_H=candidate_H.tolist())
    result = dict(pattern.register(image, exclusion, candidate_H))
    result['coarse_search'] = metadata
    return result
