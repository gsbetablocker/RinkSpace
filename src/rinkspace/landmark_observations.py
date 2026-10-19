"""Native-image marking edges, deliberately without model identity or pose.

Points are Canny pixels read from the image, not samples of a fitted ellipse.
The optional conic is a shape proposal only. A gray edge can be the inner or
outer side of an ink band; that ambiguity is retained for later association.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import cv2
import numpy as np

ALGORITHM_VERSION = 'native-multiscale-visible-marking-edges-v10-2'


@dataclass(frozen=True)
class LandmarkDetectionConfig:
    blur_sigma_source_px: float = 1.6
    canny_low: float = 4.0
    canny_high: float = 12.0
    minimum_length_source_px: float = 70.0
    sample_spacing_source_px: float = 7.0
    maximum_candidates: int = 100
    minimum_unmasked_length_source_px: float = 45.0


def polygon_mask(shape, polygons, sample_size):
    out = np.zeros(shape[:2], np.uint8)
    scale = np.array([shape[1] / sample_size[0], shape[0] / sample_size[1]])
    for polygon in polygons:
        if len(polygon) >= 3:
            cv2.fillPoly(out, [np.rint(np.asarray(polygon) * scale).astype(np.int32)], 255)
    return out


def _subsample(points, spacing):
    # findContours retraces isolated one-pixel edges. Remove exact duplicate
    # pixels before spatial subsampling so they do not inflate evidence.
    seen, unique = set(), []
    for p in points:
        key = (int(p[0]), int(p[1]))
        if key not in seen:
            seen.add(key); unique.append(p)
    if len(unique) < 2:
        return np.asarray(unique, float).reshape(-1, 2)
    selected = [unique[0]]
    for p in unique[1:]:
        if np.linalg.norm(np.asarray(p) - selected[-1]) >= spacing:
            selected.append(p)
    return np.asarray(selected, float)


def _segments(points, valid, maximum_gap=25.):
    result, start = [], None
    for i, ok in enumerate(valid):
        if start is not None and i > start and np.linalg.norm(points[i] - points[i - 1]) > maximum_gap:
            if i - start >= 5:
                result.append(points[start:i])
            start = None
        if ok and start is None:
            start = i
        if start is not None and (not ok or i == len(points) - 1):
            stop = i + 1 if ok else i
            if stop - start >= 5:
                result.append(points[start:stop])
            start = None
    return result


def _ellipse_candidate(points, shape):
    if len(points) < 12:
        return None
    try:
        (cx, cy), (a, b), angle = cv2.fitEllipse(points.astype(np.float32))
    except cv2.error:
        return None
    if not np.isfinite([cx, cy, a, b, angle]).all() or min(a, b) < 5 or max(a, b) > shape[1] * 12:
        return None
    theta = np.deg2rad(angle)
    centered = points - [cx, cy]
    q = centered @ np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    norm = np.sqrt((q[:, 0] / (a / 2)) ** 2 + (q[:, 1] / (b / 2)) ** 2)
    residual = np.abs(norm - 1) * min(a, b) / 2
    if np.median(residual) > 6 or np.percentile(residual, 90) > 12:
        return None
    return {'center_source_px': [cx, cy], 'diameters_source_px': [a, b],
            'angle_degrees': angle, 'median_approx_normal_error_source_px': float(np.median(residual)),
            'is_observation': False, 'center_is_not_physical_circle_center_projection': True,
            'occluded_or_missing_contour_is_not_completed': True}


def visible_segments(observation):
    """Resolve compact actual segments without joining an occlusion gap."""
    if 'visible_segment_ranges' in observation:
        points = observation['points_source_px']
        return [points[start:stop] for start, stop in observation['visible_segment_ranges']]
    return observation.get('visible_segments_source_px', [observation['points_source_px']])


def detect_landmarks(image, search_mask, foreground_mask=None, *, conservative_mask=None,
                     sample_size=(960, 540), config=None):
    """Extract image-supported candidates over the entire supplied ice search.

    Search masks are image/boundary candidates, never a model-prediction ROI.
    Foreground is applied after extraction so removed arcs remain auditable.
    """
    config = config or LandmarkDetectionConfig()
    if image.ndim != 3 or search_mask.shape != image.shape[:2]:
        raise ValueError('BGR image and equal-sized search mask required')
    if image.dtype != np.uint8:
        raise ValueError('native decoded uint8 BGR pixels required')
    shape = image.shape
    foreground_mask = np.zeros(shape[:2], np.uint8) if foreground_mask is None else foreground_mask
    conservative_mask = foreground_mask if conservative_mask is None else conservative_mask
    if foreground_mask.shape != shape[:2] or conservative_mask.shape != shape[:2]:
        raise ValueError('mask shape mismatch')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    smooth = cv2.GaussianBlur(gray, (0, 0), config.blur_sigma_source_px)
    edges = cv2.Canny(smooth, config.canny_low, config.canny_high)
    # Additional coarser edges rescue broad gray paint without upscaling a
    # small preview. Both branches operate on the original decoded image.
    coarse = cv2.GaussianBlur(gray, (0, 0), 3.2)
    coarser_edges = cv2.Canny(coarse, config.canny_low * .65, config.canny_high * .65)
    fine_support = cv2.dilate(edges, np.ones((9, 9), np.uint8))
    edges[(coarser_edges > 0) & (fine_support == 0)] = 255
    edges[search_mask == 0] = 0
    # Avoid manufacturing a contour along a search-mask cut.
    edges[cv2.erode(search_mask, np.ones((7, 7), np.uint8)) == 0] = 0
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours.sort(key=lambda c: cv2.arcLength(c, False), reverse=True) if isinstance(contours, list) else None
    contours = sorted(contours, key=lambda c: cv2.arcLength(c, False), reverse=True)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, saturation, _ = cv2.split(hsv)
    near_red = cv2.dilate((((hue < 12) | (hue > 166)) & (saturation > 12)).astype(np.uint8), np.ones((17, 17), np.uint8))
    near_blue = cv2.dilate(((hue >= 90) & (hue <= 137) & (saturation > 12)).astype(np.uint8), np.ones((17, 17), np.uint8))
    observations, already = [], []
    for contour in contours:
        full = contour[:, 0, :]
        if cv2.arcLength(contour, False) < config.minimum_length_source_px:
            continue
        points = _subsample(full, config.sample_spacing_source_px)
        if len(points) < 8 or np.linalg.norm(np.ptp(points, axis=0)) < config.minimum_length_source_px / 2:
            continue
        # RETR_LIST can give the same thin edge on both sides. Dedupe exact
        # spatial support, not different nearby inner/outer ink-band edges.
        fingerprint = frozenset(map(tuple, np.floor(points / 8).astype(int)))
        if any(len(fingerprint & old) / max(1, len(fingerprint)) > .82 for old in already):
            continue
        already.append(fingerprint)
        xi, yi = np.rint(points).astype(int).T
        masked = foreground_mask[yi, xi] > 0
        formerly_masked = conservative_mask[yi, xi] > 0
        segments = _segments(points, ~masked, max(25., config.sample_spacing_source_px * 2.5))
        accepted_points = np.concatenate(segments) if segments else np.empty((0, 2))
        length = sum(float(np.linalg.norm(np.diff(s, axis=0), axis=1).sum()) for s in segments)
        is_masked = length < config.minimum_unmasked_length_source_px
        if is_masked:
            accepted_points = points
            segments = _segments(points, np.ones(len(points), bool), max(25., config.sample_spacing_source_px * 2.5))
        lookup = {tuple(p): i for i, p in enumerate(accepted_points)}
        segment_ranges = [[lookup[tuple(s[0])], lookup[tuple(s[-1])] + 1] for s in segments]
        # A luminance edge is often at the desaturated outside of a blurred
        # painted stripe. Read colour in an 8px neighbourhood, and retain
        # weaker colour alternatives rather than force a stripe to be gray.
        red, blue = near_red[yi, xi], near_blue[yi, xi]
        colour = 'red' if red.mean() > .45 else 'blue' if blue.mean() > .45 else 'gray'
        centered = accepted_points - accepted_points.mean(axis=0)
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        normal_residual = np.abs(centered @ vh[-1])
        straight = float(np.percentile(normal_residual, 90)) < max(2.5, .007 * np.linalg.norm(np.ptp(accepted_points, axis=0)))
        obs_id = f'observed_{len(observations):03d}'
        observations.append({
            'observation_id': obs_id, 'kind': ('arc_or_line' if colour == 'gray' else 'line') if straight else 'arc',
            'surface_type': 'ice_marking_candidate', 'colour_class': colour,
            'colour_candidates': list(dict.fromkeys([colour] + (['red'] if red.mean() > .12 else []) + (['blue'] if blue.mean() > .12 else []))),
            'colour_neighbourhood_fraction': {'red': float(red.mean()), 'blue': float(blue.mean())},
            'linearity_p90_source_px': float(np.percentile(normal_residual, 90)),
            'line_like_appearance_does_not_resolve_physical_line_vs_shallow_arc': True,
            'points_source_px': accepted_points.tolist(),
            'visible_segment_ranges': segment_ranges,
            'visible_segment_range_convention': 'half_open_indices_into_points_source_px',
            'measurement_definition': 'visible_intensity_edge_inner_outer_band_side_unresolved',
            'image_scale_from_source': [sample_size[0] / shape[1], sample_size[1] / shape[0]],
            'ellipse_candidate': _ellipse_candidate(accepted_points, shape) if colour == 'gray' or not straight else None,
            'candidate_model_ids': [], 'selected_model_id': None,
            'association_status': 'masked_out' if is_masked else 'detected_unidentified',
            'used_by_solver': False, 'residual_summary_sample_px': None,
            'measurement_uncertainty_source_px': 4.0 if colour == 'gray' else 6.0,
            'group_weight': 1.0, 'identity_basis': [], 'manual_confirmed': False,
            'rejection_stage': 'foreground_mask' if is_masked else 'identity_pending',
            'rejection_reason': 'too_little_visible_edge_outside_current_frame_foreground_candidate' if is_masked else 'model_identity_not_assigned_by_image_detector',
            'mask_fraction': float(masked.mean()), 'v09_conservative_mask_fraction': float(formerly_masked.mean()),
            'observed_edge_length_source_px': length,
            'extraction_method': 'native_multiscale_luminance_gradient_canny_visible_pixels',
            'search_method': 'expanded_full_image_ice_candidate_no_model_projection',
            'fit_ellipse_used_to_generate_observations': False,
        })
        if len(observations) >= config.maximum_candidates:
            break
    return observations, {'algorithm_version': ALGORITHM_VERSION, 'config': asdict(config),
                          'native_size': [shape[1], shape[0]], 'sample_size': list(sample_size),
                          'search_pixel_count': int(np.count_nonzero(search_mask)),
                          'edge_pixel_count': int(np.count_nonzero(edges)),
                          'actual_image_scale_from_source': [1.0, 1.0],
                          'model_projection_used': False, 'identity_assigned': False}


def draw_landmarks(image, observations):
    out = image.copy()
    for obs in observations:
        masked = obs['association_status'] == 'masked_out'
        colour = (60, 90, 180) if masked else (50, 235, 60)
        for segment in visible_segments(obs):
            cv2.polylines(out, [np.rint(segment).astype(np.int32)], False, colour, 2, cv2.LINE_AA)
        p = np.rint(obs['points_source_px'][len(obs['points_source_px']) // 2]).astype(int)
        cv2.putText(out, obs['observation_id'].split('_')[-1], tuple(p), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 20, 255), 2)
    return out
