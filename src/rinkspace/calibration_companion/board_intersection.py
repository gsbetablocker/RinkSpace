"""Intersect an observed camera ray with an identified finite board surface.

This only computes a conditional point from an existing camera and geometry.
It neither identifies an advertisement nor turns that anchor pixel into an
independent validation observation.
"""
from __future__ import annotations

import cv2
import numpy as np

from .joint_scene import CORNERS, CORNER_START_DEGREES, SIDE_CORNERS, STRAIGHT_SIDES


def intersect_board_surface(camera, uv, side, parameters, allow_corner=None):
    """Return the nearest positive ray intersection on the requested surface.

    ``side`` names one straight side or one quarter-cylinder corner. A straight
    side may include exactly one adjacent corner through ``allow_corner``.
    Bounds, height and individual corner radii always come from ``parameters``;
    the implementation has no camera-role or original rink-size assumptions.
    Surface identity must have been established before this call.
    """
    common = {'valid': False, 'reasons': [], 'xyz': None, 'board_side': side,
              'allow_corner': allow_corner, 'surface_segment': None,
              'anchor_projection_not_a_validation_sample': True,
              'point_status': 'conditional_ray_surface_intersection_NOT_measured_XYZ'}
    if side not in (*STRAIGHT_SIDES, *CORNERS):
        return {**common, 'reasons': ['unsupported_unconfirmed_board_surface']}
    if allow_corner is not None and (side not in STRAIGHT_SIDES or allow_corner not in SIDE_CORNERS[side]):
        return {**common, 'reasons': ['allow_corner_not_adjacent_to_identified_straight_side']}
    bounds = np.asarray(parameters['bounds'], float)
    height = float(parameters['board_height'])
    radii = {c: float(parameters.get('corner_radii', {}).get(c, parameters['corner_radius'])) for c in CORNERS}
    if (bounds.shape != (4,) or not np.isfinite(bounds).all() or
            not np.isfinite([height, *radii.values()]).all() or height <= 0 or min(radii.values()) <= 0):
        return {**common, 'reasons': ['invalid_board_geometry']}
    x0, y0, x1, y1 = bounds
    if (x1 <= x0 or y1 <= y0 or
            x1 - x0 < max(radii['upper_left'] + radii['upper_right'], radii['lower_left'] + radii['lower_right']) or
            y1 - y0 < max(radii['upper_left'] + radii['lower_left'], radii['upper_right'] + radii['lower_right'])):
        return {**common, 'reasons': ['invalid_board_geometry']}
    common['board_height'] = height
    uv = np.asarray(uv, float)
    if uv.shape != (2,) or not np.isfinite(uv).all():
        return {**common, 'reasons': ['invalid_anchor_pixel']}
    try:
        und = cv2.undistortPointsIter(uv.reshape(1, 1, 2), camera.K_px,
            camera.distortion_opencv, None, None,
            (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 200, 1e-13)).reshape(2)
    except cv2.error:
        return {**common, 'reasons': ['inverse_distortion_failed']}
    if not np.isfinite(und).all():
        return {**common, 'reasons': ['nonfinite_inverse_distortion']}
    ray = camera.R.T @ np.r_[und, 1.]
    ray_norm = np.linalg.norm(ray)
    if not np.isfinite(ray_norm) or ray_norm <= 0:
        return {**common, 'reasons': ['nonfinite_inverse_camera_ray']}
    ray /= ray_norm
    origin = np.asarray(camera.center_xyz, float)
    # Validate the inverse branch independently of which surface is hit.
    projected, _ = camera.project([origin + ray])
    roundtrip = float(np.linalg.norm(projected[0] - uv))
    radial_fold = None
    distortion = camera.distortion_opencv
    if len(distortion) >= 5 and np.allclose(distortion[1:], 0):
        radial_fold = float(1 + 3 * distortion[0] * np.dot(und, und))
    common.update(inverse_distortion_roundtrip_px=roundtrip if np.isfinite(roundtrip) else None,
                  radial_mapping_derivative=radial_fold if radial_fold is None or np.isfinite(radial_fold) else None)
    inverse_reasons = []
    if not np.isfinite(roundtrip) or roundtrip > .05:
        inverse_reasons.append('inverse_distortion_roundtrip_exceeds_0.05px')
    if radial_fold is not None and radial_fold <= 0:
        inverse_reasons.append('nonmonotonic_inverse_radial_branch')
    elif radial_fold is not None and not np.isfinite(radial_fold):
        inverse_reasons.append('nonfinite_inverse_radial_branch')
    if inverse_reasons:
        return {**common, 'reasons': inverse_reasons}

    tolerance = 1e-6
    candidates, surface_failures = [], []

    def candidate(distance, segment, metadata, domain_reasons):
        xyz = origin + distance * ray
        if side in STRAIGHT_SIDES and segment.endswith('_corner'):
            # Match the latent-point schema: continuation retains world along,
            # whereas an explicitly identified direct corner uses a fraction.
            axis = 0 if side in ('upper_long_side', 'lower_long_side') else 1
            metadata = {**metadata, 'corner_fraction': metadata['along'],
                        'along_parameterization': 'world_x' if axis == 0 else 'world_y',
                        'along': float(xyz[axis])}
        reasons = list(domain_reasons(xyz))
        if distance <= 1e-9:
            reasons.append('board_intersection_behind_anchor_camera')
        if not -tolerance <= xyz[2] <= height + tolerance:
            reasons.append('intersection_outside_frozen_board_height')
        candidates.append({'xyz': xyz.tolist(), 'surface_segment': segment,
                           'ray_distance_scene_units': float(distance),
                           'valid': not reasons, 'reasons': reasons, **metadata})

    surfaces = [side] + ([allow_corner] if allow_corner is not None else [])
    for surface in surfaces:
        if surface in STRAIGHT_SIDES:
            is_long = surface in ('upper_long_side', 'lower_long_side')
            normal_axis, along_axis = (1, 0) if is_long else (0, 1)
            plane = {'upper_long_side': y1, 'lower_long_side': y0,
                     'left_end': x0, 'right_end': x1}[surface]
            low_corner, high_corner = SIDE_CORNERS[surface]
            lo = bounds[along_axis] + radii[low_corner]
            hi = bounds[along_axis + 2] - radii[high_corner]
            if abs(ray[normal_axis]) < 1e-7:
                surface_failures.append('ray_nearly_parallel_to_board_plane')
                continue
            distance = (plane - origin[normal_axis]) / ray[normal_axis]
            candidate(distance, surface,
                {'plane_axis': normal_axis, 'plane_coordinate': float(plane),
                 'straight_along_bounds': [float(lo), float(hi)],
                 'ray_plane_abs_cosine': float(abs(ray[normal_axis])),
                 'along_parameterization': 'world_x' if is_long else 'world_y',
                 'along': float(origin[along_axis] + distance * ray[along_axis])},
                lambda xyz, axis=along_axis, lo=lo, hi=hi:
                    [] if lo - tolerance <= xyz[axis] <= hi + tolerance else
                    ['intersection_outside_frozen_straight_board_segment'])
            continue

        radius = radii[surface]
        center = np.array([x1 - radius if 'right' in surface else x0 + radius,
                           y1 - radius if 'upper' in surface else y0 + radius])
        offset = origin[:2] - center
        a = float(np.dot(ray[:2], ray[:2]))
        b = 2. * float(np.dot(offset, ray[:2]))
        c = float(np.dot(offset, offset) - radius * radius)
        if a < 1e-14:
            surface_failures.append('ray_coincident_with_board_cylinder' if abs(c) < tolerance
                                    else 'ray_parallel_to_board_cylinder')
            continue
        discriminant = b * b - 4. * a * c
        if discriminant < -1e-12 * max(b * b, abs(4. * a * c), 1.):
            surface_failures.append('ray_misses_board_cylinder')
            continue
        root = np.sqrt(max(discriminant, 0.))
        q = -.5 * (b + np.copysign(root, b))
        roots = [-b / (2. * a)] if abs(q) < 1e-30 else [q / a, c / q]
        roots = sorted(set(float(distance) for distance in roots))
        for distance in roots:
            xy = origin[:2] + distance * ray[:2]
            delta = xy - center
            angle = float(np.rad2deg(np.arctan2(delta[1], delta[0])) % 360.)
            start = CORNER_START_DEGREES[surface]
            if start == 0. and abs(angle - 360.) < 1e-8:
                angle = 0.
            if angle < start - 1e-8:
                angle += 360.
            fraction = (angle - start) / 90.
            if -1e-8 <= fraction <= 1. + 1e-8:
                fraction = float(np.clip(fraction, 0., 1.))
            def domain(xyz, center=center, surface=surface):
                delta = xyz[:2] - center
                valid_x = delta[0] >= -tolerance if 'right' in surface else delta[0] <= tolerance
                valid_y = delta[1] >= -tolerance if 'upper' in surface else delta[1] <= tolerance
                return [] if valid_x and valid_y else ['intersection_outside_frozen_quarter_circle']
            candidate(distance, surface + '_corner',
                {'corner': surface, 'corner_center_xy': center.tolist(), 'corner_radius': radius,
                 'along_parameterization': 'corner_fraction', 'along': fraction,
                 'ray_surface_abs_cosine': float(abs(np.dot(ray[:2], delta / radius)))}, domain)

    accepted = sorted((row for row in candidates if row['valid']), key=lambda row: row['ray_distance_scene_units'])
    distinct = []
    for row in accepted:
        if not any(np.linalg.norm(np.asarray(row['xyz']) - other['xyz']) <= tolerance for other in distinct):
            distinct.append(row)
    if distinct:
        chosen = distinct[0]
        return {**common, **chosen, 'candidate_intersections': candidates,
                'valid_intersection_count': len(distinct),
                'selected_intersection_is_nearest_positive': True,
                'coincident_join_intersections_merged': len(accepted) - len(distinct)}
    positive = sorted((row for row in candidates if row['ray_distance_scene_units'] > 0),
                      key=lambda row: row['ray_distance_scene_units'])
    diagnostic = positive[0] if positive else (candidates[0] if candidates else {})
    reasons = list(dict.fromkeys(reason for row in candidates for reason in row['reasons']))
    reasons.extend(reason for reason in surface_failures if reason not in reasons)
    return {**common, **diagnostic, 'valid': False, 'reasons': reasons or ['no_finite_forward_board_intersection'],
            'candidate_intersections': candidates, 'valid_intersection_count': 0,
            'selected_intersection_is_nearest_positive': False}
