"""Mask-isolated print/curve re-observation near a supplied model projection.

This refines an existing reference-to-current map; it cannot initialize a rink
identity or resolve look-alike patterns. Reference pixels must already belong
to a named model anchor. Caller masks must exclude people, boards and unstable
texture. Results are always candidates or refusals, never independent checks.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np

from .masked_photometric import estimate_photometric


ALGORITHM_VERSION = 'model-print-bandpass-roi-normal-sensitivity-3'


@dataclass(frozen=True)
class CurveRefinementConfig:
    """Development safeguards in supplied-image pixels, not an ISU standard."""
    motion_model: str = 'affine'
    low_frequency_radius_px: int = 10
    minimum_bandpass_std: float = 0.004
    edge_gradient_threshold: float = 0.010
    minimum_curve_pixels: int = 120
    minimum_orientation_ratio: float = 0.025
    normal_sampling_tile_px: int = 12
    minimum_normal_tiles: int = 16
    maximum_affine_normal_condition: float = 100.0
    maximum_unit_rms_normal_sensitivity_px: float = 60.0
    minimum_support_fraction: float = 0.015
    minimum_overlap_pixels: int = 1200
    minimum_support_retained: float = 0.90
    maximum_delta_px: float = 18.0
    maximum_curve_p90_px: float = 3.0
    maximum_curve_regression_px: float = 0.35


def _erode(mask, radius):
    return cv2.erode(mask.astype(np.uint8), np.ones((2 * radius + 1,) * 2, np.uint8),
                     borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)


def _bandpass(image, mask, radius):
    """Replace protected values before any color, range or filter operation."""
    raw, mask = np.asarray(image), np.asarray(mask)
    if raw.ndim != 2 or mask.shape != raw.shape:
        raise ValueError('reference/current must be grayscale with matching masks')
    if mask.dtype.kind not in 'buif' or not np.all(np.isfinite(mask)):
        raise ValueError('masks must be finite numeric arrays')
    valid = mask > 0
    clean = np.zeros(raw.shape, np.float64)
    clean[valid] = raw[valid]
    if not np.all(np.isfinite(clean)):
        raise ValueError('usable grayscale values must be finite')
    if raw.dtype.kind == 'u':
        clean /= np.iinfo(raw.dtype).max
    elif raw.dtype.kind != 'f' or np.any(clean < 0) or np.any(clean > 1):
        raise ValueError('images must be unsigned integer or [0, 1] float arrays')
    fine = cv2.GaussianBlur(clean, (5, 5), .8, borderType=cv2.BORDER_CONSTANT)
    coarse = cv2.GaussianBlur(clean, (2 * radius + 1,) * 2, max(2., radius / 3.),
                              borderType=cv2.BORDER_CONSTANT)
    feature = np.clip(.5 + 2. * (fine - coarse), 0., 1.)
    safe = _erode(valid, radius)
    feature[~safe] = .5
    return feature, safe


def _project(h, points):
    points = np.asarray(points, np.float64)
    q = np.c_[points, np.ones(len(points))] @ h.T
    with np.errstate(divide='ignore', invalid='ignore'):
        return q[:, :2] / q[:, 2:3]


def _hull(mask):
    y, x = np.nonzero(mask)
    if len(x) < 3:
        return np.empty((0, 2), np.float64)
    return cv2.convexHull(np.c_[x, y].astype(np.float32)).reshape(-1, 2).astype(np.float64)


def _plausible(h, support):
    if h.shape != (3, 3) or not np.isfinite(h).all() or abs(np.linalg.det(h)) < 1e-12:
        return False
    if len(support) < 3:
        return False
    denom = (np.c_[support, np.ones(len(support))] @ h.T)[:, 2]
    mapped = _project(h, support)
    if (not np.isfinite(mapped).all() or np.min(np.abs(denom)) < 1e-9
            or np.min(denom) * np.max(denom) <= 0):
        return False
    before = cv2.contourArea(support.astype(np.float32), oriented=True)
    after = cv2.contourArea(mapped.astype(np.float32), oriented=True)
    return bool(before * after > 0)


def _warp_safe(image, safe, h_target_from_image, shape):
    """Bilinear remap with all four source corners checked before sampling."""
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    coordinates = _project(np.linalg.inv(h_target_from_image), np.c_[xx.ravel(), yy.ravel()])
    finite = np.isfinite(coordinates).all(axis=1)
    valid = finite & (coordinates[:, 0] >= 0) & (coordinates[:, 1] >= 0)
    valid &= (coordinates[:, 0] < image.shape[1] - 1) & (coordinates[:, 1] < image.shape[0] - 1)
    indices = np.flatnonzero(valid)
    integer = np.floor(coordinates[indices]).astype(np.int64)
    x, y = integer.T
    valid[indices] &= safe[y, x] & safe[y, x + 1] & safe[y + 1, x] & safe[y + 1, x + 1]
    # Invalid coordinates never access original image values, including NaNs.
    coordinates[~valid] = -2.
    mapx = coordinates[:, 0].reshape(shape).astype(np.float32)
    mapy = coordinates[:, 1].reshape(shape).astype(np.float32)
    warped = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=.5)
    valid = valid.reshape(shape)
    warped[~valid] = .5
    return warped, valid


def _curves(image, safe, threshold):
    """Local gradient maxima; no hysteresis connectivity across mask edges."""
    isolated = image.copy()
    isolated[~safe] = .5
    gx = cv2.Sobel(isolated, cv2.CV_64F, 1, 0, ksize=3) / 8.
    gy = cv2.Sobel(isolated, cv2.CV_64F, 0, 1, ksize=3) / 8.
    magnitude = np.hypot(gx, gy)
    angle = (np.arctan2(gy, gx) * 180 / np.pi + 180) % 180
    direction = ((angle + 22.5) // 45).astype(np.int8) % 4
    maxima = np.zeros(image.shape, bool)
    for code, (dy, dx) in enumerate(((0, 1), (1, 1), (1, 0), (1, -1))):
        maxima |= ((direction == code)
                   & (magnitude >= np.roll(magnitude, (dy, dx), axis=(0, 1)))
                   & (magnitude >= np.roll(magnitude, (-dy, -dx), axis=(0, 1))))
    edges = maxima & (magnitude >= threshold) & _erode(safe, 2)
    gradient = np.c_[gx[edges], gy[edges]]
    if len(gradient):
        gradient /= np.maximum(np.linalg.norm(gradient, axis=1, keepdims=True), 1e-12)
        values = np.linalg.eigvalsh(gradient.T @ gradient)
        orientation_ratio = float(values[0] / max(values[-1], 1e-12))
    else:
        orientation_ratio = 0.
    return edges, orientation_ratio


def _curve_diagnostics(reference_warped, current, shared, config):
    ref_edges, ref_ratio = _curves(reference_warped, shared, config.edge_gradient_threshold)
    cur_edges, cur_ratio = _curves(current, shared, config.edge_gradient_threshold)
    counts = [int(np.count_nonzero(ref_edges)), int(np.count_nonzero(cur_edges))]
    result = {'reference_curve_pixel_count': counts[0], 'current_curve_pixel_count': counts[1],
              'reference_orientation_ratio': ref_ratio, 'current_orientation_ratio': cur_ratio,
              'residual_coordinate_space': 'current_input_image_px',
              'residual_semantics': 'symmetric_nearest_bandpass_gradient_maximum_distance_not_independent_landmarks',
              'independent_validation': False}
    if min(counts) == 0:
        return result
    to_current = cv2.distanceTransform((~cur_edges).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[ref_edges]
    to_reference = cv2.distanceTransform((~ref_edges).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[cur_edges]
    distances = np.r_[to_current, to_reference]
    result.update(median_px=float(np.median(distances)), p90_px=float(np.percentile(distances, 90)),
                  max_px=float(np.max(distances)), mean_px=float(np.mean(distances)),
                  reference_to_current_p90_px=float(np.percentile(to_current, 90)),
                  current_to_reference_p90_px=float(np.percentile(to_reference, 90)))
    return result


def _affine_normal_sensitivity(image, shared, config):
    """Local affine/similarity observability from spaced curve normals.

    One curve pixel per tile limits the influence of contiguous edge samples.
    The returned bound is a linear worst-case sensitivity to a unit-RMS normal
    residual vector; it is NOT a calibrated uncertainty or confidence bound.
    Long, flattened rings may be observable despite a low global orientation
    ratio; parallel lines still leave a singular affine normal system.
    """
    edges, orientation = _curves(image, shared, config.edge_gradient_threshold)
    y, x = np.nonzero(edges)
    result = {'orientation_ratio': orientation, 'normal_sample_count': 0, 'rank': 0,
              'condition': None, 'unit_rms_normal_support_displacement_bound_px': None,
              'sampling_tile_px': config.normal_sampling_tile_px,
              'motion_model': config.motion_model,
              'geometric_parameter_count': 4 if config.motion_model == 'similarity' else 6,
              'semantics': 'linear_motion_model_sensitivity_to_unit_rms_curve_normal_residual_not_calibrated_uncertainty'}
    if not len(x):
        return result
    points = np.c_[x, y]
    _, selected = np.unique(points // config.normal_sampling_tile_px, axis=0, return_index=True)
    points = points[selected]
    isolated = image.copy()
    isolated[~shared] = .5
    gx = cv2.Sobel(isolated, cv2.CV_64F, 1, 0, ksize=3) / 8.
    gy = cv2.Sobel(isolated, cv2.CV_64F, 0, 1, ksize=3) / 8.
    normals = np.c_[gx[points[:, 1], points[:, 0]], gy[points[:, 1], points[:, 0]]]
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    center = np.mean(points, axis=0)
    scale = max(float(np.max(np.ptp(points, axis=0))) / 2., 1.)
    normalized = (points - center) / scale
    nx, ny = normals.T
    px, py = normalized.T
    design = np.c_[nx * px, nx * py, nx, ny * px, ny * py, ny]
    # Affine differential [h00,h01,tx,h10,h11,ty] projected onto
    # [uniform_scale,rotation,tx,ty]. No shear/anisotropic scale can enter.
    basis = None
    if config.motion_model == 'similarity':
        basis = np.array([[1., 0, 0, 0], [0, -1., 0, 0], [0, 0, 1., 0],
                          [0, 1., 0, 0], [1., 0, 0, 0], [0, 0, 0, 1.]])
        design = design @ basis
    singular = np.linalg.svd(design, compute_uv=False)
    rank = int(np.count_nonzero(singular > singular[0] * 1e-7))
    result.update(normal_sample_count=len(points), rank=rank,
                  normal_sample_xy=points.tolist(), normal_sample_vectors=normals.tolist())
    if rank < result['geometric_parameter_count']:
        return result
    result['condition'] = float(singular[0] / singular[-1])
    inverse = np.linalg.pinv(design)
    bounds = []
    for x, y in (_hull(shared) - center) / scale:
        at_probe = np.array([[x, y, 1., 0, 0, 0], [0, 0, 0, x, y, 1.]])
        if basis is not None:
            at_probe = at_probe @ basis
        bounds.append(float(np.linalg.svd(at_probe @ inverse, compute_uv=False)[0] * np.sqrt(len(points))))
    result['unit_rms_normal_support_displacement_bound_px'] = max(bounds)
    return result


def _normal_support_adequate(diagnostics, config):
    return (diagnostics['normal_sample_count'] >= config.minimum_normal_tiles
            and diagnostics['rank'] == diagnostics['geometric_parameter_count']
            and diagnostics['condition'] <= config.maximum_affine_normal_condition
            and diagnostics['unit_rms_normal_support_displacement_bound_px'] <= config.maximum_unit_rms_normal_sensitivity_px)


def _local_photometric(warped, current, warped_safe, current_safe, shared, config):
    """Use the observed overlap ROI, then conjugate its delta back to full pixels."""
    y, x = np.nonzero(shared)
    margin = int(np.ceil(config.maximum_delta_px)) + 8
    x0, y0 = max(0, int(x.min()) - margin), max(0, int(y.min()) - margin)
    x1, y1 = min(current.shape[1], int(x.max()) + margin + 1), min(current.shape[0], int(y.max()) + margin + 1)
    roi = np.s_[y0:y1, x0:x1]
    audit = estimate_photometric(warped[roi], current[roi], warped_safe[roi], current_safe[roi],
                                  np.eye(3), motion_model=config.motion_model)
    audit.update(optimizer_roi_full_image_xyxy=[x0, y0, x1, y1],
                 fitting_coordinates='optimizer_roi_px',
                 roi_origin_full_image_xy=[x0, y0],
                 support_scope='local_observed_overlap_only_not_entire_frame')
    if audit['status'] != 'candidate':
        return audit, None
    roi_from_full = np.array([[1., 0, -x0], [0, 1., -y0], [0, 0, 1.]])
    full_map = np.linalg.inv(roi_from_full) @ np.asarray(audit['H_reference_from_current']) @ roi_from_full
    audit['H_prewarpedReference_full_from_current_full'] = full_map.tolist()
    return audit, full_map


def refine_model_projection(reference, current, reference_mask, current_mask,
                            H_current_from_reference, *, config=None):
    """Return a bounded re-observation candidate with explicit direction/evidence.

    Prewarp the model anchor by H0. Existing mask-isolated photometric fitting
    estimates D = H_prewarpedReference_from_current near identity. Compose
    H_current_from_reference = inverse(D) @ H0. An affine or similarity delta preserves
    the supplied project's perspective terms; it does not replace H0 by an
    affine approximation. Similarity constrains D to translation, rotation and
    uniform scale; it is a local approximation after H0, not a camera model.
    A refusal never returns the initial H as a result.
    """
    config = config or CurveRefinementConfig()
    result = {'schema_version': 'rinkspace.model-curve-refinement.v1', 'algorithm_version': ALGORITHM_VERSION,
              'status': 'unavailable', 'reason': None, 'H_current_from_reference': None,
              'matrix_direction': 'H_current_from_reference', 'independent_validation': False,
              'global_initialization_performed': False, 'physical_landmark_identity_verified': False,
              'thresholds': asdict(config), 'threshold_basis': 'declared_development_safeguards_not_an_industry_standard',
              'motion_model': config.motion_model,
              'model_assumption': ('four_dof_local_similarity_delta_translation_rotation_uniform_scale_not_a_physical_camera_or_metric_model'
                                   if config.motion_model == 'similarity' else 'local_affine_or_projective_delta_not_a_physical_camera_or_metric_model'),
              'photometric_audit': None,
              'protection': {'outside_mask_replaced_before_processing': True,
                             'bandpass_filter_support_radius_px': config.low_frequency_radius_px,
                             'prewarp_interpolation': 'bilinear_all_four_reference_corners_safe',
                             'optimizer': 'existing_masked_photometric_with_both_masks',
                             'mask_semantics': 'caller_supplied_stable_ice_print_regions_not_automatically_verified'},
              'limitations': ['requires_a_credible_model_anchor_and_initial_projection',
                              'look_alike_curve_identity_is_not_established_here',
                              'scratch_reflection_and_people_exclusion_depend_on_caller_masks',
                              'not_an_independent_geometric_accuracy_check',
                              'outside_observed_support_is_extrapolation']}

    def fail(reason):
        result['reason'] = reason
        return result

    numeric = [value for name, value in asdict(config).items() if name != 'motion_model']
    if (not all(isinstance(value, (int, float)) and not isinstance(value, bool)
                and np.isfinite(value) for value in numeric)
            or not isinstance(config.low_frequency_radius_px, int)
            or not isinstance(config.normal_sampling_tile_px, int) or config.normal_sampling_tile_px < 2
            or config.minimum_normal_tiles < 6
            or config.maximum_affine_normal_condition <= 1
            or config.maximum_unit_rms_normal_sensitivity_px <= 0
            or config.motion_model not in {'affine', 'similarity', 'homography', 'homography_regularized'}
            or config.low_frequency_radius_px < 3 or config.low_frequency_radius_px > 30
            or config.minimum_overlap_pixels < 160 or config.minimum_curve_pixels < 20
            or not 0 < config.minimum_support_retained <= 1
            or not 0 < config.minimum_orientation_ratio < 1
            or not 0 < config.minimum_support_fraction < 1
            or config.minimum_bandpass_std <= 0 or config.edge_gradient_threshold <= 0
            or config.maximum_curve_regression_px < 0
            or config.maximum_delta_px <= 0 or config.maximum_curve_p90_px <= 0):
        return fail('invalid_refinement_configuration')
    try:
        ref, ref_safe = _bandpass(reference, reference_mask, config.low_frequency_radius_px)
        cur, cur_safe = _bandpass(current, current_mask, config.low_frequency_radius_px)
        h0 = np.asarray(H_current_from_reference, np.float64)
        ref_hull = _hull(ref_safe)
        if h0.shape != (3, 3) or not np.isfinite(h0).all() or abs(h0[2, 2]) < 1e-12:
            return fail('invalid_initial_homography')
        h0 = h0 / h0[2, 2]
        if not _plausible(h0, ref_hull):
            return fail('initial_homography_horizon_reflection_or_empty_support')
        warped, warped_safe = _warp_safe(ref, ref_safe, h0, cur.shape)
    except (ValueError, TypeError, cv2.error, np.linalg.LinAlgError) as error:
        result['input_error'] = str(error)
        return fail('invalid_input')
    result['initial_H_current_from_reference'] = h0.tolist()
    shared = warped_safe & cur_safe
    count = int(np.count_nonzero(shared))
    initial_hull = _hull(shared)
    hull_fraction = float(cv2.contourArea(initial_hull.astype(np.float32)) / cur.size) if len(initial_hull) else 0.
    result['initial_support'] = {'pixel_count': count, 'hull_current_xy': initial_hull.tolist(), 'hull_fraction': hull_fraction}
    if count < config.minimum_overlap_pixels or hull_fraction < config.minimum_support_fraction:
        return fail('insufficient_bidirectional_mask_overlap')
    result['initial_texture_std'] = {'reference': float(np.std(warped[shared])), 'current': float(np.std(cur[shared]))}
    if min(result['initial_texture_std'].values()) < config.minimum_bandpass_std:
        return fail('insufficient_stable_print_texture')
    before = _curve_diagnostics(warped, cur, shared, config)
    result['initial_curve_residual'] = before
    if min(before['reference_curve_pixel_count'], before['current_curve_pixel_count']) < config.minimum_curve_pixels:
        return fail('insufficient_observed_curve_pixels')
    normal_diagnostics = {'reference': _affine_normal_sensitivity(warped, shared, config),
                          'current': _affine_normal_sensitivity(cur, shared, config)}
    result['initial_affine_normal_sensitivity'] = normal_diagnostics
    if not all(_normal_support_adequate(value, config) for value in normal_diagnostics.values()):
        if min(before['reference_orientation_ratio'], before['current_orientation_ratio']) < config.minimum_orientation_ratio:
            return fail('curve_orientation_aperture_problem')
        return fail('curve_similarity_normal_system_underconstrained' if config.motion_model == 'similarity'
                    else 'curve_affine_normal_system_underconstrained')
    sensitivity_model = 'similarity' if config.motion_model == 'similarity' else 'affine'
    result['orientation_guard'] = (f'anisotropic_normals_but_bounded_{sensitivity_model}_sensitivity_on_local_support'
        if min(before['reference_orientation_ratio'], before['current_orientation_ratio']) < config.minimum_orientation_ratio
        else f'distributed_normals_and_bounded_{sensitivity_model}_sensitivity_on_local_support')
    audit, full_map = _local_photometric(warped, cur, warped_safe, cur_safe, shared, config)
    result['photometric_audit'] = audit
    result['photometric_residual_semantics'] = 'gain_bias_fitted_bandpass_intensity_not_geometric_pixels'
    if audit['status'] != 'candidate':
        return fail('photometric_refinement_refused:' + audit['reason'])
    delta = np.linalg.inv(full_map)
    h = delta @ h0
    h /= h[2, 2]
    correction = np.linalg.norm(_project(delta, initial_hull) - initial_hull, axis=1)
    result['maximum_observed_support_delta_px'] = float(np.max(correction))
    if (not _plausible(h, ref_hull) or not _plausible(delta, initial_hull)
            or not np.isfinite(correction).all() or np.max(correction) > config.maximum_delta_px):
        return fail('model_projection_correction_out_of_bounds')
    refined, refined_safe = _warp_safe(ref, ref_safe, h, cur.shape)
    final_shared = refined_safe & cur_safe
    retained = np.count_nonzero(final_shared & shared) / count
    result['safe_support_retained'] = float(retained)
    if retained < config.minimum_support_retained:
        return fail('bidirectional_mask_support_lost')
    # Compare the same spatial footprint; discarding hard regions is no gain.
    common = final_shared & shared
    comparable_before = _curve_diagnostics(warped, cur, common, config)
    after = _curve_diagnostics(refined, cur, common, config)
    result['comparison_before_curve_residual'] = comparable_before
    result['final_curve_residual'] = after
    if min(after['reference_curve_pixel_count'], after['current_curve_pixel_count']) < config.minimum_curve_pixels:
        return fail('refined_curve_support_insufficient')
    final_normal = {'reference': _affine_normal_sensitivity(refined, common, config),
                    'current': _affine_normal_sensitivity(cur, common, config)}
    result['final_affine_normal_sensitivity'] = final_normal
    if not all(_normal_support_adequate(value, config) for value in final_normal.values()):
        return fail('refined_curve_similarity_normal_system_underconstrained' if config.motion_model == 'similarity'
                    else 'refined_curve_affine_normal_system_underconstrained')
    if max(after['reference_to_current_p90_px'], after['current_to_reference_p90_px']) > config.maximum_curve_p90_px:
        return fail('refined_curve_distance_exceeds_declared_limit')
    if after['mean_px'] > comparable_before['mean_px'] + config.maximum_curve_regression_px:
        return fail('curve_residual_regressed_on_common_support')
    yy, xx = np.nonzero(common)
    result.update(status='candidate', reason='bounded_model_print_reobservation_candidate',
                  H_current_from_reference=h.tolist(), delta_current_from_initial_projection=delta.tolist(),
                  support_current_polygon=_hull(common).tolist(), support_pixel_count=len(xx),
                  support_coordinate_space='current_input_image_px',
                  composition='inverse(H_prewarpedReference_from_current) @ initial_H_current_from_reference')
    return result
