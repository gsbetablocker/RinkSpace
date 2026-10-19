"""Image-only brightness-side evidence for gray ink-band association.

The gradient points towards brighter pixels. It alone does not identify an
outer boundary: the caller must supply independently proposed, correctly
oriented model-curve normals in the same raw/distorted image coordinate space.
No fitted ellipse centre, model ID, or predicted location enters measurement.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib

import cv2
import numpy as np

ALGORITHM_VERSION = 'native-gray-band-gradient-profile-v10-1'


@dataclass(frozen=True)
class BandProfileConfig:
    gradient_sigmas_source_px: tuple[float, ...] = (2.0, 4.0, 8.0)
    sample_offsets_source_px: tuple[float, ...] = (-24., -16., -12., -8., -4., 0., 4., 8., 12., 16., 24.)
    minimum_gradient_level_per_source_px: float = .20
    minimum_contrast_levels: float = 3.0
    minimum_scale_direction_agreement: float = .70
    maximum_gray_saturation: float = 35.0


def point_array_sha256(points):
    return hashlib.sha256(np.asarray(points, dtype='<f8').tobytes(order='C')).hexdigest()


def _sample(image, points):
    p = np.asarray(points, np.float32)
    return cv2.remap(image, p[:, 0].reshape(-1, 1), p[:, 1].reshape(-1, 1),
                     interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                     borderValue=0).reshape(-1)


def measure_band_profiles(image, observations, config=None):
    """Return actual pixel profiles bound to each unchanged observation array."""
    config = config or BandProfileConfig()
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError('Native decoded uint8 BGR image required')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    saturation = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:, :, 1].astype(np.float32)
    gradient_maps = []
    for sigma in config.gradient_sigmas_source_px:
        blurred = cv2.GaussianBlur(gray, (0, 0), sigma)
        # Sobel's 3x3 derivative has an eightfold gain on a unit linear ramp.
        gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3, scale=.125)
        gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3, scale=.125)
        gradient_maps.append((gx, gy))
    profiles = []
    for observation in observations:
        points = np.asarray(observation['points_source_px'], np.float64)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) == 0:
            raise ValueError('Observation must contain original Nx2 points')
        multiscale = np.asarray([np.c_[_sample(gx, points), _sample(gy, points)] for gx, gy in gradient_maps])
        norms = np.linalg.norm(multiscale, axis=2)
        units = multiscale / np.maximum(norms[:, :, None], 1e-8)
        primary = len(gradient_maps) // 2
        gradient, unit = multiscale[primary], units[primary]
        agreement = np.min(np.sum(units * unit[None, :, :], axis=2), axis=0)
        offsets = np.asarray(config.sample_offsets_source_px)
        query = points[:, None, :] + unit[:, None, :] * offsets[None, :, None]
        in_bounds = ((query[:, :, 0] >= 0) & (query[:, :, 0] <= image.shape[1] - 1) &
                     (query[:, :, 1] >= 0) & (query[:, :, 1] <= image.shape[0] - 1))
        values = _sample(gray, query.reshape(-1, 2)).reshape(len(points), len(offsets))
        sat = _sample(saturation, query.reshape(-1, 2)).reshape(len(points), len(offsets))
        near = (abs(offsets) <= 12)
        positive, negative = (offsets >= 8) & (offsets <= 12), (offsets <= -8) & (offsets >= -12)
        contrast = values[:, positive].mean(axis=1) - values[:, negative].mean(axis=1)
        reasons, reliable = [], []
        for i in range(len(points)):
            rejected = []
            if not in_bounds[i, near].all():
                rejected.append('profile_exits_source_image')
            if norms[primary, i] < config.minimum_gradient_level_per_source_px:
                rejected.append('weak_luminance_gradient')
            if contrast[i] < config.minimum_contrast_levels:
                rejected.append('weak_or_nonmonotonic_local_contrast')
            if agreement[i] < config.minimum_scale_direction_agreement:
                rejected.append('gradient_direction_changes_with_scale')
            if np.median(sat[i, near]) > config.maximum_gray_saturation or observation.get('colour_class') in ('red', 'blue'):
                rejected.append('coloured_band_gray_brightness_assumption_unknown')
            if observation.get('association_status') == 'masked_out':
                rejected.append('foreground_exclusion')
            reliable.append(not rejected); reasons.append(rejected)
        reliable = np.asarray(reliable, bool)
        profiles.append({
            'observation_id': observation['observation_id'], 'point_count': len(points),
            'point_array_sha256': point_array_sha256(points),
            'point_array_encoding_for_hash': 'little_endian_float64_Nx2_C_order',
            'point_coordinates_are_original_and_unchanged': True,
            'coordinate_space': 'source_distorted_image_px',
            'gradient_intensity_unit': 'uint8_luminance_level_per_source_pixel',
            'gradients_source_xy': gradient.astype(float).tolist(),
            'gradient_unit_source_xy': unit.astype(float).tolist(),
            'gradient_magnitude_source': norms[primary].astype(float).tolist(),
            'gradient_sigma_source_px': config.gradient_sigmas_source_px[primary],
            'multiscale_direction_agreement': agreement.astype(float).tolist(),
            'contrast_source_levels': contrast.astype(float).tolist(),
            'profile_offsets_source_px': offsets.tolist(),
            'profile_offset_direction': 'positive_along_measured_brightness_gradient_not_model_normal',
            'profile_luminance_source': values.astype(float).tolist(),
            'profile_in_bounds': in_bounds.tolist(),
            'gradient_reliable': reliable.tolist(), 'unknown_reasons': reasons,
            'reliable_point_count': int(reliable.sum()),
            'median_gradient_source_xy': np.median(gradient[reliable], axis=0).astype(float).tolist() if reliable.any() else None,
            'median_contrast_source_levels': float(np.median(contrast[reliable])) if reliable.any() else None,
            'outer_or_inner_identity': 'unknown_without_independent_model_outward_normals',
            'model_projection_or_id_used_in_measurement': False,
            'brightness_side_is_not_marking_identity_or_independent_validation': True,
            'config': asdict(config),
        })
    return profiles


def evaluate_outer_boundary_cue(profile, outward_normals_source_xy):
    """Compare measured brightness direction with external model outward side.

    The external normals must include the full projection/lens coordinate
    chain. A nearest curve or its residual does not enter this calculation.
    This is a gray-paint side cue, not proof of circle ID or correct camera.
    """
    normals = np.asarray(outward_normals_source_xy, np.float64)
    gradients = np.asarray(profile['gradient_unit_source_xy'], np.float64)
    if normals.shape != gradients.shape or not np.isfinite(normals).all():
        raise ValueError('One finite source-coordinate normal per original point required')
    normal_lengths = np.linalg.norm(normals, axis=1)
    usable = np.asarray(profile['gradient_reliable'], bool) & (normal_lengths > 1e-8)
    cosine = np.sum(normals / np.maximum(normal_lengths[:, None], 1e-8) * gradients, axis=1)
    n = int(usable.sum())
    positive = float(np.mean(cosine[usable] > .35)) if n else None
    negative = float(np.mean(cosine[usable] < -.35)) if n else None
    status = 'unknown'
    if n >= 5 and n >= .4 * len(normals):
        if positive >= .7 and negative <= .15:
            status = 'brightness_supports_outer_side_of_this_model_hypothesis'
        elif negative >= .7 and positive <= .15:
            status = 'brightness_supports_inner_side_of_this_model_hypothesis'
        else:
            status = 'mixed_brightness_side_cue'
    return {'status': status, 'reliable_point_count': n,
            'outward_alignment_cosine': cosine.tolist(), 'usable_points': usable.tolist(),
            'outward_positive_fraction': positive, 'outward_negative_fraction': negative,
            'median_outward_alignment': float(np.median(cosine[usable])) if n else None,
            'model_normals_supplied_by_caller': True, 'nearest_model_distance_used': False,
            'is_circle_identity_or_geometry_validation': False}
