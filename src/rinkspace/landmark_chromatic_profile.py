"""Measured low-saturation ink colour relative to neighbouring actual ice.

The result is colour evidence, never a rink identity or a projected contour.
A neutral or weak response does not prove that a desaturated mark is gray.
"""
from dataclasses import asdict, dataclass

import cv2
import numpy as np

from .landmark_band_profile import point_array_sha256


ALGORITHM_VERSION = 'source-relative-chromatic-profile-v1'


@dataclass(frozen=True)
class ChromaticProfileConfig:
    gradient_sigma_source_px: float = 3.
    minimum_gradient: float = .15
    minimum_luminance_contrast: float = 4.
    minimum_blue_excess_levels: float = 3.
    minimum_blue_fraction: float = .65
    minimum_reliable_fraction: float = .4
    minimum_reliable_points: int = 8


def relative_chromatic_excess(dark_bgr, reference_bgr):
    """Remove the local ice colour cast under a neutral-darkening hypothesis."""
    dark = np.asarray(dark_bgr, float)
    reference = np.asarray(reference_bgr, float)
    scale = dark.mean(axis=-1) / np.maximum(reference.mean(axis=-1), 1.)
    br = dark[..., 0] - dark[..., 2] - scale * (reference[..., 0] - reference[..., 2])
    bg = dark[..., 0] - dark[..., 1] - scale * (reference[..., 0] - reference[..., 1])
    return np.stack((br, bg), axis=-1)


def measure_chromatic_profiles(image, observations, *, config=None):
    """Read a short native-pixel colour profile at every frozen observed point."""
    config = config or ChromaticProfileConfig()
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError('Native uint8 BGR source image required')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    smooth = cv2.GaussianBlur(gray, (0, 0), config.gradient_sigma_source_px)
    gx = cv2.Sobel(smooth, -1, 1, 0, ksize=3) / 8.
    gy = cv2.Sobel(smooth, -1, 0, 1, ksize=3) / 8.
    offsets = np.asarray([-20., -12., -8., -4., 0., 4., 8., 12., 20.])
    profiles = []
    h, w = image.shape[:2]
    for observation in observations:
        points = np.asarray(observation['points_source_px'], float).reshape(-1, 2)
        if not len(points) or not np.isfinite(points).all():
            raise ValueError('Finite observed source points required')
        xy = np.rint(points).astype(int)
        if ((xy[:, 0] < 0) | (xy[:, 0] >= w) | (xy[:, 1] < 0) | (xy[:, 1] >= h)).any():
            raise ValueError('Observed points outside source image')
        gradients = np.c_[gx[xy[:, 1], xy[:, 0]], gy[xy[:, 1], xy[:, 0]]]
        magnitude = np.linalg.norm(gradients, axis=1)
        unit = gradients / np.maximum(magnitude[:, None], 1e-6)
        locations = points[:, None, :] + unit[:, None, :] * offsets[None, :, None]
        mx, my = locations.transpose(2, 0, 1).astype(np.float32)
        colours = cv2.remap(image.astype(np.float32), mx, my, cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REFLECT101)
        luminance = colours.mean(axis=2)
        dark_index, bright_index = luminance.argmin(axis=1), luminance.argmax(axis=1)
        index = np.arange(len(points))
        dark, bright = colours[index, dark_index], colours[index, bright_index]
        contrast = bright.mean(axis=1) - dark.mean(axis=1)
        inside = ((locations[..., 0] >= 0) & (locations[..., 0] <= w - 1)
                  & (locations[..., 1] >= 0) & (locations[..., 1] <= h - 1)).all(axis=1)
        reliable = inside & (magnitude >= config.minimum_gradient) & (contrast >= config.minimum_luminance_contrast)
        if observation.get('association_status') == 'masked_out':
            reliable[:] = False
        excess = relative_chromatic_excess(dark, bright)
        count = int(reliable.sum())
        sufficient = count >= config.minimum_reliable_points and float(reliable.mean()) >= config.minimum_reliable_fraction
        # Low positive excess can also arise on nominally gray marks. Only a
        # repeated excess above quantization-scale noise is positive support.
        blue_fraction = float(np.mean(excess[reliable, 0] > config.minimum_blue_excess_levels)) if count else None
        median = np.median(excess[reliable], axis=0).tolist() if count else None
        blue = bool(sufficient and blue_fraction >= config.minimum_blue_fraction)
        profiles.append({
            'observation_id': observation['observation_id'],
            'point_array_sha256': point_array_sha256(points),
            'point_count': len(points), 'profile_offsets_source_px': offsets.tolist(),
            'gradient_unit_source_xy': np.round(unit, 4).tolist(),
            'dark_sample_offset_source_px': offsets[dark_index].tolist(),
            'reference_sample_offset_source_px': offsets[bright_index].tolist(),
            'dark_bgr': np.round(dark, 2).tolist(), 'local_reference_bgr': np.round(bright, 2).tolist(),
            'luminance_contrast_levels': np.round(contrast, 3).tolist(),
            'chromatic_excess_br_bg_levels': np.round(excess, 3).tolist(),
            'reliable_flags': reliable.tolist(), 'reliable_point_count': count,
            'median_chromatic_excess_br_bg_levels': median,
            'blue_excess_point_fraction': blue_fraction,
            'colour_evidence': 'blue_ink_supported' if blue else 'unknown',
            'absence_of_blue_support_is_not_gray_identity': True,
            'observed_pixels_or_model_points_changed': False,
            'model_camera_or_landmark_identity_used': False,
            'config': asdict(config),
        })
    return profiles
