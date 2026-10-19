"""Explicit uniform conversion of an unchanged scene to reference metres.

This module supplies units, never recalibrates geometry, cameras or motion.
Fit geometry first; a single coefficient then makes its spans closest to the
chosen reference size. Historical v1 scale records retain their old policy.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .storage import AssetStore

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCALE = ROOT/'configs/rink_model/ankara_2026_metric_scale_assumed_v1.json'


def file_sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def boundary_span(scene):
    xyz = np.asarray(scene['ice_boundary']['points_xyz'], dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
        raise ValueError('Finite declared XYZ ice boundary required')
    span = np.ptp(xyz[:, :2], axis=0)
    if np.any(span <= 0):
        raise ValueError('Ice boundary needs positive X and Y spans')
    return span


SQUARED_DIMENSION_ERROR = 'equal_weight_squared_dimension_error'
LEGACY_LOG_RELATIVE_ERROR = 'equal_weight_log_relative_dimension_error'


def estimate_uniform_scale(native_span_xy, design_span_m=(60., 30.), *,
                           policy=SQUARED_DIMENSION_ERROR):
    """Closest dimensions under the explicit policy, without axis stretching.

    Default minimises (s*L - reference_L)^2 + (s*W - reference_W)^2.
    The legacy policy exists only to read historical records unchanged.
    """
    native = np.asarray(native_span_xy, dtype=float)
    design = np.asarray(design_span_m, dtype=float)
    if (native.shape != (2,) or design.shape != (2,)
            or not np.isfinite(np.r_[native, design]).all()
            or np.any(native <= 0) or np.any(design <= 0)):
        raise ValueError('Two positive finite native/design dimensions required')
    if policy == SQUARED_DIMENSION_ERROR:
        return float(np.dot(native, design) / np.dot(native, native))
    if policy == LEGACY_LOG_RELATIVE_ERROR:
        return float(math.exp(float(np.mean(np.log(design/native)))))
    raise ValueError('Unknown uniform metric-scale objective')


def build_uniform_scale_manifest(scene_path, reference_source_path, *, scale_id,
                                 reference_dimensions_m=(60., 30.)):
    """Bind post-fit conversion to an unchanged scene and user requirement.

    Returns data only. This does not edit geometry, impose size priors, claim a
    measured venue size, or decide whether the scene aligns with real footage.
    """
    assets = AssetStore(ROOT)
    scene_path = assets.source(scene_path)
    reference_source_path = assets.source(reference_source_path)
    scene = json.loads(scene_path.read_text(encoding='utf8'))
    coordinate = scene['coordinate_system']
    if (coordinate['unit'] == 'm' or coordinate.get('metres_per_unit') is not None
            or coordinate.get('verified_metre_scale') is True):
        raise ValueError('Use a native unscaled scene; do not convert metres twice')
    span = boundary_span(scene)
    reference = np.asarray(reference_dimensions_m, float)
    scale = estimate_uniform_scale(span, reference)
    converted = span * scale
    return {
        'schema_version': 'rinkspace.assumed-uniform-metric-scale.v2',
        'scale_id': scale_id,
        'geometry_level': 'metric_assumed',
        'scale_interpretation': 'post_geometry_uniform_conversion_closest_to_reference',
        'unit': 'm',
        'native_unit': coordinate['unit'],
        'verified_metre_scale': False,
        'source_coordinate_system_id': coordinate['id'],
        'source_span_xy_scene_units': span.tolist(),
        'design_dimensions_m': reference.tolist(),
        'reference_dimensions_status': 'chosen_conversion_reference_not_hard_geometry_target',
        'reference_source': {
            'path': str(reference_source_path),
            'sha256': file_sha256(reference_source_path),
            'authority': 'current_user_conversation',
        },
        'scale_policy': SQUARED_DIMENSION_ERROR,
        'scale_formula': 's=(reference_length*L+reference_width*W)/(L*L+W*W)',
        'metres_per_scene_unit': scale,
        'conversion': 'XYZ_m = uniform_scale * XYZ_scene',
        'converted_span_xy_m': converted.tolist(),
        'reference_residual_xy_m': (converted - reference).tolist(),
        'reference_squared_dimension_error_m2': float(np.sum((converted - reference)**2)),
        'native_aspect_ratio': float(span[0] / span[1]),
        'metric_aspect_ratio': float(converted[0] / converted[1]),
        'force_two_to_one_aspect_ratio': False,
        'anisotropic_rescaling': False,
        'geometry_fit_uses_reference_size_constraint': False,
        'compatible_native_scenes': [{'path': str(scene_path),
                                      'sha256': file_sha256(scene_path)}],
        'solver_policy': {
            'post_fit_unit_conversion_only': True,
            'geometry_and_native_scene_file_unchanged': True,
            'camera_translation_scales_with_world': True,
            'intrinsics_rotation_Brown_and_PTS_unchanged': True,
        },
        'uncertainty': {
            'actual_venue_size_independently_measured': False,
            'geometry_estimation_uncertainty': None,
            'metres_per_scene_unit_statistical_uncertainty': None,
        },
        'geometry_accuracy_or_programme_alignment_promoted': False,
    }


@dataclass(frozen=True)
class AssumedMetricScale:
    metres_per_scene_unit: float
    manifest: dict

    def __post_init__(self):
        if not math.isfinite(self.metres_per_scene_unit) or self.metres_per_scene_unit <= 0:
            raise ValueError('Uniform metric scale must be finite and positive')
        if (self.manifest.get('geometry_level') != 'metric_assumed'
                or self.manifest.get('verified_metre_scale') is not False):
            raise ValueError('Design-derived metres must remain metric_assumed')

    @classmethod
    def load(cls, path=DEFAULT_SCALE, *, scene_path=None):
        assets = AssetStore(ROOT)
        record = json.loads(assets.source(path).read_text(encoding='utf8'))
        version = record.get('schema_version')
        if version not in ('rinkspace.assumed-uniform-metric-scale.v1',
                           'rinkspace.assumed-uniform-metric-scale.v2'):
            raise ValueError('Unknown uniform metric-scale schema')
        if record.get('conversion') != 'XYZ_m = uniform_scale * XYZ_scene':
            raise ValueError('Only a uniform XYZ conversion is allowed')
        native = np.asarray(record['source_span_xy_scene_units'], float)
        design = np.asarray(record['design_dimensions_m'], float)
        scale = float(record['metres_per_scene_unit'])
        policy = record.get('scale_policy')
        expected_policy = LEGACY_LOG_RELATIVE_ERROR if version.endswith('.v1') else SQUARED_DIMENSION_ERROR
        if policy != expected_policy:
            raise ValueError('Uniform metric-scale policy does not match its record version')
        if not math.isclose(scale, estimate_uniform_scale(native, design, policy=policy), rel_tol=1e-13):
            raise ValueError('Scale does not implement its declared dimension-error objective')
        if not np.allclose(record['converted_span_xy_m'], native*scale, rtol=1e-13, atol=0.):
            raise ValueError('Recorded metric dimensions do not preserve geometry')
        evidence = record['design_source'] if version.endswith('.v1') else record['reference_source']
        if file_sha256(assets.source(evidence['path'])) != evidence['sha256']:
            raise ValueError('User design-size provenance changed')
        result = cls(scale, record)
        if scene_path is not None:
            result.require_native_scene(scene_path)
        return result

    def require_native_scene(self, path):
        """Bind units to an actual scene; reject changed or already metric data."""
        path = AssetStore(ROOT).source(path)
        digest = file_sha256(path)
        if digest not in {r['sha256'] for r in self.manifest['compatible_native_scenes']}:
            raise ValueError('Scale is not bound to this scene; review or re-estimate')
        scene = json.loads(Path(path).read_text(encoding='utf8'))
        coordinate = scene['coordinate_system']
        if (coordinate['unit'] != self.manifest.get('native_unit', 'conditional_scene_units')
                or coordinate['id'] != self.manifest['source_coordinate_system_id']):
            raise ValueError('Cannot scale a different coordinate frame or metres twice')
        if not np.allclose(boundary_span(scene), self.manifest['source_span_xy_scene_units'], rtol=1e-13, atol=0.):
            raise ValueError('Native rink dimensions changed')
        return scene

    def quantity_to_metres(self, value, *, length_power=1):
        """s for XYZ/distance/speed/acceleration; s² for area; 1/s for curvature.

        This converts an existing quantity only. It preserves NaN gaps and
        supplies no contacts, times, derivatives or accuracy certification.
        """
        if length_power not in (-1, 0, 1, 2, 3):
            raise ValueError('Declare the quantity length dimension explicitly')
        return np.asarray(value, dtype=float)*self.metres_per_scene_unit**length_power

    def points_to_metres(self, xyz):
        value = np.asarray(xyz, dtype=float)
        if value.ndim == 0 or value.shape[-1] not in (2, 3):
            raise ValueError('Points must end in XY or XYZ')
        return self.quantity_to_metres(value)

    def points_to_scene(self, xy_or_xyz_m):
        value = np.asarray(xy_or_xyz_m, dtype=float)
        if value.ndim == 0 or value.shape[-1] not in (2, 3):
            raise ValueError('Points must end in XY or XYZ')
        return value/self.metres_per_scene_unit

    def camera25_to_metres(self, camera25):
        value = np.asarray(camera25, dtype=float).copy()
        if value.ndim == 0 or value.shape[-1] != 25:
            raise ValueError('Expected camera25 or an array of camera25')
        value[..., :3] *= self.metres_per_scene_unit
        return value

    def camera25_to_scene(self, camera25_m):
        value = np.asarray(camera25_m, dtype=float).copy()
        if value.ndim == 0 or value.shape[-1] != 25:
            raise ValueError('Expected metric camera25')
        value[..., :3] /= self.metres_per_scene_unit
        return value

    def homography_to_metres(self, native_from_image):
        """Change only an existing map's output units, not its geometry status."""
        matrix = np.asarray(native_from_image, dtype=float)
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
            raise ValueError('Finite 3x3 native-world-from-image map required')
        return np.diag([self.metres_per_scene_unit, self.metres_per_scene_unit, 1.]) @ matrix
