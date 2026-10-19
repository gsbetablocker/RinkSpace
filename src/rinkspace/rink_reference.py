"""Small model/frame bridge for the model-first integration, not a rink solver.

Reuses existing geometry.py. Nothing in this module estimates a real landmark,
calibrates a camera, undistorts pixels, segments a person or validates accuracy.
Legacy sample transforms are converted explicitly; grids remain model-bound.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Mapping
import re

import numpy as np
from .geometry import homography


@dataclass(frozen=True)
class RinkReference:
    """Minimal identity/units view of a full source-backed rink model.

    Evidence entries are nonempty reference IDs, not evidence verification.
    The full model stores the cited observations separately.
    """
    model_id: str
    revision: str
    geometry_level: str
    unit: str
    coordinate_definition: str
    geometry_evidence: tuple[str, ...] = ()
    scale_evidence: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ('model_id', 'revision', 'coordinate_definition'):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f'{name} must be a nonempty string')
        expected = {'projective': 'reference_px', 'euclidean_unscaled': 'arbitrary_unit',
                    'metric_assumed': 'm', 'metric_verified': 'm'}
        if self.geometry_level not in expected or self.unit != expected[self.geometry_level]:
            raise ValueError('geometry_level and unit disagree')
        for refs in (self.geometry_evidence, self.scale_evidence):
            if not isinstance(refs, (tuple, list)) or any(not isinstance(r, str) or not r.strip() for r in refs):
                raise ValueError('Evidence must be a sequence of reference IDs')
        if self.geometry_level != 'projective' and not self.geometry_evidence:
            raise ValueError('Euclidean geometry needs evidence references')
        if self.geometry_level.startswith('metric_') and not self.scale_evidence:
            raise ValueError('Metres need scale evidence or a declared assumption reference')

    def grid_label(self, kind: str = 'coordinate_net') -> str:
        if kind not in ('coordinate_net', 'equal_spacing', 'metre'):
            raise ValueError('Unknown grid kind')
        if kind == 'equal_spacing' and self.geometry_level == 'projective':
            raise ValueError('A projective coordinate net is not an equal-distance grid')
        if kind == 'metre' and not self.geometry_level.startswith('metric_'):
            raise ValueError('Metre grid requires metric geometry')
        if self.geometry_level == 'projective':
            return '位置坐标网（非等距，非米制）'
        if self.geometry_level == 'euclidean_unscaled':
            return '等距网格（任意单位，非米制）'
        if self.geometry_level == 'metric_assumed':
            return '米制网格（尺度为条件假设）'
        return '米制网格（尺度证据另查；不表示定位精度已通过）'


def compose_model_from_image(
    h_model_from_anchor_image: Any,
    h_anchor_sample_from_current_sample: Any,
    sample_from_current_image: Any,
    sample_from_anchor_image: Any,
) -> np.ndarray:
    """C_raw @ inv(S_anchor) @ L @ S_current, in one declared lens domain.

    These can be crop/resize matrices, not just scalar scales. Caller must
    verify anchor/component identities and consistent distortion coordinates.
    """
    c = homography(h_model_from_anchor_image)
    l = homography(h_anchor_sample_from_current_sample)
    st = homography(sample_from_current_image)
    sa = homography(sample_from_anchor_image)
    return homography(c @ np.linalg.inv(sa) @ l @ st)


def validate_frame_map(record: Mapping[str, Any], model: RinkReference) -> dict[str, Any]:
    """Validate identity/coordinate contracts, never semantic or accuracy claims."""
    if record.get('model_id') != model.model_id or record.get('model_revision') != model.revision:
        raise ValueError('Frame map belongs to another model or revision')
    digest = record.get('source_sha256', '')
    if not isinstance(digest, str) or re.fullmatch(r'[0-9a-fA-F]{64}', digest) is None:
        raise ValueError('A real source_sha256 reference is required')
    pts = record.get('pts')
    if isinstance(pts, bool) or not isinstance(pts, int):
        raise ValueError('pts must be an integer source timestamp')
    try:
        tb = Fraction(record['time_base'])
    except (ValueError, ZeroDivisionError, TypeError, KeyError) as exc:
        raise ValueError('time_base must be a positive rational') from exc
    if tb <= 0:
        raise ValueError('time_base must be positive')
    if record.get('matrix_direction') != 'H_model_from_image':
        raise ValueError('Explicit H_model_from_image direction is required')
    if record.get('image_domain') not in ('undistorted_px', 'raw_px_distortion_unchecked'):
        raise ValueError('Declare the image lens domain')
    status = record.get('status')
    if status not in ('observed', 'predicted', 'candidate', 'unavailable', 'ambiguous'):
        raise ValueError('Unknown localization status')
    value = record.get('H_model_from_image')
    if status in ('unavailable', 'ambiguous'):
        if value is not None:
            raise ValueError('Unavailable/ambiguous output cannot hide an accepted transform')
        h = None
    else:
        if not isinstance(record.get('anchor_id'), str) or not record['anchor_id'].strip():
            raise ValueError('A global model anchor is required; a local identity is not enough')
        if status == 'predicted' and not record.get('propagated_from'):
            raise ValueError('Predicted output must name propagation provenance')
        h = homography(value)
    return {'H_model_from_image': h, 'source_time': pts * tb, 'status': status,
            'semantic_validation_performed': False, 'accuracy_validation_performed': False}


def _clip_viewport(a: np.ndarray, b: np.ndarray, width: int, height: int):
    """Liang-Barsky clipping in image pixels, not an ice-region mask."""
    # Homography inversion can put an exact boundary at -1e-14. Snap only
    # sub-nanopixel roundoff; do not discard a valid line on the image border.
    a, b = a.copy(), b.copy()
    for endpoint in (a, b):
        for axis, limit in enumerate((width, height)):
            if abs(endpoint[axis]) < 1e-9:
                endpoint[axis] = 0.0
            elif abs(endpoint[axis] - (limit - 1)) < 1e-9:
                endpoint[axis] = float(limit - 1)
    d = b - a
    lo, hi = 0.0, 1.0
    for p, q in ((-d[0], a[0]), (d[0], width - 1 - a[0]),
                 (-d[1], a[1]), (d[1], height - 1 - a[1])):
        if abs(p) < 1e-14:
            if q < 0:
                return None
            continue
        r = q / p
        if p < 0:
            lo = max(lo, r)
        else:
            hi = min(hi, r)
        if lo > hi:
            return None
    return np.stack([a + lo * d, a + hi * d])


def project_model_segments(
    model: RinkReference, record: Mapping[str, Any], segments_model: Any,
    image_size: tuple[int, int], *, grid_kind: str = 'coordinate_net',
    allow_approximate_raw: bool = False,
) -> dict[str, Any]:
    """Project model segments to the original-size image coordinate domain.

    Returns line segments, NOT a rendered or verified overlay. The application
    must mask real ice, foreground occlusion and unsupported extrapolation.
    Raw lens distortion is not implemented: callers must explicitly opt into
    the approximation, or supply undistorted coordinates and redistort later.
    Horizon-crossing segments are rejected as a whole, never connected across
    infinity. This conservative helper is not a full projective mesh renderer.
    """
    if (len(image_size) != 2 or any(isinstance(v, bool) or not isinstance(v, int)
                                  or v <= 0 for v in image_size)):
        raise ValueError('image_size must be two positive integer dimensions')
    state = validate_frame_map(record, model)
    label = model.grid_label(grid_kind)
    segments = np.asarray(segments_model, dtype=np.float64)
    if segments.size == 0:
        segments = np.empty((0, 2, 2))
    if segments.ndim != 3 or segments.shape[1:] != (2, 2) or not np.isfinite(segments).all():
        raise ValueError('Expected finite N x 2 x 2 model segments')
    result = {'segments_image': [], 'grid_label': label, 'status': state['status'],
              'model_id': model.model_id, 'model_revision': model.revision,
              'rejected_horizon_count': 0, 'rejected_viewport_count': 0,
              'requires_ice_support_and_foreground_clipping': True,
              'independent_accuracy_validated': False,
              'lens_distortion_approximation': record['image_domain'] == 'raw_px_distortion_unchecked'}
    if state['H_model_from_image'] is None:
        return result
    if result['lens_distortion_approximation'] and not allow_approximate_raw:
        raise ValueError('Raw lens distortion is unchecked; opt in to approximation explicitly')
    h = homography(np.linalg.inv(state['H_model_from_image']))
    for segment in segments:
        q = np.c_[segment, np.ones(2)] @ h.T
        if np.any(np.abs(q[:, 2]) < 1e-10) or np.sign(q[0, 2]) != np.sign(q[1, 2]):
            result['rejected_horizon_count'] += 1
            continue
        xy = q[:, :2] / q[:, 2:3]
        clipped = _clip_viewport(xy[0], xy[1], *image_size)
        if clipped is None:
            result['rejected_viewport_count'] += 1
        else:
            result['segments_image'].append(clipped.tolist())
    return result


def check_source_group_split(model_source_groups: list[str], heldout_source_groups: list[str]) -> None:
    """Reject known duplicate-broadcast leakage; does not discover duplicates."""
    if not all(isinstance(g, str) and g.strip() for g in model_source_groups + heldout_source_groups):
        raise ValueError('Source group IDs must be nonempty strings')
    overlap = set(model_source_groups) & set(heldout_source_groups)
    if overlap:
        raise ValueError(f'Broadcast-image groups leak into holdout: {sorted(overlap)}')
