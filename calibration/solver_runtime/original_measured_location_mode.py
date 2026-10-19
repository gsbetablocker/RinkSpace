"""Typed authority for measured positions in existing absolute source rows.

The installer qualifies the original RGB record and calls this constructor.
Evaluators retain the existing raw finite/typed residual and camera Jacobian;
only the optical footprint dead zone is bypassed for authenticated locations.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

MODE_FIELD = 'original_boundary_location_observation'
AUTHORITY_FIELD = 'authenticated_measured_location'
MODE = 'measured_transition_location'
SCHEMA = 'rinkspace.authenticated-original-measured-location-mode.v1'
_ARRAY_FIELDS = ('uv', 'sigma', 'normal', 'normal_interval', 'source_uv',
    'normal_interval_px', 'point_support_bounds_xy_px', 'source_tangent_interval_px')
_ATTRIBUTE_FIELDS = ('id', 'pts', 'split', 'point_correspondence', 'curve_correspondence',
    'curve_observation_kind', 'must_preserve_upper_branch', 'scanline_fixed_axis',
    'scanline_branch', 'extent_side', 'clip_boundary_x', 'clip_blur_allowance_px',
    'source_uv_is_image_boundary_context_only', 'original_outer_source_qualification_operator')


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
        allow_nan=False, default=lambda x: x.tolist() if isinstance(x, np.ndarray)
        else x.item() if isinstance(x, np.generic) else _unsupported(x)).encode('utf8')


def _unsupported(value):
    raise TypeError('Measured location authority requires explicit JSON, not ' + type(value).__name__)


@dataclass(frozen=True)
class _ObservedArray:
    field: str
    dtype: str
    shape: tuple
    data: bytes

    def array(self):
        return np.frombuffer(self.data, dtype=self.dtype).reshape(self.shape)

    def identity(self):
        return dict(field=self.field, dtype=self.dtype, shape=list(self.shape),
            sha256=hashlib.sha256(self.data).hexdigest())


@dataclass(frozen=True)
class AuthenticatedMeasuredLocation:
    """Pickle-safe immutable certificate; a JSON dict is never sufficient."""
    authority_path: str
    authority_sha256: str
    source_row_index: int
    qualified_receipt_record_sha256: str
    original_point_count: int
    observed_arrays: tuple
    original_attributes_json: bytes

    def source_identity(self):
        return dict(schema=SCHEMA,
            authority_reference=dict(path=self.authority_path, sha256=self.authority_sha256),
            original_source_row_index=self.source_row_index,
            qualified_receipt_record_sha256=self.qualified_receipt_record_sha256,
            original_point_count=self.original_point_count,
            observed_fields=[field.identity() for field in self.observed_arrays],
            original_operator_attributes=json.loads(self.original_attributes_json))

    def validate_runtime_row(self, row):
        attributes = {k: {'present': k in row, 'value': row.get(k)} for k in _ATTRIBUTE_FIELDS}
        if _canonical(attributes) != self.original_attributes_json:
            raise ValueError('Authenticated measured location original role or typed operator changed')
        mapping = row.get('active_original_absolute_input_compaction')
        if mapping is None:
            keep = np.arange(self.original_point_count)
        else:
            keep = np.asarray(mapping.get('retained_original_point_indices'), dtype=np.int64)
            if (mapping.get('original_point_count') != self.original_point_count
                    or keep.ndim != 1 or not len(keep) or np.any(keep < 0)
                    or np.any(keep >= self.original_point_count)
                    or len(keep) > 1 and np.any(np.diff(keep) <= 0)):
                raise ValueError('Measured location original point compaction identity differs')
        present = {k for k in _ARRAY_FIELDS if k in row}
        if present != {snapshot.field for snapshot in self.observed_arrays}:
            raise ValueError('Authenticated measured location original observed fields changed')
        for snapshot in self.observed_arrays:
            original = snapshot.array()
            expected = original[keep] if original.ndim and original.shape[0] == self.original_point_count else original
            actual = np.asarray(row[snapshot.field])
            if (actual.dtype.str != expected.dtype.str or actual.shape != expected.shape
                    or np.ascontiguousarray(actual).tobytes() != np.ascontiguousarray(expected).tobytes()):
                raise ValueError('Authenticated measured location UV/sigma/normal/support changed: ' + snapshot.field)


def install_measured_location_mode(row, *, authority_ref, source_row_index, qualified_receipt_record):
    """Return a copied row after the caller's complete source qualification."""
    if MODE_FIELD in row or AUTHORITY_FIELD in row:
        raise ValueError('An existing original location mode may not be replaced')
    if 'normal_interval' not in row:
        raise ValueError('Measured location mode requires an original optical normal interval')
    if row.get('curve_observation_kind') == 'confirmed_lower_long_side_top_horizontal_support':
        raise ValueError('Typed nearboard horizontal intervals require their own measured-axis qualification')
    if type(source_row_index) is not int or source_row_index < 0:
        raise ValueError('Exact nonnegative original source row index required')
    path = Path(authority_ref['path']).resolve()
    if hashlib.sha256(path.read_bytes()).hexdigest() != authority_ref['sha256']:
        raise ValueError('Measured location qualification authority bytes changed')
    if not isinstance(qualified_receipt_record, dict) or not qualified_receipt_record:
        raise ValueError('Complete qualified original RGB receipt record required')
    uv = np.asarray(row['uv'])
    if uv.ndim != 2 or uv.shape[1:] != (2,) or not len(uv) or not np.isfinite(uv).all():
        raise ValueError('Original finite measured source UV required')
    for key in ('sigma', 'normal', 'normal_interval'):
        value = np.asarray(row[key])
        if (value.shape != ((len(uv),) if key == 'sigma' else uv.shape)
                or not np.isfinite(value).all() or key == 'sigma' and np.any(value <= 0)):
            raise ValueError('Original measured source sigma/normal/support layout required')
    snapshots = []
    for key in _ARRAY_FIELDS:
        if key not in row:
            continue
        value = np.asarray(row[key])
        if value.dtype.hasobject or not np.isfinite(value).all():
            raise ValueError('Explicit finite numeric observed source field required: ' + key)
        snapshots.append(_ObservedArray(key, value.dtype.str, value.shape,
                                        np.ascontiguousarray(value).tobytes()))
    attributes = {k: {'present': k in row, 'value': row.get(k)} for k in _ATTRIBUTE_FIELDS}
    certificate = AuthenticatedMeasuredLocation(str(path), authority_ref['sha256'], source_row_index,
        hashlib.sha256(_canonical(qualified_receipt_record)).hexdigest(), len(uv), tuple(snapshots),
        _canonical(attributes))
    result = dict(row)
    result[MODE_FIELD] = MODE
    result[AUTHORITY_FIELD] = certificate
    return result


def uses_original_normal_support(row):
    """Old rows remain untouched; naked new mode metadata fails explicitly."""
    mode = row.get(MODE_FIELD)
    certificate = row.get(AUTHORITY_FIELD)
    if mode is None:
        if certificate is not None:
            raise ValueError('Measured location authority requires its explicit observation mode')
        return True
    if mode != MODE or type(certificate) is not AuthenticatedMeasuredLocation:
        raise ValueError('Measured location mode requires its authenticated installer certificate')
    certificate.validate_runtime_row(row)
    return False
