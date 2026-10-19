"""Keep source-reviewed false pixel samples in history, outside active FIT.

This is an explicit source qualification, never a residual-based weight. A
derived packet must preserve its complete parent packet and bind a separately
hashed review manifest to original rows, UV bytes and original RGB evidence.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np


SCHEMA = 'rinkspace.source-only-point-qualification.v1'
PACKET_FIELD = 'source_only_review_qualification'


def file_sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def original_uv_sha256(row):
    """Hash original UV in C-order, little-endian float64; shape stays in row."""
    return hashlib.sha256(np.ascontiguousarray(row['uv'], dtype='<f8').tobytes()).hexdigest()


def _reference(ref, owner, label):
    if not isinstance(ref, dict) or not {'path', 'sha256'} <= set(ref):
        raise ValueError(label + ' path and SHA256 required')
    path = Path(ref['path'])
    if not path.is_absolute():
        path = owner.parent / path
    path = path.resolve()
    if file_sha256(path) != ref['sha256']:
        raise ValueError(label + ' hash changed')
    return path


def _direct_user_annotation(row):
    # An identity confirmed from a photo is not automatically a user-drawn UV.
    # Actual drawing/registration provenance is protected even if an optional
    # convenience flag was omitted from that historical row.
    source = row.get('source')
    return any(row.get(key) for key in (
        'source_user_annotation', 'source_annotation_registration',
        'source_annotation_attachment', 'direct_user_annotation',
        'user_directly_annotated_pixels')) or (isinstance(source, dict) and source.get('kind') in (
            'digitized_user_stroke', 'direct_user_annotation', 'registered_user_pixels')
        )


def mask_scalar_rows(residual, matrix, indices):
    """Source-role zero rows, preserving scalar layout and caller buffers."""
    indices = np.asarray(indices)
    if (indices.ndim != 1 or (indices.size and indices.dtype.kind not in 'iu')
            or np.any(indices < 0) or np.any(indices >= len(residual))
            or len(np.unique(indices)) != len(indices)
            or (matrix is not None and matrix.shape[0] != len(residual))):
        raise ValueError('Unique original scalar slots and unchanged source row layout required')
    if not len(indices):
        return residual, matrix
    result = np.asarray(residual).copy()
    result[indices] = 0.
    if matrix is not None:
        matrix = matrix.tocsr(copy=True)
        for index in indices:
            matrix.data[matrix.indptr[index]:matrix.indptr[index + 1]] = 0.
        matrix.eliminate_zeros()
    return result, matrix


class SourcePointQualification:
    def __init__(self, derived_path, packet, *, source_family):
        derived_path = Path(derived_path).resolve()
        manifest_path = _reference(packet[PACKET_FIELD], derived_path, 'Source qualification manifest')
        manifest = json.loads(manifest_path.read_text(encoding='utf8'))
        if manifest.get('schema') != SCHEMA or manifest.get('source_family') != source_family:
            raise ValueError('Explicit same-family source-only point qualification required')
        parent_ref = manifest.get('original_packet')
        parent_path = _reference(parent_ref, manifest_path, 'Original source packet')
        parent = json.loads(parent_path.read_text(encoding='utf8'))
        if PACKET_FIELD in parent:
            raise ValueError('Qualification must bind the immutable original packet, not another mask')
        preserved = {key: value for key, value in packet.items() if key != PACKET_FIELD}
        if preserved != parent:
            raise ValueError('Qualification changed original packet UV, sigma, split, source or history')
        records = manifest.get('records')
        if not isinstance(records, list):
            raise ValueError('Explicit reviewed source row records required')
        self.records = {}
        self.original_rows = {}
        for record in records:
            index = record.get('row_index')
            if type(index) is not int or not 0 <= index < len(parent['rows']):
                raise ValueError('Original source row index outside immutable packet')
            row = parent['rows'][index]
            identity = row['id']
            if identity in self.records or record.get('row_id') != identity:
                raise ValueError('Original source row ID changed or qualification duplicated')
            if row.get('split') != 'fit':
                raise ValueError('Only original FIT samples may be excluded; HELD roles stay unchanged')
            if _direct_user_annotation(row):
                raise ValueError('Direct user annotation cannot be excluded by source qualification')
            if (type(record.get('source_pts')) is not int
                    or record['source_pts'] != int(row['pts'])
                    or record.get('original_uv_float64_sha256') != original_uv_sha256(row)):
                raise ValueError('Original source PTS or UV hash changed')
            indices = record.get('excluded_point_indices')
            if (not isinstance(indices, list) or not indices
                    or any(type(i) is not int or not 0 <= i < len(row['uv']) for i in indices)
                    or len(set(indices)) != len(indices)):
                raise ValueError('Unique in-range original point indices required')
            if (record.get('review_basis') != 'original_rgb_only'
                    or not isinstance(record.get('reason'), str) or not record['reason'].strip()):
                raise ValueError('Source-only original RGB review and explicit reason required')
            evidence = record.get('native_rgb_evidence')
            if not isinstance(evidence, list) or not evidence:
                raise ValueError('Original native RGB evidence required')
            original_path = row.get('source_raw_frame', row.get('source_path'))
            original_hash = row.get('source_file_sha256', row.get('source_frame_sha256'))
            if not original_path or not original_hash:
                raise ValueError('Original source row RGB provenance absent')
            exact_native = False
            for ref in evidence:
                path = _reference(ref, manifest_path, 'Native RGB evidence')
                if (type(ref.get('source_pts')) is not int):
                    raise ValueError('Native RGB evidence needs exact integer source PTS')
                exact_native |= (ref['source_pts'] == int(row['pts'])
                    and path == Path(original_path).resolve() and ref['sha256'] == original_hash)
            if not exact_native:
                raise ValueError('Review must include this row exact original native RGB and SHA256')
            self.records[identity] = deepcopy(record)
            self.original_rows[identity] = deepcopy(row)
        self.contract = dict(schema=SCHEMA, source_family=source_family,
            manifest_path=str(manifest_path), manifest_sha256=file_sha256(manifest_path),
            implementation_source=dict(path=str(Path(__file__).resolve()), sha256=file_sha256(__file__)),
            original_packet=dict(path=str(parent_path), sha256=parent_ref['sha256']),
            reviewed_records=deepcopy(records), source_UV_sigma_split_and_history_unchanged=True,
            scalar_slots_removed=0, new_FIT_samples_added=0,
            exclusion_is_source_identity_not_cost_weighting=True,
            inactive_zero_residual_is_accuracy_measurement=False,
            excluded_points_not_in_active_FIT_accuracy_statistics=True,
            direct_user_annotation_and_HELD_unchanged=True,
            fresh_source_profiles_and_J_required=True, accuracy_accepted=False)

    def point_mask(self, row):
        mask = np.ones(len(row['uv']), bool)
        record = self.records.get(row['id'])
        if record is not None:
            if (row['split'] != 'fit' or int(row['pts']) != record['source_pts']
                    or original_uv_sha256(row) != record['original_uv_float64_sha256']):
                raise ValueError('Runtime source row identity or UV changed after qualification')
            mask[record['excluded_point_indices']] = False
        return mask

    def scalar_mask(self, rows):
        return np.concatenate([np.repeat(self.point_mask(row), 2) for row in rows])

    def mask_values_and_jacobian(self, rows, residual, matrix):
        mask = self.scalar_mask(rows)
        if np.shape(residual) != mask.shape or (matrix is not None and matrix.shape[0] != len(mask)):
            raise ValueError('Original source scalar row layout changed')
        residual, matrix = mask_scalar_rows(residual, matrix, np.flatnonzero(~mask))
        return residual, matrix, np.flatnonzero(~mask)

    def review_metrics(self, row, distance):
        mask = self.point_mask(row)
        distance = np.asarray(distance)[mask]
        return dict(source_uv_count=len(row['uv']), active_FIT_accuracy_uv_count=int(mask.sum()),
            excluded_point_indices=np.flatnonzero(~mask).tolist(),
            included_accuracy_point_indices=np.flatnonzero(mask).tolist(),
            mean_native_px=float(distance.mean()) if len(distance) else None,
            p90_native_px=float(np.percentile(distance, 90)) if len(distance) else None,
            max_native_px=float(distance.max()) if len(distance) else None,
            inactive_zero_residual_is_accuracy_measurement=False)


def load_source_point_qualification(packet_path, packet, *, source_family):
    """No manifest means no qualification object and no numerical path change."""
    if PACKET_FIELD not in packet:
        return None
    return SourcePointQualification(packet_path, packet, source_family=source_family)
