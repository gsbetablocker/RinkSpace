"""Load explicit active source factors while retaining the complete raw ledger.

Admission is source/provenance based.  This module never selects observations
using projected residuals, combines different reader UVs, or alters held rows.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def row_signature(row):
    """Exact physical role/time, finite geometry, native UV and original sigma."""
    return (int(row['pts']), row['id'], bool(row.get('point_correspondence')),
            np.asarray(row['uv'], np.float64).tobytes(),
            np.asarray(row['world'], np.float64).tobytes(),
            np.asarray(row['sigma'], np.float64).tobytes())


def active_fit_rows(packet, *, expected_scene_sha=None, expected_source_sha=None):
    """Return validated canonical fit rows, plus the original-row factor map.

    Legacy packets retain their exact multiplicity. New packets require a
    hash-bound record that explicitly accounts for every inactive raw fit row.
    """
    binding = packet.get('source_factor_admission')
    if binding is None:
        return packet['fit_rows'], dict(policy='legacy_exact_multiplicity',
            original_fit_rows=len(packet['fit_rows']), active_fit_factors=len(packet['fit_rows']))
    if digest(binding['path']) != binding['sha256']:
        raise ValueError('Source factor admission changed')
    record = json.loads(Path(binding['path']).read_text(encoding='utf8'))
    original_binding = record['original_authority_packet']
    if digest(original_binding['path']) != original_binding['sha256']:
        raise ValueError('Original source authority packet changed')
    original = json.loads(Path(original_binding['path']).read_text(encoding='utf8'))
    for role in ('fit_rows', 'held_rows', 'pending_rows'):
        if packet[role] != original[role]:
            raise ValueError('Source admission altered the original source rows or roles')
    if record['rows_selected_by_residual'] or record['camera_or_projection_used_for_admission']:
        raise ValueError('Source admission cannot be selected by projected error')
    for expected, field in ((expected_scene_sha, 'scene_sha256'), (expected_source_sha, 'source_sha256')):
        if expected is not None and record[field] != expected:
            raise ValueError('Source factor source/scene binding changed')
    parents = list(range(len(packet['fit_rows'])))
    reasons = {}
    for group in record['confirmed_same_measurement_duplicate_imports']:
        indices = group['raw_fit_row_indices']
        canonical = group['canonical_raw_fit_row_index']
        if canonical not in indices or group['basis'] != 'same_automatic_measurement_imported_twice':
            raise ValueError('Unsupported source duplicate admission')
        if not all(row_signature(packet['fit_rows'][i]) == row_signature(packet['fit_rows'][canonical]) for i in indices):
            raise ValueError('Duplicate grouping combined differing source UV/geometry/sigma')
        for evidence in group['provenance_inputs']:
            if digest(evidence['path']) != evidence['sha256']:
                raise ValueError('Duplicate original provenance changed')
        for i in indices:
            parents[i] = canonical
            if i != canonical:
                reasons[i] = 'confirmed_same_measurement_duplicate_import'
    for overlap in record['original_held_feature_overlaps_disabled_in_fit']:
        i, j = overlap['raw_fit_row_index'], overlap['raw_held_row_index']
        for evidence in overlap['provenance_inputs']:
            if digest(evidence['path']) != evidence['sha256']:
                raise ValueError('Original held-overlap provenance changed')
        if row_signature(packet['fit_rows'][i]) != row_signature(packet['held_rows'][j]):
            raise ValueError('Held overlap did not preserve the exact same source feature')
        if parents[i] != i:
            raise ValueError('Held overlap and duplicate admission collide')
        parents[i] = None
        reasons[i] = 'original_development_sampling_overlap_with_predeclared_held_feature'
    indices = [i for i, parent in enumerate(parents) if parent == i]
    selected = [packet['fit_rows'][i] for i in indices]
    lookup = {raw: factor for factor, raw in enumerate(indices)}
    mapping = [dict(raw_fit_row_index=i, canonical_raw_fit_row_index=parent,
                    active_factor_index=None if parent is None else lookup[parent],
                    factor_role='inactive_historical_fit' if parent is None else 'active_canonical' if parent == i else 'duplicate_of_active_canonical',
                    reason=reasons.get(i, 'original_active_source_row'))
               for i, parent in enumerate(parents)]
    if selected != packet['active_fit_rows'] or indices != record['active_fit_raw_indices']:
        raise ValueError('Explicit active factors differ from provenance admission')
    if mapping != packet['original_fit_row_to_factor'] or mapping != record['original_fit_row_to_factor']:
        raise ValueError('Original-to-canonical source factor map changed')
    return selected, dict(binding=binding, original_fit_rows=len(parents),
        active_fit_factors=len(selected), original_fit_row_to_factor=mapping,
        confirmed_import_duplicates=sum(v == 'confirmed_same_measurement_duplicate_import' for v in reasons.values()),
        held_feature_overlaps_disabled_in_fit=sum(v != 'confirmed_same_measurement_duplicate_import' for v in reasons.values()),
        original_held_rows=len(packet['held_rows']), original_pending_rows=len(packet['pending_rows']),
        historical_held_contamination=record['original_held_feature_overlaps_disabled_in_fit'])
