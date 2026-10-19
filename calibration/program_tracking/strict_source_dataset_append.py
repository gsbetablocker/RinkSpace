"""Append source-reviewed deltas to a frozen index view without pixel copies."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from .compact_source_header_inventory import read_minimal_compact_source_headers
from .strict_absolute_source import StrictAbsoluteSourcePack
from .strict_model_scope import authenticate_source_reference
from .strict_source_fit_dataset import SCHEMA, partition_support, reference


def append_reviewed_deltas(base_reference, delta_references, audit_reference, output_directory):
    """Require literal zero-duplicate/operator-conflict preflight for all packs.

    Existing support and pending facts retain their immutable references. New
    source arrays are validated by the public consumer; only small indices are
    written. Deltas with HELD require a complete protection rebuild, not this
    append route. A delta cannot cross existing feature/material HELD support.
    """
    for ref in (base_reference, audit_reference, *delta_references):
        authenticate_source_reference(ref)
    base = json.loads(Path(base_reference['path']).read_text(encoding='utf8'))
    audit = json.loads(Path(audit_reference['path']).read_text(encoding='utf8'))
    if base.get('schema') != SCHEMA or not delta_references:
        raise ValueError('A frozen source dataset and explicit reviewed deltas are required')
    selected = [c['source_pack'] for c in base['cuts']] + list(delta_references)
    key = lambda r: (str(Path(r['path']).resolve()), r['sha256'])
    wanted = {key(ref) for ref in selected}
    audited = {key(item['source_reference']) for record in audit['records'] for item in record['inputs']}
    if (audit.get('schema') != 'rinkspace.source-input-cross-pack-exact-pixel-operator-audit.v1'
            or len(wanted) != len(selected) or wanted != audited
            or any(v for k, v in audit['counts'].items() if 'duplicate' in k or 'conflict' in k)
            or any(r.get('operator_conflicts') for r in audit['records'])
            or audit['points'] != audit['unique_exact_pixel_target_count']):
        raise ValueError('Complete literal source preflight must prove no duplicates or operator conflicts')
    output = Path(output_directory).resolve()
    if (output / 'fit_dataset.json').exists():
        raise FileExistsError('Preserve earlier dataset revisions')
    output.mkdir(parents=True, exist_ok=True)
    coverage = base['original_native_coverage_reference']
    authenticate_source_reference(coverage)
    with np.load(coverage['path'], allow_pickle=False) as stored:
        ids = np.flatnonzero(stored['representative'])
        bindings = {int(stored['actual_pts'][i]): (int(stored['native_pts'][i]),
            bytes(stored['BGR_sha256'][i]).decode('ascii')) for i in ids}
    if set(bindings) != set(base['expected_actual_pts']):
        raise ValueError('Original full native/canonical exposure catalogue changed')
    exposures, held_features, held_materials = set(), set(), set()
    for cut in base['cuts']:
        for ref in (cut['source_pack'], cut['support_index']):
            authenticate_source_reference(ref)
        headers, offsets = read_minimal_compact_source_headers(cut['source_pack']['path'])
        with np.load(cut['support_index']['path'], allow_pickle=False) as saved:
            selected_fit, selected_held = saved['fit_point_indices'], saved['held_point_indices']
            if not np.array_equal(saved['row_offsets'], offsets):
                raise ValueError('Original row boundaries changed')
        for ri, h in enumerate(headers):
            lo, hi = offsets[ri:ri + 2]
            first, last = np.searchsorted(selected_fit, [lo, hi])
            if last > first:
                exposures.add(h['actual_pts'])
            first, last = np.searchsorted(selected_held, [lo, hi])
            if last > first:
                held_features.add((h['actual_pts'], h['feature_id']))
                held_materials.add((h['actual_pts'], h['physical_material_id']))
    if len(exposures) != base['actual_exposures_with_finite_fit']:
        raise ValueError('Frozen FIT exposure coverage changed')
    result = deepcopy(base)
    materials = Counter(base['finite_fit_points_by_material'])
    appended = []
    for number, ref in enumerate(delta_references, start=len(base['cuts'])):
        pack = StrictAbsoluteSourcePack(ref['path'], expected_sha256=ref['sha256'])
        if (pack.metadata['source'] != base['source']
                or not set(pack.metadata['expected_actual_pts']).issubset(bindings)
                or not pack.metadata['all_exposures_processed']
                or not pack.metadata['all_exposures_exact_native_BGR_bound']):
            raise ValueError('Reviewed delta belongs to another source or exposure clock')
        if pack.held_rows:
            raise ValueError('New HELD evidence requires a complete source-protection rebuild')
        for stamp, binding in pack.frame_bindings.items():
            if bindings.get(stamp) != (binding['native_pts'], binding['decoded_BGR_sha256']):
                raise ValueError('Reviewed delta changed the original exposure or decoded pixels')
        for h in pack.rows:
            if ((h['actual_pts'], h['feature_id']) in held_features or
                    (h['actual_pts'], h['physical_material_id']) in held_materials):
                raise ValueError('New FIT would cross a protected material/feature exposure')
        offsets = pack.values['row_offsets']
        partition = partition_support(pack.rows, offsets, pack.values['sigma_px'])
        partition['pending_named_point_indices'] = np.array([], np.int64)
        path = output / f'source_{number:02d}_support_indices.npz'
        if path.exists():
            raise FileExistsError('Preserve earlier source index')
        np.savez_compressed(path, row_offsets=offsets, promoted_held_rows=np.array([], np.int64), **partition)
        cut = dict(source_pack=ref, support_index=reference(path), pending_filter_references=[],
            source_rows=len(pack.rows), source_points=len(pack.values['sigma_px']),
            fit_points=len(partition['fit_point_indices']), held_points=0,
            pending_support_points=len(partition['pending_support_point_indices']), pending_named_points=0,
            held_protected_rows_promoted=0)
        for field in ('fit_points', 'held_points', 'pending_support_points', 'pending_named_points'):
            result[field] += cut[field]
        selection = partition['fit_point_indices']
        for ri, h in enumerate(pack.rows):
            lo, hi = offsets[ri:ri + 2]
            first, last = np.searchsorted(selection, [lo, hi])
            if last > first:
                exposures.add(h['actual_pts'])
                materials[h['physical_material_id']] += int(last - first)
        result['cuts'].append(cut)
        appended.append(cut)
        del pack
    result.update(finite_fit_points_by_material=dict(sorted(materials.items())),
        actual_exposures_with_finite_fit=len(exposures),
        actual_exposures_without_finite_fit=len(bindings) - len(exposures),
        previous_frozen_dataset_reference=base_reference,
        exact_duplicate_operator_audit_reference=audit_reference)
    result['source_review_authorities'] = [*base['source_review_authorities'], audit_reference, reference(__file__)]
    manifest = output / 'fit_dataset.json'
    with manifest.open('x', encoding='utf8') as stream:
        json.dump(result, stream, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    receipt = dict(schema='rinkspace.reviewed-source-fit-dataset-delta-append.v1',
        previous_dataset_reference=base_reference, revised_dataset_reference=reference(manifest),
        literal_all_source_preflight_reference=audit_reference, appended_sources=appended,
        reused_unchanged_support_indices=len(base['cuts']),
        original_UV_sigma_operators_clock_and_HELD_unchanged=True, raw_source_arrays_copied=False,
        solver_started=False, independent_accuracy_accepted=False)
    receipt_path = output / 'reviewed_delta_append_receipt.json'
    with receipt_path.open('x', encoding='utf8') as stream:
        json.dump(receipt, stream, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    return dict(dataset_reference=reference(manifest), append_receipt_reference=reference(receipt_path))
