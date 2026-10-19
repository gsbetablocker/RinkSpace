"""Apply new exact identity evidence to immutable source dataset indices."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from .compact_source_header_inventory import read_minimal_compact_source_headers
from .strict_model_scope import authenticate_source_reference
from .strict_source_fit_dataset import SCHEMA, _named_pending_indices, partition_support, reference


def revise_named_pending(base_reference, filter_references, output_directory):
    """Keep unchanged indices by reference; publish only newly changed masks.

    New facts can retire exact original FIT or missing-support locations to
    named pending. They cannot change UV, weights, clocks, operators or HELD.
    Overlapping facts fail rather than double-counting a source location.
    """
    for ref in (base_reference, *filter_references):
        authenticate_source_reference(ref)
    base = json.loads(Path(base_reference['path']).read_text(encoding='utf8'))
    if base.get('schema') != SCHEMA or not filter_references:
        raise ValueError('Frozen dataset and explicit new identity evidence required')
    output = Path(output_directory).resolve()
    if (output / 'fit_dataset.json').exists():
        raise FileExistsError('Preserve earlier frozen revision')
    output.mkdir(parents=True, exist_ok=True)
    source_keys = {(str(Path(c['source_pack']['path']).resolve()), c['source_pack']['sha256'])
                   for c in base['cuts']}
    additions = {}
    for ref in filter_references:
        fact = json.loads(Path(ref['path']).read_text(encoding='utf-8-sig'))
        if fact.get('schema') != 'rinkspace.exact-source-named-identity-pending-filter.v1':
            raise ValueError('New facts must use the exact original-index admission interface')
        for group in fact['source_pack_groups']:
            bound = group['source_pack_reference']
            key = str(Path(bound['path']).resolve()), bound['sha256']
            if key not in source_keys:
                raise ValueError('New identity evidence refers to an absent original source')
            additions.setdefault(key, []).append(ref)
    result = deepcopy(base)
    counts, materials, exposure_pts = Counter(), Counter(), set()
    changes = []
    for number, cut in enumerate(result['cuts']):
        ref = cut['source_pack']
        for bound in (ref, cut['support_index']):
            authenticate_source_reference(bound)
        key = str(Path(ref['path']).resolve()), ref['sha256']
        headers, offsets = read_minimal_compact_source_headers(ref['path'])
        with np.load(ref['path'], allow_pickle=False) as data:
            metadata = json.loads(str(data['metadata_json'].item()))
            sigma = data['sigma_px']
            bindings = {int(p): (int(n), bytes(h).decode('ascii')) for p, n, h in zip(
                data['source_frame_actual_pts'], data['source_frame_native_pts'],
                data['source_frame_BGR_sha256']) if n >= 0 and h}
        if metadata['source'] != base['source'] or metadata['expected_actual_pts'] != base['expected_actual_pts']:
            raise ValueError('Original source or full exposure catalogue changed')
        with np.load(cut['support_index']['path'], allow_pickle=False) as saved:
            arrays = {name: saved[name] for name in saved.files}
        if not np.array_equal(arrays['row_offsets'], offsets):
            raise ValueError('Frozen source row boundaries changed')
        roles = [dict(split=h['split'], point_count=h['point_count']) for h in headers]
        for ri in arrays['promoted_held_rows']:
            if roles[int(ri)]['split'] != 'fit':
                raise ValueError('Invalid inherited HELD promotion')
            roles[int(ri)]['split'] = 'held'
        old_facts = cut.get('pending_filter_references', [])
        old_named = _named_pending_indices(headers, offsets, ref, metadata['source'], bindings, old_facts)
        if not np.array_equal(old_named, arrays['pending_named_point_indices']):
            raise ValueError('Frozen named identity evidence changed')
        original = partition_support(roles, offsets, sigma)
        for name, indices in original.items():
            if not np.array_equal(np.setdiff1d(indices, old_named, assume_unique=True), arrays[name]):
                raise ValueError('Frozen source role/support masks changed')
        new_facts = [*old_facts, *additions.get(key, [])]
        new_named = _named_pending_indices(headers, offsets, ref, metadata['source'], bindings, new_facts)
        if np.intersect1d(new_named, original['held_point_indices'], assume_unique=True).size:
            raise ValueError('New identity facts cannot remove protected HELD evidence')
        if not np.array_equal(new_named, old_named):
            changed = np.setdiff1d(new_named, old_named, assume_unique=True)
            for name, indices in original.items():
                arrays[name] = np.setdiff1d(indices, new_named, assume_unique=True)
            arrays['pending_named_point_indices'] = new_named
            partitions = [arrays[name] for name in original] + [new_named]
            if not np.array_equal(np.sort(np.concatenate(partitions)), np.arange(len(sigma))):
                raise ValueError('New masks discarded or duplicated original source locations')
            path = output / f'source_{number:02d}_support_indices.npz'
            if path.exists():
                raise FileExistsError('Preserve earlier source index')
            np.savez_compressed(path, **arrays)
            changes.append(dict(source_pack=ref, original_support_index=cut['support_index'],
                revised_support_index=reference(path), new_pending_original_point_indices=changed.tolist(),
                original_fit_removed=int(np.intersect1d(changed, original['fit_point_indices']).size),
                original_missing_support_reclassified=int(np.intersect1d(
                    changed, original['pending_support_point_indices']).size)))
            cut['support_index'] = reference(path)
            cut['pending_filter_references'] = new_facts
        for name, field in (('fit_point_indices', 'fit_points'), ('held_point_indices', 'held_points'),
                            ('pending_support_point_indices', 'pending_support_points'),
                            ('pending_named_point_indices', 'pending_named_points')):
            cut[field] = len(arrays[name])
            counts[field] += cut[field]
        selected = arrays['fit_point_indices']
        for ri, header in enumerate(headers):
            lo, hi = offsets[ri:ri + 2]
            first, last = np.searchsorted(selected, [lo, hi])
            if last > first:
                materials[header['physical_material_id']] += int(last - first)
                exposure_pts.add(header['actual_pts'])
    if not changes or counts['held_points'] != base['held_points']:
        raise ValueError('Revision must change named pending while preserving every HELD location')
    result.update(counts)
    result.update(finite_fit_points_by_material=dict(sorted(materials.items())),
        actual_exposures_with_finite_fit=len(exposure_pts),
        actual_exposures_without_finite_fit=len(base['expected_actual_pts']) - len(exposure_pts),
        previous_frozen_dataset_reference=base_reference,
        exact_identity_revision_filter_references=filter_references)
    result['source_review_authorities'] = [*base['source_review_authorities'], *filter_references,
                                           reference(__file__)]
    manifest = output / 'fit_dataset.json'
    with manifest.open('x', encoding='utf8') as stream:
        json.dump(result, stream, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    receipt = dict(schema='rinkspace.exact-source-fit-dataset-identity-revision.v1',
        previous_dataset_reference=base_reference, revised_dataset_reference=reference(manifest),
        filter_references=filter_references, changed_sources=changes,
        reused_unchanged_support_indices=len(base['cuts']) - len(changes),
        counts=dict(counts), original_UV_sigma_operators_clock_and_HELD_unchanged=True,
        pending_is_not_negative_or_confirmed_reflection=True, solver_started=False,
        independent_accuracy_accepted=False)
    receipt_path = output / 'identity_revision_receipt.json'
    with receipt_path.open('x', encoding='utf8') as stream:
        json.dump(receipt, stream, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    return dict(dataset_reference=reference(manifest), revision_receipt_reference=reference(receipt_path))
