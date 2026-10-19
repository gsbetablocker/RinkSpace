"""Independently check a frozen exact-index revision against its parent.

Unchanged source bytes and earlier duplicate/operator audits remain authoritative.
This check accounts for every point and rechecks new source-bound exclusions,
whole source clock, finite support and complete HELD conservation.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from calibration.program_tracking.compact_source_header_inventory import read_minimal_compact_source_headers
from calibration.program_tracking.strict_model_scope import authenticate_source_reference, file_sha


def reference(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=file_sha(path))


def identity(ref):
    return str(Path(ref['path']).resolve()).casefold(), ref['sha256']


def read_json(ref):
    authenticate_source_reference(ref)
    return json.loads(Path(ref['path']).read_text(encoding='utf-8-sig'))


def audit(parent_ref, revised_ref, keep_facts_ref=None):
    started = time.perf_counter()
    parent, revised = read_json(parent_ref), read_json(revised_ref)
    if (parent['schema'] != 'rinkspace.immutable-reviewed-finite-source-fit-dataset.v1'
            or revised['schema'] != parent['schema']
            or revised['previous_frozen_dataset_reference'] != parent_ref
            or revised['source'] != parent['source']
            or revised['expected_actual_pts'] != parent['expected_actual_pts']):
        raise ValueError('Revision must retain its exact source and whole parent clock')
    before = {identity(c['source_pack']): c for c in parent['cuts']}
    after = {identity(c['source_pack']): c for c in revised['cuts']}
    if len(before) != len(parent['cuts']) or set(before) != set(after) or len(after) != len(revised['cuts']):
        raise ValueError('Revision changed original source packs or duplicated a source')
    filters = [(ref, read_json(ref)) for ref in revised['exact_identity_revision_filter_references']]
    for _, fact in filters:
        if (fact['schema'] != 'rinkspace.exact-source-named-identity-pending-filter.v1'
                or fact['retain_pending_source_readouts_full_literal_evidence'] is not True):
            raise ValueError('Exact literal identity evidence required')
    groups = {}
    for ref, fact in filters:
        for group in fact['source_pack_groups']:
            key = identity(group['source_pack_reference'])
            if key not in after:
                raise ValueError('New facts refer to an absent original source')
            groups.setdefault(key, []).append((ref, group))
    keeps = {}
    if keep_facts_ref is not None:
        keep_facts = read_json(keep_facts_ref)
        if (keep_facts['schema'] != 'rinkspace.exact-specific-inner-original-source-independent-keepfacts.v1'
                or keep_facts['additional_FIT_points_created'] != 0
                or keep_facts['exact_existing_point_keep_count'] != len(keep_facts['points'])):
            raise ValueError('Independently confirmed keeps must refer to unchanged original points')
        for point in keep_facts['points']:
            key = identity(point['source_pack_reference'])
            if (key not in after or point['decision'] != 'allow'
                    or point['source_feature_physical_identity_confirmed'] is not True
                    or point['original_source_image_actually_reviewed'] is not True
                    or point['model_projection_or_fit_residual_used_for_identity'] is not False):
                raise ValueError('Specific material keep requires original-image identity evidence')
            keeps.setdefault(key, []).append(point)
    totals, materials, fit_exposures, bindings = Counter(), Counter(), set(), {}
    per_cut, per_cut_fit, per_cut_source = {}, {}, {}
    records, changed_points = [], 0
    partitions = {'fit_point_indices': 'fit_points', 'held_point_indices': 'held_points',
                  'pending_support_point_indices': 'pending_support_points',
                  'pending_named_point_indices': 'pending_named_points'}
    allowed = {*partitions, 'row_offsets', 'promoted_held_rows'}
    for key, cut in after.items():
        prior = before[key]
        ref = cut['source_pack']
        authenticate_source_reference(ref)
        headers, offsets = read_minimal_compact_source_headers(ref['path'])
        with np.load(ref['path'], allow_pickle=False) as data:
            sigma = data['sigma_px']
            original = data['original_point_indices']
            metadata = json.loads(str(data['metadata_json'].item()))
            current = {int(p): (int(n), bytes(h).decode('ascii')) for p, n, h in zip(
                data['source_frame_actual_pts'], data['source_frame_native_pts'],
                data['source_frame_BGR_sha256']) if n >= 0 and h}
            keep_uv = data['uv'] if key in keeps else None
        if (metadata['source'] != revised['source']
                or metadata['expected_actual_pts'] != revised['expected_actual_pts']
                or sorted(current) != metadata['processed_actual_pts']):
            raise ValueError('A source lost its authentic processed native RGB clock')
        for pts, binding in current.items():
            if pts in bindings and bindings[pts] != binding:
                raise ValueError('Overlapping packs disagree on actual original RGB')
            bindings[pts] = binding
        saved = []
        for idx_ref in (prior['support_index'], cut['support_index']):
            authenticate_source_reference(idx_ref)
            with np.load(idx_ref['path'], allow_pickle=False) as data:
                if set(data.files) != allowed or not np.array_equal(data['row_offsets'], offsets):
                    raise ValueError('Index layout or source row boundaries changed')
                saved.append({name: data[name] for name in allowed})
        old, new = saved
        for name in allowed - {'row_offsets'}:
            if new[name].dtype != np.dtype('int64') or new[name].ndim != 1:
                raise ValueError('Original source indices must be one-dimensional int64')
        if (not np.array_equal(old['held_point_indices'], new['held_point_indices'])
                or not np.array_equal(old['promoted_held_rows'], new['promoted_held_rows'])):
            raise ValueError('Revision changed HELD protection')
        pending = set()
        for _, group in groups.get(key, []):
            for witness in group['source_review_witnesses']:
                authenticate_source_reference(witness)
            count = 0
            for entry in group['entries']:
                ri = entry['pack_row_index']
                row = headers[ri]
                lo, hi = map(int, offsets[ri:ri + 2])
                indices = entry['pack_point_indices']
                bind = entry['source_binding']
                if (row['split'] != 'fit' or entry['physical_material_id'] != row['physical_material_id']
                        or bind['actual_pts'] != row['actual_pts'] or bind['native_pts'] != row['native_pts']
                        or bind['decoded_native_BGR_sha256'] != row['decoded_BGR_sha256']
                        or current[row['actual_pts']] != (row['native_pts'], row['decoded_BGR_sha256'])
                        or any(type(i) is not int or not lo <= i < hi for i in indices)
                        or original[indices].tolist() != entry['original_candidate_indices']
                        or pending.intersection(indices)):
                    raise ValueError('New fact lost exact original row/material/native RGB/candidate binding')
                pending.update(indices)
                count += len(indices)
            if count != group['pending_named_point_count']:
                raise ValueError('Literal new fact count changed')
        added = np.asarray(sorted(pending), np.int64)
        if np.intersect1d(added, old['pending_named_point_indices']).size:
            raise ValueError('Already pending evidence was counted again')
        if not np.array_equal(new['pending_named_point_indices'],
                              np.sort(np.r_[old['pending_named_point_indices'], added])):
            raise ValueError('Revised named mask differs from exact new facts')
        keep_ids = set()
        for point in keeps.get(key, []):
            ri, pi = point['pack_row_index'], point['pack_point_index']
            row = headers[ri]
            lo, hi = map(int, offsets[ri:ri + 2])
            if (pi in keep_ids or not lo <= pi < hi or point['scope'] != row['scope']
                    or point['physical_material_id'] != row['physical_material_id']
                    or (point['actual_pts'], point['native_pts'], point['decoded_BGR_sha256']) !=
                       (row['actual_pts'], row['native_pts'], row['decoded_BGR_sha256'])
                    or point['original_candidate_index'] != int(original[pi])
                    or point['original_sigma_px'] != float(sigma[pi])
                    or not np.array_equal(point['uv'], keep_uv[pi])
                    or not np.isin(pi, old['fit_point_indices'])
                    or not np.isin(pi, new['fit_point_indices'])):
                raise ValueError('Independent keep lost its exact original UV/support/role/source identity')
            authenticate_source_reference(point['original_raw_native_image_reference'])
            authenticate_source_reference(point['exact_frame_review_reference'])
            keep_ids.add(pi)
        for name in ('fit_point_indices', 'held_point_indices', 'pending_support_point_indices'):
            if not np.array_equal(new[name], old[name][~np.isin(old[name], added)]):
                raise ValueError('Revision changed a role beyond its exact new evidence')
        arrays = [new[name] for name in partitions]
        if (any(len(a) > 1 and np.any(np.diff(a) <= 0) for a in arrays)
                or not np.array_equal(np.sort(np.concatenate(arrays)), np.arange(len(sigma)))):
            raise ValueError('Revised roles duplicated or discarded original source points')
        finite = np.isfinite(sigma) & (sigma > 0)
        if not finite[new['fit_point_indices']].all() or finite[new['pending_support_point_indices']].any():
            raise ValueError('Missing original uncertainty became finite FIT support')
        promoted = set(new['promoted_held_rows'].tolist())
        selected = new['fit_point_indices']
        for ri, row in enumerate(headers):
            pts, shot = row['actual_pts'], row['shot_id']
            lo, hi = map(int, offsets[ri:ri + 2])
            if row['scope'] not in ('ice', 'board') or pts not in current:
                raise ValueError('Model-external or source-unbound point entered a source pack')
            count = per_cut.setdefault(shot, Counter())
            per_cut_source.setdefault(shot, set()).add(pts)
            for name, field in partitions.items():
                left, right = np.searchsorted(new[name], [lo, hi])
                count[field] += int(right - left)
                if field == 'held_points' and right > left and row['split'] != 'held' and ri not in promoted:
                    raise ValueError('HELD indices disagree with source row role')
            left, right = np.searchsorted(selected, [lo, hi])
            if right > left:
                if row['split'] != 'fit' or ri in promoted:
                    raise ValueError('A protected HELD row entered FIT')
                materials[row['physical_material_id']] += int(right - left)
                fit_exposures.add(pts)
                per_cut_fit.setdefault(shot, set()).add(pts)
        amounts = {field: len(new[name]) for name, field in partitions.items()}
        if any(cut[field] != count for field, count in amounts.items()):
            raise ValueError('Source manifest totals differ from actual indices')
        totals.update(amounts)
        changed_points += len(added)
        records.append(dict(source_pack_reference=ref, support_index_reference=cut['support_index'],
                            new_pending_points=len(added), **amounts))
    if (sorted(bindings) != revised['expected_actual_pts']
            or dict(sorted(materials.items())) != revised['finite_fit_points_by_material']
            or any(revised[name] != count for name, count in totals.items())
            or len(fit_exposures) != revised['actual_exposures_with_finite_fit']
            or len(bindings) - len(fit_exposures) != revised['actual_exposures_without_finite_fit']):
        raise ValueError('Whole source clock/material/exposure/role aggregate differs')
    coverage = [dict(shot_id=shot, cut=shot.rsplit('_', 1)[-1], **dict(count),
                     row_bound_actual_exposures=len(per_cut_source[shot]),
                     actual_exposures_with_finite_fit=len(per_cut_fit.get(shot, set())))
                for shot, count in sorted(per_cut.items())]
    return dict(schema='rinkspace.independent-exact-source-dataset-revision-audit.v1',
                parent_dataset_reference=parent_ref, revised_dataset_reference=revised_ref,
                implementation_reference=reference(__file__), passed=True, records=records,
                counts=dict(totals), new_exact_pending_points=changed_points,
                actual_exposures=len(bindings), actual_exposures_with_finite_fit=len(fit_exposures),
                source_cut_point_coverage=coverage,
                independently_verified_original_points_kept=sum(map(len, keeps.values())),
                independent_keep_facts_reference=keep_facts_ref,
                original_UV_sigma_operators_and_all_HELD_unchanged=True,
                all_original_point_locations_accounted=True, solver_started=False,
                independent_pixel_accuracy_accepted=False, seconds=time.perf_counter() - started)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--revised', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--keep-facts', type=Path)
    args = parser.parse_args()
    result = audit(reference(args.parent), reference(args.revised),
                   reference(args.keep_facts) if args.keep_facts else None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf8') as stream:
        json.dump(result, stream, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    print(json.dumps(dict(report_reference=reference(args.output), passed=True,
                          counts=result['counts'], seconds=result['seconds'])), flush=True)


if __name__ == '__main__':
    main()
