"""Independently audit a logical original-source FIT view and its index arrays.

Recompute original exact masks, whole source clock, ordinary-merge HELD
protection and literal finite support. Never evaluate or install a solver.
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
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
from calibration.program_tracking.compact_source_header_inventory import read_minimal_compact_source_headers
from calibration.program_tracking.strict_model_scope import authenticate_source_reference, file_sha


def reference(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=file_sha(path))


def identity(ref):
    return str(Path(ref['path']).resolve()).casefold(), ref['sha256']


def audit_frozen_indices(dataset_ref, precheck_ref):
    """Check every frozen index against unchanged independently checked inputs.

    The precheck already proves no HELD promotion and no named/HELD overlap in
    this actual source selection. Only packets containing HELD need their row
    headers read again; all finite/missing and named masks are checked exactly.
    """
    started = time.perf_counter()
    for ref in (dataset_ref, precheck_ref):
        authenticate_source_reference(ref)
    previous = json.loads(Path(precheck_ref['path']).read_text(encoding='utf8'))
    dataset = json.loads(Path(dataset_ref['path']).read_text(encoding='utf8'))
    if (previous['schema'] != 'rinkspace.independent-logical-finite-source-dataset-audit.v1'
            or dataset['schema'] != 'rinkspace.immutable-reviewed-finite-source-fit-dataset.v1'
            or previous['counts']['promoted_held_rows'] != 0
            or previous['counts']['literal_current_named_pending_points'] != previous['counts']['pending_named_points']):
        raise ValueError('Fast final audit requires this independent zero-promotion/no-HELD-mask-overlap precheck')
    logical_ref = previous['logical_source_reference']
    authenticate_source_reference(logical_ref)
    logical = json.loads(Path(logical_ref['path']).read_text(encoding='utf8'))
    prior = {identity(record['source_reference']):record for record in previous['records']}
    logical_records = {identity(record['absolute_inputs']):record for record in logical['records']}
    cuts = {identity(record['source_pack']):record for record in dataset['cuts']}
    if set(cuts) != set(prior) or set(cuts) != set(logical_records) or len(cuts) != len(dataset['cuts']):
        raise ValueError('Frozen dataset changed the independently checked original source identities')
    records, counts = [], Counter()
    allowed = {'row_offsets', 'fit_point_indices', 'held_point_indices',
        'pending_support_point_indices', 'pending_named_point_indices', 'promoted_held_rows'}
    for key, cut in cuts.items():
        ref = cut['source_pack']
        authenticate_source_reference(ref)
        with np.load(ref['path'], allow_pickle=False) as z:
            sigma, offsets = z['sigma_px'], z['row_offsets']
        index_ref = cut['support_index']
        authenticate_source_reference(index_ref)
        with np.load(index_ref['path'], allow_pickle=False) as z:
            if set(z.files) != allowed or not np.array_equal(z['row_offsets'], offsets):
                raise ValueError('Frozen view duplicates evidence or changes original row boundaries')
            arrays = {name:z[name] for name in allowed if name != 'row_offsets'}
        if any(value.dtype != np.dtype('int64') or value.ndim != 1 for value in arrays.values()):
            raise ValueError('Exact original index arrays must be one-dimensional int64')
        if len(arrays['promoted_held_rows']):
            raise ValueError('Frozen view added HELD promotions absent from the independent precheck')
        partitions = [value for name,value in arrays.items() if name != 'promoted_held_rows']
        if (any(len(value)>1 and np.any(np.diff(value)<=0) for value in partitions)
                or not np.array_equal(np.sort(np.concatenate(partitions)), np.arange(len(sigma)))):
            raise ValueError('Frozen index partitions duplicate or discard original source pixels')
        named = np.asarray(logical_records[key]['original_pack_point_indices_to_exclude_from_named_FIT'], np.int64)
        if not np.array_equal(arrays['pending_named_point_indices'], named):
            raise ValueError('Frozen index changed the independent exact current named mask')
        held = []
        if prior[key]['held_points']:
            headers, header_offsets = read_minimal_compact_source_headers(ref['path'])
            if not np.array_equal(offsets, header_offsets):
                raise ValueError('Literal HELD header boundaries changed')
            for ri,row in enumerate(headers):
                if row['split'] == 'held':
                    lo,hi = map(int, offsets[ri:ri+2])
                    held.extend(range(lo,hi))
        if not np.array_equal(arrays['held_point_indices'], np.asarray(held,np.int64)):
            raise ValueError('Frozen index changed complete original HELD observations')
        finite = np.isfinite(sigma) & (sigma > 0)
        if not finite[arrays['fit_point_indices']].all() or finite[arrays['pending_support_point_indices']].any():
            raise ValueError('Frozen index changed literal original measured support missingness')
        amounts = {name.removesuffix('_point_indices')+'_points':len(value)
            for name,value in arrays.items() if name != 'promoted_held_rows'}
        for name,n in amounts.items():
            if n != prior[key][name] or n != cut[name]:
                raise ValueError('Frozen index quantity changed independent support partition')
        counts.update(amounts)
        records.append(dict(source_reference=ref, support_index_reference=index_ref,
            all_point_index_arrays_independently_verified=True, **amounts))
    for name,n in counts.items():
        if dataset[name] != n or previous['counts'][name] != n:
            raise ValueError('Frozen dataset aggregate differs from independent source partition')
    if (dataset['actual_exposures'] != previous['actual_exposures'] or
            dataset['actual_exposures_with_finite_fit'] != previous['actual_exposures_with_finite_fit'] or
            dataset['finite_fit_points_by_material'] != previous['finite_fit_points_by_material']):
        raise ValueError('Frozen dataset changed independent source coverage/material quantities')
    authenticate_source_reference(dataset_ref)
    return dict(previous, schema='rinkspace.final-logical-finite-source-index-audit.v1',
        independent_source_partition_precheck_reference=precheck_ref,
        frozen_fit_dataset_reference=dataset_ref, frozen_index_records=records,
        all_frozen_point_index_arrays_independently_verified=True,
        final_index_verification_seconds=time.perf_counter()-started,
        final_index_audit_implementation_reference=reference(__file__))


def audit(logical_ref, dataset_ref=None):
    started = time.perf_counter()
    authenticate_source_reference(logical_ref)
    logical = json.loads(Path(logical_ref['path']).read_text(encoding='utf8'))
    if logical['schema'] != 'rinkspace.logical-current-qualified-source-inputs.v1':
        raise ValueError('Current immutable logical source manifest required')
    pre_ref = logical['operator_and_duplicate_preflight_reference']
    authenticate_source_reference(pre_ref)
    preflight = json.loads(Path(pre_ref['path']).read_text(encoding='utf8'))
    for name in ('exact_duplicate_occurrences', 'operator_flag_conflicting_occurrences',
            'same_pixel_target_conflicting_split_occurrences', 'duplicate_operator_array_values_require_literal_check'):
        if preflight['counts'].get(name, 0):
            raise ValueError('Logical view requires resolved duplicate/operator/split preflight')
    source_refs = [record['absolute_inputs'] for record in logical['records']]
    prior_refs = [pack['source_reference'] for cut in preflight['records'] for pack in cut['inputs']]
    if len(set(map(identity, source_refs))) != len(source_refs) or set(map(identity, source_refs)) != set(map(identity, prior_refs)):
        raise ValueError('Logical inputs differ from the literal duplicate/operator preflight')
    for ref in logical['original_authorities'].values():
        authenticate_source_reference(ref)
    authenticate_source_reference(logical['model_identity_reference'])
    dataset = None
    by_source = {}
    if dataset_ref:
        authenticate_source_reference(dataset_ref)
        dataset = json.loads(Path(dataset_ref['path']).read_text(encoding='utf8'))
        if dataset['schema'] != 'rinkspace.immutable-reviewed-finite-source-fit-dataset.v1':
            raise ValueError('Frozen finite FIT dataset schema required')
        by_source = {identity(cut['source_pack']):cut for cut in dataset['cuts']}
        if len(by_source) != len(source_refs) or set(by_source) != set(map(identity, source_refs)):
            raise ValueError('Dataset did not retain exactly the logical source packs')
    held_features, held_materials = set(), set()
    for ref in source_refs:
        authenticate_source_reference(ref)
        rows, _ = read_minimal_compact_source_headers(ref['path'])
        for row in rows:
            if row['split'] == 'held':
                held_features.add((row['actual_pts'], row['feature_id']))
                held_materials.add((row['actual_pts'], row['physical_material_id']))
    frames, records = {}, []
    totals, per_material = Counter(), Counter()
    fit_exposures, fit_groups = set(), set()
    for record in logical['records']:
        ref = record['absolute_inputs']
        rows, offsets = read_minimal_compact_source_headers(ref['path'])
        with np.load(ref['path'], allow_pickle=False) as z:
            sigma = z['sigma_px']
            original_indices = z['original_point_indices']
            metadata = json.loads(str(z['metadata_json'].item()))
            own_frames = {int(a):(int(n), bytes(h).decode('ascii')) for a,n,h in zip(
                z['source_frame_actual_pts'], z['source_frame_native_pts'], z['source_frame_BGR_sha256']) if n >= 0 and h}
        if metadata['source'] != logical['source'] or metadata['expected_actual_pts'] != logical['expected_actual_pts']:
            raise ValueError('Logical source identity or complete clock changed')
        if sorted(own_frames) != metadata['processed_actual_pts']:
            raise ValueError('A logical source packet lost a processed own RGB binding')
        for pts, binding in own_frames.items():
            if pts in frames and frames[pts] != binding:
                raise ValueError('Overlapping sources disagree on canonical own RGB pixels')
            frames[pts] = binding
        if (len(sigma) != int(offsets[-1]) or len(sigma) != record['source_points_before_current_pending_filter']
                or np.isinf(sigma).any() or np.any(sigma[np.isfinite(sigma)] <= 0)):
            raise ValueError('Literal source quantity or working support changed')
        selected_mask = record['original_pack_point_indices_to_exclude_from_named_FIT']
        literal_pending = set()
        for filter_record in record['current_original_pending_filters']:
            fr = filter_record['filter_reference']
            authenticate_source_reference(fr)
            fact = json.loads(Path(fr['path']).read_text(encoding='utf8'))
            matches = [g for g in fact['source_pack_groups'] if identity(g['source_pack_reference']) == identity(ref)]
            if len(matches) != 1 or matches[0] != filter_record['group']:
                raise ValueError('Logical point-mask authority differs from original literal filter')
            group = matches[0]
            for witness in group['source_review_witnesses']:
                authenticate_source_reference(witness)
            amount = 0
            for entry in group['entries']:
                ri = entry['pack_row_index']
                row = rows[ri]
                lo, hi = map(int, offsets[ri:ri+2])
                values = entry['pack_point_indices']
                if (entry['physical_material_id'] != row['physical_material_id']
                        or entry['source_binding']['actual_pts'] != row['actual_pts']
                        or entry['source_binding']['native_pts'] != row['native_pts']
                        or entry['source_binding']['decoded_native_BGR_sha256'] != row['decoded_BGR_sha256']
                        or any(type(i) is not int or not lo <= i < hi for i in values)
                        or original_indices[values].tolist() != entry['original_candidate_indices']):
                    raise ValueError('Current mask lost its exact original row/material/source/candidate binding')
                if literal_pending.intersection(values):
                    raise ValueError('Current exact point-mask was applied twice')
                literal_pending.update(values)
                amount += len(values)
            if amount != group['pending_named_point_count']:
                raise ValueError('Literal point-mask count changed')
        if sorted(literal_pending) != selected_mask or len(literal_pending) != record['exact_pending_points']:
            raise ValueError('Logical named mask differs from authenticated original indices')
        named = np.zeros(len(sigma), bool)
        named[selected_mask] = True
        held = np.zeros(len(sigma), bool)
        promoted = []
        original_held_points = 0
        for ri, row in enumerate(rows):
            lo, hi = map(int, offsets[ri:ri+2])
            if row['point_count'] != hi-lo or row['scope'] not in ('ice', 'board') or row['split'] not in ('fit', 'held'):
                raise ValueError('Logical row scope/order/split invalid')
            if own_frames[row['actual_pts']] != (row['native_pts'], row['decoded_BGR_sha256']):
                raise ValueError('Logical row changed its own original pixels')
            protected = row['split'] == 'held' or (row['actual_pts'], row['feature_id']) in held_features or (
                row['actual_pts'], row['physical_material_id']) in held_materials
            if protected:
                held[lo:hi] = True
                if row['split'] == 'fit':
                    promoted.append(ri)
                else:
                    original_held_points += hi-lo
        finite = np.isfinite(sigma) & (sigma > 0)
        expected = dict(fit_point_indices=np.flatnonzero(~held & ~named & finite),
            held_point_indices=np.flatnonzero(held),
            pending_support_point_indices=np.flatnonzero(~held & ~named & ~finite),
            pending_named_point_indices=np.flatnonzero(~held & named))
        if not np.array_equal(np.sort(np.concatenate(list(expected.values()))), np.arange(len(sigma))):
            raise ValueError('Independent logical partition does not preserve every original pixel')
        index_ref = None
        if dataset:
            indexed = by_source[identity(ref)]
            index_ref = indexed['support_index']
            authenticate_source_reference(index_ref)
            with np.load(index_ref['path'], allow_pickle=False) as z:
                if not np.array_equal(z['row_offsets'], offsets):
                    raise ValueError('Frozen logical index changed original row offsets')
                for name, values in expected.items():
                    if not np.array_equal(z[name], values):
                        raise ValueError('Frozen logical index differs from independent partition: '+name)
                if not np.array_equal(z['promoted_held_rows'], np.asarray(promoted, np.int64)):
                    raise ValueError('Frozen logical index differs from ordinary-merge HELD protection')
                if any('uv' in name.lower() or 'sigma' in name.lower() for name in z.files):
                    raise ValueError('Logical index stores duplicated source numeric evidence')
        chosen = np.zeros(len(sigma), bool)
        chosen[expected['fit_point_indices']] = True
        for ri, row in enumerate(rows):
            lo, hi = map(int, offsets[ri:ri+2])
            amount = int(chosen[lo:hi].sum())
            if amount:
                fit_exposures.add(row['actual_pts'])
                fit_groups.add((row['actual_pts'], row['physical_material_id']))
                per_material[row['physical_material_id']] += amount
        amounts = {name.removesuffix('_point_indices')+'_points':len(values) for name, values in expected.items()}
        totals.update(amounts)
        totals['original_points'] += len(sigma)
        totals['original_rows'] += len(rows)
        totals['original_held_points'] += original_held_points
        totals['literal_current_named_pending_points'] += len(literal_pending)
        totals['promoted_held_rows'] += len(promoted)
        totals['promoted_held_points'] += len(expected['held_point_indices'])-original_held_points
        totals['missing_held_working_sigma_points'] += int((held & ~finite).sum())
        records.append(dict(cut=record['cut'], source_kind=record['source_kind'], source_reference=ref,
            support_index_reference=index_ref, source_points=len(sigma), source_rows=len(rows),
            literal_current_named_pending_points=len(literal_pending), original_held_points=original_held_points,
            promoted_held_rows=len(promoted), **amounts))
        authenticate_source_reference(ref)
        print(json.dumps(dict(cut=record['cut'], packet=Path(ref['path']).name, **amounts), separators=(',', ':')), flush=True)
    coverage_ref = logical['original_native_and_retirement_coverage_reference']
    authenticate_source_reference(coverage_ref)
    with np.load(coverage_ref['path'], allow_pickle=False) as z:
        canonical = np.flatnonzero(z['representative'])
        original_frames = {int(z['actual_pts'][i]):(int(z['native_pts'][i]), bytes(z['BGR_sha256'][i]).decode('ascii')) for i in canonical}
        native_count = len(z['native_pts'])
        retired_count = len(z['old_pixel_disposition'])
        if not bool(z['native_material_measurement_attempted'].all()):
            raise ValueError('Original complete native processing coverage incomplete')
    if frames != original_frames or set(frames) != set(logical['expected_actual_pts']):
        raise ValueError('Logical source view lost original canonical source/RGB coverage')
    if totals['literal_current_named_pending_points'] != logical['original_base_exact_pending_points']:
        raise ValueError('Logical exact named pending quantity changed')
    if totals['original_points'] != logical['source_points_before_current_pending_filters']:
        raise ValueError('Logical original point denominator changed')
    if dataset:
        for name in ('fit_points', 'held_points', 'pending_support_points', 'pending_named_points'):
            if dataset[name] != totals[name]:
                raise ValueError('Dataset aggregate changed independent per-packet quantity: '+name)
        if dataset['actual_exposures'] != len(frames) or dataset['actual_exposures_with_finite_fit'] != len(fit_exposures):
            raise ValueError('Dataset finite FIT exposure denominator changed')
        if dataset['finite_fit_points_by_material'] != dict(sorted(per_material.items())):
            raise ValueError('Dataset finite FIT material quantities changed')
    authenticate_source_reference(logical_ref)
    if dataset_ref:
        authenticate_source_reference(dataset_ref)
    return dict(schema='rinkspace.independent-logical-finite-source-dataset-audit.v1',
        logical_source_reference=logical_ref, frozen_fit_dataset_reference=dataset_ref,
        preflight_reference=pre_ref, original_coverage_reference=coverage_ref, records=records,
        counts=dict(totals), finite_fit_points_by_material=dict(sorted(per_material.items())),
        native_frames=native_count, actual_exposures=len(frames), native_alias_frames=native_count-len(frames),
        actual_exposures_with_finite_fit=len(fit_exposures), actual_exposures_without_finite_fit=len(frames)-len(fit_exposures),
        finite_fit_physical_material_exposure_groups=len(fit_groups), original_retired_points_not_restored=retired_count,
        scalar_residual_rows_for_finite_fit_points=2*totals['fit_points'],
        all_original_current_named_masks_literal_and_applied_once=True,
        complete_original_HELD_and_ordinary_merge_protection_verified=True,
        finite_fit_support_literal_original_sigma=True, original_source_sigma_reconstructed_points=0,
        original_source_pixel_or_loss_weights_changed=False, own_native_alias_pixels_added_to_fit=False,
        all_original_canonical_actual_native_RGB_bindings_bit_exact=True,
        exact_source_duplicates=0, operator_conflicts=0,
        full_solver_graph_adapter_installed=False, entire_iteration_speed_verified=False,
        whole_shot_visual_accuracy_accepted=False, solver_started=False,
        elapsed_seconds=time.perf_counter()-started, implementation_reference=reference(__file__))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--logical-sources', type=Path)
    parser.add_argument('--dataset', type=Path)
    parser.add_argument('--partition-precheck', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve an earlier independent logical dataset audit')
    if args.partition_precheck:
        if not args.dataset:
            raise ValueError('Frozen dataset is required for final index audit')
        result = audit_frozen_indices(reference(args.dataset), reference(args.partition_precheck))
    else:
        if not args.logical_sources:
            raise ValueError('Current logical source manifest required')
        result = audit(reference(args.logical_sources), reference(args.dataset) if args.dataset else None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf8') as f:
        json.dump(result, f, ensure_ascii=False, separators=(',', ':'))
    print(json.dumps(dict(output=str(args.output), counts=result['counts'], actual_exposures=result['actual_exposures'],
        fit_exposures=result['actual_exposures_with_finite_fit']), separators=(',', ':')), flush=True)
