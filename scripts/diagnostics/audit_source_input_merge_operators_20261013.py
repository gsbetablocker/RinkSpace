"""Audit exact source merge identities across immutable inputs, one cut at a time.

This records quantity, explicit split, working support and typed operators.
It does not qualify pixels, remove evidence, install weights, or start FIT.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
from calibration.program_tracking.compact_source_header_inventory import read_minimal_compact_source_headers
from calibration.program_tracking.strict_model_scope import authenticate_source_reference, file_sha

OPERATOR_FIELDS = ('curve_observation_kind', 'point_correspondence',
    'must_preserve_upper_branch', 'scanline_branch', 'scanline_fixed_axis',
    'extent_side', 'clip_boundary_x', 'clip_blur_allowance_px',
    'source_operator_array_fields_present')


def reference(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=file_sha(path))


def _operator(row):
    value = {name: row.get(name) for name in OPERATOR_FIELDS}
    typed = (row.get('point_correspondence') is True or
        row.get('must_preserve_upper_branch') is True or
        row.get('curve_observation_kind') not in (None, 'measured_transition_location') or
        any(row.get(k) is not None for k in ('scanline_branch', 'scanline_fixed_axis',
            'extent_side', 'clip_boundary_x', 'clip_blur_allowance_px')) or
        bool(row.get('source_operator_array_fields_present')))
    return json.dumps(value, sort_keys=True, separators=(',', ':')), typed


def audit_cut(cut, references):
    groups = {}
    choices = {}
    pack_records = []
    conflicts = []
    conflicts_seen = set()
    totals = Counter()
    for pi, ref in enumerate(references):
        authenticate_source_reference(ref)
        headers, offsets = read_minimal_compact_source_headers(ref['path'])
        with np.load(ref['path'], allow_pickle=False) as z:
            uv, sigma = z['uv'], z['sigma_px']
            metadata = json.loads(str(z['metadata_json'].item()))
        if len(uv) != len(sigma) or offsets[-1] != len(uv):
            raise ValueError('Original source arrays and ordered rows disagree')
        finite = np.isfinite(sigma) & (sigma > 0)
        if not np.isfinite(uv).all() or np.isinf(sigma).any() or np.any(sigma[np.isfinite(sigma)] <= 0):
            raise ValueError('Invalid original source values')
        split_points, materials, operators = Counter(), Counter(), Counter()
        all_missing_board_rows = all_missing_board_points = 0
        for ri, row in enumerate(headers):
            lo, hi = map(int, offsets[ri:ri+2])
            count = hi-lo
            physical = (row['actual_pts'], row['scope'], row['physical_material_id'],
                json.dumps(row['model_target'], sort_keys=True, separators=(',', ':')),
                row['observation_semantics'])
            gi = groups.setdefault(physical, len(groups))
            operator, typed = _operator(row)
            split_points[row['split']] += count
            materials[row['physical_material_id']] += count
            operators[operator] += count
            if row['scope'] == 'board' and not finite[lo:hi].any():
                all_missing_board_rows += 1
                all_missing_board_points += count
            totals['typed_input_points'] += count if typed else 0
            for global_index in range(lo, hi):
                key = (gi, np.asarray(uv[global_index], np.float64).tobytes())
                current = (pi, ri, global_index, bool(finite[global_index]), row['split'], operator, typed)
                if key in choices:
                    old = choices[key]
                    cross = old[0] != pi
                    totals['exact_duplicate_occurrences'] += 1
                    totals['cross_pack_exact_duplicate_occurrences'] += int(cross)
                    totals['same_pixel_target_conflicting_split_occurrences'] += int(old[4] != row['split'])
                    totals['missing_support_replaced_by_finite_occurrences'] += int(current[3] and not old[3])
                    totals['typed_exact_duplicate_occurrences'] += int(typed or old[6])
                    totals['typed_cross_pack_exact_duplicate_occurrences'] += int(cross and (typed or old[6]))
                    conflict = old[5] != operator
                    totals['operator_flag_conflicting_occurrences'] += int(conflict)
                    totals['cross_pack_operator_flag_conflicting_occurrences'] += int(cross and conflict)
                    unresolved_array = bool(row.get('source_operator_array_fields_present'))
                    totals['duplicate_operator_array_values_require_literal_check'] += int(unresolved_array)
                    if conflict:
                        identity = (gi, key[1], old[0], old[1], pi, ri, old[5], operator)
                        if identity not in conflicts_seen:
                            conflicts_seen.add(identity)
                            conflicts.append(dict(actual_pts=row['actual_pts'],
                                material=row['physical_material_id'], model_target=row['model_target'],
                                uv=uv[global_index].tolist(), first_pack_reference=references[old[0]],
                                first_row_index=old[1], first_global_point_index=old[2],
                                first_operator=json.loads(old[5]), repeated_pack_reference=ref,
                                repeated_row_index=ri, repeated_global_point_index=global_index,
                                repeated_operator=json.loads(operator), cross_pack=cross))
                    if current[3] and not old[3]:
                        choices[key] = current
                else:
                    choices[key] = current
        authenticate_source_reference(ref)
        record = dict(source_reference=ref, points=len(uv), rows=len(headers),
            finite_working_sigma_points=int(finite.sum()), missing_working_sigma_points=int((~finite).sum()),
            split_points=dict(split_points), material_points=dict(materials),
            operator_point_inventory=[dict(operator=json.loads(op), points=n) for op, n in operators.items()],
            all_missing_board_rows_eligible_for_saved_readout_reconstruction=all_missing_board_rows,
            all_missing_board_points_before_any_reconstruction=all_missing_board_points,
            expected_actual_exposures=len(metadata['expected_actual_pts']),
            processed_actual_exposures=len(metadata['processed_actual_pts']),
            source_uncertainty_ready_for_fit=metadata['source_uncertainty_ready_for_fit'])
        pack_records.append(record)
        print(json.dumps(dict(cut=cut, pack=Path(ref['path']).name, points=len(uv),
            current_cross_pack_operator_conflicts=totals['cross_pack_operator_flag_conflicting_occurrences']),
            separators=(',', ':')), flush=True)
        del headers, offsets, uv, sigma, finite
        gc.collect()
    totals['unique_finite_working_sigma_points_before_saved_readout_reconstruction'] = sum(v[3] for v in choices.values())
    totals['unique_missing_working_sigma_points_before_saved_readout_reconstruction'] = sum(not v[3] for v in choices.values())
    return dict(cut=cut, inputs=pack_records, points=sum(r['points'] for r in pack_records),
        rows=sum(r['rows'] for r in pack_records), physical_target_exposure_groups=len(groups),
        unique_exact_pixel_target_count=len(choices), counts=dict(totals), operator_conflicts=conflicts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--deltas', type=Path, action='append', default=[])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve an earlier source input audit')
    authorities = [reference(args.plan), *[reference(p) for p in args.deltas]]
    plan = json.loads(args.plan.read_text(encoding='utf8'))
    by_cut = {entry['shot_id'].rsplit('_', 1)[-1]: [entry['absolute_inputs']] for entry in plan['entries']}
    for path in args.deltas:
        for record in json.loads(path.read_text(encoding='utf8'))['records']:
            by_cut[record['cut']].append(record['absolute_inputs'])
    started = time.perf_counter()
    records = [audit_cut(cut, refs) for cut, refs in by_cut.items()]
    for ref in authorities:
        authenticate_source_reference(ref)
    total = Counter()
    for record in records:
        total.update(record['counts'])
    report = dict(schema='rinkspace.source-input-cross-pack-exact-pixel-operator-audit.v1',
        input_manifest_references=authorities, records=records, counts=dict(total),
        points=sum(r['points'] for r in records), rows=sum(r['rows'] for r in records),
        unique_exact_pixel_target_count=sum(r['unique_exact_pixel_target_count'] for r in records),
        original_source_arrays_changed=False, weights_installed=False,
        typed_array_values_not_inferred_from_presence_flags=True,
        pixel_accuracy_qualified=False, solver_started=False, elapsed_seconds=time.perf_counter()-started,
        implementation_reference=reference(__file__))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf8') as f:
        json.dump(report, f, ensure_ascii=False, separators=(',', ':'))
    print(json.dumps(dict(output=str(args.output), counts=dict(total), points=report['points'],
        rows=report['rows']), separators=(',', ':')), flush=True)


if __name__ == '__main__':
    main()
