"""Append per-shot source and finite-FIT coverage to the current index audit."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
from calibration.program_tracking.compact_source_header_inventory import read_minimal_compact_source_headers
from calibration.program_tracking.strict_model_scope import authenticate_source_reference, file_sha


def identity(ref):
    return str(Path(ref['path']).resolve()).casefold(), ref['sha256']


def supplement(audit_path):
    audit_path = Path(audit_path).resolve()
    original_sha = file_sha(audit_path)
    report = json.loads(audit_path.read_text(encoding='utf8'))
    if report['schema'] != 'rinkspace.final-logical-finite-source-index-audit.v1':
        raise ValueError('Current final logical index audit required')
    logical_ref = report['logical_source_reference']
    dataset_ref = report['frozen_fit_dataset_reference']
    for ref in (logical_ref, dataset_ref):
        authenticate_source_reference(ref)
    logical = json.loads(Path(logical_ref['path']).read_text(encoding='utf8'))
    dataset = json.loads(Path(dataset_ref['path']).read_text(encoding='utf8'))
    selected = {identity(c['source_pack']):c for c in dataset['cuts']}
    source_cuts, fit_cuts, totals = {}, {}, {}
    for record in logical['records']:
        cut = record['cut']
        source_cuts.setdefault(cut, set())
        fit_cuts.setdefault(cut, set())
        count = totals.setdefault(cut, Counter())
        ref = record['absolute_inputs']
        authenticate_source_reference(ref)
        headers, offsets = read_minimal_compact_source_headers(ref['path'])
        with np.load(ref['path'], allow_pickle=False) as z:
            metadata = json.loads(str(z['metadata_json'].item()))
        source_cuts[cut].update(metadata['processed_actual_pts'])
        indexed = selected[identity(ref)]
        authenticate_source_reference(indexed['support_index'])
        with np.load(indexed['support_index']['path'], allow_pickle=False) as z:
            fit = z['fit_point_indices']
            if not np.array_equal(z['row_offsets'], offsets):
                raise ValueError('Original source row boundaries changed during cut coverage supplement')
        count['fit_points'] += len(fit)
        for name in ('held_points', 'pending_support_points', 'pending_named_points'):
            count[name] += indexed[name]
        for ri, row in enumerate(headers):
            lo, hi = map(int, offsets[ri:ri+2])
            a,b = np.searchsorted(fit, [lo,hi])
            if b > a:
                fit_cuts[cut].add(row['actual_pts'])
        if not fit_cuts[cut].issubset(source_cuts[cut]):
            raise ValueError('Finite-FIT exposure belongs outside its own shot source clock')
        print(json.dumps(dict(cut=cut, packet=Path(ref['path']).name,
            finite_fit_source_exposures_so_far=len(fit_cuts[cut]))), flush=True)
    plan_ref = logical['original_authorities']['plan']
    authenticate_source_reference(plan_ref)
    plan = json.loads(Path(plan_ref['path']).read_text(encoding='utf8'))
    by_cut = {entry['shot_id'].rsplit('_',1)[-1]:entry for entry in plan['entries']}
    all_source, all_fit = set(), set()
    rows = []
    for cut in sorted(source_cuts):
        if all_source.intersection(source_cuts[cut]):
            raise ValueError('Distinct shot clocks overlap canonical source exposures')
        all_source.update(source_cuts[cut])
        all_fit.update(fit_cuts[cut])
        entry = by_cut[cut]
        if len(source_cuts[cut]) != entry['canonical_exposures']:
            raise ValueError('A shot lost its original complete canonical processing coverage')
        rows.append(dict(cut=cut, shot_id=entry['shot_id'], native_frames=entry['native_frames'],
            source_clock_actual_exposures=len(source_cuts[cut]),
            actual_exposures_with_finite_fit=len(fit_cuts[cut]),
            actual_exposures_without_finite_fit=len(source_cuts[cut]-fit_cuts[cut]),
            **dict(totals[cut])))
    if (len(all_source) != report['actual_exposures'] or len(all_fit) != report['actual_exposures_with_finite_fit']
            or sum(r['fit_points'] for r in rows) != report['counts']['fit_points']
            or sum(r['held_points'] for r in rows) != report['counts']['held_points']):
        raise ValueError('Per-shot unique coverage or point totals changed final independent audit')
    report.update(source_cut_coverage=rows,
        source_cut_exposures_deduplicated_by_actual_pts=True,
        source_cut_coverage_supplement_implementation_reference=dict(path=str(Path(__file__).resolve()), sha256=file_sha(__file__)),
        base_index_audit_sha256_before_cut_supplement=original_sha)
    if file_sha(audit_path) != original_sha:
        raise ValueError('Another writer changed the current report during supplement')
    audit_path.write_text(json.dumps(report, ensure_ascii=False, separators=(',', ':')), encoding='utf8')
    return dict(path=str(audit_path), sha256=file_sha(audit_path), source_cut_coverage=rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--audit', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(supplement(args.audit), ensure_ascii=False, separators=(',', ':')), flush=True)
