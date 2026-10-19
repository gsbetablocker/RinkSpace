"""Assemble successful full-clock cuts and measured missing-material deltas.

The original nine-cut coverage is kept separately. Overlapping source deltas
use the existing exact-pixel merge within their cut; disjoint cuts are then
written sequentially. This operation does not fit, train, or accept accuracy.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]

from calibration.program_tracking.strict_model_scope import authenticate_source_reference, file_sha
from calibration.program_tracking.strict_source_pack_merge import merge_strict_absolute_source_packs
from calibration.program_tracking.strict_source_disjoint_merge import merge_disjoint_strict_absolute_source_packs


def reference(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=file_sha(path))


def metadata(ref):
    authenticate_source_reference(ref)
    with np.load(ref['path'], allow_pickle=False) as saved:
        return json.loads(str(saved['metadata_json'].item()))


def require_exact_base_successor(previous_reference, current_reference):
    """A filtered base keeps the original exposure authority and clock.

    Its removed source pixels must not be reintroduced through a previously
    assembled cut. This verifies lineage only, never pixel accuracy.
    """
    previous = metadata(previous_reference)
    current = metadata(current_reference)
    identity = lambda ref: (str(Path(ref['path']).resolve()), ref['sha256'])
    if identity(previous_reference) not in {
            identity(ref) for ref in current['original_references']}:
        raise ValueError('A replacement base must retain its exact original source authority')
    for name in ('source', 'expected_actual_pts', 'processed_actual_pts'):
        if current[name] != previous[name]:
            raise ValueError('A replacement base changed original source or exposure clock')
    with np.load(previous_reference['path'], allow_pickle=False) as old:
        with np.load(current_reference['path'], allow_pickle=False) as new:
            for name in ('source_frame_actual_pts', 'source_frame_native_pts', 'source_frame_BGR_sha256'):
                a, b = old[name], new[name]
                if a.dtype != b.dtype or a.shape != b.shape or a.tobytes() != b.tobytes():
                    raise ValueError('A replacement base changed exact native exposure binding')


def assemble_cut(entry, additions, output_dir):
    """Keep a completed cut reusable while later measured deltas finish."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    base_ref = entry['absolute_inputs']
    base = metadata(base_ref)
    cut_clock = set(base['processed_actual_pts'])
    for ref in additions:
        new = metadata(ref)
        if (new['source'] != base['source'] or
                new['expected_actual_pts'] != base['expected_actual_pts'] or
                not set(new['processed_actual_pts']).issubset(cut_clock)):
            raise ValueError('A source delta changed video, programme clock or cut scope')
    filename = re.sub(r'[^A-Za-z0-9_.-]', '_', entry['shot_id'])
    receipt_path = out/(filename+'_assembly_receipt.json')
    previous_reference = None
    merge_inputs = [base_ref, *additions]
    if receipt_path.exists():
        previous = json.loads(receipt_path.read_text(encoding='utf8'))
        def identity(ref):
            return str(Path(ref['path']).resolve()), ref['sha256']
        same_base = identity(previous['original_cut']) == identity(base_ref)
        if not same_base:
            require_exact_base_successor(previous['original_cut'], base_ref)
        old = list(map(identity, previous['selected_delta_references']))
        current = list(map(identity, additions))
        receipt = previous['merged_cut']
        merged_previous = dict(path=receipt['path'], sha256=receipt['sha256'])
        authenticate_source_reference(merged_previous)
        if same_base and old == current:
            return merged_previous, previous
        # A later missing-material layer gets an immutable successor file.
        # The completed cut and its receipt are never replaced or discarded.
        digest_input = current if same_base else [identity(base_ref), current]
        digest = hashlib.sha256(json.dumps(digest_input, separators=(',', ':')).encode('utf8')).hexdigest()[:16]
        previous_reference = reference(receipt_path)
        receipt_path = out/(filename+'_assembly_receipt_'+digest+'.json')
        filename += '_'+digest
        if receipt_path.exists():
            successor = json.loads(receipt_path.read_text(encoding='utf8'))
            if (identity(successor['original_cut']) != identity(base_ref) or
                    list(map(identity, successor['selected_delta_references'])) != current):
                raise ValueError('A source assembly input digest identifies different bytes')
            selected = successor['merged_cut']
            ref = dict(path=selected['path'], sha256=selected['sha256'])
            authenticate_source_reference(ref)
            return ref, successor
        if same_base and current[:len(old)] == old:
            merge_inputs = [merged_previous, *additions[len(old):]]
    if additions:
        path = out/(filename+'_with_measured_deltas.npz')
        if path.exists():
            raise FileExistsError('Retain an incomplete earlier cut output for diagnosis')
        receipt = merge_strict_absolute_source_packs(merge_inputs, path)
        ref = dict(path=receipt['path'], sha256=receipt['sha256'])
        if set(metadata(ref)['processed_actual_pts']) != cut_clock:
            raise ValueError('An augmented cut lost its full source clock')
    else:
        ref = base_ref
        receipt = dict(path=ref['path'], sha256=ref['sha256'], existing_cut_reused=True,
            exact_duplicate_points_removed=0)
    record = dict(shot_id=entry['shot_id'], original_cut=base_ref,
        selected_delta_references=additions, merged_cut=receipt)
    if previous_reference is not None:
        record['earlier_completed_cut_assembly_reference'] = previous_reference
        record['earlier_original_cut_reference'] = previous['original_cut']
        record['previous_merged_pixels_reused'] = same_base and current[:len(old)] == old
    with receipt_path.open('x', encoding='utf8') as saved:
        json.dump(record, saved, ensure_ascii=False, separators=(',', ':'))
    gc.collect()
    return ref, record


def assemble(plan_reference, delta_reference, coverage_reference, output_dir, *, authorities=()):
    for ref in (plan_reference, delta_reference, coverage_reference, *authorities):
        authenticate_source_reference(ref)
    plan = json.loads(Path(plan_reference['path']).read_text(encoding='utf8'))
    deltas = json.loads(Path(delta_reference['path']).read_text(encoding='utf8'))
    if not plan['ready'] or len(plan['entries']) != 9:
        raise ValueError('Nine successful terminal cuts required')
    out = Path(output_dir).resolve()
    if (out/'full_programme_measured_source_pool.npz').exists() or (out/'full_programme_measured_source_pool_assembly.json').exists():
        raise FileExistsError('Keep previous source assembly unchanged')
    out.mkdir(parents=True, exist_ok=True)
    by_cut = {}
    allowed_cuts = {entry['shot_id'].rsplit('_', 1)[-1] for entry in plan['entries']}
    supplemental_references = [delta_reference, plan_reference, coverage_reference, *authorities]
    for record in deltas['records']:
        cut = record['cut']
        if cut not in allowed_cuts:
            raise ValueError('A measured delta names a cut outside this programme')
        ref = record['absolute_inputs']
        authenticate_source_reference(ref)
        by_cut.setdefault(cut, []).append(ref)
        for field in ('receipt', 'native_alias_readouts'):
            if field in record:
                authenticate_source_reference(record[field])
                supplemental_references.append(record[field])
    selected, cut_receipts = [], []
    for entry in plan['entries']:
        cut = entry['shot_id'].rsplit('_', 1)[-1]
        additions = by_cut.get(cut, [])
        ref, record = assemble_cut(entry, additions, out)
        selected.append(ref)
        cut_receipts.append(record)
        print(json.dumps(dict(cut=entry['shot_id'], **record['merged_cut']), separators=(',', ':')), flush=True)
        gc.collect()
    selected_path = out/'selected_cut_inputs.json'
    selected_manifest = dict(schema='rinkspace.selected-current-measured-cut-inputs.v1',
        records=[dict(shot_id=entry['shot_id'], absolute_inputs=ref)
            for entry,ref in zip(plan['entries'],selected)],
        model_identity_reference=plan.get('identity_reference'),
        terminal_nine_cut_plan_reference=plan_reference,
        original_native_and_retirement_coverage_reference=coverage_reference,
        selected_missing_material_delta_reference=delta_reference,
        expected_native_frames=plan['expected_native_frames'],
        expected_actual_exposures=plan['expected_canonical_exposures'],
        current_measured_source_review_complete_is_not_accuracy_acceptance=True,
        own_native_QA_is_not_new_canonical_FIT=True,
        whole_programme_source_fit_ready=False, solver_started=False)
    if selected_path.exists():
        if json.loads(selected_path.read_text(encoding='utf8')) != selected_manifest:
            raise ValueError('Earlier selected cut inputs bind different source bytes')
    else:
        with selected_path.open('x',encoding='utf8') as saved:
            json.dump(selected_manifest,saved,ensure_ascii=False,separators=(',', ':'))
    print(json.dumps(dict(stage='all_nine_measured_cut_inputs_ready',
        selected_cut_inputs_reference=reference(selected_path)),separators=(',', ':')),flush=True)
    pool = merge_disjoint_strict_absolute_source_packs(selected,
        out/'full_programme_measured_source_pool.npz', additional_references=supplemental_references)
    if (not pool['all_exposures_processed'] or
            pool['processed_actual_exposures'] != plan['expected_canonical_exposures']):
        raise ValueError('The final measured source pool lost canonical exposures')
    with np.load(pool['path'], allow_pickle=False) as saved:
        sigma = saved['sigma_px']
        finite = np.isfinite(sigma) & (sigma > 0)
        raw = dict(points=len(sigma), finite_source_working_sigma_points=int(finite.sum()),
            missing_source_working_sigma_points=int((~finite).sum()))
    result = dict(schema='rinkspace.full-programme-measured-source-pool-assembly.v1',
        terminal_nine_cut_plan_reference=plan_reference,
        original_native_and_retirement_coverage_reference=coverage_reference,
        selected_missing_material_delta_reference=delta_reference,
        user_and_validation_authorities=list(authorities), cut_assemblies=cut_receipts,
        final_absolute_source_pack=pool, source_quantity=raw,
        expected_native_frames=plan['expected_native_frames'],
        expected_actual_exposures=plan['expected_canonical_exposures'],
        old_detector_points_retired_not_appended=True,
        native_alias_pixels_retained_in_own_review_parts_not_new_canonical_points=True,
        missing_material_or_precision_not_converted_to_FIT=True,
        whole_programme_source_fit_ready=False, manual_visual_review_complete=False,
        accuracy_accepted=False, training_started=False, solver_started=False)
    path = out/'full_programme_measured_source_pool_assembly.json'
    with path.open('x', encoding='utf8') as saved:
        json.dump(result, saved, ensure_ascii=False, separators=(',', ':'))
    return dict(receipt_reference=reference(path), source_quantity=raw,
        final_absolute_source_pack=pool, solver_started=False, accuracy_accepted=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--deltas', type=Path, required=True)
    parser.add_argument('--coverage', type=Path, required=True)
    parser.add_argument('--authority', type=Path, action='append', default=[])
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(assemble(reference(args.plan), reference(args.deltas), reference(args.coverage),
        args.output_dir, authorities=[reference(path) for path in args.authority]),
        ensure_ascii=False, separators=(',', ':')), flush=True)


if __name__ == '__main__':
    main()
