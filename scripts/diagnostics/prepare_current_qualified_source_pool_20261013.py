"""Bind current exact point filters once to each original nine-cut source pack.

This prepares the measured review pool. It does not admit missing source
support, promote own-native QA pixels to canonical observations, or solve.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import gc
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]

from calibration.program_tracking.strict_model_scope import authenticate_source_reference, file_sha
from calibration.program_tracking.strict_source_named_identity_filter_with_exclusions import (
    FILTER_SCHEMA, export_exact_named_identity_pending_pack,
)


def reference(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=file_sha(path))


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def identity(ref):
    return str(Path(ref['path']).resolve()), ref['sha256']


def write_once(path, value):
    path = Path(path)
    if path.exists():
        if read(path) != value:
            raise ValueError('Existing preparation output identifies different input bytes: ' + str(path))
        return
    with path.open('x', encoding='utf8') as saved:
        json.dump(value, saved, ensure_ascii=False, separators=(',', ':'))


def prepare_logical(review_root, output_dir):
    """Reference original arrays and exact pending masks without copying pixels.

    The consumer must apply each original index mask before declaring FIT.
    This is usable only with the cited duplicate/operator preflight audit.
    """
    import numpy as np
    work, out = Path(review_root).resolve(), Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    named = dict(
        plan=work/'nine_cut_terminal_core_coverage_20261013_v1/terminal_nine_cut_plan.json',
        coverage=work/'nine_cut_terminal_core_coverage_20261013_v1/terminal_nine_cut_core_coverage.npz',
        gray=work/'all_nine_current_gray_source_qualified_selected_refs_20261013_v1.json',
        board=work/'validated_board_delta_selected_refs_final_20261013_v1.json',
        f1=work/'original_F1_final_current_source_qualification_refs_20261013_v1.json',
        f2=work/'034_004_original_F2_current_point_and_native_qualification_refs_20261013_v1.json',
        audit=work/'original_terminal_plus_current_gray_board_merge_operator_audit_20261013_v1.json',
    )
    refs = {key:reference(path) for key,path in named.items()}
    for ref in refs.values():
        authenticate_source_reference(ref)
    plan, gray, board, audit = (read(named[k]) for k in ('plan','gray','board','audit'))
    filters_by_source = {}
    for kind in ('f1','f2'):
        for record in read(named[kind])['records']:
            fr = record['current_named_pending_filter_reference']
            authenticate_source_reference(fr)
            fact = read(fr['path'])
            if (fact['schema'] != FILTER_SCHEMA or
                    not fact['retain_pending_source_readouts_full_literal_evidence'] or
                    not fact['original_packs_or_alias_clock_evidence_must_not_be_changed']):
                raise ValueError('Literal current original-index pending filter required')
            sr = record['source_pack_reference']
            matches = [g for g in fact['source_pack_groups'] if identity(g['source_pack_reference']) == identity(sr)]
            if len(matches) != 1:
                raise ValueError('A current point mask must bind the exact original base')
            group = matches[0]
            for witness in group['source_review_witnesses']:
                authenticate_source_reference(witness)
            filters_by_source.setdefault(identity(sr), []).append(dict(
                filter_reference=fr, qualification_manifest_reference=refs[kind], group=group))
    records, seen, total_pending = [], set(), 0
    base_keys = {identity(e['absolute_inputs']) for e in plan['entries']}
    if not set(filters_by_source).issubset(base_keys):
        raise ValueError('A current original point mask belongs outside the terminal bases')
    for entry in plan['entries']:
        sr = entry['absolute_inputs']
        groups = filters_by_source.get(identity(sr), [])
        exact = set()
        for record in groups:
            indices = [i for value in record['group']['entries'] for i in value['pack_point_indices']]
            if len(indices) != record['group']['pending_named_point_count'] or exact.intersection(indices):
                raise ValueError('Joint F1/F2 original mask duplicates an exact point')
            exact.update(indices)
        total_pending += len(exact)
        records.append(dict(cut=entry['shot_id'].rsplit('_',1)[-1], shot_id=entry['shot_id'],
            source_kind='original_base_with_current_joint_exact_pending_mask', absolute_inputs=sr,
            current_original_pending_filters=groups,
            original_pack_point_indices_to_exclude_from_named_FIT=sorted(exact),
            exact_pending_points=len(exact), original_base_mask_must_be_applied_once=True))
        seen.add(identity(sr))
    for kind, manifest in (('gray',gray),('board',board)):
        for record in manifest['records']:
            sr = record['absolute_inputs']
            if identity(sr) in seen:
                raise ValueError('A source pack is selected twice')
            seen.add(identity(sr))
            selected = dict(cut=record['cut'], source_kind=kind, absolute_inputs=sr,
                current_original_pending_filters=[], original_pack_point_indices_to_exclude_from_named_FIT=[],
                exact_pending_points=0)
            for field in ('receipt','native_alias_readouts'):
                if field in record:
                    selected[field]=record[field]
            records.append(selected)
    source = expected = None
    total_points = 0
    for record in records:
        sr=record['absolute_inputs'];authenticate_source_reference(sr)
        with np.load(sr['path'],allow_pickle=False) as z:
            metadata=json.loads(str(z['metadata_json'].item()))
            offsets=z['row_offsets']
        if source is None:
            source,expected=metadata['source'],metadata['expected_actual_pts']
        if (source!=metadata['source'] or expected!=metadata['expected_actual_pts'] or
                metadata.get('legacy_family_fallback')!='forbidden' or
                metadata.get('physically_compacted_approved_pixels_only') is not True):
            raise ValueError('All logical source packets require the same authenticated strict source clock')
        count=int(offsets[-1]);total_points+=count
        if any(type(i) is not int or i<0 or i>=count
                for i in record['original_pack_point_indices_to_exclude_from_named_FIT']):
            raise ValueError('A current original index mask lies outside its literal source array')
        record['source_points_before_current_pending_filter']=count
        record['source_rows']=len(offsets)-1
    result=dict(schema='rinkspace.logical-current-qualified-source-inputs.v1',records=records,
        original_authorities=refs, model_identity_reference=plan['identity_reference'],
        original_native_and_retirement_coverage_reference=refs['coverage'],
        source=source,expected_actual_pts=expected,expected_native_frames=plan['expected_native_frames'],
        expected_actual_exposures=plan['expected_canonical_exposures'],
        source_points_before_current_pending_filters=total_points,
        original_base_exact_pending_points=total_pending,
        qualified_source_points_before_support_partition=total_points-total_pending,
        operator_and_duplicate_preflight_reference=refs['audit'],
        exact_source_point_masks_mandatory_before_consumer_FIT=True,
        original_base_and_filtered_replacement_never_both_selected=True,
        current_gray_pending_evidence_reference=refs['gray'],
        own_native_alias_readouts_are_QA_only_not_additional_canonical_points=True,
        complete_HELD_propagation_per_source_material_and_feature_required=True,
        missing_support_not_FIT_and_original_sigma_not_modified=True,
        source_processing_is_not_manual_visual_completion=True,
        accuracy_accepted=False,solver_started=False)
    path=out/'logical_qualified_sources.json';write_once(path,result)
    return dict(logical_qualified_source_inputs_reference=reference(path),packs=len(records),
        input_points=total_points,exact_pending_points=total_pending,
        current_points=total_points-total_pending,solver_started=False)


def prepare(review_root, output_dir):
    work = Path(review_root).resolve()
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    named = dict(
        plan=work / 'nine_cut_terminal_core_coverage_20261013_v1/terminal_nine_cut_plan.json',
        coverage=work / 'nine_cut_terminal_core_coverage_20261013_v1/terminal_nine_cut_core_coverage.npz',
        gray=work / 'all_nine_current_gray_source_qualified_selected_refs_20261013_v1.json',
        board=work / 'validated_board_delta_selected_refs_final_20261013_v1.json',
        f1=work / 'original_F1_final_current_source_qualification_refs_20261013_v1.json',
        f2=work / '034_004_original_F2_current_point_and_native_qualification_refs_20261013_v1.json',
    )
    refs = {key: reference(path) for key, path in named.items()}
    for ref in refs.values():
        authenticate_source_reference(ref)
    plan, gray, board = (read(named[key]) for key in ('plan', 'gray', 'board'))
    if not plan['ready'] or len(plan['entries']) != 9 or gray['base_replacements']:
        raise ValueError('Expected the current nine-cut original base plan and unassembled replacements')
    expected_cuts = {'001', '003', '004', '034', '035', '036', '037', '038', '039'}
    if {r['cut'] for r in gray['records']} != expected_cuts:
        raise ValueError('The current qualified gray partitions omit an authorized cut')
    original_bases = {identity(entry['absolute_inputs']): entry for entry in plan['entries']}
    filters_by_source = {}
    for key in ('f1', 'f2'):
        manifest = read(named[key])
        for record in manifest['records']:
            sr = record['source_pack_reference']
            if identity(sr) not in original_bases:
                raise ValueError('Current point qualification refers to a different original base')
            fr = record['current_named_pending_filter_reference']
            authenticate_source_reference(sr)
            authenticate_source_reference(fr)
            fact = read(fr['path'])
            if (fact['schema'] != FILTER_SCHEMA or
                    fact.get('retain_pending_source_readouts_full_literal_evidence') is not True or
                    fact.get('original_packs_or_alias_clock_evidence_must_not_be_changed') is not True):
                raise ValueError('Current filter does not preserve the literal original evidence')
            matching = [g for g in fact['source_pack_groups'] if identity(g['source_pack_reference']) == identity(sr)]
            if len(matching) != 1:
                raise ValueError('Current filter must bind exactly one original base')
            group = matching[0]
            for witness in group['source_review_witnesses']:
                authenticate_source_reference(witness)
            filters_by_source.setdefault(identity(sr), []).append((fr, group, refs[key]))

    prepared_plan = deepcopy(plan)
    base_replacements = []
    pending_banks = []
    for entry in prepared_plan['entries']:
        sr = entry['absolute_inputs']
        selected_filters = filters_by_source.get(identity(sr), [])
        if not selected_filters:
            continue
        cut = entry['shot_id'].rsplit('_', 1)[-1]
        cut_dir = out / ('base_' + cut)
        cut_dir.mkdir(exist_ok=True)
        entries, witnesses, exact_indices = [], {}, set()
        for fr, group, mr in selected_filters:
            for value in group['entries']:
                indices = value['pack_point_indices']
                if exact_indices.intersection(indices):
                    raise ValueError('F1/F2 current filters overlap an original point index')
                exact_indices.update(indices)
                entries.append(deepcopy(value))
            for witness in [fr, mr, *group['source_review_witnesses']]:
                witnesses[identity(witness)] = witness
        fact = dict(schema=FILTER_SCHEMA,
            retain_pending_source_readouts_full_literal_evidence=True,
            original_packs_or_alias_clock_evidence_must_not_be_changed=True,
            source_pack_groups=[dict(source_pack_reference=sr,
                pending_named_point_count=len(exact_indices),
                source_review_witnesses=list(witnesses.values()), entries=entries)])
        filter_path = cut_dir / 'joint_current_original_F1_F2_pending_filter.json'
        write_once(filter_path, fact)
        filtered_path = cut_dir / 'joint_current_source_qualified.npz'
        receipt_path = filtered_path.with_name(filtered_path.stem + '.named_identity_filter_receipt.json')
        if receipt_path.exists():
            receipt = read(receipt_path)
            authenticate_source_reference(dict(path=receipt['path'], sha256=receipt['sha256']))
            if (receipt['source_pack_reference'] != sr or
                    receipt['pending_filter_reference'] != reference(filter_path)):
                raise ValueError('Existing filtered base has different immutable authorities')
        else:
            receipt = export_exact_named_identity_pending_pack(sr, reference(filter_path), filtered_path)
        replacement = dict(cut=cut, source_pack_reference=sr,
            derived_pack_reference=dict(path=receipt['path'], sha256=receipt['sha256']),
            pending_filter_reference=reference(filter_path),
            pending_evidence_bank_reference=receipt['pending_evidence_bank_reference'],
            exact_export_receipt_reference=reference(receipt_path),
            pending_points=len(exact_indices), retained_points=receipt['points'],
            joint_F1_F2_original_index_filter_applied_once=True)
        base_replacements.append(replacement)
        pending_banks.append(replacement)
        entry['original_absolute_inputs_before_current_point_qualification'] = sr
        entry['absolute_inputs'] = replacement['derived_pack_reference']
        entry['current_point_qualification_receipt_reference'] = replacement['exact_export_receipt_reference']
        print(json.dumps(dict(stage='base_joint_filter_complete', cut=cut,
            retained_points=receipt['points'], pending_points=len(exact_indices),
            reference=replacement['derived_pack_reference']), separators=(',', ':')), flush=True)
        gc.collect()

    deltas = []
    seen = set()
    for kind, manifest in (('board', board), ('gray', gray)):
        for record in manifest['records']:
            ref = record['absolute_inputs']
            authenticate_source_reference(ref)
            if identity(ref) in original_bases or identity(ref) in seen:
                raise ValueError('An original base or selected delta was added twice')
            seen.add(identity(ref))
            selected = dict(cut=record['cut'], source_kind=kind, absolute_inputs=ref)
            for field in ('receipt', 'native_alias_readouts'):
                if field in record:
                    authenticate_source_reference(record[field])
                    selected[field] = record[field]
            selected['own_native_readouts_are_QA_only_not_new_canonical_FIT'] = True
            deltas.append(selected)
    for record in gray['current_effective_pending_banks']:
        selected = dict(record,
            source_pack_reference=record['source_original_reference'],
            derived_pack_reference=record['absolute_inputs'],
            pending_filter_reference=record['pending_filter'],
            pending_evidence_bank_reference=record['pending_evidence_bank'])
        for field in ('source_pack_reference', 'derived_pack_reference',
                'pending_filter_reference', 'pending_evidence_bank_reference'):
            authenticate_source_reference(selected[field])
        pending_banks.append(selected)
    prepared_plan.update(current_original_point_qualification_manifest_references=[refs['f1'], refs['f2']],
        original_nine_cut_plan_reference=refs['plan'], base_replacements=base_replacements,
        original_base_and_filtered_replacement_not_both_selected=True,
        whole_programme_source_fit_ready=False, solver_started=False)
    delta_manifest = dict(schema='rinkspace.current-qualified-board-gray-source-deltas.v1',
        records=deltas, current_gray_selected_reference=refs['gray'],
        current_board_selected_reference=refs['board'],
        canonical_source_measurements_only=True, own_native_readouts_QA_only=True,
        historical_merged_pack_selected=False, solver_started=False)
    pending_manifest = dict(schema='rinkspace.current-joint-base-and-gray-pending-source-evidence.v1',
        current_effective_pending_banks=pending_banks,
        current_qualification_authorities=[refs['gray'], refs['f1'], refs['f2']],
        expected_pending_points=sum(r['pending_points'] for r in pending_banks),
        original_and_failure_history_preserved=True, solver_started=False)
    paths = dict(plan=out/'filtered_nine_cut_plan.json', deltas=out/'combined_selected_delta_refs.json',
        pending=out/'current_effective_pending_manifest.json')
    for key, value in (('plan', prepared_plan), ('deltas', delta_manifest), ('pending', pending_manifest)):
        write_once(paths[key], value)
    result = dict(schema='rinkspace.current-source-pool-preparation.v1',
        original_authorities=refs, prepared_plan_reference=reference(paths['plan']),
        combined_delta_reference=reference(paths['deltas']), pending_manifest_reference=reference(paths['pending']),
        coverage_reference=refs['coverage'], base_replacements=base_replacements,
        base_points_pending=sum(r['pending_points'] for r in base_replacements),
        current_gray_points=gray['current_source_qualified_points'],
        current_gray_pending_points=gray['current_exact_pending_points'],
        own_native_not_canonical_FIT=True, solver_started=False)
    write_once(out/'preparation_receipt.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--review-root', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--logical-only',action='store_true')
    args = parser.parse_args()
    action=prepare_logical if args.logical_only else prepare
    print(json.dumps(action(args.review_root, args.output_dir), ensure_ascii=False,
        separators=(',', ':')), flush=True)
