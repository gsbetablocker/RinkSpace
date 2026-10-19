"""Export a derivative strict pack after exact named-identity quarantine.

The original pack, source clock, proof table and graph references remain
immutable. Pending pixels retain a separate complete numeric evidence bank.
This module never reviews pixels, supplies identities, changes weights, or fits.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from .compact_source_json import pack_source_json
from .strict_absolute_source import StrictAbsolutePackWriter, StrictAbsoluteSourcePack
from .strict_model_scope import ExactSourceScopeReview, REVIEW_SCHEMA, authenticate_source_reference, file_sha
from .strict_source_pack_merge import _slice_row


FILTER_SCHEMA = 'rinkspace.exact-source-named-identity-pending-filter.v1'
POINT_FIELDS = ('uv', 'sigma_px', 'source_sides_uv', 'raw_candidate_uv',
    'source_side_quantiles_uv', 'original_point_indices', 'source_proof_template_index')
FRAME_FIELDS = ('source_frame_actual_pts', 'source_frame_native_pts', 'source_frame_BGR_sha256')


def _reference(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=file_sha(path))


def _literal(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _literal(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_literal(v) for v in value]
    return value


def _slice_current_row(row, indices):
    """Reuse the current slicer and retain additional measured quantiles."""
    result = _slice_row(row, indices)
    uncertainty = result.get('source_readout_uncertainty', {})
    original = row.get('source_readout_uncertainty', {})
    name = 'actual_side_gradient_quantiles_uv'
    if name in original:
        value = np.asarray(original[name])
        if value.shape != (len(row['uv']), 2, 3, 2):
            raise ValueError('Source uncertainty quantile count differs from current points')
        uncertainty[name] = value[np.asarray(indices, np.int64)].copy()
    return result


def _exact_equal(first, second):
    first, second = np.asarray(first), np.asarray(second)
    return (first.dtype == second.dtype and first.shape == second.shape
        and first.tobytes() == second.tobytes())


def current_row_source_binding(pack, ri):
    """Derive only authenticated source/clock fields, never a missing scope."""
    row=pack.rows[ri];actual=row['actual_pts'];frame=pack.frame_bindings[actual]
    if (row['native_pts']!=frame['native_pts']
            or row['decoded_BGR_sha256']!=frame['decoded_BGR_sha256']):
        raise ValueError('Row differs from the exact saved native exposure')
    lo,hi=map(int,pack.values['row_offsets'][ri:ri+2])
    paths=set()
    for code in np.unique(pack.values['source_proof_template_index'][lo:hi]):
        proof=pack.proof_templates[int(code)]
        image=proof['source_image_evidence']
        if (proof['actual_pts']!=actual or proof['native_pts']!=frame['native_pts']
                or proof['decoded_BGR_sha256']!=frame['decoded_BGR_sha256']
                or image['sha256']!=pack.metadata['source']['video_sha256']):
            raise ValueError('Original source proof differs from the row clock')
        paths.add(str(Path(image['path']).resolve()))
    if len(paths)!=1:raise ValueError('One exact original image source required')
    return dict(source_sha256=pack.metadata['source']['video_sha256'],
        source_path=paths.pop(),time_base=pack.metadata['source']['time_base'],
        actual_pts=actual,native_pts=frame['native_pts'],shot_id=row['shot_id'],
        decoded_native_BGR_sha256=frame['decoded_BGR_sha256'])


def _binding_matches(supplied, expected):
    if not isinstance(supplied,dict):return False
    aliases={'source_sha256':'video_sha256','source_path':'path',
        'decoded_native_BGR_sha256':'decoded_BGR_sha256'}
    for name,wanted in expected.items():
        value=supplied.get(name,supplied.get(aliases.get(name,'')))
        if name=='shot_id' and value is None:continue
        if name=='source_path':
            if not value or Path(value).resolve()!=Path(wanted).resolve():return False
        elif value!=wanted:return False
        if name in supplied and aliases.get(name) in supplied:
            other=supplied[aliases[name]]
            if (Path(other).resolve()!=Path(wanted).resolve() if name=='source_path' else other!=wanted):return False
    return True


def _pending_indices(pack, group):
    pending = {}
    for entry in group['entries']:
        ri = entry['pack_row_index']
        if type(ri) is not int or not 0 <= ri < len(pack.rows):
            raise ValueError('Pending row index is outside the authenticated pack')
        row = pack.rows[ri]
        if row['split'] != 'fit':
            raise ValueError('A named-FIT quarantine cannot discard or change HELD')
        expected_binding=current_row_source_binding(pack,ri)
        if 'source_binding' in row and not _binding_matches(row['source_binding'],expected_binding):
            raise ValueError('Header own source binding differs from numeric clock/proofs')
        if (row['physical_material_id'] != entry['physical_material_id']
                or not _binding_matches(entry['source_binding'],expected_binding)
                or entry.get('exclude_from_final_named_FIT') is not True
                or entry.get('status') != 'source_material_measured_named_identity_pending'
                or not isinstance(entry.get('reason'), str) or not entry['reason']):
            raise ValueError('Pending material, own source binding or reason differs')
        lo, hi = map(int, pack.values['row_offsets'][ri:ri+2])
        absolute = entry['pack_point_indices']
        if (not absolute or any(type(i) is not int or not lo <= i < hi for i in absolute)
                or absolute != sorted(set(absolute))):
            raise ValueError('Pending exact point indices must be unique and inside their row')
        local = np.asarray(absolute, np.int64)-lo
        for key in ('original_candidate_indices', 'source_query_indices'):
            supplied = entry.get(key)
            if supplied is None:
                if key in row:
                    raise ValueError('A current source locator must not be omitted')
                continue
            values = np.asarray(row.get(key, []))
            # The fact inventory may preserve all locators of the original row
            # or only the exact pending subset; both are checked against it.
            wanted = values if len(supplied) == len(values) else values[local]
            if not np.array_equal(np.asarray(supplied), wanted):
                raise ValueError('Pending current source candidate/query locator differs')
        current = pending.setdefault(ri, set())
        if current.intersection(absolute):
            raise ValueError('Pending source point is listed twice')
        current.update(absolute)
    total = sum(map(len, pending.values()))
    if total != group['pending_named_point_count']:
        raise ValueError('Pending named point total differs from exact inventory')
    return pending


def export_exact_named_identity_pending_pack(source_reference, filter_reference, output_path):
    """Preserve exact non-pending FIT and all HELD through the public writer.

    Inputs are authenticated ``{path, sha256}`` references. The filter matches
    an exact source pack and exact rows/point indices, never a nearest point or
    material-wide deletion. It cannot mark new points or identities admissible.
    """
    source_path, _ = authenticate_source_reference(source_reference)
    fact_path, _ = authenticate_source_reference(filter_reference)
    output_path = Path(output_path).resolve()
    bank_path = output_path.with_name(output_path.stem+'.named_identity_pending.npz')
    receipt_path = output_path.with_name(output_path.stem+'.named_identity_filter_receipt.json')
    if (output_path == source_path or any(p.exists() for p in (output_path, bank_path, receipt_path))):
        raise ValueError('Preserve original and previously exported source artifacts')
    fact = json.loads(fact_path.read_text(encoding='utf-8-sig'))
    if (fact.get('schema') != FILTER_SCHEMA
            or fact.get('retain_pending_source_readouts_full_literal_evidence') is not True
            or fact.get('original_packs_or_alias_clock_evidence_must_not_be_changed') is not True):
        raise ValueError('Exact immutable-source named identity quarantine required')
    matching = [g for g in fact['source_pack_groups'] if
        Path(g['source_pack_reference']['path']).resolve() == source_path
        and g['source_pack_reference']['sha256'] == source_reference['sha256']]
    if len(matching) != 1:
        raise ValueError('Filter must bind exactly one matching immutable source pack')
    group = matching[0]
    witnesses = group.get('source_review_witnesses', [])
    if not witnesses:
        raise ValueError('Named identity quarantine requires authenticated source review facts')
    for reference in witnesses:
        authenticate_source_reference(reference)
    pack = StrictAbsoluteSourcePack(source_path, expected_sha256=source_reference['sha256'])
    pending = _pending_indices(pack, group)
    pending_points = np.asarray(sorted(i for values in pending.values() for i in values), np.int64)
    keep = np.ones(len(pack.values['uv']), bool)
    keep[pending_points] = False
    kept_points = np.flatnonzero(keep)
    source_rows = sorted(pending)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bank = {key: pack.values[key][pending_points].copy() for key in POINT_FIELDS}
    bank.update({key: pack.values[key].copy() for key in FRAME_FIELDS})
    bank.update(original_pack_point_indices=pending_points,
        original_pack_row_indices=np.asarray(source_rows, np.int64),
        original_rows_json=pack_source_json([pack.rows[ri] for ri in source_rows]),
        original_source_proof_templates_json=pack_source_json(pack.proof_templates),
        pending_filter_json=pack_source_json(group),
        metadata_json=json.dumps(dict(schema='rinkspace.source-named-identity-pending-evidence-bank.v1',
            source_pack=source_reference, filter_reference=filter_reference,
            points_not_FIT_admitted=True, source_pixel_values_changed=False), separators=(',', ':')))
    np.savez_compressed(bank_path, **bank)
    original_refs = {}
    for reference in [*pack.metadata['original_references'], source_reference, filter_reference,
            *witnesses, _reference(bank_path), _reference(__file__)]:
        key = str(Path(reference['path']).resolve())
        if key in original_refs and original_refs[key]['sha256'] != reference['sha256']:
            raise ValueError('One source authority is bound to conflicting bytes')
        original_refs[key] = dict(reference, path=key)
    writer = StrictAbsolutePackWriter(output_path, source=pack.metadata['source'],
        expected_actual_pts=pack.metadata['expected_actual_pts'],
        original_references=list(original_refs.values()),
        legacy_pixel_families=pack.metadata['legacy_pixel_families'])
    writer.exclusions=deepcopy(pack.exclusions)
    for entry in group['entries']:
        writer.exclusions.append(dict(deepcopy(entry),
            source_pack_reference=deepcopy(source_reference),
            pending_filter_reference=deepcopy(filter_reference),
            pending_evidence_bank_reference=_reference(bank_path),
            source_review_witnesses=deepcopy(witnesses),
            source_pixel_identity_preserved=True,
            source_pixel_values_changed=False))
    # Preserve the complete historical proof table and the original numeric
    # template IDs, including unused proofs for quarantined source pixels.
    writer.proof_templates = deepcopy(pack.proof_templates)
    writer.proof_lookup = {json.dumps(proof, sort_keys=True, separators=(',', ':')): i
        for i, proof in enumerate(writer.proof_templates)}
    if len(writer.proof_lookup) != len(writer.proof_templates):
        raise ValueError('Original proof table has duplicate literal templates')
    for pts, binding in pack.frame_bindings.items():
        writer._register_frame(pts, binding['native_pts'], binding['decoded_BGR_sha256'])
    for pts in pack.metadata['processed_actual_pts']:
        writer.mark_exposure_processed(pts)
    retained_rows = []
    for ri, header in enumerate(pack.rows):
        lo, hi = map(int, pack.values['row_offsets'][ri:ri+2])
        indices = np.flatnonzero(keep[lo:hi])
        if not len(indices):
            continue
        row = dict(deepcopy(header), **{key: pack.values[key][lo:hi].copy()
            for key in POINT_FIELDS if key != 'source_proof_template_index'})
        row = _slice_current_row(row, indices)
        proof_ids = pack.values['source_proof_template_index'][lo:hi][indices]
        proofs = [dict(deepcopy(pack.proof_templates[int(code)]), uv=point.tolist())
            for code, point in zip(proof_ids, row['uv'])]
        review = ExactSourceScopeReview(dict(schema=REVIEW_SCHEMA,
            source=pack.metadata['source'], reviewed_endpoints=proofs))
        if writer.add_row(row, review) != len(indices):
            raise ValueError('An unchanged source pixel lost its authenticated admission proof')
        retained_rows.append(dict(source_row_index=ri, kept_original_pack_point_indices=(lo+indices).tolist()))
    exported = writer.finish()
    result = StrictAbsoluteSourcePack(output_path, expected_sha256=exported['sha256'])
    for name in POINT_FIELDS:
        if not _exact_equal(result.values[name], pack.values[name][kept_points]):
            raise ValueError('Retained literal source array changed: '+name)
    for name in FRAME_FIELDS:
        if not _exact_equal(result.values[name], pack.values[name]):
            raise ValueError('Source exposure/native clock changed: '+name)
    if json.dumps(result.proof_templates, sort_keys=True, separators=(',', ':')) != json.dumps(pack.proof_templates, sort_keys=True, separators=(',', ':')):
        raise ValueError('Historical source proof table changed')
    if result.exclusions!=writer.exclusions:
        raise ValueError('Original and precise pending exclusions changed')
    for output_row, record in zip(result.rows, retained_rows):
        prior = pack.rows[record['source_row_index']]
        if prior['split'] == 'held' and json.dumps(output_row, sort_keys=True, separators=(',', ':')) != json.dumps(prior, sort_keys=True, separators=(',', ':')):
            raise ValueError('HELD header changed')
    if file_sha(source_path) != source_reference['sha256'] or file_sha(fact_path) != filter_reference['sha256']:
        raise ValueError('Immutable input source evidence changed during export')
    receipt = dict(exported, source_pack_reference=source_reference, pending_filter_reference=filter_reference,
        pending_evidence_bank_reference=_reference(bank_path), pending_named_points=len(pending_points),
        input_points=len(pack.values['uv']), retained_original_pack_point_indices=kept_points.tolist(),
        retained_rows=retained_rows, all_retained_numeric_values_bitexact=True,
        full_original_proof_table_and_template_ids_unchanged=True,
        full_actual_native_BGR_clock_unchanged=True, HELD_unchanged=True,
        original_source_bank_and_graph_references_retained=True, source_admission_gate_relaxed=False,
        original_exclusions_preserved=True, precise_pending_exclusions_with_bank_and_fact_references_retained=True,
        new_independent_accuracy_holdout_claimed=False,
        original_pack_modified=False, manual_visual_coverage_complete=False, solver_started=False)
    with receipt_path.open('x', encoding='utf8') as stream:
        json.dump(_literal(receipt), stream, ensure_ascii=False, indent=2)
    return dict(receipt, receipt_reference=_reference(receipt_path))


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Export an exact named-identity-pending derivative; no fit')
    parser.add_argument('--pack', nargs=2, metavar=('PATH', 'SHA256'), required=True)
    parser.add_argument('--filter', nargs=2, metavar=('PATH', 'SHA256'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    receipt = export_exact_named_identity_pending_pack(
        dict(path=args.pack[0], sha256=args.pack[1]),
        dict(path=args.filter[0], sha256=args.filter[1]), args.output)
    print(json.dumps({k: receipt[k] for k in ('path', 'sha256', 'points', 'rows',
        'pending_named_points', 'receipt_reference')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
