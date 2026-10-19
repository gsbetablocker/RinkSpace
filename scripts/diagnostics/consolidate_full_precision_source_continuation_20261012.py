"""Close the nine source cuts through the existing strict pack/index APIs.

Only the four retained packs and five successful successor packs enter the
merge. The old detector archive supplies retirement locators, never fallback
UV. This script does not fit, train, or mark manual visual review complete.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]

from calibration.program_tracking.strict_model_scope import authenticate_source_reference, file_sha
from calibration.program_tracking.strict_source_pack_merge import merge_strict_absolute_source_packs
from calibration.program_tracking.strict_source_review_coverage import consolidate_source_review_coverage
from scripts.diagnostics.prepare_full_precision_source_continuation_20261012 import WORK, INVENTORY


def _reference(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=file_sha(path))


def source_closeout_plan(manifest_path, batch_dirs):
    manifest_reference = _reference(manifest_path)
    manifest = json.loads(Path(manifest_path).read_text(encoding='utf8'))
    if manifest.get('schema') != 'rinkspace.exact-source-cut-batch.v1':
        raise ValueError('Frozen exact-source continuation manifest required')
    retained = manifest.get('preserved_completed_source_cuts', [])
    if len(retained) != 4 or len(manifest['cuts']) != 5:
        raise ValueError('This continuation retains four cuts and processes five cuts')
    common = manifest['common_arguments']
    identity_reference = next(reference for reference in manifest['references']
        if Path(reference['path']).resolve() == Path(common[common.index('--identity')+1]).resolve())
    authenticate_source_reference(identity_reference)
    identity = json.loads(Path(identity_reference['path']).read_text(encoding='utf8'))
    camera_reference = dict(path=identity['camera_path'], sha256=identity['camera_sha256'])
    authenticate_source_reference(camera_reference)
    with np.load(camera_reference['path'], allow_pickle=False) as camera:
        native = camera['pts'].copy()
        actual = camera['exposure_representative_pts'].copy()
        camera_metadata = json.loads(str(camera['metadata_json'].item()))
    source = dict(video_sha256=camera_metadata['source_sha256'],
        time_base=camera_metadata['source_time_base'], pixel_domain='native1920x1080')
    receipts = {}
    batch_references = []
    for directory in batch_dirs:
        path = Path(directory)/'batch_receipt.json'
        if not path.exists():
            continue
        reference = _reference(path)
        batch = json.loads(path.read_text(encoding='utf8'))
        if batch.get('schema') != 'rinkspace.exact-source-cut-batch-receipt.v1':
            raise ValueError('Source-cut terminal batch receipt required')
        # A later explicit retry may cite a reduced successor manifest. Keep
        # its immutable receipt and authenticate that manifest as provenance.
        authenticate_source_reference(batch['manifest_reference'])
        batch_references.append(reference)
        for terminal in batch['cuts']:
            if terminal.get('exit_code') == 0:
                receipts[terminal['shot_id']] = terminal
    pending = []
    entries = []
    cuts = sorted([*retained, *manifest['cuts']], key=lambda cut:cut['start_pts'])
    seen = set()
    for cut in cuts:
        identifier = cut['shot_id']
        mask = (native >= cut['start_pts']) & (native < cut['stop_pts'])
        cut_native = native[mask]
        cut_actual = np.unique(actual[mask])
        if (not len(cut_native) or len(cut_native) != cut['native_count']
                or len(cut_actual) != cut['canonical_count']
                or seen.intersection(map(int,cut_native))):
            raise ValueError('The nine cuts must partition the exact source clock')
        seen.update(map(int,cut_native))
        if 'absolute_inputs' in cut:
            reference = cut['absolute_inputs']
            alias_reference = cut['native_alias_review']
            retirement_reference = cut['old_retirement_indices']
            retained_cut = True
        else:
            terminal = receipts.get(identifier)
            if terminal is None:
                pending.append(dict(shot_id=identifier, reason='no_successful_terminal_batch_receipt',
                    expected_filename=re.sub(r'[^A-Za-z0-9_.-]', '_',identifier)+'.npz'))
                continue
            if any(terminal[key] != cut[key] for key in ('start_pts','stop_pts','native_count','canonical_count')):
                raise ValueError('Successor cut receipt changed its exact source range')
            reference = dict(path=terminal['output_path'], sha256=terminal['sha256'])
            path = Path(reference['path'])
            alias_reference = _reference(path.with_name(path.stem+'_native_alias_review.npz'))
            retirement_reference = _reference(path.with_name(path.stem+'_retired_old_source_indices.npz'))
            retained_cut = False
        for ref in (reference,alias_reference,retirement_reference):
            authenticate_source_reference(ref)
        with np.load(reference['path'],allow_pickle=False) as pack:
            metadata = json.loads(str(pack['metadata_json'].item()))
            if (metadata.get('source') != source or metadata.get('legacy_family_fallback') != 'forbidden'
                    or metadata.get('physically_compacted_approved_pixels_only') is not True
                    or not np.array_equal(metadata['expected_actual_pts'],np.unique(actual))
                    or not np.array_equal(metadata['processed_actual_pts'],cut_actual)):
                raise ValueError('A source pack has an incomplete or differing canonical cut clock')
            frame_lookup = {int(pts):index for index,pts in enumerate(pack['source_frame_actual_pts'])}
            for pts in cut_actual:
                index = frame_lookup[int(pts)]
                if int(pack['source_frame_native_pts'][index]) != int(pts) or len(pack['source_frame_BGR_sha256'][index]) != 64:
                    raise ValueError('A processed canonical exposure lacks its exact BGR binding')
            originals = metadata['original_references']
            precision = [ref for ref in originals if Path(ref['path']).name.endswith('_source_precision_pending.npz')]
            if not retained_cut:
                graph_dir = Path(reference['path']).with_name(Path(reference['path']).stem+'_pending_material_graphs')
                sidecars = list(graph_dir.glob('*_source_precision_pending.npz'))
                if len(sidecars) != 1:
                    raise ValueError('Each successor cut must preserve its precision pending sidecar')
                sidecar = _reference(sidecars[0])
                if sidecar not in precision:
                    raise ValueError('The source pack does not hash-reference its precision pending sidecar')
            precision_pending_candidates = 0
            for ref in precision:
                authenticate_source_reference(ref)
                with np.load(ref['path'],allow_pickle=False) as pending_pack:
                    pending_metadata = json.loads(pending_pack['pending_actual_source_receipts_utf8'].tobytes().decode('utf8'))
                if (pending_metadata.get('schema') != 'rinkspace.final-q50-source-precision-pending.v1'
                        or pending_metadata.get('missing_nodes_forged') is not False
                        or pending_metadata.get('source_points_not_admitted') is not True):
                    raise ValueError('Precision pending receipts must retain missing actual support without admission')
                precision_pending_candidates += sum(len(frame['pending_actual_source_candidates'])
                    for frame in pending_metadata['source_precision_receipts'])
        with np.load(alias_reference['path'],allow_pickle=False) as alias:
            alias_metadata = json.loads(str(alias['metadata_json'].item()))
            representative = cut_native == actual[mask]
            if (alias_metadata.get('source') != source
                    or alias_metadata.get('new_FIT_observations_created') is not False
                    or not np.array_equal(alias['native_pts'],cut_native)
                    or not np.array_equal(alias['actual_pts'],actual[mask])
                    or not np.array_equal(alias['representative'],representative)
                    or not np.array_equal(alias['alias_native_pts'],cut_native[~representative])
                    or not np.array_equal(alias['alias_actual_pts'],actual[mask][~representative])):
                raise ValueError('A source cut omitted or changed exact native alias clock coverage')
            statuses = alias['canonical_candidate_status']
            alias_pending = int(np.sum(statuses == 0))
        entries.append(dict(shot_id=identifier, absolute_inputs=reference,
            native_alias_review=alias_reference, old_retirement_indices=retirement_reference,
            retained_original_path=retained_cut, precision_pending_references=precision,
            native_frames=len(cut_native), canonical_exposures=len(cut_actual),
            alias_identity_pending_candidates=alias_pending,
            precision_pending_candidates=precision_pending_candidates,
            source_uncertainty_ready_for_fit=metadata['source_uncertainty_ready_for_fit']))
    if seen != set(map(int,native)):
        raise ValueError('The requested nine cuts omit original native source frames')
    return dict(schema='rinkspace.strict-source-nine-cut-closeout-plan.v1',
        manifest_reference=manifest_reference, batch_receipt_references=batch_references,
        identity_reference=identity_reference, camera_clock_reference=camera_reference,
        entries=entries, pending_cuts=pending, ready=len(entries)==9 and not pending,
        expected_native_frames=len(native), expected_canonical_exposures=len(np.unique(actual)),
        source_processing_is_not_manual_visual_completion=True,
        old_detector_pixels_are_not_merge_inputs=True, additional_board_delta_included=False,
        whole_programme_source_fit_ready=False, solver_started=False)


def consolidate(plan, output_dir):
    if not plan['ready']:
        raise ValueError('All nine cuts must have complete successful source receipts before closeout')
    output_dir = Path(output_dir).resolve()
    source_path = output_dir/'full_programme_strict_source_9cut.npz'
    coverage_path = output_dir/'full_programme_strict_source_9cut_coverage.npz'
    receipt_path = output_dir/'full_programme_strict_source_9cut_closeout.json'
    if any(path.exists() for path in (source_path,coverage_path,receipt_path)):
        raise FileExistsError('Preserve an earlier nine-cut source closeout')
    inputs = [entry['absolute_inputs'] for entry in plan['entries']]
    source_receipt = merge_strict_absolute_source_packs(inputs,source_path)
    if (source_receipt['processed_actual_exposures'] != plan['expected_canonical_exposures']
            or source_receipt['all_exposures_processed'] is not True):
        raise ValueError('Merged canonical source clock is incomplete')
    coverage_receipt = consolidate_source_review_coverage(
        [entry['native_alias_review'] for entry in plan['entries']],
        [entry['old_retirement_indices'] for entry in plan['entries']],
        identity_reference=plan['identity_reference'],output_path=coverage_path)
    if (coverage_receipt['processed_native_frames'] != plan['expected_native_frames']
            or coverage_receipt['all_native_frames_RGB_processed'] is not True):
        raise ValueError('Merged native source clock is incomplete')
    with np.load(source_path,allow_pickle=False) as pack:
        metadata = json.loads(str(pack['metadata_json'].item()))
        unresolved = int(np.sum(~np.isfinite(pack['sigma_px'])))
    result = dict(plan, schema='rinkspace.strict-source-nine-cut-closeout-receipt.v1',
        absolute_source_pack=source_receipt, native_and_retirement_coverage=coverage_receipt,
        source_uncertainty_ready_for_fit=metadata['source_uncertainty_ready_for_fit'],
        admitted_points_without_finite_source_working_sigma=unresolved,
        precision_pending_candidates=sum(entry['precision_pending_candidates'] for entry in plan['entries']),
        manual_visual_review_complete=False, accuracy_accepted=False,
        missing_material_or_precision_not_converted_to_FIT=True,
        old_detector_pixel_fallback_added=False, training_started=False, solver_started=False)
    output_dir.mkdir(parents=True,exist_ok=True)
    with receipt_path.open('x',encoding='utf8') as saved:
        json.dump(result,saved,ensure_ascii=False,separators=(',',':'))
    return dict(receipt_reference=_reference(receipt_path), absolute_source_pack=source_receipt,
        native_and_retirement_coverage=coverage_receipt,
        source_uncertainty_ready_for_fit=result['source_uncertainty_ready_for_fit'],
        whole_programme_source_fit_ready=False,
        manual_visual_review_complete=False,accuracy_accepted=False,solver_started=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest',type=Path,
        default=INVENTORY/'source_full_precision_continuation5_FROZEN_v1.json')
    parser.add_argument('--batch-dir',type=Path,action='append')
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument('--check-only',action='store_true')
    operation.add_argument('--output-dir',type=Path)
    args = parser.parse_args()
    plan = source_closeout_plan(args.manifest,
        args.batch_dir or [WORK/'source_full_precision_continuation5_v1'])
    result = plan if args.check_only else consolidate(plan,args.output_dir)
    print(json.dumps(result,ensure_ascii=False,separators=(',',':')),flush=True)


if __name__ == '__main__':
    main()
