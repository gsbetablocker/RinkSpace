"""Audit literal reviewed-point coverage in a completed source training ledger."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from calibration.program_tracking.strict_model_scope import authenticate_source_reference, file_sha


def reference(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=file_sha(path))


def audit(report_path, output_path, crop_size):
    report = json.loads(report_path.read_text(encoding='utf8'))
    if report.get('schema') != 'rinkspace.current-logical-source-vision-training-consumption.v1':
        raise ValueError('A completed current-source training report is required')
    for key in ('bundle_reference', 'consumption_reference', 'candidate_checkpoint', 'candidate_model'):
        authenticate_source_reference(report[key])
    bundle = json.loads(Path(report['bundle_reference']['path']).read_text(encoding='utf8'))
    ledger = [json.loads(line) for line in Path(report['consumption_reference']['path']).read_text(encoding='utf8').splitlines()]
    by_native = {}
    attempts = []
    for frame in ledger:
        native = int(frame['native_pts'])
        if native in by_native:
            raise ValueError('A source native frame was recorded twice')
        by_native[native] = frame
        attempts.extend(frame.get('crop_batches', ()))
    committed = sum(a['optimizer_step_committed'] for a in attempts)
    if committed != report['successful_optimizer_steps'] or len(attempts) != report['source_crops_consumed']:
        raise ValueError('Actual accepted/rejected proposals differ from the final report')
    if len(ledger) != report['frames_consumed']:
        raise ValueError('Actual source frames differ from the final report')
    sources = []
    for admission_ref in bundle['exact_review_training_admission_references']:
        authenticate_source_reference(admission_ref)
        admission = json.loads(Path(admission_ref['path']).read_text(encoding='utf8'))
        source_ref = admission['source_pack_reference']
        authenticate_source_reference(source_ref)
        with np.load(source_ref['path'], allow_pickle=False) as pack:
            uv = pack['uv']
        totals = Counter()
        frame_states = {k: set() for k in ('admitted', 'scheduled', 'attempted', 'committed')}
        source_attempts, source_commits = set(), set()
        for entry in admission['entries']:
            native = int(entry['native_pts'])
            actual = int(entry['actual_pts'])
            points = uv[entry['pack_point_indices']]
            frame_states['admitted'].add(actual)
            active = {key: np.zeros(len(points), bool) for key in ('scheduled', 'attempted', 'committed')}
            for x, y in bundle['crop_origins_by_native'].get(str(native), ()):
                active['scheduled'] |= ((points[:, 0] >= x) & (points[:, 0] < x + crop_size)
                    & (points[:, 1] >= y) & (points[:, 1] < y + crop_size))
            frame = by_native.get(native)
            if frame and int(frame['actual_pts']) != actual:
                raise ValueError('Actual training exposure changed the admitted point source clock')
            for attempt in (frame or {}).get('crop_batches', ()):
                x, y = attempt['crop_xy']
                hits = ((points[:, 0] >= x) & (points[:, 0] < x + crop_size)
                    & (points[:, 1] >= y) & (points[:, 1] < y + crop_size))
                active['attempted'] |= hits
                if hits.any():
                    source_attempts.add((native, x, y))
                if attempt['optimizer_step_committed']:
                    active['committed'] |= hits
                    if hits.any():
                        source_commits.add((native, x, y))
            totals['admitted_points'] += len(points)
            for key, flags in active.items():
                totals[key + '_points'] += int(flags.sum())
                if flags.any():
                    frame_states[key].add(actual)
        if totals['admitted_points'] != admission['points']:
            raise ValueError('Admitted source point inventory changed')
        sources.append(dict(source_pack_reference=source_ref, training_admission_reference=admission_ref,
            counts=dict(totals), actual_exposures={key: len(value) for key, value in frame_states.items()},
            attempted_reviewed_source_crops=len(source_attempts), committed_reviewed_source_crops=len(source_commits)))
    result = dict(schema='rinkspace.completed-source-training-actual-reviewed-point-coverage.v1',
        training_report_reference=reference(report_path), actual_crop_proposals=len(attempts),
        committed_optimizer_steps=committed, rejected_crop_proposals=len(attempts) - committed,
        actual_native_frames=len(ledger), scheduled_crop_proposals=report['scheduled_source_crops'],
        rejection_reasons=dict(Counter(a.get('reason') for a in attempts if not a['optimizer_step_committed'])),
        reviewed_sources=sources, model_accuracy_not_inferred_from_training_loss=True,
        original_native_guard_retained=all(r['retained_without_new_classification_errors']
            for r in report['original_native_retention_comparisons']),
        formal_model_replaced=report['formal_model_replaced'],
        independent_whole_shot_accuracy_accepted=report['independent_whole_shot_accuracy_accepted'])
    if output_path.exists():
        raise FileExistsError('Preserve earlier completed-source training audit')
    output_path.write_text(json.dumps(result, indent=2) + '\n', encoding='utf8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--crop-size', type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.report, args.output, args.crop_size)))
