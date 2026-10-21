"""Read committed joint-fit checkpoints into feedback jobs, without changing a fit.

The native formal_receipt is the checkpoint boundary. Raw fixed-station RS
residuals are discovery signals only; they never supply replacement UV or labels.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import threading

import numpy as np

from .io_utils import file_sha256, read_json, write_json

PROJECT = Path(__file__).resolve().parents[4]
VERSION = "rinkfit.feedback-checkpoint.v2"


def _ref(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=file_sha256(path))


def _array(base, descriptor):
    path = Path(descriptor['file'])
    if not path.is_absolute():
        path = Path(base) / path
    if file_sha256(path) != descriptor['sha256']:
        raise ValueError('Checkpoint feedback array SHA mismatch: ' + str(path))
    shape = tuple(descriptor['shape'])
    dtype = np.dtype(descriptor['dtype'])
    offset = int(descriptor.get('offset_bytes', 0))
    if path.stat().st_size < offset + int(np.prod(shape)) * dtype.itemsize:
        raise ValueError('Incomplete checkpoint feedback array: ' + str(path))
    return np.memmap(path, dtype=dtype, mode='r', offset=offset, shape=shape)


def infer_context(packet_path, output):
    """Resolve only explicit programme scope/source ownership; no video defaults."""
    for root in dict.fromkeys([Path(output).resolve(), *Path(output).resolve().parents,
                               Path(packet_path).resolve().parent, *Path(packet_path).resolve().parents]):
        scope_path = root / 'programme_scope.json'
        if not scope_path.is_file():
            continue
        source = read_json(scope_path)['source']
        parts = Path(source['path']).parts
        try:
            event = parts[parts.index('events') + 1]
            programme = parts[parts.index('programmes') + 1]
        except (ValueError, IndexError):
            continue
        return dict(event_id=event, programme_id=programme,
                    source=dict(video_sha256=source['sha256'], time_base=source['time_base'],
                                coordinate_domain='native_pixels'),
                    scope_reference=_ref(scope_path))
    raise ValueError('No explicit event/programme source binding for checkpoint feedback')


@contextmanager
def _store(context, database=None):
    if database is None:
        from rinkspace.feedback_runtime import feedback_store
        with feedback_store(PROJECT, context['event_id']) as store:
            yield store
    else:
        from rink_feedback.core import Store
        with Store(database) as store:
            yield store


def checkpoint_receipt_path(generation):
    generation = Path(generation).resolve()
    formal = generation / 'formal_receipt.json'
    engineering = generation / 'engineering_checkpoint_receipt.json'
    if formal.is_file() and engineering.is_file():
        raise ValueError('Ambiguous checkpoint objective receipt')
    if engineering.is_file():
        receipt = read_json(engineering)
        if receipt.get('objective_id') != 'engineering_recovery_native_pixel_v1' or receipt.get('formal_acceptance_claimed') is not False:
            raise ValueError('Invalid engineering checkpoint objective identity')
        return engineering
    return formal


def _diagnostics(generation, packet_path):
    """Validate the complete artifact set before reading any discovery values."""
    generation = Path(generation).resolve()
    receipt_path = checkpoint_receipt_path(generation)
    receipt = read_json(receipt_path)
    packet_path = Path(packet_path).resolve()
    packet_sha = file_sha256(packet_path)
    if receipt['source_packet_sha256'] != packet_sha:
        raise ValueError('Checkpoint receipt belongs to another source packet')
    packet = read_json(packet_path)
    report = read_json(generation / 'value_report.json')
    for name, descriptor in report['files'].items():
        path = generation / name
        if path.stat().st_size != descriptor['size_bytes'] or file_sha256(path) != descriptor['sha256']:
            raise ValueError('Incomplete or changed committed checkpoint artifact: ' + name)
    model = read_json(generation / 'model.json')
    arrays = {name: _array(packet_path.parent, packet['arrays'][name]) for name in
              ('native_pts', 'shot_index', 'material_index', 'admission', 'fit_role', 'E', 'uv', 'profile_normal', 'kind')}
    n = int(packet['row_count'])
    if len(packet['source_row_ids']) != n or any(len(x) != n for x in arrays.values()):
        raise ValueError('Checkpoint feedback source row identity is incomplete')
    raw = generation / 'raw_station_RS_error_native_uv.f64'
    valid_path = generation / 'raw_station_RS_valid.u8'
    if raw.name not in report['files'] or valid_path.name not in report['files']:
        raise ValueError('Fixed-current-station diagnostics required; profiled residuals are not substituted')
    error = np.memmap(raw, dtype='<f8', mode='r', shape=(n, 2))
    valid = np.memmap(valid_path, dtype='u1', mode='r', shape=(n,))
    return receipt, report, packet, model, arrays, error, valid


def collect_findings(generation, packet_path, context):
    receipt, report, packet, model, arrays, error, valid = _diagnostics(generation, packet_path)
    active = (arrays['admission'] == 1) & (arrays['fit_role'] == 1)
    distance = np.linalg.norm(error, axis=1)
    query_rows = np.asarray([q['row_index'] for q in packet.get('queries', [])], dtype=np.int64)
    if len(query_rows):
        distance[query_rows] = np.abs(np.sum(error[query_rows] * arrays['profile_normal'][query_rows], axis=1))
    finite = np.isfinite(distance)
    invalid = active & ((valid != 1) | ~finite)
    over = active & ~invalid & (distance > arrays['E'])
    generation = Path(generation).resolve()
    evidence = [_ref(checkpoint_receipt_path(generation)), _ref(generation / 'value_report.json')]
    snapshot_sha = file_sha256(packet_path)
    findings = []
    # One actual worst row per shot/material/signal is a review representative.
    # Every active row is scanned; counts and membership digest retain the scope.
    for signal, selected in [('native_RS_support_failure', invalid), ('over_original_E', over)]:
        rows = np.flatnonzero(selected)
        grouped = {}
        for index in rows:
            key = (int(arrays['shot_index'][index]), int(arrays['material_index'][index]))
            grouped.setdefault(key, []).append(int(index))
        for (shot_index, material_index), indices in sorted(grouped.items()):
            index = max(indices, key=lambda i: (float(distance[i]) if finite[i] else float('inf'), -i))
            sid = packet['shot_ids'][shot_index]
            label = model['materials'][material_index]['label']
            rid = packet['source_row_ids'][index]
            dataset = packet.get('source_dataset_ids', [''] * len(active))[index]
            observation_id = dataset + '::' + rid
            pts = int(arrays['native_pts'][index])
            source = dict(context['source'], pts=pts)
            tags = ['fit_checkpoint', 'native_source_recheck']
            if label.startswith('Ad'):
                tags += ['advertisement', 'physical_instance']
            if 'board' in label:
                tags += ['material_boundary']
            finding = dict(event_id=context['event_id'], programme_id=context['programme_id'],
                source=source, camera_id=model['shots'][sid].get('rig_id'), entity_ids=[label],
                signal_type=signal, problem_key=signal, observation_ids=[observation_id], context_tags=tags,
                evidence_refs=evidence, checkpoint_state_token=receipt['state_token'],
                observation_snapshot_sha256=snapshot_sha, source_row_index=index,
                review_scope=dict(shot_id=sid, material_index=material_index,
                    affected_rows=len(indices), row_indices_sha256=hashlib.sha256(np.asarray(indices, '<u8').tobytes()).hexdigest(),
                    representative_selection='largest actual raw error; stable first row on ties'),
                diagnostic=dict(raw_fixed_station_RS_error_px=float(distance[index]) if finite[index] else None,
                    original_E_px=float(arrays['E'][index]), source_uv=arrays['uv'][index].tolist(),
                    raw_RS_valid=bool(valid[index]), is_truth_label=False,
                    source_identity_certificate_recalled=any((Path(packet_path).parent / name).is_file() for name in ('SOURCE_IDENTITY_CERTIFICATE_RECALLED.json', 'SOURCE_IDENTITY_EVIDENCE_RECALL.json')),
                    native_projection_domain_pass_is_not_source_RGB_identity_proof=True),
                visual_resolution='unknown_until_current_native_source_review',
                native_projection_is_not_source_identity_proof=True)
            # Reuse only the actual source/reader dependency record produced by
            # a completed native review of this exact observation subject.
            from .packet_bound_feedback_review import cached_dependencies
            deps = cached_dependencies(source, packet, arrays, index,
                review_config=context.get('source_review_configuration'))
            if deps is not None:
                finding['review_dependencies'] = deps
            findings.append(finding)
    return findings, dict(active_rows=int(active.sum()), scanned_rows=len(active),
                          over_original_E_rows=int(over.sum()), invalid_raw_RS_rows=int(invalid.sum()),
                          source_packet_sha256=snapshot_sha, state_token=receipt['state_token'])


def consume_generation(generation, packet_path, *, context=None, database=None, output=None, run_id=None):
    from rink_feedback.cycle import FeedbackCycle
    generation = Path(generation).resolve()
    context = context or infer_context(packet_path, generation)
    findings, summary = collect_findings(generation, packet_path, context)
    run_id = run_id or str(generation.parent)
    with _store(context, database) as store:
        result = FeedbackCycle(store).checkpoint(run_id, findings)
    for task, finding in zip(result['tasks'], findings):
        task['source_row_index'] = finding['source_row_index']
        task['source_dataset_id'], _, task['source_row_id'] = finding['observation_ids'][0].partition('::')
        task['review_scope'] = finding['review_scope']
        task['diagnostic'] = finding['diagnostic']
        task['regression_probe'] = dict(status='pending_current_source_review',
            previous_resolution_refs=task['previous_resolution_refs'],
            affected_observation_ids=task['observation_ids'], visual_regression_executed=False,
            numeric_signal_is_not_visual_regression=True)
    result.update(schema=VERSION, generation_reference=_ref(checkpoint_receipt_path(generation)),
                  summary=summary, automatic_observation_installation=False,
                  visual_quality_improved=False, source_clock=context['source'],
                  source_packet_reference=_ref(packet_path),
                  review_reuse_policy='reuse completed exact-subject native RGB/mask/observation/reader/fact dependencies; new PTS rereads pixels')
    if output is None:
        from rinkspace.storage import AssetStore
        folder = AssetStore(PROJECT).data / 'events' / context['event_id'] / 'feedback' / 'checkpoints'
        output = folder / (hashlib.sha256((str(generation) + summary['source_packet_sha256']).encode()).hexdigest()[:24] + '.json')
    from rinkspace.feedback_runtime import atomic_json
    atomic_json(output, result)
    return dict(output=str(Path(output).resolve()), **summary, task_count=len(result['tasks']),
                idempotent_findings=sum(t['exact_finding_repeated'] for t in result['tasks']),
                visual_quality_improved=False)


class CheckpointObserver:
    """Observe closed native checkpoint receipts; never run inside an optimizer."""
    def __init__(self, output, packet, *, context=None):
        self.output = Path(output).resolve()
        self.packet = Path(packet).resolve()
        self.context = context
        self.stop = threading.Event()
        self.seen = set()
        self.errors = []
        self.consumed_task_documents = []
        self.thread = threading.Thread(target=self._run, name='rinkspace-feedback-checkpoints', daemon=True)

    def scan(self, *, final=False):
        receipts = list(self.output.glob('generation_*/formal_receipt.json'))
        receipts += list(self.output.glob('independent_RS_state/engineering_checkpoint_receipt.json'))
        for receipt in sorted(receipts):
            key = str(receipt)
            if key in self.seen:
                continue
            try:
                result = consume_generation(receipt.parent, self.packet, context=self.context)
            except Exception as exc:
                # A live native stream may not have closed yet. Retry at the
                # next boundary, then make an unresolved connection explicit.
                if final:
                    self.errors.append(dict(generation=str(receipt.parent), error=str(exc),
                                            type=type(exc).__name__, feedback_not_consumed=True))
                continue
            self.seen.add(key)
            self.consumed_task_documents.append(result["output"])

    def _run(self):
        while not self.stop.wait(1.0):
            self.scan()

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join()
        self.scan(final=True)
        if self.output.exists():
            write_json(self.output / 'feedback_checkpoint_receipt.json', dict(schema=VERSION,
                committed_generations_consumed=len(self.seen), errors=self.errors,
                consumed_task_documents=self.consumed_task_documents,
                optimization_or_observation_mutated=False, visual_quality_improved=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generation', required=True)
    parser.add_argument('--packet', required=True)
    parser.add_argument('--database')
    parser.add_argument('--out')
    parser.add_argument('--run-id')
    args = parser.parse_args()
    print(json.dumps(consume_generation(args.generation, args.packet, database=args.database,
                                       output=args.out, run_id=args.run_id), ensure_ascii=False))


if __name__ == '__main__':
    main()
