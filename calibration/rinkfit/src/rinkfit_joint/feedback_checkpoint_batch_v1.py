"""Versioned checkpoint consumer using complete, exact-source dependency batches.

The native replay scan and representative rule match feedback_checkpoint.v2.
Every representative remains present, including unknown reviews. The old reader
and live fits remain immutable; only this explicitly selected entry uses batching.
"""
from __future__ import annotations
import hashlib
import importlib
from pathlib import Path
import numpy as np
from .feedback_checkpoint import (PROJECT, _diagnostics, _ref, _store,
    checkpoint_receipt_path, infer_context, CheckpointObserver as _FrozenObserver)
from .io_utils import file_sha256, read_json, write_json

VERSION = 'rinkfit.feedback-checkpoint.batch.v1'


def batch_class(config):
    reference = config if isinstance(config, dict) else dict(file=str(config))
    path = Path(reference.get('file', reference.get('path'))).resolve()
    if reference.get('sha256') and file_sha256(path) != reference['sha256']:
        raise ValueError('Changed actual source-review configuration')
    config_data = read_json(path)
    helper = config_data.get('actual_batch_dependency_helper_ref')
    if helper is None:
        from .batch_review_dependencies import ExactReviewDependencyBatch
        return ExactReviewDependencyBatch
    helper_path = Path(helper.get('file', helper.get('path'))).resolve()
    if helper_path.parent != Path(__file__).parent or not helper_path.stem.startswith('batch_review_dependencies'):
        raise ValueError('Batch helper must be a versioned project dependency reader')
    if file_sha256(helper_path) != helper['sha256']:
        raise ValueError('Changed actual batch dependency helper')
    module = importlib.import_module('.' + helper_path.stem, __package__)
    if Path(module.__file__).resolve() != helper_path:
        raise ValueError('Batch helper runtime path differs')
    return module.ExactReviewDependencyBatch


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
            findings.append(finding)
    config = context.get('source_review_configuration')
    if config is None:
        raise ValueError('Batch checkpoint reader requires explicit current source-review configuration')
    batch = batch_class(config)(packet, arrays, config,
        [dict(row_index=f['source_row_index'], source=f['source']) for f in findings],
        project=PROJECT, event=context['event_id'])
    dependencies = batch.resolve_all()
    for finding, deps in zip(findings, dependencies):
        if deps is not None:
            finding['review_dependencies'] = deps
    return findings, dict(active_rows=int(active.sum()), scanned_rows=len(active),
        over_original_E_rows=int(over.sum()), invalid_raw_RS_rows=int(invalid.sum()),
        source_packet_sha256=snapshot_sha, state_token=receipt['state_token'],
        actual_representatives=len(findings), exact_source_review_reusable=sum(d is not None for d in dependencies),
        unknown_source_review=sum(d is None for d in dependencies)), batch


def consume_generation(generation, packet_path, *, context=None, database=None, output=None, run_id=None):
    generation = Path(generation).resolve()
    context = context or infer_context(packet_path, generation)
    findings, summary, batch = collect_findings(generation, packet_path, context)
    run_id = run_id or str(generation.parent)
    with _store(context, database) as store:
        result = batch.checkpoint(store, run_id, findings)
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
        output = folder / (hashlib.sha256((VERSION + str(generation) + summary['source_packet_sha256']).encode()).hexdigest()[:24] + '.json')
    from rinkspace.feedback_runtime import atomic_json
    dependency_output = Path(output).with_suffix('.dependencies.json')
    atomic_json(dependency_output, batch.receipt())
    result['actual_batch_dependency_receipt'] = _ref(dependency_output)
    result['actual_checkpoint_collector'] = _ref(__file__)
    atomic_json(output, result)
    return dict(output=str(Path(output).resolve()), **summary, task_count=len(result['tasks']),
                idempotent_findings=sum(t['exact_finding_repeated'] for t in result['tasks']),
                visual_quality_improved=False)


class CheckpointObserver(_FrozenObserver):
    """Observe the same closed checkpoints using the versioned batch consumer."""
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
                if final:
                    self.errors.append(dict(generation=str(receipt.parent), error=str(exc),
                        type=type(exc).__name__, feedback_not_consumed=True))
                continue
            self.seen.add(key)
            self.consumed_task_documents.append(result['output'])

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join()
        self.scan(final=True)
        if self.output.exists():
            write_json(self.output / 'feedback_checkpoint_receipt.json', dict(schema=VERSION,
                actual_checkpoint_collector=_ref(__file__),
                committed_generations_consumed=len(self.seen), errors=self.errors,
                consumed_task_documents=self.consumed_task_documents,
                optimization_or_observation_mutated=False, visual_quality_improved=False))
