"""Versioned exact-source collector writing lossless interned task documents."""
from pathlib import Path
import hashlib
from .feedback_checkpoint_batch_v1 import PROJECT,collect_findings,batch_class,infer_context,_store,_ref,checkpoint_receipt_path
from .feedback_checkpoint_batch_v1 import CheckpointObserver as _FrozenObserver
from .io_utils import write_json
from . import feedback_task_codec_v1 as task_codec
VERSION='rinkfit.feedback-checkpoint.batch.v2'


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
    result['task_codec_module'] = _ref(task_codec.__file__)
    task_codec.write_task_document(output, result)
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
