"""Dispatch committed checkpoint findings to the current native RGB reader.

Only the existing board reader has a source-specific review adapter here.
Other material tasks remain explicit pending work. Reviews never mutate a
solver generation or promote an observation from its numerical residual.
"""
from pathlib import Path
from .io_utils import read_json


def dispatch_checkpoint_reviews(task_documents, review_run, output, *, device='cuda'):
    from rinkspace.feedback_review import review_existing_board_frame
    from rinkspace.feedback_runtime import atomic_json
    output=Path(output).resolve()
    if output.exists():
        raise FileExistsError('immutable checkpoint source review generation exists')
    output.mkdir(parents=True)
    grouped={};pending=[]
    for document in task_documents:
        tasks=read_json(document)
        for task in tasks['tasks']:
            context=task['context']
            if not any(label.startswith('board.') for label in context.get('entity_ids', [])):
                pending.append(dict(finding_id=task['finding_id'], status='pending_material_reader_adapter', source=context['source']))
                continue
            pts=int(context['source']['pts'])
            grouped.setdefault(pts, set()).add(task['finding_id'])
    reviews=[]
    for pts, finding_ids in sorted(grouped.items()):
        target=output/str(pts)
        try:
            result=review_existing_board_frame(review_run,pts,target,device=device,predecessor_findings=sorted(finding_ids))
            reviews.append(dict(native_pts=pts, finding_ids=sorted(finding_ids), status='current_RGB_review_executed',
                receipt=str(target/'receipt.json'), counts=result['counts'], resolutions=len(result['resolutions']),
                source_feedback=result['source_feedback'], observation_generation_mutated=False))
        except Exception as error:
            reviews.append(dict(native_pts=pts, finding_ids=sorted(finding_ids), status='pending_exact_source_review',
                error=str(error), error_type=type(error).__name__, observation_generation_mutated=False))
        atomic_json(output/'progress.json', dict(reviews=reviews,pending=pending))
    receipt=dict(schema='rinkfit.current_checkpoint_source_dispatch.v1',task_documents=list(map(str,task_documents)),
        existing_native_reader_run=str(Path(review_run).resolve()), device=device,reviews=reviews,pending=pending,
        observation_or_qualification_generation_mutated=False,visual_accuracy_improvement_claimed=False)
    atomic_json(output/'receipt.json',receipt)
    return receipt
