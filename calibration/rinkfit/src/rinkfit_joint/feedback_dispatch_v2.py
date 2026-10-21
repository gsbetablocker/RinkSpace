"""Dispatch current native RGB reviews in the validated source-reader runtime.

Versioned current-packet tasks cover existing source readers and the qualified
S09 shared board textures. Unsupported new C operators remain explicit pending.
The original reader/dispatcher modules remain immutable.
"""
from pathlib import Path
import os
import subprocess
import sys
from .io_utils import read_json, write_json


from .feedback_dispatch import source_reader_python


def dispatch_checkpoint_reviews(task_documents, review_run, output, *, device='cuda', packet_path=None,
                                interpreter=None, _in_worker=False):
    output=Path(output).resolve();review_run=Path(review_run).resolve()
    if output.exists():raise FileExistsError('immutable checkpoint source review generation exists')
    config=review_run/'source_review_config.json'
    documents=[read_json(p) for p in task_documents]
    if packet_path is None:
        refs={x['source_packet_reference']['path'] for x in documents if x.get('source_packet_reference')}
        if len(refs)==1:packet_path=refs.pop()
    if packet_path is not None and not config.is_file():
        raise FileNotFoundError('Exact current-packet review requires source_review_config.json: '+str(config))
    if not _in_worker and source_reader_python(interpreter)!=Path(sys.executable):
        project=Path(__file__).resolve().parents[4]
        output.parent.mkdir(parents=True,exist_ok=True)
        job=output.parent/(output.name+'.job.json')
        write_json(job,dict(task_documents=list(map(str,task_documents)),review_run=str(review_run),
                           output=str(output),device=device,packet_path=None if packet_path is None else str(packet_path)))
        env=os.environ.copy();env['PYTHONPATH']=os.pathsep.join([str(project),str(project/'src'),
            str(project/'calibration/rinkfit/src'),str(project/'.venv/Lib/site-packages'),env.get('PYTHONPATH','')])
        worker=source_reader_python(interpreter)
        # Preflight once per worker, avoiding one repeated dependency failure per frame.
        probe=subprocess.run([str(worker),'-c','import torch,cv2,av,scipy; print(torch.__version__)'],env=env,capture_output=True,text=True)
        if probe.returncode:raise RuntimeError('Native source reader environment preflight failed: '+probe.stderr[-3000:])
        log=output.parent/(output.name+'.worker.log')
        with log.open('w',encoding='utf8') as stream:
            proc=subprocess.run([str(worker),'-m','rinkfit_joint.feedback_dispatch_v2',str(job)],env=env,stdout=stream,stderr=subprocess.STDOUT)
        if proc.returncode:raise RuntimeError('Native source reader worker failed; see '+str(log))
        return read_json(output/'receipt.json')
    grouped={};pending=[];reused=[]
    current=None
    if config.is_file() and packet_path is not None:
        from .packet_feedback_facade_v2 import PacketBoundCurrentReview
        current=PacketBoundCurrentReview(packet_path,config,requested_row_indices={int(t['source_row_index'])for d in documents for t in d['tasks']})
    for tasks in documents:
        for task in tasks['tasks']:
            context=task['context']
            if task.get('source_review_required') is False and (current is None or current.has_current_cached_review(task)):
                reused.append(dict(finding_id=task['finding_id'],resolution=task.get('reuse_resolution_id'),status='exact_native_review_reused'));continue
            if current is None and not any(label.startswith('board.') for label in context.get('entity_ids',[])):
                pending.append(dict(finding_id=task['finding_id'],status='pending_current_packet_review_configuration',source=context['source']));continue
            pts=int(context['source']['pts']);grouped.setdefault(pts,[]).append(task)
    output.mkdir(parents=True);reviews=[]
    for pts,tasks in sorted(grouped.items()):
        target=output/str(pts)
        try:
            if current is not None:result=current.read_frame(pts,tasks,target,device=device)
            else:
                from rinkspace.feedback_review import review_existing_board_frame
                result=review_existing_board_frame(review_run,pts,target,device=device,predecessor_findings=sorted({t['finding_id']for t in tasks}))
            reviews.append(dict(native_pts=pts,finding_ids=sorted({t['finding_id']for t in tasks}),
                status='current_RGB_review_executed',receipt=str(target/'receipt.json'),counts=result['counts'],
                resolutions=len(result['resolutions']),source_feedback=result.get('source_feedback'),observation_generation_mutated=False))
        except Exception as error:
            reviews.append(dict(native_pts=pts,finding_ids=sorted({t['finding_id']for t in tasks}),
                status='pending_exact_source_review',error=str(error),error_type=type(error).__name__,observation_generation_mutated=False))
        write_json(output/'progress.json',dict(reviews=reviews,pending=pending,reused=reused))
    receipt=dict(schema='rinkfit.current_checkpoint_source_dispatch.v3',task_documents=list(map(str,task_documents)),
        existing_native_reader_run=str(review_run),device=device,interpreter=sys.executable,
        current_packet_reader=current is not None,reviews=reviews,pending=pending,reused=reused,
        current_reader_version=None if current is None else current.reader_version,
        current_reader_module=None if current is None else current.reader_module_ref,
        observation_or_qualification_generation_mutated=False,visual_accuracy_improvement_claimed=False)
    write_json(output/'receipt.json',receipt);return receipt


if __name__=='__main__':
    dispatch_checkpoint_reviews(**read_json(sys.argv[1]),_in_worker=True)
