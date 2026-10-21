"""Engineering recovery with the versioned exact-source batch observer.

This entry keeps the native optimizer/PTX and source identity guard unchanged.
The selected dispatcher and observer are SHA-bound in the next run config.
"""
from pathlib import Path
import importlib
import json
import os
import subprocess
from .joint_solver import assert_source_identity_current, sha
from .joint_solver_batch_v2 import source_review_runtime
from .io_utils import write_json


def _module(reference):
    name = reference['module']
    if not name.startswith('rinkfit_joint.'):
        raise ValueError('Checkpoint dispatcher must be a project reader module')
    module = importlib.import_module(name)
    path = Path(module.__file__).resolve()
    if path != Path(reference['file']).resolve() or sha(path) != reference['sha256']:
        raise ValueError('Changed actual checkpoint dispatcher module: ' + name)
    return module


def validate_task_consumers(review):
    """Bind actual new consumers plus the three frozen helpers they delegate to."""
    dispatcher=_module(review['dispatcher_module_ref'])
    collector=_module(review['collector_module_ref'])
    codec=_module(review['task_codec_module_ref'])
    batch_module=_module(review['batch_dependency_module_ref'])
    if collector.task_codec is not codec or dispatcher.task_codec is not codec:
        raise ValueError('Collector/dispatcher codec differs from bound actual module')
    required={collector.collect_findings.__module__,dispatcher.prepare_reuse_decisions.__module__,source_review_runtime.__module__}
    references=review['frozen_dependency_module_refs']
    if len(references)!=len(required)or {r['module']for r in references}!=required:
        raise ValueError('Incomplete/extra frozen feedback helper module registry')
    for reference in references:_module(reference)
    actual=[review[k]for k in ('collector_module_ref','dispatcher_module_ref','task_codec_module_ref','batch_dependency_module_ref')]+references
    return collector,dispatcher,batch_module,actual


def _execute_engineering_bound(executable, model, packet, output, *, config):
    """Run a bound immutable packet; only closed states feed source review."""
    assert_source_identity_current(packet)
    exe = Path(executable).resolve()
    recipe = json.loads((exe.parent / 'recipe.json').read_text(encoding='utf-8'))
    if sha(exe) != recipe['executable_sha256']:
        raise ValueError('Compiled backend SHA changed')
    cfg = json.loads(Path(config).read_text(encoding='utf-8'))
    if cfg.get('screen_grid_manifest') or cfg.get('screen_grid_initialization_only'):
        raise ValueError('Batch physical recovery requires a translated native seed')
    review = cfg['checkpoint_source_review']
    collector,dispatcher,batch_module,actual_modules = validate_task_consumers(review)
    out = Path(output).resolve()
    if out.exists():
        raise FileExistsError('Immutable engineering generation exists')
    source_config = Path(review['run']).resolve() / 'source_review_config.json'
    if not source_config.is_file():
        raise FileNotFoundError('Configured current source reader inputs missing: ' + str(source_config))
    source_config_sha = sha(source_config)
    selected_batch_class = collector.batch_class(dict(file=str(source_config), sha256=source_config_sha))
    if selected_batch_class.__module__ != batch_module.__name__:
        raise ValueError('Configured collector helper differs from run-bound actual helper')
    context = collector.infer_context(packet, out)
    context['source_review_configuration'] = dict(file=str(source_config), sha256=source_config_sha)
    env = os.environ.copy()
    env['PATH'] = ';'.join(recipe['runtime_bin_paths']) + ';' + env.get('PATH', '')
    env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    log = out.parent / (out.name + '.log')
    log.parent.mkdir(parents=True, exist_ok=True)
    command = [str(exe), '--engineering-recovery', str(Path(model).resolve()),
               str(Path(packet).resolve()), str(out), str(Path(config).resolve())]
    with collector.CheckpointObserver(out, packet, context=context) as observer:
        with log.open('w', encoding='utf-8') as stream:
            result = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
    if observer.consumed_task_documents:
        dispatcher.dispatch_checkpoint_reviews(observer.consumed_task_documents, review['run'],
            out / 'source_reviews', device=review.get('device', 'cuda'), packet_path=packet)
    for reference in actual_modules:
        if sha(reference['file']) != reference['sha256']:
            raise ValueError('Actual checkpoint module changed during recovery')
    if sha(source_config) != source_config_sha:
        raise ValueError('Actual source reader configuration changed during recovery')
    receipt = dict(schema='rinkfit.joint_engineering_batch_execution.v3', command=command,
        backend_sha256=sha(exe), model_sha256=sha(model), packet_sha256=sha(packet),
        configuration_sha256=sha(config), actual_checkpoint_modules=actual_modules,
        source_review_configuration=dict(file=str(source_config), sha256=source_config_sha),
        returncode=result.returncode, log=str(log), engineering_recovery=True,
        objective_id='engineering_recovery_native_pixel_v1',
        committed_feedback_errors=observer.errors, quality_accepted=False)
    if out.exists():
        (out / 'execution.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(receipt), flush=True)
    return result.returncode


def execute_engineering(executable,model,packet,output,*,config):
    """Native recovery unchanged; explicitly bind the compact collector/decoder."""
    out=Path(output).resolve();receipt=out.parent/(out.name+'_runtime_imports.json')
    if out.exists()or receipt.exists():raise FileExistsError('Immutable engineering generation exists')
    with source_review_runtime()as runtime:
        write_json(receipt,dict(schema='rinkfit.engineering_source_review_runtime.v3',actual_entry=dict(file=str(Path(__file__).resolve()),sha256=sha(__file__)),**runtime))
        return _execute_engineering_bound(executable,model,packet,out,config=config)
