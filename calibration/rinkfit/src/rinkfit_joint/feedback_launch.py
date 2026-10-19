"""Project-owned launch of an already prepared native formal continuation.

No source is measured or promoted here. The checkpoint observer is mandatory;
current source recall guards remain authoritative before any executable starts.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import time

from .feedback_checkpoint import CheckpointObserver, PROJECT, infer_context
from .io_utils import file_sha256, read_json

VERSION = 'rinkfit.feedback-launch.v1'


def _ref(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=file_sha256(path))


def _verify(ref):
    path = Path(ref['path']).resolve()
    if file_sha256(path) != ref['sha256']:
        raise ValueError('Prepared feedback launch reference changed: ' + str(path))
    return path


def prepare(execution, checkpoint, output, manifest, *, current_model=None):
    """Stage a reviewable continuation only; never start or resume the solver."""
    from rinkspace.feedback_runtime import atomic_json
    from rinkspace.storage import AssetStore
    execution, checkpoint = Path(execution).resolve(), Path(checkpoint).resolve()
    output, manifest = Path(output).resolve(), Path(manifest).resolve()
    data = AssetStore(PROJECT).data
    if not output.is_relative_to(data) or not manifest.is_relative_to(data):
        raise ValueError('Prepared runtime launch artifacts belong on registered Data')
    original = read_json(execution)
    command = list(original['command'])
    if len(command) != 6 or command[1] != '--formal':
        raise ValueError('Expected the exact existing six-argument joint formal command')
    executable, model, packet, _, config_path = map(Path, [command[0], *command[2:]])
    if executable.name != 'joint_fit.exe':
        raise ValueError('Expected registered joint_fit native executable')
    if output.exists() or manifest.exists():
        raise FileExistsError('Prepared launch and numerical output must be new immutable paths')
    config = read_json(config_path)
    config['resume_checkpoint'] = str(checkpoint)
    config['stop_file'] = str(output) + '.stop'
    configfile = manifest.parent / 'candidate_config.json'
    if configfile.exists():
        raise FileExistsError('Immutable candidate configuration already exists')
    atomic_json(configfile, config)
    command[4], command[5] = str(output), str(configfile)
    context = infer_context(packet, output)
    receipt = dict(schema=VERSION, status='prepared_only_not_started', command=command,
        original_execution=_ref(execution), executable=_ref(executable), model=_ref(model),
        source_packet=_ref(packet), complete_resume_checkpoint=_ref(checkpoint),
        prepared_config=_ref(configfile), original_config=_ref(config_path),
        feedback_context=context, launcher=_ref(Path(__file__)),
        existing_driver_preserved=True, original_native_flags_preserved=True,
        configuration_changes=['resume_checkpoint', 'stop_file'],
        native_command_path_changes=['output', 'configuration'],
        numerical_limits_weights_and_source_observations_changed=False,
        automatically_resumed=False, source_qualification_recheck_required_before_launch=True)
    if current_model:
        receipt['current_complete_checkpoint_model'] = _ref(current_model)
    atomic_json(manifest, receipt)
    return receipt


def inspect(manifest):
    """Authenticate prepared bytes without executing native code or changing pause."""
    doc = read_json(manifest)
    if doc['schema'] != VERSION or len(doc['command']) != 6 or doc['command'][1] != '--formal':
        raise ValueError('Unsupported prepared formal launch')
    for key in ('executable', 'model', 'source_packet', 'complete_resume_checkpoint',
                'prepared_config', 'original_execution', 'original_config', 'launcher'):
        _verify(doc[key])
    command = doc['command']
    if [str(Path(command[i]).resolve()) for i in (0, 2, 3, 5)] != [
            str(Path(doc[k]['path']).resolve()) for k in ('executable', 'model', 'source_packet', 'prepared_config')]:
        raise ValueError('Prepared command differs from its authenticated inputs')
    config = read_json(command[5])
    if Path(config['resume_checkpoint']).resolve() != Path(doc['complete_resume_checkpoint']['path']).resolve():
        raise ValueError('Prepared configuration names another complete checkpoint')
    markers = [str(Path(command[3]).parent / name) for name in
               ('SOURCE_IDENTITY_CERTIFICATE_RECALLED.json', 'SOURCE_IDENTITY_EVIDENCE_RECALL.json')
               if (Path(command[3]).parent / name).exists()]
    return dict(schema=VERSION, prepared_manifest=_ref(manifest), native_executable_started=False,
                command=command, source_recall_markers=markers,
                explicit_authorized_continuation_required=True,
                current_source_qualification_blocks_launch=bool(markers))


def run(manifest):
    """Explicit launch API; current source guard and observer cannot be skipped."""
    from rinkspace.feedback_runtime import atomic_json
    from .joint_solver import assert_source_identity_current
    inspect(manifest)
    doc = read_json(manifest)
    command = doc['command']
    assert_source_identity_current(command[3])
    out = Path(command[4]).resolve()
    if out.exists():
        raise FileExistsError('Immutable native generation output exists')
    executable = Path(command[0])
    recipe = read_json(executable.parent / 'recipe.json')
    if file_sha256(executable) != recipe['executable_sha256']:
        raise ValueError('Native executable differs from its original production build')
    env = dict(os.environ)
    env['PATH'] = ';'.join(recipe['runtime_bin_paths']) + ';' + env.get('PATH', '')
    env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    logfile = out.parent / (out.name + '.log')
    if logfile.exists():
        raise FileExistsError('Native continuation log already exists')
    logfile.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with CheckpointObserver(out, command[3], context=doc['feedback_context']):
        with logfile.open('x', encoding='utf8') as stream:
            result = subprocess.run(command, cwd=PROJECT, env=env, stdout=stream, stderr=subprocess.STDOUT)
    receipt = dict(schema=VERSION, prepared_manifest=_ref(manifest), command=command,
                   returncode=result.returncode, elapsed_seconds=time.monotonic() - start,
                   log=str(logfile), checkpoint_observer_mandatory=True,
                   native_parameters_or_observations_changed_by_launcher=False)
    if out.exists():
        atomic_json(out / 'execution.json', receipt)
    return receipt
