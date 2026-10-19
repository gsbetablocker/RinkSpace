"""Explicit output-only adapter; retain the public Windows replace policy."""
from pathlib import Path
import hashlib
import importlib
import json
import time

_execution_references = None


def _reference(path):
    path = Path(path)
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def _check_reference(expected):
    actual = _reference(expected['path'])
    if actual['sha256'] != expected['sha256']:
        raise ValueError('Frozen atomic-output producer changed: ' + str(expected['path']))
    return actual


def write(path, value):
    """Keep the original runtime JSON bytes; retry only Windows sharing errors."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n', encoding='utf8')
    # Literal policy of z190_joint_source_solver._write_json. Only serialization
    # differs: this function preserves the original stage-runtime serialization.
    for attempt in range(64):
        try:
            temporary.replace(path)
            return
        except OSError as error:
            if getattr(error, 'winerror', None) not in (5, 32) or attempt == 63:
                raise
            time.sleep(min(.05 * 2**attempt, .5))


def install(expected_runtime_reference, expected_solver_reference):
    """Replace only the loaded stage runtime's writer and return that writer."""
    global _execution_references
    runtime_ref = _check_reference(expected_runtime_reference)
    solver_ref = _check_reference(expected_solver_reference)
    runtime = importlib.import_module('calibration.program_tracking.first_shot_joint_stage_runtime')
    if Path(runtime.__file__).resolve() != Path(runtime_ref['path']).resolve():
        raise ValueError('Atomic writer addressed a different loaded stage runtime')
    _execution_references = dict(runtime=runtime_ref, public_solver=solver_ref)
    runtime.write = write
    return write


def provenance():
    return dict(schema='rinkspace.explicit-atomic-stage-output.v1',
                producer=_reference(__file__), inherited=_execution_references,
                retries=64, retry_winerrors=[5, 32],
                backoff_seconds='min(0.05*2**attempt,0.5)',
                original_stage_JSON_serialization_preserved=True,
                source_model_parameters_or_objective_modified=False)


def cached_worker_init(*args, execution, original_initializer):
    """Restore the already verified same-value cache after original worker init."""
    result = original_initializer(*args)
    from . import first_shot_fit as fit
    from .first_shot_camera_initialization import _install_execution
    _install_execution(execution, type(fit._source_worker[0].fixed_head))
    return result
