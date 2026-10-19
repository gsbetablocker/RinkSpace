"""Capture the first ORIGINAL current priority-domain inputs, then unwind.

Invoke with the unchanged runtime arguments, for example:
  python scripts/diagnostics/capture_priority_domain_current_state.py named66 --output ...

The ordinary driver reconstructs its checkpoint/source domain and owns all
worker cleanup. The patched boundary writes inputs BEFORE any LP evaluation
and raises a dedicated control exception; no J or LP is evaluated here.
"""
from __future__ import annotations

import functools
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


class OriginalPriorityDomainCaptured(BaseException):
    """Diagnostic control, deliberately outside numerical Exception fallbacks."""


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _argument(name):
    for index, value in enumerate(sys.argv[1:], 1):
        if value == name and index + 1 < len(sys.argv):
            return sys.argv[index + 1]
        if value.startswith(name + '='):
            return value[len(name) + 1:]
    return None


def _file_identity(path):
    path = Path(path).resolve()
    return dict(path=str(path), bytes=path.stat().st_size, sha256=_sha(path))


def main():
    # This import installs the ORIGINAL package paths before the boundary is
    # patched. Windows spawn imports this module without executing main().
    import calibration.solver_runtime.__main__ as entry
    from calibration.solver_runtime import install
    install()
    from structured_programme_source_candidate import feasibility_priority_restoration as source
    import numpy as np

    directory_text = os.environ.get('RINKSPACE_BENCHMARK_DIAGNOSTIC_DIR')
    if directory_text is None:
        output = _argument('--output')
        if output is None:
            raise ValueError('Original --output or RINKSPACE_BENCHMARK_DIAGNOSTIC_DIR required')
        directory_text = output + '.bootstrap'
    destination = Path(directory_text).resolve() / 'priority_domain_capture'
    data_root = Path('D:/CodexData/RinkSpace_Data').resolve()
    if not destination.is_relative_to(data_root):
        raise ValueError('Diagnostic output must stay within the existing Data root')
    original = source.contract_original_support_bounds
    signature = inspect.signature(original)
    captured = False

    @functools.wraps(original)
    def capture(*args, **kwargs):
        nonlocal captured
        if captured:
            raise RuntimeError('Diagnostic priority-domain boundary called again after capture')
        bound = signature.bind(*args, **kwargs)
        explicitly_supplied = list(bound.arguments)
        bound.apply_defaults()
        arrays = {}
        for name in ('lower', 'upper', 'constraint_matrix', 'original_constraint_lower',
                     'row_group_index', 'current_group_violations'):
            value = np.asarray(bound.arguments[name])
            if value.dtype.hasobject:
                raise TypeError('Original numeric priority-domain array required: ' + name)
            arrays[name] = value
        metric = bound.arguments['matrix']
        if metric is not None:
            arrays['matrix'] = np.asarray(metric)
            if arrays['matrix'].dtype.hasobject:
                raise TypeError('Original optional metric must be numeric')
        arrays['matrix_supplied'] = np.asarray(metric is not None, dtype=np.bool_)

        destination.mkdir(parents=True, exist_ok=True)
        artifact = destination / 'original_priority_domain_inputs.npz'
        metadata_path = destination / 'original_priority_domain_capture.json'
        if artifact.exists() or metadata_path.exists():
            raise FileExistsError('A prior capture is already preserved in this diagnostic destination')
        with artifact.open('xb') as stream:
            # Uncompressed original arrays: preserve dtype/shape/values, avoid
            # object serialization and a separate compression workload.
            np.savez(stream, **arrays)
        config = destination.parent / 'input_config.json'
        checkpoint = _argument('--camera-seed')
        inputs = dict(
            schema='rinkspace.original-current-priority-domain-capture.v1',
            captured_before_original_LP=True, original_LP_calls=0,
            fresh_J_evaluated_by_capture=False, diagnostic_only=True,
            checkpoint_source_and_guards_reconstructed_by_original_driver=True,
            capture_pid=os.getpid(), wall_time_ns=time.time_ns(),
            original_forward_argv=sys.argv[1:], explicitly_supplied_arguments=explicitly_supplied,
            matrix_was_None=metric is None,
            original_contract_function_sha256=hashlib.sha256(inspect.getsource(original).encode()).hexdigest(),
            original_contract_module=_file_identity(inspect.getsourcefile(original)),
            capture_script=_file_identity(__file__), inputs_file=_file_identity(artifact),
            array_inventory={name:dict(dtype=value.dtype.str, shape=list(value.shape),
                bytes=int(value.nbytes), C_contiguous=bool(value.flags.c_contiguous),
                sha256=hashlib.sha256(value.tobytes(order='C')).hexdigest())
                for name, value in arrays.items()},
            input_config=None if not config.is_file() else _file_identity(config),
            original_runtime_installation_receipt=str(destination.parent/'execution_acceleration_receipt.json'),
            registered_source_manifest=_file_identity(Path(entry.__file__).with_name('source_manifest.json')),
            accuracy_accepted=False,
        )
        if checkpoint is not None:
            inputs['checkpoint'] = _file_identity(checkpoint)
            with np.load(checkpoint, allow_pickle=False) as saved:
                q = np.asarray(saved['q'])
                inputs['checkpoint_q'] = dict(dtype=q.dtype.str, shape=list(q.shape),
                    sha256=hashlib.sha256(q.tobytes()).hexdigest())
                inputs['checkpoint_profile_fields'] = [key for key in saved.files if key.startswith('profile__')]
        metadata_path.write_text(json.dumps(inputs, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        captured = True
        raise OriginalPriorityDomainCaptured(str(metadata_path))

    source.contract_original_support_bounds = capture
    try:
        entry.main()
    except OriginalPriorityDomainCaptured as signal:
        # Reached only after the original driver has unwound its worker
        # finally cleanup. Every unrelated numeric/control error propagates.
        print(json.dumps(dict(stage='original_priority_domain_capture_complete',
            capture_metadata=str(signal), original_LP_calls=0,
            diagnostic_control_unwound_original_driver=True)), flush=True)
        return 0
    finally:
        source.contract_original_support_bounds = original
    raise RuntimeError('Original driver completed without reaching the priority-domain capture boundary')


if __name__ == '__main__':
    raise SystemExit(main())
