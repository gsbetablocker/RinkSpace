"""Engineering entry with explicit project import roots for real source review.

The native solver and SHA-bound v1 feedback consumers are unchanged.  The
observer and its subprocess readers inherit the same canonical import roots,
including the project namespace used by calibration.program_tracking.runtime.
"""
from contextlib import contextmanager
import importlib.util
import os
from pathlib import Path
import sys
from .io_utils import file_sha256, write_json


def project_import_roots():
    project = Path(__file__).resolve().parents[4]
    roots = [project, project / 'calibration/rinkfit/src', project / 'src']
    expected = project / 'calibration/program_tracking/runtime/bootstrap.py'
    if not expected.is_file():
        raise FileNotFoundError('Shared project runtime bootstrap missing: ' + str(expected))
    return roots, expected


@contextmanager
def source_review_runtime():
    roots, expected = project_import_roots()
    previous_path = list(sys.path)
    previous_environment = os.environ.get('PYTHONPATH')
    prefix = [str(path) for path in roots]
    try:
        sys.path[:] = prefix + [p for p in previous_path if p not in prefix]
        inherited = [p for p in (previous_environment or '').split(os.pathsep) if p and p not in prefix]
        os.environ['PYTHONPATH'] = os.pathsep.join(prefix + inherited)
        spec = importlib.util.find_spec('calibration.program_tracking.runtime.bootstrap')
        if spec is None or spec.origin is None or Path(spec.origin).resolve() != expected.resolve():
            raise ImportError('Actual source-review bootstrap resolves outside the current project')
        yield dict(project_import_roots=prefix,
                   bootstrap=dict(file=str(expected), sha256=file_sha256(expected)),
                   parent_and_worker_environment_consistent=True)
    finally:
        sys.path[:] = previous_path
        if previous_environment is None:
            os.environ.pop('PYTHONPATH', None)
        else:
            os.environ['PYTHONPATH'] = previous_environment


def execute_engineering(executable, model, packet, output, *, config):
    """Use the existing real solver/observer after validating their import roots."""
    out = Path(output).resolve()
    receipt = out.parent / (out.name + '_runtime_imports.json')
    if out.exists() or receipt.exists():
        raise FileExistsError('Immutable engineering generation or runtime receipt exists')
    with source_review_runtime() as runtime:
        from .joint_solver_batch_v1 import execute_engineering as execute_bound
        write_json(receipt, dict(schema='rinkfit.engineering_source_review_runtime.v2',
            actual_entry=dict(file=str(Path(__file__).resolve()), sha256=file_sha256(__file__)),
            **runtime))
        return execute_bound(executable, model, packet, out, config=config)
