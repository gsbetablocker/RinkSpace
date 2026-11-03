"""Run an existing body entry point with process-local performance settings."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError('must be positive')
    return value


def command_environment(project, data_root, profile, backend, script, script_args,
                        *, workers=None, native_threads=None, python=None):
    selected = profile[backend]
    workers = selected['workers'] if workers is None else workers
    native_threads = selected['native_threads'] if native_threads is None else native_threads
    if workers < 1 or native_threads < 1:
        raise ValueError('worker counts must be positive')
    if backend == 'cuda' and workers != 1:
        raise ValueError('CUDA uses one body owner; --workers must be 1')
    base = project if selected['python_base'] == 'project' else data_root
    executable = Path(python) if python else base / selected['python']
    library = data_root / profile['native_library']
    target = Path(script)
    if not target.is_absolute():
        target = project / target
    for label, candidate in (('Python', executable), ('native library', library), ('script', target)):
        if not candidate.is_file():
            raise FileNotFoundError(f'{label}: {candidate}')
    env = os.environ.copy()
    env.update(RINKSPACE_DATA_ROOT=str(data_root), PYTHONDONTWRITEBYTECODE='1',
        RINKSPACE_BODY_BACKEND=backend, RINKSPACE_BODY_WORKERS=str(workers),
        RINKSPACE_BODY_BLAS_THREADS='1', RINKSPACE_BODY_NATIVE_THREADS=str(native_threads),
        RINKSPACE_BODY_NATIVE_LIBRARY=str(library),
        RINKSPACE_BODY_GPU_BATCH_SIZE=str(selected.get('gpu_batch_size', 512)),
        RINKSPACE_BODY_PERTURBATION_BATCH=str(selected.get('perturbation_batch', 4)),
        RINKSPACE_BODY_MAX_BATCH_SAMPLES=str(selected.get('max_batch_samples', 4096)),
        OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', OMP_NUM_THREADS='1')
    if profile.get('cuda_path') and not env.get('CUDA_PATH'):
        env['CUDA_PATH'] = profile['cuda_path']
    # Select this checkout even if the caller was reviewing an isolated baseline.
    env['PYTHONPATH'] = os.pathsep.join((str(project/'src'), str(project)))
    args = list(script_args)
    if args[:1] == ['--']:
        args = args[1:]
    return [str(executable), '-B', '-X', 'utf8', str(target), *args], env


def main(argv=None):
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--workers', type=positive_int)
    parser.add_argument('--native-threads', type=positive_int)
    parser.add_argument('--python', type=Path)
    parser.add_argument('--profile', type=Path, default=project/'environments/body-performance-local.json')
    parser.add_argument('--show-command', action='store_true', help='Print settings without starting the target')
    parser.add_argument('script', help='Existing Python entry point; arguments after it pass through unchanged')
    parser.add_argument('script_args', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    sys.path.insert(0, str(project/'src'))
    from rinkspace.storage import AssetStore
    data_root = Path(AssetStore(project).data)
    profile = json.loads(args.profile.read_text(encoding='utf8'))
    try:
        command, env = command_environment(project, data_root, profile, args.backend, args.script,
            args.script_args, workers=args.workers, native_threads=args.native_threads, python=args.python)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    record = dict(event='body_performance_launcher', command=command, backend=args.backend,
        settings={k: v for k, v in env.items() if k.startswith('RINKSPACE_BODY_')},
        environment_scope='child process only', started=not args.show_command)
    print(json.dumps(record, ensure_ascii=False), flush=True)
    if args.show_command:
        return 0
    return subprocess.run(command, cwd=project, env=env).returncode


if __name__ == '__main__':
    raise SystemExit(main())
