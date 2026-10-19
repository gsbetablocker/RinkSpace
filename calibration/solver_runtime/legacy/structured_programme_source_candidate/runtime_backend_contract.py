"""Restore and verify the existing exact native backends after audit imports.

This module does not build libraries, construct the source graph, change q,
evaluate observations, or choose the source-actor/pattern worker count. Call
after the driver's module imports (also in Windows spawn), before prepare,
and again after prepare to detect any later import side effect.
"""
from __future__ import annotations
from calibration.solver_runtime.config import data_root as _runtime_data_root, data_module_dir as _runtime_data_module_dir

import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import pickle
import sys
import time


BASE=_runtime_data_root()
from calibration.solver_runtime.config import PROGRAM_SOURCE_ROOT
SNAPSHOT = PROGRAM_SOURCE_ROOT
REQUIRED_ENV = {
    'RINKSPACE_MOTION_BACKEND': 'native',
    'RINKSPACE_PROJECTION_BACKEND': 'native',
    'RINKSPACE_SCATTER_BACKEND': 'native',
    'RINKSPACE_PROJECTION_VALUE_BACKEND': 'opencv',
    'RINKSPACE_SOLVER_ACCELERATION': 'auto',
    'RINKSPACE_GEOMETRY_SPARSE_FD': '1',
    'RINKSPACE_GEOMETRY_FD_STEP_M': '1e-7',
    'RINKSPACE_PROFILE_MOTION_ONLY': '1',
}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _library_identity(module, metadata):
    library = getattr(module, '_library', None)
    library_name = getattr(library, '_name', None)
    if library_name is None:
        raise RuntimeError(f'Native library is not loaded: {module.__name__}')
    library_path = Path(library_name).resolve()
    source_path = Path(module.SOURCE).resolve()
    build = metadata.get('native_build') or {}
    source_sha = _sha(source_path)
    library_sha = _sha(library_path)
    if (build.get('source_sha256') != source_sha or
            build.get('dll_sha256') != library_sha):
        raise RuntimeError(f'Loaded native source/DLL identity changed: {module.__name__}')
    return dict(module=module.__name__, module_path=str(Path(module.__file__).resolve()),
                source_path=str(source_path), source_sha256=source_sha,
                library_path=str(library_path), library_sha256=library_sha,
                source_and_library_match_manifest=True, metadata=metadata)


def restore_native_backend_contract(*, stage, native_threads=3, pattern_workers=None):
    """Return actual loaded backend identities; fail on missing native kernels.

    MODE in acceleration.py is frozen at import. Restore that module global
    in place, so existing imported function references retain the same module
    and kernel. Never reload it or fabricate success from environment values.
    Existing whitening instances are not rebuilt: call before graph creation.
    """
    if not stage:
        raise ValueError('A named restoration stage is required')
    native_threads = int(native_threads)
    if native_threads < 1:
        raise ValueError('Positive native thread count required')
    if pattern_workers is not None and int(pattern_workers) < 1:
        raise ValueError('Positive explicit pattern worker count required')
    before = {key: os.environ.get(key) for key in REQUIRED_ENV}
    os.environ.update(REQUIRED_ENV)
    os.environ['RINKSPACE_MOTION_THREADS'] = str(native_threads)
    os.environ['RINKSPACE_SCATTER_THREADS'] = str(native_threads)
    if pattern_workers is not None:
        os.environ['RINKSPACE_PATTERN_WORKERS'] = str(int(pattern_workers))

    acceleration = importlib.import_module('acceleration')
    aliases = []
    for name, module in tuple(sys.modules.items()):
        if (module is None or name.rsplit('.', 1)[-1] != 'acceleration' or
                not hasattr(module, 'nearest_segments') or not hasattr(module, 'MODE')):
            continue
        if Path(module.__file__).resolve() != (SNAPSHOT / 'acceleration.py').resolve():
            raise RuntimeError(f'Unexpected solver acceleration module: {name}: {module.__file__}')
        if getattr(module, '_kernel', None) is None:
            raise RuntimeError('Verified nearest-segment DLL was not loaded before factor imports: '
                               f'{name}: {module.metadata()}')
        previous_mode = module.MODE
        module.MODE = 'auto'
        aliases.append(dict(name=name, previous_mode=previous_mode,
                            mode=module.MODE, module_object_id=id(module)))
    acceleration_metadata = acceleration.metadata()
    if (acceleration_metadata['mode'] != 'auto' or
            acceleration_metadata['curve_search'] != 'strict_c'):
        raise RuntimeError(f'Nearest-segment native acceleration is inactive: {acceleration_metadata}')

    libraries = {'nearest_segment': _library_identity(acceleration, acceleration_metadata)}
    for label, name in (('motion', 'z190_motion_native'),
                        ('projection_guard', 'z190_projection_guard_native'),
                        ('csr_scatter', 'z190_csr_scatter_native')):
        module = importlib.import_module(name)
        metadata = module.metadata()
        if not metadata.get('available'):
            raise RuntimeError(f'Required exact native backend unavailable: {name}: {metadata}')
        libraries[label] = _library_identity(module, metadata)

    brown4 = importlib.import_module('continuous_central_brown4_optin')
    projection_hook = brown4.enable_projection_backend()
    actual_env = {key: os.environ.get(key) for key in REQUIRED_ENV}
    if actual_env != REQUIRED_ENV:
        raise RuntimeError(f'Backend configuration changed during verification: {actual_env}')
    projection = importlib.import_module('projection')
    if projection._central_projection_backend is not brown4.safe_project_compatible:
        raise RuntimeError('Reviewed Brown4 central-projection hook is not installed')

    return dict(schema='rinkspace.actual-native-runtime-backend-contract.v1',
                stage=str(stage), restored_environment_before=before,
                required_environment=actual_env,
                native_threads=native_threads,
                pattern_workers=os.environ.get('RINKSPACE_PATTERN_WORKERS'),
                pattern_worker_count_owned_by_caller=True,
                acceleration_loaded_module_aliases=aliases,
                actual_native_libraries=libraries,
                projection_hook=projection_hook,
                dispatch=dict(motion='SchurMotionKernel reads environment on each call',
                              projection='project_fast/project_central_valid read environment on each call',
                              scatter='scatter_fixed_xyz_source_csr reads environment on each call',
                              nearest_segment='same loaded acceleration module globals and verified strict C kernel'),
                graph_created=False, residual_or_jacobian_evaluated=False,
                numerical_equivalence_or_speedup_claimed=False,
                source_observations_q_masks_scales_profiles_unchanged=True)


def _verified_source_worker_bootstrap(task_root, native_threads, dynamic_modules):
    """Pickleable wrapper: original bootstrap first, then verify its imports."""
    module = importlib.import_module('z190_process_source_evaluation')
    original = getattr(module, '_rinkspace_original_worker_bootstrap', None)
    if original is None or original is _verified_source_worker_bootstrap:
        raise RuntimeError('Original source worker bootstrap was not retained')
    original_result = original(task_root, native_threads, dynamic_modules)
    contract = restore_native_backend_contract(
        stage='source_worker_after_dynamic_module_imports', native_threads=native_threads)
    contract.update(pid=os.getpid(), parent_pid=os.getppid(), task_root=str(task_root),
                    dynamic_module_names=sorted(dynamic_modules))
    module._rinkspace_worker_bootstrap_backend_contract = contract
    return original_result


def _finalize_source_worker_backend(task_root, native_threads, dynamic_modules):
    """Run after the original pickle load; q/source buffers are not inspected."""
    module = importlib.import_module('z190_process_source_evaluation')
    contract = restore_native_backend_contract(
        stage='source_worker_after_block_unpickle', native_threads=native_threads)
    contract.update(pid=os.getpid(), parent_pid=os.getppid(), task_root=str(task_root),
                    dynamic_module_names=sorted(dynamic_modules),
                    source_actor_block_unpickled_before_final_restoration=True,
                    original_worker_algorithm_preserved=True)
    first = getattr(module, '_rinkspace_worker_bootstrap_backend_contract', None)
    contract['bootstrap_verification'] = None if first is None else dict(
        stage=first['stage'], required_environment=first['required_environment'],
        restored_environment_before=first['restored_environment_before'],
        actual_native_library_hashes={
            key: value['library_sha256'] for key, value in first['actual_native_libraries'].items()})
    runtime = os.environ.get('RINKSPACE_SOURCE_RUNTIME_DIR')
    contract['metadata_file_path'] = None
    if runtime:
        directory = Path(runtime).resolve()
        if not directory.is_relative_to(BASE.resolve()):
            raise RuntimeError(f'Source worker runtime metadata must remain under {BASE}: {directory}')
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / f'actor_backend_{os.getpid()}_{time.time_ns()}.json'
        contract['metadata_file_path'] = str(destination)
        with destination.open('x', encoding='utf8') as handle:
            json.dump(contract, handle, ensure_ascii=False, indent=2)
    module._rinkspace_worker_final_backend_contract = contract
    return contract


def install_source_worker_backend_contract():
    """Install bootstrap and post-unpickle checks without changing old source.

    Call at driver module scope after imports so Windows spawn installs the
    same top-level actor target. The original worker text is copied exactly;
    only one backend-verification call is inserted after its pickle load.
    Its original module/name and all original function globals are retained.
    """
    module = importlib.import_module('z190_process_source_evaluation')
    expected = (SNAPSHOT / 'z190_process_source_evaluation.py').resolve()
    if Path(module.__file__).resolve() != expected:
        raise RuntimeError(f'Unexpected source worker module: {module.__file__}')
    installed = getattr(module, '_rinkspace_worker_backend_installation', None)
    if installed is not None:
        if (module._worker_bootstrap is not _verified_source_worker_bootstrap or
                module._actor_worker is not module._rinkspace_verified_actor_worker):
            raise RuntimeError('Installed source worker backend contract was replaced')
        return dict(installed, already_installed=True)

    original_bootstrap = module._worker_bootstrap
    original_actor = module._actor_worker
    original_source = inspect.getsource(original_actor)
    marker = '        block=pickle.loads(payload);del payload\n'
    added = '        _rinkspace_finalize_worker_backend(task_root,native_threads,dynamic_modules)\n'
    if original_source.count(marker) != 1:
        raise RuntimeError('Exact original source worker unpickle marker changed')
    modified_source = original_source.replace(marker, marker + added, 1)
    if modified_source.replace(added, '', 1) != original_source:
        raise RuntimeError('Source actor runtime patch changed original algorithm text')
    source_bytes = expected.read_bytes()
    module._rinkspace_original_worker_bootstrap = original_bootstrap
    module._rinkspace_original_actor_worker = original_actor
    module._rinkspace_finalize_worker_backend = _finalize_source_worker_backend
    exec(compile(modified_source, str(expected) + '::<runtime-backend-contract>', 'exec'),
         module.__dict__)
    verified_actor = module._actor_worker
    if (verified_actor.__module__ != original_actor.__module__ or
            verified_actor.__name__ != original_actor.__name__):
        raise RuntimeError('Source worker runtime target lost its original pickle identity')
    module._worker_bootstrap = _verified_source_worker_bootstrap
    module._rinkspace_verified_actor_worker = verified_actor
    pickle.dumps(verified_actor, protocol=5)
    if expected.read_bytes() != source_bytes:
        raise RuntimeError('Historical source worker file changed during installation')
    installation = dict(schema='rinkspace.source-worker-native-backend-installation.v1',
                        worker_module=module.__name__, worker_source_path=str(expected),
                        worker_source_sha256=hashlib.sha256(source_bytes).hexdigest(),
                        original_actor_function_text_sha256=hashlib.sha256(original_source.encode()).hexdigest(),
                        runtime_actor_function_text_sha256=hashlib.sha256(modified_source.encode()).hexdigest(),
                        insertion_exactly_after_original_block_unpickle=True,
                        original_algorithm_text_otherwise_exact=True,
                        dynamic_module_imports_checked=True,
                        post_unpickle_imports_checked=True,
                        original_top_level_pickle_module_and_name_preserved=True,
                        pickle_target_serialization_verified=True,
                        historical_source_file_unchanged=True,
                        numerical_equivalence_or_speedup_claimed=False,
                        already_installed=False)
    module._rinkspace_worker_backend_installation = installation
    return dict(installation)
