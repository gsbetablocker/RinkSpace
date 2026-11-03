"""Evaluate saved body problems; never fit, infer, render or replace a candidate.

Run with the project's existing native/body Python environment.  Full-body,
shared-boot and opening modes reconstruct the real 78dba30 problem classes and
consume their existing observation/state files.  The only numerical output is
a profile plus a residual/CSR snapshot in a NEW output directory.  Reusing the
existing viewer catalog may create its empty workbench/results3d directory;
no existing input array, candidate, checkpoint or viewer result is written.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time
from types import SimpleNamespace


PUBLIC_BASELINE = "78dba301449d49c429b7c3fee7717aea1a61cf7a"


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def nonnegative_float(value):
    value = float(value)
    if not 0 <= value < float('inf'):
        raise argparse.ArgumentTypeError("must be finite and nonnegative")
    return value


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kind', choices=('fullbody', 'boot', 'opening', 'motion', 'timing'), required=True)
    p.add_argument('--root', type=Path, required=True, help='Existing S01 body data root containing input.npz')
    p.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--output', type=Path, required=True, help='New directory; an existing path is rejected')
    p.add_argument('--backend', choices=('cpu', 'cuda'), default='cpu')
    p.add_argument('--workers', type=positive_int, default=1)
    p.add_argument('--blas-threads', type=positive_int, default=1)
    p.add_argument('--gpu-batch-size', type=positive_int, default=512)
    p.add_argument('--perturbation-batch', type=positive_int, default=4)
    p.add_argument('--max-batch-samples', type=positive_int, default=4096)
    p.add_argument('--native-library', type=Path, help='Read-only native DLL/SO override for this evaluation; camera record is unchanged')
    p.add_argument('--native-threads', type=positive_int, help='Native projection worker request; default: --workers')
    p.add_argument('--repeats', type=positive_int, default=3)
    p.add_argument('--state', type=Path, help='Existing NPZ with x; otherwise use the saved candidate for this kind')
    p.add_argument('--window', type=positive_int, default=1, help='Fullbody window number, 1..5')
    p.add_argument('--mode', choices=('full', 'legacy'), default='full', help='Fullbody state layout')
    p.add_argument('--compare-snapshot', type=Path, help='Snapshot from identical inputs and x')
    p.add_argument('--residual-atol', type=nonnegative_float, default=1e-8)
    p.add_argument('--residual-rtol', type=nonnegative_float, default=1e-10)
    p.add_argument('--jacobian-atol', type=nonnegative_float, default=5e-5)
    p.add_argument('--jacobian-rtol', type=nonnegative_float, default=2e-4)
    return p


def sha256_file(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def json_value(value):
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, 'tolist'):
        return value.tolist()
    raise TypeError(type(value).__name__)


def write_json(path, value):
    path = Path(path)
    pending = path.with_suffix(path.suffix + '.pending')
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=json_value, allow_nan=False), encoding='utf8')
    pending.replace(path)


def array_digest(array):
    import numpy as np
    a = np.ascontiguousarray(array)
    h = hashlib.sha256(str(a.shape).encode() + a.dtype.str.encode())
    h.update(a.tobytes())
    return h.hexdigest()


def input_identity(args, input_records, x, interpretation):
    # DLL/SO hashes belong to implementation provenance. Changing them while
    # keeping this exact camera state is precisely an intended comparison.
    excluded = {'camera_native_library', 'native_library', 'native_library_override'}
    return dict(kind=args.kind, mode=args.mode if args.kind == 'fullbody' else None,
                window=args.window if args.kind == 'fullbody' else None,
                input_files={name: value['sha256'] for name, value in input_records.items() if name not in excluded},
                x_sha256=array_digest(x), state_interpretation=interpretation)


def clear_result_caches(problem):
    """Remove evaluated states, keeping prepared immutable model/basis caches."""
    seen = set()
    def clear(obj):
        if obj is None or id(obj) in seen:
            return
        seen.add(id(obj))
        for key in ('cache', 'cache_x', 'cached', 'cached_x', 'binding_key'):
            if key in vars(obj):
                setattr(obj, key, None)
        for child in getattr(obj, 'windows', ()):
            clear(child)
        # Current bindings have no result cache; cover explicit result-cache
        # fields if the already selected implementation introduces them.
        binding = getattr(obj, 'binding', None)
        if binding is not None:
            for key in ('cache', 'cache_x', 'cached', 'cached_x', 'cache_key'):
                if key in vars(binding):
                    setattr(binding, key, None)
    clear(problem)


def process_peak_rss_bytes():
    """Process-lifetime peak, not an invented per-evaluation allocation peak."""
    if sys.platform == 'win32':
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD),
                        ('PeakWorkingSetSize', ctypes.c_size_t), ('WorkingSetSize', ctypes.c_size_t),
                        ('QuotaPeakPagedPoolUsage', ctypes.c_size_t), ('QuotaPagedPoolUsage', ctypes.c_size_t),
                        ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t), ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
                        ('PagefileUsage', ctypes.c_size_t), ('PeakPagefileUsage', ctypes.c_size_t)]
        get_current = ctypes.windll.kernel32.GetCurrentProcess
        get_current.restype = wintypes.HANDLE
        counters = Counters(); counters.cb = ctypes.sizeof(counters)
        get_memory = ctypes.windll.psapi.GetProcessMemoryInfo
        get_memory.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        if get_memory(get_current(), ctypes.byref(counters), counters.cb):
            return int(counters.PeakWorkingSetSize)
        return None
    try:
        import resource
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(value if sys.platform == 'darwin' else value * 1024)
    except (ImportError, AttributeError):
        return None


def _existing(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f'Required existing input is missing: {path}; this profiler does not prepare or fit it')
    return path


def _load_x(path):
    import numpy as np
    path = _existing(path)
    with np.load(path, allow_pickle=False) as data:
        if 'x' not in data:
            raise ValueError(f'State file has no x array: {path}')
        saved = {k: data[k].copy() for k in data.files if k in ('x', 'knots', 'active', 'offsets', 'seconds')}
    x = np.asarray(saved['x'], dtype=np.float64)
    if x.ndim != 1 or not np.isfinite(x).all():
        raise ValueError(f'Expected a finite one-dimensional saved x: {path}')
    return x, saved


def execution_classes(project, backend):
    """Use selected source hooks, or only a CPU resource scope for 78dba30."""
    selected = Path(project) / 'calibration/program_tracking/body_tracking/body_execution.py'
    if selected.is_file():
        from calibration.program_tracking.body_tracking import body_execution
        if Path(body_execution.__file__).resolve() != selected.resolve():
            raise ValueError('Imported body_execution is not from --project; use a clean interpreter without a competing PYTHONPATH')
        return body_execution.BodyExecution, body_execution.BodyExecutionConfig, 'selected_project_execution_hooks'
    if backend != 'cpu':
        raise ValueError('This baseline has no body execution hooks; only --backend cpu is valid. CUDA requires the patched batch integration')
    # Loading this one file does not put the optimized repository on sys.path:
    # every model/problem/camera import must still come from --project.
    helper = Path(__file__).resolve().parents[1] / 'calibration/program_tracking/body_tracking/body_execution.py'
    if not helper.is_file():
        raise FileNotFoundError('Baseline profiling requires this script from the optimized checkout, whose body_execution.py supplies CPU resource limits')
    name = '_rinkspace_profile_cpu_execution'
    spec = importlib.util.spec_from_file_location(name, helper)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses resolves its defining module here.
    spec.loader.exec_module(module)
    return module.BodyExecution, module.BodyExecutionConfig, 'baseline_without_execution_hooks'


def configure_native(camera, info, args):
    """Override only the loaded implementation, preserving all camera arrays."""
    from rinkspace.core.native import NativeCore
    declared = _existing(info['native_library'])
    selected = _existing(args.native_library) if args.native_library else declared
    if args.native_library:
        camera.core = NativeCore(selected)
    requested = args.native_threads if args.native_threads is not None else args.workers
    if hasattr(camera.core, 'motion_projection_threads'):
        camera.core.motion_projection_threads = requested
    def capability(name):
        value = getattr(camera.core, name, False)
        return bool(value() if callable(value) else value)
    return dict(declared_library=str(declared), declared_library_sha256=sha256_file(declared),
                evaluation_library=str(selected), evaluation_library_sha256=sha256_file(selected),
                override=bool(args.native_library), camera_record_modified=False,
                supports_motion_xyz_jacobian=capability('supports_motion_xyz_jacobian'),
                openmp_compiled=capability('supports_parallel_motion_projection'),
                requested_projection_threads=requested,
                configured_projection_threads=getattr(camera.core, 'motion_projection_threads', None),
                measured_openmp_team_size=None,
                thread_count_note='capability/configuration only; this ABI does not report the actual OpenMP team or machine utilization')


def verify_selected_modules(project):
    """Fail instead of labeling mixed checkouts as a baseline measurement."""
    project = Path(project).resolve()
    for name, module in tuple(sys.modules.items()):
        if not (name.startswith('calibration.program_tracking.body_tracking.') or name == 'rinkspace.core.native'):
            continue
        source = getattr(module, '__file__', None)
        if source is None:
            continue
        try:
            Path(source).resolve().relative_to(project)
        except ValueError as error:
            raise ValueError(f'Module {name} was loaded outside --project ({source}); use a fresh interpreter without a competing PYTHONPATH') from error


def construct_problem(args, execution):
    """Mirror existing constructors without invoking any data producer."""
    import numpy as np
    from calibration.program_tracking.body_tracking.fullbody_state import FrozenReference, camera_binding
    from calibration.program_tracking.body_tracking.fullbody_fit import FullBodyProblem
    root, project, output = args.root, args.project, args.output
    programme = project / 'events/ankara_2026/programmes/okada_mei_sp'
    extra_files = {}
    catalog = None
    interpretation = 'saved complete control vector'
    if args.kind == 'fullbody':
        from calibration.program_tracking.body_tracking.fullbody_round import WINDOWS
        if args.window > len(WINDOWS):
            raise ValueError(f'--window must be in 1..{len(WINDOWS)}')
        reference = FrozenReference(root)
        camera, info, _, _ = camera_binding(project, reference)
        native_info = configure_native(camera, info, args)
        execution.attach(reference)
        observation = _existing(root / 'fullbody_1102/observations.npz')
        problem = FullBodyProblem(reference, camera, observation, WINDOWS[args.window - 1], args.mode, log=None)
        state_path = _existing(args.state or root / 'fullbody_1102' / args.mode / f'w{args.window}' / 'result.npz')
    elif args.kind == 'boot':
        from rinkspace.viewer3d.data import ResultCatalog
        from rinkspace.viewer3d.prepare import FullSurface
        from calibration.program_tracking.body_tracking.footwear_binding import IntegratedBodyState, FootwearBinding
        from calibration.program_tracking.body_tracking.boot_integration import Registration, BootWindowProblem, JointBootProblem
        reference = IntegratedBodyState(root, root / 'fullbody_1102')
        camera, info, _, _ = camera_binding(project, reference)
        native_info = configure_native(camera, info, args)
        catalog = ResultCatalog(project=project, data_root=root.parents[1], config=programme / 'viewer3d_fullbody_1102.json')
        binding = FootwearBinding(reference, FullSurface(catalog), info)
        reference.binding = binding
        joint_path = root / 'boot_integration_1102/joint_result.npz'
        registration_path = root / 'boot_integration_1102/registration.npz'
        state_path = _existing(args.state or (joint_path if joint_path.is_file() else registration_path))
        saved_x, _ = _load_x(state_path)
        if len(saved_x) < 14:
            raise ValueError('The boot state must contain all 14 shared registration values')
        binding.x = saved_x[:14].copy()
        execution.attach(reference)
        observation = _existing(root / 'boot_integration_1102/observations.npz')
        registration = Registration(reference, camera, binding, output)
        windows = [BootWindowProblem(reference, camera, observation, seconds)
                   for seconds in ((35.2, 36.8), (40., 41.5))]
        problem = JointBootProblem(registration, windows, output)
        # Do not append saved boot w1/w2 to reference.refinements: those same
        # corrections are the active x blocks and would then be applied twice.
    else:
        if args.kind == 'opening':
            from calibration.program_tracking.body_tracking.opening_repair import setup, OpeningProblem
            problem_type = OpeningProblem
            default_state = root / 'opening_repair_1102/fit/result.npz'
        else:
            import importlib
            module = importlib.import_module('calibration.program_tracking.body_tracking.opening_' + args.kind)
            setup = module.build
            problem_type = getattr(module, 'MotionProblem' if args.kind == 'motion' else 'TimingProblem')
            default_state = root / ('opening_' + args.kind + '_1103') / 'fit/result.npz'
            extra_files['opening_objective_source'] = _existing(module.__file__)
            from calibration.program_tracking.body_tracking import opening_motion
            extra_files['motion_objective_source'] = _existing(opening_motion.__file__)
            source = json.loads((root / 'input.json').read_text(encoding='utf8'))
            extra_files['motion_clock_map'] = _existing(next(p for p in source['frozen_inputs']
                if p.endswith('native60_to_official50.jsonl')))
        reference, camera, info, catalog = setup(project, root, output)
        native_info = configure_native(camera, info, args)
        native_info['opening_reference_setup_used_declared_library'] = True
        execution.attach(reference)
        observation = _existing(root / 'boot_integration_1102/observations.npz')
        problem = problem_type(reference, camera, observation, None)
        state_path = _existing(args.state or default_state)
        for relative in ('boot_integration_1102/joint_result.npz', 'boot_integration_1102/w1/result.npz',
                         'boot_integration_1102/w2/result.npz', 'source_cache/raw_shot_01.npz'):
            extra_files[relative] = _existing(root / relative)
        preparation = reference if args.kind == 'opening' else reference.old
        if preparation.preparation_path.is_file():
            extra_files['opening_preparation'] = preparation.preparation_path

    x, saved = _load_x(state_path)
    if args.kind == 'boot':
        if x.shape == (14,):
            x = np.r_[x, np.zeros(sum(problem.sizes))]
            interpretation = 'saved 14-parameter registration plus explicit zero body-window corrections'
        if x.shape != problem.x0.shape:
            raise ValueError(f'State shape {x.shape} does not match boot layout {problem.x0.shape}')
        if 'offsets' in saved:
            np.testing.assert_array_equal(saved['offsets'], problem.offsets)
    else:
        if x.shape != (problem.size,):
            raise ValueError(f'State shape {x.shape} does not match selected window {(problem.size,)}')
        for key in ('knots', 'active'):
            if key in saved:
                np.testing.assert_array_equal(saved[key], getattr(problem.state, key))

    if args.kind != 'fullbody':
        for i in range(1, 6):
            extra_files[f'fullbody_reference_w{i}'] = _existing(root / f'fullbody_1102/full/w{i}/result.npz')
    if catalog is not None:
        for name in ('model_path', 'config_path', 'camera_path', 'metric_path'):
            path = getattr(catalog, name, None)
            if path is not None:
                extra_files['catalog_' + name] = _existing(path)
        if catalog.root.resolve() != root.resolve():
            raise ValueError('Viewer catalog source_run does not match --root')
    for path in reference.hashes:
        p = _existing(path)
        try:
            label = 'reference/' + str(p.relative_to(root)).replace('\\', '/')
        except ValueError:
            label = 'reference/' + p.name
        extra_files[label] = p
    for name in ('record', 'state_path'):
        extra_files['camera_' + name] = _existing(info[name])
    windows = getattr(problem, 'windows', (problem,))
    if hasattr(problem, 'registration'):
        owners = [problem.registration, *windows]
    else:
        owners = list(windows)
    for i, owner in enumerate(owners):
        sources = getattr(owner, 'gear_sources', getattr(owner, 'sources', {}))
        for name in ('review_path', 'admission_path'):
            if name in sources:
                extra_files[f'gear_{i}_{name}'] = _existing(sources[name])
    extra_files['observation'] = observation
    extra_files['saved_state'] = state_path
    # FullBodyProblem sets its native worker request from the execution policy;
    # honor an explicit per-profile override after all windows are constructed.
    if hasattr(camera.core, 'motion_projection_threads'):
        camera.core.motion_projection_threads = args.native_threads if args.native_threads is not None else args.workers
    native_info['configured_projection_threads'] = getattr(camera.core, 'motion_projection_threads', None)
    return SimpleNamespace(problem=problem, reference=reference, camera=camera, camera_info=info,
                           state_path=state_path, observation=observation, files=extra_files, x=x,
                           state_interpretation=interpretation, native_info=native_info)


def snapshot_arrays(value, x, metadata):
    import numpy as np
    jac = value['jac'].tocsr(copy=True)
    jac.sum_duplicates(); jac.sort_indices(); jac.eliminate_zeros()
    return dict(x=np.asarray(x, dtype=np.float64), residual=np.asarray(value['residual'], dtype=np.float64),
                jac_data=jac.data, jac_indices=jac.indices, jac_indptr=jac.indptr,
                jac_shape=np.asarray(jac.shape, dtype=np.int64),
                metadata_json=np.asarray(json.dumps(metadata, sort_keys=True, default=json_value)))


def compare_snapshot(current, baseline_path, *, residual_atol, residual_rtol, jacobian_atol, jacobian_rtol):
    import numpy as np
    from scipy.sparse import csr_matrix
    with np.load(_existing(baseline_path), allow_pickle=False) as data:
        old = {k: data[k].copy() for k in data.files}
    a = json.loads(str(current['metadata_json'])); b = json.loads(str(old['metadata_json']))
    if a['input_fingerprint'] != b['input_fingerprint'] or not np.array_equal(current['x'], old['x']):
        raise ValueError('Cannot compare snapshots: source inputs, objective layout or x differ')
    r, r0 = current['residual'], old['residual']
    if r.shape != r0.shape or not np.array_equal(current['jac_shape'], old['jac_shape']):
        raise ValueError('Snapshot residual/Jacobian dimensions differ')
    J = csr_matrix((current['jac_data'], current['jac_indices'], current['jac_indptr']), shape=tuple(current['jac_shape']))
    J0 = csr_matrix((old['jac_data'], old['jac_indices'], old['jac_indptr']), shape=tuple(old['jac_shape']))
    difference = J - J0
    rdiff = r - r0
    jnorm = float(np.linalg.norm(J0.data)); dnorm = float(np.linalg.norm(difference.data))
    return dict(baseline_snapshot=str(Path(baseline_path).resolve()),
                residual_max_abs=float(np.max(np.abs(rdiff), initial=0.)),
                residual_relative_l2=float(np.linalg.norm(rdiff) / max(np.linalg.norm(r0), 1e-15)),
                jacobian_max_abs=float(np.max(np.abs(difference.data), initial=0.)),
                jacobian_relative_frobenius=dnorm / max(jnorm, 1e-15),
                residual_within_tolerance=bool(np.allclose(r, r0, atol=residual_atol, rtol=residual_rtol)),
                jacobian_within_tolerance=bool(dnorm <= jacobian_atol + jacobian_rtol * jnorm),
                thresholds=dict(residual_atol=residual_atol, residual_rtol=residual_rtol,
                                jacobian_absolute_frobenius=jacobian_atol, jacobian_relative_frobenius=jacobian_rtol),
                automatic_backend_selection=False)


def run_profile(args):
    import numpy as np
    import scipy
    BodyExecution, BodyExecutionConfig, source_implementation = execution_classes(args.project, args.backend)
    execution = BodyExecution(BodyExecutionConfig(backend=args.backend, workers=args.workers,
                              blas_threads=args.blas_threads, gpu_batch_size=args.gpu_batch_size,
                              perturbation_batch=args.perturbation_batch, max_batch_samples=args.max_batch_samples))
    setup_start = time.perf_counter()
    try:
        setup = construct_problem(args, execution)
        verify_selected_modules(args.project)
        setup_seconds = time.perf_counter() - setup_start
        problem, x = setup.problem, setup.x
        backend = execution._body_backend
        def synchronize():
            if backend is not None and backend.device.type == 'cuda':
                backend.torch.cuda.synchronize(backend.device)
        synchronize()
        if backend is not None and backend.device.type == 'cuda':
            backend.torch.cuda.reset_peak_memory_stats(backend.device)
        input_records = {name: dict(path=str(path), sha256=sha256_file(path)) for name, path in sorted(setup.files.items())}
        identity = input_identity(args, input_records, x, setup.state_interpretation)
        fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        def timed(jacobian):
            synchronize(); wall_start = time.perf_counter(); cpu_start = time.process_time()
            with execution.limits():
                value = problem.evaluate(x, jacobian)
            synchronize()
            wall, cpu = time.perf_counter() - wall_start, time.process_time() - cpu_start
            stages = value.get('phases')
            if stages is None and hasattr(problem, 'windows'):
                stages = {f'window_{i+1}': dict((w.cache or {}).get('phases', {})) for i, w in enumerate(problem.windows)}
            return value, dict(wall_seconds=wall, process_cpu_seconds=cpu,
                               process_average_cpu_cores=cpu / max(wall, 1e-15),
                               with_jacobian=jacobian, phases=stages,
                               forward_reused=value.get('forward_reused'),
                               process_lifetime_peak_rss_bytes=process_peak_rss_bytes())
        clear_result_caches(problem)
        warm_value, warmup = timed(True)
        # Do not keep a second large warm-up Jacobian alive during measurements.
        del warm_value
        records = []
        for repeat in range(args.repeats):
            if repeat:
                del value
            clear_result_caches(problem)
            value, direct = timed(True)
            direct.update(repeat=repeat + 1, path='fresh_jacobian')
            records.append(direct)
            del value
            clear_result_caches(problem)
            forward, first = timed(False)
            forward_r = np.array(forward['residual'], copy=True)
            del forward
            value, second = timed(True)
            # A legitimate cached forward is allowed only WITHIN this pair.
            first.update(repeat=repeat + 1, path='fun_then_jac_fun')
            second.update(repeat=repeat + 1, path='fun_then_jac_jac',
                          same_state_residual_max_abs=float(np.max(np.abs(forward_r - value['residual']), initial=0.)))
            records.extend((first, second))
            print(json.dumps(dict(event='profile_repeat', repeat=repeat + 1,
                                  fresh_jacobian_seconds=direct['wall_seconds'],
                                  fun_then_jac_seconds=first['wall_seconds'] + second['wall_seconds'])), flush=True)
        snapshot = snapshot_arrays(value, x, dict(input_fingerprint=fingerprint, identity=identity))
        J = value['jac']; r = value['residual']
        if not np.isfinite(r).all() or not np.isfinite(J.data).all():
            raise ValueError('Nonfinite evaluation output; refusing a speed result')
        direct_times = [v['wall_seconds'] for v in records if v['path'] == 'fresh_jacobian']
        pair_times = [sum(v['wall_seconds'] for v in records if v['repeat'] == i and v['path'].startswith('fun_then_jac'))
                      for i in range(1, args.repeats + 1)]
        children = getattr(problem, 'windows', (problem,))
        frames = [np.asarray(child.frames, dtype=np.int64) for child in children]
        if hasattr(problem, 'registration'):
            frames.append(np.asarray(problem.registration.frames, dtype=np.int64))
        frame_union = np.unique(np.concatenate(frames))
        implementation_files = [Path(__file__), *[args.project / 'calibration/program_tracking/body_tracking' / name
            for name in ('body_execution.py', 'compact_body_torch.py', 'rebuild_body_geometry.py', 'fullbody_state.py',
                         'fullbody_fit.py', 'boot_integration.py', 'footwear_binding.py', 'opening_repair.py',
                         'rebuild_exposure_body.py', 'rebuild_body_fit.py', 'footwear_pixel_factors.py')],
            args.project / 'src/rinkspace/viewer3d/prepare.py']
        execution_record = execution.record()
        hooks_active = source_implementation != 'baseline_without_execution_hooks'
        execution_record.update(body_execution_hooks_active=hooks_active,
                                effective_parallel_finite_difference_workers=args.workers if hooks_active else 1)
        report = dict(schema='rinkspace.body-evaluation-profile.v1', utc=datetime.now(timezone.utc).isoformat(),
                      public_baseline=PUBLIC_BASELINE, kind=args.kind, input_fingerprint=fingerprint,
                      state_interpretation=setup.state_interpretation, input_files=input_records,
                      implementation_sha256={str(p): sha256_file(p) for p in implementation_files if p.is_file()},
                      execution=execution_record, camera=setup.camera_info,
                      source_implementation=source_implementation, native_implementation=setup.native_info,
                      runtime=dict(python=sys.version, executable=sys.executable, numpy=np.__version__, scipy=scipy.__version__,
                                   platform=platform.platform(), processor=platform.processor(),
                                   logical_cpu_count=os.cpu_count(), cpu_accounting='this process including its threads; no machine-wide utilization'),
                      setup_seconds=setup_seconds, first_jacobian_warmup=warmup, evaluations=records,
                      median_fresh_jacobian_seconds=statistics.median(direct_times),
                      median_fun_then_jac_seconds=statistics.median(pair_times),
                      repetitions=args.repeats, cost=float(r @ r / 2), cost_parts=value.get('parts'),
                      gradient_inf=float(np.max(np.abs(J.T @ r), initial=0.)), jacobian_shape=list(J.shape),
                      jacobian_nnz=int(J.nnz), variables=len(x), residuals=len(r),
                      source_frames=len(frame_union), source_pts_sha256=array_digest(setup.reference.p['pts'][frame_union]),
                      process_lifetime_peak_rss_bytes=process_peak_rss_bytes(),
                      measured_scope='evaluation only; no linear solve or trust-region iteration was run',
                      caches='result/binding-key caches cleared before each fresh path; prepared immutable model and spline caches retained; same-state forward reuse allowed only within fun->jac',
                      input_arrays_and_existing_candidates_modified=False,
                      catalog_side_effect='existing ResultCatalog initialization may create its empty workbench/results3d directory',
                      automatic_backend_selection=False)
        try:
            from threadpoolctl import threadpool_info
            report['runtime']['thread_pools'] = threadpool_info()
        except ImportError:
            report['runtime']['thread_pools'] = None
        if backend is not None and backend.device.type == 'cuda':
            report['gpu_memory_bytes'] = dict(peak_allocated=backend.torch.cuda.max_memory_allocated(backend.device),
                                             peak_reserved=backend.torch.cuda.max_memory_reserved(backend.device),
                                             current_allocated=backend.torch.cuda.memory_allocated(backend.device),
                                             scope='PyTorch allocator on selected device after problem construction; excludes native/other-process allocations')
        if args.compare_snapshot:
            report['comparison'] = compare_snapshot(snapshot, args.compare_snapshot,
                residual_atol=args.residual_atol, residual_rtol=args.residual_rtol,
                jacobian_atol=args.jacobian_atol, jacobian_rtol=args.jacobian_rtol)
        np.savez_compressed(args.output / 'snapshot.npz', **snapshot)
        write_json(args.output / 'profile.json', report)
        return report
    finally:
        execution.close()


def main(argv=None):
    p = parser(); args = p.parse_args(argv)
    if args.backend == 'cuda' and args.workers != 1:
        p.error('--backend cuda requires --workers 1; perturbations are batched instead')
    args.project = args.project.resolve(); args.root = args.root.resolve(); args.output = args.output.resolve()
    if args.state is not None:
        args.state = args.state.resolve()
    if args.native_library is not None:
        args.native_library = args.native_library.resolve()
    if args.compare_snapshot is not None:
        args.compare_snapshot = args.compare_snapshot.resolve()
    if args.output.exists():
        p.error('--output must be a new directory, so existing candidates cannot be overwritten')
    args.output.mkdir(parents=True, exist_ok=False)
    print(json.dumps(dict(profile=str(args.output / 'profile.json'),
                          snapshot=str(args.output / 'snapshot.npz'), operation='evaluation_only')), flush=True)
    for name, value in dict(BACKEND=args.backend, WORKERS=args.workers, BLAS_THREADS=args.blas_threads,
                            GPU_BATCH_SIZE=args.gpu_batch_size, PERTURBATION_BATCH=args.perturbation_batch,
                            MAX_BATCH_SAMPLES=args.max_batch_samples).items():
        os.environ['RINKSPACE_BODY_' + name] = str(value)
    for name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
        os.environ[name] = str(args.blas_threads)
    sys.path[:0] = [str(args.project / 'src'), str(args.project)]
    started = time.perf_counter()
    try:
        # Match the established Windows launcher order: native DLL first,
        # then body/viewer imports and optional Torch. Keep this owner alive.
        from rinkspace.core.native import NativeCore
        record_path = _existing(args.project / 'events/ankara_2026/programmes/okada_mei_sp/fit_v11/camera_solver_refined_1102.json')
        record = json.loads(record_path.read_text(encoding='utf8'))
        held_core = NativeCore(args.native_library or record['native_library']['path'])
        report = run_profile(args)
        report['launcher_wall_seconds'] = time.perf_counter() - started
        write_json(args.output / 'profile.json', report)
        print(json.dumps(dict(profile=str(args.output / 'profile.json'),
                              median_fresh_jacobian_seconds=report['median_fresh_jacobian_seconds'],
                              median_fun_then_jac_seconds=report['median_fun_then_jac_seconds'],
                              comparison=report.get('comparison'))), flush=True)
        return 0
    except BaseException as error:
        write_json(args.output / 'profile_failed.json', dict(error=repr(error), kind=args.kind,
                   elapsed_seconds=time.perf_counter() - started, no_solver_run=True))
        raise


if __name__ == '__main__':
    raise SystemExit(main())
