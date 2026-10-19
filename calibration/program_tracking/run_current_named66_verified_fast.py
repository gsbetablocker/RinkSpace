"""Lightweight Windows spawn entry for the verified complete programme solver."""
from pathlib import Path
import os
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
os.environ.update(RINKSPACE_MOTION_BACKEND='native', RINKSPACE_MOTION_THREADS='3',
    RINKSPACE_SCATTER_BACKEND='native', RINKSPACE_SCATTER_THREADS='3',
    RINKSPACE_PROJECTION_BACKEND='native', RINKSPACE_PROJECTION_VALUE_BACKEND='opencv',
    RINKSPACE_SOLVER_ACCELERATION='auto')
from calibration.solver_runtime import install
BASE = install()
from structured_programme_source_candidate.runtime_backend_contract import install_source_worker_backend_contract
install_source_worker_backend_contract()


def main():
    from calibration.solver_runtime.strict_model_scope_inputs import require_solver_scope_policy
    require_solver_scope_policy(PROJECT_ROOT / 'configs' / 'solver_runtime' / 'okada_named66.json',
        require_policy=True)
    import ctypes as ct
    from ctypes import wintypes as wt
    import json
    import math
    import threading
    import time
    import run_current_named66_full_programme as driver
    from rinkspace.runtime_memory import process_identity, sample_owned_tree, _windows_api, _open_process
    output = Path(sys.argv[sys.argv.index('--output') + 1])
    root = process_identity()
    stop = threading.Event()
    samples = []
    previous = {}
    last = time.perf_counter()

    def monitor():
        nonlocal previous, last
        while not stop.wait(.25):
            now = time.perf_counter()
            current = {}
            tree = sample_owned_tree([root])
            kernel, _ = _windows_api()
            for p in tree.processes:
                try:
                    handle = _open_process(p.identity.pid)
                    try:
                        values = [wt.FILETIME() for _ in range(4)]
                        if kernel.GetProcessTimes(handle, *[ct.byref(v) for v in values]):
                            current[(p.identity.pid, p.identity.creation_token)] = sum(
                                (v.dwHighDateTime << 32) | v.dwLowDateTime for v in values[2:]) / 1e7
                    finally:
                        kernel.CloseHandle(handle)
                except OSError:
                    pass
            cpu = sum(max(0, value - previous.get(pid, value)) for pid, value in current.items()) / (now - last) * 100 / os.cpu_count()
            samples.append(dict(t=now, cpu_percent_of_machine=cpu,
                owned_processes=len(current), rss_bytes=tree.rss_bytes))
            previous, last = current, now

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    original = driver.solve_joint_source

    def measured_solve(evaluate, q, *args, **kwargs):
        from z190_streaming_source_normal import _block_cost
        import cProfile
        import pstats
        before = time.perf_counter()
        costs = []
        rows = 0
        for block in kwargs['source_block_factory'](q, False):
            costs.append(_block_cost(block.r, block.quadratic))
            rows += len(block.r)
        warmup = time.perf_counter() - before
        report = dict(stage='all_current_source_actors_warmed', seconds=warmup,
            source_rows=rows, full_cost=math.fsum(costs), source_formula_or_observations_changed=False)
        print(json.dumps(report), flush=True)
        (output / 'actual_pool_warmup.json').write_text(json.dumps(report, indent=2))
        started = time.perf_counter()
        profiler = cProfile.Profile()
        profiling = os.environ.get('RINKSPACE_SOLVER_PROFILE') == '1'
        if profiling:
            profiler.enable()
        try:
            return original(evaluate, q, *args, **kwargs)
        finally:
            if profiling:
                profiler.disable()
            elapsed = time.perf_counter() - started
            if profiling:
                profiler.dump_stats(str(output / 'actual_warm_cycle_profile.pstats'))
                stats = pstats.Stats(profiler)
                hot = [dict(file=file, line=line, function=function, calls=v[1],
                    self_seconds=v[2], cumulative_seconds=v[3])
                    for (file, line, function), v in stats.stats.items()]
                (output / 'actual_warm_cycle_hot_calls.json').write_text(json.dumps(
                    sorted(hot, key=lambda row: row['cumulative_seconds'], reverse=True)[:80], indent=2))
            peak = max((x['cpu_percent_of_machine'] for x in samples if x['t'] >= started), default=0)
            data = dict(schema='rinkspace.current-full-source-warm-cycle.v1', solve_seconds=elapsed,
                pool_warmup_seconds=warmup, cold_cycle_including_pool_seconds=warmup + elapsed,
                cpu_peak_percent_of_machine=peak, logical_cpu_count=os.cpu_count(),
                same_q_normal_and_full_current_source_inventory=True)
            (output / 'actual_complete_solver_wall.json').write_text(json.dumps(data, indent=2))
            print(json.dumps(dict(stage='actual_complete_solver_wall', **data)), flush=True)

    driver.solve_joint_source = measured_solve
    try:
        driver.main()
    finally:
        stop.set()
        thread.join(2)
        if output.exists():
            (output / 'actual_owned_cpu_samples.json').write_text(json.dumps(samples, indent=2))
            (output / 'actual_loaded_solver_sources.json').write_text(json.dumps({
                name: str(Path(module.__file__).resolve())
                for name, module in tuple(sys.modules.items())
                if getattr(module, '__file__', None) and (
                    str(BASE).lower() in str(Path(module.__file__).resolve()).lower()
                    or 'RinkSpace' in str(Path(module.__file__).resolve()))
            }, indent=2))


if __name__ == '__main__':
    main()
