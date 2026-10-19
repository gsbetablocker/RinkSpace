"""Read-only Windows RAM/commit telemetry for explicitly owned solver PIDs.

No solver graph, material owner, residual, Jacobian, native backend, admission
threshold or process-control API is used. PID creation tokens pin ownership.
Seen descendants remain tracked if their launcher exits before they do.
"""
from __future__ import annotations

import argparse
import ctypes as ct
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'src')]
from rinkspace.runtime_memory import (
    ProcessIdentity, _PerformanceInfo, _process_table, _windows_api,
    process_identity, read_system_memory, sample_owned_tree,
)

DATA = Path('D:/CodexData/RinkSpace_Data').resolve()


def _global_commit():
    """Exact system totals from GetPerformanceInfo; boot peak is marked apart."""
    performance = _PerformanceInfo()
    performance.cb = ct.sizeof(performance)
    _, psapi = _windows_api()
    if not psapi.GetPerformanceInfo(ct.byref(performance), ct.sizeof(performance)):
        raise ct.WinError(ct.get_last_error())
    page = int(performance.PageSize)
    return dict(
        global_committed_bytes=int(performance.CommitTotal)*page,
        global_commit_limit_bytes=int(performance.CommitLimit)*page,
        global_commit_available_bytes=int(performance.CommitLimit-performance.CommitTotal)*page,
        OS_boot_commit_peak_bytes=int(performance.CommitPeak)*page,
        OS_boot_peak_is_not_this_run_peak=True,
    )


def _atomic_json(path, value):
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, separators=(',',':'))+'\n', encoding='utf8')
    # Same existing Windows read-handle retry used by the solver checkpoint.
    # This is telemetry I/O only; it neither delays nor controls solver work.
    for attempt in range(64):
        try:
            temporary.replace(path)
            return
        except OSError as exc:
            if getattr(exc,'winerror',None) not in (5,32) or attempt == 63:
                raise
            time.sleep(min(.05*2**attempt,.5))


def _actor_roles(run):
    if run is None:
        return {}
    path = run/'source_runtime_memory_live.json'
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding='utf8'))
    except (OSError,ValueError,UnicodeError):
        return {}
    return {int(pid):str(block) for pid,block in zip(value.get('actor_pids',[]),value.get('source_blocks',[]))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pid', action='append', type=int, required=True,
        help='Explicit owned launcher/main PID; may repeat to pin both.')
    parser.add_argument('--creation-token', action='append', type=int,
        help='Optional exact creation tokens in the same order as --pid.')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path,
        help='Read only small resident actor PID telemetry for source-block labels.')
    parser.add_argument('--interval-ms', type=int, default=500,
        help='Sampling cadence only; never an execution budget or memory threshold.')
    args = parser.parse_args()
    if args.interval_ms < 1:
        parser.error('Positive sampling cadence required')
    if args.creation_token is not None and len(args.creation_token) != len(args.pid):
        parser.error('One exact creation token per explicit PID required')
    output = args.output_dir.resolve()
    if not output.is_relative_to(DATA):
        parser.error('Solver telemetry belongs in the existing RinkSpace Data tree')
    output.mkdir(parents=True, exist_ok=True)
    roots = tuple(process_identity(pid) for pid in args.pid)
    if args.creation_token is not None:
        expected = tuple(ProcessIdentity(pid,token) for pid,token in zip(args.pid,args.creation_token))
        if roots != expected:
            raise RuntimeError('Owned PID creation token changed before monitoring; no new process attached')
    if os.getpid() in args.pid:
        raise ValueError('The independent monitor cannot own its own lifetime')
    pinned = {item.pid:item for item in roots}
    all_seen = dict(pinned)
    process_peaks = {}
    sample_count = 0
    first_system = None
    max_rss = max_private = max_global_commit = max_RAM_used = 0
    min_physical = min_commit = None
    sample_errors = 0
    started = time.time()
    journal = output/'owned_memory_samples.jsonl'
    if journal.exists():
        raise FileExistsError('Use a fresh owned telemetry destination; existing samples are preserved')
    with journal.open('x', encoding='utf8') as log:
        while True:
            tick = time.perf_counter()
            table = _process_table()
            live = {}
            uncertain = []
            for pid,identity in pinned.items():
                if pid not in table:
                    continue  # An absent PID is terminal for this exact identity.
                try:
                    current = process_identity(pid)
                except OSError as exc:
                    live[pid] = identity
                    uncertain.append(dict(identity=asdict(identity),error=str(exc)))
                    continue  # An observation failure is not process termination.
                if current == identity:
                    live[pid] = identity
                # A reused PID does not transfer ownership to the new process.
            pinned = live
            owned = sample_owned_tree(tuple(live.values())) if live else None
            system = read_system_memory()
            commit = _global_commit()
            first_system = first_system or dict(system=asdict(system),commit=commit)
            sample_count += 1
            rows = []
            actor_roles = _actor_roles(args.run_dir)
            if owned is not None:
                sample_errors += bool(owned.errors or uncertain)
                for process in owned.processes:
                    if process.identity.pid == os.getpid():
                        continue
                    pinned[process.identity.pid] = process.identity
                    all_seen[process.identity.pid] = process.identity
                    item = asdict(process)
                    item['role'] = ('explicit_owned_root' if process.identity in roots else
                        'residual_actor:'+actor_roles[process.identity.pid] if process.identity.pid in actor_roles else
                        'owned_descendant_role_not_assumed')
                    rows.append(item)
                    key = str(process.identity.pid)+':'+str(process.identity.creation_token)
                    previous = process_peaks.get(key)
                    if previous is None:
                        previous = dict(identity=asdict(process.identity),parent_pid=process.parent_pid,
                            name=process.name,role=item['role'],max_sampled_rss_bytes=0,
                            max_sampled_private_bytes=0,OS_process_peak_rss_bytes=0,
                            OS_process_peak_private_bytes=0,first_seen_at=system.sampled_at)
                        process_peaks[key] = previous
                    if item['role'].startswith('residual_actor:'):
                        previous['role'] = item['role']
                    previous['last_seen_at'] = system.sampled_at
                    previous['max_sampled_rss_bytes'] = max(previous['max_sampled_rss_bytes'],process.rss_bytes)
                    previous['max_sampled_private_bytes'] = max(previous['max_sampled_private_bytes'],process.private_bytes)
                    previous['OS_process_peak_rss_bytes'] = max(previous['OS_process_peak_rss_bytes'],process.peak_rss_bytes)
                    previous['OS_process_peak_private_bytes'] = max(previous['OS_process_peak_private_bytes'],process.peak_private_bytes)
            rss = sum(item['rss_bytes'] for item in rows)
            private = sum(item['private_bytes'] for item in rows)
            max_rss = max(max_rss,rss)
            max_private = max(max_private,private)
            max_global_commit = max(max_global_commit,commit['global_committed_bytes'])
            RAM_used = system.physical_total_bytes-system.physical_available_bytes
            max_RAM_used = max(max_RAM_used,RAM_used)
            min_physical = system.physical_available_bytes if min_physical is None else min(min_physical,system.physical_available_bytes)
            min_commit = commit['global_commit_available_bytes'] if min_commit is None else min(min_commit,commit['global_commit_available_bytes'])
            alive = bool(pinned)
            sample = dict(sample=sample_count,sampled_at=system.sampled_at,system=asdict(system),
                commit=commit,owned_processes=rows,owned_sampled_rss_bytes=rss,
                owned_sampled_private_bytes=private,owned_observation_errors=[]if owned is None else list(owned.errors),
                uncertain_owned_identity_reads=uncertain,any_owned_identity_alive=alive)
            log.write(json.dumps(sample,ensure_ascii=False,separators=(',',':'))+'\n')
            log.flush()
            summary = dict(schema='rinkspace.explicit-owned-solver-real-Windows-memory-peaks.v2',
                explicit_roots=[asdict(item) for item in roots],monitor_pid=os.getpid(),
                any_owned_identity_alive=alive,sample_count=sample_count,interval_ms=args.interval_ms,
                started_at=started,latest_sample=sample,first_system_sample=first_system,
                owned_coincident_sampled_peak_rss_bytes=max_rss,
                owned_coincident_sampled_peak_private_bytes=max_private,
                sampled_run_global_commit_peak_bytes=max_global_commit,
                sampled_run_system_RAM_used_peak_bytes=max_RAM_used,
                minimum_global_commit_available_bytes=min_commit,
                minimum_physical_available_bytes=min_physical,
                per_owned_process_measured_peaks=list(process_peaks.values()),
                incomplete_or_uncertain_sample_count=sample_errors,
                monitor_only_no_process_or_solver_mutation=True,
                no_memory_threshold_admission_or_task_termination=True,
                source_graph_material_models_residuals_J_or_native_backends_loaded=False,
                only_explicit_roots_and_authenticated_descendants_attached=True,
                seen_descendants_tracked_after_launcher_exit=True,
                RSS_sum_can_double_count_shared_pages=True,
                global_commit_peak_includes_other_apps_and_is_not_owned_private_sum=True,
                peaks_are_sampled_except_explicitly_marked_OS_lifetime_peaks=True)
            _atomic_json(output/'owned_memory_live.json',summary)
            if not alive:
                _atomic_json(output/'owned_memory_terminal_summary.json',summary)
                print(json.dumps(dict(stage='owned_memory_monitor_complete',samples=sample_count,
                    summary_path=str(output/'owned_memory_terminal_summary.json')),ensure_ascii=False),flush=True)
                return
            time.sleep(max(0.,args.interval_ms/1000.-(time.perf_counter()-tick)))


if __name__ == '__main__':
    main()
