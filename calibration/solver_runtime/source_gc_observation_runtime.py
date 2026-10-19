"""Observe parent cyclic GC without changing its execution policy.

Only a caller's explicit install adds a callback. No objects being scanned
are retained. The atexit summary ends at that callback, not actual OS exit;
later generation-2 callbacks append best-effort events during finalization.
"""
from __future__ import annotations

import atexit
import gc
import hashlib
import json
import os
from pathlib import Path
import time


_STATE = None


def _totals(state):
    return [dict(generation=generation, collections=state['counts'][generation],
        collection_seconds=state['nanoseconds'][generation]/1e9,
        collected=state['collected'][generation],
        uncollectable=state['uncollectable'][generation]) for generation in range(3)]


def _append_event(state, event):
    # Output allocates only a small scalar record after collection completes.
    # A guard in the callback excludes observer recursion; no object census,
    # explicit collection, GC policy change or source-object retention occurs.
    try:
        with state['events_path'].open('a', encoding='utf8') as stream:
            stream.write(json.dumps(event, separators=(',', ':'))+'\n')
    except BaseException:
        state['observation_write_failures'] += 1


def _on_collection(phase, info):
    state = _STATE
    if state is None:
        return
    try:
        if state['in_callback'] or os.getpid() != state['pid']:
            return
    except BaseException:
        # Module teardown can remove supporting globals after atexit.
        return
    state['in_callback'] = True
    try:
        generation = int(info['generation'])
        if generation not in (0, 1, 2):
            return
        now = time.perf_counter_ns()
        if phase == 'start':
            state['started_ns'][generation] = now
            return
        if phase != 'stop':
            return
        start = state['started_ns'][generation]
        state['started_ns'][generation] = 0
        if not start:
            state['unpaired_stops'] += 1
            return
        elapsed = now-start
        state['counts'][generation] += 1
        state['nanoseconds'][generation] += elapsed
        collected = int(info.get('collected', 0))
        uncollectable = int(info.get('uncollectable', 0))
        state['collected'][generation] += collected
        state['uncollectable'][generation] += uncollectable
        if generation == 2:
            _append_event(state, dict(schema='rinkspace.parent-GC-observation.v1',
                event='generation_2_completed', pid=state['pid'],
                generation=generation, collection_seconds=elapsed/1e9,
                collected=collected, uncollectable=uncollectable,
                elapsed_since_install_seconds=(now-state['install_monotonic_ns'])/1e9,
                wall_time_ns=time.time_ns(), after_atexit_summary=state['atexit_reported'],
                generations=_totals(state)))
    except BaseException:
        # Observability must never replace or suppress a solver exception.
        state['observation_callback_failures'] += 1
    finally:
        state['in_callback'] = False


def _write_atexit_summary():
    state = _STATE
    if state is None or os.getpid() != state['pid']:
        return
    try:
        state['atexit_reported'] = True
        now = time.perf_counter_ns()
        event = dict(schema='rinkspace.parent-GC-observation.v1',
            event='atexit_summary', pid=state['pid'],
            boundary='through_this_atexit_callback_only',
            actual_OS_exit_or_all_finalization_GC_not_claimed=True,
            later_generation_2_events_are_best_effort=True,
            elapsed_since_install_seconds=(now-state['install_monotonic_ns'])/1e9,
            wall_time_ns=time.time_ns(), generations=_totals(state),
            entry_gc_enabled=state['entry_gc_enabled'], exit_gc_enabled=gc.isenabled(),
            entry_gc_thresholds=state['entry_gc_thresholds'], exit_gc_thresholds=list(gc.get_threshold()),
            unpaired_stops=state['unpaired_stops'],
            observation_write_failures=state['observation_write_failures'],
            observation_callback_failures=state['observation_callback_failures'],
            collect_disable_freeze_or_threshold_changes_performed=False)
        with state['summary_path'].open('w', encoding='utf8') as stream:
            stream.write(json.dumps(event, indent=2)+'\n')
    except BaseException:
        state['observation_write_failures'] += 1


def install(report_directory):
    """Observe this explicitly installing process, without changing GC state."""
    global _STATE
    if _STATE is not None:
        return _STATE['receipt']
    destination = Path(report_directory).resolve()
    try:
        destination.mkdir(parents=True, exist_ok=True)
    except OSError:
        # Missing optional telemetry storage cannot stop original execution.
        pass
    enabled, thresholds = gc.isenabled(), list(gc.get_threshold())
    receipt = dict(schema='rinkspace.parent-GC-observation-install.v1',
        helper_path=str(Path(__file__).resolve()),
        helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        pid=os.getpid(), report_directory=str(destination),
        only_explicit_installing_process_observed=True,
        original_GC_enabled_state_and_thresholds_retained=True,
        no_collect_disable_freeze_or_object_census=True,
        generation_2_completed_events_and_atexit_summary=True,
        atexit_summary_is_not_OS_exit_or_finalization_completion=True)
    _STATE = dict(receipt=receipt, pid=receipt['pid'],
        events_path=destination/'source_gc_observation.jsonl',
        summary_path=destination/'source_gc_observation_summary.json',
        install_monotonic_ns=time.perf_counter_ns(), entry_gc_enabled=enabled,
        entry_gc_thresholds=thresholds, started_ns=[0, 0, 0], counts=[0, 0, 0],
        nanoseconds=[0, 0, 0], collected=[0, 0, 0], uncollectable=[0, 0, 0],
        in_callback=False, atexit_reported=False, unpaired_stops=0,
        observation_write_failures=0, observation_callback_failures=0)
    gc.callbacks.append(_on_collection)
    atexit.register(_write_atexit_summary)
    return receipt


install_source_gc_observation_runtime = install
