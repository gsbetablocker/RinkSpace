"""Short GC scheduling scope for original source assembly and deserialization.

Only automatic cyclic-GC scheduling is suspended. Original explicit collect
calls, reference counting, source IO and object reducers remain unchanged.
This context must end before actor ready or normal/solver execution begins.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import gc
import json
from pathlib import Path
import time


_DEPTH = ContextVar('rinkspace_source_assembly_gc_depth', default=0)


@contextmanager
def defer_automatic_source_gc(label, report_directory=None):
    """Restore the exact entry enable state in finally, including nested errors.

    Do not freeze, collect, change thresholds, or retain assembled objects.
    The caller owns the narrow pre-execution boundary; no global installer.
    """
    enabled = gc.isenabled()
    depth = _DEPTH.get()
    token = _DEPTH.set(depth+1)
    counts, thresholds = gc.get_count(), gc.get_threshold()
    start = time.perf_counter_ns()
    if enabled:
        gc.disable()
    succeeded = False
    try:
        yield
        succeeded = True
    finally:
        if enabled:
            gc.enable()
        else:
            gc.disable()
        _DEPTH.reset(token)
        if report_directory is not None:
            value = dict(schema='rinkspace.original-source-assembly-GC-scheduling.v1',
                label=label, nested_depth=depth, elapsed_seconds=(time.perf_counter_ns()-start)/1e9,
                entry_gc_enabled=enabled, exit_gc_enabled=gc.isenabled(),
                original_thresholds=list(thresholds), entry_counts=list(counts), exit_counts=list(gc.get_count()),
                scope_succeeded=succeeded, explicit_original_collect_unchanged=True,
                extra_collect_freeze_or_threshold_change=False, wall_time_ns=time.time_ns())
            try:
                destination = Path(report_directory)
                destination.mkdir(parents=True, exist_ok=True)
                with (destination/'source_assembly_gc_events.jsonl').open('a', encoding='utf8') as stream:
                    stream.write(json.dumps(value, sort_keys=True)+'\n')
            except Exception:
                # Optional observability must not replace an original error.
                pass


source_assembly_gc_scope = defer_automatic_source_gc
