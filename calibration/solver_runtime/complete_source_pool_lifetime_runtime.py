"""Keep the original owned geometry pool through one complete fresh source J.

The programme boundary used to destroy workers that later native source
columns immediately needed again. Only that normal boundary is deferred;
failure cleanup, explicit close and other release requests remain immediate.
The full-source generator releases pending pools on completion or failure,
before a subsequent residual/guard command may restart its separate actors.
"""
from __future__ import annotations

from contextvars import ContextVar
import functools
import hashlib
import inspect
import json
from pathlib import Path
import sys
import time

_SCOPE = ContextVar('rinkspace_complete_fresh_source_pool_lifetime', default=None)
_RECEIPT = None


def install_complete_source_pool_lifetime_runtime(driver, *, report_directory=None):
    global _RECEIPT
    if _RECEIPT is not None:
        return _RECEIPT
    from .absolute_geometry_fd_pool_candidate import AbsoluteGeometryGroupPool
    from structured_programme_source_candidate.paired_programme_chart import CanonicalCompleteProgrammeSource
    from structured_programme_source_candidate.human_stroke_truth import SourceWithDirectManualCentres
    original_release = AbsoluteGeometryGroupPool.release_idle
    block_owners = (CanonicalCompleteProgrammeSource, SourceWithDirectManualCentres)
    original_blocks = {owner: owner.blocks for owner in block_owners}
    destination = None if report_directory is None else Path(report_directory)
    if destination is not None:
        destination.mkdir(parents=True, exist_ok=True)

    def record(**value):
        value.update(stage='complete_source_geometry_pool_lifetime', wall_time_ns=time.time_ns())
        print(json.dumps(value), flush=True)
        if destination is not None:
            with (destination/'pool_lifetime.jsonl').open('a', encoding='utf8') as stream:
                stream.write(json.dumps(value)+'\n')

    @functools.wraps(original_release)
    def release(pool):
        scope = _SCOPE.get()
        caller = sys._getframe(1)
        programme_boundary = (caller.f_code.co_name == 'release_finished_geometry_fd_stage'
                              and caller.f_globals.get('__name__') == driver.__name__)
        del caller
        if (scope is not None and programme_boundary and pool.owns_executor
                and not pool.closed and pool.executor is not None
                and not getattr(pool, '_rinkspace_active_geometry_group_iterators', 0)
                and not any(getattr(pool, name, 0) for name in (
                    '_rinkspace_confirmed_columns_pending_jobs',
                    '_rinkspace_native_columns_pending_jobs',
                    '_rinkspace_shared_material_bank_pending_jobs'))):
            scope['pending'][id(pool)] = pool
            record(action='deferred_original_programme_boundary_release',
                   original_workers=pool.workers, executor_identity=id(pool.executor),
                   source_rows_or_derivative_operations_changed=False)
            return False
        return original_release(pool)

    def wrap_blocks(original):
        @functools.wraps(original)
        def blocks(owner, q, jacobian=True):
            if not jacobian or _SCOPE.get() is not None:
                yield from original(owner, q, jacobian)
                return
            scope = dict(pending={})
            token = _SCOPE.set(scope)
            success = False
            try:
                yield from original(owner, q, jacobian)
                success = True
            finally:
                _SCOPE.reset(token)
                cleanup_errors = []
                for pool in scope['pending'].values():
                    try:
                        started = time.perf_counter()
                        result = original_release(pool)
                        record(action='released_after_complete_source_iterator',
                               source_iterator_succeeded=success, original_workers=pool.workers,
                               released=bool(result), seconds=time.perf_counter()-started)
                    except BaseException as error:
                        cleanup_errors.append(error)
                # An original numerical/control failure stays primary. The
                # driver's original outer finally also owns complete pool.close.
                if success and cleanup_errors:
                    raise cleanup_errors[0]
        return blocks

    AbsoluteGeometryGroupPool.release_idle = release
    for owner, original in original_blocks.items():
        owner.blocks = wrap_blocks(original)
    _RECEIPT = dict(schema='rinkspace.complete-fresh-source-owned-pool-lifetime.v1',
        helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        original_blocks_sha256={owner.__name__: hashlib.sha256(inspect.getsource(original).encode()).hexdigest()
                               for owner, original in original_blocks.items()},
        only_original_driver_programme_release_deferred=True,
        residual_actor_lifetime_owned_by_separate_lifecycle_policy=True,
        complete_source_iterator_finally_releases_owned_pool=True,
        failure_cleanup_and_other_release_calls_immediate=True,
        original_derivative_jobs_rows_order_and_normal_accumulation_unchanged=True,
        actual_complete_cycle_speedup_pending=True)
    return _RECEIPT
