"""Run original typed pattern jobs on an already idle owned geometry pool.

No FD pool is created for this route.  The original pattern context prepares
and validates every serialized factor, q9, camera and frame, and consumes the
original frame order.  Only its executor is borrowed for one context; private
pattern jobs are drained before returning that executor to its original owner.
"""
from __future__ import annotations

from contextvars import ContextVar
import functools
import hashlib
import inspect
import json
from pathlib import Path
import threading
import time

_SCOPE = ContextVar('rinkspace_pattern_existing_fd_executor', default=None)
_RECEIPT = None
_PATTERN_WORKER_READY = False
_REPORT_ERRORS = []

_BUSY_ATTRIBUTES = (
    '_rinkspace_active_geometry_group_iterators',
    '_rinkspace_confirmed_columns_pending_jobs',
    '_rinkspace_native_columns_pending_jobs',
    '_rinkspace_shared_material_bank_pending_jobs',
    '_rinkspace_shared_material_bank_precompute_active',
    '_rinkspace_pattern_borrow_active',
    '_rinkspace_pattern_borrow_pending_jobs',
)
_BORROW_ATTRIBUTES = (
    '_rinkspace_pattern_borrow_active',
    '_rinkspace_pattern_borrow_pending_jobs',
    '_rinkspace_pattern_borrow_executor_identity',
)
_MISSING = object()


def evaluate_pattern_on_existing_absolute_worker(*args):
    """Spawn-importable target retaining the original pattern bootstrap/math."""
    global _PATTERN_WORKER_READY
    from calibration.solver_runtime.worker_entry import (
        initialize_pattern_worker, evaluate_pattern_worker)
    if not _PATTERN_WORKER_READY:
        from structured_programme_source_candidate.current_geometry_pattern_pool import BASE
        initialize_pattern_worker(str(BASE))
        _PATTERN_WORKER_READY = True
    return evaluate_pattern_worker(*args)


def _idle_pool(manager, pool_class):
    # An already existing private pattern pool remains under its own original
    # lifecycle.  In particular this helper never leaves two pools resident.
    if manager.closed or not manager.enabled or manager.pool is not None:
        return None, 'pattern_manager_closed_disabled_or_owns_pool'
    state = getattr(manager.owner, 'state', None)
    blocks = getattr(state, 'blocks', None)
    if not blocks:
        return None, 'no_original_programme_blocks'
    pools = [getattr(block, '_absolute_geometry_fd_pool', None) for block in blocks]
    pool = pools[0]
    if not isinstance(pool, pool_class) or any(value is not pool for value in pools):
        return None, 'programme_blocks_do_not_share_one_original_fd_pool'
    if pool.closed or not pool.owns_executor or pool.executor is None:
        return None, 'no_existing_owned_fd_executor'
    if any(getattr(pool, name, 0) for name in _BUSY_ATTRIBUTES):
        return None, 'original_fd_pool_has_active_context_or_jobs'
    # CPython's original ProcessPoolExecutor removes work items before waking
    # Future.result().  Unknown executor implementations are not borrowed.
    pending = getattr(pool.executor, '_pending_work_items', None)
    if not isinstance(pending, dict) or pending:
        return None, 'executor_pending_work_not_proven_empty'
    completion = getattr(pool, 'last_completion_scheduling', None)
    if isinstance(completion, dict) and (
        completion.get('active_futures_at_exit', 0)
        or completion.get('buffered_batches_at_exit', 0)
    ):
        return None, 'last_original_group_iterator_not_drained'
    return pool, None


class _PatternExecutorLease:
    """Forward only this context's submit, tracking only its own futures."""
    def __init__(self, manager, pool, original_worker):
        self.manager, self.pool = manager, pool
        self.executor = pool.executor
        self.original_worker = original_worker
        self.thread = threading.current_thread()
        self.futures = []
        self.pending = 0
        self.released = False
        self.lock = threading.RLock()
        self.saved = {name: getattr(pool, name, _MISSING) for name in _BORROW_ATTRIBUTES}
        pool._rinkspace_pattern_borrow_active = True
        pool._rinkspace_pattern_borrow_pending_jobs = 0
        pool._rinkspace_pattern_borrow_executor_identity = id(self.executor)

    def submit(self, function, *args, **kwargs):
        if threading.current_thread() is not self.thread or function is not self.original_worker:
            raise RuntimeError('Only the original context thread and pattern worker may use this lease')
        if self.pool.executor is not self.executor or self.pool.closed:
            raise RuntimeError('Original FD executor changed during a pattern context')
        with self.lock:
            self.pending += 1
            self.pool._rinkspace_pattern_borrow_pending_jobs = self.pending
        try:
            future = self.executor.submit(evaluate_pattern_on_existing_absolute_worker, *args, **kwargs)
        except BaseException:
            with self.lock:
                self.pending -= 1
                self.pool._rinkspace_pattern_borrow_pending_jobs = self.pending
            raise
        self.futures.append(future)

        def completed(_future):
            with self.lock:
                self.pending -= 1
                if not self.released:
                    self.pool._rinkspace_pattern_borrow_pending_jobs = self.pending
        future.add_done_callback(completed)
        return future

    def drain(self):
        # Do not cancel or await any pre-existing/foreign future.  Error order
        # and the numerical exception raised by the original consumer remain
        # primary; later submitted pattern jobs merely finish before reuse.
        errors = []
        for future in self.futures:
            try:
                future.result()
            except BaseException as error:
                errors.append(error)
        with self.lock:
            # Future.result may wake before its done callbacks finish.  After
            # every result is delivered there is no work left for this lease;
            # callbacks cannot subsequently write into restored pool metadata.
            self.released = True
            for name, value in self.saved.items():
                if value is _MISSING:
                    vars(self.pool).pop(name, None)
                else:
                    setattr(self.pool, name, value)
        return errors


def install_pattern_fd_pool_reuse_runtime(driver=None, *, report_directory=None):
    """Install before source evaluation; the registered factor code is intact."""
    global _RECEIPT
    if _RECEIPT is not None:
        return _RECEIPT
    from .absolute_geometry_fd_pool_candidate import AbsoluteGeometryGroupPool
    from calibration.solver_runtime.worker_entry import evaluate_pattern_worker
    from structured_programme_source_candidate.current_geometry_pattern_pool import CurrentGeometryPatternPool
    original_executor = CurrentGeometryPatternPool._executor
    original_evaluate = CurrentGeometryPatternPool.evaluate_current
    destination = None if report_directory is None else Path(report_directory)
    if destination is not None:
        destination.mkdir(parents=True, exist_ok=True)

    def record(value):
        try:
            value.update(stage='pattern_existing_fd_executor', wall_time_ns=time.time_ns())
            print(json.dumps(value), flush=True)
            if destination is not None:
                with (destination/'pattern_fd_pool_reuse.jsonl').open('a', encoding='utf8') as stream:
                    stream.write(json.dumps(value)+'\n')
        except Exception as error:
            # Optional evidence I/O never overrides the original math failure.
            _REPORT_ERRORS.append(dict(type=type(error).__name__, message=str(error)))

    @functools.wraps(original_executor)
    def executor(manager):
        lease = _SCOPE.get()
        if (lease is not None and lease.manager is manager
                and threading.current_thread() is lease.thread):
            return lease
        return original_executor(manager)

    @functools.wraps(original_evaluate)
    def evaluate(manager, q, jacobian, held):
        if _SCOPE.get() is not None:
            return original_evaluate(manager, q, jacobian, held)
        pool, reason = _idle_pool(manager, AbsoluteGeometryGroupPool)
        if pool is None:
            record(dict(action='original_pattern_executor', reason=reason,
                        jacobian=bool(jacobian), held=bool(held)))
            return original_evaluate(manager, q, jacobian, held)
        lease = _PatternExecutorLease(manager, pool, evaluate_pattern_worker)
        token = _SCOPE.set(lease)
        succeeded = False
        started = time.perf_counter()
        try:
            result = original_evaluate(manager, q, jacobian, held)
            succeeded = True
            return result
        finally:
            errors = lease.drain()
            _SCOPE.reset(token)
            record(dict(action='borrowed_existing_owned_fd_executor',
                        jacobian=bool(jacobian), held=bool(held),
                        pattern_requested_workers=manager.workers,
                        borrowed_executor_workers=pool.workers,
                        executor_identity=id(lease.executor),
                        submitted_pattern_jobs=len(lease.futures),
                        drained_pattern_jobs=len(lease.futures),
                        drained_error_count=len(errors), original_context_succeeded=succeeded,
                        pending_pattern_jobs=0, seconds=time.perf_counter()-started,
                        original_pattern_pool_assigned=False, shared_executor_shutdown=False))
            if succeeded and errors:
                raise errors[0]

    CurrentGeometryPatternPool._executor = executor
    CurrentGeometryPatternPool.evaluate_current = evaluate
    _RECEIPT = dict(schema='rinkspace.pattern-existing-owned-fd-executor.v1',
        helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        original_executor_sha256=hashlib.sha256(inspect.getsource(original_executor).encode()).hexdigest(),
        original_evaluate_sha256=hashlib.sha256(inspect.getsource(original_evaluate).encode()).hexdigest(),
        original_worker_sha256=hashlib.sha256(inspect.getsource(evaluate_pattern_worker).encode()).hexdigest(),
        existing_owned_idle_executor_only=True, launches_fd_executor=False,
        original_pattern_initializer_before_first_borrowed_job=True,
        original_frame_factor_q9_camera_source_frame_payload_and_geometry_sha_unchanged=True,
        original_factor_math_parent_J_and_frame_consumption_order_unchanged=True,
        tracks_and_drains_only_this_context_jobs=True,
        original_pattern_close_never_receives_shared_executor=True,
        borrowed_active_attribute='_rinkspace_pattern_borrow_active',
        borrowed_pending_attribute='_rinkspace_pattern_borrow_pending_jobs',
        borrowed_executor_identity_attribute='_rinkspace_pattern_borrow_executor_identity',
        actual_full_cycle_equivalence_and_speedup_pending=True)
    _RECEIPT['report_errors'] = _REPORT_ERRORS
    return _RECEIPT
