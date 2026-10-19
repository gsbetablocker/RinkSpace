"""Persistent geometry workers with explicit per-task shared scene context.

The original physical producer, finite-difference step and domain failures are
retained. Scene context changes invalidate worker scene caches through the
model's own setter; they do not require rebuilding the physical-state graph.
"""
from concurrent.futures import ProcessPoolExecutor
import hashlib
import multiprocessing
import os
from pathlib import Path
from types import MethodType

import numpy as np


def _reference(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def _init_context_worker(binding, descriptors, model_ref, producers, execution_ref):
    from native_transverse_bundle import load_model, reference, configure_native_runtime
    from calibration.program_tracking import first_shot_fit as fit
    for ref in [*producers, execution_ref]:
        if reference(ref['path']) != ref:
            raise ValueError('Frozen geometry execution producer changed: ' + ref['path'])
    model = load_model(model_ref)
    configure_native_runtime(stage='persistent-transverse-context-world-FD', native_threads=1)
    import cv2
    cv2.setNumThreads(1)
    state = model.init_geometry_state(binding, np.zeros(5))
    original_guard = state.guard_geometry

    def guard(q):
        base = state.initial_q.copy()
        value = np.asarray(q, float)
        if value.shape == base.shape:
            base = value
        else:
            base[state.world_parameter_indices] = value
        # Read the context attached to this very sample, never an initializer's
        # captured value. The original setter clears all affected scene caches.
        delta = state.actual_transverse_geometry._transverse_delta
        try:
            model.describe(base, delta)
        except ValueError as error:
            from calibration.program_tracking.first_shot_physical_state import GeometryDomainError
            raise GeometryDomainError(str(error)) from error
        return original_guard(q)

    state.guard_geometry = guard
    fit._geometry_worker = state, descriptors


def _context_sample(item):
    from calibration.program_tracking import first_shot_fit as fit
    task, delta = item
    state, _ = fit._geometry_worker
    state.actual_transverse_geometry.set_deviations(delta)
    result = fit._geometry_sample(task)
    return result, dict(pid=os.getpid(), context_sha256=hashlib.sha256(delta.tobytes()).hexdigest())


class _ContextPool:
    def __init__(self, adapter, delta):
        self.context = delta.copy()
        self.active = False
        self.receipts = {}
        self.pool = ProcessPoolExecutor(adapter.bundle.geometry_workers,
            mp_context=multiprocessing.get_context('spawn'),
            initializer=_init_context_worker,
            initargs=(adapter.bundle.state.binding_reference['path'],
                adapter.bundle.target_descriptors, adapter.model_reference,
                adapter.source_producers, adapter.geometry_execution_receipt['producer']))

    def set_context(self, delta):
        if self.active:
            raise RuntimeError('Geometry context cannot change during an active sample map')
        self.context = delta.copy()

    def map(self, function, tasks, *, chunksize=1):
        from calibration.program_tracking import first_shot_fit as fit
        if function is not fit._geometry_sample:
            raise ValueError('Only the unchanged original geometry sampler is supported')
        delta = self.context.copy()
        expected = hashlib.sha256(delta.tobytes()).hexdigest()
        if self.active:
            raise RuntimeError('Concurrent geometry sample maps are unsupported')
        self.active = True
        try:
            for result, receipt in self.pool.map(_context_sample,
                    ((task, delta) for task in tasks), chunksize=chunksize):
                if receipt['context_sha256'] != expected:
                    raise ValueError('Worker evaluated a different shared geometry context')
                self.receipts[(receipt['pid'], expected)] = receipt
                yield result
        finally:
            self.active = False

    def shutdown(self, wait=True):
        return self.pool.shutdown(wait=wait)


def _ensure_context_pool(adapter, delta):
    if adapter.bundle.geometry_workers <= 1:
        return
    delta = np.asarray(delta, dtype=np.float64)
    if delta.shape != (5,) or not np.isfinite(delta).all():
        raise ValueError('Exactly five finite shared geometry deviations required')
    if _reference(__file__) != adapter.geometry_execution_receipt['producer']:
        raise ValueError('Frozen geometry execution helper changed')
    pool = adapter.bundle._geometry_pool
    if pool is None:
        adapter.bundle._geometry_pool = _ContextPool(adapter, delta)
    elif isinstance(pool, _ContextPool):
        pool.set_context(delta)
    else:
        raise ValueError('Install context execution before starting another geometry pool')
    adapter._geometry_pool_delta = delta.copy()


def install_geometry_context(adapter):
    """Attach before the adapter's first parallel geometry evaluation."""
    if adapter.bundle._geometry_pool is not None:
        raise ValueError('Existing geometry pool must be closed before changing execution')
    if not hasattr(adapter.state, 'deviations'):
        raise ValueError('A shared transverse-geometry adapter is required')
    adapter.geometry_execution_receipt = dict(
        producer=_reference(__file__),
        geometry_workers=int(adapter.bundle.geometry_workers),
        source_workers=int(adapter.workers),
        context='explicit per original finite-difference sample',
        same_original_sampler_step_world_and_domain_guard=True,
        worker_scene_cache_invalidation='actual_transverse_geometry.set_deviations',
        pool_shutdown='adapter.close only',
        concurrent_context_change_rejected=True)
    adapter._ensure_geometry_pool = MethodType(_ensure_context_pool, adapter)
    return adapter.geometry_execution_receipt
