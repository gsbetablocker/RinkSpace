"""Precompute complete adaptive reference-curve Jacobians outside the GTSAM callback.

Each worker evaluates the original curve method, including its full physical
central difference and union topology.  Only independent whole curves run in
parallel.  Graph state expansion and the factor's robust/rig chain stay in the
main process; worker failures are rerun there to preserve typed exceptions.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from queue import Empty
from dataclasses import dataclass
from importlib import import_module
import multiprocessing
import os
from threading import Lock
import time
from typing import Callable

import numpy as np


_ENV_LOCK = Lock()
_THREAD_VARIABLES = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
_STARTUP_TIMEOUT_SECONDS = 45.0
_PROBE_RELEASE = None


def _local_module(name: str):
    return import_module(f".{name}", package=__package__) if __package__ else import_module(name)


@dataclass(frozen=True)
class AdaptiveCurveJob:
    curve: object
    template25: np.ndarray
    model: object
    pts: int
    source_keys: tuple[int, ...]
    prime: Callable


def _backend_flags(curve_class=None) -> tuple[bool, bool] | None:
    """Only known original or reviewed opt-in paths may enter a worker."""
    curve_module = _local_module("secondary_nearboard_adaptive_curve")
    projection = _local_module("projection")
    central = _local_module("continuous_central_brown4_optin")
    implicit = _local_module("continuous_adaptive_curve_implicit_optin")
    from physical_projection import safe_project

    if (curve_class is not None and curve_class is not curve_module.AdaptiveProjectedPolyline
            or curve_module.project_central_valid is not projection.project_central_valid):
        return None
    method = curve_module.AdaptiveProjectedPolyline.value_and_physical_jacobian
    if implicit._ORIGINAL is None:
        if (method.__module__ != curve_module.__name__
                or method.__qualname__ != "AdaptiveProjectedPolyline.value_and_physical_jacobian"):
            return None
    elif (implicit.AdaptiveProjectedPolyline is not curve_module.AdaptiveProjectedPolyline
          or method.__module__ != implicit.__name__
          or method.__qualname__ != "install_adaptive_implicit_optin.<locals>.guarded"):
        return None
    backend = projection._central_projection_backend
    if backend is safe_project:
        central_enabled = False
    elif backend is central.safe_project_compatible:
        central_enabled = True
    else:
        return None
    implicit_enabled = (implicit._ORIGINAL is not None
                        and implicit.AdaptiveProjectedPolyline is curve_module.AdaptiveProjectedPolyline)
    return central_enabled, implicit_enabled


def _worker_init(central_enabled: bool, implicit_enabled: bool,
                 ready_queue, probe_release):
    global _PROBE_RELEASE
    for name in _THREAD_VARIABLES:
        os.environ[name] = "1"
    from runtime.bootstrap import add_geometry_dependencies
    add_geometry_dependencies()
    import cv2
    cv2.setNumThreads(1)
    _local_module("secondary_nearboard_adaptive_curve")
    if central_enabled:
        _local_module("continuous_central_brown4_optin").enable_projection_backend()
    if implicit_enabled:
        _local_module("continuous_adaptive_curve_implicit_optin").install_adaptive_implicit_optin()
    _PROBE_RELEASE = probe_release
    actual = _backend_flags()
    ready_queue.put((os.getpid(), actual))
    if actual != (central_enabled, implicit_enabled):
        raise RuntimeError(
            f"Adaptive curve worker backend differs: {actual} != "
            f"{(central_enabled, implicit_enabled)}")


def _worker_ready():
    if _PROBE_RELEASE is None or not _PROBE_RELEASE.wait(_STARTUP_TIMEOUT_SECONDS):
        raise TimeoutError("Adaptive curve startup probe was not released")
    return os.getpid()


def _stop_failed_pool(pool: ProcessPoolExecutor) -> None:
    """Bound cleanup for a broken or stalled startup."""
    # ProcessPoolExecutor has no public timed shutdown or worker handles. Read
    # the handles before shutdown discards them; never modify executor state.
    processes = tuple((getattr(pool, "_processes", None) or {}).copy().values())
    manager_thread = getattr(pool, "_executor_manager_thread", None)
    # A broken executor usually drains its own call queue promptly. Let that
    # finish before shutdown so its feeder never writes to a closed handle.
    if manager_thread is not None:
        manager_thread.join(timeout=1.0)
        if not manager_thread.is_alive():
            pool.shutdown(wait=True, cancel_futures=True)
            return
    pool.shutdown(wait=False, cancel_futures=True)
    for process in processes:
        if process.pid is None:
            continue
        try:
            if process.is_alive():
                process.terminate()
        except (AssertionError, OSError, ValueError):
            pass
    deadline = time.monotonic() + 2.0
    for process in processes:
        if process.pid is None:
            continue
        try:
            process.join(timeout=max(0.0, deadline - time.monotonic()))
            if process.is_alive():
                process.kill()
        except (AssertionError, OSError, ValueError):
            pass
    deadline = time.monotonic() + 2.0
    for process in processes:
        if process.pid is not None:
            try:
                process.join(timeout=max(0.0, deadline - time.monotonic()))
            except (AssertionError, OSError, ValueError):
                pass
    if manager_thread is not None:
        manager_thread.join(timeout=2.0)


def _worker_curve(task):
    index, curve, physical, template25, source_keys = task
    implicit = _local_module("continuous_adaptive_curve_implicit_optin")
    before = dict(implicit._COUNTS)
    try:
        bundle = curve.value_and_physical_jacobian(
            physical, template25, source_keys=source_keys)
    except Exception as exc:
        # Some native/OpenCV exceptions cannot be pickled. The owner process
        # re-evaluates the original factor and raises its original typed error.
        return index, False, None, type(exc).__name__ + ": " + str(exc), {}
    delta = {key: value - before.get(key, 0)
             for key, value in implicit._COUNTS.items()
             if value > before.get(key, 0)}
    return index, True, bundle, None, delta


def _original_bundle(job: AdaptiveCurveJob, physical: np.ndarray):
    curve_module = _local_module("secondary_nearboard_adaptive_curve")
    try:
        return job.curve.value_and_physical_jacobian(
            physical, job.template25, source_keys=job.source_keys)
    except (curve_module.BoardIntersectionError,
            curve_module.AdaptiveCurveResolutionError) as exc:
        exc.evaluation_stage = "jacobian"
        exc.factor_family = job.curve.factor_family
        exc.source_keys = job.source_keys
        raise


class _CurvePrecomputer:
    def __init__(self, jobs: list[AdaptiveCurveJob], workers: int):
        self.jobs = jobs
        self.requested_workers = workers
        self.workers = workers
        self.cached: dict[int, tuple[bytes, tuple]] = {}
        self.pool: ProcessPoolExecutor | None = None
        self.flags = _backend_flags(type(jobs[0].curve))
        self.startup_error: str | None = (
            "unsupported adaptive curve backend; original main-thread evaluation"
            if self.flags is None and workers > 1 else None)
        self.worker_fallbacks = 0
        self.last_worker_error: str | None = None
        self.preparations = 0
        self.worker_implicit_factors = 0
        self.worker_fd_factors = 0
        self.worker_pids: tuple[int, ...] = ()
        self.startup_worker_pids: list[int] = []
        self.readiness_task_pids: list[int] = []

    def _start_pool(self):
        if self.pool is not None or self.workers <= 1 or self.flags is None:
            return
        with _ENV_LOCK:
            previous = {name: os.environ.get(name) for name in _THREAD_VARIABLES}
            for name in _THREAD_VARIABLES:
                os.environ[name] = "1"
            ready_queue = None
            probe_release = None
            try:
                context = multiprocessing.get_context("spawn")
                ready_queue = context.Queue()
                probe_release = context.Event()
                pool = ProcessPoolExecutor(
                    max_workers=self.workers,
                    mp_context=context,
                    initializer=_worker_init,
                    initargs=(*self.flags, ready_queue, probe_release))
                try:
                    deadline = time.monotonic() + _STARTUP_TIMEOUT_SECONDS
                    probes = []
                    pids: set[int] = set()
                    # Each submission may spawn one process. Hold its probe
                    # until every initializer reports, so an idle first worker
                    # cannot prevent later submissions from spawning workers.
                    # Stage submissions to avoid filling the call queue when
                    # the first initializer fails immediately.
                    for count in range(1, self.workers + 1):
                        probes.append(pool.submit(_worker_ready))
                        while len(pids) < count:
                            # The manager may not mark a freshly spawned
                            # failed initializer broken until another result
                            # arrives. Detect its exit while the first probe
                            # is intentionally held at the startup barrier.
                            for probe in probes:
                                if probe.done():
                                    error = probe.exception()
                                    if error is not None:
                                        raise error
                            for process in tuple((getattr(pool, "_processes", None)
                                                  or {}).copy().values()):
                                if process.exitcode is not None:
                                    raise RuntimeError(
                                        f"Adaptive curve worker {process.pid} "
                                        f"exited during startup ({process.exitcode})")
                            remaining = deadline - time.monotonic()
                            if remaining <= 0:
                                raise TimeoutError(
                                    f"Only {len(pids)}/{self.workers} adaptive "
                                    "curve workers initialized")
                            try:
                                pid, actual = ready_queue.get(timeout=min(.1, remaining))
                            except Empty:
                                continue
                            self.startup_worker_pids.append(pid)
                            if actual != self.flags:
                                raise RuntimeError(
                                    f"Adaptive curve worker {pid} backend differs: "
                                    f"{actual} != {self.flags}")
                            if pid in pids:
                                raise RuntimeError(
                                    f"Adaptive curve worker {pid} reported readiness twice")
                            pids.add(pid)
                    probe_release.set()
                    self.readiness_task_pids = [
                        probe.result(timeout=max(0., deadline - time.monotonic()))
                        for probe in probes]
                    if not set(self.readiness_task_pids).issubset(pids):
                        raise RuntimeError("Adaptive curve readiness task used an unknown worker")
                    self.worker_pids = tuple(sorted(pids))
                except BaseException:
                    probe_release.set()
                    _stop_failed_pool(pool)
                    raise
                self.pool = pool
            except Exception as exc:
                # A process-pool failure cannot remove an observation. The
                # complete original calculation runs in the main process.
                self.startup_error = repr(exc)
                self.workers = 1
                self.worker_pids = ()
            finally:
                if ready_queue is not None:
                    ready_queue.close()
                    ready_queue.join_thread()
                for name, value in previous.items():
                    if value is None:
                        os.environ.pop(name, None)
                    else:
                        os.environ[name] = value

    def prepare(self, states) -> dict:
        started = time.perf_counter()
        self.preparations += 1
        states_by_frame = {}
        pending = []
        bundles = {}
        for index, job in enumerate(self.jobs):
            key = (id(job.model), int(job.pts))
            if key not in states_by_frame:
                states_by_frame[key] = np.asarray(
                    job.model.physical_state(job.pts, states), np.float64)
            physical = states_by_frame[key]
            token = physical.tobytes()
            cached = self.cached.get(index)
            if cached is not None and cached[0] == token:
                bundles[index] = cached[1]
            else:
                pending.append((index, physical.copy(), token))

        submitted = 0
        if pending:
            self._start_pool()
            futures = {}
            if self.pool is not None:
                try:
                    for index, physical, _ in pending:
                        job = self.jobs[index]
                        futures[index] = self.pool.submit(
                            _worker_curve,
                            (index, job.curve, physical, job.template25,
                             job.source_keys))
                    submitted = len(futures)
                except Exception as exc:
                    self.startup_error = repr(exc)
                    self.close()
                    self.workers = 1
            for index, physical, token in pending:
                job = self.jobs[index]
                future = futures.get(index)
                if future is None:
                    bundle = _original_bundle(job, physical)
                else:
                    try:
                        returned_index, success, bundle, worker_error, count_delta = future.result()
                        if returned_index != index:
                            raise RuntimeError("Adaptive curve worker returned another factor")
                        if not success:
                            raise RuntimeError(worker_error)
                        implicit = _local_module("continuous_adaptive_curve_implicit_optin")
                        implicit._COUNTS.update(count_delta)
                        if "jacobian_method" in bundle[2]:
                            self.worker_implicit_factors += 1
                        else:
                            self.worker_fd_factors += 1
                    except Exception as exc:
                        self.worker_fallbacks += 1
                        self.last_worker_error = repr(exc)
                        bundle = _original_bundle(job, physical)
                bundles[index] = bundle
                self.cached[index] = (token, bundle)

        # Prime only after all requested factors completed. This never holds a
        # partial new accepted state if one factor has a physical-domain error.
        for index, job in enumerate(self.jobs):
            key = (id(job.model), int(job.pts))
            job.prime(states_by_frame[key], bundles[index])
        return dict(factors=len(self.jobs), computed=len(pending),
                    submitted=submitted,
                    workers=(len(self.worker_pids) if self.pool else 1),
                    requested_workers=self.requested_workers,
                    worker_pids=list(self.worker_pids),
                    startup_worker_pids=list(self.startup_worker_pids),
                    readiness_task_pids=list(self.readiness_task_pids),
                    startup_error=self.startup_error,
                    fallback_count=self.worker_fallbacks,
                    last_worker_error=self.last_worker_error,
                    worker_implicit_factors=self.worker_implicit_factors,
                    worker_fd_factors=self.worker_fd_factors,
                    seconds=time.perf_counter() - started)

    def close(self):
        if self.pool is not None:
            self.pool.shutdown(wait=True, cancel_futures=True)
            self.pool = None
            self.worker_pids = ()
        self.cached.clear()


def prepare_adaptive_curve_jobs(graph, states) -> dict:
    """Prime accepted-state reference callbacks before a GTSAM linearization."""
    jobs = getattr(graph, "_adaptive_curve_jobs", ())
    if not jobs:
        return dict(factors=0, computed=0, submitted=0, workers=0, seconds=0.)
    workers = int(getattr(graph, "_adaptive_curve_worker_count", 8))
    if workers < 0:
        raise ValueError("Adaptive curve worker count must be nonnegative")
    if workers == 0:
        close_adaptive_curve_workers(graph)
        return dict(factors=len(jobs), computed=0, submitted=0, workers=0,
                    seconds=0.)
    manager = getattr(graph, "_adaptive_curve_manager", None)
    if (manager is None or manager.jobs is not jobs
            or manager.requested_workers != workers):
        close_adaptive_curve_workers(graph)
        manager = _CurvePrecomputer(jobs, workers)
        graph._adaptive_curve_manager = manager
    report = manager.prepare(states)
    graph._adaptive_curve_parallel_stats = report
    return report


def close_adaptive_curve_workers(graph) -> None:
    """Release this graph's worker processes and accepted-state bundles."""
    manager = getattr(graph, "_adaptive_curve_manager", None)
    if manager is not None:
        manager.close()
        graph._adaptive_curve_manager = None
