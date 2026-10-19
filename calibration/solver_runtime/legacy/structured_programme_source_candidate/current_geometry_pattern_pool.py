"""Per-frame original typed pattern factors on private current geometry.

Only the frame-factor call is scheduled in owned processes. The existing
course context order, camera providers, finite guards, global J assembly,
course FD widths, and finally restoration remain the original adapter's.
Pool processes persist; every job carries a fresh serialized factor binding.
No source graph, owner or mutable programme bundle is sent to the pool.
"""
from __future__ import annotations
from calibration.solver_runtime.config import data_root as _runtime_data_root, data_module_dir as _runtime_data_module_dir

import atexit
import copy
from concurrent.futures import ProcessPoolExecutor
import hashlib
import multiprocessing
import os
from pathlib import Path
import pickle
import time
from types import MethodType, SimpleNamespace

import cv2
import numpy as np
from scipy.sparse import csr_matrix, vstack

from z190_shared_board_pattern_source import fixed_pivot_jacobian
from z190_continuous_source_bundle import SCENE_UNIT_M

BASE=_runtime_data_root()
_WORKER_CONTRACT = None
_WORKER_FACTOR_CLASSES = set()


def _sha_array(value):
    value = np.asarray(value)
    return hashlib.sha256(str((value.dtype.str, value.shape)).encode() + value.tobytes()).hexdigest()


def _factor_geometry(factor):
    visited = set()
    while not hasattr(factor, 'geometry'):
        if id(factor) in visited or not hasattr(factor, 'factor'):
            raise TypeError('Original typed pattern factor has no private geometry binding')
        visited.add(id(factor)); factor = factor.factor
    return factor.geometry


def geometry_signature(factor):
    """Exact numeric identity of the finite currently bound course geometry."""
    geometry = _factor_geometry(factor)
    if not hasattr(geometry, 'q_course'):
        raise TypeError('Current course geometry required; old fixed geometry is not a pool job')
    digest = hashlib.sha256((type(geometry).__module__ + '.' + type(geometry).__qualname__).encode())
    for name in ('q_course', 'course_xy_m', 'course_station_m', 'course_tangents',
                 'station_domain_m', 'z_domain_m', 'x_m', 'y_domain_m', 'scene_unit_m',
                 'course_source_boundary_indices', '_course_xy_jacobian',
                 '_course_edges', '_course_edges_j', '_course_lengths', '_course_prefix_j'):
        if not hasattr(geometry, name):
            raise TypeError('Missing original finite geometry field: ' + name)
        digest.update(name.encode()); digest.update(_sha_array(getattr(geometry, name)).encode())
    if hasattr(geometry, 'independent_pair_station_z_offsets_m'):
        digest.update(b'independent_pair_station_z_offsets_m')
        digest.update(_sha_array(geometry.independent_pair_station_z_offsets_m).encode())
    return digest.hexdigest()


def _worker_initialize(task_root):
    import sys
    sys.dont_write_bytecode = True
    from isolated_solver_bootstrap import install
    install()
    from prepared_cache_helpers import install_mappingproxy_pickle, install_geometry_unpickle
    install_mappingproxy_pickle(); install_geometry_unpickle()
    cv2.setNumThreads(1)
    from .runtime_backend_contract import restore_native_backend_contract
    global _WORKER_CONTRACT
    _WORKER_CONTRACT = restore_native_backend_contract(
        stage='current_geometry_pattern_worker_initializer', native_threads=1)


def _worker_evaluate(payload, payload_sha256, expected_geometry_sha256):
    global _WORKER_CONTRACT
    if hashlib.sha256(payload).hexdigest() != payload_sha256:
        raise ValueError('Current pattern job payload changed during transport')
    unpickle_begin = time.perf_counter()
    factor, q9, camera, frame, jacobian = pickle.loads(payload)
    unpickle_seconds = time.perf_counter() - unpickle_begin
    geometry = _factor_geometry(factor)
    factor_classes = (type(factor).__module__, type(factor).__qualname__,
                      type(geometry).__module__, type(geometry).__qualname__)
    from .runtime_backend_contract import restore_native_backend_contract, REQUIRED_ENV
    if factor_classes not in _WORKER_FACTOR_CLASSES:
        _WORKER_CONTRACT = restore_native_backend_contract(
            stage='current_geometry_pattern_worker_after_typed_factor_unpickle', native_threads=1)
        _WORKER_FACTOR_CLASSES.add(factor_classes)
    import acceleration
    if any(os.environ.get(k) != v for k, v in REQUIRED_ENV.items()) or acceleration.MODE != 'auto':
        raise RuntimeError('Current pattern worker backend changed after its verified bootstrap')
    actual_geometry = geometry_signature(factor)
    if actual_geometry != expected_geometry_sha256:
        raise ValueError('Current pattern factor has stale/different finite geometry')
    begin = time.perf_counter(); cpu = time.process_time()
    result = factor.evaluate(q9, camera, frame, with_jacobian=jacobian)
    stats = dict(pid=os.getpid(), parent_pid=os.getppid(),
                 payload_sha256=payload_sha256, geometry_sha256=actual_geometry,
                 q9_sha256=_sha_array(q9), camera25_sha256=_sha_array(camera),
                 actual_pts=int(frame['source_actual_pts']), split=frame['split'],
                 physical_ad_id=int(frame['physical_ad_id']), jacobian=bool(jacobian),
                 unpickle_seconds=unpickle_seconds,
                 evaluate_wall_seconds=time.perf_counter()-begin,
                 evaluate_cpu_seconds=time.process_time()-cpu,
                 backend_environment=dict(_WORKER_CONTRACT['required_environment']),
                 acceleration_mode=acceleration.MODE,
                 library_sha256={k:v['library_sha256'] for k,v in _WORKER_CONTRACT['actual_native_libraries'].items()})
    return result, stats


def _evaluate_current_parallel(owner, q, jacobian, held):
    manager = owner._current_geometry_pattern_pool
    if not manager.enabled:
        return manager.original_evaluate_current(q, jacobian, held)
    return manager.evaluate_current(q, jacobian, held)


class CurrentGeometryPatternPool:
    def __init__(self, owner, *, workers=4, enabled=True):
        workers = int(workers)
        if not 1 <= workers <= 4:
            raise ValueError('Explicit current-pattern process count must be 1..4')
        self.owner = owner; self.workers = workers; self.enabled = bool(enabled)
        self.original_evaluate_current = owner._evaluate_current
        self.pool = None; self.closed = False; self.context_count = 0
        self.context_reports = []; self.worker_pids = set()
        atexit.register(self.close)

    def set_enabled(self, enabled):
        if self.closed and enabled:
            raise RuntimeError('Closed current pattern pool cannot restart')
        self.enabled = bool(enabled)

    def _executor(self):
        if self.closed:
            raise RuntimeError('Current pattern pool has closed')
        if self.pool is None:
            from calibration.solver_runtime.worker_entry import initialize_pattern_worker
            self.pool = ProcessPoolExecutor(max_workers=self.workers,
                mp_context=multiprocessing.get_context('spawn'),
                initializer=initialize_pattern_worker, initargs=(str(BASE),))
        return self.pool

    def evaluate_current(self, q, jacobian, held):
        owner = self.owner; q9 = q[owner.start:owner.start+9]
        owner.geometry.guard(q9)
        # All camera() calls stay on the original caller-owned route. A fresh
        # typed factor/geometry is serialized while this context is bound.
        pending = []; serialized_bytes = 0; pickle_seconds = 0.; begin = time.perf_counter()
        for frame in owner.held if held else owner.fit:
            cam, slots, view, lever = owner.camera(q, frame, held=held)
            factor = owner.frame_factor(frame)
            geometry_sha = geometry_signature(factor)
            tick = time.perf_counter()
            payload = pickle.dumps((factor, q9.copy(), np.asarray(cam).copy(),
                                    copy.deepcopy(frame), bool(jacobian)), protocol=5)
            pickle_seconds += time.perf_counter()-tick; serialized_bytes += len(payload)
            payload_sha = hashlib.sha256(payload).hexdigest()
            from calibration.solver_runtime.worker_entry import evaluate_pattern_worker
            future = self._executor().submit(evaluate_pattern_worker, payload, payload_sha, geometry_sha)
            pending.append((frame, cam, slots, view, lever, future, geometry_sha))
        values, matrices, reports, jobs = [], [], [], []
        # Consume original frame order, regardless of process completion order.
        for frame, cam, slots, view, lever, future, geometry_sha in pending:
            result, stats = future.result(); jobs.append(stats); self.worker_pids.add(stats['pid'])
            values.append(result.residual)
            reports.append(dict(physical_ad_id=frame['physical_ad_id'],
                source_actual_pts=frame['source_actual_pts'], source_native_pts=frame['source_native_pts'],
                split=frame['split'], observer_state_owner=frame['observer_state_owner'],
                source_reports=result.source_reports, source_pixels_roles_sigma_unchanged=True))
            if jacobian:
                jxyz, jview = fixed_pivot_jacobian(result.camera_state11_jacobian, cam, lever, SCENE_UNIT_M)
                matrix = np.zeros((len(result.residual), len(q)))
                matrix[:, slots], matrix[:, view:view+4] = jxyz, jview
                matrix[:, owner.start:owner.start+9] = result.shared_q9_jacobian
                chart = owner.chart_for_frame(frame)
                matrix[:, chart.indices] = result.optics_jacobian
                rotation = cv2.Rodrigues(np.asarray(cam[3:6]))[0]
                matrix[:, chart.mount_indices] = np.asarray(result.camera_state11_jacobian)[:,:3]@rotation.T/(100.*SCENE_UNIT_M)
                matrices.append(csr_matrix(matrix))
        residual = np.concatenate(values) if values else np.empty(0)
        self.context_count += 1
        self.context_reports.append(dict(context_ordinal=self.context_count, q_sha256=_sha_array(q),
            jacobian=bool(jacobian), held=bool(held), source_frames=len(pending),
            serialized_job_bytes=serialized_bytes, parent_pickle_seconds=pickle_seconds,
            context_wall_seconds=time.perf_counter()-begin, jobs=jobs))
        if len(self.context_reports) > 64:
            self.context_reports = self.context_reports[-64:]
        if held:
            return SimpleNamespace(residual=residual, source_reports=reports)
        return (residual, vstack(matrices, format='csr') if matrices else csr_matrix((0,len(q)))) if jacobian else residual

    def report(self):
        return dict(schema='rinkspace.current-geometry-private-typed-pattern-pool.v1',
            enabled=self.enabled, requested_workers=self.workers,
            actual_worker_pids=sorted(self.worker_pids), contexts_evaluated=self.context_count,
            original_context_order_preserved=True,
            factor_and_geometry_snapshot_transmitted_on_every_job=True,
            full_programme_owner_or_bundle_transmitted=False,
            q9_camera25_frame_source_support_and_original_factor_math_unchanged=True,
            current_geometry_digest_checked_in_worker=True,
            global_J_and_course_FD_assembled_by_original_parent_adapter=True,
            context_reports=list(self.context_reports),
            numerical_equivalence_or_speedup_claimed=False)

    def release_idle(self):
        """Finish this stage's pool; keep the enabled manager restartable."""
        if self.closed or self.pool is None:
            return False
        self.pool.shutdown(wait=True, cancel_futures=True)
        self.pool = None
        return True

    def close(self):
        if self.closed:return
        self.enabled = False; self.closed = True
        if self.pool is not None:
            self.pool.shutdown(wait=True, cancel_futures=True); self.pool = None


def install_current_geometry_pattern_pool(owner, *, workers=4, enabled=True):
    """Attach only to this CourseMountOpticsPatterns/CurrentPatterns instance.

    Main hook: call after CurrentProgrammeSource has installed CurrentPatterns,
    before its first evaluation/review. set_enabled(False) uses the original
    serial method; close() releases only processes created by this manager.
    """
    from shared_board_course_candidate.course_pattern_adapter import CourseMountOpticsPatterns
    if not isinstance(owner, CourseMountOpticsPatterns):
        raise TypeError('Actual current-course typed pattern owner required')
    existing = vars(owner).get('_current_geometry_pattern_pool')
    if existing is not None:
        if existing.workers != int(workers):
            raise ValueError('An existing pattern pool has a different requested process count')
        existing.set_enabled(enabled); return existing
    manager = CurrentGeometryPatternPool(owner, workers=workers, enabled=enabled)
    owner._current_geometry_pattern_pool = manager
    owner._evaluate_current = MethodType(_evaluate_current_parallel, owner)
    return manager
