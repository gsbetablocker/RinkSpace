"""Small original-context routing contract; no process pool or source graph."""
from types import SimpleNamespace
from concurrent.futures import Future
import hashlib
import pickle

import numpy as np

from calibration.solver_runtime import install
install()
from calibration.solver_runtime import pattern_fd_pool_reuse_runtime as runtime
from calibration.solver_runtime.absolute_geometry_fd_pool_candidate import AbsoluteGeometryGroupPool
from calibration.solver_runtime.worker_entry import evaluate_pattern_worker
from structured_programme_source_candidate import current_geometry_pattern_pool as source


class _Geometry:
    def __init__(self):
        for name in ('q_course', 'course_xy_m', 'course_station_m', 'course_tangents',
                     'station_domain_m', 'z_domain_m', 'x_m', 'y_domain_m', 'scene_unit_m',
                     'course_source_boundary_indices', '_course_xy_jacobian',
                     '_course_edges', '_course_edges_j', '_course_lengths', '_course_prefix_j'):
            setattr(self, name, np.arange(2, dtype=np.float64))

    def guard(self, q):
        assert np.asarray(q).shape == (9,)


class _Factor:
    def __init__(self, geometry):
        self.geometry = geometry


class _RecordingFuture(Future):
    def __init__(self, ordinal, calls):
        super().__init__()
        self.ordinal, self.calls = ordinal, calls

    def result(self, *args, **kwargs):
        self.calls.append(self.ordinal)
        return super().result(*args, **kwargs)


class _Executor:
    def __init__(self, *args, **kwargs):
        self._pending_work_items = {}
        self.calls, self.targets = [], []
        self.error = None
        self.closed = False

    def submit(self, function, payload, payload_sha, geometry_sha):
        factor, q9, camera, frame, jacobian = pickle.loads(payload)
        assert hashlib.sha256(payload).hexdigest() == payload_sha
        assert source.geometry_signature(factor) == geometry_sha
        assert q9.tobytes() == np.arange(9, dtype=float).tobytes()
        assert camera.tobytes() == np.arange(25, dtype=float).tobytes()
        assert not jacobian
        ordinal = frame['source_actual_pts']
        future = _RecordingFuture(ordinal, self.calls)
        self.targets.append(function)
        if self.error is not None and ordinal == 1:
            future.set_exception(self.error)
        else:
            future.set_result((SimpleNamespace(residual=np.array([ordinal], dtype=float),
                                               source_reports=['original']), dict(pid=17)))
        return future

    def shutdown(self, *args, **kwargs):
        self.closed = True


def _manager(pool):
    geometry = _Geometry()
    frames = [dict(source_actual_pts=i, source_native_pts=i+7, split='FIT',
                   physical_ad_id=13, observer_state_owner='source') for i in range(3)]
    owner = SimpleNamespace(start=0, geometry=geometry, fit=frames, held=frames,
        state=SimpleNamespace(blocks=[SimpleNamespace(_absolute_geometry_fd_pool=pool)]),
        _evaluate_current=lambda *args: None,
        camera=lambda q, frame, held: (np.arange(25, dtype=float), None, None, None),
        frame_factor=lambda frame: _Factor(geometry))
    return source.CurrentGeometryPatternPool(owner)


def test_original_context_borrow_fallback_and_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(source, 'ProcessPoolExecutor', _Executor)
    runtime.install_pattern_fd_pool_reuse_runtime(report_directory=tmp_path)
    executor = _Executor()
    pool = AbsoluteGeometryGroupPool({}, workers=8)
    pool.executor = executor
    manager = _manager(pool)
    q = np.arange(9, dtype=float)
    result = manager.evaluate_current(q, False, False)
    assert result.tobytes() == np.arange(3, dtype=float).tobytes()
    assert executor.calls == [0, 1, 2, 0, 1, 2]
    assert executor.targets == [runtime.evaluate_pattern_on_existing_absolute_worker]*3
    assert manager.pool is None and pool.executor is executor
    assert not any(name in vars(pool) for name in runtime._BORROW_ATTRIBUTES)
    manager.close()
    assert not executor.closed

    executor = _Executor()
    original_error = ValueError('original middle-frame error')
    executor.error = original_error
    pool.executor = executor
    manager = _manager(pool)
    # A reporting failure cannot replace the original frame exception.
    original_dumps = runtime.json.dumps
    monkeypatch.setattr(runtime.json, 'dumps', lambda *args, **kwargs: (_ for _ in ()).throw(OSError('evidence IO')))
    try:
        manager.evaluate_current(q, False, False)
    except ValueError as error:
        assert error is original_error
    else:
        raise AssertionError('Original error was swallowed')
    monkeypatch.setattr(runtime.json, 'dumps', original_dumps)
    assert executor.calls == [0, 1, 0, 1, 2]
    assert not any(name in vars(pool) for name in runtime._BORROW_ATTRIBUTES)
    assert not executor.closed
    assert runtime._REPORT_ERRORS[-1]['type'] == 'OSError'

    # A foreign pending job or active original iterator must retain the old
    # pattern executor route; no shared task is inspected or awaited.
    for busy in ('foreign_pending', 'active_iterator', 'no_fd_executor'):
        pool.executor = _Executor()
        if busy == 'foreign_pending':
            pool.executor._pending_work_items[77] = object()
        elif busy == 'active_iterator':
            pool._rinkspace_active_geometry_group_iterators = 1
        else:
            pool.executor = None
        manager = _manager(pool)
        assert np.array_equal(manager.evaluate_current(q, False, False), np.arange(3))
        assert manager.pool is not None
        assert manager.pool.targets == [evaluate_pattern_worker]*3
        manager.close()
        vars(pool).pop('_rinkspace_active_geometry_group_iterators', None)


def test_late_callback_does_not_recreate_pool_metadata():
    class _LateFuture(_RecordingFuture):
        def add_done_callback(self, callback):
            self.callback = callback

    class _LateExecutor:
        def submit(self, function, *args, **kwargs):
            self.future = _LateFuture(0, [])
            self.future.set_result('ok')
            return self.future

    executor = _LateExecutor()
    pool = SimpleNamespace(executor=executor, closed=False)
    original_worker = object()
    lease = runtime._PatternExecutorLease(object(), pool, original_worker)
    lease.submit(original_worker, b'payload', 'sha', 'geometry')
    assert not lease.drain()
    executor.future.callback(executor.future)
    assert not any(name in vars(pool) for name in runtime._BORROW_ATTRIBUTES)
