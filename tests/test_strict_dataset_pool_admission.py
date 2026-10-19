from concurrent.futures import Future
from types import SimpleNamespace

import numpy as np
import pytest

from calibration.solver_runtime import strict_dataset_execution_runtime as runtime
from calibration.solver_runtime.absolute_geometry_fd_pool_candidate import AbsoluteGeometryGroupPool
from calibration.solver_runtime.native_factor_columns_runtime import _pool_available
from calibration.solver_runtime.strict_source_dataset_adapter import _key


class TrackingFuture(Future):
    def __init__(self, result, *, deferred=False):
        super().__init__()
        self.result_calls = 0
        self.outcome = result
        if deferred:
            self.set_running_or_notify_cancel()
        elif isinstance(result, BaseException):
            self.set_exception(result)
        else:
            self.set_result(result)

    def result(self, *args, **kwargs):
        self.result_calls += 1
        if not self.done():
            self.set_result(self.outcome)
        return super().result(*args, **kwargs)


class CheckedExecutor:
    def __init__(self, pool, factor, mode):
        self.pool, self.factor, self.mode = pool, factor, mode
        self.futures = []
        self._pending_work_items = {}
        self.shutdown_called = False

    def submit(self, function, job):
        assert self.pool._rinkspace_native_columns_active
        assert '_rinkspace_native_columns_idle_receipt' not in vars(self.pool)
        if self.mode == 'submit_failure' and self.futures:
            raise RuntimeError('intentional submit failure')
        nominal = [self.factor._produce_world(job['world'], target)[0] for target in job['targets']]
        values = [np.empty((*world.shape, len(job['positions']))) for world in nominal]
        for offset, column in enumerate(job['positions']):
            plus, minus = job['world'].copy(), job['world'].copy()
            plus[column] += 1e-6
            minus[column] -= 1e-6
            for ordinal, target in enumerate(job['targets']):
                values[ordinal][:, :, offset] = (
                    self.factor._produce_world(plus, target)[0]
                    - self.factor._produce_world(minus, target)[0]) / (2e-6)
        outcome = dict(positions=job['positions'], values=values, failures=[],
                       world_identity=job['world_identity'])
        if self.mode == 'worker_failure' and not self.futures:
            outcome = RuntimeError('intentional worker failure')
        if self.mode == 'cancelled_turn' and not self.futures:
            outcome = KeyboardInterrupt('intentional turn cancellation')
        if self.mode == 'identity_failure' and not self.futures:
            outcome['world_identity'] = None
        if self.mode == 'qualification_failure':
            for value in values:
                value += .01
        if self.mode == 'foreign_queued_after_submit':
            self._pending_work_items['foreign'] = object()
        future = TrackingFuture(outcome, deferred=self.mode == 'submit_failure')
        self.futures.append(future)
        return future

    def shutdown(self, *, wait, cancel_futures):
        assert wait and cancel_futures
        assert all(future.done() and future.result_calls for future in self.futures)
        self.shutdown_called = True


def configured_pool(monkeypatch, mode='success'):
    q = np.array([2., 3.])
    targets = [dict(kind='current_scene_ice_curve', curve_id=str(i)) for i in range(2)]
    def produce(value, target):
        offset = int(target['curve_id'])
        return np.array([[value[0], value[1], offset],
                         [value[0]+1, value[1]+2, offset]], float), None
    factor = SimpleNamespace(rows=[dict(model_target=t) for t in targets],
        world_indices=np.arange(2), _world_cache={}, _derivative_cache={},
        _produce_world=produce)
    pool = AbsoluteGeometryGroupPool(dict(path='authenticated-test-owner', sha256='0'*64),
                                     workers=2, max_inflight=2)
    executor = CheckedExecutor(pool, factor, mode)
    pool.executor = executor
    # A valid earlier private stage gives admission; strict jobs invalidate
    # this receipt during their own active/pending execution interval.
    pool._rinkspace_native_columns_idle_receipt = dict(
        executor_identity=id(executor), pending_jobs=0, current_q_sha256='earlier')
    factor.state = SimpleNamespace(blocks=[SimpleNamespace(_absolute_geometry_fd_pool=pool)])
    assert _pool_available(pool)
    return q, targets, factor, pool, executor


def test_success_consumes_all_jobs_and_reopens_existing_station_and_confirmed_admission(monkeypatch):
    q, targets, factor, pool, executor = configured_pool(monkeypatch)
    assert runtime._prepare_all_ice_derivatives(factor, q)
    assert all(f.done() and f.result_calls == 1 for f in executor.futures)
    assert pool.executor is executor and not executor.shutdown_called
    assert not pool._rinkspace_native_columns_active
    assert pool._rinkspace_native_columns_pending_jobs == 0
    assert pool._rinkspace_native_columns_idle_receipt['executor_identity'] == id(executor)
    assert _pool_available(pool)
    for target in targets:
        columns, derivative = factor._derivative_cache[_key(target)]
        assert columns.tolist() == [0, 1]
        plus, minus = q.copy(), q.copy()
        plus[1] += 1e-6
        minus[1] -= 1e-6
        expected = (factor._produce_world(plus, target)[0] - factor._produce_world(minus, target)[0])/(2e-6)
        assert np.array_equal(expected, derivative[:, :, 1])


@pytest.mark.parametrize('mode,error', [
    ('submit_failure', RuntimeError), ('worker_failure', RuntimeError),
    ('identity_failure', ValueError), ('qualification_failure', ValueError),
    ('cancelled_turn', KeyboardInterrupt),
    ('foreign_queued_after_submit', RuntimeError),
])
def test_every_failure_drains_or_cancels_all_futures_before_release_and_never_mints_idle(monkeypatch, mode, error):
    q, targets, factor, pool, executor = configured_pool(monkeypatch, mode)
    with pytest.raises(error):
        runtime._prepare_all_ice_derivatives(factor, q)
    assert all(f.done() and f.result_calls for f in executor.futures)
    assert executor.shutdown_called and pool.executor is None
    assert '_rinkspace_native_columns_idle_receipt' not in vars(pool)
    assert pool._rinkspace_native_columns_pending_jobs == 0
    assert not pool._rinkspace_native_columns_active
    assert factor._derivative_cache == {}


def test_existing_busy_or_unreceipted_executor_is_not_admitted(monkeypatch):
    q, targets, factor, pool, executor = configured_pool(monkeypatch)
    del pool._rinkspace_native_columns_idle_receipt
    assert runtime._existing_pool(factor) is None
    assert not runtime._prepare_all_ice_derivatives(factor, q)
    assert executor.futures == []
    pool._rinkspace_native_columns_pending_jobs = 1
    assert runtime._existing_pool(factor) is None


def test_same_executor_receipt_cannot_admit_foreign_queued_work(monkeypatch):
    q, targets, factor, pool, executor = configured_pool(monkeypatch)
    executor._pending_work_items['foreign'] = object()
    assert runtime._existing_pool(factor) is None
    assert not runtime._prepare_all_ice_derivatives(factor, q)
    assert executor.futures == []


def test_execution_installs_only_ice_pool_and_retains_original_camera_and_preflight():
    camera = lambda *args: None
    preflight = lambda *args: None
    factor = SimpleNamespace(_world=lambda *args: None, _camera=camera, preflight=preflight)
    receipt = runtime.install_strict_dataset_execution(factor)
    assert factor._camera is camera and factor.preflight is preflight
    assert receipt['candidate_camera_mapping_enabled'] is False
    assert 'original camera-coordinate FD retained' in receipt['candidate_camera_mapping_disabled_reason']
