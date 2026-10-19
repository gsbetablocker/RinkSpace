"""Check scheduling order/byte identity without a source graph or new process."""
from concurrent.futures import ThreadPoolExecutor
import threading

import numpy as np
import pytest

from calibration.solver_runtime import absolute_geometry_fd_pool_candidate as source
from calibration.solver_runtime.completion_driven_absolute_fd_runtime import (
    completion_driven_ordered_groups, _with_explicit_group_count,
    install_completion_driven_absolute_fd_runtime,
)


def groups(count=6):
    return [(100 + i, [dict(id=str(i), uv=[[i + .25, i - .5]])], np.arange(25, dtype=float) + i)
            for i in range(count)]


def result_for_job(job, *, route='unchanged_test_producer'):
    # An existing independent producer's output is deliberately nontrivial;
    # the scheduling helper may only carry these exact bytes and source IDs.
    row = job['rows'][0]
    values = np.asarray(row['uv'], float).ravel()[:, None]
    geo = (values + job['camera25'][0]) * np.asarray(job['world66'])[None, :]
    return dict(ordinal=job['ordinal'], actual_pts=job['actual_pts'],
                source_group_sha256=job['source_group_sha256'], geo=geo,
                stats=dict(actual_route=route, scalar_rows=len(geo)))


def pool_for(executor, *, groups_per_job=1):
    return source.AbsoluteGeometryGroupPool(dict(path='no_snapshot_is_loaded', sha256='unchanged'),
        executor=executor, workers=2, max_inflight=2, groups_per_job=groups_per_job, parent_geometry=object())


def test_refills_completed_slot_before_first_batch_returns_and_preserves_every_byte(monkeypatch):
    first_released = threading.Event()
    third_started_before_first_returned = threading.Event()
    def worker(batch):
        first = batch['groups'][0]['ordinal']
        if first == 0:
            assert first_released.wait(2), 'FIFO wait prevented the next independent task from starting'
        if first == 2:
            third_started_before_first_returned.set()
            first_released.set()
        common = {k: v for k, v in batch.items() if k != 'groups'}
        return [result_for_job(dict(common, **group)) for group in batch['groups']]
    monkeypatch.setattr(source, 'evaluate_absolute_geometry_worker', worker)
    world = np.array([.125, -2., 0.])
    original_inputs = groups()
    with ThreadPoolExecutor(max_workers=2) as executor:
        pool = pool_for(executor)
        actual = list(completion_driven_ordered_groups(pool, original_inputs, world, worker, 1e-7))
    assert third_started_before_first_returned.is_set()
    assert [result['ordinal'] for result in actual] == list(range(6))
    for i, (pts, rows, camera) in enumerate(original_inputs):
        expected = result_for_job(dict(ordinal=i, actual_pts=pts, rows=rows, camera25=camera,
            world66=world, source_group_sha256=source._source_group_digest(rows)))
        assert expected['source_group_sha256'] == actual[i]['source_group_sha256']
        assert expected['geo'].dtype == actual[i]['geo'].dtype
        assert expected['geo'].shape == actual[i]['geo'].shape
        assert expected['geo'].tobytes() == actual[i]['geo'].tobytes()
    assert pool.last_completion_scheduling['original_order_output_batches'] == 6
    assert pool.last_completion_scheduling['active_futures_at_exit'] == 0
    assert len(pool.last_stats) == 6


def test_later_job_error_cannot_replace_first_original_error(monkeypatch):
    first_released = threading.Event()
    def worker(batch):
        first = batch['groups'][0]['ordinal']
        if first == 0:
            assert first_released.wait(2)
            raise ValueError('first original error')
        if first == 1:
            raise RuntimeError('later completion error')
        first_released.set()
        common = {k: v for k, v in batch.items() if k != 'groups'}
        return [result_for_job(dict(common, **group)) for group in batch['groups']]
    monkeypatch.setattr(source, 'evaluate_absolute_geometry_worker', worker)
    with ThreadPoolExecutor(max_workers=2) as executor:
        pool = pool_for(executor)
        with pytest.raises(ValueError, match='first original error'):
            list(completion_driven_ordered_groups(pool, groups(), np.array([1., 2.]), worker, 1e-7))


def test_original_group_identity_still_checked(monkeypatch):
    def worker(batch):
        common = {k: v for k, v in batch.items() if k != 'groups'}
        result = [result_for_job(dict(common, **group)) for group in batch['groups']]
        result[0]['actual_pts'] += 1
        return result
    monkeypatch.setattr(source, 'evaluate_absolute_geometry_worker', worker)
    with ThreadPoolExecutor(max_workers=2) as executor:
        with pytest.raises(ValueError, match='Original absolute source group order/identity changed'):
            list(completion_driven_ordered_groups(pool_for(executor), groups(), np.array([1., 2.]), worker, 1e-7))


def test_existing_headroom_parent_fallback_does_not_start_workers(monkeypatch):
    seen = []
    def parent(job, geometry, certificates, route):
        seen.append(job['ordinal'])
        return result_for_job(job, route=route)
    def forbidden(batch):
        raise AssertionError('Original parent fallback must not submit a worker')
    monkeypatch.setattr(source, '_evaluate_group_on_owner', parent)
    monkeypatch.setattr(source, 'evaluate_absolute_geometry_worker', forbidden)
    with ThreadPoolExecutor(max_workers=2) as executor:
        pool = pool_for(executor, groups_per_job=2)
        pool._headroom = lambda phase: False
        actual = list(completion_driven_ordered_groups(pool, groups(), np.array([1., 2.]), forbidden, 1e-7))
    assert seen == list(range(6))
    assert all(result['stats']['actual_route'] == 'parent_low_headroom' for result in actual)


def test_actor_busy_activation_parent_execution_is_explicit_scoped_and_reversible(monkeypatch):
    pool = pool_for(None, groups_per_job=2)
    seen = []
    def parent(job, geometry, certificates, route):
        assert geometry is pool.parent_geometry
        assert certificates is pool.parent_certificates
        assert job['batched_projection'] is True
        assert job['step'] == 1e-7
        seen.append(job['ordinal'])
        return result_for_job(job, route=route)
    def worker(batch):
        common = {k: v for k, v in batch.items() if k != 'groups'}
        return [result_for_job(dict(common, **group)) for group in batch['groups']]
    def forbidden(phase):
        raise AssertionError('Explicit actor-busy activation must not fabricate a headroom check')
    monkeypatch.setattr(source, '_evaluate_group_on_owner', parent)
    monkeypatch.setattr(source, 'evaluate_absolute_geometry_worker', worker)
    pool._headroom = forbidden
    pool._rinkspace_force_parent_activation_FD = True
    world = np.array([.125, -2.])
    iterator = completion_driven_ordered_groups(pool, groups(), world, worker, 1e-7)
    first = next(iterator)
    assert pool._rinkspace_active_geometry_group_iterators == 1
    with pytest.raises(RuntimeError, match='requires an idle geometry FD manager'):
        next(completion_driven_ordered_groups(pool, groups(), world, worker, 1e-7))
    actual = [first, *iterator]
    assert pool._rinkspace_active_geometry_group_iterators == 0
    assert pool.executor is None
    assert seen == list(range(6))
    assert all(result['stats']['actual_route'] == 'parent_actor_busy_activation' for result in actual)
    assert pool.last_completion_scheduling['submitted_batches'] == 0
    assert pool.last_completion_scheduling['parent_output_batches'] == 3
    assert pool.last_completion_scheduling['force_parent_activation'] is True
    del pool._rinkspace_force_parent_activation_FD
    del pool._headroom
    with ThreadPoolExecutor(max_workers=2) as executor:
        pool.executor = executor
        pool.owns_executor = False
        parallel = list(completion_driven_ordered_groups(pool, groups(), world, worker, 1e-7))
    assert pool._rinkspace_active_geometry_group_iterators == 0
    assert pool.last_completion_scheduling['force_parent_activation'] is False
    assert pool.last_completion_scheduling['submitted_batches'] == 3
    for expected, observed in zip(actual, parallel):
        assert expected['geo'].tobytes() == observed['geo'].tobytes()


@pytest.mark.parametrize('count,effective,transport_jobs', [(39, 3, 15), (154, 10, 19), (512, 16, 32)])
def test_low_group_chunks_preserve_original_group_bytes_and_delivery_envelopes(monkeypatch, count, effective, transport_jobs):
    seen_chunks = []
    def worker(batch):
        seen_chunks.append([group['ordinal'] for group in batch['groups']])
        assert batch['step'] == 1e-7
        assert batch['batched_projection'] is True
        common = {k: v for k, v in batch.items() if k != 'groups'}
        return [result_for_job(dict(common, **group)) for group in batch['groups']]
    monkeypatch.setattr(source, 'evaluate_absolute_geometry_worker', worker)
    actual_groups = groups(count)
    world = np.array([.125, -2., 0.])
    with ThreadPoolExecutor(max_workers=8) as executor:
        reference_pool = source.AbsoluteGeometryGroupPool(dict(path='not_loaded', sha256='unchanged'),
            executor=executor, workers=8, max_inflight=8, groups_per_job=16, parent_geometry=object())
        expected = list(source.AbsoluteGeometryGroupPool.ordered_groups(reference_pool, actual_groups, world, worker, 1e-7))
        seen_chunks.clear()
        candidate_pool = source.AbsoluteGeometryGroupPool(dict(path='not_loaded', sha256='unchanged'),
            executor=executor, workers=8, max_inflight=8, groups_per_job=16, parent_geometry=object())
        actual = list(completion_driven_ordered_groups(candidate_pool, iter(actual_groups), world, worker, 1e-7, group_count=count))
    assert len(seen_chunks) == transport_jobs
    assert max(map(len, seen_chunks)) == effective
    assert sorted(i for chunk in seen_chunks for i in chunk) == list(range(count))
    assert all(chunk == list(range(chunk[0], chunk[-1] + 1)) for chunk in seen_chunks)
    for old, new in zip(expected, actual):
        assert old['ordinal'] == new['ordinal']
        assert old['actual_pts'] == new['actual_pts']
        assert old['source_group_sha256'] == new['source_group_sha256']
        assert old['geo'].dtype == new['geo'].dtype
        assert old['geo'].shape == new['geo'].shape
        assert old['geo'].tobytes() == new['geo'].tobytes()
    profile = candidate_pool.last_completion_scheduling
    assert profile['requested_groups_per_job'] == candidate_pool.groups_per_job == 16
    assert profile['effective_groups_per_job'] == effective
    assert profile['submitted_batches'] == transport_jobs
    assert profile['original_order_output_batches'] == (count + 15) // 16
    assert candidate_pool._rinkspace_active_geometry_group_iterators == 0


def test_dynamic_chunks_do_not_deliver_partial_original_batch_or_later_error_first(monkeypatch):
    for evaluate in ('original', 'candidate'):
        later_error_seen = threading.Event()
        def worker(batch):
            common = {k: v for k, v in batch.items() if k != 'groups'}
            result = []
            for group in batch['groups']:
                if group['ordinal'] == 4:
                    assert later_error_seen.wait(2)
                    raise ValueError('first original group error')
                if group['ordinal'] == 20:
                    later_error_seen.set()
                    raise RuntimeError('later independent group error')
                result.append(result_for_job(dict(common, **group)))
            return result
        monkeypatch.setattr(source, 'evaluate_absolute_geometry_worker', worker)
        with ThreadPoolExecutor(max_workers=8) as executor:
            pool = source.AbsoluteGeometryGroupPool(dict(path='not_loaded', sha256='unchanged'),
                executor=executor, workers=8, max_inflight=8, groups_per_job=16, parent_geometry=object())
            values = []
            iterator = (source.AbsoluteGeometryGroupPool.ordered_groups(pool, groups(39), np.array([1., 2.]), worker, 1e-7)
                if evaluate == 'original' else completion_driven_ordered_groups(pool, groups(39), np.array([1., 2.]), worker, 1e-7, group_count=39))
            with pytest.raises(ValueError, match='first original group error'):
                for value in iterator:
                    values.append(value)
            assert values == []
            assert pool.last_stats == []
            if evaluate == 'candidate':
                assert pool._rinkspace_active_geometry_group_iterators == 0


def test_explicit_group_count_does_not_split_actor_busy_parent_route(monkeypatch):
    def parent(job, geometry, certificates, route):
        assert route == 'parent_actor_busy_activation'
        return result_for_job(job, route=route)
    monkeypatch.setattr(source, '_evaluate_group_on_owner', parent)
    pool = source.AbsoluteGeometryGroupPool(dict(path='not_loaded', sha256='unchanged'),
        workers=8, max_inflight=8, groups_per_job=16, parent_geometry=object())
    pool._rinkspace_force_parent_activation_FD = True
    pool._headroom = lambda phase: pytest.fail('Explicit activation must not test headroom')
    pool._executor = lambda: pytest.fail('Explicit activation must not start workers')
    actual = list(completion_driven_ordered_groups(pool, groups(39), np.array([1., 2.]), parent, 1e-7, group_count=39))
    assert [result['ordinal'] for result in actual] == list(range(39))
    assert pool.last_completion_scheduling['effective_groups_per_job'] == 16
    assert pool.last_completion_scheduling['parent_output_batches'] == 3
    assert pool.last_completion_scheduling['submitted_batches'] == 0


class TinyOriginalCaller:
    def evaluate_camera_geometry(self, q, jacobian=True):
        pool = self.pool
        original_groups = ((p, rows, camera) for p, rows, camera in self.groups)
        return list(pool.ordered_groups(original_groups, q, self.absolute, 1e-7, batched_projection=True))


def test_caller_adapter_only_supplies_known_count_without_consuming_generator():
    class CapturePool:
        def ordered_groups(self, original_groups, world, absolute, step, *, batched_projection, group_count):
            assert group_count == 39
            assert iter(original_groups) is original_groups
            assert batched_projection is True
            assert step == 1e-7
            return ['unchanged original callback result']
    adapted, source_hash = _with_explicit_group_count(TinyOriginalCaller.evaluate_camera_geometry)
    owner = TinyOriginalCaller()
    owner.pool = CapturePool()
    owner.groups = groups(39)
    owner.absolute = result_for_job
    assert adapted(owner, np.array([1., 2.])) == ['unchanged original callback result']
    assert len(source_hash) == 64


def test_installer_patches_real_caller_count_without_constructing_any_graph(tmp_path, monkeypatch):
    from calibration.solver_runtime import install
    import inspect
    install()
    from structured_programme_source_candidate.paired_programme_blocks import StructuredProgrammeBlock
    original_pool = source.AbsoluteGeometryGroupPool.ordered_groups
    original_caller = StructuredProgrammeBlock.evaluate_camera_geometry
    monkeypatch.setattr(source.AbsoluteGeometryGroupPool, 'ordered_groups', original_pool)
    monkeypatch.setattr(source.AbsoluteGeometryGroupPool, '_rinkspace_completion_driven_FD_runtime', None, raising=False)
    monkeypatch.setattr(source.AbsoluteGeometryGroupPool, '_rinkspace_original_ordered_groups', None, raising=False)
    monkeypatch.setattr(StructuredProgrammeBlock, 'evaluate_camera_geometry', original_caller)
    monkeypatch.setattr(StructuredProgrammeBlock, '_rinkspace_original_geometry_group_count_caller', None, raising=False)
    receipt = install_completion_driven_absolute_fd_runtime(report_directory=tmp_path)
    assert source.AbsoluteGeometryGroupPool.ordered_groups is completion_driven_ordered_groups
    assert inspect.signature(StructuredProgrammeBlock.evaluate_camera_geometry) == inspect.signature(original_caller)
    assert StructuredProgrammeBlock._rinkspace_original_geometry_group_count_caller is original_caller
    assert receipt['input_generator_not_preconsumed_and_source_groups_not_split'] is True
    assert receipt['original_delivery_envelopes_and_first_exception_priority_retained'] is True
    assert install_completion_driven_absolute_fd_runtime(report_directory=tmp_path) is receipt
