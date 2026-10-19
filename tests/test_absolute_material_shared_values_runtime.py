"""Finite original-stencil cache fill/ownership; no complete graph or OS pool."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import hashlib
import json
import pickle
from types import SimpleNamespace

import numpy as np

from calibration.solver_runtime import absolute_geometry_fd_pool_candidate as source
from calibration.solver_runtime import absolute_material_value_runtime as runtime


class TinyOwner:
    parameter_count = 2
    calls = []

    def __init__(self):
        self.bindings = {'a': dict(offset=.5), 'b': dict(offset=1.5)}

    def _q(self, q):
        value = np.asarray(q, float)
        if value.shape != (2,) or not np.isfinite(value).all():
            raise ValueError('original complete domain')
        return value.copy()

    validate_parameters = _q

    def _rinkspace_original_material_value(self, binding, q):
        self.calls.append((binding['offset'], q.tobytes()))
        return np.array([[binding['offset'] + q[0], q[1], 3.]])


class TinyPool:
    max_inflight = 2

    def __init__(self, geometry, executor):
        self.parent_geometry = geometry
        self.executor = executor
        self.material_owner = dict(path='unchanged_saved_private_owner', sha256='unchanged_sha')
        self.owns_executor = False
        self.closed = False

    def _executor(self):
        return self.executor

    def _headroom(self, phase):
        return True


@contextmanager
def scope(geometry, world, bank=None):
    value = dict(job=dict(world66=world), geometry=geometry, root_world_bytes=None,
        shared_bank=bank, hits=0, original_value_calls=0, uncacheable_results=0, root_cache_resets=0)
    token = runtime._SCOPE.set(value)
    try:
        yield value
    finally:
        runtime._SCOPE.reset(token)


def test_each_original_world_material_once_and_byte_copyout(tmp_path, monkeypatch):
    geometry = TinyOwner()
    monkeypatch.setattr(TinyOwner, 'calls', [])
    monkeypatch.setattr(runtime, '_REPORT_DIRECTORY', tmp_path)
    monkeypatch.setattr(runtime, 'initialize_absolute_material_value_worker', lambda: None)
    monkeypatch.setitem(source._PRIVATE_GEOMETRIES,
        ('unchanged_saved_private_owner', 'unchanged_sha'), (geometry, set()))
    world = np.array([.125, .25])
    with ThreadPoolExecutor(max_workers=2) as executor:
        pool = TinyPool(geometry, executor)
        bank = runtime._prepare_shared_material_bank(pool, world, 1e-7)
        assert bank['entries'] == 10  # nominal + two original central samples per column
        assert len(geometry.calls) == len(set(geometry.calls)) == 10
        assert runtime._prepare_shared_material_bank(pool, world, 1e-7) is bank
        assert len(geometry.calls) == 10
        assert pool._rinkspace_shared_material_bank_pending_jobs == 0
        assert pool._rinkspace_shared_material_bank_precompute_active is False
        assert pool._rinkspace_shared_material_bank_idle_receipt['executor_identity'] == id(executor)
        wrapped = runtime._material_wrapper(TinyOwner._rinkspace_original_material_value)
        with scope(geometry, world, bank) as live:
            first = wrapped(geometry, geometry.bindings['a'], world)
            first[0, 0] = 999.
            second = wrapped(geometry, geometry.bindings['a'], world)
            assert second.tobytes() == np.array([[.625, .25, 3.]]).tobytes()
            assert live['shared_hits'] == 2
            assert live['original_value_calls'] == 0
        assert len(geometry.calls) == 10
        with scope(geometry, world + np.array([1., 0.]), bank) as live:
            wrapped(geometry, geometry.bindings['a'], world + np.array([1., 0.]))
            assert live['original_value_calls'] == 1
    report = json.loads((tmp_path / 'shared_material_preparation.jsonl').read_text().splitlines()[0])
    assert report['worker_jobs'] == 5
    assert report['original_material_calls'] == 10


def test_existing_parent_nominal_is_not_precomputed_again(tmp_path, monkeypatch):
    geometry = TinyOwner()
    monkeypatch.setattr(TinyOwner, 'calls', [])
    monkeypatch.setattr(runtime, '_REPORT_DIRECTORY', tmp_path)
    monkeypatch.setattr(runtime, 'initialize_absolute_material_value_worker', lambda: None)
    monkeypatch.setitem(source._PRIVATE_GEOMETRIES,
        ('unchanged_saved_private_owner', 'unchanged_sha'), (geometry, set()))
    world = np.array([.125, .25])
    wrapped = runtime._material_wrapper(TinyOwner._rinkspace_original_material_value)
    with scope(geometry, world):
        wrapped(geometry, geometry.bindings['a'], world)
    with ThreadPoolExecutor(max_workers=2) as executor:
        bank = runtime._prepare_shared_material_bank(TinyPool(geometry, executor), world, 1e-7)
        assert bank['entries'] == 10
    assert len(geometry.calls) == len(set(geometry.calls)) == 10
    report = json.loads((tmp_path / 'shared_material_preparation.jsonl').read_text().splitlines()[0])
    assert report['parent_seed_values'] == 1
    assert report['original_material_calls'] == 9


def test_nested_types_readonly_and_non_c_strides_roundtrip(tmp_path):
    values = (np.arange(24.).reshape(2, 3, 4).transpose(1, 0, 2),
              np.asfortranarray(np.arange(6.).reshape(2, 3)), np.float64(.5))
    values[1].flags.writeable = False
    owned = runtime._owned_value(values)
    path = tmp_path / 'values.bin'
    with path.open('wb') as stream:
        packed = runtime._shared_pack(owned, stream)
    mapping = np.memmap(path, dtype=np.uint8, mode='r')
    reconstructed = runtime._copy_out(runtime._shared_unpack(packed, mapping))
    assert type(reconstructed) is tuple
    assert type(reconstructed[2]) is np.float64
    for actual, original in zip(reconstructed[:2], values[:2]):
        assert actual.dtype == original.dtype and actual.shape == original.shape
        assert actual.flags.writeable == original.flags.writeable
        assert actual.tobytes() == original.tobytes()


def test_force_parent_never_opens_precompute_executor(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, '_REPORT_DIRECTORY', tmp_path)
    pool = SimpleNamespace(_rinkspace_force_parent_activation_FD=True)
    assert runtime._prepare_shared_material_bank(pool, np.arange(2.), 1e-7) is None


def test_programme_scope_is_lazy_and_preserves_current_original_call(monkeypatch):
    geometry = TinyOwner()
    monkeypatch.setattr(TinyOwner, 'calls', [])
    monkeypatch.setattr(runtime, '_REPORT_DIRECTORY', None)
    seen = []
    wrapped_value = runtime._material_wrapper(TinyOwner._rinkspace_original_material_value)
    class Programme:
        name = 'original_tiny'
        def __init__(self):self.geometry = geometry
        def geometry_values(self, q):
            seen.append('original_world_getter')
            return q
    def original(block, q, jacobian=True):
        seen.append('original_admission')
        for _ in range(2):wrapped_value(block.geometry, block.geometry.bindings['a'], q)
        return ('original_result', jacobian)
    programme = Programme()
    wrapped = runtime._programme_material_scope(original)
    assert wrapped.__wrapped__ is original
    assert wrapped(programme, np.array([.125, .25])) == ('original_result', True)
    assert seen == ['original_admission', 'original_world_getter']
    assert len(geometry.calls) == 1
    assert runtime._SCOPE.get() is None
    assert programme._rinkspace_last_material_value_scope['hits'] == 1
