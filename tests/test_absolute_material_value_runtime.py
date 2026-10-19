"""Original FD/rebind fixtures and array ownership; no complete source graph."""
from contextlib import contextmanager
import pickle

import numpy as np
import pytest

from calibration.solver_runtime import install
from calibration.solver_runtime import absolute_geometry_fd_pool_candidate as source
from calibration.solver_runtime import absolute_material_value_runtime as runtime

install()
from structured_programme_source_candidate.structured_programme_geometry import StructuredProgrammeGeometry


class TinyGeometry(StructuredProgrammeGeometry):
    parameter_count = 2

    def __init__(self):
        self.bindings = {'same_material': dict(kind='ad', fields=('world',), offset=.5)}
        self.scene_unit_m = 2.
        self.original_value_calls = 0
        self.validations = []

    def _key(self, row):
        return row['material_id']

    def validate_parameters(self, q):
        value = np.asarray(q, float)
        self.validations.append(value.tobytes())
        if value.shape != (2,) or not np.isfinite(value).all():
            raise ValueError('original complete world guard')
        return value

    def _q(self, q):
        return self.validate_parameters(q)

    def source_parameter_activity(self, row, q):
        self.validate_parameters(q)
        return np.array([True, False])

    def _ad_value(self, binding, q, jacobian):
        self.original_value_calls += 1
        world = np.array([[binding['offset'] + q[0], 0., 3.]])
        return world, np.zeros((1, 3, 2))


def tiny_absolute(camera, rows, jacobian):
    return np.concatenate([np.array([np.asarray(row['world'])[0, 0] * 3.,
                                    np.asarray(row['world'])[0, 0] ** 2]) for row in rows])


def job_for(world, rows, ordinal):
    return dict(material_owner=dict(path='no_owner_loaded', sha256='unchanged'),
        world66=np.asarray(world, float), camera25=np.zeros(25), rows=rows,
        source_group_sha256=source._source_group_digest(rows), absolute=tiny_absolute,
        step=1e-7, batched_projection=True, ordinal=ordinal, actual_pts=100 + ordinal)


@contextmanager
def direct_scope(geometry, root):
    scope = dict(geometry=geometry, job=dict(world66=root), root_world_bytes=None,
        hits=0, original_value_calls=0, uncacheable_results=0, root_cache_resets=0)
    token = runtime._SCOPE.set(scope)
    try:
        yield scope
    finally:
        runtime._SCOPE.reset(token)


def test_original_group_FD_stencils_rebind_companions_and_bytes_preserved(monkeypatch):
    rows = [dict(id=str(index), material_id='same_material', uv=[[index, .25]],
                 world=np.array([[.5, 0., 3.]]), world_xyz=[[.5, 0., 3.]],
                 world_xyz_scene_units=[[1., 0., 6.]]) for index in range(2)]
    original_group = source._evaluate_group_on_owner
    original_value = StructuredProgrammeGeometry._material_value
    baseline = TinyGeometry()
    baseline_certificates = set()
    expected = [original_group(job_for([.125, 0.], rows, index), baseline, baseline_certificates, 'parent_actor_busy_activation')
                for index in range(2)]
    candidate = TinyGeometry()
    candidate_certificates = set()
    monkeypatch.setattr(TinyGeometry, '_material_value', runtime._material_wrapper(original_value))
    monkeypatch.setattr(runtime, '_ORIGINAL_GROUP', original_group)
    actual = [runtime._evaluate_group_with_material_values(job_for([.125, 0.], rows, index), candidate,
              candidate_certificates, 'parent_actor_busy_activation') for index in range(2)]
    assert baseline.validations == candidate.validations
    assert baseline_certificates == candidate_certificates
    assert candidate.original_value_calls < baseline.original_value_calls
    for old, new in zip(expected, actual):
        assert old['geo'].dtype == new['geo'].dtype
        assert old['geo'].shape == new['geo'].shape
        assert old['geo'].tobytes() == new['geo'].tobytes()
        assert old['stats']['physical_domain_FD_stencils'] == new['stats']['physical_domain_FD_stencils']
        assert old['source_group_sha256'] == new['source_group_sha256']
        assert old['source_row_ids'] == new['source_row_ids']
        assert new['stats']['absolute_material_value_runtime']['hits'] > 0
    assert runtime._SCOPE.get() is None
    assert not any('rinkspace' in key for key in vars(candidate))
    restored = pickle.loads(pickle.dumps(candidate, protocol=5))
    assert restored not in runtime._VALUES
    # Source companions still fail even with a populated material cache.
    changed = dict(rows[0], world_xyz=[[999., 0., 3.]])
    with pytest.raises(ValueError, match='companion'):
        runtime._evaluate_group_with_material_values(job_for([.125, 0.], [changed], 3), candidate,
            candidate_certificates, 'parent_actor_busy_activation')
    assert runtime._SCOPE.get() is None


def test_full_sample_world_and_binding_identity_keys_root_reset_and_copy_out():
    geometry = TinyGeometry()
    binding = geometry.bindings['same_material']
    wrapped = runtime._material_wrapper(StructuredProgrammeGeometry._material_value)
    with direct_scope(geometry, np.array([.125, 0.])) as scope:
        first = wrapped(geometry, binding, np.array([.125, 0.]))
        first[0, 0] = 999.
        second = wrapped(geometry, binding, np.array([.125, 0.]))
        assert second[0, 0] == .625
        wrapped(geometry, binding, np.array([.125, 1.]))
        wrapped(geometry, dict(binding), np.array([.125, 0.]))
        assert scope['hits'] == 1
        assert scope['original_value_calls'] == 3
        assert len(runtime._VALUES[geometry]['entries']) == 3
    with direct_scope(geometry, np.array([.25, 0.])) as scope:
        wrapped(geometry, binding, np.array([.125, 0.]))
        assert scope['hits'] == 0
        assert scope['root_cache_resets'] == 1
        assert len(runtime._VALUES[geometry]['entries']) == 1
    calls = geometry.original_value_calls
    wrapped(geometry, binding, np.array([.125, 0.]))
    assert geometry.original_value_calls == calls + 1


def test_original_return_type_flags_and_exception_not_cached():
    geometry = TinyGeometry()
    binding = {}
    calls = []
    def original(owner, material, world):
        calls.append(world.tobytes())
        if material.get('fail'):
            raise ValueError('original material error')
        readonly = np.array([3., 4.])
        readonly.flags.writeable = False
        return (np.array([1., 2.]), readonly, np.float64(5.))
    wrapped = runtime._material_wrapper(original)
    with direct_scope(geometry, np.array([.125, 0.])):
        first = wrapped(geometry, binding, np.array([.125, 0.]))
        first[0][0] = 999.
        second = wrapped(geometry, binding, np.array([.125, 0.]))
        assert type(second) is tuple
        assert type(second[0]) is np.ndarray
        assert type(second[2]) is np.float64
        assert second[0].flags.writeable is True
        assert second[1].flags.writeable is False
        assert second[0].tobytes() == np.array([1., 2.]).tobytes()
        failure = {'fail': True}
        for _ in range(2):
            with pytest.raises(ValueError, match='original material error'):
                wrapped(geometry, failure, np.array([.125, 0.]))
    assert len(calls) == 3


def test_picklable_spawn_initializer_calls_original_before_install(monkeypatch):
    order = []
    monkeypatch.setattr(source, 'initialize_absolute_geometry_worker', lambda: order.append('original'))
    monkeypatch.delattr(source, '_rinkspace_original_material_worker_initializer', raising=False)
    monkeypatch.setattr(runtime, '_install_runtime', lambda: order.append('install'))
    initializer = pickle.loads(pickle.dumps(runtime.initialize_absolute_material_value_worker))
    initializer()
    assert order == ['original', 'install']


def test_real_installer_keeps_original_initializer_and_scope_parent_owner(monkeypatch, tmp_path):
    original_material = StructuredProgrammeGeometry._material_value
    original_group = source._evaluate_group_on_owner
    original_initializer = source.initialize_absolute_geometry_worker
    monkeypatch.setattr(runtime, '_INSTALLED', False)
    monkeypatch.setattr(runtime, '_ORIGINAL_GROUP', None)
    monkeypatch.setattr(runtime, '_RECEIPT', None)
    monkeypatch.setattr(runtime, '_REPORT_DIRECTORY', None)
    monkeypatch.setattr(StructuredProgrammeGeometry, '_material_value', original_material)
    monkeypatch.setattr(StructuredProgrammeGeometry, '_rinkspace_original_material_value', None, raising=False)
    monkeypatch.setattr(source, '_evaluate_group_on_owner', original_group)
    monkeypatch.setattr(source, 'initialize_absolute_geometry_worker', original_initializer)
    monkeypatch.setattr(source, '_rinkspace_original_material_group_evaluate', None, raising=False)
    monkeypatch.setattr(source, '_rinkspace_original_material_worker_initializer', None, raising=False)
    receipt = runtime.install_absolute_material_value_runtime(report_directory=tmp_path)
    assert source._rinkspace_original_material_group_evaluate is original_group
    assert source._rinkspace_original_material_worker_initializer is original_initializer
    assert source.initialize_absolute_geometry_worker is runtime.initialize_absolute_material_value_worker
    assert source._evaluate_group_on_owner is runtime._evaluate_group_with_material_values
    assert StructuredProgrammeGeometry._rinkspace_original_material_value is original_material
    assert receipt['cache_outside_geometry_pickle'] is True
    assert runtime.install_absolute_material_value_runtime(report_directory=tmp_path) is receipt
    assert pickle.loads(pickle.dumps(source.initialize_absolute_geometry_worker)) is source.initialize_absolute_geometry_worker
