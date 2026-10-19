from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from calibration.solver_runtime.strict_reviewed_programme_source import (
    FrozenArchivedProfileUpdate, StrictReviewedProgrammeSource,
)


def _runtime():
    from calibration.solver_runtime import install
    install()


def _forbidden(*args, **kwargs):
    raise AssertionError('Archived pixel/profile evaluator must not be called')


def test_archived_profile_auxiliary_is_frozen_and_retained_byte_for_byte():
    arrays = {'weights_0': np.array([[.25, .9], [1., .4]]),
              'present_0': np.array(True), 'robust_0': np.array(True)}
    class Archived:
        __call__ = _forbidden
        def checkpoint_arrays(self):
            return {key: value.copy() for key, value in arrays.items()}
        def iter_checkpoint_arrays(self):
            return iter(arrays.items())
        def restore_checkpoint(self, values):
            arrays.update({key: value.copy() for key, value in values.items()})
    update = FrozenArchivedProfileUpdate(Archived(), 7)
    before = {key: (value.dtype.str, value.shape, value.tobytes()) for key, value in arrays.items()}
    update(np.arange(7, dtype=float))
    assert before == {key: (value.dtype.str, value.shape, value.tobytes())
                      for key, value in update.iter_checkpoint_arrays()}
    update.restore_checkpoint(update.checkpoint_arrays())
    assert before == {key: (value.dtype.str, value.shape, value.tobytes()) for key, value in arrays.items()}
    with pytest.raises(ValueError, match='Complete finite'):
        update(np.zeros(8))


def test_roll_prior_complete_rotation_and_bias_columns_match_nonlinear_FD():
    _runtime()
    source = StrictReviewedProgrammeSource.__new__(StrictReviewedProgrammeSource)
    source.count = 25; source.sigma = np.deg2rad(.5); source.bias_slots = np.array([22, 23, 24])
    q = np.zeros(25); q[2:5] = [.25, -.22, .11]; q[9:12] = [-.24, .31, -.15]; q[22] = .01
    positions = np.array([2, 9])
    block = source._roll_block(q, positions, 0, 'prior', True)
    assert block.J.shape == (2, 25)
    assert np.array_equal(block.quadratic, [True, True])
    h = 1e-7
    for column in [2, 3, 4, 9, 10, 11, 22]:
        direction = np.eye(25)[column]
        plus = source._roll_block(q+h*direction, positions, 0, 'prior', False).r
        minus = source._roll_block(q-h*direction, positions, 0, 'prior', False).r
        assert np.allclose((plus-minus)/(2*h), block.J[:, column].toarray().ravel(), rtol=0, atol=1e-6)


def test_fit_iterator_cannot_consume_any_archived_pixel_or_profile_route():
    _runtime()
    from z190_streaming_source_normal import SourceRowBlock
    source = StrictReviewedProgrammeSource.__new__(StrictReviewedProgrammeSource)
    source.count = 5; source.prior_count = 1; source.scales = np.array([1., 2., 3.])
    source.base = SimpleNamespace(blocks=_forbidden, guard=_forbidden, updater=_forbidden)
    source.raw = SimpleNamespace(evaluations=_forbidden, tail_evaluations=_forbidden)
    source._engineering_blocks = lambda q, jacobian: iter([
        SourceRowBlock(np.array([.5]), csr_matrix([[1., 0., 0., 0., 0.]]) if jacobian else None,
                       np.ones(1, bool), np.ones(1), dict(family='explicit_prior'))])
    measured = np.array([7., -8.])
    factor = SimpleNamespace(iter_blocks=lambda q, jacobian: iter([
        SourceRowBlock(measured, csr_matrix((2, 5)) if jacobian else None,
                       np.ones(2, bool), np.array([2., 3.]), dict(family='strict_dataset'))]))
    source.strict_dataset_factor = factor
    blocks = list(source.blocks(np.zeros(5), True))
    assert [block.row_ledger['family'] for block in blocks] == ['explicit_prior', 'strict_dataset']
    assert np.array_equal(blocks[1].r, measured)
    assert blocks[1].row_ledger['original_start'] == 1
    assert blocks[1].row_ledger['original_stop'] == 3
    assert np.array_equal(np.concatenate([block.scales for block in blocks]), source.scales)
    assert all(block.J.shape[1] == 5 for block in blocks)


def test_actual_engineering_builder_does_not_evaluate_archived_programme_external_or_patterns():
    _runtime()
    from z190_streaming_source_normal import SourceRowBlock
    source = StrictReviewedProgrammeSource.__new__(StrictReviewedProgrammeSource)
    source.count = source.core_count = 30; source.sigma = np.deg2rad(.5)
    source.bias_slots = np.array([24, 25, 26]); source.soft_pivot = False
    source.external_roll_positions = np.array([0])
    names = ('main', 'secondary', 'low')
    blocks = []
    for index, name in enumerate(names):
        camera = np.zeros(25); camera[2] = 100.
        calibration = SimpleNamespace(camera25_and_jacobians=lambda view, camera=camera:
            (camera, np.zeros((15, 6)), np.zeros((15, 4))))
        bundle = SimpleNamespace(stamps=[100, 200], reference_pts=100, index={100: 0, 200: 1},
            calibration=SimpleNamespace(installation_id=name),
            current_calibration=lambda q, calibration=calibration: calibration,
            view=lambda q, pts: None, evaluate=_forbidden)
        block = SimpleNamespace(name=name, start=8*index, bundle=bundle, evaluate=_forbidden,
            camera_local=lambda q: q, optics_chart=SimpleNamespace(mount_indices=[27, 28, 29]))
        blocks.append(block)
    state = SimpleNamespace(ids=names, blocks=blocks,
        slots={name: np.array([27, 28, 29]) for name in names},
        external=SimpleNamespace(evaluate=_forbidden), patterns=SimpleNamespace(evaluate=_forbidden))
    source.state = state; source.raw = SimpleNamespace(refs=dict(zip(names, blocks)),
        evaluations=_forbidden, tail_evaluations=_forbidden)
    source.reference_factor = SimpleNamespace(parameter_count=30, block=lambda q, jacobian:
        SourceRowBlock(np.zeros(18), csr_matrix((18, 30)) if jacobian else None,
                       np.ones(18, bool), np.ones(18), dict(family='reference18')))
    q = np.zeros(30)
    for position in (0, 4, 8, 12, 16, 20):
        q[position:position+3] = [.3, -.2, .15]
    blocks = list(source._engineering_blocks(q, True))
    assert sum(len(block.r) for block in blocks) == 32
    assert sum(block.row_ledger['family'].startswith('programme_engineering_roll:') for block in blocks) == 3
    assert all(block.J.shape[1] == 30 for block in blocks)


def test_nonempty_reviewed_motion_cannot_fall_back_to_unknown_depth(monkeypatch):
    import calibration.solver_runtime.strict_reviewed_programme_source as module
    motion = dict(main1=dict(stamps=np.array([100]), previous_active_to_original=np.array([0]),
                             a=np.array([[12., 14.]])))
    monkeypatch.setattr(module, 'load_strict_programme_motion_inputs', lambda *a, **k: (motion, {}))
    bundle = SimpleNamespace(stamps=np.array([100]), a=np.array([[12., 14.]]))
    state = SimpleNamespace(blocks=[SimpleNamespace(name='main1', bundle=bundle)])
    with pytest.raises(ValueError, match='UNKNOWN_DEPTH is never admitted'):
        StrictReviewedProgrammeSource(SimpleNamespace(count=4), SimpleNamespace(state=state),
            SimpleNamespace(core_count=4), SimpleNamespace(parameter_count=4),
            dict(path='unused.json', sha256='0'*64))


def test_strict_ice_existing_pool_derivatives_match_complete_serial_arithmetic(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from calibration.solver_runtime import strict_dataset_execution_runtime as execution
    from calibration.solver_runtime.strict_source_dataset_adapter import _key
    targets = [dict(kind='current_scene_ice_curve', curve_id='one', feature_id='one', world_curve_role='segment0'),
               dict(kind='current_scene_ice_curve', curve_id='two', feature_id='two', world_curve_role='segment0')]
    def world(q, target):
        offset = 1. if target['curve_id'] == 'one' else -2.
        return np.array([[q[0]+offset, q[1]**2, 0.], [q[0]-offset, q[1], 0.]])
    transmitted = []
    def worker(job):
        transmitted.append(set(job))
        values = []
        for target in job['targets']:
            columns = []
            for position in job['positions']:
                plus, minus = job['world'].copy(), job['world'].copy()
                plus[position] += 1e-6; minus[position] -= 1e-6
                columns.append((world(plus, target)-world(minus, target))/(2e-6))
            values.append(np.stack(columns, axis=2))
        return dict(positions=job['positions'], values=values, failures=[], world_identity=job['world_identity'])
    monkeypatch.setattr(execution, 'evaluate_strict_ice_columns_worker', worker)
    with ThreadPoolExecutor(2) as executor:
        from calibration.solver_runtime.absolute_geometry_fd_pool_candidate import AbsoluteGeometryGroupPool
        pool = AbsoluteGeometryGroupPool(dict(path='existing', sha256='saved'), workers=2, max_inflight=2)
        pool.executor = executor
        # The test thread executor models the owned process executor's empty
        # queue; production admits the existing exact idle protocol only.
        executor._pending_work_items = {}
        pool._rinkspace_native_columns_idle_receipt = dict(executor_identity=id(executor), pending_jobs=0)
        q = np.array([.25, -.4])
        factor = SimpleNamespace(state=SimpleNamespace(blocks=[SimpleNamespace(_absolute_geometry_fd_pool=pool)]),
            rows=[dict(model_target=target) for target in targets], world_indices=np.array([0, 1]),
            _world_cache={}, _derivative_cache={}, _produce_world=lambda value, target: (world(value, target), None))
        assert execution._prepare_all_ice_derivatives(factor, q)
        for target in targets:
            columns, derivative = factor._derivative_cache[_key(target)]
            for position, column in enumerate(columns):
                plus, minus = q.copy(), q.copy(); plus[column] += 1e-6; minus[column] -= 1e-6
                assert np.array_equal(derivative[:, :, position], (world(plus,target)-world(minus,target))/(2e-6))
        assert all('state' not in keys and 'rows' not in keys and 'pixels' not in keys for keys in transmitted)
        assert factor.strict_execution_last_ice_FD['new_executor_manager_created'] is False
