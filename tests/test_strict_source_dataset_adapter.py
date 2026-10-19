from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from calibration.solver_runtime.strict_source_dataset_adapter import (
    compact_dataset_rows, partition_dataset_rows_by_family, StrictDatasetFactor)


def source_row(uv, sigma, *, role='fit', index=0, actual=100):
    return dict(split=role, scope='ice', actual_pts=actual, native_pts=actual,
        decoded_BGR_sha256='0' * 64, observation_semantics='paint_centerline',
        physical_material_id='C:segment0',
        model_target=dict(kind='current_scene_ice_curve', curve_id='C:segment0',
            feature_id='C', world_curve_role='segment0'),
        uv=np.asarray(uv, np.float64), sigma_px=np.asarray(sigma, np.float64),
        source_visible_paths=[list(range(len(uv)))],
        source_dataset_locator=dict(source_pack=dict(path='immutable.npz', sha256='a' * 64),
            source_row_index=index, local_point_indices=list(range(len(uv)))))


def test_same_primitive_group_preserves_all_pixels_sigma_and_independent_source_paths():
    first = source_row([[12., 7.], [14., 8.]], [1., 3.])
    second = source_row([[99., 40.]], [2.], index=3)
    frozen = deepcopy([first, second])
    rows = compact_dataset_rows([first, second], split='fit')
    assert len(rows) == 1
    assert np.array_equal(rows[0]['uv'], np.r_[first['uv'], second['uv']])
    assert np.array_equal(rows[0]['sigma'], [1., 3., 2.])
    assert [r['group_point_range'] for r in rows[0]['source_locators']] == [[0, 2], [2, 3]]
    assert [r['source_visible_paths'] for r in rows[0]['source_locators']] == [[[0, 1]], [[0]]]
    assert not rows[0]['uv'].flags.writeable
    for before, after in zip(frozen, [first, second]):
        assert np.array_equal(before['uv'], after['uv'])
        assert before['source_dataset_locator'] == after['source_dataset_locator']


def test_missing_support_and_typed_operator_cannot_silently_enter_fit():
    row = source_row([[1., 2.]], [np.nan])
    with pytest.raises(ValueError, match='Missing support'):
        compact_dataset_rows([row], split='fit')
    row['sigma_px'][:] = 1.
    row['normal_interval'] = [[-5., 5.]]
    with pytest.raises(ValueError, match='typed source operator'):
        compact_dataset_rows([row], split='fit')
    row.pop('normal_interval'); row['point_correspondence'] = True
    with pytest.raises(ValueError, match='point-correspondence'):
        compact_dataset_rows([row], split='fit')


@pytest.mark.parametrize('field', ['quadratic', 'loss_weights', 'loss_function',
    'robust_loss', 'huber_delta', 'source_integration_weights', 'source_integration_mass_rule'])
def test_future_explicit_loss_is_never_silently_replaced_by_default_policy(field):
    row = source_row([[1., 2.]], [1.]); row[field] = 2.
    with pytest.raises(ValueError, match='Explicit dataset loss policy'):
        compact_dataset_rows([row], split='fit')


def test_held_missing_sigma_and_point_semantics_survive_without_fit_admission():
    row = source_row([[1., 2.]], [np.nan], role='held')
    row['point_correspondence'] = True
    selected = compact_dataset_rows([row], split='held')[0]
    assert selected['point_correspondence'] is True
    assert np.isnan(selected['sigma'][0])
    with pytest.raises(ValueError, match='role/surface'):
        compact_dataset_rows([row], split='fit')


def test_other_exposure_or_material_does_not_reuse_same_primitive_group():
    one = source_row([[1., 2.]], [1.])
    two = source_row([[1., 2.]], [1.], actual=200)
    three = deepcopy(one); three['model_target']['world_curve_role'] = 'segment1'
    groups = compact_dataset_rows([one, two, three], split='fit')
    assert len(groups) == 3


def test_requested_current_ice_producer_matches_original_complete_inventory_knots_exactly():
    from rinkspace.vision_programme_ice_review import ice_curve_inventory
    features = [dict(id='C0', kind='circle', center_xyz=[0., 0., 0.], radius=2.),
        dict(id='F2', kind='circle', rings=[dict(role='outer', center_xyz=[1., 2., 0.], radius=3.),
            dict(role='inner', center_xyz=[1.1, 2.2, 0.], radius=.75)]),
        dict(id='C', kind='polyline', segments_xyz=[[[-3., 0., 0.], [3., 0., 0.]]]),
        dict(id='lane2', kind='polygon', paint_segments_xyz=[[[0., 0., 0.], [0., 2., 0.]],
            [[0., 2., 0.], [1., 2., 0.]], [[1., 2., 0.], [1., 0., 0.]]]),
        dict(id='goal_crease_left', kind='polygon', polygon_xyz=[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])]
    inventory = ice_curve_inventory(dict(ice_features=features))
    factor = StrictDatasetFactor.__new__(StrictDatasetFactor)
    factor.ice = SimpleNamespace(features=lambda q: {f['id']: f for f in features})
    factor.world_indices = np.arange(3)
    for curve in inventory:
        target = {k: curve[k] for k in ('curve_id', 'feature_id', 'world_curve_role')}
        actual = factor._ice_world(np.zeros(3), target)
        assert actual.shape == curve['xyz'].shape
        assert actual.tobytes() == curve['xyz'].tobytes()


class NumericalFactor(StrictDatasetFactor):
    """Two actual physical curve coordinates and the original camera15 chart."""
    def __init__(self):
        self.parameter_count = 17
        self.rows = compact_dataset_rows([source_row([[935., 438.]], [1.3]),
            source_row([[1075., 500.]], [.7], index=1)], split='fit')
        self.held = []
        self.scales = np.repeat([1.3, .7], 2)
        self.quadratic = np.zeros(4, bool); self.row_count = 4
        self._world_cache_q = self._camera_cache_q = self._evaluation_q = None
        self._world_cache = {}; self._derivative_cache = {}; self._camera_cache = {}
        block = SimpleNamespace(name='original_family_A')
        self.state = SimpleNamespace(blocks=[block], bypts={100:block})

    def _produce_world(self, q, target):
        world = np.array([[-1., -1., 0.], [1., -1., 0.], [1., 1., 0.]])
        world[:, 0] += q[15]; world[:, 1] += q[16]
        jacobian = np.zeros((3, 3, 2)); jacobian[:, 0, 0] = 1.; jacobian[:, 1, 1] = 1.
        return world, (np.array([15, 16]), jacobian)

    def _camera(self, q, actual, jacobian):
        camera = np.zeros(25); camera[:15] = q[:15]
        camera[:3] *= 100.; camera[8:11] *= 1000.
        return camera, np.arange(15), np.eye(15)


def test_full_brown_sparse_complete_chart_derivatives_match_real_nonlinear_trials(monkeypatch):
    monkeypatch.setenv('RINKSPACE_PROJECTION_BACKEND', 'numpy')
    from calibration.solver_runtime import install
    install()
    factor = NumericalFactor()
    q = np.zeros(17); q[2] = -.1
    q[6:8] = np.log(1000.); q[8:10] = [.96, .54]
    block = list(factor.iter_blocks(q, True))[0]
    assert block.J.shape == (4, 17)
    assert np.array_equal(block.scales, factor.scales)
    assert np.array_equal(block.quadratic, factor.quadratic)
    from calibration.solver_runtime.source_loss import weighted_source_cost
    assert weighted_source_cost(np.array([10., 0., 0., 0.]), block.quadratic, None) == 18.
    assert block.row_ledger['original_stop'] == 4
    value = list(factor.iter_blocks(q, False))[0].r
    assert np.allclose(block.r, value, rtol=0, atol=1e-12)
    direction = np.random.default_rng(7).normal(size=17)
    h = 1e-7
    plus = list(factor.iter_blocks(q + h * direction, False))[0].r
    minus = list(factor.iter_blocks(q - h * direction, False))[0].r
    measured = (plus - minus) / (2 * h)
    assert np.allclose(block.J @ direction, measured, rtol=2e-6, atol=2e-5)
    # A changed current q must not inherit cached old world geometry.
    later = q.copy(); later[15] = .25
    assert not np.array_equal(list(factor.iter_blocks(later, False))[0].r, value)


def test_actual_camera_chain_retains_conditioned_scales_and_positive_height_stencil(monkeypatch):
    from calibration.solver_runtime import install
    install()
    from structured_programme_source_candidate import fixed_pivot_positive_domain_fd as fd
    monkeypatch.setattr(fd, 'positive_pivot_height_columns', lambda blocks: np.array([2]))
    q = np.zeros(21); q[2] = 1e-8; q[16] = np.log(1000.)
    def camera(value, stamp):
        assert value[2] > 0
        result = np.zeros(25)
        result[:3] = 100 * value[:3] + value[9:12]
        result[3:6] = value[13:16]
        result[6:8] = value[16]
        result[8] = 1000 * value[6]; result[9] = 1000 * value[7]
        result[11] = .001 * value[8]
        return result
    block = SimpleNamespace(xyz_slots=np.array([0, 1, 2]), start=13,
        optics_chart=SimpleNamespace(indices=np.array([6, 7, 8]), mount_indices=np.array([9, 10, 11])),
        bundle=SimpleNamespace(camera=camera, index={100: 0}), camera_local=lambda core: core)
    factor = StrictDatasetFactor.__new__(StrictDatasetFactor)
    factor.core_count = 21; factor.state = SimpleNamespace(bypts={100: block})
    factor._world_cache_q = factor._camera_cache_q = factor._evaluation_q = None
    factor._world_cache = {}; factor._derivative_cache = {}; factor._camera_cache = {}
    factor._prepare_evaluation(q)
    nominal, columns, chain = factor._camera(q, 100, True)
    expected = np.zeros_like(chain)
    for i, column in enumerate(columns):
        if column in (0, 1, 2): expected[column, i] = 1.
        elif column in (6, 7): expected[column + 2, i] = 1.
        elif column == 8: expected[11, i] = .001
        elif column in (9, 10, 11): expected[column - 9, i] = .01
        elif column in (13, 14, 15): expected[column - 10, i] = 1.
        elif column == 16: expected[6:8, i] = 1.
    assert np.allclose(chain, expected, rtol=0, atol=2e-9)
    assert np.array_equal(nominal, camera(q, 100))
    assert np.array_equal(factor._camera(q, 100, True)[2], chain)


def test_original_programme_family_blocks_strictly_stack_all_r_J_sigma_and_masks(monkeypatch):
    monkeypatch.setenv('RINKSPACE_PROJECTION_BACKEND', 'numpy')
    from calibration.solver_runtime import install
    install()
    from scipy.sparse import vstack
    factor = NumericalFactor()
    a = SimpleNamespace(name='original_family_A'); b = SimpleNamespace(name='original_family_B')
    factor.state = SimpleNamespace(blocks=[a,b], bypts={100:a,200:b})
    entries = [source_row([[936.,439.]], [.9], actual=200),
        source_row([[940.,437.]], [1.1], actual=100),
        source_row([[1078.,510.]], [2.3], actual=200, index=1),
        source_row([[1074.,505.]], [.8], actual=100, index=1)]
    # Separate model primitives must stay separate even though the family
    # source block concatenates their complete unchanged numerical outputs.
    entries[2]['model_target']['world_curve_role'] = 'segment1'
    entries[3]['model_target']['world_curve_role'] = 'segment1'
    groups = compact_dataset_rows(entries, split='fit')
    original_uv = [r['uv'].copy() for r in groups]
    factor.family_groups = partition_dataset_rows_by_family(groups, factor.state)
    assert list(factor.family_groups) == ['original_family_A','original_family_B']
    factor.rows = [r for rows in factor.family_groups.values() for r in rows]
    factor.scales = np.concatenate([np.repeat(row['sigma'],2) for row in factor.rows])
    factor.quadratic = np.zeros(len(factor.scales),bool); factor.row_count=len(factor.scales)
    q = np.zeros(17); q[2] = -.1; q[6:8]=np.log(1000.); q[8:10]=[.96,.54]
    factor._prepare_evaluation(q)
    before = [factor._evaluate_row(q,row,True) for row in factor.rows]
    blocks = list(factor.iter_blocks(q,True))
    assert len(blocks)==2
    assert [block.row_ledger['programme_family'] for block in blocks] == list(factor.family_groups)
    expected_r=np.concatenate([pair[0] for pair in before])
    expected_J=vstack([pair[1] for pair in before],format='csr')
    actual_r=np.concatenate([block.r for block in blocks]); actual_J=vstack([block.J for block in blocks],format='csr')
    assert actual_r.tobytes()==expected_r.tobytes()
    for field in ('data','indices','indptr'):
        assert getattr(actual_J,field).tobytes()==getattr(expected_J,field).tobytes()
    assert np.concatenate([block.scales for block in blocks]).tobytes()==factor.scales.tobytes()
    assert np.concatenate([block.quadratic for block in blocks]).tobytes()==factor.quadratic.tobytes()
    assert [block.row_ledger['original_start'] for block in blocks]==[0,4]
    assert [block.row_ledger['original_stop'] for block in blocks]==[4,8]
    assert all(np.array_equal(old,row['uv']) for old,row in zip(original_uv,groups))
    assert factor.last_timing['source_blocks']==2
