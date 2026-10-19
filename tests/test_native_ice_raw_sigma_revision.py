from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from calibration.solver_runtime import install

install()
from structured_programme_source_candidate import check_native_ice_raw_sigma_revision as revision
from structured_programme_source_candidate import native_ice_directional_diagnostic as directional


def test_qualified_normalization_uses_original_N_and_the_same_active_slots(monkeypatch):
    rows = [dict(id='curve-a', split='fit', actual_pts=100, uv=np.array([[1., 2.], [3., 4.], [5., 6.]]),
                 sigma=np.array([2., 3., 4.]), curve=dict(kind='current_ice_segment')),
            dict(id='curve-b', split='fit', actual_pts=200, uv=np.array([[7., 8.], [9., 10.]]),
                 sigma=np.array([5., 6.]), curve=dict(kind='current_ice_segment'))]
    qualification = SimpleNamespace(point_mask=lambda row: np.array([False, True, True])
                                   if row['id'] == 'curve-a' else np.array([True, True]))
    raw_r = np.arange(1., 11.)
    raw_J = np.arange(20., dtype=float).reshape(10, 2)
    sigma = np.concatenate([np.repeat(row['sigma'], 2) for row in rows])
    multiplier = np.concatenate([np.full(2 * len(row['uv']), np.sqrt(len(row['uv']))) for row in rows])
    selected = np.arange(2, 10)

    class Historical:
        def __init__(self, *args):
            self.rows, self.held = rows, []
            self.scales = sigma * multiplier
            self.row_count = len(selected)
            self.quadratic = np.ones(len(selected), bool)
            self.source_point_qualification = qualification
            self.contract = dict(schema='original-v1')

        def block(self, q, jacobian):
            assert self.source_point_qualification is None
            assert self.row_count == len(raw_r)
            assert np.array_equal(self.scales, sigma * multiplier)
            return SimpleNamespace(r=raw_r / multiplier, J=csr_matrix(raw_J / multiplier[:, None]),
                                   quadratic=self.quadratic)

    active = SimpleNamespace(r=raw_r[selected], J=csr_matrix(raw_J[selected]),
                             quadratic=np.ones(len(selected), bool), scales=sigma[selected])
    extra = SimpleNamespace(rows=rows, held=[], state=object(), packet_path='original.json', parameter_count=2,
                            row_count=len(selected), source_point_qualification=qualification,
                            contract=dict(schema='raw-v2', packet_sha256='source'), block=lambda *args: active)
    monkeypatch.setattr(revision, 'MeanV1OnCurrentGeometry', Historical)
    result = revision.check_raw_sigma_revision(extra, np.array([1., 2.]))
    assert result['passed']
    assert result['original_scalar_rows'] == 10
    assert result['qualified_active_scalar_rows'] == 8
    assert result['source_only_qualified_scalar_rows_removed'] == 2
    assert result['historical_group_sqrt_N_uses_original_source_count']
    assert result['actual_r_vs_v1_times_sqrt_N_max_absolute'] < 1e-13
    assert np.array_equal(extra.source_point_qualification.point_mask(rows[0]), [False, True, True])

    active.scales = sigma[:-2]
    with pytest.raises(ValueError, match='Raw per-UV sigma'):
        revision.check_raw_sigma_revision(extra, np.array([1., 2.]))


def test_empty_or_wrong_shape_qualification_does_not_silently_reindex():
    row = dict(uv=np.ones((3, 2)))
    for mask in (np.empty(0, bool), np.array([[True, True, True]]), np.ones(3)):
        extra = SimpleNamespace(rows=[row], source_point_qualification=SimpleNamespace(point_mask=lambda row: mask))
        with pytest.raises(ValueError, match='invalid original point mask'):
            revision._active_original_scalar_indices(extra)

    extra = SimpleNamespace(rows=[row], source_point_qualification=None)
    assert np.array_equal(revision._active_original_scalar_indices(extra), np.arange(6))


def test_directional_branch_ledger_uses_active_slots_and_original_UV_identity(monkeypatch):
    from pathlib import Path
    import tempfile

    data_root = Path('D:/CodexData/RinkSpace_Data')
    if not data_root.is_dir():
        pytest.skip('Actual runtime diagnostic location is unavailable')
    q = np.array([1., 2.])
    direction = np.array([.1, -.2])
    step = 1e-6
    original_row = dict(id='curve', split='fit', feature_id='C0', curve=dict(kind='hockey_circle'),
                        pts=100, actual_pts=100, uv=np.array([[3., 4.], [5., 6.]]), sigma=np.array([4., 4.]),
                        source_raw_frame='actual.png', source_file_sha256='source', _source_optical_support=None)
    active_row = dict(original_row, uv=original_row['uv'][1:], sigma=original_row['sigma'][1:])
    active = SimpleNamespace(rows=[active_row], parameter_count=2, row_count=2, scales=np.array([4., 4.]),
                             quadratic=np.ones(2, bool), contract=dict(packet_sha256='packet'),
                             _qualified_original_index_map=[dict(id='curve', retained_original_point_indices=[1])],
                             _cameras=lambda value: {100: np.asarray(value)},
                             _world=lambda value, curve: np.zeros((2, 3)))
    extra = SimpleNamespace(rows=[original_row], _qualified_active_fit_view=lambda: active)
    block = SimpleNamespace(r=q, J=csr_matrix(np.eye(2)), scales=active.scales, quadratic=active.quadratic)

    def actual_branch(camera, world, uv):
        return dict(nearest_uv=uv + 4 * camera, index=np.array([0]), raw_t=np.array([.5]),
                    category=np.array([0]), segment_valid=np.array([True]), central=np.array([True, True]),
                    midpoint_central=np.array([True]))

    monkeypatch.setattr(directional, '_branch', actual_branch)
    monkeypatch.setattr(directional, '_curve_error_on_association',
                        lambda camera, world, uv, association: (np.tile(4 * camera, len(uv)), association))
    with tempfile.TemporaryDirectory(prefix='native_ice_active_diagnostic_test_', dir=data_root) as temporary:
        out = Path(temporary).resolve()
        assert out.is_relative_to(data_root.resolve())
        report = directional.record_native_ice_directional_diagnostic(
            extra, q, q, block, direction, step, q + step * direction, q - step * direction,
            direction, out, dict(source='actual-current-layout-test'))
        assert report['branch_replay_completed']
        assert report['source_review_excluded_scalar_rows'] == 2
        assert report['row_ledger'][0]['original_source_UV_indices'] == [1]
        assert report['original_argmax']['source_UV_index'] == 1
        assert report['original_argmax']['active_source_UV_index'] == 0
        assert max(report['branch_value_replay_max_difference'].values()) < 1e-13
