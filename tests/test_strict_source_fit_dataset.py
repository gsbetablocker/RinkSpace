from copy import deepcopy

import numpy as np
import pytest

from calibration.program_tracking.strict_source_fit_dataset import partition_support, finite_source_row


def test_missing_fit_support_does_not_delete_held_or_finite_stations():
    rows = [dict(split='fit', point_count=5), dict(split='held', point_count=3)]
    sigma = np.array([1., np.nan, 2., 3., np.nan, np.nan, 1., 1.])
    result = partition_support(rows, np.array([0,5,8]), sigma)
    assert result['fit_point_indices'].tolist() == [0,2,3]
    assert result['held_point_indices'].tolist() == [5,6,7]
    assert result['pending_support_point_indices'].tolist() == [1,4]


def test_retained_paths_cannot_bridge_missing_readout_and_typed_arrays_follow_uv():
    uv = np.c_[np.arange(5.), np.zeros(5)]
    row = dict(split='fit', scope='ice', uv=uv, sigma_px=np.array([1.,1.,np.nan,1.,1.]),
        point_count=5, original_point_indices=np.arange(5),
        source_visible_paths=[[0,1,2,3,4]], source_query_indices=np.arange(5),
        normal_interval=np.arange(10).reshape(5,2),
        source_readout_uncertainty=dict(actual_side_gradient_quantiles_uv=np.arange(60).reshape(5,2,3,2)),
        curve_observation_kind='clipped_extent', clip_boundary_x=10.)
    original = deepcopy(row)
    result = finite_source_row(row, [0,1,3,4], {'path':'source.npz','sha256':'frozen'}, 3)
    assert result['source_visible_paths'] == [[0,1],[2,3]]
    assert np.array_equal(result['normal_interval'], original['normal_interval'][[0,1,3,4]])
    assert np.array_equal(result['source_readout_uncertainty']['actual_side_gradient_quantiles_uv'],
        original['source_readout_uncertainty']['actual_side_gradient_quantiles_uv'][[0,1,3,4]])
    assert np.array_equal(result['uv'], original['uv'][[0,1,3,4]])
    assert result['clip_boundary_x'] == 10.
    assert row['source_visible_paths'] == [[0,1,2,3,4]]
    assert np.isnan(row['sigma_px'][2])
    with pytest.raises(ValueError, match='Missing source support'):
        finite_source_row(row, [2], {'path':'source.npz','sha256':'frozen'}, 3)
    row['split'] = 'held'
    with pytest.raises(ValueError, match='HELD'):
        finite_source_row(row, [0,1,3,4], {'path':'source.npz','sha256':'frozen'}, 3)


def test_complete_held_row_preserves_missing_support_and_typed_source_values():
    row = dict(split='held', scope='board', point_count=3,
        uv=np.arange(6., dtype=np.float64).reshape(3,2),
        sigma_px=np.array([np.nan,1.,np.nan]), original_point_indices=np.array([7,8,9]),
        source_visible_paths=[[2,1],[0]], anchor_point_indices=np.array([20,21,22]),
        source_sides_uv=np.arange(12.).reshape(3,2,2),
        source_side_quantiles_uv=np.arange(36.).reshape(3,2,3,2),
        raw_candidate_uv=np.arange(6.).reshape(3,2),
        normal_interval=np.arange(6.).reshape(3,2),
        point_support_bounds_xy_px=np.arange(12.).reshape(3,2,2),
        source_tangent_interval_px=np.arange(6.).reshape(3,2),
        curve_observation_kind='axis_scanline_branch', scanline_branch='min_y',
        temporal_evidence={'kind':'frozen_source_witness','source_pts':100})
    original = deepcopy(row)
    result = finite_source_row(row, [0,1,2], {'path':'source.npz','sha256':'frozen'}, 4)
    for name, value in original.items():
        if isinstance(value,np.ndarray):
            assert result[name].dtype == value.dtype
            assert result[name].shape == value.shape
            assert result[name].tobytes() == value.tobytes()
        else:
            assert result[name] == value
    assert result['split'] == 'held'
    assert result['working_sigma_status'] == 'missing_or_partial_source_working_sigma_not_fit_ready'
    assert row['source_visible_paths'] == [[2,1],[0]]


def test_query_gaps_and_missing_stations_remain_separate_after_finite_selection():
    row = dict(split='fit', scope='ice', point_count=5,
        uv=np.c_[np.arange(5.),np.zeros(5)], sigma_px=np.array([1.,np.nan,1.,1.,1.]),
        source_query_indices=np.array([10,11,20,21,22]),
        final_current_RGB_pixel_precision_checked=True)
    result = finite_source_row(row, [0,2,3,4], {'path':'source.npz','sha256':'frozen'}, 0)
    assert result['source_visible_paths'] == [[0],[1,2,3]]
    assert result['source_query_indices'].tolist() == [10,20,21,22]
    assert result['source_dataset_original_path_basis'] == 'source_query_indices'
    assert 'source_visible_paths' not in row
