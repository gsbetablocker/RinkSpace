"""CPU checks for normal-matrix scaling ranges and FP64 value preservation."""

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from calibration.program_tracking.body_tracking.body_gpu_linear import _coalesce_gram_tiles


@pytest.mark.parametrize(('tiles', 'options', 'expected'), [
    pytest.param([], {}, [], id='empty'),
    pytest.param([(0, 41, 0, 125)], {}, [(0, 41, 0, 125)], id='partial-tile'),
    pytest.param([(0, 0, 0, 0)], {}, [(0, 0, 0, 0)], id='zero-length'),
    pytest.param([(0, 96, 0, 0), (96, 150, 0, 0)], {}, [(0, 150, 0, 0)], id='zero-nnz'),
    pytest.param([(0, 96, 0, 100), (96, 192, 100, 200), (192, 221, 200, 230)],
                 {'max_rows': 192}, [(0, 192, 0, 200), (192, 221, 200, 230)], id='row-budget'),
    pytest.param([(0, 96, 0, 1_200_000), (96, 192, 1_200_000, 2_000_000),
                  (192, 200, 2_000_000, 2_000_001)], {},
                 [(0, 192, 0, 2_000_000), (192, 200, 2_000_000, 2_000_001)], id='nnz-budget'),
    pytest.param([(0, 96, 0, 2_000_001), (96, 192, 2_000_001, 2_000_020)], {},
                 [(0, 96, 0, 2_000_001), (96, 192, 2_000_001, 2_000_020)], id='oversized-nnz'),
    pytest.param([(0, 2048, 0, 12), (2048, 2049, 12, 13)], {},
                 [(0, 2048, 0, 12), (2048, 2049, 12, 13)], id='oversized-rows'),
])
def test_scaling_groups_preserve_contiguous_ranges(tiles, options, expected):
    assert _coalesce_gram_tiles(tiles, **options) == expected


@pytest.mark.parametrize(('tiles', 'options', 'message'), [
    ([(0, 2, 0, 2), (3, 4, 2, 3)], {}, 'contiguous'),
    ([(0, 2, 0, 2), (1, 4, 2, 3)], {}, 'contiguous'),
    ([(0, 2, 0, 2), (2, 4, 1, 3)], {}, 'contiguous'),
    ([(0, 0, 0, 2)], {}, 'Invalid Gram tile'),
    ([], {'max_rows': 0}, 'budgets must be positive'),
    ([], {'max_nnz': -1}, 'budgets must be positive'),
])
def test_scaling_groups_reject_invalid_ranges_and_budgets(tiles, options, message):
    with pytest.raises(ValueError, match=message):
        _coalesce_gram_tiles(tiles, **options)


@pytest.mark.parametrize(('columns', 'observations', 'max_nnz'), [
    (197, 273, 2_000_000),
    (1537, 1873, 2_000_000),
    (197, 273, 1_000),
])
def test_grouped_scaling_preserves_fp64_values_and_explicit_diagonal(columns, observations, max_nnz):
    rng = np.random.default_rng(934)
    rows = np.repeat(np.arange(observations), 7)
    # Reserve one structurally empty column to exercise the damping slot.
    indices = rng.integers(0, columns - 1, len(rows))
    jacobian = csr_matrix((rng.normal(size=len(rows)), (rows, indices)), shape=(observations, columns))
    normal = (jacobian.T @ jacobian).tocsr()
    normal.setdiag(normal.diagonal())
    normal.sum_duplicates()
    normal.sort_indices()
    rowptr = normal.indptr
    tile_edges = [*range(0, columns, 96), columns]
    tiles = [(first, last, rowptr[first], rowptr[last]) for first, last in zip(tile_edges, tile_edges[1:])]
    groups = _coalesce_gram_tiles(tiles, max_nnz=max_nnz)
    if max_nnz == 2_000_000:
        assert len(groups) < len(tiles)

    scale = np.exp(rng.uniform(-3.0, 3.0, columns))
    diagonal = rng.uniform(0.0, 1.0, columns)
    penalty = np.full(columns, 0.07 ** 2) + diagonal ** 2
    slots = np.array([rowptr[i] + np.searchsorted(normal.indices[rowptr[i]:rowptr[i + 1]], i)
                      for i in range(columns)])
    np.testing.assert_array_equal(normal.indices[slots], np.arange(columns))
    assert normal.data[slots[-1]] == 0.0

    def scaled(ranges):
        values = normal.data.copy()
        for first, last, start, end in ranges:
            row_scale = np.repeat(scale[first:last], np.diff(rowptr[first:last + 1]))
            # Same FP64 multiplication order as GPUJacobian.solve_scaled.
            values[start:end] *= row_scale * scale[normal.indices[start:end]]
        values[slots] += penalty
        return values

    grouped = scaled(groups)
    reference = normal.data * (np.repeat(scale, np.diff(rowptr)) * scale[normal.indices])
    reference[slots] += penalty
    assert grouped.dtype == np.float64
    np.testing.assert_array_equal(grouped.view(np.uint64), scaled(tiles).view(np.uint64))
    np.testing.assert_array_equal(grouped.view(np.uint64), reference.view(np.uint64))
    assert grouped[slots[-1]] == penalty[-1]
