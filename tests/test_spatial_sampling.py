import importlib
from pathlib import Path

import numpy as np
import pytest

from rinkspace.session.spatial_sampling import SamplingReason, sample_source_pixels


def _sample(uv, *, pts=None, paths=None, ids=None, **options):
    n = len(uv)
    return sample_source_pixels(
        np.asarray(uv, dtype=float).reshape(n, 2),
        np.full(n, 256, dtype=np.int64) if pts is None else pts,
        np.full(n, "physical/curve") if paths is None else paths,
        np.array([f"source-{i:04}" for i in range(n)]) if ids is None else ids,
        **options,
    )


def test_duplicate_multiplicity_and_row_permutation_preserve_selected_provenance():
    uv = np.c_[np.arange(0., 121., 3.), np.zeros(41)]
    ids = np.array([f"original-{i:04}" for i in range(len(uv))])
    baseline = _sample(uv, ids=ids)
    repeated = np.r_[np.arange(len(uv)), np.repeat(np.arange(10, 20), 17)]
    dense_uv = uv[repeated]
    dense_ids = np.r_[ids, [f"z-copy-{i:04}" for i in range(len(repeated) - len(uv))]]
    expected = set(ids[baseline.selected_indices])
    permutation = np.random.default_rng(7).permutation(len(dense_uv))
    dense = _sample(dense_uv, ids=dense_ids)
    shuffled = _sample(dense_uv[permutation], ids=dense_ids[permutation])
    assert set(dense_ids[dense.selected_indices]) == expected
    assert set(dense_ids[permutation][shuffled.selected_indices]) == expected
    assert np.count_nonzero(dense.reason_codes == SamplingReason.DUPLICATE_UV) == len(dense_uv) - len(uv)


def test_explicit_stable_key_decides_duplicate_representative():
    result = _sample([[0, 0], [0, 0], [80, 0]], ids=np.array(["a", "z", "end"]),
                     stable_keys=np.array([20, 10, 30]))
    np.testing.assert_array_equal(result.selected_indices, [1, 2])
    assert result.reason_codes[0] == SamplingReason.DUPLICATE_UV


def test_subset_covers_curved_and_disconnected_visible_support_without_editing_uv():
    theta = np.linspace(0, np.pi, 301)
    uv = np.r_[np.c_[100*np.cos(theta), 100*np.sin(theta)],
               np.c_[np.linspace(600, 720, 301), np.full(301, 20.)]]
    original = uv.copy()
    uv.flags.writeable = False
    result = _sample(uv, spacing_px=24.)
    chosen = uv[result.selected_indices]
    assert len(chosen) < len(uv)
    assert np.any(chosen[:, 0] > 500) and np.any(chosen[:, 0] < 500)
    for point in uv:
        assert np.hypot(*(chosen - point).T).min() <= 24.
    np.testing.assert_array_equal(uv, original)
    assert np.all(np.diff(result.selected_indices) > 0)


def test_pts_and_physical_paths_cannot_cover_each_other():
    uv = np.tile([[0., 0.], [12., 0.], [48., 0.]], (4, 1))
    pts = np.repeat([256, 256, 512, 512], 3)
    paths = np.repeat(["one", "two", "one", "two"], 3)
    result = _sample(uv, pts=pts, paths=paths)
    for first in range(0, 12, 3):
        np.testing.assert_array_equal(result.selected_indices[
            (result.selected_indices >= first) & (result.selected_indices < first + 3)], [first, first + 2])


def test_all_fixed_rows_survive_even_near_curve_or_coincident_fixed_rows():
    uv = [[0, 0], [50, 0], [50, 0], [50, 0], [51, 0], [100, 0]]
    kind = np.array([1, 0, 0, 1, 0, 1])
    result = _sample(uv, kind=kind)
    assert {1, 2, 4}.issubset(result.selected_indices)
    assert np.all(result.reason_codes[[1, 2, 4]] == SamplingReason.FIXED_POINT)
    assert result.reason_codes[3] == SamplingReason.DUPLICATE_UV


def test_explicit_finite_fragments_keep_their_own_source_extrema():
    uv = [[0, 0], [2, 0], [3, 0], [4, 0], [8, 0], [10, 0]]
    result = _sample(uv, fragment_ids=np.array([0, 0, 1, 1, 2, 2]))
    np.testing.assert_array_equal(result.selected_indices, np.arange(6))
    # Shared UVs belong to one retained source record even across fragments.
    result = _sample([[0, 0], [0, 0]], ids=np.array(["z", "a"]), fragment_ids=np.array([0, 1]))
    np.testing.assert_array_equal(result.selected_indices, [1])


def test_geometric_ties_and_closed_contours_are_permutation_stable():
    uv = np.array([[0, 0], [0, 100], [100, 0], [100, 100], [50, 50], [50, 25], [50, 75]])
    ids = np.array([f"id-{i}" for i in range(len(uv))])
    expected = set(ids[_sample(uv, ids=ids).selected_indices])
    for seed in range(5):
        order = np.random.default_rng(seed).permutation(len(uv))
        selected = _sample(uv[order], ids=ids[order]).selected_indices
        assert set(ids[order][selected]) == expected


def test_farthest_coverage_has_no_arbitrary_point_limit():
    uv = np.c_[np.arange(4097)*24., np.zeros(4097)]
    result = _sample(uv)
    chosen = uv[result.selected_indices, 0]
    assert len(chosen) > 1000
    assert np.diff(chosen).max() <= 48.


def test_empty_and_singleton_groups():
    result = sample_source_pixels(np.empty((0, 2)), np.empty(0, dtype=np.int64),
                                  np.empty(0, dtype="U1"), np.empty(0, dtype="U1"))
    assert result.selected_indices.shape == result.reason_codes.shape == (0,)
    result = _sample([[10, 20]])
    np.testing.assert_array_equal(result.selected_indices, [0])
    assert result.reason_codes[0] == SamplingReason.EXTREME_SEED


@pytest.mark.parametrize("spacing", [0, -1, np.inf, np.nan, [24]])
def test_invalid_spacing_is_rejected(spacing):
    with pytest.raises(ValueError, match="spacing_px"):
        _sample([[0, 0]], spacing_px=spacing)


def test_invalid_pixels_timestamp_kind_and_shapes_are_rejected():
    with pytest.raises(ValueError, match="finite"):
        _sample([[np.nan, 0]])
    with pytest.raises(ValueError, match="integer"):
        _sample([[0, 0]], pts=np.array([1.5]))
    with pytest.raises(ValueError, match="kind"):
        _sample([[0, 0]], kind=np.array([2]))
    with pytest.raises(ValueError, match="shape"):
        _sample([[0, 0]], paths=np.array(["one", "two"]))


@pytest.fixture
def spatial_preparer(monkeypatch):
    programme = Path(__file__).resolve().parents[1] / "events/ankara_2026/programmes/okada_mei_sp"
    monkeypatch.syspath_prepend(str(programme))
    return importlib.import_module("prepare_spatial_refinement")


def _prepared_sample(preparer, uv, path_ids, path_indices, *, enabled, **options):
    groups, policy = preparer._sampling_group_map(path_ids, print_instance_coverage=enabled)
    result = _sample(uv, paths=groups[np.asarray(path_indices)], **options)
    return result, groups, policy


def test_print_instance_cover_is_unchanged_when_one_measured_contour_is_split_into_paths(spatial_preparer):
    theta = np.linspace(0., 2*np.pi, 240, endpoint=False)
    uv = np.c_[100.+60.*np.cos(theta), 100.+60.*np.sin(theta)]
    ids = np.array([f"same-source-{i:04}" for i in range(len(uv))])
    single, _, _ = _prepared_sample(spatial_preparer, uv,
        ["print_instance/Ad1::whole_contour"], np.zeros(len(uv), dtype=int), enabled=True, ids=ids)
    paths = [f"print_instance/Ad1::piece_{i}" for i in range(12)]
    path_indices = np.repeat(np.arange(12), 20)
    split, _, _ = _prepared_sample(spatial_preparer, uv, paths, path_indices, enabled=True, ids=ids)
    legacy, _, _ = _prepared_sample(spatial_preparer, uv, paths, path_indices, enabled=False, ids=ids)
    np.testing.assert_array_equal(split.selected_indices, single.selected_indices)
    np.testing.assert_array_equal(split.reason_codes, single.reason_codes)
    assert len(split.selected_indices) < len(legacy.selected_indices)


def test_print_instances_and_exposures_cannot_cover_each_other(spatial_preparer):
    uv = np.tile([[0., 0.], [12., 0.], [48., 0.]], (4, 1))
    pts = np.repeat([256, 256, 512, 512], 3)
    path_ids = ["print_instance/Ad1::left", "print_instance/Ad1::right",
                "print_instance/Ad2::left", "print_instance/Ad2::right"]
    path_indices = np.tile([0, 0, 1, 2, 2, 3], 2)
    result, groups, _ = _prepared_sample(spatial_preparer, uv, path_ids, path_indices,
                                         enabled=True, pts=pts)
    assert groups[0] == groups[1] and groups[2] == groups[3] and groups[0] != groups[2]
    np.testing.assert_array_equal(result.selected_indices, [0, 2, 3, 5, 6, 8, 9, 11])


def test_print_instance_option_preserves_exact_B1_B2_and_board_selection_and_source_fields(spatial_preparer):
    x = np.arange(0., 241., 3.)
    uv = np.r_[np.c_[x, np.sin(x)*.05], np.c_[x, np.full(len(x), 100.)],
               np.c_[x, np.full(len(x), 180.)], np.c_[x, np.full(len(x), 260.)]]
    path_ids = ["print_instance/Ad1::first", "print_instance/Ad1::second",
                "ice/B1", "ice/B2", "board/continuous/top"]
    paths = np.r_[np.arange(len(x)) % 2, np.full(len(x), 2),
                  np.full(len(x), 3), np.full(len(x), 4)].astype(np.int32)
    n = len(uv)
    source = dict(uv=uv, pts=np.full(n, 256, dtype=np.int64), path_indices=paths,
        observation_ids=np.array([f"original-{i:04}" for i in range(n)]),
        kind=np.ones(n, dtype=np.int64), stations=np.linspace(0., 1., n),
        source_rows=np.arange(n, dtype=np.int64))
    originals = {name: value.copy() for name, value in source.items()}
    original_path_ids = list(path_ids)
    for value in source.values():
        value.flags.writeable = False
    common = dict(pts=source["pts"], ids=source["observation_ids"], kind=source["kind"])
    legacy, default_groups, policy = _prepared_sample(spatial_preparer, uv, path_ids,
        source["path_indices"], enabled=False, **common)
    original = _sample(uv, paths=source["path_indices"], **common)
    np.testing.assert_array_equal(legacy.selected_indices, original.selected_indices)
    np.testing.assert_array_equal(legacy.reason_codes, original.reason_codes)
    np.testing.assert_array_equal(default_groups, np.arange(len(path_ids)))
    combined, _, new_policy = _prepared_sample(spatial_preparer, uv, path_ids,
        source["path_indices"], enabled=True, **common)
    for path in (2, 3, 4):
        old_rows = legacy.selected_indices[paths[legacy.selected_indices] == path]
        new_rows = combined.selected_indices[paths[combined.selected_indices] == path]
        np.testing.assert_array_equal(new_rows, old_rows)
    assert policy["sampling_grouping_strategy"] == "source_pts_native_path"
    assert new_policy["sampling_grouping_strategy"] == "source_pts_print_instance_else_native_path"
    assert new_policy["sampling_group_map_sha256"] != policy["sampling_group_map_sha256"]
    assert new_policy["sampling_establishes_direction_coverage"] is False
    assert new_policy["sampling_establishes_source_accuracy"] is False
    _, repeated_policy = spatial_preparer._sampling_group_map(path_ids, print_instance_coverage=True)
    assert repeated_policy["sampling_group_map_sha256"] == new_policy["sampling_group_map_sha256"]
    assert path_ids == original_path_ids
    for name, value in source.items():
        np.testing.assert_array_equal(value, originals[name])


def test_print_instance_cover_retains_every_fixed_observation(spatial_preparer):
    uv = [[0., 0.], [50., 0.], [50., 0.], [50., 0.], [100., 0.]]
    paths = [f"print_instance/Ad1::component_{i}" for i in range(4)]
    kind = np.array([1, 0, 0, 1, 1])
    result, _, _ = _prepared_sample(spatial_preparer, uv, paths, [0, 1, 2, 3, 0],
                                    enabled=True, kind=kind)
    assert {1, 2}.issubset(result.selected_indices)
    assert np.all(result.reason_codes[[1, 2]] == SamplingReason.FIXED_POINT)
    assert result.reason_codes[3] == SamplingReason.DUPLICATE_UV


def test_non_ad_paths_keep_original_index_groups_even_if_labels_repeat(spatial_preparer):
    paths = ["ice/B1", "ice/B1", "board/top", "board/top",
             "print_instance/Other::a", "print_instance/Other::b",
             "print_instance/Ad1::a", "print_instance/Ad1::b"]
    default, _ = spatial_preparer._sampling_group_map(paths)
    np.testing.assert_array_equal(default, np.arange(len(paths)))
    combined, _ = spatial_preparer._sampling_group_map(paths, print_instance_coverage=True)
    assert len(np.unique(combined[:6])) == 6
    assert combined[6] == combined[7]
    assert combined[6] not in combined[:6]
