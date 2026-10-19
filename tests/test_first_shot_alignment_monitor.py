"""Raw all-point checks must be independent of cost and retain point coverage."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from calibration.program_tracking.first_shot_alignment_monitor import AlignmentMonitor


class Bundle:
    def __init__(self):
        self.state = SimpleNamespace(initial_q=np.zeros(1))
        self.frames = []
        for index in (1, 2, 3):
            self.frames.append(dict(index=index, native_pts=256000+256*index, source_point_count=4,
                rows=[dict(target=dict(kind="board_edge", role="top", part="upper_long_side"),
                    source_point_ids=["source a", "source b"], uv=np.array([[1, 2], [3, 4]], float),
                    sigma=np.array([.4, .6])),
                      dict(target=dict(kind="ad_corner", advertisement_id=2, role="left_top"),
                    source_point_ids=["source corner"], uv=np.array([[5, 6]], float), sigma=np.array([.5])),
                      dict(target=dict(kind="ice_curve", curve_id="approved ice circle"),
                    source_point_ids=["source curve"], uv=np.array([[7, 8]], float), sigma=np.array([.7]))]))
        self.offset = np.zeros(12)
        self.calls = 0

    def raw_errors(self, q):
        self.calls += 1
        return np.arange(1, 13, dtype=float) + float(q[0]) * self.offset

    def evaluate(self, *args, **kwargs):
        raise AssertionError("The raw monitor must never evaluate weighted cost")


def test_every_point_id_and_frame_is_saved_with_unweighted_group_statistics(tmp_path):
    bundle = Bundle()
    monitor = AlignmentMonitor(bundle, tmp_path)
    bundle.offset[:] = -.5
    result = monitor.evaluate([1], dict(iteration=1, accepted=True, cost=1e-30))
    report = json.loads(open(result["report"]["path"], encoding="utf8").read())
    inventory = json.loads((tmp_path/"source_inventory.json").read_text(encoding="utf8"))
    assert result["source_points"] == 12 and result["source_frames"] == 3
    assert inventory["point_count"] == 12
    assert all(frame["all_original_ids_covered"] and len(frame["source_point_ids"]) == 4
               for frame in inventory["frames"])
    assert report["global_comparison"]["current"]["mean"] == 6
    assert report["by_kind"]["board_edge"]["current"]["points"] == 6
    assert report["by_target"]["Ad2.left_top"]["current"]["points"] == 3
    assert report["weighted_cost_used_as_alignment"] is False
    assert report["independent_HELD"] is False
    assert report["source_points_dropped"] == 0 and result["stop_branch"] is False
    assert "cost" not in report["event"] and bundle.calls == 2


def test_small_floating_deterioration_requires_review_without_automatic_stopping(tmp_path):
    bundle = Bundle(); monitor = AlignmentMonitor(bundle, tmp_path)
    bundle.offset[:] = 1e-9
    result = monitor.evaluate([1], dict(iteration=1))
    assert result["global_mean_median_and_tail_all_worse"]
    assert result["requires_visual_review"] and not result["stop_branch"]
    assert result["worsened_beyond_own_source_support"] == 0


def test_consistent_deterioration_beyond_native_source_support_recommends_stop(tmp_path):
    bundle = Bundle(); monitor = AlignmentMonitor(bundle, tmp_path)
    bundle.offset[:] = 2
    result = monitor.evaluate([1], dict(iteration=1, accepted=True))
    assert result["stop_branch"] and result["worsened_beyond_own_source_support"] == 12
    assert result["consistently_worse_frames"] == 3
    report = json.loads(open(result["report"]["path"], encoding="utf8").read())
    assert report["stop_basis"] == "single_native_exposure_deteriorates_beyond_its_own_measured_source_support"


def test_explicit_visual_counterevidence_can_confirm_consistent_raw_deterioration(tmp_path):
    bundle = Bundle(); monitor = AlignmentMonitor(bundle, tmp_path)
    bundle.offset[:] = .05
    result = monitor.evaluate([1], dict(iteration=1, visual_alignment_worse=True))
    assert result["stop_branch"]


def test_a_physical_target_cannot_be_hidden_by_improved_frame_and_global_majorities(tmp_path):
    bundle = Bundle(); monitor = AlignmentMonitor(bundle, tmp_path)
    bundle.offset[:] = -.8; bundle.offset[2::4] = .8
    result = monitor.evaluate([1], dict(iteration=1))
    report = json.loads(open(result["report"]["path"], encoding="utf8").read())
    assert report["by_target"]["Ad2.left_top"]["mean_change_px"] > 0
    assert result["stop_branch"]
    assert result["physical_target_source_contradictions"] == ["Ad2.left_top"]
    assert report["by_target"]["Ad2.left_top"]["same_original_point_set_as_initial"]
    assert report["by_target"]["Ad2.left_top"]["all_original_source_point_IDs_covered"]
    inventory = json.loads((tmp_path/"target_inventory.json").read_text(encoding="utf8"))
    assert inventory["targets"]["Ad2.left_top"]["global_point_offsets"] == [2, 6, 10]
    assert report["stop_basis"] == "physical_target_deteriorates_beyond_its_own_measured_source_support"


def test_fifty_pixel_advertisement_edge_regression_stops_even_when_other_points_improve(tmp_path):
    bundle = Bundle()
    # One native advertisement edge per exposure is a small minority.  Its
    # source supports cannot be hidden by many initially worse printed pixels.
    for frame in bundle.frames:
        frame["rows"][1]["target"] = dict(kind="ad_edge", advertisement_id=2, role="left_vertical")
    baseline = np.tile([100., 100., 1., 100.], 3)
    offset = np.tile([-80., -80., 50., -80.], 3)
    bundle.raw_errors = lambda q: baseline + float(q[0]) * offset
    monitor = AlignmentMonitor(bundle, tmp_path)
    result = monitor.evaluate([1], dict(iteration=1))
    assert result["unweighted_native_pixel_summary"]["mean"] < result["initial_unweighted_native_pixel_summary"]["mean"]
    assert result["unweighted_native_pixel_summary"]["median"] < result["initial_unweighted_native_pixel_summary"]["median"]
    assert result["local_source_contradiction_frames"] == []
    assert result["physical_target_source_contradictions"] == ["Ad2.left_vertical"]
    assert result["stop_branch"] and result["requires_visual_review"]


def test_one_catastrophic_frame_cannot_be_hidden_by_an_improved_global_median(tmp_path):
    bundle = Bundle(); monitor = AlignmentMonitor(bundle, tmp_path)
    bundle.offset[:] = -4.8
    bundle.offset[:4] = 195
    result = monitor.evaluate([1], dict(iteration=22))
    assert result["unweighted_native_pixel_summary"]["median"] < result["initial_unweighted_native_pixel_summary"]["median"]
    assert not result["global_mean_median_and_tail_all_worse"]
    assert result["local_source_contradiction_frames"] == [1]
    assert result["requires_visual_review"] and result["stop_branch"]


def test_visual_counterevidence_can_stop_a_frame_despite_global_improvement(tmp_path):
    bundle = Bundle(); monitor = AlignmentMonitor(bundle, tmp_path)
    bundle.offset[:] = -.5
    result = monitor.evaluate([1], dict(iteration=2, visual_alignment_worse=True))
    assert result["stop_branch"]


@pytest.mark.parametrize("mutation", ["duplicate_point", "missing_point", "duplicate_frame", "missing_support"])
def test_point_inventory_cannot_be_reduced_or_repeated(tmp_path, mutation):
    bundle = Bundle()
    if mutation == "duplicate_point":
        bundle.frames[0]["rows"][0]["source_point_ids"][1] = "source a"
    elif mutation == "missing_point":
        bundle.frames[0]["source_point_count"] = 5
    elif mutation == "duplicate_frame":
        bundle.frames[1]["native_pts"] = bundle.frames[0]["native_pts"]
    else:
        bundle.frames[0]["rows"][0]["sigma"][0] = np.nan
    with pytest.raises(ValueError):
        AlignmentMonitor(bundle, tmp_path)


@pytest.mark.parametrize("invalid", [np.ones(11), np.r_[np.ones(11), np.nan], np.r_[np.ones(11), -1]])
def test_raw_errors_must_cover_all_finite_original_points(tmp_path, invalid):
    bundle = Bundle(); monitor = AlignmentMonitor(bundle, tmp_path)
    bundle.raw_errors = lambda q: invalid
    with pytest.raises(ValueError, match="every original finite point"):
        monitor.evaluate([1], dict(iteration=1))


def test_authorized_target_regression_remains_visible_without_stopping(tmp_path):
    bundle = Bundle()
    bundle.offset[:] = -.8
    bundle.offset[2::4] = .8
    monitor = AlignmentMonitor(bundle, tmp_path/"allow", allow_local_regression=True)
    result = monitor.evaluate([1], dict(iteration=1, accepted=True))
    report = json.loads(open(result["report"]["path"], encoding="utf8").read())
    assert result["physical_target_source_contradictions"] == ["Ad2.left_top"]
    assert result["local_source_contradiction_frames"] == []
    assert result["requires_visual_review"] and not result["stop_branch"]
    assert result["allow_local_regression"] and report["stop_basis"] is None
    assert report["by_target"]["Ad2.left_top"]["worsened_beyond_own_source_support"] == 3
    assert report["by_target"]["Ad2.left_top"]["all_original_source_point_IDs_covered"]
    np.testing.assert_array_equal(monitor.baseline, np.arange(1, 13, dtype=float))
    np.testing.assert_array_equal(monitor.source_support, np.tile([.4, .6, .5, .7], 3))
    assert result["source_points"] == 12 and report["source_points_dropped"] == 0
    # Default callers retain the original explicit stop policy.
    strict = AlignmentMonitor(bundle, tmp_path/"strict")
    assert strict.evaluate([1], dict(iteration=1))["stop_branch"]


def test_authorized_single_frame_regression_remains_visible_without_stopping(tmp_path):
    bundle = Bundle()
    bundle.offset[:] = -.8
    bundle.offset[:4] = 2.
    monitor = AlignmentMonitor(bundle, tmp_path/"allow", allow_local_regression=True)
    result = monitor.evaluate([1], dict(iteration=1))
    report = json.loads(open(result["report"]["path"], encoding="utf8").read())
    assert result["local_source_contradiction_frames"] == [1]
    assert result["physical_target_source_contradictions"] == []
    assert not result["global_mean_median_and_tail_all_worse"]
    assert result["requires_visual_review"] and not result["stop_branch"]
    assert report["frames"][0]["worsened_beyond_own_source_support"] == 4
    assert report["source_points_dropped"] == 0 and report["stop_basis"] is None
    np.testing.assert_array_equal(monitor.source_support, np.tile([.4, .6, .5, .7], 3))
    strict = AlignmentMonitor(bundle, tmp_path/"strict")
    assert strict.evaluate([1], dict(iteration=1))["stop_branch"]


@pytest.mark.parametrize("evidence", ["global", "visual"])
def test_authorized_local_continuation_still_stops_global_or_visual_failure(tmp_path, evidence):
    bundle = Bundle()
    bundle.offset[:] = 2. if evidence == "global" else -.5
    monitor = AlignmentMonitor(bundle, tmp_path, allow_local_regression=True)
    event = dict(iteration=1, visual_alignment_worse=evidence == "visual")
    result = monitor.evaluate([1], event)
    report = json.loads(open(result["report"]["path"], encoding="utf8").read())
    assert result["stop_branch"] and result["requires_visual_review"]
    assert result["source_points"] == 12 and report["source_points_dropped"] == 0
    if evidence == "global":
        assert result["global_mean_median_and_tail_all_worse"]
        assert result["worsened_beyond_own_source_support"] == 12
        assert report["stop_basis"] == "raw_global_and_frame_deterioration_beyond_measured_source_support"
    else:
        assert not result["global_mean_median_and_tail_all_worse"]
        assert report["stop_basis"] == "explicit_visual_counterevidence"
