import numpy as np

from rinkspace.camera_evidence import align_near_offset, mapping_segments, replay_segments


def test_observed_pts_alignment_does_not_invent_unmatched_tail():
    query = {"rgb": np.array([[0, 0], [10, 10], [255, 255]], np.uint8), "pts": np.array([0, 10, 20]), "time_base": "1/10"}
    target = {"rgb": np.array([[0, 0], [10, 10], [40, 40]], np.uint8), "pts": np.array([100, 110, 120]), "time_base": "1/10"}
    anchors = align_near_offset(query, target, 10, radius=.01)
    assert anchors[0]["target_pts"] == 100
    assert anchors[-1]["status"] == "unresolved_pixel_match"
    segments = mapping_segments(anchors)
    assert segments[0]["query_interval_s"] == [0, 1]
    assert segments[0]["frame_exact_between_anchors"] is False


def test_mapping_does_not_bridge_failed_or_discontinuous_anchors():
    anchors = [dict(query_time_s=i, target_time_s=i+100, offset_s=100,
                    status="pixel_duplicate_anchor" if i != 2 else "unresolved_pixel_match") for i in range(5)]
    segments = mapping_segments(anchors)
    assert [x["query_interval_s"] for x in segments] == [[0, 1], [3, 4]]


def test_replay_mapping_retains_slow_motion_and_splits_edit_jumps():
    rows = [dict(replay_time_s=100+i*.2, original_time_s=10+i*.15 if i<4 else 50+(i-4)*.15,
                 masked_thumbnail_mse=1) for i in range(8)]
    segments = replay_segments(rows)
    assert len(segments) == 2
    assert all(abs(s["original_seconds_per_replay_second"]-.75)<1e-6 for s in segments)
    assert all(s["independent_camera_evidence"] is False for s in segments)
