"""Load reviewed far-rail *source* UV for an opt-in complete-main graph.

The source identity rule is complete over the original four main shots, while
individual unknown frames remain unknown. This loader does not evaluate a
camera, create observations, or alter the frozen detector/source packets.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from cache_far_rail_adjacent_links import CandidateCache, _sha


IDENTITY = "board_upper_blue_cap"
ALLOWED_STATUS = {1: "source_visual_review",
                  2: "bidirectional_source_match_to_reviewed_seeds"}


def load_reviewed_main_rail(data: dict, selection_manifest_path: Path, *,
                            expected_manifest_sha256: str) -> tuple[list[dict], dict]:
    """Return exact measured UV rows and a strict provenance/coverage summary.

    Each row: native_pts, observed_uv, vertical_transition_interval_px,
    physical_identity, identity_status, source_sha256, source_pixel_sha256,
    global_uv_indices, identity_evidence. Unselected candidate UV are absent,
    with explicit unknown frame counts in metadata.
    """
    selection_manifest_path = Path(selection_manifest_path)
    actual_manifest_hash = _sha(selection_manifest_path)
    if actual_manifest_hash != expected_manifest_sha256:
        raise ValueError("Reviewed rail manifest SHA differs from explicit opt-in pin")
    report = json.loads(selection_manifest_path.read_text(encoding="utf-8"))
    if (report.get("schema") != "rinkspace.far_rail_reviewed_source_selection.v1"
            or report.get("source_identity_review_complete") is not True):
        raise ValueError("Reviewed rail selection has no complete source identity audit")
    if data.get("camera_id") != "main_long_side":
        raise ValueError("Reviewed main rail only applies to main_long_side")
    expected_source = data.get("diag", {}).get("source_sha256")
    captured_source = data.get("captured", {}).get("source_hash")
    if (expected_source != report["source_video_sha256"]
            or captured_source != report["source_video_sha256"]):
        raise ValueError("Reviewed rail source differs from prepared graph source")
    packet_path = Path(report["packet"])
    if _sha(packet_path) != report["packet_sha256"]:
        raise ValueError("Reviewed source selection packet SHA differs from manifest")
    candidate_path = Path(report["candidate_packet_path"])
    if _sha(candidate_path) != report["candidate_packet_sha256"]:
        raise ValueError("Original full-main candidate UV packet changed")
    cache = CandidateCache(candidate_path.parent)
    if (cache.packet_sha256 != report["candidate_packet_sha256"]
            or cache.manifest["source_video_sha256"] != report["source_video_sha256"]):
        raise ValueError("Reviewed selection source provenance differs from candidate cache")
    link_path = Path(report["adjacent_links_path"])
    if _sha(link_path) != report["adjacent_links_sha256"]:
        raise ValueError("Native source bidirectional-link packet changed")
    seed_path = Path(report["pointwise_reviewed_seeds"])
    if _sha(seed_path) != report["pointwise_reviewed_seeds_sha256"]:
        raise ValueError("Pointwise source review seeds changed")
    with np.load(packet_path, allow_pickle=False) as selection:
        arrays = {key: selection[key].copy() for key in selection.files}
    if (not np.array_equal(arrays["frame_pts"], cache.frame_pts)
            or not np.array_equal(arrays["frame_shot_index"], cache.frame_shot)
            or arrays["candidate_packet_sha256"].item() != cache.packet_sha256
            or arrays["adjacent_links_sha256"].item() != report["adjacent_links_sha256"]
            or arrays["pointwise_reviewed_seeds_sha256"].item()
            != report["pointwise_reviewed_seeds_sha256"]):
        raise ValueError("Reviewed selection arrays do not match frozen raw source")
    programme_pts = set(map(int, data["pts"]))
    if programme_pts != set(map(int, cache.frame_pts)):
        raise ValueError("Complete main graph source PTS differ from reviewed four shots")
    status = arrays["frame_identity_status_code"]
    offsets = arrays["frame_selected_offsets"]
    chosen = arrays["selected_global_uv_indices"]
    if (len(status) != len(cache.frame_pts) or len(offsets) != len(status) + 1
            or offsets[0] != 0 or offsets[-1] != len(chosen)
            or np.any(np.diff(offsets) < 0)
            or np.any(~np.isin(status, (0, 1, 2)))):
        raise ValueError("Reviewed source selection offsets/status are invalid")
    rows = []
    for index, native_pts in enumerate(cache.frame_pts):
        first, last = map(int, offsets[index:index+2])
        selected = chosen[first:last]
        if status[index] == 0:
            if len(selected):
                raise ValueError(f"Unknown PTS {native_pts} contains selected source UV")
            continue
        if len(selected) < 2 or len(np.unique(selected)) != len(selected):
            raise ValueError(f"Qualified source UV under-supported or duplicate at PTS {native_pts}")
        comp_first, comp_last = map(int, cache.frame_offset[index:index+2])
        uv_first = int(cache.uv_offset[comp_first])
        uv_last = int(cache.uv_offset[comp_last])
        if np.any(selected < uv_first) or np.any(selected >= uv_last):
            raise ValueError(f"Selection crosses original frame UV bounds at {native_pts}")
        identity_status = ALLOWED_STATUS[int(status[index])]
        source_pixel_sha = str(cache.arrays["frame_source_pixel_sha256"][index])
        evidence = dict(
            source_only=True,
            observed_uv_source="frozen_original_video_blue_colour_transition",
            selection_manifest_sha256=actual_manifest_hash,
            candidate_packet_sha256=cache.packet_sha256,
            adjacent_source_links_sha256=report["adjacent_links_sha256"],
            pointwise_reviewed_seeds_sha256=report["pointwise_reviewed_seeds_sha256"],
            direct_reviewed_seed_points=int(arrays["frame_direct_reviewed_seed_points"][index]),
            left_reviewed_seed_pts=int(arrays["frame_left_reviewed_seed_pts"][index]),
            right_reviewed_seed_pts=int(arrays["frame_right_reviewed_seed_pts"][index]),
            forward_matched_source_uv=int(arrays["frame_forward_matched_source_uv"][index]),
            backward_matched_source_uv=int(arrays["frame_backward_matched_source_uv"][index]),
            colour_interval_semantics="q10/q90 transition extent, not a confidence interval",
            no_projection_or_flow_created_uv=True,
        )
        rows.append(dict(
            native_pts=int(native_pts),
            observed_uv=cache.arrays["observed_uv"][selected].copy(),
            vertical_transition_interval_px=(
                cache.arrays["vertical_transition_interval_px"][selected].copy()),
            physical_identity=IDENTITY,
            identity_status=identity_status,
            source_sha256=cache.manifest["source_video_sha256"],
            source_pixel_sha256=source_pixel_sha,
            global_uv_indices=selected.astype(np.int64, copy=True),
            identity_evidence=evidence,
        ))
    metadata = dict(
        source_identity_review_complete=True,
        source_accuracy_verified=False,
        selection_manifest=str(selection_manifest_path),
        selection_manifest_sha256=actual_manifest_hash,
        selection_packet_sha256=report["packet_sha256"],
        candidate_packet_sha256=cache.packet_sha256,
        adjacent_source_links_sha256=report["adjacent_links_sha256"],
        pointwise_reviewed_seeds_sha256=report["pointwise_reviewed_seeds_sha256"],
        source_video_sha256=cache.manifest["source_video_sha256"],
        complete_main_native_frames=len(cache.frame_pts),
        selected_exposures=len(rows),
        unknown_exposures=int(np.count_nonzero(status == 0)),
        selected_raw_uv=sum(len(row["observed_uv"]) for row in rows),
        source_visual_review_exposures=int(np.count_nonzero(status == 1)),
        bidirectional_source_match_exposures=int(np.count_nonzero(status == 2)),
        qualification=("Source visual seeds and colour-contiguous real UV, with "
                       "adjacent LK+FB links and both-side reviewed-seed agreement. "
                       "Unknown frames and unselected UV are not observations."),
    )
    if metadata["selected_raw_uv"] != report["counts"]["selected_uv"]:
        raise ValueError("Reviewed source UV count differs from pinned report")
    return rows, metadata
