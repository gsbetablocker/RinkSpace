"""Read-only input assembly for a depth-aware, same-camera programme graph.

This module constructs no solver graph and runs no optimization.  The original
v13 frame states and physical 11-state callbacks are kept; the four legacy
temporal pair loops are replaced by actual native UV on the existing board.
Board identity outside reviewed source support remains conditional, and absent
adjacent pairs remain explicitly unknown.
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import numpy as np

import conditioned_camera_comparison as driver
import low_end_board_edges_interval_diagnostic as low_end_edges
import low_end_board_edges_interval_pivot as reviewed_intervals
import low_end_board_edges_qualified_only as qualified_only
import low_end_calibration_transfer_profile as transfer
import low_end_confirmed_exposure_aliases as exposure_evidence
from source_supported_motion import load_supported_motion
from surface_pair import SurfacePairConstraint


SOURCE_SHOTS = Path(__file__).with_name("source_shots.json")
V12_REVIEW_VIDEO = Path(__file__).with_name("result") / "okada_sp_candidate_grid.mp4"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _selected_shots(camera_id: str) -> tuple[int, ...]:
    _require(driver.digest(SOURCE_SHOTS) == exposure_evidence.SOURCE_SHOTS_SHA256,
             "Reviewed source-shot timeline changed")
    source = json.loads(SOURCE_SHOTS.read_text(encoding="utf8"))
    _require(source["source_sha256_from_timeline"] == exposure_evidence.SOURCE_SHA256
             and source["native_frame_count"] == 10140
             and len(source["shots"]) == 10,
             "Source-shot identity or full programme coverage changed")
    selected = tuple(index for index, shot in enumerate(source["shots"])
                     if shot["camera_id"] == camera_id)
    _require(bool(selected), f"Unknown programme camera: {camera_id}")
    return selected


def _restore_dynamic_keyframe_links(data: dict) -> int:
    """Keep original long-span UV factors whose ice/board plane hit moves with C."""
    _require(data["config"].get("pair_model") == "surface",
             "Keyframe links require the original dynamic surface pair model")
    originals = Counter()
    for shot in data["shots"]:
        for a, b, pair in shot["links"]:
            _require(isinstance(pair, SurfacePairConstraint)
                     and pair.fixed_mount_rays is False,
                     "A selected keyframe link is a fixed-centre ray factor")
            originals[int(a), int(b)] += 1
    restored = [spec for spec in data["captured"]["specs"]
                if spec["file"] == "joint_tracking_refine.py"
                and spec["loop"] == "links"
                and set(spec["keys"]) <= data["pts"]]
    _require(Counter(tuple(map(int, spec["keys"])) for spec in restored) == originals
             and data["excluded"].get("replaced_links", 0) == len(restored),
             "Captured dynamic keyframe link count or source PTS changed")
    data["retained"].extend(restored)
    data["excluded"].pop("replaced_links", None)
    return len(restored)


def _add_reviewed_low_end_inputs(data: dict) -> tuple[dict, dict]:
    """Reuse the frozen source-only UV and edge decisions, without solved poses."""
    _require(data["shot_indices"] == (3, 5, 8),
             "Reviewed low_end inputs require all three original shots")
    data["supported_motion"], data["supported_motion_metadata"] = (
        load_supported_motion(data, transfer.SUPPORT, transfer.seed_driver.MANIFEST,
                              transfer.SUPPORT_SHA256))
    extra, data["supported_motion_fallback_metadata"] = load_supported_motion(
        data, transfer.FALLBACK, transfer.seed_driver.MANIFEST,
        transfer.FALLBACK_SHA256)
    data["supported_motion"].extend(extra)
    _require(data["supported_motion_metadata"]["pair_count"] == 248
             and data["supported_motion_metadata"]["point_count"] == 3761
             and data["supported_motion_fallback_metadata"]["pair_count"] == 6
             and data["supported_motion_fallback_metadata"]["point_count"] == 54,
             "Reviewed shot05 original-UV support differs")

    phase_path = reviewed_intervals.FIXED_DIR / "phase_00.json"
    _require(driver.digest(phase_path) == reviewed_intervals.FIXED_PHASE_SHA256,
             "Reviewed low_end edge-source manifest changed")
    phase = json.loads(phase_path.read_text(encoding="utf8"))
    _require(phase["source_sha256"] == data["captured"]["source_hash"]
             and phase["scene_sha256"] == data["diag"]["model_sha256"],
             "Reviewed edge source/scene binding changed")
    if "shot_indices" in phase:
        _require(phase["shot_indices"] == [3, 5, 8],
                 "Reviewed edge source shot group changed")
    edge_data, shot_by_pts = low_end_edges.source_edges(data)
    indices = low_end_edges.factor_indices(data, edge_data, shot_by_pts)
    packet_sha = {(number, edge_id): phase["support"][
        f"shot{number}_{low_end_edges.EDGE_NAMES[edge_id]}"]["sha256"]
        for number in (5, 8) for edge_id in range(2)}
    supports, manifest = low_end_edges.load_supports(
        data, edge_data, packet_sha, phase["support_manifest"]["sha256"])
    _require(manifest == phase["support_manifest"],
             "Reviewed low_end edge source manifest differs")
    interval_data = low_end_edges.changed_data(
        data, edge_data, shot_by_pts, indices, supports)
    qualified = qualified_only.qualified_data(
        data, interval_data, edge_data, shot_by_pts, indices, supports)
    coverage = qualified_only.source_coverage(edge_data, supports)
    return qualified, dict(
        policy="source_qualified_board_top_and_contact_only; unknown_rows_zeroed",
        replacement_factors=len(indices), coverage=coverage,
        support_manifest_path=str(low_end_edges.ADAPTIVE_REPORT.resolve()),
        support_manifest_sha256=phase["support_manifest"]["sha256"],
        reviewed_phase_sha256=reviewed_intervals.FIXED_PHASE_SHA256,
        source_uv_changed=False)


def prepare_inputs(output: Path, camera_id: str = "low_end") -> tuple[dict, list, dict]:
    """Return conditioned physical callbacks, dynamic board pairs, and limits.

    ``output`` is used only by the upstream capture/preflight constructor; it
    must obey that constructor's outside-repository cache rule. No update,
    batch polish, camera export, or source-pixel selection is run here.
    """
    shots = _selected_shots(camera_id)
    data = driver.prepare(
        transfer.seed_driver.CONFIG, transfer.seed_driver.CANDIDATE,
        transfer.seed_driver.MANIFEST, shots, Path(output),
        complete_camera_group=True)
    _require(data["camera_id"] == camera_id and data["shot_indices"] == shots,
             "Prepared physical camera group differs from source timeline")
    restored_keyframes = _restore_dynamic_keyframe_links(data)
    edge_metadata = None
    if camera_id == "low_end":
        data, edge_metadata = _add_reviewed_low_end_inputs(data)

    has_auxiliary = any(data["manifest"]["coverage_by_shot"][str(index)][
        "auxiliary_pair_count"] for index in shots)
    if has_auxiliary:
        from continuous_camera_auxiliary import build_dynamic_pairs
        pairs, counts, counts_by_shot, components, validity = build_dynamic_pairs(data)
    else:
        pairs, counts, counts_by_shot, components, validity = driver.native_pairs(data)
    _require(validity["invalid_pair_count"] == 0,
             f"Dynamic board input has invalid source/target geometry: {validity['reason_counts']}")
    adjacent = [(int(a), int(b)) for a, b, _ in pairs if int(b)-int(a) == 256]
    observed = set(adjacent)
    _require(len(adjacent) == len(observed),
             "Dynamic board input duplicates an adjacent source PTS pair")
    per_shot = {}
    for shot in data["shots"]:
        index = int(shot["index"])
        stamps = tuple(map(int, shot["pts"]))
        expected = [(a, b) for a, b in zip(stamps[:-1], stamps[1:])]
        unknown = [list(pair) for pair in expected if pair not in observed]
        per_shot[str(index)] = dict(
            frames=len(stamps), **counts_by_shot[str(index)],
            unknown_adjacent_count=len(unknown), unknown_adjacent_pts=unknown,
            native_components=len(components[str(index)]))
    _require(len(observed) + sum(row["unknown_adjacent_count"]
                                for row in per_shot.values())
             == sum(len(shot["pts"])-1 for shot in data["shots"]),
             "Native board groups duplicate or cross a camera cut")
    metadata = dict(
        camera_id=camera_id, shot_indices=list(shots), frames=len(data["pts"]),
        state_source="frozen_v13_physical_11state", source_sha256=data["diag"]["source_sha256"],
        scene_sha256=data["diag"]["model_sha256"],
        frozen_manifest_path=str(transfer.seed_driver.MANIFEST.resolve()),
        frozen_manifest_sha256=driver.digest(transfer.seed_driver.MANIFEST),
        frozen_packet_path=str(data["packet_path"].resolve()),
        frozen_packet_sha256=driver.digest(data["packet_path"]),
        dynamic_board_factor="NativeBoardPair: actual xy0/xy1, current source-ray board hit",
        native_counts=counts, per_shot=per_shot, native_input_validity=validity,
        excluded_original_temporal_factors=data["excluded"],
        restored_dynamic_keyframe_surface_factor_count=restored_keyframes,
        retained_callback_mode="physical_11state; no derived fixed-C wrapper",
        scene_and_other_cameras="conditioned constants at v13; not a whole-programme joint solve",
        missing_native_policy="unknown observation; other data and priors may estimate states",
        source_qualification=("reviewed v2 shot05 support plus frozen conditional board packet"
                              if camera_id == "low_end" else
                              "frozen v13 region_hint and geometry association only; independent board texture qualification absent"),
        edge_source_support=edge_metadata,
        v12_baseline_video=str(V12_REVIEW_VIDEO.resolve()),
        v12_baseline_role="independent visual comparison, never a fit input",
        solver_started=False)
    data["continuous_input_metadata"] = metadata
    return data, pairs, metadata


def load_exposure_aliases(data: dict) -> tuple[dict[int, int], dict]:
    """Read confirmed full-programme repeated exposures for one camera group."""
    groups_path = exposure_evidence.CADENCE / "exposure_groups.json"
    _require(driver.digest(groups_path) == exposure_evidence.GROUPS_SHA256
             and driver.digest(SOURCE_SHOTS) == exposure_evidence.SOURCE_SHOTS_SHA256,
             "Frozen full-programme exposure or cut record changed")
    source = json.loads(SOURCE_SHOTS.read_text(encoding="utf8"))
    packet = json.loads(groups_path.read_text(encoding="utf8"))
    _require(packet["schema"] == "rinkspace.official50-confirmed-exposure-groups.v1"
             and packet["source_binding"]["full60"]["sha256"] ==
                 data["captured"]["source_hash"]
             and packet["validation"]["all_native_pts_exactly_once"] is True
             and packet["validation"]["all_groups_single_shot"] is True
             and packet["validation"]["all_nine_cut_pairs_different_groups"] is True
             and packet["validation"]["confirmed_pair_count"] == 1620
             and source["source_sha256_from_timeline"] == data["captured"]["source_hash"],
             "Confirmed exposure packet belongs to another source or cut timeline")
    shot_at = {}
    for index, shot in enumerate(source["shots"]):
        for pts in range(int(shot["start_pts"]), int(shot["end_pts"]), 256):
            _require(pts not in shot_at, "Source shots overlap")
            shot_at[pts] = index
    _require(len(shot_at) == source["native_frame_count"] == 10140,
             "Source-shot frames are incomplete")
    selected = set(data["shot_indices"])
    aliases: dict[int, int] = {}
    all_seen = set()
    repeats = Counter()
    for group in packet["groups"]:
        stamps = tuple(map(int, group["full_pts"]))
        index = int(group["shot_index"])
        _require(bool(stamps) and stamps == tuple(sorted(stamps))
                 and all(shot_at.get(pts) == index and pts not in all_seen
                         for pts in stamps),
                 "Exposure group repeats a PTS or crosses a camera cut")
        all_seen.update(stamps)
        if len(stamps) > 1:
            official = group["confirmed_official_pts"]
            evidence = group["confirmed_pair_evidence"]
            _require(len(stamps) in (2, 3)
                     and group["criterion"] == "confirmed_same_official_exposure_pair_chain"
                     and official is not None
                     and group["frame_accepted_official_pts"] == [official]*len(stamps)
                     and group["frame_match_status"] == ["unique_pixel_match"]*len(stamps)
                     and len(evidence) == len(stamps)-1,
                     "Repeated exposure lacks confirmed original-pixel identity")
            for (a, b), row in zip(zip(stamps[:-1], stamps[1:]), evidence):
                _require(b-a == 256 and int(row["previous_full_pts"]) == a
                         and int(row["full_pts"]) == b
                         and int(row["official_pts"]) == int(official)
                         and row["criterion"] ==
                         "accepted_official_pts_equal_and_adjacent_within_shot_and_full_pair_roi_gray_mae_le_0.45"
                         and float(row["full_pair_roi_gray_mae"]) <= .45,
                         "Same-exposure link fails reviewed native-pixel rule")
        if index in selected:
            for pts in stamps:
                aliases[pts] = stamps[0]
            repeats[index] += len(stamps)-1
    _require(all_seen == set(shot_at) and set(aliases) == data["pts"]
             and all(rep in aliases and aliases[rep] == rep
                     and shot_at[pts] == shot_at[rep]
                     for pts, rep in aliases.items()),
             "Exposure aliases omit a selected native PTS or cross shots")
    metadata = dict(
        path=str(groups_path.resolve()), sha256=exposure_evidence.GROUPS_SHA256,
        source_shots_sha256=exposure_evidence.SOURCE_SHOTS_SHA256,
        source_sha256=data["captured"]["source_hash"],
        selected_frames=len(aliases), representative_frames=len(set(aliases.values())),
        confirmed_duplicate_links_by_shot={str(k): int(v) for k, v in sorted(repeats.items())},
        policy="Only confirmed exact same-shot exposure groups; ambiguous frames stay distinct",
        factor_rule="Map duplicate keys once per factor and sum their Jacobian blocks",
        solver_started=False)
    return aliases, metadata
