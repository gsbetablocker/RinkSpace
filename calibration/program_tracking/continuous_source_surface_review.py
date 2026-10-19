"""Apply a frozen original-pixel non-board review to secondary board inputs.

Filtering happens on original UV, before quality scaling, Huber weighting, and
correlated-pair whitening.  The saved source packet and UV stay untouched.
Keyframe/auxiliary/source-edge families are checked against the same mask;
this frozen review has zero hits in those families and aborts if that changes.
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import numpy as np

import conditioned_camera_comparison as driver
from native_board_pair import NativeBoardPair
import secondary_foreground_source_mask as source_mask
from secondary_foreground_source_mask import classify_uv, digest


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def _classify_endpoints(a: int, b: int, xy0: np.ndarray, xy1: np.ndarray,
                        boundaries: dict[int, dict]) -> tuple[np.ndarray, np.ndarray]:
    return classify_uv(a, xy0, boundaries), classify_uv(b, xy1, boundaries)


def _verify_other_families(data: dict, boundaries: dict[int, dict],
                           inventory: dict) -> dict:
    """Do not let excluded foreground return via another observation family."""
    by_family = {}
    links = inventory["keyframe_surface"]
    expected = links["counts"]
    _require(not links["hits"] and expected.get("rows_excluded", 0) == 0,
             "Reviewed keyframe surface links need raw-UV factor reconstruction")
    total_pairs = total_rows = 0
    for shot in data["shots"]:
        for a, b, pair in shot["links"]:
            total_pairs += 1
            total_rows += len(pair.xy0)
            left, right = _classify_endpoints(int(a), int(b), pair.xy0, pair.xy1,
                                              boundaries)
            _require(not np.any(left | right),
                     f"Keyframe surface link overlaps excluded foreground: {a}->{b}")
    _require(total_pairs == expected["pairs_total"]
             and total_rows == expected["rows_total"]
             and data["continuous_input_metadata"]["restored_dynamic_keyframe_surface_factor_count"]
                 == total_pairs,
             "Original restored dynamic keyframe family differs from source review")
    by_family["keyframe_surface"] = dict(pairs=total_pairs, rows=total_rows,
                                          excluded_rows=0, treatment="unchanged")

    edges = inventory["source_edge"]
    _require(not edges["hits"] and edges["counts"].get("rows_excluded", 0) == 0,
             "Reviewed source-edge family needs raw-UV factor reconstruction")
    config_shots = {int(row["shot_index"]): row for row in data["config"]["shots"]}
    total_rows = 0
    for shot in data["shots"]:
        number = int(shot["index"])
        path = Path(config_shots[number]["source_edges"])
        with np.load(path, allow_pickle=False) as z:
            _require(str(z["source_sha256"]) == data["captured"]["source_hash"]
                     and np.array_equal(z["pts"], shot["pts"]),
                     f"Source-edge archive source/PTS differs: shot{number:02d}")
            stamps = z["pts"].copy()
            offsets = z["offsets"].copy()
            observed = z["observed_uv"].copy()
        _require(int(offsets[0]) == 0 and int(offsets[-1]) == len(observed),
                 f"Source-edge offsets changed: shot{number:02d}")
        for frame_index, stamp in enumerate(stamps):
            lo, hi = map(int, offsets[frame_index:frame_index+2])
            total_rows += hi-lo
            _require(not np.any(classify_uv(int(stamp), observed[lo:hi], boundaries)),
                     f"Source-edge UV overlaps excluded foreground: shot{number:02d}, PTS{stamp}")
    _require(total_rows == edges["counts"]["rows_total"],
             "Original source-edge row count differs from source review")
    by_family["source_edge"] = dict(rows=total_rows, excluded_rows=0,
                                      treatment="unchanged")
    return by_family


def _subset_pair(pair: NativeBoardPair, keep: np.ndarray) -> NativeBoardPair:
    _require(type(pair) is NativeBoardPair,
             "Filter original NativeBoardPair before Brown factor wrapping")
    _require(np.count_nonzero(keep) >= 8,
             "Source review left fewer than eight actual board correspondences")
    subset = NativeBoardPair(pair.xy0[keep], pair.xy1[keep], pair.scene,
                             pair.camera0_template, pair.camera1_template,
                             allowed_panels=pair.allowed_panels,
                             seam_fraction_margin=pair.seam_fraction_margin)
    for name, value in vars(pair).items():
        if name.startswith("rinkspace_"):
            setattr(subset, name,
                    value.copy() if isinstance(value, np.ndarray) else value)
    return subset


def apply_source_surface_review(data: dict, pairs: list, manifest_path: Path
                                ) -> tuple[dict, list, dict]:
    """Return (data, board_pairs, metadata) for an opt-in source-corrected fit."""
    path = Path(manifest_path)
    review = json.loads(path.read_text(encoding="utf8"))
    _require(review["schema"] == "rinkspace.secondary-foreground-source-mask.v1"
             and data["camera_id"] == review["selected_camera_id"]
             == "secondary_long_side"
             and list(data["shot_indices"]) == review["selected_shot_indices"]
             and review["source_video_sha256"] == data["captured"]["source_hash"]
             and review["frozen_packet_sha256"] == driver.digest(data["packet_path"])
             and review["source_shots_sha256"] == driver.digest(
                 Path(review["source_shots_path"]))
             and Path(review["detector_module_path"]).resolve() ==
                 Path(source_mask.__file__).resolve()
             and review["detector_module_sha256"] ==
                 digest(Path(source_mask.__file__)),
             "Source surface review camera/timeline/frozen packet binding differs")
    _require(review["boundary_metadata"]["shot06_native_frames"] == 582
             and review["boundary_metadata"]["observed_diagonal_frames"] == 71
             and review["boundary_metadata"]["exclusion_policy"] ==
                 "both definite interior and uncertain boundary excluded as known board",
             "Source mask coverage or interior/boundary policy changed")
    boundaries = {int(k): v for k, v in review["boundary_by_pts"].items()}
    _require(set(boundaries) == set(range(230409984, 230428160, 256)),
             "Source mask is missing a reviewed shot06 native frame")
    for case in review["boundary_metadata"]["identity_checks"]:
        actual = int(classify_uv(int(case["pts"]),
                                 np.asarray([case["xy"]]), boundaries)[0])
        _require(actual == int(case["mask_class"]),
                 f"Original-pixel identity review changed: {case}")

    inventory = review["input_families"]
    other_families = _verify_other_families(data, boundaries, inventory)
    frozen = inventory["native_auxiliary"]
    native_counts = frozen["counts"]
    _require(native_counts.get("auxiliary_rows_excluded", 0) == 0,
             "Auxiliary board factors overlap reviewed foreground")
    _require(all(row["family"] == "native" for row in frozen["hits"]),
             "Auxiliary board rows require a new reviewed raw-UV adapter")
    packet = data["packet"]
    selected = set(data["shot_indices"])
    pair_lookup = {}
    for i, (a, b) in enumerate(packet["pair_pts"]):
        if int(packet["shot_index"][i]) not in selected:
            continue
        key = (int(a), int(b))
        _require(key not in pair_lookup,
                 f"Strict board packet repeats source pair PTS {key}")
        pair_lookup[key] = i
    _require(len(pair_lookup) == len(pairs),
             "Prepared dynamic factors do not cover each selected strict pair")
    hits_by_pair: dict[int, set[int]] = {}
    for hit in frozen["hits"]:
        i = int(hit["packet_pair_index"])
        j = int(hit["packet_point_global_index"])
        _require(i in pair_lookup.values(), "Excluded packet pair is outside selected rig")
        lo, hi = map(int, packet["offsets"][i:i+2])
        source_file = data["manifest"]["source_files"][int(packet["source_file_id"][i])]
        _require(lo <= j < hi and int(hit["packet_point_local_index"]) == j-lo
                 and int(packet["shot_index"][i]) == int(hit["shot_index"])
                 and int(packet["source_file_id"][i]) == int(hit["source_file_id"])
                 and int(packet["source_pair_index"][i]) == int(hit["source_pair_index"])
                 and source_file["kind"] == "native"
                 and source_file["sha256"] == hit["source_file_sha256"]
                 and int(packet["source_point_index"][j]) == hit["source_point_index"]
                 and int(packet["source_global_point_index"][j]) ==
                     hit["source_global_point_index"]
                 and packet["xy0"][j].tolist() == hit["xy0"]
                 and packet["xy1"][j].tolist() == hit["xy1"]
                 and packet["pair_pts"][i].tolist() ==
                     [hit["source_pts"], hit["target_pts"]],
                 "Excluded point lost its frozen original-source index or UV")
        left, right = _classify_endpoints(
            int(hit["source_pts"]), int(hit["target_pts"]),
            packet["xy0"][j:j+1], packet["xy1"][j:j+1], boundaries)
        _require(int(left[0]) == hit["source_mask_class"]
                 and int(right[0]) == hit["target_mask_class"]
                 and bool(left[0] or right[0]),
                 "Source/target foreground classification differs from manifest")
        seen = hits_by_pair.setdefault(i, set())
        _require(j not in seen, "Foreground manifest repeats an original point")
        seen.add(j)

    adapted = []
    counts = Counter()
    touched = []
    for a, b, pair in pairs:
        key = (int(a), int(b))
        _require(key in pair_lookup, f"Dynamic pair absent from frozen packet: {key}")
        i = pair_lookup.pop(key)
        lo, hi = map(int, packet["offsets"][i:i+2])
        _require(len(pair.xy0) == hi-lo
                 and np.array_equal(pair.xy0, packet["xy0"][lo:hi].astype(np.float64))
                 and np.array_equal(pair.xy1, packet["xy1"][lo:hi].astype(np.float64)),
                 f"Prepared pair UV/order differs from strict source packet: {key}")
        left, right = _classify_endpoints(int(a), int(b), pair.xy0, pair.xy1,
                                          boundaries)
        excluded_local = set(np.flatnonzero((left > 0) | (right > 0)).tolist())
        recorded_local = {j-lo for j in hits_by_pair.get(i, set())}
        _require(excluded_local == recorded_local,
                 f"Foreground manifest incompletely covers pair {key}")
        family = data["manifest"]["source_files"][int(packet["source_file_id"][i])]["kind"]
        counts[family+"_rows_original"] += len(pair.xy0)
        counts[family+"_rows_excluded"] += len(excluded_local)
        if excluded_local:
            keep = np.ones(len(pair.xy0), bool)
            keep[list(excluded_local)] = False
            pair = _subset_pair(pair, keep)
            _require(np.array_equal(pair.xy0,
                                    packet["xy0"][lo:hi][keep].astype(np.float64))
                     and np.array_equal(pair.xy1,
                                        packet["xy1"][lo:hi][keep].astype(np.float64)),
                     "Review changed a retained source/target UV")
            touched.append(dict(packet_pair_index=i, source_pts=int(a),
                                target_pts=int(b), original_rows=hi-lo,
                                excluded_rows=len(excluded_local),
                                remaining_rows=len(pair.xy0)))
        counts[family+"_rows_retained"] += len(pair.xy0)
        adapted.append((a, b, pair))
    _require(not pair_lookup and len(adapted) == len(pairs)
             and counts["native_rows_original"] == native_counts["native_rows_total"]
             and counts["auxiliary_rows_original"] == native_counts["auxiliary_rows_total"]
             and counts["native_rows_excluded"] == native_counts["native_rows_excluded"]
             and counts["auxiliary_rows_excluded"] == 0,
             "Source review totals differ from the original board factor input")
    metadata = dict(
        manifest_path=str(path.resolve()), manifest_sha256=digest(path),
        source_sha256=review["source_video_sha256"],
        frozen_packet_sha256=review["frozen_packet_sha256"],
        mask_definition=review["boundary_metadata"],
        raw_uv_changed=False, raw_uv_subset_before_quality_and_whitening=True,
        counts=dict(counts), affected_native_pair_count=len(touched),
        affected_native_pairs=touched, other_families=other_families,
        policy="remove definite foreground and uncertain boundary from known board factors",
        solver_started=False)
    return data, adapted, metadata
