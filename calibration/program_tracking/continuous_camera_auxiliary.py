"""Dynamic board pairs with verified original multiframe source indices.

The legacy conditional driver rejects the 42 auxiliary packet rows because
their source files have no single-shot ID.  This adapter keeps its native-pair
path unchanged, then checks every auxiliary row against its original cache and
constructs the same moving-centre NativeBoardPair.  It performs no solve.
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import numpy as np

import conditioned_camera_comparison as driver
from native_board_pair import NativeBoardPair
from projection import from_state
from reference_board_pair import BoardIntersectionError
from source_supported_motion import grouped_native_pairs
from track_sequence import _distributed_indices


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def _native_packet(data: dict) -> dict:
    """Make an in-memory native-only view for the original driver."""
    packet = data["packet"]
    source = {int(row["id"]): row for row in data["manifest"]["source_files"]}
    selected_shots = set(map(int, data["shot_indices"]))
    pair_ids = [i for i, file_id in enumerate(packet["source_file_id"])
                if source[int(file_id)]["kind"] == "native"
                and int(packet["shot_index"][i]) in selected_shots]
    sizes = [int(packet["offsets"][i+1]-packet["offsets"][i]) for i in pair_ids]
    point_ids = np.concatenate([np.arange(packet["offsets"][i],
                                          packet["offsets"][i+1], dtype=np.int64)
                                for i in pair_ids]) if pair_ids else np.empty(0, np.int64)
    return dict(
        shot_index=packet["shot_index"][pair_ids].copy(),
        source_file_id=packet["source_file_id"][pair_ids].copy(),
        source_pair_index=packet["source_pair_index"][pair_ids].copy(),
        pair_pts=packet["pair_pts"][pair_ids].copy(),
        offsets=np.r_[0, np.cumsum(sizes, dtype=np.int64)],
        source_point_index=packet["source_point_index"][point_ids].copy(),
        xy0=packet["xy0"][point_ids].copy(),
        xy1=packet["xy1"][point_ids].copy())


def _auxiliary_sources(data: dict) -> dict[int, tuple[dict, dict, dict]]:
    """Bind each auxiliary archive to its frozen manifest and v13 noise spec."""
    specified = {Path(row["path"]).resolve(): row
                 for row in data["config"].get("additional_source_links", [])}
    _require(len(specified) == len(data["config"].get("additional_source_links", [])),
             "Additional source specifications repeat one archive")
    result = {}
    for row in data["manifest"]["source_files"]:
        if row["kind"] != "auxiliary":
            continue
        path = Path(row["path"])
        spec = specified.get(path.resolve())
        _require(spec is not None and driver.digest(path) == row["sha256"]
                 and float(spec["sigma_px"]) == 3.
                 and int(spec["max_matches"]) == int(row["max_matches"]),
                 f"Auxiliary archive or original noise spec changed: {path}")
        similarity = np.asarray(spec["similarity_noise_px"], np.float64)
        _require(similarity.shape == (4,) and np.isfinite(similarity).all()
                 and np.all(similarity > 0),
                 f"Auxiliary similarity noise is invalid: {path}")
        with np.load(path, allow_pickle=False) as z:
            raw = {name: z[name].copy() for name in
                   ("pair_pts", "offsets", "xy0", "xy1", "fb_error_px",
                    "homography_inlier", "region_hint")}
            source_sha = str(z["source_sha256"])
            provenance = json.loads(str(z["metadata_json"]))
        count = len(raw["xy0"])
        _require(source_sha == data["captured"]["source_hash"]
                 and raw["pair_pts"].ndim == 2 and raw["pair_pts"].shape[1] == 2
                 and raw["offsets"].shape == (len(raw["pair_pts"])+1,)
                 and int(raw["offsets"][0]) == 0
                 and int(raw["offsets"][-1]) == count
                 and np.all(np.diff(raw["offsets"]) >= 0)
                 and all(len(raw[name]) == count for name in
                         ("xy1", "fb_error_px", "homography_inlier", "region_hint"))
                 and provenance.get("model_projection_used_to_make_matches") is False
                 and provenance.get("end_ad7_corners_used_to_make_matches") is False
                 and provenance.get("sam2_people_and_hud_excluded") is True,
                 f"Auxiliary source index, video, or original-pixel provenance changed: {path}")
        result[int(row["id"])] = (row, spec, raw)
    return result


def _checked_auxiliary_group(data: dict, group: dict, shot: dict,
                             sources: dict) -> tuple[int, int, NativeBoardPair, dict]:
    packet = data["packet"]
    file_id = int(group["source_file_id"])
    source, spec, raw = sources[file_id]
    i = group["packet_pair_index"]
    _require(i is not None, "Auxiliary rows require their original frozen packet")
    i = int(i)
    pair_index = int(group["source_pair_index"])
    a, b = map(int, group["pts"])
    own = set(map(int, shot["pts"]))
    _require(0 <= pair_index < len(raw["pair_pts"])
             and int(packet["source_file_id"][i]) == file_id
             and int(packet["source_pair_index"][i]) == pair_index
             and int(packet["shot_index"][i]) == int(shot["index"])
             and np.array_equal(raw["pair_pts"][pair_index], [a, b])
             and np.array_equal(packet["pair_pts"][i], [a, b])
             and a in own and b in own and b-a >= 512 and (b-a) % 256 == 0,
             f"Auxiliary pair {i} does not belong to one original camera shot")
    lo, hi = map(int, raw["offsets"][pair_index:pair_index+2])
    local = np.asarray(group["source_point_indices"], np.int64)
    global_ids = lo + local
    frozen_lo, frozen_hi = map(int, packet["offsets"][i:i+2])
    _require(local.ndim == 1 and len(local) == frozen_hi-frozen_lo
             and len(np.unique(local)) == len(local)
             and np.all((local >= 0) & (local < hi-lo))
             and np.array_equal(local, packet["source_point_index"][frozen_lo:frozen_hi])
             and np.array_equal(global_ids,
                                packet["source_global_point_index"][frozen_lo:frozen_hi])
             and np.array_equal(group["xy0"], packet["xy0"][frozen_lo:frozen_hi])
             and np.array_equal(group["xy1"], packet["xy1"][frozen_lo:frozen_hi])
             and np.array_equal(group["xy0"], raw["xy0"][global_ids])
             and np.array_equal(group["xy1"], raw["xy1"][global_ids]),
             f"Auxiliary pair {i} differs from original cache indices or UV")
    good = (raw["homography_inlier"][lo:hi].astype(bool)
            & (raw["fb_error_px"][lo:hi] < 2.5)
            & np.isfinite(raw["xy0"][lo:hi]).all(axis=1)
            & np.isfinite(raw["xy1"][lo:hi]).all(axis=1))
    pool = np.flatnonzero(good)
    original_selected = pool[_distributed_indices(
        raw["xy0"][lo:hi][pool], raw["fb_error_px"][lo:hi][pool],
        int(source["max_matches"]))]
    _require(set(local.tolist()) <= set(original_selected.tolist())
             and np.all(raw["region_hint"][global_ids] == 2)
             and np.all(packet["region_hint"][frozen_lo:frozen_hi] == 2),
             f"Auxiliary pair {i} is outside original LK/board candidate selection")
    templates = data["captured"]["templates"]
    states = data["states"]
    c0 = from_state(states[a], templates[a])
    c1 = from_state(states[b], templates[b])
    panels = np.arange(len(data["scene"]["board_surface"]["bottom_edge_xyz"]))
    pair = NativeBoardPair(group["xy0"], group["xy1"], data["scene"], c0, c1,
                           allowed_panels=panels)
    pair.rinkspace_sigma_px = float(spec["sigma_px"])
    pair.rinkspace_similarity_noise_px = np.asarray(spec["similarity_noise_px"],
                                                    np.float64).copy()
    pair.rinkspace_quality_kind = "multiframe_motion"
    pair.rinkspace_source_file_id = file_id
    pair.rinkspace_source_pair_index = pair_index
    pair.rinkspace_source_file_sha256 = source["sha256"]
    return a, b, pair, dict(shot_index=int(shot["index"]), packet_pair_index=i,
                            source_pair_index=pair_index, source_pts=a,
                            target_pts=b, pixel_count=len(pair.xy0))


def _geometry_failures(pair: NativeBoardPair, states: list[np.ndarray]) -> list[dict]:
    valid, reasons = pair.initial_validity(states)
    failures = [dict(row_index=int(row), reason=reasons[row])
                for row in np.flatnonzero(~valid)]
    if not failures:
        try:
            raw = pair._raw_residual(states)
            matrices = pair._raw_jacobians(states)
        except BoardIntersectionError as exc:
            failures = [dict(row_index=int(row), reason="jacobian_"+exc.reason)
                        for row in exc.indices]
        else:
            if not np.isfinite(raw).all() or any(not np.isfinite(m).all()
                                                  for m in matrices):
                failures = [dict(row_index=int(row),
                                 reason="nonfinite_residual_or_jacobian")
                            for row in range(len(pair.xy0))]
    return failures


def build_dynamic_pairs(data: dict) -> tuple[list, dict, dict, dict, dict]:
    """Return driver.native_pairs' tuple with authenticated auxiliary board UV.

    Native rows use the original driver and its original point selection.
    Auxiliary rows retain their different v13 sigma and similarity modes on
    each factor for the caller's noise construction. Conditional board identity
    does not become independently validated just because the source is bound.
    """
    native_data = {**data, "packet": _native_packet(data)}
    native, native_counts, by_shot, _, native_validity = driver.native_pairs(native_data)
    sigma = float(data["config"].get("native_pair_sigma_px", 2.))
    similarity = data["config"].get("native_pair_similarity_noise_px")
    for _, _, pair in native:
        pair.rinkspace_sigma_px = sigma
        pair.rinkspace_similarity_noise_px = (None if similarity is None else
                                              np.asarray(similarity, np.float64).copy())
        pair.rinkspace_quality_kind = "native_motion"

    sources = _auxiliary_sources(data)
    count = Counter(native_counts)
    invalid = list(native_validity["pairs"])
    native_by_shot = {int(shot["index"]): [] for shot in data["shots"]}
    shot_at = {int(pts): int(shot["index"]) for shot in data["shots"]
               for pts in shot["pts"]}
    for a, b, pair in native:
        _require(shot_at[int(a)] == shot_at[int(b)],
                 "Original native board pair crosses a camera cut")
        native_by_shot[shot_at[int(a)]].append((a, b, pair))
    result = []
    components = {}
    seen_aux = set()
    expected_aux = sum(int(data["manifest"]["coverage_by_shot"][str(shot["index"])][
        "auxiliary_pair_count"]) for shot in data["shots"])
    for shot in data["shots"]:
        index = int(shot["index"])
        own = Counter(by_shot[str(index)])
        own_pairs = list(native_by_shot[index])
        for group in grouped_native_pairs(data, index):
            file_id = int(group["source_file_id"])
            if file_id not in sources:
                continue
            key = file_id, int(group["source_pair_index"])
            _require(key not in seen_aux,
                     "One original auxiliary source pair appears twice")
            seen_aux.add(key)
            a, b, pair, identity = _checked_auxiliary_group(
                data, group, shot, sources)
            failures = _geometry_failures(pair, [data["states"][a], data["states"][b]])
            if failures:
                invalid.append({**identity, "invalid_rows": failures})
            own_pairs.append((a, b, pair))
            own["pixels"] += len(pair.xy0)
            own["auxiliary_pairs"] += 1
            count["pixels"] += len(pair.xy0)
            count["auxiliary_pairs"] += 1
        result.extend(own_pairs)
        by_shot[str(index)] = dict(own)
        components[str(index)] = driver.joint._components(
            shot["pts"], [(a, b) for a, b, _ in own_pairs])
    _require(len(seen_aux) == expected_aux,
             "Selected auxiliary pairs differ from frozen manifest coverage")
    reasons = dict(Counter(row["reason"] for entry in invalid
                           for row in entry["invalid_rows"]))
    validity = dict(all_selected_pixels_geometry_valid=not invalid,
                    invalid_pair_count=len(invalid),
                    invalid_row_count=sum(len(entry["invalid_rows"]) for entry in invalid),
                    reason_counts=reasons, pairs=invalid)
    return result, dict(count), by_shot, components, validity
