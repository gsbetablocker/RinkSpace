"""Evidence-bound exposure times for the continuous programme camera model.

Native source PTS remain frame identities, factor keys, and output timestamps.
This sidecar supplies a separate time coordinate for camera motion.  A unique
pixel match to the official 50 fps source supplies its presentation time;
unmatched frames get explicitly estimated coordinates, never an invented
official exposure identity.  Conservative alias groups are not changed.
"""
from __future__ import annotations

from bisect import bisect_left
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

import low_end_confirmed_exposure_aliases as evidence


SOURCE_TICKS_PER_SECOND = 15360
OFFICIAL_TICKS_PER_SECOND = 12800
OFFICIAL_FRAME_TICKS = 256
LEGACY_POLICY = "legacy_accepted_official_v1"
COLLISION_SAFE_POLICY = "exclude_unconfirmed_official_collisions_v1"
CLOCK_POLICIES = (LEGACY_POLICY, COLLISION_SAFE_POLICY)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_resume_clock_policy(output: Path, requested_policy: str) -> None:
    """Reject a different clock before reconstructing an existing graph."""
    _require(requested_policy in CLOCK_POLICIES, "Unknown requested resume clock policy")
    phase = json.loads((Path(output)/"phase_00.json").read_text(encoding="utf8"))
    recorded = phase.get("exposure_clock_policy")
    metadata_policy = phase.get("exposure_clock", {}).get("clock_policy")
    _require(recorded is None or metadata_policy is None or recorded == metadata_policy,
             "Saved phase declares inconsistent exposure-clock policies")
    actual = recorded or metadata_policy or LEGACY_POLICY
    _require(actual == requested_policy,
             f"Resume exposure-clock policy differs: saved {actual}, requested {requested_policy}; use a fresh output")


def _official_for_group(group: dict[str, Any], rows: dict[int, dict]) -> int | None:
    stamps = tuple(map(int, group["full_pts"]))
    _require(bool(stamps) and stamps == tuple(sorted(stamps)),
             "Exposure group has unordered or empty native PTS")
    reported = tuple(group["frame_accepted_official_pts"])
    statuses = tuple(group["frame_match_status"])
    _require(len(reported) == len(statuses) == len(stamps),
             "Exposure group mapping fields have unequal lengths")
    for pts, official, status in zip(stamps, reported, statuses):
        row = rows[pts]
        _require(int(row["shot_index"]) == int(group["shot_index"])
                 and row["accepted_official_pts"] == official
                 and row["match_status"] == status,
                 f"Official source mapping disagrees at native PTS {pts}")
        if official is not None:
            _require(status == "unique_pixel_match",
                     f"Non-unique official mapping accepted at native PTS {pts}")
    if len(stamps) == 1:
        _require(group["confirmed_official_pts"] is None,
                 "Singleton was incorrectly labelled a confirmed alias chain")
        return None if reported[0] is None else int(reported[0])
    official = group["confirmed_official_pts"]
    _require(official is not None
             and all(p == official for p in reported)
             and group["criterion"] == "confirmed_same_official_exposure_pair_chain"
             and len(group["confirmed_pair_evidence"]) == len(stamps) - 1,
             "Aliased source frames lack a confirmed common official exposure")
    for (left, right), pair in zip(zip(stamps[:-1], stamps[1:]),
                                   group["confirmed_pair_evidence"]):
        _require(right - left == 256
                 and int(pair["previous_full_pts"]) == left
                 and int(pair["full_pts"]) == right
                 and int(pair["official_pts"]) == int(official)
                 and rows[right]["previous_pair_status"] ==
                 "confirmed_same_official_exposure",
                 "Aliased frames do not have confirmed adjacent-pair evidence")
    return int(official)


def build_exposure_clock(
    data: dict[str, Any], aliases: dict[int, int], *,
    policy: str = LEGACY_POLICY, source_shots_path: Path | None = None,
) -> tuple[dict[int, float], dict[str, Any], set[tuple[int, int]]]:
    """Return native-equivalent virtual PTS, audit metadata, and valid motion edges.

    ``virtual_pts`` contains every original selected source PTS, in floating
    ticks at 15360/s.  Aliased native frames have exactly the same time.  For
    confirmed unique matches, ``t = 1.2*official_pts + shot_offset_ticks``;
    each shot offset is the median alignment to its canonical native PTS.

    ``valid_adjacent_pairs`` contains consecutive canonical keys *within a
    shot* only when both have unique accepted official pixel matches exactly
    one official frame (256/12800 s) apart.  An acceleration factor on
    ``(a,b,c)`` is justified only if both ``(a,b)`` and ``(b,c)`` are present.
    The explicit collision-safe policy also excludes every canonical group
    sharing an accepted official key with another conservative group in the
    same shot. Their source identities and original mapping remain separate;
    their clock coordinates are estimates, not direct official anchors.

    This excludes unknown times, unaliased same-time frames, and official
    clock gaps without changing factor/frame identity.
    """
    _require(policy in CLOCK_POLICIES, f"Unknown exposure-clock policy: {policy}")
    groups_path = evidence.CADENCE / "exposure_groups.json"
    rows_path = evidence.CADENCE / "native60_to_official50.jsonl"
    shots_path = Path(source_shots_path or evidence.SOURCE_SHOTS)
    expected = ((groups_path, evidence.GROUPS_SHA256),
                (rows_path, evidence.MAPPING_SHA256))
    for path, digest in expected:
        _require(_sha256(path) == digest, f"Frozen exposure evidence changed: {path}")

    packet = json.loads(groups_path.read_text(encoding="utf8"))
    source = json.loads(shots_path.read_text(encoding="utf8"))
    source_manifest_sha = _sha256(shots_path)
    scope_only_manifest_change = False
    if source_manifest_sha != evidence.SOURCE_SHOTS_SHA256:
        # The current user scope excludes the historical 17 preprogramme
        # frames. No changed source/cut fields may slip past the frozen proof.
        original = dict(source)
        scope_only_manifest_change = "analysis_scope" in original
        original.pop("analysis_scope", None)
        frozen = json.dumps(original, ensure_ascii=False, indent=2).replace(
            "\n", "\r\n").encode("utf8")
        _require(policy == COLLISION_SAFE_POLICY and scope_only_manifest_change
                 and hashlib.sha256(frozen).hexdigest() == evidence.SOURCE_SHOTS_SHA256,
                 f"Frozen exposure source manifest changed: {shots_path}")
    _require(packet["schema"] == "rinkspace.official50-confirmed-exposure-groups.v1"
             and packet["source_binding"]["full60"]["sha256"] ==
                 evidence.SOURCE_SHA256
             and packet["source_binding"]["full60"]["native_pts_time_base"] ==
                 "1/15360"
             and packet["source_binding"]["official50"]["native_pts_time_base"] ==
                 "1/12800"
             and packet["validation"]["all_native_pts_exactly_once"] is True
             and packet["validation"]["all_groups_single_shot"] is True
             and packet["validation"]["all_nine_cut_pairs_different_groups"] is True
             and source["source_sha256_from_timeline"] == evidence.SOURCE_SHA256
             and source["source_time_base"] == "1/15360"
             and data["captured"]["source_hash"] == evidence.SOURCE_SHA256,
             "Exposure packet, source timebase, or graph source hash disagrees")

    selected_pts = set(map(int, data["pts"]))
    alias_map = {int(k): int(v) for k, v in aliases.items()}
    _require(alias_map.keys() == selected_pts,
             "Alias map does not cover exactly the selected original PTS")
    shot_pts: dict[int, tuple[int, ...]] = {}
    for shot in data["shots"]:
        index = int(shot["index"])
        pts = tuple(map(int, shot["pts"]))
        _require(index not in shot_pts and pts == tuple(sorted(set(pts)))
                 and bool(pts) and 0 <= index < len(source["shots"]),
                 "Selected shot PTS or index is invalid")
        bounds = source["shots"][index]
        _require(pts[0] == int(bounds["start_pts"])
                 and pts[-1] + 256 == int(bounds["end_pts"])
                 and len(pts) == int(bounds["source_native_frame_count"])
                 and all(b-a == 256 for a,b in zip(pts,pts[1:])),
                 "Selected shot no longer matches the audited original PTS")
        shot_pts[index] = pts
    _require(set().union(*(set(p) for p in shot_pts.values())) == selected_pts
             and sum(len(p) for p in shot_pts.values()) == len(selected_pts),
             "Selected source PTS are omitted or repeated across shots")

    rows: dict[int, dict] = {}
    with rows_path.open(encoding="utf8") as stream:
        for line in stream:
            row = json.loads(line)
            pts = int(row["full_pts"])
            if pts in selected_pts:
                _require(pts not in rows, "Official mapping repeated a source PTS")
                rows[pts] = row
    _require(rows.keys() == selected_pts, "Official mapping omitted selected source PTS")

    by_shot: dict[int, list[dict[str, Any]]] = defaultdict(list)
    selected_seen: set[int] = set()
    for group in packet["groups"]:
        index = int(group["shot_index"])
        if index not in shot_pts:
            continue
        stamps = tuple(map(int, group["full_pts"]))
        _require(all(p in selected_pts and p not in selected_seen for p in stamps),
                 "Selected exposure group is partial, repeated, or crosses shots")
        selected_seen.update(stamps)
        key = stamps[0]
        _require(all(alias_map[p] == key for p in stamps),
                 "Clock would disagree with the existing conservative aliases")
        official = _official_for_group(group, rows)
        by_shot[index].append(dict(key=key, stamps=stamps, official=official))
    _require(selected_seen == selected_pts, "Exposure groups omitted selected source PTS")

    collisions = []
    for index, groups in by_shot.items():
        by_official = defaultdict(list)
        for group in groups:
            if group["official"] is not None:
                by_official[group["official"]].append(group)
        for official, members in sorted(by_official.items()):
            if len(members) < 2:
                continue
            keys = [g["key"] for g in members]
            collisions.append(dict(shot_index=index, accepted_official_pts=official,
                canonical_source_pts=keys,
                full_source_pts=[p for g in members for p in g["stamps"]],
                source_pair_evidence=[dict(full_pts=p,
                    previous_full_pts=rows[p].get("previous_full_pts"),
                    previous_pair_status=rows[p].get("previous_pair_status"),
                    full_pair_roi_gray_mae=rows[p].get("full_pair_roi_gray_mae"))
                    for g in members for p in g["stamps"]]))
            if policy == COLLISION_SAFE_POLICY:
                for group in members:
                    group["original_official"] = group["official"]
                    group["collision_canonical_source_pts"] = keys
                    group["official"] = None

    virtual_pts: dict[int, float] = {}
    valid_adjacent_pairs: set[tuple[int, int]] = set()
    by_key: dict[str, dict[str, Any]] = {}
    shot_report: dict[str, dict[str, Any]] = {}
    for index, expected_pts in shot_pts.items():
        groups = by_shot[index]
        _require(tuple(p for g in groups for p in g["stamps"]) == expected_pts,
                 f"Exposure groups disagree with source order in shot {index}")
        mapped_idx = [i for i,g in enumerate(groups) if g["official"] is not None]
        mapped_official = [groups[i]["official"] for i in mapped_idx]
        _require(all(b >= a for a,b in zip(mapped_official,mapped_official[1:])),
                 f"Official accepted times reverse within shot {index}")
        if mapped_idx:
            offset = float(np.median([
                groups[i]["key"] -
                groups[i]["official"] * SOURCE_TICKS_PER_SECOND /
                    OFFICIAL_TICKS_PER_SECOND
                for i in mapped_idx]))
        else:
            offset = None
        known_times = {
            i: groups[i]["official"] * SOURCE_TICKS_PER_SECOND /
                OFFICIAL_TICKS_PER_SECOND + offset for i in mapped_idx
        }
        times: list[float] = []
        statuses = Counter()
        for i, group in enumerate(groups):
            key = group["key"]
            if i in known_times:
                t = float(known_times[i])
                status = "matched_unique_official"
                interval = [t,t]
                anchors = [key,key]
            elif not mapped_idx:
                t = float(key)
                status = "estimated_native_no_official_anchor"
                interval = [None,None]
                anchors = [None,None]
            else:
                pos = bisect_left(mapped_idx,i)
                left = mapped_idx[pos-1] if pos else None
                right = mapped_idx[pos] if pos < len(mapped_idx) else None
                if left is not None and right is not None:
                    p0,p1 = groups[left]["key"],groups[right]["key"]
                    t0,t1 = known_times[left],known_times[right]
                    t = float(t0 + (t1-t0)*(key-p0)/(p1-p0))
                    status = "estimated_between_official_anchors"
                    interval = [float(t0),float(t1)]
                elif left is not None:
                    t = float(known_times[left] + key-groups[left]["key"])
                    status = "estimated_native_after_official_anchor"
                    interval = [float(known_times[left]),None]
                else:
                    assert right is not None
                    t = float(known_times[right] - (groups[right]["key"]-key))
                    status = "estimated_native_before_official_anchor"
                    interval = [None,float(known_times[right])]
                anchors = [groups[left]["key"] if left is not None else None,
                           groups[right]["key"] if right is not None else None]
            estimation_status = status
            if "original_official" in group:
                status = "estimated_official_match_collision"
            times.append(t)
            statuses[status] += 1
            for pts in group["stamps"]:
                virtual_pts[pts] = t
            by_key[str(key)] = dict(
                shot_index=index, source_pts=list(group["stamps"]),
                virtual_pts=t, status=status,
                accepted_official_pts=group["official"],
                admissible_interval_virtual_pts=interval,
                interpolation_anchor_source_pts=anchors)
            if "original_official" in group:
                by_key[str(key)].update(
                    original_accepted_official_pts=group["original_official"],
                    original_match_status=[rows[p]["match_status"] for p in group["stamps"]],
                    original_previous_pair_status=[rows[p].get("previous_pair_status")
                                                   for p in group["stamps"]],
                    official_anchor_eligible=False,
                    collision_canonical_source_pts=group["collision_canonical_source_pts"],
                    clock_estimation_method=estimation_status)

        _require(all(b >= a-1e-7 for a,b in zip(times,times[1:])),
                 f"Exposure clock reverses within shot {index}")
        zero_pairs, skipped_pairs = [], []
        for a,b in zip(groups,groups[1:]):
            oa,ob = a["official"],b["official"]
            pair = (a["key"],b["key"])
            if oa is not None and ob is not None and ob-oa == OFFICIAL_FRAME_TICKS:
                valid_adjacent_pairs.add(pair)
            elif oa is not None and ob is not None and ob == oa:
                zero_pairs.append(list(pair))
            elif oa is not None and ob is not None and ob-oa > OFFICIAL_FRAME_TICKS:
                skipped_pairs.append(dict(source_pts=list(pair),
                                          official_delta_pts=int(ob-oa)))
        native_disagreement = [abs(g["key"]-t)/SOURCE_TICKS_PER_SECOND
                               for g,t in zip(groups,times)
                               if g["official"] is not None]
        shot_report[str(index)] = dict(
            native_frame_count=len(expected_pts), canonical_frame_count=len(groups),
            matched_canonical_count=len(mapped_idx),
            estimated_canonical_count=len(groups)-len(mapped_idx),
            timestamp_status_counts=dict(statuses),
            official_to_native_offset_ticks=offset,
            matched_native_clock_disagreement_seconds_max=
                max(native_disagreement,default=None),
            zero_time_unaliased_pairs=zero_pairs,
            official_gap_pairs=skipped_pairs,
            valid_adjacent_motion_pair_count=sum(
                (a["key"],b["key"]) in valid_adjacent_pairs
                for a,b in zip(groups,groups[1:])),
            official_collision_canonical_count=sum(len(c["canonical_source_pts"])
                for c in collisions if c["shot_index"] == index),
            official_collision_excluded_count=sum("original_official" in g for g in groups))

    _require(virtual_pts.keys() == selected_pts,
             "Virtual clock omitted original source PTS")
    metadata = dict(
        schema="rinkspace.continuous-exposure-clock.v1",
        clock_policy=policy,
        source_pts_role="unchanged original frame identity and output timestamp",
        virtual_pts_units="floating native-equivalent ticks at 15360 ticks/second",
        official_time_role="matched official presentation time; capture time is not independently established",
        estimated_time_role="interpolated/extrapolated coordinate only; no official exposure identity inferred",
        motion_prior_rule="use three canonical states only when both adjacent pairs are in valid_adjacent_pairs",
        evidence_sha256=dict(exposure_groups=evidence.GROUPS_SHA256,
                             official_mapping=evidence.MAPPING_SHA256,
                             source_shots=evidence.SOURCE_SHOTS_SHA256),
        shots=shot_report, timestamps_by_canonical_source_pts=by_key)
    metadata.update(
        source_manifest_binding=dict(path=str(shots_path.resolve()),
            actual_sha256=source_manifest_sha,
            frozen_sha256=evidence.SOURCE_SHOTS_SHA256,
            scope_only_manifest_change=scope_only_manifest_change),
        official_anchor_collisions=collisions,
        official_collision_canonical_count=sum(len(c["canonical_source_pts"]) for c in collisions),
        collision_anchor_rule=("retain all historical accepted anchors" if policy == LEGACY_POLICY else
            "exclude every distinct conservative canonical group sharing a shot/accepted official key"))
    return virtual_pts, metadata, valid_adjacent_pairs


def candidate_clock_for_saved_phase(data: dict[str, Any], aliases: dict[int, int],
                                    phase: dict[str, Any], *,
                                    frozen_source_shots_path: Path):
    """Create a clock sidecar while retaining saved source-image pose groups.

    This is not a graph reconstruction or solve. The frozen source proof must
    reproduce the old direct anchors. Existing estimated pose-clock choices
    survive; a change reaching such an estimate requires a separate explicit
    regrouping rather than silently replacing the saved pose-time policy.
    """
    legacy, legacy_meta, old_pairs = build_exposure_clock(data, aliases,
        policy=LEGACY_POLICY, source_shots_path=frozen_source_shots_path)
    proposed, proposed_meta, new_pairs = build_exposure_clock(data, aliases,
        policy=COLLISION_SAFE_POLICY, source_shots_path=frozen_source_shots_path)
    saved = {int(p): float(t) for p,t in phase["frame_clock_virtual_pts"].items()}
    _require(saved.keys() == legacy.keys(), "Saved clock has different source coverage")
    _require(phase["source_sha256"] == data["captured"]["source_hash"],
             "Saved clock phase identifies another source video")
    old_meta = phase["exposure_clock"]
    _require(old_meta.get("clock_policy", LEGACY_POLICY) == LEGACY_POLICY
             and not old_meta.get("saved_graph_clock_candidate"),
             "Expected an original legacy graph phase, not a relabelled clock candidate")
    records = deepcopy(old_meta["timestamps_by_canonical_source_pts"])
    pose_representatives = {}
    for key, record in records.items():
        for pts in record["source_pts"]:
            _require(int(pts) not in pose_representatives, "Saved pose clock repeats a source PTS")
            pose_representatives[int(pts)] = int(key)
        _require(all(saved[int(p)] == record["virtual_pts"] for p in record["source_pts"]),
                 "Saved pose group clock disagrees with its recorded time")
        _require(all(record["accepted_official_pts"] ==
            legacy_meta["timestamps_by_canonical_source_pts"][str(aliases[int(p)])]["accepted_official_pts"]
            for p in record["source_pts"]),
            "Saved pose group changes an original accepted official mapping")
        if record["accepted_official_pts"] is not None:
            _require(all(saved[int(p)] == legacy[int(p)] for p in record["source_pts"]),
                     "Saved direct official anchor differs from frozen legacy evidence")
    _require(pose_representatives.keys() == saved.keys(), "Saved pose groups omit source PTS")
    binding = phase.get("source_camera_rebinding", {}).get(
        "source_binding_metadata", {}).get("frame_key_by_source_pts")
    if binding is not None:
        _require({int(p):int(k) for p,k in binding.items()} == pose_representatives,
                 "Saved pose clock differs from actual frame-key binding")
    affected = {p for p in saved if proposed[p] != legacy[p]}
    collision_pts = {p for c in proposed_meta["official_anchor_collisions"]
                     for p in c["full_source_pts"]}
    _require(affected <= collision_pts,
             "Collision-safe rebuild affects other estimates; explicit pose-clock re-estimation required")
    clock = dict(saved)
    for pts in affected:
        clock[pts] = proposed[pts]
    for key, record in list(records.items()):
        members = list(map(int, record["source_pts"]))
        _require(len({clock[p] for p in members}) == 1,
                 "New clock would split a saved source-image pose group")
        if set(members) & collision_pts:
            _require(set(members) <= collision_pts and len({aliases[p] for p in members}) == 1,
                     "Collision clock would re-label a saved source-image merged pose")
            replacement = deepcopy(proposed_meta["timestamps_by_canonical_source_pts"][str(aliases[members[0]])])
            _require(replacement["source_pts"] == members, "Collision changes a saved pose group")
            records[key] = replacement
    saved_pairs = {tuple(map(int,p)) for p in phase["valid_temporal_acceleration_pairs"]}
    mapped_old = {(pose_representatives[a],pose_representatives[b]) for a,b in old_pairs}
    _require(saved_pairs <= mapped_old, "Saved graph has unexplained strict motion edges")
    mapped_new = {(pose_representatives[a],pose_representatives[b]) for a,b in new_pairs}
    pairs = saved_pairs & mapped_new
    _require(not any(a in collision_pts or b in collision_pts for a,b in pairs),
             "A collision retained a strict motion edge")
    for shot in data["shots"]:
        times = [clock[int(p)] for p in shot["pts"]]
        _require(all(b >= a-1e-7 for a,b in zip(times,times[1:])),
                 "Candidate saved pose clock reverses within a shot")
    for a,b in pairs:
        _require(abs(clock[b]-clock[a]-307.2) <= 1e-4,
                 "Retained official motion edge changed its 20ms clock")
    metadata = deepcopy(old_meta)
    metadata.update({key: deepcopy(proposed_meta[key]) for key in (
        "clock_policy", "source_manifest_binding", "official_anchor_collisions",
        "official_collision_canonical_count", "collision_anchor_rule")})
    metadata["timestamps_by_canonical_source_pts"] = records
    for shot in data["shots"]:
        index = int(shot["index"])
        selected = [r for r in records.values() if r["shot_index"] == index]
        report = metadata["shots"][str(index)]
        report["legacy_zero_time_unaliased_pairs"] = report["zero_time_unaliased_pairs"]
        report["zero_time_unaliased_pairs"] = proposed_meta["shots"][str(index)]["zero_time_unaliased_pairs"]
        report["timestamp_status_counts"] = dict(Counter(r["status"] for r in selected))
        report["matched_canonical_count"] = sum(r["status"] == "matched_unique_official" for r in selected)
        report["estimated_canonical_count"] = len(selected)-report["matched_canonical_count"]
        report["official_collision_canonical_count"] = sum(r["status"] == "estimated_official_match_collision" for r in selected)
        report["official_collision_excluded_count"] = report["official_collision_canonical_count"]
        report["valid_adjacent_motion_pair_count"] = sum(records[str(a)]["shot_index"] == index for a,b in pairs)
    metadata["saved_graph_clock_candidate"] = dict(
        clock_only=True, camera_graph_rebuilt=False, camera_states_solved=False,
        saved_pose_group_identity_preserved=True,
        saved_noncollision_estimated_clocks_preserved=True,
        candidate_route="collision-only saved pose-clock derivation; distinct from a fresh official-group clock build",
        time_gauge="saved graph gauge retained; any noncollision clock change requires explicit reconstruction",
        strict_motion_edges_removed=len(saved_pairs-pairs), strict_motion_edges_added=0,
        changed_source_pts=sorted(affected),
        graph_reconstruction_and_real_source_validation_required=True)
    return clock, metadata, pairs
