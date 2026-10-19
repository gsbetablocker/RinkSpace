"""Explicit candidate GVHMR slots on the verified official 25 Hz clock.

Source PTS identify the real input pixels. Official PTS identify the model
sequence cadence; neither timestamps nor delta time are added to the network.
Unknown exposure identity, missing canonical H and missing official frames
break context, rather than becoming irregular adjacent model observations.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .exposure_observations import (GROUPS_SHA256, MAPPING_SHA256,
                                    confirmed_representatives)


POLICY = "official_25hz"
OFFICIAL_TIME_BASE = 12800
OFFICIAL_FRAME_STEP = 256
MODEL_STEP = 512
TEMPORAL_FIELDS = (
    "shot_index", "shot_slot", "source_pts",
    "official_exposure_representative_pts", "official_pts", "sequence_id",
    "source_presentation_time_s", "model_sample_time_s",
)


def json_digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def temporal_rows(rows: list[dict]) -> list[dict]:
    return [{key: row[key] for key in TEMPORAL_FIELDS} for row in rows]


def temporal_plan_digest(plan: dict) -> str:
    return json_digest(dict(time_policy=plan["time_policy"],
                            model_time_contract=plan["model_time_contract"],
                            slots=temporal_rows(plan["slots"]),
                            unavailable_exposures=plan["unavailable_exposures"],
                            sequences=plan["sequences"]))


def build_official_slots(camera, observations: dict, groups: dict,
                         mapping: list[dict], hashes: dict) -> dict:
    """Select every second official frame inside complete, supported runs."""
    if (hashes.get("groups") != GROUPS_SHA256
            or hashes.get("mapping") != MAPPING_SHA256
            or groups["source_binding"]["official50"]["native_pts_time_base"] != "1/12800"
            or groups["source_binding"]["full60"]["native_pts_time_base"] != "1/15360"):
        raise ValueError("Official 25 Hz needs the frozen complete official-clock evidence")
    pts = np.asarray(camera.pts)
    aliases, proof = confirmed_representatives(
        groups, mapping, pts, camera.shot_index,
        camera.source["source_sha256_from_timeline"])
    if not np.array_equal(aliases, camera.official_exposure_representative_pts):
        raise ValueError("Camera exposure groups differ from verified official evidence")
    valid = np.asarray(observations["valid"], bool)
    if valid.shape != pts.shape:
        raise ValueError("Canonical observation validity must cover every source PTS")
    by_pts = {int(row["full_pts"]): row for row in mapping}
    canonical = np.flatnonzero(pts == aliases)
    accepted_counts = {}
    for idx in canonical:
        row = by_pts[int(pts[idx])]
        if row["match_status"] == "unique_pixel_match" and row["accepted_official_pts"] is not None:
            key = (int(camera.shot_index[idx]), int(row["accepted_official_pts"]))
            accepted_counts[key] = accepted_counts.get(key, 0) + 1

    slots, unavailable, sequences = [], [], []
    for shot in range(1, 10):
        members = canonical[camera.shot_index[canonical] == shot]
        run, reason, shot_slot = [], "shot_start", 0

        def flush():
            nonlocal run, shot_slot
            if not run:
                return
            sequence = len(sequences)
            chosen = run[::2]
            sequences.append(dict(sequence_id=sequence, shot_index=shot,
                break_before_reason=reason, canonical_exposure_count=len(run),
                selected_slots=len(chosen), first_official_pts=run[0][1],
                last_official_pts=run[-1][1],
                last_selected_official_pts=chosen[-1][1],
                omitted_alternate_official_frames=len(run)-len(chosen)))
            for idx, official in chosen:
                source = int(pts[idx])
                slots.append(dict(slot_id=len(slots), shot_index=shot,
                    shot_slot=shot_slot, source_pts=source,
                    official_exposure_representative_pts=int(aliases[idx]),
                    official_pts=official, sequence_id=sequence,
                    source_presentation_time_s=source / 15360,
                    model_sample_time_s=official / OFFICIAL_TIME_BASE,
                    model_time_semantics="verified official reference clock; checkpoint rhythm compatibility unverified",
                    observation_valid=True, observation_reuses_confirmed_exposure=False,
                    camera_pose_representative_pts=int(camera.source_pose_representative_pts[idx])))
                shot_slot += 1
            # An explicit unused slot separates runs even when their reference
            # timestamps happen to be 40 ms apart. Existing window/gap logic
            # then also prevents interpolation through unavailable exposures.
            shot_slot += 1
            run = []

        for idx in members:
            source = int(pts[idx])
            row = by_pts[source]
            official = row["accepted_official_pts"]
            why = None
            if row["match_status"] != "unique_pixel_match" or official is None:
                why = "unknown_official_exposure"
            elif not isinstance(official, int) or official < 0 or official % OFFICIAL_FRAME_STEP:
                raise ValueError("Accepted official PTS is outside the verified 50 Hz clock")
            elif accepted_counts[(shot, official)] != 1:
                why = "official_match_repeated_without_confirmed_group"
            elif not valid[idx]:
                why = "missing_canonical_observation"
            if why is not None:
                flush()
                unavailable.append(dict(source_pts=source, shot_index=shot,
                    accepted_official_pts=official, match_status=row["match_status"], reason=why))
                reason = why
                continue
            if run and official - run[-1][1] != OFFICIAL_FRAME_STEP:
                previous = run[-1][1]
                flush()
                reason = f"nonconsecutive_official50_pts:{official-previous}"
            run.append((int(idx), int(official)))
        flush()

    result = dict(slots=slots, unavailable_exposures=unavailable, sequences=sequences,
        model_time_contract=dict(schema="rinkspace.official_25hz_model_time.v1",
            official_time_base="1/12800", official_frame_step_pts=OFFICIAL_FRAME_STEP,
            model_step_official_pts=MODEL_STEP, model_step_seconds=.04,
            selection="every second unique accepted official frame within complete canonical runs",
            unknown_missing_or_nonconsecutive_official_frames="break_sequence",
            preparation_missing_crop="shot_slot gap breaks windows and materialization interpolation",
            last_off_phase_frame="omitted; no 20 ms boundary slot",
            source_pts_semantics="actual native presentation PTS retained",
            actual_timestamps_or_delta_time_passed_to_network=False,
            checkpoint_training_rhythm_compatibility="unverified",
            groups_sha256=hashes["groups"], mapping_sha256=hashes["mapping"],
            official_source_sha256=groups["source_binding"]["official50"]["sha256"]),
        official_evidence_sha256=hashes, official_exposure_proof=proof)
    validate_rows(slots)
    return result


def validate_rows(rows: list[dict]) -> None:
    """Fail before inference if an irregular pair is labelled one sequence."""
    for row in rows:
        if (any(not isinstance(row[key], int) or row[key] < 0 for key in (
                "source_pts", "official_exposure_representative_pts", "official_pts",
                "sequence_id", "shot_slot", "shot_index"))
                or row["shot_index"] not in range(1, 10)
                or row["source_pts"] != row["official_exposure_representative_pts"]
                or row["official_pts"] % OFFICIAL_FRAME_STEP
                or row["model_sample_time_s"] != row["official_pts"] / OFFICIAL_TIME_BASE
                or row["source_presentation_time_s"] != row["source_pts"] / 15360):
            raise ValueError("Official 25 Hz slot has inconsistent real source/reference time")
    for left, right in zip(rows[:-1], rows[1:]):
        if right["source_pts"] <= left["source_pts"]:
            raise ValueError("Official 25 Hz source PTS must strictly advance")
        if left["shot_index"] != right["shot_index"]:
            if left["sequence_id"] == right["sequence_id"]:
                raise ValueError("Official 25 Hz sequence crosses a camera cut")
            continue
        slot_delta = right["shot_slot"] - left["shot_slot"]
        contiguous = slot_delta == 1
        same_sequence = left["sequence_id"] == right["sequence_id"]
        if (slot_delta <= 0 or (contiguous and not same_sequence)
                or (same_sequence and right["official_pts"] - left["official_pts"]
                    != MODEL_STEP * slot_delta)):
            raise ValueError("Official 25 Hz adjacent model slots require exactly 40 ms; otherwise break sequence")


def plan_binding(plan: dict) -> dict:
    if plan.get("time_policy") != POLICY:
        if "model_time_contract" in plan or "temporal_plan_sha256" in plan:
            raise ValueError("Official cadence metadata cannot be relabelled as a legacy policy")
        return {}
    if plan.get("temporal_plan_sha256") != temporal_plan_digest(plan):
        raise ValueError("Official 25 Hz slot plan changed after temporal binding")
    validate_rows(plan["slots"])
    return {key: plan[key] for key in (
        "temporal_plan_sha256", "model_time_contract", "official_evidence_sha256",
        "official_evidence_inputs")}


def validate_plan_evidence(plan: dict, camera, observations: dict | None = None):
    if not plan_binding(plan):
        return
    from .exposure_observations import load_confirmed_evidence
    inputs = plan.get("official_evidence_inputs", {})
    if not inputs.get("groups") or not inputs.get("mapping"):
        raise ValueError("Official 25 Hz preparation needs the checked proof input paths")
    groups, mapping, hashes = load_confirmed_evidence(
        Path(inputs["groups"]), Path(inputs["mapping"]), camera.source_shots_path)
    if hashes != plan["official_evidence_sha256"]:
        raise ValueError("Official 25 Hz proof changed after planning")
    by_pts = {row["full_pts"]: row for row in mapping}
    for slot in plan["slots"]:
        row = by_pts[slot["source_pts"]]
        if (row["match_status"] != "unique_pixel_match"
                or row["accepted_official_pts"] != slot["official_pts"]
                or row["shot_index"] != slot["shot_index"]):
            raise ValueError("Official 25 Hz selected source no longer has its accepted reference exposure")
    if observations is not None:
        actual = build_official_slots(camera, observations, groups, mapping, hashes)
        if (temporal_rows(actual["slots"]) != temporal_rows(plan["slots"])
                or any(actual[key] != plan[key] for key in (
                    "sequences", "unavailable_exposures", "model_time_contract"))):
            raise ValueError("Official 25 Hz plan differs from actual canonical observations/proof")
    return groups, mapping, hashes


def interpolation_fractions(camera, mapping: list[dict], left_source_pts: int,
                            right_source_pts: int, source_pts: np.ndarray) -> np.ndarray:
    """Official-clock fractions only for a complete real 50 Hz exposure chain.

    Equal native presentation spacing is not evidence of exposure time. A
    two-slot 25 Hz interval must have exactly its one accepted middle exposure;
    unknown/missing reference identity leaves the entire bracket unavailable.
    """
    wanted = np.asarray(source_pts)
    result = np.full(len(wanted), np.nan, np.float32)
    pts = np.asarray(camera.pts)
    aliases = np.asarray(camera.official_exposure_representative_pts)
    by_pts = {int(row["full_pts"]): row for row in mapping}
    endpoints = np.searchsorted(pts, [left_source_pts, right_source_pts])
    if (np.any(endpoints >= len(pts))
            or not np.array_equal(pts[endpoints], [left_source_pts, right_source_pts])
            or not np.array_equal(aliases[endpoints], pts[endpoints])
            or camera.shot_index[endpoints[0]] != camera.shot_index[endpoints[1]]):
        return result
    member = (pts >= left_source_pts) & (pts <= right_source_pts)
    representatives = np.unique(aliases[member])
    rows = [by_pts.get(int(p), {}) for p in representatives]
    if (len(rows) != 3 or representatives[0] != left_source_pts
            or representatives[-1] != right_source_pts
            or any(row.get("match_status") != "unique_pixel_match"
                   or row.get("accepted_official_pts") is None
                   or row.get("shot_index") != int(camera.shot_index[endpoints[0]])
                   for row in rows)):
        return result
    official = np.asarray([row["accepted_official_pts"] for row in rows])
    if not np.array_equal(np.diff(official), [OFFICIAL_FRAME_STEP, OFFICIAL_FRAME_STEP]):
        return result
    indices = np.searchsorted(pts, wanted)
    available = indices < len(pts)
    available &= pts[np.minimum(indices, len(pts)-1)] == wanted
    fraction = {int(p): np.float32((int(t)-int(official[0])) / MODEL_STEP)
                for p, t in zip(representatives, official)}
    for i in np.flatnonzero(available):
        representative = int(aliases[indices[i]])
        if representative in fraction:
            result[i] = fraction[representative]
    return result


def cache_binding(plan: dict, rows: list[dict]) -> dict:
    if plan.get("time_policy") != POLICY:
        return {}
    result = plan_binding(plan)
    selected = temporal_rows(rows)
    result.update(temporal_slots=selected, temporal_slots_sha256=json_digest(selected))
    return result


def validate_cache(metadata: dict, data: dict, label: str, plan: dict | None = None,
                   camera=None) -> None:
    if metadata.get("time_policy") != POLICY:
        if ("model_time_contract" in metadata or "temporal_plan_sha256" in metadata
                or "official_pts" in data or "sequence_id" in data):
            raise ValueError(f"{label} official cadence cannot be relabelled as a legacy policy")
        return
    contract = metadata.get("model_time_contract", {})
    if (contract.get("model_step_official_pts") != MODEL_STEP
            or contract.get("official_time_base") != "1/12800"
            or contract.get("groups_sha256") != GROUPS_SHA256
            or contract.get("mapping_sha256") != MAPPING_SHA256
            or contract.get("actual_timestamps_or_delta_time_passed_to_network") is not False
            or not isinstance(metadata.get("temporal_plan_sha256"), str)
            or len(metadata["temporal_plan_sha256"]) != 64):
        raise ValueError(f"{label} lacks the official 25 Hz temporal contract")
    rows = metadata.get("temporal_slots")
    if not isinstance(rows, list) or metadata.get("temporal_slots_sha256") != json_digest(rows):
        raise ValueError(f"{label} official temporal slots changed after binding")
    if len(rows) != len(data["source_pts"]):
        raise ValueError(f"{label} official temporal slots do not cover the NPZ")
    if any(row["shot_index"] != metadata["shot_index"] for row in rows):
        raise ValueError(f"{label} official temporal slots cross the selected shot")
    for key in ("source_pts", "shot_slot", "official_exposure_representative_pts",
                "official_pts", "sequence_id", "model_sample_time_s"):
        if key not in data or not np.array_equal(data[key], np.asarray([r[key] for r in rows])):
            raise ValueError(f"{label} {key} differs from bound official temporal slots")
    validate_rows(rows)
    if plan is not None:
        binding = plan_binding(plan)
        if any(metadata.get(key) != value for key, value in binding.items()):
            raise ValueError(f"{label} official cadence is bound to another slot plan")
        planned = {json_digest(row) for row in temporal_rows(plan["slots"])}
        if any(json_digest(row) not in planned for row in rows):
            raise ValueError(f"{label} invents an official model slot")
    else:
        # Standalone inference must establish actual exposure identity before
        # loading weights. A self-consistent sidecar/NPZ is not proof that its
        # claimed official PTS belongs to those real source pixels.
        if camera is None:
            raise ValueError(f"{label} needs checked camera/source inputs for official exposure validation")
        from .exposure_observations import load_confirmed_evidence
        inputs = metadata.get("official_evidence_inputs", {})
        if not inputs.get("groups") or not inputs.get("mapping"):
            raise ValueError(f"{label} lacks original official evidence input paths")
        groups, mapping, hashes = load_confirmed_evidence(
            Path(inputs["groups"]), Path(inputs["mapping"]), camera.source_shots_path)
        if hashes != metadata.get("official_evidence_sha256"):
            raise ValueError(f"{label} frozen official proof changed after preparation")
        aliases, _ = confirmed_representatives(groups, mapping, camera.pts,
            camera.shot_index, camera.source["source_sha256_from_timeline"])
        if not np.array_equal(aliases, camera.official_exposure_representative_pts):
            raise ValueError(f"{label} camera canonical exposures differ from frozen proof")
        by_pts = {row["full_pts"]: row for row in mapping}
        for slot in rows:
            idx = int(np.searchsorted(camera.pts, slot["source_pts"]))
            reference = by_pts.get(slot["source_pts"], {})
            if (idx == len(camera.pts) or camera.pts[idx] != slot["source_pts"]
                    or aliases[idx] != slot["source_pts"]
                    or reference.get("match_status") != "unique_pixel_match"
                    or reference.get("accepted_official_pts") != slot["official_pts"]
                    or reference.get("shot_index") != slot["shot_index"]):
                raise ValueError(f"{label} source PTS lacks its unique accepted official exposure")
