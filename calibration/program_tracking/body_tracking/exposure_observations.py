"""Candidate-only selection of real H caches within confirmed source exposures.

Every selected value comes from the first actual source PTS in a frozen,
uniquely matched official exposure group.  Presentation identity and per-PTS
review exclusions stay distinct.  No network, averaging, interpolation or
alias-to-canonical backfill is involved.  This module is not used by production
localization until its candidate has passed source-image review.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


OBSERVATION_FIELDS = (
    "coco17", "native_coco17_xy", "keypoint_scores", "keypoint_valid",
    "keypoint_in_source", "source_clipped_target", "visibility_status",
    "bbox_native_xyxy", "crop_affine", "crop_center", "crop_scale_200",
    "hip_midpoint_proxy_xy", "valid", "target_status", "track_id",
    "observation_origin",
)
PAIR_CRITERION = (
    "accepted_official_pts_equal_and_adjacent_within_shot_and_full_pair_roi_gray_mae_le_0.45")
GROUPS_SHA256 = "5d43945541c5d6621827a49f9c23f025f0a90941bdc6480b059decab64eb74f2"
MAPPING_SHA256 = "a9be37308b9a39fb55519a0be69c741c53a8216a81a60a2066fff25bdcf54dcc"


def digest(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def confirmed_representatives(groups: dict, mapping: list[dict], pts: np.ndarray,
                              shot_index: np.ndarray, source_sha256: str
                              ) -> tuple[np.ndarray, dict]:
    """Check actual accepted matches and pair evidence; positive keys alone fail."""
    pts, shot_index = np.asarray(pts), np.asarray(shot_index)
    if (pts.ndim != 1 or pts.dtype.kind not in "iu" or len(pts) == 0
            or np.any(np.diff(pts) <= 0) or shot_index.shape != pts.shape):
        raise ValueError("Exact increasing source PTS and matching shots required")
    if (groups.get("schema") != "rinkspace.official50-confirmed-exposure-groups.v1"
            or groups["source_binding"]["full60"]["sha256"] != source_sha256):
        raise ValueError("Official grouping source identity differs")
    by_pts = {int(row["full_pts"]): row for row in mapping}
    if len(by_pts) != len(mapping):
        raise ValueError("Official mapping has duplicate source PTS")
    covered, canonical, roi_mae, groups_selected = set(), {}, [], 0
    for group in groups["groups"]:
        stamps = list(map(int, group["full_pts"]))
        if not stamps or stamps != sorted(set(stamps)) or covered.intersection(stamps):
            raise ValueError("Official groups repeat or reorder source PTS")
        covered.update(stamps)
        rows = [by_pts[p] for p in stamps]
        if any(int(row["shot_index"]) != int(group["shot_index"]) for row in rows):
            raise ValueError("Official group crosses a source shot")
        if len(stamps) > 1:
            official = group["confirmed_official_pts"]
            if (group["criterion"] != "confirmed_same_official_exposure_pair_chain"
                    or official is None
                    or len(group["confirmed_pair_evidence"]) != len(stamps)-1):
                raise ValueError("Repeated source exposure lacks confirmation")
            if any(row["match_status"] != "unique_pixel_match"
                   or row["accepted_official_pts"] != official for row in rows):
                raise ValueError("Repeated exposure lacks unique accepted official matches")
            for evidence, left, right in zip(group["confirmed_pair_evidence"], rows[:-1], rows[1:]):
                a, b = int(left["full_pts"]), int(right["full_pts"])
                values = (float(evidence["full_pair_roi_gray_mae"]),
                          float(right["full_pair_roi_gray_mae"]))
                if (b-a != 256 or evidence["previous_full_pts"] != a
                        or evidence["full_pts"] != b
                        or evidence["official_pts"] != official
                        or evidence["criterion"] != PAIR_CRITERION
                        or right["previous_full_pts"] != a
                        or right["previous_pair_status"] != "confirmed_same_official_exposure"
                        or not np.isfinite(values).all() or max(values) > .45
                        or values[0] != values[1]):
                    raise ValueError("Repeated exposure failed frozen pixel/adjacency evidence")
                roi_mae.append(values[0])
            if any(p in pts for p in stamps):
                groups_selected += 1
        canonical.update({p: stamps[0] for p in stamps})
    if covered != set(by_pts):
        raise ValueError("Official groups and mapping have different source coverage")
    if not set(map(int, pts)).issubset(covered):
        raise ValueError("Official groups omit requested source PTS")
    representatives = np.asarray([canonical[int(p)] for p in pts], np.int64)
    indices = np.searchsorted(pts, representatives)
    if (np.any(indices >= len(pts))
            or not np.array_equal(pts[np.minimum(indices, len(pts)-1)], representatives)
            or not np.array_equal(shot_index[indices], shot_index)
            or any(int(by_pts[int(p)]["shot_index"]) != int(s) for p,s in zip(pts,shot_index))):
        raise ValueError("Canonical source PTS unavailable or across a cut")
    return representatives, dict(
        selected_source_frames=len(pts), repeated_groups=groups_selected,
        alias_presentations=int(np.sum(pts != representatives)),
        full_mapping_rows_checked=len(mapping), confirmed_pair_evidence_checked=len(roi_mae),
        confirmed_pair_roi_gray_mae_max=float(max(roi_mae)) if roi_mae else None,
        ambiguous_and_singleton_rows_never_merged=True,
        canonical_rule="first real native source PTS of uniquely confirmed official exposure group")


def select_canonical_observations(observations: dict, representatives: np.ndarray,
                                 reviewed_joint_valid: np.ndarray
                                 ) -> tuple[dict, np.ndarray, dict]:
    """Select the complete canonical H record, preserving each review mask.

    A missing canonical H stays missing even when an alias has usable H.  All
    original records remain exact under original_*.  No original reviewed
    exclusion is enabled by selection, and canonical rows are unchanged.
    """
    missing = set(OBSERVATION_FIELDS)-set(observations)
    if missing:
        raise ValueError(f"Incomplete raw H observation package: {sorted(missing)}")
    pts = np.asarray(observations["pts"])
    representatives = np.asarray(representatives)
    if (representatives.shape != pts.shape or representatives.dtype.kind not in "iu"
            or pts.dtype.kind not in "iu" or np.any(np.diff(pts) <= 0)):
        raise ValueError("Exact ordered source identities required")
    indices = np.searchsorted(pts, representatives)
    if (np.any(indices >= len(pts))
            or not np.array_equal(pts[np.minimum(indices,len(pts)-1)], representatives)
            or not np.array_equal(representatives[indices], representatives)
            or not np.array_equal(observations["shot_index"][indices], observations["shot_index"])):
        raise ValueError("Canonical must be present, self-representing and in same shot")
    reviewed = np.asarray(reviewed_joint_valid, bool)
    if reviewed.shape != (len(pts),17):
        raise ValueError("Reviewed exclusions must cover each source PTS and COCO17 joint")
    output = {key: np.asarray(value).copy() for key,value in observations.items()}
    for key in OBSERVATION_FIELDS:
        if np.asarray(observations[key]).shape[0] != len(pts):
            raise ValueError(f"Raw H field {key} does not cover source timeline")
        output["original_"+key] = np.asarray(observations[key]).copy()
        output[key] = np.asarray(observations[key])[indices].copy()
    output["effective_observation_source_pts"] = representatives.copy()
    output["effective_observation_alias_selection"] = pts != representatives
    alias = pts != representatives
    output["canonical_cached_observation_origin"] = output["observation_origin"].copy()
    output["observation_origin"] = output["observation_origin"].astype("<U64")
    output["observation_origin"][alias & output["valid"]] = (
        "confirmed_exposure_cached_representative_H")
    output["observation_origin"][alias & ~output["valid"]] = "missing_canonical_exposure_H"
    # Keep raw per-PTS decisions and apply the conservative intersection: a
    # canonical reviewed exclusion cannot disappear when an alias selects H.
    effective_review = reviewed & reviewed[indices]
    output["original_reviewed_joint_valid"] = reviewed.copy()
    output["effective_reviewed_joint_valid"] = effective_review.copy()
    recovered = alias & ~observations["valid"] & output["valid"]
    dropped = alias & observations["valid"] & ~output["valid"]
    return output, effective_review, dict(
        policy="confirmed_official_exposure_canonical_cached_H_package_candidate_v1",
        alias_presentations=int(alias.sum()),
        missing_canonical_observation_presentations=int((~output["valid"]).sum()),
        alias_valid_H_not_used_when_canonical_missing=int(dropped.sum()),
        alias_valid_H_not_used_source_pts=pts[dropped].astype(int).tolist(),
        original_missing_alias_H_reused_from_canonical_count=int(recovered.sum()),
        original_missing_alias_H_reused_source_pts=pts[recovered].astype(int).tolist(),
        original_per_pts_review_mask_exactly_preserved=True,
        effective_review_policy="own_PTS_review_AND_selected_canonical_review",
        effective_review_new_exclusions=int(np.sum(reviewed & ~effective_review)),
        reviewed_exclusions_restored=0,
        different_exposure_observations_modified=False,
        source_pts_and_shot_identity_preserved=True,
        missing_canonical_never_backfilled=True, averaging=False,
        network_inference=False, interpolation=False, production_default_changed=False)


def load_confirmed_evidence(groups_path: Path, mapping_path: Path,
                            source_shots_path: Path) -> tuple[dict,list[dict],dict]:
    groups = json.loads(Path(groups_path).read_text(encoding="utf-8-sig"))
    mapping = [json.loads(line) for line in Path(mapping_path).read_text(
        encoding="utf-8-sig").splitlines() if line]
    shots = json.loads(Path(source_shots_path).read_text(encoding="utf-8-sig"))
    hashes = {"groups":digest(groups_path),"mapping":digest(mapping_path),
              "source_shots":digest(source_shots_path)}
    expected = groups["input_sha256"]
    # Recorded paths can move; bind content by each evidence filename.
    for name,path in (("mapping",mapping_path),):
        values = [value for key,value in expected.items()
                  if Path(key).name == Path(path).name]
        if values != [hashes[name]]:
            raise ValueError(f"Official evidence {name} hash differs from frozen grouping")
    if groups["source_binding"]["full60"]["sha256"] != shots["source_sha256_from_timeline"]:
        raise ValueError("Official evidence and source manifest identify different videos")
    if hashes["groups"] != GROUPS_SHA256 or hashes["mapping"] != MAPPING_SHA256:
        raise ValueError("Frozen complete official grouping/mapping proof changed")
    if (shots["source_resolution"] != [1920,1080]
            or shots["source_time_base"] != "1/15360"
            or len(mapping) != shots["native_frame_count"]
            or [int(row["full_pts"]) for row in mapping] != list(
                range(*shots["window_pts"],256))):
        raise ValueError("Current source manifest changes original geometry/PTS window")
    for row in mapping:
        shot = shots["shots"][int(row["shot_index"])]
        if not shot["start_pts"] <= row["full_pts"] < shot["end_pts"]:
            raise ValueError("Current source manifest changes official per-PTS shot assignment")
    old_shot_hashes = [value for key,value in expected.items()
                       if Path(key).name == Path(source_shots_path).name]
    if len(old_shot_hashes) != 1:
        raise ValueError("Frozen grouping lacks original source-manifest hash")
    hashes["source_shots_recorded_at_official_grouping"] = old_shot_hashes[0]
    hashes["current_source_shots_file_differs_from_recorded"] = (
        hashes["source_shots"] != old_shot_hashes[0])
    hashes["current_manifest_validation"] = (
        "Frozen groups/mapping hashes checked; same source SHA, native dimensions/time base, "
        "complete native PTS window and every actual shot PTS assignment checked. "
        "Current analysis_scope excludes historical first 17 frames; recorded manifest SHA retained.")
    return groups,mapping,hashes
