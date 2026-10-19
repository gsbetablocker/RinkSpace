"""Derive a source-image-reviewed correction from a frozen board-support packet.

The reviewed source columns are removed, never replaced by a projection or an
interpolated yellow edge.  Original video PTS, LK UV, masks and old packets stay
unchanged.  The review file names exact source PNGs and explicit x/foot ranges;
those ranges are evidence identities, not a universal board-height threshold.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

try:
    from . import source_board_support as support
    from .render_result import DEFAULT_MASKS, DEFAULT_PEOPLE, PersonMasks
except ImportError:
    import source_board_support as support
    from render_result import DEFAULT_MASKS, DEFAULT_PEOPLE, PersonMasks


def correct(parent_path: Path, parent_sha256: str, evidence_path: Path,
            motion_path: Path, review_path: Path, output_dir: Path) -> dict:
    if support._sha256(parent_path) != parent_sha256:
        raise ValueError("Parent support SHA mismatch")
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    if parent["schema"] != "rinkspace-source-board-motion-support-v1":
        raise ValueError("Unexpected parent schema")
    if parent["detector_module"]["sha256"] != support._sha256(Path(support.__file__)):
        raise ValueError("Frozen pixel detector implementation changed")
    if len(parent["source_files"]) != 1:
        raise ValueError("Expected one native motion source")
    if (Path(parent["source_files"][0]["path"]).resolve() != motion_path.resolve()
            or parent["source_files"][0]["sha256"] != support._sha256(motion_path)):
        raise ValueError("Native LK source SHA mismatch")
    review = json.loads(review_path.read_text(encoding="utf-8"))
    if (review["schema"] != "rinkspace-source-board-band-review-v1"
            or review["parent_source_support_sha256"] != parent_sha256):
        raise ValueError("Review does not bind frozen parent")
    with np.load(evidence_path, allow_pickle=False) as z:
        if any(Path(row["source_support_evidence_path"]).resolve() != evidence_path.resolve()
               or row["source_support_evidence_sha256"] != support._sha256(evidence_path)
               for row in parent["frames"]):
            raise ValueError("Frozen accepted-column evidence differs")
        pts = z["pts"].copy()
        raw_offsets = z["raw_offsets"].copy()
        raw_columns = z["raw_columns"].copy()
        accepted_offsets = z["accepted_offsets"].copy()
        accepted_columns = z["accepted_columns"].copy()
    if list(map(int, pts)) != [int(row["pts"]) for row in parent["frames"]]:
        raise ValueError("Exact frame PTS differs from evidence")
    by_pts = {int(row["pts"]): row for row in parent["frames"]}
    if len(by_pts) != len(parent["frames"]):
        raise ValueError("Duplicate frame PTS")
    rejected = {stamp: np.zeros(int(accepted_offsets[i + 1] - accepted_offsets[i]), bool)
                for i, stamp in enumerate(pts)}
    reviewed_cases = []
    seen = set()
    for case in review["cases"]:
        stamp = int(case["pts"])
        if stamp not in by_pts or stamp in seen:
            raise ValueError("Review PTS absent or duplicated")
        seen.add(stamp)
        image_path = Path(case["source_png_path"])
        if support._sha256(image_path) != case["source_png_sha256"]:
            raise ValueError(f"Reviewed PNG SHA mismatch at {stamp}")
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None or image.shape != (1080, 1920, 3):
            raise ValueError(f"Reviewed source size differs at {stamp}")
        if hashlib.sha256(image.tobytes()).hexdigest() != by_pts[stamp]["raw_bgr_sha256"]:
            raise ValueError(f"Reviewed PNG differs from decoded original PTS {stamp}")
        i = int(by_pts[stamp]["evidence_index"])
        part = accepted_columns[accepted_offsets[i]:accepted_offsets[i + 1]]
        columns = []
        for band in case["false_band_intervals"]:
            x0, x1 = map(float, band["x_range_inclusive_px"])
            foot0, foot1 = map(float, band["observed_false_foot_y_range_px"])
            if x0 > x1 or foot0 > foot1:
                raise ValueError("Invalid reviewed source-band range")
            ids = np.flatnonzero((part[:, 0] >= x0) & (part[:, 0] <= x1)
                                  & (part[:, 2] >= foot0) & (part[:, 2] <= foot1))
            if not len(ids) or np.any(rejected[stamp][ids]):
                raise ValueError("Review band empty or overlaps another band")
            rejected[stamp][ids] = True
            columns.append({"x_range_inclusive_px": [x0, x1],
                            "observed_false_foot_y_range_px": [foot0, foot1],
                            "accepted_column_indices_in_frame": ids.astype(int).tolist(),
                            "source_x_px": part[ids, 0].astype(float).tolist(),
                            "source_false_foot_y_px": part[ids, 2].astype(float).tolist()})
        reviewed_cases.append({"pts": stamp, "source_png_path": str(image_path),
                               "source_png_sha256": case["source_png_sha256"],
                               "reason": case["reason"], "columns": columns})
    if len(reviewed_cases) != len(review["cases"]):
        raise ValueError("Incomplete source review")
    corrected_parts = []
    corrected_offsets = [0]
    for i, stamp in enumerate(pts):
        part = accepted_columns[accepted_offsets[i]:accepted_offsets[i + 1]]
        kept = part[~rejected[int(stamp)]]
        corrected_parts.append(kept)
        corrected_offsets.append(corrected_offsets[-1] + len(kept))
    output_dir.mkdir(parents=True, exist_ok=True)
    corrected_evidence = output_dir / "source_board_support_columns_reviewed.npz"
    np.savez_compressed(corrected_evidence, pts=pts,
                        raw_offsets=raw_offsets, raw_columns=raw_columns,
                        accepted_offsets=np.asarray(corrected_offsets, np.int64),
                        accepted_columns=np.concatenate(corrected_parts))
    evidence_sha = support._sha256(corrected_evidence)
    people = PersonMasks(DEFAULT_MASKS, DEFAULT_PEOPLE)
    masks = {}
    for stamp in seen:
        mask, status = people.get(stamp, require_sam2=True)
        if (status != by_pts[stamp]["sam2_status"]
                or hashlib.sha256(mask.tobytes()).hexdigest()
                != by_pts[stamp]["sam2_mask_sha256"]):
            raise ValueError("SAM2 mask changed for reviewed PTS")
        masks[stamp] = support._source_mask(mask)
    with np.load(motion_path, allow_pickle=False) as z:
        motion = {key: z[key].copy() for key in ("pair_pts", "offsets", "xy0", "xy1")}
    corrected_pairs = []
    removed_points = []
    for pair in parent["pairs"]:
        row = copy.deepcopy(pair)
        a, b = map(int, row["pts"])
        i = int(row["source_pair_index"])
        if list(map(int, motion["pair_pts"][i])) != [a, b]:
            raise ValueError("Original LK pair PTS mismatch")
        local = np.asarray(row["source_point_indices"], int)
        global_ids = np.asarray(row["source_global_point_indices"], int)
        if not np.array_equal(local + int(motion["offsets"][i]), global_ids):
            raise ValueError("Original LK local/global index mismatch")
        keep = np.ones(len(local), bool)
        for stamp, key in ((a, "xy0"), (b, "xy1")):
            if stamp not in seen:
                continue
            frame_index = int(by_pts[stamp]["evidence_index"])
            part = corrected_parts[frame_index]
            keep &= support.classify_points(motion[key][global_ids],
                                            {"accepted": part}, masks[stamp])
        for old_id in global_ids[~keep]:
            removed_points.append({"source_file_id": row["source_file_id"],
                                   "source_pair_index": i,
                                   "source_global_point_index": int(old_id),
                                   "pts": [a, b]})
        retained = global_ids[keep]
        row["source_point_indices"] = local[keep].astype(int).tolist()
        row["source_global_point_indices"] = retained.astype(int).tolist()
        row["support_count"] = len(retained)
        for key, uv_key in (("source_x_range", "xy0"),
                            ("target_x_range", "xy1")):
            x = motion[uv_key][retained, 0]
            row[key] = [float(x.min()), float(x.max())] if len(x) else None
        row["coverage_note"] = ("local_only_below8" if 0 < len(retained) < 8
                                else "no_supported_pair" if not len(retained)
                                else "spans_to_be_reviewed")
        corrected_pairs.append(row)
    result = copy.deepcopy(parent)
    result["parent_source_support_sha256"] = parent_sha256
    result["source_point_selection"] = "original_v13_selected_hint0_with_reviewed_source_columns"
    result["source_support_correction_review"] = {
        "path": str(review_path), "sha256": support._sha256(review_path),
        "cases": reviewed_cases,
        "false_column_count": int(sum(np.count_nonzero(value) for value in rejected.values())),
        "removed_point_count": len(removed_points),
        "removed_points": removed_points}
    result["correction_module"] = {"path": str(Path(__file__).resolve()),
                                   "sha256": support._sha256(Path(__file__))}
    result["frames"] = copy.deepcopy(parent["frames"])
    for row in result["frames"]:
        stamp = int(row["pts"])
        row["accepted_source_columns"] -= int(np.count_nonzero(rejected[stamp]))
        row["source_support_evidence_path"] = str(corrected_evidence)
        row["source_support_evidence_sha256"] = evidence_sha
    result["pairs"] = corrected_pairs
    result["both_end_supported_pixels"] = int(sum(row["support_count"]
                                                    for row in corrected_pairs))
    result["qa_overlays"] = {}  # Parent overlays predate this review.
    result["limits"].append("Reviewed false color runs removed; no substitute lower edge inferred")
    path = output_dir / "source_board_motion_support_reviewed.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return {"path": str(path), "sha256": support._sha256(path),
            "evidence_path": str(corrected_evidence), "evidence_sha256": evidence_sha,
            "false_columns": result["source_support_correction_review"]["false_column_count"],
            "removed_points": len(removed_points),
            "remaining_points": result["both_end_supported_pixels"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--parent-sha256", required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(correct(args.parent, args.parent_sha256, args.evidence,
                             args.motion, args.review, args.output), indent=2))


if __name__ == "__main__":
    main()
