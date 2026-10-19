"""Restore saved RGB midpoint semantics without changing source observations.

This event-specific revision uses already stored profiles and reviewed native
images. It performs no decoding, detection, resampling, projection, or fitting.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[4]
DATA = Path("D:/CodexData/RinkSpace_Data/events/ankara_2026")
H = DATA / "diagnostics/current56_shared_active_set_filter_20261009/systematic_continuation_20261010"
PARENT = H / "source_pixel_review_additional/B2_original_center_and_LOW_Ad4_missing_anchor_systematic/uniform_full_cut_actual_RGB_candidates/FINAL_native_board191_original185_plus6_LOW_Ad4_actual_RGB_location_FIT_20261011.json"
PARENT_SHA256 = "09236c9f93b8daf73af34b949de3acb1c86623fecdf712d4906a029c7d4caea2"
PACKET = ROOT / "events/ankara_2026/observations/source_packets/board191_saved_RGB_midpoint_locations_20261011/packet.json"
RECEIPT = H / "source_pixel_review_additional/board_top_contact_material_semantics/board191_saved_RGB_midpoint_location_revision_20261011.json"
ROW_INDICES = (7, 8, 11, 12, 13, 14, 15, 16, 17, 18, 21, 22, 23, 104, 105, 106, 107)
MODE_FIELDS = ("boundary_location_observation", "boundary_location_measurement")
RECEIPT_SCHEMA = "rinkspace.saved-RGB-transition-midpoint-location-revision.v1"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf8")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def reference(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": digest(path)}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def exact(first, second, name):
    require(np.array_equal(np.asarray(first, float), np.asarray(second, float)), name + " changed")


def arithmetic_equal(first, second, name):
    first, second = np.asarray(first, float), np.asarray(second, float)
    tolerance = 64 * np.finfo(float).eps * max(1., float(np.max(np.abs(first))), float(np.max(np.abs(second))))
    require(first.shape == second.shape and np.max(np.abs(first - second)) <= tolerance, name + " differs from saved q50 arithmetic")


def visual_review(index):
    if index in (7, 8):
        return {"reviewer": "source_cache_audit", "date": "2026-10-11", "basis": "existing_native_RGB_opened_without_model", "point_scope": "all three stored x500/650/800 upper or lower outer-boundary scans", "finding": "samples lie on the visible purple panel outer boundary; athlete and internal lettering do not cover these sample windows"}
    if index in (12, 13):
        return {"reviewer": "source_cache_audit", "date": "2026-10-11", "basis": "existing_native_RGB_opened_without_model", "point_scope": "all three stored x1500/1650/1800 upper or lower scans", "finding": "sample windows lie on the single source-visible external panel transition; no body or interior lettering in those windows"}
    if index in (15, 16):
        return {"reviewer": "source_cache_audit", "date": "2026-10-11", "basis": "existing_native_RGB_opened_without_model", "point_scope": "both stored x1820/1880 upper or lower scans", "finding": "partial panel outer boundary is visible; right edge outside the image is not measured"}
    if index in (17, 18):
        return {"reviewer": "source_cache_audit", "date": "2026-10-11", "basis": "existing_native_RGB_opened_without_model", "point_scope": "left y310/322/334 or right y306/316 scans only", "finding": "listed scan windows show the magenta left or cyan right external transition; sampled right windows are above the head, lower head-overlapped right boundary is excluded"}
    if index in (11, 14):
        return {"reviewer": "root", "date": "2026-10-11", "basis": "existing_native_RGB_opened_without_model", "point_scope": "all stored y570/640/710/760 scans", "finding": "continuous white-to-magenta outer left transition; horizontal blur is present, no interior lettering or athlete overlaps the listed scans"}
    if index in (21, 22, 23):
        return {"reviewer": "root", "date": "2026-10-11", "basis": "existing_native_RGB_opened_without_model", "point_scope": "left y365/455; upper and lower x600/1650", "finding": "listed windows are on the blue printed block external boundary with no athlete or interior lettering; horizontal blur remains"}
    return {"basis": "existing_packet_actual_source_visual_review", "scope": "all original output points, excluding declared hidden graphic/body/corner spans"}


def qualify_row(row, index):
    require(row["split"].lower() == "fit", "HELD cannot be revised")
    require(not any(key in row for key in MODE_FIELDS), "row already has a location operator")
    uv = np.asarray(row["uv"], float)
    normals = np.asarray(row["normal"], float)
    intervals = np.asarray(row["normal_interval_px"], float)
    require(uv.ndim == 2 and uv.shape[1] == 2 and uv.shape == normals.shape == intervals.shape, "source point arrays disagree")
    proofs = []
    if "source_scan_profiles" in row:
        field = "source_scan_profiles"
        identity = row["source_identity_authority"]
        require("external" in str(identity).lower(), "no external material identity")
        require(row.get("model_pixels_used_in_source_measurement") is False, "model-derived pixels are ineligible")
        for point, scan in enumerate(row[field]):
            require(scan.get("selection_uses_model_camera_or_residual") is False, "scan selected by model")
            axis = "x" if "q10_q50_q90_x" in scan else "y"
            q = np.asarray(scan["q10_q50_q90_" + axis], float)
            measured = [q[1], scan["source_y"]] if axis == "x" else [scan["source_x"], q[1]]
            exact(uv[point], measured, "source_scan UV")
            exact(normals[point], [1, 0] if axis == "x" else [0, 1], "source_scan normal")
            arithmetic_equal(intervals[point], q[[0, 2]] - q[1], "source_scan interval")
            mass_keys = [key for key in scan if "gradient_mass" in key]
            require(len(mass_keys) == 1 and np.sum(scan[mass_keys[0]]) > 0, "missing signed source transition mass")
            proofs.append({"point_index": point, "source_profile_canonical_json_sha256": hashlib.sha256(canonical(scan)).hexdigest(), "profile_pointer": f"/rows/{index}/{field}/{point}", "scan_axis": axis, "source_roi_xyxy": scan["source_roi_xyxy"], "q10_q50_q90": q.tolist(), "source_quality_flags": None, "quality_flags_absent_not_assumed_false": True})
    elif "measured_RGB_sources" in row:
        field = "measured_RGB_sources"
        identity = row["physical_identity_authority"]
        require(identity.get("authority_is_not_model_projection_or_print_similarity_alone") is True, "unqualified material identity")
        require("external" in str(identity).lower(), "no external material identity")
        for point, scan in enumerate(row[field]):
            require(scan["curve_role"] == "printed_colour_block_external_border", "different material edge")
            require(scan["interval_is_optical_transition_support_not_accuracy_bound"] is True, "different interval meaning")
            q = np.asarray(scan["actual_source_transition_q10_q50_q90"], float)
            axis = scan["source_scan_axis"]
            require(axis in ("x", "y") and scan["source_transition_native_gray_strength"] > 0, "missing RGB transition")
            exact(uv[point], scan["actual_source_uv"], "actual RGB UV")
            exact(uv[point, 0 if axis == "x" else 1], q[1], "actual RGB q50")
            exact(normals[point], scan["normal"], "actual RGB normal")
            exact(intervals[point], scan["normal_interval_px"], "actual RGB interval")
            arithmetic_equal(intervals[point], q[[0, 2]] - q[1], "actual RGB q50 interval")
            proofs.append({"point_index": point, "source_profile_canonical_json_sha256": hashlib.sha256(canonical(scan)).hexdigest(), "profile_pointer": f"/rows/{index}/{field}/{point}", "scan_axis": axis, "q10_q50_q90": q.tolist(), "source_native_RGB_profile_sha256": scan["source_native_RGB_profile_sha256"], "source_quality_flags": None, "quality_flags_absent_not_assumed_false": True})
    elif "reader_source_proof" in row:
        field = "reader_source_proof"
        identity = row["physical_identity_basis"]
        require(row["measurement_operator"] == "integrated_actual_signed_RGB_luminance_transition_q50", "different reader operator")
        require(row["direct_user_UV"] is False and row["model_UV_not_used_as_measurements"] is True, "not independent measured RGB")
        review = row["actual_source_visual_review"]
        require(review["all_output_points_actually_viewed"] is True and review["no_visible_point_source_pixel_error_found"] is True, "missing existing output-point review")
        for point, scan in enumerate(row[field]):
            flags = scan["source_flags"]
            require(flags and all(value is False for value in flags.values()), "selected source transition has an adverse quality flag")
            require(scan["disposition"] == "measured_material_transition" and scan["peak_position_rule"] == "integrated_actual_signed_luminance_transition_q50", "ambiguous source transition")
            exact(uv[point], scan["actual_source_uv"], "reader actual UV")
            exact(normals[point], scan["source_search_normal"], "reader normal")
            exact(intervals[point], scan["observed_normal_interval_px"], "reader observed interval")
            q = np.asarray(scan["source_q10_q50_q90_normal_offsets_px"], float)
            arithmetic_equal(uv[point], np.asarray(scan["source_search_point_uv"]) + normals[point] * q[1], "reader q50 UV")
            arithmetic_equal(intervals[point], q[[0, 2]] - q[1], "reader q50 interval")
            proofs.append({"point_index": point, "source_query_index": scan["source_query_index"], "source_profile_canonical_json_sha256": hashlib.sha256(canonical(scan)).hexdigest(), "profile_pointer": f"/rows/{index}/{field}/{point}", "q10_q50_q90_normal_offsets_px": q.tolist(), "source_quality_flags": flags, "disposition": scan["disposition"], "source_branch_selection_rule": scan.get("source_branch_selection_rule"), "broad_mixed_signed_mass_not_used_as_output_UV": scan.get("broad_mixed_signed_mass_not_used_as_output_UV")})
    else:
        raise ValueError("No saved q50 measurement proof")
    require(len(proofs) == len(uv), "one saved source proof is required per original point")
    for proof in proofs:
        q = np.asarray(proof.get("q10_q50_q90", proof.get("q10_q50_q90_normal_offsets_px")), float)
        require(q.shape == (3,) and np.isfinite(q).all() and q[0] <= q[1] <= q[2], "invalid transition quantiles")
    measured = {key: deepcopy(row[key]) for key in ("id", "split", "curve", "uv", "normal", "normal_interval_px", "sigma_px", "source_raw_frame", "source_file_sha256", "source_visible_paths", "point_integration_mass") if key in row}
    measured["actual_pts"] = int(row.get("actual_pts", row["pts"]))
    return {"original_row_index": index, "id": row["id"], "split": row["split"], "source_row_canonical_json_sha256": hashlib.sha256(canonical(row)).hexdigest(), "measurement_field": field, "measured_boundary_row": measured, "physical_identity_authority": deepcopy(identity), "quality_review": visual_review(index), "saved_q50_point_proofs": proofs, "source_sigma_and_optical_support_unchanged": True, "sigma_is_not_new_measured_covariance": True}


def save_new(path, value):
    path = Path(path)
    encoded = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf8")
    if path.exists():
        require(path.read_bytes() == encoded, "refusing to overwrite different frozen revision: " + str(path))
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(encoded)


def build_revision(parent_path=PARENT, packet_path=PACKET, receipt_path=RECEIPT, row_indices=ROW_INDICES):
    parent_ref = reference(parent_path)
    require(parent_ref["sha256"] == PARENT_SHA256, "this review belongs to a different parent packet")
    parent = json.loads(Path(parent_path).read_text(encoding="utf8"))
    require(len(parent["rows"]) == 191, "unexpected parent source row count")
    require(len(set(row_indices)) == len(row_indices) and all(index in ROW_INDICES for index in row_indices), "unreviewed or duplicate source row")
    records = [qualify_row(parent["rows"][index], index) for index in row_indices]
    images = {}
    for record in records:
        measured = record["measured_boundary_row"]
        path = measured["source_raw_frame"]
        if path not in images:
            images[path] = reference(path)
        require(images[path]["sha256"] == measured["source_file_sha256"], "original native PNG changed")
        record["source_row_reference"] = {**parent_ref, "pointer": f"/rows/{record['original_row_index']}"}
    receipt = {"schema": RECEIPT_SCHEMA, "parent_packet_reference": parent_ref, "reproducer_reference": reference(__file__), "changed_row_indices": list(row_indices), "changed_row_ids": [record["id"] for record in records], "points": sum(len(record["measured_boundary_row"]["uv"]) for record in records), "rows": records, "native_image_references": list(images.values()), "source_image_decode_or_new_scan_performed": False, "model_state_projection_or_residual_used_for_qualification": False, "all_original_UV_sigma_normal_optical_support_split_and_point_order_unchanged": True, "human_support_and_HELD_roles_unchanged": True, "qualification_is_soft_measured_location_not_subpixel_accuracy_or_hard_guard": True}
    save_new(receipt_path, receipt)
    receipt_ref = reference(receipt_path)
    revised = deepcopy(parent)
    for index in row_indices:
        revised["rows"][index][MODE_FIELDS[0]] = "measured_transition_midpoint"
        revised["rows"][index][MODE_FIELDS[1]] = receipt_ref
    revised["source_location_operator_revision"] = {"parent_packet_reference": parent_ref, "receipt_reference": receipt_ref, "changed_row_indices": list(row_indices), "changed_row_ids": [record["id"] for record in records], "all_original_source_rows_except_two_location_fields_unchanged": True}
    strip = lambda row: {key: value for key, value in row.items() if key not in MODE_FIELDS}
    require(all(canonical(strip(before)) == canonical(strip(after)) for before, after in zip(parent["rows"], revised["rows"])), "original observation fields changed")
    require(all(canonical(before) == canonical(after) for index, (before, after) in enumerate(zip(parent["rows"], revised["rows"])) if index not in row_indices), "unselected or HELD source changed")
    require(canonical({key: value for key, value in revised.items() if key != "source_location_operator_revision"}) == canonical({**parent, "rows": revised["rows"]}), "other packet top fields changed")
    save_new(packet_path, revised)
    return {"parent_packet": parent_ref, "packet": reference(packet_path), "receipt": receipt_ref, "changed_row_indices": list(row_indices), "points": receipt["points"], "all_original_source_fields_except_two_location_fields_unchanged": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, default=PARENT)
    parser.add_argument("--packet", type=Path, default=PACKET)
    parser.add_argument("--receipt", type=Path, default=RECEIPT)
    args = parser.parse_args()
    print(json.dumps(build_revision(args.parent, args.packet, args.receipt), ensure_ascii=False))


if __name__ == "__main__":
    main()
