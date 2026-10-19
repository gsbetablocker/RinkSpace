"""Independently verify source-only cut5 RGB review union against original rows.

Loads JSON and numeric NPZ arrays only. No model, camera, solver, or residual.
The certificate confirms material identity; it does not recertify flow precision.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def checked(ref):
    assert sha(ref["path"]) == ref["sha256"], ref["path"]
    return Path(ref["path"])


def numeric(path, names):
    with np.load(path, allow_pickle=False) as z:
        return {name: z[name].copy() for name in names}


def verify(review_dir, candidate_path):
    receipt_path = review_dir / "actual_original_cut5_9288_region_candidates_RGB_receipt_v1.json"
    assert sha(receipt_path) == "baf8a32a2bedc7a9aaf4e3c4e2a9260242091398be99e1dff42fe00f5311ee07"
    receipt, candidate = read(receipt_path), read(candidate_path)
    assert sha(candidate_path) == "cac37b8c14d72bf46421b38a845736f5535668deeab6e41598396835a6e16954"
    contracts = [
        ("independent_pages000_086_material_review_v1.json", "5e1d81963c20046451ab6000d90c8cd4d017e8d7343a822b13d2c43b082e2708", "qualified_source_only_rows", range(0, 87)),
        ("independent_middle_087_173_material_review_SOURCE_ONLY_v1.json", "ec9fc38d288d2a3383256d6f7d12b9510f06c3d5ece65f5c0a0b6619048cc8b6", "qualified_material_points", range(87, 174)),
        ("root_pages174_259_all_original_points_actual_RGB_material_QUALIFIED_v1.json", "545e645cf75d595c93eccb572414fcbfdcc7a58fdfae510bf46b3c5d798fb4c1", "records", range(174, 260)),
    ]
    expected = {int(r["current_original_prefix_bundle_index"]): r for r in receipt["point_to_pair_sheet"]}
    assert len(expected) == len(receipt["point_to_pair_sheet"]) == 9288
    original_candidates = {int(r["current_original_prefix_bundle_index"]): r for r in candidate["records"]}
    assert len(original_candidates) == 15218 and set(expected) <= set(original_candidates)
    merged, group_reports, review_sets = {}, [], []
    metadata = {"material_identity_review", "both_end_material_reviewed"}
    for filename, digest, key, pages in contracts:
        path = review_dir / filename
        assert sha(path) == digest, filename
        doc = read(path)
        if key == "qualified_source_only_rows":
            assert not doc["uncertain_source_only_rows"] and not doc["remaining_sheet_indices"]
            assert doc["source_model_residual_not_used_for_material_selection"] is True
            assert doc["source_UV_sigma_PTS_weights_rows_and_runtime_roles_modified"] is False
            assert doc["exact_subpixel_matching_uncertainty_not_certified_from_blurred_RGB"] is True
        elif key == "qualified_material_points":
            assert not doc["unknown_points"] and not doc["rejected_points"]
            assert doc["review_scope"]["model_geometry_camera_residual_J_or_HELD_used_to_select_or_qualify_points"] is False
            assert doc["review_scope"]["subpixel_flow_match_accuracy_or_sigma_remeasured"] is False
        else:
            assert doc["unknown_material_points"] == doc["nearfield_equipment_points"] == doc["people_or_UI_or_ice_points"] == 0
            assert doc["original_source_row_UV_sigma_PTS_split_weights_changed"] is False
            assert doc["subpixel_correspondence_or_exposure_centroid_accuracy_certified"] is False
        rows = doc[key]
        page_set = set(pages)
        ids = set()
        for row in rows:
            index = int(row["current_original_prefix_bundle_index"])
            assert index in expected and index not in ids and index not in merged
            base = expected[index]
            assert int(row["sheet_index"]) in page_set
            for field, value in base.items():
                if field not in metadata:
                    assert row[field] == value, (filename, index, field)
            for field, value in original_candidates[index].items():
                if field not in metadata:
                    assert row[field] == value, (filename, index, field)
            assert row["role_status"] == "original_unknown_depth_FIT; both_native_RGB_material_inspection_pending"
            ids.add(index)
            merged[index] = row
        assert ids == {i for i, r in expected.items() if int(r["sheet_index"]) in page_set}
        assert {int(r["sheet_index"]) for r in rows} == page_set
        review_sets.append(ids)
        group_reports.append({"path": str(path), "sha256": digest, "point_count": len(ids), "sheet_count": len(page_set)})
    assert set(merged) == set(expected) and sum(map(len, review_sets)) == 9288
    for sheet in receipt["sheets"]:
        assert sha(sheet["path"]) == sheet["sha256"]

    admission = numeric(checked(candidate["original_source_ledger"]), ("active_original_global_index", "source_uv_a", "source_uv_b", "original_sigma_xy"))
    selected = numeric(checked(candidate["original_selected_rows"]), ("original_global_row_indices", "native_pair_pts"))
    raw = numeric(checked(candidate["original_raw_motion"]), ("pair_pts", "offsets", "xy0", "xy1"))
    aliases_npz = numeric(checked(candidate["clock_alias_authority"]), ("pts", "exposure_representative_pts"))
    aliases = dict(zip(map(int, aliases_npz["pts"]), map(int, aliases_npz["exposure_representative_pts"])))
    for index, row in merged.items():
        full = int(row["original_full_low_bundle_row"])
        selected_row = int(row["original_selected_cut5_row"])
        raw_row = int(row["original_raw_cut5_point_index"])
        pair = int(row["original_raw_pair_index"])
        assert int(admission["active_original_global_index"][index]) == full == 31987 + selected_row
        assert int(selected["original_global_row_indices"][selected_row]) == raw_row
        assert raw_row == int(raw["offsets"][pair]) + int(row["original_raw_pair_local_index"])
        for name, source in (("source_uv", "source_uv_a"), ("target_uv", "source_uv_b"), ("source_sigma_px", "original_sigma_xy")):
            assert np.array_equal(np.asarray(row[name]), admission[source][full]), (index, name)
        assert np.array_equal(row["source_uv"], raw["xy0"][raw_row])
        assert np.array_equal(row["target_uv"], raw["xy1"][raw_row])
        assert np.array_equal(row["source_native_pair_pts"], raw["pair_pts"][pair])
        assert np.array_equal(selected["native_pair_pts"][pair], raw["pair_pts"][pair])
        assert row["source_pts"] == aliases[int(raw["pair_pts"][pair, 0])]
        assert row["target_pts"] == aliases[int(raw["pair_pts"][pair, 1])]

    old274 = {int(r["global_bundle_index"]) for r in read(checked(candidate["original_typed_prefix_contract"]))}
    low53 = {int(r.get("current_original_prefix_bundle_index", r.get("original_global_bundle_index"))) for r in read(checked(candidate["existing_low53_role_packet"]))["records"]}
    retired = {int(r["current_bundle_motion_row"]) for r in read(checked(candidate["static_UI_retirement"]))["records"]}
    ice_raw = {int(r["original_global_index"]) for r in read(checked(candidate["original_ice_source"]))["qualified_rows"] if r["split"] == "fit"}
    original_ice3 = set(map(int, candidate["existing_typed_indices_excluded"]["original_ice3"]))
    assert old274 == set(candidate["existing_typed_indices_excluded"]["old274"]) and len(old274) == 274
    assert low53 == set(candidate["existing_typed_indices_excluded"]["admitted_low53"]) and len(low53) == 53
    assert original_ice3 == {113623, 113625, 113636}
    assert not set(merged) & (old274 | low53 | retired | original_ice3)
    assert not {int(r["original_raw_cut5_point_index"]) for r in merged.values()} & ice_raw
    return {
        "schema": "independent.original-cut5-material-review-union-source-identity.v1",
        "status": "all_assertions_passed_source_only_no_runtime_admission",
        "source_receipt": {"path": str(receipt_path), "sha256": sha(receipt_path)},
        "candidate_packet": {"path": str(candidate_path), "sha256": sha(candidate_path)},
        "reviews": group_reports,
        "counts": {"qualified_union": len(merged), "actual_pairs": len({r["original_raw_pair_index"] for r in merged.values()}), "pages": len(receipt["sheets"]), "overlaps": 0, "missing_expected_points": 0, "unreviewed_points_added": 0, "unknown_or_rejected_rows_silently_added": 0},
        "original_UV_sigma_native_and_actual_PTS_prefix_raw_selected_indices_role_status_bitexact": True,
        "metadata_only_material_review_status_fields_may_differ": sorted(metadata),
        "existing_source_exclusion_counts": {"old274": len(old274), "low53": len(low53), "original_ice3": len(original_ice3), "retired": len(retired), "known_ice_raw_FIT": len(ice_raw)},
        "intersections_with_all_existing_source_sets": 0,
        "subpixel_correspondence_or_sigma_recaptured": False,
        "geometry_camera_model_residual_J_or_solver_loaded": False,
        "source_runtime_manifest_or_packet_modified": False,
        "qualified_original_prefix_indices": sorted(merged),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--review-dir", type=Path, required=True)
    ap.add_argument("--candidate", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    out = verify(a.review_dir, a.candidate)
    out["verifier"] = {"path": str(Path(__file__).resolve()), "sha256": sha(__file__)}
    a.output.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"path": str(a.output), "sha256": sha(a.output), "counts": out["counts"]}))


if __name__ == "__main__":
    main()
