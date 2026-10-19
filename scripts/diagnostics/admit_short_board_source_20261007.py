"""Append reviewed native short-board sources; retain every original row."""
from pathlib import Path
from copy import deepcopy
import json,hashlib
from short_board_global_source_20261007 import OUT,DATA

def read(p):return json.loads(Path(p).read_text(encoding="utf-8-sig"))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,a):Path(p).write_text(json.dumps(a,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")

def main():
    candidate=OUT/"short_board_multitime_source_candidate.json";source=read(candidate)
    circle_path=DATA/"observations/vision_training/continuous_review/cut6_clear_source_constraints/complete_source_clear_anchor_append_20261007/circle_parent_plus_actual_clear_anchors.json"
    material_path=DATA/"calibration/runs/solver_acceleration_20261003/structured_programme_source_candidate/additional_Ad12_25_native_material_points_FIT_v1.json"
    circle=read(circle_path);material=read(material_path)
    circle_out=OUT/"circle_parent_plus_short_board_multitime_20261007.json"
    material_out=OUT/"material_parent_plus_short_board_Ad25_Ad26_multitime_20261007.json"
    if circle_out.exists() or material_out.exists():raise FileExistsError("Preserve admitted derived source identity")
    identity_catalog=Path(r"D:/OneDrive/codex_projs/RinkSpace/configs/rink_model/ankara_2026_advertisements_user_confirmed_20260924.json")
    evidence=dict(kind="native_RGB_source_read_and_marker_review_with_permanent_same_event_user_print_catalog",
        identity_catalog_path=str(identity_catalog),identity_catalog_sha256=sha(identity_catalog),
        source_candidate_path=str(candidate),source_candidate_sha256=sha(candidate),
        native_marker_review_image=source["source_marker_review_image"],
        all_source_points_agent_RGB_measurements_not_human_pixel_GT=True,
        pixel_ellipse_centres_or_sizes_not_added_as_world_factors=True,
        current_user_request="all visible source-identified advertisements join shared world/camera correction; preserve physical IDs and internal/external material roles")
    new_circles=[]
    for r in source["circle_rows"]:
        if len(r["uv"])<3:continue
        row=deepcopy(r);row.update(source_review_authority=evidence,solver_admitted=True)
        if row["curve"]["advertisement_id"]==24:new_circles.append(row)
        else:material["rows"].append(row)
    circle["rows"].extend(new_circles)
    material["rows"].extend(dict(deepcopy(r),source_review_authority=evidence,solver_admitted=True) for r in source["shield_rows"])
    old_same_print=circle["geometry_contract"]["same_print_source_evidence"]
    if sha(old_same_print["path"])!=old_same_print["sha256"]:raise ValueError("Existing same-print authority changed")
    same_print=read(old_same_print["path"])
    same_print["source_frames"].extend(dict(advertisement_number=24,pts=r["pts"],source_file_sha256=r["source_file_sha256"],
        source_raw_frame=r["source_raw_frame"],additional_native_source_review=evidence) for r in new_circles)
    same_print_out=OUT/"same_print_Ad11_24_source_evidence_plus_short_board_20261007.json";write(same_print_out,same_print)
    circle["geometry_contract"]["same_print_source_evidence"]=dict(path=str(same_print_out),sha256=sha(same_print_out))
    for value,parent,path in ((circle,circle_path,circle_out),(material,material_path,material_out)):
        original=read(parent)
        if value["rows"][:len(original["rows"])]!=original["rows"]:raise AssertionError("Original source changed")
        value.update(status="explicit_new_fit_target_authorized",source_installed_in_solver=False,original_packets_modified=False,
            derived_from_native_source_parent=dict(path=str(parent),sha256=sha(parent)),additional_source_review=evidence)
        write(path,value)
    record=dict(schema="rinkspace.short-board-native-multitime-derived-source-admission.v1",active_config_or_state_modified=False,
        circle_packet=dict(path=str(circle_out),sha256=sha(circle_out),original_rows=len(read(circle_path)["rows"]),new_rows=len(new_circles)),
        material_packet=dict(path=str(material_out),sha256=sha(material_out),original_rows=len(read(material_path)["rows"]),new_rows=len(material["rows"])-len(read(material_path)["rows"])),
        original_all_rows_UV_sigma_roles_and_source_clock_preserved=True,
        source_scope="Ad24 outermost circular print stroke; Ad25 existing three named shield print points; Ad26 original shared q9 circle outermost print stroke",
        Ad26_requires_existing_material_factor_new_circle_kind_and_explicit_independent_station_Z_suffix=True,
        user_confirmed_template_dimensions_or_positions_not_rewritten=True,unobserved_or_occluded_source_points_not_created=True)
    write(OUT/"short_board_source_admission_record.json",record);print(json.dumps(record))

if __name__=="__main__":main()
