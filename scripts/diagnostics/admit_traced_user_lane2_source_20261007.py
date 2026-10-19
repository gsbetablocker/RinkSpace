"""Prepare inactive FIT/HELD source-only Lane2 profile correspondences."""
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import av
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
DATA=Path(r"D:/CodexData/RinkSpace_Data/events/ankara_2026")
BASE=DATA/"observations/vision_training/continuous_review/cut6_clear_source_constraints/complete_source_clear_anchor_append_20261007"
PARENT=BASE/"ice_clear_anchors_user_lane2_lane3_20261007.json"
TRACE=DATA/"diagnostics/external_expert_20261007/user_lane_source_interval_trace/user_lane_source_normal_profile_trace.json"
SOURCE=DATA/"media/originals/oxXu1avhrvs.mp4"
OUTPUT=BASE/"ice_clear_anchors_user_lane2_lane3_traced_lane2_v2_20261007.json"
STAMPS=[230502144,230503168,230504192,230505216,230506240,230506752,230507264,230507776]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def main():
    if OUTPUT.exists():raise FileExistsError("Preserve derived source packet identity")
    original=read(PARENT);trace=read(TRACE)
    if trace["original_user_source_packet_sha256"]!=sha(PARENT):raise ValueError("Original user input changed")
    decoded={}
    with av.open(str(SOURCE)) as container:
        stream=container.streams.video[0];stream.thread_count=1
        container.seek(min(STAMPS)-15360,stream=stream,backward=True,any_frame=False)
        for f in container.decode(stream):
            if f.pts>max(STAMPS):break
            if f.pts in STAMPS:decoded[int(f.pts)]=f.to_ndarray(format="bgr24")
    if set(decoded)!=set(STAMPS):raise ValueError("Missing actual source frames")
    receipts={r["pts"]:r for r in trace["decoded_native_receipts"]}
    evidence=DATA/"evidence/user_annotations_20261007/lane2_source_profile_correspondence"
    evidence.mkdir(parents=True,exist_ok=True)
    user=next(r for r in original["rows"] if r["id"]=="user_native230501120_lane2_upper_long_20261007")
    rows=[]
    for j,pts in enumerate(STAMPS):
        h=next(h for h in trace["per_pair_source_profile_trace"] if h["to_pts"]==pts)
        measured=[r for r in h["source_axis_normal_tracking"] if r["original_user_index"] in (0,1) and r["chain_alive"]]
        if not measured:raise ValueError(f"No supported Lane2 source points at {pts}")
        bgr=decoded[pts]
        if hashlib.sha256(bgr.tobytes()).hexdigest()!=receipts[pts]["native_BGR_sha256"]:
            raise ValueError("Profile correspondence actual RGB changed")
        frame=evidence/f"native_{pts}.png"
        if frame.exists():
            if not np.array_equal(cv2.imread(str(frame)),bgr):raise ValueError("Existing source evidence differs")
        elif not cv2.imwrite(str(frame),bgr):raise RuntimeError("Could not preserve actual source frame")
        uv=np.array([r["source_axis_uv"] for r in measured])
        if len(uv)==2:
            tangent=uv[1]-uv[0];tangent/=np.linalg.norm(tangent)
            normal=np.tile([-tangent[1],tangent[0]],(len(uv),1))
            normal_origin="TLS tangent of two independently RGB-profile-corresponded axis points"
        else:
            normal=np.array([r["profile_match"]["source_normal"] for r in measured])
            normal_origin="actual source normal-profile scan axis; tangent aperture uncertainty retained"
        split="fit" if j%2==0 else "holdout"
        rows.append(dict(id=f"user_traced_native{pts}_lane2_upper_long_{split}_20261007",pts=pts,actual_pts=pts,
            feature_id="lane2",curve=deepcopy(user["curve"]),geometry_role=user["geometry_role"],split=split,
            uv=uv.tolist(),normal=normal.tolist(),source_normal_origin=normal_origin,
            sigma_px=[4.]*len(uv),sigma_semantics="4 native px original user-axis engineering floor with source-profile correspondence; not calibrated variance",
            source_raw_frame=str(frame),source_file_sha256=sha(frame),
            source_native_RGB_sha256=hashlib.sha256(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB).tobytes()).hexdigest(),
            sourceUV_origin="original registered user side axis transported by actual native RGB normal-profile correspondence",
            source_RGB_actual_measurement=True,user_confirmed_source_axis=False,
            source_axis_identity_from_confirmed_user_via_RGB_continuity=True,
            physical_identity_authority=dict(kind="direct_user_side_identity_plus_independent_actual_RGB_normal_profile_sequence",
                original_user_row_id=user["id"],original_user_authority=deepcopy(user["physical_identity_authority"]),
                source_trace_path=str(TRACE),source_trace_sha256=sha(TRACE),
                identity_not_from_model_or_agent_arrow_interpretation=True,later_frame_not_direct_new_user_confirmation=True),
            source_profile_correspondences=measured,
            source_optical_normal_support_authorized=False,
            profile_matching_support_is_not_physical_paint_width_or_statistical_accuracy=True,
            model_projection_used_to_generate_or_select_uv=False,endpoint_observation=False,
            inactive_candidate_packet_only=True))
    result=deepcopy(original);result["rows"].extend(rows)
    if result["rows"][:len(original["rows"])]!=original["rows"]:raise AssertionError("Original active rows changed")
    counts={role:sum(2*len(r["uv"]) for r in rows if r["split"]==role) for role in ("fit","holdout")}
    result.update(status="INACTIVE_SOURCE_CANDIDATE_ORIGINAL_ALL_ROWS_PLUS_USER_SIDE_RGB_PROFILE_TRACE",
        source_installed_in_solver=False,original_packets_modified=False,
        source_continuous_user_axis_trace=dict(path=str(TRACE),sha256=sha(TRACE),source_only=True,
            parent_packet_path=str(PARENT),parent_packet_sha256=sha(PARENT),
            native_display_interval=[230501120,230507776],raw_frame_time_base="1/15360",
            appended_rows=len(rows),appended_scalar_rows_by_split=counts,
            Lane3_unstable_interval_not_filled=True,not_generated_by_camera_or_world_projection=True))
    OUTPUT.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    record=dict(schema="rinkspace.user-confirmed-side-RGB-continuous-axis-candidate.v1",status="INACTIVE_DERIVED_SOURCE_CANDIDATE",
        source_packet_path=str(OUTPUT),source_packet_sha256=sha(OUTPUT),parent_packet_path=str(PARENT),parent_packet_sha256=sha(PARENT),
        source_trace_path=str(TRACE),source_trace_sha256=sha(TRACE),source_times=STAMPS,
        appended_rows=len(rows),appended_scalar_rows_by_split=counts,all_original_rows_preserved=True,
        Lane3_unstable_segment_left_missing=True,original_user_identity_authority=user["physical_identity_authority"],
        solver_and_active_config_unchanged=True,whole_video_accuracy_accepted=False)
    previous=BASE/"ice_clear_anchors_user_lane2_lane3_traced_lane2_20261007.json"
    if previous.exists():
        record["superseded_inactive_candidate"]=dict(path=str(previous),sha256=sha(previous),
            reason="NativeIceCurveFactor split enum is holdout, not held; only derived split string corrected, all UV/sigma and original rows preserved")
    fact=ROOT/"events/ankara_2026/model_constraints/lane2_continuous_source_axis_trace_20261007.json"
    fact.write_text(json.dumps(record,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(record,ensure_ascii=False))


if __name__=="__main__":main()
