"""Immutable state native-source normal errors along reliable RGB Lane2 tracks."""
from pathlib import Path
import os
for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):os.environ[k]="1"
import argparse
import json
import sys
import hashlib
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/"src"),str(ROOT/"scripts/diagnostics")]
from evaluate_user_lane_states_20261007 import accepted,read,sha,DATA,RUNS,SOURCE_DIR
from rinkspace.vision_camera_initialization import frozen_scene_curve
from z190_source_camera_diagnostic import absolute_value_jac15

TRACE=DATA/"diagnostics/external_expert_20261007/user_lane_source_interval_trace/user_lane_source_normal_profile_trace.json"
PACKET=SOURCE_DIR/"complete_source_clear_anchor_append_20261007/ice_clear_anchors_user_lane2_lane3_20261007.json"


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--output",required=True);ap.add_argument("--state-dir",action="append");args=ap.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    trace=read(TRACE);packet=read(PACKET);user=next(r for r in packet["rows"] if r["id"]=="user_native230501120_lane2_upper_long_20261007")
    frozen=read(DATA/"programmes/okada_mei_sp/results/shared_radius_05db_pelvis_priority_full_programme_20261005/FROZEN_INPUTS_UNACCEPTED.json")
    if sha(frozen["scene"]["path"])!=frozen["scene"]["sha256"] or sha(frozen["camera"]["path"])!=frozen["camera"]["sha256"]:raise ValueError("Old immutable source changed")
    old=dict(label="old4075_camera917c204",q_sha256=frozen["q_sha256"],scene_path=frozen["scene"]["path"],scene_sha256=frozen["scene"]["sha256"],camera_path=frozen["camera"]["path"],camera_sha256=frozen["camera"]["sha256"])
    clear=accepted(RUNS/"source_corrected_clear_actual_complete_joint_20261007/accepted_states/0eb5c6d4d2b0a24ab201f65361ea0ed2de409b4bdfdddb1fc68c6b331ae54e63","clear0eb5")
    run=RUNS/"okada_lane23_Ad7_fast_complete_20261007/accepted_states"
    q478=accepted(run/"478afaf776150c398c48e7492da911e1219036cfad3295629a3df309b7d1089c","q478")
    states=[old,clear,q478]
    targets=[Path(p) for p in args.state_dir] if args.state_dir else [max((p for p in run.iterdir() if (p/"state_identity.json").exists()),key=lambda p:p.stat().st_mtime)]
    states += [accepted(p) for p in targets if p.resolve()!=Path(q478["state_directory"]).resolve()]
    observations=[]
    initial=np.array(user["uv"][:2],float);t=initial[1]-initial[0];t/=np.linalg.norm(t);n=np.tile([-t[1],t[0]],(2,1))
    observations.append(dict(pts=230501120,indices=[0,1],uv=initial.tolist(),normal=n.tolist(),source="registered direct user source"))
    for frame in trace["per_pair_source_profile_trace"]:
        supported=[r for r in frame["source_axis_normal_tracking"] if r["original_user_index"] in (0,1) and r["chain_alive"]]
        if not supported:continue
        uv=np.array([r["source_axis_uv"] for r in supported])
        if len(uv)==2:
            t=uv[1]-uv[0];t/=np.linalg.norm(t);n=np.tile([-t[1],t[0]],(2,1))
        else:n=np.array([r["profile_match"]["source_normal"] for r in supported])
        observations.append(dict(pts=frame["to_pts"],indices=[r["original_user_index"] for r in supported],uv=uv.tolist(),normal=n.tolist(),source="native RGB normal profile correspondence",source_quality=[dict(ncc=r["profile_match"]["ncc"],FB_normal_px=r["step_forward_backward_normal_px"],original_RGB_template_delta_px=r["independent_original_template_normal_delta_px"]) for r in supported]))
    measurements=[];summaries=[]
    canonical_clock=None
    for state in states:
        scene=read(state["scene_path"])
        with np.load(state["camera_path"],allow_pickle=False) as arrays:
            stamps=arrays["pts"].copy();cameras=arrays["camera25"].copy();aliases=arrays["exposure_representative_pts"].copy()
        clock={int(p):int(q) for p,q in zip(stamps,aliases)}
        if canonical_clock is None:canonical_clock=clock
        elif clock!=canonical_clock:raise ValueError("State source exposure mapping changed")
        world=frozen_scene_curve(scene,user["curve"])
        group=[]
        for source in observations:
            pts=source["pts"];i=int(np.flatnonzero(stamps==pts)[0])
            uv=np.array(source["uv"]);normal=np.array(source["normal"])
            row=dict(curve=user["curve"],uv=uv,world=world,normal=normal,sigma=np.ones(len(uv)))
            raw=absolute_value_jac15(cameras[i],[row],False).reshape(-1,2)
            signed=np.einsum("ni,ni->n",raw,normal)
            measurement=dict(state=state["label"],q_sha256=state["q_sha256"],pts=pts,actual_pts=clock[pts],
                native_exposure_unique=clock[pts]==pts,source_point_indices=source["indices"],source_uv=uv.tolist(),
                unit_source_normal=normal.tolist(),signed_model_minus_source_normal_px=signed.tolist(),
                mean_abs_normal_px=float(np.abs(signed).mean()),maximum_abs_normal_px=float(np.abs(signed).max()),
                raw_model_minus_source_uv=raw.tolist())
            measurements.append(measurement)
            if clock[pts]==pts:group.extend(np.abs(signed).tolist())
        summaries.append(dict(state=state["label"],all_unique_exposure_points=len(group),
            unique_exposure_mean_abs_normal_px=float(np.mean(group)),unique_exposure_p90_abs_normal_px=float(np.percentile(group,90)),
            unique_exposure_max_abs_normal_px=float(np.max(group)),
            start_mean_abs_normal_px=next(m["mean_abs_normal_px"] for m in measurements if m["state"]==state["label"] and m["pts"]==230501120),
            last_mean_abs_normal_px=next(m["mean_abs_normal_px"] for m in measurements if m["state"]==state["label"] and m["pts"]==230507776)))
    old_axis=[]
    for oldrow in packet["rows"]:
        if oldrow["id"] not in ("cut6_source_230506752_lane2_upper_long_holdout","cut6_source_230507776_lane2_upper_long_fit"):continue
        source=next(r for r in observations if r["pts"]==oldrow["pts"])
        uv=np.array(source["uv"]);old_uv=np.array(oldrow["uv"])
        if len(uv)!=2:
            old_axis.append(dict(id=oldrow["id"],comparison_not_supported="only one reliable tracked point; cannot assert source-line slope to remote old UV"));continue
        tangent=uv[1]-uv[0];tangent/=np.linalg.norm(tangent);normal=np.array([-tangent[1],tangent[0]])
        old_axis.append(dict(id=oldrow["id"],source_observed_tracked_uv=uv.tolist(),old_source_uv=old_uv.tolist(),
            remote_old_UV_normal_to_tracked_two_point_TLS_line_px=((old_uv-uv[0])@normal).tolist(),
            old_UV_outside_actual_tracked_pixel_domain=True,image_line_extension_navigation_only_not_pixel_GT=True,
            this_does_not_authorize_retiring_or_relabeling_old_source=True))
    report=dict(schema="rinkspace.immutable-user-lane2-source-interval-normal-comparison.v1",source_trace_path=str(TRACE),source_trace_sha256=sha(TRACE),
        states=states,source_observations=observations,measurements=measurements,unique_exposure_summaries=summaries,
        old_later_source_axis_navigation_comparison=old_axis,
        source_operator="unchanged frozen_scene_curve 1025 samples and absolute_value_jac15 full native Brown",source_pts_exposure_map_identical_across_all_states=True,
        state_source_and_active_solver_config_unchanged=True,unsupported_Lane3_not_filled=True,whole_video_accuracy_accepted=False)
    path=out/"lane2_native_source_interval_state_normal_errors.json";path.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps(dict(report=str(path),summaries=summaries,old_axis=old_axis),ensure_ascii=False))


if __name__=="__main__":main()
