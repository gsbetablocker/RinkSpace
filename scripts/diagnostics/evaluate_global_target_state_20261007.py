"""Independent source regression audit of immutable global-target states."""
from pathlib import Path
from collections import Counter
import argparse,json
import numpy as np
from evaluate_user_lane_states_20261007 import accepted,read,sha,ROOT,DATA,RUNS
from verify_current_native_primitives_20261007 import state_camera_adapter
from rinkspace.vision_camera_initialization import current_complete_source_geometry,frozen_scene_curve
from z190_source_camera_diagnostic import absolute_value_jac15
from structured_programme_source_candidate.native_ice_curve_source import NativeIceCurveFactor
from structured_programme_source_candidate.native_board_track_source import NativeBoardTrackFactor
from structured_programme_source_candidate.original_review_source import evaluate_original_review_rows,load_original_review_rows
from structured_programme_source_candidate.current_ice_material_scene import install_source_ice_material_view
from wide_anchor_candidate.full_brown_projection import curve_vector_error_full_brown
from audit_camera_rf_history_20261007 import compare as camera_compare,load_camera
from z190_shared_board_pattern_source import _project

BASE=RUNS/"okada_lane23_Ad7_material_accelerated_20261007/accepted_states/fe814cbe02a54f0ddef0427a654e9776f7af4b3f19c80adc506fc0da31e12daa"
OUT=DATA/"diagnostics/external_expert_20261007/global_target_state_QA"
CONFIG=ROOT/"configs/solver_runtime/okada_global_geometry_crease_ads26_boardtracks_20261007.json"
SHOTS=read(ROOT/"calibration/program_tracking/source_shots.json")["shots"][1:10]


def cut(pts):
    for i,s in enumerate(SHOTS,1):
        if s["start_pts"]<=pts<s["end_pts"]:return i
    return None


def actual_ice(directory,packet,auxiliary,board_packet=None,native_board_edge_packet=None):
    identity=accepted(directory)
    keys=("q","aux__canonical_pts","aux__block_lengths","aux__block_start","aux__block_names",
        "aux__installation_xyz_slots","aux__current_shared_mount_parameter_indices",
        "aux__current_all_installation_optics_parameter_indices","aux__correlated_roll_prior_contract_json")
    with np.load(Path(directory)/"complete_state_UNACCEPTED.npz",allow_pickle=False) as saved:
        arrays={k:saved[k].copy() for k in keys}
        if 'aux__current_complete_parameter_layout_json'in saved:arrays['aux__current_complete_parameter_layout_json']=saved['aux__current_complete_parameter_layout_json'].copy()
    with np.load(identity["camera_path"],allow_pickle=False) as saved:cameras={k:saved[k].copy() for k in saved.files}
    provider,world,proof=current_complete_source_geometry(Path(directory)/"complete_state_UNACCEPTED.npz",auxiliary)
    install_source_ice_material_view(provider)
    state,camera_error=state_camera_adapter(arrays,cameras,provider)
    factor=NativeIceCurveFactor(state,packet,len(arrays["q"]));q=state.evaluation_q(arrays["q"])
    review=factor.review(q)["rows"];cam=factor._cameras(q)
    scene=read(Path(directory)/"scene_UNACCEPTED.json");features={f["id"]:f for f in scene["ice_features"]}
    measurements=[]
    byid={r["id"]:r for r in factor.rows+factor.held}
    for stat in review:
        row=byid[stat["id"]];current_world=factor._world(q,row["curve"])
        if row["curve"]["kind"]=="current_goal_crease":
            exported=np.asarray(features[row["feature_id"]]["polygon_xyz"])
            if not np.array_equal(exported[0],exported[-1]):exported=np.vstack((exported,exported[0]))
            if not np.array_equal(current_world,exported):raise ValueError("Crease factor does not use exported current polygon")
        raw=curve_vector_error_full_brown(cam[row["actual_pts"]],current_world,row["uv"],with_jacobian=False)
        if isinstance(raw,tuple):raw=raw[0]
        raw=np.asarray(raw).reshape(-1,2);dist=np.linalg.norm(raw,axis=1)
        record=dict(stat,shot=cut(row["pts"]),raw_native_mean_px=float(dist.mean()),raw_native_p90_px=float(np.percentile(dist,90)),
            raw_native_max_px=float(dist.max()),model_minus_source_vectors_native_px=raw.tolist(),source_uv=row["uv"].tolist(),
            curve=row["curve"],source_sigma_px=row["sigma"].tolist(),source_raw_frame=row["source_raw_frame"],source_file_sha256=sha(row["source_raw_frame"]))
        if "normal" in row:
            n=np.asarray(row["normal"],float);n/=np.linalg.norm(n,axis=1)[:,None];d=np.einsum("ni,ni->n",raw,n)
            record.update(source_signed_normal_px=d.tolist(),raw_normal_mean_abs_px=float(abs(d).mean()),raw_normal_max_abs_px=float(abs(d).max()))
        measurements.append(record)
    result=dict(state=identity,world_proof=proof,original_physical_camera_reproduced_maximum_difference=camera_error,measurements=measurements)
    original_rows=load_original_review_rows()
    for row in original_rows:
        provider.register_row(row["source_row"],actual_pts=int(row["group_actual_pts"]),allow_previously_qualified_held_ad2_representation=row["collection_role"]=="HELD")
    result["original225_review"]=evaluate_original_review_rows(provider,world,identity["camera_path"],rows=original_rows,complete_q_sha256=identity["q_sha256"])
    if native_board_edge_packet is not None:
        from structured_programme_source_candidate.native_board_edge_source import NativeBoardEdgeFactor
        edges=NativeBoardEdgeFactor(state,native_board_edge_packet,len(q))
        result["real_native_board_edge_review"]=edges.review(q)
        source_rows={row["id"]:row for row in edges.rows+edges.held}
        for row in result["real_native_board_edge_review"]["rows"]:
            source=source_rows[row["id"]]
            row.update(shot=cut(source["pts"]),curve=source["curve"],source_uv=source["uv"].tolist(),
                source_sigma_px=source["sigma"].tolist(),source_normal=source["normal"].tolist(),
                source_normal_interval_px=source["normal_interval"].tolist(),
                source_raw_frame=source["source_raw_frame"],source_file_sha256=sha(source["source_raw_frame"]))
    if board_packet is not None:
        tracks=NativeBoardTrackFactor(state,board_packet,len(q))
        result["real_static_board_track_review"]=tracks.review(q)
        for row in result["real_static_board_track_review"]["rows"]:row["shot"]=cut(row["target_pts"])
    return result


def original225_regressions(base,current,before_report=None,after_report=None):
    old=read(Path(base)/"original225_review_UNACCEPTED.json") if before_report is None else before_report
    new=read(Path(current)/"original225_review_UNACCEPTED.json") if after_report is None else after_report
    if old["source_packet_sha256"]!=new["source_packet_sha256"]:raise ValueError("Original source packet identity changed")
    if old["complete_q_sha256"]!=accepted(base)["q_sha256"]:raise ValueError("Original225 baseline report belongs to another q")
    if new["complete_q_sha256"]!=accepted(current)["q_sha256"]:raise ValueError("Stored original225 report belongs to another q")
    if len(old["rows"])!=len(new["rows"]):raise ValueError("Original225 source rows omitted")
    if not old["complete_numeric_source_evaluation"] or not new["complete_numeric_source_evaluation"]:raise ValueError("Original225 evaluation failed; missing rows cannot count as zero regression")
    def key(r):return (r["block"],r["group_order"],r["row_order"],r["id"],r["source_row_pts"])
    bykey={key(r):r for r in old["rows"]};comparisons=[]
    for b in new["rows"]:
        a=bykey[key(b)]
        if a["observation_signature"]!=b["observation_signature"] or a["actual_pts"]!=b["actual_pts"]:raise ValueError("Original row UV/sigma/clock/operator changed")
        if a["evaluation_status"]!="evaluated" or b["evaluation_status"]!="evaluated":continue
        fields=("original_operator_outside_support_median_px","original_operator_outside_support_p95_px","original_operator_outside_support_maximum_px")
        if any(a.get(f) is None or b.get(f) is None for f in fields):continue
        record=dict(id=b["id"],pts=b["source_row_pts"],actual_pts=b["actual_pts"],shot=cut(b["source_row_pts"]),block=b["block"],
            original_split=b["original_source_split"],UV_count=b["UV_count"],original_source_authority=b["original_source_authority"],
            before_median_px=a[fields[0]],after_median_px=b[fields[0]],median_delta_px=b[fields[0]]-a[fields[0]],
            before_p95_px=a[fields[1]],after_p95_px=b[fields[1]],p95_delta_px=b[fields[1]]-a[fields[1]],
            before_max_px=a[fields[2]],after_max_px=b[fields[2]],max_delta_px=b[fields[2]]-a[fields[2]])
        comparisons.append(record)
    cuts=[]
    for i,s in enumerate(SHOTS,1):
        group=[r for r in comparisons if r["shot"]==i]
        cuts.append(dict(shot=i,start_pts=s["start_pts"],end_pts_exclusive=s["end_pts"],camera_id=s["camera_id"],evaluated_original_rows=len(group),
            sampled_exposures=len({r["actual_pts"] for r in group}),median_of_source_row_median_before_px=float(np.median([r["before_median_px"] for r in group])) if group else None,
            median_of_source_row_median_after_px=float(np.median([r["after_median_px"] for r in group])) if group else None,
            greatest_source_p95_regressions=sorted(group,key=lambda r:r["p95_delta_px"],reverse=True)[:3],
            absent_original_source_is_not_success=not bool(group)))
    return dict(original_packet_sha256=old["source_packet_sha256"],original_rows_UV_sigma_roles_clock_and_operator_preserved=True,
        rows=sorted(comparisons,key=lambda r:r["p95_delta_px"],reverse=True),per_shot=cuts,
        greatest_p95_regressions=sorted(comparisons,key=lambda r:r["p95_delta_px"],reverse=True)[:12],
        every_original_source_row_is_not_independent_or_user_verified=True,source_retirement_numeric_zero_not_alignment=True)


def print_sources(directory,options):
    identity=accepted(directory);scene=read(identity["scene_path"]);camera=load_camera(identity["camera_path"])
    lookup={int(p):i for i,p in enumerate(camera["pts"])};measurements=[]
    for option in ("--additional-board-circle-source","--additional-board-material-point-source"):
        packet=Path(options[option])
        for row in read(packet)["rows"]:
            i=lookup[int(row["pts"])];at=int(camera["exposure_representative_pts"][i])
            if row.get("actual_pts",at)!=at:raise ValueError("Printed source native exposure binding failed")
            source=Path(row.get("source_raw_frame",row.get("source_path","")))
            if sha(source)!=row.get("source_file_sha256",row.get("source_frame_sha256")):raise ValueError("Actual printed source changed")
            uv=np.asarray(row["uv"],float);ad=row["curve"]["advertisement_id"]
            if row.get("point_correspondence"):
                points=scene["shared_board_patterns"]["shield_fixed_material_points_xyz_scene_units"][str(ad)]
                world=np.array([points[k] for k in row["material_roles"]],float);raw=_project(camera["camera25"][i],world,False)[0]-uv
            else:
                if row["curve"]["kind"]=="shared_board_pattern_circle_curve":world=np.asarray(scene["shared_board_patterns"]["circle_outer_stroke_paths_xyz_scene_units"][str(ad)])
                else:world=frozen_scene_curve(scene,row["curve"])
                raw=curve_vector_error_full_brown(camera["camera25"][i],world,uv,with_jacobian=False)
                if isinstance(raw,tuple):raw=raw[0]
                raw=np.asarray(raw).reshape(-1,2)
            distance=np.linalg.norm(raw,axis=1)
            record=dict(id=row["id"],advertisement_id=ad,pts=row["pts"],actual_pts=at,split=row["split"],shot=cut(row["pts"]),
                packet_path=str(packet),packet_sha256=sha(packet),source_uv=uv.tolist(),model_minus_source_UV=raw.tolist(),
                mean_native_px=float(distance.mean()),max_native_px=float(distance.max()))
            if "normal" in row:
                normal=np.asarray(row["normal"],float);normal/=np.linalg.norm(normal,axis=1)[:,None]
                signed=np.einsum("ni,ni->n",raw,normal);record.update(signed_normal_px=signed.tolist(),mean_abs_normal_px=float(abs(signed).mean()))
            measurements.append(record)
    return measurements


def existing_review_families(base,current,baseline_edge_review=None,current_edge_review=None):
    a=read(Path(base)/"source_review_UNACCEPTED.json");b=read(Path(current)/"source_review_UNACCEPTED.json")
    baseline_identity=accepted(base)
    if a["scene_sha256"]!=baseline_identity["scene_sha256"] or a["camera_sha256"]!=baseline_identity["camera_sha256"]:raise ValueError("Stored baseline source reviews do not bind frozen state")
    identity=accepted(current)
    if b["scene_sha256"]!=identity["scene_sha256"] or b["camera_sha256"]!=identity["camera_sha256"]:raise ValueError("Stored source reviews do not bind frozen state")
    specification={"additional_native_board_edges":("rows","mean_outside_source_support_px","max_outside_source_support_px"),
        "additional_native_persistent_ice_long_baseline":("rows","native_error_norm_px","native_error_norm_px"),
        "wide":("raw_source_native_pixel_error_groups","median_native_px","p95_native_px")}
    result={}
    for family,(listkey,metric,high) in specification.items():
        historical_previous={r["id"]:r for r in a.get(family,{}).get(listkey,[])}
        fresh_edges=family=="additional_native_board_edges" and baseline_edge_review is not None and current_edge_review is not None
        previous={r["id"]:r for r in (baseline_edge_review["rows"] if fresh_edges else a.get(family,{}).get(listkey,[]))}
        current_rows=current_edge_review["rows"] if fresh_edges else b.get(family,{}).get(listkey,[])
        comparisons=[];new_rows_without_prior=[]
        for new in current_rows:
            if new["id"] not in historical_previous:new_rows_without_prior.append(dict(new))
            if new["id"] not in previous:continue
            old=previous[new["id"]]
            comparisons.append(dict(id=new["id"],split=new.get("split"),pts=new.get("pts",new.get("target_native_pts")),
                metric=metric,before=old[metric],after=new[metric],delta=new[metric]-old[metric],
                high_metric=high,before_high=old[high],after_high=new[high],delta_high=new[high]-old[high]))
        result[family]=dict(comparisons=sorted(comparisons,key=lambda r:r["delta_high"],reverse=True),
            matched_previous_rows=len(comparisons),new_rows_not_in_previous=len(new_rows_without_prior),
            new_rows_without_prior=new_rows_without_prior,
            same_current_packet_evaluated_at_both_states=fresh_edges)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--state-dir",action="append");parser.add_argument("--config",type=Path,default=CONFIG);parser.add_argument("--baseline",type=Path,default=BASE);args=parser.parse_args()
    config=read(args.config);options=config["source_options"];packet=Path(options["--additional-ice-source"])
    # The frozen launch config may retain the parent packet's historical hash.
    # An explicit option identity authenticates the actual selected append packet.
    identities=config.get("other_source_packet_sha256s",{})
    registered=identities.get("--additional-ice-source",config.get("additional_ice_source_packet_sha256",identities.get("additional_ice_source_packet_sha256")))
    if registered is None or sha(packet)!=registered:raise ValueError("New actual ice source changed or has no registered identity")
    if args.state_dir:directories=[Path(d) for d in args.state_dir]
    else:
        directory=RUNS/"okada_global_geometry_crease_ads26_boardtracks_joint_20261007/accepted_states"
        candidates=[] if not directory.exists() else [d for d in directory.iterdir() if (d/"state_identity.json").exists() and (d/"original225_review_UNACCEPTED.json").exists()]
        if not candidates:print(json.dumps(dict(status="waiting_for_complete_immutable_accepted_state",run=str(directory.parent))));return
        directories=[max(candidates,key=lambda d:d.stat().st_mtime)]
    board_packet=options.get("--additional-native-board-track-source")
    edge_packet=options.get("--additional-board-source")
    baseline=actual_ice(args.baseline,packet,options["--world-init"],board_packet,edge_packet);old={r["id"]:r for r in baseline["measurements"]}
    baseline_prints={r["id"]:r for r in print_sources(args.baseline,options)}
    basecam=load_camera(baseline["state"]["camera_path"]);OUT.mkdir(parents=True,exist_ok=True)
    for directory in directories:
        current=actual_ice(directory,packet,options["--world-init"],board_packet,edge_packet)
        for r in current["measurements"]:
            a=old[r["id"]];r.update(before_raw_native_mean_px=a["raw_native_mean_px"],mean_delta_px=r["raw_native_mean_px"]-a["raw_native_mean_px"],
                before_raw_native_max_px=a["raw_native_max_px"],maximum_delta_px=r["raw_native_max_px"]-a["raw_native_max_px"])
        original=original225_regressions(args.baseline,directory,baseline["original225_review"],current["original225_review"])
        prints=print_sources(directory,options)
        for r in prints:
            oldprint=baseline_prints[r["id"]];r.update(before_mean_native_px=oldprint["mean_native_px"],before_max_native_px=oldprint["max_native_px"],
                mean_delta_px=r["mean_native_px"]-oldprint["mean_native_px"],max_delta_px=r["max_native_px"]-oldprint["max_native_px"])
        newcam=load_camera(current["state"]["camera_path"])
        track_comparisons=[];track_cuts=[]
        if board_packet:
            oldtracks={r["id"]:r for r in baseline["real_static_board_track_review"]["rows"]}
            for row in current["real_static_board_track_review"]["rows"]:
                a=oldtracks[row["id"]]
                track_comparisons.append(dict(row,before_median_native_px=a["median_native_px"],before_maximum_native_px=a["maximum_native_px"],
                    median_delta_px=row["median_native_px"]-a["median_native_px"],maximum_delta_px=row["maximum_native_px"]-a["maximum_native_px"]))
            for i in range(1,10):
                group=[r for r in track_comparisons if r["shot"]==i]
                held=[r for r in group if r["split"]=="holdout"]
                track_cuts.append(dict(shot=i,groups=len(group),fit_points=sum(r["points"] for r in group if r["split"]=="fit"),holdout_points=sum(r["points"] for r in held),
                    held_group_medians_before=[r["before_median_native_px"] for r in held],held_group_medians_after=[r["median_native_px"] for r in held],
                    largest_held_maximum_before=max((r["before_maximum_native_px"] for r in held),default=None),
                    largest_held_maximum_after=max((r["maximum_native_px"] for r in held),default=None)))
        report=dict(schema="rinkspace.immutable-global-target-source-regression-QA.v1",baseline=baseline["state"],current=current["state"],
            ice_packet=dict(path=str(packet),sha256=sha(packet)),actual_ice_source_operator="NativeIceCurveFactor._world/_residual/review and original full Brown finite-curve operator",
            new_and_existing_ice_FIT_and_holdout=current["measurements"],original225_source_regression=original,
            all_circle_and_material_FIT_and_holdout=prints,
            preserved_board_edges_wide_and_persistent_source_comparisons=existing_review_families(args.baseline,directory,
                baseline.get("real_native_board_edge_review"),current.get("real_native_board_edge_review")),
            original225_evaluated_again_at_both_complete_states=True,
            nine_shot_static_board_tracks=dict(packet_path=board_packet,packet_sha256=sha(board_packet) if board_packet else None,rows=track_comparisons,per_shot=track_cuts,
                operator="NativeBoardTrackFactor: actual full Brown inverse ray, current far finite board intersection, actual target full Brown projection; same nonzero-lever fixed-pivot rig"),
            raw_camera_exposure_changes_all_nine_shots=camera_compare(basecam,newcam,SHOTS),
            changed_state_is_not_verified_video_alignment=True,numerical_cost_used_as_accuracy=False,
            regression_policy="Report every signed change in actual native source pixels; largest outside-support and holdout regressions determine further source review. No new automatic solver stop or invented confidence threshold.",
            whole_video_accuracy_accepted=False)
        path=OUT/f'{current["state"]["q_sha256"][:12]}_source_regression_QA.json'
        path.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
        important=[dict(id=r["id"],split=r["split"],mean=r["raw_native_mean_px"],before=r["before_raw_native_mean_px"],max=r["raw_native_max_px"]) for r in current["measurements"] if r["curve"]["kind"]=="current_goal_crease" or "traced" in r["id"] or "trace" in r["id"]]
        print(json.dumps(dict(report=str(path),q=current["state"]["q_sha256"],ice_important=important,
            nine_shot_board_HELD=track_cuts,
            short_prints=[dict(id=r["id"],split=r["split"],before=r["before_mean_native_px"],after=r["mean_native_px"],max=r["max_native_px"]) for r in prints if r["id"].startswith("short_board")],
            old_source_cut_summary=[dict(shot=s["shot"],n=s["evaluated_original_rows"],before=s["median_of_source_row_median_before_px"],after=s["median_of_source_row_median_after_px"]) for s in original["per_shot"]],
            largest_old_regressions=original["greatest_p95_regressions"][:5])))


if __name__=="__main__":main()
