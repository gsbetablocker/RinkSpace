"""Every existing absolute FIT/HELD row on one immutable complete state.

Reuses the original six actual source evaluators and material owner; it starts
no solver/motion graph and does not assign image identities from projection.
"""
from pathlib import Path
from copy import deepcopy
from collections import Counter
import argparse,json,pickle,time
import numpy as np
from evaluate_global_target_state_20261007 import accepted,read,sha,ROOT,DATA,RUNS,cut,SHOTS,load_camera
from rinkspace.vision_camera_initialization import current_complete_source_geometry
from structured_programme_source_candidate.original_review_source import (PACKET,PACKET_SHA256,
    observation_signature,load_original_absolute_evaluators)
from reference_board_pair import BoardIntersectionError
from structured_programme_source_candidate.current_ice_material_scene import install_source_ice_material_view

OUT=DATA/"diagnostics/external_expert_20261007/all_original_absolute_source_audit"


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--state-dir",type=Path,required=True)
    ap.add_argument("--config",type=Path,required=True);args=ap.parse_args();start=time.perf_counter()
    identity=accepted(args.state_dir);options=read(args.config)["source_options"]
    binder,world,proof=current_complete_source_geometry(args.state_dir/"complete_state_UNACCEPTED.npz",options["--world-init"])
    install_source_ice_material_view(binder)
    if sha(PACKET)!=PACKET_SHA256:raise ValueError("Original absolute source packet changed")
    source=pickle.loads(PACKET.read_bytes())["programme_absolute_rows"]
    if Counter(r["collection_role"] for r in source)!={"FIT":11995,"HELD":225}:raise ValueError("Full original absolute inventory changed")
    calls,call_proof=load_original_absolute_evaluators();archive=load_camera(identity["camera_path"])
    camera={int(p):archive["camera25"][i] for i,p in enumerate(archive["exposure_representative_pts"])}
    aliases=dict(zip(map(int,archive["pts"]),map(int,archive["exposure_representative_pts"])))
    correction_path=options.get("--original-ice-pixel-correction-source")
    correction={} if not correction_path else {r["original_packet_row_index"]:r for r in read(correction_path)["records"]}
    if correction_path:
        for ref in read(correction_path)["source_evidence"]:
            if sha(ref["path"])!=ref["sha256"]:raise ValueError("Active pixel correction RGB changed")
    metrics=[];failures=[]
    for index,item in enumerate(source):
        original=item["source_row"];signature=observation_signature(original);at=int(item["group_actual_pts"])
        row=original;active_correction=False
        if index in correction:
            change=correction[index];row=deepcopy(original);points=change["point_indices"]
            row["uv"]=np.asarray(row["uv"]).copy();row["normal_interval"]=np.asarray(row["normal_interval"]).copy()
            row["uv"][points]=change["corrected_uv"];row["normal_interval"][points]=change["corrected_normal_interval_px"]
            if "source_uv" in row and np.array_equal(row["source_uv"],original["uv"]):row["source_uv"]=row["uv"].copy()
            active_correction=True
        record=dict(original_packet_row_index=index,block=item["block"],collection_role=item["collection_role"],
            id=row["id"],source_pts=int(row["pts"]),source_native_pts=row.get("source_native_pts"),actual_pts=at,shot=cut(at),
            original_split=row.get("split"),UV_count=len(row["uv"]),source_method=row.get("method"),source_origin=row.get("origin"),
            source_branch_operator_authority=row.get("source_branch_operator_authority"),active_pixel_correction=active_correction,
            original_pending_identity="pending" in str(row.get("split")),source_classification_not_reverified_by_this_numeric_loop=True)
        locator=(item["block"],item["collection_role"],at,int(item["group_order"]),int(item["row_order"]))
        if locator==("secondary","FIT",230445824,131,2):
            record.update(status="historical_retired_false_rail_as_print_corner",included_in_active_accuracy_statistics=False)
            metrics.append(record);continue
        try:
            if at not in camera:raise ValueError("Source has no immutable actual exposure camera")
            if row["pts"] in aliases and aliases[int(row["pts"])]!=at:raise ValueError("Original native/actual source clock mismatch")
            binder.register_row(original,actual_pts=at,allow_previously_qualified_held_ad2_representation=item["collection_role"]=="HELD")
            bound=binder.rebind(row,world)
            if observation_signature(bound)!=observation_signature(row):raise ValueError("Material rebinding changed source pixels/role")
            binding=binder._binding(row)[1];contract=binding["contract"]
            r=np.asarray(calls[item["block"]](camera[at],[bound],False)).reshape(-1,2)*np.asarray(row["sigma"])[:,None]
            rawbound=dict(bound);rawbound.pop("normal_interval",None);rawbound.pop("point_support_bounds_xy_px",None)
            raw=None
            try:raw=np.asarray(calls[item["block"]](camera[at],[rawbound],False)).reshape(-1,2)*np.asarray(row["sigma"])[:,None]
            except KeyError as error:
                record["raw_metric_unavailable_reason"]="Original qualified source operator requires preserved support: "+str(error)
            distance=np.linalg.norm(r,axis=1);rawdistance=None if raw is None else np.linalg.norm(raw,axis=1)
            record.update(status="evaluated",physical_identity=contract.get("physical_identity"),material_kind=binding["kind"],
                mean_outside_source_support_px=float(distance.mean()),median_outside_source_support_px=float(np.median(distance)),
                p95_outside_source_support_px=float(np.quantile(distance,.95)),maximum_outside_source_support_px=float(distance.max()),
                raw_mean_native_px=None if raw is None else float(rawdistance.mean()),raw_median_native_px=None if raw is None else float(np.median(rawdistance)),raw_p95_native_px=None if raw is None else float(np.quantile(rawdistance,.95)),
                raw_maximum_native_px=None if raw is None else float(rawdistance.max()),raw_mean_model_minus_source_XY_px=None if raw is None else raw.mean(axis=0).tolist(),
                raw_median_model_minus_source_XY_px=None if raw is None else np.median(raw,axis=0).tolist(),included_in_active_accuracy_statistics=True)
        except (ValueError,TypeError,IndexError,KeyError,FloatingPointError,BoardIntersectionError) as error:
            record.update(status="original_source_operator_or_binding_failed",error_type=type(error).__name__,error=str(error),generic_fallback_used=False)
            failures.append(record)
        if observation_signature(original)!=signature:raise ValueError("Original source mutated")
        metrics.append(record)
        if index%2000==0:print(json.dumps(dict(progress=index,total=len(source),elapsed_seconds=time.perf_counter()-start)),flush=True)
    cuts=[]
    for i,shot in enumerate(SHOTS,1):
        group=[r for r in metrics if r["shot"]==i and r["status"]=="evaluated"]
        fit=[r for r in group if r["collection_role"]=="FIT"];held=[r for r in group if r["collection_role"]=="HELD"]
        kinds=Counter(r["id"] for r in fit)
        cuts.append(dict(shot=i,start_pts=shot["start_pts"],end_pts_exclusive=shot["end_pts"],camera_id=shot["camera_id"],
            evaluated_FIT_rows=len(fit),evaluated_HELD_rows=len(held),source_ids_and_FIT_counts=dict(kinds),
            largest_FIT_outside_support=sorted(fit,key=lambda r:r["p95_outside_source_support_px"],reverse=True)[:3],
            largest_HELD_outside_support=sorted(held,key=lambda r:r["p95_outside_source_support_px"],reverse=True)[:3],
            largest_contact_raw_native=sorted([r for r in fit if r["id"] in ("far_contact","ice_board_contact") and r["raw_p95_native_px"] is not None],key=lambda r:r["raw_p95_native_px"],reverse=True)[:3]))
    frozen=read(args.state_dir/"original225_review_UNACCEPTED.json");stored={(r["block"],r["group_order"],r["row_order"],r["source_row_pts"]):r for r in frozen["rows"]}
    comparisons=[]
    for item,measured in zip(source,metrics):
        if item["collection_role"]!="HELD" or measured["status"]!="evaluated":continue
        key=(item["block"],item["group_order"],item["row_order"],int(item["source_row"]["pts"]));before=stored[key]
        comparisons.append(abs(before["original_operator_outside_support_p95_px"]-measured["p95_outside_source_support_px"]))
    report=dict(schema="rinkspace.all-original-absolute-source-immutable-numeric-audit.v1",state=identity,world_proof=proof,
        original_packet=dict(path=str(PACKET),sha256=PACKET_SHA256),source_evaluators=call_proof,
        row_count=len(metrics),source_roles=dict(Counter(r["collection_role"] for r in metrics)),status_counts=dict(Counter(r["status"] for r in metrics)),
        original225_same_q_p95_reproduction_maximum_native_px=max(comparisons),original225_replayed_rows=len(comparisons),
        active_original_ice_pixel_correction=dict(path=correction_path,sha256=sha(correction_path) if correction_path else None,corrected_original_rows=sorted(correction)),
        full_source_rows=metrics,per_shot=cuts,failures=failures,elapsed_seconds=time.perf_counter()-start,
        relative_motion_not_used_as_absolute_accuracy=True,source_pending_and_development_identity_not_promoted_to_verified=True,
        source_raw_pixel_context_not_all_independent_point_correspondences=True,source_or_solver_not_modified=True,accuracy_accepted=False)
    OUT.mkdir(parents=True,exist_ok=True);path=OUT/f'{identity["q_sha256"][:12]}_all_absolute_source_audit.json'
    path.write_text(json.dumps(report,ensure_ascii=False,separators=(",",":"),allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else v.tolist())+"\n",encoding="utf-8")
    print(json.dumps(dict(report=str(path),statuses=report["status_counts"],seconds=report["elapsed_seconds"],
        cuts=[dict(shot=r["shot"],FIT=r["evaluated_FIT_rows"],HELD=r["evaluated_HELD_rows"],FIT_worst=[dict(id=t["id"],pts=t["source_pts"],p95=t["p95_outside_source_support_px"],raw=t["raw_p95_native_px"],xy=t["raw_mean_model_minus_source_XY_px"]) for t in r["largest_FIT_outside_support"]],
            HELD_worst=[dict(id=t["id"],pts=t["source_pts"],pending=t["original_pending_identity"],p95=t["p95_outside_source_support_px"],xy=t["raw_mean_model_minus_source_XY_px"]) for t in r["largest_HELD_outside_support"]]) for r in cuts])))


if __name__=="__main__":main()
