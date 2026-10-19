"""Concrete nine-shot preparation and an honest backend handoff contract."""
from __future__ import annotations
from collections import defaultdict, Counter
from pathlib import Path
import shutil
from .io_utils import read_json, write_json, read_jsonl, write_jsonl, file_sha256, object_sha256
from .timeline import programme_spec, control_plan, verify_video
from .materials import build_registry, plan_admission, semantic_key
from .reservation import reserve_native_windows, split_jobs_by_reservation


def dense_jobs(scope, selected_frames):
    """One task for every shot-local anchor interval, including both end margins.

    Frames are assigned exactly once to jobs. Intervals do not share counted
    evidence, even though adjacent intervals may use the same anchor as context.
    """
    by_shot=defaultdict(list)
    for f in selected_frames: by_shot[f["shot_id"]].append(f["native_pts"])
    jobs=[]
    for shot in scope["shots"]:
        sid=shot["id"]; first,last=shot["first_pts"],shot["last_pts"]
        if sid=="s01":
            jobs.append({"id":"s01/import_replay","shot_id":sid,"rig_id":shot["rig_id"],
                "first_pts":first,"last_pts":last,"frame_count":shot["native_frame_count"],
                "action":"import_latest_complete_s01_replay_same_geometry_and_raw_rows",
                "status":"pending_latest_s01_snapshot"});continue
        anchors=sorted(set(by_shot[sid]));cursor=first
        for index,right in enumerate(anchors+[last+256]):
            end=min(right,last)
            if cursor>end:continue
            left_context=max([p for p in anchors if p<=cursor], default=None)
            right_context=min([p for p in anchors if p>=end], default=None)
            jobs.append({"id":f"{sid}/{cursor}-{end}","shot_id":sid,"rig_id":shot["rig_id"],
                "first_pts":cursor,"last_pts":end,"frame_count":(end-cursor)//256+1,
                "left_anchor_pts":left_context,"right_anchor_pts":right_context,
                "action":"bidirectional_source_search_and_current_frame_relocalisation",
                "prediction_is_measurement":False,"cross_cut_tracking_allowed":False,
                "mask_then_relocalise":True,"reset_at_occlusion":True,
                "paint_center_requires_two_material_edges":True,
                "frame_receipt_required_even_if_no_visible_static_support":True,
                "stop_at_locked_validation_window":True,"status":"pending_source_worker"})
            cursor=end+256
        if cursor!=last+256: raise AssertionError("Dense jobs do not cover a complete shot")
    if sum(j["frame_count"] for j in jobs)!=scope["included_frame_count"]:
        raise AssertionError("Dense jobs dropped or counted exposures twice")
    return jobs


def dependency_recipes(rows, groups, registry):
    """Symbolic factor ownership over actual data; never a fake numeric problem.

    The eventual joint_v1 compiler resolves finite paths, exact cubic support,
    gauge substitutions, W memberships and typed nuisance dimensions from these
    receipts before allocating a solver state.
    """
    by_group=defaultdict(list)
    for row in rows:
        if row.get("target_id"): by_group[row["provenance_group_id"]].append(row)
    group_index={g["group_id"]:g for g in groups}
    materials={m["id"]:m for m in registry["materials"]}
    for gid,members in by_group.items():
        sid=members[0]["shot_id"];rig=members[0]["rig_id"]
        material_ids=sorted({semantic_key(r) for r in members})
        refs=sorted({name for m in material_ids for name in (materials[m].get("seed_match") or {}).get("parameter_block_names",[])})
        yield {"id":gid,"shot_id":sid,"rig_id":rig,"native_pts":members[0]["native_pts"],
            "source_row_ids":[r["source_row_id"] for r in members],"material_ids":material_ids,
            "shared_parameter_routes":["scene/rink_and_board",f"rigs/{rig}/shared_head",
                f"rigs/{rig}/optics",f"rigs/{rig}/rolling_shutter"],
            "shot_parameter_routes":[f"shots/{sid}/reference_angles",f"shots/{sid}/camera_controls"],
            "reference_geometry_block_names":refs,
            "source_correlation_declarations":group_index.get(gid,{}).get("declared_correlation_ids",[]),
            "physical_corner_rows":sum(bool(r.get("point_correspondence")) for r in members),
            "finite_curve_rows":sum(not r.get("point_correspondence") for r in members),
            "numeric_W_status":"requires_source_statistical_compiler",
            "allowed_cubic_support":"union over entire permitted rolling-shutter row interval plus gauge-substitution dependencies",
            "nuisance_elimination":"connected component of actual W/order/query factors, not independent point by assumption",
            "numeric_factor_emitted":False}


def prepare(bundle_root, output, *, normalized=None, video=None, verify_source=True, control_dt=.4):
    bundle=Path(bundle_root).resolve();out=Path(output).resolve()
    if out.exists():raise ValueError("Use a new run directory; active shot1/joint states must not be overwritten")
    if bundle==out or bundle in out.parents:
        raise ValueError("Run output belongs in the data directory, outside this code bundle")
    normalized=Path(normalized).resolve() if normalized else bundle/"data/prepared"
    rows=list(read_jsonl(normalized/"observations.jsonl.gz"))
    groups=list(read_jsonl(normalized/"groups.jsonl.gz"))
    frames=list(read_jsonl(normalized/"frames.jsonl.gz"))
    if len({r["source_row_id"] for r in rows})!=len(rows):raise ValueError("Duplicate source row identities")
    inventory=read_json(bundle/"data/context/shot_inventory_20261014.json")
    scope,graph,schedule=programme_spec(inventory,frames)
    reservation,roles=reserve_native_windows(scope,frames)
    for f in schedule:
        f["validation_reservation_role"]=roles[f["native_pts"]]
        f["use_pixels_for_training_or_templates"]=roles[f["native_pts"]] in {
            "development_candidate","historical_regression"}
    reference=bundle/"data/reference/s01_i20_model.json"
    registry=build_registry(rows,read_json(reference),file_sha256(reference))
    decisions,auto_jobs,admission=plan_admission(rows,groups,registry)
    video_audit=verify_video(video,scope) if video and verify_source else {
        "passed":False,"reason":"video verification not requested or video path not supplied"}
    out.mkdir(parents=True)
    write_json(out/"programme_scope.json",scope)
    write_json(out/"joint_graph.json",graph)
    write_json(out/"control_plan.json",control_plan(scope,control_dt))
    write_json(out/"video_audit.json",video_audit)
    write_json(out/"validation_reservation.json",reservation)
    write_json(out/"material_registry.json",registry)
    write_jsonl(out/"frame_schedule.jsonl.gz",schedule)
    write_jsonl(out/"auto_admission.jsonl.gz",decisions)
    write_jsonl(out/"source_resolution_jobs.jsonl.gz",auto_jobs)
    jobs=split_jobs_by_reservation(dense_jobs(scope,frames),roles)
    write_jsonl(out/"dense_jobs.jsonl.gz",jobs)
    write_jsonl(out/"dependency_recipes.jsonl.gz",dependency_recipes(rows,groups,registry))
    write_json(out/"auto_admission_summary.json",admission)
    inputs={p.name:{"path":str(p),"sha256":file_sha256(p)} for p in [
        normalized/"observations.jsonl.gz",normalized/"groups.jsonl.gz",normalized/"frames.jsonl.gz",reference]}
    remaining=[
        "validation_lineage: finalise whole-source-family roles before templates, flow or camera initialisation; honour reserved windows now",
        "source_worker: current native pixel identity/occlusion/operator decisions and automatic repairs",
        "material_compiler: operator-aware crosswalk, s09 assembly and full shared-scene geometry",
        "numeric_statistics: genuine group covariance/W and one-count source lineage",
        "s01_bridge: latest complete repaired checkpoint plus verified objective dependencies",
        "camera_initialiser: two additional physical rigs and eight shot-local native trajectories",
        "joint_backend: dynamic layout, sparse derivatives, constraints and actual joint solver",
        "dense_worker: every included native frame has source evidence or explicit unsupported status",
        "acceptance: frozen-state full-source projection, coverage and withheld evidence reports"]
    summary={"schema":"rinkfit.joint9_preparation_receipt.v1","version":"0.1.0",
        "architecture_prepared":True,"production_joint_fit_complete":False,
        "normalised_source_rows":len(rows),"source_groups":len(groups),"selected_native_frames":len(frames),
        "included_programme_frames":len(schedule),"dense_interval_jobs":len(jobs),
        "validation_reservation":{k:reservation[k] for k in ("reservation_sha256","frame_role_counts","shots_without_eligible_reservation","independence_certified")},
        "rigs":{k:[s for s,v in graph["shots"].items() if v["rig_id"]==k] for k in graph["rigs"]},
        "material_registry_summary":registry["summary"],"auto_admission":admission,
        "video_verified":bool(video_audit.get("passed")),"inputs":inputs,
        "source_rows_deleted":0,"current_s01_worktree_modified":False,
        "completed_capabilities":["lossless source ingest","integer PTS routing","three-rig camera graph",
            "complete checkpoint import adapter","operator-aware material registry","automatic source triage",
            "all-frame task ledger","source-correlated factor recipes","validation and Schur-topology modules"],
        "remaining_codex_work":remaining}
    write_json(out/"prepare_summary.json",summary)
    return summary


def status(run):
    run=Path(run)
    summary=read_json(run/"prepare_summary.json")
    results={"architecture_prepared":summary["architecture_prepared"],
        "included_programme_frames":summary["included_programme_frames"],
        "production_joint_fit_complete":False,"remaining_codex_work":summary["remaining_codex_work"]}
    for name in ("video_audit.json","screen_summary.json","graph_audit.json","full_replay_receipt.json"):
        if (run/name).exists(): results[name]=read_json(run/name)
    # A text file called final or a good subset residual does not complete this contract.
    return results
