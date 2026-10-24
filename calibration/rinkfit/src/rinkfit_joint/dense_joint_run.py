"""Qualified dense-six fresh native GPU entry and current-source finding binding.

This is the public form of the actual joint6 launcher. Native entry performs
its full source/domain gate and builds the graph; a stored probe is read-only
supporting evidence and never substitutes for actual execution.
"""
from __future__ import annotations
from collections import defaultdict
from copy import deepcopy
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np
from .effective_input_prepare_v1 import actual_source_qualification
from .io_utils import read_json,file_sha256,object_sha256
from .dense_joint_packet import _array
from .dense_qualification_packet import current_dense_packet_source_bindings

ROOT=Path(__file__).resolve().parents[4]
SHOTS=["s01","s02","s03","s04","s05","s06"]


def reference(path):
    path=Path(path).resolve()
    return {"file":str(path),"sha256":file_sha256(path)}


def _check(ref,base=None):
    path=Path(ref["file"])
    path=(path if path.is_absolute()else Path(base or ROOT)/path).resolve()
    if file_sha256(path)!=ref["sha256"]:raise ValueError("Pinned dense run input changed: "+str(path))
    return path


def _same_ref(left,right):
    return (isinstance(left, dict) and isinstance(right, dict) and {"file", "sha256"} <= left.keys() and {"file", "sha256"} <= right.keys()
            and left["sha256"]==right["sha256"] and Path(left["file"]).resolve()==Path(right["file"]).resolve())


def _write(path,value):
    path=Path(path).resolve()
    if len(str(path).encode("utf-16-le"))//2>=260 or any(len(x.encode("utf-16-le"))//2>255 for x in path.parts):raise ValueError("Run artifact exceeds Windows path limits")
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x",encoding="utf8")as f:json.dump(value,f,ensure_ascii=False,separators=(",",":"),allow_nan=False)
    return reference(path)


def _load_spec(spec):
    if isinstance(spec,dict):return deepcopy(spec),dict(kind="in_memory_spec",object_sha256=object_sha256(spec))
    path=Path(spec).resolve();return read_json(path),reference(path)


def _run_packet_header(path):
    """Read inherited run metadata without expanding millions of source IDs."""
    wanted = {"shot_ids", "engineering_scope", "native_ad_source_binding", "dense_source_binding", "row_count", "active_rows", "dense_qualification_binding"}
    result = {}; seen = set(); path = Path(path)
    while True:
        key = str(path.resolve()).casefold()
        if key in seen: raise ValueError("Cyclic dense parent packet")
        seen.add(key); doc = json.loads(path.read_text(encoding="utf-8-sig"))
        for name in wanted:
            if name not in result and name in doc: result[name] = doc[name]
        if "dense_parent_packet" not in doc: return result
        path = _check(doc["dense_parent_packet"], path.parent)


def representative_identity_ancestry(saved_sha, packet_ref, revision_ref, qualification_ref):
    """Verify every mixed rebind/withdrawal generation; native source guard still runs."""
    chain = []; seen = set(); raw_rows = None; proofs = set()
    def document(ref): return json.loads(_check(ref).read_text(encoding="utf-8-sig"))
    def proof(ref):
        key = (str(Path(ref["file"]).resolve()), ref["sha256"])
        if key not in proofs: _check(ref); proofs.add(key)
    def inherited(doc, name, array=False):
        visited = set()
        while name not in (doc.get("arrays", {}) if array else doc):
            ref = doc.get("dense_parent_packet")
            if not ref or ref["sha256"] in visited: raise ValueError("Missing/cyclic revision ancestry field: " + name)
            visited.add(ref["sha256"]); doc = document(ref)
        return doc["arrays"][name] if array else doc[name]
    def current_revision(doc):
        refs = [doc[k] for k in ("native_print_identity_rebind", "native_material_identity_revision") if k in doc]
        if len(refs) != 1: raise ValueError("Exactly one current source revision is required")
        return refs[0]
    while packet_ref["sha256"] != saved_sha:
        if packet_ref["sha256"] in seen: raise ValueError("Cyclic continuation source revision ancestry")
        seen.add(packet_ref["sha256"])
        packet = document(packet_ref); revision = document(revision_ref); qualification = document(qualification_ref)
        parent_ref = revision.get("parent_packet", {}); parent = document(parent_ref)
        parent_qual_ref = revision.get("parent_qualification", {}); parent_qual = document(parent_qual_ref)
        rebind = revision.get("schema") == "rinkfit.native_print_identity_rebind.v1"
        withdrawal = revision.get("schema") == "rinkfit.native_material_identity_revision.v1"
        expected = "rinkfit.native_print_identity_rebind_qualification.v1" if rebind else "rinkfit.native_material_identity_revision_qualification.v1"
        if (not (rebind or withdrawal) or revision.get("source_rows_deleted") != 0
                or revision.get("source_UV_ID_PTS_and_membership_unchanged" if rebind else "source_UV_ID_PTS_and_operators_unchanged") is not True
                or not _same_ref(current_revision(packet), revision_ref)
                or not _same_ref(packet.get("dense_parent_packet", {}), parent_ref)
                or qualification.get("schema") != expected
                or not _same_ref(qualification.get("source_packet", {}), packet_ref)
                or not _same_ref(qualification.get("revision", {}), revision_ref)
                or not _same_ref(qualification.get("parent_packet", {}), parent_ref)
                or not _same_ref(qualification.get("parent_qualification", {}), parent_qual_ref)
                or not _same_ref(parent_qual.get("source_packet", {}), parent_ref)):
            raise ValueError("Continuation mixed revision ancestry/qualification mismatch")
        count = inherited(packet, "row_count")
        if raw_rows is None: raw_rows = count
        if count != raw_rows or inherited(parent, "row_count") != count or revision.get("raw_rows") != count:
            raise ValueError("Continuation revision changed raw rows")
        review = document(revision["review"]); rows = review["rows"]; ids = [int(row["raw_row"]) for row in rows]
        declared = revision.get("rebound_rows" if rebind else "withheld_rows", [])
        if not ids or len(set(ids)) != len(ids) or sorted(ids) != sorted(declared) or any(i < 0 or i >= count for i in ids):
            raise ValueError("Continuation review row partition differs from revision")
        if rebind:
            if review.get("source_pixels_changed") is not False or review.get("source_model_projection_used") is not False:
                raise ValueError("Continuation review changed or projected source pixels")
            if any(not np.isfinite(row["new_station"]) or not 0 <= row["new_station"] <= 1 for row in rows):
                raise ValueError("Continuation rebind station invalid")
        else:
            if review.get("schema") != "rinkfit.native_material_identity_review.v1" or review.get("qualification_uses_model_residual") is not False:
                raise ValueError("Continuation withdrawal needs original RGB qualification review")
            proof(review["native_frame_extractor"])
            for row in rows:
                if row.get("qualification") != "unknown" or not row.get("finding") or not row.get("native_images"):
                    raise ValueError("Continuation withdrawal lacks native source finding")
                for image in row["native_images"]: proof(image)
        for name in ("active_rows", "admitted_rows"):
            if inherited(packet, name) != inherited(parent, name) - (0 if rebind else len(ids)):
                raise ValueError("Continuation revision changed unreviewed membership")
        permitted = ("material_index", "path") if rebind else ("admission", "fit_role")
        for name, value in packet.get("arrays", {}).items():
            if name not in permitted and value != inherited(parent, name, True):
                raise ValueError("Continuation revision changed source array: " + name)
        metadata = {"schema", "dense_parent_packet", "row_count", "arrays", "active_rows", "admitted_rows", "native_print_identity_rebind", "native_material_identity_revision", "dense_parent_append"}
        for name, value in packet.items():
            if name not in metadata and value != inherited(parent, name):
                raise ValueError("Continuation revision changed source semantics: " + name)
        chain.append(dict(kind="rebind" if rebind else "withdrawal", packet=packet_ref, revision=revision_ref, review=revision["review"], rows=rows))
        packet_ref, qualification_ref = parent_ref, parent_qual_ref
        if packet_ref["sha256"] != saved_sha: revision_ref = current_revision(parent)
    merged = {}; withdrawn = {}
    for generation in reversed(chain):
        for row in generation["rows"]:
            i = int(row["raw_row"])
            if i in withdrawn: raise ValueError("Continuation revised an already withdrawn observation")
            previous = merged.get(i)
            if generation["kind"] == "withdrawal":
                if previous and any(previous.get(k) != row.get(k) for k in ("source_row_id", "native_pts", "uv")):
                    raise ValueError("Continuation withdrawal source identity is discontinuous")
                withdrawn[i] = row; continue
            if previous:
                if (previous["new_material_index"] != row["old_material_index"]
                        or previous.get("new_path", previous.get("old_path", 0)) != row.get("old_path", 0)
                        or any(previous.get(k) != row.get(k) for k in ("source_row_id", "native_pts", "uv"))):
                    raise ValueError("Continuation review identity transition is discontinuous")
                row = dict(row, old_material_index=previous["old_material_index"], old_path=previous.get("old_path", 0))
            merged[i] = row
    return dict(generations=[{k:v for k,v in g.items() if k != "rows"} for g in reversed(chain)],
                rows=[merged[i] for i in sorted(merged)], withheld_rows=sorted(withdrawn),
                withdrawal_rows=[withdrawn[i] for i in sorted(withdrawn)], raw_rows=raw_rows)


def prepare_run(spec,*,output_must_be_new=True,source_probe=None):
    """Read pinned actual inputs and prepare the exact command; never launch."""
    spec,spec_ref=_load_spec(spec);out=Path(spec["output"]).resolve()
    if output_must_be_new and out.exists():raise FileExistsError("Immutable solve output already exists: "+str(out))
    paths={key:_check(spec[key])for key in ("model","packet","native","configuration")}
    cfg=read_json(paths["configuration"])
    if "current_input_v1" in cfg:
        if source_probe is not None:raise ValueError("Current input cannot consume a historical source probe")
        from .current_input_v1 import prepare_current_run
        return prepare_current_run(spec,spec_ref,paths)
    if cfg.get("historical_source_replay_audit_only_v1") is not True:
        raise ValueError("Current input is not prepared: current_input_v1 is required; historical replay is an explicit offline audit entry")
    recipe_path=paths["native"].parent/"recipe.json";recipe=read_json(recipe_path)
    if file_sha256(paths["native"])!=recipe["executable_sha256"]:raise ValueError("Native executable differs from actual build recipe")
    for name, sha in recipe["source_files"].items():
        source=ROOT/"calibration/rinkfit/cpp"/name
        if file_sha256(source)!=sha:raise ValueError("Rebuild native executable; source changed: "+name)
    cfg=read_json(paths["configuration"])
    if cfg.get("engineering_active_shot_ids")!=SHOTS:raise ValueError("Six-shot configured source scope differs")
    linear_backend=cfg.get("linear_solver_backend")
    if linear_backend not in ("cuda_cgnr","cpu_eigen_sparse_schur") or cfg.get("gpu_raw_uv",{}).get("batch_reuse")is not True:raise ValueError("Qualified native linear backend and GPU raw-UV batch cache must be enabled")
    objective=cfg.get("source_uv_capture_objective");provisional=objective=="rinkfit.sourceUV_observed_row_finite_initialization.v1"
    if cfg.get("dense_source_fresh_capture_v1")is not True or objective not in ("rinkfit.sourceUV_trueRS_finite_capture.v3","rinkfit.sourceUV_observed_row_finite_initialization.v1"):raise ValueError("Qualified fresh dense capture or explicit observed-row initialization entry required")
    if provisional and (cfg.get("reinitialize_native_auxiliaries",False)or cfg.get("source_uv_capture_reseed_new_sources",False)):raise ValueError("Observed-row initialization must use the qualified packet without historical auxiliary reinitialization")
    if cfg.get("capture_primal_restart_v2") or cfg.get("capture_primal_transition_v2"):
        from .capture_restart import validate
        validate(cfg,spec["model"],spec["packet"])
    if cfg.get("representative_full_source_transition_v1") and not cfg.get("representative_continuation_v1"):
        raise ValueError("Full-source transition requires complete representative continuation")
    if cfg.get("complete_stage_primal_v1"):
        from .complete_stage_primal import validate
        validate(cfg,spec["model"],spec["packet"])
    elif cfg.get("representative_continuation_v1"):
        # Match the native entry priority: an exact saved representative primal
        # supersedes the historical initializer retained in the configuration.
        saved = read_json(_check(cfg["representative_continuation_v1"]))
        if saved.get("complete_checkpoint") is not True or saved.get("schema") != "rinkfit.representative_complete_primal_continuation.v1":
            raise ValueError("A complete representative continuation is required")
        for name in ("model", "checkpoint", "explicit_nuisance_manifest", "representative_source_selection_v1"):
            _check(saved[name])
        if cfg.get("representative_full_source_transition_v1"):
            from .representative_full_source import validate
            validate(cfg, spec["packet"], saved)
        if saved["source_packet_sha256"] != spec["packet"]["sha256"]:
            if cfg.get('representative_source_append_primal_v1'):
                from .training_source_append_v2 import validate_append_primal_policy
                validate_append_primal_policy(saved,cfg,spec['packet'])
            else:
                packet_document = json.loads(paths["packet"].read_text(encoding="utf-8-sig"))
                rebind = "native_print_identity_rebind" in packet_document
                revision_key = "native_print_identity_rebind" if rebind else "native_material_identity_revision"
                execution_key = "representative_material_rebind_v1" if rebind else "representative_source_withdrawal_v1"
                if not _same_ref(packet_document.get(revision_key, {}), cfg.get(execution_key, {})):
                    raise ValueError("Continuation must authorize the current packet's actual revision")
                representative_identity_ancestry(saved["source_packet_sha256"], spec["packet"],
                                                 cfg[execution_key], actual_source_qualification(cfg))
    elif cfg.get("observed_primal_restart_v1") or cfg.get("observed_primal_warm_start_v1"):
        from .observed_primal_restart import validate_restart_manifest
        validate_restart_manifest(cfg, spec["model"], spec["packet"])
    if cfg.get("source_uv_capture_source_append_v4")or cfg.get("source_uv_capture_source_append_v5")or any(cfg.get(k)for k in ("resume_engineering_outer_state","resume_engineering_checkpoint","resume_source_uv_capture_checkpoint")):raise ValueError("Fresh source generation cannot import historical AL/primal resume state")
    graph_only = cfg.get("source_uv_capture_construct_problem_only", False) is True
    complete_observed_import=any(cfg.get(k) for k in ('observed_primal_restart_v1','observed_primal_warm_start_v1','capture_primal_restart_v2','capture_primal_transition_v2','representative_continuation_v1','complete_stage_primal_v1'))
    full_graph_preflight=cfg.get('source_uv_capture_preflight_only') is True
    if cfg.get('preflight_only',False) or (full_graph_preflight and not (cfg.get('effective_input_v1') and complete_observed_import and not graph_only)) or (graph_only and not complete_observed_import):
        raise ValueError("Graph preflight requires its complete observed import and explicit effective full-graph mode")
    effective_report=None
    if 'effective_input_v1' in cfg:
        from .effective_input_prepare_v1 import validate_effective_configuration
        effective_report=validate_effective_configuration(cfg,spec['packet'],spec['model'],recipe)
        refs={key:cfg[key] for key in ('effective_input_v1','capture_certified_numeric_domain','archived_dependency_aliases')}
    else:
        refs={key:cfg[key]for key in ("dense_source_qualification","engineering_dense_source_weights_v1","capture_certified_numeric_domain","archived_dependency_aliases")}
    pinned={key:_check(ref,paths["configuration"].parent)for key,ref in refs.items()}
    gpu=cfg["gpu_raw_uv"];ptx=Path(gpu["ptx"]).resolve()
    if file_sha256(ptx)!=gpu["ptx_sha256"]:raise ValueError("Pinned GPU PTX changed")
    for key in ("shadow_receipt","batch_reuse_qualification"):
        if key not in gpu:raise ValueError("Actual GPU qualification reference missing: "+key)
        _check(gpu[key],paths["configuration"].parent)
    packet=_run_packet_header(paths["packet"])
    if effective_report:
        qual=read_json(_check(effective_report['original_qualification']));weights=None
        if not _same_ref(qual['source_packet'],spec['packet']):raise ValueError('Effective source proof binds another original packet')
    else:
        qual=read_json(pinned["dense_source_qualification"]);weights=read_json(pinned["engineering_dense_source_weights_v1"])
        if not _same_ref(qual["source_packet"],spec["packet"])or not _same_ref(weights["source_packet"],spec["packet"]):raise ValueError("Source and budget proofs bind a different immutable packet")
        if not _same_ref(weights["dense_source_qualification"],refs["dense_source_qualification"]):raise ValueError("Engineering weights bind a different qualification generation")
    domain=read_json(pinned["capture_certified_numeric_domain"])
    source_activity_model=effective_report["source_model"] if effective_report else spec["model"]
    if not _same_ref(domain["activity_model"],source_activity_model):raise ValueError("Numerical domain binds a different immutable source activity model")
    if packet["shot_ids"]!=SHOTS or packet["engineering_scope"]["shot_ids"]!=SHOTS:raise ValueError("Actual packet six-shot source scope differs")
    native_ad=qual["schema"] in ("rinkfit.native_ad_source_append_qualification.v1","rinkfit.native_surface_source_append_qualification.v2","rinkfit.native_named_print_append_qualification.v3") or (qual["schema"] in ("rinkfit.native_print_identity_rebind_qualification.v1","rinkfit.native_material_identity_revision_qualification.v1","rinkfit.native_material_recovery_qualification.v1") and qual["registry_kind"]=="native_ad_source_binding")
    registry=packet["native_ad_source_binding" if native_ad else "dense_source_binding"]
    n=int(packet["row_count"]);n0=int(registry["row_start"]);nnew=int(registry["count"])
    active=int(packet["active_rows"] if effective_report else weights["active_rows"]);newactive=int(qual.get("active_new_rows",nnew) if effective_report else weights["new_active_rows"]);unknown=nnew-newactive
    if n0+nnew!=n or int(qual["raw_rows"])!=n or int(qual["new_rows"])!=nnew or (not effective_report and int(weights["raw_rows"])!=n):raise ValueError("Full raw-source denominator differs")
    if int(packet["active_rows"])!=active or int(qual.get("active_new_rows",nnew))!=newactive or int(qual.get("unknown_new_rows",nnew-int(qual.get("active_new_rows",nnew))))!=unknown:raise ValueError("Current active/unknown qualification denominator differs")
    identity_verdict = cfg.get("source_identity_verdicts_v1")
    identity_report = None
    if identity_verdict:
        from .source_identity_guard import validate_run_binding
        selection = read_json(_check(cfg["representative_source_selection_v1"]))
        if not _same_ref(selection["source_packet"], spec["packet"]):
            raise ValueError("Identity verdict must use the current representative source packet")
        identity_report = validate_run_binding(paths["packet"], selection,
            read_json(_check(identity_verdict)), lambda ref: _check(ref, paths["configuration"].parent))
    current_generation=None
    if packet.get("dense_qualification_binding") and not native_ad:
        qbinding=_check(packet["dense_qualification_binding"],paths["packet"].parent);current=read_json(qbinding)
        if int(current["total_raw_rows"])!=nnew or int(current["total_active_rows"])!=newactive or int(current["total_unknown_rows"])!=unknown:raise ValueError("Current source generation differs from qualified membership")
        _check(current["source_observation_generation"],qbinding.parent);current_generation=current["source_observation_generation"]
    live_review=cfg.get("accepted_source_anchor_review")
    if live_review:
        review_path=_check(live_review,paths["configuration"].parent);review=read_json(review_path)
        if review.get("schema")!="rinkfit.dense6.accepted_source_anchor_review.v1" or not isinstance(cfg.get("write_accepted_physical_preview"),bool):raise ValueError("Actual accepted-source review requires an explicit preview policy")
        _check(review["baseline"],review_path.parent)
        from .dense_live_review import validate_anchor_review_binding
        validate_anchor_review_binding(live_review,paths["packet"])
        for key,filename in (("guard","dense_fixed_anchor_guard.py"),("policy","dense_live_review.py"),("strata","fixed_local_anchor_audit.py")):
            actual=_check(review[key],review_path.parent)
            if actual!=Path(__file__).with_name(filename).resolve():raise ValueError("Accepted source review must use its actual installed consumer")
    cache=Path(spec.get("cuda_cache_path")or os.environ.get("CUDA_CACHE_PATH")or(ptx.parent/"cuda_driver_cache")).resolve()
    command=[str(paths["native"]),"--engineering-recovery",str(paths["model"]),str(paths["packet"]),str(out),str(paths["configuration"])]
    info=dict(schema="rinkfit.dense6_public_native_run_preparation.v1",spec=spec_ref,consumer=reference(__file__),
        output=str(out),model=spec["model"],packet=spec["packet"],native=spec["native"],configuration=spec["configuration"],
        native_recipe=reference(recipe_path),proofs=refs,command=command,runtime_bin_paths=recipe["runtime_bin_paths"],
        cuda_driver_cache_path=str(cache),native_raw_UV_batch_reuse=True,linear_backend=linear_backend,linear_solve_on_GPU=linear_backend=="cuda_cgnr",PTX=dict(file=str(ptx),sha256=gpu["ptx_sha256"]),
        raw_rows=n,parent_raw_rows=n0,new_raw_rows=nnew,active_rows=active,new_active_rows=newactive,unknown_new_rows=unknown,
        current_source_generation=current_generation,source_full_denominator_retained=True,
        source_identity_verdict_check=identity_report,
        objective_id=objective,numerical_initialization_only=provisional,true_RS_verification_required_after_this_run=provisional,phase_II_allowed=False,
        actual_native_source_gate_required_on_every_execution=True,source_accuracy_or_model_quality_accepted=False,
        native_launched=False,accepted_source_anchor_review=live_review if cfg.get("write_accepted_physical_preview") else None,source_anchor_review_baseline=live_review,effective_input_preflight=effective_report,graph_verification_only=graph_only or full_graph_preflight,full_graph_preflight_without_solver=full_graph_preflight)
    if source_probe is not None:
        if effective_report:raise ValueError("Legacy source/weight probes cannot certify an effective-input ABI")
        probe_path=_check(source_probe)if isinstance(source_probe,dict)else Path(source_probe).resolve();probe=read_json(probe_path)
        if probe.get("passed")is not True or int(probe["raw_rows"])!=n or not _same_ref(probe["source"]["binding"],refs["dense_source_qualification"])or not _same_ref(probe["engineering_weights"]["binding"],refs["engineering_dense_source_weights_v1"]):raise ValueError("Stored native probe does not consume this qualification and budget generation")
        if probe["source"].get("all_source_UV_ID_PTS_material_and_store_ordinals_verified")is not True or int(probe["source"]["new_unknown_rows_retained"])!=unknown or int(probe["engineering_weights"]["active_rows"])!=active:raise ValueError("Stored native probe source denominator differs")
        info["existing_native_probe"]=reference(probe_path);info["existing_probe_reused_without_reexecuting_gate"]=True
    return info


def record_initial_capture_rejection(result, rejection):
    actual=read_json(rejection)
    if actual.get("solver_started") is not False or actual.get("quality_accepted") is not False or actual.get("complete_denominator_retained") is not True:
        raise ValueError("Initial capture rejection must preserve its no-solve/full-denominator boundary")
    seeds=actual["seeds"];active=int(seeds["active_rows"]);failed=int(seeds["failed_RS_rows_retained"])
    if active!=result["source_active_rows"] or not 0<failed<=active:
        raise ValueError("Initial capture rejection source denominator differs")
    result.pop("missing_native_graph_receipt",None);result.pop("missing_final_native_receipt",None)
    result.update(initial_source_capture_rejection=reference(rejection),solver_started=False,
                  initialization_ready=False,failed_RS_rows_retained=failed,
                  converged=False,solution_usable=False,requires_feasible_observed_initialization=True)
    return result


def execute(spec):
    """Launch one actual fresh native process and persist actual execution receipts."""
    info=prepare_run(spec);out=Path(info["output"]);log=out.parent/(out.name+".log");launch_path=out.parent/(out.name+"_launch.json");execution_path=out.parent/(out.name+"_execution.json")
    for path in (log,launch_path,execution_path):
        if path.exists():raise FileExistsError("Immutable run artifact already exists: "+str(path))
    env=os.environ.copy();env["PATH"]=os.pathsep.join(info["runtime_bin_paths"])+os.pathsep+env.get("PATH","")
    env.update(OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1",CUDA_CACHE_PATH=info["cuda_driver_cache_path"])
    Path(info["cuda_driver_cache_path"]).mkdir(exist_ok=True,parents=True);out.parent.mkdir(exist_ok=True,parents=True)
    guardian=None
    if info.get("accepted_source_anchor_review"):
        from .dense_live_review import AcceptedSourceReview
        guardian=AcceptedSourceReview(out,info["packet"]["file"],info["accepted_source_anchor_review"])
    started=time.perf_counter();launch=dict(info,schema="rinkfit.joint6_actual_GPU_launch.v1",driver_pid=os.getpid(),started_unix=time.time(),quality_accepted=False)
    with log.open("x",encoding="utf8")as stream:
        proc=subprocess.Popen(info["command"],cwd=info.get("runtime_working_directory",ROOT),env=env,stdout=stream,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
        launch.update(native_pid=proc.pid,native_launched=True)
        try:launch_ref=_write(launch_path,launch)
        except BaseException:
            proc.terminate();proc.wait();raise
        while True:
            if guardian:
                try:guardian.poll()
                except Exception as error:guardian.fail_closed(error)
            try:code=proc.wait(timeout=.5);break
            except subprocess.TimeoutExpired:pass
        if guardian:
            try:guardian.poll()
            except Exception as error:guardian.fail_closed(error)
    result=dict(schema="rinkfit.joint6_actual_GPU_execution.v1",returncode=code,seconds=time.perf_counter()-started,
        launch=launch_ref,log=reference(log),source_raw_rows=info["raw_rows"],source_active_rows=info["active_rows"],
        source_unknown_rows_retained=info["unknown_new_rows"],objective_id=info["objective_id"],numerical_initialization_only=info["numerical_initialization_only"],quality_accepted=False,phase_II_allowed=False)
    receipt=out/("source_uv_observed_row_initialization_receipt.json"if info["numerical_initialization_only"]else"source_uv_capture_receipt.json")
    if info.get("graph_verification_only"):
        graph = out / "actual_production_graph_preflight.json"
        if graph.is_file():
            checked_graph = read_json(graph)
            if checked_graph.get("Solve_started") is not False or checked_graph.get("quality_accepted") is not False:
                raise ValueError("Native graph verification violated its no-solve boundary")
            result.update(solver_started=False, graph_verification_only=True, graph_receipt=reference(graph),
                          converged=False, solution_usable=None, complete_state_token=None)
        else:
            result.update(solver_started=None, graph_verification_only=True, missing_native_graph_receipt=True)
    elif receipt.is_file():
        actual=read_json(receipt);result.update(solver_started=actual.get("solver_started"),actual_receipt=reference(receipt),solution_usable=actual.get("solution_usable"),complete_state_token=actual.get("complete_state_token"),stopped=actual.get("stopped"),callback_error=actual.get("callback_error"),termination_type=actual.get("termination_type"),converged=actual.get("converged",False))
        if info["numerical_initialization_only"]:
            if actual.get("true_RS_verified")is not False or actual.get("fresh_trueRS_v3_required")is not True or actual.get("phase_II_allowed")is not False:raise ValueError("Native provisional result omitted its true-RS verification boundary")
            result.update(true_RS_verified=False,fresh_trueRS_v3_required=True)
    else:result.update(solver_started=None,actual_receipt=None,missing_final_native_receipt=True)
    if info.get("accepted_source_anchor_review"):
        result["accepted_source_anchor_review"]=info["accepted_source_anchor_review"]
        if (out/"accepted_source_review/latest.json").is_file():result["actual_live_source_reviews"]=reference(out/"accepted_source_review/latest.json")
        if (out/"accepted_source_review/error.json").is_file():result["live_source_review_error"]=reference(out/"accepted_source_review/error.json")
    rejection=out/"source_uv_capture_initial_rejection.json"
    if rejection.is_file():record_initial_capture_rejection(result,rejection)
    _write(execution_path,result)
    if result.get("initialization_ready") is False:
        raise RuntimeError("True-RS initialization is not ready: "+str(result["failed_RS_rows_retained"])+" of "+str(result["source_active_rows"])+" active rows have no valid initial RS state; no Solve started and all sources retained. See "+str(rejection))
    if code:raise RuntimeError("Native dense run exited "+str(code)+"; see "+str(log))
    if result.get("missing_native_graph_receipt"):
        raise RuntimeError("Native graph verification omitted its required no-solve receipt; see " + str(log))
    if result.get("missing_final_native_receipt"):
        raise RuntimeError("Native process omitted its required final receipt; see " + str(log))
    if result.get("live_source_review_error") or result.get("callback_error"):
        raise RuntimeError("Source observer or native callback failed; see " + str(execution_path))
    if result.get("termination_type") in ("FAILURE", "USER_FAILURE") or result.get("solution_usable") is False:
        raise RuntimeError("Native solver did not return a usable state; see " + str(execution_path))
    return result


def bind_current_generation_rows(packet_path,row_indices):
    """Return actual current semantic-stream locators, retaining unknown source IDs."""
    path=Path(packet_path).resolve();doc=read_json(path)
    links=current_dense_packet_source_bindings(path,row_indices)
    admission=_array(doc["arrays"]["admission"],path.parent);role=_array(doc["arrays"]["fit_role"],path.parent)
    for key,link in links.items():
        index=int(key);link["current_fit_active"]=bool(admission[index]and role[index])
        link["qualification_packet"]=reference(path)
        if doc.get("dense_qualification_binding"):link["current_qualification_binding"]=doc["dense_qualification_binding"]
    return links


def bind_current_generation_findings(packet_path,task_document,output):
    """Reuse real task codec and dense review adapter with current source streams."""
    from .feedback_task_codec_v1 import read_task_document,write_task_document
    from .dense_fit_feedback import bind_packet_findings
    tasks=read_task_document(task_document);indices=[int(t["source_row_index"])for t in tasks["tasks"]]
    links=bind_current_generation_rows(packet_path,indices);parts=defaultdict(list);pending=[]
    for task in tasks["tasks"]:
        link=links.get(str(task["source_row_index"]))
        if link is None:pending.append(dict(finding_id=task["finding_id"],source_row_index=task["source_row_index"],reason="retained_parent_source_requires_its_existing_source_reader"));continue
        parts[object_sha256(link["rows_ref"])].append(task)
    out=Path(output).resolve();out.mkdir(exist_ok=False,parents=True);deliveries=[]
    for number,group in enumerate(parts.values()):
        subset=dict(deepcopy(tasks),tasks=group,parent_task_document=reference(task_document));task_path=out/("tasks_"+str(number)+".json")
        write_task_document(task_path,subset);indices=[str(t["source_row_index"])for t in group];mapping={i:links[i]for i in indices}
        rows_ref=mapping[indices[0]]["rows_ref"]
        result=bind_packet_findings(task_path,rows_ref,mapping,out/("findings_"+str(number)+".json"))
        deliveries.append(dict(task_document=reference(task_path),source_findings=result,source_rows=rows_ref,tasks=len(group)))
    result=dict(schema="rinkfit.current_dense_generation_finding_bindings.v1",packet=reference(packet_path),
        parent_task_document=reference(task_document),consumer=reference(__file__),deliveries=deliveries,pending=pending,
        original_source_UV_changed=False,qualification_or_packet_mutated=False,review_dispatch_started=False)
    _write(out/"manifest.json",result);return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("spec");parser.add_argument("--inspect",action="store_true");parser.add_argument("--source-probe")
    args=parser.parse_args();result=prepare_run(args.spec,output_must_be_new=False,source_probe=args.source_probe)if args.inspect else execute(args.spec)
    print(json.dumps(result,ensure_ascii=False))

if __name__=="__main__":main()
