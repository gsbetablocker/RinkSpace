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
    return left["sha256"]==right["sha256"] and Path(left["file"]).resolve()==Path(right["file"]).resolve()


def _write(path,value):
    path=Path(path).resolve()
    if len(str(path).encode("utf-16-le"))//2>=260 or any(len(x.encode("utf-16-le"))//2>255 for x in path.parts):raise ValueError("Run artifact exceeds Windows path limits")
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x",encoding="utf8")as f:json.dump(value,f,ensure_ascii=False,separators=(",",":"),allow_nan=False)
    return reference(path)


def _load_spec(spec):
    if isinstance(spec,dict):return deepcopy(spec),dict(kind="in_memory_spec",object_sha256=object_sha256(spec))
    path=Path(spec).resolve();return read_json(path),reference(path)


def prepare_run(spec,*,output_must_be_new=True,source_probe=None):
    """Read pinned actual inputs and prepare the exact command; never launch."""
    spec,spec_ref=_load_spec(spec);out=Path(spec["output"]).resolve()
    if output_must_be_new and out.exists():raise FileExistsError("Immutable solve output already exists: "+str(out))
    paths={key:_check(spec[key])for key in ("model","packet","native","configuration")}
    recipe_path=paths["native"].parent/"recipe.json";recipe=read_json(recipe_path)
    if file_sha256(paths["native"])!=recipe["executable_sha256"]:raise ValueError("Native executable differs from actual build recipe")
    cfg=read_json(paths["configuration"])
    if cfg.get("engineering_active_shot_ids")!=SHOTS:raise ValueError("Six-shot configured source scope differs")
    linear_backend=cfg.get("linear_solver_backend")
    if linear_backend not in ("cuda_cgnr","cpu_eigen_sparse_schur") or cfg.get("gpu_raw_uv",{}).get("batch_reuse")is not True:raise ValueError("Qualified native linear backend and GPU raw-UV batch cache must be enabled")
    objective=cfg.get("source_uv_capture_objective");provisional=objective=="rinkfit.sourceUV_observed_row_finite_initialization.v1"
    if cfg.get("dense_source_fresh_capture_v1")is not True or objective not in ("rinkfit.sourceUV_trueRS_finite_capture.v3","rinkfit.sourceUV_observed_row_finite_initialization.v1"):raise ValueError("Qualified fresh dense capture or explicit observed-row initialization entry required")
    if provisional and (cfg.get("reinitialize_native_auxiliaries",False)or cfg.get("source_uv_capture_reseed_new_sources",False)):raise ValueError("Observed-row initialization must use the qualified packet without historical auxiliary reinitialization")
    if cfg.get("source_uv_capture_source_append_v4")or cfg.get("source_uv_capture_source_append_v5")or any(cfg.get(k)for k in ("resume_engineering_outer_state","resume_engineering_checkpoint","resume_source_uv_capture_checkpoint")):raise ValueError("Fresh source generation cannot import historical AL/primal resume state")
    if any(cfg.get(k,False)for k in ("preflight_only","source_uv_capture_preflight_only","source_uv_capture_construct_problem_only")):raise ValueError("This public run entry requires actual execution, not graph-only or preflight-only consumption")
    refs={key:cfg[key]for key in ("dense_source_qualification","engineering_dense_source_weights_v1","capture_certified_numeric_domain","archived_dependency_aliases")}
    pinned={key:_check(ref,paths["configuration"].parent)for key,ref in refs.items()}
    gpu=cfg["gpu_raw_uv"];ptx=Path(gpu["ptx"]).resolve()
    if file_sha256(ptx)!=gpu["ptx_sha256"]:raise ValueError("Pinned GPU PTX changed")
    for key in ("shadow_receipt","batch_reuse_qualification"):
        if key not in gpu:raise ValueError("Actual GPU qualification reference missing: "+key)
        _check(gpu[key],paths["configuration"].parent)
    packet=read_json(paths["packet"]);qual=read_json(pinned["dense_source_qualification"]);weights=read_json(pinned["engineering_dense_source_weights_v1"])
    if not _same_ref(qual["source_packet"],spec["packet"])or not _same_ref(weights["source_packet"],spec["packet"]):raise ValueError("Source and budget proofs bind a different immutable packet")
    if not _same_ref(weights["dense_source_qualification"],refs["dense_source_qualification"]):raise ValueError("Engineering weights bind a different qualification generation")
    domain=read_json(pinned["capture_certified_numeric_domain"])
    if not _same_ref(domain["activity_model"],spec["model"]):raise ValueError("Numerical domain binds a different activity model")
    if packet["shot_ids"]!=SHOTS or packet["engineering_scope"]["shot_ids"]!=SHOTS:raise ValueError("Actual packet six-shot source scope differs")
    n=int(packet["row_count"]);n0=int(packet["dense_source_binding"]["row_start"]);nnew=int(packet["dense_source_binding"]["count"])
    active=int(weights["active_rows"]);newactive=int(weights["new_active_rows"]);unknown=nnew-newactive
    if n0+nnew!=n or int(qual["raw_rows"])!=n or int(qual["new_rows"])!=nnew or int(weights["raw_rows"])!=n:raise ValueError("Full raw-source denominator differs")
    if int(packet["active_rows"])!=active or int(qual.get("active_new_rows",nnew))!=newactive or int(qual.get("unknown_new_rows",0))!=unknown:raise ValueError("Current active/unknown qualification denominator differs")
    current_generation=None
    if packet.get("dense_qualification_binding"):
        qbinding=_check(packet["dense_qualification_binding"],paths["packet"].parent);current=read_json(qbinding)
        if int(current["total_raw_rows"])!=nnew or int(current["total_active_rows"])!=newactive or int(current["total_unknown_rows"])!=unknown:raise ValueError("Current source generation differs from qualified membership")
        _check(current["source_observation_generation"],qbinding.parent);current_generation=current["source_observation_generation"]
    live_review=cfg.get("accepted_source_anchor_review")
    if live_review:
        review_path=_check(live_review,paths["configuration"].parent);review=read_json(review_path)
        if review.get("schema")!="rinkfit.dense6.accepted_source_anchor_review.v1" or cfg.get("write_accepted_physical_preview")is not True:raise ValueError("Actual accepted-source review requires stable physical previews")
        _check(review["baseline"],review_path.parent)
        for key,filename in (("guard","dense_fixed_anchor_guard.py"),("policy","dense_live_review.py")):
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
        objective_id=objective,numerical_initialization_only=provisional,true_RS_verification_required_after_this_run=provisional,phase_II_allowed=False,
        actual_native_source_gate_required_on_every_execution=True,source_accuracy_or_model_quality_accepted=False,
        native_launched=False,accepted_source_anchor_review=live_review)
    if source_probe is not None:
        probe_path=_check(source_probe)if isinstance(source_probe,dict)else Path(source_probe).resolve();probe=read_json(probe_path)
        if probe.get("passed")is not True or int(probe["raw_rows"])!=n or not _same_ref(probe["source"]["binding"],refs["dense_source_qualification"])or not _same_ref(probe["engineering_weights"]["binding"],refs["engineering_dense_source_weights_v1"]):raise ValueError("Stored native probe does not consume this qualification and budget generation")
        if probe["source"].get("all_source_UV_ID_PTS_material_and_store_ordinals_verified")is not True or int(probe["source"]["new_unknown_rows_retained"])!=unknown or int(probe["engineering_weights"]["active_rows"])!=active:raise ValueError("Stored native probe source denominator differs")
        info["existing_native_probe"]=reference(probe_path);info["existing_probe_reused_without_reexecuting_gate"]=True
    return info


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
        proc=subprocess.Popen(info["command"],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
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
    if receipt.is_file():
        actual=read_json(receipt);result.update(solver_started=actual.get("solver_started"),actual_receipt=reference(receipt),solution_usable=actual.get("solution_usable"),complete_state_token=actual.get("complete_state_token"))
        if info["numerical_initialization_only"]:
            if actual.get("true_RS_verified")is not False or actual.get("fresh_trueRS_v3_required")is not True or actual.get("phase_II_allowed")is not False:raise ValueError("Native provisional result omitted its true-RS verification boundary")
            result.update(true_RS_verified=False,fresh_trueRS_v3_required=True)
    else:result.update(solver_started=None,actual_receipt=None,missing_final_native_receipt=True)
    if info.get("accepted_source_anchor_review"):
        result["accepted_source_anchor_review"]=info["accepted_source_anchor_review"]
        if (out/"accepted_source_review/latest.json").is_file():result["actual_live_source_reviews"]=reference(out/"accepted_source_review/latest.json")
        if (out/"accepted_source_review/error.json").is_file():result["live_source_review_error"]=reference(out/"accepted_source_review/error.json")
    rejection=out/"source_uv_capture_initial_rejection.json"
    if rejection.is_file():result["initial_source_capture_rejection"]=reference(rejection);result["solver_started"]=False
    _write(execution_path,result)
    if code:raise RuntimeError("Native dense run exited "+str(code)+"; see "+str(log))
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
