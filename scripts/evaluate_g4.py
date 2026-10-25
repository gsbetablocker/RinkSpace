"""Run the same native LM controller on the G3 accepted window. No Python optimizer."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import runpy
import time
import traceback
import numpy as np
from rinkspace.core.native import NativeCore
from rinkspace.core.gpu import ResidentResidualProblem
from rinkspace.session.gpu_problem import load_linearization_problem
from rinkspace.path_policy import guard_project_write

g3=runpy.run_path(str(Path(__file__).with_name("evaluate_g3.py")))
read,write,sha,memory,quantile,inputs,PLAN_KEYS,available_host,objective,compare=(
    g3[n] for n in ("read","write","sha","memory","quantile","inputs","PLAN_KEYS","available_host","objective","compare"))

def note(out,message):
    print(message,flush=True)
    with (out/"main.log").open("a",encoding="utf8") as f:f.write(message+"\n")

def save_state(root,p,state,result):
    for name in ("controls.npy","stations.npy","camera.json","state.json"):guard_project_write(root/name)
    root.mkdir(parents=True,exist_ok=True)
    for key in state:np.save(root/(key+".npy"),state[key],allow_pickle=False)
    write(root/"camera.json",{**p["camera"],"controls_ref":"controls.npy",
        "g4_origin":{"geometry_updated":False,"original_control_grid":True,"source_state":"../g3/state.json"}})
    write(root/"state.json",{"schema":"rinkspace.g4.saved_state.v1","user_approval":"not_reviewed",
        "optimizer_result":result,"camera_ref":"camera.json","scene_updated":False,
        "controls_ref":{"path":"controls.npy","sha256":sha(root/"controls.npy")},
        "row_station_ref":{"path":"stations.npy","sha256":sha(root/"stations.npy")},
        "station_row_order":"same G1 inputs.npz rows and G2 original local identities",
        "activity_ref":"../g4.json","full_resume_checkpoint":False})

def statistics(p,raw):
    result={}
    for label,mask in [("all",np.ones(len(p["arrays"]["pts"]),bool))]+[
        (x["label"],p["arrays"]["codes"]==x["native_code"]) for x in p["problem"]["participating_entities"]]:
        result[label]={key:quantile(np.linalg.norm(value[mask,9:11],axis=1)) for key,value in raw.items()}
    frames=[]
    for pts in p["problem"]["native_pts"]:
        mask=p["arrays"]["pts"]==pts
        frames.append({"pts":pts,"source_seconds":pts/12800,"rows":int(mask.sum()),
                       **{key:quantile(np.linalg.norm(value[mask,9:11],axis=1)) for key,value in raw.items()}})
    return result,frames

def run(args):
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    note(out,f"G4 main log: {out/'main.log'}")
    if (out/"iterations.jsonl").exists():raise ValueError("An actual run already exists; use --review-existing for display only")
    p=load_linearization_problem(args.g1,args.g2);started=time.perf_counter()
    protected=read(args.g3/"protected_before.json")
    identity=read(args.g3/"input_identity.json")
    for item in protected:
        if sha(item["path"])!=item["sha256"]:raise ValueError("protected native model/camera changed")
    for path,h in identity.items():
        if sha(path)!=h:raise ValueError("G1/G2 original input changed")
    saved=read(args.start_state/"state.json")
    before={}
    for key,refkey in (("controls","controls_ref"),("stations","row_station_ref")):
        ref=saved[refkey];path=args.start_state/ref["path"]
        if sha(path)!=ref["sha256"]:raise ValueError("G3 state identity mismatch")
        before[key]=np.load(path,allow_pickle=False);identity[str(path.resolve())]=sha(path)
    if before["controls"].shape!=p["controls"].shape or before["stations"].shape!=p["arrays"]["stations"].shape:
        raise ValueError("G3 state row/control binding mismatch")
    previous=read(args.g3/"decision.json")["summary"]
    write(out/"protected_before.json",protected);write(out/"input_identity.json",identity);write(out/"activity.json",p["activity"])
    core=NativeCore(args.library);host=available_host();mem0=memory()
    history=[]
    with ResidentResidualProblem(core,**inputs(p,**before)) as problem:
        lin=problem.configure_linearization(**{k:p[k] for k in PLAN_KEYS},huber_delta=p["activity"]["loss"]["delta"])
        options={"device_budget_bytes":problem.creation["free_after_alloc_bytes"]-lin["device_bytes"],
            "host_budget_bytes":min(host.values()),"initial_lambda":previous["lambda_next"]}
        solver=problem.configure_solver(lower=p["lower"],upper=p["upper"],**options)
        meta={"native_creation":problem.creation,"linearization_layout":lin,"solver_creation":solver,
              "options":options,"available_host_at_start":host,"process_before_resident":mem0}
        write(out/"runtime_metadata.json",meta)
        raw_before=problem.evaluate("cuda")
        def progress(event):
            history.append(event)
            with (out/"iterations.jsonl").open("a",encoding="utf8") as f:f.write(json.dumps(event,allow_nan=False)+"\n")
            m=event["summary"]
            if len(history)<=5 or len(history)%20==0 or not m["accepted"]:
                note(out,f'candidate={int(m["trial_count"])} accepted={m["accepted"]} cost={m["old_cost"]:.9g}->{m["trial_cost"]:.9g} lambda={m["lambda_used"]:.6g} alpha={m["feasible_alpha"]:.6g}')
        note(out,"phase=native_continuous_started; G3 saved state; same objective/W/Q3 activity; geometry frozen")
        try:
            result=problem.solve(progress=progress)
        except BaseException:
            # Preserve actual accepted output on failure; this is not a G5 restart checkpoint.
            np.savez(out/"accepted_on_failure.npz",**problem.export_state())
            meta["process_after_failure"]=memory();write(out/"runtime_metadata.json",meta)
            raise
        write(out/"solve.json",result)
        after=problem.export_state();np.savez(out/"accepted.npz",**after)
        raw_after=problem.evaluate("cuda");cpu=problem.evaluate("cpu")
        np.testing.assert_array_equal(raw_after["status"],0)
        np.testing.assert_array_equal(cpu["status"],0)
        checks={"final_projection_cpu_gpu":compare(raw_after["values"],cpu["values"],atol=1e-8),
            "original_objective_cpu_gpu":compare(np.array([result["final_cost"]]),np.array([objective(p,cpu["values"][:,9:11])]))}
        checks["original_station_supports_preserved"]=bool(np.all(after["stations"]>=p["arrays"]["support"][:,0]) and np.all(after["stations"]<=p["arrays"]["support"][:,1]))
        if not checks["original_station_supports_preserved"]:raise AssertionError("station support changed")
        delta=after["controls"]-before["controls"]
        checks["same_constant_camera_activity_max_deviation"]=float(np.max(np.abs(delta-delta[0])))
        np.testing.assert_allclose(delta, np.tile(delta[0],(len(delta),1)),rtol=0,atol=1e-12)
        meta["process_after_solve"]=memory();write(out/"runtime_metadata.json",meta)
        original=np.load(args.g3/"raw_before_after.npz",allow_pickle=False)["before"]
        raw={"original_G1":original,"before_G3":raw_before["values"],"after_G4":raw_after["values"]}
        np.savez(out/"raw_comparison.npz",**raw,status=raw_after["status"])
        stats,frames=statistics(p,raw);write(out/"frame_residuals.json",frames);write(out/"entity_residuals.json",stats)
        write(out/"numerical_checks.json",checks)
        full={"schema":"rinkspace.g4.result.v1","status":"continuous_iteration_completed",
            "selected_shot_ids":["s01"],"required_shot_ids":read(args.g2/"result.json")["required_shot_ids"],
            "scope":"same G1-G3 S01 24.00-26.00 second window, 101 source exposures",
            "rows":len(p["arrays"]["pts"]),"native_pts":p["problem"]["native_pts"],
            "participating_entities":p["problem"]["participating_entities"],"unsupported":p["problem"]["unsupported"],
            "geometry_updated":False,"source_uv_pts_updated":False,"activity":p["activity"],
            "solve":result,"raw_pixel_residuals":stats,"camera_delta_from_G3":delta[0].tolist(),
            "camera_delta_from_G1":(after["controls"][0]-p["controls"][0]).tolist(),
            "station_delta_from_G3":quantile(np.abs(after["stations"]-before["stations"])),
            "improved_frames_vs_G3":sum(x["after_G4"]["median"]<x["before_G3"]["median"] for x in frames),
            "improved_frames_vs_G1":sum(x["after_G4"]["median"]<x["original_G1"]["median"] for x in frames),
            "numerical_checks":checks,"user_approval":"not_reviewed","full_resume_checkpoint":False,
            "memory":{"owned_device_bytes":problem.creation["owned_device_bytes"]+lin["device_bytes"]+solver["owned_device_bytes"],
                      "runtime_ref":"runtime_metadata.json","cudss_estimates":{k:v for k,v in history[-1]["summary"].items() if "estimate" in k} if history else {},
                      "GPU_peak_measured":False},
            "limits":["Only the original six R3/R4 contours constrain this run. Other complete model entities are displayed for inspection, not certified.",
                      "Continuous native convergence is not source accuracy or human approval.",
                      "The original temporal motion is retained with one shared constant pan/tilt/logf correction. No arbitrary per-frame/per-circle freedom.",
                      "G5 full restart checkpoints are not implemented."],
            "main_log":str(out/"main.log"),"state_path":str(args.state_out.resolve())}
        save_state(args.state_out,p,after,result)
    for item in protected:
        if sha(item["path"])!=item["sha256"]:raise AssertionError("protected file changed")
    for path,h in identity.items():
        if sha(path)!=h:raise AssertionError("input changed")
    full["protected_files_unchanged"]=len(protected);full["input_files_unchanged"]=len(identity)
    full["solve_and_checks_seconds"]=time.perf_counter()-started
    write(out/"result.json",full)
    note(out,"phase=native_complete; "+json.dumps(result))
    note(out,"phase=raw_summary; "+json.dumps(stats["all"]))
    return full

def review(args):
    from rinkspace.review.window import render_gpu_window
    out=args.out.resolve();p=load_linearization_problem(args.g1,args.g2)
    for path,h in read(out/"input_identity.json").items():
        if sha(path)!=h:raise ValueError("input changed before review")
    before={k:np.load(args.start_state/(k+".npy"),allow_pickle=False) for k in ("controls","stations")}
    after=dict(np.load(out/"accepted.npz",allow_pickle=False))
    full=read(out/"result.json")
    full["review"]=render_gpu_window(NativeCore(args.library),p,before,after,args.scene,out,
                                    lambda x:note(out,x))
    write(out/"result.json",full)
    guard_project_write(args.state_out.parent/"g4.json")
    write(args.state_out.parent/"g4.json",full)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    for key in ("g1","g2","g3","start-state","state-out","out","library","scene"):parser.add_argument("--"+key,type=Path,required=True)
    parser.add_argument("--review-existing",action="store_true")
    parser.add_argument("--solve-only",action="store_true")
    args=parser.parse_args()
    try:
        if not args.review_existing:run(args)
        if not args.solve_only:review(args)
    except BaseException as exc:
        args.out.mkdir(parents=True,exist_ok=True)
        write(args.out/"failure.json",{"error":str(exc),"traceback":traceback.format_exc(),"phase":"review" if args.review_existing else "run"})
        raise
