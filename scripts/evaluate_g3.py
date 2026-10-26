"""One real candidate through the resident native LM engine (G3).
NumPy solves below are independent diagnostics AFTER the native decision.
"""
from __future__ import annotations
import argparse
import ctypes as c
import json
from pathlib import Path
import runpy
import time
import traceback
import numpy as np
from rinkspace.core.native import NativeCore
from rinkspace.core.gpu import ResidentResidualProblem
from rinkspace.session.gpu_problem import load_linearization_problem
from rinkspace.review.render import source_frames

g2=runpy.run_path(str(Path(__file__).with_name("evaluate_g2.py")))
read,write,sha,memory,quantile,inputs,PLAN_KEYS=(g2[n] for n in ("read","write","sha","memory","quantile","inputs","PLAN_KEYS"))

def finite_record(value):
    if isinstance(value,float) and not np.isfinite(value):return None
    if isinstance(value,dict):return {k:finite_record(v) for k,v in value.items()}
    if isinstance(value,list):return [finite_record(v) for v in value]
    return value

def available_host():
    class Status(c.Structure):
        _fields_=[("length",c.c_ulong),("load",c.c_ulong)]+[(s,c.c_ulonglong) for s in
            ("total_phys","available_phys","total_page","available_page","total_virtual","available_virtual","extended_virtual")]
    m=Status();m.length=c.sizeof(m)
    if not c.windll.kernel32.GlobalMemoryStatusEx(c.byref(m)):raise c.WinError()
    return {"available_physical_bytes":m.available_phys,"available_commit_bytes":m.available_page}

def objective(p,r):
    cost=0.;delta=p["activity"]["loss"]["delta"]
    for i,rank in enumerate(p["ranks"]):
        rows=p["members"][p["group_offsets"][i]:p["group_offsets"][i+1]]
        W=p["weights"][p["weight_offsets"][i]:p["weight_offsets"][i+1]].reshape(rank,2*len(rows))
        wr=W@r[rows].ravel();s=wr@wr
        cost+=.5*(s if s<=rank*delta**2 else rank*(2*delta*np.sqrt(s/rank)-delta**2))
    return float(cost)

def compare(actual,expected,rtol=2e-8,atol=2e-9):
    np.testing.assert_allclose(actual,expected,rtol=rtol,atol=atol)
    return {"max_abs":float(np.max(np.abs(actual-expected))),"relative_l2":float(np.linalg.norm(actual-expected)/max(1.,np.linalg.norm(expected)))}

def system_check(p,layout,blocks,system,step):
    q=p["directions"].shape[2];nl=len(p["local_scales"]);lam=step["summary"]["lambda_used"]
    S=blocks["shared_H"].copy()+lam*np.eye(q);rhs=-blocks["shared_g"].copy();parts=[]
    for ci,comp in enumerate(layout["components"]):
        lb,n,_,_,d,h0,g0=map(int,comp);ids=layout["locals"][lb:lb+n];free=np.flatnonzero(system["blocked"][ids]==0)
        H=blocks["H"][h0:h0+d*d].reshape(d,d);g=blocks["g"][g0:g0+d]
        cols=g2["component_columns"](layout,ci,q);nq=len(cols)
        B=H[:nq,nq+free];C=H[np.ix_(nq+free,nq+free)]+lam*np.eye(len(free))
        solved=np.linalg.solve(C,np.c_[B.T,g[nq+free]]) if len(free) else np.empty((0,nq+1))
        S[np.ix_(cols,cols)]-=B@solved[:,:nq];rhs[cols]+=B@solved[:,nq]
        parts.append((ids,free,cols,solved,H,g))
    dq=np.linalg.solve(S,rhs);full=np.r_[dq,np.zeros(nl)]
    for ids,free,cols,solved,H,g in parts:full[q+ids[free]]=-(solved[:,len(cols)]+solved[:,:len(cols)]@dq[cols])
    pred=0.
    for ids,free,cols,solved,H,g in parts:
        d=np.r_[system["applied"][cols],system["applied"][q+ids]]
        pred-=g@d+.5*d@H@d
    return {"reduced_S":compare(system["S"],S),"reduced_rhs":compare(system["rhs"],rhs),
        "full_step":compare(system["step"],full),"prediction":compare(np.array([step["summary"]["predicted_decrease"]]),np.array([pred])),
        "reduced_eigenvalues":np.linalg.eigvalsh(S).tolist(),
        "method":"independent CPU block algebra and solve only after GPU/native decision; no production fallback"}

def save_state(root,p,state,step):
    names=("controls.npy","stations.npy","camera.json","state.json")
    for name in names:
        path=(root/name).resolve()
        if len(str(path).encode("utf-16-le"))//2>=260 or len(path.name.encode("utf-16-le"))//2>255:raise ValueError("OneDrive path limit")
    root.mkdir(parents=True,exist_ok=True)
    np.save(root/"controls.npy",state["controls"],allow_pickle=False)
    np.save(root/"stations.npy",state["stations"],allow_pickle=False)
    camera={**p["camera"],"controls_ref":"controls.npy","g3_origin":{"optimizer_accepted":step["summary"]["accepted"],
        "same_original_control_grid":True,"scene_updated":False}}
    write(root/"camera.json",camera)
    write(root/"state.json",{"schema":"rinkspace.g3.saved_state.v1","optimizer_accepted":step["summary"]["accepted"],
        "user_approval":"not_reviewed","camera_ref":"camera.json","controls_ref":{"path":"controls.npy","sha256":sha(root/"controls.npy")},
        "row_station_ref":{"path":"stations.npy","sha256":sha(root/"stations.npy")},
        "station_row_order":"same G1 inputs.npz rows; G2 local_ids retain original shared identities",
        "activity_ref":"../g3.json","full_resume_checkpoint":False})

def render(core,p,before,after,out,step,note):
    import cv2
    from PIL import Image,ImageDraw
    from rinkspace.review.fonts import review_font
    pts=p["problem"]["native_pts"];source=p["problem"]["source"];camera=p["camera"];paths=p["problem"]["participating_entities"]
    ns=257;per=len(paths)*ns
    codes=np.tile(np.repeat([x["native_code"] for x in paths],ns),len(pts)).astype(np.int32)
    stations=np.tile(np.linspace(0,1,ns),len(paths)*len(pts))
    times=np.repeat((np.array(pts)-int(camera["reference_pts"]))*camera["time_base_num"]/camera["time_base_den"],per)
    def project(state):
        with ResidentResidualProblem(core,**inputs(p,controls=state["controls"],codes=codes,stations=stations,times=times,observed=np.zeros((len(codes),2)))) as display:
            return display.evaluate("cuda")
    old,new=project(before),project(after);font=review_font(25);small=review_font(19)
    dst=out/"frames";dst.mkdir(exist_ok=True);records=[];selected=set([pts[0],pts[len(pts)//2],pts[-1]])
    def strokes(image,data,offset,color):
        for j in range(len(paths)):
            start=offset+j*ns
            for k in range(ns-1):
                row=start+k
                if data["status"][row] or data["status"][row+1]:continue
                xy=data["values"][row:row+2,3:5]
                if np.max(np.abs(xy))>1e7:continue
                cv2.line(image,tuple(np.rint(xy[0]).astype(int)),tuple(np.rint(xy[1]).astype(int)),color,2,cv2.LINE_AA)
    accepted=step["summary"]["accepted"];title="本次接受候选" if accepted else "本次拒绝候选"
    for i,(point,frame) in enumerate(source_frames(Path(source["asset_path"]),source,pts)):
        if point!=pts[i]:raise AssertionError("source PTS order changed")
        overlay=frame.copy();strokes(overlay,old,i*per,(255,210,20));strokes(overlay,new,i*per,(30,160,255))
        rows=np.flatnonzero(p["arrays"]["pts"]==point)
        for uv in p["arrays"]["uv"][rows]:cv2.circle(overlay,tuple(np.rint(uv).astype(int)),2,(230,230,230),-1,cv2.LINE_AA)
        rgb=Image.fromarray(cv2.cvtColor(overlay,cv2.COLOR_BGR2RGB));panel=Image.new("RGB",(1920,1160),(18,23,31));panel.paste(rgb,(0,80));d=ImageDraw.Draw(panel)
        d.text((16,5),f"G3 一次 LM 候选 | S01 {point/12800:.2f} s / PTS {point}",font=font,fill="white")
        d.text((16,44),f"青色：更新前；橙色：{title}；白点：本帧原 UV。仅 R3/R4 参与此步，当前几何冻结，未人工验收。",font=small,fill=(220,225,230))
        path=f"frames/pts_{point}.jpg";panel.save(out/path,quality=94)
        if point in selected:
            left=frame.copy();right=frame.copy();strokes(left,old,i*per,(255,210,20));strokes(right,new,i*per,(30,160,255))
            pair=Image.new("RGB",(1920,620),(18,23,31));pair.paste(Image.fromarray(cv2.cvtColor(left,cv2.COLOR_BGR2RGB)).resize((960,540)),(0,80))
            pair.paste(Image.fromarray(cv2.cvtColor(right,cv2.COLOR_BGR2RGB)).resize((960,540)),(960,80))
            draw=ImageDraw.Draw(pair);draw.text((15,10),f"更新前 | {point/12800:.2f} s",font=font,fill="white")
            draw.text((975,10),title+" | 同一原片曝光",font=font,fill="white");draw.text((15,46),"同一 native 几何；只显示本次相机变化，不表示完整画面验收。",font=small,fill="white")
            pair.save(out/f"comparison_{point}.jpg",quality=94)
        records.append({"pts":point,"source_seconds":point/12800,"image":path,"visible_observation_rows":len(rows)})
    write(out/"frames.json",records)
    m=step["summary"]
    html="""<!doctype html><meta charset="utf-8"><title>G3：一次真实 LM 候选</title><style>
body{background:#141b24;color:#edf3fa;font:16px system-ui;max-width:1600px;margin:24px auto;padding:0 20px}
p{line-height:1.65;color:#d2dce9}img{width:100%;border:1px solid #38495d}a{color:#89caff}.box{background:#253143;padding:15px;border-radius:9px}input{width:75%}button{padding:8px}</style>
<h1>G3：一次真实 LM 候选</h1><div class="box">__SUMMARY__</div>
<p>S01 原片 24–26 秒的全部 101 个源曝光；同一 G1/G2 输入，5,920 条原观测与完整相关组。当前场地、广告、畸变、安装、读出冻结，活动方向仍为 pan / tilt / log(f) 三个常量修正及原 station。没有继续迭代。</p>
<p>青色为更新前，橙色为本次候选，白点为本帧原 UV。画面中的线条只显示本次参与的 R3/R4 六条轮廓；目标下降不等于完整镜头或其它实体通过验收。</p>
<button onclick="go(i-1)">上一帧</button><input id="slider" type="range" min="0" max="100" value="50" oninput="go(+this.value)"><button onclick="go(i+1)">下一帧</button><p id="where"></p>
<a id="full"><img id="frame"></a><p><a href="result.json">完整结果与耗时</a> · <a href="main.log">主日志</a> · <a href="comparison_320000.jpg">25 秒左右对照</a></p>
<script>const frames=__FRAMES__;let i=50;function go(n){i=Math.max(0,Math.min(frames.length-1,n));const f=frames[i];document.getElementById('slider').value=i;document.getElementById('where').textContent='原片 '+f.source_seconds.toFixed(2)+' 秒 · PTS '+f.pts;document.getElementById('frame').src=f.image;document.getElementById('full').href=f.image;}go(i);</script>"""
    summary=f'{title}：{m["reason"]}。目标 {m["old_cost"]:.6f} → {m["trial_cost"]:.6f}；增益比 {m["gain_ratio"]:.6f}。<br>GPU 原生闭环 {step["timings"]["wall_ms"]:.3f} ms（本次进程首次调用，含单次符号分析及库加载成本）；未人工验收。'
    metrics=step["timings"]
    names=[("线性化（native 墙钟）","linearization_wall_ms"),("边界初始化","bounds_setup_event_ms"),
        ("阻尼＋局部消元","damp_local_eliminate_event_ms"),("cuDSS 符号分析","cudss_analysis_event_ms"),
        ("cuDSS 数值分解","cudss_factor_event_ms"),("cuDSS 求解","cudss_solve_event_ms"),
        ("回代与边界检查","backsub_bounds_event_ms"),("可行更新与预测下降","retract_prediction_event_ms"),
        ("候选原目标评估","trial_objective_event_ms"),("摘要回传","summary_d2h_event_ms"),
        ("接受状态提交","accept_commit_event_ms")]
    timing_html="<details><summary>实际分项耗时</summary><table>"+ "".join(f"<tr><td>{label}</td><td>{metrics[key]:.3f} ms</td></tr>" for label,key in names)+"</table><p>除线性化墙钟外为 CUDA 事件跨度，包含库调用提交与同步；不能当作纯 kernel 时间。总 native 闭环 "+f'{metrics["wall_ms"]:.3f}'+" ms。首次准备和进程峰值记录未保存；精确自有 GPU 缓冲约 9.93 MiB，cuDSS 另报告约 25 KiB 分解内存估计。</p></details>"
    if (out/"frame_residuals.json").is_file():
        per=read(out/"frame_residuals.json");bad=[x for x in per if x["after"]["median"]>x["before"]["median"]]
        if bad:
            idx=pts.index(bad[0]["pts"])
            timing_html+=f'<p><b>逐帧限制：{len(per)-len(bad)} 帧中位残差下降，{len(bad)} 帧上升。</b>本次退步段从 {bad[0]["pts"]/12800:.2f} 秒开始；总体下降不代表每帧改善。<button onclick="go({idx})">查看退步段起点</button><button onclick="go({len(pts)-1})">查看末帧</button></p>'
    html=html.replace("<p>S01 原片",timing_html+"<p>S01 原片")
    (out/"index.html").write_text(html.replace("__SUMMARY__",summary).replace("__FRAMES__",json.dumps(records,ensure_ascii=False)),encoding="utf8")
    note(f"phase=review_saved; frames={len(records)}; original_pts_preserved")
    return {"source_frames":len(records),"review":"index.html","original_frame_pts":pts}

def result_record(args,p,step,checks,raw_before,raw_after,meta):
    out=args.out.resolve();options=meta["options"];lin_info=meta["linearization_layout"];solver_info=meta["solver_creation"];native_creation=meta["native_creation"]
    mem0=meta.get("process_before_resident");mem1=meta.get("process_after_candidate");host=meta.get("available_host_at_start")
    result={"schema":"rinkspace.g3.result.v1","run_id":out.name,"status":"completed_candidate","reason_code":step["summary"]["reason"],
        "selected_shot_ids":["s01"],"required_shot_ids":read(args.g2/"result.json")["required_shot_ids"],
        "scope_status":"same_G1_G2_window_factor_subset","native_pts":p["problem"]["native_pts"],
        "rows":len(p["arrays"]["rows"]),"native_frame_count":len(p["problem"]["native_pts"]),
        "participating_entities":p["problem"]["participating_entities"],"unsupported":p["problem"]["unsupported"],
        "fit_executed":True,"cuda_executed":True,"cudss_executed":True,"candidate_count":1,
        "parameters_updated":step["summary"]["accepted"],"geometry_updated":False,"source_uv_pts_updated":False,
        "optimizer_accepted":step["summary"]["accepted"],"user_approval":"not_reviewed",
        "full_resume_checkpoint":False,"last_checkpoint":None,"objective_terms":{"data":"same rank-normalized Huber delta=3 with complete original W",
            "priors":"none; unchanged from G2","cost_before":step["summary"]["old_cost"],"cost_trial":step["summary"]["trial_cost"]},
        "activity":{k:v for k,v in p["activity"].items() if k!="original_local_indices"},"solver_options":options,
        "native_creation":native_creation,"linearization_layout":lin_info,"solver_creation":solver_info,"decision":step["summary"],
        "timings":step["timings"],"memory":{"process_before_resident":mem0,"process_after_candidate":mem1,
            "available_host_at_start":host,"owned_G1_G2_G3_device_bytes":native_creation["owned_device_bytes"]+lin_info["device_bytes"]+solver_info["owned_device_bytes"],
            "cudss_memory":"separate symbolic estimates in decision; not measured process GPU peak"},
        "raw_pixel_residual_before":quantile(np.linalg.norm(raw_before["values"][:,9:11],axis=1)),
        "raw_pixel_residual_after":quantile(np.linalg.norm(raw_after["values"][:,9:11],axis=1)),
        "validation":checks,"state_path":str(args.state_out.resolve()),"main_log":str(out/"main.log"),
        "artifacts":{"root":str(out),"review":"index.html","candidate":"candidate.npz","accepted":"accepted.npz","system":"step_system.npz","raw":"raw_before_after.npz"},
        "limitations":["One candidate only, not convergence.","Only the same six native R3/R4 contours constrain this candidate; other entities and whole S01 accuracy are not certified.",
            "Full solve/resume lifecycle remains absent.","Numerical acceptance does not change human approvals.","Timing is this actual first call; no repeated real updates or speedup claim."]}
    result["activity"]["parameter_update_executed"]=bool(step["summary"]["accepted"])
    system=np.load(out/"step_system.npz",allow_pickle=False)
    q=p["directions"].shape[2]
    result["activity"]["candidate_delta_scaled"]=system["applied"][:q].tolist()
    result["activity"]["candidate_delta_physical"]=(system["applied"][:q]*np.array([x["physical_scale"] for x in p["activity"]["parameters"]])).tolist()
    result["memory"]["explicit_application_step_d2h_bytes"]=step["summary"]["explicit_application_d2h_bytes"]+(32 if step["timings"]["linearization_wall_ms"] else 0)
    result["memory"]["d2h_note"]="solver scalar/status copies plus G2 summary; vendor-library internal transfers are not measured"
    frame_stats=[]
    for pts in p["problem"]["native_pts"]:
        mask=p["arrays"]["pts"]==pts
        frame_stats.append({"pts":pts,"rows":int(mask.sum()),"before":quantile(np.linalg.norm(raw_before["values"][mask,9:11],axis=1)),
            "after":quantile(np.linalg.norm(raw_after["values"][mask,9:11],axis=1))})
    write(out/"frame_residuals.json",frame_stats)
    result["frame_residual_summary"]={"frames":len(frame_stats),"median_error_decreased":sum(x["after"]["median"]<x["before"]["median"] for x in frame_stats),
        "per_frame_ref":"frame_residuals.json","meaning":"raw point residual diagnostic, not independent accuracy acceptance"}
    return result

def main(args):
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    def note(message):
        print(message,flush=True)
        with (out/"main.log").open("a",encoding="utf8") as f:f.write(message+"\n")
    note(f"G3 main log: {out/'main.log'}")
    if (out/"decision.json").exists():raise ValueError("A candidate already exists; use --review-existing to finish display without another LM call")
    start=time.perf_counter();p=load_linearization_problem(args.g1,args.g2)
    protected={x["path"]:sha(x["path"]) for x in read(args.g2/"protected_before.json")}
    for x in read(args.g2/"protected_before.json"):
        if protected[x["path"]]!=x["sha256"]:raise ValueError("current native geometry/camera differs from G2")
    write(out/"protected_before.json",[{"path":k,"sha256":v} for k,v in protected.items()])
    identity={str(root/name):sha(root/name) for root,names in ((args.g1,("inputs.npz","scene.json","camera.json","controls.npy","problem.json")),(args.g2,("plan.npz","activity.json","noise_groups.json"))) for name in names}
    write(out/"input_identity.json",identity);write(out/"activity.json",p["activity"])
    core=NativeCore(args.library);reference=NativeCore(args.reference_library);host=available_host()
    mem0=memory();note("phase=resident; original G2 activity, full W and finite supports; one candidate only")
    with ResidentResidualProblem(core,**inputs(p)) as problem:
        lin_info=problem.configure_linearization(**{k:p[k] for k in PLAN_KEYS},huber_delta=p["activity"]["loss"]["delta"])
        options={"device_budget_bytes":problem.creation["free_after_alloc_bytes"]-lin_info["device_bytes"],
            "host_budget_bytes":min(host.values()),"initial_lambda_relative":.001,"initial_lambda":0.,
            "acceptance_eta":.0001,"linear_tolerance":1e-8}
        solver_info=problem.configure_solver(lower=p["lower"],upper=p["upper"],**options)
        meta={"options":options,"linearization_layout":lin_info,"solver_creation":solver_info,"native_creation":problem.creation,"process_before_resident":mem0,"available_host_at_start":host}
        write(out/"runtime_metadata.json",meta)
        before={"controls":p["controls"].copy(),"stations":p["arrays"]["stations"].copy()}
        raw_before=problem.evaluate("cuda");note("phase=one_candidate_started")
        # Exactly one native call/decision on this real source problem.
        step=problem.lm_step()
        note("phase=one_candidate_decided; "+json.dumps(finite_record(step),ensure_ascii=False,allow_nan=False))
        write(out/"decision.json",finite_record(step))
        accepted=problem.export_state();trial=problem.export_state(trial=True);system=problem.export_step_system()
        raw_after=problem.evaluate("cuda");mem1=memory()
        meta["process_after_candidate"]=mem1;write(out/"runtime_metadata.json",meta)
        np.savez(out/"candidate.npz",**trial);np.savez(out/"accepted.npz",**accepted)
        np.savez(out/"step_system.npz",**system)
        np.savez(out/"raw_before_after.npz",before=raw_before["values"],after=raw_after["values"],before_status=raw_before["status"],after_status=raw_after["status"])
        # Replay the unchanged G2 linearization in a separate handle only for diagnostics.
        with ResidentResidualProblem(core,**inputs(p)) as check:
            check.configure_linearization(**{k:p[k] for k in PLAN_KEYS},huber_delta=p["activity"]["loss"]["delta"])
            check.linearize("cuda");blocks=check.export_blocks()
            checks=system_check(p,check.layout,blocks,system,step)
        with ResidentResidualProblem(reference,**inputs(p,controls=trial["controls"],stations=trial["stations"])) as check:
            cpu_trial=check.evaluate("cpu")
        np.testing.assert_array_equal(raw_before["status"],0)
        if step["summary"]["trial_domain_failures"]==0:
            cpu_cost=objective(p,cpu_trial["values"][:,9:11])
            checks["trial_objective"]=compare(np.array([step["summary"]["trial_cost"]]),np.array([cpu_cost]))
            if step["summary"]["accepted"]:
                checks["accepted_projection_cpu_gpu"]=compare(raw_after["values"],cpu_trial["values"],atol=1e-8)
        if step["summary"]["accepted"]:
            for key in accepted:np.testing.assert_array_equal(accepted[key],trial[key])
        else:
            for key in accepted:np.testing.assert_array_equal(accepted[key],before[key])
        checks["accepted_state_matches_decision"]=True
        checks["original_station_supports_preserved"]=bool(np.all(trial["stations"]>=p["arrays"]["support"][:,0]) and np.all(trial["stations"]<=p["arrays"]["support"][:,1]))
        if not checks["original_station_supports_preserved"]:raise AssertionError("trial outside original finite support")
        write(out/"numerical_checks.json",checks)
        save_state(args.state_out,p,accepted,step)
        result=result_record(args,p,step,checks,raw_before,raw_after,meta)
        write(out/"result.json",finite_record(result));note("phase=numerical_checks_passed; rendering same-source comparison")
        render_start=time.perf_counter();result["artifacts"].update(render(core,p,before,trial,out,step,note))
        result["timings"]["render_seconds"]=time.perf_counter()-render_start
    for path,h in protected.items():
        if sha(path)!=h:raise AssertionError("protected model/camera changed")
    for path,h in identity.items():
        if sha(path)!=h:raise AssertionError("original G1/G2 input changed")
    result["protected_files_unchanged"]=len(protected);result["input_files_unchanged"]=len(identity)
    result["timings"]["end_to_end_seconds"]=time.perf_counter()-start
    result["memory"]["process_after_review"]=memory()
    write(out/"result.json",finite_record(result));note("phase=complete; one candidate only; original model and human approvals preserved")


def review_existing(args):
    """Finish artifacts from a saved decision; this path NEVER calls lm_step."""
    out=args.out.resolve();p=load_linearization_problem(args.g1,args.g2)
    def note(s):
        print(s,flush=True)
        with (out/"main.log").open("a",encoding="utf8") as f:f.write(s+"\n")
    note(f"G3 existing-decision review log: {out/'main.log'}")
    for path,h in read(out/"input_identity.json").items():
        if sha(path)!=h:raise ValueError("saved decision input changed")
    for x in read(out/"protected_before.json"):
        if sha(x["path"])!=x["sha256"]:raise ValueError("protected model changed")
    started=time.perf_counter();step=read(out/"decision.json");checks=read(out/"numerical_checks.json")
    trial=dict(np.load(out/"candidate.npz",allow_pickle=False));accepted=dict(np.load(out/"accepted.npz",allow_pickle=False))
    raw=np.load(out/"raw_before_after.npz",allow_pickle=False)
    if not (args.state_out/"state.json").is_file():raise ValueError("saved state is missing; not an optimizer resume")
    if (out/"runtime_metadata.json").is_file():meta=read(out/"runtime_metadata.json")
    else:
        # Preserve the first successful candidate rather than repeat it to recover telemetry.
        g2_result=read(args.g2/"result.json")
        meta={"options":{"initial_lambda_relative":.001,"initial_lambda":0.,"acceptance_eta":step["summary"]["acceptance_eta"],
            "linear_tolerance":1e-8,"device_budget_bytes":None,"host_budget_bytes":None,
            "budget_note":"available capacity was used and checked by native; exact setup values were not saved before the report failure"},
            "linearization_layout":g2_result["layout"],
            "solver_creation":{"owned_device_bytes":step["summary"]["solver_device_bytes"],"setup_wall_ms":None,
                "padded_local_dimension":46,"reduced_CSR_nnz":6,"cudss_major":0,"cudss_minor":8,"cudss_patch":0},
            "native_creation":{"owned_device_bytes":g2_result["memory"]["resident_G1_buffers_bytes"]},
            "process_before_resident":None,"process_after_candidate":None,"available_host_at_start":None}
        meta["linearization_layout"]["device_bytes"]=g2_result["memory"]["G2_buffers_bytes"]
    result=result_record(args,p,step,checks,{"values":raw["before"]},{"values":raw["after"]},meta)
    result["postprocessing_recovery"]={"reason":"required_shot_ids is stored in G2 result, not G1 problem; result assembly failed after accepted state and checks were saved",
        "new_candidates_executed":0,"setup_and_process_memory_telemetry_missing":not (out/"runtime_metadata.json").is_file(),
        "preserved_failure_record":"failure.json"}
    result["artifacts"].update(render(NativeCore(args.library),p,{"controls":p["controls"],"stations":p["arrays"]["stations"]},trial,out,step,note))
    for x in read(out/"protected_before.json"):
        if sha(x["path"])!=x["sha256"]:raise ValueError("protected model changed")
    result["protected_files_unchanged"]=len(read(out/"protected_before.json"));result["input_files_unchanged"]=len(read(out/"input_identity.json"))
    result["timings"]["postprocessing_recovery_seconds"]=time.perf_counter()-started
    result["timings"]["end_to_end_seconds"]=None
    result["memory"]["process_after_review"]=memory()
    write(out/"result.json",finite_record(result))
    failure=read(out/"failure.json");failure.update({"phase":"post-decision report assembly","resolved":True,"resolution":"saved candidate reused; no new LM call"})
    write(out/"failure.json",failure)
    note("phase=complete; recovered report/display only; total real candidate_count=1")

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("g1","g2","library","reference-library","out","state-out"):parser.add_argument("--"+key,type=Path,required=True)
    parser.add_argument("--review-existing",action="store_true",help="Complete display from an already saved decision; never run another candidate")
    args=parser.parse_args()
    try:
        if args.review_existing:review_existing(args)
        else:main(args)
    except Exception:
        args.out.mkdir(parents=True,exist_ok=True)
        with (args.out/"main.log").open("a",encoding="utf8") as f:f.write(traceback.format_exc())
        write(args.out/"failure.json",{"status":"failed","error":traceback.format_exc()})
        raise
