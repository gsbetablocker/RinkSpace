"""Evaluate frozen real source UV residuals on the current native CPU/CUDA core.

G1 only: no fitting, whitening, re-identification or parameter updates.
Existing imported stations are reused byte-for-byte at their source-row identity.
"""
from __future__ import annotations
import argparse
import ctypes as c
import hashlib
import json
from pathlib import Path
import shutil
import time
import traceback
import numpy as np
from rinkspace.core.native import NativeCore
from rinkspace.core.gpu import ResidentResidualProblem
from rinkspace.compat.d78_entities import native_definition
from rinkspace.compat.d78_state import project_native_pts
from rinkspace.review.render import source_frames

def read(p): return json.loads(Path(p).read_text(encoding="utf-8-sig"))
def write(p,v): Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""): h.update(b)
    return h.hexdigest()

def memory():
    class Counters(c.Structure):
        _fields_=[("cb",c.c_ulong),("PageFaultCount",c.c_ulong)]+[(n,c.c_size_t) for n in
            ("PeakWorkingSetSize","WorkingSetSize","QuotaPeakPagedPoolUsage","QuotaPagedPoolUsage",
             "QuotaPeakNonPagedPoolUsage","QuotaNonPagedPoolUsage","PagefileUsage","PeakPagefileUsage","PrivateUsage")]
    get=c.windll.kernel32.GetCurrentProcess;get.restype=c.c_void_p
    fn=c.windll.psapi.GetProcessMemoryInfo;fn.argtypes=[c.c_void_p,c.POINTER(Counters),c.c_ulong]
    p=Counters();p.cb=c.sizeof(p)
    if not fn(get(),c.byref(p),p.cb):raise c.WinError()
    return {"working_set_bytes":p.WorkingSetSize,"peak_working_set_bytes":p.PeakWorkingSetSize,"private_bytes":p.PrivateUsage}

def quantile(x):
    x=np.asarray(x)
    return dict(zip(("min","median","p95","max"),np.quantile(x,[0,.5,.95,1]).tolist())) if x.size else None

def main(args):
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    def note(message):
        print(message,flush=True)
        with (out/"main.log").open("a",encoding="utf-8") as f:f.write(message+"\n")
    t0=time.perf_counter();note("phase=prepare; frozen original UV, integer PTS and stations; fit_executed=false")
    initial_memory=memory()
    scene=read(args.scene);camera=read(args.camera);window=read(args.window)
    prepared=read(args.prepared/"manifest.json")
    controls_path=args.camera.parent/camera["controls_ref"];controls=np.load(controls_path,allow_pickle=False)
    input_refs=[args.scene,args.camera,args.window,controls_path,args.local_state/"local_layout.json"]
    expected_camera={"schema":"rinkspace.camera.motion.v1","mount_family":"orthogonal_pan_tilt_pivot",
        "lens_family":"brown_zoom_linear","pupil_model":"zoom_quadratic_camera_z","readout_mode":"predicted_row_implicit_unique"}
    if any(camera.get(k)!=v for k,v in expected_camera.items()):raise ValueError("unsupported native camera family")
    arrays={}
    for k,ref in prepared["arrays"].items():
        p=args.prepared/ref["path"]
        if sha(p)!=ref["sha256"]:raise ValueError("prepared array changed: "+k)
        arrays[k]=np.load(p,allow_pickle=False);input_refs.append(p)
    pts=np.asarray(window["native_pts"],dtype=np.int64)
    source=prepared["sources"][0]
    if sha(source["asset_path"])!=source["sha256"]:raise ValueError("source video identity changed")
    registry=np.load(args.prepared/source["pts_ref"]["path"])
    if not np.all(np.isin(pts,registry)) or len(np.unique(pts))!=len(pts):raise ValueError("window is not original PTS")
    if source["id"]!=camera["source_id"] or window["shot_id"]!=camera["shot_id"]:raise ValueError("source/shot mismatch")
    in_window=np.isin(arrays["pts"],pts)
    current={p["id"]:p for p in scene["paths"]}
    supported={}
    excluded=[]
    for i,f in enumerate(prepared["features"]):
        nr=int(np.count_nonzero(in_window&(arrays["feature_index"]==i)))
        if not nr:continue
        path=current.get(f["id"])
        ok=(path is not None and 100<=path.get("generated_code",-1)<132
            and path.get("measurement")=="contour" and path["label"]==f["label"]
            and f["measurement"] in ("inner_boundary","outer_boundary","contour"))
        if ok:supported[i]=path
        else:excluded.append({"feature_id":f["id"],"label":f["label"],"rows":nr,
            "reason":"G1 supports shared curling contours only; other current-native families or changed measurement semantics are unsupported"})
    selected=in_window&np.isin(arrays["feature_index"],list(supported))&(arrays["active"]!=0)
    rows=np.flatnonzero(selected);n=len(rows)
    if not n:raise ValueError("no eligible actual contour observations")
    # Importer stores one original station per raw source row, not per compact local_index.
    local=read(args.local_state/"local_layout.json")["station.f64"]
    stations_all=np.load(args.local_state/local["path"],allow_pickle=False)
    if hashlib.sha256(stations_all.astype("<f8").tobytes()).hexdigest()!=local["source_ref"]["sha256"]:
        raise ValueError("frozen station source identity changed")
    input_refs.append(args.local_state/local["path"])
    source_rows=arrays["legacy_source_rows"].astype(np.int64)
    if not np.array_equal(source_rows,np.arange(len(source_rows))):raise ValueError("requires explicit original row-order binding")
    if len(stations_all)!=len(source_rows):raise ValueError("frozen station row count mismatch")
    stations=stations_all[source_rows[rows]].copy()
    if not np.all((stations>=arrays["support"][rows,0])&(stations<=arrays["support"][rows,1])):
        raise ValueError("frozen station outside original finite support; not clamped")
    if not np.all(arrays["kind"][rows]==1):raise ValueError("G1 contour requires existing curve factors")
    codes=np.array([supported[int(i)]["generated_code"] for i in arrays["feature_index"][rows]],np.int32)
    times=np.array([(int(p)-int(camera["reference_pts"]))*camera["time_base_num"]/camera["time_base_den"]
                    for p in arrays["pts"][rows]],np.float64)
    observed=arrays["uv"][rows].copy()
    immutable={str(p.resolve()):sha(p) for p in input_refs}
    write(out/"input_identity.json",immutable)
    # Preserve original statistical groups in full, never crop or diagonalize W.
    groups=[]
    selected_set=set(rows.tolist())
    for gi in np.unique(arrays["noise_group_index"][rows]):
        if gi<0:continue
        g=prepared["noise_groups"][int(gi)]
        members=arrays["noise_group_members"][g["member_start"]:g["member_start"]+g["member_count"]]
        groups.append({"original_index":int(gi),"group":g,"original_member_rows":members.tolist(),
                       "selected_member_rows":[int(x) for x in members if int(x) in selected_set],
                       "evaluated_as_whitened_group":False})
    write(out/"noise_groups.json",{"groups":groups,"independent_rows":rows[arrays["noise_group_index"][rows]<0].tolist(),
          "weights_source":prepared["arrays"]["noise_group_weights"],"whitening_executed":False,
          "note":"Raw UV evaluation only. Complete original groups remain referenced; no independent-point likelihood is claimed."})
    np.savez(out/"inputs.npz",rows=rows,source_rows=source_rows[rows],codes=codes,stations=stations,times=times,
             **{k:arrays[k][rows] for k in arrays if len(arrays[k])==len(selected)})
    with (args.prepared/"observations.jsonl").open(encoding="utf-8") as src,(out/"observations.jsonl").open("w",encoding="utf-8") as dst:
        kept=0
        for i,line in enumerate(src):
            if i in selected_set:dst.write(line);kept+=1
    if kept!=n:raise ValueError("observation provenance row mismatch")
    shutil.copyfile(args.scene,out/"scene.json")
    shutil.copyfile(args.camera,out/"camera.json")
    shutil.copyfile(controls_path,out/camera["controls_ref"])
    binding=[{"feature_id":p["id"],"label":p["label"],"native_code":p["generated_code"],
        "measurement":p["measurement"],"measurement_policy_id":p["measurement_policy_id"],
        "rows":int(np.count_nonzero(arrays["feature_index"][rows]==i))} for i,p in supported.items()]
    write(out/"problem.json",{"schema":"rinkspace.g1.problem.v1","source":source,"prepared_path":str(args.prepared.resolve()),
        "prepared_revision":prepared["revision"],"camera_ref":"camera.json","scene_ref":"scene.json","arrays_ref":"inputs.npz",
        "native_pts":pts.tolist(),"rows":n,"participating_entities":binding,"unsupported":excluded,
        "window_all_rows":int(in_window.sum()),"inactive_supported_rows":int(np.count_nonzero(in_window&~selected&np.isin(arrays["feature_index"],list(supported)))),
        "station_policy":"exact frozen station per legacy_source_rows; no reinitialization or update",
        "local_state_source":local,"residual":"native_project(current_native_entity(frozen_station))-original_source_uv",
        "whitening_executed":False,"fit_executed":False})
    note(f"phase=resident_prepare; frames={len(pts)} real_observations={n} features={len(binding)}")
    core=NativeCore(args.library);old=NativeCore(args.cpu_reference)
    kwargs=dict(board=scene["board"],layout=scene["shared_layout"],options=scene["shared_geometry_options"],
        rig=camera["rig"],controls=controls,first_control_time=camera["first_control_time"],control_dt=camera["control_dt"],
        image_height=camera["image_height"],codes=codes,stations=stations,times=times,observed=observed)
    before_cuda=memory()
    with ResidentResidualProblem(core,**kwargs) as problem:
        resident_memory=memory();note("phase=evaluate_cpu_cuda")
        cpu_cold=problem.evaluate("cpu");gpu_cold=problem.evaluate("cuda")
        cpu_runs=[problem.evaluate("cpu") for _ in range(5)]
        gpu_runs=[problem.evaluate("cuda") for _ in range(7)]
        cpu=cpu_runs[-1];gpu=gpu_runs[-1];creation=problem.creation
        # Compare against the previous production CPU DLL as well, so extracting
        # a shared sampler cannot make CPU and GPU agree on an accidental change.
        reference_xyz=np.full((n,3),np.nan)
        for fi,p in supported.items():
            ix=np.flatnonzero(arrays["feature_index"][rows]==fi)
            kind,params=native_definition(scene,p,old)
            vertices=p.get("vertices",[])
            reference_xyz[ix]=old.sample_path(scene["board"],kind,params,vertices,stations[ix].tolist())
        original=project_native_pts(old,camera,controls,arrays["pts"][rows],reference_xyz)
        np.savez(out/"native_comparison_debug.npz",cpu=cpu["values"],gpu=gpu["values"],cpu_status=cpu["status"],gpu_status=gpu["status"],original_cpu_uv=original["uv"],original_cpu_status=original["status"],reference_xyz=reference_xyz)
        note("status_counts="+repr({name:dict(zip(*np.unique(v,return_counts=True))) for name,v in (("cpu",cpu["status"]),("cuda",gpu["status"]),("original_cpu",original["status"]))}))
        if not np.array_equal(cpu["status"],gpu["status"]) or not np.array_equal(cpu["status"],original["status"]):
            raise AssertionError("CPU/CUDA/original-native status disagreement")
        valid=cpu["status"]==0
        if not np.array_equal(np.unique(arrays["pts"][rows]),pts):raise AssertionError("real factors do not cover every window frame")
        if not np.any(valid):raise AssertionError("all selected projections invalid")
        np.testing.assert_allclose(cpu["values"][valid,:3],reference_xyz[valid],rtol=0,atol=2e-15)
        np.testing.assert_allclose(cpu["values"][valid,3:5],original["uv"][valid],rtol=0,atol=1e-8)
        np.testing.assert_allclose(gpu["values"][valid,3:5],cpu["values"][valid,3:5],rtol=0,atol=1e-7)
        np.testing.assert_allclose(gpu["values"][valid,9:11],cpu["values"][valid,9:11],rtol=0,atol=1e-7)
        np.testing.assert_array_equal(cpu_cold["status"],cpu["status"])
        for x in cpu_runs+[cpu_cold]:
            np.testing.assert_array_equal(x["values"],cpu["values"])
        for x in gpu_runs+[gpu_cold]:
            np.testing.assert_array_equal(x["values"],gpu["values"])
        evaluation_memory=memory()
        np.savez(out/"evaluation.npz",cpu=cpu["values"],gpu=gpu["values"],cpu_status=cpu["status"],gpu_status=gpu["status"],
                 original_cpu_xyz=reference_xyz,original_cpu_uv=original["uv"],original_cpu_status=original["status"])
    note("phase=numerical_check_passed; rendering exact source frames")
    delta=np.linalg.norm(cpu["values"][valid,3:5]-gpu["values"][valid,3:5],axis=1)
    residual=np.linalg.norm(gpu["values"][valid,9:11],axis=1)
    per_frame=[]
    for p in pts:
        ix=arrays["pts"][rows]==p;ok=ix&valid
        per_frame.append({"pts":int(p),"source_time_seconds":float(p*source["time_base_num"]/source["time_base_den"]),
            "rows":int(ix.sum()),"valid":int(ok.sum()),"invalid":int((ix&~valid).sum()),
            "residual_norm_px":quantile(np.linalg.norm(gpu["values"][ok,9:11],axis=1)),
            "cpu_gpu_uv_max_px":float(np.max(np.abs(cpu["values"][ok,3:5]-gpu["values"][ok,3:5]))) if ok.any() else None})
    result={"schema":"rinkspace.g1.result.v1","run_id":out.name,"status":"evaluated_pending_render","reason_code":"readonly_residual_evaluation",
        "last_checkpoint":None,"selected_shot_ids":[window["shot_id"]],"required_shot_ids":prepared["required_shot_ids"],
        "scope_status":"subset_window_and_factor_family","native_pts":pts.tolist(),"native_frame_count":len(pts),
        "rows":n,"participating_entities":binding,"unsupported":excluded,"fit_executed":False,"cuda_executed":True,
        "parameters_updated":False,"stations_updated":False,"whitening_executed":False,"user_approval":"not_reviewed",
        "objective_terms":{"raw_uv_residual_dimension":2,"valid_rows":int(valid.sum()),"invalid_rows":int((~valid).sum()),
            "status_counts":{str(k):int(v) for k,v in zip(*np.unique(cpu["status"],return_counts=True))},
            "residual_norm_px":quantile(residual),"raw_squared_sum_diagnostic_only":float(np.sum(residual**2)),
            "statistical_objective":None,"group_policy":"full original groups preserved; whitening deferred to G2"},
        "numerical":{"cpu_gpu_uv_norm_px":quantile(delta),"cpu_gpu_residual_abs_max_px":float(np.max(np.abs(cpu["values"][valid,9:11]-gpu["values"][valid,9:11]))),
            "original_cpu_uv_abs_max_px":float(np.max(np.abs(cpu["values"][valid,3:5]-original["uv"][valid]))),
            "status_agreement":True,"repeat_values_identical":True,"tolerance_abs_px":1e-7},
        "timings":{"resident_creation":creation,"cpu_cold":cpu_cold["timings"],"cuda_cold":gpu_cold["timings"],
            "cpu_warm_runs":[x["timings"] for x in cpu_runs],"cuda_warm_runs":[x["timings"] for x in gpu_runs],
            "cpu_wall_ms":quantile([x["timings"]["wall_ms"] for x in cpu_runs]),
            "cuda_wall_ms":quantile([x["timings"]["wall_ms"] for x in gpu_runs]),
            "cuda_kernel_ms":quantile([x["timings"]["kernel_event_ms"] for x in gpu_runs]),
            "cuda_d2h_ms":quantile([x["timings"]["d2h_event_ms"] for x in gpu_runs]),
            "cold_cuda_note":"First CUDA event span may include lazy module loading / host launch delay; reported separately from warm kernel time.",
            "timing_boundary":"Native call, entire 101-frame batch; CPU serial same-source reference. CUDA wall includes kernel and D2H. Python copies/render/IO excluded."},
        "memory":{"process_initial":initial_memory,"process_before_cuda":before_cuda,"process_resident":resident_memory,"process_after_evaluation":evaluation_memory,
            "owned_gpu_bytes":creation["owned_device_bytes"],"allocator_free_delta_bytes":creation["free_before_alloc_bytes"]-creation["free_after_alloc_bytes"],
            "note":"cudaMemGetInfo includes device-global changes, not exclusive process peak; owned buffers are exact. Process working set includes preparation and driver."},
        "artifacts":{"problem":"problem.json","evaluation":"evaluation.npz","frames":"frames.json","review":"index.html"},
        "limitations":["Only current shared curling contour factors have CUDA support.",
                       "Frozen historical stations are evaluated against current geometry; residual size is not a new fit or point-quality acceptance.",
                       "No Jacobian, whitening, block accumulation, parameter update or nonlinear optimizer executed."]}
    write(out/"frames.json",per_frame)
    write(out/"result.json",result)
    render(args,out,core,scene,camera,controls,source,pts,arrays,rows,supported,cpu,gpu,result,note)
    for p,h in immutable.items():
        if sha(p)!=h:raise AssertionError("input mutated: "+p)
    protected=out/"protected_before.json"
    if protected.exists():
        old_hashes=read(protected);changed=[x["path"] for x in old_hashes if sha(x["path"])!=x["sha256"]]
        if changed:raise AssertionError("protected model data changed: "+repr(changed))
        result["protected_model_files_unchanged"]=len(old_hashes)
    result["status"]="completed"
    result["input_files_unchanged"]=len(immutable)
    result["timings"]["whole_run_seconds"]=time.perf_counter()-t0
    result["memory"]["process_after_render"]=memory()
    write(out/"result.json",result)
    note("phase=complete; inputs_unchanged=true; fit_executed=false; see result.json and index.html")

def render(args,out,core,scene,camera,controls,source,pts,arrays,rows,supported,cpu,gpu,result,note):
    # Same native GPU evaluator is used for dense DISPLAY samples, separately
    # from the actual observation factors measured above. Display is not evidence.
    import cv2
    from PIL import Image,ImageDraw
    from rinkspace.review.fonts import review_font
    paths=list(supported.values());ns=257
    draw_codes=np.repeat([p["generated_code"] for p in paths],ns).astype(np.int32)
    draw_stations=np.tile(np.linspace(0,1,ns),len(paths))
    draw_kwargs=dict(board=scene["board"],layout=scene["shared_layout"],options=scene["shared_geometry_options"],
        rig=camera["rig"],controls=controls,first_control_time=camera["first_control_time"],control_dt=camera["control_dt"],
        image_height=camera["image_height"],codes=draw_codes,stations=draw_stations)
    image_dir=out/"frames";image_dir.mkdir(exist_ok=True)
    tiles=[];image_records=[];max_display_delta=0.
    # Start, middle, end and actual worst residual time; no accuracy-based omission.
    worst=max(read(out/"frames.json"),key=lambda x:x["residual_norm_px"]["max"])["pts"]
    picks={int(pts[0]),int(pts[len(pts)//2]),int(pts[-1]),worst}
    font=review_font(24);small=review_font(19)
    for frame_number,(p,frame) in enumerate(source_frames(Path(source["asset_path"]),source,pts)):
        times=np.full(len(draw_codes),(int(p)-int(camera["reference_pts"]))*camera["time_base_num"]/camera["time_base_den"])
        with ResidentResidualProblem(core,**draw_kwargs,times=times,observed=np.zeros((len(draw_codes),2))) as display:
            dc=display.evaluate("cpu");dg=display.evaluate("cuda")
        if not np.array_equal(dc["status"],dg["status"]):raise AssertionError("display status mismatch")
        ok=dc["status"]==0
        if ok.any():max_display_delta=max(max_display_delta,float(np.max(np.abs(dc["values"][ok,3:5]-dg["values"][ok,3:5]))))
        panels=[]
        for backend,data,clr in (("CPU",dc,(255,210,20)),("GPU",dg,(30,170,255))):
            image=frame.copy()
            for j in range(len(paths)):
                a=j*ns;b=a+ns
                uv=data["values"][a:b,3:5];status=data["status"][a:b]
                for k in range(ns-1):
                    if status[k] or status[k+1]:continue
                    q=uv[k:k+2]
                    if np.max(np.abs(q))>1e7:continue
                    cv2.line(image,tuple(np.rint(q[0]).astype(int)),tuple(np.rint(q[1]).astype(int)),clr,2,cv2.LINE_AA)
            rr=np.flatnonzero(arrays["pts"][rows]==p)
            for k in rr:
                uv=arrays["uv"][rows[k]]
                cv2.circle(image,tuple(np.rint(uv).astype(int)),3,(30,245,245),-1,cv2.LINE_AA)
                if cpu["status"][k]==0:
                    prediction=(cpu if backend=="CPU" else gpu)["values"][k,3:5]
                    cv2.line(image,tuple(np.rint(uv).astype(int)),tuple(np.rint(prediction).astype(int)),(180,180,180),1,cv2.LINE_AA)
            rgb=Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
            header=Image.new("RGB",(1920,74),(18,23,31));d=ImageDraw.Draw(header)
            d.text((16,5),f"{backend} | S01  {p/12800:.2f} s  | PTS {p} | R3/R4: inner / middle / outer",font=font,fill="white")
            d.text((16,39),"黄点：原始观测    灰线：冻结局部坐标的残差    曲线：当前 native 投影（未更新参数）",font=small,fill=(220,220,220))
            full=Image.new("RGB",(1920,1154));full.paste(header,(0,0));full.paste(rgb,(0,74))
            filename=f"{p}_{backend.lower()}.jpg";full.save(image_dir/filename,quality=91)
            panels.append(full)
        row=Image.new("RGB",(3840,1154));row.paste(panels[0],(0,0));row.paste(panels[1],(1920,0))
        if p in picks:
            row.resize((1920,577)).save(out/f"comparison_{p}.jpg",quality=95)
            tiles.append((p,row.resize((1920,577))))
        image_records.append({"pts":p,"seconds":p/12800,"cpu":f"frames/{p}_cpu.jpg","gpu":f"frames/{p}_gpu.jpg"})
        if frame_number%25==0:note(f"phase=render; decoded={frame_number+1}/{len(pts)}")
    result["numerical"]["dense_display_cpu_gpu_abs_max_px"]=max_display_delta
    result["rendered_source_frames"]=len(image_records)
    tiles.sort()
    strip=Image.new("RGB",(1920,577*len(tiles)))
    for i,(_,img) in enumerate(tiles):strip.paste(img,(0,i*577))
    strip.save(out/"comparison.jpg",quality=94)
    write(out/"image_index.json",image_records)
    html="""<!doctype html><meta charset="utf-8"><title>G1 — 原片 CPU/GPU 投影</title>
<style>body{background:#121821;color:#edf2f6;font:16px system-ui;margin:24px}h1{font-size:23px}.bar{display:flex;gap:14px;align-items:center;margin:14px 0}input{width:55vw}button{padding:7px 12px}#pair{display:grid;grid-template-columns:1fr 1fr;gap:8px}img{width:100%}p{line-height:1.6;color:#ccd5e0}a{color:#8ec9ff}.metric{padding:14px;background:#202a38;border-radius:8px}</style>
<h1>G1 · 同一窗口原片上的 CPU / GPU 对照</h1>
<p>范围：S01 原片 24.00–26.00 秒，101 个原生 PTS，R3/R4 六条靶圈轮廓、5,920 条原始观测。当前 native 几何与 R2 相机冻结；没有参数更新。其它线、围挡、广告尚未进入 G1 CUDA。</p>
<div class="metric">__METRICS__</div>
<p>左：CPU；右：GPU。黄点是原始观测，灰线连接其冻结局部坐标的模型投影。CPU/GPU 重合只验证计算一致，不表示模型已经贴合原片。大残差原样保留。点击图像查看原分辨率。</p>
<div class="bar"><button id="prev">上一帧</button><input id="slider" type="range" min="0" max="100" value="0"><button id="next">下一帧</button><span id="label"></span></div>
<div id="pair"><a id="ca"><img id="cpu"></a><a id="ga"><img id="gpu"></a></div>
<p><a href="result.json">完整耗时、内存与数值结果</a> · <a href="problem.json">参与实体与未支持清单</a> · <a href="frames.json">逐帧残差与失效分母</a></p>
<script>const frames=__FRAMES__;const slider=document.getElementById('slider');function show(){const f=frames[+slider.value];cpu.src=f.cpu;gpu.src=f.gpu;ca.href=f.cpu;ga.href=f.gpu;label.textContent=f.seconds.toFixed(2)+' 秒 / PTS '+f.pts;}
slider.oninput=show;prev.onclick=()=>{slider.value=Math.max(0,+slider.value-1);show()};next.onclick=()=>{slider.value=Math.min(frames.length-1,+slider.value+1);show()};show();</script>"""
    timing=result["timings"];owned=result["memory"]["owned_gpu_bytes"]/1024**2
    metrics=(f"整批 5,920 条残差：CPU {timing['cpu_wall_ms']['median']:.3f} ms；GPU 含回传 {timing['cuda_wall_ms']['median']:.3f} ms；"
             f"GPU 内核 {timing['cuda_kernel_ms']['median']:.3f} ms；GPU 自有缓冲 {owned:.3f} MiB。"
             f"原始残差中位数 {result['objective_terms']['residual_norm_px']['median']:.3f} px，最大 {result['objective_terms']['residual_norm_px']['max']:.3f} px。")
    (out/"index.html").write_text(html.replace("__FRAMES__",json.dumps(image_records)).replace("__METRICS__",metrics),encoding="utf-8")

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("scene","camera","window","prepared","local-state","library","cpu-reference","out"):
        parser.add_argument("--"+name,type=Path,required=True)
    args=parser.parse_args()
    try:main(args)
    except Exception:
        args.out.mkdir(parents=True,exist_ok=True)
        with (args.out/"main.log").open("a",encoding="utf-8") as f:f.write(traceback.format_exc())
        write(args.out/"failure.json",{"status":"failed","error":traceback.format_exc(),"fit_executed":False})
        raise
