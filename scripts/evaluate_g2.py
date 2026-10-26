"""Audit G2 on the frozen G1 window. No parameter updates or optimizer.

Finite differences and NumPy algebra below are explicit verification only.
Production linearize stays native/CUDA and returns only a scalar summary.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import runpy
import time
import traceback
import numpy as np
from rinkspace.core.native import NativeCore
from rinkspace.core.gpu import ResidentResidualProblem
from rinkspace.session.gpu_problem import prepare_g2
from rinkspace.compat.d78_entities import native_definition
from rinkspace.compat.d78_state import project_native_pts
from rinkspace.review.render import source_frames

helpers=runpy.run_path(str(Path(__file__).with_name("evaluate_g1.py")))
read,write,sha,memory,quantile=(helpers[n] for n in ("read","write","sha","memory","quantile"))
PLAN_KEYS=("directions","local_ids","local_scales","group_offsets","members","ranks","weight_offsets","weights")

def inputs(p, *, controls=None, stations=None, codes=None, times=None, observed=None):
    a=p["arrays"];c=p["camera"];s=p["scene"]
    return dict(board=s["board"],layout=s["shared_layout"],options=s["shared_geometry_options"],
        rig=c["rig"],controls=p["controls"] if controls is None else controls,
        first_control_time=c["first_control_time"],control_dt=c["control_dt"],image_height=c["image_height"],
        codes=a["codes"] if codes is None else codes,stations=a["stations"] if stations is None else stations,
        times=a["times"] if times is None else times,observed=a["uv"] if observed is None else observed)

def component_columns(layout,index,q):
    if "component_columns" not in layout:return np.arange(q)
    begin,count=map(int,layout["component_columns"][index])
    return layout["columns"][begin:begin+count]

def close(a,b,rtol=1e-9,atol=1e-8):
    np.testing.assert_allclose(a,b,rtol=rtol,atol=atol)
    return {"max_abs":float(np.max(np.abs(a-b),initial=0.)),"relative_l2":float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-30))}

def cpu_projection(core,p,rows,controls=None,stations=None):
    a=p["arrays"];s=p["scene"];rows=np.asarray(rows,np.int64)
    stations=a["stations"][rows] if stations is None else np.asarray(stations)
    xyz=np.empty((len(rows),3));paths={x["generated_code"]:x for x in s["paths"] if "generated_code" in x}
    for code in np.unique(a["codes"][rows]):
        ix=np.flatnonzero(a["codes"][rows]==code);path=paths[int(code)]
        kind,params=native_definition(s,path,core)
        xyz[ix]=core.sample_path(s["board"],kind,params,path.get("vertices",[]),stations[ix].tolist())
    result=project_native_pts(core,p["camera"],p["controls"] if controls is None else controls,a["pts"][rows],xyz)
    if np.any(result["status"]):raise AssertionError("CPU finite-difference probe left certified native domain")
    return result["uv"]

def numerical_derivatives(problem,reference,p,out):
    a=p["arrays"];q=p["directions"].shape[2]
    selected=[]
    for code in np.unique(a["codes"]):
        rr=np.flatnonzero(a["codes"]==code)
        selected.extend(rr[np.linspace(0,len(rr)-1,5).astype(int)].tolist())
    rows=np.array(sorted(set(selected)),np.int32)
    analytic=problem.probe_rows(rows)
    cpu_analytic=problem.probe_rows(rows,"cpu")
    cpu_gpu=close(analytic,cpu_analytic,rtol=1e-10,atol=1e-8)
    checks=[];saved=[]
    for step in (.002,.001,.0005):
        fd=np.empty_like(analytic)
        for col in range(q):
            delta=step*p["directions"][:,:,col]
            plus=cpu_projection(reference,p,rows,p["controls"]+delta)
            minus=cpu_projection(reference,p,rows,p["controls"]-delta)
            fd[:,:,col]=(plus-minus)/(2*step)
        scales=p["local_scales"][p["local_ids"][rows]]
        plus=cpu_projection(reference,p,rows,stations=a["stations"][rows]+step*scales)
        minus=cpu_projection(reference,p,rows,stations=a["stations"][rows]-step*scales)
        fd[:,:,-1]=(plus-minus)/(2*step)
        error=np.abs(analytic-fd)
        scaled=error/np.maximum(1.,np.abs(analytic))
        if scaled.max()>3e-6:raise AssertionError(f"actual native central difference mismatch {scaled.max()}")
        checks.append({"scaled_step":step,"max_abs_px_per_scaled_unit":float(error.max()),
                       "max_error_over_max_1_abs_J":float(scaled.max())})
        saved.append(fd)
    np.savez(out/"derivative_checks.npz",rows=rows,original_rows=a["rows"][rows],pts=a["pts"][rows],
             analytic_cuda=analytic,analytic_cpu=cpu_analytic,finite_differences=np.array(saved))
    return {"rows":len(rows),"source_pts":np.unique(a["pts"][rows]).tolist(),"columns_per_row":q+1,
        "method":"central differences of the previous native CPU DLL, only for engineering verification",
        "checks":checks,"cpu_gpu":cpu_gpu}

def whitening_checks(problem,p,out):
    records=p["group_records"];sizes=np.diff(p["group_offsets"])
    chosen=sorted(set([0,len(records)-1,int(sizes.argmax()),int(np.argmin(p["ranks"])),len(records)//2]))
    checks=[]
    residual=problem.evaluate("cpu")["values"][:,9:11]
    for gi in chosen:
        begin,end=p["group_offsets"][gi:gi+2];rows=p["members"][begin:end];rank=int(p["ranks"][gi])
        w=p["weights"][p["weight_offsets"][gi]:p["weight_offsets"][gi+1]].reshape(rank,2*len(rows))
        raw=problem.probe_rows(rows,"cpu")
        record=problem.layout["groups"][gi];component=problem.layout["components"][record[4]]
        lb,nl,_,_,dim,_,_=map(int,component);local_order=problem.layout["locals"][lb:lb+nl]
        local_col={int(v):j for j,v in enumerate(local_order)}
        cols=component_columns(problem.layout,int(record[4]),problem._q);nq=len(cols)
        J=np.zeros((2*len(rows),dim));J[:,:nq]=raw[:,:,cols].reshape(2*len(rows),nq)
        for j,row in enumerate(rows):
            lid=int(p["local_ids"][row])
            if lid>=0:J[2*j:2*j+2,nq+local_col[lid]]=raw[j,:,-1]
        expected_r=w@residual[rows].ravel();expected_J=w@J
        got=problem.probe_group(gi);got_cpu=problem.probe_group(gi,"cpu")
        rc=close(got["white_r"],expected_r);jc=close(got["white_J"],expected_J)
        close(got["white_J"],got_cpu["white_J"])
        squares=float(expected_r@expected_r);delta=p["activity"]["loss"]["delta"]
        rms=np.sqrt(squares/rank) if rank else 0
        weight=min(1.,delta/rms) if rms else (1. if rank else 0.)
        cost=.5*(squares if rms<=delta else rank*(2*delta*rms-delta*delta))
        close(got["loss"],np.array([squares,weight,cost]))
        checks.append({"group":gi,"id":records[gi]["id"],"members":len(rows),"rank":rank,
                       "residual":rc,"jacobian":jc,"loss_checked":True})
    write(out/"whitening_checks.json",checks)
    return checks

def blocks_and_rank(blocks,problem,p,out):
    q=problem._q;groups=problem.layout["groups"];components=problem.layout["components"]
    ranks=[];profile=blocks["shared_H"].copy();max_asym=0.;minimum_eigen=0.;cross_nonzero=0
    for ci,component in enumerate(components):
        lb,nl,_,_,dim,h0,g0=map(int,component)
        cols=component_columns(problem.layout,ci,q);nq=len(cols)
        if dim==0:continue
        H=blocks["H"][h0:h0+dim*dim].reshape(dim,dim);C=H[nq:,nq:];B=H[:nq,nq:]
        max_asym=max(max_asym,float(np.max(np.abs(H-H.T))))
        eig=np.linalg.eigvalsh((H+H.T)/2)
        minimum_eigen=min(minimum_eigen,float(eig[0]))
        if eig[0]<-1e-9*max(1.,abs(eig[-1])):raise AssertionError("non-PSD Gram block")
        if nl:
            ev=np.linalg.eigvalsh((C+C.T)/2);tol=max(float(ev[-1]),1e-300)*nl*np.finfo(float).eps*64
            rank=int(np.count_nonzero(ev>tol));ranks.append({"component":ci,"locals":nl,"rank":rank,"tolerance":tol})
            profile[np.ix_(cols,cols)]-=B@np.linalg.pinv(C,rcond=nl*np.finfo(float).eps*64,hermitian=True)@B.T
            if nl>1:cross_nonzero+=int(np.count_nonzero(np.triu(np.abs(C)>1e-14,1)))
    eig=np.linalg.eigvalsh((profile+profile.T)/2)
    nonzero=2*len(problem.layout["csr_columns"])-q if "csr_columns" in problem.layout else q*q
    nonzero+=sum(2*len(component_columns(problem.layout,ci,q))*int(c[1])+int(c[1])**2 for ci,c in enumerate(components))
    result={"shared_variables":q,"station_variables":int(problem.linearization_info["local_count"]),
        "connected_components":len(components),"max_station_component":int(problem.linearization_info["max_local_component"]),
        "formal_whitened_residual_dimension":int(problem.linearization_info["whitened_rank"]),
        "packed_H_values":len(blocks["H"]),"packed_gradient_values":len(blocks["g"]),
        "assembled_structural_H_nonzeros":nonzero,"local_cross_entries_nonzero":cross_nonzero,
        "maximum_asymmetry":max_asym,"minimum_Gram_eigenvalue":minimum_eigen,
        "rank_deficient_local_components":sum(x["rank"]<x["locals"] for x in ranks),
        "local_rank_records":ranks,"profiled_shared_eigenvalues":eig.tolist(),
        "profiled_shared_condition":float(eig[-1]/eig[0]),
        "rank_method":"CPU diagnostic eig/pseudoinverse of GPU Gram blocks, rcond=local_count*eps64*64; not production Schur or update",
        "production_elimination_executed":False}
    if eig[0]<=0:raise AssertionError("selected shared activity has no positive conditional information")
    write(out/"block_structure.json",result)
    return result

def directional_check(reference,p,blocks,problem):
    a=p["arrays"];q=problem._q;nlocal=len(p["local_scales"])
    dq=np.linspace(.2,.7,q);dl=np.zeros(nlocal)
    for row in np.linspace(0,len(a["rows"])-1,7).astype(int):dl[p["local_ids"][row]]=.3
    step=.001
    controls_delta=np.einsum("mcq,q->mc",p["directions"],dq)
    station_delta=np.array([0. if i<0 else dl[i]*p["local_scales"][i] for i in p["local_ids"]])
    rows=np.arange(len(a["rows"]))
    base=cpu_projection(reference,p,rows)-a["uv"]
    plus=cpu_projection(reference,p,rows,p["controls"]+step*controls_delta,a["stations"]+step*station_delta)-a["uv"]
    minus=cpu_projection(reference,p,rows,p["controls"]-step*controls_delta,a["stations"]-step*station_delta)-a["uv"]
    def objective(r):
        total=0.;delta=p["activity"]["loss"]["delta"]
        for gi,rank in enumerate(p["ranks"]):
            rr=p["members"][p["group_offsets"][gi]:p["group_offsets"][gi+1]]
            w=p["weights"][p["weight_offsets"][gi]:p["weight_offsets"][gi+1]].reshape(rank,2*len(rr))
            white=w@r[rr].ravel();s=white@white
            total+=.5*(s if s<=rank*delta**2 else rank*(2*delta*np.sqrt(s/rank)-delta**2))
        return total
    expected_gradient=(objective(plus)-objective(minus))/(2*step)
    expected_curvature=0.
    dr=(plus-minus)/(2*step)
    for gi,rank in enumerate(p["ranks"]):
        rr=p["members"][p["group_offsets"][gi]:p["group_offsets"][gi+1]]
        w=p["weights"][p["weight_offsets"][gi]:p["weight_offsets"][gi+1]].reshape(rank,2*len(rr))
        r=w@base[rr].ravel();d=w@dr[rr].ravel();rms=np.sqrt(r@r/rank) if rank else 0.
        weight=min(1.,p["activity"]["loss"]["delta"]/rms) if rms else (1. if rank else 0.)
        expected_curvature+=weight*(d@d)
    gradient=0.;curvature=0.
    for ci,component in enumerate(problem.layout["components"]):
        lb,nl,_,_,dim,h0,g0=map(int,component);ids=problem.layout["locals"][lb:lb+nl]
        direction=np.r_[dq[component_columns(problem.layout,ci,q)],dl[ids]]
        H=blocks["H"][h0:h0+dim*dim].reshape(dim,dim)
        gradient+=direction@blocks["g"][g0:g0+dim]
        curvature+=direction@H@direction
    close(np.array([gradient,curvature]),np.array([expected_gradient,expected_curvature]),rtol=3e-6,atol=3e-5)
    return {"scaled_step":step,"gradient_from_GPU_blocks":gradient,"gradient_from_CPU_objective_difference":expected_gradient,
        "curvature_from_GPU_blocks":curvature,"curvature_from_CPU_directional_residual_difference":expected_curvature,
        "parameter_update_executed":False}

def main(args):
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    def note(text):
        print(text,flush=True)
        with (out/"main.log").open("a",encoding="utf-8") as f:f.write(text+"\n")
    note(f"G2 main log: {out/'main.log'}")
    started=time.perf_counter();p=prepare_g2(args.g1,motion_activity="level")
    before={x["path"]:sha(x["path"]) for x in read(args.g1/"protected_before.json")}
    expected={x["path"]:x["sha256"] for x in read(args.g1/"protected_before.json")}
    if before!=expected:raise AssertionError("current native model/camera data differ from G1")
    write(out/"activity.json",p["activity"])
    write(out/"noise_groups.json",p["group_records"])
    np.savez(out/"plan.npz",**{k:p[k] for k in PLAN_KEYS})
    identities={str(args.g1/name):sha(args.g1/name) for name in ("inputs.npz","scene.json","camera.json","controls.npy","problem.json")}
    write(out/"input_identity.json",identities)
    note("phase=resident; 5920 original rows; three camera levels; original stations and full W retained")
    core=NativeCore(args.library);reference=NativeCore(args.reference_library)
    process_before=memory()
    with ResidentResidualProblem(core,**inputs(p)) as problem:
        info=problem.configure_linearization(**{k:p[k] for k in PLAN_KEYS},huber_delta=p["activity"]["loss"]["delta"])
        resident_memory=memory()
        cpu_cold=problem.linearize("cpu");cuda_cold=problem.linearize("cuda")
        cpu_runs=[problem.linearize("cpu") for _ in range(5)]
        cuda_runs=[problem.linearize("cuda") for _ in range(7)]
        cpu=cpu_runs[-1];gpu=cuda_runs[-1];after_linearize=memory()
        note(f"phase=linearized; components={int(info['component_count'])} rank={int(info['whitened_rank'])} summary_d2h={gpu['timings']['summary_d2h_bytes']}")
        if problem.diagnostic_download_bytes!=0:raise AssertionError("normal path downloaded Jacobians")
        cb=problem.export_blocks("cpu");gb=problem.export_blocks("cuda")
        comparisons={k:close(gb[k],cb[k]) for k in cb}
        for x in cpu_runs+cuda_runs:
            close(np.array(list(x["summary"].values())),np.array(list(gpu["summary"].values())))
        np.savez(out/"linearization_cpu_gpu.npz",**{"cpu_"+k:v for k,v in cb.items()},**{"gpu_"+k:v for k,v in gb.items()},
                 **problem.layout)
        # The G1 raw problem remains unchanged after G2 activity compilation.
        old=np.load(args.g1/"evaluation.npz")
        current=problem.evaluate("cuda")
        close(current["values"],old["gpu"],rtol=0,atol=1e-9)
        np.testing.assert_array_equal(current["status"],old["gpu_status"])
        note("phase=finite_difference_and_whitening_checks")
        derivatives=numerical_derivatives(problem,reference,p,out)
        white=whitening_checks(problem,p,out)
        structure=blocks_and_rank(gb,problem,p,out)
        direction=directional_check(reference,p,gb,problem)
        note("phase=checks_passed; no update; rendering explanatory perturbations")
        gpu_transfer=problem.diagnostic_download_bytes
        result={"schema":"rinkspace.g2.result.v1","run_id":out.name,"status":"verified_pending_render",
            "reason_code":"readonly_linearization","last_checkpoint":None,"selected_shot_ids":["s01"],
            "required_shot_ids":read(args.g1/"result.json")["required_shot_ids"],"scope_status":"same_G1_window_factor_subset",
            "native_pts":p["problem"]["native_pts"],"rows":len(p["arrays"]["rows"]),"native_frame_count":len(p["problem"]["native_pts"]),
            "participating_entities":p["problem"]["participating_entities"],"unsupported":p["problem"]["unsupported"],
            "activity":p["activity"],"layout":info,"objective_terms":gpu["summary"],"precision":"float64",
            "whitening_executed":True,"jacobian_executed":True,"block_accumulation_executed":True,
            "fit_executed":False,"cuda_executed":True,"parameters_updated":False,"station_values_updated":False,"user_approval":"not_reviewed",
            "timings":{"cpu_first":cpu_cold["timings"],"cuda_first_this_process":cuda_cold["timings"],
                "cpu_warm_runs":[x["timings"] for x in cpu_runs],"cuda_warm_runs":[x["timings"] for x in cuda_runs],
                "cpu_wall_ms":quantile([x["timings"]["wall_ms"] for x in cpu_runs]),
                "cuda_wall_ms":quantile([x["timings"]["wall_ms"] for x in cuda_runs]),
                "cuda_stage_ms":{k:quantile([x["timings"][k] for x in cuda_runs]) for k in
                    ("raw_jacobian_event_ms","whitening_event_ms","loss_event_ms","block_reduce_event_ms","summary_d2h_event_ms")},
                "boundary":"native linearization of entire 101-frame problem; includes raw J, full W, rank-Huber IRLS, connected blocks and scalar summary; excludes setup, diagnostics and rendering"},
            "memory":{"process_before_cuda":process_before,"process_resident":resident_memory,"process_after_linearization":after_linearize,
                "resident_G1_buffers_bytes":problem.creation["owned_device_bytes"],"G2_buffers_bytes":info["device_bytes"],
                "total_owned_GPU_buffers_bytes":problem.creation["owned_device_bytes"]+info["device_bytes"],
                "normal_linearization_d2h_bytes":32,"explicit_diagnostic_download_bytes":gpu_transfer,
                "note":"exact owned GPU buffers; process working set includes driver and preparation, not an exclusive GPU peak"},
            "validation":{"blocks_cpu_gpu":comparisons,"finite_difference":derivatives,"whitening_numpy":white,
                "directional_objective_and_Gram":direction,"block_structure":structure,"G1_raw_forward_unchanged":True},
            "artifacts":{"root":str(out),"activity":"activity.json","structure":"block_structure.json","review":"index.html",
                "derivatives":"derivative_checks.npz","blocks":"linearization_cpu_gpu.npz"},
            "limitations":["Only G1 shared curling contours participate; other factor families remain unsupported.",
                "Local station Gram blocks may be rank deficient. No damping, production elimination, cuDSS or LM update was executed.",
                "Activity is three conditional camera levels; temporal slopes, shared geometry, installation, optics and readout remain frozen.",
                "Huber delta=3 is explicit G2 configuration. Motion/measured priors are not introduced without weights."] }
    write(out/"result.json",result)
    render_perturbations(core,p,out,note)
    for path,h in before.items():
        if sha(path)!=h:raise AssertionError("protected model or camera changed")
    for path,h in identities.items():
        if sha(path)!=h:raise AssertionError("G1 input was modified")
    result["protected_files_unchanged"]=len(before);result["status"]="completed"
    result["timings"]["whole_run_seconds"]=time.perf_counter()-started
    result["memory"]["process_after_render"]=memory()
    first=out/"first_probe.json"
    if first.exists():
        result["timings"]["first_observed_new_kernel_run"]={
            "activity_dimension":6, "scope":"earlier affine-activity startup probe, not the final Q=3 benchmark",
            "timings":read(first)["gpu"]["timings"]}
    write(out/"result.json",result)
    note("phase=complete; same raw residuals; no optimization or source/model changes")

def render_perturbations(core,p,out,note):
    import cv2
    from PIL import Image,ImageDraw
    from rinkspace.review.fonts import review_font
    source=p["problem"]["source"];c=p["camera"];a=p["arrays"]
    native_pts=p["problem"]["native_pts"];pts=native_pts[len(native_pts)//2]
    _,frame=next(source_frames(Path(source["asset_path"]),source,[pts]))
    paths=p["problem"]["participating_entities"];ns=257
    codes=np.repeat([x["native_code"] for x in paths],ns).astype(np.int32)
    stations=np.tile(np.linspace(0,1,ns),len(paths))
    times=np.full(len(codes),(pts-int(c["reference_pts"]))*c["time_base_num"]/c["time_base_den"])
    def project(controls):
        with ResidentResidualProblem(core,**inputs(p,controls=controls,codes=codes,stations=stations,times=times,observed=np.zeros((len(codes),2)))) as r:
            return r.evaluate("cuda")
    base=project(p["controls"]);outputs=[]
    choices=[("pan",0,1.,"pan +1 mrad"),("tilt",1,1.,"tilt +1 mrad"),("zoom",2,5.,"log(f) +0.005（约 +0.5%）")]
    font=review_font(25);small=review_font(20)
    records=[]
    for name,column,amplitude,label in choices:
        shifted=project(p["controls"]+amplitude*p["directions"][:,:,column])
        image=frame.copy()
        for data,color in ((base,(255,210,20)),(shifted,(30,160,255))):
            for j in range(len(paths)):
                start=j*ns
                for k in range(ns-1):
                    i=start+k
                    if data["status"][i] or data["status"][i+1]:continue
                    xy=data["values"][i:i+2,3:5]
                    if np.max(np.abs(xy))>1e7:continue
                    cv2.line(image,tuple(np.rint(xy[0]).astype(int)),tuple(np.rint(xy[1]).astype(int)),color,2,cv2.LINE_AA)
        rgb=Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
        panel=Image.new("RGB",(1920,1160),(18,23,31));panel.paste(rgb,(0,80));d=ImageDraw.Draw(panel)
        d.text((16,4),f"G2 测试扰动：{label}  | S01 {pts/12800:.2f} s / PTS {pts}",font=font,fill="white")
        d.text((16,43),"青色：当前 native 投影；橙色：只改变所示相机方向。共享几何保持原值。这不是优化结果。",font=small,fill=(220,220,220))
        panel.save(out/f"perturb_{name}.jpg",quality=94)
        outputs.append(panel.resize((1280,773)))
        valid=(base["status"]==0)&(shifted["status"]==0)
        displacement=shifted["values"][valid,3:5]-base["values"][valid,3:5]
        records.append({"name":name,"column":column,"scaled_amplitude":amplitude,"physical_delta":amplitude*.001,
            "source_pts":pts,"valid_display_samples":int(valid.sum()),"displacement_px":quantile(np.linalg.norm(displacement,axis=1)),
            "source_state_updated":False,"image":f"perturb_{name}.jpg"})
    strip=Image.new("RGB",(1280,773*len(outputs)))
    for i,img in enumerate(outputs):strip.paste(img,(0,773*i))
    strip.save(out/"perturbations.jpg",quality=94)
    write(out/"perturbations.json",records)
    html="""<!doctype html><meta charset="utf-8"><title>G2：局部导数与真实块系统</title>
<style>body{background:#141b24;color:#edf3fa;font:16px system-ui;max-width:1500px;margin:24px auto;padding:0 20px}h1{font-size:25px}p{line-height:1.65;color:#cdd8e6}img{width:100%;border:1px solid #38495d}a{color:#89caff}.box{background:#253143;padding:15px;border-radius:9px}button{font-size:16px;padding:9px;margin:6px}</style>
<h1>G2 已执行：局部 Jacobian、完整相关组白化、连通组块累计</h1>
<p>沿用 G1 的 24–26 秒、101 帧、5,920 条真实观测。197 个原相关组和 73 个独立观测共形成 270 个连通组；白化后有效维数 1,300。没有优化或参数更新。</p>
<div class="box">活动方向仅为窗口 pan、tilt、log(f) 三个常量修正，加原有有限 station。时间斜率因条件性差而冻结。当前场地、广告、安装、畸变、读出全部冻结。小扰动只用于显示影响，不写回相机或模型。</div>
<p>按钮切换同一张 25 秒原片上的测试扰动。青色是当前模型投影，橙色是所标相机扰动的实际 GPU 投影；不代表贴合改善。</p>
<button onclick="show(0)">pan +1 mrad</button><button onclick="show(1)">tilt +1 mrad</button><button onclick="show(2)">焦距约 +0.5%</button>
<a id="full"><img id="picture"></a>
<p>导数检查、CPU/GPU 块比较、条件性与耗时由工程检查负责。<a href="result.json">完整结果</a> · <a href="activity.json">活动参数</a> · <a href="block_structure.json">块系统及秩亏记录</a></p>
<script>const imgs=["perturb_pan.jpg","perturb_tilt.jpg","perturb_zoom.jpg"];function show(i){document.getElementById("picture").src=imgs[i];document.getElementById("full").href=imgs[i]}show(0)</script>"""
    measured=read(out/"result.json")
    stats="<div class=\"box\"><b>实测：整个 101 帧窗口的一次线性化</b><br>GPU 热运行中位数 {gpu_ms:.3f} ms；CPU {cpu_ms:.3f} ms。自有 GPU 数据与计算缓冲 {gpu_mib:.2f} MiB；每次正常调用只下载 32 字节摘要。<br>CPU/GPU 块累计、完整 W 和小样本数值导数检查通过。192 个局部块存在秩亏；没有执行阻尼、消元或参数更新。</div>\n"
    stats=stats.format(gpu_ms=measured["timings"]["cuda_wall_ms"]["median"],cpu_ms=measured["timings"]["cpu_wall_ms"]["median"],gpu_mib=measured["memory"]["total_owned_GPU_buffers_bytes"]/1024**2)
    html=html.replace('<p>按钮切换',stats+'<p>按钮切换')
    (out/"index.html").write_text(html,encoding="utf-8")
    note("phase=perturbation_views_saved; fixed scene; no accepted update")

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("g1","library","reference-library","out"):parser.add_argument("--"+key,type=Path,required=True)
    args=parser.parse_args()
    try:main(args)
    except Exception:
        args.out.mkdir(parents=True,exist_ok=True)
        with (args.out/"main.log").open("a",encoding="utf-8") as f:f.write(traceback.format_exc())
        write(args.out/"failure.json",{"status":"failed","error":traceback.format_exc(),"fit_executed":False})
        raise
