"""Engineering capacity/solve check on current native geometry and synthetic UV.

No observation dataset is read or written. Outputs belong outside OneDrive.
Every requested row enters the same complete correlation partition and LM.
"""
from __future__ import annotations
import argparse
import ctypes as c
import json
import hashlib
import os
from pathlib import Path
import time
import numpy as np
from rinkspace.core.native import NativeCore
from rinkspace.session.gpu_geometry import NativeSceneGPU


_cuda_memory = None
_device_samples = []

def memory():
    if os.name!="nt":return {}
    class Status(c.Structure):
        _fields_=[("length",c.c_ulong),("load",c.c_ulong)]+[(n,c.c_ulonglong) for n in
            ("total_physical","available_physical","total_page","available_page","total_virtual","available_virtual","extended")]
    status=Status();status.length=c.sizeof(status)
    c.windll.kernel32.GlobalMemoryStatusEx(c.byref(status))
    class Counters(c.Structure):
        _fields_=[("cb",c.c_ulong),("faults",c.c_ulong)]+[(n,c.c_size_t) for n in
            ("peak_working_set","working_set","peak_paged","paged","peak_nonpaged","nonpaged","pagefile","peak_pagefile")]
    counters=Counters();counters.cb=c.sizeof(counters)
    current=c.windll.kernel32.GetCurrentProcess;current.restype=c.c_void_p
    get=c.windll.psapi.GetProcessMemoryInfo;get.argtypes=[c.c_void_p,c.c_void_p,c.c_ulong]
    get(current(),c.byref(counters),c.sizeof(counters))
    result={"available_host_bytes":int(status.available_physical),"host_working_set":int(counters.working_set),"host_peak_working_set":int(counters.peak_working_set)}
    if _cuda_memory is not None:
        free=c.c_size_t();total=c.c_size_t();error=_cuda_memory(c.byref(free),c.byref(total))
        if error:raise RuntimeError(f"cudaMemGetInfo failed: {error}")
        used=total.value-free.value;_device_samples.append(used)
        result.update(device_global_used_bytes=used,device_global_free_bytes=free.value,device_total_bytes=total.value)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library",required=True,type=Path);parser.add_argument("--scene",required=True,type=Path)
    parser.add_argument("--out",required=True,type=Path);parser.add_argument("--rows",type=int,default=1_000_000)
    parser.add_argument("--controls",type=int,default=160);parser.add_argument("--group-size",type=int,default=4)
    args=parser.parse_args();n=args.rows;nc=args.controls;q=3*nc;m=args.group_size
    if n<=0 or nc<8 or m<2 or m%2 or n%m:parser.error("positive rows divisible by an even group size, and at least 8 controls, required")
    root=Path(__file__).resolve().parents[1];out=args.out.resolve()
    if out==root or root in out.parents:parser.error("benchmark outputs must be outside the project/OneDrive tree")
    out.mkdir(parents=True,exist_ok=True);log=out/"iterations.jsonl"
    if log.exists() or (out/"result.json").exists():parser.error("output already contains a run; preserve its evidence and select another directory")
    started=time.perf_counter()
    def emit(kind,**fields):
        record={"kind":kind,"elapsed_seconds":time.perf_counter()-started,**fields}
        with log.open("a",encoding="utf8") as f:f.write(json.dumps(record,allow_nan=False)+"\n")
        print(json.dumps(record,allow_nan=False),flush=True)
    emit("start",rows=n,control_points=nc,retained_parameters=q,group_size=m,source="synthetic native projections; no dataset",**memory())
    core=NativeCore(args.library)
    global _cuda_memory
    runtime=c.CDLL("cudart64_13.dll")
    _cuda_memory=runtime.cudaMemGetInfo;_cuda_memory.argtypes=[c.POINTER(c.c_size_t),c.POINTER(c.c_size_t)];_cuda_memory.restype=c.c_int
    emit("device_baseline",**memory())
    geometry=NativeSceneGPU(args.scene,core);npairs=n//2
    pairs=np.arange(npairs);path_names=list(geometry.paths)
    path_ids=np.repeat(np.asarray(path_names,dtype=object)[pairs%len(path_names)],2).tolist()
    stations=np.repeat(.1+.8*np.mod(pairs*.6180339887498949,1.),2)
    first=-2.;dt=1.;t=np.repeat(np.linspace(-.8,nc-4.2,npairs),2)+np.tile([-.02,.02],npairs)
    reference=2**53+17;pts=reference+np.rint(t*12800).astype(np.int64)
    rig=[1.,0.,0.,0.,1.,0.,0.,0.,1.,0.,0.,1.,1.,0.,0.,0.,0.,-3.,0.,0.,0.,0.,0.,0.,100.,1.,50.,50.,0.,0.,0.,0.,0.,0.,0.,0.,0.,.02]
    camera=dict(rig=rig,first_control_time=first,control_dt=dt,image_height=101,reference_pts=reference,time_base_num=1,time_base_den=12800)
    knots=np.arange(nc);controls=np.c_[.01*np.sin(knots/30),.01*np.cos(knots/25),np.log(100.)+.02*np.sin(knots/40)]
    inputs=dict(camera=camera,controls=controls,path_ids=path_ids,stations=stations,pts=pts,observed=np.zeros((n,2)))
    with geometry.resident(**inputs) as target:
        projection=target.evaluate("cuda")
        if np.any(projection["status"]):raise RuntimeError("synthetic source has invalid native geometry/camera rows")
        wanted=projection["values"][:,3:5].copy()
    del projection
    emit("target_ready",native_paths=geometry.inventory["path_count"],**memory())
    inputs["observed"]=wanted;inputs["controls"]=controls+.0003*np.sin(np.arange(q).reshape(nc,3)+.3)
    active=pairs%11!=0;pair_local=np.full(npairs,-1,np.int32);pair_local[active]=np.arange(np.count_nonzero(active),dtype=np.int32)
    local_ids=np.repeat(pair_local,2);nl=int(np.count_nonzero(active));ng=n//m
    # Off-diagonal W and repeated material stations exercise connected Schur blocks.
    W=np.eye(2*m)+.07*np.tril(np.ones((2*m,2*m)),-1)
    weights=np.tile(W.ravel(),ng)
    with geometry.resident(**inputs) as p:
        before=time.perf_counter()
        p.configure_projection_time_support(half_seconds=abs(rig[37])/2)
        info=p.configure_linearization(directions=np.eye(q).reshape(nc,3,q)*.001,local_ids=local_ids,
            local_scales=np.full(nl,.001),group_offsets=np.arange(ng+1,dtype=np.int64)*m,members=np.arange(n),
            ranks=np.full(ng,2*m),weight_offsets=np.arange(ng+1,dtype=np.int64)*W.size,weights=weights,huber_delta=3.)
        planning=time.perf_counter()-before
        emit("linearization_plan",planning_seconds=planning,info=info,**memory())
        before=time.perf_counter();initial=p.linearize();linearization=time.perf_counter()-before
        solver=p.configure_solver(lower=np.zeros(nl),upper=np.ones(nl),device_budget_bytes=p.creation["free_after_alloc_bytes"],host_budget_bytes=memory()["available_host_bytes"])
        owned=p.creation["owned_device_bytes"]+p.geometry_device_bytes+info["device_bytes"]+solver["owned_device_bytes"]
        emit("ready",initial=initial,linearization_seconds=linearization,solver=solver,owned_device_bytes=owned,**memory())
        history=[]
        def progress(record):
            history.append(record);emit("iteration",**record,**memory())
        result=p.solve(progress=progress)
        final=p.evaluate("cuda");errors=np.linalg.norm(final["values"][:,3:5]-wanted,axis=1)
        final_memory=memory()
        report={"library":str(args.library.resolve()),"library_sha256":hashlib.sha256(args.library.read_bytes()).hexdigest(),"rows":n,"source":"synthetic native projections; no observation dataset used","scene":str(args.scene.resolve()),
            "native_paths":geometry.inventory["path_count"],"control_points":nc,"retained_parameters":q,"local_stations":nl,
            "complete_groups":ng,"group_size":m,"full_non_diagonal_whitening":True,"shared_station_pairs":True,
            "planning_seconds":planning,"initial_linearization_seconds":linearization,"linearization_info":info,"solver_info":solver,
            "owned_device_bytes":owned,"cuDSS_peak_estimate_bytes":max((e["summary"]["cudss_device_peak_estimate"] for e in history),default=0),
            "initial_cost":initial["summary"]["huber_cost"],"result":result,"final_failed_rows":int(np.count_nonzero(final["status"])),
            "pixel_error":{"median":float(np.median(errors)),"p99":float(np.quantile(errors,.99)),"max":float(np.max(errors))},
            "elapsed_seconds":time.perf_counter()-started,**final_memory,
            "device_memory_sampling":{"scope":"device-global cudaMemGetInfo samples; includes other activity; not a process-exclusive peak",
                "baseline_used_bytes":_device_samples[0],"sampled_peak_used_bytes":max(_device_samples),
                "sampled_peak_growth_bytes":max(_device_samples)-_device_samples[0],"samples":len(_device_samples)}}
        (out/"result.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf8")
        emit("complete",report=str(out/"result.json"),result=result,pixel_error=report["pixel_error"],owned_device_bytes=owned,**memory())
        if not result["numerically_converged"] or report["final_failed_rows"]:raise RuntimeError("engineering solve did not converge with every requested row valid; see saved report")


if __name__=="__main__":main()
