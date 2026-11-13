"""Run the authors' PVA-Net on prepared single-camera native source crops."""
from argparse import ArgumentParser
from fractions import Fraction
from pathlib import Path
import sys
import time

import numpy as np
import torch

from calibration.program_tracking.body_tracking.camera_bridge import rectified_to_native
from experiments.skate_video_v1.inputs import atomic_json, atomic_npz, digest, emit, read


def infer_many(paths, source, checkpoint, output, window, overlap):
    checkpoint_sha=digest(checkpoint)
    if checkpoint_sha!=read(checkpoint.parent/"weight_receipt.json")["sha256"]:
        raise ValueError("Official checkpoint bytes differ from its acquisition receipt")
    sys.path.insert(0,str(source))
    from pvanet.model_io import load_full_model
    from scripts.demo_pvanet import run_pvanet_inference
    device=torch.device("cuda")
    start=time.perf_counter()
    model=load_full_model(checkpoint,device=device)
    count=sum(p.numel() for p in model.parameters())
    emit("pva_model_loaded",parameters=count,elapsed_s=time.perf_counter()-start,device=torch.cuda.get_device_name())
    for path in paths:
        if len(paths)==1 and output.suffix==".npz":
            target=output
        else:
            target=output/(path.stem+".npz")
        target.parent.mkdir(exist_ok=True,parents=True)
        infer_clip(path,model,run_pvanet_inference,source,checkpoint_sha,device,target,window,overlap)


def infer_clip(path,model,run_inference,source,checkpoint_sha,device,output,window,overlap):
    start=time.perf_counter()
    if output.exists() and output.with_suffix(".json").exists():
        prior=read(output.with_suffix(".json"))
        if (prior["source_input_sha256"]==digest(path) and prior["checkpoint_sha256"]==checkpoint_sha
                and prior["output_sha256"]==digest(output)):
            emit("pva_verified_existing_clip",source=path.stem)
            return
    with np.load(path,allow_pickle=False) as archive:
        batch={k:archive[k] for k in archive.files}
    images=torch.from_numpy(batch["crops_rgb"]).permute(0,3,1,2)
    sample=dict(imgs=images[None].to(device=device,dtype=torch.float32)/255.,
                bbx_xyxy=torch.as_tensor(batch["bbx_xyxy"],device=device,dtype=torch.float32)[None],
                K_fullimg=torch.as_tensor(batch["K_rect"],device=device,dtype=torch.float32)[None])
    emit("pva_clip_begin",clip=path.stem,source_exposures=len(images),window=window,overlap=overlap)
    for w in dict.fromkeys((window,min(window,90),min(window,60),min(window,45))):
        torch.cuda.reset_peak_memory_stats()
        try:
            with torch.inference_mode():
                pred,_,conf,v,a,_,_,window_info=run_inference(model,sample,w,min(overlap,w//3),False)
            break
        except torch.cuda.OutOfMemoryError:
            emit("pva_window_out_of_memory",clip=path.stem,window=w,allocated=torch.cuda.max_memory_allocated())
            if w<=45:
                raise
        torch.cuda.empty_cache()
    else:
        raise RuntimeError("No PVA-Net inference window completed")
    torch.cuda.synchronize()
    pred,conf,v,a=[x[0].float().cpu().numpy() for x in (pred,conf,v,a)]
    native=[]
    good=[]
    size=tuple(batch["native_resolution"]) if "native_resolution" in batch else (1920,1088 if path.stem=="fs_axel_cam7" else 1080)
    for uv,cam,k in zip(pred,batch["camera25"],batch["K_rect"],strict=True):
        u,g=rectified_to_native(uv,cam,K_rect=k,native_size_wh=size)
        native.append(u)
        good.append(g)
    R=batch["R_w2c"]
    pts=batch["sample_pts"]
    tb=batch["source_time_base"]
    t=pts*float(Fraction(*tb))
    finite=lambda value:np.isfinite(value).all()
    if not all(map(finite,(conf,v,a))):
        raise ValueError("PVA returned nonfinite motion")
    atomic_npz(output,sample_pts=pts,source_seconds=t,nominal_seconds=batch["nominal_seconds"],
               source_time_base=tb,velocity_start_pts=pts[:-1],velocity_end_pts=pts[1:],
               acceleration_source_pts=np.column_stack((pts[:-2],pts[1:-1],pts[2:])),
               velocity_camera_mps=v,acceleration_camera_mps2=a,
               velocity_world_mps=np.einsum("nji,nkj->nki",R[:-1],v),
               acceleration_world_mps2=np.einsum("nji,nkj->nki",R[:-2],a),
               rectified_image_keypoints=pred,source_image_keypoints=np.array(native),
               source_keypoints_valid=np.array(good),heatmap_confidence=conf[...,0],
               camera_R_w2c=R,bbx_xyxy=batch["bbx_xyxy"],K_rect=batch["K_rect"],
               box_inferred=batch["box_inferred"],input_sha256=np.array(digest(path)),
               checkpoint_sha256=np.array(checkpoint_sha))
    report=dict(frames=len(t),velocity_rows=len(v),acceleration_rows=len(a),
         source_pts=pts[[0,-1]].tolist(),windows=window_info,source_input=str(path),
         source_input_sha256=digest(path),checkpoint_sha256=checkpoint_sha,library=str(source),
         runtime=torch.__version__,device=torch.cuda.get_device_name(),seconds=time.perf_counter()-start,
         peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
         native_time_mapping="nearest 30 Hz image clock; indices keep exact source PTS",
         dynamics="m/s and m/s^2 per upstream at nominal 30 Hz; first-frame camera orientation",
         output_sha256=digest(output))
    atomic_json(output.with_suffix(".json"),report)
    emit("pva_complete",**report)


if __name__=="__main__":
    p=ArgumentParser(description=__doc__)
    for k in ("input","source","checkpoint","output"):
        p.add_argument("--"+k,type=Path,required=True)
    p.add_argument("--window",type=int,default=120)
    p.add_argument("--overlap",type=int,default=30)
    opt=p.parse_args()
    opt.output.parent.mkdir(exist_ok=True,parents=True)
    infer_many([opt.input],opt.source,opt.checkpoint,opt.output,opt.window,opt.overlap)
