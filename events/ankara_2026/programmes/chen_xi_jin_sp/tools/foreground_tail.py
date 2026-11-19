"""Delineate the closing shot's skater where a box covers usable board print."""
from __future__ import annotations

import json
import sys

import cv2
import numpy as np
import torch

from common import DATA, HEIGHT, OUTPUT, ROOT, WIDTH, read, save, scope, source_frames, source_pts


def main():
    sys.path.insert(0,str(ROOT))
    from calibration.program_tracking.demo_mask_repair import sam_mask
    from transformers import Sam2VideoModel, Sam2VideoProcessor

    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    model_path=DATA/"cache/pretrained/sam2.1-hiera-tiny"
    processor=Sam2VideoProcessor.from_pretrained(model_path,local_files_only=True)
    model=Sam2VideoModel.from_pretrained(model_path,local_files_only=True).to("cuda").eval()
    preprocess=read(model_path/"video_preprocessor_config.json")
    record=scope()
    shot=record["shots"][-1]
    pts=source_pts(shot)
    with np.load(OUTPUT/"foreground/s20.npz",allow_pickle=False) as p:
        boxes=p["boxes"]; offsets=p["offsets"]
        np.testing.assert_array_equal(pts,p["pts"])
    result=[]
    for i,(p,frame) in enumerate(source_frames(record["source"]["registered_path"],pts)):
        bb=boxes[offsets[i]:offsets[i+1]]
        extent=np.prod(bb[:,2:4]-bb[:,:2],axis=1)
        chosen=bb[int(np.argmax(extent*bb[:,4]))][:4]
        xyxy=np.clip(chosen+[-12,-12,12,12],0.,[WIDTH,HEIGHT,WIDTH,HEIGHT])
        x0,y0,x1,y1=chosen
        prompt=[[float((x0+x1)*.5),float(y0+(y1-y0)*.60)]]
        mask=sam_mask(processor,model,preprocess,frame.to_ndarray(format="bgr24"),xyxy.tolist(),prompt)
        if not 15000<np.count_nonzero(mask)<800000:
            raise ValueError(f"Invalid closing-shot silhouette at {p}: {mask.sum()}")
        result.append(np.packbits(mask,axis=1))
        if i%15==0:
            print(p, int(mask.sum()), flush=True)
    save(OUTPUT/"foreground/s20_sam.npz",pts=pts,packed=np.stack(result),
         source_sha256=record["source"]["sha256"])
    print("Full closing shot masks",len(pts),flush=True)


if __name__=="__main__":
    main()
