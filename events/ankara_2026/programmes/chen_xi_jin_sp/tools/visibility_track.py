"""Track a confirmed foreground mask through nearby original exposures."""
from __future__ import annotations

import argparse
import cv2
import numpy as np

from common import DATA, HEIGHT, OUTPUT, WIDTH, save, scope, sha, source_frames, source_pts, write


def track_order(model,processor,source_images,indices,seed):
    import torch
    from torch.nn.functional import interpolate

    session=processor.init_video_session(inference_device="cuda",inference_state_device="cuda",
            video_storage_device="cpu",max_vision_features_cache_size=1,dtype=torch.bfloat16)
    processor.add_inputs_to_inference_session(session,frame_idx=0,obj_ids=[1],input_masks=seed)
    mean=np.array([.485,.456,.406],np.float32)
    norm=np.array([.229,.224,.225],np.float32)
    for k,i in enumerate(indices):
        raw=source_images[i]
        scaled=cv2.resize(raw[:,:,::-1],(1024,1024),interpolation=cv2.INTER_LINEAR).astype(np.float32)/255.
        value=np.ascontiguousarray(((scaled-mean)/norm).transpose(2,0,1))
        tensor=torch.from_numpy(value).to(device="cuda",dtype=torch.bfloat16)
        with torch.inference_mode():
            result=model(inference_session=session,frame_idx=k,frame=tensor)
            mask=(interpolate(result.pred_masks.float(),(HEIGHT,WIDTH),
                               mode="bilinear",align_corners=False)[0,0]>0.).cpu().numpy()
        if k==0:
            mask=seed.astype(bool)
        yield i,mask
        before=k-max(model.config.num_maskmem+2,model.config.max_object_pointers_in_encoder+2)
        for index, outputs in session.output_dict_per_obj.items():
            old=outputs["non_cond_frame_outputs"]
            for j in list(old):
                if j<before:
                    del old[j]
            tracked=session.frames_tracked_per_obj[index]
            for j in list(tracked):
                if j<before:
                    del tracked[j]
        if session.processed_frames:
            session.processed_frames={k:session.processed_frames[k]}


def extend_dark_hems(frame,mask):
    """Recover visible dark dress at the image edge where it adjoins the matte."""
    y,x=np.nonzero(mask)
    if not len(x) or y.max()<HEIGHT-115:
        return mask
    left=max(0,x.min()-80)
    right=min(WIDTH,x.max()+80)
    top=max(0,int(np.quantile(y,.72))-8)
    stay=mask[top:,left:right].copy()
    luma=cv2.cvtColor(frame[top:,left:right],cv2.COLOR_BGR2GRAY)
    expanded=cv2.distanceTransform(np.ascontiguousarray(~stay,dtype=np.uint8),cv2.DIST_L2,5)<72.
    target=(luma<130)&expanded
    _,label=cv2.connectedComponents(np.ascontiguousarray(stay|target,dtype=np.uint8))
    active=np.unique(label[stay])
    active=active[active>0]
    mask[top:,left:right]|=np.isin(label,active)
    return mask


def validate_silhouette(mask,box,minimum):
    region=mask.nonzero()
    if not len(region[0]):
        return False
    area=len(region[0])
    limits=np.array([region[1].min(),region[0].min(),region[1].max()+1,region[0].max()+1])
    barea=np.prod(box[2:]-box[:2])
    intersection=np.maximum(0.,np.minimum(box[2:],limits[2:])-np.maximum(box[:2],limits[:2])).prod()
    union=barea+np.prod(limits[2:]-limits[:2])-intersection
    return minimum*barea < area < 1.04*barea and intersection/union>.52


def run(model,processor,record,shot,output,seed_path,index):
    pts=source_pts(shot)
    sid=shot["id"]
    with np.load(seed_path,allow_pickle=False) as a:
        assert a["source_sha256"].item()==record["source"]["sha256"]
        np.testing.assert_array_equal(pts,a["pts"])
        boxes=a["search_boxes"]
        direct=a["source_detection_direct"]
        assert a["valid"][index]
        seed=np.unpackbits(a["packed"][index],axis=1)
    frames=[frame.to_ndarray(format="bgr24") for _,frame in source_frames(record["source"]["registered_path"],pts)]
    packed=np.zeros((len(pts),HEIGHT,WIDTH//8),np.uint8)
    facts=[None]*len(pts)
    schedules=(range(index,len(pts)),range(index,-1,-1)) if index else (range(len(pts)),)
    for indices in schedules:
        for j,mask in track_order(model,processor,frames,indices,seed):
            if sid=="s20":
                mask=extend_dark_hems(frames[j],mask)
            valid=validate_silhouette(mask,boxes[j],.19 if sid=="s20" else .30)
            packed[j]=np.packbits(mask,axis=1)
            facts[j]=dict(pts=int(pts[j]),valid=bool(valid),area=int(mask.sum()))
            if j%20==0:
                print(sid,j,len(pts),facts[j],flush=True)
    valid=np.array([z["valid"] for z in facts],bool)
    save(output/f"{sid}.npz",pts=pts,packed=packed,valid=valid,
         source_sha256=record["source"]["sha256"],source_detection_direct=direct,search_boxes=boxes)
    write(output/f"{sid}.json",dict(shot=sid,frames=len(pts),valid=int(valid.sum()),records=facts,
          seed_pts=int(pts[index]),seed_packet=str(seed_path),seed_sha256=sha(seed_path)))
    print(sid,"verified bounding support",int(valid.sum()),"of",len(pts),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed",required=True)
    p.add_argument("--out",required=True)
    p.add_argument("--frames",nargs="+",required=True,help="shot:zero-based reviewed keyframe")
    a=p.parse_args()
    import torch
    from transformers import Sam2VideoModel, Sam2VideoProcessor

    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    base=DATA/"cache/pretrained/sam2.1-hiera-tiny"
    model=Sam2VideoModel.from_pretrained(str(base),local_files_only=True).eval().requires_grad_(False)
    model=model.to(device="cuda",dtype=torch.bfloat16)
    proc=Sam2VideoProcessor.from_pretrained(str(base),local_files_only=True)
    record=scope()
    folder=OUTPUT/a.out
    folder.mkdir(exist_ok=True)
    for token in a.frames:
        sid,frame=token.split(":")
        s=next(t for t in record["shots"] if t["id"]==sid)
        run(model,proc,record,s,folder,OUTPUT/a.seed/f"{sid}.npz",int(frame))


if __name__=="__main__":
    main()
