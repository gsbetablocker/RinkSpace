"""Read the foreground skater silhouette from every original source exposure."""
from __future__ import annotations

import argparse
from itertools import islice
import os
import sys
from time import monotonic

import cv2
import numpy as np

from common import DATA, HEIGHT, JOINT, OUTPUT, WIDTH, frozen_scene, save, scope, sha, source_frames, source_pts, write
from measure import DT, FIRST, TOP, load_camera, motion_clock


def athlete_boxes(record, shot, guide, native, course, inward):
    pid = shot["id"]
    pts, controls, rig = load_camera(shot, guide)
    t = motion_clock(shot,guide,pts)
    views = native.core.evaluate_motion_camera(rig,controls,FIRST,DT,t)
    assert np.all(views["status"]==0)
    with np.load(OUTPUT/"foreground"/f"{pid}.npz",allow_pickle=False) as source:
        np.testing.assert_array_equal(source["pts"],pts)
        assert source["source_sha256"].item() == record["source"]["sha256"]
        box, offset = source["boxes"],source["offsets"]
    usable=[]
    strengths=[]
    for i in range(len(pts)):
        b=box[offset[i]:offset[i+1]]
        height=b[:,3]-b[:,1]
        width=b[:,2]-b[:,0]
        choose=(height>=62.)&(width>=22.)
        b=b[choose]
        if not len(b):
            usable.append(np.empty((0,4)))
            strengths.append(np.empty(0))
            continue
        center=.5*(b[:,:2]+b[:,2:4])
        prod=(b[:,2]-b[:,0])*(b[:,3]-b[:,1])
        q=native.core.project_motion(rig,controls,FIRST,DT,HEIGHT,t[i],course)
        face=np.sum((course[:,:2]-views["centre"][i,:2])*inward,axis=1)<=0.
        uv=q["uv"][(q["status"]==0)&face]
        foot=b[:,3]
        near=np.abs(uv[:,None,0]-center[None,:,0]) if len(uv) else np.empty((0,len(b)))
        if len(near):
            nearest=np.argmin(near,axis=0)
            distance=near[nearest,np.arange(len(b))]
            reference=uv[nearest,1]
            penalty=np.where(distance<105.,np.clip((reference-foot-35.)/26.,0.,5.),0.)
        else:
            penalty=np.zeros(len(b))
        center_bonus=(np.abs(center[:,0]-.5*WIDTH)/550.)**1.5
        strength=2. + .65*np.log(prod/(65.*105.)) + np.log(np.maximum(.1,b[:,4])) - center_bonus - penalty
        usable.append(b[:,:4].astype(np.float64))
        strengths.append(strength)
    paths=[]
    last_box=None
    last_score=None
    for b, strength in zip(usable,strengths,strict=True):
        # Include an empty detection for frames where the model sees only
        # spectators. Missing boxes can guide a new read from adjacent prompts.
        options=np.r_[strength,-2.5]
        if last_box is None:
            score=options
            back=np.zeros(len(options),np.int64)
        else:
            d=np.zeros((len(last_box)+1,len(b)+1))
            if len(b) and len(last_box):
                lp=.5*(last_box[:,:2]+last_box[:,2:])
                np_= .5*(b[:,:2]+b[:,2:])
                scale=np.sqrt(np.prod(last_box[:,2:]-last_box[:,:2],axis=1))
                size=np.sqrt(np.prod(b[:,2:]-b[:,:2],axis=1))
                move=np.linalg.norm(lp[:,None]-np_[None,:],axis=2)
                d[:-1,:-1]=np.minimum(12.,3.7*move/(38.+.25*scale[:,None])+
                                      3.2*np.abs(np.log(size[None,:]/scale[:,None])))
            d[-1,:-1]=1.7
            d[:-1,-1]=1.7
            previous=last_score[:,None]-d
            back=np.argmax(previous,axis=0)
            score=options+previous[back,np.arange(len(options))]
        paths.append(back)
        last_box,last_score=b,score-score.max()
    track=np.full((len(usable),4),np.nan)
    k=int(np.argmax(last_score))
    for i in range(len(usable)-1,-1,-1):
        if k<len(usable[i]):
            track[i]=usable[i][k]
        k=paths[i][k]
    direct=np.isfinite(track[:,0])
    found=np.flatnonzero(direct)
    if not len(found):
        raise ValueError("No plausible athlete in "+pid)
    for axis in range(4):
        track[:,axis]=np.interp(np.arange(len(track)),found,track[found,axis])
    return track,direct


class Silhouette:
    def __init__(self):
        import torch
        from safetensors.torch import load_file
        from transformers import Sam2VideoConfig

        self.torch=torch
        torch.set_num_threads(2)
        torch.set_float32_matmul_precision("high")
        self.path=DATA/"cache/pretrained/sam2.1-hiera-tiny"
        self.config=Sam2VideoConfig.from_pretrained(str(self.path),local_files_only=True)
        self.weights=load_file(str(self.path/"model.safetensors"))
        self.mean=np.array([.485,.456,.406],np.float32).reshape(1,3,1,1)
        self.std=np.array([.229,.224,.225],np.float32).reshape(1,3,1,1)
        self.models={}

    def model(self,side):
        from transformers import Sam2Config, Sam2Model
        if side not in self.models:
            c=self.config
            config=Sam2Config(vision_config=c.vision_config.to_dict(),mask_decoder_config=c.mask_decoder_config.to_dict(),
                              prompt_encoder_config=c.prompt_encoder_config.to_dict())
            config.vision_config.backbone_feature_sizes=[[side//i,side//i] for i in (4,8,16)]
            config.prompt_encoder_config.image_size=side
            m=Sam2Model(config).eval().requires_grad_(False)
            names=set(m.state_dict())
            m.load_state_dict({k:v for k,v in self.weights.items() if k in names},strict=True)
            self.models[side]=m.to(device="cuda",dtype=self.torch.bfloat16)
        return self.models[side]

    def masks(self,images,boxes,side,increase=.32,points=None):
        torch=self.torch
        inputs=[]
        prompts=[]
        sam_points=[]
        sam_labels=[]
        bounds=[]
        for i,(im,b) in enumerate(zip(images,boxes,strict=True)):
            mid=.5*(b[:2]+b[2:])
            width=max(b[2:]-b[:2])
            edge=max(90,int(np.ceil(width*(1.+increase)+14.)))
            x0,y0=np.floor(mid-.5*edge).astype(int)
            left,top=max(0,x0),max(0,y0)
            right,bottom=min(WIDTH,x0+edge),min(HEIGHT,y0+edge)
            crop=im[top:bottom,left:right]
            # Replicate only context beyond the native sensor. The output
            # silhouette is always clipped back to the true source domain.
            rgb=cv2.copyMakeBorder(crop,top-y0,y0+edge-bottom,left-x0,x0+edge-right,cv2.BORDER_REPLICATE)
            rgb=cv2.resize(rgb,(side,side),interpolation=cv2.INTER_AREA if edge>=side else cv2.INTER_LINEAR)
            inputs.append(rgb[:,:,::-1].transpose(2,0,1))
            prompts.append(np.clip((b-[x0,y0,x0,y0])*(side/edge),0.,side-1.))
            bounds.append((x0,y0,edge,left,top,right,bottom))
            if points is not None:
                xy=points[i]
                n=min(8,len(xy))
                scaled=np.zeros((8,2),np.float32)
                scaled[:n]=np.clip((xy[:n]-[x0,y0])*(side/edge),0.,side-1.)
                sam_points.append(scaled)
                sam_labels.append(np.r_[np.ones(n,np.int64),np.full(8-n,-10,np.int64)])
        color=np.array(inputs,dtype=np.float32)*(1./255.)
        tensors=torch.from_numpy(np.ascontiguousarray((color-self.mean)/self.std)).to(device="cuda",dtype=torch.bfloat16)
        # Keep prompt coordinates in float32 even at 1024 input resolution.
        prompt=torch.from_numpy(np.array(prompts,dtype=np.float32)[:,None,:]).cuda()
        keypoints={}
        if points is not None:
            keypoints=dict(input_points=torch.from_numpy(np.asarray(sam_points)[:,None]).cuda(),
                           input_labels=torch.from_numpy(np.asarray(sam_labels)[:,None]).cuda())
        model=self.model(side)
        with torch.inference_mode():
            prediction=model(pixel_values=tensors,input_boxes=prompt,multimask_output=True,**keypoints)
            masks=prediction.pred_masks[:,0].float().cpu().numpy()
            score=prediction.iou_scores[:,0].float().cpu().numpy()
        hints=points if points is not None else [np.empty((0,2)) for _ in boxes]
        return [choose_mask(arr,iou,region,box,hint)
                for arr,iou,region,box,hint in zip(masks,score,bounds,boxes,hints,strict=True)]


def choose_mask(logits,quality,region,box,hints):
    x,y,side,left,top,right,bottom=region
    mask=np.zeros((HEIGHT,WIDTH),np.uint8)
    area=np.prod(box[2:]-box[:2])
    choices=[]
    for score,prob in zip(quality,logits,strict=True):
        bound=cv2.resize(prob,(side,side),interpolation=cv2.INTER_LINEAR)[top-y:bottom-y,left-x:right-x]>0.
        count=int(bound.sum())
        if count<.12*area or count>1.20*area:
            continue
        rows=np.flatnonzero(bound.any(axis=1))+top
        columns=np.flatnonzero(bound.any(axis=0))+left
        limits=np.array([columns[0],rows[0],columns[-1]+1,rows[-1]+1])
        common=np.maximum(0.,np.minimum(box[2:],limits[2:])-np.maximum(box[:2],limits[:2]))
        union=area+np.prod(limits[2:]-limits[:2])-np.prod(common)
        iou=np.prod(common)/union
        if iou<.47 or score<.48:
            continue
        hits=np.rint(hints).astype(int)-[left,top]
        inside=((hits>=0)&(hits<[right-left,bottom-top])).all(axis=1)
        seen=bound[hits[inside,1],hits[inside,0]]
        support=float(seen.mean()) if len(seen) else 1.
        if len(seen) and (not seen[0] or support < .56):
            continue
        choices.append((.52*score+.31*iou+.17*support,bound,int(count),limits.tolist(),float(score),
                        float(iou),support))
    if not choices:
        return mask,None
    _,bound,count,limits,score,iou,support=max(choices,key=lambda z:z[0])
    # The foreground is an opaque silhouette. Fill enclosed holes in the
    # model output only when tiny relative to the person.
    n,labels,stats,_=cv2.connectedComponentsWithStats(np.ascontiguousarray(~bound,dtype=np.uint8))
    for label in range(1,n):
        ux,uy,w,h,size=stats[label]
        touches=ux==0 or uy==0 or ux+w==bound.shape[1] or uy+h==bound.shape[0]
        if not touches and size<max(26,.007*count):
            bound[labels==label]=True
    mask[top:bottom,left:right]=bound
    return mask,dict(area=int(bound.sum()),box=limits,model_score=score,box_iou=iou,source_prompt_support=support)


def pose_hints(detector,raw,boxes,closeup):
    results=detector.predict(raw,device=0,imgsz=960 if closeup else 640,half=True,conf=.19,
                             max_det=12,verbose=False)
    proposals=[]
    use_pairs=((5,6),(11,12),(5,11),(6,12))
    for sample,box in zip(results,boxes,strict=True):
        b=sample.boxes.xyxy.cpu().numpy()
        if len(b):
            intersect=np.maximum(0.,np.minimum(b[:,2:],box[2:])-np.maximum(b[:,:2],box[:2])).prod(1)
            total=np.prod(b[:,2:]-b[:,:2],axis=1)+np.prod(box[2:]-box[:2])-intersect
            overlap=intersect/total
            person=int(np.argmax(overlap))
        else:
            overlap=np.array([0.])
            person=0
        if overlap[person] < .50:
            proposals.append(np.empty((0,2)))
            continue
        p=sample.keypoints.xy[person].cpu().numpy()
        conf=sample.keypoints.conf[person].cpu().numpy()
        extent=box[2:]-box[:2]
        good=(conf>=.55)&((p>=box[:2]-3.)&(p<=box[2:]+3.)).all(1)
        torso=[j for j in (5,6,11,12) if good[j]]
        keep=[]
        if len(torso)>=3:
            keep.append(np.mean(p[torso],axis=0))
        elif len(torso)==2 and torso in ([5,11],[6,12]):
            keep.append(np.mean(p[torso],axis=0))
        elif closeup:
            keep.append(box[:2]+[.52*extent[0],.63*extent[1]])
        for i,j in use_pairs:
            if good[i] and good[j]:
                keep.append(.5*(p[i]+p[j]))
        for j in (0,7,8,9,10,13,14,15,16):
            if good[j] and conf[j]>=.75:
                keep.append(p[j])
        selected=[]
        for point in keep:
            if (point>=[2,2]).all() and (point<[WIDTH-2,HEIGHT-2]).all():
                if all(np.linalg.norm(point-x)>max(6.,.041*extent[1]) for x in selected):
                    selected.append(point)
        proposals.append(np.asarray(selected[:8],np.float32).reshape(-1,2))
    return proposals


def run_shot(record,shot,guide,folder,segmenter,native,course,inward,batch,detector=None):
    sid=shot["id"]
    pts=source_pts(shot)
    boxes,direct=athlete_boxes(record,shot,guide,native,course,inward)
    source=source_frames(record["source"]["registered_path"],pts)
    started=monotonic()
    records=[]
    pack=np.zeros((len(pts),HEIGHT,WIDTH//8),np.uint8)
    i=0
    while (block:=list(islice(source,batch))):
        raw=[frame.to_ndarray(format="bgr24") for _,frame in block]
        ends=i+len(block)
        prompts=boxes[i:ends]
        large=(sid in {"s01","s20"} or np.max(prompts[:,3]-prompts[:,1])>=720.)
        hints=pose_hints(detector,raw,prompts,large) if detector is not None else None
        masks=segmenter.masks(raw,prompts,1024 if large else 512,points=hints)
        missing=[j for j,(_,fact) in enumerate(masks) if fact is None]
        if missing:
            retry=[hints[j][:1] for j in missing] if hints is not None else None
            retried=segmenter.masks([raw[j] for j in missing],prompts[missing],1024,increase=.60,points=retry)
            for j,replacement in zip(missing,retried,strict=True):
                masks[j]=replacement
        for j,((p,_),(mask,fact)) in enumerate(zip(block,masks,strict=True)):
            assert p==pts[i+j]
            row=dict(pts=int(p),source_detector_box=bool(direct[i+j]))
            if fact is not None:
                pack[i+j]=np.packbits(mask,axis=1)
                row.update(fact)
            records.append(row)
        i=ends
        if i%240==0:
            print(sid,i,"of",len(pts),"valid",sum("area" in x for x in records),f"{monotonic()-started:.1f}s",flush=True)
    assert len(records)==len(pts)
    valid=np.array(["area" in x for x in records],bool)
    save(folder/f"{sid}.npz",pts=pts,packed=pack,source_sha256=record["source"]["sha256"],
         valid=valid,source_detection_direct=direct,search_boxes=boxes)
    write(folder/f"{sid}.json",dict(shot=sid,verified_current_detection_boxes=int(direct.sum()),
                                   source_pose_prompts=detector is not None,
                                   model_sha256=sha(segmenter.path/"model.safetensors"),
                                   frames=len(pts),valid=int(valid.sum()),records=records))
    print(sid,"silhouettes",int(valid.sum()),"/",len(pts),"source boxes",int(direct.sum()),f"{monotonic()-started:.1f}s",flush=True)


def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument("--guide",default="steady08")
    a.add_argument("--out",required=True)
    a.add_argument("--batch",type=int,default=8)
    a.add_argument("--shots",nargs="+")
    a.add_argument("--pose-prompts",action="store_true")
    args=a.parse_args()
    cv2.setNumThreads(2)
    folder=OUTPUT/args.out
    folder.mkdir(exist_ok=True)
    record=scope()
    native=frozen_scene(record,verify=True)
    sys.path.insert(0,str(JOINT))
    from joint_inputs import native_paths
    course=native_paths(native,[TOP],count=141)[TOP][0]
    along=np.gradient(course[:,:2],axis=0)
    inward=np.c_[along[:,1],-along[:,0]]
    inward*=np.where((inward*(-course[:,:2])).sum(1)<0,-1.,1.)[:,None]
    model=Silhouette()
    detector=None
    if args.pose_prompts:
        os.environ["YOLO_CONFIG_DIR"]=str(folder/"runtime")
        from ultralytics import YOLO
        detector=YOLO(str(DATA/"cache/pretrained/yolo11n-pose.pt"))
    shots=sorted(record["shots"],key=lambda q:(q["id"] not in ("s01","s20"),q["id"]))
    for s in shots:
        k=s["id"]
        if (not args.shots or k in args.shots) and not (folder/f"{k}.json").is_file():
            run_shot(record,s,OUTPUT/args.guide,folder,model,native,course,inward,args.batch,detector)


if __name__=="__main__":
    main()
