"""Local demo-only SAM2 resegmentation on exact official source PTS.

Historical masks, body observations, rink and cameras remain unchanged. This
presentation override is limited to the two user-reported severe losses.
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
DATA=Path("D:/CodexData/RinkSpace_Data/events/ankara_2026/programmes/okada_mei_sp")
OUT=DATA/"results/demo_full_20261016/v2"
PACKET=DATA/"results/demo_last_shot_20261016/source_packet.npz"
MODEL=ROOT.parent/"MotionTrail_Codex_Starter_v1/models/sam2.1-hiera-tiny"
YOLO_MODEL=ROOT.parent/"FSpathing/models/yolo11n-pose.pt"
WINDOWS=((1300224,1306880),(1456384,1469952))
FPS=50
WH=(1920,1080)

sys.path[:0]=[str(ROOT),str(ROOT/"src"),str(ROOT/"calibration/program_tracking")]
from calibration.program_tracking.runtime.bootstrap import add_geometry_dependencies
add_geometry_dependencies()
os.environ["YOLO_CONFIG_DIR"]=str(OUT/"yolo_runtime")
os.environ.setdefault("OMP_NUM_THREADS","1")

def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()

def write_json(path,data):
    pending=Path(path).with_suffix(".pending.json")
    pending.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf8")
    os.replace(pending,path)

def source_frames(source,targets):
    import av
    targets=set(map(int,targets))
    with av.open(str(source)) as c:
        s=c.streams.video[0]
        assert str(s.time_base)=="1/12800" and s.average_rate==50
        for first,last in WINDOWS:
            needed=sorted(t for t in targets if first<=t<=last)
            if not needed:continue
            c.seek(needed[0],stream=s,backward=True)
            for frame in c.decode(s):
                if frame.pts is None or frame.pts<needed[0]:continue
                if frame.pts>needed[-1]:break
                if int(frame.pts) in targets:
                    yield int(frame.pts),frame.to_ndarray(format="bgr24")

def sam_mask(processor,model,preprocess,bgr,box,points):
    import cv2,numpy as np,torch
    import torch.nn.functional as functional
    from PIL import Image
    height,width=bgr.shape[:2]
    session=processor.init_video_session(inference_device="cuda",inference_state_device="cuda",
        processing_device="cpu",video_storage_device="cpu",dtype=torch.bfloat16)
    processor.add_inputs_to_inference_session(session,frame_idx=0,obj_ids=[1],
        input_boxes=[[box]],input_points=[[points]],input_labels=[[[1]*len(points)]],
        original_size=(height,width))
    size=(preprocess["size"]["width"],preprocess["size"]["height"])
    rgb=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
    values=np.asarray(rgb.resize(size,resample=Image.Resampling.BILINEAR),np.float32)
    values*=float(preprocess["rescale_factor"])
    values=(values-np.asarray(preprocess["image_mean"],np.float32))/np.asarray(preprocess["image_std"],np.float32)
    tensor=torch.from_numpy(np.ascontiguousarray(values.transpose(2,0,1))).to("cuda")
    with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
        result=model(inference_session=session,frame=tensor,frame_idx=0)
        full=functional.interpolate(result.pred_masks.float(),(height,width),mode="bilinear",align_corners=False)>0
    return full.detach().cpu().numpy().reshape(-1,height,width).any(axis=0).astype(np.uint8)

def find_box_and_points(detector,bgr):
    import numpy as np
    result=detector.predict(bgr,verbose=False,device=0,conf=.20,imgsz=1280)[0]
    boxes=result.boxes.xyxy.detach().cpu().numpy()
    conf=result.boxes.conf.detach().cpu().numpy()
    if not len(boxes):raise ValueError("No person at current source PTS")
    heights=boxes[:,3]-boxes[:,1]
    areas=(boxes[:,2]-boxes[:,0])*heights
    eligible=np.flatnonzero(heights>=140)
    if not len(eligible):raise ValueError("No foreground athlete candidate")
    chosen=int(eligible[np.argmax(areas[eligible]*conf[eligible])])
    raw_box=boxes[chosen].astype(float)
    box=raw_box+np.array([-12,-12,12,12])
    box[:2]=np.maximum(box[:2],[0,0]);box[2:]=np.minimum(box[2:],WH)
    xy=result.keypoints.xy[chosen].detach().cpu().numpy()
    quality=result.keypoints.conf[chosen].detach().cpu().numpy()
    # Reliable torso and source-native pose points are SAM search prompts,
    # never accepted anatomical ground truth or ice contact observations.
    torso_ids=[i for i in (5,6,11,12) if quality[i]>=.40]
    if len(torso_ids)<2:raise ValueError("Current athlete torso is not located")
    points=[xy[torso_ids].mean(axis=0)]
    for i in (0,5,6,7,8,9,10,11,12,13,14,15,16):
        point=xy[i]
        if quality[i]<.72:continue
        if not (box[0]<=point[0]<=box[2] and box[1]<=point[1]<=box[3]):continue
        if all(np.linalg.norm(point-p)>18 for p in points):points.append(point)
    return box.tolist(),np.asarray(points[:1]).astype(float).tolist(),dict(
        detector_raw_box_xyxy=raw_box.tolist(),detector_confidence=float(conf[chosen]),
        detected_person_count=len(boxes),selected_person_index=chosen,
        keypoints_xy=xy.tolist(),keypoints_confidence=quality.tolist())

def qa(source,targets,packed,rows,out):
    import cv2,numpy as np
    from calibration.program_tracking.render_demo_last_shot import Sample
    sample=Sample("v2",contact_path=OUT.parent/"contact_packet.npz")
    lookup={int(p):i for i,p in enumerate(targets)}
    chosen=[1300224,1301248,1306880,1456384,1461760,1469952]
    tiles=[]
    for pts,bgr in source_frames(source,[p for p in chosen if p in lookup]):
        row=lookup[pts]
        new=cv2.resize(np.unpackbits(packed[row],axis=1)[:,:960],WH,interpolation=cv2.INTER_NEAREST)>0
        old_i=sample.lookup[pts];legacy=int(sample.packet["legacy_mask_pts"][old_i])
        sample.masks.get(legacy,True)
        _,mrow,_=sample.masks.lookup[legacy]
        old=cv2.resize(np.unpackbits(sample.masks.active["masks"][mrow],axis=1)[:,:960],WH,interpolation=cv2.INTER_NEAREST)>0
        pair=[]
        for title,mask in [("old",old),("native SAM2",new)]:
            image=bgr.copy()
            image[mask]=image[mask]*.50+np.array([25,40,230])*.50
            contours,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image,contours,-1,(0,255,255),2)
            tile=cv2.resize(image,(960,540))
            cv2.putText(tile,f"source {pts/12800:.2f}s {title}",(12,27),0,.65,(30,30,230),2,cv2.LINE_AA)
            pair.append(tile)
        tiles.append(np.hstack(pair))
    for j in range(0,len(tiles),3):
        cv2.imwrite(str(out/f"mask_repair_qa_{j//3+1}.jpg"),np.vstack(tiles[j:j+3]),
                    [cv2.IMWRITE_JPEG_QUALITY,94])

def build(samples=False):
    import cv2,numpy as np,torch
    cv2.setNumThreads(1);torch.set_num_threads(2)
    sys.path.append(str(ROOT.parent/"FSpathing/.venv/Lib/site-packages"))
    from ultralytics import YOLO
    from transformers import Sam2VideoModel,Sam2VideoProcessor
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"yolo_runtime").mkdir(parents=True,exist_ok=True)
    with np.load(PACKET,allow_pickle=False) as p:
        source=Path(str(p["source_path"]))
        source_pts=p["pts"].copy()
    target=np.concatenate([np.arange(a,b+256,256,dtype=np.int64) for a,b in WINDOWS])
    assert len(target)==81 and np.isin(target,source_pts).all()
    if samples:
        target=np.array([1300224,1301248,1306880,1456384,1461760,1469952],np.int64)
    detector=YOLO(str(YOLO_MODEL)).to("cuda")
    model,loading=Sam2VideoModel.from_pretrained(str(MODEL),local_files_only=True,output_loading_info=True)
    errors={k:loading[k] for k in ("missing_keys","unexpected_keys","mismatched_keys","error_msgs") if loading.get(k)}
    if errors:raise ValueError(errors)
    model.requires_grad_(False).eval().to("cuda",dtype=torch.bfloat16)
    processor=Sam2VideoProcessor.from_pretrained(str(MODEL),local_files_only=True)
    preprocess=json.loads((MODEL/"preprocessor_config.json").read_text(encoding="utf8"))
    rows=[];masks=[];started=time.perf_counter()
    for pts,bgr in source_frames(source,target):
        box,points,details=find_box_and_points(detector,bgr)
        mask=sam_mask(processor,model,preprocess,bgr,box,points)
        half=cv2.resize(mask,(960,540),interpolation=cv2.INTER_NEAREST)
        if half.sum()<100:raise ValueError(f"Empty segmentation {pts}")
        area=int(mask.sum());boxarea=(box[2]-box[0])*(box[3]-box[1])
        if not .03<area/boxarea<1.6:raise ValueError(f"Implausible mask {pts}: {area/boxarea}")
        masks.append(np.packbits(half,axis=1))
        rows.append(dict(pts=pts,source_seconds=pts/12800.,demo_frame_1based=round((pts/12800.-20.02)*50)+411,
            status="native_sam2_resegmented_inferred",source_bgr_sha256=hashlib.sha256(bgr.tobytes()).hexdigest(),
            prompt_box_xyxy=box,prompt_points_uv=points,mask_area_native_px=area,
            halfres_foreground_px=int(half.sum()),**details))
        write_json(OUT/"mask_progress.json",dict(frames=len(rows),total=len(target),last_pts=pts,
            elapsed_seconds=time.perf_counter()-started))
    assert [r["pts"] for r in rows]==target.tolist()
    packed=np.stack(masks).astype(np.uint8)
    out=OUT/"mask_samples.npz" if samples else OUT/"foreground_masks.npz"
    np.savez_compressed(out,pts=target,masks=packed,wh=np.array([960,540],np.int32),
        shape=np.array([540,960],np.int32),status=np.array([r["status"] for r in rows]),
        source_time_base=np.array("1/12800"),source_path=np.array(str(source)))
    receipt=dict(schema="rinkspace.demo-native-foreground-repair.v1",output=str(out),sha256=sha(out),
        source_packet=str(PACKET),source_packet_sha256=sha(PACKET),
        source_path=str(source),source_sha256=sha(source),source_time_base="1/12800",native_fps=50,
        requested_demo_frames_1based=[[4489,4515],[5099,5152]],
        native_windows_pts=[list(v) for v in WINDOWS],source_windows_seconds=[[a/12800.,b/12800.] for a,b in WINDOWS],
        frames=len(rows),masks_shape=list(packed.shape),mask_encoding="np.packbits axis=1 width960; no dilation; nearest half-resolution sampling",
        status="native_sam2_resegmented_inferred",
        method="Per-frame native-source YOLO11n-pose foreground box plus torso search prompt, fresh SAM2 decoder session every source frame",
        crosses_shot_cut_at_pts=1301248,temporal_memory_reused=False,
        original_masks_and_body_and_camera_unchanged=True,all_other_pts_unchanged=True,
        human_pixel_ground_truth=False,model_path=str(MODEL),model_sha256=sha(MODEL/"model.safetensors"),
        detector_path=str(YOLO_MODEL),detector_sha256=sha(YOLO_MODEL),rows=rows,
        code=str(Path(__file__)),code_sha256=sha(__file__),elapsed_seconds=time.perf_counter()-started)
    write_json(OUT/("mask_samples.json" if samples else "foreground_masks.json"),receipt)
    qa(source,target,packed,rows,OUT)
    print(json.dumps({k:receipt[k] for k in ("output","sha256","frames","elapsed_seconds")}),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--samples",action="store_true")
    args=p.parse_args()
    build(args.samples)
