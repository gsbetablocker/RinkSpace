"""Independent current-source person masks for remaining official first shot.

Reuse local YOLO proposals and SAM2. Boxes and pose points only navigate the
person search; fitting features always use current native RGB outside masks.
"""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/"src")]
from calibration.program_tracking.runtime.bootstrap import add_geometry_dependencies
add_geometry_dependencies()
import cv2
import numpy as np
import torch
from rinkspace.vision_instances import Sam2SourceInstances,SourcePersonBoxes
from rinkspace.programme_source_stream import file_sha
from calibration.program_tracking.infer_current_person_exclusion_masks import DEFAULT_MODEL
from calibration.program_tracking.infer_continuation_person_exclusion import source_materials,current_point
from calibration.program_tracking.prepare_first_shot_review import DATA,OUT,read,write,navigation,PRIORITIES

YOLO_WEIGHTS=ROOT / "models/cache/yolo11n-pose.pt"
YOLO_SITE=ROOT / ".venv-gpu/Lib/site-packages"
ANCHORS={201:[1100,620],250:[870,620],300:[960,590],400:[1020,490],500:[1010,570],600:[800,580],800:[720,650],1000:[890,540],1200:[1140,540],1248:[790,510]}
# Verified in the current native exposure, not coordinates imported from a
# neighbouring mask: these two raised white boots were locally omitted by SAM.
LOCAL_BOOT_PROMPTS={index:dict(box_xyxy=[839.,672.,910.,773.],
    points_uv=[[881.,710.],[855.,750.],[856.,685.],[826.,720.],[920.,720.],[870.,786.]],
    point_labels=[1,1,1,0,0,0],conservative_native_scope_xyxy=[833,663,917,780]) for index in (1005,1006)}


def expected_body(index):
    keys=sorted(ANCHORS);left=max(k for k in keys if k<=index);right=min(k for k in keys if k>=index)
    t=0 if left==right else (index-left)/(right-left)
    return np.array(ANCHORS[left])*(1-t)+np.array(ANCHORS[right])*t


def prompt_for_current(detector,rgb,binding,index):
    boxes=detector.read(rgb,binding)
    options=[r for r in boxes if r["box_xyxy"][3]>580 and r["box_xyxy"][3]-r["box_xyxy"][1]>130]
    target=expected_body(index)
    dark,skin,hsv,spread=source_materials(rgb)
    if options:
        # Native current detector distinguishes the foreground skating person
        # from the crowd standing behind the upper board. Manual landmarks
        # navigate only the rare multiple-foreground association.
        def score(r):
            box=np.array(r["box_xyxy"]);centre=(box[:2]+box[2:])/2
            return np.linalg.norm((centre-target)/[450,450])-(box[3]-box[1])/1200
        selected=min(options,key=score);box=np.array(selected["box_xyxy"],float)
        source_box=box.copy()
        box[:2]=np.maximum([0,0],box[:2]-[18,18]);box[2:]=np.minimum([1920,1080],box[2:]+[18,18])
        target=[float((source_box[0]+source_box[2])/2),float(source_box[1]+.58*(source_box[3]-source_box[1]))]
        proposal=dict(origin="current_original_RGB_local_YOLO_person_box",box_xyxy=source_box.tolist(),provenance=selected["provenance"])
    else:
        body=current_point(dark,target,180)
        box=np.array([max(0,body[0]-260),max(0,body[1]-400),min(1920,body[0]+260),min(1080,body[1]+500)],float)
        proposal=dict(origin="reviewed_source_landmark_navigation_then_current_RGB_dark_material",local_YOLO_foreground_box_missing=True)
    scoped=np.zeros_like(dark);x0,y0,x1,y1=np.rint(box).astype(int);scoped[max(0,y0):min(1080,y1),max(0,x0):min(1920,x1)]=dark[max(0,y0):min(1080,y1),max(0,x0):min(1920,x1)]
    body=current_point(scoped,target,170)
    positives=[body]
    # A current person box and actual dark torso point suffice for SAM2. Dark
    # material extremes can lie on the blurred upper board rail; they are not
    # admissible positive person prompts merely because they touch the body.
    negatives=[]
    for x,y in [(box[0]-30,(box[1]+box[3])/2),(box[2]+30,(box[1]+box[3])/2),((box[0]+box[2])/2,box[1]-30)]:
        if 0<=x<1920 and 0<=y<1080:negatives.append([float(x),float(y)])
    prompt=dict(object_id=1,kind="occluder",origin="original_rgb_candidate",source_binding=binding,box_xyxy=box.tolist(),
        points_uv=positives+negatives,point_labels=[1]*len(positives)+[0]*len(negatives),exclusion_authority=None)
    return prompt,body,proposal,dark,skin,hsv,spread


def mask_one(provider,detector,row,force_box=False):
    index=int(row["index"]);raw=cv2.imread(row["image_path"]);rgb=cv2.cvtColor(raw,cv2.COLOR_BGR2RGB)
    if file_sha(row["image_path"])!=row["png_sha256"] or hashlib.sha256(rgb.tobytes()).hexdigest()!=row["decoded_native_RGB_sha256"]:
        raise ValueError("Authenticated current source changed")
    binding=dict(source_sha256=row["source"]["source_video_sha256"],source_pts=row["native_pts"],time_base="1/12800",native_hw=[1080,1920],
        shot_id="official50_first_shot_current_person",decoded_native_rgb_sha256=row["decoded_native_RGB_sha256"])
    prompt,body,proposal,dark,skin,hsv,spread=prompt_for_current(detector,rgb,binding,index)
    result=provider.read(rgb,binding,prompts=[prompt],reset=True)
    candidate=result["occluder_region_candidate_mask"].detach().cpu().numpy()
    encoded_prompts=[prompt]
    # A strongly inclined pose can put the box-centre dark proposal on the
    # background. Re-encode the same current RGB with its current detector box
    # alone, then locate the actual foreground dark material inside the result.
    ys,xs=np.nonzero(candidate)
    box_height=prompt["box_xyxy"][3]-prompt["box_xyxy"][1]
    candidate_extent=float(ys.max()-ys.min()+1)/box_height if len(ys) else 0.
    box_reencoding=force_box or not candidate[int(body[1]),int(body[0])] or candidate_extent<.72
    if box_reencoding:
        box_prompt=dict(prompt);box_prompt.pop("points_uv",None);box_prompt.pop("point_labels",None)
        box_result=provider.read(rgb,binding,prompts=[box_prompt],reset=True)
        candidate=box_result["occluder_region_candidate_mask"].detach().cpu().numpy()
        encoded_prompts.append(box_prompt)
        nc,lc,sc,_=cv2.connectedComponentsWithStats(candidate.astype(np.uint8),8)
        if nc>1:
            dominant=max(range(1,nc),key=lambda label:sc[label,cv2.CC_STAT_AREA])
            candidate=lc==dominant
        support=candidate&dark
        if support.any():
            yy,xx=np.nonzero(support);points=np.c_[xx,yy]
            point=points[np.argmin(np.sum((points-np.array(body))**2,axis=1))]
            body=point.astype(float).tolist()
    mask=candidate.astype(np.uint8)*255
    # Repair current foreground material omitted by SAM's translucent-sleeve
    # preference only where it is attached to this current native candidate.
    x0,y0,x1,y1=np.rint(prompt["box_xyxy"]).astype(int)
    local=np.zeros_like(mask);local[y0:y1,x0:x1]=255
    material=(dark|skin)&(local>0)
    nearby=cv2.dilate(mask,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(21,21)))>0
    addition=(material&nearby&(mask==0)).astype(np.uint8)*255
    added=[int((addition>0).sum())]
    mask[addition>0]=255
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9)))
    n,labels,stats,_=cv2.connectedComponentsWithStats((mask>0).astype(np.uint8),8)
    foreground=int(labels[int(body[1]),int(body[0])])
    if not foreground:raise ValueError(f"Current SAM2 body omitted {index}")
    mask=(labels==foreground).astype(np.uint8)*255
    local_boot=LOCAL_BOOT_PROMPTS.get(index)
    boot_pixels=0
    if local_boot:
        # The current white skate has a different material from the torso.
        # Encode its reviewed native region separately rather than substituting
        # a neighbouring frame or expanding a detector box into the background.
        boot_prompt=dict(object_id=1,kind="occluder",origin="original_rgb_candidate",source_binding=binding,
            box_xyxy=local_boot["box_xyxy"],points_uv=local_boot["points_uv"],point_labels=local_boot["point_labels"],exclusion_authority=None)
        boot_result=provider.read(rgb,binding,prompts=[boot_prompt],reset=True)
        boot=boot_result["occluder_region_candidate_mask"].detach().cpu().numpy()
        x0,y0,x1,y1=local_boot["conservative_native_scope_xyxy"]
        scope=np.zeros_like(boot);scope[y0:y1,x0:x1]=True;boot&=scope
        boot_pixels=int(boot.sum());mask[boot]=255;encoded_prompts.append(boot_prompt)
        # The source-visible ankle between the independently read calf and
        # skate can remain outside both SAM components. Include its actual
        # skin/dark material only next to the current components, in this
        # reviewed native ankle search region; it is an exclusion, not a point.
        ankle_scope=np.zeros_like(boot);ankle_scope[666:710,805:873]=True
        ankle_near=cv2.dilate(mask,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(25,25)))>0
        ankle_addition=ankle_scope&(skin|dark)&ankle_near
        mask[ankle_addition]=255
    mask=cv2.dilate(mask,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(21,21)))
    path=DATA/"masks_current"/f"{index:06d}.png";cv2.imwrite(str(path),mask,[cv2.IMWRITE_PNG_COMPRESSION,1])
    receipt=dict(schema="rinkspace.current-native-source-person-exclusion.v1",index=index,source_pts=row["native_pts"],time_base="1/12800",
        source_time=row["source_time"],original_png_sha256=row["png_sha256"],original_RGB_sha256=row["decoded_native_RGB_sha256"],
        mask_path=str(path),mask_sha256=file_sha(path),current_original_RGB_independently_encoded=True,current_SAM2_inference_executed=True,
        prior_frame_memory_used=False,historical_masks_used=False,independent_current_SAM_encodings=len(encoded_prompts),current_source_prompt=prompt,
        executed_current_source_prompts=encoded_prompts,current_box_only_reencoding_used=box_reencoding,
        reviewed_current_native_local_foreground_region=local_boot,independent_current_native_local_boot_pixels=boot_pixels,
        current_source_person_search_proposal=proposal,current_foreground_body_material_uv=body,current_near_foreground_material_component_areas=added,
        generator_revision="current_torso_box_only_local_material_pocket_v3",native_dilation_radius_px=10,dilation_kernel_wh=[21,21],dilation_kernel_shape="ellipse",foreground_pixels=int((mask>0).sum()),
        mask_is_conservative_exclusion_candidate_not_pixel_ground_truth=True,source_prompt_coordinates_are_boundary_measurements=False,
        human_acceptance=False,source_points_or_solver_modified=False,checkpoint=dict(path=str(DEFAULT_MODEL),sha256=provider.model_sha256,download_performed=False))
    write(DATA/"mask_provenance"/f"{index:06d}.json",receipt)
    if index in PRIORITIES or index%50==0:
        overlay=raw.copy();overlay[mask>0]=(overlay[mask>0].astype(float)*.45+np.array([150,190,40])*.55).astype(np.uint8)
        contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(overlay,contours,-1,(0,255,255),2)
        cv2.imwrite(str(DATA/"masks_current"/f"review_{index:06d}.png"),overlay)
    return receipt


def finalize_mask_manifest():
    manifest=read(DATA/"frame_manifest.json");rows={r["index"]:r for r in manifest["frames"]}
    records=[read(p) for p in sorted((DATA/"mask_provenance").glob("[0-9]*.json"))]
    for record in records:
        index=record["index"];row=rows[index];row["mask_sha256"]=record["mask_sha256"]
        row["source"].update(mask_path=record["mask_path"],mask_sha256=record["mask_sha256"])
        row["current_person_mask_provenance"]=dict(path=str(DATA/"mask_provenance"/f"{index:06d}.json"),sha256=file_sha(DATA/"mask_provenance"/f"{index:06d}.json"))
    write(DATA/"frame_manifest.json",manifest)
    write(DATA/"masks_current/current_mask_manifest.json",dict(schema="rinkspace.current-native-source-person-exclusion.v1",
        source_manifest=dict(path=str(DATA/"frame_manifest.json"),sha256=file_sha(DATA/"frame_manifest.json")),rows=records,
        current_source_inference_per_frame=True,prior_frame_memory_used=False,historical_60fps_masks_used=False,mask_count=len(records),expected_frame_count=len(rows),
        all_requested_native_frames_processed=len(records)==len(rows),checkpoint=dict(path=str(DEFAULT_MODEL),download_performed=False)))
    tiles=[]
    for p in sorted((DATA/"masks_current").glob("review_[0-9]*.png")):
        tile=cv2.resize(cv2.imread(str(p)),(384,216));cv2.putText(tile,p.stem[7:],(4,17),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1);tiles.append(tile)
    sheet=np.zeros((216*((len(tiles)+4)//5),1920,3),np.uint8)
    for i,tile in enumerate(tiles): y,x=divmod(i,5);sheet[y*216:(y+1)*216,x*384:(x+1)*384]=tile
    cv2.imwrite(str(DATA/"masks_current/contact_sheet.png"),sheet)
    return records


def review_all_masks():
    """Native mask integrity and per-frame visibility review sheets."""
    manifest=read(DATA/"frame_manifest.json");records=[];tiles=[];flags=[]
    review_dir=DATA/"masks_current/review_pages";review_dir.mkdir(parents=True,exist_ok=True)
    for row in manifest["frames"]:
        index=row["index"];receipt=read(DATA/"mask_provenance"/f"{index:06d}.json")
        source=cv2.imread(row["image_path"]);mask=cv2.imread(receipt["mask_path"],0)
        if mask.shape!=(1080,1920) or set(np.unique(mask))!={0,255} or file_sha(receipt["mask_path"])!=receipt["mask_sha256"]:
            raise ValueError(f"Native mask integrity failed {index}")
        if receipt.get("generator_revision")!="current_torso_box_only_local_material_pocket_v3": raise ValueError(f"Stale mask generator {index}")
        yy,xx=np.nonzero(mask);bbox=[int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)]
        proposal=receipt["current_source_person_search_proposal"].get("box_xyxy")
        coverage=None
        if proposal:
            box=np.array(proposal);coverage=(bbox[3]-bbox[1])/(box[3]-box[1])
            if coverage<.72 or bbox[3]<box[3]-70:flags.append(dict(index=index,status="visible_detector_height_or_boot_extent_needs_review",mask_bbox=bbox,detector_box=proposal,vertical_extent_ratio=float(coverage)))
        source[mask>0]=(source[mask>0].astype(float)*.45+np.array([150,190,40])*.55).astype(np.uint8)
        contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(source,contours,-1,(0,255,255),2)
        tile=cv2.resize(source,(320,180));cv2.putText(tile,str(index),(4,17),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1);tiles.append(tile)
        records.append(dict(index=index,mask_sha256=receipt["mask_sha256"],mask_bbox=bbox,foreground_pixels=int((mask>0).sum()),
            current_source_torso_seed_in_mask=bool(mask[int(receipt["current_foreground_body_material_uv"][1]),int(receipt["current_foreground_body_material_uv"][0])]),
            current_original_RGB_independently_encoded=receipt["current_original_RGB_independently_encoded"],prior_frame_memory_used=receipt["prior_frame_memory_used"]))
    pages=[]
    for page in range((len(tiles)+59)//60):
        canvas=np.zeros((180*10,320*6,3),np.uint8)
        for ordinal,tile in enumerate(tiles[page*60:(page+1)*60]):y,x=divmod(ordinal,6);canvas[y*180:(y+1)*180,x*320:(x+1)*320]=tile
        path=review_dir/f"{page+1:03d}.png";cv2.imwrite(str(path),canvas);pages.append(str(path))
    result=dict(schema="rinkspace.current-native-mask-integrity-and-visibility-review.v1",frame_count=len(records),first_index=201,last_index=1248,
        native_binary_mask_integrity_passed=True,current_torso_seeds_all_in_mask=all(r["current_source_torso_seed_in_mask"] for r in records),
        current_source_RGB_independently_encoded_all=True,historical_mask_used=False,records=records,flags_requiring_visual_review=flags,review_pages=pages,
        visual_review_completed=False,masks_are_conservative_exclusions_not_ground_truth=True)
    write(DATA/"masks_current/integrity_review.json",result)
    print(f"mask_integrity_ready {len(records)} review_pages={len(pages)} flags={len(flags)}",flush=True)
    return result


def run(indices=None,priority_only=False,force=False):
    started=time.perf_counter();torch.set_num_threads(4)
    manifest=read(DATA/"frame_manifest.json");rows={r["index"]:r for r in manifest["frames"]}
    selected=list(sorted(rows)) if indices is None else indices
    selected=[i for i in PRIORITIES if i in selected]+[i for i in selected if i not in PRIORITIES]
    if priority_only:selected=[i for i in selected if i in PRIORITIES]
    provider=Sam2SourceInstances(DEFAULT_MODEL,device="cuda",reread_rgb=False,contour_stride=16)
    detector=SourcePersonBoxes(YOLO_WEIGHTS,dependency_paths=[YOLO_SITE],runtime_directory=DATA/"person_detector_runtime",device="cuda")
    failed=[]
    for ordinal,index in enumerate(selected,1):
        prior=DATA/"mask_provenance"/f"{index:06d}.json"
        if not force and prior.is_file() and read(prior).get("generator_revision")=="current_torso_box_only_local_material_pocket_v3" and file_sha(DATA/"masks_current"/f"{index:06d}.png")==read(prior)["mask_sha256"]:
            continue
        try:
            receipt=mask_one(provider,detector,rows[index],force_box=force and index not in LOCAL_BOOT_PROMPTS)
        except Exception as e:
            failed.append(dict(index=index,error=str(e)));print(f"mask_failed {index} {e}",flush=True);continue
        if ordinal<=10 or ordinal%50==0:print(f"current_mask_ready {index} {ordinal}/{len(selected)} pixels={receipt['foreground_pixels']} elapsed={time.perf_counter()-started:.2f}",flush=True)
        if ordinal==10:finalize_mask_manifest();navigation(PRIORITIES)
    records=finalize_mask_manifest()
    write(DATA/"mask_inference_failures.json",dict(failures=failed,elapsed_seconds=time.perf_counter()-started))
    if failed:raise RuntimeError(f"Current mask inference failures: {failed}")
    print(f"current_masks_complete {len(records)}",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--indices",type=int,nargs="*");parser.add_argument("--priority-only",action="store_true");parser.add_argument("--review-only",action="store_true");parser.add_argument("--force",action="store_true")
    args=parser.parse_args()
    if args.review_only:review_all_masks()
    else:run(args.indices,args.priority_only,args.force)
