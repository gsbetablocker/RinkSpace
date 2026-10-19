"""Infer one authenticated native-source person mask from explicit source prompts.

The prompt/colour scope specification is data, not a rink correspondence.
Each SAM object is independently reset and encoded; no historical masks read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/"src")]
from calibration.program_tracking.runtime.bootstrap import add_geometry_dependencies
add_geometry_dependencies()
import cv2
import numpy as np
from rinkspace.vision_instances import Sam2SourceInstances
from rinkspace.programme_source_stream import file_sha
from calibration.program_tracking.infer_current_person_exclusion_masks import DEFAULT_MODEL


def apply_reviewed_source_constraints(mask,spec):
    mask=mask.copy()
    receipt=[]
    for constraint in spec.get("reviewed_current_source_silhouette_constraints",[]):
        allowed=np.zeros_like(mask)
        cv2.fillPoly(allowed,[np.asarray(constraint["source_outline_xy"],np.int32)],255)
        allowed=cv2.dilate(allowed,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(21,21)))
        x0,y0,x1,y1=constraint["apply_scope_xyxy"]
        removed=int(((mask[y0:y1,x0:x1]>0)&(allowed[y0:y1,x0:x1]==0)).sum())
        mask[y0:y1,x0:x1][allowed[y0:y1,x0:x1]==0]=0
        receipt.append(dict(region=constraint["region"],source_outline_xy=constraint["source_outline_xy"],
            apply_scope_xyxy=constraint["apply_scope_xyxy"],native_outline_margin_px=10,
            rejected_candidate_pixels=removed,authority="current original visible silhouette visual inspection; exclusion only"))
    return mask,receipt


def save_preview(destination,bgr,mask):
    preview=bgr.copy()
    preview[mask>0]=(preview[mask>0].astype(float)*.45+np.array([150,190,40])*.55).astype(np.uint8)
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(preview,contours,-1,(0,255,255),2)
    cv2.imwrite(str(destination/"person_mask_review.png"),preview)


def infer(source_binding_path,prompt_spec_path,model=DEFAULT_MODEL):
    source_binding_path,prompt_spec_path=Path(source_binding_path),Path(prompt_spec_path)
    source=json.loads(source_binding_path.read_text(encoding="utf-8"))
    spec=json.loads(prompt_spec_path.read_text(encoding="utf-8"))
    source_path=Path(source["image_path"])
    if file_sha(source_path)!=source["png_sha256"] or spec["source_png_sha256"]!=source["png_sha256"]:
        raise ValueError("Authenticated current source PNG changed")
    bgr=cv2.imread(str(source_path))
    rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
    if hashlib.sha256(rgb.tobytes()).hexdigest()!=source["decoded_RGB_sha256"]:
        raise ValueError("Authenticated current native RGB changed")
    binding=dict(source_sha256=source["source_video_sha256"],source_pts=source["native_pts"],
        time_base=source["time_base"],native_hw=source["native_hw"],
        shot_id="single_current_source_person_exclusion",decoded_native_rgb_sha256=source["decoded_RGB_sha256"])
    provider=Sam2SourceInstances(Path(model),device="cuda",reread_rgb=False,contour_stride=16)
    mask=np.zeros(rgb.shape[:2],np.uint8)
    executed=[]
    for proposed in spec["independent_SAM_prompts"]:
        prompt=dict(object_id=1,kind="occluder",origin="original_rgb_candidate",
            source_binding=binding,box_xyxy=proposed["box_xyxy"],
            points_uv=proposed["points_uv"],point_labels=proposed["point_labels"],exclusion_authority=None)
        result=provider.read(rgb,binding,prompts=[prompt],reset=True)
        candidate=result["occluder_region_candidate_mask"].detach().cpu().numpy()
        mask[candidate]=255
        executed.append(dict(region=proposed["region"],prompt=prompt,mask_pixels=int(candidate.sum())))
    # Keep only the currently authenticated foreground person's component,
    # before any dilation can connect a spectator/background island to it.
    seed_x,seed_y=spec["foreground_body_seed"]
    _,labels,stats,_=cv2.connectedComponentsWithStats((mask>0).astype(np.uint8),8)
    label=int(labels[seed_y,seed_x])
    if label==0:
        raise ValueError("Authenticated current foreground body omitted")
    removed=int((mask>0).sum()-stats[label,cv2.CC_STAT_AREA])
    mask=(labels==label).astype(np.uint8)*255
    spread=rgb.max(2).astype(int)-rgb.min(2).astype(int)
    source_black=(rgb.max(2)<110)&(spread<55)
    torso_spec=spec["current_source_black_material_scope_xyxy"]
    x0,y0,x1,y1=torso_spec
    scoped=np.zeros_like(mask)
    scoped[y0:y1,x0:x1]=source_black[y0:y1,x0:x1]
    n,labels,stats,_=cv2.connectedComponentsWithStats(scoped,8)
    near=cv2.dilate(mask,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(21,21)))>0
    black=np.zeros_like(mask)
    source_components=[]
    for other in range(1,n):
        area=int(stats[other,cv2.CC_STAT_AREA])
        component=labels==other
        if area>=100 and bool((component&near).any()):
            black[component]=255
            source_components.append(area)
    black=cv2.morphologyEx(black,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)))
    contours,_=cv2.findContours(black,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(mask,contours,-1,255,cv2.FILLED)
    # Foreground-coloured pockets repair skin/chiffon/ornament omissions; they
    # do not expand into external white letters or purple advertisement areas.
    hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
    ycc=cv2.cvtColor(rgb,cv2.COLOR_RGB2YCrCb)
    skin=(hsv[...,0]<=27)&(hsv[...,1]>=25)&(hsv[...,1]<=185)&(hsv[...,2]>=90)&(ycc[...,1]>=132)&(ycc[...,1]<=188)&(ycc[...,2]>=75)&(ycc[...,2]<=128)
    closed=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(31,31)))>0
    pocket=((rgb.max(2)<150)&(spread<48)|skin)&closed&(mask==0)
    mask[pocket]=255
    filled_holes=[]
    n,labels,stats,_=cv2.connectedComponentsWithStats((mask==0).astype(np.uint8),8)
    purple=(hsv[...,0]>=135)&(hsv[...,0]<=175)&(spread>55)&(hsv[...,2]>75)
    white=(hsv[...,1]<60)&(hsv[...,2]>145)
    for other in range(1,n):
        x,y,w,h,area=stats[other]
        if x==0 or y==0 or x+w==mask.shape[1] or y+h==mask.shape[0]:
            continue
        hole=labels==other
        if float((purple|white)[hole].mean())<.6:
            mask[hole]=255
            filled_holes.append(int(area))
    # Conservative, uniform native 10 px motion-blur exclusion margin.
    mask=cv2.dilate(mask,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(21,21)))
    mask,constraint_receipt=apply_reviewed_source_constraints(mask,spec)
    for x,y in spec["current_source_foreground_check_points"]:
        if not bool(mask[y,x]):
            raise ValueError(f"Current source foreground check point not excluded: {(x,y)}")
    destination=source_path.parent
    output=destination/"person_mask.png"
    cv2.imwrite(str(output),mask,[cv2.IMWRITE_PNG_COMPRESSION,1])
    save_preview(destination,bgr,mask)
    receipt=dict(schema="rinkspace.single-current-source-person-exclusion.v1",
        programme_frame_index_1based=source["programme_frame_index_1based"],source_pts=source["native_pts"],
        source_time=source["source_time"],time_base=source["time_base"],native_hw=source["native_hw"],
        original_png_sha256=source["png_sha256"],original_RGB_sha256=source["decoded_RGB_sha256"],
        source_binding_sha256=file_sha(source_binding_path),mask_path=str(output.resolve()),mask_sha256=file_sha(output),
        local_checkpoint=dict(path=str(model),sha256=provider.model_sha256,download_performed=False),
        independent_current_SAM_encodings=executed,prior_frame_memory_used=False,historical_masks_used=False,
        disconnected_background_pixels_removed=removed,current_source_black_components_areas=source_components,
        current_source_black_material_scope_xyxy=torso_spec,source_colour_pocket_pixels_added=int(pocket.sum()),
        foreground_internal_holes_filled_areas=filled_holes,native_dilation_radius_px=10,dilation_kernel_wh=[21,21],
        reviewed_current_source_silhouette_constraints=constraint_receipt,
        dilation_kernel_shape="ellipse",source_prompt_spec_sha256=file_sha(prompt_spec_path),
        generator_script_sha256=file_sha(Path(__file__)),fitting_points_or_solver_modified=False,
        mask_is_conservative_exclusion_candidate_not_pixel_ground_truth=True,human_acceptance=False,
        neighbouring_frames_processed=False)
    (destination/"person_mask_provenance.json").write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(dict(mask_path=str(output),mask_sha256=receipt["mask_sha256"],pixels=int((mask>0).sum())),ensure_ascii=False),flush=True)


def repair_existing(source_binding_path,prompt_spec_path):
    source=json.loads(Path(source_binding_path).read_text(encoding="utf-8"))
    spec=json.loads(Path(prompt_spec_path).read_text(encoding="utf-8"))
    source_path=Path(source["image_path"])
    destination=source_path.parent
    receipt_path=destination/"person_mask_provenance.json"
    receipt=json.loads(receipt_path.read_text(encoding="utf-8"))
    mask_path=destination/"person_mask.png"
    if file_sha(source_path)!=source["png_sha256"] or file_sha(mask_path)!=receipt["mask_sha256"]:
        raise ValueError("Current original or candidate mask changed before source review repair")
    mask,constraint_receipt=apply_reviewed_source_constraints(cv2.imread(str(mask_path),0),spec)
    for x,y in spec["current_source_foreground_check_points"]:
        if not mask[y,x]:
            raise ValueError("Source-foreground check failed after silhouette constraint")
    cv2.imwrite(str(mask_path),mask,[cv2.IMWRITE_PNG_COMPRESSION,1])
    save_preview(destination,cv2.imread(str(source_path)),mask)
    receipt["reviewed_current_source_silhouette_constraints"]=constraint_receipt
    receipt["mask_sha256"]=file_sha(mask_path)
    receipt["source_prompt_spec_sha256"]=file_sha(Path(prompt_spec_path))
    receipt["generator_script_sha256"]=file_sha(Path(__file__))
    receipt_path.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(dict(mask_path=str(mask_path),mask_sha256=receipt["mask_sha256"],removed_pixels=sum(r["rejected_candidate_pixels"] for r in constraint_receipt)),ensure_ascii=False),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_binding",type=Path)
    parser.add_argument("prompt_spec",type=Path)
    parser.add_argument("--model",type=Path,default=DEFAULT_MODEL)
    parser.add_argument("--repair-existing",action="store_true")
    args=parser.parse_args()
    if args.repair_existing:
        repair_existing(args.source_binding,args.prompt_spec)
    else:
        infer(args.source_binding,args.prompt_spec,args.model)
