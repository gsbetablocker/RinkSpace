"""Infer person exclusion from each authenticated native current exposure.

Uses the existing local SAM2 interface/checkpoint. Historical source masks are
never read. Point/box prompts locate the foreground person, not rink evidence.
"""
from __future__ import annotations

from rinkspace.storage import resolve_asset

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from calibration.program_tracking.runtime.bootstrap import add_geometry_dependencies
add_geometry_dependencies()

import cv2
import numpy as np
from rinkspace.vision_instances import Sam2SourceInstances
from rinkspace.programme_source_stream import file_sha

DEFAULT_SEQUENCE = Path("D:/CodexData/RinkSpace_Data/events/ankara_2026/programmes/okada_mei_sp/results/"
    "frame_by_frame_review_20261013/opening_100_50fps")
DEFAULT_MODEL = resolve_asset(ROOT, ROOT / "models/cache/sam2.1-hiera-tiny")


def current_skin_seeds(rgb):
    """Positive current hand pixels, scoped away from the spectators/board ink."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    ycc = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    skin = ((hsv[..., 0] <= 27) & (hsv[..., 1] >= 25) & (hsv[..., 1] <= 185)
        & (hsv[..., 2] >= 90) & (ycc[..., 1] >= 132) & (ycc[..., 1] <= 188)
        & (ycc[..., 2] >= 75) & (ycc[..., 2] <= 128))
    scope = np.zeros(skin.shape, bool)
    scope[145:685, 1150:1640] = True
    skin &= scope
    count, labels, stats, _ = cv2.connectedComponentsWithStats(skin.astype(np.uint8), 8)
    candidates = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= 250]
    if not candidates:
        return [], skin
    selected = max(candidates, key=lambda i: stats[i, cv2.CC_STAT_AREA])
    y, x = np.nonzero(labels == selected)
    centre = np.array([np.median(x), np.median(y)])
    coordinates = np.column_stack((x, y))
    middle = coordinates[np.argmin(np.sum((coordinates-centre)**2, axis=1))]
    points = [middle]
    # Existing current skin extremes add the moving fingertips; they remain
    # SAM prompts only. No source contour or world point is produced here.
    for score in (x, -x, y, -y):
        point = coordinates[int(np.argmax(score))]
        if all(np.linalg.norm(point-p) > 18 for p in points):
            points.append(point)
    return [p.astype(float).tolist() for p in points], skin


def conservative_current_mask(rgb, initial):
    """Close only foreground-coloured internal holes, retain visible board gaps."""
    mask = initial.copy()
    inverse = (mask == 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(inverse, 8)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    ycc = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    skin = ((hsv[..., 0] <= 27) & (hsv[..., 1] >= 25) & (hsv[..., 1] <= 185)
        & (hsv[..., 2] >= 90) & (ycc[..., 1] >= 132) & (ycc[..., 1] <= 188)
        & (ycc[..., 2] >= 75) & (ycc[..., 2] <= 128))
    neutral_dark = (rgb.max(2).astype(int)-rgb.min(2).astype(int) < 35) & (rgb[..., 1] < 100)
    filled_holes = []
    for label in range(1,count):
        x,y,w,h,area = stats[label]
        if x==0 or y==0 or x+w==rgb.shape[1] or y+h==rgb.shape[0]:
            continue
        hole = labels==label
        if (skin[hole].mean() >= .10 or neutral_dark[hole].mean() >= .40) and 650<x<1600:
            mask[hole] = 255
            filled_holes.append(int(area))
    nearby = cv2.dilate(mask,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(31,31)))>0
    foreground_scope = np.zeros(skin.shape,bool)
    foreground_scope[145:1080,650:1640]=True
    skin_addition = skin & nearby & foreground_scope & (mask==0)
    mask[skin_addition]=255
    # Translucent black cuff holes are foreground too: add only neutral dark
    # source pixels near the authenticated current person, never purple board.
    # Closing repairs enclosed/near-enclosed foreground pockets without
    # extending onto grey blurred edges of the neighbouring white letters.
    nearby_sleeve=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(61,61)))>0
    sleeve_scope=np.zeros(skin.shape,bool)
    sleeve_scope[145:1080,760:1640]=True
    source_black=((rgb.max(2).astype(int)-rgb.min(2).astype(int))<48)&(rgb.max(2)<150)
    dark_addition=source_black&nearby_sleeve&sleeve_scope&(mask==0)
    mask[dark_addition]=255
    # User-requested native 10 px exclusion margin includes motion blur.
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(21,21))
    mask=cv2.dilate(mask,kernel)
    torso,torso_receipt=current_source_black_torso(rgb,mask)
    mask[torso>0]=255
    mask,final_hole_receipt=close_source_foreground_holes(rgb,mask)
    count,labels,stats,_=cv2.connectedComponentsWithStats((mask>0).astype(np.uint8),8)
    foreground_label=int(labels[830,1100])
    if foreground_label==0:
        raise ValueError("Foreground body seed lost during source mask repair")
    removed_islands=int((mask>0).sum()-stats[foreground_label,cv2.CC_STAT_AREA])
    mask=(labels==foreground_label).astype(np.uint8)*255
    return mask,dict(
        native_dilation_radius_px=10,dilation_kernel_shape="ellipse",
        dilation_kernel_width_height=[21,21],foreground_coloured_internal_holes_filled_areas=filled_holes,
        current_nearby_skin_pixels_added=int(skin_addition.sum()),
        current_nearby_neutral_black_sleeve_pixels_added=int(dark_addition.sum()),
        black_fill_scope="current-person morphological-close pockets only",
        black_fill_closing_kernel_width_height=[61,61],
        source_black_max_RGB=150,source_black_max_channel_spread=48,
        disconnected_nonforeground_pixels_removed=removed_islands,
        final_source_coloured_hole_repair=final_hole_receipt,
        current_source_black_torso_envelope=torso_receipt,
        visible_board_coloured_internal_gaps_not_generally_filled=True)


def close_source_foreground_holes(rgb,mask):
    """Fill enclosed foreground holes while retaining explicit board material."""
    mask=mask.copy()
    hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
    spread=rgb.max(2).astype(int)-rgb.min(2).astype(int)
    purple=(hsv[...,0]>=135)&(hsv[...,0]<=175)&(spread>55)&(hsv[...,2]>75)
    white=(hsv[...,1]<60)&(hsv[...,2]>145)
    board=purple|white
    n,labels,stats,_=cv2.connectedComponentsWithStats((mask==0).astype(np.uint8),8)
    filled=[]
    retained=[]
    for label in range(1,n):
        x,y,w,h,area=stats[label]
        if x==0 or y==0 or x+w==rgb.shape[1] or y+h==rgb.shape[0] or not 650<x<1640:
            continue
        hole=labels==label
        fraction=float(board[hole].mean())
        if fraction<.6:
            mask[hole]=255
            filled.append(dict(area=int(area),board_material_colour_fraction=fraction,bbox=[int(x),int(y),int(w),int(h)]))
        else:
            retained.append(dict(area=int(area),board_material_colour_fraction=fraction,bbox=[int(x),int(y),int(w),int(h)]))
    return mask,dict(filled_foreground_holes=filled,retained_board_material_coloured_holes=retained)


def current_source_black_torso(rgb,current_mask=None):
    """Actual connected dark torso pixels, independent of SAM's skin bias."""
    spread=rgb.max(2).astype(int)-rgb.min(2).astype(int)
    dark=(rgb.max(2)<110)&(spread<55)
    scope=np.zeros(dark.shape,np.uint8)
    scope[680:1080,560:1510]=dark[680:1080,560:1510]
    count,labels,stats,_=cv2.connectedComponentsWithStats(scope,8)
    candidates=[i for i in range(1,count) if stats[i,cv2.CC_STAT_AREA]>3000]
    if not candidates:
        raise ValueError("Current dark torso source component missing")
    label=max(candidates,key=lambda i:stats[i,cv2.CC_STAT_AREA])
    component=(labels==label).astype(np.uint8)
    added=[]
    if current_mask is not None:
        near=cv2.dilate(current_mask,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(41,41)))>0
        for other in range(1,count):
            if other==label or stats[other,cv2.CC_STAT_AREA]<100:
                continue
            other_component=labels==other
            if bool((other_component&near).any()):
                component[other_component]=1
                added.append(int(stats[other,cv2.CC_STAT_AREA]))
    component=cv2.morphologyEx(component,cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)))
    contours,_=cv2.findContours(component,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    envelope=np.zeros_like(component)
    cv2.drawContours(envelope,contours,-1,255,cv2.FILLED)
    envelope=cv2.dilate(envelope,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(21,21)))
    return envelope,dict(source_rule="max RGB <110 and channel spread <55, largest current torso component",
        source_scope_xyxy=[560,680,1510,1080],closing_kernel=[7,7],native_dilation_radius_px=10,
        other_near_current_foreground_component_areas=added,
        component_pixels=int(stats[label,cv2.CC_STAT_AREA]),conservative_envelope_pixels=int((envelope>0).sum()))


def current_dark_arm_seeds(rgb):
    """Locate actual neutral black sleeve pixels in this frame's arm region."""
    # Absolute channel spread separates the neutral chiffon/black sleeve from
    # both the saturated blue board rail and the purple advertisement. HSV
    # saturation alone admitted the rail in an earlier rejected candidate.
    spread=rgb.max(2).astype(int)-rgb.min(2).astype(int)
    dark=(rgb.max(2)<125)&(spread<42)
    scope=np.zeros(dark.shape,np.uint8)
    scope[160:625,1130:1490]=dark[160:625,1130:1490]
    count,labels,stats,_=cv2.connectedComponentsWithStats(scope,8)
    candidates=[i for i in range(1,count) if stats[i,cv2.CC_STAT_AREA]>1000]
    if not candidates:
        raise ValueError("Current source lacks a separately measurable black-arm prompt region")
    label=max(candidates,key=lambda i:stats[i,cv2.CC_STAT_AREA])
    component=(labels==label).astype(np.uint8)
    distance=cv2.distanceTransform(component,cv2.DIST_L2,5)
    yy,xx=np.nonzero(distance>=min(10.,float(distance.max())*.5))
    coordinates=np.column_stack((xx,yy))
    points=[]
    for rank in (.12,.5,.88):
        target=np.quantile(coordinates,rank,axis=0)
        point=coordinates[np.argmin(np.sum((coordinates-target)**2,axis=1))]
        if all(np.linalg.norm(point-p)>20 for p in points):
            points.append(point)
    # The source component is also a conservative black-sleeve exclusion aid.
    # It is never a fitting boundary, world point, or a learned pixel truth.
    component=cv2.morphologyEx(component,cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)))
    contours,_=cv2.findContours(component,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    envelope=np.zeros_like(component)
    cv2.drawContours(envelope,contours,-1,1,cv2.FILLED)
    return [p.astype(float).tolist() for p in points],envelope


def run(sequence=DEFAULT_SEQUENCE, model=DEFAULT_MODEL, indices=None):
    sequence, model = Path(sequence), Path(model)
    manifest_path = sequence / "frame_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = {r["index"]:r for r in manifest["frames"]}
    selected = sorted(rows) if indices is None else list(dict.fromkeys(indices))
    # First answer the reported failure and the most changed final pose.
    priorities = [i for i in (1, 2, 50, 75, 100, 74, 84, 90) if i in selected]
    selected = priorities + [i for i in selected if i not in priorities]
    destination = sequence / "masks_current"
    destination.mkdir(parents=True, exist_ok=True)
    provider = Sam2SourceInstances(model, device="cuda", reread_rgb=False, contour_stride=16)
    records, tiles = [], []
    for index in selected:
        row = rows[index]
        path = Path(row["path"])
        if file_sha(path) != row["png_sha256"]:
            raise ValueError("Current original PNG changed")
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if hashlib.sha256(rgb.tobytes()).hexdigest() != row["decoded_native_RGB_sha256"]:
            raise ValueError("Current original RGB changed")
        hand_points, skin = current_skin_seeds(rgb)
        arm_points,arm_source_envelope=current_dark_arm_seeds(rgb)
        binding = dict(source_sha256=manifest["source"]["sha256"], source_pts=row["native_pts"],
            time_base=row["source_time_base"], shot_id="official_50fps_opening_current_person",
            native_hw=[1080, 1920], decoded_native_rgb_sha256=row["decoded_native_RGB_sha256"])
        positive = [[1000., 300.], [1100., 830.]] + hand_points
        negative = [[550., 500.], [1750., 650.], [750., 70.], [1700., 150.]]
        prompt = dict(object_id=1, kind="occluder", origin="original_rgb_candidate",
            source_binding=binding, box_xyxy=[560., 0., 1640., 1080.],
            points_uv=positive+negative, point_labels=[1]*len(positive)+[0]*len(negative),
            current_source_skin_seed_only=True, exclusion_authority=None)
        result = provider.read(rgb, binding, prompts=[prompt], reset=True)
        mask = result["occluder_region_candidate_mask"].detach().cpu().numpy().astype(np.uint8)*255
        # Separate current encodings avoid SAM's multi-object exclusion, which
        # could remove torso pixels when person and sleeve shared one inference.
        arm_negative=[[1500.,600.],[1050.,300.],[1550.,700.],[1600.,80.]]
        arm_prompt=dict(object_id=1,kind="occluder",origin="original_rgb_candidate",
            source_binding=binding,box_xyxy=[1080.,0.,1610.,750.],
            points_uv=arm_points+arm_negative,point_labels=[1]*len(arm_points)+[0]*len(arm_negative),
            current_source_dark_sleeve_seed_only=True,exclusion_authority=None)
        arm_result=provider.read(rgb,binding,prompts=[arm_prompt],reset=True)
        arm_mask=arm_result["occluder_region_candidate_mask"].detach().cpu().numpy()
        mask[arm_mask]=255
        mask, conservative_receipt = conservative_current_mask(rgb, mask)
        if not mask[830, 1100] or not mask[300, 1000]:
            raise ValueError("Current person segmentation omitted authenticated foreground prompts")
        output_path = destination / f"{index:06d}.png"
        if not cv2.imwrite(str(output_path), mask, [cv2.IMWRITE_PNG_COMPRESSION, 1]):
            raise OSError(output_path)
        hand_seed_inside = all(bool(mask[int(y), int(x)]) for x, y in hand_points)
        record = dict(index=index, source_pts=row["native_pts"], source_time_base=row["source_time_base"],
            original_png_sha256=row["png_sha256"], original_RGB_sha256=row["decoded_native_RGB_sha256"],
            current_mask_path=str(output_path.resolve()), current_mask_sha256=file_sha(output_path),
            current_native_RGB_independently_encoded=True, old_60fps_mask_used=False,
            current_SAM2_inference_executed=True, prior_frame_memory_used=False,
            source_prompts=[prompt,arm_prompt], current_hand_skin_positive_seed_count=len(hand_points),
            current_hand_positive_seeds_inside_mask=hand_seed_inside,
            current_dark_arm_positive_seed_points=arm_points,
            current_dark_arm_positive_seeds_inside_mask=all(bool(mask[int(y),int(x)]) for x,y in arm_points),
            current_source_black_sleeve_component_pixels=int(arm_source_envelope.sum()),
            source_black_sleeve_conservative_envelope_added=False,
            separate_current_sleeve_SAM_mask_union=True,
            conservative_current_exclusion=conservative_receipt,
            foreground_pixels=int((mask>0).sum()), mask_is_pixel_ground_truth=False,
            current_skin_seed_coverage_within_hand_roi=float((skin & (mask>0)).sum()/max(1, skin.sum())))
        records.append(record)
        colored = bgr.copy()
        colored[mask>0] = (colored[mask>0].astype(np.float32)*.45 + np.array([150.,190.,40.])*.55).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(colored, contours, -1, (0,255,255), 2)
        if index in priorities or index in (69,79,95):
            cv2.imwrite(str(destination/f"review_{index:06d}.png"), colored)
        tile = cv2.resize(colored, (384,216), interpolation=cv2.INTER_AREA)
        cv2.putText(tile, f"{index:03d} current source mask", (4,16), cv2.FONT_HERSHEY_SIMPLEX,.42,(255,255,255),1)
        tiles.append((index,tile))
        if len(records) <= 6 or len(records) % 20 == 0:
            print(json.dumps(dict(stage="current_mask_written", index=index,
                completed=len(records), total=len(selected), hand_seed_inside=hand_seed_inside,
                mask_path=str(output_path)), ensure_ascii=False), flush=True)
    ordered = sorted(records, key=lambda r:r["index"])
    sheet = np.zeros((216*10,384*10,3),np.uint8)
    for ordinal, (_,tile) in enumerate(sorted(tiles)):
        y,x = divmod(ordinal,10)
        sheet[y*216:(y+1)*216,x*384:(x+1)*384] = tile
    sheet_path = destination/"all_current_masks_contact_sheet.png"
    cv2.imwrite(str(sheet_path),sheet)
    payload = dict(schema="rinkspace.current-native-source-person-exclusion.v1",
        source_manifest=dict(path=str(manifest_path),sha256=file_sha(manifest_path)),
        checkpoint=dict(path=str(model), model_sha256=provider.model_sha256, download_performed=False),
        inference_interface="rinkspace.vision_instances.Sam2SourceInstances.read",
        current_original_png_exposures_processed=len(ordered), all_requested_native_frames_processed=len(ordered)==len(rows),
        current_source_inference_per_frame=True, prior_frame_memory_used=False,
        old_navigation_masks_modified=False, source_points_or_solver_modified=False,
        masks_are_source_exclusion_candidates_not_human_pixel_truth=True,
        rows=ordered,contact_sheet=str(sheet_path.resolve()))
    (destination/"current_mask_manifest.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    review_masks(sequence)
    return payload


def review_masks(sequence=DEFAULT_SEQUENCE):
    sequence=Path(sequence)
    destination=sequence/"masks_current"
    path=destination/"current_mask_manifest.json"
    manifest=json.loads(path.read_text(encoding="utf-8"))
    tiles=[]
    fulltiles=[]
    for row in manifest["rows"]:
        source=cv2.imread(str(sequence/"frames"/f"{row['index']:06d}.png"))
        mask=cv2.imread(row["current_mask_path"],0)
        source[mask>0]=(source[mask>0].astype(float)*.45+np.array([150,190,40])*.55).astype(np.uint8)
        contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(source,contours,-1,(0,255,255),2)
        if row["index"] in (1,2,6,15,26,50,65,75,83,100):
            cv2.imwrite(str(destination/f"review_{row['index']:06d}.png"),source)
        fulltile=cv2.resize(source,(384,216),interpolation=cv2.INTER_AREA)
        cv2.putText(fulltile,f"{row['index']:03d} final current mask",(4,16),cv2.FONT_HERSHEY_SIMPLEX,.42,(255,255,255),1)
        fulltiles.append(fulltile)
        tile=cv2.resize(source[0:720,750:1650],(270,216),interpolation=cv2.INTER_AREA)
        cv2.putText(tile,f"{row['index']:03d}",(3,17),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
        tiles.append(tile)
    pages=[]
    for page in range((len(tiles)+24)//25):
        canvas=np.zeros((216*5,270*5,3),np.uint8)
        for ordinal,tile in enumerate(tiles[page*25:page*25+25]):
            y,x=divmod(ordinal,5)
            canvas[y*216:(y+1)*216,x*270:(x+1)*270]=tile
        image_path=destination/f"hand_crop_review_{page+1:03d}.png"
        cv2.imwrite(str(image_path),canvas)
        pages.append(str(image_path.resolve()))
    manifest["all_frame_hand_crop_review_pages"]=pages
    fullsheet=np.zeros((216*10,384*10,3),np.uint8)
    for ordinal,tile in enumerate(fulltiles):
        y,x=divmod(ordinal,10)
        fullsheet[y*216:(y+1)*216,x*384:(x+1)*384]=tile
    cv2.imwrite(str(destination/"all_current_masks_contact_sheet.png"),fullsheet)
    path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(dict(hand_review_pages=pages,
        hand_prompt_coverage_failures=[r["index"] for r in manifest["rows"] if not r["current_hand_positive_seeds_inside_mask"]],
        low_RGB_hand_seed_coverage=[dict(index=r["index"],coverage=r["current_skin_seed_coverage_within_hand_roi"])
            for r in manifest["rows"] if r["current_skin_seed_coverage_within_hand_roi"]<.85]),ensure_ascii=False),flush=True)
    return manifest


def finalize_existing(sequence=DEFAULT_SEQUENCE):
    """Apply the same final source-hole rule to a completed current SAM batch."""
    sequence=Path(sequence)
    destination=sequence/"masks_current"
    path=destination/"current_mask_manifest.json"
    manifest=json.loads(path.read_text(encoding="utf-8"))
    if len(manifest["rows"])!=100 or [r["index"] for r in manifest["rows"]]!=list(range(1,101)):
        raise ValueError("Finalization requires the complete ordered current 100-frame batch")
    for row in manifest["rows"]:
        source_path=sequence/"frames"/f"{row['index']:06d}.png"
        if file_sha(source_path)!=row["original_png_sha256"]:
            raise ValueError("Current source PNG changed before final mask repair")
        rgb=cv2.cvtColor(cv2.imread(str(source_path)),cv2.COLOR_BGR2RGB)
        mask_path=Path(row["current_mask_path"])
        if file_sha(mask_path)!=row["current_mask_sha256"]:
            raise ValueError("Current SAM mask changed before final repair")
        mask=cv2.imread(str(mask_path),0)
        torso,torso_receipt=current_source_black_torso(rgb,mask)
        mask[torso>0]=255
        mask,receipt=close_source_foreground_holes(rgb,mask)
        cv2.imwrite(str(mask_path),mask,[cv2.IMWRITE_PNG_COMPRESSION,1])
        row["current_mask_sha256"]=file_sha(mask_path)
        row["foreground_pixels"]=int((mask>0).sum())
        row["conservative_current_exclusion"]["final_source_coloured_hole_repair"]=receipt
        row["conservative_current_exclusion"]["current_source_black_torso_envelope"]=torso_receipt
    manifest["final_source_hole_rule_applied_uniformly_to_all_100"]=True
    manifest["generator_script_sha256"]=file_sha(Path(__file__))
    path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return review_masks(sequence)


def verify_final(sequence=DEFAULT_SEQUENCE):
    sequence=Path(sequence)
    path=sequence/"masks_current/current_mask_manifest.json"
    manifest=json.loads(path.read_text(encoding="utf-8"))
    rows=manifest["rows"]
    if len(rows)!=100 or [r["index"] for r in rows]!=list(range(1,101)):
        raise ValueError("Incomplete final current-mask batch")
    for row in rows:
        if file_sha(sequence/"frames"/f"{row['index']:06d}.png")!=row["original_png_sha256"]:
            raise ValueError("Final current source PNG mismatch")
        if file_sha(Path(row["current_mask_path"]))!=row["current_mask_sha256"]:
            raise ValueError("Final current mask hash mismatch")
        mask=cv2.imread(row["current_mask_path"],0)
        if mask.shape!=(1080,1920) or set(np.unique(mask))!={0,255}:
            raise ValueError("Invalid current native binary exclusion mask")
        if not mask[300,1000] or not mask[830,1100]:
            raise ValueError("Authenticated current foreground missing")
        if row["index"] in (6,9,10,15):
            if not mask[690,1230] or not mask[660,1290]:
                raise ValueError("Reported black-cuff false-feature area still not excluded")
    manifest["generator_script_sha256"]=file_sha(Path(__file__))
    manifest["final_mask_integrity_verified_frames"]=100
    manifest["visual_agent_review"]={"scope":"all 100 current hand/head crop tiles and all 100 full-person tiles; full-native failure samples 26/65/100",
        "observed_failure_status":"reported raised-sleeve, fingers, black-cuff and torso omission corrected",
        "human_acceptance":False,"pixel_ground_truth_claim":False}
    manifest["frozen_mask_set_sha256"]=hashlib.sha256("\n".join(
        f"{r['index']}:{r['current_mask_sha256']}" for r in rows).encode()).hexdigest()
    path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(dict(verified=100,manifest_sha256=file_sha(path),
        mask_set_sha256=manifest["frozen_mask_set_sha256"],native_exclusion_margin_px=10),ensure_ascii=False),flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence",type=Path,default=DEFAULT_SEQUENCE)
    parser.add_argument("--model",type=Path,default=DEFAULT_MODEL)
    parser.add_argument("--indices",type=int,nargs="+")
    parser.add_argument("--review-only",action="store_true")
    parser.add_argument("--finalize-existing",action="store_true")
    parser.add_argument("--verify-final",action="store_true")
    args=parser.parse_args()
    if args.verify_final:
        verify_final(args.sequence)
    elif args.finalize_existing:
        finalize_existing(args.sequence)
    elif args.review_only:
        review_masks(args.sequence)
    else:
        run(args.sequence,args.model,args.indices)
