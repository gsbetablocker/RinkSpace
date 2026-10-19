"""Current native-image SAM2 person exclusion for official frames101–199.

Reviewed source-image landmarks navigate prompts, followed by nearest actual
current skin/dark pixels. Every region independently encodes the current RGB;
no historical mask, frame memory, or rink projection is read.
"""
from __future__ import annotations
import argparse
import hashlib
import json
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
from rinkspace.vision_instances import Sam2SourceInstances
from rinkspace.programme_source_stream import file_sha
from calibration.program_tracking.infer_current_person_exclusion_masks import DEFAULT_MODEL
from calibration.program_tracking.prepare_continuation_frame_sources import DATA,OUT,read,write,navigation

# Source-image visual prompt navigation only, not measured contours or rink points.
# Entries are head, body, two hands, two shoulders, and visible lower-leg seeds.
ANCHORS={
101: [[1020,435],[1100,900],[1470,230],[950,1050],[1155,360],[850,760],[]],
125: [[1080,465],[1080,880],[1450,90],[1070,1020],[1160,335],[900,780],[]],
150: [[1090,400],[1020,800],[1430,85],[925,910],[1150,305],[950,580],[]],
158: [[1120,340],[1000,760],[1190,100],[950,915],[1150,290],[1000,570],[]],
165: [[1190,320],[1170,730],[1120,105],[1030,910],[1190,365],[1050,580],[[930,1060],[1280,1060]]],
175: [[1240,285],[1210,730],[695,155],[1450,495],[1040,365],[1320,420],[[1090,1020],[1400,1010]]],
181: [[1180,245],[1230,720],[650,300],[1700,370],[1050,375],[1340,360],[[1180,1000],[1370,1000]]],
183: [[1160,245],[1230,710],[690,350],[1730,345],[1050,385],[1340,350],[[1190,1000],[1340,1000]]],
185: [[1135,255],[1220,695],[770,420],[1740,315],[1060,400],[1340,345],[[1190,1000],[1330,1000]]],
187: [[1130,250],[1210,675],[900,450],[1690,315],[1080,420],[1310,340],[[1180,1000],[1310,1000]]],
190: [[1080,245],[1180,650],[1100,490],[1500,320],[1100,415],[1240,335],[[1170,1000],[1250,1000]]],
192: [[1080,245],[1150,660],[1330,345],[1280,535],[1220,340],[1160,440],[[1130,1000],[1200,1000]]],
193: [[1080,245],[1150,650],[1110,420],[1380,535],[1090,420],[1220,405],[[1110,1000],[1210,1000]]],
194: [[1080,245],[1150,670],[1120,420],[1480,540],[1090,420],[1230,420],[[1110,1000],[1200,1000]]],
195: [[1110,245],[1115,720],[860,420],[1530,540],[1060,365],[1230,405],[[1110,1015],[1200,1015]]],
199: [[1150,235],[1100,650],[600,445],[1520,550],[960,325],[1240,345],[[1070,940],[1190,975]]],
}
ARM_ELBOWS={101:[1310,85],125:[1325,35],150:[1335,165],158:[1180,205],165:[1100,235],175:[865,285],185:[855,430],195:[970,545],199:[835,400]}
PHOTOGRAPHER_NEGATIVES={101:[1580,80],125:[1500,70],150:[1320,60],158:[1300,60],165:[1260,65],175:[1250,70],185:[1180,70],195:[1110,70],199:[1090,95]}

def navigation_point(table,index):
    keys=sorted(table);left=max(k for k in keys if k<=index);right=min(k for k in keys if k>=index)
    t=0 if left==right else (index-left)/(right-left)
    return (np.array(table[left])*(1-t)+np.array(table[right])*t).tolist()

def interpolated(index):
    keys=sorted(ANCHORS);left=max(k for k in keys if k<=index);right=min(k for k in keys if k>=index)
    t=0 if left==right else (index-left)/(right-left)
    result=[]
    for a,b in zip(ANCHORS[left][:6],ANCHORS[right][:6]):result.append((np.array(a)*(1-t)+np.array(b)*t).tolist())
    result.append(ANCHORS[left][6] if t<.5 else ANCHORS[right][6])
    return result

def source_materials(rgb):
    hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV);ycc=cv2.cvtColor(rgb,cv2.COLOR_RGB2YCrCb)
    spread=rgb.max(2).astype(int)-rgb.min(2).astype(int)
    dark=(rgb.max(2)<125)&(spread<40)&(hsv[...,1]<100)
    skin=(hsv[...,0]<=22)&(hsv[...,1]>=20)&(hsv[...,1]<=185)&(hsv[...,2]>=80)&(ycc[...,1]>=138)&(ycc[...,1]<=195)&(ycc[...,2]>=95)&(ycc[...,2]<=135)
    return dark,skin,hsv,spread

def current_point(material,target,radius=100):
    x,y=map(float,target);x0=max(0,int(x-radius));x1=min(1920,int(x+radius)+1);y0=max(0,int(y-radius));y1=min(1080,int(y+radius)+1)
    crop=material[y0:y1,x0:x1].astype(np.uint8)
    n,labels,stats,centroids=cv2.connectedComponentsWithStats(crop,8)
    choices=[i for i in range(1,n)if stats[i,cv2.CC_STAT_AREA]>=25
        and not(min(stats[i,cv2.CC_STAT_WIDTH],stats[i,cv2.CC_STAT_HEIGHT])<45
            and max(stats[i,cv2.CC_STAT_WIDTH],stats[i,cv2.CC_STAT_HEIGHT])>5*min(stats[i,cv2.CC_STAT_WIDTH],stats[i,cv2.CC_STAT_HEIGHT]))]
    if not choices:raise ValueError(f"No current source prompt material near {target}")
    label=min(choices,key=lambda i:np.linalg.norm(centroids[i]+[x0,y0]-[x,y]))
    dist=cv2.distanceTransform((labels==label).astype(np.uint8),cv2.DIST_L2,5)
    yy,xx=np.nonzero(dist>=min(5.,max(1.,float(dist.max())*.5)))
    points=np.c_[xx+x0,yy+y0];point=points[np.argmin(np.sum((points-[x,y])**2,axis=1))]
    return point.astype(float).tolist()

def mask_one(provider,row):
    raw=cv2.imread(row["image_path"]);rgb=cv2.cvtColor(raw,cv2.COLOR_BGR2RGB)
    if file_sha(row["image_path"])!=row["png_sha256"] or hashlib.sha256(rgb.tobytes()).hexdigest()!=row["decoded_native_RGB_sha256"]:
        raise ValueError("Authenticated current source changed")
    index=row["index"];head,body,hand1,hand2,shoulder1,shoulder2,legs=interpolated(index)
    dark,skin,hsv,spread=source_materials(rgb)
    prompt_ycc=cv2.cvtColor(rgb,cv2.COLOR_RGB2YCrCb)
    current_hand_skin=(hsv[...,0]<=25)&(hsv[...,1]>=25)&(hsv[...,1]<=130)&(hsv[...,2]>=80)&(prompt_ycc[...,1]>=130)&(prompt_ycc[...,1]<=190)&(prompt_ycc[...,2]>=110)&(prompt_ycc[...,2]<=135)
    headp=current_point(skin,head,90);bodyp=current_point(dark,body,120)
    hands=[];hand_skin_missing=[];hand_dark_missing=[]
    for ordinal,point in enumerate((hand1,hand2),1):
        try:hands.append(current_point(current_hand_skin,point,100))
        except ValueError:
            try:hands.append(current_point(dark,point,120))
            except ValueError:
                hands.append(bodyp)
                hand_dark_missing.append(ordinal)
            hand_skin_missing.append(ordinal)
    legp=[];missing_leg_searches=[]
    for point in legs:
        try:legp.append(current_point(skin,point,100))
        except ValueError:missing_leg_searches.append(point)
    binding=dict(source_sha256=row["source"]["source_video_sha256"],source_pts=row["native_pts"],
        time_base="1/12800",native_hw=[1080,1920],shot_id="official50_continuation_current_person",
        decoded_native_rgb_sha256=row["decoded_native_RGB_sha256"])
    # Negative source RGB prompts distinguish foreground from spectators and
    # the white/purple advertisement on both sides of the moving person.
    photo_negative=navigation_point(PHOTOGRAPHER_NEGATIVES,index)
    negatives=[[500.,400.],[1800.,650.],[600.,700.],[1700.,940.],photo_negative]
    for p in [[1250.,60.],[1550.,60.],[900.,100.],[1100.,100.],[1600.,260.]]:
        if min(np.linalg.norm(np.array(p)-q)for q in hands+[headp])>150:negatives.append(p)
    positives=[headp,bodyp]+hands+legp
    allp=np.array([headp,bodyp]+hands+legp+[hand1,hand2])
    box=[max(400.,float(allp[:,0].min())-150),max(0.,float(allp[:,1].min())-160),
         min(1920.,float(allp[:,0].max())+150),1080.]
    specs=[dict(region="whole_current_person",box_xyxy=box,points_uv=positives+negatives,
                point_labels=[1]*len(positives)+[0]*len(negatives))]
    for ordinal,(hand,shoulder)in enumerate(zip(hands,[shoulder1,shoulder2])):
        source_arm=[]
        for fraction in (.25,.55,.8):
            target=np.array(hand)*(1-fraction)+np.array(shoulder)*fraction
            try:source_arm.append(current_point(dark,target,65))
            except ValueError:pass
        elbow=(np.array(hand)+np.array(shoulder))*.5
        if index<=158:
            elbow=(np.array(navigation_point(ARM_ELBOWS,index)) if ordinal==0 else
                np.array([max(650.,shoulder2[0]-170),min(1079.,shoulder2[1]+200)]))
        try:source_arm.append(current_point(dark,elbow,75))
        except ValueError:pass
        pp=[hand]+source_arm
        endpoints=np.array([hand,shoulder,elbow]);lo=np.maximum([400,0],endpoints.min(0)-100);hi=np.minimum([1920,1080],endpoints.max(0)+100)
        local_neg=[p for p in negatives if lo[0]<=p[0]<hi[0]and lo[1]<=p[1]<hi[1]]
        specs.append(dict(region=f"current_sleeve_hand_{ordinal+1}",box_xyxy=np.r_[lo,hi].tolist(),
            points_uv=pp+local_neg,point_labels=[1]*len(pp)+[0]*len(local_neg)))
    mask=np.zeros((1080,1920),np.uint8);executed=[]
    for spec in specs:
        prompt=dict(object_id=1,kind="occluder",origin="original_rgb_candidate",source_binding=binding,
            box_xyxy=spec["box_xyxy"],points_uv=spec["points_uv"],point_labels=spec["point_labels"],exclusion_authority=None)
        result=provider.read(rgb,binding,prompts=[prompt],reset=True)
        candidate=result["occluder_region_candidate_mask"].detach().cpu().numpy()
        mask[candidate]=255;executed.append(dict(region=spec["region"],prompt=prompt,mask_pixels=int(candidate.sum())))
    n,labels,stats,_=cv2.connectedComponentsWithStats((mask>0).astype(np.uint8),8)
    bodylabel=int(labels[int(bodyp[1]),int(bodyp[0])])
    if not bodylabel:raise ValueError("Current torso omitted by SAM2")
    keep={bodylabel}
    for x,y in positives:
        label=int(labels[int(y),int(x)])
        if label:keep.add(label)
    mask=np.isin(labels,list(keep)).astype(np.uint8)*255
    # SAM's skin preference omits the actual translucent black sleeves. Add
    # current neutral-dark material only in prompt-navigation corridors around
    # these same current arms. The corridors supply no boundary coordinates.
    corridor=np.zeros_like(mask)
    elbow1=(np.array(hand1)+np.array(shoulder1))*.5
    elbow2=(np.array(hand2)+np.array(shoulder2))*.5
    if index<=158:
        elbow1=np.array(navigation_point(ARM_ELBOWS,index))
        elbow2=np.array([max(650.,shoulder2[0]-170),min(1079.,shoulder2[1]+200)])
    for shoulder,elbow,hand in [(shoulder1,elbow1,hand1),(shoulder2,elbow2,hand2)]:
        line=np.rint([shoulder,elbow,hand]).astype(np.int32)
        cv2.polylines(corridor,[line],False,255,135,lineType=cv2.LINE_8)
        for point in line:cv2.circle(corridor,tuple(point),67,255,-1)
    source_black_scope=((rgb.max(2)<110)&(spread<40)&(hsv[...,1]<100)&(corridor>0)).astype(np.uint8)
    n,black_labels,black_stats,_=cv2.connectedComponentsWithStats(source_black_scope,8)
    source_black=np.zeros_like(mask);black_areas=[]
    existing_near=cv2.dilate(mask,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(41,41)))>0
    for label in range(1,n):
        x,y,w,h,area=black_stats[label]
        if int(area)<100 or (min(w,h)<15 and max(w,h)>4*min(w,h)):continue
        component=(black_labels==label).astype(np.uint8)
        if not bool((component.astype(bool)&existing_near).any()):continue
        nx,ny=np.rint(photo_negative).astype(int)
        if component[max(0,ny-20):min(1080,ny+21),max(0,nx-20):min(1920,nx+21)].any():continue
        contours,_=cv2.findContours(component,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(source_black,contours,-1,255,cv2.FILLED)
        black_areas.append(int(black_stats[label,cv2.CC_STAT_AREA]))
    mask[source_black>0]=255
    # Repair only current dark/skin pockets close to SAM foreground. Saturated
    # blue/purple board and remote white print cannot become person material.
    ycc=cv2.cvtColor(rgb,cv2.COLOR_RGB2YCrCb)
    visible_skin=(hsv[...,0]<=27)&(hsv[...,1]>=20)&(hsv[...,1]<=185)&(hsv[...,2]>=80)&(ycc[...,1]>=130)&(ycc[...,1]<=195)&(ycc[...,2]>=95)&(ycc[...,2]<=135)
    material=dark|visible_skin
    near=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(31,31)))>0
    pocket=near&material&(mask==0);mask[pocket]=255
    count,hlabels,hstats,_=cv2.connectedComponentsWithStats((mask==0).astype(np.uint8),8)
    filled=[]
    board=((hsv[...,0]>=125)&(hsv[...,0]<=178)&(spread>55))|((hsv[...,1]<60)&(hsv[...,2]>155))
    for label in range(1,count):
        x,y,w,h,area=hstats[label]
        if x==0 or y==0 or x+w==1920 or y+h==1080:continue
        hole=hlabels==label
        if float(board[hole].mean())<.6:mask[hole]=255;filled.append(int(area))
    # Remove SAM-admitted remote advertisement ink using current RGB only.
    material_near=cv2.dilate(material.astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)))>0
    remote_board=(mask>0)&board&~material_near
    mask[remote_board]=0
    # Connected native white print/board material extending outside the SAM
    # person establishes an external background component. Reject its admitted
    # blurred fragments before adding the requested foreground margin.
    white=(hsv[...,1]<95)&(hsv[...,2]>140)
    white[:180]=False;white[700:]=False;white[:,:450]=False
    white_protection=material.copy()
    if 178<=index<=193:
        arm_zone=np.zeros_like(mask,bool)
        arm_zone[:,int(max(shoulder2[0]+25,bodyp[0]+150)):]=True
        arm_zone[:,:int(min(shoulder1[0]-25,bodyp[0]-150))]=True
        sn,sl,_,_=cv2.connectedComponentsWithStats(current_hand_skin.astype(np.uint8),8)
        current_hands=np.zeros_like(mask,np.uint8)
        for hx,hy in hands:
            label=int(sl[int(hy),int(hx)])
            if label:current_hands[sl==label]=1
        hand_support=cv2.dilate(current_hands,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11)))>0
        white_protection[arm_zone]=(dark|hand_support)[arm_zone]
    wn,wl,ws,_=cv2.connectedComponentsWithStats(white.astype(np.uint8),8)
    rejected_white=0
    for label in range(1,wn):
        if ws[label,cv2.CC_STAT_AREA]<200:continue
        component=wl==label
        if int((component&(mask==0)).sum())<500:continue
        support=cv2.dilate(component.astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11)))>0
        reject=(mask>0)&support&~white_protection
        rejected_white+=int(reject.sum());mask[reject]=0
    yellow=(hsv[...,0]>=15)&(hsv[...,0]<=40)&(hsv[...,1]>=80)&(hsv[...,2]>=100)
    yn,yl,ys,_=cv2.connectedComponentsWithStats(yellow.astype(np.uint8),8)
    rejected_yellow=0
    for label in range(1,yn):
        x,y,w,h,area=ys[label]
        if w<500 or h>100 or not 350<y<700:continue
        support=cv2.dilate((yl==label).astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11)))>0
        reject=(mask>0)&support&~dark
        rejected_yellow+=int(reject.sum());mask[reject]=0
    mask=cv2.dilate(mask,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(21,21)))
    reviewed_static_patches=[]
    if index==178:
        # Native source review places this T fragment above the visible sleeve
        # and its 10px foreground margin. This rejects background only.
        x1,y1,x2,y2=1442,345,1480,384
        removed=int((mask[y1:y2,x1:x2]>0).sum())
        mask[y1:y2,x1:x2]=0
        reviewed_static_patches.append(dict(rectangle_xyxy=[x1,y1,x2,y2],
            source_material="static advertisement T upper fragment, visibly above current sleeve by more than native10px margin",
            removed_pixels=removed,authority="current original native visual inspection; exclusion only"))
    for x,y in positives:
        if not mask[int(y),int(x)]:raise ValueError(f"Current source foreground prompt omitted in {index}: {(x,y)}")
    output=DATA/"masks_current"/f"{index:06d}.png";cv2.imwrite(str(output),mask,[cv2.IMWRITE_PNG_COMPRESSION,1])
    overlay=raw.copy();overlay[mask>0]=(overlay[mask>0].astype(float)*.45+np.array([150,190,40])*.55).astype(np.uint8)
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(overlay,contours,-1,(0,255,255),2)
    cv2.imwrite(str(DATA/"masks_current"/f"review_{index:06d}.png"),overlay)
    receipt=dict(schema="rinkspace.current-native-source-person-exclusion.v1",index=index,source_pts=row["native_pts"],
        time_base="1/12800",source_time=row["source_time"],original_png_sha256=row["png_sha256"],
        original_RGB_sha256=row["decoded_native_RGB_sha256"],mask_path=str(output),mask_sha256=file_sha(output),
        independent_current_SAM_encodings=executed,current_original_RGB_independently_encoded=True,
        current_SAM2_inference_executed=True,prior_frame_memory_used=False,historical_masks_used=False,
        native_dilation_radius_px=10,dilation_kernel_wh=[21,21],dilation_kernel_shape="ellipse",
        source_colour_pocket_pixels_added=int(pocket.sum()),filled_foreground_hole_areas=filled,
        current_source_neutral_dark_sleeve_component_areas=black_areas,
        neutral_dark_sleeve_navigation_corridor_only=True,
        remote_source_board_material_pixels_removed=int(remote_board.sum()),foreground_pixels=int((mask>0).sum()),
        connected_external_current_white_print_or_board_pixels_removed=rejected_white,
        long_static_current_yellow_board_foot_pixels_removed=rejected_yellow,
        reviewed_static_source_patch_rejections=reviewed_static_patches,
        prompt_navigation_origin="visually_reviewed_current-source-landmarks_interpolated_then_snapped_to_each_current_RGB_skin_or_neutral_dark_material",
        leg_navigation_searches_without_current_skin_pixels=missing_leg_searches,
        hand_navigation_searches_using_visible_current_dark_cuff_instead_of_unavailable_skin=hand_skin_missing,
        hand_navigation_searches_without_visible_skin_or_dark_material_body_seed_reused=hand_dark_missing,
        source_prompt_coordinates_are_boundary_measurements=False,foreground_check_points=positives,
        mask_is_conservative_exclusion_candidate_not_pixel_ground_truth=True,human_acceptance=False,
        source_points_or_solver_modified=False,checkpoint=dict(path=str(DEFAULT_MODEL),sha256=provider.model_sha256,download_performed=False))
    write(DATA/"mask_provenance"/f"{index:06d}.json",receipt)
    return receipt

def run(indices=None):
    started=time.perf_counter();torch.set_num_threads(4)
    manifest=read(DATA/"frame_manifest.json");rows={r["index"]:r for r in manifest["frames"]}
    selected=list(range(101,200))if indices is None else indices
    selected=[i for i in [101,125,150,175,199]if i in selected]+[i for i in selected if i not in [101,125,150,175,199]]
    provider=Sam2SourceInstances(DEFAULT_MODEL,device="cuda",reread_rgb=False,contour_stride=16)
    for ordinal,index in enumerate(selected,1):
        receipt=mask_one(provider,rows[index])
        rows[index]["mask_sha256"]=receipt["mask_sha256"]
        rows[index]["source"].update(mask_path=receipt["mask_path"],mask_sha256=receipt["mask_sha256"])
        print(json.dumps(dict(stage="current_mask_ready",index=index,pixels=receipt["foreground_pixels"],completed=ordinal,total=len(selected),elapsed=time.perf_counter()-started)),flush=True)
    records=[read(path)for path in sorted((DATA/"mask_provenance").glob("[0-9]*.json"))]
    canonical=read(DATA.parent/"frame_000200_50fps/person_mask_provenance.json")
    canonical.update(index=200,mask_path=str(DATA/"masks_current/000200.png"),canonical200_reused_unchanged=True)
    records.append(canonical)
    for record in records:
        index=record["index"]
        rows[index]["mask_sha256"]=record["mask_sha256"]
        rows[index]["source"].update(mask_path=record["mask_path"],mask_sha256=record["mask_sha256"])
    rows[200]["mask_sha256"]=canonical["mask_sha256"]
    rows[200]["source"].update(mask_path=canonical["mask_path"],mask_sha256=canonical["mask_sha256"])
    write(DATA/"frame_manifest.json",manifest)
    write(DATA/"masks_current/current_mask_manifest.json",dict(schema="rinkspace.current-native-source-person-exclusion.v1",
        source_manifest=dict(path=str(DATA/"frame_manifest.json"),sha256=file_sha(DATA/"frame_manifest.json")),
        rows=records,current_source_inference_per_frame=True,prior_frame_memory_used=False,historical_60fps_masks_used=False,
        canonical200_reused_unchanged=True,mask_count=len(records),elapsed_seconds=time.perf_counter()-started))
    navigation(selected+[200])
    tiles=[]
    for index in sorted([r["index"]for r in records]):
        path=DATA/"masks_current"/f"review_{index:06d}.png"
        if index==200:path=DATA.parent/"frame_000200_50fps/person_mask_review.png"
        tile=cv2.resize(cv2.imread(str(path)),(384,216));cv2.putText(tile,str(index),(4,17),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1);tiles.append(tile)
    cols=5;rows_n=(len(tiles)+cols-1)//cols;sheet=np.zeros((rows_n*216,cols*384,3),np.uint8)
    for i,tile in enumerate(tiles):y,x=divmod(i,cols);sheet[y*216:(y+1)*216,x*384:(x+1)*384]=tile
    cv2.imwrite(str(DATA/"masks_current/contact_sheet.png"),sheet)

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--indices",type=int,nargs="*")
    args=parser.parse_args();run(args.indices)
