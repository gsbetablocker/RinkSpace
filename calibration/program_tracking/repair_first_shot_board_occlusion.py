"""Separate visible board print from foreground in current native SAM reads."""
from __future__ import annotations
from copy import deepcopy
import cv2
import numpy as np
from .infer_current_person_exclusion_masks import Sam2SourceInstances, DEFAULT_MODEL
from .repair_first_shot_board_patterns import (DATA,MODEL,REPAIR,TARGET,read,write,sha,
    feature_world,project_fast,sample_outer_ink)

# Directly inspected, currently visible board print beside the head/shoulders.
# These are segmentation background prompts, never fitting observations.
VISIBLE_NATIVE_BOARD_PROMPTS = {
    830: [[781,510],[774,536],[886,530],[889,552],[887,574]],
    831: [[787,515],[782,534],[899,533],[901,555],[896,579]],
    832: [[796,515],[789,530],[909,532],[913,552],[910,574]],
}


def repair_masks(indices):
    destination=REPAIR/'masks';destination.mkdir(parents=True,exist_ok=True)
    diagnostics=REPAIR/'board_diagnostics';diagnostics.mkdir(exist_ok=True)
    masks={r['index']:r for r in read(DATA/'masks_current/current_mask_manifest.json')['rows']}
    manifest={r['index']:r for r in read(DATA/'frame_manifest.json')['frames']}
    archive=np.load(DATA/'camera_seeds.npz');cameras=dict(zip(archive['programme_indices'],archive['camera25']))
    world=feature_world(read(MODEL/'scene.json'),read(MODEL/'feature_registry.json')['targets'][TARGET],read(MODEL/'ad2_material_template.json'))
    provider=Sam2SourceInstances(DEFAULT_MODEL,device='cuda',reread_rgb=False,contour_stride=16)
    receipts=[]
    for index in indices:
        record=manifest[index];raw=cv2.imread(record['image_path']);old=cv2.imread(record['mask_path'],0)
        path,depth=project_fast(cameras[index],world);center=(path.min(0)+path.max(0))/2
        rows=sample_outer_ink(raw,np.zeros(old.shape,bool),path,center,32)
        negative=[]
        for j,row in enumerate(rows):
            if row is None:continue
            uv=row[0];normal=uv-center;normal/=np.linalg.norm(normal)
            whites=[]
            for offset in (-4,5):
                x,y=np.rint(uv+normal*offset).astype(int)
                if x<2 or x>1916 or y<2 or y>1076:break
                patch=raw[y-1:y+2,x-1:x+2].astype(float)
                whites.append(bool(np.mean(patch)>165 and np.mean(np.ptp(patch,axis=2))<40))
            if whites!=[True,True]:continue
            point=uv+normal*5
            if all(np.linalg.norm(point-p)>18 for p in negative):negative.append(point)
        if len(negative)<2:raise ValueError(f'{index}: no independently visible printed-ring background prompts')
        negative.extend(np.asarray(p,float)for p in VISIBLE_NATIVE_BOARD_PROMPTS.get(index,[]))
        prompt=deepcopy(masks[index]['current_source_prompt'])
        prompt['points_uv']+=np.asarray(negative).tolist();prompt['point_labels']+=[0]*len(negative)
        prompt['current_native_thin_print_with_white_board_on_both_sides_background_only']=True
        result=provider.read(cv2.cvtColor(raw,cv2.COLOR_BGR2RGB),prompt['source_binding'],prompts=[prompt],reset=True)
        new=result['occluder_region_candidate_mask'].detach().cpu().numpy().astype('uint8')*255
        dilation=3 if index in VISIBLE_NATIVE_BOARD_PROMPTS else 10
        new=cv2.dilate(new,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*dilation+1,2*dilation+1)))
        # Do not replace valid legs, boots, hands or any other source region.
        low=np.floor(path.min(0)-20).astype(int);high=np.ceil(path.max(0)+20).astype(int)
        x0,y0=np.maximum(low,0);x1,y1=np.minimum(high,[1920,1080]);merged=old.copy();merged[y0:y1,x0:x1]=new[y0:y1,x0:x1]
        if index in VISIBLE_NATIVE_BOARD_PROMPTS:
            # Native review shows the head is foreground.  A logo-background
            # negative can fragment it; preserve the original conservative
            # head exclusion rather than expose any of those pixels.
            head_x={830:846,831:850,832:853}[index]
            merged[474:545,head_x-32:head_x+32] |= old[474:545,head_x-32:head_x+32]
        output=destination/f'{index:06d}.png';cv2.imwrite(str(output),merged,[cv2.IMWRITE_PNG_COMPRESSION,1])
        overlay=raw.copy();overlay[merged>0]=(overlay[merged>0].astype(float)*.4+np.array([0,220,220])*.6).astype('uint8')
        cv2.imwrite(str(diagnostics/f'person_mask_repair_{index:06d}.png'),overlay,[cv2.IMWRITE_PNG_COMPRESSION,1])
        receipts.append(dict(index=index,path=str(output),sha256=sha(output),parent_mask_path=record['mask_path'],parent_mask_sha256=sha(record['mask_path']),
                             source_only=True,current_source_SAM_reset=True,model_checkpoint_sha256=provider.model_sha256,
                             native_person_dilation_radius_px=dilation,edited_native_roi_xyxy=[int(x0),int(y0),int(x1),int(y1)],
                             static_print_background_prompts=np.asarray(negative).tolist(),prompt=prompt,
                             removed_false_foreground_pixels=int(((old>0)&(merged==0)).sum()),added_foreground_pixels=int(((old==0)&(merged>0)).sum()),
                             original_mask_unchanged=True,mask_is_pixel_ground_truth=False))
        print(index,'ring prompts',len(negative),'removed',receipts[-1]['removed_false_foreground_pixels'],flush=True)
    receipt_path=destination/'Ad11_mask_repair_receipt.json'
    previous=read(receipt_path)['rows']if receipt_path.is_file()else[]
    changed=set(indices)
    combined=sorted([r for r in previous if r['index']not in changed]+receipts,key=lambda r:r['index'])
    write(receipt_path,dict(schema='rinkspace.current-source-Ad11-false-occlusion-repair.v1',rows=combined))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--indices',required=True);args=parser.parse_args()
    repair_masks([int(i)for i in args.indices.split(',')])
