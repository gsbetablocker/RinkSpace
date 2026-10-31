"""Source-bound two-boot observation preparation; no camera/scene updates."""
from pathlib import Path
import argparse,json,hashlib,time
import cv2
import numpy as np
from .rebuild_prepare import load,digest
from rinkspace.review.render import source_frames


def prepare(root,project,review_path,dataset=None):
    root,project=Path(root),Path(project);cache=root/'footwear_v2';cache.mkdir(exist_ok=True);out=Path(dataset) if dataset else cache;out.mkdir(exist_ok=True);(cache/'source_crops').mkdir(exist_ok=True)
    p=load(root/'input.npz');base=load(root/'pose/result.npz');meta=json.loads((root/'input.json').read_text(encoding='utf8'))
    review_path=Path(review_path);review=json.loads(review_path.read_text(encoding='utf8'));scope=json.loads((review_path.parent/review['source_scope']).read_text(encoding='utf8'));source={**scope['source'],**scope['source_clock']}
    # Unseen detailed boot observations in contiguous time windows. These
    # pixels do not calibrate shape, pose choices, or residual scales.
    holdouts=review['check_windows_s'];guard=review['check_guard_s']
    requested=np.asarray(review['fit_source_seconds'])
    selected=set(int(p['pts'][np.argmin(abs(p['pts']/12800-t))]) for t in requested)
    for lo,hi in holdouts:selected.update(p['pts'][(p['pts']/12800>=lo)&(p['pts']/12800<=hi)].tolist())
    rows=[]
    for pts,bgr in source_frames(source['registered_path'],source,p['pts']):
        if pts not in selected:continue
        i=int(np.searchsorted(p['pts'],pts));t=pts/12800
        hold=any(lo<=t<=hi for lo,hi in holdouts);protected=any(lo-guard<=t<=hi+guard for lo,hi in holdouts)
        role='check' if hold else ('protected' if protected else 'fit')
        qualified=p['source_qualified'][i];xy=p['source_uv'][i];bh=p['source_bbox'][i,3]-p['source_bbox'][i,1]
        foot_ids=np.array([13,14,15,16]);good=qualified[foot_ids]
        if p['lower_body_inferred'][i] or not good.all():
            rows.append(dict(pts=pts,role=role,status='unknown_lower_body',source_seconds=t));continue
        lo=np.maximum(np.floor(xy[foot_ids].min(0)-[bh*.22,bh*.08]).astype(int),[0,0]);hi=np.minimum(np.ceil(xy[foot_ids].max(0)+[bh*.22,bh*.24]).astype(int),[1920,1080])
        crop=bgr[lo[1]:hi[1],lo[0]:hi[0]].copy();file=cache/'source_crops'/f'{pts}.png'
        if not file.exists():
            if not cv2.imwrite(str(file),crop):raise OSError(file)
        ankle=xy[[15,16]];knee=xy[[13,14]];centres=ankle+(ankle-knee)*.08
        boxes=np.c_[centres-[bh*.13,bh*.085],centres+[bh*.13,bh*.16]]
        boxes[:,:2]=np.maximum(boxes[:,:2],lo);boxes[:,2:]=np.minimum(boxes[:,2:],hi)
        rows.append(dict(pts=pts,source_seconds=t,role=role,status='source_crop_available',crop=str(file),crop_xyxy=[*lo.tolist(),*hi.tolist()],
            crop_sha256=digest(file),decoded_native_crop_rgb_sha256=hashlib.sha256(crop[:,:,::-1].tobytes()).hexdigest(),
            ankle_source_uv=ankle.tolist(),knee_source_uv=knee.tolist(),search_centres_uv=centres.tolist(),search_boxes_uv=boxes.tolist(),
            source_left_right='saved COCO physical identity candidate; crossing/occlusion must be checked',body_model_projection_used_as_measurement=False))
    report=dict(schema='rinkspace.two-boot-observation-crops.v1',source=source,shot='s01',base_body_sha256=digest(root/'pose/result.npz'),
        input_sha256=digest(root/'input.npz'),camera_snapshot_sha256=digest(root/'camera_snapshot.npz'),
        detailed_boot_check_windows_s=holdouts,check_guard_s=guard,source_time_base='1/12800',rows=rows,
        inference='none at preparation; boxes navigate only; unchanged decoded RGB cropped without resizing',
        camera_and_scene_frozen=True,base_pelvis_frozen=True)
    (out/'source_index.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    # Source-only atlas omits check windows until calibration decisions freeze.
    usable=[r for r in rows if r['status']=='source_crop_available' and r['role']=='fit']
    for page,start in enumerate(range(0,len(usable),12)):
        tiles=[]
        for r in usable[start:start+12]:
            crop=cv2.imread(r['crop']);box=np.full((340,480,3),24,np.uint8);s=min(470/crop.shape[1],295/crop.shape[0]);im=cv2.resize(crop,None,fx=s,fy=s,interpolation=cv2.INTER_CUBIC)
            box[40:40+im.shape[0],5:5+im.shape[1]]=im;cv2.putText(box,f'{r["source_seconds"]:.3f}s / {r["pts"]}',(10,25),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),1,cv2.LINE_AA)
            for foot,xy in enumerate(r['ankle_source_uv']):
                at=np.rint((np.asarray(xy)-r['crop_xyxy'][:2])*s+[5,40]).astype(int);cv2.circle(box,tuple(at),5,(0,210,255) if foot==0 else (255,180,80),1,cv2.LINE_AA);cv2.putText(box,'LR'[foot],tuple(at+[8,-5]),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,255),1,cv2.LINE_AA)
            tiles.append(box)
        while len(tiles)%3:tiles.append(np.full_like(tiles[0],24))
        cv2.imwrite(str(out/f'source_atlas_{page}.jpg'),np.vstack([np.hstack(tiles[k:k+3]) for k in range(0,len(tiles),3)]),[cv2.IMWRITE_JPEG_QUALITY,94])
    print(json.dumps(dict(rows=len(rows),fit=sum(r['role']=='fit' for r in rows),check=sum(r['role']=='check' for r in rows),protected=sum(r['role']=='protected' for r in rows),atlases=(len(usable)+11)//12)),flush=True)

def candidate_atlas(root):
    out=Path(root)/'footwear_v2';index=json.loads((out/'source_index.json').read_text(encoding='utf8'))
    rows=[r for r in index['rows'] if r['role']=='fit' and (out/'mask_candidates'/str(r['pts'])).with_name(str(r['pts'])+'_qa.jpg').exists()]
    for page,start in enumerate(range(0,len(rows),12)):
        tiles=[]
        for row in rows[start:start+12]:
            image=cv2.imread(str(out/'mask_candidates'/(str(row['pts'])+'_qa.jpg')));box=np.full((300,420,3),24,np.uint8)
            scale=min(410/image.shape[1],260/image.shape[0]);image=cv2.resize(image,None,fx=scale,fy=scale)
            box[35:35+image.shape[0],5:5+image.shape[1]]=image
            cv2.putText(box,str(row['source_seconds'])+'s / '+str(row['pts']),(8,23),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1,cv2.LINE_AA);tiles.append(box)
        while len(tiles)%3:tiles.append(np.full_like(tiles[0],24))
        cv2.imwrite(str(out/('mask_atlas_'+str(page)+'.jpg')),np.vstack([np.hstack(tiles[k:k+3]) for k in range(0,len(tiles),3)]))
    print(len(rows),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--project',type=Path,required=True);ap.add_argument('--review',type=Path,required=True);ap.add_argument('--dataset',type=Path);a=ap.parse_args();prepare(a.root,a.project,a.review,a.dataset)
