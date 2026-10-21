import os
for k in('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,hashlib,time,gzip,argparse
from pathlib import Path
P=Path(r'D:/OneDrive/codex_projs/RinkSpace');R=Path(r'D:/CodexData/RinkSpace_Data/runs/okada_rec_1019/recover_v2');O=R/'dense56_footwear_v1';O.mkdir(exist_ok=True)
sys.path[:0]=[str(P),str(P/'src'),str(P/'calibration/rinkfit/src')]
from calibration.program_tracking.runtime.bootstrap import add_geometry_dependencies
add_geometry_dependencies()
import cv2,numpy as np,av,torch
from rinkspace.vision_instances import Sam2SourceInstances
cv2.setNumThreads(1);torch.set_num_threads(1)
TH=hashlib.sha256
source=json.load(open(R.parent.parent/'okada_joint9_1018a/programme_scope.json',encoding='utf8'))['source']
MASK_INDEX={json.loads(l)['native_pts']:json.loads(l)for l in gzip.open(R/'dense56_masks_v1/index.jsonl.gz','rt',encoding='utf8')}
def propose(rgb,mask,pts):
 boxes=MASK_INDEX[pts].get('current_boxes',[])
 if not boxes:return []
 box=max([p['box_xyxy']for p in boxes],key=lambda a:a[3]-a[1]);bx0=max(0,int(box[0])-35);bx1=min(1920,int(box[2])+36);by0=max(0,int(box[1])-25);by1=min(1080,int(box[3])+101);body=np.zeros_like(mask);body[by0:by1,bx0:bx1]=mask[by0:by1,bx0:bx1];ys,xs=np.nonzero(body)
 if not len(ys):return []
 x0=max(0,int(xs.min())-100);x1=min(1920,int(xs.max())+101);y0=max(0,int(box[1]+.65*(box[3]-box[1])));y1=min(1080,int(ys.max())+181);terminal_y=int(ys.max())
 rr=rgb[y0:y1,x0:x1].astype('i2');m=body[y0:y1,x0:x1];near,inds=__import__('scipy.ndimage',fromlist=['distance_transform_edt']).distance_transform_edt(~m,return_indices=True);nearestRGB=rr[inds[0],inds[1]]
 skin=(nearestRGB[:,:,0]-nearestRGB[:,:,2]>15)&(inds[0]+y0>terminal_y-35)
 warm=(rr[:,:,0]-rr[:,:,2]>8)&(abs(rr[:,:,0]-rr[:,:,1])<10)&(rr[:,:,1]>70)&(rr[:,:,2]<235)&~m
 warm=cv2.morphologyEx(warm.astype('u1'),cv2.MORPH_CLOSE,np.ones((3,3),'u1'));n,lab,stats,cen=cv2.connectedComponentsWithStats(warm,8);out=[]
 for k,(x,y,w,h,area)in enumerate(stats[1:],1):
  cc=lab==k;attach=cc&(near<18)&skin
  if area<70 or w>170 or h>190 or w<5 or h<8 or w/h>3.2 or not attach.any():continue
  core=cv2.distanceTransform(cc.astype('u1'),cv2.DIST_L2,5);core[(near<9)|(near>90)]=0
  if core.max()<2:continue
  yy,xx=np.unravel_index(np.argmax(core),core.shape);px=x0+int(xx);py=y0+int(yy)
  out.append(dict(points_uv=[[px,py]],point_labels=[1],box_xyxy=[max(0,x0+int(x)-16),max(0,y0+int(y)-12),min(1920,x0+int(x+w)+16),min(1080,y0+int(y+h)+30)],actual_RGB_at_positive_point=rgb[py,px].tolist(),current_warm_component_native_area=int(area),component_bbox=[x0+int(x),y0+int(y),x0+int(x+w),y0+int(y+h)],actual_skin_attachment_pixels=int(attach.sum()),prompt_role='current_native_warm_white_footwear_component_attached_to_visible_truncated_ankle_navigation_only'))
 return out
def main():
 start=time.monotonic();expected=list(range(948992,1425408+1,256));(O/'masks').mkdir(exist_ok=True);(O/'delta_masks').mkdir(exist_ok=True);(O/'QA').mkdir(exist_ok=True)
 sam=Sam2SourceInstances(r'D:/CodexData/MotionTrail_Archive_20261019/models/sam2.1-hiera-tiny',device='cuda',reread_rgb=False,contour_stride=16);rows=[];stats={};cells=[];qa=0;changed=0
 def saveqa():
  nonlocal cells,qa
  if not cells:return
  cells += [np.full((260,600,3),235,'u1')for _ in range(8-len(cells))];cv2.imwrite(str(O/'QA'/f'changed_{qa:03d}.jpg'),np.vstack([np.hstack(cells[k:k+2])for k in range(0,8,2)]),[cv2.IMWRITE_JPEG_QUALITY,96]);cells=[];qa+=1
 with av.open(source['path'])as c,(O/'index.stream.jsonl').open('w',encoding='utf8')as wi,gzip.open(O/'prompts.jsonl.gz','wt',encoding='utf8')as wp:
  st=c.streams.video[0];c.seek(expected[0],stream=st,backward=True)
  for fr in c.decode(st):
   pts=int(fr.pts)
   if pts>expected[-1]:break
   if pts<expected[0]:continue
   assert pts==expected[len(rows)];rgb=fr.to_ndarray(format='rgb24');bgr=rgb[:,:,::-1].copy();sid='s05'if pts<=1300992 else's06';parent=MASK_INDEX[pts];rgbsha=TH(rgb.tobytes()).hexdigest();bgrsha=TH(bgr.tobytes()).hexdigest();assert rgbsha==parent['source_RGB_sha256']and bgrsha==parent['source_BGR_sha256'];mask=cv2.imread(parent['mask_file'],0)>0;props=propose(rgb,mask,pts);bind=dict(source_sha256=source['sha256'],source_pts=pts,time_base='1/12800',native_hw=[1080,1920],shot_id=sid,decoded_native_rgb_sha256=rgbsha);prompts=[dict(object_id=i+1,kind='occluder',origin='original_rgb_candidate',source_binding=bind,exclusion_authority=None,**p)for i,p in enumerate(props)];delta=np.zeros_like(mask);state='no_current_visible_truncated_ankle_component_extra_prompt'
   if prompts:
    result=sam.read(rgb,bind,prompts=prompts,reset=True);delta=result['occluder_region_candidate_mask'].detach().cpu().numpy().astype(bool);state='actual_current_RGB_positive_point_SAM2_footwear_pixel_segmentation'
   added=delta&~mask;area=int(added.sum());comp=mask|delta;row=dict(parent);row.update(schema='rinkspace.current_source_person_footwear_composite.v1',parent_mask_file=parent['mask_file'],parent_mask_sha256=parent['mask_sha256'],source_BGR_sha256=bgrsha,source_RGB_sha256=rgbsha,added_footwear_pixels=area,footwear_delta_mask_file=None,footwear_delta_mask_sha256=None,footwear_prompt_count=len(props),source_exclusion_type='current_actual_footwear_occluder',unknown_reflection_pixels_are_not_footwear_mask=True,prompt_ROI_is_not_hard_mask=True,mask_recipe='logical_OR_unchanged_parent_SAM2_and_current_RGB_prompted_SAM2_footwear_pixels',person_guard_pixels=8,mask_origin='current_native_SAM2_footwear_composite'if area else'unchanged_parent_native_SAM2_mask_reference',pixel_masks=True,body_pose_or_ROI_used_as_hard_exclusion=False,footwear_status=state)
   if area:
    df=O/'delta_masks'/f'{pts}.png';mf=O/'masks'/f'{pts}.png';cv2.imwrite(str(df),delta.astype('u1')*255,[cv2.IMWRITE_PNG_COMPRESSION,3]);cv2.imwrite(str(mf),comp.astype('u1')*255,[cv2.IMWRITE_PNG_COMPRESSION,3]);row.update(mask_file=str(mf),mask_sha256=TH(mf.read_bytes()).hexdigest(),mask_boolean_sha256=TH(comp.tobytes()).hexdigest(),mask_pixels=int(comp.sum()),footwear_delta_mask_file=str(df),footwear_delta_mask_sha256=TH(df.read_bytes()).hexdigest());changed+=1
    if changed%8==1 or pts in [1135104,1161216,1170944]:
     ys,xs=np.nonzero(added);x0=max(0,int(xs.min())-80);x1=min(1920,int(xs.max())+81);y0=max(0,int(ys.min())-90);y1=min(1080,int(ys.max())+55);im=bgr.copy();cnt,_=cv2.findContours(mask.astype('u1'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(im,cnt,-1,(255,0,0),2);cnt,_=cv2.findContours(delta.astype('u1'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(im,cnt,-1,(0,255,0),2)
     for p in props:cv2.circle(im,tuple(p['points_uv'][0]),3,(0,0,255),-1)
     crop=im[y0:y1,x0:x1];hh,ww=crop.shape[:2];scale=min(600/ww,230/hh);crop=cv2.resize(crop,(int(ww*scale),int(hh*scale)));cell=np.full((260,600,3),235,'u1');cell[:crop.shape[0],:crop.shape[1]]=crop;cv2.putText(cell,f'{pts} added {area} prompts {len(props)}',(8,252),0,.55,(0,0,255),1);cells.append(cell)
     if len(cells)==8:saveqa()
   rows.append(row);wi.write(json.dumps(row,separators=(',',':'))+'\n');wp.write(json.dumps(dict(native_pts=pts,shot_id=sid,source_RGB_sha256=rgbsha,source_BGR_sha256=bgrsha,prompts=props,output_footwear_added_pixels=area,positive_points_are_current_visible_RGB_pixels=True),separators=(',',':'))+'\n');stats[state]=stats.get(state,0)+1
   if len(rows)%64==0:wi.flush();wp.flush();p=dict(done=len(rows),total=1862,changed_frames=changed,last_pts=pts,elapsed_seconds=time.monotonic()-start);(O/'progress.json').write_text(json.dumps(p));print(json.dumps(p),flush=True)
 saveqa();assert len(rows)==1862
 with gzip.open(O/'index.jsonl.gz','wt',encoding='utf8')as f:
  for row in rows:f.write(json.dumps(row,separators=(',',':'))+'\n')
 manifest=dict(schema='rinkspace.current_source_person_footwear_composite.v1',complete=True,accepted_for_final_source_qualification=False,frame_count=len(rows),changed_frames=changed,added_footwear_pixels=sum(r['added_footwear_pixels']for r in rows),counts=stats,index_file=str(O/'index.jsonl.gz'),index_sha256=TH((O/'index.jsonl.gz').read_bytes()).hexdigest(),prompts_file=str(O/'prompts.jsonl.gz'),prompts_sha256=TH((O/'prompts.jsonl.gz').read_bytes()).hexdigest(),source_video_sha256=source['sha256'],parent_index_file=str(R/'dense56_masks_v1/index.jsonl.gz'),parent_index_sha256=TH((R/'dense56_masks_v1/index.jsonl.gz').read_bytes()).hexdigest(),current_native_source_hashes_checked_every_frame=True,source_model_sha256=sam.model_sha256,original_masks_untouched=True,prompt_ROI_not_hard_exclusion=True,unknown_gray_reflection_not_automatically_foreground=True,person_guard_pixels=8,pose_sideways_ankle_coverage_audit_pending=True,elapsed_seconds=time.monotonic()-start)
 (O/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8');print(json.dumps(manifest),flush=True)
if __name__=='__main__':main()
