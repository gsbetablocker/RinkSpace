import os
for k in('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,gzip,hashlib,time
from pathlib import Path
P=Path(r'D:/OneDrive/codex_projs/RinkSpace');O=Path(r'D:/CodexData/RinkSpace_Data/runs/okada_rec_1019/recover_v2/dense56_v1');L=O/'leg_fix_v1';sys.path[:0]=[str(P),str(P/'src')]
from calibration.program_tracking.runtime.bootstrap import add_geometry_dependencies
add_geometry_dependencies()
import cv2,numpy as np,av,torch
from rinkspace.vision_instances import Sam2SourceInstances
cv2.setNumThreads(1);torch.set_num_threads(1)
sam=Sam2SourceInstances(r'D:/CodexData/MotionTrail_Archive_20261019/models/sam2.1-hiera-tiny',device='cuda',reread_rgb=False,contour_stride=16)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
rows={r['native_pts']:r for r in map(json.loads,gzip.open(O/'raisedshoe_v1/s05_index.jsonl.gz','rt',encoding='utf8'))}
source=json.load(open(O/'parts/s05/video_config.json',encoding='utf8'))['source'];prom=json.load(open(L/'source_prompts.json',encoding='utf8'))['frames'];ps={int(x) for x in prom};F=L/'attempt_v1';F.mkdir(exist_ok=False);(F/'raw_native').mkdir();cells=[];page=0
with av.open(source['file'])as c,gzip.open(F/'candidates.jsonl.gz','wt',encoding='utf8')as w:
 st=c.streams.video[0];c.seek(min(ps),stream=st,backward=True)
 for fr in c.decode(st):
  p=int(fr.pts)
  if p>max(ps):break
  if p not in ps:continue
  b=fr.to_ndarray(format='bgr24');rgb=b[:,:,::-1].copy();r=rows[p];bh=hashlib.sha256(b.tobytes()).hexdigest();rh=hashlib.sha256(rgb.tobytes()).hexdigest();assert bh==r['source_BGR_sha256']and rh==r['source_RGB_sha256'];m=cv2.imread(r['mask_file'],0)>0;u=cv2.imread(r['source_unknown_exclusion_mask_ref']['file'],0)>0 if r.get('source_unknown_exclusion_mask_ref')else np.zeros_like(m);old=m|u;pr=prom[str(p)];x0,y0,x1,y1=pr['crop_xyxy'];cr=rgb[y0:y1,x0:x1].copy();h,wi=cr.shape[:2];pos=pr['positive_uv'];neg=[]
  for x,y in pr['negative_uv']:
   if not old[max(0,y-3):min(1080,y+4),max(0,x-3):min(1920,x+4)].any():neg.append([x,y])
  pt=(np.array(pos+neg)-[x0,y0]).tolist();ss=sam.processor.init_video_session(inference_device=sam.device,inference_state_device=sam.device,video_storage_device='cpu',max_vision_features_cache_size=1,dtype=torch.bfloat16);sam.processor.add_inputs_to_inference_session(inference_session=ss,frame_idx=0,obj_ids=1,original_size=(h,wi),input_points=[[pt]],input_labels=[[[1]*len(pos)+[0]*len(neg)]])
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
   enc=sam.processor(images=cr,device=str(sam.device),return_tensors='pt')['pixel_values'].to(sam.device);out=sam.model(inference_session=ss,frame=enc,frame_idx=0);sm=sam.processor.post_process_masks([out.pred_masks],original_sizes=[[h,wi]],binarize=True)[0][0,0].cpu().numpy().astype('u1')
  full=np.zeros((1080,1920),'u1');full[y0:y1,x0:x1]=sm;f=F/'raw_native'/f'{p}.png';cv2.imwrite(str(f),full*255,[cv2.IMWRITE_PNG_COMPRESSION,3]);add=(full>0)&~old;rec=dict(native_pts=p,shot_id='s05',time_base='1/12800',source_video_sha256=source['sha256'],source_BGR_sha256=bh,source_RGB_sha256=rh,source_crop_RGB_sha256=hashlib.sha256(cr.tobytes()).hexdigest(),crop_xyxy=pr['crop_xyxy'],crop_is_unmodified_native_RGB=True,source_prompt_positive_uv=pos,source_prompt_negative_uv=neg,source_prompt_ref=dict(file=str(L/'source_prompts.json'),sha256=sha(L/'source_prompts.json')),SAM_box_navigation_only=None,raw_current_source_SAM_ref=dict(file=str(f),sha256=sha(f)),raw_pixels=int(sm.sum()),new_unknown_candidate_pixels=int(add.sum()),model_sha256=sam.model_sha256,qualification='pending_actual_current_native_source_visual_review',no_bbox_mask_or_interpolation_fill=True);w.write(json.dumps(rec,separators=(',',':'))+'\n');w.flush()
  crop=b[y0:y1,x0:x1].copy();cs,_=cv2.findContours(old[y0:y1,x0:x1].astype('u1'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(crop,cs,-1,(0,255,0),1);cs,_=cv2.findContours(add[y0:y1,x0:x1].astype('u1'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(crop,cs,-1,(255,0,255),1);cv2.circle(crop,tuple((np.array(pos[0])-[x0,y0]).astype(int)),4,(0,0,255),1);tile=np.full((240,490,3),240,'u1');tile[25:25+h,:wi]=crop;cv2.putText(tile,f'{p} +{int(add.sum())}',(3,19),0,.5,(0,0,0),1);cells.append(tile)
  if len(cells)==12:
   cv2.imwrite(str(F/f'QA_{page:02}.jpg'),np.vstack([np.hstack(cells[k:k+3])for k in(0,3,6,9)]),[cv2.IMWRITE_JPEG_QUALITY,97]);page+=1;cells=[]
  print('SAM',p,int(add.sum()),flush=True)
if cells:
 cells+=[np.full((240,490,3),240,'u1') for _ in range(12-len(cells))];cv2.imwrite(str(F/f'QA_{page:02}.jpg'),np.vstack([np.hstack(cells[k:k+3])for k in(0,3,6,9)]),[cv2.IMWRITE_JPEG_QUALITY,97])
print('DONE',len(ps),flush=True)
