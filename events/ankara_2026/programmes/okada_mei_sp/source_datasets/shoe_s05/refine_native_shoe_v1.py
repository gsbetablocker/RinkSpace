import os
for k in('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,gzip,hashlib
from pathlib import Path
P=Path(r'D:/OneDrive/codex_projs/RinkSpace');O=Path(r'D:/CodexData/RinkSpace_Data/runs/okada_rec_1019/recover_v2/dense56_v1');D=O/'raisedshoe_v1';sys.path[:0]=[str(P),str(P/'src')]
from calibration.program_tracking.runtime.bootstrap import add_geometry_dependencies
add_geometry_dependencies()
import cv2,numpy as np,av
cv2.setNumThreads(1)
def rz(p):return[json.loads(l)for l in gzip.open(p,'rt',encoding='utf8')]
import torch,time
from rinkspace.vision_instances import Sam2SourceInstances
torch.set_num_threads(1);sam=Sam2SourceInstances(r'D:/CodexData/MotionTrail_Archive_20261019/models/sam2.1-hiera-tiny',device='cuda',reread_rgb=False,contour_stride=16)
A=D/'attempt_v3';F=D/'refine_v1';F.mkdir(exist_ok=True);(F/'raw_native').mkdir(exist_ok=True);rows={r['native_pts']:r for r in rz(O/'shoe_fix/effective_v2/s05_index.jsonl.gz')};cands={r['native_pts']:r for r in rz(A/'candidates.jsonl.gz')};ps={int(p):r for p,r in json.load(open(D/'refine_source_prompts_crop.json'))['frames'].items()};source=json.load(open(O/'parts/s05/video_config.json'))['source'];outrows=[];cells=[];page=0
def emit():
 global cells,page
 if not cells:return
 cells+=[np.full((310,380,3),240,'u1')for _ in range(12-len(cells))];cv2.imwrite(str(F/f'QA_{page:02}.jpg'),np.vstack([np.hstack(cells[k:k+4])for k in(0,4,8)]),[cv2.IMWRITE_JPEG_QUALITY,97]);page+=1;cells=[]
with av.open(source['file'])as c,gzip.open(F/'candidates.jsonl.gz','wt',encoding='utf8')as w:
 st=c.streams.video[0];c.seek(min(ps),stream=st,backward=True)
 for fr in c.decode(st):
  p=int(fr.pts)
  if p>max(ps):break
  if p not in ps:continue
  b=fr.to_ndarray(format='bgr24');r=cands[p];x0,y0,x1,y1=r['crop_xyxy'];rgb=b[y0:y1,x0:x1,::-1].copy();h,wi=rgb.shape[:2];pos=ps[p]['p'];neg=ps[p]['n'];pts=pos+neg;ss=sam.processor.init_video_session(inference_device=sam.device,inference_state_device=sam.device,video_storage_device='cpu',max_vision_features_cache_size=1,dtype=torch.bfloat16);sam.processor.add_inputs_to_inference_session(inference_session=ss,frame_idx=0,obj_ids=1,original_size=(h,wi),input_points=[[pts]],input_labels=[[[1]*len(pos)+[0]*len(neg)]])
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
   enc=sam.processor(images=rgb,device=str(sam.device),return_tensors='pt')['pixel_values'].to(sam.device);out=sam.model(inference_session=ss,frame=enc,frame_idx=0);m=sam.processor.post_process_masks([out.pred_masks],original_sizes=[[h,wi]],binarize=True)[0][0,0].cpu().numpy().astype('u1')
  full=np.zeros((1080,1920),'u1');full[y0:y1,x0:x1]=m;f=F/'raw_native'/f'{p}.png';cv2.imwrite(str(f),full*255,[cv2.IMWRITE_PNG_COMPRESSION,3]);rec=dict(r);rec.update(raw_current_source_SAM_ref=dict(file=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest()),source_prompt_positive_uv=(np.array(pos)+[x0,y0]).tolist(),source_prompt_negative_uv=(np.array(neg)+[x0,y0]).tolist(),SAM_box_navigation_only=None,source_prompt_refinement='fresh_full_phase_actual_source_QA_against_static_board_and_ice_counterexamples');w.write(json.dumps(rec,separators=(',',':'))+chr(10));outrows.append(rec)
  old=cv2.imread(rows[p]['mask_file'],0)>0;new=(full>0)&~old;cr=b[y0:y1,x0:x1].copy();cs,_=cv2.findContours(new[y0:y1,x0:x1].astype('u1'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(cr,cs,-1,(255,0,255),1)
  for x,y in pos:cv2.circle(cr,(x,y),3,(0,255,255),1)
  for x,y in neg:cv2.circle(cr,(x,y),3,(0,0,255),1)
  tile=np.full((310,380,3),240,'u1');tile[25:25+cr.shape[0],:cr.shape[1]]=cr;cv2.putText(tile,f'{p} +{int(new.sum())}',(3,19),0,.5,(0,0,0),1);cells.append(tile)
  if len(cells)==12:emit()
  print('REFINE',p,int(new.sum()),flush=True)
emit();print('DONE',len(outrows),flush=True)
