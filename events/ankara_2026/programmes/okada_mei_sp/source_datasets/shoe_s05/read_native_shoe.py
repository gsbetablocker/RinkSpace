import os
for k in('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,gzip,hashlib
from pathlib import Path
P=Path(r'D:/OneDrive/codex_projs/RinkSpace');O=Path(r'D:/CodexData/RinkSpace_Data/runs/okada_rec_1019/recover_v2/dense56_v1');S=O/'raisedshoe_v1';D=S/'attempt_v3';D.mkdir(exist_ok=False);sys.path[:0]=[str(P),str(P/'src')]
from calibration.program_tracking.runtime.bootstrap import add_geometry_dependencies
add_geometry_dependencies()
import cv2,numpy as np,av
cv2.setNumThreads(1)
def rz(p):return[json.loads(l)for l in gzip.open(p,'rt',encoding='utf8')]
import torch,time
from rinkspace.vision_instances import Sam2SourceInstances
torch.set_num_threads(1)
sam=Sam2SourceInstances(r'D:/CodexData/MotionTrail_Archive_20261019/models/sam2.1-hiera-tiny',device='cuda',reread_rgb=False,contour_stride=16)
rows={r['native_pts']:r for r in rz(O/'shoe_fix/effective_v2/s05_index.jsonl.gz')};source=json.load(open(O/'parts/s05/video_config.json',encoding='utf8'))['source'];nav={r['native_pts']:r for r in json.load(open(S/'LK_navigation.json',encoding='utf8'))};anchors={int(p):uv for p,uv in json.load(open(S/'navigation_anchors.json',encoding='utf8'))['anchors'].items()};ak=np.array(sorted(anchors));avals=np.array([anchors[p]for p in ak]);(D/'raw_native').mkdir(exist_ok=True);(D/'candidate_QA').mkdir(exist_ok=True);cells=[];page=0;proofs=[];start=time.time()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def native_json(x):
 if isinstance(x,np.generic):return x.item()
 if isinstance(x,np.ndarray):return x.tolist()
 raise TypeError(type(x).__name__)
def emit():
 global page,cells
 if not cells:return
 cells +=[np.full((310,380,3),240,'u1')for _ in range(16-len(cells))];f=D/'candidate_QA'/f'page_{page:02}.jpg';cv2.imwrite(str(f),np.vstack([np.hstack(cells[k:k+4])for k in(0,4,8,12)]),[cv2.IMWRITE_JPEG_QUALITY,96]);page+=1;cells=[]
with av.open(source['file'])as c,gzip.open(D/'candidates.jsonl.gz','wt',encoding='utf8')as w:
 st=c.streams.video[0];c.seek(min(nav),stream=st,backward=True)
 for fr in c.decode(st):
  p=int(fr.pts)
  if p>max(nav):break
  if p not in nav:continue
  b=fr.to_ndarray(format='bgr24');r=rows[p];n=nav[p];rh=hashlib.sha256(b[:,:,::-1].copy().tobytes()).hexdigest();bh=hashlib.sha256(b.tobytes()).hexdigest();assert rh==r['source_RGB_sha256']==n['source_RGB_sha256']and bh==r['source_BGR_sha256']==n['source_BGR_sha256'];old=cv2.imread(r['mask_file'],0)>0
  nn=np.array([np.interp(p,ak,avals[:,j])for j in(0,1)]);uv=np.array(n['source_prompt_uv'])if n['accepted_navigation']else nn;cx,cy=np.rint(uv).astype(int);x0=max(0,cx-185);y0=max(0,cy-140);x1=min(1920,cx+186);y1=min(1080,cy+141);rgb=b[y0:y1,x0:x1,::-1].copy();h,wi=rgb.shape[:2];neg=[]
  for u in[[x0+18,y0+18],[x1-19,y0+18],[x0+18,y1-19],[x1-19,y1-19]]:
   x,y=u
   if not old[max(0,y-10):min(1080,y+11),max(0,x-10):min(1920,x+11)].any():neg.append(u)
  pos=[uv.tolist()]if n['accepted_navigation']else[]
  if p==1038592:pos=[[620,615],[640,640],[668,641],[596,612],[600,650]]
  if p==1057792:pos=[[550,357],[514,345],[488,358],[537,356]]
  pts=(np.array(pos+neg)-[x0,y0]).tolist();ss=sam.processor.init_video_session(inference_device=sam.device,inference_state_device=sam.device,video_storage_device='cpu',max_vision_features_cache_size=1,dtype=torch.bfloat16);kwargs=dict(inference_session=ss,frame_idx=0,obj_ids=1,original_size=(h,wi),input_boxes=[[[max(0,cx-x0-105),max(0,cy-y0-70),min(wi-1,cx-x0+105),min(h-1,cy-y0+70)]]])
  if len(pts):kwargs.update(input_points=[[pts]],input_labels=[[[1]*len(pos)+[0]*len(neg)]])
  kwargs['input_boxes']=[[[int(v)for v in kwargs['input_boxes'][0][0]]]];sam.processor.add_inputs_to_inference_session(**kwargs)
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
   enc=sam.processor(images=rgb,device=str(sam.device),return_tensors='pt')['pixel_values'].to(sam.device);out=sam.model(inference_session=ss,frame=enc,frame_idx=0);m=sam.processor.post_process_masks([out.pred_masks],original_sizes=[[h,wi]],binarize=True)[0][0,0].cpu().numpy().astype('u1')
  full=np.zeros((1080,1920),'u1');full[y0:y1,x0:x1]=m;f=D/'raw_native'/f'{p}.png';cv2.imwrite(str(f),full*255,[cv2.IMWRITE_PNG_COMPRESSION,3]);added=(full>0)&~old;rec=dict(native_pts=p,shot_id='s05',time_base='1/12800',source_video_sha256=source['sha256'],source_BGR_sha256=bh,source_RGB_sha256=rh,source_crop_RGB_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),crop_xyxy=[int(x0),int(y0),int(x1),int(y1)],crop_is_unmodified_native_RGB=True,source_prompt_positive_uv=pos,source_prompt_negative_uv=neg,SAM_box_navigation_only=kwargs['input_boxes'][0][0],LK_navigation=n,navigation_fallback='linear_neighbor_source_anchor_search_location_only_not_observation'if not n['accepted_navigation']else None,raw_current_source_SAM_ref=dict(file=str(f),sha256=sha(f)),raw_pixels=int(m.sum()),new_pixels=int(added.sum()),source_qualification_status='pending_complete_continuous_current_source_visual_review',model_sha256=sam.model_sha256,no_mask_polygon_dilation_or_rectangle_fill=True);w.write(json.dumps(rec,separators=(',',':'),default=native_json)+'\n');w.flush();proofs.append(rec)
  crop=b[y0:y1,x0:x1].copy();cs,_=cv2.findContours(old[y0:y1,x0:x1].astype('u1'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(crop,cs,-1,(0,255,0),1);cs,_=cv2.findContours(added[y0:y1,x0:x1].astype('u1'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(crop,cs,-1,(255,0,255),1);tile=np.full((310,380,3),240,'u1');tile[25:25+crop.shape[0],:crop.shape[1]]=crop;cv2.putText(tile,f'{p} +{int(added.sum())}'+(' LK'if n['accepted_navigation']else' box'),(3,19),0,.48,(0,0,0),1);cells.append(tile)
  if len(cells)==16:emit()
  if len(proofs)%20==0:print('SAM',len(proofs),p,round(time.time()-start,2),flush=True)
emit();print('DONE',len(proofs),flush=True)
