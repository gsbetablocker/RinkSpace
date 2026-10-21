import os
for k in('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
exec(open(r'D:/CodexData/RinkSpace_Data/runs/okada_rec_1019/recover_v2/dense56_v1/shoe_fix/full_body_v3.py',encoding='utf8').read().split("P=Path")[0])
from pathlib import Path
P=Path(r'D:/OneDrive/codex_projs/RinkSpace');R=Path(r'D:/CodexData/RinkSpace_Data/runs/okada_rec_1019/recover_v2');O=R/'dense56_v1/shoe_fix/background_v1';sys.path[:0]=[str(P),str(P/'src'),str(P/'calibration/rinkfit/src')]
from calibration.program_tracking.runtime.bootstrap import add_geometry_dependencies
add_geometry_dependencies()
import cv2,numpy as np,av,torch
from rinkspace.vision_instances import Sam2SourceInstances
cv2.setNumThreads(1);torch.set_num_threads(1)
def sh(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rz(p):return[json.loads(l)for l in gzip.open(p,'rt',encoding='utf8')]
prfile=O/'proposals.jsonl.gz';props={q['native_pts']:q for q in rz(prfile)};actor={q['native_pts']:q for q in rz(R/'dense56_v1/shoe_fix/full_v3/index.stream.jsonl.gz')};source=json.load(open(R/'dense56_v1/parts/s05/video_config.json',encoding='utf8'))['source'];(O/'instances').mkdir(exist_ok=False);(O/'QA').mkdir(exist_ok=False)
sam=Sam2SourceInstances(r'D:/CodexData/MotionTrail_Archive_20261019/models/sam2.1-hiera-tiny',device='cuda',reread_rgb=False,contour_stride=16);records=[];seq=[];start=time.perf_counter();cells=[];pages=[];groups={};gid=0;last=[];lastpts=None;lastshot=None

def emit():
 global cells
 if not cells:return
 cells +=[np.full((280,360,3),240,'u1')for _ in range(12-len(cells))];ff=O/'QA'/f'page_{len(pages):03}.jpg';cv2.imwrite(str(ff),np.vstack([np.hstack(cells[k:k+4])for k in(0,4,8)]),[cv2.IMWRITE_JPEG_QUALITY,96]);pages.append(str(ff));cells=[]
def iou(a,b):
 x0=max(a[0],b[0]);y0=max(a[1],b[1]);x1=min(a[2],b[2]);y1=min(a[3],b[3]);inter=max(0,x1-x0)*max(0,y1-y0);return inter/max((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter,1)
with av.open(source['file'])as c,gzip.open(O/'instances.jsonl.gz','wt',encoding='utf8')as w:
 st=c.streams.video[0];c.seek(min(props),stream=st,backward=True)
 for fr in c.decode(st):
  p=int(fr.pts)
  if p>max(props):break
  if p not in props:continue
  q=props[p];prs=q['proposals']
  if not prs:last=[];lastpts=p;lastshot=q['shot_id'];continue
  bgr=fr.to_ndarray(format='bgr24');rgb=bgr[:,:,::-1].copy();assert hashlib.sha256(bgr.tobytes()).hexdigest()==q['source_BGR_sha256']and hashlib.sha256(rgb.tobytes()).hexdigest()==q['source_RGB_sha256'];bind=dict(source_sha256=source['sha256'],source_pts=p,time_base='1/12800',native_hw=[1080,1920],shot_id=q['shot_id'],decoded_native_rgb_sha256=q['source_RGB_sha256']);prompts=[]
  for j,a in enumerate(prs):
   bb=np.array(a['box_xyxy']);pts=[];k=np.array(a['keypoints_uv']);co=np.array(a['keypoints_confidence'])
   if co[0]>.5:pts.append(k[0].tolist())
   if co[5]>.35 and co[6]>.35:pts.append(((k[5]+k[6])/2).tolist())
   pr=dict(object_id=j+1,kind='occluder',origin='original_rgb_candidate',source_binding=bind,box_xyxy=[max(0,float(bb[0])-4),max(0,float(bb[1])-4),min(1920,float(bb[2])+4),min(1080,float(bb[3])+4)],exclusion_authority=None)
   if pts:pr.update(points_uv=pts,point_labels=[1]*len(pts))
   prompts.append(pr)
  rr=sam.read(rgb,bind,prompts=prompts,reset=True);masks={a['object_id']:a['region_mask'].detach().cpu().numpy().astype(bool)for a in rr['instances']};ids=[None]*len(prs)
  if lastpts is not None and p-lastpts==256 and q['shot_id']==lastshot:
   pairs=sorted([(iou(a['box_xyxy'],b['box_xyxy']),j,k)for j,a in enumerate(prs)for k,b in enumerate(last)],reverse=True);used=set()
   for score,j,k in pairs:
    if score>.2 and ids[j]is None and k not in used:ids[j]=last[k]['tracklet_id'];used.add(k)
  current=[]
  for j,a in enumerate(prs):
   if ids[j]is None:gid+=1;ids[j]=gid
   m=masks[j+1];ff=O/'instances'/f'{p}_{j}.png';cv2.imwrite(str(ff),m.astype('u1')*255,[cv2.IMWRITE_PNG_COMPRESSION,3]);yy,xx=np.where(m);bbox=[]if not len(xx)else[int(xx.min()),int(yy.min()),int(xx.max())+1,int(yy.max())+1];record=dict(native_pts=p,shot_id=q['shot_id'],object_index=j,tracklet_id=ids[j],source_BGR_sha256=q['source_BGR_sha256'],source_RGB_sha256=q['source_RGB_sha256'],mask_file=str(ff),mask_sha256=sh(ff),pixel_count=int(m.sum()),mask_bbox_xyxy=bbox,proposal=a,prompt=prompts[j],model_sha256=sam.model_sha256,source_person_qualification='pending_actual_source_visual_and_continuity_review',bbox_is_not_mask=True);w.write(json.dumps(record,separators=(',',':'))+chr(10));records.append(record);groups.setdefault(ids[j],[]).append(record);current.append(dict(box_xyxy=a['box_xyxy'],tracklet_id=ids[j]))
   bb=a['box_xyxy'];x0=max(0,int(min(bb[0],bbox[0]if bbox else bb[0]))-20);x1=min(1920,int(max(bb[2],bbox[2]if bbox else bb[2]))+21);y0=max(0,int(min(bb[1],bbox[1]if bbox else bb[1]))-20);y1=min(1080,int(max(bb[3],bbox[3]if bbox else bb[3]))+31);ov=bgr.copy();cs,_=cv2.findContours(m.astype('u1'),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(ov,cs,-1,(0,255,0),1);crop=np.hstack([bgr[y0:y1,x0:x1],ov[y0:y1,x0:x1]]);sc=min(350/crop.shape[1],245/crop.shape[0]);crop=cv2.resize(crop,None,fx=sc,fy=sc);tile=np.full((280,360,3),240,'u1');tile[25:25+crop.shape[0],:crop.shape[1]]=crop;cv2.putText(tile,f'{p}/{j} T{ids[j]} {a["confidence"]:.2f}',(3,20),0,.45,(0,0,0),1);cells.append(tile)
   if len(cells)==12:emit()
  seq.append(dict(native_pts=p,shot_id=q['shot_id'],instances=len(prs)));last=current;lastpts=p;lastshot=q['shot_id']
  if len(seq)%25==0:w.flush();(O/'sam_progress.json').write_text(json.dumps(dict(frames=len(seq),instances=len(records),tracklets=len(groups),native_pts=p,seconds=time.perf_counter()-start)),encoding='utf8');print(len(seq),p,len(records),len(groups),round(time.perf_counter()-start,2),flush=True)
emit();assert len(records)==sum(len(q['proposals'])for q in props.values());(O/'sam_manifest_candidate.json').write_text(json.dumps(dict(complete=True,accepted=False,frames_with_boundary_person_proposal=len(seq),instances=len(records),tracklets=len(groups),proposal_index=dict(file=str(prfile),sha256=sh(prfile)),index=dict(file=str(O/'instances.jsonl.gz'),sha256=sh(O/'instances.jsonl.gz')),source_video_sha256=source['sha256'],all_candidate_original_RGB_BGR_verified=True,all_masks_are_current_SAM_not_box_fill=True,all_raw_actor_masks_unchanged=True,QA=pages,elapsed_seconds=time.perf_counter()-start),indent=2),encoding='utf8');print('COMPLETE',flush=True)
