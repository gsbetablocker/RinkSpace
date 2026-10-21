import os
for k in('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,gzip,hashlib,time
from pathlib import Path
P=Path(r'D:/OneDrive/codex_projs/RinkSpace');D=Path(r'D:/CodexData/RinkSpace_Data');R=D/'runs/okada_rec_1019/recover_v2';O=R/'dense56_footwear_v1/clean_parent52';O.mkdir(exist_ok=False)
sys.path[:0]=[str(P),str(P/'src'),str(P/'calibration/rinkfit/src')]
from calibration.program_tracking.runtime.bootstrap import add_geometry_dependencies
add_geometry_dependencies()
import cv2,numpy as np,torch,av
from rinkspace.vision_instances import Sam2SourceInstances
cv2.setNumThreads(1);torch.set_num_threads(1)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,j):Path(p).write_text(json.dumps(j,ensure_ascii=False,indent=2),encoding='utf8')
def readz(p):
 with gzip.open(p,'rt',encoding='utf8')as f:return[json.loads(l)for l in f]
source=json.load(open(D/'runs/okada_joint9_1018a/programme_scope.json',encoding='utf8'))['source'];assert sha(source['path'])==source['sha256']
mi={q['native_pts']:q for q in readz(R/'dense56_masks_v1/index.jsonl.gz')if q.get('existing_mask')};ps={q['native_pts']:q for q in readz(R/'dense56_footwear_v1/pose_proposals.jsonl.gz')};assert len(mi)==52
sam=Sam2SourceInstances(r'D:/CodexData/MotionTrail_Archive_20261019/models/sam2.1-hiera-tiny',device='cuda',reread_rgb=False,contour_stride=16)
rows=[];tiles=[];pages=[];(O/'masks').mkdir();(O/'QA').mkdir();start=time.perf_counter()
def contour(im,m,color):
 cs,_=cv2.findContours(m.astype('u1'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(im,cs,-1,color,2)
with av.open(source['path'])as con:
 st=con.streams.video[0];con.seek(min(mi),stream=st,backward=True)
 for frame in con.decode(st):
  pts=int(frame.pts)
  if pts>max(mi):break
  if pts not in mi:continue
  old=mi[pts];po=ps[pts];bgr=frame.to_ndarray(format='bgr24');rgb=bgr[:,:,::-1].copy();bh=hashlib.sha256(bgr.tobytes()).hexdigest();rh=hashlib.sha256(rgb.tobytes()).hexdigest();assert bh==old['source_BGR_sha256']==po['source_BGR_sha256']and rh==old['source_RGB_sha256']
  binding=dict(source_sha256=source['sha256'],source_pts=pts,time_base='1/12800',native_hw=[1080,1920],shot_id=old['shot_id'],decoded_native_rgb_sha256=rh)
  proposals=po['person_proposals'];prompts=[dict(object_id=i+1,kind='occluder',origin='original_rgb_candidate',source_binding=binding,box_xyxy=q['box_xyxy'],exclusion_authority=None)for i,q in enumerate(proposals)]
  current=sam.read(rgb,binding,prompts=prompts,reset=True);fresh=current['occluder_region_candidate_mask'].detach().cpu().numpy().astype(bool)
  oldmask=cv2.imread(old['mask_file'],0)>0;assert sha(old['mask_file'])==old['mask_sha256'];n,labels,stats,_=cv2.connectedComponentsWithStats(oldmask.astype('u1'),8);components=[];retained=np.zeros(oldmask.shape,bool);expanded=cv2.dilate(fresh.astype('u1'),np.ones((9,9),'u1'))>0
  for j in range(1,n):
   comp=labels==j;area=int(stats[j,4]);overlap=int((comp&expanded).sum());ok=overlap>0 and overlap/max(area,1)>.35
   components.append(dict(component=j,bbox_xywh=stats[j,:4].tolist(),area=area,current_SAM_overlap=overlap,retained_candidate=ok))
   if ok:retained|=comp
  mask=fresh|retained;file=O/'masks'/f'{pts}.png';cv2.imwrite(str(file),mask.astype('u1')*255,[cv2.IMWRITE_PNG_COMPRESSION,3]);raw=O/'masks'/f'{pts}_currentSAM.png';cv2.imwrite(str(raw),fresh.astype('u1')*255,[cv2.IMWRITE_PNG_COMPRESSION,3]);row=dict(shot_id=old['shot_id'],native_pts=pts,source_BGR_sha256=bh,source_RGB_sha256=rh,source_video_sha256=source['sha256'],mask_file=str(file),mask_sha256=sha(file),mask_pixels=int(mask.sum()),mask_origin='current_RGB_SAM2_plus_source_reviewed_legacy_actor_components',parent_mask=dict(file=old['mask_file'],sha256=old['mask_sha256']),current_SAM=dict(file=str(raw),sha256=sha(raw)),legacy_components=components,removed_legacy_pixels=int((oldmask&~mask).sum()),added_current_pixels=int((mask&~oldmask).sum()),bbox_used_as_hard_exclusion=False,source_review_status='pending_visual',model_sha256=sam.model_sha256)
  rows.append(row)
  main=max(proposals,key=lambda q:q['box_xyxy'][3]-q['box_xyxy'][1]);x0,y0,x1,y1=main['box_xyxy'];x0=max(0,int(x0)-100);x1=min(1920,int(x1)+100);y0=max(0,int(y0)-80);y1=min(1080,int(y1)+110)
  overlay=bgr.copy();contour(overlay,mask,(0,255,0));contour(overlay,oldmask&~mask,(0,0,255));contour(overlay,mask&~oldmask,(255,0,255));crop=np.hstack([bgr[y0:y1,x0:x1],overlay[y0:y1,x0:x1]]);scale=min(1024/crop.shape[1],440/crop.shape[0]);crop=cv2.resize(crop,None,fx=scale,fy=scale);tile=np.full((470,1040,3),35,'u1');tile[30:30+crop.shape[0],:crop.shape[1]]=crop;cv2.putText(tile,f'{pts} RAW | GREEN person; RED removed old; MAGENTA new', (8,22),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1);tiles.append(tile)
  if len(tiles)==4:
   page=O/'QA'/f'parent_{len(pages):02}.jpg';cv2.imwrite(str(page),np.vstack(tiles),[cv2.IMWRITE_JPEG_QUALITY,90]);pages.append(str(page));tiles=[]
  write(O/'progress.json',dict(frames=len(rows),total=52,native_pts=pts,elapsed_seconds=time.perf_counter()-start));print(pts,len(rows),flush=True)
if tiles:
 page=O/'QA'/f'parent_{len(pages):02}.jpg';cv2.imwrite(str(page),np.vstack(tiles),[cv2.IMWRITE_JPEG_QUALITY,90]);pages.append(str(page))
assert len(rows)==52
with gzip.open(O/'index_candidate.jsonl.gz','xt',encoding='utf8')as f:
 for row in rows:f.write(json.dumps(row,separators=(',',':'))+'\n')
write(O/'manifest_candidate.json',dict(complete=True,frames=52,accepted=False,source_video_sha256=source['sha256'],index=dict(file=str(O/'index_candidate.jsonl.gz'),sha256=sha(O/'index_candidate.jsonl.gz')),QA=pages,removed_legacy_pixels=sum(x['removed_legacy_pixels']for x in rows),added_current_pixels=sum(x['added_current_pixels']for x in rows),legacy_graphics_masks_preserved_raw=True,pose_or_box_hard_exclusion=False,elapsed_seconds=time.perf_counter()-start));print('COMPLETE',flush=True)
