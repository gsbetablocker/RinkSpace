import os
for k in('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,gzip,hashlib,time
from pathlib import Path
P=Path(r'D:/OneDrive/codex_projs/RinkSpace');R=Path(r'D:/CodexData/RinkSpace_Data/runs/okada_rec_1019/recover_v2');O=R/'dense56_v1/shoe_fix/full_v3';O.mkdir(exist_ok=False);(O/'masks').mkdir();(O/'raw_SAM').mkdir();(O/'QA').mkdir()
sys.path[:0]=[str(P),str(P/'src'),str(P/'calibration/rinkfit/src')]
from calibration.program_tracking.runtime.bootstrap import add_geometry_dependencies
add_geometry_dependencies()
import cv2,numpy as np,av,torch
from rinkspace.vision_instances import Sam2SourceInstances
cv2.setNumThreads(1);torch.set_num_threads(1)
def sh(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rz(p):return[json.loads(l)for l in gzip.open(p,'rt',encoding='utf8')]
parentfile=R/'dense56_footwear_v1/index.jsonl.gz';mi={q['native_pts']:q for q in rz(parentfile)};poses={q['native_pts']:q for q in rz(R/'dense56_footwear_v1/pose_selection_v2.jsonl.gz')};source=json.load(open(R/'dense56_v1/parts/s05/video_config.json',encoding='utf8'))['source'];assert len(mi)==1862
sam=Sam2SourceInstances(r'D:/CodexData/MotionTrail_Archive_20261019/models/sam2.1-hiera-tiny',device='cuda',reread_rgb=False,contour_stride=16)
manual={1096192:[[654,417],[728,418],[848,495],[1060,472],[801,660]],1135104:[[964,484],[1024,435],[1053,574],[981,295],[1023,723],[1000,110]],1153792:[[1041,530],[1024,671],[1003,508],[1119,290],[1168,117]],1161216:[[937,472],[1025,434],[1065,530],[1066,294],[1068,188],[994,645]],1170944:[[944,472],[1037,433],[1077,525],[1073,296],[1064,188]]}
rows=[];cells=[];pages=[];start=time.perf_counter()
def emit():
 global cells
 if not cells:return
 cells +=[np.full((350,440,3),240,'u1')for _ in range(12-len(cells))];page=O/'QA'/f'page_{len(pages):03}.jpg';cv2.imwrite(str(page),np.vstack([np.hstack(cells[k:k+4])for k in(0,4,8)]),[cv2.IMWRITE_JPEG_QUALITY,96]);pages.append(str(page));cells=[]
with av.open(source['file'])as c,gzip.open(O/'index.stream.jsonl.gz','wt',encoding='utf8')as w:
 st=c.streams.video[0];c.seek(min(mi),stream=st,backward=True)
 for fr in c.decode(st):
  p=int(fr.pts)
  if p>max(mi):break
  if p not in mi:continue
  par=mi[p];bgr=fr.to_ndarray(format='bgr24');rgb=bgr[:,:,::-1].copy();rh=hashlib.sha256(rgb.tobytes()).hexdigest();bh=hashlib.sha256(bgr.tobytes()).hexdigest();assert rh==par['source_RGB_sha256']and bh==par['source_BGR_sha256']and sh(par['mask_file'])==par['mask_sha256'];base=cv2.imread(par['mask_file'],0)>0
  bind=dict(source_sha256=source['sha256'],source_pts=p,time_base='1/12800',native_hw=[1080,1920],shot_id=par['shot_id'],decoded_native_rgb_sha256=rh)
  po=poses[p];pp=po['person_proposals'][po['selected_person_list_index']];pb=np.array(pp['box_xyxy']);n0,l0,st0,_=cv2.connectedComponentsWithStats(base.astype('u1'),8);reg=np.zeros_like(base);reg[max(0,int(pb[1])):min(1080,int(pb[3])+1),max(0,int(pb[0])):min(1920,int(pb[2])+1)]=True;scores=[int(((l0==j)&reg).sum())for j in range(1,n0)];actor= l0==(int(np.argmax(scores))+1);yy,xx=np.where(actor);box=[max(0,min(int(xx.min()),int(pb[0]))-130),max(0,min(int(yy.min()),int(pb[1]))-80),min(1920,max(int(xx.max()),int(pb[2]))+130),min(1080,max(int(yy.max()),int(pb[3]))+40)]
  dist=cv2.distanceTransform(actor.astype('u1'),cv2.DIST_L2,5);yy2,xx2=np.where(dist>8);xy=np.stack([xx2,yy2],1);chosen=[int(np.argmax(dist[yy2,xx2]))];dd=np.linalg.norm(xy-xy[chosen[0]],axis=1)
  for j in range(7):
   ix=int(np.argmax(dd));chosen.append(ix);dd=np.minimum(dd,np.linalg.norm(xy-xy[ix],axis=1))
  points=xy[chosen].tolist();origpoints=list(points);skin=(rgb[:,:,0].astype('i2')>rgb[:,:,1].astype('i2')+8)&(rgb[:,:,1].astype('i2')>rgb[:,:,2].astype('i2')+6)&(rgb[:,:,0]>70)&(rgb[:,:,0]<235)
  region=np.zeros_like(base);region[int(box[1]):int(box[3]),int(box[0]):int(box[2])]=True;skin &=region;n,la,stats,_=cv2.connectedComponentsWithStats(skin.astype('u1'),8);bd=cv2.distanceTransform((~actor).astype('u1'),cv2.DIST_L2,5);skinnav=[]
  for j in range(1,n):
   comp=la==j
   if stats[j,4]<120 or bd[comp].min()>3:continue
   di=cv2.distanceTransform(comp.astype('u1'),cv2.DIST_L2,5);iy,ix=np.unravel_index(np.argmax(di),di.shape)
   if di[iy,ix]>3:points.append([int(ix),int(iy)]);skinnav.append([int(ix),int(iy)])
  if p in manual:points+=manual[p]
  frmeta=json.load(open(R/'dense56_print_v1/final_frames_v5'/f'{p}.json',encoding='utf8'));mr=frmeta.get('static_motion_ref');neg=[];motion_ref=None
  if mr and Path(mr['file']).exists():
   assert sh(mr['file'])==mr['sha256'];mo=json.load(open(mr['file'],encoding='utf8'));assert mo['native_pts']==p and mo['shot_id']==par['shot_id'] and not mo['cross_cut'];uvs=np.array(mo['current_uv'],float).reshape(-1,2);fb=np.array(mo['forward_backward_error_px'],float);outside=(uvs[:,0]<box[0]-15)|(uvs[:,0]>box[2]+15)|(uvs[:,1]<box[1]-15)|(uvs[:,1]>box[3]+15);ok=outside&(fb<.75);ext=uvs[ok]
   if len(ext)>=3:
    hull=cv2.convexHull(ext.astype('f4'),returnPoints=False).reshape(-1);neg=ext[hull].tolist()
   elif len(ext):neg=ext.tolist()
   motion_ref=mr
  labels=[1]*len(points)+[0]*len(neg);points+=neg
  pr=dict(object_id=1,kind='occluder',origin='original_rgb_candidate',source_binding=bind,points_uv=points,point_labels=labels,box_xyxy=box,exclusion_authority=None);rr=sam.read(rgb,bind,prompts=[pr],reset=True);raw=rr['occluder_region_candidate_mask'].detach().cpu().numpy().astype(bool)
  # Reject disconnected source SAM fragments with no genuine parent-person support.
  n,la,stats,_=cv2.connectedComponentsWithStats(raw.astype('u1'),8);fresh=np.zeros_like(base);cc=[]
  for j in range(1,n):
   comp=la==j;ov=int((comp&actor).sum());supported=ov>=50;cc.append(dict(label=j,bbox_xywh=stats[j,:4].tolist(),pixels=int(stats[j,4]),parent_true_person_overlap_pixels=ov,retained_person_candidate=supported))
   if supported:fresh|=comp
  final=base|fresh;delta=fresh&~base;rf=O/'raw_SAM'/f'{p}.png';ff=O/'masks'/f'{p}.png';cv2.imwrite(str(rf),raw.astype('u1')*255,[cv2.IMWRITE_PNG_COMPRESSION,3]);cv2.imwrite(str(ff),final.astype('u1')*255,[cv2.IMWRITE_PNG_COMPRESSION,3]);row=dict(par);row.update(mask_file=str(ff),mask_sha256=sh(ff),base_mask_ref=dict(file=par['mask_file'],sha256=par['mask_sha256']),mask_pixels=int(final.sum()),parent_mask_ref=dict(file=par['mask_file'],sha256=par['mask_sha256']),raw_current_SAM_ref=dict(file=str(rf),sha256=sh(rf)),source_video_sha256=source['sha256'],source_RGB_sha256=rh,source_BGR_sha256=bh,mask_origin='parent_true_person_OR_current_native_RGB_full_silhouette_SAM_components_with_true_person_support',delta_pixels=int(delta.sum()),removed_parent_true_person_pixels=int((base&~final).sum()),points_uv=points,parent_deep_interior_positive_points_uv=origpoints,current_source_skin_material_search_points_uv=skinnav,source_inspected_positive_points_uv=manual.get(p,[]),bbox_search_only=box,bbox_is_hard_mask=False,static_source_negative_points_uv=neg,static_source_motion_ref=motion_ref,components=cc,person_semantics='actual_person_only_source_SAM_mask_no_box_fill_no_mask_dilation',model_sha256=sam.model_sha256,source_review_status='pending_visual_qualification',parent_unknown_exclusion_ref=par.get('source_unknown_exclusion_mask_ref'));w.write(json.dumps(row,separators=(',',':'))+'\n');w.flush();rows.append(row)
  # Every actual native frame has a crop with original RGB and source contour; no crop clipping at old mask extents.
  yy3,xx3=np.where(final);x0=max(0,int(xx3.min())-40);x1=min(1920,int(xx3.max())+41);y0=max(0,int(yy3.min())-40);y1=min(1080,int(yy3.max())+41);ov=bgr.copy();cs,_=cv2.findContours(final.astype('u1'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(ov,cs,-1,(0,255,0),2);cs,_=cv2.findContours(delta.astype('u1'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(ov,cs,-1,(255,0,255),1);crop=ov[y0:y1,x0:x1];sc=min(430/crop.shape[1],315/crop.shape[0]);crop=cv2.resize(crop,None,fx=sc,fy=sc);tile=np.full((350,440,3),240,'u1');tile[25:25+crop.shape[0],:crop.shape[1]]=crop;cv2.putText(tile,f'{p} +{int(delta.sum())}',(4,20),0,.55,(0,0,0),1);cells.append(tile)
  if len(cells)==12:emit()
  if p in manual or len(rows)%25==1:
   cv2.imwrite(str(O/'QA'/f'full_{p}.jpg'),np.hstack([bgr[y0:y1,x0:x1],ov[y0:y1,x0:x1]]),[cv2.IMWRITE_JPEG_QUALITY,97])
  if len(rows)%25==0:
   (O/'progress.json').write_text(json.dumps(dict(frames=len(rows),total=len(mi),native_pts=p,elapsed_seconds=time.perf_counter()-start)),encoding='utf8');print(p,len(rows),round(time.perf_counter()-start,2),flush=True)
emit();assert len(rows)==1862
(O/'manifest_candidate.json').write_text(json.dumps(dict(complete=True,accepted=False,frames=len(rows),source=source,parent_index=dict(file=str(parentfile),sha256=sh(parentfile)),index=dict(file=str(O/'index.stream.jsonl.gz'),sha256=sh(O/'index.stream.jsonl.gz')),frames_with_person_delta=sum(q['delta_pixels']>0 for q in rows),total_delta_pixels=sum(q['delta_pixels']for q in rows),no_parent_person_pixels_removed=True,all_native_RGB_reread=True,source_prompt_and_box_not_hard_mask=True,source_SAM_model_sha256=sam.model_sha256,raw_UV_or_fitting_generations_unchanged=True,QA=pages,elapsed_seconds=time.perf_counter()-start),indent=2),encoding='utf8');print('COMPLETE',flush=True)
