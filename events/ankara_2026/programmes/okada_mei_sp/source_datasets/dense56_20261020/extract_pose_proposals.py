"""Current native pose proposals for footwear search; never pixel exclusion truth."""
import os
for k in('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,gzip,hashlib,time
from pathlib import Path
P=Path(r'D:/OneDrive/codex_projs/RinkSpace');D=Path(r'D:/CodexData/RinkSpace_Data');R=D/'runs/okada_rec_1019/recover_v2';O=R/'dense56_footwear_v1';O.mkdir(exist_ok=True)
sys.path[:0]=[str(P),str(P/'src'),str(P/'calibration/rinkfit/src')]
from calibration.program_tracking.runtime.bootstrap import add_geometry_dependencies
add_geometry_dependencies()
import cv2,numpy as np,torch,av
from rinkspace.vision_instances import SourcePersonBoxes
cv2.setNumThreads(1);torch.set_num_threads(1)
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb')as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(p,j):Path(p).write_text(json.dumps(j,ensure_ascii=False,indent=2),encoding='utf8')
source=json.load(open(D/'runs/okada_joint9_1018a/programme_scope.json',encoding='utf8'))['source'];expected=list(range(948992,1425408+1,256));assert sha(source['path'])==source['sha256'];masks={}
with gzip.open(R/'dense56_masks_v1/index.jsonl.gz','rt',encoding='utf8')as f:
 for line in f:j=json.loads(line);masks[j['native_pts']]=j
model=SourcePersonBoxes(r'D:/OneDrive/codex_projs/FSpathing/models/yolo11n-pose.pt',dependency_paths=[r'D:/CodexData/SkatingProjects_Archive_20261019/FSpathing/.venv/Lib/site-packages'],runtime_directory=O/'pose_runtime',device='cuda');file=O/'pose_proposals.jsonl.gz';n=0;found=0;t=time.perf_counter()
with gzip.open(file,'xt',encoding='utf8',compresslevel=4)as out,av.open(source['path'])as con:
 st=con.streams.video[0];assert str(st.time_base)=='1/12800';con.seek(expected[0],stream=st,backward=True)
 for frame in con.decode(st):
  p=int(frame.pts)
  if p<expected[0]:continue
  if p>expected[-1]:break
  assert p==expected[n];bgr=frame.to_ndarray(format='bgr24');rgb=bgr[:,:,::-1].copy();bh=hashlib.sha256(bgr.tobytes()).hexdigest();rh=hashlib.sha256(rgb.tobytes()).hexdigest();mi=masks[p];assert bh==mi['source_BGR_sha256']and rh==mi['source_RGB_sha256']
  pred=model.model.predict(source=bgr,device=model.device.index or 0,verbose=False,save=False)[0];boxes=pred.boxes.xyxy.cpu().numpy();confidence=pred.boxes.conf.cpu().numpy();classes=pred.boxes.cls.cpu().numpy();keypoints=pred.keypoints.xy.cpu().numpy()if pred.keypoints is not None else None;kc=pred.keypoints.conf.cpu().numpy()if pred.keypoints is not None and pred.keypoints.conf is not None else None
  mask=cv2.imread(mi['mask_file'],0);assert mask is not None and sha(mi['mask_file'])==mi['mask_sha256'];nn,labels,stats,_=cv2.connectedComponentsWithStats((mask>0).astype('u1'),8);main=np.zeros(mask.shape,bool)
  if nn>1:main=labels==(1+int(np.argmax(stats[1:,4])))
  area=int(main.sum());persons=[]
  for i,(box,conf,cls)in enumerate(zip(boxes,confidence,classes)):
   if int(cls)!=0:continue
   x0,y0,x1,y1=box;lo=np.maximum(np.floor([x0,y0]).astype(int),0);hi=np.minimum(np.ceil([x1,y1]).astype(int),[1920,1080]);intersection=int(main[lo[1]:hi[1],lo[0]:hi[0]].sum());score=intersection/max(area+float(max(0,x1-x0)*max(0,y1-y0))-intersection,1)
   persons.append(dict(prediction_index=i,box_xyxy=box.tolist(),box_confidence=float(conf),keypoints_uv=keypoints[i].tolist()if keypoints is not None else None,keypoint_confidence=kc[i].tolist()if kc is not None else None,original_main_SAM_pixel_overlap=int(intersection),original_main_SAM_pixel_IoU=float(score)))
  selected=int(np.argmax([q['original_main_SAM_pixel_IoU']for q in persons]))if persons and max(q['original_main_SAM_pixel_overlap']for q in persons)>0 else None;found+=selected is not None
  row=dict(shot_id=mi['shot_id'],native_pts=p,time_base='1/12800',source_BGR_sha256=bh,source_RGB_sha256=rh,source_video_sha256=source['sha256'],person_proposals=persons,selected_person_list_index=selected,selected_keypoints_uv=persons[selected]['keypoints_uv']if selected is not None else None,selected_keypoint_confidence=persons[selected]['keypoint_confidence']if selected is not None else None,selection_is_source_search_proposal_only=True,original_person_mask=dict(file=mi['mask_file'],sha256=mi['mask_sha256']),keypoints_are_image_model_proposals_not_pixel_or_body_truth=True,bbox_or_pose_used_as_hard_exclusion=False,physical_world_points_or_contacts_issued=False,model_sha256=model.weights_sha256)
  out.write(json.dumps(row,separators=(',',':'))+'\n');n+=1
  if n%50==0:out.flush();write(O/'pose_progress.json',dict(frames=n,total=1862,native_pts=p,selected_proposals=found,elapsed_seconds=time.perf_counter()-t))
assert n==1862
manifest=dict(schema='rinkspace.native_footwear_pose_search_proposals.v1',complete=True,frames=n,selected_person_proposals=found,source_video_sha256=source['sha256'],source_native_RGB_BGR_verified_every_frame=True,rows=dict(file=str(file),sha256=sha(file)),model_sha256=model.weights_sha256,source_reader=dict(file=str(Path(__file__).resolve()),sha256=sha(__file__)),all_poses_are_current_image_model_search_proposals=True,no_pose_or_box_used_as_exclusion=True,no_ground_truth_or_physical_contact_claim=True,elapsed_seconds=time.perf_counter()-t);write(O/'pose_manifest.json',manifest);print(json.dumps(manifest),flush=True)
