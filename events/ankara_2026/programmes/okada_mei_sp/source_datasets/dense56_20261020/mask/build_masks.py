import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,gzip,hashlib,time
from pathlib import Path
P=Path(r'D:/OneDrive/codex_projs/RinkSpace');D=Path(r'D:/CodexData/RinkSpace_Data');R=D/'runs/okada_rec_1019/recover_v2';J=D/'runs/okada_joint9_1018a';O=R/'dense56_masks_v1'
sys.path[:0]=[str(P),str(P/'src'),str(P/'calibration/rinkfit/src')]
from calibration.program_tracking.runtime.bootstrap import add_geometry_dependencies
add_geometry_dependencies()
import numpy as np,cv2,torch,av
from rinkspace.vision_instances import Sam2SourceInstances,SourcePersonBoxes
from rinkfit_joint.io_utils import file_sha256
cv2.setNumThreads(1);torch.set_num_threads(1)
def loadj(p):return json.loads(Path(p).read_text(encoding='utf8'))
def writej(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf8')
def main():
 start=time.perf_counter();ledger={}
 for dn in ('dense_s03_v1','dense_remaining_v1'):
  with gzip.open(J/dn/'frame_ledger.jsonl.gz','rt',encoding='utf8')as f:
   for l in f:
    q=json.loads(l)
    if 948992<=q['native_pts']<=1425408:ledger[q['native_pts']]=q
 assert len(ledger)==1862
 scope=loadj(J/'programme_scope.json');source=scope['source'];assert file_sha256(source['path'])==source['sha256']
 sam=Sam2SourceInstances(r'D:/CodexData/MotionTrail_Archive_20261019/models/sam2.1-hiera-tiny',device='cuda',reread_rgb=False,contour_stride=16)
 detector=None;rows=[json.loads(l) for l in (O/'index.stream.jsonl').read_text(encoding='utf8').splitlines() if l.strip()];counts={};
 for r in rows:
  assert file_sha256(r['mask_file'])==r['mask_sha256'];counts[r['mask_origin']]=counts.get(r['mask_origin'],0)+1
 expected=list(range(948992,1425408+1,256))
 with av.open(source['path'])as c,(O/'index.stream.jsonl').open('a',encoding='utf8')as index:
  st=c.streams.video[0];assert str(st.time_base)=='1/12800';c.seek(expected[len(rows)],stream=st,backward=True)
  for frame in c.decode(st):
   pts=int(frame.pts)
   if len(rows)>=len(expected)or pts>expected[-1]:break
   if pts<expected[len(rows)]:continue
   assert pts==expected[len(rows)]
   old=ledger[pts];sid=old['shot_id'];bgr=frame.to_ndarray(format='bgr24');rgb=bgr[:,:,::-1].copy();bgrsha=hashlib.sha256(bgr.tobytes()).hexdigest();rgbsha=hashlib.sha256(rgb.tobytes()).hexdigest()
   if old.get('decoded_native_BGR_sha256'):assert old['decoded_native_BGR_sha256']==bgrsha
   binding=dict(source_sha256=source['sha256'],source_pts=pts,time_base='1/12800',native_hw=[1080,1920],shot_id=sid,decoded_native_rgb_sha256=rgbsha)
   prog=(pts-256256)//256+1;existing=D/f'events/ankara_2026/programmes/okada_mei_sp/results/shot{int(sid[1:])}_review_20261014/masks_current/{prog:06}.png'
   boxes=old.get('current_person_boxes');reuse=False
   if existing.is_file():
    mask=cv2.imread(str(existing),cv2.IMREAD_GRAYSCALE)>0;kind='reused_existing_native_SAM2_pixel_mask';original=dict(file=str(existing),sha256=file_sha256(existing));reuse=True
   else:
    if boxes is None:
     if detector is None:detector=SourcePersonBoxes(r'D:/OneDrive/codex_projs/FSpathing/models/yolo11n-pose.pt',dependency_paths=[r'D:/CodexData/SkatingProjects_Archive_20261019/FSpathing/.venv/Lib/site-packages'],runtime_directory=O/'detector_runtime',device='cuda')
     boxes=detector.read(rgb,binding)
    for box in boxes:
     assert box['source_binding']['source_pts']==pts and box['source_binding']['decoded_native_rgb_sha256']==rgbsha
    prompts=[dict(object_id=i+1,kind='occluder',origin='original_rgb_candidate',source_binding=binding,box_xyxy=box['box_xyxy'],exclusion_authority=None)for i,box in enumerate(boxes)]
    if prompts:
     current=sam.read(rgb,binding,prompts=prompts,reset=True);mask=current['occluder_region_candidate_mask'].detach().cpu().numpy().astype(bool)
    else:mask=np.zeros((1080,1920),bool)
    kind='current_RGB_SAM2_pixel_mask_existing_model';original=None
   assert mask.shape==(1080,1920)
   path=O/'masks'/f'{pts}.png';cv2.imwrite(str(path),mask.astype(np.uint8)*255,[cv2.IMWRITE_PNG_COMPRESSION,3])
   row=dict(shot_id=sid,native_pts=pts,programme_frame_index_1based=prog,mask_file=str(path),mask_sha256=file_sha256(path),mask_boolean_sha256=hashlib.sha256(mask.tobytes()).hexdigest(),source_BGR_sha256=bgrsha,source_RGB_sha256=rgbsha,source_video_sha256=source['sha256'],mask_origin=kind,existing_mask=original,old_frame_role=old['role'],current_boxes=boxes,mask_pixels=int(mask.sum()),bbox_used_as_hard_exclusion=False,model_sha256=sam.model_sha256,graphics_mask_included=False,task_current_all_frames_source_annotation=True)
   rows.append(row);index.write(json.dumps(row,separators=(',',':'))+'\n');index.flush();counts[kind]=counts.get(kind,0)+1
   if len(rows)%25==0:
    progress=dict(done=len(rows),total=1862,native_pts=pts,shot_id=sid,counts=counts,elapsed_seconds=time.perf_counter()-start);writej(O/'progress.json',progress);print(json.dumps(progress),flush=True)
 with gzip.open(O/'index.jsonl.gz','wt',encoding='utf8')as f:
  for r in rows:f.write(json.dumps(r,separators=(',',':'))+'\n')
 assert len(rows)==1862
 writej(O/'manifest.json',dict(frames=len(rows),counts=counts,source=source,pixel_masks=True,bbox_hard_exclusion=False,current_all_frames_annotation=True,old_source_roles_preserved=True,solver_run=False,model_weights_reused=True,elapsed_seconds=time.perf_counter()-start))
 print('COMPLETE',len(rows),flush=True)
if __name__=='__main__':main()
