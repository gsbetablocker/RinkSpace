import os
for k in('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
exec(open(r'D:/CodexData/RinkSpace_Data/runs/okada_rec_1019/recover_v2/dense56_v1/shoe_fix/test_bg_detector.py',encoding='utf8').read().split("yolo=SourcePersonBoxes")[0])
import gzip,time,av
from rinkspace.vision_instances import SourcePersonBoxes
O=O/'background_v1';O.mkdir(exist_ok=False)
def sh(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rz(p):return[json.loads(l)for l in gzip.open(p,'rt',encoding='utf8')]
mi={q['native_pts']:q for q in rz(R/'dense56_footwear_v1/index.jsonl.gz')};bc={q['native_pts']:q for q in rz(R/'dense56_board_v1/material_v5/qualified_coverage.jsonl.gz')};source=json.load(open(R/'dense56_v1/parts/s05/video_config.json',encoding='utf8'))['source']
yolo=SourcePersonBoxes(r'D:/OneDrive/codex_projs/FSpathing/models/yolo11n-pose.pt',dependency_paths=[r'D:/CodexData/SkatingProjects_Archive_20261019/FSpathing/.venv/Lib/site-packages'],runtime_directory=O/'runtime',device='cuda');rows=[];start=time.perf_counter()
with av.open(source['file'])as c,gzip.open(O/'proposals.jsonl.gz','wt',encoding='utf8')as w:
 st=c.streams.video[0];c.seek(min(mi),stream=st,backward=True)
 for fr in c.decode(st):
  p=int(fr.pts)
  if p>max(mi):break
  if p not in mi:continue
  par=mi[p];bgr=fr.to_ndarray(format='bgr24');rgb=bgr[:,:,::-1].copy();bh=hashlib.sha256(bgr.tobytes()).hexdigest();rh=hashlib.sha256(rgb.tobytes()).hexdigest();assert bh==par['source_BGR_sha256']and rh==par['source_RGB_sha256'];trace=np.array(bc[p]['roles']['board.top']['trace_uv'],float).reshape(-1,2);props=[]
  pred=yolo.model.predict(source=bgr,device=0,imgsz=1280,conf=.05,verbose=False,save=False)[0]
  for k,(bb,conf,cl)in enumerate(zip(pred.boxes.xyxy.cpu().numpy(),pred.boxes.conf.cpu().numpy(),pred.boxes.cls.cpu().numpy())):
   if int(cl)!=0:continue
   x0,y0,x1,y1=bb;inspan=trace[(trace[:,0]>=x0-8)&(trace[:,0]<=x1+8)];yy=float(np.median(inspan[:,1]))if len(inspan)else float(np.interp((x0+x1)/2,trace[:,0],trace[:,1]))if len(trace)else None
   if yy is None:continue
   relevant=y0<yy+8 and y1>yy-20 and y1<yy+80 and y1-y0>40
   if relevant:props.append(dict(proposal_id=k,box_xyxy=bb.tolist(),confidence=float(conf),keypoints_uv=pred.keypoints.xy[k].cpu().numpy().tolist(),keypoints_confidence=pred.keypoints.conf[k].cpu().numpy().tolist(),source_board_top_navigation_y=yy,source_relevance='person_search_proposal_reaches_actual_board_top_material_readout_footprint',not_person_truth=True))
  row=dict(native_pts=p,shot_id=par['shot_id'],source_BGR_sha256=bh,source_RGB_sha256=rh,proposals=props,board_qualified_ref=dict(file=bc[p]['qualified_file'],sha256=bc[p]['qualified_file_sha256']),box_never_hard_mask=True,model_sha256=yolo.weights_sha256);w.write(json.dumps(row,separators=(',',':'))+chr(10));rows.append(row)
  if len(rows)%50==0:w.flush();(O/'progress.json').write_text(json.dumps(dict(frames=len(rows),total=1862,proposal_instances=sum(len(r['proposals'])for r in rows),native_pts=p,seconds=time.perf_counter()-start)),encoding='utf8');print(len(rows),p,sum(len(r['proposals'])for r in rows),round(time.perf_counter()-start,2),flush=True)
assert len(rows)==1862
(O/'manifest_proposals.json').write_text(json.dumps(dict(complete=True,frames=len(rows),proposal_instances=sum(len(r['proposals'])for r in rows),source_video_sha256=source['sha256'],index=dict(file=str(O/'proposals.jsonl.gz'),sha256=sh(O/'proposals.jsonl.gz')),person_truth_or_mask_issued=False,all_native_BGR_RGB_verified=True,source_reader=dict(file=__file__,sha256=sh(__file__)),elapsed_seconds=time.perf_counter()-start),indent=2),encoding='utf8');print('COMPLETE',flush=True)
