import sys,json,gzip,hashlib,time
from pathlib import Path
P=Path(r'D:/OneDrive/codex_projs/RinkSpace');R=Path(r'D:/CodexData/RinkSpace_Data/runs/okada_rec_1019/recover_v2');O=R/'dense56_v1/shoe_fix';sys.path[:0]=[str(P),str(P/'src'),str(P/'calibration/rinkfit/src')]
from calibration.program_tracking.runtime.bootstrap import add_geometry_dependencies
add_geometry_dependencies()
import cv2,numpy as np
cv2.setNumThreads(1)
def sh(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rz(p):
 out=[]
 with gzip.open(p,'rt',encoding='utf8')as f:
  try:
   for l in f:out.append(json.loads(l))
  except EOFError:pass
 return out
mode=sys.argv[1]if len(sys.argv)>1 else'full_v3';a=rz(O/mode/'index.stream.jsonl.gz');rows=[];risks=[]
for r in a:
 p=r['native_pts'];new=cv2.imread(r['mask_file'],0)>0;base=cv2.imread(r['parent_mask_ref']['file'],0)>0;delta=new&~base;mr=r.get('static_source_motion_ref');morph=None;ct=0;conflicts=[];outside_hits=0
 if mr:
  assert sh(mr['file'])==mr['sha256'];mo=json.load(open(mr['file'],encoding='utf8'));prev=np.array(mo['source_uv'],'f4').reshape(-1,2);curr=np.array(mo['current_uv'],'f4').reshape(-1,2);fb=np.array(mo['forward_backward_error_px']);valid=np.isfinite(prev).all(1)&np.isfinite(curr).all(1)&(fb<.75)
  if valid.sum()>=6:
   mat,ins=cv2.estimateAffinePartial2D(prev[valid],curr[valid],method=cv2.RANSAC,ransacReprojThreshold=.75,maxIters=2000,confidence=.995);morph=None if mat is None else mat.tolist()
   if mat is not None:
    err=np.linalg.norm(np.c_[prev,np.ones(len(prev))]@mat.T-curr,axis=1);x=np.clip(np.rint(curr[:,0]).astype(int),0,1919);y=np.clip(np.rint(curr[:,1]).astype(int),0,1079);box=r['bbox_search_only'];outside=(curr[:,0]<box[0])|(curr[:,0]>box[2])|(curr[:,1]<box[1])|(curr[:,1]>box[3]);stat=valid&(err<.35);ct=int(stat.sum());bad=stat&delta[y,x];outside_hits=int((bad&outside).sum());bd=cv2.distanceTransform((~base).astype('u1'),cv2.DIST_L2,5)
    for j in np.flatnonzero(bad):
     conflicts.append(dict(previous_uv=prev[j].tolist(),current_uv=curr[j].tolist(),FB_error=float(fb[j]),static_model_residual=float(err[j]),distance_parent_person=float(bd[y[j],x[j]]),outside_actor_navigation_domain=bool(outside[j])))
 rr=dict(native_pts=p,shot_id=r['shot_id'],source_RGB_sha256=r['source_RGB_sha256'],new_mask_sha256=r['mask_sha256'],parent_mask_sha256=r['parent_mask_ref']['sha256'],old_person_pixels_lost=int((base&~new).sum()),added_pixels=int(delta.sum()),static_motion_ref=mr,actual_static_affine=morph,actual_static_source_inlier_corners=ct,new_person_conflicting_actual_static_material=conflicts,outside_actor_navigation_domain_conflicts=outside_hits);rows.append(rr)
 if outside_hits or any(q['distance_parent_person']>20 for q in conflicts):risks.append(dict(native_pts=p,delta=r['delta_pixels'],conflict_count=len(conflicts),outside=outside_hits,max_distance=max([q['distance_parent_person']for q in conflicts],default=0)))
report=dict(frames=len(a),old_person_pixels_lost=sum(r['old_person_pixels_lost']for r in rows),frames_with_static_motion=sum(r['static_motion_ref']is not None for r in rows),frames_with_actual_static_affine=sum(r['actual_static_affine']is not None for r in rows),static_source_corners=sum(r['actual_static_source_inlier_corners']for r in rows),outside_conflict_frames=sum(r['outside_actor_navigation_domain_conflicts']>0 for r in rows),risks=risks,not_acceptance=True)
(O/mode/'static_counterexample_audit.json').write_text(json.dumps(report,indent=2),encoding='utf8')
with gzip.open(O/mode/'static_counterexample_audit.jsonl.gz','wt',encoding='utf8')as f:
 for r in rows:f.write(json.dumps(r,separators=(',',':'))+chr(10))
print(json.dumps(report))
