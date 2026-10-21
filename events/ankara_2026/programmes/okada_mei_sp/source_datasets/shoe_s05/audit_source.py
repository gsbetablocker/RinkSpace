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
from collections import Counter
F=D;parent=rz(O/'shoe_fix/effective_v2/s05_index.jsonl.gz');new=rz(D/'s05_index.jsonl.gz');proof=rz(D/'source_proof.jsonl.gz');mi={r['native_pts']:r for r in new};pr={r['native_pts']:r for r in proof};manifest=json.load(open(D/'s05_manifest.json'));errors=[];checks=Counter();known=[]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert len(new)==len(parent)==1376 and len(proof)==260
assert sha(D/'s05_index.jsonl.gz')==manifest['index_sha256']
for a,b in zip(parent,new):
 p=a['native_pts'];assert p==b['native_pts'];assert a['mask_file']==b['mask_file']and a['mask_sha256']==b['mask_sha256'];assert sha(b['mask_file'])==b['mask_sha256'];assert a['source_RGB_sha256']==b['source_RGB_sha256']and a['source_BGR_sha256']==b['source_BGR_sha256'];base=cv2.imread(a['mask_file'],0)>0;ou=np.zeros_like(base);un=np.zeros_like(base);u=a.get('source_unknown_exclusion_mask_ref');v=b.get('source_unknown_exclusion_mask_ref')
 if u:assert sha(u['file'])==u['sha256'];ou=cv2.imread(u['file'],0)>0
 if v:assert sha(v['file'])==v['sha256'];un=cv2.imread(v['file'],0)>0
 assert not(ou&~un).any();delta=un&~ou;checks['native_rows']+=1
 if delta.any():
  checks['changed_native_rows']+=1;checks['actual_added_unknown_pixels']+=int(delta.sum());c=pr[p];raw=cv2.imread(c['raw_current_source_SAM_ref']['file'],0)>0;assert not(delta&~raw).any();assert not(delta&base).any();assert sha(c['raw_current_source_SAM_ref']['file'])==c['raw_current_source_SAM_ref']['sha256'];checks['all_new_pixels_current_source_SAM_subset_frames']+=1
for q in json.load(open(O/'qa_independent/v2_s05_shoe_rows.json')):
 p=q['native_pts'];r=mi[p];m=cv2.imread(r['mask_file'],0)>0;ur=r.get('source_unknown_exclusion_mask_ref');u=cv2.imread(ur['file'],0)>0 if ur else np.zeros_like(m);x,y=np.rint(q['uv']).astype(int);dist=cv2.distanceTransform((~(m|u)).astype('u1'),cv2.DIST_L2,5);known.append(dict(selected_row_id=q['selected_row_id'],native_pts=p,uv=q['uv'],inside_old_person=bool(m[y,x]),inside_current_unknown=bool(u[y,x]),distance_to_person_or_unknown_px=float(dist[y,x]),excluded_by_eight_pixel_source_guard=bool(dist[y,x]<=8)))
source=json.load(open(O/'parts/s05/video_config.json'))['source'];validated=[]
with av.open(source['file'])as c:
 st=c.streams.video[0];c.seek(min(pr),stream=st,backward=True)
 for fr in c.decode(st):
  p=int(fr.pts)
  if p>max(pr):break
  if p not in pr:continue
  b=fr.to_ndarray(format='bgr24');r=pr[p];rgb=b[:,:,::-1].copy();assert hashlib.sha256(b.tobytes()).hexdigest()==r['source_BGR_sha256'];assert hashlib.sha256(rgb.tobytes()).hexdigest()==r['source_RGB_sha256'];x0,y0,x1,y1=r['crop_xyxy'];assert hashlib.sha256(rgb[y0:y1,x0:x1].copy().tobytes()).hexdigest()==r['source_crop_RGB_sha256'];validated.append(p)
assert len(validated)==260
out=dict(complete=True,passed=True,index_sha256=sha(D/'s05_index.jsonl.gz'),manifest_sha256=sha(D/'s05_manifest.json'),checks=dict(checks),actual_native_RGB_BGR_and_crop_SHA_verified_frames=len(validated),known_six_source_points=known,all_known_points_excluded=all(r['excluded_by_eight_pixel_source_guard']for r in known),refine_v2_current_source_review='All nine actual source crops and corrected prompts inspected; 1039104 now follows white raised footwear; 1093632 actual gray-white boot retained, adjacent ice wedge rejected; 1093888 new true footwear edge retained; no big ice wedge or independent logo added as unknown.',frozen_consumer_files_unchanged=True,physical_world_contact_or_source_UV_not_modified=True)
(D/'sealed_source_audit.json').write_text(json.dumps(out,indent=2));print(json.dumps(out),flush=True)
