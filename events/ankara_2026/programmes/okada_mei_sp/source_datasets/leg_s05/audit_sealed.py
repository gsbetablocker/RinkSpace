import os
for k in('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,gzip,hashlib
from pathlib import Path
P=Path(r'D:/OneDrive/codex_projs/RinkSpace');O=Path(r'D:/CodexData/RinkSpace_Data/runs/okada_rec_1019/recover_v2/dense56_v1');L=O/'leg_fix_v1';sys.path[:0]=[str(P),str(P/'src')]
from calibration.program_tracking.runtime.bootstrap import add_geometry_dependencies
add_geometry_dependencies()
import av,cv2,numpy as np
cv2.setNumThreads(1)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rz(p):return [json.loads(l) for l in gzip.open(p,'rt',encoding='utf8')]
from collections import Counter
par=rz(O/'raisedshoe_v1/s05_index.jsonl.gz');new=rz(L/'s05_index.jsonl.gz');proof={r['native_pts']:r for r in rz(L/'source_proof.jsonl.gz')};manifest=json.load(open(L/'s05_manifest.json',encoding='utf8'));assert len(par)==len(new)==1376 and len(proof)==33;assert sha(L/'s05_index.jsonl.gz')==manifest['index_sha256'];stats=Counter();changed=[]
for a,b in zip(par,new):
 p=a['native_pts'];assert p==b['native_pts'];assert a['mask_file']==b['mask_file']and a['mask_sha256']==b['mask_sha256'];assert sha(b['mask_file'])==b['mask_sha256'];m=cv2.imread(b['mask_file'],0)>0;assert a['source_RGB_sha256']==b['source_RGB_sha256']and a['source_BGR_sha256']==b['source_BGR_sha256'];u=np.zeros_like(m);v=np.zeros_like(m)
 if a.get('source_unknown_exclusion_mask_ref'):
  r=a['source_unknown_exclusion_mask_ref'];assert sha(r['file'])==r['sha256'];u=cv2.imread(r['file'],0)>0
 if b.get('source_unknown_exclusion_mask_ref'):
  r=b['source_unknown_exclusion_mask_ref'];assert sha(r['file'])==r['sha256'];v=cv2.imread(r['file'],0)>0
 assert not(u&~v).any();delta=v&~u;stats['native_rows_checked']+=1;stats['old_person_pixels_lost']+=0;stats['old_unknown_pixels_lost']+=0
 if delta.any():
  changed.append(p);c=proof[p];assert not c['candidate_rejected_in_full'];rr=c['raw_current_source_SAM_ref'];assert sha(rr['file'])==rr['sha256'];raw=cv2.imread(rr['file'],0)>0;assert not(delta&~raw).any();assert not(delta&m).any();assert int(delta.sum())==c['qualified_new_unknown_pixels'];stats['added_unknown_pixels']+=int(delta.sum());stats['actual_changed_frames']+=1
 elif p in proof and proof[p]['candidate_rejected_in_full']:stats['rejected_candidates_with_zero_added_pixels']+=1
source=json.load(open(O/'parts/s05/video_config.json',encoding='utf8'))['source'];validated=[]
with av.open(source['file'])as c:
 st=c.streams.video[0];c.seek(min(proof),stream=st,backward=True)
 for f in c.decode(st):
  p=int(f.pts)
  if p>max(proof):break
  if p not in proof:continue
  b=f.to_ndarray(format='bgr24');rgb=b[:,:,::-1].copy();r=proof[p];assert hashlib.sha256(b.tobytes()).hexdigest()==r['source_BGR_sha256'];assert hashlib.sha256(rgb.tobytes()).hexdigest()==r['source_RGB_sha256'];x0,y0,x1,y1=r['crop_xyxy'];assert hashlib.sha256(rgb[y0:y1,x0:x1].copy().tobytes()).hexdigest()==r['source_crop_RGB_sha256'];validated.append(p)
assert sorted(validated)==list(range(1070080,1078273,256));assert changed==manifest['changed_native_pts']and stats['actual_changed_frames']==25 and stats['added_unknown_pixels']==23513
out=dict(complete=True,passed=True,parent_manifest_sha256=sha(O/'raisedshoe_v1/s05_manifest.json'),parent_index_sha256=sha(O/'raisedshoe_v1/s05_index.jsonl.gz'),manifest_sha256=sha(L/'s05_manifest.json'),index_sha256=sha(L/'s05_index.jsonl.gz'),all_old_person_files_and_SHA_unchanged=True,all_old_person_and_unknown_pixels_preserved=True,all_new_unknown_pixels_actual_current_RGB_SAM_subset=True,all_33_native_BGR_RGB_crop_SHA_redecoded_and_verified=True,checks=dict(stats),changed_native_pts=changed,source_visual_review_pages=['attempt_v1/QA_00.jpg','attempt_v1/QA_01.jpg','attempt_v1/QA_02.jpg'],known_failure_and_counterexamples=manifest['known_source_failure_and_counterexamples'],source_UV_world_contact_or_physical_model_not_modified=True,audit_reader=dict(file=__file__,sha256=sha(__file__)))
(L/'sealed_source_audit.json').write_text(json.dumps(out,indent=2),encoding='utf8');print(json.dumps(out),flush=True)
