"""Actual native RGB affine context support gate for immutable wide readouts."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys
sys.path[:0]=[r'D:/OneDrive/codex_projs/RinkSpace/calibration/rinkfit/src',r'D:/OneDrive/codex_projs/RinkSpace']
from pathlib import Path
from collections import defaultdict
import json,av,cv2,numpy as np,hashlib
from rinkfit_joint.source_texture_tracklets import _source_mask
from rinkfit_joint.io_utils import read_jsonl,file_sha256

def audit(config,readouts,output):
 out=Path(output);out.mkdir(exist_ok=True,parents=True);cfg=json.loads(Path(config).read_text(encoding='utf8'));rr=[json.loads(s) for s in Path(readouts).read_text(encoding='utf8').splitlines()];asks=defaultdict(list)
 if file_sha256(cfg['source']['path'])!=cfg['source']['sha256']:raise ValueError('Changed official source')
 for q in rr:
  if q['status']=='wide_context_direct_source_supported':asks[q['native_pts']].append((q,'forward'));asks[q['birth_native_pts']].append((q,'reverse'))
 led={x['native_pts']:x for f in cfg['actor_ledgers'] for x in read_jsonl(f) if x['native_pts'] in asks};cases=[];proof=[];perrow=defaultdict(list)
 with av.open(cfg['source']['path']) as c:
  st=c.streams.video[0];assert str(st.time_base)=='1/12800';c.seek(min(asks),stream=st,backward=True)
  for fr in c.decode(st):
   p=int(fr.pts)
   if p<min(asks):continue
   if p>max(asks):break
   if p not in asks:continue
   rgb=fr.to_ndarray(format='rgb24');mask,dig=_source_mask(rgb,led[p],cfg['source'],p,'s02')
   for guv in cfg.get('guard_source_uv_by_pts',{}).get(str(p),[]):cv2.circle(mask,tuple(np.rint(guv).astype(int)),33,255,-1)
   for rg in cfg['regions']:
    if rg['native_pts']==p and rg.get('native_person_mask'):
     mr=rg['native_person_mask'];assert file_sha256(mr['file'])==mr['sha256'];mm=cv2.imread(mr['file'],cv2.IMREAD_GRAYSCALE);mask=np.maximum(mask,cv2.dilate((mm>0).astype(np.uint8)*255,np.ones((33,33),np.uint8)))
   for q,direction in asks[p]:
    cor=np.asarray(q[direction]['registered_corners_native_uv']);lo=np.floor(cor.min(0)-1).astype(int);hi=np.ceil(cor.max(0)+1).astype(int);edge=lo.min()<0 or np.any(hi>=[1920,1080]);num=-1 if edge else int(np.count_nonzero(mask[lo[1]:hi[1]+1,lo[0]:hi[0]+1]));item=dict(source_row_id=q['source_row_id'],native_pts=p,direction=direction,actual_RGB_sha256=dig,mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest(),registered_corners_native_uv=cor.tolist(),conservative_affine_support_bbox_plus_1px_halo=[lo.tolist(),hi.tolist()],excluded_pixels_in_bbox=num,status='pass_full_affine_context_support' if num==0 else 'unknown_actual_affine_context_support');proof.append(item);perrow[q['source_row_id']].append(item)
    if num:cases.append(item)
 bad={q['source_row_id'] for q in cases}
 for q in rr:
  if q['source_row_id'] in perrow:q['full_affine_context_support']=perrow[q['source_row_id']]
  if q['source_row_id'] in bad:q.update(prior_wide_registration_status=q['status'],status='unknown_full_affine_context_support',unknown_reason='actual_registered_full_context_plus1px_sampler_halo_intersects_exclusion',current_uv=q['original_uv'],source_UV_changed=False)
 with (out/'final_readout_rows.jsonl').open('w',encoding='utf8') as f:
  for q in rr:f.write(json.dumps(q,separators=(',',':'),allow_nan=False)+'\n')
 report=dict(schema='rinkfit.actual_wide_context_affine_support_audit.v1',input_readout_records=dict(file=str(Path(readouts).resolve()),sha256=file_sha256(readouts)),reader=dict(file=str(Path(__file__).resolve()),sha256=file_sha256(__file__)),output_final_readout_records=dict(file=str((out/'final_readout_rows.jsonl').resolve()),sha256=file_sha256(out/'final_readout_rows.jsonl')),original_total_rows=len(rr),accepted_readouts_audited=len(perrow),actual_direction_supports=len(proof),failed_direction_supports=len(cases),failed_source_rows=len(bad),all_direction_supports=proof,failed=cases,policy='registered affine corners conservative complete bounding rectangle plus1px sampler halo against exact RGB-bound actor/graphics/native-mask and existingguard exclusions; actual current and birth frames separately read; no model')
 (out/'full_affine_support_mask_QA.json').write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps({k:report[k] for k in ('original_total_rows','actual_direction_supports','failed_source_rows','output_final_readout_records')}));return report
if __name__=='__main__':audit(*sys.argv[1:])
