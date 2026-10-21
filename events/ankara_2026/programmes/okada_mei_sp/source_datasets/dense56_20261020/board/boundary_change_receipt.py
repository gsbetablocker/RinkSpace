import json,time,gzip,hashlib,numpy as np
from pathlib import Path
O=Path(r'D:/CodexData/RinkSpace_Data/runs/okada_rec_1019/recover_v2/dense56_board_v1/material_v5');B=Path(r'D:/CodexData/RinkSpace_Data/runs/okada_rec_1019/recover_v2/dense56_board_v1/shape_v4');PTS=list(range(948992,1425408+1,256))
def j(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def span(b):
 d=np.diff(np.r_[False,b,False].astype('i1'));return [[int(a),int(v-1)]for a,v in zip(np.flatnonzero(d==1),np.flatnonzero(d==-1))]
maxdiff={'board.top':0.,'board.contact':0.};framecounts={'board.top':0,'board.contact':0};oldmissing=[];n=0
with gzip.open(O/'v4_v5_boundary_changes.stream.jsonl.gz','wt',encoding='utf8')as f:
 for p in PTS:
  a=B/'anchored_frames'/(str(p)+'.json');b=O/'anchored_frames'/(str(p)+'.json')
  while not b.exists():time.sleep(.2)
  nr=j(b)
  if not a.exists():oldmissing.append(p);f.write(json.dumps(dict(native_pts=p,shot_id=nr['shot_id'],old_generation_frame_absent=True,new_anchored_file=str(b),new_anchored_sha256=sha(b)),separators=(',',':'))+'\n');continue
  old=j(a);rr={};expanded=np.zeros(1920,bool)
  for role in ['board.top','board.contact']:
   x=np.array(old['roles'][role]['full_native_y']);y=np.array(nr['roles'][role]['full_native_y']);d=abs(y-x);changed=d>.5;maxdiff[role]=max(maxdiff[role],float(d.max()));framecounts[role]+=bool(changed.any());rr[role]=dict(max_abs_delta_px=float(d.max()),median_abs_delta_px=float(np.median(d)),changed_x_spans=span(changed));expanded|=y<x-.5 if role=='board.top'else y>x+.5
  f.write(json.dumps(dict(native_pts=p,shot_id=nr['shot_id'],old_anchored_file=str(a),old_anchored_sha256=sha(a),new_anchored_file=str(b),new_anchored_sha256=sha(b),same_source_BGR=old['decoded_native_BGR_sha256']==nr['decoded_native_BGR_sha256'],roles=rr,new_board_interior_expansion_x_spans=span(expanded),newly_admitted_board_band_requires_current_RGB_print_requalification=True,source_UV_not_transferred=True),separators=(',',':'))+'\n');n+=1
(O/'v4_v5_boundary_changes.stream.jsonl.gz').replace(O/'v4_v5_boundary_changes.jsonl.gz')
z=dict(complete=True,frames=1862,compared_frames=n,old_missing_frames=oldmissing,max_difference_px=maxdiff,frames_changed_over_half_px=framecounts,receipt_file=str(O/'v4_v5_boundary_changes.jsonl.gz'),sha256=sha(O/'v4_v5_boundary_changes.jsonl.gz'));(O/'v4_v5_boundary_changes_manifest.json').write_text(json.dumps(z,separators=(',',':')),encoding='utf8');print(json.dumps(z),flush=True)
