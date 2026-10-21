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
rows={r['native_pts']:r for r in rz(O/'shoe_fix/effective_v2/s05_index.jsonl.gz')};source=json.load(open(O/'parts/s05/video_config.json'))['source'];anchors={int(p):uv for p,uv in json.load(open(D/'navigation_anchors.json'))['anchors'].items()};first=min(anchors)-512;last=max(anchors)+512;frames={};grays={};records=[]
with av.open(source['file'])as c:
 st=c.streams.video[0];c.seek(first,stream=st,backward=True)
 for fr in c.decode(st):
  p=int(fr.pts)
  if p>last:break
  if p<first:continue
  b=fr.to_ndarray(format='bgr24');frames[p]=b;grays[p]=cv2.cvtColor(b,cv2.COLOR_BGR2GRAY)
for p,b in frames.items():
 a=min(anchors,key=lambda a:abs(a-p));uv=np.array(anchors[a],np.float32).reshape(1,1,2);q=uv.copy();fb=0.;errors=[];valid=True
 for k in range(a,p,256 if p>a else-256):
  z=k+(256 if p>a else-256);n,st,err=cv2.calcOpticalFlowPyrLK(grays[k],grays[z],q,None,winSize=(31,31),maxLevel=4,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,40,.001));back,stb,erb=cv2.calcOpticalFlowPyrLK(grays[z],grays[k],n,None,winSize=(31,31),maxLevel=4,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,40,.001));e=float(np.linalg.norm(back-q));valid=valid and bool(st[0,0]and stb[0,0])and e<1.5;fb=max(fb,e);errors.append(float(err[0,0]));q=n
 uv=q[0,0];rec=dict(native_pts=p,anchor_native_pts=a,source_prompt_uv=uv.tolist(),anchor_source_prompt_uv=anchors[a],max_step_forward_backward_error_px=fb,step_patch_L1_error=errors,accepted_navigation=valid,native_frame_steps=abs(a-p)//256,actual_RGB_LK_navigation_only=True,model_interpolation_used=False,source_BGR_sha256=hashlib.sha256(b.tobytes()).hexdigest(),source_RGB_sha256=hashlib.sha256(b[:,:,::-1].copy().tobytes()).hexdigest());records.append(rec)
(D/'LK_navigation.json').write_text(json.dumps(records,indent=2));print('VALID',sum(r['accepted_navigation']for r in records),'TOTAL',len(records));print('BAD',[(r['native_pts'],r['max_step_forward_backward_error_px'])for r in records if not r['accepted_navigation']])
cells=[];j=0
for r in records:
 p=r['native_pts'];x,y=np.rint(r['source_prompt_uv']).astype(int);b=frames[p];x0=max(0,x-90);y0=max(0,y-75);crop=b[y0:min(1080,y+76),x0:min(1920,x+91)].copy();cv2.drawMarker(crop,(x-x0,y-y0),(0,0,255),cv2.MARKER_CROSS,9,1);tile=np.full((180,190,3),240,'u1');tile[25:25+crop.shape[0],:crop.shape[1]]=crop;cv2.putText(tile,f'{p} FB{r["max_step_forward_backward_error_px"]:.1f}',(2,18),0,.4,(0,0,0),1);cells.append(tile)
 if len(cells)==24:
  cv2.imwrite(str(D/f'LK_QA_{j:02}.jpg'),np.vstack([np.hstack(cells[k:k+6])for k in(0,6,12,18)]));j+=1;cells=[]
if cells:
 cells+=[np.full((180,190,3),240,'u1')for _ in range(24-len(cells))];cv2.imwrite(str(D/f'LK_QA_{j:02}.jpg'),np.vstack([np.hstack(cells[k:k+6])for k in(0,6,12,18)]))
