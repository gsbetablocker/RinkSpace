import sys,json,gzip,hashlib
from pathlib import Path
P=Path(r'D:/OneDrive/codex_projs/RinkSpace');R=Path(r'D:/CodexData/RinkSpace_Data/runs/okada_rec_1019/recover_v2');O=R/'dense56_v1/shoe_fix'
sys.path[:0]=[str(P),str(P/'src'),str(P/'calibration/rinkfit/src')]
from calibration.program_tracking.runtime.bootstrap import add_geometry_dependencies
add_geometry_dependencies()
import cv2,numpy as np,av
cv2.setNumThreads(1)
def rz(p):return[json.loads(l)for l in gzip.open(p,'rt',encoding='utf8')]
mi={q['native_pts']:q for q in rz(R/'dense56_footwear_v1/index.jsonl.gz')};frames={q['native_pts']:q for q in rz(R/'dense56_v1/parts/s05/selected_frames.jsonl.gz')}
wanted={1096192,1135104,1153792,1161216,1170944};source=json.load(open(R/'dense56_v1/parts/s05/video_config.json',encoding='utf8'))['source']
with av.open(source['file'])as c:
 st=c.streams.video[0];c.seek(min(wanted),stream=st,backward=True)
 for fr in c.decode(st):
  p=int(fr.pts)
  if p>max(wanted):break
  if p not in wanted:continue
  rgb=fr.to_ndarray(format='rgb24');bgr=rgb[:,:,::-1].copy();m=cv2.imread(mi[p]['mask_file'],0)>0;over=bgr.copy();cs,_=cv2.findContours(m.astype('u1'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(over,cs,-1,(0,255,0),2)
  print(p,frames[p].keys(),mi[p]['mask_file'],flush=True)
  for r in frames[p].get('points',[]):
   uv=r.get('uv',r.get('source_uv'))
   if uv:cv2.circle(over,tuple(np.rint(uv).astype(int)),2,(255,255,0),1)
  cv2.imwrite(str(O/f'diag_{p}.jpg'),np.vstack([bgr,over]),[cv2.IMWRITE_JPEG_QUALITY,96]);cv2.imwrite(str(O/f'raw_{p}.png'),bgr)
