import av,cv2,numpy as np,json,sys
from source_reader_anchored import current_graphics
from pathlib import Path
O=Path(r'D:/CodexData/RinkSpace_Data/runs/okada_rec_1019/recover_v2/dense56_board_v1/material_v5');J=Path(r'D:/CodexData/RinkSpace_Data/runs/okada_joint9_1018a')
pts=list(map(int,sys.argv[1].split(',')));name=sys.argv[2];tiles=[]
scope=json.loads((J/'programme_scope.json').read_text(encoding='utf8'))
with av.open(scope['source']['path'])as c:
 s=c.streams.video[0];c.seek(min(pts),stream=s,backward=True)
 for f in c.decode(s):
  p=int(f.pts)
  if p>max(pts):break
  if p not in pts:continue
  qf=O/'qualified_frames'/(str(p)+'.json');af=O/'anchored_frames'/(str(p)+'.json');raw=json.loads(af.read_text(encoding='utf8'));q=json.loads(qf.read_text(encoding='utf8'))if qf.exists()else None;im=f.to_ndarray(format='bgr24');out=im.copy();mask=cv2.imread(raw['mask_path'],0)>0;mask|=current_graphics(im)[0]
  for role,col in [('board.top',(0,255,0)),('board.contact',(0,60,255))]:
   uv=q['roles'][role]['trace_uv']if q is not None else [[x,y]for x,y in enumerate(raw['roles'][role]['full_native_y'])if x%4==0 and 0<=y<1080 and not mask[int(round(y)),x]and not raw['roles'][role]['material_band']['full_material_occlusion'][x]]
   for v in uv:
    x,y=np.rint(v).astype(int)
    if 0<=x<1920 and 0<=y<1080:cv2.circle(out,(x,y),2,col,-1)
  y0=max(0,int(min(np.percentile(rr['full_native_y'],5)for rr in raw['roles'].values()))-25);y1=min(1080,int(max(np.percentile(rr['full_native_y'],95)for rr in raw['roles'].values()))+30);crop=out[y0:y1];tile=cv2.resize(crop,(960,320));cv2.putText(tile,str(p)+' '+('qualified'if q else'anchored'),(10,22),0,.6,(0,255,255),2);tiles.append(tile)
while len(tiles)%2:tiles.append(np.zeros((320,960,3),'u1'))
cv2.imwrite(str(O/name),np.vstack([np.hstack(tiles[i:i+2])for i in range(0,len(tiles),2)]))
