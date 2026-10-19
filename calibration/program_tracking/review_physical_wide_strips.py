"""Current-source strips for sequential inspection of measured wide paint."""
from __future__ import annotations
import cv2
import numpy as np
from .extract_first_shot_ice import read,write,sha,DATA

RD=DATA/'repair_20261014'

def render(first=201,last=728):
 out=RD/'wide_line_diagnostics/physical_front_strips';out.mkdir(parents=True,exist_ok=True)
 cells=[];records=[]
 for i in range(first,last+1):
  path=RD/f'physical_wide_proposals/{i:06d}.json';packet=read(path)
  if not packet['features']:continue
  raw=cv2.imread(packet['source']['image_path'])
  for f in packet['features']:
   points=np.vstack([s['uv']for s in f['segments']]);center=points.mean(0)
   _,_,vt=np.linalg.svd(points-center);t=vt[0]
   if t[0]<0:t=-t
   n=np.array([-t[1],t[0]])
   along=(points-center)@t;lo,hi=along.min()-25,along.max()+25
   rows=f['source_line_fit']['direct_readouts']
   sides=np.vstack([r['source_evidence']['actual_source_sides_uv']for r in rows])
   margin=max(45.,min(95.,float(np.max(abs((sides-center)@n)))+18))
   length=int(np.ceil(hi-lo));height=int(np.ceil(margin*2))
   u=np.linspace(lo,hi,length);v=np.linspace(-margin,margin,height)
   grid=center+u[None,:,None]*t+v[:,None,None]*n
   strip=cv2.remap(raw,grid[...,0].astype(np.float32),grid[...,1].astype(np.float32),cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=(45,45,45))
   marked=strip.copy()
   def xy(uv):
    p=np.asarray(uv)-center
    return np.rint(np.c_[((p@t)-lo)*(length-1)/(hi-lo),((p@n)+margin)*(height-1)/(2*margin)]).astype(int)
   for row in rows:
    ends=xy(row['source_evidence']['actual_source_sides_uv'])
    for p in ends:cv2.circle(marked,tuple(p),1,(0,220,220),-1)
    cv2.circle(marked,tuple(xy([row['uv']])[0]),1,(0,0,255),-1)
   for seg in f['segments']:
    uv=xy(seg['uv']);cv2.polylines(marked,[uv],False,(255,0,255),1,cv2.LINE_AA)
    for p in uv:cv2.circle(marked,tuple(p),2,(255,0,255),-1)
   cell=np.full((246,960,3),30,np.uint8)
   cv2.putText(cell,f'{i}  {f["target_id"]}  {f["point_count"]} rep / {len(rows)} source',(8,19),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1,cv2.LINE_AA)
   for pic,y in ((strip,26),(marked,136)):
    ratio=min(944/pic.shape[1],104/pic.shape[0]);small=cv2.resize(pic,None,fx=ratio,fy=ratio)
    cell[y:y+small.shape[0],8:8+small.shape[1]]=small
   cells.append(cell);records.append(dict(index=i,target_id=f['target_id'],proposal_sha256=sha(path)))
 pages=[]
 for pos in range(0,len(cells),12):
  selected=cells[pos:pos+12];canvas=np.full((246*6,1920,3),20,np.uint8)
  for k,c in enumerate(selected):canvas[(k//2)*246:(k//2+1)*246,(k%2)*960:(k%2+1)*960]=c
  dest=out/f'page_{pos//12+1:02d}.jpg';cv2.imwrite(str(dest),canvas,[cv2.IMWRITE_JPEG_QUALITY,96])
  pages.append(dict(path=str(dest),records=records[pos:pos+12]))
 write(out/'render_manifest.json',dict(pages=pages,rendering_only_not_visual_acceptance=True))
 print(len(pages),len(records),flush=True)

if __name__=='__main__':render()
