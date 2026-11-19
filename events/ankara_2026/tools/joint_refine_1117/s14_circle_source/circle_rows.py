import csv, os,sys
from pathlib import Path
import cv2 as cv,numpy as np
sys.path.insert(0,'events/ankara_2026/tools/joint_refine_1117')
from review import native_for,camera_state,source_frames,ji
from shared import PREPARED,read,arrays
work=Path(os.environ['RS_CIRCLE_AUDIT']); name='jin_s14'; s=read(PREPARED/'setup.json')['shots'][name]
a=arrays(PREPARED/(name+'.npz')); m=arrays(work/'measure.npz'); q=arrays(Path(s['source_packets']['8']['path'])); n=native_for('s14_boards_1118/reviewed','jin'); b=camera_state('s14_boards_1118/reviewed')
paths=['d78_material_145/path0','d78_material_142/path0','d78_material_143/path0']
frames=m['idx']; pts=m['pts']; rows=m['rows']; sr=m['source_rows']; probes=q['bracket'][sr]; unit=q['normal'][sr];
tab=ji.native_paths(n,paths,count=257); jobs=[(i,k,b[name+'_times'][i]) for i in frames for k in paths]; projs=ji.project_groups(n,b[name+'_rigs'],b[name+'_controls'],s['first'],s['dt'],jobs,{k:v[0] for k,v in tab.items()}); projarr=np.array([x[0] for x in projs]).reshape(len(frames),len(paths),257,2)
segclass=np.full(len(rows),'unknown',dtype='U20'); side=np.full((len(rows),3),np.nan); chroma=side.copy(); darkness=side.copy(); evidence=[]; panels=[]
for j,(pt,img) in enumerate(source_frames(s['source'],pts)):
 i=frames[j]; rr=np.flatnonzero(a['pts'][rows]==pt); uv=a['uv'][rows[rr]]; en=probes[rr]; un=unit[rr]
 def get(positions):
  x=positions[:,:,0].astype(np.float32); y=positions[:,:,1].astype(np.float32); return cv.remap(img,x,y,cv.INTER_LINEAR).astype(float).mean(1)[:,::-1]
 left=get(en[:,[0]]-un[:,None]*np.array([2.,3.,4.])[None,:,None]); right=get(en[:,[1]]+un[:,None]*np.array([2.,3.,4.])[None,:,None]); mid=get(en[:,[0]]+(en[:,[1]]-en[:,[0]])*np.array([.35,.5,.65])[None,:,None]);
 for ii,flank in enumerate([left,right,(left+right)/2]):
  dd=mid-flank; chroma[rr,ii]=dd[:,0]-dd[:,2]; darkness[rr,ii]=-dd.mean(1)
 for k,(poly,stat) in enumerate(projs[j*len(paths):(j+1)*len(paths)]):
  nn=ji.nearest(poly,stat,uv)
  if nn is None:continue
  _,_,norm,delta,_=nn;sgn=np.sign(delta[:,1]); side[rr,k]=sgn*np.linalg.norm(delta,axis=1)
 png=img.copy();
 for curve,color in [(m['projection'][1,j],(20,35,245)),(projarr[j,0],(230,80,220)),(projarr[j,1],(255,200,0))]:
  good=np.isfinite(curve).all(1)&((curve>-1200)&(curve<[3120,2280])).all(1); segments=np.split(np.flatnonzero(good),np.where(np.diff(np.flatnonzero(good))!=1)[0]+1)
  for ss in segments:
   if len(ss)>1:cv.polylines(png,[np.rint(curve[ss]).astype(np.int32)],False,color,1,cv.LINE_AA)
 for ri in rr:
  u=a['uv'][rows[ri]]; sf=abs(m['signed'][1,ri]); sc=abs(side[ri,0]); sb=abs(side[ri,1]);
  # The colors here are a visual inspection aid only.
  if sc<=8 and sf>9: col=(240,90,240);tag='C?'
  elif sb<=10 and sf>10: col=(255,220,0);tag='B1?'
  elif sf<=9 and min(sc,sb)>10: col=(0,250,240);tag='C0?'
  else:col=(0,255,0);tag='?'
  cv.circle(png,tuple(u.round().astype(int)),3,col,-1,cv.LINE_AA)
  if (ri-rr[0])%3==0:
   p=tuple(u.round().astype(int)+[-6,(-1)**(ri-rr[0])*11+8]); cv.putText(png,str(rows[ri]),p,cv.FONT_HERSHEY_SIMPLEX,.32,(25,25,25),2,cv.LINE_AA); cv.putText(png,str(rows[ri]),p,cv.FONT_HERSHEY_SIMPLEX,.32,col,1,cv.LINE_AA)
 cy=np.clip(round(np.median(uv[:,1])),70,1010); k=75; cy=np.clip(cy,k,1080-k);
 panel=np.concatenate([im[cy-k:cy+k] for im in (img,png)]); cv.putText(panel,f'{pt/12800:.3f}s PTS {pt} S14 #{i}, y={cy-k}-{cy+k-1}',(9,20),cv.FONT_HERSHEY_SIMPLEX,.6,(20,20,20),1,cv.LINE_AA); panels.append(panel)
 print('frame',i,pt/12800)
 for ri in rr:
  print(rows[ri],*a['uv'][rows[ri]].round(0).astype(int), 'w',round(float(np.linalg.norm(probes[ri,1]-probes[ri,0])),1),'c0 C B1',np.round([m['signed'][1,ri],*side[ri,:2]],1),'chroma',chroma[ri].round(1),'dark',darkness[ri,:2].round(1))
for j in range(0,len(panels),3): cv.imwrite(str(work/f'row_context_{j//3+1}.png'),np.concatenate(panels[j:j+3]));
np.savez_compressed(work/'material_properties.npz',rows=rows,distance_lines=side,proj_lines=projarr,chroma_delta=chroma,darkness_delta=darkness)
