import os,sys
from pathlib import Path
import cv2 as cv,numpy as np
sys.path.insert(0,'events/ankara_2026/tools/joint_refine_1117')
from review import native_for,camera_state,source_frames,ji
from shared import PREPARED,read,arrays
work=Path(os.environ['RS_CIRCLE_AUDIT']); name='jin_s14'; s=read(PREPARED/'setup.json')['shots'][name]
a=arrays(PREPARED/(name+'.npz')); m=arrays(work/'measure.npz'); n=native_for('s14_boards_1118/reviewed','jin'); b=camera_state('s14_boards_1118/reviewed')
ks={k:v['label'] for k,v in n.paths.items() if v['layer']=='ice'}
print('path samples',sorted(set(x.split(':')[0] for x in ks.values())))
frames=np.array([42,70,98,140,154,168,182,210,238,252,280,308,336,392,420]); pts=b[name+'_pts'][frames]
paths=list(ks); tab=ji.native_paths(n,paths,count=257); jobs=[(i,k,b[name+'_times'][i]) for i in frames for k in paths]; projs=ji.project_groups(n,b[name+'_rigs'],b[name+'_controls'],s['first'],s['dt'],jobs,{k:v[0] for k,v in tab.items()});
plots=[]; projarr=np.array([x[0] for x in projs]).reshape(len(frames),len(paths),257,2); np.savez_compressed(work/'contexts.npz',pts=pts,paths=paths,labels=list(ks.values()),projection=projarr)
for f,(pt,img) in enumerate(source_frames(s['source'],pts)):
 i=frames[f]; r=np.flatnonzero(a['pts'][m['rows']]==pt); near=np.argsort(np.abs(m['signed'][1][r]))[-3:]; print('frame',i,pt/12800, 'bad sample nearest projected entities');
 for ri in r[near]:
  u=a['uv'][m['rows'][ri]]; d=[(ji.nearest(uv,st,u[None])[0][0],ks[k]) for k,(uv,st) in zip(paths,projs[f*len(paths):(f+1)*len(paths)]) if (st==0).all()]; print(m['rows'][ri],u.round(1),[(round(x,2),k) for x,k in sorted(d)[:6]])
 group=np.array([k for k in paths if any(q in ks[k].split(':')[0] for q in ['C0','.C','.B1','.B2','.A1','.A2','.A3','.A4'])]); png=img.copy();
 colors={'C0':(0,40,230),'C':(220,70,200),'B1':(255,190,30),'B2':(255,190,30),'A1':(70,220,70),'A2':(70,220,70),'A3':(70,220,70),'A4':(70,220,70)}
 for k in group:
  name_=ks[k].split(':')[0].split('.')[-1]; color=colors.get(name_,(180,200,100)); u=projarr[f,paths.index(k)]; cv.polylines(png,[np.rint(u).astype(np.int32)],False,color,1,cv.LINE_AA)
  available=np.flatnonzero(((u>=25)&(u<[1895,1055])).all(1));
  if len(available):
   xy=u[available[len(available)//2]].round().astype(int);cv.putText(png,name_,tuple(xy),(cv.FONT_HERSHEY_SIMPLEX),.55,color,2,cv.LINE_AA)
 for ri in r:cv.circle(png,tuple(a['uv'][m['rows'][ri]].round().astype(int)),3,(0,255,255),-1,cv.LINE_AA)
 xys=a['uv'][m['rows'][r]]; cy=round(np.median(xys[:,1])); cy=np.clip(cy,110,970); result=np.concatenate([cv.resize(j[cy-110:cy+110],(1344,154),interpolation=cv.INTER_AREA) for j in (img,png)]);
 cv.putText(result,f'{pt/12800:.3f} s  PTS {pt}  source y={cy-110}..{cy+109}',(8,22),cv.FONT_HERSHEY_SIMPLEX,.65,(0,0,0),1,cv.LINE_AA); plots.append(result)
for j in range(0,len(plots),3): cv.imwrite(str(work/f'context_{j//3+1}.jpg'),np.concatenate(plots[j:j+3]),[cv.IMWRITE_JPEG_QUALITY,97]);
