import os,sys
from pathlib import Path
import numpy as np,cv2 as cv
sys.path.insert(0,'events/ankara_2026/tools/joint_refine_1117')
from review import native_for, camera_state, source_frames, ji
from shared import PREPARED,read,arrays
work=Path(os.environ['RS_CIRCLE_AUDIT']); name='jin_s14'
s=read(PREPARED/'setup.json')['shots'][name]; a=arrays(PREPARED/(name+'.npz')); e=arrays(PREPARED.parent/'point_review'/(name+'.npz')); q=arrays(Path(s['source_packets']['8']['path']))
rows=np.flatnonzero((a['family']=='C0') & np.isin(a['fold'],range(5,9))); sr=a['source_row'][rows]
assert np.all(q['uv'][sr]==a['uv'][rows]) and np.all(q['pts'][sr]==a['pts'][rows]); assert np.allclose(a['uv'][rows],q['bracket'][sr].mean(1),atol=1e-10,rtol=0)
key=a['path'][rows[0]]; indices=np.unique(a['idx'][rows]); pts=e['pts'][indices];
proj=[]; outlines=[]; side=[]; station=[]; nearest=[]; diffs=[]
for candidate in ['lane4_side_1118/verified','s14_boards_1118/reviewed']:
 n=native_for(candidate,'jin'); b=camera_state(candidate); tab=ji.native_paths(n,[key],count=2049); w,tt=tab[key]; center=np.array(n.paths[key]['parameters'][:3]); wout=center+1.002*(w-center)
 keys=[key,'radius_out']; jobs=[(i,k,b[name+'_times'][i]) for i in indices for k in keys]; uv_all=ji.project_groups(n,b[name+'_rigs'],b[name+'_controls'],s['first'],s['dt'],jobs,{key:w,'radius_out':wout}); assert np.all([np.all(x[1]==0) for x in uv_all]); uv_all=np.array([j[0] for j in uv_all]); p,outer=uv_all[::2],uv_all[1::2]; proj.append(p); outlines.append(outer)
 current=np.zeros(len(rows)); st=current.copy(); close=np.zeros((len(rows),2)); dif=np.zeros((len(rows),2));
 for i,curve,out in zip(indices,p,outer):
  ix=np.flatnonzero(a['idx'][rows]==i); u=a['uv'][rows[ix]]; base=curve[:-1]; vec=np.diff(curve,axis=0); v=u[:,None,:]-base; f=np.clip((v*vec).sum(-1)/(vec**2).sum(-1),0,1); inset=v-f[:,:,None]*vec; best=(inset**2).sum(-1).argmin(1); d=inset[np.arange(len(ix)),best]; signed=np.sign((d*(out-curve)[best]).sum(-1));
  current[ix]=signed*np.linalg.norm(d,axis=-1); st[ix]=tt[best]+f[np.arange(len(ix)),best]*(tt[1]-tt[0]); close[ix]=u-d; dif[ix]=d
 side.append(current); station.append(st); nearest.append(close); diffs.append(dif)
print('dense/stored distance difference p50,p90,max', [np.percentile(abs(np.abs(side[i])-e[k][rows]),[50,90,100]).round(5).tolist() for i,k in enumerate(['previous','error'])]); print('signed', [np.percentile(v,[0,10,25,50,75,90,100]).round(2).tolist() for v in side]);
width=np.linalg.norm(q['bracket'][sr,1]-q['bracket'][sr,0],axis=1); print('width',np.percentile(width,[0,10,50,90,100])); print('station difference', [np.percentile(np.abs((t-a['station'][rows]+.5)%1-.5),[50,90,100]).round(4).tolist() for t in station]);
print('frame idx, s, before P50, after P50, N, n above 15 abs');
for i,pt in zip(indices,pts):
 rr=(a['idx'][rows]==i); print(i,pt/12800, *((np.median(v[rr]).round(2)) for v in side),rr.sum(),(abs(side[1][rr])>15).sum())
selected=np.array([42,70,98,140,154,168,182,210,238,252,280,308,336,392,420]);
plot=[]; crop_samples=[]
for pt,img in source_frames(s['source'],pts[np.isin(indices,selected)]):
 j=np.searchsorted(pts,pt); i=indices[j]; group=np.flatnonzero(a['pts'][rows]==pt); rgb=img.copy();
 def draw_path(arr,c,w):
  seg=np.rint(arr*16).astype(np.int32); cv.polylines(rgb,[seg],False,c,w,cv.LINE_AA,shift=4)
 draw_path(proj[0][j],(210,130,25),2); draw_path(proj[1][j],(30,70,255),2)
 for ridx in group:
  xy=lambda uv: tuple(np.rint(uv*16).astype(int)); u=a['uv'][rows[ridx]]; es=q['bracket'][sr[ridx]]
  cv.line(rgb,xy(es[0]),xy(es[1]),(40,255,40),1,cv.LINE_AA,shift=4)
  cv.circle(rgb,xy(u),2*16,(30,255,255),-1,cv.LINE_AA,shift=4)
 horiz=np.concatenate([cv.resize(im,(640,360),interpolation=cv.INTER_AREA) for im in [img,rgb]],axis=1)
 cv.putText(horiz,f'{pt/12800:.3f} s  PTS {pt}  S14 f{i} | source / point yellow, edges green, before blue, now red',(12,25),cv.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1,cv.LINE_AA)
 plot.append(horiz)
 if i in [42,70,98,140,182,238,252,280,308,336,392,420]:
  near=group[np.argmax(abs(side[1][group]))]; u=a['uv'][rows[near]]; x,y=np.rint(u).astype(int); x=np.clip(x,112,1920-112); y=np.clip(y,84,1080-84)
  crop=lambda im: im[y-84:y+84,x-112:x+112]
  tile=np.concatenate([crop(img),crop(rgb)],axis=1)
  cv.putText(tile,f'{pt/12800:.3f} s #{rows[near]} {side[1][near]:+.1f}px',(6,17),cv.FONT_HERSHEY_SIMPLEX,.43,(0,0,0),1,cv.LINE_AA)
  crop_samples.append(tile)
for start in range(0,len(plot),3): cv.imwrite(str(work/f'overview_{start//3+1}.jpg'),np.concatenate(plot[start:start+3]),[cv.IMWRITE_JPEG_QUALITY,95]);
for start in range(0,len(crop_samples),6):
 tiles=crop_samples[start:start+6]; cv.imwrite(str(work/f'close_{start//6+1}.png'),np.concatenate([np.concatenate(tiles[j:j+2],axis=1) for j in range(0,len(tiles),2)]));
np.savez_compressed(work/'measure.npz',rows=rows,source_rows=sr,idx=indices,pts=pts,projection=proj,projection_outside=outlines,signed=side,station=station,nearest=nearest,delta=diffs)
