import csv, os, sys
from pathlib import Path
import cv2 as cv
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
sys.path.insert(0,'events/ankara_2026/tools/joint_refine_1117')
from review import camera_state, ji, native_for, source_frames
from shared import PREPARED, arrays, read

work=Path(os.environ['RS_CIRCLE_AUDIT']); name='jin_s14'; key='d78_material_144/path0'; s=read(PREPARED/'setup.json')['shots'][name]
original=np.genfromtxt(work/'paired_reread_candidates.csv',delimiter=',',names=True)
original=original[~((original['s14_index']==252)&(original['source_x']==1545))]
held=np.array([70,168,182]); source_idx=np.unique(original['s14_index'].astype(int))
# Neighbouring exposures stay outside the three read-only control clusters.
expand={f:([f] if f in held else list(np.array([-3,-1,0,1,3])+f)) for f in source_idx}
frames=np.array(sorted({i for x in expand.values() for i in x})); owner={f:ref for ref,numbers in expand.items() for f in numbers}; pts=None
natives=[]; projections=[]; outward=[]
for candidate in ['lane4_side_1118/verified','s14_boards_1118/reviewed']:
 n=native_for(candidate,'jin'); b=camera_state(candidate); g=ji.native_paths(n,[key],count=2049); xyz,t=g[key]
 center=np.array(n.paths[key]['parameters'][:3]); outer_xyz=center+1.002*(xyz-center)
 pts=b[name+'_pts'][frames]; keys=[key,'radial_outer']; jobs=[(f,k,b[name+'_times'][f]) for f in frames for k in keys]
 uv=ji.project_groups(n,b[name+'_rigs'],b[name+'_controls'],s['first'],s['dt'],jobs,{key:xyz,'radial_outer':outer_xyz}); assert all(np.all(status==0) for _,status in uv)
 data=np.array([v for v,_ in uv]); projections.append(data[::2]); outward.append(data[1::2]); natives.append(n)
proj=np.asarray(projections); outward=np.asarray(outward); status=np.zeros(len(t),int)

def edge_pair(raw_rgb,x):
    profile=gaussian_filter1d(raw_rgb.mean(1),1.5); grad=np.gradient(profile); best=None
    for l in find_peaks(-grad)[0]:
        for r in find_peaks(grad)[0]:
            if not 3.0<=x[r]-x[l]<=16.0 or l<5 or r>=len(x)-5:continue
            if -grad[l]<.23 or grad[r]<.23:continue
            left=raw_rgb[l-5:l-2].mean(0); right=raw_rgb[r+2:r+5].mean(0); inside=raw_rgb[(3*l+r)//4:(l+3*r)//4+1].mean(0)
            strength=np.minimum(left-inside,right-inside).mean()
            if strength<1.6:continue
            mid=(x[l]+x[r])/2
            if abs(mid)>8.:continue
            score=min(-grad[l],grad[r])+.12*strength-.04*abs((left-right).mean())-.015*abs(mid)
            if best is None or best[0]<score:
                def edge(k,sgn):
                    prev,mean,nxt=sgn*grad[k-1:k+2]; sub=np.clip(.5*(prev-nxt)/min(-1e-8,prev+nxt-2*mean),-.5,.5);return x[k]+sub*(x[1]-x[0]);
                best=(score,edge(l,-1),edge(r,1),strength,left,inside,right)
    return best

records=[]; image_tiles={0:[],1:[]}; calib=np.linspace(-16.,16.,65)
for j,(pt,img) in enumerate(source_frames(s['source'],pts)):
 f=frames[j]; ff=owner[f]; jr=np.searchsorted(frames,ff); ref=original[original['s14_index']==ff]; source_p=np.c_[ref['source_x'],ref['midpoint_y']]; seg, frac, _, _,_=ji.nearest(proj[1,jr],status,source_p); estimate=source_p+((1-frac[:,None])*(proj[1,j,seg]-proj[1,jr,seg])+frac[:,None]*(proj[1,j,seg+1]-proj[1,jr,seg+1]));
 local_slope=np.polyfit(source_p[:,0],source_p[:,1],1)[0] if len(source_p)>1 else ((proj[1,jr,seg+5,1]-proj[1,jr,seg-5,1])/(proj[1,jr,seg+5,0]-proj[1,jr,seg-5,0]))[0]
 tangent=np.array([1.,local_slope]); tangent/=np.linalg.norm(tangent); unit=np.array([-tangent[1],tangent[0]]); patch_rows=[]
 for p in estimate:
  gp=p+calib[:,None,None]*unit+tangent*np.array([-2.,-1.,0.,1.,2.])[None,:,None]
  rgb=cv.remap(img.astype(np.float32),gp[:,:,0].astype('float32'),gp[:,:,1].astype('float32'),cv.INTER_LINEAR).mean(1)[:,::-1]; pair=edge_pair(rgb,calib)
  if pair is None:continue
  _,d0,d1,k,flank0,inside,flank1=pair; e0=p+unit*d0; e1=p+unit*d1; u=(e0+e1)/2; signed=[]
  for state in range(2):
   pred=proj[state,j]; ss,_,_,delta,_=ji.nearest(pred,status,u[None]); direction=(outward[state,j]-pred)[ss[0]]; assert np.sign(direction@unit)>=0
   signed.append(float(np.sign(delta[0]@direction)*np.linalg.norm(delta[0])))
  patch_rows.append((u,e0,e1)); records.append((f,pt,ff,*u,*e0,*e1,*unit,*signed,k,*flank0,*inside,*flank1))
 xlo,ylo=np.floor(estimate.min(0)-np.array([36,24])).astype(int); xhi,yhi=np.ceil(estimate.max(0)+np.array([36,24])).astype(int)
 xlo=max(0,xlo); ylo=max(0,ylo); xhi=min(img.shape[1],xhi); yhi=min(img.shape[0],yhi); patch=img[ylo:yhi,xlo:xhi]; pic=np.ascontiguousarray(patch.copy())
 for u,e0,e1 in patch_rows:
  rel=lambda z: tuple(np.rint(z-[xlo,ylo]).astype(int));cv.line(pic,rel(e0),rel(e1),(0,220,15),1);cv.circle(pic,rel(u),1,(0,250,255),-1)
 g=cv.cvtColor(patch,cv.COLOR_BGR2GRAY); gg=np.clip((g.astype(np.float32)-np.percentile(g,10))*16+48,0,255).astype(np.uint8); gg=cv.cvtColor(gg,cv.COLOR_GRAY2BGR)
 show=np.concatenate([cv.resize(im,(260,99)) for im in [patch,gg,pic]]); cv.putText(show,f'{pt/12800:.3f}s PTS {pt} f{f} (ref {ff}) n={len(patch_rows)}',(4,17),cv.FONT_HERSHEY_SIMPLEX,.53,(5,5,5),1,cv.LINE_AA); image_tiles[0 if ff<=154 else 1].append(show)
for group,tiles in image_tiles.items():
 for st in range(0,len(tiles),8):cv.imwrite(str(work/f'neighbor_edges_{group}_{st//8}.png'),np.concatenate(tiles[st:st+8]));
header=['s14_index','pts','reference_frame','u','v','edge_inner_u','edge_inner_v','edge_outer_u','edge_outer_v','outward_u','outward_v','signed_previous_px','signed_reviewed_px','contrast_min_dn','flank_inner_R','flank_inner_G','flank_inner_B','mid_R','mid_G','mid_B','flank_outer_R','flank_outer_G','flank_outer_B']
with (work/'neighbor_reread_candidates.csv').open('w',encoding='utf-8',newline='') as fo:
 wr=csv.writer(fo);wr.writerow(header);wr.writerows(records)
print(len(records),'points',len(np.unique(np.asarray(records)[:,1])),'frames')
