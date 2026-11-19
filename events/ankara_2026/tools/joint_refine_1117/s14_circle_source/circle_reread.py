import csv,os,sys
from pathlib import Path
import cv2 as cv,numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
sys.path.insert(0,'events/ankara_2026/tools/joint_refine_1117')
from review import source_frames,ji
from shared import PREPARED,read,arrays
work=Path(os.environ['RS_CIRCLE_AUDIT']); s=read(PREPARED/'setup.json')['shots']['jin_s14']; m=arrays(work/'measure.npz')
# Pixel intervals hand-selected on the native, unpainted C0 image, lower sloping branch,
# from the visibly bending left tip, and away from C/B1 and skater. Only profiles in
# these intervals are passed to the edge detector. Projection is used for scoring later.
patches={
 56: [(95,936,960),(150,943,968),(225,951,976)],
 70: [(356,907,935),(387,912,940),(417,916,945)],
 98: [(890,854,885),(918,857,894),(950,867,900)],
 112: [(1121,845,882),(1152,853,886),(1184,857,888)],
 126: [(1223,820,850),(1249,826,853),(1270,829,856)],
 168: [(684,702,733),(713,709,739),(747,714,744)],
 182: [(603,676,703),(625,679,708),(654,684,712)],
 210: [(900,648,674),(922,652,680),(968,662,689)],
 238: [(1356,619,644),(1389,624,649),(1421,629,653)],
 252: [(1460,601,628),(1504,608,632),(1545,610,636)],
}
status=np.zeros(m['projection'].shape[-2],int); results=[]; sheets=[]
indices=m['idx'][np.isin(m['idx'],list(patches))]; pts=m['pts'][np.isin(m['idx'],list(patches))];
for f,(pt,img) in zip(indices,source_frames(s['source'],pts),strict=True):
 index=np.searchsorted(m['pts'],pt); select=patches[f]; xl=select[0][0]-48; xr=select[-1][0]+48; yl=min(v[1] for v in select)-18; yr=max(v[2] for v in select)+12; a=img[yl:yr,xl:xr]; panel=a.copy();
 for x,lo,hi in select:
  # Average five columns, sample source brightness along y. The paired gradient
  # picks the two paint edges, and the reported read is their 2D midpoint.
  rgb=img[lo:hi+1,x-2:x+3].astype(float).mean(1)[:,::-1]; profile=gaussian_filter1d(rgb.mean(1),1.1); grad=np.gradient(profile);
  lcan=find_peaks(-grad)[0]; rcan=find_peaks(grad)[0]; best=None
  for l in lcan:
   for r in rcan:
    if not 3<=r-l<=20 or l<3 or r>=len(grad)-3:continue
    if -grad[l]<.4 or grad[r]<.4:continue
    left=rgb[l-3:l-1].mean(0);right=rgb[r+2:r+4].mean(0);mid=rgb[(3*l+r)//4:(l+3*r)//4+1].mean(0); depth=np.minimum(left-mid,right-mid).mean();
    strength=min(-grad[l],grad[r])+depth*.18-abs((left-right).mean())*.09
    if depth>1.25 and (best is None or best[0]<strength):best=(strength,l,r,depth,left,mid,right)
  if best is None:
   print('no clear paired edge',int(f),x,lo,hi); continue
  _,l,r,depth,left,mid,right=best
  def interp(y,sign):
   v=grad[y-1:y+2]*sign
   return y+.5*(v[0]-v[2])/max(1.e-8,v[0]+v[2]-2*v[1]) if False else y+np.clip(.5*(v[0]-v[2])/min(-1e-8,v[0]+v[2]-2*v[1]),-.5,.5)
  edge_lo=lo+interp(l,-1); edge_hi=lo+interp(r,1); u=np.array([x,(edge_lo+edge_hi)/2]); signed=[]; offsets=[]
  for p,out in zip(m['projection'][:,index],m['projection_outside'][:,index],strict=True):
   seg,_,_,delta,_=ji.nearest(p,status,u[None]); off=delta[0]; sign=np.sign(off @ (out-p)[seg[0]]); signed.append(float(sign*np.linalg.norm(off))); offsets.append(list(off));
  res=(int(f),int(pt),x,edge_lo,edge_hi,float(u[1]),*signed,depth,*left,*mid,*right); results.append(res);
  print(int(f),pt/12800,x,np.round(res[3:9],2),'rgb L/M/R',left.round(1),mid.round(1),right.round(1));
  pt1=(x-xl,round(edge_lo-yl)); pt2=(x-xl,round(edge_hi-yl));cv.line(panel,pt1,pt2,(16,170,4),1); cv.circle(panel,pt1,2,(20,160,5),1); cv.circle(panel,pt2,2,(20,160,5),1);cv.circle(panel,(x-xl,round(u[1]-yl)),1,(8,190,220),-1)
  cx=x-xl; cv.putText(panel,str(x),(cx-9,min(panel.shape[0]-3,round(u[1]-yl)+14)),cv.FONT_HERSHEY_SIMPLEX,.28,(0,0,0),1,cv.LINE_AA)
 gray=cv.cvtColor(a,cv.COLOR_BGR2GRAY); gscale=((gray.astype(float)-np.percentile(gray,10))*16+48).clip(0,255).astype(np.uint8); gscale=cv.cvtColor(gscale,cv.COLOR_GRAY2BGR);
 both=np.concatenate([cv.resize(im,(300,145)) for im in (a,gscale,panel)],axis=1); cv.putText(both,f'{pt/12800:.3f}s | PTS {pt} | source / contrast / 2 edges + midpoint',(8,17),cv.FONT_HERSHEY_SIMPLEX,.49,(20,20,20),1,cv.LINE_AA); sheets.append(both)
for j in range(0,len(sheets),5): cv.imwrite(str(work/f'material_edges_{j//5+1}.png'),np.concatenate(sheets[j:j+5]));
with (work/'paired_reread_candidates.csv').open('w',encoding='utf-8',newline='') as fo:
 wr=csv.writer(fo); wr.writerow(['s14_index','pts','source_x','edge1_y','edge2_y','midpoint_y','signed_previous_px','signed_reviewed_px','contrast_min_dn','left_R','left_G','left_B','mid_R','mid_G','mid_B','right_R','right_G','right_B']); wr.writerows(results)
