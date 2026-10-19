import json,numpy as np,cv2
from pathlib import Path
from calibration.program_tracking.extract_first_shot_ice import *
from calibration.program_tracking.native_ice_material_repair import sample_chroma,gaussian_filter1d
from rinkspace.vision_programme_ice_review import _actual_profile_candidates
ex=Extractor(DATA/'repair_20261014');out=ex.output_root/'source_profile_diagnostics';out.mkdir(exist_ok=True)
report={}
for i in [201,300,500,1219]:
 bgr=cv2.imread(str(DATA/f'frames/{i:06d}.png'));src=ex.rows[i]['source'];person=cv2.imread(src['mask_path'],0);bc,_=current_broadcast_exclusion(bgr);dom,_=current_ice_visible_domain(i);native=NativeProfiles(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB),src,(person>0)|bc|dom)
 fid='R4' if i==201 else 'C0';g=ex.house_reader.groups[fid];role='outer';th,q,n,inter=ex.house_reader.queries(g,i,role,np.zeros(6));off,prof,val=sample_chroma(native.frame.rgb,q,n,inter,'dark')
 rr=[];canvas=bgr.copy()
 for j in range(len(q)):
  oo=off[j,val[j]];pp=prof[j,val[j]]
  cs=_actual_profile_candidates(q[j],n[j],oo,pp,gaussian_filter1d(pp,1),operator='paint_center',signal='dark',transition=None,with_quantiles=True) if len(oo)>10 else []
  candidates=[]
  for pt,sides,bad,qu in cs:
   w=float(np.linalg.norm(sides[1]-sides[0]));sig=float(np.max(np.linalg.norm(qu[:,2]-qu[:,0],axis=1))/1.349);qo=(qu-q[j])@n[j];co=[abs(float(np.interp(v[2],oo,pp)-np.interp(v[0],oo,pp))) for v in qo]
   candidates.append(dict(uv=pt.tolist(),width=w,sigma=sig,contrast=co,bad=bad))
  rr.append(dict(query=q[j].tolist(),normal=n[j].tolist(),theta=th[j],candidates=candidates,off=oo.tolist(),profile=pp.tolist()))
  if (q[j]>=0).all() and (q[j]<[1920,1080]).all():cv2.circle(canvas,tuple(np.rint(q[j]).astype(int)),3,(0,255,255),-1);cv2.putText(canvas,str(j),tuple(np.rint(q[j]).astype(int)),0,.35,(0,0,0),1)
 report[str(i)]=rr;cv2.imwrite(str(out/f'queries_{i:06d}.png'),canvas)
(out/'queries_profiles.json').write_text(json.dumps(report),encoding='utf8')
print(out)


