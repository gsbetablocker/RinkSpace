from calibration.program_tracking.extract_first_shot_ice import *
ex=Extractor(DATA/'repair_20261014');out=ex.output_root/'source_profile_diagnostics'
for i in [300,500,678,1219]:
 b=cv2.imread(str(DATA/f'frames/{i:06d}.png'));s=ex.rows[i]['source'];m=cv2.imread(s['mask_path'],0);bc,_=current_broadcast_exclusion(b);dom,_=current_ice_visible_domain(i);n=NativeProfiles(cv2.cvtColor(b,cv2.COLOR_BGR2RGB),s,(m>0)|bc|dom)
 fid='F4' if i==678 else 'C0';g=ex.house_reader.groups[fid];result,diag=ex.house_reader.read(fid,i,n);cv=b.copy();info={}
 for role,rr in result.items():
  p=np.asarray(rr['curve']);vis=(p>=[0,0]).all(1)&(p<=[1919,1079]).all(1);p=p[vis]
  for p0 in p:cv2.circle(cv,tuple(np.rint(p0).astype(int)),1,(0,255,255),-1)
  for r in rr['rows']:cv2.circle(cv,tuple(np.rint(r['uv']).astype(int)),4,(0,0,255),-1)
  info[role]=rr['joint_group_navigation']
 cv2.imwrite(str(out/f'fit_{i:06d}.png'),cv)
 print(i,diag,info)
