from calibration.program_tracking.extract_first_shot_ice import *
p=DATA/'repair_20261014';bad=read(p/'circle_diagnostics/circle_material_audit.json')['issues'];mx=0;out=[]
for r in bad:
 f=next(f for f in read(p/f"circle_candidates/{r['frame']:06d}.json")['features_by_curve'].values() if f['target_id']==r['target_id']);g=f['geometric_interpretation'];e=(g['center_uv'],g['axes_px'],g['angle_degrees']);raw=np.array([q['uv'] for q in f['source_circle_fit']['direct_readouts']],np.float32);near=cv2.getClosestEllipsePoints(e,raw).reshape(-1,2);err=np.linalg.norm(near-raw,axis=1);mx=max(mx,float(err.max()));out.append([r['frame'],r['error'],float(err.max())])
print(mx,out[:20])
