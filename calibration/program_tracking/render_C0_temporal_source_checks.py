"""Show the actual current C0 material reads and common-source motion checks."""
from calibration.program_tracking.extract_first_shot_ice import *
p=DATA/'repair_20261014';out=p/'circle_diagnostics/C0_motion_source';out.mkdir(exist_ok=True)
for i in (504,505,509,510,583,1164,1165,1166,1167):
 a=read(p/f'ice_candidates/{i:06d}.json');f=next(f for f in a['features'] if f['target_id']=='ice.C0:outer');raw=cv2.imread(str(DATA/f'frames/{i:06d}.png'))
 mark=raw.copy()
 for s in f['segments']:
  q=np.rint(s['uv']).astype(int);cv2.polylines(mark,[q],False,(0,255,255),1)
  for uv in q:cv2.circle(mark,tuple(uv),4,(0,0,255),-1)
 for row in f['source_circle_fit']['direct_readouts']:cv2.circle(mark,tuple(np.rint(row['uv']).astype(int)),2,(255,0,255),-1)
 canvas=np.vstack([raw[480:],mark[480:]]);cv2.putText(canvas,f'C0 {i}; top native / bottom sparse fitted red, raw magenta',(5,18),0,.65,(0,0,255),2)
 cv2.imwrite(str(out/f'{i:06d}.png'),canvas)
 print(i,[(s['uv'][0],s['uv'][-1]) for s in f['segments']],f['joint_group_navigation']['shared_source_affine_parameters'])
