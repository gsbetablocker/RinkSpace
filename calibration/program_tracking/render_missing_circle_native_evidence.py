from calibration.program_tracking.extract_first_shot_ice import *
p=DATA/'repair_20261014';out=p/'circle_diagnostics/missing_source_checks';out.mkdir(exist_ok=True)
cases=[(293,'ice.R3:outer'),(328,'ice.R4:outer'),(329,'ice.R4:outer'),(330,'ice.R4:outer'),(331,'ice.R4:outer'),(642,'ice.F4:outer'),(763,'ice.F4:outer'),(768,'ice.F4:outer'),(772,'ice.F4:outer'),(973,'ice.F4:outer'),(974,'ice.F4:outer'),(975,'ice.F4:outer'),(1054,'ice.F4:outer'),(1055,'ice.F4:outer'),(1056,'ice.F4:outer')]
for i,tid in cases:
 old=read(DATA/f'ice_candidates/{i:06d}.json');f=next((f for f in old['features'] if f['target_id']==tid),None)
 if f is None:continue
 uv=np.concatenate([np.array(s['uv']) for s in f['segments']]);lo=np.maximum([0,0],uv.min(0)-[50,45]).astype(int);hi=np.minimum([1919,1079],uv.max(0)+[50,45]).astype(int);raw=cv2.imread(str(DATA/f'frames/{i:06d}.png'));mark=raw.copy()
 for q in uv:cv2.circle(mark,tuple(np.rint(q).astype(int)),3,(0,0,255),-1)
 c1=raw[lo[1]:hi[1]+1,lo[0]:hi[0]+1];c2=mark[lo[1]:hi[1]+1,lo[0]:hi[0]+1];canvas=np.vstack([c1,c2]);cv2.putText(canvas,str(i)+' '+tid,(8,20),0,.5,(0,0,255),1);cv2.imwrite(str(out/f'{i:06d}_{tid.replace(":","_")}.png'),canvas)
print(out)
