from calibration.program_tracking.extract_first_shot_ice import *
root=DATA/'repair_20261014';folder=root/'circle_diagnostics';folder.mkdir(exist_ok=True)
canvas=np.zeros((2*340,4*480,3),np.uint8)
for slot,i in enumerate([230,237,240,242,243,244,247]):
 p=read(root/f'circle_candidates/{i:06d}.json');f=p['features_by_curve']['A2:outer'];b=cv2.imread(str(DATA/f'frames/{i:06d}.png'))
 uv=np.concatenate([s['uv'] for s in f['segments']]);lo=np.maximum([0,0],uv.min(0)-[80,80]).astype(int);hi=np.minimum([1919,1079],uv.max(0)+[80,80]).astype(int)
 for seg in f['segments']:
  for q in seg['uv']:cv2.circle(b,tuple(np.rint(q).astype(int)),3,(0,0,255),-1)
 crop=b[lo[1]:hi[1]+1,lo[0]:hi[0]+1];scale=min(480/crop.shape[1],310/crop.shape[0]);crop=cv2.resize(crop,None,fx=scale,fy=scale);y,x=divmod(slot,4);canvas[y*340:y*340+crop.shape[0],x*480:x*480+crop.shape[1]]=crop;cv2.putText(canvas,str(i),(x*480+12,y*340+330),0,.6,(255,255,255),1)
cv2.imwrite(str(folder/'A2_all_remaining_native_crops.jpg'),canvas)
