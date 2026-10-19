from calibration.program_tracking.extract_first_shot_ice import *
p=DATA/'repair_20261014';out=p/'circle_diagnostics/A3_identity_review';out.mkdir(exist_ok=True);items=[]
for i in range(201,1249):
 a=read(p/f'ice_candidates/{i:06d}.json');f=next((f for f in a['features'] if f['target_id']=='ice.A3:outer'),None)
 if not f:continue
 raw=cv2.imread(str(DATA/f'frames/{i:06d}.png'));panels=[];boxes=[]
 for j,seg in enumerate(f['segments']):
  uv=np.array(seg['uv']);lo=np.maximum([0,0],uv.min(0)-[70,50]).astype(int);hi=np.minimum([1919,1079],uv.max(0)+[70,50]).astype(int);crop=raw[lo[1]:hi[1]+1,lo[0]:hi[0]+1].copy();mark=crop.copy()
  for v in uv:cv2.circle(mark,tuple(np.rint(v-lo).astype(int)),3,(0,0,255),-1)
  panel=np.vstack([crop,mark]);cv2.putText(panel,str(lo.tolist())+'-'+str(hi.tolist()),(5,16),0,.35,(0,0,255),1);panels.append(panel);boxes.append([lo.tolist(),hi.tolist()])
 height=max(im.shape[0] for im in panels);canvas=np.zeros((height+25,sum(im.shape[1] for im in panels),3),np.uint8);x=0
 for im in panels:canvas[25:25+im.shape[0],x:x+im.shape[1]]=im;x+=im.shape[1]
 cv2.putText(canvas,'A3 source '+str(i),(5,18),0,.5,(255,255,255),1);file=out/f'{i:06d}.png';cv2.imwrite(str(file),canvas);items.append(dict(frame=i,path=str(file),source_boxes=boxes,candidate_sha256=sha(p/f'ice_candidates/{i:06d}.json')))
for start in range(0,len(items),4):
 page=np.zeros((2*350,2*640,3),np.uint8)
 for slot,item in enumerate(items[start:start+4]):
  image=cv2.imread(item['path']);scale=min(640/image.shape[1],340/image.shape[0]);image=cv2.resize(image,None,fx=scale,fy=scale);y,x=divmod(slot,2);page[y*350:y*350+image.shape[0],x*640:x*640+image.shape[1]]=image
 cv2.imwrite(str(out/f'page_{start//4+1:02d}.jpg'),page,[cv2.IMWRITE_JPEG_QUALITY,94])
write(out/'manifest.json',dict(items=items,note='Raw current source top; red representatives bottom; no inference geometry drawn. Current candidate identity still needs visual source confirmation.'))
print(len(items),str(out))
