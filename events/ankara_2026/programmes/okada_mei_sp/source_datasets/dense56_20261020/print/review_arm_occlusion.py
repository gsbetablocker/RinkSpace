import av,cv2,json,gzip,base64,numpy as np,hashlib
from pathlib import Path
O=Path(r'D:/CodexData/RinkSpace_Data/runs/okada_rec_1019/recover_v2/dense56_print_v1');R=O.parent;out=O/'arm_occlusion_review';out.mkdir(exist_ok=True);V=json.loads(Path(r'D:/CodexData/RinkSpace_Data/runs/okada_joint9_1018a/programme_scope.json').read_text(encoding='utf8'))['source']['path'];tiles=[];records=[]
with av.open(V)as c:
 st=c.streams.video[0];c.seek(1186816,stream=st,backward=True)
 for frame in c.decode(st):
  pts=int(frame.pts)
  if pts<1186816:continue
  if pts>1189888:break
  rgb=frame.to_ndarray(format='rgb24');bgr=rgb[:,:,::-1].copy();j=json.loads((O/'final_frames_v4_final'/f'{pts}.json').read_text(encoding='utf8'));rows=[]
  for m in j['source_members']:
   with open(m['file'],'rb')as f:f.seek(m['offset_bytes']);bb=f.read(m['length_bytes'])
   assert hashlib.sha256(bb).hexdigest()==m['sha256'];rows.extend(json.loads(s)for s in gzip.decompress(bb).decode('utf8').splitlines())
  el=np.frombuffer(base64.b64decode(j['eligible_uint8_base64']),dtype='u1')>0;code=np.frombuffer(base64.b64decode(j['status_uint8_base64']),dtype='u1');xy=np.asarray([r['uv']for r in rows]);roi=(xy[:,0]>520)&(xy[:,0]<800)&(xy[:,1]>350)&(xy[:,1]<440);idx=np.flatnonzero(roi&el);show=bgr.copy();mask=cv2.imread(str(R/'dense56_masks_v1/masks'/f'{pts}.png'),0);guard=cv2.dilate((mask>0).astype('u1'),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(17,17)));contours,_=cv2.findContours(guard,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(show,contours,-1,(0,0,255),1)
  for i in idx:cv2.circle(show,tuple(np.rint(xy[i]).astype(int)),2,(0,255,255),-1)
  pair=cv2.hconcat([bgr[300:480,480:840],show[300:480,480:840]]);cv2.putText(pair,f'{pts} prior eligible in ROI={len(idx)}',(3,17),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,0,255),1);tiles.append(pair);records.append(dict(native_pts=pts,RGB_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),prior_v4_eligible_ROI_rows=len(idx),rows=[dict(row_index=int(i),point_id=rows[i]['point_id'],uv=rows[i]['uv'],status=int(code[i]),normal=rows[i].get('pixel_support',{}).get('normal'),source_kind=rows[i]['source_kind'])for i in idx]));cv2.imwrite(str(out/f'{pts}_source_and_prior_eligible.png'),pair)
cv2.imwrite(str(out/'near_1188352_source_and_prior.jpg'),cv2.vconcat(tiles),[cv2.IMWRITE_JPEG_QUALITY,93]);(out/'near_1188352_prior_eligibility.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps([dict(pts=j['native_pts'],eligible_ROI=j['prior_v4_eligible_ROI_rows'])for j in records]))
