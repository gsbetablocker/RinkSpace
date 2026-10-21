from pathlib import Path
import sys,json,gzip,hashlib,cv2,numpy as np,av
P=Path(r'D:/OneDrive/codex_projs/RinkSpace');sys.path.insert(0,str(P/'calibration/rinkfit/src'))
from rinkfit_joint.dense_source_review import COLORS
O=Path(r'D:/CodexData/RinkSpace_Data/runs/okada_rec_1019/recover_v2/dense56_v1');Q=O/'qa_independent/v4_delta_native'
config=json.load(open(O/'parts_v4/s05/video_config.json',encoding='utf8'))
with gzip.open(config['selected_frame_file']['file'],'rt',encoding='utf8')as f: rows={r['native_pts']:r for r in map(json.loads,f)}
with gzip.open(O/'leg_fix_v1/source_proof.jsonl.gz','rt',encoding='utf8')as f: proofs={r['native_pts']:r for r in map(json.loads,f)}
pts=list(range(1070080,1078272+1,256));src=av.open(config['source']['file']);st=src.streams.video[0];src.seek(pts[0],stream=st,backward=True);tiles=[];pages=[];recs=[]
for fr in src.decode(st):
 p=int(fr.pts)
 if p>pts[-1]:break
 if p not in proofs:continue
 raw=fr.to_ndarray(format='bgr24');r=rows[p];pr=proofs[p];assert hashlib.sha256(raw.tobytes()).hexdigest()==r['decoded_native_BGR_sha256']==pr['source_BGR_sha256']
 im=raw.copy()
 for point in r['points']:cv2.circle(im,tuple(np.rint(point['uv']).astype(int)),2,COLORS[point['display_class']],1,cv2.LINE_AA)
 ref=pr.get('qualified_new_source_unknown_delta_ref');pixels=0
 if ref:
  data=Path(ref['file']).read_bytes();assert hashlib.sha256(data).hexdigest()==ref['sha256'];mask=cv2.imdecode(np.frombuffer(data,'u1'),0);pixels=int(np.count_nonzero(mask));cs,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(im,cs,-1,(0,0,255),1)
 x0,y0,x1,y1=pr['crop_xyxy'];pair=np.hstack([raw[y0:y1,x0:x1],im[y0:y1,x0:x1]]);tile=np.zeros((241,962,3),'u1');tile[:211]=pair
 cv2.putText(tile,f'PTS {p}: source | v4 points + delta contour red ({pixels}px)',(5,233),0,.5,(255,255,255),1,cv2.LINE_AA);tiles.append(tile);recs.append(dict(native_pts=p,crop_xyxy=pr['crop_xyxy'],new_unknown_pixels=pixels,delta_ref=ref))
 if len(tiles)==6:
  path=Q/f'leg_delta_pairs_{len(pages):02}.jpg';cv2.imwrite(str(path),np.vstack(tiles),[cv2.IMWRITE_JPEG_QUALITY,97]);pages.append(str(path));tiles=[]
if tiles:
 while len(tiles)<6:tiles.append(np.zeros((241,962,3),'u1'))
 path=Q/f'leg_delta_pairs_{len(pages):02}.jpg';cv2.imwrite(str(path),np.vstack(tiles),[cv2.IMWRITE_JPEG_QUALITY,97]);pages.append(str(path))
assert len(recs)==33
json.dump(dict(frames=recs,pages=pages,points_are_exact_current_v4=True,mask_contours_are_exclusion_evidence_not_truth=True),open(Q/'delta_index.json','w',encoding='utf8'),indent=2)
print(json.dumps(dict(frames=len(recs),pages=len(pages))))
