from pathlib import Path
import sys,json,gzip,hashlib,cv2,numpy as np,av
P=Path(r'D:/OneDrive/codex_projs/RinkSpace');sys.path.insert(0,str(P/'calibration/rinkfit/src'))
from rinkfit_joint.dense_source_review import COLORS
O=Path(r'D:/CodexData/RinkSpace_Data/runs/okada_rec_1019/recover_v2/dense56_v1')
Q=O/'qa_independent/v4_delta_native'
config=json.load(open(O/'parts_v4/s05/video_config.json',encoding='utf8'))
with gzip.open(config['selected_frame_file']['file'],'rt',encoding='utf8')as f: rows={r['native_pts']:r for r in map(json.loads,f)}
with gzip.open(O/'leg_fix_v1/source_proof.jsonl.gz','rt',encoding='utf8')as f: proofs={r['native_pts']:r for r in map(json.loads,f)}
pts=list(range(1070080,1078272+1,256)); assert len(pts)==33
src=av.open(config['source']['file']);st=src.streams.video[0];src.seek(pts[0],stream=st,backward=True)
tiles=[];pages=[];recs=[];details=[]
for fr in src.decode(st):
 p=int(fr.pts)
 if p>pts[-1]:break
 if p not in pts:continue
 raw=fr.to_ndarray(format='bgr24');r=rows[p];pr=proofs[p]
 assert hashlib.sha256(raw.tobytes()).hexdigest()==r['decoded_native_BGR_sha256']==pr['source_BGR_sha256']
 assert hashlib.sha256(raw[:,:,::-1].copy().tobytes()).hexdigest()==pr['source_RGB_sha256']
 im=raw.copy()
 for point in r['points']:cv2.circle(im,tuple(np.rint(point['uv']).astype(int)),2,COLORS[point['display_class']],1,cv2.LINE_AA)
 crop=[450,160,1450,900];x0,y0,x1,y1=crop
 pair=np.hstack([cv2.resize(raw[y0:y1,x0:x1],(650,481)),cv2.resize(im[y0:y1,x0:x1],(650,481))]);tile=np.zeros((511,1300,3),'u1');tile[:481]=pair
 cv2.putText(tile,f'PTS {p} / source {p/12800:.3f}s / original left | v4 points right / crop {crop}',(8,503),0,.55,(255,255,255),1,cv2.LINE_AA)
 tiles.append(tile)
 recs.append(dict(native_pts=p,source_RGB_sha256=pr['source_RGB_sha256'],source_BGR_sha256=pr['source_BGR_sha256'],native_crop_xyxy=crop,view_scale=.65,selected_rows=len(r['points'])))
 if len(tiles)==3:
  page=Q/f'body_pair_{len(pages):02}.jpg';cv2.imwrite(str(page),np.vstack(tiles),[cv2.IMWRITE_JPEG_QUALITY,97]);pages.append(str(page));tiles=[]
 if p in {1071872,1074688,1074944,1076224,1076480,1076736}:
  x0,y0,x1,y1=pr['crop_xyxy'];cropr=raw[y0:y1,x0:x1];cropi=im[y0:y1,x0:x1];pair=np.hstack([cropr,cropi]);tile=np.zeros((pair.shape[0]+30,pair.shape[1],3),'u1');tile[:-30]=pair
  cv2.putText(tile,f'PTS {p} native crop {pr["crop_xyxy"]}: source | current v4',(5,tile.shape[0]-9),0,.52,(255,255,255),1,cv2.LINE_AA)
  path=Q/f'leg_native_{p}.png';cv2.imwrite(str(path),tile);details.append(dict(file=str(path),native_pts=p,crop_xyxy=pr['crop_xyxy']))
assert len(recs)==33 and not tiles
json.dump(dict(schema='source_QA_current_native_pixel_pairs',source=config['source'],selected_frame_ref=config['selected_frame_file'],manifest_sha256=hashlib.sha256((O/'parts_v4/s05/manifest.json').read_bytes()).hexdigest(),frames=recs,pages=pages,native_details=details,model_used=False,points_moved=False),open(Q/'index.json','w',encoding='utf8'),indent=2)
print(json.dumps(dict(frames=len(recs),pages=len(pages),native_details=len(details))))
