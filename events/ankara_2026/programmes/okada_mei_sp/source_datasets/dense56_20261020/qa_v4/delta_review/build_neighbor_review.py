from pathlib import Path
import sys,json,gzip,hashlib,cv2,numpy as np,av
P=Path(r'D:/OneDrive/codex_projs/RinkSpace');sys.path.insert(0,str(P/'calibration/rinkfit/src'))
from rinkfit_joint.dense_source_review import COLORS
O=Path(r'D:/CodexData/RinkSpace_Data/runs/okada_rec_1019/recover_v2/dense56_v1');Q=O/'qa_independent/v4_delta_native';c=json.load(open(O/'parts_v4/s05/video_config.json',encoding='utf8'))
with gzip.open(c['selected_frame_file']['file'],'rt',encoding='utf8')as f: rows={r['native_pts']:r for r in map(json.loads,f)}
with gzip.open(O/'leg_fix_v1/s05_index.jsonl.gz','rt',encoding='utf8')as f: masks={r['native_pts']:r for r in map(json.loads,f)}
pts=[1069824,1078528];src=av.open(c['source']['file']);st=src.streams.video[0];src.seek(pts[0],stream=st,backward=True);tiles=[];recs=[]
for fr in src.decode(st):
 p=int(fr.pts)
 if p>pts[-1]:break
 if p not in pts:continue
 raw=fr.to_ndarray(format='bgr24');r=rows[p];assert hashlib.sha256(raw.tobytes()).hexdigest()==r['decoded_native_BGR_sha256']
 mr=masks[p];bgr=hashlib.sha256(raw.tobytes()).hexdigest();rgb=hashlib.sha256(raw[:,:,::-1].copy().tobytes()).hexdigest();assert bgr==mr['source_BGR_sha256'] and rgb==mr['source_RGB_sha256']
 im=raw.copy()
 for point in r['points']:cv2.circle(im,tuple(np.rint(point['uv']).astype(int)),2,COLORS[point['display_class']],1,cv2.LINE_AA)
 # 1:1 native crop retaining all skater and both supporting board/ice regions.
 crop=[450,160,1450,900];x0,y0,x1,y1=crop;pair=np.hstack([raw[y0:y1,x0:x1],im[y0:y1,x0:x1]]);tile=np.zeros((770,2000,3),'u1');tile[:740]=pair;cv2.putText(tile,f'PTS {p}: source left | exact v4 right; crop {crop}',(8,760),0,.65,(255,255,255),1,cv2.LINE_AA)
 path=Q/f'neighbor_{p}.jpg';cv2.imwrite(str(path),tile,[cv2.IMWRITE_JPEG_QUALITY,98]);recs.append(dict(native_pts=p,file=str(path),source_RGB_sha256=rgb,source_BGR_sha256=bgr,crop_xyxy=crop,source_actor_index_RGB_BGR_verified=True))
assert len(recs)==2
json.dump(dict(frames=recs,full_native_scale=True),open(Q/'neighbor_index.json','w',encoding='utf8'),indent=2);print(json.dumps(dict(frames=len(recs))))
