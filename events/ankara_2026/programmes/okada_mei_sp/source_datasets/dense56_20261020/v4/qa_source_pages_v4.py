"""Actual native source point review pages; no model or moved UV."""
from pathlib import Path
import sys,json,gzip,hashlib,cv2,numpy as np,av
P=Path(r'D:/OneDrive/codex_projs/RinkSpace');sys.path.insert(0,str(P/'calibration/rinkfit/src'))
from rinkfit_joint.dense_source_review import COLORS
O=Path(__file__).resolve().parent
config=json.load(open(sys.argv[1],encoding='utf8'));out=Path(sys.argv[2]);out.mkdir(parents=True,exist_ok=False)
rows={}
with gzip.open(config['selected_frame_file']['file'],'rt',encoding='utf8')as f:
 for line in f:
  r=json.loads(line);rows[r['native_pts']]=r
special={1036800,1038592,1057792,1068032,1071872,1078272,1093632,1103104,1342208,1044992,1096192,1135104,1153792,1161216,1170944,1333760,1334016,1334272,1339648,1346048,1351424,1351680,1351936,1414912,1188096,1188352,1235200,986624,987648,988672,989696,1368064,1135104,1161216,1170944,948992,1007616,1015552,1024768,1071616,1124864,1290240,1300992,1301248,1332736,1363456,1368576,1425408}
for s in config['shots']:special.update(range(s['first_pts'],s['last_pts']+1,25*256));special.add(s['last_pts'])
source=av.open(config['source']['file']);st=source.streams.video[0];source.seek(min(rows),stream=st,backward=True);tiles=[];pages=[];summ=[]
for fr in source.decode(st):
 p=int(fr.pts)
 if p>max(rows):break
 if p not in special or p not in rows:continue
 r=rows[p];raw=fr.to_ndarray(format='bgr24');assert hashlib.sha256(raw.tobytes()).hexdigest()==r['decoded_native_BGR_sha256'];im=raw.copy()
 for point in r['points']:cv2.circle(im,tuple(np.rint(point['uv']).astype(int)),2,COLORS[point['display_class']],1,cv2.LINE_AA)
 if p in {1036800,1038592,1057792,1068032,1071872,1078272,1093632,1103104,1342208,1044992,1096192,1135104,1153792,1161216,1170944,1333760,1334016,1334272,1339648,1346048,1351424,1351680,1351936,1363456,1188096,1188352,1235200,986624,987648,988672,989696,1368064,1135104,1161216,1170944,1124864,1414912,948992,1007616,1015552,1024768,1071616,1290240,1300992,1301248,1332736,1368576,1425408}:cv2.imwrite(str(out/f'{p}.jpg'),im,[cv2.IMWRITE_JPEG_QUALITY,97])
 tile=np.zeros((574,960,3),dtype='u1');tile[:540]=cv2.resize(im,(960,540));cv2.putText(tile,f'{r["shot_id"]}  PTS {p}   source {p/12800:.3f}s   {len(r["points"])} source rows',(12,564),0,.7,(255,255,255),1,cv2.LINE_AA);tiles.append(tile);summ.append(dict(native_pts=p,source_seconds=p/12800,rows=len(r['points']),source_BGR_sha256=r['decoded_native_BGR_sha256']))
 if len(tiles)==6:
  page=np.vstack([np.hstack(tiles[i:i+2])for i in range(0,6,2)]);q=out/f'page_{len(pages):02}.jpg';cv2.imwrite(str(q),page,[cv2.IMWRITE_JPEG_QUALITY,94]);pages.append(str(q));tiles=[]
if tiles:
 while len(tiles)<6:tiles.append(np.zeros((574,960,3),dtype='u1'))
 page=np.vstack([np.hstack(tiles[i:i+2])for i in range(0,6,2)]);q=out/f'page_{len(pages):02}.jpg';cv2.imwrite(str(q),page,[cv2.IMWRITE_JPEG_QUALITY,94]);pages.append(str(q))
source.close();json.dump(dict(actual_source_review_frames=summ,pages=pages,source_points_moved=False,model_used=False),open(out/'index.json','w',encoding='utf8'),indent=2);print(json.dumps(dict(frames=len(summ),pages=len(pages),directory=str(out))))
