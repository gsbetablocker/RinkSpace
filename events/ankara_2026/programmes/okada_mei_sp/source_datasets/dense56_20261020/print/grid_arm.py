import av,cv2,numpy as np,json
from pathlib import Path
O=Path(r'D:/CodexData/RinkSpace_Data/runs/okada_rec_1019/recover_v2/dense56_print_v1');R=O.parent;V=json.loads(Path(r'D:/CodexData/RinkSpace_Data/runs/okada_joint9_1018a/programme_scope.json').read_text(encoding='utf8'))['source']['path']
for pts in(1187584,1187840,1188096,1188352,1188608,1188864,1189120):
 with av.open(V)as c:
  st=c.streams.video[0];c.seek(pts,stream=st,backward=True)
  for frame in c.decode(st):
   if frame.pts==pts:im=frame.to_ndarray(format='bgr24');break
 crop=im[200:500,450:900].copy()
 for x in range(460,901,20):cv2.line(crop,(x-450,0),(x-450,299),(90,90,90),1);cv2.putText(crop,str(x),(x-450,13),cv2.FONT_HERSHEY_SIMPLEX,.3,(0,0,255),1)
 for y in range(200,501,20):cv2.line(crop,(0,y-200),(449,y-200),(90,90,90),1);cv2.putText(crop,str(y),(2,y-200+11),cv2.FONT_HERSHEY_SIMPLEX,.3,(0,0,255),1)
 cv2.imwrite(str(O/'arm_occlusion_review'/f'{pts}_native_grid.png'),crop)
