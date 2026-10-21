import cv2,json,numpy as np
from pathlib import Path
O=Path(r'D:/CodexData/RinkSpace_Data/runs/okada_rec_1019/recover_v2/dense56_print_v1');j=json.loads((O/'body_direct_delta_v5/manifest.json').read_text(encoding='utf8'));tiles=[]
for q in j['changed']:
 pts=q['native_pts'];im=cv2.imread(str(O/f'{pts}_final_v5_QA.jpg'));im=im[im.shape[0]//2:];im=cv2.resize(im,(640,200));cv2.putText(im,str(pts),(4,18),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,255),1);tiles.append(im)
while len(tiles)%3:tiles.append(np.zeros_like(tiles[0]))
cv2.imwrite(str(O/'direct22_final_qualified_QA.jpg'),cv2.vconcat([cv2.hconcat(tiles[i:i+3])for i in range(0,len(tiles),3)]),[cv2.IMWRITE_JPEG_QUALITY,94])
