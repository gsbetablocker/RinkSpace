import sys,json,gzip
from pathlib import Path
P=Path(r'D:/OneDrive/codex_projs/RinkSpace');sys.path.insert(0,str(P))
from calibration.program_tracking.runtime.bootstrap import add_geometry_dependencies
add_geometry_dependencies()
import cv2,numpy as np,av
R=Path(r'D:/CodexData/RinkSpace_Data/runs/okada_rec_1019/recover_v2');O=R/'dense56_footwear_v1/clean_parent52';rows={q['native_pts']:q for q in map(json.loads,gzip.open(O/'index_candidate.jsonl.gz','rt'))};poses={q['native_pts']:q for q in map(json.loads,gzip.open(O.parent/'pose_proposals.jsonl.gz','rt'))};source=json.load(open(R.parent.parent/'okada_joint9_1018a/programme_scope.json',encoding='utf8'))['source'];tiles=[];pages=[]
with av.open(source['path'])as c:
 s=c.streams.video[0];c.seek(min(rows),stream=s,backward=True)
 for frame in c.decode(s):
  p=int(frame.pts)
  if p>max(rows):break
  if p not in rows:continue
  bgr=frame.to_ndarray(format='bgr24');row=rows[p];mask=cv2.imread(row['current_SAM']['file'],0);overlay=bgr.copy();cs,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(overlay,cs,-1,(0,255,0),2);main=max(poses[p]['person_proposals'],key=lambda q:q['box_xyxy'][3]-q['box_xyxy'][1]);x0,y0,x1,y1=main['box_xyxy'];x0=max(0,int(x0)-100);x1=min(1920,int(x1)+100);y0=max(0,int(y0)-60);y1=min(1080,int(y1)+120);crop=np.hstack([bgr[y0:y1,x0:x1],overlay[y0:y1,x0:x1]]);scale=min(1024/crop.shape[1],440/crop.shape[0]);crop=cv2.resize(crop,None,fx=scale,fy=scale);tile=np.full((470,1040,3),35,'u1');tile[30:30+crop.shape[0],:crop.shape[1]]=crop;cv2.putText(tile,f'{p} RAW | CURRENT SAM only',(8,22),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1);tiles.append(tile)
  if len(tiles)==4:
   page=O/'QA'/f'fresh_{len(pages):02}.jpg';cv2.imwrite(str(page),np.vstack(tiles),[cv2.IMWRITE_JPEG_QUALITY,90]);pages.append(str(page));tiles=[]
print(len(pages))
