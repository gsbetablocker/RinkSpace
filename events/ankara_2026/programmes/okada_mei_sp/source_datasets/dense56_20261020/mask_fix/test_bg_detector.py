import os,sys,json,hashlib
for k in('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
P=Path(r'D:/OneDrive/codex_projs/RinkSpace');R=Path(r'D:/CodexData/RinkSpace_Data/runs/okada_rec_1019/recover_v2');O=R/'dense56_v1/shoe_fix';sys.path[:0]=[str(P),str(P/'src'),str(P/'calibration/rinkfit/src')]
from calibration.program_tracking.runtime.bootstrap import add_geometry_dependencies
add_geometry_dependencies()
import cv2,numpy as np,torch
from rinkspace.vision_instances import SourcePersonBoxes
cv2.setNumThreads(1);torch.set_num_threads(1)
yolo=SourcePersonBoxes(r'D:/OneDrive/codex_projs/FSpathing/models/yolo11n-pose.pt',dependency_paths=[r'D:/CodexData/SkatingProjects_Archive_20261019/FSpathing/.venv/Lib/site-packages'],runtime_directory=O/'bg_pose_runtime',device='cuda');bgr=cv2.imread(str(O/'background1235200.png'))
for name,im in[('full',bgr),('board_strip',bgr[:380])]:
 pp=yolo.model.predict(source=im,device=0,imgsz=1280,conf=.05,verbose=False,save=False)[0];out=[]
 for box,conf,cl in zip(pp.boxes.xyxy.cpu().numpy(),pp.boxes.conf.cpu().numpy(),pp.boxes.cls.cpu().numpy()):
  if int(cl)==0:out.append(dict(box_xyxy=box.tolist(),confidence=float(conf)))
 print(name,json.dumps(out),flush=True)
