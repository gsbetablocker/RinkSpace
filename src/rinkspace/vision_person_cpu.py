"""CPU reuse of the existing local YOLO person model for evaluation masks.
Boxes are conservative source occlusion proposals, never physical fit points.
"""
from pathlib import Path
import hashlib,os,sys
import numpy as np
class SourcePersonBoxesCPU:
 def __init__(self,weights_path,*,dependency_paths=(),runtime_directory):
  import torch
  self.weights=Path(weights_path).resolve()
  if not self.weights.is_file():raise FileNotFoundError(self.weights)
  runtime=Path(runtime_directory).resolve();runtime.mkdir(parents=True,exist_ok=True)
  os.environ['YOLO_CONFIG_DIR']=str(runtime);os.environ['MPLCONFIGDIR']=str(runtime/'mpl');os.environ['YOLO_AUTOINSTALL']='false'
  for p in dependency_paths:
   p=Path(p).resolve()
   if not p.is_dir():raise FileNotFoundError(p)
   if str(p)not in sys.path:sys.path.append(str(p))
  from ultralytics import YOLO
  self.model=YOLO(str(self.weights));self.weights_sha256=hashlib.sha256(self.weights.read_bytes()).hexdigest();self.device='cpu'
 def read(self,rgb,binding):
  image=np.asarray(rgb)
  if image.shape!=(1080,1920,3)or image.dtype!=np.uint8:raise ValueError('Native uint8 source RGB required')
  expected=binding['decoded_native_rgb_sha256']
  if hashlib.sha256(image.tobytes()).hexdigest()!=expected:raise ValueError('Actual person-mask RGB differs')
  pred=self.model.predict(source=np.ascontiguousarray(image[:,:,::-1]),device='cpu',verbose=False)[0];rows=[]
  for box,conf,cls in zip(pred.boxes.xyxy.cpu().numpy(),pred.boxes.conf.cpu().numpy(),pred.boxes.cls.cpu().numpy()):
   if int(cls)==0:rows.append(dict(source_binding=dict(binding),box_xyxy=box.tolist(),provenance=dict(kind='same_local_pretrained_person_model_CPU_box_prediction',model_sha256=self.weights_sha256,confidence=float(conf),body_pose_used=False,is_pixel_ground_truth=False,evaluation_only=True)))
  return rows
