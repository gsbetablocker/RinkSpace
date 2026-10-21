"""Same-state review drawing; source dots are immutable current UV, never predictions."""
from pathlib import Path
from collections import defaultdict
import gzip,hashlib,json
import cv2,numpy as np
from .io_utils import read_json,file_sha256
from .joint_metrics import PacketArrays
from rinkfit_start.native_camera import rotation


def material_color(label):
 return (40,225,240) if label.startswith(('Ad','print_component.')) else (80,230,80) if label.startswith('board') else (255,140,50)

class CurrentSourcePoints:
 def __init__(self,packet_file,model,scope,*,sealed_evaluation=None):
  reader=PacketArrays(packet_file);packet=reader.packet
  self.packet_ref=dict(path=str(Path(packet_file).resolve()),sha256=file_sha256(packet_file));self.by_pts=defaultdict(list);self.ledger={};self.evaluation_ref=None
  uv=reader.load('uv');pts=reader.load('native_pts');shot=reader.load('shot_index');mat=reader.load('material_index');active=(reader.load('admission')==1)&(reader.load('fit_role')==1)
  ranges={s['id']:(s['first_pts'],s['last_pts']) for s in scope['shots']};labels=[m['label'] for m in model['materials']]
  for i in np.flatnonzero(active):
   sid=packet['shot_ids'][int(shot[i])];stamp=int(pts[i]);lo,hi=ranges[sid]
   if not lo<=stamp<=hi or (stamp-lo)%256 or int(mat[i])>=len(labels):raise ValueError('Current source point shot/material/native PTS mismatch')
   if not np.isfinite(uv[i]).all() or (uv[i]<0).any() or (uv[i]>[1919,1079]).any():raise ValueError('Active current source point is outside the native sensor')
   self.by_pts[stamp].append((uv[i].copy(),material_color(labels[int(mat[i])]),int(i),'fit'))
  if sealed_evaluation is not None:
   evalfile=Path(sealed_evaluation);evaluation=read_json(evalfile)
   if evaluation.get('schema')!='rinkfit.joint9.frozen_native_board_validation.v1':raise ValueError('Only sealed current-exposure evaluation points are displayable')
   if evaluation['source']['sha256']!=scope['source']['sha256']:raise ValueError('Evaluation dots belong to another native source')
   # Exact file hash is supplied by caller separately; keep the original frozen model ref.
   self.evaluation_model=evaluation['model'];ref=evaluation['files']['source_rows.jsonl.gz']
   if file_sha256(ref['path'])!=ref['sha256']:raise ValueError('Sealed actual source points changed')
   with gzip.open(ref['path'],'rt',encoding='utf8') as f:
    for line in f:
     row=json.loads(line)
     if row.get('role')!='sealed_validation' or row.get('training_eligible') is not False or not row.get('current_RGB_measured') or row.get('current_uv')!=row.get('source_uv'):raise ValueError('Evaluation display must preserve its first actual current measurement')
     stamp=int(row['native_pts']);sid=row['shot_id'];lo,hi=ranges[sid]
     if not lo<=stamp<=hi or (stamp-lo)%256:raise ValueError('Sealed source point native clock mismatch')
     xy=np.asarray(row['current_uv'],float)
     if not np.isfinite(xy).all() or (xy<0).any() or (xy>[1919,1079]).any():raise ValueError('Sealed source point outside native sensor')
     self.by_pts[stamp].append((xy,material_color(row['material_label']),row['source_row_id'],'sealed_evaluation'))
   self.evaluation_ref=dict(path=str(evalfile.resolve()),sha256=file_sha256(evalfile))
  for stamp,values in self.by_pts.items():
   raw=np.asarray([v[0] for v in values],dtype='<f8');self.ledger[stamp]=dict(current_actual_source_points=len(values),fit_points=sum(v[3]=='fit' for v in values),sealed_evaluation_points=sum(v[3]=='sealed_evaluation' for v in values),current_native_UV_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),source_points_from_model=False,prior_exposure_points_drawn=False)
 def draw(self,image,native_pts):
  for xy,color,_,_ in self.by_pts.get(int(native_pts),[]):
   cv2.circle(image,tuple(np.rint(xy).astype(int)),2,color,1,cv2.LINE_AA)
  return self.ledger.get(int(native_pts),dict(current_actual_source_points=0,fit_points=0,sealed_evaluation_points=0,current_native_UV_sha256=hashlib.sha256(b'').hexdigest(),source_points_from_model=False,prior_exposure_points_drawn=False))

class SharedModelTopView:
 def __init__(self,world,records,scene,*,width=640,height=1080):
  self.width=width;self.height=height;self.scale=min((height-140)/1.06,(width-100)/(scene.width+.08));self.center=np.array([width/2,height/2+18]);self.base=np.full((height,width,3),(25,34,40),np.uint8)
  cv2.putText(self.base,'SAME PHYSICAL MODEL  L=1',(22,32),cv2.FONT_HERSHEY_SIMPLEX,.6,(235,240,240),1,cv2.LINE_AA)
  for r in records:
   if r.get('latent_source_track_point',False):continue
   q=world[r['begin']:r['end']];px=self.project(q);color=r['color']
   if r['material_index']<0:
    if np.isfinite(px).all():cv2.putText(self.base,r['label'],tuple(np.rint(px[0]).astype(int)),cv2.FONT_HERSHEY_SIMPLEX,.35,color,1,cv2.LINE_AA)
   elif len(q)>1 and np.isfinite(px).all():cv2.polylines(self.base,[np.rint(px).astype(np.int32).reshape(-1,1,2)],False,color,1,cv2.LINE_AA)
  cv2.putText(self.base,'left dots: current original native UV',(22,height-24),cv2.FONT_HERSHEY_SIMPLEX,.47,(220,230,230),1,cv2.LINE_AA)
 def project(self,xyz):
  q=np.asarray(xyz);return self.center+np.c_[q[...,1],-q[...,0]]*self.scale
 def frame(self,camera,local,sid):
  image=self.base.copy();a=np.asarray(camera.spline(np.array([local])))[0];R=rotation(camera.b,a[1])@camera.R0@rotation(camera.n,a[0]);z=a[2]-np.log(float(camera.optics['f_ref']));wb=R.T@camera.b;h=np.cross(camera.n,wb);h/=np.linalg.norm(h);D=camera.d.copy();D[2]+=camera.head.get('pupil_linear',0)*z+camera.head.get('pupil_quadratic',0)*z*z;C=camera.P+R.T@D+camera.a*h;direction=R.T@np.array([0.,0.,1.]);at=self.project(C[None])[0];tip=self.project((C+.08*direction)[None])[0]
  if np.isfinite([at,tip]).all():
   # Clip the actual camera ray to the map; never move its world location to fit.
   ok,a,b=cv2.clipLine((0,0,self.width,self.height),tuple(np.rint(at).astype(int)),tuple(np.rint(tip).astype(int)))
   if ok:cv2.arrowedLine(image,a,b,(0,180,255),2,cv2.LINE_AA,tipLength=.25)
  cv2.putText(image,sid+'  shared physical rig',(22,64),cv2.FONT_HERSHEY_SIMPLEX,.5,(235,240,240),1,cv2.LINE_AA);return image
