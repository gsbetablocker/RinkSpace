"""Convert a documented navigation pose into current shared optical coordinates.

Targets here are old navigation only. They never enter a source likelihood.
Only official native PTS routes the proposed shot control patch; source-native
re-capture and fitting must accept/reject it before any actual delivery.
"""
from pathlib import Path
import argparse,copy,hashlib,json
import cv2,numpy as np
from scipy.optimize import least_squares
from .camera_graph import CameraGraph,SHOT_TO_RIG,REFERENCE_SHOTS
from .joint_model import graph_from_joint,as_event_view
from .initialize_cameras import spline_seed,angles_for
from .role_guard import FrameRoleGuard
from .io_utils import read_json,write_json,file_sha256
from rinkfit_start.native_camera import NativeCamera,rotation
from rinkfit_start.native_scene import NativeScene

def old_project(c,X):
 R=cv2.Rodrigues(c[3:6])[0];K=np.array([[np.exp(c[6]),0,c[8]],[0,np.exp(c[7]),c[9]],[0,0,1.]])
 uv=cv2.projectPoints(np.asarray(X,float),c[3:6],-R@c[:3],K,c[11:25])[0].reshape(-1,2);uv[:,0]+=c[10]*(uv[:,1]-c[9])/K[1,1];depth=(X-c[:3])@R[2];return uv,depth

def current_at(q,rig,X):
 head=copy.deepcopy(rig['shared_head']);head['R0']=(rotation(head['b'],q[1])@np.asarray(head['R0'])@rotation(head['n'],q[0])).tolist()
 source=dict(shared_head=head,optics=rig['optics'],rolling_shutter={'signed_readout_seconds':0.},camera_controls=dict(basis=dict(first_control_time=-2.,dt=1.,n_controls=5),controls=[[0.,0.,float(q[2])]]*5))
 return NativeCamera(source).at(X,0.)

def convert(project,run,model_file,out,shot_numbers=(4,7,9)):
 project=Path(project);run=Path(run);out=Path(out)
 if out.exists():raise ValueError('New immutable optical conversion generation required')
 model=read_json(model_file);graph=graph_from_joint(model);cg=CameraGraph(graph);scene=NativeScene(as_event_view(model,'s01'));guard=FrameRoleGuard(run/'roles/frame_roles.json');plans={s['shot_id']:s for s in read_json(run/'control_plan.json')['shots']};patch={};proof=[]
 board=scene.board(np.linspace(0,scene.perimeter,512,endpoint=False),scene.height*.5)
 for number in shot_numbers:
  sid=f's{number:02d}';rig=graph['rigs'][SHOT_TO_RIG[sid]];navdir=Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/programmes/okada_mei_sp/results')/f'shot{number}_review_20261014';navpath=(navdir/'r2' if number==2 else navdir)/'navigation_mapping.json';nav=read_json(navpath);ref=nav['model_seed']['inherited_camera_archive'];archive_path=Path(ref['path'])
  if file_sha256(archive_path)!=ref['sha256']:raise ValueError('Navigation archive hash mismatch')
  with np.load(archive_path,allow_pickle=False) as z:cams=np.asarray([z['camera25'][f['inherited_camera_archive_row_0based']] for f in nav['frames']])
  scene_ref=nav['model_seed']['scene'];actual_scene_sha=file_sha256(scene_ref['path']);normalization=dict(navigation_scene=scene_ref,navigation_scene_actual_sha256=actual_scene_sha,uses_scene_geometry_or_relations=False)
  if actual_scene_sha==scene_ref['sha256']:source_scene=read_json(scene_ref['path'])
  else:
   # Some old display scenes were later augmented in place. Do not silently
   # trust those bytes: exact accepted q and camera archive identify the frozen
   # parent scene from which only the coordinate origin/scale is consumed.
   reference=read_json(run/'reference_8a28.json');exact=reference['scene']
   if nav['model_seed']['q_sha256']!=reference['q_sha256'] or ref['sha256']!=reference['camera_archive_sha256'] or file_sha256(exact['path'])!=exact['sha256']:raise ValueError('Navigation normalization has no exact q/scene/camera provenance')
   source_scene=read_json(exact['path']);normalization.update(exact_q=reference['q_sha256'],frozen_scene=exact,mutated_display_scene_not_consumed=True)
  bounds=np.asarray(source_scene['parameters']['bounds']);scale=float(bounds[2]-bounds[0]);origin=np.r_[(bounds[:2]+bounds[2:])/2,0.];pt=np.asarray([f['official_pts'] for f in nav['frames']],np.int64);guard.assert_training_pixels(pt.tolist(),shot_id=sid);times=cg.local_seconds(sid,pt);current=cg.native_camera(sid).spline(times);start=current.copy();start[:,:2]+=np.asarray(graph['shots'][sid]['reference_angles']);values=[];anchors=[]
  for i,(c,f) in enumerate(zip(cams,nav['frames'])):
   if hashlib.sha256(np.ascontiguousarray(c,dtype='<f8').tobytes()).hexdigest()!=f['camera25_float64_sha256']:raise ValueError('Exact source navigation row mismatch')
   R=cv2.Rodrigues(c[3:6])[0];K=np.array([[np.exp(c[6]),0,c[8]],[0,np.exp(c[7]),c[9]],[0,0,1.]])
   pixels=np.asarray([(u,v) for v in np.linspace(180,990,6) for u in np.linspace(160,1760,8)],float);unskew=pixels.copy();unskew[:,0]-=c[10]*(unskew[:,1]-c[9])/K[1,1];rays=np.c_[cv2.undistortPoints(unskew[:,None],K,c[11:25])[:,0],np.ones(len(pixels))]@R;lam=-c[2]/rays[:,2];world=c[:3]+lam[:,None]*rays;norm=(world-origin)/scale;inside=(lam>0)&np.isfinite(norm).all(1)&(scene.containment(norm[:,:2])<-.001)
   X=np.r_[norm[inside],board];target,depth=old_project(c,X*scale+origin);choose=(depth>.05)&np.isfinite(target).all(1)&(target[:,0]>=100)&(target[:,0]<=1820)&(target[:,1]>=80)&(target[:,1]<=1000);X=X[choose];target=target[choose]
   if len(X)<8:raise ValueError('Not enough finite navigation witnesses '+sid+'/'+str(pt[i]))
   initial=start[i].copy()
   def fun(q):
    p=current_at(q,rig,X);r=(p['uv']-target).ravel();return np.nan_to_num(r,nan=1e8,posinf=1e8,neginf=-1e8)
   fit=least_squares(fun,initial,bounds=(initial-[1,1,.7],initial+[1,1,.7]),x_scale=[.1,.1,.1],max_nfev=80,ftol=1e-11,xtol=1e-11,gtol=1e-9)
   values.append(fit.x);before=np.linalg.norm(fun(initial).reshape(-1,2),axis=1);after=np.linalg.norm(fun(fit.x).reshape(-1,2),axis=1);anchors.append(dict(native_pts=int(pt[i]),witnesses=len(X),before_navigation_rms_px=float(np.sqrt(np.mean(before*before))),after_navigation_rms_px=float(np.sqrt(np.mean(after*after))),total_yaw_pitch_logf=fit.x.tolist(),success=bool(fit.success),current_principal_point=rig['optics']['principal_point'],navigation_principal_point=c[8:10].tolist()))
  values=np.asarray(values);values[:,:2]=np.unwrap(values[:,:2],axis=0);shot=graph['shots'][sid];duration=(shot['scope']['last_pts']-shot['scope']['first_pts'])/12800.;control,offset=spline_seed(times,values,{**plans[sid],'duration':duration});patch[sid]=dict(camera_controls=control,reference_angles=offset.tolist(),anchors=anchors)
  proof.append(dict(shot_id=sid,coordinate_normalization=normalization,navigation_mapping=dict(path=str(navpath),sha256=file_sha256(navpath)),navigation_archive=ref,source_pts_field='official_pts only; legacy exposure fields never used',old_values_role='unaccepted numerical navigation target only; no source measurement emitted',witness_source='conditional finite ice-ray intersections and same normalized finite board; no source pixels created'))
 # The reference shot has zero offsets by definition. Transfer its reference
 # angles to R0 and subtract them from every member's offset: the combined
 # rotation Rb(beta) R0 Rn(alpha) is exactly preserved by this coordinate change.
 head_patch={}
 for rid in sorted({SHOT_TO_RIG[s] for s in patch}):
  reference=REFERENCE_SHOTS[rid]
  if reference not in patch:continue
  members={s for s,v in graph['shots'].items() if v['rig_id']==rid}
  if not members.issubset(patch):raise ValueError('Reference head rebase requires all rig member shot motions')
  ref=np.asarray(patch[reference]['reference_angles']);head=graph['rigs'][rid]['shared_head'];old=np.asarray(head['R0']);new=rotation(head['b'],ref[1])@old@rotation(head['n'],ref[0]);error=0.
  for sid in sorted(members):
   offset=np.asarray(patch[sid]['reference_angles']);replacement=offset-ref
   for q in patch[sid]['camera_controls']['controls']:
    q=np.asarray(q);before=rotation(head['b'],q[1]+offset[1])@old@rotation(head['n'],q[0]+offset[0]);after=rotation(head['b'],q[1]+replacement[1])@new@rotation(head['n'],q[0]+replacement[0]);error=max(error,float(np.max(np.abs(before-after))))
   patch[sid]['reference_angles']=replacement.tolist()
  if error>1e-12:raise ValueError('Reference gauge transfer did not preserve the proposed rotations')
  head_patch[rid]=dict(R0_before=old.tolist(),R0_after=new.tolist(),subtracted_reference_angles=ref.tolist(),rotation_equivalence_max_abs=error,position_axes_lever_pupil_and_optics_changed=False,reference_shot=reference)
 out.mkdir(parents=True);doc=dict(schema='rinkfit.joint9.optical_coordinate_seed_patch.v1',source_model=dict(path=str(Path(model_file).resolve()),sha256=file_sha256(model_file)),shared_head_reference_gauge_patch=head_patch,shared_main_head_or_optics_changed=False,s01_changed=False,source_pixel_likelihood_changed=False,patch=patch,provenance=proof,physical_calibration_complete=False,required_next='apply bound shot motion and necessary exact R0 reference-gauge patch to candidate; full original native source recapture/evaluation/fitting must decide acceptance');write_json(out/'patch.json',doc);return doc

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--project',required=True);p.add_argument('--run',required=True);p.add_argument('--model',required=True);p.add_argument('--out',required=True);p.add_argument('--shots',type=int,nargs='+',default=[4,7,9]);a=p.parse_args();d=convert(a.project,a.run,a.model,a.out,a.shots);print(json.dumps({s:dict(anchors=len(x['anchors']),before_rms=np.mean([r['before_navigation_rms_px'] for r in x['anchors']]),after_rms=np.mean([r['after_navigation_rms_px'] for r in x['anchors']])) for s,x in d['patch'].items()}))
