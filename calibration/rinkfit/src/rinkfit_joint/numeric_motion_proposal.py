"""Numerical shot-motion proposals from existing admitted fixed native readouts.

These 2D targets generate an initial camera state only.  They are not new W,
constraints, source means, identities, or final likelihood.  A complete saved
AL problem must accept/reject the proposal after its actual source auxiliaries
are optimized; no source or existing dual is changed here.
"""
from pathlib import Path
import argparse,copy,hashlib,json,collections
import numpy as np
from scipy.optimize import least_squares
from .camera_graph import CameraGraph
from .joint_model import graph_from_joint,as_event_view
from .initialize_cameras import spline_seed
from rinkfit_start.native_scene import NativeScene

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb')as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def propose(parent,packet,source_metadata,out,shot_id='s09'):
 parent=Path(parent);packet=Path(packet);source_metadata=Path(source_metadata);out=Path(out)
 if out.exists():raise ValueError('immutable numerical proposal output already exists')
 model=json.loads((parent/'model.json').read_text());graph=graph_from_joint(model);cg=CameraGraph(graph);camera=cg.native_camera(shot_id);scene=NativeScene(as_event_view(model,'s01'))
 with packet.open('r',encoding='utf8')as f:head=f.read(32768)
 desc=json.JSONDecoder().raw_decode(head[head.index('"arrays":')+len('"arrays":'):])[0]
 def ar(name):
  d=desc[name];return np.memmap(d['file'],mode='r',dtype=d['dtype'],shape=tuple(d['shape']),offset=d.get('offset_bytes',0))
 uv=ar('uv');pts=ar('native_pts');mat=ar('material_index');kind=ar('kind');ad=ar('admission');role=ar('fit_role');si=ar('shot_index')
 shot_ids=json.JSONDecoder().raw_decode(head[head.index('"shot_ids":')+len('"shot_ids":'):])[0];shot_index=shot_ids.index(shot_id)
 rows=[r for r in json.loads(source_metadata.read_text())['rows'] if r['source_row_id'].startswith(shot_id+'/')]
 byframe=collections.defaultdict(list)
 for r in rows:
  i=r['row'];assert int(pts[i])==r['native_pts'] and int(mat[i])==r['material_index'] and int(kind[i])==1 and int(ad[i])==1 and int(role[i])==1 and int(si[i])==shot_index
  assert np.array_equal(uv[i],np.asarray(r['uv'],float))
  byframe[int(pts[i])].append(r)
 base_controls=camera.controls.copy();anchors=[];underidentified=[]
 for pt,rr in sorted(byframe.items()):
  if len(rr)<2:
   underidentified.append(dict(native_pts=pt,rows=[r['row']for r in rr],reason='single point does not identify three instantaneous pose variables; source stays active'));continue
  X=np.array([scene.world(r['material_index'],0,0.)for r in rr]);mu=np.array([r['uv']for r in rr]);t=cg.local_seconds(shot_id,np.array([pt],np.int64))[0];readout_times=t+camera.tau*(mu[:,1]/1079.-.5)
  def fun(delta):
   camera.controls=base_controls+delta[None,:]
   q=camera.at(X,readout_times);res=q['uv']-mu
   return np.nan_to_num(res,nan=1e8,posinf=1e8,neginf=-1e8).ravel()
  before=fun(np.zeros(3));fit=least_squares(fun,np.zeros(3),bounds=(-np.ones(3)*.8,np.ones(3)*.8),x_scale=np.ones(3)*.1,max_nfev=100,ftol=1e-11,xtol=1e-11,gtol=1e-9)
  rank=int(np.linalg.matrix_rank(fit.jac));singular=np.linalg.svd(fit.jac,compute_uv=False)
  if rank!=3:
   underidentified.append(dict(native_pts=pt,rows=[r['row']for r in rr],reason='instantaneous 2D numerical pose Jacobian is rank deficient; source stays active',rank=rank));continue
  anchors.append(dict(native_pts=pt,local_seconds=float(t),rows=[r['row']for r in rr],source_row_ids=[r['source_row_id']for r in rr],delta_yaw_pitch_logf=fit.x.tolist(),before_RMS_px=float(np.sqrt(np.mean(np.sum(before.reshape(-1,2)**2,axis=1)))),after_RMS_px=float(np.sqrt(np.mean(np.sum(fit.fun.reshape(-1,2)**2,axis=1)))),jacobian_singular_values=singular.tolist(),success=bool(fit.success),fit_message=fit.message))
 camera.controls=base_controls
 if len(anchors)<2:raise ValueError('fewer than two independently identified actual pose anchors')
 full_times=np.array([a['local_seconds']for a in anchors]);full_deltas=np.array([a['delta_yaw_pitch_logf']for a in anchors]);full_logf=camera.spline(full_times)[:,2]+full_deltas[:,2]
 for item in underidentified:
  camera.controls=base_controls
  rr=byframe[item['native_pts']]
  if len(rr)!=1:continue
  r=rr[0];pt=item['native_pts'];t=cg.local_seconds(shot_id,np.array([pt],np.int64))[0];mu=np.array(r['uv'],float);X=scene.world(r['material_index'],0,0.);readout_time=t+camera.tau*(mu[1]/1079.-.5)
  conditional_logf=float(np.interp(t,full_times,full_logf));delta_logf=conditional_logf-float(camera.spline(np.array([readout_time]))[0,2]);start=np.array([np.interp(t,full_times,full_deltas[:,j])for j in range(2)])
  def two_angles(q):
   delta=np.r_[q,delta_logf];camera.controls=base_controls+delta[None,:];v=camera.at(X,readout_time)['uv']-mu;return np.nan_to_num(v,nan=1e8,posinf=1e8,neginf=-1e8)
  initial=two_angles(start);fit=least_squares(two_angles,start,bounds=(-np.ones(2)*.8,np.ones(2)*.8),x_scale=np.ones(2)*.1,max_nfev=100,ftol=1e-11,xtol=1e-11,gtol=1e-9)
  if np.linalg.matrix_rank(fit.jac)!=2:continue
  item['conditioned_two_angle_numeric_proposal']=True
  anchors.append(dict(native_pts=pt,local_seconds=float(t),rows=[r['row']],source_row_ids=[r['source_row_id']],delta_yaw_pitch_logf=np.r_[fit.x,delta_logf].tolist(),before_RMS_px=float(np.linalg.norm(initial)),after_RMS_px=float(np.linalg.norm(fit.fun)),jacobian_singular_values=np.linalg.svd(fit.jac,compute_uv=False).tolist(),success=bool(fit.success),fit_message=fit.message,numeric_conditioning='Single actual corner identifies yaw/pitch conditional on interpolated neighbouring numerical focal proposal only; final focal is not fixed',conditional_numerical_logf=conditional_logf))
 camera.controls=base_controls;anchors.sort(key=lambda a:a['native_pts'])
 s=graph['shots'][shot_id];basis=s['camera_controls']['basis'];duration=(s['scope']['last_pts']-s['scope']['first_pts'])/12800.;times=np.array([a['local_seconds']for a in anchors]);deltas=np.array([a['delta_yaw_pitch_logf']for a in anchors]);samples=np.unique(np.r_[np.linspace(0.,duration,max(4*len(base_controls),len(times))),times])
 interpolated=np.column_stack([np.interp(samples,times,deltas[:,j])for j in range(3)]);total_values=camera.spline(samples)+np.r_[s['reference_angles'],0.]+interpolated
 control,ref=spline_seed(samples,total_values,dict(basis,duration=duration));proposed=copy.deepcopy(model);shot=proposed['shots'][shot_id];changes=[]
 replacements={shot['reference_angles_block']:ref.tolist()}
 if 'control_coordinates'in shot:raise ValueError('packed shot controls require explicit coordinate-map proposal encoder')
 for block,v in zip(shot['control_blocks'],control['controls']):replacements[block]=v
 for block,values in replacements.items():
  b=proposed['blocks'][block];before=b['values'].copy();values=list(values)
  for j in b.get('fixed_indices',[]):values[j]=before[j]
  if b.get('constant',False):
   if values!=before:raise ValueError('proposal would change an original constant block')
  if not np.isfinite(values).all()or np.any(np.array(values)<b['lower'])or np.any(np.array(values)>b['upper']):raise ValueError('numerical proposal outside original box: '+b['name'])
  if values!=before:changes.append(dict(block=block,name=b['name'],before_values=before,after_values=values));b['values']=values
 verify_camera=CameraGraph(graph_from_joint(proposed)).native_camera(shot_id);X=np.array([scene.world(r['material_index'],0,0.)for r in rows]);mu=np.array([r['uv']for r in rows]);tt=cg.local_seconds(shot_id,np.array([r['native_pts']for r in rows],np.int64));projected=verify_camera.project_rs(X,tt);d=np.linalg.norm(projected['uv']-mu,axis=1)
 cp=parent/('numerical_resume_checkpoint.json'if(parent/'numerical_resume_checkpoint.json').exists()else'resume_checkpoint.json')
 doc=dict(schema='rinkfit.joint_shot_motion_numerical_proposal.v1',parent_model=dict(file=str(parent/'model.json'),sha256=sha(parent/'model.json')),parent_checkpoint=dict(file=str(cp),sha256=sha(cp)),source_packet=dict(file=str(packet),sha256=sha(packet)),source_fixed_readout_metadata=dict(file=str(source_metadata),sha256=sha(source_metadata)),shot_id=shot_id,changes=changes,anchors=anchors,underidentified_numeric_frames=underidentified,all_source_rows_remain_active=True,source_UV_W_E_identity_clock_aux_and_duals_changed=False,all_shared_scene_rig_optics_and_other_shot_values_unchanged=True,proposal_fit='unweighted actual native fixed-entity 2D residual only; independent per-frame pose/logf increment at the measured readout row time',interpolation='existing spline_seed on parent motion plus linearly interpolated actual-pose increments; endpoint/halo continuation is numerical initialization only, no model smoothness factor or asserted motion fact',final_problem_unchanged=True,requires_original_complete_AL_and_E_RS_Q_domain_acceptance=True,proposal_fixed_source_diagnostic=dict(count=len(rows),independent_sufficient_RS_valid=int(projected['valid'].sum()),RMS_all_finite_px=float(np.sqrt(np.mean(d*d))),max_all_finite_px=float(np.max(d)),original_E2_pass=int(np.sum(projected['valid']&(d<=2.)))),not_an_accepted_joint_fit=True)
 doc['proposal_fixed_source_rows']=[dict(row=r['row'],native_pts=r['native_pts'],source_row_id=r['source_row_id'],material=r['material'],source_uv=r['uv'],predicted_uv=projected['uv'][i].tolist(),distance_px=float(d[i]),independent_sufficient_RS_valid=bool(projected['valid'][i]))for i,r in sorted(enumerate(rows),key=lambda p:float(d[p[0]]),reverse=True)]
 out.mkdir(parents=True);(out/'proposal.json').write_text(json.dumps(doc,indent=2),encoding='utf8');return doc

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--parent',required=True);p.add_argument('--packet',required=True);p.add_argument('--source-metadata',required=True);p.add_argument('--out',required=True);p.add_argument('--shot',default='s09');a=p.parse_args();d=propose(a.parent,a.packet,a.source_metadata,a.out,a.shot);print(json.dumps(dict(anchors=len(d['anchors']),changes=len(d['changes']),proposal_fixed_source_diagnostic=d['proposal_fixed_source_diagnostic'])))
