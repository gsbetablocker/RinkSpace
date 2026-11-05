import os,sys,json,copy,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.sparse import csr_matrix
P=Path(r'D:/OneDrive/codex_projs/RinkSpace');D=Path(r'D:/CodexData/RinkSpace_Data');O=D/'runs/body_s01_20261029/blade_support_profile_1105';O.mkdir(exist_ok=True);sys.path[:0]=[str(P),str(P/'src')];os.environ['RINKSPACE_BODY_NATIVE_LIBRARY']=str(D/'runs/body_perf_1103/native_cpu/rinkspace_core.dll')
from calibration.program_tracking.body_tracking.blade_current_reference import restore_current_reference
from calibration.program_tracking.body_tracking.blade_boot_refit import _mass,_prior,ExistingActionsWithBootAxes
from calibration.program_tracking.body_tracking.spin_continuous_refit import ContinuousJoint
from calibration.program_tracking.body_tracking.absolute_motion_curve import AbsoluteMotionCurve,greville_times,_exp,_log
from calibration.program_tracking.body_tracking.absolute_motion_state import curve_from_packet,evaluate_curve
from calibration.program_tracking.body_tracking.skate_physics import fk_state,batch_features
from calibration.program_tracking.body_tracking.body_execution import BodyExecution,BodyExecutionConfig
from calibration.program_tracking.body_tracking.support_image import SupportImageFactors
from calibration.program_tracking.body_tracking.rebuild_prepare import load,digest
base=D/'runs/body_s01_20261029/blade_orientation_1105/entry_81_2_82_0';parameter=P/'events/ankara_2026/programmes/okada_mei_sp/spin_boot_axis_entry_1105.npz'
ref,camera,info,saved=restore_current_reference(P,base,parameter,seed=True);execution=BodyExecution(BodyExecutionConfig(workers=12,blas_threads=1,local_derivatives='analytic',compatibility_backend='cuda_resident'));execution.attach(ref)
doc=json.loads((base/'support_and_free_shoe_regions.json').read_text(encoding='utf-8'));runtime=copy.deepcopy(doc);mult=ref.grasp_relations['factor_parameters']['shoe_region_weight_multiplier']
for r in runtime['observations']:r['objective_weight']*=mult
for r in runtime['boot_axis_observations']:r['objective_weight']*=mult
mass,_=_mass(ref,base);initial=load(base/'common/exmr_hand_nofoot/initial.npz');prior=_prior(initial,ref)
with (O/'objective_probe.jsonl').open('w',encoding='utf-8',buffering=1) as log, execution.limits():
 problem=ContinuousJoint(ref,camera,ref.observations_path,(86.4,92.86),doc['supports'],O/'not_a_solve',log,'absolute_motion_v1',optimize_shared=False,ice_policy=ref.ice_policy)
 class NoOpening:
  def evaluate(self,x,jacobian=False):return dict(residual=np.empty(0),jac=csr_matrix((0,problem.body.size)),parts={},metrics={})
 problem.source_motion=NoOpening();problem.support_image=SupportImageFactors(problem,runtime);problem.action_factors=ExistingActionsWithBootAxes(problem,prior,mass,runtime)
 seed=curve_from_packet(saved,seed=True);actual=curve_from_packet(saved);gv=greville_times(actual.knots);n=len(gv);before_state=fk_state(ref,evaluate_curve(actual,gv),np.zeros((n,69)))
 packet=ref.binding.packet();eligible=set(packet['template']['support_feature_ids']['glide']);features=batch_features(ref.binding,before_state['R'],before_state['O']);ids=[i for i,name in enumerate(features['ids']) if name in eligible]
 before_h=features['height'][:,1,ids].min(-1)
 selected=(actual.knots[:n]+76>=86.8)&(actual.knots[4:]+76<=92.46);ci=np.flatnonzero(selected)
 records=[]
 for scale in [0.,.25,.5,1.]:
  rotations=actual.rotation_controls.copy();positions=actual.position_controls.copy()
  R=before_state['R'][:,1];forward=R[:,:,0];horizontal=forward.copy();horizontal[:,2]=0;horizontal/=np.linalg.norm(horizontal,axis=1)[:,None]
  axis=np.cross(np.broadcast_to([0.,0.,1.],horizontal.shape),horizontal);pitch=np.arctan2(forward[:,2],np.linalg.norm(forward[:,:2],axis=1));ankle=before_state['Q']@before_state['frames'][:,8]
  localaxis=np.einsum('nji,nj->ni',ankle,axis)*pitch[:,None]*scale
  rotations[ci,8]=rotations[ci,8]@_exp(localaxis[ci])
  newcurve=AbsoluteMotionCurve(actual.knots,positions,rotations,actual.edge_reference_quaternion)
  after_state=fk_state(ref,evaluate_curve(newcurve,gv),np.zeros((n,69)));feat=batch_features(ref.binding,after_state['R'],after_state['O']);after_h=feat['height'][:,1,ids].min(-1)
  correction=(before_h-after_h)/ref.binding.ratio;B=actual.basis(gv[ci])[:,ci];positions[ci,2]+=np.linalg.solve(B,correction[ci])
  controls=saved['x'][17:].reshape(n,69).copy();controls[ci,2]=positions[ci,2]-seed.position_controls[ci,2];controls[ci,27:30]=_log(seed.rotation_controls[ci,8].swapaxes(-1,-2)@rotations[ci,8])
  x=saved['x'].copy();x[17:]=controls.ravel();t=time.perf_counter();value=problem.evaluate(x,False)
  source=ref.p['pts']/12800;use=(source>=86.8)&(source<=92.46);d=problem.body.state.increments(x[17:],ref.times[use]);s=fk_state(ref,ref.at(ref.times[use]),d);f=batch_features(ref.binding,s['R'],s['O']);h=f['height'][:,1,ids].min(-1)
  pitchafter=np.degrees(np.arctan2(s['R'][:,1,2,0],np.linalg.norm(s['R'][:,1,:2,0],axis=1)))
  r=dict(flatten_fraction=scale,seconds=time.perf_counter()-t,parts={k:float(v) for k,v in value['parts'].items()},joint_cost=float(value['residual']@value['residual']/2),image_cost=float(value['image_cost']),raw_data_parts=value['raw_data_parts'],pitch_after_deg=np.quantile(pitchafter,[.05,.5,.95]).tolist(),runner_min_height_after_m=np.quantile(h,[.05,.5,.95]).tolist(),structural_max_violation_m=value['gate']['max_violation_m'],baseline_contact_height_preserved_at_control_midpoints=True,not_a_candidate=True)
  records.append(r);print(json.dumps(r),flush=True)
(O/'objective_pitch_probe.json').write_text(json.dumps(dict(schema='rinkspace.fixed-footwear-pitch-objective-probe.v1',parameter=str(parameter),parameter_sha256=digest(parameter),same_body_ankle_root_variables=True,no_new_shoe_DOF=True,freeze_template_camera=True,notes='Whole-blade x-axis neutralisation is a diagnostic counterfactual, not a desired spin pose or a measured contact law. Existing runner minimum retained approximately by root-z compensation.',records=records),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
execution.close()
