"""Current-state S05 blade/boot projections and three-rotation counterfactuals.
No candidate export, new shoe state, or change to production objectives.
"""
import sys,os,json,csv,time,hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.optimize import least_squares
from threadpoolctl import threadpool_limits
P=Path(r'D:/OneDrive/codex_projs/RinkSpace');D=Path(r'D:/CodexData/RinkSpace_Data');O=D/'runs/body_s01_20261029/blade_orientation_1105';A=O.parent/'blade_observability_1105'
sys.path[:0]=[str(P),str(P/'src')]
os.environ['RINKSPACE_BODY_NATIVE_LIBRARY']=str(D/'runs/body_perf_1103/native_cpu/rinkspace_core.dll')
from calibration.program_tracking.body_tracking.blade_projection import SavedBladeProjection
from calibration.program_tracking.body_tracking.skate_template import profile_height

def load(f):return json.loads(Path(f).read_text(encoding='utf-8-sig'))
def write(name,value):
 (O/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def unit(x):return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)
def angle(a,b):return float(np.degrees(np.arccos(np.clip(abs(unit(a)@unit(b)),-1,1))))
def stats(a):
 a=np.asarray(a,float);return dict(n=len(a),median=float(np.median(a)),p95=float(np.quantile(a,.95)),maximum=float(a.max())) if len(a) else dict(n=0)
def row_q(model,r):
 packet=model.binding.packet();p=packet['parameters'];k=r['kind']
 if k in ['blade_endpoint_region','blade_midpoint_region']:
  x={'rear':-p.runner_rear_extension_m,'front':p.sole_length_m-p.runner_front_inset_m}.get(r.get('blade_endpoint'),.5*(-p.runner_rear_extension_m+p.sole_length_m-p.runner_front_inset_m))
  return np.array([x,0.,float(profile_height(x,p))])
 return packet['adapter'][np.asarray(r.get('adapter_point_ids',list(range(14))),int)].mean(0)
def region_error(uv,r):
 d=np.asarray(uv)-r['centre_uv'];rad=float(np.sqrt(d@np.linalg.solve(np.asarray(r['region_shape_px2']),d)))
 return dict(centre_px=float(np.linalg.norm(d)),outside_px=float(np.linalg.norm(d))*max(0,1-1/max(rad,1e-30)),dx=float(d[0]),dy=float(d[1]))

def source_axis_key(item):
 s=item.get('visible_steel_line_segment');return None if not s else np.asarray(s['point_b_native_xy_px'])-np.asarray(s['point_a_native_xy_px'])

audit=load(A/'blade_observability_audit.json');assembly=audit['factor_assembly'];geo=load(O/'source_axis_geometry_review.json')
gmap={(r['pts'],r['foot_proposal']):r for r in geo['items']}
vis={(e['pts'],r['foot']):r['visual_review'] for e in audit['reviewed_source_exposures'] for r in e['feet']}
allpts=np.array([e['pts'] for e in audit['reviewed_source_exposures']],np.int64)
source_rows=[r['original_row'] for r in assembly['candidates']['C6']['rows'] if r['actual_fit_index_0based'] is not None and r['kind']!='support_contact_region']
row_index=defaultdict(list)
for r in source_rows:row_index[(r['pts'],r['foot'])].append(r)
models={};wide=[];row_records=[];projections={}
tick=time.perf_counter()
for label,c in assembly['candidates'].items():
 model=SavedBladeProjection(P,c['stage_root'],c['input_files'][-1]['path']);models[label]=model
 packet=model.binding.packet();p=packet['parameters'];xx=np.array([-p.runner_rear_extension_m,.5*(-p.runner_rear_extension_m+p.sole_length_m-p.runner_front_inset_m),p.sole_length_m-p.runner_front_inset_m]);q=np.c_[xx,np.zeros(3),profile_height(xx,p)]
 # Boot heel and toe are the same fixed adapter means used by current source factors.
 for feature,defaultids in [('heel',[0,6,7,13]),('toe',[1,2,8,9])]:
  rr=next((r for r in source_rows if r.get('feature')==feature and r['foot']==1),None)
  q=np.r_[q,packet['adapter'][np.asarray(rr.get('adapter_point_ids',defaultids) if rr else defaultids,int)].mean(0)[None]]
 for foot in [0,1]:
  projected=model.project_points(allpts,foot,q)['uv'];projections[(label,foot)]=dict(pts=allpts,uv=projected,q=q)
  for pt,uv in zip(allpts,projected):
   pt=int(pt);v=vis[(pt,foot)];sr=row_index[(pt,foot)];bykind=defaultdict(list)
   for r in sr:bykind[r['kind']].append(r)
   ep={r.get('blade_endpoint'):r for r in bykind['blade_endpoint_region']}
   heel=next((r for r in sr if r.get('feature')=='heel'),None);toe=next((r for r in sr if r.get('feature')=='toe'),None)
   g=gmap.get((pt,foot));gs=g.get('visible_steel_line_segment') if g else None
   rec=dict(candidate=label,pts=pt,seconds=pt/12800,foot=foot,foot_identity=v['anatomical_identity'],source_frame=v.get('source_frame'),source_frame_sha256=v.get('source_frame_sha256'),blade_axis_visibility=v['blade_axis_observable'],blade_rear_u=float(uv[0,0]),blade_rear_v=float(uv[0,1]),blade_middle_u=float(uv[1,0]),blade_middle_v=float(uv[1,1]),blade_front_u=float(uv[2,0]),blade_front_v=float(uv[2,1]),boot_heel_u=float(uv[3,0]),boot_heel_v=float(uv[3,1]),boot_toe_u=float(uv[4,0]),boot_toe_v=float(uv[4,1]),blade_axis_error_deg=None,blade_axis_error_status='orientation_unobservable',source_segment_to_model_whole_axis_deg=None,source_segment_is_whole_axis='unknown',conditional_endpoint_regions_axis_error_deg=None,boot_contour_axis_error_deg=None)
   # A source-visible axis plus both finite endpoint labels yields only a conditional centre-defined axis.
   if set(ep)=={'front','rear'} and v['blade_axis_observable']=='visible':
    s=np.asarray(ep['front']['centre_uv'])-ep['rear']['centre_uv'];rec['conditional_endpoint_regions_axis_error_deg']=angle(uv[2]-uv[0],s)
    rec['blade_axis_error_deg']=rec['conditional_endpoint_regions_axis_error_deg'];rec['blade_axis_error_status']='conditional_finite_endpoint_region_centres_not_certified_material_or_exact_axis'
   if gs and g['axis_direction_usable']:
    rec['source_segment_to_model_whole_axis_deg']=angle(uv[2]-uv[0],source_axis_key(g));rec['source_segment_is_whole_axis']=gs.get('representative_of_full_blade_axis','unknown')
    rec['source_segment_role']=g['role'];rec['source_segment_angle_interval_deg']=json.dumps(gs['undirected_angle_interval_degrees_image_xy'])
    # Unknown mapping of a visible subsegment to the full rear-front chord is not promoted into a verified blade-axis error.
   if heel and toe and v['boot_heel_visible']=='visible' and v['boot_toe_visible']=='visible':rec['boot_contour_axis_error_deg']=angle(uv[4]-uv[3],np.asarray(toe['centre_uv'])-heel['centre_uv'])
   if g and g.get('boot_contour_regions',{}).get('heel') and g.get('boot_contour_regions',{}).get('toe'):
    bc=g['boot_contour_regions'];rec['new_source_boot_contour_axis_error_deg']=angle(uv[4]-uv[3],np.asarray(bc['toe']['center_native_xy_px'])-bc['heel']['center_native_xy_px']);rec['new_source_boot_role']=g['role']
   for endpoint,index in [('rear',0),('front',2)]:
    r=ep.get(endpoint)
    if r:
     for k,val in region_error(uv[index],r).items():rec[f'{endpoint}_region_{k}']=val
   wide.append(rec)
 # Project actual row q, not a centre or surface invented from another semantic.
 for foot in [0,1]:
  rows=[r for r in source_rows if r['foot']==foot];pt=np.array([r['pts'] for r in rows]);local=np.array([row_q(model,r) for r in rows])[:,None,:]
  uv=model.project_points(pt,foot,local)['uv'][:,0]
  for r,u in zip(rows,uv):row_records.append(dict(candidate=label,id=r['id'],pts=r['pts'],foot=foot,kind=r['kind'],feature=r.get('feature'),blade_endpoint=r.get('blade_endpoint'),uv=u.tolist(),**region_error(u,r),role=r['role'],material_identity_confirmed=r.get('material_identity_confirmed')))
 print(json.dumps(dict(event='projected_current_candidate',candidate=label,frames=len(allpts),elapsed=time.perf_counter()-tick)),flush=True)
fields=sorted(set(k for r in wide for k in r));fields=['candidate','pts','seconds','foot']+[k for k in fields if k not in ['candidate','pts','seconds','foot']]
with (O/'blade_axis_projection_comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fields);w.writeheader();w.writerows(wide)
write('current_projection_details.json',dict(schema='rinkspace.blade-current-native-projection.v1',same_evidence_document=assembly['candidates']['C6']['document'],all_four_same_camera_pts_units=True,source_rows=row_records,wide_rows=wide,source_geometry=geo,role_note='New manual lines remain diagnostic only; short subsegment orientation is not full-blade-axis truth.',no_candidate_export=True))
# Static registration and existing-ankle-only source-region-centre studies.
# These tiny counterfactual least squares do NOT alter any production objective/state.
counter=[]
for label in ['local_281']:
 model=models[label]
 for foot in [0,1]:
  rows=[r for r in source_rows if r['foot']==foot];pt=np.array([r['pts'] for r in rows]);q=np.array([row_q(model,r) for r in rows])[:,None,:];target=np.array([r['centre_uv'] for r in rows]);S=np.array([r['region_shape_px2'] for r in rows]);W=np.linalg.inv(np.linalg.cholesky(S));base_uv=model.project_points(pt,foot,q)['uv'][:,0]
  def scaled(uv):return np.einsum('nij,nj->ni',W,uv-target).ravel()
  def static_res(d):return scaled(model.project_points(pt,foot,q,static_delta=d)['uv'][:,0])
  result=least_squares(static_res,np.zeros(3),loss='linear',bounds=(-np.pi,np.pi))
  trial=model.project_points(pt,foot,q,static_delta=result.x)['uv'][:,0]
  counter.append(dict(candidate=label,foot=foot,kind='fixed_body_static_R_FB_only',rows=len(rows),source_unique_pts=len(set(pt)),delta_right_R_FB_rotvec=result.x.tolist(),rotation_degrees=float(np.linalg.norm(result.x)*180/np.pi),centre_normalized_cost_before=float(scaled(base_uv)@scaled(base_uv)/2),centre_normalized_cost_after=float(result.fun@result.fun/2),centre_error_before=stats(np.linalg.norm(base_uv-target,axis=1)),centre_error_after=stats(np.linalg.norm(trial-target,axis=1)),outside_before=stats([region_error(u,r)['outside_px'] for u,r in zip(base_uv,rows)]),outside_after=stats([region_error(u,r)['outside_px'] for u,r in zip(trial,rows)]),nfev=result.nfev,status=result.status,message=result.message,body_is_identical=True,source_region_centres_are_not_exact_material_truth=True,not_a_candidate=True))
  print(json.dumps(dict(event='static_counterfactual',foot=foot,nfev=result.nfev,rotation_deg=float(np.linalg.norm(result.x)*180/np.pi))),flush=True)
  # Only exposure groups with at least two finite source regions; singleton points do not determine orientation.
  for point in sorted(set(pt)):
   ix=np.flatnonzero(pt==point)
   if len(ix)<2:continue
   rlocal=[rows[i] for i in ix];qq=q[ix];tt=target[ix];ww=W[ix]
   def ankle_res(d):return np.einsum('nij,nj->ni',ww,model.project_points(pt[ix],foot,qq,ankle_delta=np.broadcast_to(d,(len(ix),3)))['uv'][:,0]-tt).ravel()
   rr=least_squares(ankle_res,np.zeros(3),loss='linear',bounds=(-np.pi,np.pi))
   before=model.project_points(np.array([point]),foot,model.binding.packet()['adapter'][[0,1]],include_body=True)
   after=model.project_points(np.array([point]),foot,model.binding.packet()['adapter'][[0,1]],ankle_delta=rr.x[None],include_body=True)
   bodychange=after['coco_uv'][0]-before['coco_uv'][0]
   sv=np.linalg.svd(rr.jac,compute_uv=False)
   trial=model.project_points(pt[ix],foot,qq,ankle_delta=np.broadcast_to(rr.x,(len(ix),3)))['uv'][:,0]
   counter.append(dict(candidate=label,foot=foot,pts=int(point),seconds=point/12800,kind='fixed_static_registration_existing_ankle_only',rows=len(ix),delta_right_ankle_rotvec=rr.x.tolist(),rotation_degrees=float(np.linalg.norm(rr.x)*180/np.pi),normalized_centre_before=float(ankle_res(np.zeros(3))@ankle_res(np.zeros(3))/2),normalized_centre_after=float(rr.fun@rr.fun/2),centre_before=stats(np.linalg.norm(base_uv[ix]-tt,axis=1)),centre_after=stats(np.linalg.norm(trial-tt,axis=1)),outside_before=stats([region_error(u,r)['outside_px'] for u,r in zip(base_uv[ix],rlocal)]),outside_after=stats([region_error(u,r)['outside_px'] for u,r in zip(trial,rlocal)]),source_jacobian_singular_values=sv.tolist(),body_pixel_change_all17=stats(np.linalg.norm(bodychange,axis=1)),ankle_coco_pixel_changes=np.linalg.norm(bodychange[[15,16]],axis=1).tolist(),nfev=rr.nfev,message=rr.message,source_geometry_is_conditional=True,not_a_joint_candidate=True,body_root_and_other_joint_local_rotations_frozen=True))
  print(json.dumps(dict(event='ankle_counterfactuals_complete',foot=foot,exposures=sum(r.get('foot')==foot and r['kind']=='fixed_static_registration_existing_ankle_only' for r in counter))),flush=True)
write('counterfactual_projection_study.json',dict(schema='rinkspace.blade-existing-state-counterfactual.v1',mathematical_objective='Read-only finite-region-centre explanation study scaled by hand-picked shape, not production loss/accuracy; no probability or material truth inferred.',state_rule='Static R_FB (3) per foot globally versus existing body ankle local rotation (3) per observed exposure. No translations, no independent shoe state, no new DOF registered, no trial exported.',counterfactuals=counter,no_joint_solve=True,no_new_default=True,elapsed_seconds=time.perf_counter()-tick))
write('stage2_projection_summary.json',dict(candidates=list(models),pts=len(allpts),projection_csv_rows=len(wide),source_region_rows=len(row_records),projection_elapsed_and_counterfactual_seconds=time.perf_counter()-tick,source_axis_not_certified=True,counterfactuals=len(counter),new_body_or_shoe_candidate=False))
print(json.dumps(dict(event='complete',csv_rows=len(wide),counterfactuals=len(counter),elapsed=time.perf_counter()-tick)),flush=True)
