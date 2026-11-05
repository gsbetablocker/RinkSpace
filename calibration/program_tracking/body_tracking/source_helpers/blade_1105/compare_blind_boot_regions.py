import sys,os,json,time
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
P=Path(r'D:/OneDrive/codex_projs/RinkSpace');O=Path(r'D:/CodexData/RinkSpace_Data/runs/body_s01_20261029/blade_orientation_1105');sys.path[:0]=[str(P),str(P/'src')]
from calibration.program_tracking.body_tracking.blade_projection import SavedBladeProjection
from calibration.program_tracking.body_tracking.skate_template import profile_height
os.environ['RINKSPACE_BODY_NATIVE_LIBRARY']=r'D:/CodexData/RinkSpace_Data/runs/body_perf_1103/native_cpu/rinkspace_core.dll'
a=json.loads((O.parent/'blade_observability_1105/factor_assembly_audit.json').read_text(encoding='utf-8'));c=a['candidates']['local_281'];m=SavedBladeProjection(P,c['stage_root'],c['input_files'][-1]['path']);geo=json.loads((O/'source_axis_geometry_review.json').read_text(encoding='utf-8'))
p=m.binding.packet();adapter=p['adapter'];q=np.stack([adapter[[0,7]].mean(0),adapter[[1,8]].mean(0)])
rows=[]
for g in geo['items']:
 regions=g.get('boot_contour_regions') or {}
 if not regions.get('heel') or not regions.get('toe'):continue
 pt=g['pts'];foot=g['foot_proposal'];target=np.array([regions[k]['center_native_xy_px'] for k in ['heel','toe']]);half=np.array([regions[k]['half_extent_native_xy_px'] for k in ['heel','toe']]);W=1/(np.sqrt(2)*half)
 def uv(d,kind):return m.project_points(np.array([pt]),foot,q,**({'static_delta':d} if kind=='static' else {'ankle_delta':d[None]}))['uv'][0]
 def residual(d,kind):
  offset=(uv(d,kind)-target)*W;norm=np.linalg.norm(offset,axis=-1);return (offset*np.maximum(0,1-1/np.maximum(norm,1e-30))[:,None]).ravel()
 b=m.project_points(np.array([pt]),foot,q,include_body=True)
 for kind in ['static','ankle']:
  result=least_squares(lambda d:residual(d,kind),np.zeros(3),bounds=(-np.pi,np.pi));v=uv(result.x,kind)
  body=m.project_points(np.array([pt]),foot,q,include_body=True,**({'static_delta':result.x} if kind=='static' else {'ankle_delta':result.x[None]}))
  def ang(x,y):return float(np.degrees(np.arccos(np.clip(abs(x@y)/(np.linalg.norm(x)*np.linalg.norm(y)),-1,1))))
  # Combined source-centre derivatives show whether the two mechanisms can be distinguished.
  h=1e-5;J=[]
  for mode in ['static','ankle']:
   for i in range(3):
    d=np.zeros(3);d[i]=h;J.append(((uv(d,mode)-uv(-d,mode))*W/(2*h)).ravel())
  sv=np.linalg.svd(np.stack(J,1),compute_uv=False)
  r=dict(pts=pt,seconds=pt/12800,foot=foot,source_role=g['role'],kind=kind,delta=result.x.tolist(),rotation_deg=float(np.linalg.norm(result.x)*180/np.pi),nfev=result.nfev,base_uv=b['uv'][0].tolist(),trial_uv=v.tolist(),source_centres=target.tolist(),source_half_extent=half.tolist(),outside_residual_before=residual(np.zeros(3),kind).tolist(),outside_residual_after=result.fun.tolist(),boot_axis_before_deg=ang(b['uv'][0,1]-b['uv'][0,0],target[1]-target[0]),boot_axis_after_deg=ang(v[1]-v[0],target[1]-target[0]),centre_error_before=np.linalg.norm(b['uv'][0]-target,axis=1).tolist(),centre_error_after=np.linalg.norm(v-target,axis=1).tolist(),body_pixel_change17=np.linalg.norm(body['coco_uv'][0]-b['coco_uv'][0],axis=-1).tolist(),combined_static_ankle_source_jacobian_singular_values=sv.tolist(),not_a_candidate=True)
  rows.append(r);print(json.dumps({k:r[k] for k in ['seconds','kind','rotation_deg','boot_axis_before_deg','boot_axis_after_deg','outside_residual_after','centre_error_after']}),flush=True)
(O/'new_boot_counterfactuals.json').write_text(json.dumps(dict(schema='rinkspace.blind-boot-region-counterfactual.v1',rows=rows,source_is_finite_contour_region_not_material=True,no_pose_or_binding_modified=True,diagnostic_trials_not_for_playback=True),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
