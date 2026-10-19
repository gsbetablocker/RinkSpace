"""Read all existing actual-view roll angles; no model update, J, or images."""
from pathlib import Path
import hashlib
import json
import numpy as np
import cv2

ROOT=Path(__file__).resolve().parents[2]
RUNS=Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/calibration/runs/solver_acceleration_20261003')
STATES={
 'q8':('okada_review86_q276_vertical_print_station_one_cycle_20261011','8a15a4ed0c82beb490993db9c58294af592096e8159d9652b5bd5c50217c71eb'),
 'q8dcd':('okada_review86_q8_bounded_motion_nuisance_fix_20261011','8dcd3b9104dd1281bfcb6c09e2f7edb62af4e2712351ec2e7acfba1c4ce94817'),
 'q441':('okada_review86_q8_saved_boundary_location_fix_20261011','441aa15fdfe8c21870399389125a8610de3d65577e8d53db22020ab6e9ccaf91')}


def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def sha(path):
 with Path(path).open('rb') as s:return hashlib.file_digest(s,'sha256').hexdigest()
def emit(stage,**value):print(json.dumps(dict(stage=stage,**value),ensure_ascii=False),flush=True)


def load(run,which):
 directory=RUNS/run/'accepted_states'/which;identity=read(directory/'state_identity.json');path=directory/'complete_state_UNACCEPTED.npz'
 if sha(path)!=identity['state_npz_sha256']:raise ValueError('Saved state bytes changed')
 with np.load(path,allow_pickle=False) as z:
  q=z['q'][z['aux__current_evaluation_to_canonical_parameter_indices']]
  contract=json.loads(str(z['aux__correlated_roll_prior_contract_json'].item()))
  names=z['aux__block_names'].tolist();lengths=z['aux__block_lengths']//4;starts=z['aux__block_start'];stamps=z['aux__canonical_pts'].copy()
  xyz=np.asarray(z['aux__installation_xyz_slots'])
  optics=json.loads(str(z['aux__all_installation_optics_contract_json'].item()))
  instmap={name:i for i,name in enumerate(contract['installation_order'])}
  # Determine installation from the saved public camera package, rather than
  # inferring it from a programme name or temporal shot number.
 camera=Path(identity['camera_path'])
 if sha(camera)!=identity['camera_sha256']:raise ValueError('Saved camera bytes changed')
 with np.load(camera,allow_pickle=False) as z:
  pts=z['pts'];meta=json.loads(str(z['metadata_json'].item()))
  installations=z['installation_id'];export=z['camera25']
 theta=[];owners=[];rvec=[];cursor=0;errors=[]
 for name,length,start in zip(names,lengths,starts):
  p=stamps[cursor:cursor+length];cursor+=int(length);positions=int(start)+4*np.arange(length)
  views=q[positions[:,None]+np.arange(4)]
  ix=np.searchsorted(pts,p)
  if not np.array_equal(pts[ix],p):raise ValueError('Actual state exposures missing from original export')
  values=np.asarray([cv2.Rodrigues(v[:3])[0] for v in views]);x,y=values[:,0,2],-values[:,1,2]
  if np.any(x*x+y*y<1e-8):raise ValueError('Original roll domain undefined')
  rotations=np.asarray([cv2.Rodrigues(v[3:6])[0] for v in export[ix]])
  errors.append(float(np.max(abs(values-rotations))))
  theta.extend(np.arctan2(x,y));owners.extend(instmap[str(v)] for v in installations[ix]);rvec.extend(views[:,:3])
 if cursor!=len(stamps) or max(errors)>1e-12:raise ValueError('Source and exported world-to-camera rotations differ')
 bias=q[np.asarray(contract['statistical_bias_parameter_indices'],int)]
 return dict(path=str(path),sha256=sha(path),theta=np.asarray(theta),owners=np.asarray(owners,int),stamps=stamps,bias=bias,sigma=contract['component_sigma_radians'],rotation_export_max_error=max(errors),installation_order=contract['installation_order'])


def main():
 states={name:load(*pair) for name,pair in STATES.items()};base=states['q8']
 for name,s in states.items():
  if not np.array_equal(s['stamps'],base['stamps']) or not np.array_equal(s['owners'],base['owners']) or s['sigma']!=base['sigma']:raise ValueError('Actual roll source exposure/installation/sigma changed')
  emit('state',name=name,path=s['path'],sha256=s['sha256'],sigma_degrees=float(np.rad2deg(s['sigma'])),
       bias_degrees=np.rad2deg(s['bias']).tolist(),source_camera_rotation_max_error=s['rotation_export_max_error'],
       branch_wrap_points=int((abs(s['theta'])>np.pi/2).sum()),actual_views=len(s['theta']))
 shots=read(ROOT/'calibration/program_tracking/source_shots.json')['shots'][1:10]
 assigned=np.zeros(len(base['stamps']),bool)
 for shot,rec in enumerate(shots,1):
  mask=(base['stamps']>=int(rec['start_pts']))&(base['stamps']<int(rec['end_pts']))
  if np.any(assigned&mask) or not mask.any():raise ValueError('Actual shot source assignment is not an exact partition')
  assigned|=mask;inst=np.unique(base['owners'][mask])
  if len(inst)!=1:raise ValueError('One source shot spans physical installations')
  inst=int(inst[0]);out={}
  for name,s in states.items():
   angle=np.rad2deg(s['theta'][mask]);r=(s['theta'][mask]-s['bias'][inst])/s['sigma'];change=np.rad2deg(s['theta'][mask]-base['theta'][mask])
   out[name]=dict(actual_roll_degrees_p05_p50_p95=np.quantile(angle,[.05,.5,.95]).tolist(),
       original_prior_residual_radians_mean=float(np.mean(s['theta'][mask]-s['bias'][inst])),
       original_prior_residual_sigma_mean=float(r.mean()),prior_cost=float(.5*r@r),
       roll_change_from_q8_degrees_p05_p50_p95=np.quantile(change,[.05,.5,.95]).tolist(),
       signed_roll_change_points=dict(negative=int((change<0).sum()),zero=int((change==0).sum()),positive=int((change>0).sum())))
  emit('whole_shot',shot=shot,start_pts=int(rec['start_pts']),stop_pts=int(rec['end_pts']),actual_exposures=int(mask.sum()),installation=base['installation_order'][inst],states=out)
 if not assigned.all():raise ValueError('Actual roll source views outside nine shots')
 emit('complete',source_J=False,fit=False,image_decode=False,image_writes=0,report_files_written=0)


if __name__=='__main__':main()
