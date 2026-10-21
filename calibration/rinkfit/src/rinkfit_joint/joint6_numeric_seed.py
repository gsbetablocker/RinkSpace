"""Bound only added-shot log-f to an existing engineering enclosure.

Targets are the inherited model function clipped to that enclosure, not source
observations. This is a changed numerical starting state, never a preservation
or camera-calibration claim. Shared geometry/optics and all other scalars stay.
"""
from __future__ import annotations
import copy
from pathlib import Path
import numpy as np
from scipy.optimize import lsq_linear
from .io_utils import read_json,write_json,file_sha256
from .source_quality_compare_v1 import _camera

def _ref(p):
 p=Path(p).resolve();return dict(file=str(p),sha256=file_sha256(p),bytes=p.stat().st_size)
def _stats(x):
 x=np.asarray(x,float);return dict(count=int(x.size),min=float(x.min()),max=float(x.max()),rms=float(np.sqrt(np.mean(x*x))),max_abs=float(np.max(np.abs(x))))
def _basis(camera,times):
 x=(np.asarray(times)-camera.first)/camera.dt;i=np.floor(x).astype(int);i=np.where(i==len(camera.controls)-2,i-1,i);u=x-i
 if np.any(i<1)or np.any(i>len(camera.controls)-3):raise ValueError('Full original RS endpoint outside allocated cubic support')
 w=np.stack(((1-u)**3,3*u**3-6*u*u+4,-3*u**3+3*u*u+3*u+1,u**3),axis=-1)/6
 A=np.zeros((len(times),len(camera.controls)));rr=np.arange(len(times))
 for k in range(4):A[rr,i+k-1]=w[:,k]
 return A

def build_bounded_added_shot_logf_seed(model_file,output,*,existing_ranges,parent_certified_domain_file):
 parent=read_json(model_file);m=copy.deepcopy(parent);out=Path(output).resolve()
 if out.exists():raise FileExistsError('Immutable numerical seed output exists')
 if set(m['shots'])!=set('s%02d'%i for i in range(1,7)):raise ValueError('Numerical seed is scoped to the existing six-shot activity view')
 report=dict(schema='rinkfit.changed_added_shot_bounded_logf_numeric_seed.v1',parent=_ref(model_file),existing_domain=_ref(parent_certified_domain_file),existing_logf_ranges=existing_ranges,per_shot={},source_UV_or_qualification_changed=False,new_source_observations=0,hardware_values_claimed=False,original_function_preserved=False,source_fit_accuracy_verified=False,quality_accepted=False,domain_expanded=False)
 values={};changed=[];expected=set()
 for sid in ('s05','s06'):
  shot=m['shots'][sid];camera=_camera(parent,sid);rig=shot['rig_id'];lo,hi=map(float,existing_ranges[rig])
  if not np.isfinite([lo,hi]).all()or not lo<hi:raise ValueError('Invalid existing closed logf range')
  if 'control_coordinates'in shot:raise ValueError('Added shot is not in original direct logf control chart')
  pts=np.arange(shot['scope']['first_pts'],shot['scope']['last_pts']+1,256,dtype=np.int64);centres=(pts-shot['time_origin_pts'])/12800.
  offsets=np.array([-.5,0.,.5])*camera.tau;times=(centres[:,None]+offsets).reshape(-1);A=_basis(camera,times)
  old=camera.controls[:,2];original=camera.spline(times)[:,2];basis_error=float(np.max(np.abs(A@old-original)))
  if basis_error>8e-15:raise ValueError('Numerical basis differs from original native evaluator')
  lower=np.maximum(lo,np.array([parent['blocks'][i]['lower'][2]for i in shot['control_blocks']]));upper=np.minimum(hi,np.array([parent['blocks'][i]['upper'][2]for i in shot['control_blocks']]))
  if np.any(lower>=upper):raise ValueError('No nonempty existing bounded logf control interval')
  target=np.clip(original,lo,hi);fit=lsq_linear(A,target,bounds=(lower,upper),method='trf',lsq_solver='exact',tol=1e-12,max_iter=None)
  if not fit.success or not np.isfinite(fit.x).all()or np.any(fit.x<lower)or np.any(fit.x>upper):raise ValueError('Bounded numerical logf seed failed')
  for k,bi in enumerate(shot['control_blocks']):
   block=m['blocks'][bi]
   if block['name']!='shot/'+sid+'/control/'+str(k)or len(block['values'])!=3:raise ValueError('Unexpected added-shot control layout')
   before=block['values'][2];block['values'][2]=float(fit.x[k]);expected.add((bi,2));changed.append(dict(block=bi,name=block['name'],ambient_index=2,before=before,after=float(fit.x[k])))
  actual=_camera(m,sid).spline(times)[:,2]
  if not np.allclose(actual,A@fit.x,rtol=0,atol=8e-15):raise ValueError('New function differs from original native evaluator')
  values[sid+'_native_pts']=pts;values[sid+'_row_time_offsets']=offsets;values[sid+'_times']=times;values[sid+'_inherited_logf']=original;values[sid+'_clipped_engineering_target_logf']=target;values[sid+'_new_logf']=actual
  report['per_shot'][sid]=dict(rig=rig,native_frames=len(pts),RS_offsets_seconds=offsets.tolist(),rows_of_original_cubic_basis=len(times),basis_roundoff_max=basis_error,existing_closed_logf_range=[lo,hi],old_controls=_stats(old),new_controls=_stats(fit.x),control_change=_stats(fit.x-old),inherited_function=_stats(original),new_function=_stats(actual),function_change=_stats(actual-original),difference_from_clipped_engineering_target=_stats(actual-target),max_relative_focal_change=float(np.max(np.abs(np.exp(actual-original)-1))),LS_cost=float(fit.cost),LS_optimality=float(fit.optimality),LS_status=int(fit.status),target_definition='inherited native camera logf clipped to existing engineering enclosure; not measured source data')
 for i,(a,b)in enumerate(zip(parent['blocks'],m['blocks'])):
  for k,(x,y)in enumerate(zip(a['values'],b['values'])):
   if x!=y and(i,k)not in expected:raise ValueError('Non-logf/shared/S01 scalar changed')
 if parent['rigs']!=m['rigs']or parent['materials']!=m['materials']or parent['shots']!=m['shots']or parent['linear_control_manifolds']!=m['linear_control_manifolds']:raise ValueError('Shared physical metadata changed')
 report['all_other_physical_scalars_and_S01_exact']=True;report['changed_controls']=changed
 m['joint6_bounded_added_motion_numeric_seed']={k:copy.deepcopy(report[k])for k in ('schema','parent','existing_domain','existing_logf_ranges','source_UV_or_qualification_changed','original_function_preserved','domain_expanded')}
 out.mkdir(parents=True);write_json(out/'model6.json',m);np.savez_compressed(out/'function_differences.npz',**values)
 report['model']=_ref(out/'model6.json');report['function_differences']=_ref(out/'function_differences.npz');report['implementation']=_ref(__file__)
 domain=dict(schema='rinkfit.dense6_capture_numeric_domain.v1',parent_certified_domain=_ref(parent_certified_domain_file),activity_model=report['model'],direct_logf_shots=['s02','s03','s04','s05','s06'],whole_six_g0_claimed=False,certificate_expanded=False)
 write_json(out/'capture_numeric_domain.json',domain);report['capture_numeric_domain']=_ref(out/'capture_numeric_domain.json');write_json(out/'report.json',report);return report
