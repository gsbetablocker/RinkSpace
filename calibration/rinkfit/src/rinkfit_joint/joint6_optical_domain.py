"""Source-supported six-shot optical math box; no hardware or full-g0 claim."""
from __future__ import annotations
import copy,hashlib,json
from pathlib import Path
import numpy as np
from .io_utils import read_json,write_json,file_sha256
from .source_quality_compare_v1 import _camera
from .dense_source_motion import _inverse_brown_rows

def _ref(p):
 p=Path(p).resolve();return dict(file=str(p),sha256=file_sha256(p),bytes=p.stat().st_size)
def _stats(x):
 x=np.asarray(x,float);good=x[np.isfinite(x)];r=dict(denominator=int(x.size),finite=int(good.size),unknown=int(x.size-good.size))
 if good.size:r.update(min=float(good.min()),max=float(good.max()),median=float(np.median(good)),p99=float(np.quantile(good,.99)))
 return r

def build_source_supported_optical_seed(activity_model,bounded_model,packet_file,old_domain,out):
 out=Path(out).resolve()
 if out.exists():raise FileExistsError(out)
 parent=read_json(activity_model);bounded=read_json(bounded_model);m=copy.deepcopy(parent);changed=[]
 for bi in m['shots']['s06']['control_blocks']:
  a=m['blocks'][bi]['values'];b=bounded['blocks'][bi]['values']
  if a[2]!=b[2]:changed.append(dict(block=bi,name=m['blocks'][bi]['name'],before=a[2],after=b[2]));a[2]=b[2]
 # Strictly preserve every other actual scalar and metadata, including complete S05.
 verify=copy.deepcopy(m)
 for item in changed:verify['blocks'][item['block']]['values'][2]=item['before']
 if verify!=parent:raise AssertionError('Unexpected physical/model change')
 old=read_json(old_domain);ranges={'main':[8.042490207840718,10.3643895631867],'secondary':[8.319989835896669,13.815510557964274]}
 cams={sid:_camera(m,sid)for sid in m['shots']};ranges['end']=[min(float(c.controls[:,2].min())for sid,c in cams.items()if m['shots'][sid]['rig_id']=='end'),max(float(c.controls[:,2].max())for sid,c in cams.items()if m['shots'][sid]['rig_id']=='end')]
 raw=read_json(packet_file);arrays={}
 for key in ('uv','native_pts','shot_index','admission','fit_role'):
  a=raw['arrays'][key]
  if file_sha256(a['file'])!=a['sha256']:raise ValueError('Changed source column '+key)
  arrays[key]=np.memmap(a['file'],dtype=np.dtype(a['dtype']),mode='r',shape=tuple(a['shape']))
 active=(arrays['admission']!=0)&(arrays['fit_role']!=0);per={};rigs={}
 for sid,c in cams.items():
  mask=(arrays['shot_index']==raw['shot_ids'].index(sid))&active;uv=np.array(arrays['uv'][mask]);pts=np.array(arrays['native_pts'][mask]);t=(pts-m['shots'][sid]['time_origin_pts'])/12800.+c.tau*(uv[:,1]/1079.-.5)
  lf=c.spline(t)[:,2];z=lf-np.log(c.optics['f_ref']);principal=np.asarray(c.optics['principal_point'])+z[:,None]*np.asarray(c.optics['principal_linear']);fxy=np.c_[np.exp(lf+c.optics['eta']/2),np.exp(lf-c.optics['eta']/2)]
  coeff=np.c_[np.polynomial.polynomial.polyval(z,c.optics['k1']),np.polynomial.polynomial.polyval(z,c.optics['k2']),np.full(len(z),c.optics['p1']),np.full(len(z),c.optics['p2'])]+np.maximum(c.wide_boundary-lf,0.)[:,None]**2*c.wide_coefficients
  xy,valid=_inverse_brown_rows((uv-principal)/fxy,coeff);radius=np.linalg.norm(xy,axis=1);radius[~valid]=np.nan
  rig=m['shots'][sid]['rig_id'];lo,hi=ranges[rig];per[sid]=dict(rig=rig,active_source_rows=int(mask.sum()),native_exposures=int(np.unique(pts).size),source_log_f=_stats(lf),source_focal_native_px=_stats(np.exp(lf)),source_inverse_ray_radius=_stats(radius),inverse_Brown_failed=int((~valid).sum()),outside_frozen_ray_disk=int((valid&(radius>c.optics['radius_max'])).sum()),outside_new_zoom_box=int(((lf<lo)|(lf>hi)).sum()),outside_historical_zoom_box=int(((lf<({'end':[9.179826821732064,9.927670536803493],'main':ranges['main'],'secondary':ranges['secondary']}[rig][0]))|(lf>({'end':[9.179826821732064,9.927670536803493],'main':ranges['main'],'secondary':ranges['secondary']}[rig][1]))).sum()),source_ray_reconstructed_from_model=True,model_inverse_is_not_hardware_measurement=True)
 for rid,rr in m['rigs'].items():
  cam=_camera(m,rr['reference_shot']);cam.controls[0,2]=ranges[rid][0];cam.controls[1,2]=ranges[rid][1];cert=cam.optical_domain_certificate()
  if not cert['passed']:raise ValueError('Whole optical box failed '+rid)
  rigs[rid]=dict(log_f_range=ranges[rid],ray_radius=float(cam.optics['radius_max']),python_optical_certificate=cert,optics_initial_assumption=('temporary zero-Brown candidate; source-supported coverage, not calibrated hardware'if rid!='main'else'preserved source-supported main Brown and wide extension; complete S01 projection unchanged'))
 out.mkdir(parents=True);write_json(out/'model6.json',m)
 coverage=dict(schema='rinkfit.dense6_actual_source_optical_coverage.v1',model=_ref(out/'model6.json'),source_packet=_ref(packet_file),per_shot=per,per_rig=rigs,full_denominator=int(active.sum()),source_UV_PTS_or_qualification_changed=False,hardware_calibration_claimed=False,quality_accepted=False)
 write_json(out/'source_coverage.json',coverage)
 domain=dict(schema='rinkfit.dense6_source_supported_optical_domain.v1',activity_model=_ref(out/'model6.json'),historical_certified_domain=_ref(old_domain),source_packet=_ref(packet_file),source_coverage=_ref(out/'source_coverage.json'),direct_logf_shots=['s02','s03','s04','s05','s06'],candidate_math_box_not_hardware_calibration=True,whole_six_g0_claimed=False,historical_certificate_unchanged=True,scene_policy='fixed_to_current_source_supported_candidate_scene',rigs=rigs)
 write_json(out/'capture_optical_domain.json',domain)
 receipt=dict(schema='rinkfit.dense6_source_supported_optical_seed_receipt.v1',parent=_ref(activity_model),bounded_s06_reference=_ref(bounded_model),model=_ref(out/'model6.json'),domain=_ref(out/'capture_optical_domain.json'),source_coverage=_ref(out/'source_coverage.json'),only_changed_scalar_coordinates=changed,S05_function_exactly_preserved=True,S01_complete_projection_dependencies_exactly_preserved=True,shared_scene_rigs_optics_exactly_preserved=True,new_source_accuracy_verified=False,quality_accepted=False)
 write_json(out/'receipt.json',receipt);return receipt
