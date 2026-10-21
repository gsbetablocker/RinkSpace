"""Rebind an accepted numerical candidate to the same closed optical math box.

This issues neither source identity nor a physical-quality or hardware claim.
It does not change model scalars, source pixels, eligibility, or box boundaries.
"""
from __future__ import annotations
import copy
from pathlib import Path
import numpy as np
from .io_utils import read_json,write_json,file_sha256
from .joint6_optical_domain import _ref,_stats
from .source_quality_compare_v1 import _camera
from .dense_source_motion import _inverse_brown_rows


def rebind_source_supported_optical_candidate(candidate_model,packet_file,parent_domain,out,*,allowed_head_rigs=()):
 out=Path(out).resolve()
 if out.exists():raise FileExistsError(out)
 previous=read_json(parent_domain)
 if previous['schema']!='rinkfit.dense6_source_supported_optical_domain.v1':raise ValueError('Explicit source-supported parent optical math box required')
 parent_ref=previous['activity_model']
 if file_sha256(parent_ref['file'])!=parent_ref['sha256']:raise ValueError('Changed optical parent model')
 parent=read_json(parent_ref['file']);model=read_json(candidate_model)
 if sorted(model['shots'])!=['s01','s02','s03','s04','s05','s06'] or len(model['rigs'])!=3:raise ValueError('Expected one scene / three rigs / six shots')
 allowed=set();heads=set(allowed_head_rigs)
 if not heads<={'end','secondary'}:raise ValueError('Only explicitly selected non-main heads may change')
 head_blocks={model['rigs'][rid]['head_block']for rid in heads}
 allowed.update(head_blocks)
 for sid in ('s02','s03','s04','s05','s06'):
  allowed.update(model['shots'][sid]['control_blocks'])
  allowed.add(model['shots'][sid]['reference_angles_block'])
 test=copy.deepcopy(model);changed=[]
 if len(model['blocks'])!=len(parent['blocks']):raise ValueError('Changed physical block registry')
 for i,(a,b) in enumerate(zip(model['blocks'],parent['blocks'])):
  if a['values']!=b['values']:
   if i not in allowed or a['name']!=b['name'] or (i not in head_blocks and a['name'].split('/')[:1]!=['shot']):raise ValueError('Optical rebind only permits non-S01 motion and explicitly selected end/secondary heads: '+a['name'])
   changed.append(dict(block=i,name=a['name'],maximum_absolute_scalar_change=float(np.max(np.abs(np.asarray(a['values'])-b['values'])))))
   test['blocks'][i]['values']=copy.deepcopy(b['values'])
 if test!=parent:raise ValueError('Optical rebind cannot change scene/main/optics/S01 or projection metadata')
 raw=read_json(packet_file);arrays={}
 for key in ('uv','native_pts','shot_index','admission','fit_role'):
  spec=raw['arrays'][key]
  if file_sha256(spec['file'])!=spec['sha256']:raise ValueError('Changed source column '+key)
  arrays[key]=np.memmap(spec['file'],dtype=np.dtype(spec['dtype']),mode='r',shape=tuple(spec['shape']))
 active=(arrays['admission']!=0)&(arrays['fit_role']!=0)
 per={};rigs={};cams={sid:_camera(model,sid)for sid in model['shots']}
 for sid,cam in cams.items():
  rid=model['shots'][sid]['rig_id'];lo,hi=previous['rigs'][rid]['log_f_range']
  if sid!='s01' and (np.min(cam.controls[:,2])<lo or np.max(cam.controls[:,2])>hi):raise ValueError('Candidate control left the unchanged closed optical box: '+sid)
  take=(arrays['shot_index']==raw['shot_ids'].index(sid))&active
  uv=np.array(arrays['uv'][take]);pts=np.array(arrays['native_pts'][take])
  t=(pts-model['shots'][sid]['time_origin_pts'])/12800.+cam.tau*(uv[:,1]/1079.-.5)
  lf=cam.spline(t)[:,2];z=lf-np.log(cam.optics['f_ref'])
  pp=np.asarray(cam.optics['principal_point'])+z[:,None]*np.asarray(cam.optics['principal_linear'])
  fxy=np.c_[np.exp(lf+cam.optics['eta']/2),np.exp(lf-cam.optics['eta']/2)]
  co=np.c_[np.polynomial.polynomial.polyval(z,cam.optics['k1']),np.polynomial.polynomial.polyval(z,cam.optics['k2']),np.full(len(z),cam.optics['p1']),np.full(len(z),cam.optics['p2'])]
  co+=np.maximum(cam.wide_boundary-lf,0.)[:,None]**2*cam.wide_coefficients
  xy,valid=_inverse_brown_rows((uv-pp)/fxy,co);radius=np.linalg.norm(xy,axis=1);radius[~valid]=np.nan
  per[sid]=dict(rig=rid,active_source_rows=int(take.sum()),native_exposures=int(np.unique(pts).size),source_log_f=_stats(lf),source_focal_native_px=_stats(np.exp(lf)),source_inverse_ray_radius=_stats(radius),inverse_Brown_failed=int((~valid).sum()),outside_frozen_ray_disk=int((valid&(radius>cam.optics['radius_max'])).sum()),outside_unchanged_zoom_box=int(((lf<lo)|(lf>hi)).sum()),source_ray_reconstructed_from_model=True,model_inverse_is_not_hardware_measurement=True)
 for rid,rr in model['rigs'].items():
  cam=_camera(model,rr['reference_shot']);box=previous['rigs'][rid]['log_f_range']
  cam.controls[0,2]=box[0];cam.controls[1,2]=box[1];cert=cam.optical_domain_certificate()
  if not cert['passed'] or cert['full_cubic_log_f_hull']!=box:raise ValueError('Whole unchanged optical box failed: '+rid)
  rigs[rid]=dict(log_f_range=box,ray_radius=previous['rigs'][rid]['ray_radius'],python_optical_certificate=cert,optics_initial_assumption=previous['rigs'][rid]['optics_initial_assumption'])
 out.mkdir(parents=True)
 coverage=dict(schema='rinkfit.dense6_actual_source_optical_coverage.v1',model=_ref(candidate_model),source_packet=_ref(packet_file),per_shot=per,per_rig=rigs,full_denominator=int(active.sum()),raw_denominator=int(len(active)),source_UV_PTS_or_qualification_changed=False,hardware_calibration_claimed=False,quality_accepted=False,source_supported_parent_domain=_ref(parent_domain),reader=_ref(__file__))
 write_json(out/'source_coverage.json',coverage)
 domain=dict(schema=previous['schema'],activity_model=_ref(candidate_model),historical_certified_domain=previous['historical_certified_domain'],source_packet=_ref(packet_file),source_coverage=_ref(out/'source_coverage.json'),direct_logf_shots=previous['direct_logf_shots'],candidate_math_box_not_hardware_calibration=True,whole_six_g0_claimed=False,historical_certificate_unchanged=True,scene_policy=previous['scene_policy'],rigs=rigs)
 write_json(out/'capture_optical_domain.json',domain)
 receipt=dict(schema='rinkfit.dense6_same_optical_box_candidate_rebind.v1',candidate=_ref(candidate_model),parent_domain=_ref(parent_domain),domain=_ref(out/'capture_optical_domain.json'),source_coverage=_ref(out/'source_coverage.json'),changed_motion_blocks=changed,model_or_source_changed=False,complete_S01_scene_main_rig_and_all_optics_preserved=True,permitted_head_rigs=sorted(heads),actual_head_changes=[r for r in changed if r['block']in head_blocks],zoom_ray_box_expanded=False,true_RS_verified=False,quality_accepted=False,reader=_ref(__file__))
 write_json(out/'receipt.json',receipt)
 return receipt
