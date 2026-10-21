"""Reuse a verified optical source superset only for unchanged model/raw UV.

Qualification withdrawals change no measured ray. This path rechecks exact raw
column identity and active subset, and preserves explicit superset bounds rather
than pretending to recompute current-source ray quantiles.
"""
from __future__ import annotations
import copy,hashlib,json
from pathlib import Path
import numpy as np


def _sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def _ref(path):
 p=Path(path).resolve();return dict(file=str(p),sha256=_sha(p),bytes=p.stat().st_size)

def _json(path):return json.loads(Path(path).read_text(encoding='utf-8'))

def _pin(ref):
 if _sha(ref['file'])!=ref['sha256']:raise ValueError('Changed exact optical reuse dependency: '+ref['file'])
 return Path(ref['file'])

def _view(path):
 arrays={};result={};seen=set();dependencies=[];current=Path(path).resolve()
 while True:
  if str(current)in seen:raise ValueError('Recursive compact source packet')
  seen.add(str(current));raw=_json(current);dependencies.append(_ref(current))
  for k,v in raw.get('arrays',{}).items():arrays.setdefault(k,v)
  for k in ('row_count','shot_ids','dense_source_binding','dense_joint_source_append'):
   if k in raw:result.setdefault(k,raw[k])
  if all(k in result for k in ('row_count','shot_ids','dense_source_binding','dense_joint_source_append')) and len(arrays)==18:break
  parent=raw.get('dense_parent_packet')
  if not parent:raise ValueError('Incomplete columnar dense source packet')
  current=_pin(parent).resolve()
 result['arrays']=arrays;result['dependencies']=dependencies;return result

def rebind_unchanged_model_source_subset(candidate_model,packet_file,parent_domain,out):
 out=Path(out).resolve()
 if out.exists():raise FileExistsError(out)
 previous=_json(parent_domain)
 if previous['schema']!='rinkfit.dense6_source_supported_optical_domain.v1':raise ValueError('Expected existing source-supported box')
 candidate=_ref(candidate_model)
 if candidate['sha256']!=previous['activity_model']['sha256'] or _pin(previous['activity_model']).resolve()!=Path(candidate_model).resolve():raise ValueError('Subset reuse requires exactly unchanged physical model')
 parent_coverage_ref=previous['source_coverage'];coverage=_json(_pin(parent_coverage_ref))
 if coverage['model']!=previous['activity_model'] or coverage['source_packet']!=previous['source_packet']:raise ValueError('Parent source coverage binding mismatch')
 if any(x[k]!=0 for x in coverage['per_shot'].values() for k in ('inverse_Brown_failed','outside_frozen_ray_disk','outside_unchanged_zoom_box')):raise ValueError('Parent source superset is not completely inside the existing optical box')
 old=_view(_pin(previous['source_packet']));new=_view(packet_file)
 for k in ('row_count','shot_ids','dense_source_binding','dense_joint_source_append'):
  if old[k]!=new[k]:raise ValueError('Changed raw source registry: '+k)
 if old['dense_joint_source_append']['actual_column_identity_sha256']!=new['dense_joint_source_append']['actual_column_identity_sha256']:raise ValueError('Changed exact source column digest')
 if set(old['arrays'])!=set(new['arrays']):raise ValueError('Changed raw source column registry')
 for k in old['arrays']:
  if k not in ('admission','fit_role') and old['arrays'][k]!=new['arrays'][k]:raise ValueError('Changed raw numeric/source column '+k)
 def column(view,key):
  spec=view['arrays'][key];p=_pin(spec)
  return np.memmap(p,dtype=np.dtype(spec['dtype']),mode='r',shape=tuple(spec['shape']))
 # Same pin means same original UV/clock; no camera projection/inverse Brown.
 for k in ('uv','native_pts','shot_index'):_pin(new['arrays'][k])
 old_active=(column(old,'admission')==1)&(column(old,'fit_role')==1)
 new_active=(column(new,'admission')==1)&(column(new,'fit_role')==1)
 if np.any(new_active&~old_active):raise ValueError('New active rows lack parent optical source coverage')
 if int(old_active.sum())!=coverage['full_denominator']:raise ValueError('Parent actual active denominator differs')
 shots=column(new,'shot_index');pts=column(new,'native_pts');per={}
 for si,sid in enumerate(new['shot_ids']):
  before=coverage['per_shot'][sid];take=new_active&(shots==si)
  per[sid]=dict(rig=before['rig'],active_source_rows=int(take.sum()),native_exposures=int(np.unique(pts[take]).size),inverse_Brown_failed=0,outside_frozen_ray_disk=0,outside_unchanged_zoom_box=0,verified_source_superset_rows=before['active_source_rows'],source_log_f_superset_bounds=before.get('source_log_f',before.get('source_log_f_superset_bounds')),source_focal_native_px_superset_bounds=before.get('source_focal_native_px',before.get('source_focal_native_px_superset_bounds')),source_inverse_ray_radius_superset_bounds=before.get('source_inverse_ray_radius',before.get('source_inverse_ray_radius_superset_bounds')),exact_current_ray_quantiles_recomputed=False)
 packet=_ref(packet_file);out.mkdir(parents=True)
 proof=dict(schema='rinkfit.same_model_identical_raw_source_active_subset.v1',candidate_model=candidate,parent_domain=_ref(parent_domain),parent_source_coverage=parent_coverage_ref,old_source_packet=previous['source_packet'],source_packet=packet,raw_rows=new['row_count'],old_active_rows=int(old_active.sum()),new_active_rows=int(new_active.sum()),newly_inactive_rows=int(np.count_nonzero(old_active&~new_active)),source_UV_PTS_shot_array_refs={k:new['arrays'][k]for k in ('uv','native_pts','shot_index')},old_membership_refs={k:old['arrays'][k]for k in ('admission','fit_role')},new_membership_refs={k:new['arrays'][k]for k in ('admission','fit_role')},raw_registry_exact=True,raw_source_digest_unchanged=True,active_subset_verified=True,parent_dependencies=old['dependencies'],current_dependencies=new['dependencies'],model_or_source_reprojection_performed=False,passed=True)
 def write(name,doc):(out/name).write_text(json.dumps(doc,ensure_ascii=False,separators=(',',':')),encoding='utf-8')
 write('source_subset.json',proof)
 current=dict(schema=coverage['schema'],model=candidate,source_packet=packet,per_shot=per,per_rig=copy.deepcopy(coverage['per_rig']),full_denominator=int(new_active.sum()),raw_denominator=new['row_count'],source_UV_PTS_or_qualification_changed=True,source_UV_PTS_unchanged=True,qualification_membership_withdrawals_only=True,hardware_calibration_claimed=False,quality_accepted=False,source_supported_parent_domain=_ref(parent_domain),verified_source_superset=parent_coverage_ref,actual_active_subset=_ref(out/'source_subset.json'),current_ray_statistics_are_verified_superset_bounds=True,reader=_ref(__file__))
 write('source_coverage.json',current)
 domain=copy.deepcopy(previous);domain.update(activity_model=candidate,source_packet=packet,source_coverage=_ref(out/'source_coverage.json'))
 write('capture_optical_domain.json',domain)
 receipt=dict(schema='rinkfit.same_closed_optical_box_source_subset_rebind.v1',candidate=candidate,source_packet=packet,parent_domain=_ref(parent_domain),domain=_ref(out/'capture_optical_domain.json'),source_coverage=_ref(out/'source_coverage.json'),source_subset=_ref(out/'source_subset.json'),raw_rows=new['row_count'],active_rows=int(new_active.sum()),source_model_and_box_unchanged=True,source_UV_PTS_unchanged=True,qualification_withdrawals_only=True,optical_or_physical_reprojection_performed=False,new_certificate_or_hardware_calibration_issued=False,whole_six_g0_claimed=False,reader=_ref(__file__),quality_accepted=False,passed=True)
 write('receipt.json',receipt)
 return receipt
