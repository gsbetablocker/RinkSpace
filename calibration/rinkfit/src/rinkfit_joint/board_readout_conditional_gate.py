"""Admit exact current board readouts with same-mean conditional RGB statistics."""
from pathlib import Path
import argparse,json,shutil,os
import numpy as np
from .io_utils import read_json,read_jsonl,write_json,write_jsonl,file_sha256,object_sha256
from .measurement_contract import EventMeasurementContract
from .role_guard import FrameRoleGuard


class _StreamingRecords:
 """Append one validated record; the enclosing admission manifest closes last."""
 def __init__(self,path):
  import gzip
  self.handle=gzip.open(path,'xt',encoding='utf8');self.count=0
 def append(self,value):
  self.handle.write(json.dumps(value,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n');self.count+=1
 def __len__(self):return self.count
 def close(self):self.handle.close()

def _compact_solver_record(row,suffix,conditional_model):
 # Full actual pixels/operators remain SHA-bound in the immutable source journal.
 keep=('source_row_id','source_family_id','shot_id','native_pts','source_uv','current_uv','joint_material_id','joint_material_index','measurement_operator_resolved','target_id','kind','entity_station_fixed')
 return dict({k:row[k]for k in keep if k in row},**suffix,conditional_model=conditional_model,physical_total_error_calibrated=False,parent_full_covariance_exact_marginal_required=True)

class AdmissionSession:
 """Reuse immutable evidence digests while streaming original source exposures."""
 def __init__(self,actual_source_aliases):
  self.index_path=Path(actual_source_aliases).resolve();self.index_sha256=file_sha256(self.index_path);self.files={};index=read_json(self.index_path)
  for ref in (index['audit_manifest'],index['aliases'],index['producer_script']):
   p=Path(ref['path']).resolve();digest=file_sha256(p)
   if digest!=ref['sha256']:raise ValueError('Actual alias evidence changed')
   self.files[str(p)]=digest
  audit=read_json(index['audit_manifest']['path'])
  for ref in audit['sources']:
   p=Path(ref['path']).resolve();digest=file_sha256(p)
   if digest!=ref['sha256']:raise ValueError('Alias original source journal changed')
   self.files[str(p)]=digest
  self.aliases={q['source_row_id']:q for q in read_jsonl(index['aliases']['path'])}
 def verify(self,path,digest):
  p=Path(path).resolve();key=str(p)
  if key not in self.files:self.files[key]=file_sha256(p)
  if self.files[key]!=digest:raise ValueError('Immutable source evidence changed '+key)
  return p
 def verify_unchanged(self):
  for p,digest in self.files.items():
   if file_sha256(p)!=digest:raise ValueError('Evidence changed during streaming admission '+p)


def promote(project,run,source_gate,statistics,outdir,*,actual_source_aliases=None,stream_output=False,compact_solver_rows=False,link_source_rows=True,session=None,source_records=None,membership_records=None,source_records_ref=None):
 run=Path(run);source=Path(source_gate);statsfile=Path(statistics);out=Path(outdir)
 data=next(p for p in run.parents if p.name=='RinkSpace_Data');contract=EventMeasurementContract(project,'ankara_2026');guard=FrameRoleGuard(run/'roles/frame_roles.json')
 if out.exists():raise ValueError('New immutable admission generation required')
 registry=read_json(statsfile);single_exposure_receipt=('source_group_id'in registry and 'row_ids'in registry and 'W'in registry)
 if single_exposure_receipt:registry={registry['source_group_id']:registry}
 members={q['source_row_id']:q for q in (membership_records if membership_records is not None else read_jsonl(source/'promotion_decisions.jsonl.gz'))};lookup={};cache={};proofcache={};reviewcache={};solver=[];decisions=[];receipts=[];physical=0;rawcount=0;alias_count=0
 if stream_output:
  out.mkdir(parents=True)
  solver=_StreamingRecords(out/'solver_rows.jsonl.gz');decisions=_StreamingRecords(out/'promotion_decisions.jsonl.gz');receipts=_StreamingRecords(out/'receipts.jsonl.gz')

 aliases={}
 if source_records is not None:
  if not isinstance(session,AdmissionSession)or source_records_ref is None or membership_records is None:raise ValueError('Original source streaming records require a verified admission session and refs')
  if Path(source_records_ref['file']).resolve()not in {(source/name).resolve()for name in ('source_rows.jsonl.gz','geometry_rows.jsonl.gz')}:raise ValueError('Streaming records do not refer to this original source journal')
  session.verify(source_records_ref['file'],source_records_ref['sha256'])
 if session is not None:
  if not isinstance(session,AdmissionSession)or Path(actual_source_aliases).resolve()!=session.index_path or file_sha256(actual_source_aliases)!=session.index_sha256:raise ValueError('Streaming alias session differs')
  aliases=session.aliases
 elif actual_source_aliases is not None:
  index=read_json(actual_source_aliases)
  for ref in (index['audit_manifest'],index['aliases'],index['producer_script']):
   if file_sha256(ref['path'])!=ref['sha256']:raise ValueError('Actual alias audit changed')
  audit=read_json(index['audit_manifest']['path'])
  for ref in audit['sources']:
   digest=file_sha256(ref['path'])
   if digest!=ref['sha256']:raise ValueError('Alias original source journal changed')
   cache[str(Path(ref['path']).resolve())]=digest
  aliases={q['source_row_id']:q for q in read_jsonl(index['aliases']['path'])}
 for gid,g in registry.items():
  for rid in g['row_ids']:
   if rid in lookup:raise ValueError('Readout appears in two noise groups')
   lookup[rid]=gid
 def verify(ref):
  p=Path(ref.get('path',ref.get('file')))
  if not p.is_absolute():p=data/p
  key=str(p.resolve())
  if key in cache:sha=cache[key]
  elif session is not None:session.verify(p,ref['sha256']);sha=ref['sha256'];cache[key]=sha
  else:sha=file_sha256(p);cache[key]=sha
  if sha!=ref['sha256']:raise ValueError('Changed source/statistic artifact '+key)
  return p
 def proof_for(row):
  p=verify(dict(path=row['source_packet'],sha256=row['source_packet_sha256']))
  if proofcache.get('path')!=str(p):
   proofcache.clear();v=read_json(p);proofcache.update(path=str(p),proof=v,rows={q['source_row_id']:q for q in v.get('qualified_source_rows',[])})
  proof=proofcache['proof']
  if 'row_records_sha256'in proof:
   from .board_native_compact import source_record_payload
   if proof['row_records_sha256'].get(row['source_row_id'])!=object_sha256(source_record_payload(row)):raise ValueError('Compact actual source record differs')
   a=row['source_measurement_audit'];verify(a['profile']['intensity_reference']);verify(a['continuous_cubic_material_profile']['actual_profile_intensity_reference'])
   return proof,row
  return proof,proofcache['rows'][row['source_row_id']]
 for row in (source_records if source_records is not None else read_jsonl(source/'source_rows.jsonl.gz')):
  rawcount+=1;rid=row['source_row_id'];d=members[rid]
  if rid in aliases:
   from .source_readout_identity import actual_readout_key
   alias=aliases[rid];alias_count+=1
   if actual_readout_key(row)!=alias['exact_same_actual_measurement_input_sha256']or int(alias['native_pts'])!=int(row['native_pts']):raise ValueError('Alias proof is not this actual readout operator')
   decisions.append(dict(d,source_geometry_eligible=False,solver_eligible=False,statistical_objective_eligible=False,numeric_statistics_ready=False,actual_source_readout_alias_of=alias['actual_source_readout_alias_of'],source_material_identity_withdrawn=False,actual_alias_audit_index=dict(path=str(actual_source_aliases),sha256=file_sha256(actual_source_aliases))));continue
  if not d.get('source_geometry_eligible',False) or row.get('actual_source_readout_alias_of'):
   decisions.append(dict(d,solver_eligible=False,statistical_objective_eligible=False,numeric_statistics_ready=False,actual_source_readout_alias_of=row.get('actual_source_readout_alias_of'),source_material_identity_withdrawn=False if row.get('actual_source_readout_alias_of')else d.get('source_material_identity_withdrawn',False)))
   continue
  physical+=1;pts=int(row['native_pts']);guard.assert_training_pixels([pts],shot_id=row['shot_id']);cert=row['measurement_contract_certificate'];expected=contract.expected(row['target_id']);ch=object_sha256(dict(native_pts=pts,current_uv=row['current_uv'],source_row_id=rid,measurement_operator=row['measurement_operator_resolved']));sh=object_sha256(dict(native_pts=pts,source_uv=row['source_uv'],source_row_id=rid))
  if cert['contract_sha256']!=contract.digest or cert['source_row_id']!=rid or cert['native_pts']!=pts or cert['current_measurement_sha256']!=ch or cert['operator']!=expected['operator']:raise ValueError('Current source does not bind authoritative measurement definition')
  if not cert['actual_RGB_material_identity_verified']or not cert['operator_selected_correct_material_transition']or cert['pixels_from_model']:raise ValueError('Current material identity not proved')
  gid=lookup.get(rid)
  if gid is None:
   decisions.append(dict(d,numeric_statistics_ready=False,statistical_objective_eligible=False,solver_eligible=False,statistics_use='explicit_source_family_balanced_geometry_initialization_only',statistical_pending_reason='same_mean_conditional_native_RGB_group_not_yet_closed',source_physical_membership_retained_from_parent=True,new_statistical_source_promotion=False))
   continue
  for ref in cert['source_evidence']:verify(ref)
  proof,actual=proof_for(row)
  for k in ('source_row_id','native_pts','source_uv','current_uv','normal','navigation_seed_uv','target_id','joint_material_id','measurement_operator_resolved','source_measurement_audit'):
   if actual[k]!=row[k]:raise ValueError('Row differs from current physical source proof: '+k)
  if proof['native_pts']!=pts or proof['shot_id']!=row['shot_id'] or proof['decoded_native_BGR_sha256']!=row['decoded_native_BGR_sha256'] or proof['source_pixels_from_model']:raise ValueError('Actual source exposure lineage differs')
  if proof.get('native_png'):
   verify(dict(path=proof['native_png'],sha256=proof['native_png_sha256']))
  else:
   nr=proof['native_video_decode_receipt']
   if int(nr['native_pts'])!=pts or nr['decoded_native_BGR_sha256']!=row['decoded_native_BGR_sha256']or not nr['training_role_verified']:raise ValueError('Native source receipt differs')
  verify(proof['current_mask']);ctx=proof['current_board_instance_context'];rp=str(verify(ctx['source_review']))
  if rp not in reviewcache:
   reviewcache[rp]=read_json(rp);reviewcache[rp]['_frame_lookup']={int(q['native_pts']):q for q in reviewcache[rp]['frames']}
  review=reviewcache[rp];fr=review['_frame_lookup'][pts]
  if fr['shot_id']!=row['shot_id']or fr['decoded_native_BGR_sha256']!=row['decoded_native_BGR_sha256']or not fr['ROI_is_search_domain_only']:raise ValueError('Object search evidence is not the same native exposure')
  if review['schema']=='rinkfit.joint9.same_native_readout_local_board_navigation.v1':
   verify(fr['source_navigation_index'])
   if any(int(q['native_pts'])!=pts or q['shot_id']!=row['shot_id']or not q['navigation_only']for q in fr['source_navigation_anchors']):raise ValueError('Navigation identity or time changed')
  elif review['schema']=='rinkfit.joint9.current_joint_model_board_navigation.v1':
   verify(fr['source_model_navigation'])
   if not fr['source_object_identity_not_proved_by_navigation']or fr.get('fit_UV_created',False):raise ValueError('Model proposal is not an actual readout')
  else:raise ValueError('This gate requires explicit algorithmic current object-source lineage')
  a=row['source_measurement_audit'];ci=cert['actual_source_certificate_index'];witness=proof['source_definition_certificates'][ci]
  if witness['material_role']!=row['material_role']or not witness['same_current_board_body']or not witness['missing_or_occluded_boundary_is_not_invented']or len(witness['current_color_run_samples'])<4:raise ValueError('Actual finite board material sequence witness missing')
  if not a['actual_outer_material_order_verified']or not a['reflection_excluded']or a['internal_interface_used']or a.get('global_color_rule_used_as_object_identity',True):raise ValueError('Wrong current outer material/object boundary')
  p=a['profile'];numeric=a['continuous_cubic_material_profile']
  if p['channel']!=2 or p['polarity']!=1 or p['interpolation']!='continuous_cubic'or not numeric['numeric_valid']or numeric['synthetic_window_boundary_count']:raise ValueError('Actual frozen signed material profile is not regular')
  q=np.asarray(row['navigation_seed_uv'])+np.asarray(row['normal'])*float(numeric['normal_offset_px'])
  if np.max(np.abs(q-np.asarray(row['current_uv'])))>1e-9:raise ValueError('Current mean is not the stored actual profile measurement')
  s=registry[gid]
  if s['statistic_kind']!='native_joint_readout'or not s['numeric_whitening_verified']or s['actual_mean_replay_max_px']>1e-9 or s['float32_channel_pipeline_error']>0 or s.get('conditional_model_is_total_error_calibration',s.get('physical_total_error_calibrated'))is not False:raise ValueError('Conditional exact-mean native statistics invalid')
  if s['source_measurements_sha256'][rid]!=sh or s['current_measurements_sha256'][rid]!=ch:raise ValueError('W and exact current/source means differ')
  for k in ('W','frozen_operator','replay_arrays'):verify(s[k])
  nr=s['native_receipt']
  if nr['native_pts']!=pts or nr['shot_id']!=row['shot_id']or nr['decoded_native_BGR_sha256']!=row['decoded_native_BGR_sha256']or nr['decoded_native_RGB_sha256']!=proof['actual_native_RGB_sha256']or nr['native_RGB_resampled']or not nr['training_role_verified']:raise ValueError('Shared native RGB covariance exposure differs')
  receipt=dict(schema='rinkfit.current_board_readout_conditional_source_admission.v1',source_row_id=rid,native_pts=pts,source_measurement_sha256=sh,current_measurement_sha256=ch,source_packet_sha256=row['source_packet_sha256'],source_membership=dict(path=str(source/'manifest.json'),sha256=file_sha256(source/'manifest.json')),source_membership_definition_sha256=object_sha256(d),measurement_contract_sha256=contract.digest,measurement_contract_certificate_sha256=object_sha256(cert),source_physical_identity_verified=True,operator_verified=True,statistical_group_id=gid,statistics_definition_sha256=object_sha256(s),statistic_kind='native_joint_readout',conditional_model=s['conditional_model'],physical_total_error_calibrated=False,parent_full_covariance_exact_marginal_required=True,raw_UV_or_current_UV_changed=False,old_recalled_source_W_reused=False)
  receipts.append(receipt);suffix=dict(solver_eligible=True,numeric_statistics_ready=True,statistical_objective_eligible=True,statistical_group_id=gid,statistic_kind='native_joint_readout',statistics_use='conditional_noise_objective',receipt_sha256=object_sha256(receipt));decisions.append(dict(d,**suffix));solver.append(_compact_solver_record(row,suffix,s['conditional_model'])if compact_solver_rows else dict(row,**suffix,conditional_model=s['conditional_model'],physical_total_error_calibrated=False,parent_full_covariance_exact_marginal_required=True))
 for p,sha in cache.items():
  if file_sha256(p)!=sha:raise ValueError('Source changed during admission')
 contract.verify_unchanged();out.mkdir(parents=True,exist_ok=stream_output);raw_storage='immutable_parent_reference'
 if link_source_rows:
  raw_storage='immutable_parent_hardlink'
  try:os.link(source/'source_rows.jsonl.gz',out/'source_rows.jsonl.gz')
  except OSError:
   shutil.copyfile(source/'source_rows.jsonl.gz',out/'source_rows.jsonl.gz');raw_storage='closed_parent_byte_copy_hardlink_unavailable'
 else:
  write_json(out/'source_rows.ref.json',dict(file=str((source/'source_rows.jsonl.gz').resolve()),sha256=file_sha256(source/'source_rows.jsonl.gz')))
 if stream_output:
  for records in (solver,decisions,receipts):records.close()
 else:
  write_jsonl(out/'solver_rows.jsonl.gz',solver);write_jsonl(out/'promotion_decisions.jsonl.gz',decisions);write_jsonl(out/'receipts.jsonl.gz',receipts)
 report=dict(schema='rinkfit.current_board_readout_conditional_admission.v1',raw_rows=rawcount,source_identity_accepted_rows=physical,exact_alias_rows_not_independently_consumed=alias_count,material_identity_of_exact_source_aliases_withdrawn=False,statistical_objective_eligible_rows=len(solver),statistics_pending_source_rows=physical-len(solver),all_rows_accounted=len(decisions)==rawcount,source_membership=dict(path=str(source/'manifest.json'),sha256=file_sha256(source/'manifest.json')),statistics_registry=dict(path=str(statsfile),sha256=file_sha256(statsfile)),measurement_contract_sha256=contract.digest,statistic_kind='native_joint_readout',statistics_use='conditional_noise_objective',physical_total_error_calibrated=False,exact_full_parent_covariance_marginal_required=True,duplicate_geometryI_and_conditionalW_consumption_allowed=False,raw_UV_or_current_UV_or_W_changed=False,exact_actual_source_alias_index=dict(path=str(actual_source_aliases),sha256=file_sha256(actual_source_aliases))if actual_source_aliases is not None else None,full_physical_proof_reverified_rows=len(solver),raw_source_storage=raw_storage,physical_source_membership_for_nonstatistical_rows_retained_from_parent=True,unclosed_W_rows_never_statistically_promoted=True)
 report.update(streamed_output_records=stream_output,compact_solver_projection=compact_solver_rows,full_actual_source_journal_retained_by_SHA_reference=not link_source_rows,original_source_exposure_subset_streamed=source_records is not None,source_subset_native_pts=sorted({int(q['native_pts'])for q in source_records})if source_records is not None else None,statistics_input_kind='single_closed_whole_exposure_receipt'if single_exposure_receipt else'group_registry',original_streaming_source_records_ref=source_records_ref)
 write_json(out/'manifest.json',report);print(json.dumps(report));return report
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--project',required=True);p.add_argument('--run',required=True);p.add_argument('--source-gate',required=True);p.add_argument('--statistics',required=True);p.add_argument('--out',required=True);a=p.parse_args();promote(a.project,a.run,a.source_gate,a.statistics,a.out)
