"""Admit finite actual ink/target sources under exact same-exposure covariance.

This gate does not reweight a group. All admitted members of a mixed exposure
must later be consumed once using the exact parent-covariance marginal.
"""
from pathlib import Path
import argparse,json,os,shutil
import numpy as np
from .io_utils import read_json,read_jsonl,write_json,write_jsonl,file_sha256,object_sha256
from .measurement_contract import EventMeasurementContract
from .role_guard import FrameRoleGuard

class SourceVerifier:
 def __init__(self,project,run):
  self.run=Path(run);self.data=next(p for p in self.run.parents if p.name=='RinkSpace_Data');self.contract=EventMeasurementContract(project,'ankara_2026');self.guard=FrameRoleGuard(self.run/'roles/frame_roles.json');self.hashes={};self.docs={}
 def verify(self,ref):
  p=Path(ref.get('path',ref.get('file')))
  if not p.is_absolute():p=self.data/p
  key=str(p.resolve())
  if key not in self.hashes:self.hashes[key]=file_sha256(p)
  if self.hashes[key]!=ref['sha256']:raise ValueError('Changed actual source artifact '+key)
  return p
 def doc(self,ref):
  p=self.verify(ref);key=str(p.resolve())
  if key not in self.docs:self.docs[key]=read_json(p)
  return self.docs[key]
 def source(self,row):
  c=self.contract;rid=row['source_row_id'];pts=int(row['native_pts']);self.guard.assert_training_pixels([pts],shot_id=row['shot_id']);cert=row['measurement_contract_certificate'];op=row['measurement_operator_resolved'];expected=c.expected(row['target_id'],row.get('category'))
  ch=object_sha256(dict(native_pts=pts,current_uv=row['current_uv'],source_row_id=rid,measurement_operator=op));sh=object_sha256(dict(native_pts=pts,source_uv=row['source_uv'],source_row_id=rid))
  if expected['operator']!=op or cert['operator']!=op or cert['contract_sha256']!=c.digest or cert['source_row_id']!=rid or cert['native_pts']!=pts or cert['current_measurement_sha256']!=ch or cert['joint_material_id']!=row['joint_material_id']:raise ValueError('Existing physical definition/certificate mismatch')
  if not cert['actual_RGB_material_identity_verified']or not cert['operator_selected_correct_material_transition']or cert['pixels_from_model']or row.get('pixels_from_model',False):raise ValueError('Actual finite material identity not verified')
  for ref in cert['source_evidence']:self.verify(ref)
  self.verify(dict(path=row['native_png'],sha256=row['native_png_sha256']))
  receipt=row['source_measurement_audit']['source_material_receipt'];frozen=receipt['frozen_point_operator'];actual=receipt['actual_point_reading'];q=np.asarray(frozen['navigation_seed_uv'])+np.asarray(frozen['normal'])*actual['normal_offset_px']
  if not actual['numeric_valid']or actual['synthetic_window_boundary_count']or np.max(np.abs(q-np.asarray(row['current_uv'])))>1e-9 or actual['current_uv']!=row['current_uv']or frozen['normal']!=row['normal']:raise ValueError('Source mean is not frozen actual finite profile')
  if row.get('point_correspondence')or row.get('entity_station_fixed'):raise ValueError('This gate admits ordinary finite curves only')
  if op=='material_interface' and row['target_id'].startswith('ice.R3:'):
   review=self.doc(row['source_identity_review']);layer=row['target_id'].split(':',1)[1];ld=review['actual_material_layer_definitions'][layer]
   if frozen['source_measurement_audit']['profile']!={'channel':0,'polarity':ld['luminance_polarity']}or receipt['layer_role']!=layer:raise ValueError('Wrong declared target material layer polarity')
   if review['geometry_model_radius_or_ratio_used_as_observed_pixels']or not row['finite_physical_identity_source_verified']:raise ValueError('World geometry is not observed pixel evidence')
   if row['shot_id']not in ('s02','s04'):raise ValueError('Only actually resolved R3 main/secondary instances are admitted')
   candidates=[self.doc(ref)for ref in cert['source_evidence']if str(ref['path']).endswith('.json')and '/frame_sources/'in str(ref['path']).replace('\\','/')]
   proof=next(p for p in candidates if 'qualified_local_material_rows'in p);original=next(q for q in proof['qualified_local_material_rows']if q['source_row_id']==rid)
   for k in ('native_pts','shot_id','source_uv','current_uv','normal','target_id'):
    if original[k]!=row[k]:raise ValueError('R3 actual source profile changed '+k)
   if original['source_material_receipt']['frozen_point_operator']!=frozen or original['source_material_receipt']['actual_point_reading']!=actual:raise ValueError('R3 frozen operator changed')
   self.verify(proof['current_mask']);self.verify(receipt['actual_five_profile_channels_reference'])
  elif op=='ink_boundary' and row.get('source_generation')=='Ad4_R_current_source_v2':
   if row['target_id']!='Ad4.print.word_ANKARA_R_arc1'or row['joint_material_index']!=73 or row['shot_id']!='s09' or not row['same_physical_s01_material_identity']:raise ValueError('Unqualified new finite R source binding')
   review=self.doc(row['same_physical_material_crosswalk'])
   if pts not in review['accepted_actual_R_exposures']or review['model_projection_used_to_create_UV']or review['model_ratio_used_for_source_selection']or review['new_geometry_or_shape_parameters']:raise ValueError('Different actual R material instance/source generation')
   for k in ('source_context_identity','source_segment_audit','actual_current_visual_review'):self.verify(review[k])
   proof=self.doc(dict(path=row['source_packet'],sha256=row['source_packet_sha256']));original=next(q for q in proof['qualified_local_material_rows']if q['source_row_id']==rid)
   for k in ('native_pts','shot_id','source_uv','current_uv','normal','target_id','joint_material_index'):
    if original[k]!=row[k]:raise ValueError('Actual finite R source changed '+k)
   if original['source_measurement_audit']['source_material_receipt']!=receipt:raise ValueError('Actual finite R operator changed')
   self.verify(proof['current_mask'])
   for ref in proof['code_dependencies']:self.verify(ref)
   if receipt['operator_version']!='finite_white_ink_purple_substrate_signed_cubic_v1'or frozen['source_measurement_audit']['profile']['channel']!=0:raise ValueError('R material must use the frozen signed white/purple channel')
   palette=receipt['actual_material_palette'];white=np.asarray(palette['white_channels']);purple=np.asarray(palette['purple_channels']);pol=frozen['source_measurement_audit']['profile']['polarity']
   if pol!=palette['white_side_normal_sign']or white[0]-purple[0]<12 or max(abs(white[1:]))>30 or purple[1]<12 or purple[2]<-2:raise ValueError('Actual R white/purple ordered transition not established')
   identity=receipt['actual_finite_material_certificate']
   if identity['source_current_exposure']!=pts or not identity['source_current_local_R_identity_actual_RGB_review']or not identity['current_source_board_object_enclosed']or identity['original_finite_template_source_edge_0based']not in(2,3,8,9,10,11):raise ValueError('Current R finite branch/object not established')
   anchor=identity.get('same_current_qualified_Ad4_material_substrate_anchor')
   if anchor:
    if anchor['method']!='same_current_exposure_actual_connected_purple_substrate_into_already_qualified_Ad4_N_or_K'or not anchor['actual_source_RGB_only']or len(anchor['current_anchor_rows'])<3:raise ValueError('Current R substrate anchor invalid')
    for ar in anchor['current_anchor_rows']:
     if ar['joint_material_index']not in(68,70):raise ValueError('R object anchor is not actual existing Ad4 N/K')
     # source_packet in the original N/K row is its historical navigation
     # packet, not its current cubic mean proof. Resolve the accepted same-
     # exposure row and reuse the complete existing ink-source verifier.
     cachekey='__actual_Ad4_current_qualified_geometry__'
     if cachekey not in self.docs:
      ap=self.run/'ad4_source_crosswalk_v2/geometry_rows.jsonl.gz';self.hashes[str(ap.resolve())]=file_sha256(ap);self.docs[cachekey]={q['source_row_id']:q for q in read_jsonl(ap)}
     qa=self.docs[cachekey].get(ar['source_row_id'])
     if qa is None or qa['native_pts']!=pts or qa['current_uv']!=ar['current_uv']or qa['joint_material_index']!=ar['joint_material_index']or not qa['source_geometry_eligible']:raise ValueError('Actual qualified N/K anchor is not the same current source')
     checked='__verified_actual_anchor__'+ar['source_row_id']
     if checked not in self.docs:self.docs[checked]=self.source(qa)
   elif not {'board.top','board.contact'}.issubset({q['target_id']for q in identity['source_scope_current_board_outer_readouts']}):raise ValueError('Actual current R board object scope missing')
  elif op=='ink_boundary':
   review=self.doc(row['same_physical_material_crosswalk']);uid=row['physical_component_uid'];identity=review['existing_source_material_indices'][uid]
   if identity['target_id']!=row['target_id']or identity['index']!=row['joint_material_index']or not row['same_physical_s01_material_identity']:raise ValueError('Different actual print/glyph source identity')
   proof=self.doc(row['actual_current_print_frame_proof']);original=next(q for q in proof['qualified_source_rows']if q['source_row_id']==rid)
   for k in ('native_pts','shot_id','source_uv','current_uv','normal'):
    if original[k]!=row[k]:raise ValueError('Actual ink profile changed '+k)
   if original['source_measurement_audit']['source_material_receipt']!=receipt:raise ValueError('Actual ink source operator differs')
   scope=self.doc(row['source_object_scope_reaudit']);fact=next(q for q in scope['actual_source_scope_facts']if q['component_uid']==uid)
   if not fact['actual_board_material_scope_passed']or fact['previous_global_object_scope_inherited']or scope['old_global_color_object_certificate_inherited']or scope['actual_ink_mean_source_UV_and_operator_changed']:raise ValueError('Old recalled global body certificate cannot establish board range')
   board=self.doc(fact['source_board_frame_proof'])
   if board['native_pts']!=pts or board['shot_id']!=row['shot_id']or board['decoded_native_BGR_sha256']!=row['decoded_native_BGR_sha256']:raise ValueError('Print body context is another exposure')
   if not any(q['material_role']=='board.top'for q in board['source_definition_certificates'])or not any(q['material_role']=='board.contact'for q in board['source_definition_certificates']):raise ValueError('Actual own cap/foot context absent')
  else:raise ValueError('Unsupported finite source operator/material; no generic edge fallback')
  return dict(source_measurement_sha256=sh,current_measurement_sha256=ch,source_profile_query_uv=frozen['navigation_seed_uv'],source_profile_normal=frozen['normal'],source_profile_current_alpha=actual['normal_offset_px'],source_profile_alpha_bounds=[-3.,3.],current_RGB_material_source_verified=True)
 def finish(self):
  for p,sha in self.hashes.items():
   if file_sha256(p)!=sha:raise ValueError('Actual source changed during promotion')
  self.contract.verify_unchanged()

def promote(project,run,source_gate,statistics,outdir):
 source=Path(source_gate);statsfile=Path(statistics);out=Path(outdir)
 if out.exists():raise ValueError('New immutable finite admission generation required')
 v=SourceVerifier(project,run);registry=read_json(statsfile);members={q['source_row_id']:q for q in read_jsonl(source/'promotion_decisions.jsonl.gz')};lookup={};decisions=[];solver=[];receipts=[];rawcount=0;physical=0
 for gid,g in registry.items():
  for rid in g['row_ids']:
   if rid in lookup:raise ValueError('Actual readout duplicated across covariance groups')
   lookup[rid]=gid
 for row in read_jsonl(source/'source_rows.jsonl.gz'):
  rawcount+=1;rid=row['source_row_id'];d=members[rid];gid=lookup.get(rid)
  if not d.get('source_geometry_eligible')or not d.get('physical_identity_passed')or row.get('actual_source_readout_alias_of'):
   decisions.append(dict(d,solver_eligible=False,numeric_statistics_ready=False,statistical_objective_eligible=False));continue
  physical+=1
  if gid is None:
   decisions.append(dict(d,solver_eligible=False,numeric_statistics_ready=False,statistical_objective_eligible=False,statistical_pending_reason='same_mean_whole_exposure_covariance_not_closed'));continue
  sourceproof=v.source(row);g=registry[gid];pts=int(row['native_pts']);nr=g['native_receipt']
  if g['statistic_kind']!='native_joint_readout'or not g['numeric_whitening_verified']or g.get('actual_mean_replay_max_px',g.get('actual_base_replay_max_px',1.))>1e-9 or g.get('float32_channel_pipeline_error',0)>0 or g.get('physical_total_error_calibrated',g.get('conditional_model_is_total_error_calibration',False)):raise ValueError('Conditional exact native readout covariance invalid')
  if g['source_measurements_sha256'][rid]!=sourceproof['source_measurement_sha256']or g['current_measurements_sha256'][rid]!=sourceproof['current_measurement_sha256']:raise ValueError('Finite source and W current/source hashes differ')
  if int(nr['native_pts'])!=pts or nr['shot_id']!=row['shot_id']or nr['decoded_native_BGR_sha256']!=row['decoded_native_BGR_sha256']or nr.get('native_RGB_resampled',False)or not nr['training_role_verified']:raise ValueError('Finite readout and covariance are different exposures')
  if row.get('native_RGB_sha256')and nr['decoded_native_RGB_sha256']!=row['native_RGB_sha256']:raise ValueError('Finite source RGB hash differs')
  for k in ('W','frozen_operator','replay_arrays'):v.verify(g[k])
  receipt=dict(schema='rinkfit.actual_finite_material_conditional_admission.v1',source_row_id=rid,native_pts=pts,source_identity_manifest=dict(path=str(source/'manifest.json'),sha256=file_sha256(source/'manifest.json')),source_membership_sha256=object_sha256(d),measurement_contract_sha256=v.contract.digest,measurement_contract_certificate_sha256=object_sha256(row['measurement_contract_certificate']),statistical_group_id=gid,parent_statistics_sha256=object_sha256(g),statistic_kind='native_joint_readout',physical_total_error_calibrated=False,exact_full_exposure_covariance_marginal_required=True,source_uv_current_uv_or_W_changed=False,**sourceproof);receipts.append(receipt)
  flags=dict(solver_eligible=True,numeric_statistics_ready=True,statistical_objective_eligible=True,statistical_group_id=gid,statistic_kind='native_joint_readout',statistics_use='conditional_noise_objective',receipt_sha256=object_sha256(receipt));decisions.append(dict(d,**flags));solver.append(dict(row,**flags,**sourceproof,conditional_model=g['conditional_model'],physical_total_error_calibrated=False,exact_full_exposure_covariance_marginal_required=True))
 v.finish();out.mkdir(parents=True)
 try:os.link(source/'source_rows.jsonl.gz',out/'source_rows.jsonl.gz')
 except OSError:shutil.copyfile(source/'source_rows.jsonl.gz',out/'source_rows.jsonl.gz')
 write_jsonl(out/'solver_rows.jsonl.gz',solver);write_jsonl(out/'promotion_decisions.jsonl.gz',decisions);write_jsonl(out/'receipts.jsonl.gz',receipts);report=dict(schema='rinkfit.actual_finite_material_conditional_admission.v1',raw_rows=rawcount,source_identity_accepted_rows=physical,statistical_objective_eligible_rows=len(solver),all_raw_rows_accounted=len(decisions)==rawcount,source_membership=dict(path=str(source/'manifest.json'),sha256=file_sha256(source/'manifest.json')),statistics_registry=dict(path=str(statsfile),sha256=file_sha256(statsfile)),measurement_contract_sha256=v.contract.digest,statistic_kind='native_joint_readout',physical_total_error_calibrated=False,mixed_exposure_members_consumed_once_jointly=True,exact_full_parent_covariance_marginal_required=True,duplicate_geometry_I_and_W_for_same_readout_allowed=False);write_json(out/'manifest.json',report);print(json.dumps(report));return report

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--project',required=True);p.add_argument('--run',required=True);p.add_argument('--source-gate',required=True);p.add_argument('--statistics',required=True);p.add_argument('--out',required=True);a=p.parse_args();promote(a.project,a.run,a.source_gate,a.statistics,a.out)
