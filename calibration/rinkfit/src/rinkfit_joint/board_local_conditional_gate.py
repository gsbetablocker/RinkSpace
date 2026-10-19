"""Bind actual local-board membership to same-mean conditional native readout W."""
from pathlib import Path
import argparse,json
from .io_utils import read_json,read_jsonl,write_json,write_jsonl,file_sha256,object_sha256
from .measurement_contract import EventMeasurementContract
from .role_guard import FrameRoleGuard

def promote(project,run,source_gate,statistics,outdir):
 run=Path(run);source=Path(source_gate);statsfile=Path(statistics);out=Path(outdir);data=next(p for p in run.parents if p.name=='RinkSpace_Data');contract=EventMeasurementContract(project,'ankara_2026');guard=FrameRoleGuard(run/'roles/frame_roles.json');registry=read_json(statsfile);member={d['source_row_id']:d for d in read_jsonl(source/'promotion_decisions.jsonl.gz')};raw=list(read_jsonl(source/'source_rows.jsonl.gz'));stats_by_rid={rid:gid for gid,s in registry.items()for rid in s['row_ids']};cache={};solver=[];decisions=[];receipts=[]
 if out.exists():raise ValueError('Immutable new conditional gate directory required')
 out.mkdir(parents=True)
 def verify(ref):
  path=Path(ref.get('path',ref.get('file')))
  if not path.is_absolute():path=data/path
  key=str(path.resolve())
  if key not in cache:cache[key]=file_sha256(path)
  if cache[key]!=ref['sha256']:raise ValueError('Source/numeric artifact changed: '+key)
  return path
 for row in raw:
  rid=row['source_row_id'];d=member[rid];errors=[]
  if not d['source_geometry_eligible']:
   decisions.append(dict(d,statistical_objective_eligible=False,numeric_statistics_ready=False,solver_eligible=False,retained_actual_source_rejection=True));continue
  pts=int(row['native_pts']);guard.assert_training_pixels([pts],shot_id=row['shot_id']);cert=row['measurement_contract_certificate'];expected=contract.expected(row['target_id'])
  if cert['contract_sha256']!=contract.digest or cert['operator']!=expected['operator']or cert['source_row_id']!=rid:raise ValueError('Actual measured definition is not user boundary')
  if not cert['actual_RGB_material_identity_verified'] or not cert['operator_selected_correct_material_transition'] or cert['pixels_from_model']:raise ValueError('Current source material not certified')
  for ref in cert['source_evidence']:verify(ref)
  verify(row['source_membership_recheck']['counterexample']);counter=read_json(verify(row['source_membership_recheck']['counterexample']))
  if rid in counter['affected_source_row_ids']:raise ValueError('Actually refuted source cannot regain membership from matching numeric hash')
  proofpath=Path(row['source_packet']);verify(dict(path=str(proofpath),sha256=row['source_packet_sha256']));proof=read_json(proofpath);native=verify(dict(path=proof['native_png'],sha256=proof['native_png_sha256']));record=next(q for q in proof['qualified_source_rows']if q['source_row_id']==rid)
  for k in ('native_pts','source_uv','current_uv','normal','target_id','joint_material_id','measurement_operator_resolved'):
   if record[k]!=row[k]:raise ValueError('Actual native material row differs from physical proof: '+k)
  context=proof['current_board_instance_context'];verify(context['source_review']);verify(proof['current_mask']);a=row['source_measurement_audit']
  if not a['actual_outer_material_order_verified']or not a['reflection_excluded']or a['internal_interface_used']or a.get('global_color_rule_used_as_object_identity',True):raise ValueError('Wrong current material boundary or object scope')
  gid=stats_by_rid.get(rid)
  if gid is None:
   decisions.append(dict(d,solver_eligible=False,numeric_statistics_ready=False,statistical_objective_eligible=False,source_physical_membership_retained_from_parent=True,statistical_pending_reason='same_mean_whole_exposure_group_not_in_this_statistics_generation'));continue
  s=registry[gid]
  if s['statistic_kind']!='native_joint_readout'or not s['numeric_whitening_verified']or s['actual_mean_replay_max_px']>1e-9 or s['float32_channel_pipeline_error']>0 or s.get('conditional_model_is_total_error_calibration',s.get('physical_total_error_calibrated'))is not False:raise ValueError('Conditional native readout statistics invalid')
  sh=object_sha256(dict(native_pts=pts,source_uv=row['source_uv'],source_row_id=rid));ch=object_sha256(dict(native_pts=pts,current_uv=row['current_uv'],source_row_id=rid,measurement_operator=row['measurement_operator_resolved']))
  if s['source_measurements_sha256'][rid]!=sh or s['current_measurements_sha256'][rid]!=ch or cert['current_measurement_sha256']!=ch:raise ValueError('Conditional W and actual current/source means differ')
  for k in ('W','frozen_operator','replay_arrays'):verify(s[k])
  native_receipt=s['native_receipt']
  if native_receipt['native_pts']!=pts or native_receipt['shot_id']!=row['shot_id'] or native_receipt['decoded_native_BGR_sha256']!=row['decoded_native_BGR_sha256'] or native_receipt['decoded_native_RGB_sha256']!=proof['actual_native_RGB_sha256'] or native_receipt['native_RGB_resampled'] or not native_receipt['training_role_verified']:raise ValueError('Native current source statistic exposure differs')
  receipt=dict(schema='rinkfit.actual_local_board_conditional_source_promotion.v1',source_row_id=rid,native_pts=pts,source_measurement_sha256=sh,current_measurement_sha256=ch,source_packet_sha256=row['source_packet_sha256'],source_membership_generation='board_local_source_v5',source_membership_definition_sha256=object_sha256(d),measurement_contract_sha256=contract.digest,measurement_contract_certificate_sha256=object_sha256(cert),source_physical_identity_verified=True,operator_verified=True,statistical_group_id=gid,statistics_definition_sha256=object_sha256(s),statistic_kind='native_joint_readout',conditional_model=s['conditional_model'],physical_total_error_calibrated=False,parent_full_covariance_exact_marginal_required=True,raw_UV_or_current_UV_changed=False,old_recalled_source_W_reused=False)
  receipts.append(receipt);decisions.append(dict(d,solver_eligible=True,numeric_statistics_ready=True,statistical_objective_eligible=True,statistical_group_id=gid,statistic_kind='native_joint_readout',statistics_use='conditional_noise_objective',receipt_sha256=object_sha256(receipt)));solver.append(dict(row,solver_eligible=True,numeric_statistics_ready=True,statistical_objective_eligible=True,statistical_group_id=gid,statistic_kind='native_joint_readout',statistics_use='conditional_noise_objective',conditional_model=s['conditional_model'],physical_total_error_calibrated=False,parent_full_covariance_exact_marginal_required=True,receipt_sha256=object_sha256(receipt)))
 for path,sha in cache.items():
  if file_sha256(path)!=sha:raise ValueError('Immutable source changed during conditional admission')
 contract.verify_unchanged();write_jsonl(out/'solver_rows.jsonl.gz',solver);write_jsonl(out/'source_rows.jsonl.gz',raw);write_jsonl(out/'promotion_decisions.jsonl.gz',decisions);write_jsonl(out/'receipts.jsonl.gz',receipts);report=dict(schema='rinkfit.actual_local_board_conditional_admission.v1',raw_rows=len(raw),source_identity_accepted_rows=sum(bool(d.get('source_geometry_eligible'))for d in member.values()),statistical_objective_eligible_rows=len(solver),statistics_pending_source_rows=sum(bool(d.get('source_geometry_eligible'))for d in member.values())-len(solver),physical_source_excluded_rows=sum(not bool(d.get('source_geometry_eligible'))for d in member.values()),all_rows_accounted=len(decisions)==len(raw),source_membership=dict(path=str(source/'manifest.json'),sha256=file_sha256(source/'manifest.json')),statistics_registry=dict(path=str(statsfile),sha256=file_sha256(statsfile)),measurement_contract_sha256=contract.digest,statistic_kind='native_joint_readout',statistics_use='conditional_noise_objective',physical_total_error_calibrated=False,exact_full_parent_covariance_marginal_required=True,duplicate_geometryI_and_conditionalW_consumption_allowed=False,raw_UV_or_current_UV_or_W_changed=False);write_json(out/'manifest.json',report);print(json.dumps(report));return report
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--project',required=True);p.add_argument('--run',required=True);p.add_argument('--source-gate',required=True);p.add_argument('--statistics',required=True);p.add_argument('--out',required=True);a=p.parse_args();promote(a.project,a.run,a.source_gate,a.statistics,a.out)
