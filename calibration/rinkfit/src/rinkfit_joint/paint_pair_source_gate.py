"""Typed two-flank physical-source gate; identity and conditional statistics stay distinct."""
from pathlib import Path
import argparse,json
from collections import Counter
from .io_utils import read_json,read_jsonl,write_json,write_jsonl,file_sha256,object_sha256
from .role_guard import FrameRoleGuard
from .measurement_contract import EventMeasurementContract

def promote(project,run,identity,statistics,material_map,outdir):
 run=Path(run);identity=Path(identity);statistics=Path(statistics);out=Path(outdir)
 if out.exists():raise ValueError('Immutable new run directory required')
 contract=EventMeasurementContract(project,'ankara_2026');guard=FrameRoleGuard(run/'roles/frame_roles.json')
 qualified=read_json(identity/'qualified_pairs.json');original_pairs=read_json(statistics/'source_pairs.json');registry=read_json(statistics/'statistics_registry.json');mapping=read_json(material_map)['semantic_to_material']
 source_map={q['source_row_id']:q for q in original_pairs};group_map={rid:gid for gid,g in registry.items() for rid in g['source_pair_row_ids']}
 if len(source_map)!=len(original_pairs):raise ValueError('Repeated original source pair ID')
 all_decisions={r['source_row_id']:r for r in read_jsonl(identity/'identity_decisions.jsonl.gz')}
 if set(all_decisions)!=set(source_map):raise ValueError('Full source ledger is incomplete')
 manual={r['source_row_id']:r for r in read_jsonl(run/'manual_source_geometry_v2/geometry_rows.jsonl.gz')}
 selected=read_json(run/'native_selected_v2/manifest.json')
 # The exact native selected manifest is also recorded in every actual-source frame proof.
 out.mkdir(parents=True);approved=[];rows=[];receipts=[];counts=Counter();hashes={}
 def verify(ref):
  path=Path(ref.get('path',ref.get('file')));wanted=ref['sha256'];key=str(path.resolve());digest=hashes.setdefault(key,file_sha256(path))
  if digest!=wanted:raise ValueError('Changed source artifact '+key)
  return dict(path=str(path),sha256=digest)
 policy=dict(schema='rinkfit.finite_paint_pair_source_admission_policy.v1',measurement_contract_sha256=contract.digest,source_pairs=dict(path=str(statistics/'source_pairs.json'),sha256=file_sha256(statistics/'source_pairs.json')),finite_identity=dict(path=str(identity/'manifest.json'),sha256=file_sha256(identity/'manifest.json')),statistics=dict(path=str(statistics/'statistics_registry.json'),sha256=file_sha256(statistics/'statistics_registry.json')),original_midpoint_rows_consumed=False,source_binding_may_be_corrected_only_by_direct_actual_manual_material_and_unbroken_current_RGB=True,world_ideal_relations_used=False,manual_source_UV_changed=False,conditional_model_is_total_physical_error_calibration=False)
 write_json(out/'policy.json',policy)
 for item in qualified:
  pair=item['pair'];rid=pair['source_row_id'];decision=all_decisions[rid];original=source_map[rid];pts=int(pair['native_pts']);sid=pair['shot_id'];guard.assert_training_pixels([pts],shot_id=sid)
  if pair!=original:raise ValueError('Finite identity uses a different actual pair')
  if not decision['finite_identity_passed'] or decision['reasons'] or decision.get('exact_same_actual_source_readout_alias_of_first'):raise ValueError('Unqualified identity or duplicate')
  target=item['actual_manual_source_target_id'];semantic=target+'|ice|paint_center|unspecified'
  spec=contract.expected(target,'ice')
  if spec.get('operator')!='paint_center':raise ValueError('Existing definition is not width centerline')
  mat=mapping[semantic]
  if not mat['geometry_ready']:raise ValueError('Shared actual material geometry missing')
  proof=verify(item['actual_finite_source_proof']);proofdoc=read_json(proof['path'])
  if proofdoc['measurement_contract_sha256']!=contract.digest or proofdoc['source_pixels_from_model'] is not False or proofdoc['manual_point_coordinates_changed'] is not False:raise ValueError('Source proof contract mismatch')
  record=next(q for q in proofdoc['actual_paint_identity_receipts'] if q['decision']['source_row_id']==rid)
  if record['decision']!=decision or record['actual_current_pair']!=pair:raise ValueError('Identity decision differs from actual source proof')
  bridge=record['actual_unbroken_current_RGB_material_path']
  if not bridge or any(q['actual_current_material_reading'].get('current_uv') is None for q in bridge):raise ValueError('Actual source material path is broken')
  if record['world_geometry_or_model_projection_used'] is not False:raise ValueError('Physical identity cannot come from model')
  png=Path(proofdoc['native_png']);verify(dict(path=str(png),sha256=proofdoc['native_png_sha256']))
  if proofdoc['native_png_sha256']!=pair['source_refs']['native_png_sha256']:raise ValueError('Actual exposure changed')
  manual_refs=[]
  for a in item['manual_source_anchors']:
   m=manual[a['source_row_id']]
   if m['source_uv']!=a['source_uv'] or m['target_id']!=target or m['native_pts']!=pts or m['receipt_sha256']!=a['receipt_sha256']:raise ValueError('Direct manual authority changed')
   cert=m['measurement_contract_certificate']
   if cert['contract_sha256']!=contract.digest or not cert['actual_RGB_material_identity_verified'] or cert['pixels_from_model']:raise ValueError('Manual source authority unclosed')
   manual_refs.append(dict(source_row_id=m['source_row_id'],source_uv=m['source_uv'],receipt_sha256=m['receipt_sha256']))
  gid=group_map[rid];g=registry[gid]
  if g['statistic_kind']!='native_joint_readout' or not g['numeric_whitening_verified'] or g['actual_two_flank_base_replay_max_px']>1e-9 or g['physical_total_error_calibrated']:raise ValueError('Conditional source statistics invalid')
  h=object_sha256(dict(native_pts=pts,source_row_id=rid,actual_flank_uv=pair['actual_flank_uv'],measurement_operator='finite_width_paint_material_flank_pair'))
  if h!=pair['source_mean_sha256'] or g['current_pair_measurements_sha256'][rid]!=h or decision['actual_4D_source_mean_sha256']!=h:raise ValueError('Wrong 4D mean or midpoint hash substituted')
  for k in ('W','replay_arrays','frozen_operator','factor_rows'):verify(g[k])
  factors=[q for q in read_jsonl(g['factor_rows']['file']) if q['original_source_row_id']==rid]
  if [q['source_row_id']for q in factors]!=pair['factor_row_ids']:raise ValueError('Paired native flank order changed')
  for side,q in enumerate(factors):
   if q['current_uv']!=pair['actual_flank_uv'][side] or q['source_uv']!=q['current_uv']:raise ValueError('Actual native flank UV changed')
   sh=object_sha256(dict(native_pts=pts,source_uv=q['source_uv'],source_row_id=q['source_row_id']));ch=object_sha256(dict(native_pts=pts,current_uv=q['current_uv'],source_row_id=q['source_row_id'],measurement_operator=q['measurement_operator_resolved']))
   if g['source_measurements_sha256'][q['source_row_id']]!=sh or g['current_measurements_sha256'][q['source_row_id']]!=ch:raise ValueError('Conditional W and actual flank means differ')
   rows.append(dict(q,target_id=target,original_target_id=q['target_id'],joint_material_id=semantic,joint_material_index=mat['material_index'],physical_role='finite_width_paint_actual_material_flank',source_pair_id=rid,width_scope_id=pair['width_scope_id'],width_scope=pair['width_scope'],source_identity_eligible=True,solver_eligible=True,statistical_objective_eligible=True,numeric_statistics_ready=True,statistic_kind='native_joint_readout',statistics_use='conditional_noise_objective',statistical_group_id=gid,conditional_model=g['conditional_model'],physical_total_error_calibrated=False,exact_parent_covariance_marginal_required=True,measurement_contract_sha256=contract.digest,actual_finite_source_proof=proof,source_material_receipt=pair['source_material_receipt']))
  receipt=dict(schema='rinkfit.actual_finite_paint_pair_source_admission_receipt.v1',source_row_id=rid,native_pts=pts,shot_id=sid,actual_4D_source_mean_sha256=h,source_pair_hash_kind='actual_flank_uv4D_not_old_midpoint2D',original_target_id=pair['target_id'],target_id=target,joint_material_id=semantic,joint_material_index=mat['material_index'],width_scope_id=pair['width_scope_id'],actual_finite_source_proof=proof,manual_source_anchors=manual_refs,measurement_contract_sha256=contract.digest,physical_identity_passed=True,source_local_two_flank_operator_passed=True,pixels_from_model=False,parent_statistical_group_id=gid,parent_statistics_definition_sha256=object_sha256(g),exact_parent_covariance_marginal_required=True,physical_total_error_calibrated=False,source_uv_or_current_uv_changed=False)
  receipts.append(receipt);approved.append(dict(item,actual_target_id=target,joint_material_id=semantic,joint_material_index=mat['material_index'],source_identity_eligible=True,statistical_group_id=gid,actual_4D_source_mean_sha256=h,source_admission_receipt_sha256=object_sha256(receipt)));counts['actual_finite_pair_source_accepted']+=1
 accepted={q['pair']['source_row_id']for q in approved};decisions=[]
 for rid,d in all_decisions.items():
  d=dict(d,source_identity_eligible=rid in accepted,solver_eligible=rid in accepted,statistical_objective_eligible=rid in accepted,numeric_statistics_ready=rid in accepted,source_mean_changed=False)
  if rid not in accepted:counts.update(d['reasons'] or ['finite_material_identity_pending'])
  decisions.append(d)
 for path,sha in hashes.items():
  if file_sha256(path)!=sha:raise ValueError('Artifact changed during admission')
 contract.verify_unchanged();write_json(out/'approved_pairs.json',approved);write_jsonl(out/'factor_rows.jsonl.gz',rows);write_jsonl(out/'solver_rows.jsonl.gz',rows);write_jsonl(out/'promotion_decisions.jsonl.gz',decisions);write_jsonl(out/'receipts.jsonl.gz',receipts)
 report=dict(schema='rinkfit.actual_finite_paint_pair_source_admission.v1',source_pairs=len(original_pairs),source_accepted_pairs=len(approved),source_accepted_flank_rows=len(rows),all_pairs_accounted=len(decisions)==len(original_pairs),counts=dict(counts),measurement_contract_sha256=contract.digest,actual_mean_dimension_per_pair=4,statistic_kind='native_joint_readout',conditional_noise_objective=True,physical_total_error_calibrated=False,raw_midpoint_rows_consumed=False,exact_parent_covariance_marginal_required=True,source_mean_or_manual_UV_changed=False,policy_sha256=file_sha256(out/'policy.json'))
 write_json(out/'manifest.json',report);print(json.dumps(report));return report
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--project',required=True);p.add_argument('--run',required=True);p.add_argument('--identity',required=True);p.add_argument('--statistics',required=True);p.add_argument('--material-map',required=True);p.add_argument('--out',required=True);a=p.parse_args();promote(a.project,a.run,a.identity,a.statistics,a.material_map,a.out)
