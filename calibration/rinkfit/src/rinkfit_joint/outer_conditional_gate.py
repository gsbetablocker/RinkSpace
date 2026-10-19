"""Promote only same-mean actual-outer groups with explicit conditional W."""
from pathlib import Path
from collections import Counter
import argparse,json
from .io_utils import read_json,read_jsonl,write_json,write_jsonl,file_sha256,object_sha256
from .materials import PromotionArtifactVerifier,apply_promotion
from .role_guard import FrameRoleGuard

def promote(run,geometry_gate,statistics,identity,material_map,outdir):
 run=Path(run);geometry_gate=Path(geometry_gate);out=Path(outdir);data=next(p for p in run.parents if p.name=='RinkSpace_Data');scope=read_json(run/'programme_scope.json');guard=FrameRoleGuard(run/'roles/frame_roles.json');stats=read_json(statistics);wanted={rid:gid for gid,s in stats.items() for rid in s['row_ids']};mats={m['id']:m for m in read_json(Path(identity)/'joint_material_registry.json')['materials']};mapping=read_json(material_map)['semantic_to_material'];verifier=PromotionArtifactVerifier(data)
 if out.exists():raise ValueError('New immutable conditional admission required')
 if len(wanted)!=sum(len(s['row_ids']) for s in stats.values()):raise ValueError('Actual source reused across conditional groups')
 rows={r['source_row_id']:r for r in read_jsonl(geometry_gate/'geometry_rows.jsonl.gz') if r['source_row_id'] in wanted}
 if set(rows)!=set(wanted):raise ValueError('Conditional W includes sources not through the actual RGB/physical gate')
 for gid,s in stats.items():
  if s.get('statistic_kind')!='native_joint_readout' or s.get('numeric_whitening_verified') is not True or s.get('actual_mean_replay_max_px',1)>1e-9:raise ValueError('Wrong or unverified same-mean conditional source model')
  if s.get('conditional_model_is_total_error_calibration') is not False:raise ValueError('Total noise calibration must not be asserted')
  for key in ('W','frozen_operator','replay_arrays'):
   ref=s[key]
   if file_sha256(ref['file'])!=ref['sha256']:raise ValueError('Immutable numeric source artifact changed')
 ledgers={};frames={}
 for n in ('dense_s03_v1','dense_remaining_v1'):
  path=run/n/'frame_ledger.jsonl.gz';sha=file_sha256(path)
  for fr in read_jsonl(path):frames[int(fr['native_pts'])]=fr;ledgers[int(fr['native_pts'])]=(path,sha)
 out.mkdir(parents=True);policy=dict(schema='rinkfit.actual_same_mean_conditional_outer_admission_policy.v1',geometry_gate=dict(path=str(geometry_gate/'manifest.json'),sha256=file_sha256(geometry_gate/'manifest.json')),statistics=dict(path=str(statistics),sha256=file_sha256(statistics)),statistics_scope='conditional encoded-RGB measurement, not calibrated full physical error',source_selection_or_RGB_remeasurement_repeated=False,old_bilinear_W_reused=False,geometry_and_W_same_source_must_not_be_double_counted=True,old_board_and_inner_sources_remain_withdrawn=True);write_json(out/'policy.json',policy);policy_sha=file_sha256(out/'policy.json')
 def art(path,role,sha=None):
  path=Path(path).resolve();return dict(path=str(path.relative_to(data)),role=role,sha256=sha or verifier.digest(path))
 profile_art={}
 for ref in read_json(geometry_gate/'manifest.json')['source_generations']:
  m=read_json(ref['path'])
  if file_sha256(ref['path'])!=ref['sha256']:raise ValueError('Completed source generation changed')
  for ch in m['chunks']:
   ff={Path(q['path']).name:q for q in ch['files']};profile_art[str(Path(ff['rows.jsonl.gz']['path']))]=[art(ff['profiles.npz']['path'],'current_native_profiles',ff['profiles.npz']['sha256']),art(ff['profile_index.json']['path'],'native_profile_index',ff['profile_index.json']['sha256'])]
 locked={};native={};receipts=[];decisions=[];solver=[];counts=Counter()
 for rid,row in rows.items():
  pts=int(row['native_pts']);sid=row['shot_id'];guard.assert_training_pixels([pts],shot_id=sid);gid=wanted[rid];s=stats[gid];material=mats[row['joint_material_id']];certificate=row['measurement_contract_certificate'];packet=Path(row['source_packet']);ledger,ledger_sha=ledgers[pts];fr=frames[pts];native[f"{scope['source']['sha256']}:{pts}"]=dict(source_video_sha256=scope['source']['sha256'],native_pts=pts,decode_sequence_verified=True,decoded_native_BGR_sha256=fr['decoded_native_BGR_sha256'],native_decode_receipt_sha256=ledger_sha,native_decode_row_sha256=object_sha256(fr));lock=dict(source_row_id=rid,source_family_id=row['source_family_id'],source_packet_sha256=row['source_packet_sha256'],native_pts=pts,fit_role='development',assignment_rule='whole actual source exposure/family, prelocked development role');locked[rid]=lock
  arts=[art(scope['source']['path'],'native_video',scope['source']['sha256']),art(out/'policy.json','source_policy',policy_sha),art(ledger,'native_decode_receipt',ledger_sha),art(packet,'source_measurement_audit',row['source_packet_sha256']),*profile_art[str(packet)],*certificate['source_evidence'],art(statistics,'conditional_statistics_registry'),art(s['W']['file'],'conditional_whitening',s['W']['sha256']),art(s['frozen_operator']['file'],'actual_frozen_source_operator',s['frozen_operator']['sha256']),art(s['replay_arrays']['file'],'shared_exposure_nonlinear_RGB_replay',s['replay_arrays']['sha256'])]
  receipt=dict(schema='rinkfit.actual_outer_same_mean_conditional_promotion_receipt.v1',source_row_id=rid,source_packet_sha256=row['source_packet_sha256'],video_sha256=scope['source']['sha256'],policy_sha256=policy_sha,joint_material_id=row['joint_material_id'],source_measurement_sha256=object_sha256(dict(native_pts=pts,source_uv=row['source_uv'],source_row_id=rid)),current_measurement_sha256=object_sha256(dict(native_pts=pts,current_uv=row['current_uv'],source_row_id=rid,measurement_operator=row['measurement_operator_resolved'])),native_decode_row_sha256=object_sha256(fr),evidence_basis='current_native_source_remeasurement',native_pixel_review_passed=True,physical_identity_passed=True,operator_passed=True,source_lineage_complete=True,numeric_statistics_ready=True,manual_authority_preserved=True,material_definition_sha256=object_sha256(material),fit_role='development',locked_role_sha256=object_sha256(lock),statistical_group_id=gid,statistics_definition_sha256=object_sha256(s),evidence_artifacts=arts,measurement_contract_sha256=certificate['contract_sha256'],measurement_contract_certificate_sha256=object_sha256(certificate),statistic_kind='native_joint_readout',conditional_model=s['conditional_model'],physical_total_error_calibrated=False,source_uv_is_first_bilinear_actual_measurement=True,new_current_mean_is_continuous_cubic=True)
  d=apply_promotion(row,{},receipt,video_sha256=scope['source']['sha256'],policy_sha256=policy_sha,material_registry=mats,statistics_registry=stats,locked_roles=locked,evidence_root=data,decoded_frame_registry=native,artifact_verifier=verifier);decisions.append(dict(source_row_id=rid,**d));receipts.append(receipt);counts.update(d['promotion_errors'] or ['conditional_source_admitted'])
  if d['solver_eligible']:solver.append({**row,'solver_eligible':True,'numeric_statistics_ready':True,'statistical_objective_eligible':True,'statistical_group_id':gid,'statistic_kind':'native_joint_readout','receipt_sha256':d['receipt_sha256'],'locked_role_sha256':object_sha256(lock),'conditional_model':s['conditional_model'],'physical_total_error_calibrated':False})
 verifier.verify_unchanged();write_jsonl(out/'solver_rows.jsonl.gz',solver);write_jsonl(out/'promotion_decisions.jsonl.gz',decisions);write_jsonl(out/'receipts.jsonl.gz',receipts);write_jsonl(out/'source_rows.jsonl.gz',rows.values());result=dict(schema='rinkfit.actual_outer_same_mean_conditional_admission.v1',rows=len(rows),solver_eligible_rows=len(solver),groups=len(stats),counts=dict(counts),all_rows_accounted=True,statistics_registry=dict(path=str(statistics),sha256=file_sha256(statistics)),geometry_source_gate=dict(path=str(geometry_gate/'manifest.json'),sha256=file_sha256(geometry_gate/'manifest.json')),measurement_contract_sha256=verifier.measurement_contract.digest,statistic_kind='native_joint_readout',conditional_source_noise_only=True,physical_total_error_calibrated=False,duplicate_geometry_and_W_consumption_allowed=False,old_bilinear_W_reused=False);write_json(out/'manifest.json',result);print(json.dumps(result));return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--geometry-gate',required=True);p.add_argument('--statistics',required=True);p.add_argument('--identity',required=True);p.add_argument('--material-map',required=True);p.add_argument('--out',required=True);a=p.parse_args();promote(a.run,a.geometry_gate,a.statistics,a.identity,a.material_map,a.out)
