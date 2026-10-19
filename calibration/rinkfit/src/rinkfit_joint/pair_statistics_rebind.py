"""Rebind an already physically signed pair to a new mixed-exposure covariance."""
from pathlib import Path
import json,argparse
import numpy as np
from .io_utils import read_json,read_jsonl,write_json,write_jsonl,file_sha256,object_sha256
from .finite_conditional_gate import SourceVerifier

def promote(project,run,identity,statistics,outdir):
 run=Path(run);identity=Path(identity);statsfile=Path(statistics);out=Path(outdir)
 if out.exists():raise ValueError('New immutable pair statistics binding required')
 v=SourceVerifier(project,run);registry=read_json(statsfile);items=read_json(identity/'approved_pairs.json');oldreceipts={q['source_row_id']:q for q in read_jsonl(identity/'receipts.jsonl.gz')};oldfactors={q['source_row_id']:q for q in read_jsonl(identity/'solver_rows.jsonl.gz')};manual={q['source_row_id']:q for q in read_jsonl(run/'manual_source_geometry_v2/geometry_rows.jsonl.gz')};lookup={rid:gid for gid,g in registry.items()for rid in g['row_ids']};rows=[];approved=[];receipts=[]
 policy=read_json(identity/'policy.json');v.verify(policy['finite_identity'])
 for item in items:
  p=item['pair'];rid=p['source_row_id'];pts=int(p['native_pts']);v.guard.assert_training_pixels([pts],shot_id=p['shot_id']);old=oldreceipts[rid];target=item['actual_target_id']
  if object_sha256(old)!=item['source_admission_receipt_sha256']or not old['physical_identity_passed']or not old['source_local_two_flank_operator_passed']or old['measurement_contract_sha256']!=v.contract.digest:raise ValueError('Original physical pair admission changed')
  if v.contract.expected(target,'ice')['operator']!='paint_center':raise ValueError('Pair source is not a declared finite-width line')
  h=object_sha256(dict(native_pts=pts,source_row_id=rid,actual_flank_uv=p['actual_flank_uv'],measurement_operator='finite_width_paint_material_flank_pair'))
  if h!=p['source_mean_sha256']or h!=old['actual_4D_source_mean_sha256']or h!=item['actual_4D_source_mean_sha256']:raise ValueError('Original actual paired UV changed')
  proof=v.doc(item['actual_finite_source_proof']);record=next(q for q in proof['actual_paint_identity_receipts']if q['decision']['source_row_id']==rid)
  if record['actual_current_pair']!=p or record['world_geometry_or_model_projection_used']or not record['actual_unbroken_current_RGB_material_path']or any(q['actual_current_material_reading'].get('current_uv')is None for q in record['actual_unbroken_current_RGB_material_path']):raise ValueError('Original actual named stroke source identity differs')
  v.verify(dict(path=proof['native_png'],sha256=proof['native_png_sha256']))
  if proof['native_png_sha256']!=p['source_refs']['native_png_sha256']or proof['source_pixels_from_model']or proof['manual_point_coordinates_changed']:raise ValueError('Original source pixels or manual authority changed')
  for a in old['manual_source_anchors']:
   m=manual[a['source_row_id']]
   if m['source_uv']!=a['source_uv']or m['receipt_sha256']!=a['receipt_sha256']or m['target_id']!=target or m['native_pts']!=pts:raise ValueError('Original actual manual material anchor changed')
  ids=p['factor_row_ids'];gids={lookup.get(k)for k in ids}
  if len(gids)!=1 or None in gids:raise ValueError('Actual pair must share one closed mixed exposure')
  gid=next(iter(gids));g=registry[gid];nr=g['native_receipt']
  if g['statistic_kind']!='native_joint_readout'or not g['numeric_whitening_verified']or g['actual_mean_replay_max_px']>1e-9 or g['physical_total_error_calibrated']or int(nr['native_pts'])!=pts or nr['shot_id']!=p['shot_id']or not nr['training_role_verified']or nr.get('native_RGB_resampled',False):raise ValueError('Conditional mixed source statistics invalid')
  if g['current_pair_measurements_sha256'][rid]!=h:raise ValueError('Mixed source covariance binds another actual pair')
  for key in ('W','frozen_operator','replay_arrays'):v.verify(g[key])
  for side,fid in enumerate(ids):
   f=oldfactors[fid]
   if f['current_uv']!=p['actual_flank_uv'][side]or f['source_uv']!=f['current_uv']:raise ValueError('Paired original actual flank order changed')
   sh=object_sha256(dict(native_pts=pts,source_uv=f['source_uv'],source_row_id=fid));ch=object_sha256(dict(native_pts=pts,current_uv=f['current_uv'],source_row_id=fid,measurement_operator=f['measurement_operator_resolved']))
   if g['source_measurements_sha256'][fid]!=sh or g['current_measurements_sha256'][fid]!=ch:raise ValueError('Mixed covariance and actual flank current/source means differ')
   rec=p['source_material_receipt'];lo,hi=rec['frozen_flank_windows'][2][side]['normal_window_offsets'];mid=.5*(lo+hi);n=np.asarray(p['source_normal']);seed=np.asarray(rec['original_navigation_seed_uv']);q=seed+mid*n
   rows.append(dict(f,statistical_group_id=gid,conditional_model=g['conditional_model'],source_profile_query_uv=q.tolist(),source_profile_normal=n.tolist(),source_profile_alpha_bounds=[lo-mid,hi-mid],source_profile_current_alpha=float(np.dot(np.asarray(f['current_uv'])-q,n)),solver_eligible=True,numeric_statistics_ready=True,statistical_objective_eligible=True,statistics_use='conditional_noise_objective',exact_full_exposure_covariance_marginal_required=True,source_statistics_rebinding_only=True))
  receipt=dict(schema='rinkfit.existing_actual_pair_mixed_statistics_rebinding.v1',source_row_id=rid,native_pts=pts,actual_4D_source_mean_sha256=h,physical_source_admission=dict(path=str(identity/'manifest.json'),sha256=file_sha256(identity/'manifest.json')),original_physical_source_receipt_sha256=object_sha256(old),original_source_identity_proof=item['actual_finite_source_proof'],statistical_group_id=gid,parent_statistics_sha256=object_sha256(g),measurement_contract_sha256=v.contract.digest,source_identity_unchanged=True,mean_or_UV_or_frozen_source_operator_changed=False,previous_statistics_replaced_not_added=True,exact_full_exposure_covariance_marginal_required=True,physical_total_error_calibrated=False);receipts.append(receipt);approved.append(dict(item,statistical_group_id=gid,source_admission_receipt_sha256=object_sha256(receipt),source_statistics_rebinding_only=True))
 v.finish();out.mkdir(parents=True);write_json(out/'approved_pairs.json',approved);write_jsonl(out/'solver_rows.jsonl.gz',rows);write_jsonl(out/'factor_rows.jsonl.gz',rows);write_jsonl(out/'receipts.jsonl.gz',receipts);report=dict(schema='rinkfit.existing_actual_pair_mixed_statistics_rebinding.v1',approved_pairs=len(approved),approved_flank_rows=len(rows),original_physical_membership_or_actual_UV_changed=False,statistics_registry=dict(path=str(statsfile),sha256=file_sha256(statsfile)),measurement_contract_sha256=v.contract.digest,previous_same_exposure_statistics_replaced_not_added=True,exact_full_exposure_covariance_marginal_required=True,physical_total_error_calibrated=False);write_json(out/'manifest.json',report);print(json.dumps(report));return report

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--project',required=True);p.add_argument('--run',required=True);p.add_argument('--identity',required=True);p.add_argument('--statistics',required=True);p.add_argument('--out',required=True);a=p.parse_args();promote(a.project,a.run,a.identity,a.statistics,a.out)
