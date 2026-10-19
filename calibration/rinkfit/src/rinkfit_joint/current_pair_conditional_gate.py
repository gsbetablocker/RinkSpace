"""Current finite-width C0 section sources with whole-exposure paired statistics."""
from pathlib import Path
import argparse,json,os,shutil
import numpy as np
from .io_utils import read_json,read_jsonl,write_json,write_jsonl,file_sha256,object_sha256
from .finite_conditional_gate import SourceVerifier

def promote(project,run,identity,statistics,outdir):
 run=Path(run);identity=Path(identity);statsfile=Path(statistics);out=Path(outdir)
 if out.exists():raise ValueError('New immutable paired admission required')
 v=SourceVerifier(project,run);contract=v.contract;registry=read_json(statsfile);qualified=read_json(identity/'approved_pairs.json');lookup={rid:gid for gid,g in registry.items()for rid in g['row_ids']};original={r['source_row_id']:r for r in read_jsonl(run/'C0_current_two_flank_source_v2/source_rows.jsonl.gz')};factor_map={r['source_row_id']:r for r in read_jsonl(identity/'factor_rows.jsonl.gz')};sourcepairs=[];factors=[];receipts=[];decisions=[]
 for item in qualified:
  p=item['pair'];rid=p['source_row_id'];pts=int(p['native_pts']);v.guard.assert_training_pixels([pts],shot_id=p['shot_id']);expected=contract.expected(item['actual_target_id'],'ice')
  if expected['operator']!='paint_center'or item['actual_target_id']!='ice.C0:outer'or not item['source_identity_eligible']:raise ValueError('Unsupported or unresolved physical paired material')
  h=object_sha256(dict(native_pts=pts,source_row_id=rid,actual_flank_uv=p['actual_flank_uv'],measurement_operator='finite_width_paint_material_flank_pair'))
  if h!=item['actual_4D_source_mean_sha256']or h!=p['source_mean_sha256']:raise ValueError('Actual 4D source hash differs')
  source=original[rid];rec=p['source_material_receipt'];review=v.doc(item['actual_finite_source_proof'])
  if source['actual_pair_4D_UV']!=p['actual_flank_uv']or source['source_material_receipt']!=rec or source['native_pts']!=pts or source['shot_id']!=p['shot_id']or source['pixels_from_model']:raise ValueError('Original actual C0 section source changed')
  if review['source_pixel_measurement_uses_model_projection']or review['source_mean_or_flank_UV_changed']or review['world_uniform_width_or_ideal_circle_radius_relation_imposed']:raise ValueError('Model cannot create actual paired pixels or impose width equality')
  for ref in review['actual_images']:v.verify(ref)
  v.verify(review['source_material_RGB_diagnostic']);v.verify(review['actual_S01_C0_source']['source_packet']);v.verify(dict(path=p['native_png'],sha256=p['native_png_sha256']));v.verify(rec['actual_five_profiles_reference'])
  if rec['material_signal']!='dark'or rec['channel']!=0 or rec['flank_polarity']!=[-1,1]or not rec['same_material_both_flanks']or not rec['opposite_ordered_transitions']or not rec['material_channel_frozen_for_replay']:raise ValueError('Actual same-stroke ordered material transition missing')
  if p['allowed_world_side_assignments']!=[[-1,1],[1,-1]]or p['width_scope_id']!=rid+'/actual_cross_section':raise ValueError('Per actual section width or unoriented side identity changed')
  ids=p['factor_row_ids'];gids={lookup.get(k)for k in ids}
  if len(gids)!=1 or None in gids:
   decisions.append(dict(source_row_id=rid,source_identity_eligible=True,solver_eligible=False,numeric_statistics_ready=False,statistical_objective_eligible=False,pending_reason='both_actual_flanks_must_share_one_closed_exposure_covariance'));continue
  gid=next(iter(gids));g=registry[gid];nr=g['native_receipt']
  if g['statistic_kind']!='native_joint_readout'or not g['numeric_whitening_verified']or g.get('actual_mean_replay_max_px',g.get('actual_base_replay_max_px',1.))>1e-9 or g.get('physical_total_error_calibrated',g.get('conditional_model_is_total_error_calibration',False)):raise ValueError('Exact conditional whole-exposure source statistics invalid')
  if int(nr['native_pts'])!=pts or nr['shot_id']!=p['shot_id']or nr['decoded_native_BGR_sha256']!=p['decoded_native_BGR_sha256']or not nr['training_role_verified']or nr.get('native_RGB_resampled',False):raise ValueError('Paired source and shared covariance exposure differs')
  if 'current_pair_measurements_sha256'in g and g['current_pair_measurements_sha256'].get(rid)!=h:raise ValueError('Whole-exposure statistics binds another pair mean')
  for k in ('W','frozen_operator','replay_arrays'):v.verify(g[k])
  rr=[]
  for side,fid in enumerate(ids):
   f=factor_map[fid]
   if f['current_uv']!=p['actual_flank_uv'][side]or f['source_uv']!=f['current_uv']or f.get('original_source_row_id',f.get('source_pair_id'))!=rid:raise ValueError('Actual two-flank order or source mean changed')
   sh=object_sha256(dict(native_pts=pts,source_uv=f['source_uv'],source_row_id=fid));ch=object_sha256(dict(native_pts=pts,current_uv=f['current_uv'],source_row_id=fid,measurement_operator=f['measurement_operator_resolved']))
   if g['source_measurements_sha256'][fid]!=sh or g['current_measurements_sha256'][fid]!=ch:raise ValueError('Flank W and exact source/current hashes differ')
   lo,hi=rec['frozen_flank_windows'][2][side]['normal_window_offsets'];n=np.asarray(p['source_normal']);seed=np.asarray(rec['original_navigation_seed_uv']);mid=.5*(lo+hi);query=seed+mid*n;alpha=float(np.dot(np.asarray(f['current_uv'])-query,n));bounds=[float(lo-mid),float(hi-mid)]
   if not bounds[0]-.26<=alpha<=bounds[1]+.26:raise ValueError('Actual paired mean outside frozen source window')
   rr.append(dict(f,target_id=item['actual_target_id'],joint_material_id=item['joint_material_id'],joint_material_index=item['joint_material_index'],source_pair_id=rid,width_scope_id=p['width_scope_id'],source_material_receipt=rec,source_profile_query_uv=query.tolist(),source_profile_normal=n.tolist(),source_profile_current_alpha=alpha,source_profile_alpha_bounds=bounds,source_identity_eligible=True,solver_eligible=True,numeric_statistics_ready=True,statistical_objective_eligible=True,statistical_group_id=gid,statistic_kind='native_joint_readout',statistics_use='conditional_noise_objective',conditional_model=g['conditional_model'],physical_total_error_calibrated=False,measurement_contract_sha256=contract.digest,exact_full_exposure_covariance_marginal_required=True))
  receipt=dict(schema='rinkfit.current_C0_actual_pair_conditional_admission.v1',source_row_id=rid,native_pts=pts,shot_id=p['shot_id'],actual_4D_source_mean_sha256=h,actual_source_identity_proof=item['actual_finite_source_proof'],source_input_manifest=dict(path=str(identity/'manifest.json'),sha256=file_sha256(identity/'manifest.json')),statistical_group_id=gid,parent_statistics_sha256=object_sha256(g),measurement_contract_sha256=contract.digest,source_identity_eligible=True,source_operator_verified=True,source_mean_or_flank_UV_changed=False,exact_full_exposure_covariance_marginal_required=True,global_uniform_width_imposed=False,raw_midpoint_consumed=False,physical_total_error_calibrated=False);receipts.append(receipt);sourcepairs.append(dict(item,solver_eligible=True,numeric_statistics_ready=True,statistical_objective_eligible=True,statistical_group_id=gid,source_admission_receipt_sha256=object_sha256(receipt)));factors.extend(rr);decisions.append(dict(source_row_id=rid,source_identity_eligible=True,solver_eligible=True,numeric_statistics_ready=True,statistical_objective_eligible=True,actual_4D_source_mean_sha256=h,statistical_group_id=gid))
 v.finish();out.mkdir(parents=True);write_json(out/'approved_pairs.json',sourcepairs);write_jsonl(out/'solver_rows.jsonl.gz',factors);write_jsonl(out/'factor_rows.jsonl.gz',factors);write_jsonl(out/'receipts.jsonl.gz',receipts);write_jsonl(out/'promotion_decisions.jsonl.gz',decisions);report=dict(schema='rinkfit.current_C0_actual_pair_conditional_admission.v1',physically_approved_pairs=len(qualified),conditional_approved_pairs=len(sourcepairs),conditional_approved_flank_rows=len(factors),physical_total_error_calibrated=False,measurement_contract_sha256=contract.digest,source_identity=dict(path=str(identity/'manifest.json'),sha256=file_sha256(identity/'manifest.json')),statistics_registry=dict(path=str(statsfile),sha256=file_sha256(statsfile)),raw_midpoint_consumed=False,source_mean_or_flank_UV_changed=False,width_scope='per_actual_transverse_source_section',mixed_exposure_covariance_consumed_once=True);write_json(out/'manifest.json',report);print(json.dumps(report));return report

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--project',required=True);p.add_argument('--run',required=True);p.add_argument('--identity',required=True);p.add_argument('--statistics',required=True);p.add_argument('--out',required=True);a=p.parse_args();promote(a.project,a.run,a.identity,a.statistics,a.out)
