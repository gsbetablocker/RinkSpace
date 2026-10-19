"""Promote first native RGB measurements with explicit video/decode lineage.

New source_uv is the first actual profile measurement in this generation. It is
never an anchor click, navigation prediction, or retroactively invented point.
"""
from pathlib import Path
from collections import Counter,defaultdict
import argparse,json,gzip
import numpy as np
from .io_utils import read_json,read_jsonl,write_json,write_jsonl,file_sha256,object_sha256
from .materials import apply_promotion,PromotionArtifactVerifier,current_physical_source_artifact_verified
from .role_guard import FrameRoleGuard

def _source_files(directory):
 d=Path(directory);m=read_json(d/'manifest.json')
 closed_subset=m.get('schema')=='rinkfit.immutable_completed_current_outer_source_subset.v1' and sum(c['frames'] for c in m.get('chunks',[]))==m.get('completed_frames')
 if not (closed_subset or m.get('all_development_frames_accounted') or m.get('all_native_pts_accounted',m.get('all_missing_frames_accounted',m.get('all_gap_frames_accounted',False)))):raise ValueError('Incomplete live dense input not admissible')
 if 'chunks' in m:
  for c in m['chunks']:
   files={Path(x['path']).name:Path(x['path']) for x in c['files']}
   for x in c['files']:
    if file_sha256(x['path'])!=x['sha256']:raise ValueError('Dense source chunk changed')
   yield files['rows.jsonl.gz'],files['profiles.npz'],files['profile_index.json'],d
 else:yield d/'rows.jsonl.gz',d/'profiles.npz',d/'profile_index.json',d

def promote_dense(run,source_dirs,statistics,identity,material_map,selected_source,out):
 run=Path(run);out=Path(out);identity=Path(identity);scope=read_json(run/'programme_scope.json');video=Path(scope['source']['path']);video_sha=scope['source']['sha256'];guard=FrameRoleGuard(run/'roles/frame_roles.json');evidence=next(x for x in run.parents if x.name=='RinkSpace_Data')
 if out.exists():raise ValueError('New immutable dense admission generation required')
 stats=read_json(statistics);stats=stats.get('groups',stats);materials={m['id']:m for m in read_json(identity/'joint_material_registry.json')['materials']};mapping=read_json(material_map)['semantic_to_material'];passed={d['source_row_id'] for d in read_jsonl(identity/'identity_decisions.jsonl.gz') if d['physical_identity_passed']};anchors={r['source_row_id']:r for r in read_jsonl(Path(selected_source)/'relocated_rows.jsonl.gz')};selected_pts={r['native_pts'] for r in anchors.values()};groups={}
 for gid,s in stats.items():
  if not s.get('numeric_whitening_verified'):continue
  for rid in s['row_ids']:
   if rid in groups:raise ValueError('Same actual measurement row in multiple numeric groups')
   groups[rid]=gid
 frames={};frame_sources={}
 for name in ['dense_s03_v1','dense_remaining_v1']:
  d=run/name;m=read_json(d/'manifest.json');ledger=d/'frame_ledger.jsonl.gz'
  if not m['all_native_pts_accounted']:raise ValueError('Native decode sequence incomplete')
  sha=file_sha256(ledger)
  for r in read_jsonl(ledger):
   pts=int(r['native_pts'])
   if pts in frames:raise ValueError('Duplicated source exposure across base decoder ledgers')
   frames[pts]=r;frame_sources[pts]=(ledger,sha)
 native_registry={}
 for pts,r in frames.items():
  if not r.get('decoded_native_BGR_sha256'):continue
  ledger,sha=frame_sources[pts];native_registry[f'{video_sha}:{pts}']={'source_video_sha256':video_sha,'native_pts':pts,'decode_sequence_verified':True,'decoded_native_BGR_sha256':r['decoded_native_BGR_sha256'],'native_decode_receipt_sha256':sha,'native_decode_row_sha256':object_sha256(r)}
 out.mkdir(parents=True);(out/'frame_audits').mkdir();files=list(x for d in source_dirs for x in _source_files(d));policy={'schema':'rinkfit.joint9.dense_native_source_admission_policy.v1','source_uv_authority':'first actual current-native RGB profile measurement in immutable source packet, not human click or anchor coordinate','navigation_and_anchor_separate':True,'source_video_sha256':video_sha,'pixel_proof':'official native source video + actual sequential PTS/BGR decode ledger + actual current source measurement/profile journal','reserved_guard_pixels_for_training':False,'same_selected_exposure_requires_joint_source_union':True,'statistics_use':'declared_per_group_conditional_source_model_or_initialization_surrogate','source_generations':[{'path':str(Path(d).resolve()),'manifest_sha256':file_sha256(Path(d)/'manifest.json')} for d in source_dirs]};write_json(out/'policy.json',policy);policy_sha=file_sha256(out/'policy.json')
 def art(path,role,sha=None):
  path=Path(path).resolve()
  if evidence!=path and evidence not in path.parents:raise ValueError('Source proof outside data evidence root')
  return {'path':str(path.relative_to(evidence)),'sha256':sha or file_sha256(path),'role':role}
 base_art=[art(video,'native_video',video_sha),art(out/'policy.json','source_policy',policy_sha)];rows=[];file_artifacts={};seen=set();verifier=PromotionArtifactVerifier(evidence)
 for packet,profiles,pindex,d in files:
  packet_sha=file_sha256(packet);topology_records={x['native_pts']:x for x in read_jsonl(d/'frame_receipts.jsonl.gz')} if (d/'frame_receipts.jsonl.gz').is_file() else {};arts=[art(packet,'source_measurement_audit',packet_sha),art(profiles,'current_native_profiles'),art(pindex,'native_profile_index')];
  if (d/'frame_receipts.jsonl.gz').is_file():arts.append(art(d/'frame_receipts.jsonl.gz','current_native_material_identity'))
  file_artifacts[str(packet)]=arts
  for raw in read_jsonl(packet):
   rid=raw['source_row_id']
   if rid in seen:raise ValueError('Duplicate actual dense source row across generations')
   seen.add(rid);pts=int(raw['native_pts']);sid=raw['shot_id'];guard.assert_training_pixels([pts],shot_id=sid);a=anchors.get(raw['anchor_source_row_id']);m=raw['source_measurement_audit'];uv=raw['current_uv'];n=m.get('normal');nav=None;flankseed=None;navnote='exact pre-corner flow query was not logged by v1; current actual corner query retained, not reconstructed as a human click'
   if raw.get('navigation_seed_uv') is not None:
    nav=raw['navigation_seed_uv'];navnote='exact frozen original material search seed retained in the actual source generation'
   elif uv is not None and n is not None and m.get('profile') and 'offset_px' in m['profile']:
    nav=(np.asarray(uv)-float(m['profile']['offset_px'])*np.asarray(n)).tolist();navnote='exact actual single-interface search seed reconstructed from signed source offset'
   elif uv is not None and n is not None and m.get('flank_profiles'):
    flankseed=[(np.asarray(q)-float(p['offset_px'])*np.asarray(n)).tolist() for q,p in zip(m['flank_uv'],m['flank_profiles'])];nav=np.mean(flankseed,axis=0).tolist();navnote='mean of two actual flank search seeds; original optical-flow center query was not logged'
   row={**raw,'source_uv':raw.get('source_uv',uv),'uv':uv,'source_packet':str(packet),'source_packet_sha256':packet_sha,'source_pointer':'source_row_id='+rid,'measurement_operator_resolved':raw['measurement_operator'],'policy_sha256':policy_sha,'navigation_seed_uv':nav,'navigation_flank_seed_uv':flankseed,'navigation_seed_provenance':navnote,'source_pixel_origin':'automatically_measured_current_native_RGB_profile','original_source_pixel_measurement_is_manual':False,'original_flags_preserved':True,'manual_reference_preserved':False,'fit_role':'development','numeric_statistics_ready':False,'solver_eligible':False,'source_measurement_generation':str(d),'source_frame_BGR_sha256':frames[pts].get('decoded_native_BGR_sha256'),'source_family_role_locked':True};topology_identity=False;topology=raw.get('source_identity_basis')=='current_native_paired_board_material_topology'
   if topology:
    tr=topology_records.get(pts,{});ci=raw.get('native_material_identity_certificate_index',-1);certs=tr.get('source_material_certificates',[])
    if isinstance(ci,int) and 0<=ci<len(certs):
     cert=certs[ci];points=np.asarray(cert.get('current_paired_material_band_points_uv',[]),float);role=raw['target_id'].replace('board.','');which=1 if role=='blue_cap_body_interface' else (2 if role=='yellow_strip_body_interface' else -1)
     if points.ndim==2 and points.shape[0]>=33 and points.shape[1]==3 and which>0:
      distance=abs(float(uv[1])-float(np.interp(float(uv[0]),points[:,0],points[:,which])));topology_identity=bool(tr.get('source_BGR_sha256')==raw.get('decoded_native_BGR_sha256') and cert.get('source_BGR_sha256')==raw.get('decoded_native_BGR_sha256') and cert.get('other_exposure_or_world_model_used') is False and cert.get('configuration',{}).get('darker_external_background_above_blue_and_neutral_ice_below_yellow_required') is True and np.all(np.diff(points[:,0])==8) and np.all(points[:,2]-points[:,1]>=24) and distance<=4 and raw.get('native_material_identity_current_RGB_verified') is True)
   if topology:topology_identity=False;row['withdrawn_inner_board_material_definition']=True
   row['current_material_topology_identity_verified']=topology_identity;row['physical_material_identity_basis']='independent_current_native_paired_board_material_topology' if topology else 'actual_source_anchor_material_identity';row['anchor_identity_verified']=bool(a and raw['anchor_source_row_id'] in passed and a['joint_material_id']==raw['joint_material_id'] and a['measurement_operator_resolved']==raw['measurement_operator']);row['current_physical_definition_artifact_verified']=current_physical_source_artifact_verified(row,row.get('measurement_contract_certificate') or {},evidence,verifier);row['physical_material_identity_verified']=row['current_physical_definition_artifact_verified'];row['current_native_pixel_binding_verified']=bool(raw.get('current_pixels_measured') is True and raw.get('pixels_from_model') is False and raw.get('decoded_native_BGR_sha256')==frames[pts].get('decoded_native_BGR_sha256') and m.get('current_uv')==uv);row['_packet']=str(packet);rows.append(row)
 write_jsonl(out/'normalized_source_journal.jsonl.gz',({k:v for k,v in r.items() if k!='_packet'} for r in rows));by=defaultdict(list)
 for r in rows:by[r['native_pts']].append(r)
 frame_art={}
 for pts,members in by.items():
  path=out/'frame_audits'/f'{pts}.json';write_json(path,{'native_pts':pts,'source_video_sha256':video_sha,'native_decode_binding':native_registry.get(f'{video_sha}:{pts}'),'rows':[{k:r.get(k) for k in ['source_row_id','source_packet_sha256','source_uv','current_uv','measurement_operator_resolved','source_measurement_audit','anchor_source_row_id','anchor_pts','anchor_identity_verified','current_native_pixel_binding_verified','navigation_seed_uv','navigation_flank_seed_uv','navigation_seed_provenance']} for r in members]});frame_art[pts]=art(path,'source_measurement_audit')
 results=[];receipts=[];solver=[];counts=Counter();locked={};output_rows=[]
 for row in rows:
  rid=row['source_row_id'];pts=row['native_pts'];anchor_pts=int(row['anchor_pts']);gid=groups.get(rid);stat=stats.get(gid,{});material=materials.get(row['joint_material_id'],{});geometry=mapping.get(row['joint_material_id'],{});reasons=[]
  try:guard.assert_training_track(min(pts,anchor_pts),max(pts,anchor_pts),shot_id=row['shot_id'])
  except ValueError:reasons.append('anchor_interval_crosses_reserved_or_guard_or_shot')
  if pts in selected_pts:reasons.append('same_native_exposure_as_selected_source_requires_joint_source_covariance_union')
  if not row['physical_material_identity_verified']:reasons.append('actual_current_or_anchor_material_identity_not_verified')
  if not row['current_native_pixel_binding_verified']:reasons.append('actual_current_RGB_or_profile_measurement_not_bound')
  if row['point_correspondence'] and row['source_measurement_audit']['status']!='entity_corner_two_direction_support':reasons.append('current_entity_not_two_direction_material_support')
  lock={'source_row_id':rid,'source_packet_sha256':row['source_packet_sha256'],'native_pts':pts,'source_family_id':row['source_family_id'],'fit_role':'development','assignment_rule':'whole actual current source family at root-protected native exposure; no random representative split'};locked[rid]=lock;source_hash=object_sha256({'native_pts':pts,'source_uv':row['source_uv'],'source_row_id':rid});current_hash=object_sha256({'native_pts':pts,'current_uv':row['current_uv'],'source_row_id':rid,'measurement_operator':row['measurement_operator_resolved']});ledger,decode_sha=frame_sources[pts];proof=native_registry.get(f'{video_sha}:{pts}',{});arts=base_art+[art(ledger,'native_decode_receipt',decode_sha),frame_art[pts]]+file_artifacts[row['_packet']];receipt={'schema':'rinkfit.joint9.dense_source_promotion_receipt.v1','source_row_id':rid,'source_packet_sha256':row['source_packet_sha256'],'video_sha256':video_sha,'policy_sha256':policy_sha,'joint_material_id':row['joint_material_id'],'source_measurement_sha256':source_hash,'current_measurement_sha256':current_hash,'native_decode_row_sha256':proof.get('native_decode_row_sha256'),'evidence_basis':'current_native_source_remeasurement','native_pixel_review_passed':not reasons,'physical_identity_passed':bool(row['physical_material_identity_verified'] and geometry.get('geometry_ready')),'operator_passed':bool(row['measurement_operator_resolved']!='unspecified'),'source_lineage_complete':bool(row['current_native_pixel_binding_verified'] and not any('anchor_interval' in r for r in reasons)),'numeric_statistics_ready':bool(gid),'manual_authority_preserved':True,'material_definition_sha256':object_sha256(material),'fit_role':'development','locked_role_sha256':object_sha256(lock),'statistical_group_id':gid,'statistics_definition_sha256':object_sha256(stat),'evidence_artifacts':arts,'native_pixel_gate_reasons':reasons,'source_uv_is_first_automatic_native_measurement':True,'manual_click_claimed':False,'statistic_kind':stat.get('statistic_kind','initialization_surrogate')};receipt['measurement_contract_sha256']=row.get('measurement_contract_sha256');receipt['measurement_contract_certificate_sha256']=object_sha256(row.get('measurement_contract_certificate') or {});receipt['evidence_artifacts'].extend((row.get('measurement_contract_certificate') or {}).get('source_evidence') or []);decision=apply_promotion(row,{},receipt,video_sha256=video_sha,policy_sha256=policy_sha,material_registry=materials,statistics_registry=stats,locked_roles=locked,evidence_root=evidence,decoded_frame_registry=native_registry,artifact_verifier=verifier);results.append({'source_row_id':rid,**decision,'dense_source_reasons':reasons});receipts.append(receipt);counts[decision['decision']]+=1
  if decision['solver_eligible']:
   clean={k:v for k,v in row.items() if k!='_packet'};solver.append({**clean,'solver_eligible':True,'joint_material_index':geometry['material_index'],'kind':'entity_corner' if row['point_correspondence'] else 'finite_material_curve','entity_station_fixed':bool(row['point_correspondence']),'statistical_group_id':gid,'receipt_sha256':decision['receipt_sha256'],'locked_role_sha256':object_sha256(lock),'statistic_kind':'initialization_surrogate'})
  if len(results)%10000==0:print(json.dumps({'processed_dense_rows':len(results),'solver_rows':len(solver)}),flush=True)
 verifier.verify_unchanged();write_jsonl(out/'solver_rows.jsonl.gz',solver);write_jsonl(out/'promotion_decisions.jsonl.gz',results);write_jsonl(out/'receipts.jsonl.gz',receipts)
 with gzip.open(out/'locked_roles.json.gz','wt',encoding='utf8') as f:json.dump(locked,f,separators=(',',':'))
 write_json(out/'native_frame_identities.json',native_registry);summary={'schema':'rinkfit.joint9.dense_source_admission.v1','source_rows':len(rows),'solver_eligible_rows':len(solver),'decision_counts':dict(counts),'all_actual_source_rows_accounted':len(results)==len(rows),'protected_pixels_consumed':False,'same_selected_exposure_rows_admitted':sum(r['native_pts'] in selected_pts for r in solver),'source_uv_claimed_human':False,'native_source_video_sha256':video_sha,'policy_sha256':policy_sha,'statistics_registry':{'path':str(statistics),'sha256':file_sha256(statistics)},'material_map':{'path':str(material_map),'sha256':file_sha256(material_map)},'artifact_batch_fully_rehashed':True,'statistical_use':'initialization_surrogate'};write_json(out/'manifest.json',summary);return summary
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--sources',nargs='+',required=True);p.add_argument('--statistics',required=True);p.add_argument('--identity',required=True);p.add_argument('--material-map',required=True);p.add_argument('--selected-source',required=True);p.add_argument('--out',required=True);a=p.parse_args();print(json.dumps(promote_dense(a.run,a.sources,a.statistics,a.identity,a.material_map,a.selected_source,a.out)))
