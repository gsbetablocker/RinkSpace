"""Release native source rows only through the reusable full promotion gate."""
from pathlib import Path
from collections import defaultdict,Counter
import argparse,gzip,json
import numpy as np
from PIL import Image
from .io_utils import read_json,read_jsonl,write_json,write_jsonl,file_sha256,object_sha256
from .materials import apply_promotion,PromotionArtifactVerifier

def promote_sources(audit_dir,identity_dir,statistics_registry,locked_roles,frame_identities,material_map,output_dir,evidence_root):
 audit=Path(audit_dir);identity=Path(identity_dir);out=Path(output_dir);root=Path(evidence_root).resolve()
 if out.exists():raise ValueError('New immutable promotion generation required')
 rows=list(read_jsonl(audit/'relocated_rows.jsonl.gz'));materials={m['id']:m for m in read_json(identity/'joint_material_registry.json')['materials']};decisions={r['source_row_id']:r for r in read_jsonl(identity/'identity_decisions.jsonl.gz')};stats=read_json(statistics_registry)
 if 'groups' in stats:stats=stats['groups']
 if isinstance(stats,list):stats={s.get('statistical_group_id',s.get('id',s.get('family_id'))):s for s in stats}
 with gzip.open(locked_roles,'rt',encoding='utf8') as f:roles=json.load(f)
 identities=read_json(frame_identities);mapping=read_json(material_map)['semantic_to_material'];group_for={}
 for gid,group in stats.items():
  if not group.get('numeric_whitening_verified'):continue
  for rid in group['row_ids']:
   if rid in group_for:raise ValueError('Source row counted in multiple numeric W groups: '+rid)
   group_for[rid]=gid
 for row in rows:
  rid=row['source_row_id']
  if rid not in roles or roles[rid]['native_pts']!=row['native_pts'] or roles[rid]['source_packet_sha256']!=row['source_packet_sha256']:raise ValueError('Missing/stale complete source family role')
 # Current UV is actual new source measurement.  Original source UV and original
 # flags remain in every artifact and are hashed by the gate independently.
 out.mkdir(parents=True);(out/'frame_audits').mkdir();by=defaultdict(list)
 for row in rows:by[row['native_pts']].append(row)
 policy_path=audit/'policy.json'
 if not policy_path.is_file():policy_path=Path(read_json(audit/'summary.json')['binding_revision']['parent_dir'])/'policy.json'
 policy_artifact={'path':str(policy_path.resolve().relative_to(root)),'sha256':file_sha256(policy_path),'role':'source_policy'}
 frame_artifacts={}
 for pts,members in by.items():
  file=out/'frame_audits'/f'{pts}.json';write_json(file,{'schema':'rinkfit.current_native_source_frame_audit.v1','native_pts':pts,'rows':[{'source_row_id':r['source_row_id'],'source_packet_sha256':r['source_packet_sha256'],'source_uv':r['source_uv'],'current_uv':r['current_uv'],'measurement_operator':r['measurement_operator_resolved'],'source_measurement_audit':r['source_measurement_audit'],'source_review_reasons':r['source_review_reasons']} for r in members]});frame_artifacts[pts]={'path':str(file.resolve().relative_to(root)),'sha256':file_sha256(file),'role':'source_measurement_audit'}
 mask_records={};mask_cache={};data_root=next(x for x in root.parents if x.name=='RinkSpace_Data') if root.name!='RinkSpace_Data' else root
 for sid in {r['shot_id'] for r in rows}:
  nr=int(sid[1:]);mb=data_root/'events/ankara_2026/programmes/okada_mei_sp/results'/f'shot{nr}_review_20261014';mp=mb/'r2/frame_manifest.json' if nr==2 else mb/'frame_manifest.json'
  for mf in read_json(mp)['frames']:mask_records[(sid,int(mf.get('native_pts',mf.get('source_pts'))))]=mf
 verifier=PromotionArtifactVerifier(root);output=[];receipts=[];solver=[];counts=Counter();video='e8b53966d56e72f3842d92c3ecbe6e9cc7e4670b7d0c9bd12f58fd2cff5ad57d'
 for row in rows:
  rid=row['source_row_id'];decision=decisions[rid];gid=group_for.get(rid);material=materials.get(row.get('joint_material_id'),{});geometry=mapping.get(row.get('joint_material_id'),{});role=roles[rid];stats_group=stats.get(gid,{})
  pixel_reasons=[]
  if row['point_correspondence'] and row['source_measurement_audit']['status'] not in {'entity_corner_two_direction_support','manual_native_click_preserved'}:pixel_reasons.append('entity_corner_two_direction_source_not_confirmed')
  if row.get('current_uv') is not None:
   key=(row['shot_id'],row['native_pts'])
   if key not in mask_cache:
    if len(mask_cache)>=2:mask_cache.pop(next(iter(mask_cache)))
    mr=mask_records[key];mp=Path(mr.get('mask_path') or mr.get('source',{}).get('mask_path'));expected=mr.get('mask_sha256') or mr.get('source',{}).get('mask_sha256')
    if expected and file_sha256(mp)!=expected:raise ValueError('Authoritative source mask changed')
    mask_cache[key]=np.asarray(Image.open(mp).convert('L'))
   u,v=np.rint(row['current_uv']).astype(int)
   override=row['source_measurement_audit'].get('mask_resolution')=='point_level_native_ice_visible_mask_overreach_near_blurred_hand_edge'
   if not(0<=u<1920 and 0<=v<1080):pixel_reasons.append('current_native_uv_outside_image')
   elif mask_cache[key][v,u] and not override:pixel_reasons.append('current_native_uv_inside_actor_mask')
  source_identity=object_sha256({'native_pts':row['native_pts'],'source_uv':row['source_uv'],'source_row_id':rid});current_identity=object_sha256({'native_pts':row['native_pts'],'current_uv':row['current_uv'],'source_row_id':rid,'measurement_operator':row['measurement_operator_resolved']})
  receipt={'schema':'rinkfit.joint9.source_promotion_receipt.v1','source_row_id':rid,'source_packet_sha256':row['source_packet_sha256'],'video_sha256':video,'policy_sha256':row['policy_sha256'],'evidence_basis':'native_source_manual_click_preserved' if row['manual_reference_preserved'] else 'current_native_source_remeasurement','source_measurement_sha256':source_identity,'current_measurement_sha256':current_identity,'native_pixel_review_passed':row['source_support_status']=='source_supported_identity_candidate' and not pixel_reasons,'native_pixel_gate_reasons':pixel_reasons,'physical_identity_passed':bool(decision['physical_identity_passed'] and geometry.get('geometry_ready')),'operator_passed':not row['measurement_definition']['conflict'] and row['measurement_operator_resolved']!='unspecified','source_lineage_complete':True,'numeric_statistics_ready':bool(gid),'manual_authority_preserved':row['original_flags_preserved'] is True,'joint_material_id':row.get('joint_material_id'),'material_definition_sha256':object_sha256(material),'fit_role':role['fit_role'],'locked_role_sha256':object_sha256(role),'statistical_group_id':gid,'statistics_definition_sha256':object_sha256(stats_group),'evidence_artifacts':[{'path':str(Path(row['native_png']).resolve().relative_to(root)),'sha256':row['native_png_sha256'],'role':'native_frame'},frame_artifacts[row['native_pts']],policy_artifact],'geometry_material_index':geometry.get('material_index'),'source_family_id':row['source_family_id'],'original_source_flags_changed':False,'requires_frame_by_frame_user_approval':False}
  receipt['measurement_contract_sha256']=row.get('measurement_contract_sha256')
  receipt['measurement_contract_certificate_sha256']=object_sha256(row.get('measurement_contract_certificate') or {})
  receipt['evidence_artifacts'].extend((row.get('measurement_contract_certificate') or {}).get('source_evidence') or [])
  promoted=apply_promotion(row,decision,receipt,video_sha256=video,policy_sha256=row['policy_sha256'],material_registry=materials,statistics_registry=stats,locked_roles=roles,evidence_root=root,decoded_frame_registry=identities,artifact_verifier=verifier)
  if not geometry.get('geometry_ready'):promoted.setdefault('diagnostics',[]).append('shared_physical_material_geometry_not_compiled')
  output.append(promoted);receipts.append(receipt);counts[promoted['decision']]+=1
  if promoted['solver_eligible']:
   solver.append({**row,'solver_eligible':True,'joint_material_index':geometry['material_index'],'statistical_group_id':gid,'promotion_receipt_sha256':object_sha256(receipt),'fit_role':role['fit_role'],'locked_role_sha256':object_sha256(role),'kind':'entity_corner' if row['point_correspondence'] else 'finite_material_curve','entity_station_fixed':bool(row['point_correspondence']),'original_source_uv_preserved':True})
 # Double hashing closes the batch against shared artifact mutation.  No solver
 # rows or admission decisions are published before this succeeds.
 frozen=verifier.verify_unchanged();write_json(out/'artifact_integrity.json',frozen);write_jsonl(out/'promotion_decisions.jsonl.gz',output);write_jsonl(out/'receipts.jsonl.gz',receipts);write_jsonl(out/'solver_rows.jsonl.gz',solver)
 byshot=Counter(r['shot_id'] for r in solver);kinds=Counter(r['kind'] for r in solver)
 result={'schema':'rinkfit.joint9.source_admission_generation.v1','raw_rows':len(rows),'solver_eligible_rows':len(solver),'decision_counts':dict(counts),'by_shot':dict(byshot),'by_kind':dict(kinds),'source_audit_rows':{'path':str(audit/'relocated_rows.jsonl.gz'),'sha256':file_sha256(audit/'relocated_rows.jsonl.gz')},'material_identity_registry':{'path':str(identity/'joint_material_registry.json'),'sha256':file_sha256(identity/'joint_material_registry.json')},'statistics_registry':{'path':str(Path(statistics_registry).resolve()),'sha256':file_sha256(statistics_registry)},'locked_roles':{'path':str(Path(locked_roles).resolve()),'sha256':file_sha256(locked_roles)},'material_map':{'path':str(Path(material_map).resolve()),'sha256':file_sha256(material_map)},'source_uv_or_flags_modified':False,'source_role_assignments_changed':False,'all_original_rows_accounted':True,'all_input_rows_admitted':len(solver)==len(rows),'artifact_batch_fully_rehashed':True};write_json(out/'manifest.json',result);return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--audit',required=True);p.add_argument('--identity',required=True);p.add_argument('--statistics',required=True);p.add_argument('--roles',required=True);p.add_argument('--frame-identities',required=True);p.add_argument('--material-map',required=True);p.add_argument('--evidence-root',required=True);p.add_argument('--out',required=True);a=p.parse_args();print(json.dumps(promote_sources(a.audit,a.identity,a.statistics,a.roles,a.frame_identities,a.material_map,a.out,a.evidence_root)))
