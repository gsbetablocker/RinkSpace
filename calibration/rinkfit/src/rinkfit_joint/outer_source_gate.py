"""Actual source admission for geometry initialization, without fabricated W."""
from pathlib import Path
from collections import Counter
import argparse,json
from .io_utils import read_json,read_jsonl,write_json,write_jsonl,file_sha256,object_sha256
from .role_guard import FrameRoleGuard
from .measurement_contract import EventMeasurementContract
from .materials import PromotionArtifactVerifier,current_physical_source_artifact_verified

def gate(project,run,source_dirs,identity,material_map,outdir):
 project=Path(project);run=Path(run);out=Path(outdir)
 if out.exists():raise ValueError('Immutable new gate directory required')
 contract=EventMeasurementContract(project,'ankara_2026');guard=FrameRoleGuard(run/'roles/frame_roles.json');data=next(q for q in run.parents if q.name=='RinkSpace_Data');scope=read_json(run/'programme_scope.json');video=Path(scope['source']['path'])
 if file_sha256(video)!=scope['source']['sha256']:raise ValueError('Official video changed')
 frames={int(r['native_pts']):r for n in ('dense_s03_v1','dense_remaining_v1') for r in read_jsonl(run/n/'frame_ledger.jsonl.gz')};mapping=read_json(material_map)['semantic_to_material'];materials={m['id']:m for m in read_json(Path(identity)/'joint_material_registry.json')['materials']};verifier=PromotionArtifactVerifier(data);rows=[];decisions=[];accepted=[];seen=set();counts=Counter();refs=[]
 for directory in source_dirs:
  directory=Path(directory);manifest=read_json(directory/'manifest.json')
  if not(manifest.get('all_original_source_rows_accounted') and manifest.get('physical_material_selection_not_redetected')):raise ValueError('Unclosed cubic generation')
  refs.append(dict(path=str(directory/'manifest.json'),sha256=file_sha256(directory/'manifest.json')))
  for chunk in manifest['chunks']:
   files={Path(a['path']).name:Path(a['path']) for a in chunk['files']}
   for a in chunk['files']:
    if file_sha256(a['path'])!=a['sha256']:raise ValueError('Source chunk changed')
   packet_sha=file_sha256(files['rows.jsonl.gz'])
   for original in read_jsonl(files['rows.jsonl.gz']):
    row=dict(original);rid=row['source_row_id'];pts=int(row['native_pts']);sid=row['shot_id'];errors=[]
    if rid in seen:raise ValueError('Duplicate actual source row')
    seen.add(rid);guard.assert_training_pixels([pts],shot_id=sid);row['category']='board_edge';row['physical_surface']='board';row['measurement_operator_resolved']=row['measurement_operator'];cert=row.get('measurement_contract_certificate') or {};expected=contract.inspect(row);errors.extend(expected['definition_problems']);fr=frames.get(pts,{})
    if row.get('current_uv') is None:errors.append('current_actual_material_or_window_pending')
    if row.get('source_material_locally_qualified') is not True:errors.append('current_material_qualification_missing')
    if row.get('decoded_native_BGR_sha256')!=fr.get('decoded_native_BGR_sha256'):errors.append('official_actual_decoder_BGR_mismatch')
    if row.get('pixels_from_model') is not False or row.get('current_pixels_measured') is not True:errors.append('actual_native_source_measurement_missing')
    if cert.get('contract_sha256')!=contract.digest or cert.get('source_row_id')!=rid or cert.get('native_pts')!=pts or cert.get('joint_material_id')!=row['joint_material_id']:errors.append('actual_source_contract_binding_mismatch')
    if cert.get('current_measurement_sha256')!=object_sha256(dict(native_pts=pts,current_uv=row.get('current_uv'),source_row_id=rid)):errors.append('current_mean_certificate_mismatch')
    for reference in cert.get('source_evidence',[]):
     path=(data/reference['path']).resolve()
     if verifier.digest(path)!=reference['sha256']:raise ValueError('Actual material proof changed')
    if row.get('current_uv') is not None and not current_physical_source_artifact_verified(row,cert,data,verifier):errors.append('actual_current_color_topology_artifact_not_verified')
    material=materials.get(row['joint_material_id'],{});geometry=mapping.get(row['joint_material_id'],{})
    if material.get('identity_accepted') is not True or material.get('surface')!='board':errors.append('common_actual_outer_material_identity_not_accepted')
    if geometry.get('geometry_ready') is not True:errors.append('finite_common_perimeter_not_ready')
    row.update(source_packet=str(files['rows.jsonl.gz']),source_packet_sha256=packet_sha,source_geometry_eligible=not errors,solver_eligible=False,numeric_statistics_ready=False,statistical_objective_eligible=False,source_measurement_semantics='source_uv=first bilinear actual measurement; current_uv=new continuous cubic measurement; no UV from model',fit_role='development',joint_material_index=geometry.get('material_index'),kind='finite_material_curve',entity_station_fixed=False,statistics_use='explicit_source_family_balanced_geometry_initialization_only',measurement_contract_sha256=contract.digest,source_measurement_generation=str(directory),source_family_role_locked=True)
    decision=dict(source_row_id=rid,native_pts=pts,shot_id=sid,source_geometry_eligible=not errors,solver_eligible=False,numeric_statistics_ready=False,physical_identity_passed=not errors,current_native_source_support_passed=not errors,geometry_errors=errors,measurement_contract_sha256=contract.digest,measurement_contract_certificate_sha256=object_sha256(cert),receipt_sha256=object_sha256(dict(source_row_id=rid,source_packet_sha256=row['source_packet_sha256'],certificate=cert,material=material,geometry=geometry,errors=errors)))
    row['receipt_sha256']=decision['receipt_sha256'];rows.append(row);decisions.append(decision);counts.update(errors or ['geometry_source_accepted'])
    if not errors:accepted.append(row)
 out.mkdir(parents=True);verifier.verify_unchanged();write_jsonl(out/'normalized_source_journal.jsonl.gz',rows);write_jsonl(out/'geometry_rows.jsonl.gz',accepted);write_jsonl(out/'promotion_decisions.jsonl.gz',decisions);result=dict(schema='rinkfit.actual_user_outer_source_geometry_gate.v1',source_rows=len(rows),geometry_rows=len(accepted),decision_counts=dict(counts),source_generations=refs,material_map=dict(path=str(material_map),sha256=file_sha256(material_map)),material_registry=dict(path=str(Path(identity)/'joint_material_registry.json'),sha256=file_sha256(Path(identity)/'joint_material_registry.json')),source_uv_preserved=True,new_mean='continuous_cubic_frozen_actual_material',old_bilinear_W_reused=False,numeric_statistics_ready=False,statistical_objective_eligible=False,old_board_measurements_must_remain_withdrawn=True,old_inner_board_height_materials_used=False,all_source_rows_accounted=True,protected_pixels_consumed=False,measurement_contract_sha256=contract.digest);write_json(out/'manifest.json',result);print(json.dumps(result));return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--project',required=True);p.add_argument('--run',required=True);p.add_argument('--sources',nargs='+',required=True);p.add_argument('--identity',required=True);p.add_argument('--material-map',required=True);p.add_argument('--out',required=True);a=p.parse_args();gate(a.project,a.run,a.sources,a.identity,a.material_map,a.out)
