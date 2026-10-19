"""Resolve actual continuous-board materials without rewriting source history."""
from pathlib import Path
import argparse,json
from .io_utils import read_json,read_jsonl,write_json,write_jsonl,file_sha256,object_sha256
from .source_worker import assemble_s09

def merge_board_sources(parent,board,identity,out):
 parent=Path(parent);board=Path(board);identity=Path(identity);out=Path(out)
 if out.exists():raise ValueError('New immutable typed-source generation required')
 overlays={r['source_row_id']:r for r in read_jsonl(board/'rows_overlay.jsonl.gz')};proofs={r['source_row_id']:r for r in read_jsonl(board/'source_decisions.jsonl.gz') if r['source_color_identity_verified']};oldpolicy=read_json(parent/'policy.json');out.mkdir(parents=True);policy={**oldpolicy,'parent_source_policy':{'path':str((parent/'policy.json').resolve()),'sha256':file_sha256(parent/'policy.json'),'content':oldpolicy},'continuous_board_material_source_extension':{'source_manifest':str(board/'manifest.json'),'source_manifest_sha256':file_sha256(board/'manifest.json'),'source_decisions_sha256':file_sha256(board/'source_decisions.jsonl.gz'),'both_material_colors_required':True,'no_source_or_current_uv_changed':True,'source_operator_equivalence':'unspecified and material_interface execute the exact same _remeasure single-interface branch; only the source material/operator identity token is resolved','new_shared_height_geometry_not_physical_measurement':True}};write_json(out/'policy.json',policy);ps=file_sha256(out/'policy.json');rewrites=[];n=0
 def revised():
  nonlocal n
  for row in read_jsonl(parent/'relocated_rows.jsonl.gz'):
   n+=1;row['policy_sha256']=ps;rid=row['source_row_id']
   if rid in overlays:
    o=overlays[rid]
    if o['source_uv']!=row['source_uv'] or o['current_uv']!=row['current_uv']:raise ValueError('Board role resolution changed measured pixels')
    old={'target_id':row.get('target_id'),'joint_material_id':row.get('joint_material_id'),'measurement_operator_resolved':row['measurement_operator_resolved'],'category':row['category'],'boundary_role':row['boundary_role']};row['source_binding_before_native_material_resolution']=old
    for k in ['target_id','joint_material_id','physical_surface','category','measurement_operator_resolved','measurement_definition','boundary_role','source_measurement_audit']:row[k]=o[k]
    row['source_support_status']='source_supported_identity_candidate';row['physical_identity_status']='actual_continuous_board_material_interface_confirmed';row['typed_material_descriptor']={'kind':'board_material_interface','role':row['boundary_role'],'continuous_course':True,'shared_height_fraction_fitted':True,'point_depth_unknown':False};row['source_review_reasons']=[];row['numeric_statistics_ready']=False;row['solver_eligible']=False
    rewrites.append({'source_row_id':rid,'native_pts':row['native_pts'],'source_uv':row['source_uv'],'current_uv':row['current_uv'],'old_operator':old['measurement_operator_resolved'],'new_operator':'material_interface','source_mean_unchanged':True,'observer_numerics_unchanged':True,'same_single_interface_source_algorithm':True,'current_measurement_sha256':object_sha256({'native_pts':row['native_pts'],'current_uv':row['current_uv'],'source_row_id':rid,'measurement_operator':'material_interface'}),'raw_source_measurement_sha256':object_sha256({'native_pts':row['native_pts'],'source_uv':row['source_uv'],'source_row_id':rid}),'native_material_proof_sha256':object_sha256(proofs[rid]),'new_joint_material_id':row['joint_material_id']})
   yield row
 write_jsonl(out/'relocated_rows.jsonl.gz',revised());write_json(out/'retained_support.ref.json',read_json(parent/'retained_support.ref.json'));write_json(out/'joint_material_registry.json',read_json(board/'joint_material_registry.json'));write_jsonl(out/'observer_semantic_equivalence.jsonl.gz',rewrites)
 def newdecisions():
  for d in read_jsonl(identity/'identity_decisions.jsonl.gz'):
   rid=d['source_row_id']
   if rid in overlays:d={**d,'joint_material_id':overlays[rid]['joint_material_id'],'physical_identity_passed':True,'current_native_source_support_passed':True,'identity_evidence_sha256':object_sha256(proofs[rid]),'source_identity_reasons':[],'source_row_reasons':[],'solver_eligible':False,'statistics_promotion_pending':True}
   yield d
 write_jsonl(out/'identity_decisions.jsonl.gz',newdecisions());summary={**read_json(parent/'summary.json'),'native_board_material_revision':{'parent_source':str(parent.resolve()),'parent_rows_sha256':file_sha256(parent/'relocated_rows.jsonl.gz'),'source_board_dir':str(board.resolve()),'rows_resolved':len(rewrites),'all_original_rows':n,'any_source_or_current_uv_or_original_flags_changed':False,'new_materials_share_one_height_fraction_per_physical_role':True}};write_json(out/'summary.json',summary);assemble_s09(out/'relocated_rows.jsonl.gz',out/'s09_packet');return summary['native_board_material_revision']
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--parent',required=True);p.add_argument('--board',required=True);p.add_argument('--identity',required=True);p.add_argument('--out',required=True);a=p.parse_args();print(json.dumps(merge_board_sources(a.parent,a.board,a.identity,a.out)))
