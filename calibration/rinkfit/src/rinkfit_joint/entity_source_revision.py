"""Append verified native entity operators to an immutable full source ledger."""
from pathlib import Path
from collections import Counter
import argparse,json
from .io_utils import read_json,read_jsonl,write_json,write_jsonl,file_sha256
from .source_worker import assemble_s09

def merge_entity_sources(parent,entity_dir,out):
 parent=Path(parent);entity=Path(entity_dir);out=Path(out)
 if out.exists():raise ValueError('New immutable source generation required')
 overlay={r['source_row_id']:r for r in read_jsonl(entity/'rows_overlay.jsonl.gz')};out.mkdir(parents=True);summary=read_json(parent/'summary.json');oldpolicy=parent/'policy.json'
 if not oldpolicy.is_file():oldpolicy=Path(summary['binding_revision']['parent_dir'])/'policy.json'
 entity_overlay_sha=file_sha256(entity/'rows_overlay.jsonl.gz');old_policy_content=read_json(oldpolicy)
 policy={**old_policy_content,'parent_source_policy':{'path':str(oldpolicy.resolve()),'sha256':file_sha256(oldpolicy),'content':old_policy_content},'native_entity_operator_extension':{'source_operator':'free_current_native_two_finite_material_edge_intersection','entity_source_dir':str(entity.resolve()),'source_manifest_sha256':file_sha256(entity/'manifest.json'),'rows_overlay_sha256':file_sha256(entity/'rows_overlay.jsonl.gz'),'source_fit_receipts_sha256':file_sha256(entity/'fit_receipts.jsonl.gz'),'manual_clicks_never_moved':True,'entity_station_may_slide':False,'numeric_covariance_claimed':False}}
 write_json(out/'policy.json',policy);sha=file_sha256(out/'policy.json');counts=Counter();n=0
 def revised():
  nonlocal n
  for r in read_jsonl(parent/'relocated_rows.jsonl.gz'):
   n+=1;rid=r['source_row_id'];r['policy_sha256']=sha
   if rid in overlay:
    o=overlay[rid]
    if o['source_uv']!=r['source_uv'] or not r['point_correspondence']:raise ValueError('Entity source identity changed')
    r['previous_current_uv']=r['current_uv'];r['current_uv']=o['current_uv'];r['source_measurement_audit']=o['source_measurement_audit'];r['entity_source_generation']=str(entity.resolve());r['entity_source_overlay_sha256']=entity_overlay_sha;r['numeric_statistics_ready']=False;r['solver_eligible']=False
    valid=bool(o['source_measurement_audit']['current_two_material_edges'].get('valid'))
    if valid or r['manual_reference_preserved']:
     r['source_support_status']='source_supported_identity_candidate';r['source_review_reasons']=[];counts['entity_source_supported']+=1
    else:
     r['source_support_status']='source_geometry_followup';r['source_review_reasons']=[o['source_measurement_audit']['current_two_material_edges'].get('reason')];counts['entity_source_pending']+=1
   yield r
 write_jsonl(out/'relocated_rows.jsonl.gz',revised());write_json(out/'retained_support.ref.json',read_json(parent/'retained_support.ref.json'));write_json(out/'joint_material_registry.json',read_json(parent/'joint_material_registry.json'))
 result={**summary,'entity_source_revision':{'parent':str(parent.resolve()),'parent_rows_sha256':file_sha256(parent/'relocated_rows.jsonl.gz'),'entity_generation':str(entity.resolve()),'counts':dict(counts),'all_source_rows':n,'all_original_source_uv_binding_flags_unchanged':True,'current_curve_uv_unchanged':True,'entity_station_fixed':True,'policy_sha256':sha}}
 write_json(out/'summary.json',result);assemble_s09(out/'relocated_rows.jsonl.gz',out/'s09_packet');return result['entity_source_revision']
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--parent',required=True);p.add_argument('--entity',required=True);p.add_argument('--out',required=True);a=p.parse_args();print(json.dumps(merge_entity_sources(a.parent,a.entity,a.out)))
