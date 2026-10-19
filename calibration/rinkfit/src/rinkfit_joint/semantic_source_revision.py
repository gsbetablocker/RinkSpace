"""Correct click-vs-editor visibility semantics using stored actual RGB reads."""
from pathlib import Path
from collections import Counter
import argparse,json,gzip
from .io_utils import read_json,read_jsonl,write_json,write_jsonl,file_sha256
from .measurement_contract import EventMeasurementContract

def revise(project,parent,outdir):
 parent=Path(parent);out=Path(outdir)
 if out.exists():raise ValueError('New immutable corrected source generation required')
 contract=EventMeasurementContract(project,'ankara_2026');out.mkdir(parents=True);counts=Counter();decisions=[]
 policy=dict(schema='rinkfit.actual_click_vs_edited_source_visibility.v1',parent_actual_source_manifest=dict(path=str(parent/'manifest.json'),sha256=file_sha256(parent/'manifest.json')),manual_click_authority='only manual_reference_preserved; whole-frame manual_authoritative denotes edited source/visible support, not a click',board_status='v9 board source withheld; separate fixed-source board_outer_full generation required',measurement_contract_sha256=contract.digest,new_RGB_reading_in_this_transform=False,actual_RGB_operator_receipts_reused_without_change=True)
 write_json(out/'policy.json',policy);psha=file_sha256(out/'policy.json')
 with gzip.open(out/'relocated_rows.jsonl.gz','wt',encoding='utf8') as f:
  for r in read_jsonl(parent/'relocated_rows.jsonl.gz'):
   original=dict(r);actual_click=bool(r.get('manual_reference_preserved'));edited=bool(r.get('manual_authoritative'));reused=False
   if actual_click:
    if r.get('current_uv')!=r['source_uv']:raise ValueError('Actual human click has been moved; explicit user revision needed')
    r['source_definition_reaudit_status']='actual_manual_click_unchanged_material_definition_and_click_noise_still_require_review';counts['actual_manual_clicks_preserved']+=1
   elif edited:
    rr=r.get('semantic_automatic_source_diagnostic')
    if not isinstance(rr,dict):raise ValueError('Edited derived representative lacks its already performed actual source reading')
    r['current_uv']=rr.get('current_uv');r['source_measurement_audit']=rr;r['measurement_operator_resolved']=rr.get('measurement_operator','unspecified');r['semantic_local_material_operator_passed']=bool(rr.get('current_uv') is not None);r['source_definition_reaudit_status']='actual_RGB_derived_representative_remeasured_not_a_human_click';reused=True;counts['edited_visibility_derived_representatives_corrected']+=1
   target=str(r.get('target_id') or '')
   if target.startswith(('board.top','board.contact')) and not actual_click:
    r['withheld_previous_outer_candidate_uv']=r.get('current_uv');r['current_uv']=None;r['semantic_local_material_operator_passed']=False;r['source_definition_reaudit_status']='v9_outer_source_withdrawn_pending_fixed_empty_plateau_guard_full_reread';counts['board_rows_withheld_for_fixed_full_source_generation']+=1
   r['edited_source_visibility_authority_preserved']=edited;r['original_source_is_actual_manual_click']=actual_click;r['v9_whole_editor_frame_misclassified_as_click']=edited and not actual_click;r['original_source_uv_unchanged']=True;r['previous_source_policy_sha256']=r.get('policy_sha256');r['policy_sha256']=psha;r['solver_eligible']=False;r['numeric_statistics_ready']=False
   if r['source_uv']!=original['source_uv']:raise ValueError('Original raw source UV changed')
   f.write(json.dumps(r,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n');counts['rows']+=1
   if r.get('current_uv') is not None and r.get('semantic_local_material_operator_passed') and not actual_click:counts['current_actual_automatic_material_candidates']+=1
   decisions.append(dict(source_row_id=r['source_row_id'],shot_id=r['shot_id'],native_pts=r['native_pts'],target_id=r.get('target_id'),actual_manual_click=actual_click,edited_source_visibility=edited,reused_exact_actual_RGB_receipt=reused,current_uv=r.get('current_uv'),operator=r.get('measurement_operator_resolved'),local_material_candidate=bool(r.get('semantic_local_material_operator_passed')),finite_physical_identity_approved=False,solver_eligible=False))
 write_jsonl(out/'operator_decisions.jsonl.gz',decisions);contract.verify_unchanged();result=dict(schema='rinkfit.source_click_visibility_semantics_correction.v1',counts=dict(counts),parent_actual_source=str(parent),parent_rows_sha256=file_sha256(parent/'relocated_rows.jsonl.gz'),manifest_rows_sha256=file_sha256(out/'relocated_rows.jsonl.gz'),source_uv_or_flags_or_actual_manual_clicks_changed=False,all_raw_rows_accounted=counts['rows']==110350,old_W_reused_for_new_mean=False,finite_physical_identity_and_new_statistics_pending=True,policy_sha256=psha);write_json(out/'manifest.json',result);return result

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--project',required=True);p.add_argument('--parent',required=True);p.add_argument('--out',required=True);a=p.parse_args();print(json.dumps(revise(a.project,a.parent,a.out)))
