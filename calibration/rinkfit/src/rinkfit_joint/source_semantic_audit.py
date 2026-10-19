"""Lossless source-definition audit and explicit withdrawal of unsafe promotion.

This is a diagnosis ledger, not an automatic source-approval shortcut. A typed
name/hash/low residual cannot establish a currently visible physical boundary.
"""
from pathlib import Path
from collections import Counter,defaultdict
import argparse,json
from .io_utils import read_json,read_jsonl,write_json,write_jsonl,file_sha256,object_sha256
from .measurement_contract import EventMeasurementContract

def _kind(r,c):
 t=str(r.get('target_id') or '');e=c.expected(t,r.get('category'))
 if t in ('board.blue_cap_body_interface','board.yellow_strip_body_interface'):return 'withdrawn_board_internal_interface'
 if t.startswith(('board.top','board.contact')):return 'board_user_outer_boundary'
 if r.get('point_correspondence'):return 'fixed_physical_entity_corner'
 if e and e.get('operator')=='paint_center':return 'finite_width_paint_midline'
 if e and e.get('physical_role')=='declared_nested_target_material_interface':return 'nested_target_material_interface'
 if r.get('category')=='ad_print' or '.print.' in t:return 'actual_print_component_or_hole'
 if r.get('category')=='ad_edge' or (t.startswith('Ad') and (r.get('measurement_operator_resolved') or r.get('measurement_operator'))=='poster_boundary'):return 'actual_installed_poster_edge'
 return 'unclosed_or_other_physical_identity'

def audit_ledger(project,run,outdir):
 project=Path(project);run=Path(run);out=Path(outdir)
 if out.exists():raise ValueError('New immutable source-definition audit directory required')
 c=EventMeasurementContract(project,'ankara_2026');out.mkdir(parents=True)
 sources=[('selected_v7',run/'source_audit_v7/relocated_rows.jsonl.gz')]
 for name in ('dense_admission_v1','dense_admission_delta_v2'):
  sources.append((name,run/name/'normalized_source_journal.jsonl.gz'))
 sources.append(('withdrawn_inner_dense_v3',run/'board_dense_recovery_v3/rows.jsonl.gz'))
 active={}
 for name in ('admission_v4','admission_v5','dense_admission_v1','dense_admission_delta_v2'):
  active[name]={r['source_row_id'] for r in read_jsonl(run/name/'solver_rows.jsonl.gz') if r.get('solver_eligible')}
 counts=Counter();groups=defaultdict(lambda:dict(rows=0,row_ids=[],frames=set(),families=set(),shots=set(),active=Counter()))
 full=[];withdraw=[];variants=set();original_variants=set();input_refs=[]
 for branch,path in sources:
  input_refs.append(dict(branch=branch,path=str(path),sha256=file_sha256(path)))
  for r in read_jsonl(path):
   rid=r['source_row_id'];kind=_kind(r,c);inspect=c.inspect(r);expected=inspect['expected'];reasons=list(inspect['definition_problems'])
   audit=r.get('source_measurement_audit') or {};op=r.get('measurement_operator_resolved') or r.get('measurement_operator')
   if kind=='withdrawn_board_internal_interface':reasons.append('internal_blue_lower_or_yellow_upper_is_not_user_requested_outer_edge')
   elif kind=='board_user_outer_boundary':reasons.append('legacy_abs_gradient_edge_has_no_actual_blue_upper_or_yellow_lower_color_order_certificate')
   elif kind=='finite_width_paint_midline':reasons.append('legacy_flank_peak_test_did_not_require_opposite_polarity_same_paint_or_material_channel')
   elif kind=='nested_target_material_interface':reasons.append('legacy_largest_abs_gradient_did_not_certify_declared_actual_ring_layer_material_pair')
   elif kind in ('actual_print_component_or_hole','actual_installed_poster_edge'):
    reasons.append('actual_finite_contour_topology_and_same_panel_identity_requires_segment_source_proof_not_peak_or_label')
   elif kind=='fixed_physical_entity_corner':
    reasons.append('two_direction_localization_is_distinct_from_same_actual_entity_and_material_role_proof')
   else:reasons.append('current_material_scope_and_finite_physical_identity_not_closed')
   if branch.startswith('dense_') and 'match' in str(r.get('source_acquisition_mode','')):reasons.append('transported_normal_covector_and_current_material_role_need_revalidation')
   relevant=[n for n,ids in active.items() if rid in ids]
   withdrawal=kind in ('withdrawn_board_internal_interface','board_user_outer_boundary') or 'actual_operator_conflicts_with_existing_user_measurement_definition' in reasons
   item=dict(branch=branch,source_row_id=rid,source_family_id=r.get('source_family_id'),shot_id=r.get('shot_id'),native_pts=r.get('native_pts'),target_id=r.get('target_id'),joint_material_id=r.get('joint_material_id'),kind=kind,actual_operator=op,expected=expected,original_source_uv=r.get('source_uv'),current_native_uv=r.get('current_uv'),manual_reference_preserved=bool(r.get('manual_reference_preserved') or r.get('manual_authoritative')),previously_active_generations=relevant,status='withdrawn_pending_correct_physical_source_remeasurement' if withdrawal else 'paused_new_admission_pending_full_source_definition_audit',reasons=sorted(set(reasons)),measurement_contract_sha256=c.digest,source_RGB_material_proof_audited=False,solver_eligible_after_this_audit=False,source_uv_or_historical_flags_modified=False)
   full.append(item)
   if withdrawal:withdraw.append(item)
   counts[(branch,kind)]+=1;key=branch+'|'+kind;g=groups[key];g['rows']+=1;g['row_ids'].append(rid);g['frames'].add(r.get('native_pts'));g['families'].add(r.get('source_family_id'));g['shots'].add(r.get('shot_id'));g['active'].update(relevant)
   if branch=='selected_v7':
    variants.add(r.get('joint_material_id'));original_variants.add((r.get('target_id'),r.get('physical_surface'),r.get('measurement_operator'),r.get('boundary_role')))
 write_jsonl(out/'all_source_definition_decisions.jsonl.gz',full)
 write_jsonl(out/'withdrawn_rows.jsonl.gz',withdraw)
 write_jsonl(out/'withdrawn_board_rows.jsonl.gz',[r for r in withdraw if r['kind'] in ('withdrawn_board_internal_interface','board_user_outer_boundary')])
 write_jsonl(out/'withdrawn_operator_conflicts.jsonl.gz',[r for r in withdraw if 'actual_operator_conflicts_with_existing_user_measurement_definition' in r['reasons']])
 summary_groups=[]
 for key,g in groups.items():
  summary_groups.append(dict(branch_and_kind=key,rows=g['rows'],source_family_count=len(g['families']),frame_count=len(g['frames']),native_pts_range=[min(v for v in g['frames'] if v is not None),max(v for v in g['frames'] if v is not None)],shots=sorted(v for v in g['shots'] if v),previously_active_by_generation=dict(g['active'])))
 write_json(out/'groups.json',summary_groups)
 write_json(out/'confirmed_fact_references.json',dict(contract_sha256=c.digest,references=c.references))
 code=Path(__file__).parent;names=('source_worker.py','materials.py','source_promotion.py','dense_promotion.py','dense_reacquire.py','dense_image_match.py','entity_corners.py','entity_reacquire.py','fixed_entity_source.py','board_interface_source.py','board_dense_recovery.py','board_outer_source.py','measurement_contract.py','material_compiler.py')
 code_refs=[dict(path=str(code/n),sha256=file_sha256(code/n)) for n in names]
 write_json(out/'code_references.json',code_refs)
 report=dict(schema='rinkfit.complete_source_definition_audit_initial_diagnosis.v1',contract_sha256=c.digest,input_references=input_refs,ledger_rows=len(full),board_withdrawal_rows=len(withdraw),groups=summary_groups,selected_current_joint_material_variants=len(variants),selected_original_descriptor_tuples=len(original_variants),all_input_rows_accounted=True,actual_RGB_audit_completed=False,source_uv_or_flags_changed=False,inner_material_indices_withdrawn=[348,349],inner_height_blocks_withdrawn=True,old_QA_limit='native gradient/hash checks and inner band QA did not validate the user-defined exterior boundary; those claims are withdrawn',shared_shape_followups=['Ad8 legacy component14 W/Y gap bridge','U/K source coverage','Ad4 degenerate components','Ad1 finite branch falls on adjacent installed panel','holes separate from occupied extents; no fictitious rectangle'],pending='Current RGB material order, every finite contour segment, and complete source-family physical identity must be re-read; this ledger is not approval')
 c.verify_unchanged();write_json(out/'manifest.json',report);return report

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--project',required=True);p.add_argument('--run',required=True);p.add_argument('--out',required=True);a=p.parse_args();print(json.dumps(audit_ledger(a.project,a.run,a.out),ensure_ascii=False))
