"""Version a source-role interpretation without changing source measurements.
Only exact, source-pinned circular outside-boundary corrections are accepted.
UI flags stay untouched; current source admission is an explicit reader gate.
"""
from pathlib import Path
from copy import deepcopy
from collections import Counter
import math,struct
from .io_utils import read_json,read_jsonl,write_json,write_jsonl,file_sha256,object_sha256
from .source_generation_input_v2 import QualifiedSourceGenerationInput

def _ref(p):
 p=Path(p).resolve();return {'path':str(p),'sha256':file_sha256(p),'bytes':p.stat().st_size}
def make_source_role_generation(parent_binding,corrections,model,out):
 out=Path(out).resolve()
 if out.exists():raise ValueError('New immutable semantic source generation required')
 old=QualifiedSourceGenerationInput(parent_binding);old.assert_dependencies();fix=read_json(corrections);by=fix['row_corrections'];rows=[];counts=Counter();seen=set();unchanged=0
 forbidden={'weak_or_search_boundary_source','outside_native_image','duplicate_retained_not_independent','frame_local_print_component_requires_shape_crosswalk'}
 for r0 in old.rows():
  r=deepcopy(r0);rid=r['source_row_id'];c=by.get(rid)
  if c:
   seen.add(rid)
   for k in ('source_row_id','source_packet','source_packet_sha256','shot_id','native_pts','source_uv','original_source_point_id'):
    if c[k]!=r[k]:raise ValueError('Source role correction redirects actual source '+k)
   if r['measurement_operator_resolved']!='material_interface' or c['measurement_operator_resolved_unchanged']!='material_interface':raise ValueError('Role correction changes an actual reader operator')
   if r['target_id']!=c['old_target_id'] or r['joint_material_id']!=c['old_joint_material_id'] or r.get('measurement_definition')!=c['original_measurement_definition']:raise ValueError('Role correction parent definition drift')
   before=list(r.get('source_review_reasons',[]));remove=c['remove_source_review_reasons']
   if remove!=['explicit_source_operator_conflicts_with_material_definition'] or before!=c['original_source_review_reasons']:raise ValueError('Source role reason mismatch')
   after=[v for v in before if v not in remove]
   if after!=c['new_source_review_reasons']:raise ValueError('Role correction removes additional source debt')
   r['source_role_history']={'parent_input_binding':_ref(parent_binding),'correction':_ref(corrections),'old_target_id':r['target_id'],'old_joint_material_id':r['joint_material_id'],'old_measurement_definition':r['measurement_definition'],'old_source_review_reasons':before,'old_physical_identity_status':r.get('physical_identity_status'),'old_source_support_status':r.get('source_support_status')}
   r.update(target_id=c['new_target_id'],material_id=c['new_material_id'],joint_material_id=c['new_joint_material_id'],boundary_role=c['new_boundary_role'],typed_material_descriptor=c['new_typed_material_descriptor'],measurement_definition=c['new_measurement_definition'],source_review_reasons=after)
   if not after and r.get('current_uv')is not None:
    audit=r['source_measurement_audit'];guards=[audit['support_guard'],audit.get('original_source_profile_guard',{})]
    if not audit.get('profile',{}).get('valid')or any(g.get('person_mask_pixels',0)or g.get('outside_native_image',False)or g.get('static_exclusion_ids',[])for g in guards):raise ValueError('Corrected role still lacks complete current/original source footprint')
    r['physical_identity_status']='typed_identity_current_pixel_supported';r['source_support_status']='source_supported_identity_candidate';counts['corrected_role_source_admitted']+=1
   else:counts['corrected_role_other_failure_retained']+=1
   counts['role_rows']+=1
  else:unchanged+=1
  r['source_admitted']=bool(r.get('source_support_status')=='source_supported_identity_candidate'and r.get('physical_identity_status')=='typed_identity_current_pixel_supported'and r.get('current_uv')is not None and not r.get('source_review_reasons'))
  r['source_admission_policy']='current_source_footprint_identity_and_all_remaining_reasons_clear_v6'
  counts['source_admitted'if r['source_admitted']else'source_not_admitted']+=1
  if struct.pack('<2d',*r['source_uv'])!=struct.pack('<2d',*r0['source_uv'])or r.get('current_uv')!=r0.get('current_uv')or r['native_pts']!=r0['native_pts']:raise ValueError('Role generation changes observed or reader UV/clock')
  for k in ('source_allowed_for_fit','original_allowed_for_fit','source_user_accepted','manual_reference_preserved','solver_eligible','numeric_statistics_ready'):
   if r.get(k)!=r0.get(k):raise ValueError('Role generation rewrites UI/manual/statistical flags')
  rows.append(r)
 if seen!=set(by)or counts['role_rows']!=161:raise ValueError('Incomplete role-correction denominator')
 out.mkdir(parents=True);write_jsonl(out/'relocated_rows.jsonl.gz',rows)
 gen=deepcopy(old.generation);gen['schema']='rinkfit.S01_S04.source_input_generation.v6';gen['parent_generation']=_ref(old.binding['generation']['path']);gen['source_semantic_role_corrections']=_ref(corrections);gen['common_model_candidate']=_ref(model);gen['same_raw_source_packet_RGB_mask_and_readout_arrays']=True
 write_json(out/'generation.json',gen)
 binding={**deepcopy(old.binding),'generation':_ref(out/'generation.json'),'rows':_ref(out/'relocated_rows.jsonl.gz'),'common_model':_ref(model),'source_role_corrections':_ref(corrections),'dependency_refs':old.binding.get('dependency_refs',[])+[_ref(parent_binding),_ref(corrections),_ref(Path(__file__))],'semantic_generation_qualification':{'source_rows':len(rows),'corrected':161,'all_original_and_current_UV_binary64_exact':True,'source_noise_or_numeric_eligibility_not_promoted':True,'remaining_bad_source_rows_retained':True}}
 for c in by.values():
  for key in ('actual_source_role_reference',):
   if key in c and c[key]not in binding['dependency_refs']:binding['dependency_refs'].append(c[key])
 write_json(out/'input_binding.json',binding);q=QualifiedSourceGenerationInput(out/'input_binding.json');n=sum(1 for _ in q.rows());supports=sum(1 for _ in q.supports());
 qa={'schema':'rinkfit.source_role_semantic_generation.v1','parent':_ref(parent_binding),'input_binding':_ref(out/'input_binding.json'),'corrections':_ref(corrections),'producer':_ref(Path(__file__)),'counts':dict(counts),'rows':n,'support_families':supports,'all161_old_definitions_and_flags_archived':True,'unchanged_other_source_roles':unchanged,'all_source_UV_currentUV_PTS_ID_exact':True,'same_original_frozen_reader_outputs_reused':True,'same_RGB_and_existing_masks_reused':True,'human_approval_not_fabricated':True,'formal_W_not_fabricated':True}
 write_json(out/'qualification.json',qa);return qa
