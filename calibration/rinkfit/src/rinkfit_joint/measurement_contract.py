"""Read existing event/user facts before proposing a source operator.

This contract checks the meaning of a measurement. Passing it is never proof
that a pixel actually belongs to that material; current RGB evidence is still
required. Historical task instructions cannot authorize current work here.
"""
from pathlib import Path
import re
from .io_utils import read_json,file_sha256,object_sha256

class EventMeasurementContract:
 def __init__(self,project,event_id):
  self.project=Path(project).resolve();self.event_id=event_id
  if event_id!='ankara_2026':raise ValueError('No event-specific confirmed measurement contract was loaded')
  base=self.project/'events'/event_id/'model_constraints';self.documents={};self.references={}
  names=('user_board_edge_semantics_20261010.json','ice_contact_material_roles_user_confirmed_20261006.json','visual_boundary_roles_user_confirmed_20261006.json','paint_roles_1018.json','source_material_operator_corrections_20261012.json','source_material_operator_corrections_right_end_20261012.json','source_material_operator_corrections_straight_connections_20261012.json','user_advertisement_pairs_20261005.json','ad7_user_1016.json','lane2_user_confirmed_20261007.json','R2_outer_user_identity_clarification_20261010.json','user_camera_setup_confirmed_20261006.json','source_clock_registry.json','user_corrections_current.json')
  for name in names:
   path=base/name;self.documents[name]=read_json(path);self.references[name]=dict(path=str(path),sha256=file_sha256(path))
  board=self.documents[names[0]]['definitions'];contact=self.documents[names[1]]['definition'];visual=self.documents[names[2]]['definition'];paint=self.documents['paint_roles_1018.json']['definitions']
  if board['board_top']!='upper edge of blue rim, top of vertical board' or contact['primary_boundary']!='visible_lower_edge_of_board_bottom_strip_at_ice' or board['band_center_is_boundary'] or contact['bottom_strip_is_ice']:raise ValueError('Existing user board facts differ from the implemented source boundary meaning')
  if visual['primary_visual_boundary']!='actual_printed_graphic_outer_contours' or visual['visible_rectangle_required']:raise ValueError('Existing printed-contour user facts differ')
  self.circular_lines=set(paint['finite_width_circular_lines']['ids']);self.target_rings=set(paint['target_rings']['ids'])
  if self.circular_lines & self.target_rings:raise ValueError('Circular width lines cannot also be target rings')
  self.curve_operators={}
  for name in names:
   if name.startswith('source_material_operator_corrections'):
    for r in self.documents[name]['records']:
     key=r['curve_id'];value=dict(operator=r['confirmed_measurement_operator'],signal=r.get('confirmed_material_signal'),authority=name,definition=r.get('source_material_definition'))
     if key in self.curve_operators and self.curve_operators[key]['operator']!=value['operator']:raise ValueError('Conflicting recorded source operators: '+key)
     self.curve_operators[key]=value
  self.digest=object_sha256(self.references)
 def verify_unchanged(self):
  for ref in self.references.values():
   if file_sha256(ref['path'])!=ref['sha256']:raise ValueError('Confirmed source facts changed within a source/admission batch')
  return True
 def expected(self,target,category=None):
  target=str(target or '');short=target[4:] if target.startswith('ice.') else target;feature=short.split(':')[0];result=None
  if target.startswith('board.top'):
   result=dict(operator='board_upper_material_edge',physical_role='blue_band_UPPER_outer_edge',model_support='existing common board perimeter at height H',required_RGB='outside/background -> blue band -> board body; actual uppermost blue edge, no inner interface',authority='user_board_edge_semantics_20261010.json')
  elif target.startswith('board.contact'):
   result=dict(operator='ice_board_interface',physical_role='yellow_band_LOWER_outer_edge',model_support='existing common board perimeter at ice height zero',required_RGB='board body -> yellow strip -> actual ice, lower strip edge with reflection excluded',authority='ice_contact_material_roles_user_confirmed_20261006.json')
  elif target in ('board.blue_cap_body_interface','board.yellow_strip_body_interface'):
   result=dict(operator=None,prohibited_as_board_boundary=True,physical_role='internal_band_body_interface_not_user_board_boundary',authority='user_board_edge_semantics_20261010.json')
  elif feature in self.circular_lines:
   result=dict(operator='paint_center',physical_role='finite_width_circular_paint_midline',required_RGB='two actual distinct flanks of the same painted stroke; midpoint, no switch to a ring exterior',authority='paint_roles_1018.json')
  elif feature in self.target_rings:
   result=dict(operator='material_interface',physical_role='declared_nested_target_material_interface',required_RGB='declared physical ring material pair and layer, not an arbitrary nearby gradient',authority='paint_roles_1018.json')
  elif short in self.curve_operators:
   result=dict(self.curve_operators[short],physical_role='confirmed_actual_paint_operator',required_RGB='same confirmed material signal and both stroke flanks in current programme source')
  elif target.startswith('ice.') and (feature in ('B1','B2','C') or re.fullmatch(r'lane[1-4]',feature)):
   result=dict(operator='paint_center',physical_role='ordinary_finite_width_paint_midline',required_RGB='two distinct material flanks of the same ordinary stroke, with declared colour/polarity; midpoint',authority='user_board_edge_semantics_20261010.json')
  elif category=='ad_corner' or re.fullmatch(r'Ad[0-9]+[.](left_top|right_top|left_bottom|right_bottom)',target):
   result=dict(operator='physical_corner',physical_role='fixed_actual_poster_corner',required_RGB='two corresponding actual material edges of the same physical installed panel, actual intersection; never sliding',authority='visual_boundary_roles_user_confirmed_20261006.json')
  elif category=='ad_edge' or re.fullmatch(r'Ad[0-9]+[.](top_edge|bottom_edge|left_vertical|right_vertical)',target):
   result=dict(operator='poster_boundary',physical_role='actual_installed_panel_edge',required_RGB='actual visible enclosing material edge; occupied extent is not an edge',authority='visual_boundary_roles_user_confirmed_20261006.json')
  elif category=='ad_print' or '.print.' in target:
   result=dict(operator='ink_boundary',physical_role='actual_printed_component_or_separate_hole_contour',required_RGB='every retained contour segment supported by the same physical printed material; no bridge between components or fabricated box',authority='visual_boundary_roles_user_confirmed_20261006.json')
  if result is not None:
   result['authority_reference']=self.references[result['authority']];result['contract_sha256']=self.digest;result['definition_match_is_not_material_identity_or_pixel_accuracy']=True
  return result
 def inspect(self,row):
  expected=self.expected(row.get('target_id'),row.get('category'));raw=row.get('measurement_operator_resolved') or row.get('measurement_operator');problems=[]
  if expected is None:problems.append('physical_measurement_definition_not_closed_against_user_facts')
  elif expected.get('prohibited_as_board_boundary'):problems.append('withdrawn_inner_board_interface_not_user_outer_boundary')
  elif raw!=expected['operator']:problems.append('actual_operator_conflicts_with_existing_user_measurement_definition')
  audit=row.get('source_measurement_audit') or {}
  if expected and expected.get('operator')=='paint_center' and row.get('current_uv') is not None and row.get('evidence_class')!='manual_native_click':
   sides=audit.get('flank_uv')
   if not isinstance(sides,list) or len(sides)!=2:problems.append('actual_midline_lacks_two_current_material_flanks')
  return dict(source_row_id=row.get('source_row_id'),target_id=row.get('target_id'),native_pts=row.get('native_pts'),actual_operator=raw,expected=expected,definition_problems=problems,current_RGB_material_identity_proof_audited_here=False,physical_source_approved=False)

 def seed_errors(self,row,decision):
  """A navigation seed must already have the same actual source contract.

  This never upgrades a definition/name match to RGB proof. The full source
  promotion verifies the bound source artifacts before publishing its receipt.
  """
  check=self.inspect(row);errors=list(check['definition_problems']);expected=check['expected'];certificate=row.get('measurement_contract_certificate') or {}
  if decision.get('solver_eligible') is not True or decision.get('admission_status')!='admitted' or decision.get('promotion_errors'):errors.append('seed_not_in_final_successful_source_admission')
  if decision.get('physical_identity_passed') is not True or decision.get('current_native_source_support_passed') is not True:errors.append('seed_current_physical_source_not_approved')
  if decision.get('measurement_contract_sha256')!=self.digest or certificate.get('contract_sha256')!=self.digest:errors.append('seed_not_bound_to_existing_user_fact_contract')
  if (certificate.get('source_row_id')!=row.get('source_row_id') or certificate.get('native_pts')!=row.get('native_pts') or certificate.get('joint_material_id')!=row.get('joint_material_id')):errors.append('seed_material_contract_row_binding_mismatch')
  if expected and (certificate.get('operator')!=expected.get('operator') or certificate.get('physical_role')!=expected.get('physical_role')):errors.append('seed_source_certificate_wrong_material_transition')
  current=object_sha256(dict(native_pts=row.get('native_pts'),current_uv=row.get('current_uv'),source_row_id=row.get('source_row_id')))
  if certificate.get('current_measurement_sha256')!=current:errors.append('seed_semantic_proof_not_for_current_actual_UV')
  if certificate.get('actual_RGB_material_identity_verified') is not True or certificate.get('operator_selected_correct_material_transition') is not True or certificate.get('pixels_from_model') is not False:errors.append('seed_missing_actual_RGB_material_operator_proof')
  if not certificate.get('source_evidence'):errors.append('seed_missing_bound_current_source_artifacts')
  if decision.get('measurement_contract_certificate_sha256')!=object_sha256(certificate):errors.append('seed_semantic_proof_not_bound_to_final_promotion')
  if not isinstance(decision.get('receipt_sha256'),str) or len(decision['receipt_sha256'])!=64:errors.append('seed_missing_final_promotion_receipt_SHA')
  return sorted(set(errors))
