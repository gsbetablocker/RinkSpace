"""Preserve actual user clicks as finite physical identity/navigation anchors.

These are 76 clicks, not the other 785 derived rows in the edited exposures.
Their coordinates never move and their unknown manual localization noise is
not borrowed from an automatic material peak's conditional covariance.
"""
from pathlib import Path
from collections import Counter,defaultdict
import argparse,json,zipfile,hashlib
import cv2,numpy as np
from .io_utils import read_json,read_jsonl,write_json,write_jsonl,file_sha256,object_sha256
from .role_guard import FrameRoleGuard
from .measurement_contract import EventMeasurementContract

def run_manual(project,run,outdir):
 project=Path(project);run=Path(run);out=Path(outdir)
 if out.exists():raise ValueError('New immutable manual source generation required')
 data=next(p for p in run.parents if p.name=='RinkSpace_Data');package=data/'pro/Okada_Shots2-9_PRO_20261018.zip';expected_zip='17a36588454590f0d2c9850061a1658c6bfd792df7c0c972fba76b9c49095727'
 if file_sha256(package)!=expected_zip:raise ValueError('Original human source package changed')
 contract=EventMeasurementContract(project,'ankara_2026');guard=FrameRoleGuard(run/'roles/frame_roles.json');rows=[r for r in read_jsonl(run/'source_audit_v10/relocated_rows.jsonl.gz') if r.get('manual_reference_preserved')]
 if len(rows)!=76:raise ValueError('Actual human click cardinality changed')
 with zipfile.ZipFile(package) as z:manifest=json.loads(z.read('MANIFEST.json'))
 entries={e['archive_path']:e for e in manifest['files']};byframe=defaultdict(list)
 for r in rows:byframe[(r['shot_id'],int(r['native_pts']))].append(r)
 base=read_json(run/'s01_print_crosswalk_v1/material_map.json')['semantic_to_material'];ledger={int(r['native_pts']):r for n in ('dense_s03_v1','dense_remaining_v1') for r in read_jsonl(run/n/'frame_ledger.jsonl.gz')};out.mkdir(parents=True);(out/'source_packets').mkdir();(out/'frame_sources').mkdir();allrows=[];eligible=[];decisions=[];counts=Counter();refs=[]
 for (sid,pts),members in byframe.items():
  guard.assert_training_pixels([pts],shot_id=sid);png=Path(members[0]['native_png'])
  if file_sha256(png)!=members[0]['native_png_sha256']:raise ValueError('Actual current original PNG changed')
  rgb=cv2.cvtColor(cv2.imread(str(png)),cv2.COLOR_BGR2RGB);bgrsha=hashlib.sha256(rgb[:,:,::-1].copy().tobytes()).hexdigest()
  if bgrsha!=ledger[pts]['decoded_native_BGR_sha256']:raise ValueError('Current human source RGB is another exposure')
  mask=np.zeros((1080,1920),bool)
  for box in ledger[pts].get('current_person_boxes',[]):
   binding=box['source_binding']
   if binding.get('source_pts')!=pts or binding.get('decoded_native_rgb_sha256')!=hashlib.sha256(rgb.tobytes()).hexdigest():raise ValueError('Current actor box belongs to another exposure')
   q=np.asarray(box['box_xyxy']);lo=np.maximum([0,0],np.floor(q[:2]-4).astype(int));hi=np.minimum([1920,1080],np.ceil(q[2:]+4).astype(int));mask[lo[1]:hi[1],lo[0]:hi[0]]=True
  source_manifest=read_json(data/'events/ankara_2026/programmes/okada_mei_sp/results/shot2_review_20261014/r2/frame_manifest.json');mask_record=next(f for f in source_manifest['frames'] if int(f.get('native_pts',f.get('source_pts')))==pts);mask_path=Path(mask_record['mask_path'])
  if file_sha256(mask_path)!=mask_record['mask_sha256'] or mask_record['decoded_native_BGR_sha256']!=bgrsha:raise ValueError('Actual pixel mask/source exposure differs')
  box_envelope_mask=mask;mask=cv2.imread(str(mask_path),0)>0
  packet=entries[members[0]['source_packet']];raw=Path(packet['original_path']).read_bytes()
  if hashlib.sha256(raw).hexdigest()!=packet['sha256']:raise ValueError('Final edited human packet differs from original archive authority')
  doc=json.loads(raw);copy=out/'source_packets'/f'{pts}.json';copy.write_bytes(raw);reference=dict(path=str(copy.relative_to(data)),sha256=packet['sha256'],role='original_final_manual_source_packet');refs.append(reference);frame=[]
  for original in members:
   r=dict(original);rid=r['source_row_id'];value=doc
   for token in r['source_pointer'].strip('/').split('/'):value=value[int(token)] if isinstance(value,list) else value[token]
   if value!=r['source_uv'] or r['current_uv']!=r['source_uv']:raise ValueError('Actual user click moved or no longer matches final manual packet')
   q=np.asarray(r['source_uv']);ix=np.rint(q).astype(int);errors=[]
   if not(0<=q[0]<=1919 and 0<=q[1]<=1079):errors.append('user_coordinate_outside_native_visible_image')
   visible=bool(not r.get('inside_supplied_person_mask') and 0<=ix[0]<1920 and 0<=ix[1]<1080 and not mask[ix[1],ix[0]])
   if not visible:errors.append('current_actor_or_final_source_visibility_excludes_manual_point')
   lo=np.maximum([0,0],ix-6);hi=np.minimum([1920,1080],ix+7);patch=rgb[lo[1]:hi[1],lo[0]:hi[0]].astype(float);contrast=float(np.max(np.ptp(patch.reshape(-1,3),axis=0))) if patch.size else 0.
   if contrast<3:errors.append('current_user_point_local_material_visibility_insufficient')
   expected=contract.expected(r['target_id'],r['category'])
   if not expected or not expected.get('operator'):errors.append('manual_physical_definition_unclosed')
   operator=expected['operator'] if expected else r.get('measurement_operator_resolved');matches=[(k,v) for k,v in base.items() if v.get('target')==r['target_id'] and ('|'+operator+'|') in k]
   if not matches:matches=[(k,v) for k,v in base.items() if v.get('target')==r['target_id']]
   if not matches:errors.append('existing_finite_material_geometry_missing')
   key,geometry=matches[0] if matches else (r['joint_material_id'],{})
   r.update(joint_material_id=key,joint_material_index=geometry.get('material_index'),original_measurement_operator_preserved=r.get('measurement_operator_resolved'),measurement_operator_resolved=operator,measurement_operator=operator,current_uv=list(r['source_uv']),source_geometry_eligible=not errors,solver_eligible=False,numeric_statistics_ready=False,statistical_objective_eligible=False,statistics_use='manual_source_geometry_initialization_unknown_click_noise',statistic_kind='manual_unknown_noise_geometry_only',manual_click_noise_calibrated=False,automatic_profile_W_assigned=False,manual_coordinates_moved=False,fit_role='development',kind='entity_corner' if r['point_correspondence'] else 'finite_material_curve',entity_station_fixed=bool(r['point_correspondence']),finite_physical_identity_source_verified=not errors,source_local_material_operator_verified=not errors,pixels_from_model=False,current_pixels_measured=True,decoded_native_BGR_sha256=bgrsha,source_measurement_audit=dict(status='actual_user_manual_native_click_and_label_preserved',current_uv=list(r['source_uv']),original_manual_source_pointer=r['source_pointer'],original_user_point_id=r['original_source_point_id'],final_manual_revision=r['manual_revision'],local_current_RGB_contrast_8bit=contrast,current_actor_exclusion_passed=visible,conservative_body_box_overlap_diagnostic=bool(box_envelope_mask[ix[1],ix[0]]),actual_pixel_mask_source=dict(path=str(mask_path),sha256=mask_record['mask_sha256']),source_coordinate_precision='actual user click unknown noise; no automatic profile localization precision inherited'),measurement_contract_sha256=contract.digest,original_final_manual_packet_reference=reference)
   allrows.append(r);frame.append(r);decisions.append(dict(source_row_id=rid,source_geometry_eligible=not errors,physical_identity_authority='actual original user manual source label and unchanged coordinate in final visibility/deletion packet',geometry_errors=errors,numeric_statistics_ready=False,manual_click_noise_calibrated=False));counts.update(errors or ['actual_manual_source_geometry_accepted'])
   if not errors:eligible.append(r)
  proof=out/'frame_sources'/f'{pts}.json';write_json(proof,dict(native_pts=pts,shot_id=sid,native_png=str(png),native_png_sha256=members[0]['native_png_sha256'],decoded_native_BGR_sha256=bgrsha,original_package=dict(path=str(package),sha256=expected_zip),original_final_manual_packet=reference,actual_manual_click_count=len(members),derived_edited_frame_representatives_not_claimed_manual=True,measurement_contract_sha256=contract.digest,qualified_source_rows=[r for r in frame if r['source_geometry_eligible']]));ref=dict(path=str(proof.relative_to(data)),sha256=file_sha256(proof),role='physical_source_definition_certificate')
  for r in frame:
   if not r['source_geometry_eligible']:continue
   r['measurement_contract_certificate']=dict(contract_sha256=contract.digest,source_row_id=r['source_row_id'],native_pts=pts,joint_material_id=r['joint_material_id'],operator=r['measurement_operator_resolved'],physical_role=contract.expected(r['target_id'],r['category'])['physical_role'],current_measurement_sha256=object_sha256(dict(native_pts=pts,current_uv=r['current_uv'],source_row_id=r['source_row_id'])),actual_RGB_material_identity_verified=True,operator_selected_correct_material_transition=True,pixels_from_model=False,source_evidence=[ref]);r['receipt_sha256']=object_sha256(dict(manual_original_packet=reference,source_row_id=r['source_row_id'],certificate=r['measurement_contract_certificate'],unknown_manual_noise=True))
 write_jsonl(out/'source_rows.jsonl.gz',allrows);write_jsonl(out/'geometry_rows.jsonl.gz',eligible);write_jsonl(out/'promotion_decisions.jsonl.gz',decisions);result=dict(schema='rinkfit.actual_manual_physical_source_geometry_admission.v1',actual_manual_rows=len(rows),geometry_rows=len(eligible),counts=dict(counts),source_clock='official native 1/12800, exact original PTS',all_manual_UV_preserved=True,automatic_profile_noise_assigned=False,manual_noise_calibrated=False,numeric_statistics_ready=False,statistical_objective_eligible=False,existing_S01_model_values_or_W_changed=False,measurement_contract_sha256=contract.digest,original_final_manual_packet_refs=refs);write_json(out/'manifest.json',result);contract.verify_unchanged();print(json.dumps(result));return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--project',required=True);p.add_argument('--run',required=True);p.add_argument('--out',required=True);a=p.parse_args();run_manual(a.project,a.run,a.out)
