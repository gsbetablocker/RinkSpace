"""Current exact RGB board-instance local source batch, geometry first."""
from pathlib import Path
import argparse,json,hashlib,os,shutil
from collections import Counter
import cv2,numpy as np
from .io_utils import read_json,read_jsonl,write_json,write_jsonl,file_sha256,object_sha256
from .role_guard import FrameRoleGuard
from .measurement_contract import EventMeasurementContract
from .board_outer_source import source_outer_boundary_rows
from .native_source import iter_training_native_rgb
from .current_person_exclusion import current_person_exclusion_mask

SOURCE_RECORD_KEYS=('source_row_id','native_pts','shot_id','target_id','joint_material_id','joint_material_index','source_uv','current_uv','normal','navigation_seed_uv','measurement_operator_resolved','source_measurement_audit')

def source_record_payload(row):
 return {k:row[k] for k in SOURCE_RECORD_KEYS}

def compact_frame_profiles(rows,out,pts):
 if not rows:return
 p=out/'profiles'/f'{pts}.npz';p.parent.mkdir(exist_ok=True)
 arrays=np.asarray([r['source_measurement_audit']['profile']['intensity'] for r in rows],np.float64);np.savez_compressed(p,actual_signed_material_profile=arrays);ref=dict(path=str(p),sha256=file_sha256(p),array='actual_signed_material_profile')
 for i,row in enumerate(rows):
  audit=row['source_measurement_audit'];v=audit['profile'].pop('intensity');w=audit['continuous_cubic_material_profile'].pop('actual_profile_intensity')
  if not np.array_equal(np.asarray(v),np.asarray(w)):raise ValueError('Actual profile arrays differ')
  q=dict(ref,row_index=i);audit['profile']['intensity_reference']=q;audit['continuous_cubic_material_profile']['actual_profile_intensity_reference']=q

def link_or_copy(source,destination):
 try:os.link(source,destination);return 'immutable_hardlink'
 except OSError:shutil.copyfile(source,destination);return 'compressed_byte_copy_hardlink_unavailable'

def measure(project,run,review,outdir):
 project=Path(project);run=Path(run);review=Path(review);out=Path(outdir);data=next(p for p in run.parents if p.name=='RinkSpace_Data');contract=EventMeasurementContract(project,'ankara_2026');guard=FrameRoleGuard(run/'roles/frame_roles.json');doc=read_json(review);reviewsha=file_sha256(review)
 if out.exists():raise ValueError('New immutable source run required')
 out.mkdir(parents=True);(out/'frame_sources').mkdir();(out/'overlays').mkdir();(out/'current_masks').mkdir();maskframes={};rows=[];ledger=[];counts=Counter();refs=[];frame_lookup={int(f['native_pts']):f for f in doc['frames']};native_ledger={int(f['native_pts']):f for n in ('dense_s03_v1','dense_remaining_v1')for f in read_jsonl(run/n/'frame_ledger.jsonl.gz')}
 for sid in {f['shot_id'] for f in doc['frames']}:
  n=int(sid[1:]);mp=data/'events/ankara_2026/programmes/okada_mei_sp/results'/f'shot{n}_review_20261014';mp=mp/'r2/frame_manifest.json'if n==2 else mp/'frame_manifest.json'
  for fr in read_json(mp)['frames']:maskframes[(sid,int(fr.get('native_pts',fr.get('source_pts'))))]=fr
 for pts,rgb,native_receipt in iter_training_native_rgb(run,frame_lookup):
  fr=frame_lookup[pts];sid=fr['shot_id'];guard.assert_training_pixels([pts],shot_id=sid);png=None;bgr=np.ascontiguousarray(rgb[:,:,::-1]);bgrsha=native_receipt['decoded_native_BGR_sha256'];rgbsha=native_receipt['decoded_native_RGB_sha256']
  if bgrsha!=fr['decoded_native_BGR_sha256']:raise ValueError('Actual native navigation exposure differs')
  context=dict(native_pts=pts,shot_id=sid,actual_native_RGB_sha256=rgbsha,source_review=dict(path=str(review),sha256=reviewsha),native_video_decode_receipt=native_receipt,evidence_directory=str(out/'material_roles'/str(pts)))
  mf=maskframes.get((sid,pts))
  if mf is not None:
   maskpath=Path(mf.get('mask_path')or mf['source']['mask_path']);masksha=mf.get('mask_sha256')or mf.get('source',{}).get('mask_sha256')
   if masksha and file_sha256(maskpath)!=masksha:raise ValueError('Actual source exclusion changed')
   if mf.get('decoded_native_BGR_sha256')and mf['decoded_native_BGR_sha256']!=bgrsha:raise ValueError('Actual source mask differs')
   mask=cv2.imread(str(maskpath),0)>0
  else:
   mask,maskreceipt=current_person_exclusion_mask(rgb,native_ledger[pts].get('current_person_boxes',[]),native_pts=pts,current_board_context=context);maskpath=out/'current_masks'/f'{pts}.png';cv2.imwrite(str(maskpath),mask.astype(np.uint8)*255)
  # Existing graphics are not transformed into scene points.
  mask[1000:,:]=True
  rr,certs=source_outer_boundary_rows(rgb,mask,current_board_context=context);frame_rows=[]
  if any(r.get('source_material_role_requalification',{}).get('status')!='pass' for r in rr):raise ValueError('Unqualified board material must not be auto-promoted')
  for i,r in enumerate(rr):
   role=r['material_role'];short='top'if role=='board.top'else'contact';operator=r['measurement_operator'];rid=f'board_instance_native/{sid}/{pts}/{short}/{i}';key='board.top|board|board_upper_material_edge|top'if short=='top'else'board.contact|board|ice_board_interface|contact'
   row=dict(r,source_row_id=rid,shot_id=sid,native_pts=pts,time_base='1/12800',source_family_id=f'board_instance_native/{sid}/{pts}/{short}',joint_material_id=key,joint_material_index=348 if short=='top' else 349,measurement_operator_resolved=operator,category='board_edge',physical_surface='board',point_correspondence=False,kind='finite_material_curve',entity_station_fixed=False,source_acquisition_mode='current_actual_board_instance_local_material_remeasurement',current_pixels_measured=True,pixels_from_model=False,source_identity_basis='current_native_RGB_board_object_review_and_same_local_blue_upper_yellow_lower_material_order',decoded_native_BGR_sha256=bgrsha,native_png_sha256=fr['native_png_sha256'],source_geometry_eligible=True,solver_eligible=False,statistical_objective_eligible=False,numeric_statistics_ready=False,statistics_use='explicit_source_family_balanced_geometry_initialization_only',fit_role='development',role='development',anchor_pts=pts,anchor_source_row_id=None,source_family_role_locked=True,actual_source_mean_is_continuous_cubic=True,old_recalled_source_UV_or_W_reused=False)
   frame_rows.append(row)
  compact_frame_profiles(frame_rows,out,pts)
  proofpath=out/'frame_sources'/f'{pts}.json';proof=dict(schema='rinkfit.actual_board_instance_current_material_source.compact.v1',native_pts=pts,shot_id=sid,native_png=None,native_png_sha256=None,native_video_decode_receipt=native_receipt,actual_native_RGB_sha256=rgbsha,decoded_native_BGR_sha256=bgrsha,current_mask=dict(path=str(maskpath),sha256=file_sha256(maskpath)),current_board_instance_context=context,source_definition_certificates=certs,row_records_sha256={r['source_row_id']:object_sha256(source_record_payload(r))for r in frame_rows},row_payload_schema='rinkfit.current_board_readout_payload.v1',actual_profiles_storage='single_per_exposure_compressed_NPZ',measurement_contract_sha256=contract.digest,source_pixels_from_model=False,source_rows=len(frame_rows));write_json(proofpath,proof);proofref=dict(path=str(proofpath.relative_to(data)),sha256=file_sha256(proofpath),role='physical_source_definition_certificate');refs.append(dict(path=str(proofpath),sha256=proofref['sha256']))
  overlay=bgr.copy()
  for row in frame_rows:
   cert=dict(contract_sha256=contract.digest,source_row_id=row['source_row_id'],native_pts=pts,joint_material_id=row['joint_material_id'],operator=row['measurement_operator_resolved'],physical_role=contract.expected(row['target_id'])['physical_role'],current_measurement_sha256=object_sha256(dict(native_pts=pts,current_uv=row['current_uv'],source_row_id=row['source_row_id'],measurement_operator=row['measurement_operator_resolved'])),actual_RGB_material_identity_verified=row.get('source_material_role_requalification',{}).get('status')=='pass',operator_selected_correct_material_transition=False,operator_qualification_status='pending_frozen_operator_transition_compatibility',pixels_from_model=False,source_evidence=[proofref],actual_source_certificate_index=row['source_definition_certificate_index']);row['measurement_contract_certificate']=cert;row['receipt_sha256']=object_sha256(dict(source_row_id=row['source_row_id'],measurement_contract_certificate=cert,source_mean=row['current_uv'],source_family_id=row['source_family_id']));row['source_packet']=str(proofpath);row['source_packet_sha256']=proofref['sha256'];rows.append(row);q=np.rint(row['current_uv']).astype(int);cv2.circle(overlay,tuple(q),3,(0,0,255)if row['material_role']=='board.top'else(0,255,0),-1)
  
  if sid not in counts:cv2.imwrite(str(out/'overlays'/f'{sid}_{pts}.png'),overlay)
  counts[sid]+=len(frame_rows);ledger.append(dict(native_pts=pts,shot_id=sid,current_board_instance_context_type=doc['schema'],current_material_rows=len(frame_rows),status='actual_local_material_sources'if frame_rows else'actual_board_instance_visible_but_material_profile_pending',actual_source_proof=refs[-1]));print(json.dumps(ledger[-1]),flush=True)
 write_jsonl(out/'source_rows.jsonl.gz',rows);row_storage=link_or_copy(out/'source_rows.jsonl.gz',out/'geometry_rows.jsonl.gz');write_jsonl(out/'frame_ledger.jsonl.gz',ledger);write_jsonl(out/'promotion_decisions.jsonl.gz',(dict(source_row_id=r['source_row_id'],source_geometry_eligible=False,solver_eligible=False,numeric_statistics_ready=False,physical_identity_passed=r.get('source_material_role_requalification',{}).get('status')=='pass',operator_qualification_status='pending_frozen_operator_transition_compatibility',statistical_objective_eligible=False,measurement_contract_sha256=contract.digest,measurement_contract_certificate_sha256=object_sha256(r['measurement_contract_certificate']),receipt_sha256=r['receipt_sha256'])for r in rows));result=dict(schema='rinkfit.actual_local_board_instance_source_batch.v1',frames=len(ledger),actual_source_rows=len(rows),by_shot=dict(counts),source_frame_proofs=refs,source_object_review=dict(path=str(review),sha256=reviewsha),all_requested_frames_accounted=True,compact_source_storage=True,per_frame_RGB_saved=False,source_geometry_storage=row_storage,old_global_source_physical_pass_or_UV_or_W_inherited=False,all_new_mean_from_current_material_RGB=True,numeric_statistics_ready=False,geometry_initialization_only=True,measurement_contract_sha256=contract.digest);write_json(out/'manifest.json',result);contract.verify_unchanged();print(json.dumps(result));return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--project',required=True);p.add_argument('--run',required=True);p.add_argument('--review',required=True);p.add_argument('--out',required=True);a=p.parse_args();measure(a.project,a.run,a.review,a.out)
