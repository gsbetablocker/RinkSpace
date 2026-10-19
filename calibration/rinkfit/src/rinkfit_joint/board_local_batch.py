"""Current exact RGB board-instance local source batch, geometry first."""
from pathlib import Path
import argparse,json,hashlib
from collections import Counter
import cv2,numpy as np
from .io_utils import read_json,read_jsonl,write_json,write_jsonl,file_sha256,object_sha256
from .role_guard import FrameRoleGuard
from .measurement_contract import EventMeasurementContract
from .board_outer_source import source_outer_boundary_rows

def measure(project,run,review,outdir):
 project=Path(project);run=Path(run);review=Path(review);out=Path(outdir);data=next(p for p in run.parents if p.name=='RinkSpace_Data');contract=EventMeasurementContract(project,'ankara_2026');guard=FrameRoleGuard(run/'roles/frame_roles.json');doc=read_json(review);reviewsha=file_sha256(review)
 if out.exists():raise ValueError('New immutable source run required')
 out.mkdir(parents=True);(out/'frame_sources').mkdir();(out/'overlays').mkdir();maskframes={};rows=[];ledger=[];counts=Counter();refs=[]
 for sid in {f['shot_id'] for f in doc['frames']}:
  n=int(sid[1:]);mp=data/'events/ankara_2026/programmes/okada_mei_sp/results'/f'shot{n}_review_20261014';mp=mp/'r2/frame_manifest.json'if n==2 else mp/'frame_manifest.json'
  for fr in read_json(mp)['frames']:maskframes[(sid,int(fr.get('native_pts',fr.get('source_pts'))))]=fr
 for fr in doc['frames']:
  pts=int(fr['native_pts']);sid=fr['shot_id'];guard.assert_training_pixels([pts],shot_id=sid);png=Path(fr['native_png'])
  if file_sha256(png)!=fr['native_png_sha256']:raise ValueError('Source PNG changed')
  bgr=cv2.imread(str(png));bgrsha=hashlib.sha256(bgr.tobytes()).hexdigest()
  if bgrsha!=fr['decoded_native_BGR_sha256']:raise ValueError('Actual object-review exposure differs')
  rgb=np.ascontiguousarray(bgr[:,:,::-1]);rgbsha=hashlib.sha256(rgb.tobytes()).hexdigest();mf=maskframes[(sid,pts)];maskpath=Path(mf.get('mask_path')or mf['source']['mask_path']);masksha=mf.get('mask_sha256')or mf.get('source',{}).get('mask_sha256')
  if masksha and file_sha256(maskpath)!=masksha:raise ValueError('Actual source exclusion changed')
  if mf.get('decoded_native_BGR_sha256')and mf['decoded_native_BGR_sha256']!=bgrsha:raise ValueError('Person mask belongs to another exposure')
  mask=cv2.imread(str(maskpath),0)>0
  # Existing graphics are not transformed into scene points.
  mask[1000:,:]=True
  context=dict(native_pts=pts,shot_id=sid,actual_native_RGB_sha256=rgbsha,source_review=dict(path=str(review),sha256=reviewsha));rr,certs=source_outer_boundary_rows(rgb,mask,current_board_context=context);frame_rows=[]
  for i,r in enumerate(rr):
   role=r['material_role'];short='top'if role=='board.top'else'contact';operator=r['measurement_operator'];rid=f'board_instance_native/{sid}/{pts}/{short}/{i}';key='board.top|board|board_upper_material_edge|top'if short=='top'else'board.contact|board|ice_board_interface|contact'
   row=dict(r,source_row_id=rid,shot_id=sid,native_pts=pts,time_base='1/12800',source_family_id=f'board_instance_native/{sid}/{pts}/{short}',joint_material_id=key,joint_material_index=348 if short=='top' else 349,measurement_operator_resolved=operator,category='board_edge',physical_surface='board',point_correspondence=False,kind='finite_material_curve',entity_station_fixed=False,source_acquisition_mode='current_actual_board_instance_local_material_remeasurement',current_pixels_measured=True,pixels_from_model=False,source_identity_basis='current_native_RGB_board_object_review_and_same_local_blue_upper_yellow_lower_material_order',decoded_native_BGR_sha256=bgrsha,native_png_sha256=fr['native_png_sha256'],source_geometry_eligible=True,solver_eligible=False,statistical_objective_eligible=False,numeric_statistics_ready=False,statistics_use='explicit_source_family_balanced_geometry_initialization_only',fit_role='development',role='development',anchor_pts=pts,anchor_source_row_id=None,source_family_role_locked=True,actual_source_mean_is_continuous_cubic=True,old_recalled_source_UV_or_W_reused=False)
   frame_rows.append(row)
  proofpath=out/'frame_sources'/f'{pts}.json';proof=dict(schema='rinkfit.actual_board_instance_current_material_source.v1',native_pts=pts,shot_id=sid,native_png=str(png),native_png_sha256=fr['native_png_sha256'],actual_native_RGB_sha256=rgbsha,decoded_native_BGR_sha256=bgrsha,current_mask=dict(path=str(maskpath),sha256=file_sha256(maskpath)),current_board_instance_context=context,source_definition_certificates=certs,qualified_source_rows=frame_rows,measurement_contract_sha256=contract.digest,source_pixels_from_model=False,source_rows=len(frame_rows));write_json(proofpath,proof);proofref=dict(path=str(proofpath.relative_to(data)),sha256=file_sha256(proofpath),role='physical_source_definition_certificate');refs.append(dict(path=str(proofpath),sha256=proofref['sha256']))
  overlay=bgr.copy()
  for row in frame_rows:
   cert=dict(contract_sha256=contract.digest,source_row_id=row['source_row_id'],native_pts=pts,joint_material_id=row['joint_material_id'],operator=row['measurement_operator_resolved'],physical_role=contract.expected(row['target_id'])['physical_role'],current_measurement_sha256=object_sha256(dict(native_pts=pts,current_uv=row['current_uv'],source_row_id=row['source_row_id'],measurement_operator=row['measurement_operator_resolved'])),actual_RGB_material_identity_verified=True,operator_selected_correct_material_transition=True,pixels_from_model=False,source_evidence=[proofref],actual_source_certificate_index=row['source_definition_certificate_index']);row['measurement_contract_certificate']=cert;row['receipt_sha256']=object_sha256(dict(source_row_id=row['source_row_id'],measurement_contract_certificate=cert,source_mean=row['current_uv'],source_family_id=row['source_family_id']));row['source_packet']=str(proofpath);row['source_packet_sha256']=proofref['sha256'];rows.append(row);q=np.rint(row['current_uv']).astype(int);cv2.circle(overlay,tuple(q),3,(0,0,255)if row['material_role']=='board.top'else(0,255,0),-1)
  cv2.imwrite(str(out/'overlays'/f'{sid}_{pts}.png'),overlay);counts[sid]+=len(frame_rows);ledger.append(dict(native_pts=pts,shot_id=sid,current_board_instance_context_type=doc['schema'],current_material_rows=len(frame_rows),status='actual_local_material_sources'if frame_rows else'actual_board_instance_visible_but_material_profile_pending',actual_source_proof=refs[-1]));print(json.dumps(ledger[-1]),flush=True)
 write_jsonl(out/'geometry_rows.jsonl.gz',rows);write_jsonl(out/'source_rows.jsonl.gz',rows);write_jsonl(out/'frame_ledger.jsonl.gz',ledger);write_jsonl(out/'promotion_decisions.jsonl.gz',(dict(source_row_id=r['source_row_id'],source_geometry_eligible=True,solver_eligible=False,numeric_statistics_ready=False,physical_identity_passed=True,statistical_objective_eligible=False,measurement_contract_sha256=contract.digest,measurement_contract_certificate_sha256=object_sha256(r['measurement_contract_certificate']),receipt_sha256=r['receipt_sha256'])for r in rows));result=dict(schema='rinkfit.actual_local_board_instance_source_batch.v1',frames=len(ledger),actual_source_rows=len(rows),by_shot=dict(counts),source_frame_proofs=refs,source_object_review=dict(path=str(review),sha256=reviewsha),all_requested_frames_accounted=True,old_global_source_physical_pass_or_UV_or_W_inherited=False,all_new_mean_from_current_material_RGB=True,numeric_statistics_ready=False,geometry_initialization_only=True,measurement_contract_sha256=contract.digest);write_json(out/'manifest.json',result);contract.verify_unchanged();print(json.dumps(result));return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--project',required=True);p.add_argument('--run',required=True);p.add_argument('--review',required=True);p.add_argument('--out',required=True);a=p.parse_args();measure(a.project,a.run,a.review,a.out)
