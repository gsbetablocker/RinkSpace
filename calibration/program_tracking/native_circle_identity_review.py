"""Apply localized, source-reviewed non-material exclusions before circle fitting."""
import json
import hashlib
from pathlib import Path
import cv2
import numpy as np
from copy import deepcopy
from scipy.optimize import least_squares

def refit_reviewed_current_material_rows(reader,fid,index,native,policy):
 archive=json.loads(Path(policy['same_frame_material_archive']).read_text(encoding='utf8'))
 if archive['source_native_sha256']!=hashlib.sha256(Path(archive['source_native_path']).read_bytes()).hexdigest():
  raise ValueError('Archived current-frame material source hash changed')
 feature=archive['feature'];group=reader.groups[fid]
 polygon=np.asarray(policy['native_polygon_uv'],np.float32)
 rows=[deepcopy(r) for r in feature['source_circle_fit']['direct_readouts'] if cv2.pointPolygonTest(polygon,tuple(map(float,r['uv'])),False)<0]
 curve=reader.projection(group,index,'outer',np.linspace(0,2*np.pi,513),None)[0];conic=reader.conic(curve)
 initial=np.asarray(feature['joint_group_navigation']['shared_source_affine_parameters']);bound=np.array([90.,60.,.25,.18,.06,.25]);prior=np.array([30.,20.,.05,.05,.015,.07]);uv=np.asarray([r['uv'] for r in rows])
 def residual(unit):
  z=unit*bound
  return np.r_[reader.transformed_conic_exact_distance(conic,uv,z),(z-initial)/prior*.1]
 def jacobian(unit):
  columns=[]
  for k in range(6):
   d=np.eye(6)[k]*.001;columns.append((residual(unit+d)-residual(unit-d))/.002)
  return np.asarray(columns).T
 fitted=initial.copy()
 for iteration in range(1):
  fit=least_squares(residual,fitted/bound,jac=jacobian,bounds=(-np.ones(6),np.ones(6)),loss='soft_l1',f_scale=1.5,max_nfev=200);fitted=fit.x*bound
 curve=reader.source_transform(curve,fit.x*bound);ellipse=cv2.fitEllipse(np.asarray(curve,np.float32));nearest=cv2.getClosestEllipsePoints(ellipse,uv.astype(np.float32)).reshape(-1,2);errors=np.linalg.norm(nearest-uv,axis=1);selected=[]
 for r,q,error in zip(rows,nearest,errors):
  if error>3:continue
  r['representative_uv']=np.round(q.astype(float),4).tolist();r['representative_to_actual_material_px']=float(error);selected.append(r)
 if len(selected)<3:return {},{}
 navigation=deepcopy(feature['joint_group_navigation']);navigation.update(shared_source_affine_parameters=(fit.x*bound).tolist(),maximum_normal_material_to_shared_circle_px=float(max(r['representative_to_actual_material_px'] for r in selected)),maximum_true_euclidean_material_to_fitted_curve_px=float(max(r['representative_to_actual_material_px'] for r in selected)),source_current_readouts_refitted_after_physical_identity_exclusion=True,raw_material_residuals_before_rejection_px=errors.tolist(),same_frame_material_archive=policy['same_frame_material_archive'],same_frame_material_archive_sha256=policy['same_frame_material_archive_sha256'],archived_current_measurements_not_temporal_or_projected_pixels=True)
 return {'outer':dict(rows=selected,curve=curve,joint_group_navigation=navigation)}, {'outer':dict(eligible_native_material_readouts=len(selected),native_source_review_exclusion=policy,refitted_after_excluding_unrelated_bottom_texture=True)}

def install_reviewed_circle_material_domains(reader,output_root):
 file=Path(output_root)/'circle_diagnostics/native_material_identity_domains.json'
 if reader is None or not file.exists():return
 policies=json.loads(file.read_text(encoding='utf8')).get('records',[])
 original=reader.measurements
 def measurements(group,index,native,parameters):
  rows,diagnostics=original(group,index,native,parameters)
  fid=group['paths'][0]['feature_id'];active=[p for p in policies if p['frame']==index and p['feature_id']==fid]
  if not active:return rows,diagnostics
  kept=[];rejected=[]
  for row in rows:
   reject=next((p for p in active if row['role'] in p.get('roles',['outer']) and cv2.pointPolygonTest(np.asarray(p['native_polygon_uv'],np.float32),tuple(map(float,row['uv'])),False)>=0),None)
   if reject is None:kept.append(row)
   else:rejected.append(dict(uv=row['uv'],role=row['role'],theta=row['theta'],reason=reject['reason'],source_review_path=reject['source_review_path']))
  for role in group['radii']:
   diagnostics.setdefault(role,{})['source_identity_rejected_current_native_readouts']=[r for r in rejected if r['role']==role]
   diagnostics[role]['eligible_native_material_readouts']=sum(r['role']==role for r in kept)
  return kept,diagnostics
 reader.measurements=measurements
 original_read=reader.read
 def read(fid,index,native):
  replay=next((p for p in policies if p['frame']==index and p['feature_id']==fid and p.get('same_frame_material_archive')),None)
  return refit_reviewed_current_material_rows(reader,fid,index,native,replay) if replay else original_read(fid,index,native)
 reader.read=read
