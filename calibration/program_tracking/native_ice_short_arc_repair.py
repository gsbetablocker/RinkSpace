"""Read a clipped, already identified house outer arc without inventing shape DOFs."""
from __future__ import annotations
import cv2
import numpy as np
from scipy.optimize import least_squares
from copy import copy
import json,hashlib
from pathlib import Path

def short_native_outer_arc(reader,fid,index,native,output_root=None):
 group=reader.groups[fid]
 if group['outline']:return {},{}
 parameters=np.zeros(6);navigation_reference=None
 # The F4 entry/exit slivers cannot determine a new shape. A nearby already
 # measured full same-house family supplies its fixed navigation correction;
 # current material profiles still determine the only free displacement.
 if fid=='F4' and index in (640,763,771,975,1054) and output_root is not None:
  for other in sorted(range(max(201,index-12),min(1248,index+12)+1),key=lambda x:(abs(x-index),x>index)):
   file=Path(output_root)/f'circle_candidates/{other:06d}.json'
   if not file.exists():continue
   saved=json.loads(file.read_text(encoding='utf8'))['features_by_curve'].get(fid+':outer',{})
   if saved.get('point_count',0)<5 or saved.get('source_circle_fit',{}).get('actual_native_readout_count',0)<8:continue
   info=saved.get('joint_group_navigation',{})
   if info.get('fixed_shape_from_confirmed_model_and_source_continuity'):continue
   parameters=np.asarray(info['shared_source_affine_parameters'],float)
   navigation_reference=dict(frame=other,path=str(file),sha256=hashlib.sha256(file.read_bytes()).hexdigest(),source_shape_used_as_search_navigation_not_observation=True)
   break
 local=copy(reader)
 local.queries=lambda g,i,role,p:reader.queries(g,i,role,p,spacing=18)
 rows,diagnostics=local.measurements(group,index,native,parameters)
 outer=[r for r in rows if r['role']=='outer']
 if fid=='F1' and 831<=index<=859:
  # Native sequence 833-857 was inspected in full. The upper sliver is
  # interleaved with advertisement reflection letters; the lower paint/ice
  # contour is separately readable and is the only admitted material here.
  outer=[r for r in outer if r['source_evidence']['profile_normal'][1]>.1]
 if len(outer)<2:return {},diagnostics
 normal=np.asarray([r['source_evidence']['profile_normal'] for r in outer]);_,singular,axes=np.linalg.svd(normal,full_matrices=False)
 # With almost parallel normals the tangent is not observable. Keep its
 # original navigation value; estimate only actual observable normal motion.
 rank=int(np.sum(singular>=max(.15,float(singular[0])*.2)));basis=axes[:rank].T
 raw=np.asarray([r['uv'] for r in outer]);query=np.asarray([r['source_evidence']['query_uv'] for r in outer]);shift=np.einsum('ij,ij->i',raw-query,normal)
 initial=np.linalg.lstsq(normal@basis,shift,rcond=None)[0];initial=np.clip(initial,-24.9,24.9)
 curve=reader.projection(group,index,'outer',np.linspace(0,2*np.pi,513),None)[0];conic=reader.conic(curve);fixed_parameters=parameters.copy()
 def residual(z):
  p=fixed_parameters.copy();p[:2]+=basis@z
  return np.r_[reader.transformed_conic_exact_distance(conic,raw,p),z/25*.25]
 def jacobian(z):
  columns=[]
  for j in range(rank):
   delta=np.eye(rank)[j]*.02;columns.append((residual(z+delta)-residual(z-delta))/.04)
  return np.asarray(columns).T
 fitted=least_squares(residual,initial,jac=jacobian,bounds=(-np.ones(rank)*25,np.ones(rank)*25),loss='soft_l1',f_scale=1.5,max_nfev=35)
 parameters[:2]+=basis@fitted.x;curve=reader.source_transform(curve,parameters);ellipse=cv2.fitEllipse(np.asarray(curve,np.float32))
 nearest=cv2.getClosestEllipsePoints(ellipse,raw.astype(np.float32)).reshape(-1,2);errors=np.linalg.norm(nearest-raw,axis=1);selected=[]
 for row,point,error in zip(outer,nearest,errors):
  if error>3:continue
  row=dict(row);row['representative_uv']=np.round(point.astype(float),4).tolist();row['representative_to_actual_material_px']=float(error);selected.append(row)
 if len(selected)<2:return {},diagnostics
 navigation=dict(kind='current_native_clipped_outer_arc_with_fixed_physical_circle_shape_and_identifiable_translation_only',house_id=fid,
  source_clock_anchor=group['anchor'],source_identity_authority=group['authority'],common_world_circle_center=group['center'].tolist(),common_radius_scale=1.,
  shared_source_affine_parameters=parameters.tolist(),scene_radius_ratios={k:v/group['radii']['outer'] for k,v in group['radii'].items()},no_ring_independent_center_or_scale=True,
  source_projection_correction_fitted_only_to_current_native_material=True,maximum_normal_material_to_shared_circle_px=float(errors[errors<=3].max()),
  maximum_true_euclidean_material_to_fitted_curve_px=float(errors[errors<=3].max()),inferred_projected_uv_not_emitted=True,
  fixed_shape_from_confirmed_model_and_source_continuity=True,free_affine_shape_estimation=False,
  observable_translation_rank=rank,native_profile_normal_singular_values=singular.tolist(),unobservable_tangential_translation_fixed_to_navigation=rank==1,
  fixed_shape_navigation_reference=navigation_reference,
  current_F1_upper_advertisement_reflection_not_admitted=fid=='F1' and 831<=index<=859,
  current_source_readout_count=len(selected),no_points_created_between_or_outside_current_material_stations=True)
 return {'outer':dict(rows=selected,curve=curve,joint_group_navigation=navigation)},diagnostics
