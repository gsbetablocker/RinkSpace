"""Source-only automatic recovery of remaining fixed advertisement entities.

Current corner and edge navigation is photometric. Shared material corners keep
exactly their established physical identity. No camera/world residual is read.
"""
from pathlib import Path
from collections import defaultdict,Counter
import argparse,json
import cv2,numpy as np
from PIL import Image
from scipy.optimize import root as solve_root
from .io_utils import read_json,read_jsonl,write_json,write_jsonl,file_sha256
from .entity_corners import EDGE_ROLES
from .source_worker import _channels,_remeasure
from .role_guard import FrameRoleGuard

PAIRS={'top_edge':('left_top','right_top'),'bottom_edge':('left_bottom','right_bottom'),'left_vertical':('left_top','left_bottom'),'right_vertical':('right_top','right_bottom')}
CONFIG={'version':'photometric_corner_and_ray_refit_v1','inward_stations_px':[3.,6.,9.,12.,15.,18.,21.,24.,27.],'ray_angle_offsets_degrees':[-8.,-4.,0.,4.,8.],'native_corner_search_window':4,'native_corner_max_move_px':5.,'edge_minimum_actual_profiles':5,'maximum_source_edge_fit_residual_px':.75,'maximum_native_corner_displacement_px':6.,'mask_required_for_all_source_profiles':True,'camera_or_world_used':False}

def remeasure_reacquired_corner(rgb,receipt,mask):
 a=np.asarray(rgb);channels=_channels(a);q=np.asarray(receipt['corner_uv_in_patch'],float);mask=np.asarray(mask);h,w=mask.shape
 if not (6<q[0]<w-7 and 6<q[1]<h-7) or mask[int(round(q[1])),int(round(q[0]))]:return {'valid':False,'reason':'entity_current_native_mask_or_edge_domain_exclusion'}
 gray=(.2126*a[:,:,0]+.7152*a[:,:,1]+.0722*a[:,:,2]).astype(np.float32);seed=q.copy()
 try:
  cq=cv2.cornerSubPix(gray,q.astype(np.float32).reshape(1,1,2),(4,4),(-1,-1),(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,40,.001))[0,0].astype(float)
  if np.linalg.norm(cq-q)<=CONFIG['native_corner_max_move_px']:seed=cq
 except cv2.error:pass
 fits=[]
 for nav in receipt['edge_tangents']:
  candidates=[];nav=np.asarray(nav,float);nav=nav/np.linalg.norm(nav)
  for angle in CONFIG['ray_angle_offsets_degrees']:
   ang=np.deg2rad(angle);t=np.array([[np.cos(ang),-np.sin(ang)],[np.sin(ang),np.cos(ang)]])@nav;n=np.array([-t[1],t[0]]);uvs=[];audits=[]
   for d in CONFIG['inward_stations_px']:
    point=seed+d*t;u,v=[int(x) for x in np.rint(point)]
    if not(5<u<w-6 and 5<v<h-6) or mask[v,u]:continue
    r=_remeasure(point,n,channels,'poster_boundary')
    if r.get('current_uv') is None:continue
    z=np.asarray(r['current_uv']);u,v=[int(x) for x in np.rint(z)]
    if mask[v,u]:continue
    uvs.append(z);audits.append({'inward_station_px':d,**r})
   if len(uvs)<CONFIG['edge_minimum_actual_profiles']:continue
   uv=np.asarray(uvs);x=(uv-seed)@t;y=(uv-seed)@n;X=np.c_[np.ones(len(x)),x,x*x];coeff=np.linalg.lstsq(X,y,rcond=None)[0];err=y-X@coeff
   # Profile channel and material polarity must identify one actual transition.
   signatures=Counter((r['profile']['channel'],r['profile']['polarity']) for r in audits);agreement=max(signatures.values())/len(audits)
   if agreement<.75 or np.max(np.abs(err))>CONFIG['maximum_source_edge_fit_residual_px']:continue
   fit={'tangent':t.tolist(),'normal':n.tolist(),'coefficients':coeff.tolist(),'current_source_uv':uv.tolist(),'fit_residuals_px':err.tolist(),'profile_audits':audits,'angle_navigation_adjustment_degrees':angle,'same_material_transition_agreement':agreement}
   candidates.append((len(uvs),-float(np.mean(err*err)),fit))
  if not candidates:return {'valid':False,'reason':'no_single_current_material_edge_supported_in_navigation_cone','edge_fits':fits,'native_corner_seed_uv':seed.tolist()}
  fits.append(max(candidates,key=lambda x:(x[0],x[1]))[2])
 def equations(z):
  out=[]
  for f in fits:
   t=np.asarray(f['tangent']);n=np.asarray(f['normal']);a=np.asarray(f['coefficients']);d=(z-seed)@t;out.append((z-seed)@n-a@[1,d,d*d])
  return np.asarray(out)
 sol=solve_root(equations,seed);point=sol.x;J=[]
 for f in fits:
  t=np.asarray(f['tangent']);n=np.asarray(f['normal']);a=np.asarray(f['coefficients']);d=(point-seed)@t;J.append(n-(a[1]+2*a[2]*d)*t)
 singular=np.linalg.svd(J,compute_uv=False);move=float(np.linalg.norm(point-q));u,v=[int(x) for x in np.rint(point)]
 valid=bool(sol.success and np.isfinite(point).all() and np.max(np.abs(equations(point)))<1e-5 and singular[-1]>.1 and move<=CONFIG['maximum_native_corner_displacement_px'] and 0<=u<w and 0<=v<h and not mask[v,u])
 return {'valid':valid,'reason':'two_reacquired_current_material_edges_intersection' if valid else 'photometric_two_edge_intersection_not_well_supported','current_intersection_uv':point.tolist() if valid else None,'native_corner_seed_uv':seed.tolist(),'original_corner_seed_uv':q.tolist(),'intersection_jacobian':np.asarray(J).tolist(),'intersection_condition':float(singular[0]/max(singular[-1],1e-30)),'source_intersection_displacement_px':move,'edge_fits':fits,'fixed_source_operator_configuration':CONFIG,'entity_station_fixed':True,'statistical_covariance_claimed':False}

def reacquire_entities(audit_dir,parent_entity,native_dir,roles,output_dir):
 audit=Path(audit_dir);parent=Path(parent_entity);native=Path(native_dir);out=Path(output_dir)
 if out.exists():raise ValueError('New immutable entity recovery generation required')
 rows=[r for r in read_jsonl(audit/'relocated_rows.jsonl.gz') if r['point_correspondence']];byid={r['source_row_id']:r for r in rows};old=list(read_jsonl(parent/'rows_overlay.jsonl.gz'));pending=[r for r in old if not r['source_measurement_audit']['current_two_material_edges'].get('valid')];lookup={};byframe=defaultdict(list)
 for r in rows:lookup[(r['shot_id'],r['native_pts'],r['target_id'])]=r
 for o in pending:
  r=byid[o['source_row_id']];byframe[(r['shot_id'],r['native_pts'])].append((r,o))
 frames={f['native_pts']:f for f in read_json(native/'manifest.json')['frames']};guard=FrameRoleGuard(roles);data_root=next(x for x in native.parents if x.name=='RinkSpace_Data');mask_records={}
 for sid in {r['shot_id'] for r in rows}:
  nr=int(sid[1:]);base=data_root/'events/ankara_2026/programmes/okada_mei_sp/results'/f'shot{nr}_review_20261014';mp=base/'r2/frame_manifest.json' if nr==2 else base/'frame_manifest.json'
  for f in read_json(mp)['frames']:mask_records[(sid,int(f.get('native_pts',f.get('source_pts'))))]=f
 out.mkdir(parents=True);overlays=[];receipts=[];patches=[];masks=[];index=[];counts=Counter()
 for (sid,pts),items in sorted(byframe.items(),key=lambda x:x[0][1]):
  guard.assert_training_pixels([pts],shot_id=sid);f=frames[pts];png=native/f['native_png']
  if file_sha256(png)!=f['native_png_sha256']:raise ValueError('Native source changed')
  rgb=np.asarray(Image.open(png).convert('RGB'));mr=mask_records[(sid,pts)];mp=Path(mr.get('mask_path') or mr.get('source',{}).get('mask_path'));expected=mr.get('mask_sha256') or mr.get('source',{}).get('mask_sha256')
  if expected and file_sha256(mp)!=expected:raise ValueError('Native mask changed')
  mask=np.asarray(Image.open(mp).convert('L')).copy();mask[:170,:620]=255;mask[900:,1600:]=255
  for row,previous in items:
   ad,role=row['target_id'].split('.',1);q=np.asarray(row['source_uv']);directions={}
   for e in previous['two_edge_direction_sources']:
    cloud=np.asarray(e['actual_direction_source_uv']);center=cloud.mean(0);_,_,V=np.linalg.svd(cloud-center,full_matrices=False);t=V[0]
    if t@(center-q)<0:t=-t
    directions[e['target'].split('.',1)[1]]=(t.tolist(),{'kind':'existing_actual_source_edge_direction','target':e['target']})
   tangents=[];direction_receipts=[]
   for edge in EDGE_ROLES[role]:
    if edge not in directions:
     others=[lookup.get((sid,pts,ad+'.'+r)) for r in PAIRS[edge] if r!=role];others=[r for r in others if r is not None]
     if others:
      c=np.asarray(others[0]['source_uv']);t=c-q
      if np.linalg.norm(t)>5:directions[edge]=((t/np.linalg.norm(t)).tolist(),{'kind':'other_actual_material_corner_navigation_only','source_row_id':others[0]['source_row_id'],'source_uv':c.tolist(),'current_native_profiles_still_required':True})
    if edge not in directions:break
    t,e=directions[edge];tangents.append(t);direction_receipts.append(e)
   u,v=[int(x) for x in np.rint(q)];origin=[int(np.clip(u-48,0,1920-97)),int(np.clip(v-48,0,1080-97))];ox,oy=origin;patch=rgb[oy:oy+97,ox:ox+97];pmask=mask[oy:oy+97,ox:ox+97];receipt={'source_row_id':row['source_row_id'],'shot_id':sid,'native_pts':pts,'native_png_sha256':f['native_png_sha256'],'native_origin_uv':origin,'original_corner_seed':q.tolist(),'corner_uv_in_patch':(q-origin).tolist(),'edge_tangents':tangents,'edge_direction_sources':direction_receipts,'fixed_source_operator_configuration':CONFIG,'operator':'remeasure_reacquired_corner(native_RGB_patch,receipt,mask_patch)','RGB_is_actual_native_source':True,'manual_reference_preserved':row['manual_reference_preserved']}
   fit=remeasure_reacquired_corner(patch,receipt,pmask) if len(tangents)==2 else {'valid':False,'reason':'both_actual_source_direction_navigators_not_available'}
   if fit.get('valid'):
    fit['current_intersection_uv']=(np.asarray(fit['current_intersection_uv'])+origin).tolist();current=row['source_uv'] if row['manual_reference_preserved'] else fit['current_intersection_uv'];newaudit={**previous['source_measurement_audit'],'status':'manual_native_click_preserved' if row['manual_reference_preserved'] else 'entity_corner_two_direction_support','current_uv':current,'current_two_material_edges':fit,'entity_station_may_slide':False,'source_reacquisition_version':'photometric_corner_and_ray_refit_v1'};overlays.append({**previous,'current_uv':current,'source_measurement_audit':newaudit,'source_remeasurement_version':'entity_reacquire_v1'});counts['actual_entity_recovered']+=1
   else:counts['still_pending']+=1
   receipt['patch_index']=len(patches);patches.append(patch);masks.append(pmask);index.append(receipt);receipts.append({'source_row_id':row['source_row_id'],'target_id':row['target_id'],'shot_id':sid,'native_pts':pts,'result':fit,'patch_receipt':receipt,'physical_binding_unchanged':True})
  print(json.dumps({'native_pts':pts,'completed':len(receipts),'counts':dict(counts)}),flush=True)
 write_jsonl(out/'rows_overlay.jsonl.gz',overlays);write_jsonl(out/'fit_receipts.jsonl.gz',receipts);write_json(out/'patch_index.json',index);np.savez_compressed(out/'corner_patches.npz',RGB=np.asarray(patches,np.uint8),mask=np.asarray(masks,np.uint8));result={'schema':'rinkfit.joint9.entity_source_reacquisition.v1','input_pending':len(pending),'counts':dict(counts),'all_pending_accounted':len(receipts)==len(pending),'source_parent_dir':str(parent.resolve()),'source_parent_manifest_sha256':file_sha256(parent/'manifest.json'),'geometry_or_camera_used':False,'original_UV_or_entity_identity_changed':False,'noise_covariance_claimed':False};write_json(out/'manifest.json',result);return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--audit',required=True);p.add_argument('--parent-entity',required=True);p.add_argument('--native',required=True);p.add_argument('--roles',required=True);p.add_argument('--out',required=True);a=p.parse_args();print(json.dumps(reacquire_entities(a.audit,a.parent_entity,a.native,a.roles,a.out)))
