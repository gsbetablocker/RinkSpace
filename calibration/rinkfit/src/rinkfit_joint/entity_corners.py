"""Fixed physical advertisement entities from two current finite material edges.

The finite source edge fits share one intersection.  A corner can never slide
along an ordinary edge.  Manual clicks are preserved even when a source fit is
available as a diagnostic.
"""
from pathlib import Path
from collections import defaultdict,Counter
import argparse,json,re
import numpy as np
from PIL import Image
from scipy.optimize import root as solve_root
from .io_utils import read_json,read_jsonl,write_json,write_jsonl,file_sha256,object_sha256
from .source_worker import _channels,_remeasure,_sample
from .role_guard import FrameRoleGuard

CORNER_SOURCE_OPERATOR={'version':'native_two_edge_quadratic_intersection_v1','inward_stations_px':[5.,8.,11.,14.,17.,20.],'minimum_points_per_edge':4,'edge_profile_operator':'poster_boundary','profile_normal_range_px':[-3.,3.],'profile_normal_step_px':.25,'channel_smoothing_sigma_px':.8,'intersection_max_displacement_px':4.,'source_edge_max_fit_residual_px':1.,'intersection_min_singular':.05}

EDGE_ROLES={'left_top':('top_edge','left_vertical'),'right_top':('top_edge','right_vertical'),'left_bottom':('bottom_edge','left_vertical'),'right_bottom':('bottom_edge','right_vertical')}

def remeasure_corner(channels,corner_uv,edge_tangents,mask):
 q=np.asarray(corner_uv,float);fits=[]
 for tangent in edge_tangents:
  t=np.asarray(tangent,float);t=t/np.linalg.norm(t);n=np.asarray([-t[1],t[0]]);uvs=[];audits=[]
  for d in CORNER_SOURCE_OPERATOR['inward_stations_px']:
   seed=q+d*t;u,v=np.rint(seed).astype(int)
   if not(4<u<channels.shape[2]-5 and 4<v<channels.shape[1]-5) or mask[v,u]:continue
   measured=_remeasure(seed,n,channels,'poster_boundary')
   if measured['current_uv'] is None:continue
   current=np.asarray(measured['current_uv']);u,v=np.rint(current).astype(int)
   if mask[v,u]:continue
   uvs.append(current);audits.append({'inward_station_px':d,**measured})
  if len(uvs)<4:return {'valid':False,'reason':'insufficient_current_native_points_on_one_entity_edge','edge_fits':fits,'support_profiles':audits}
  uv=np.asarray(uvs);x=(uv-q)@t;y=(uv-q)@n;X=np.c_[np.ones(len(x)),x,x*x];coeff=np.linalg.lstsq(X,y,rcond=None)[0];residual=y-X@coeff
  fits.append({'tangent':t.tolist(),'normal':n.tolist(),'coefficients':coeff.tolist(),'design_matrix':X.tolist(),'current_source_uv':uv.tolist(),'profile_audits':audits,'fit_residuals_px':residual.tolist(),'source_operator':'free_current_native_local_quadratic_material_edge'})
 def equations(z):
  z=np.asarray(z);out=[]
  for f in fits:
   t=np.asarray(f['tangent']);n=np.asarray(f['normal']);a=np.asarray(f['coefficients']);x=(z-q)@t;out.append((z-q)@n-(a[0]+a[1]*x+a[2]*x*x))
  return np.asarray(out)
 solution=solve_root(equations,q,method='hybr',options={'xtol':1e-9});z=solution.x;J=[]
 for f in fits:
  a=np.asarray(f['coefficients']);t=np.asarray(f['tangent']);n=np.asarray(f['normal']);x=(z-q)@t;J.append(n-(a[1]+2*a[2]*x)*t)
 J=np.asarray(J);singular=np.linalg.svd(J,compute_uv=False);condition=float(singular[0]/max(singular[-1],1e-30));maxfit=max(abs(v) for f in fits for v in f['fit_residuals_px']);distance=float(np.linalg.norm(z-q));valid=bool(solution.success and np.isfinite(z).all() and np.max(np.abs(equations(z)))<1e-5 and singular[-1]>.05 and distance<=4. and maxfit<=1.)
 return {'valid':valid,'current_intersection_uv':z.tolist() if valid else None,'intersection_seed_uv':q.tolist(),'edge_fits':fits,'intersection_jacobian':J.tolist(),'intersection_condition':condition,'max_source_edge_fit_residual_px':maxfit,'source_intersection_displacement_px':distance,'reason':'two_current_material_edges_intersection' if valid else 'current_material_intersection_weak_curved_or_outside_local_source','statistical_covariance_claimed':False,'entity_station_fixed':True}

def remeasure_corner_from_patch(rgb,receipt,mask):
 """Pure source replay; no file access or world/camera values."""
 return remeasure_corner(_channels(np.asarray(rgb)),receipt['corner_uv_in_patch'],receipt['edge_tangents'],np.asarray(mask,np.uint8))

def audit_corners(source_audit,native_dir,roles,output_dir):
 audit=Path(source_audit);native=Path(native_dir);out=Path(output_dir)
 if out.exists():raise ValueError('New immutable entity source generation required')
 rows=list(read_jsonl(audit/'relocated_rows.jsonl.gz'));corners=[r for r in rows if r['point_correspondence']];frames={f['native_pts']:f for f in read_json(native/'manifest.json')['frames']};guard=FrameRoleGuard(roles);byframe=defaultdict(list);bytarget=defaultdict(list)
 for r in rows:
  bytarget[(r['shot_id'],r['native_pts'],r.get('target_id'))].append(r)
 for r in corners:byframe[(r['shot_id'],r['native_pts'])].append(r)
 # Real readout curves remain useful for estimating edge direction even when
 # the representative packet encoded its endpoints as fixed entity corners.
 support_ref=read_json(audit/'retained_support.ref.json');support_path=Path(support_ref['path'])
 if file_sha256(support_path)!=support_ref['sha256']:raise ValueError('Actual source support changed')
 groups={};members=defaultdict(list)
 for r in rows:members[r['provenance_group_id']].append(r)
 raw_by_target=defaultdict(list)
 for s in read_jsonl(support_path):
  r=members[s['source_group_id']][0];targets={x.get('target_id') for x in members[s['source_group_id']] if x.get('category')=='ad_edge'}
  points=[x['current_uv'] for x in s['actual_source_audits'] if x.get('current_uv') is not None]
  for target in targets:raw_by_target[(r['shot_id'],r['native_pts'],target)].extend(points)
 data_root=next(x for x in native.parents if x.name=='RinkSpace_Data');mask_records={}
 for sid in {r['shot_id'] for r in corners}:
  nr=int(sid[1:]);mb=data_root/'events/ankara_2026/programmes/okada_mei_sp/results'/f'shot{nr}_review_20261014';mp=mb/'r2/frame_manifest.json' if nr==2 else mb/'frame_manifest.json'
  for f in read_json(mp)['frames']:mask_records[(sid,int(f.get('native_pts',f.get('source_pts'))))]=f
 out.mkdir(parents=True);overlays=[];receipts=[];counts=Counter();patches=[];patchmask=[];patchmeta=[]
 for (sid,pts),items in sorted(byframe.items(),key=lambda x:x[0][1]):
  guard.assert_training_pixels([pts],shot_id=sid);fr=frames[pts];png=native/fr['native_png']
  if file_sha256(png)!=fr['native_png_sha256']:raise ValueError('Current source pixels changed')
  rgb=np.asarray(Image.open(png).convert('RGB'));channels=_channels(rgb);mr=mask_records[(sid,pts)];mp=Path(mr.get('mask_path') or mr.get('source',{}).get('mask_path'));expected=mr.get('mask_sha256') or mr.get('source',{}).get('mask_sha256')
  if expected and file_sha256(mp)!=expected:raise ValueError('Current source mask changed')
  mask=np.asarray(Image.open(mp).convert('L'));mask=mask.copy();mask[:170,:620]=255;mask[900:,1600:]=255
  for row in items:
   target=row['target_id'];ad,role=target.split('.',1);q=np.asarray(row['source_uv'],float);edge_roles=EDGE_ROLES.get(role);tangents=[];evidence=[];reason=None
   if not edge_roles:reason='unsupported_physical_entity_role'
   else:
    for edge in edge_roles:
     key=(sid,pts,ad+'.'+edge);cloud=np.asarray(raw_by_target[key]+[r['current_uv'] for r in bytarget[key] if r.get('current_uv') is not None],float)
     if cloud.size==0 or len(cloud)<2:reason='one_entity_edge_has_no_current_source_direction';break
     distance=np.linalg.norm(cloud-q,axis=1);cloud=cloud[np.argsort(distance)[:min(20,len(cloud))]];center=cloud.mean(0);_,sv,V=np.linalg.svd(cloud-center,full_matrices=False);t=V[0]
     if np.dot(t,center-q)<0:t=-t
     if np.max(np.abs((cloud-center)@t))<1.:reason='one_entity_edge_direction_not_supported';break
     tangents.append(t.tolist());evidence.append({'target':ad+'.'+edge,'actual_direction_source_uv':cloud.tolist(),'direction_from_current_material_only':True})
   u,v=[int(x) for x in np.rint(q)]
   if not(32<=u<1888 and 32<=v<1048):reason='entity_source_patch_outside_native_image'
   elif mask[v,u] and not row['manual_reference_preserved']:reason='entity_current_source_occluded'
   fit=remeasure_corner(channels,q,tangents,mask) if reason is None and len(tangents)==2 else {'valid':False,'reason':reason}
   manual=row['manual_reference_preserved'];current=row['source_uv'] if manual else fit.get('current_intersection_uv');valid=bool(fit['valid'])
   status='manual_native_click_preserved' if manual else ('entity_corner_two_direction_support' if valid else 'entity_corner_requires_source_geometry_review')
   newaudit={**row['source_measurement_audit'],'status':status,'current_uv':current,'current_two_material_edges':fit,'physical_entity_role':role,'entity_station_may_slide':False,'manual_click_moved':False}
   overlay={'source_row_id':row['source_row_id'],'source_uv':row['source_uv'],'previous_current_uv':row['current_uv'],'current_uv':current,'source_measurement_audit':newaudit,'two_edge_direction_sources':evidence,'physical_identity_retained':True,'point_correspondence':True,'entity_station_fixed':True,'native_png_sha256':fr['native_png_sha256'],'source_packet_sha256':row['source_packet_sha256'],'source_family_id':row['source_family_id'],'fit_role':'development','manual_reference_preserved':manual,'source_remeasurement_version':'entity_corners_v2','numeric_statistics_ready':False,'solver_eligible':False}
   overlays.append(overlay);receipts.append({'source_row_id':row['source_row_id'],'target_id':target,'shot_id':sid,'native_pts':pts,'result':fit,'edge_direction_sources':evidence,'manual_reference_preserved':manual,'original_uv_unchanged':True});counts['source_two_edge_intersections' if valid else 'source_two_edge_pending']+=1;counts['manual_entities_unchanged']+=int(manual)
   if 32<=u<1888 and 32<=v<1048 and len(tangents)==2:
    pi=len(patches);patches.append(rgb[v-32:v+33,u-32:u+33]);patchmask.append(mask[v-32:v+33,u-32:u+33]);patchmeta.append({'patch_index':pi,'source_row_id':row['source_row_id'],'native_pts':pts,'shot_id':sid,'native_png_sha256':fr['native_png_sha256'],'native_origin_uv':[u-32,v-32],'original_corner_seed':q.tolist(),'source_operator_configuration':CORNER_SOURCE_OPERATOR,'corner_uv_in_patch':(q-[u-32,v-32]).tolist(),'edge_tangents':tangents,'operator':'remeasure_corner_from_patch(native_RGB_patch,patch_receipt,mask_patch)','RGB_is_actual_native_source':True,'gaussian_border_halo_px':12,'manual_reference_preserved':manual})
  print(json.dumps({'shot_id':sid,'native_pts':pts,'entity_rows_complete':len(overlays)}),flush=True)
 write_jsonl(out/'rows_overlay.jsonl.gz',overlays);write_jsonl(out/'fit_receipts.jsonl.gz',receipts);write_json(out/'patch_index.json',patchmeta);np.savez_compressed(out/'corner_patches.npz',RGB=np.asarray(patches,np.uint8),mask=np.asarray(patchmask,np.uint8));result={'schema':'rinkfit.joint9.entity_native_remeasurement.v1','entity_rows':len(corners),'counts':dict(counts),'native_frames':len(byframe),'patches_for_source_resampling':len(patches),'original_UV_or_binding_changed':False,'entity_station_freed':False,'noise_covariance_claimed':False,'source_audit_rows_sha256':file_sha256(audit/'relocated_rows.jsonl.gz'),'role_file_sha256':file_sha256(roles),'all_entities_accounted':len(overlays)==len(corners)};write_json(out/'manifest.json',result);return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--audit',required=True);p.add_argument('--native',required=True);p.add_argument('--roles',required=True);p.add_argument('--out',required=True);a=p.parse_args();print(json.dumps(audit_corners(a.audit,a.native,a.roles,a.out)))
