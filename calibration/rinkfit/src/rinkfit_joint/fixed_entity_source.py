"""Conditioned native observation operator for already identified physical edges.

Source classification is frozen in a receipt. Pixel perturbations never change
which material, profile, contrast channel, polarity, or finite ray is measured.
Conditional localization covariance is distinct from material-selection and
whole-source-family nuisance uncertainty.
"""
from pathlib import Path
from collections import defaultdict,Counter
import argparse,json
import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates
from .io_utils import read_json,read_jsonl,write_json,write_jsonl,file_sha256
from .source_worker import _channels
from .role_guard import FrameRoleGuard

CONFIG={'version':'fixed_physical_edge_cubic_profile_intersection_v1','normal_offsets_px':np.linspace(-1.5,1.5,25).tolist(),'channel_interpolation_order':3,'gaussian_sigma_px':.8,'material_channel_polarity_ray_and_station_selection':'conditioned on original current RGB source review, separately recorded','synthetic_noise_reselects_material':False,'synthetic_noise_quality_threshold_censoring':False,'family_nuisance_or_material_selection_covariance_claimed':False}

def remeasure_fixed_entity(rgb,receipt,mask=None):
 channels=_channels(np.asarray(rgb));q=np.asarray(receipt['corner_seed_in_patch'],float);off=np.asarray(CONFIG['normal_offsets_px']);fits=[];boundary=[]
 for ef in receipt['fixed_physical_edges']:
  t=np.asarray(ef['tangent']);n=np.asarray(ef['normal']);points=[];profile_details=[]
  for profile in ef['profiles']:
   center=np.asarray(profile['center_in_patch']);coords=center+off[:,None]*n;c=int(profile['channel']);pol=int(profile['polarity']);intensity=map_coordinates(channels[c],[coords[:,1],coords[:,0]],order=3,mode='nearest',prefilter=True);gradient=pol*np.gradient(intensity,off);k=int(gradient.argmax());delta=0.
   if 0<k<len(off)-1:
    a=gradient[k-1:k+2];den=a[0]-2*a[1]+a[2]
    if abs(den)>1e-12:delta=float(np.clip(.5*(a[0]-a[2])/den,-1,1))*(off[1]-off[0])
   shift=float(off[k]+delta);point=center+shift*n;points.append(point);boundary.append(k in (0,len(off)-1));profile_details.append({'center_in_patch':center.tolist(),'channel':c,'polarity':pol,'offset_px':shift,'peak_at_window_boundary':bool(k in (0,len(off)-1)),'current_uv_in_patch':point.tolist(),'profile_intensity':intensity.tolist(),'signed_gradient':gradient.tolist()})
  uv=np.asarray(points);x=(uv-q)@t;y=(uv-q)@n;X=np.c_[np.ones(len(x)),x,x*x];coeff=np.linalg.lstsq(X,y,rcond=None)[0];res=y-X@coeff;fits.append({'tangent':t.tolist(),'normal':n.tolist(),'coefficients':coeff.tolist(),'current_source_uv_in_patch':uv.tolist(),'fit_residuals_px':res.tolist(),'profiles':profile_details})
 point=q.copy();singular=False
 for iteration in range(24):
  values=[];J=[]
  for f in fits:
   t=np.asarray(f['tangent']);n=np.asarray(f['normal']);a=np.asarray(f['coefficients']);x=(point-q)@t;values.append((point-q)@n-a@[1,x,x*x]);J.append(n-(a[1]+2*a[2]*x)*t)
  J=np.asarray(J);values=np.asarray(values)
  if not np.isfinite(J).all() or not np.isfinite(values).all():singular=True;break
  try:step=np.linalg.solve(J,values)
  except np.linalg.LinAlgError:singular=True;break
  if np.linalg.norm(step)>4:step*=4/np.linalg.norm(step)
  point-=step
  if np.linalg.norm(step)<1e-10:break
 final_equations=[]
 for f in fits:
  t=np.asarray(f['tangent']);n=np.asarray(f['normal']);a=np.asarray(f['coefficients']);x=(point-q)@t;final_equations.append(float((point-q)@n-a@[1,x,x*x]))
 numeric_residual=float(np.max(np.abs(final_equations))) if final_equations else float('inf');converged=bool(np.isfinite(numeric_residual) and numeric_residual<=1e-7)
 valid=bool(not singular and np.isfinite(point).all() and converged);return {'valid':valid,'numeric_converged':converged,'numeric_equation_residual_px':numeric_residual,'current_intersection_uv':point.tolist() if valid else None,'edge_fits':fits,'intersection_jacobian':J.tolist(),'numeric_iterations':iteration+1,'profile_window_boundary_count':sum(boundary),'synthetic_source_quality_gates_applied':False,'entity_station_fixed':True,'conditioning_uncertainty_included':False}

def build_fixed_sources(run,source_audit,outdir):
 run=Path(run);audit=Path(source_audit);out=Path(outdir)
 if out.exists():raise ValueError('New immutable conditioned entity source generation required')
 decisions=read_json(run/'entity_statistics_v1/decisions.json')+read_json(run/'entity_reacquire_statistics_v1/decisions.json');ids={d['source_row_id'] for d in decisions if 'nonlinear source resampling crosses' in d.get('reason','')};source_audit_sha=file_sha256(audit/'relocated_rows.jsonl.gz');rows={r['source_row_id']:r for r in read_jsonl(audit/'relocated_rows.jsonl.gz') if r['source_row_id'] in ids};parent_origins={r['source_row_id']:r['native_origin_uv'] for r in read_json(run/'entity_reacquire_v1/patch_index.json')};frames={f['native_pts']:f for f in read_json(run/'native_selected_v2/manifest.json')['frames']};guard=FrameRoleGuard(run/'roles/frame_roles.json');by=defaultdict(list);data_root=next(x for x in run.parents if x.name=='RinkSpace_Data');masks={}
 for r in rows.values():by[(r['shot_id'],r['native_pts'])].append(r)
 for sid in {r['shot_id'] for r in rows.values()}:
  n=int(sid[1:]);base=data_root/'events/ankara_2026/programmes/okada_mei_sp/results'/f'shot{n}_review_20261014';mp=base/'r2/frame_manifest.json' if n==2 else base/'frame_manifest.json'
  for f in read_json(mp)['frames']:masks[(sid,int(f.get('native_pts',f.get('source_pts'))))]=f
 out.mkdir(parents=True);overlays=[];receipts=[];indices=[];patches=[];patchmasks=[];counts=Counter()
 for (sid,pts),members in sorted(by.items(),key=lambda x:x[0][1]):
  guard.assert_training_pixels([pts],shot_id=sid);fr=frames[pts];path=run/'native_selected_v2'/fr['native_png']
  if file_sha256(path)!=fr['native_png_sha256']:raise ValueError('Actual current native PNG changed')
  rgb=np.asarray(Image.open(path).convert('RGB'));mr=masks[(sid,pts)];mp=Path(mr.get('mask_path') or mr.get('source',{}).get('mask_path'));sha=mr.get('mask_sha256') or mr.get('source',{}).get('mask_sha256')
  if sha and file_sha256(mp)!=sha:raise ValueError('Actual source mask changed')
  mask=np.asarray(Image.open(mp).convert('L')).copy();mask[:170,:620]=255;mask[900:,1600:]=255
  for row in members:
   rid=row['source_row_id'];q=np.asarray(row['current_uv'],float);u,v=np.rint(q).astype(int);origin=np.asarray([int(np.clip(u-48,0,1823)),int(np.clip(v-48,0,983))]);ox,oy=origin;patch=rgb[oy:oy+97,ox:ox+97];pmask=mask[oy:oy+97,ox:ox+97];old=row['source_measurement_audit']['current_two_material_edges'];old_origin=np.asarray(parent_origins[rid]) if row.get('entity_source_generation','').endswith('entity_reacquire_v1') else np.zeros(2);edges=[];source_admitted=True
   for f in old['edge_fits']:
    t=np.asarray(f['tangent']);n=np.asarray(f['normal']);plist=[]
    for point,pa in zip(f['current_source_uv'],f['profile_audits']):
     native_point=np.asarray(point)+old_origin;center=native_point-origin;ch=pa['profile']['channel'];pol=pa['profile']['polarity'];plist.append({'center_in_patch':center.tolist(),'native_center_uv':native_point.tolist(),'channel':int(ch),'polarity':int(pol),'original_current_RGB_selection':pa,'source_normal_window_px':[-1.5,1.5]});coords=center+np.asarray(CONFIG['normal_offsets_px'])[:,None]*n;ix=np.rint(coords).astype(int)
     if np.any(ix[:,0]<4) or np.any(ix[:,0]>=93) or np.any(ix[:,1]<4) or np.any(ix[:,1]>=93) or np.any(pmask[np.clip(ix[:,1],0,96),np.clip(ix[:,0],0,96)]):source_admitted=False
    edges.append({'tangent':t.tolist(),'normal':n.tolist(),'profiles':plist,'one_physical_material_edge_identity_fixed':True})
   receipt={'patch_index':len(patches),'source_row_id':rid,'shot_id':sid,'native_pts':pts,'native_png_sha256':fr['native_png_sha256'],'native_origin_uv':origin.tolist(),'original_corner_seed':row['source_uv'],'previous_actual_corner_uv':q.tolist(),'corner_seed_in_patch':(q-origin).tolist(),'fixed_physical_edges':edges,'source_operator_configuration':CONFIG,'operator':'remeasure_fixed_entity(actual_RGB_patch,receipt,mask_patch)','manual_reference_preserved':row['manual_reference_preserved'],'conditional_localization_only':True,'conditioning_source_generation':row.get('entity_source_generation'),'conditioning_source_audit_sha256':source_audit_sha};result=remeasure_fixed_entity(patch,receipt,pmask);result['valid']=bool(result['valid'] and source_admitted and result['profile_window_boundary_count']==0)
   if result['valid']:
    uv=(np.asarray(result['current_intersection_uv'])+origin).tolist();result['current_intersection_uv']=uv;result['reason']='two_current_fixed_physical_edges_conditional_intersection';newaudit={**row['source_measurement_audit'],'status':'entity_corner_two_direction_support','current_uv':uv,'current_two_material_edges':result,'fixed_operator_conditioning_receipt_index':len(receipts),'source_selection_uncertainty_pending':True};overlays.append({'source_row_id':rid,'source_uv':row['source_uv'],'current_uv':uv,'source_measurement_audit':newaudit,'physical_identity_retained':True,'point_correspondence':True,'entity_station_fixed':True,'native_png_sha256':fr['native_png_sha256'],'source_packet_sha256':row['source_packet_sha256'],'source_family_id':row['source_family_id'],'fit_role':'development','manual_reference_preserved':False,'source_remeasurement_version':'fixed_entity_source_v1','numeric_statistics_ready':False,'solver_eligible':False});counts['actual_fixed_source_operators_ready']+=1
   else:counts['actual_fixed_window_source_pending']+=1
   patches.append(patch);patchmasks.append(pmask);indices.append(receipt);receipts.append({'source_row_id':rid,'shot_id':sid,'native_pts':pts,'base_source_result':result,'source_conditioning_receipt':receipt,'source_admitted_before_synthetic_sampling':bool(result['valid']),'old_UV_and_identity_unchanged':True})
  print(json.dumps({'native_pts':pts,'completed':len(receipts),'counts':dict(counts)}),flush=True)
 write_jsonl(out/'rows_overlay.jsonl.gz',overlays);write_jsonl(out/'fit_receipts.jsonl.gz',receipts);write_json(out/'patch_index.json',indices);np.savez_compressed(out/'corner_patches.npz',RGB=np.asarray(patches,np.uint8),mask=np.asarray(patchmasks,np.uint8));result={'schema':'rinkfit.joint9.fixed_physical_entity_source_operator.v1','input_old_unstable_operator_rows':len(ids),'counts':dict(counts),'all_input_rows_accounted':len(receipts)==len(ids),'conditioning_uncertainty_included':False,'conditional_localization_covariance_claimed':False,'new_operator_version':CONFIG['version'],'old_source_uv_or_identity_modified':False,'synthetic_quality_censoring':False};write_json(out/'manifest.json',result);return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--audit',required=True);p.add_argument('--out',required=True);a=p.parse_args();print(json.dumps(build_fixed_sources(a.run,a.audit,a.out)))
