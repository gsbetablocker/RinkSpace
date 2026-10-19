"""Recover the actual width midline of C0 from both current paint interfaces.

The old unbound inner edge stays an immutable source record. New observations
are actual two-flank pixel reads, verified against the same-frame C0 source
curve. An image ellipse is a source identity check, never world geometry.
"""
from pathlib import Path
from collections import defaultdict,Counter
import argparse,json
import cv2,numpy as np
from PIL import Image
from scipy.signal import find_peaks
from scipy.spatial import cKDTree
from .io_utils import read_json,read_jsonl,write_json,write_jsonl,file_sha256,object_sha256
from .source_worker import _channels,_sample,_remeasure
from .role_guard import FrameRoleGuard

CONFIG={'version':'current_C0_two_material_flank_center_v1','normal_range_px':[-16.,16.],'normal_step_px':.25,'actual_inner_edge_navigation_tolerance_px':3.,'source_ellipse_normal_identity_tolerance_px':2.,'minimum_flank_separation_px':1.5,'maximum_flank_separation_px':24.,'both_flanks_required':True,'world_geometry_used':False,'old_single_inner_edge_is_midline':False}

def _ellipse_distance(points,ellipse):
 (cx,cy),(w,h),angle=ellipse;ang=np.deg2rad(angle);R=np.array([[np.cos(ang),-np.sin(ang)],[np.sin(ang),np.cos(ang)]]);q=(np.asarray(points)-[cx,cy])@R;a=w/2;b=h/2;v=q[:,0]**2/a**2+q[:,1]**2/b**2-1.;g=2*np.sqrt(q[:,0]**2/a**4+q[:,1]**2/b**4);return abs(v)/np.maximum(g,1e-12)

def remeasure_circular_line(channels,q,normal,source_ellipse,mask):
 q=np.asarray(q,float);n=np.asarray(normal,float);n=n/np.linalg.norm(n);off=np.arange(-16,16.0001,.25);coords=q+off[:,None]*n;inside=(coords[:,0]>=5)&(coords[:,0]<channels.shape[2]-6)&(coords[:,1]>=5)&(coords[:,1]<channels.shape[1]-6)
 if not inside.all():return {'valid':False,'reason':'both_C0_material_flanks_not_inside_actual_profile_domain'}
 index=np.rint(coords).astype(int)
 if np.any(mask[index[:,1],index[:,0]]):return {'valid':False,'reason':'C0_wide_profile_crosses_current_actor_or_graphics_mask'}
 val=_sample(channels,coords);grad=np.gradient(val,.25,axis=0);candidates=[]
 for ch in range(3):
  g=grad[:,ch];peaks=find_peaks(abs(g),height=1.,prominence=.5,distance=4)[0]
  for i in peaks:
   if abs(off[i])>CONFIG['actual_inner_edge_navigation_tolerance_px']:continue
   for j in peaks:
    if i==j or g[i]*g[j]>=0:continue
    width=abs(off[i]-off[j])
    if not(CONFIG['minimum_flank_separation_px']<=width<=CONFIG['maximum_flank_separation_px']):continue
    mid=q+.5*(off[i]+off[j])*n;identity=float(_ellipse_distance([mid],source_ellipse)[0])
    if identity>CONFIG['source_ellipse_normal_identity_tolerance_px']:continue
    flank=[(q+off[i]*n).tolist(),(q+off[j]*n).tolist()];r=_remeasure(mid,n,channels,'paint_center',flank)
    if r.get('current_uv') is None:continue
    observed_width=np.linalg.norm(np.diff(np.asarray(r['flank_uv']),axis=0))
    if observed_width<1.5:continue
    score=float(min(abs(g[i]),abs(g[j]))/(1+identity));candidates.append((score,{'valid':True,**r,'source_identity_normal_distance_px':identity,'source_ellipse':source_ellipse,'wide_profile_selected_channel':ch,'wide_profile_seed_offsets_px':[float(off[i]),float(off[j])],'physical_material':'ice.C0_finite_width_circular_line','source_operator':'actual_two_material_flank_width_midline','source_coordinate_from_world':False}))
 if not candidates:return {'valid':False,'reason':'C0_second_actual_material_flank_or_shared_curve_identity_not_supported'}
 candidates.sort(key=lambda x:x[0],reverse=True);best=candidates[0][1];centers=[np.asarray(x[1]['current_uv']) for x in candidates if x[0]>.9*candidates[0][0]]
 if any(np.linalg.norm(x-np.asarray(best['current_uv']))>1. for x in centers):return {'valid':False,'reason':'ambiguous_multiple_actual_C0_flank_pairs','candidate_pairs':len(candidates)}
 return {**best,'candidate_pair_count':len(candidates),'fixed_source_configuration':CONFIG}

def recover_c0(audit_dir,native_dir,roles,outdir):
 audit=Path(audit_dir);native=Path(native_dir);out=Path(outdir)
 if out.exists():raise ValueError('New immutable circular-line source generation required')
 allrows=list(read_jsonl(audit/'relocated_rows.jsonl.gz'));targets=[r for r in allrows if 'C0.inner_material_boundary' in r['source_feature_id']];by=defaultdict(list);groups=defaultdict(list);named=defaultdict(list)
 for r in allrows:groups[r['provenance_group_id']].append(r)
 for r in targets:by[(r['shot_id'],r['native_pts'])].append(r)
 for r in allrows:
  if r.get('target_id')=='ice.C0:outer' and r.get('current_uv') is not None:named[(r['shot_id'],r['native_pts'])].append(r)
 ref=read_json(audit/'retained_support.ref.json');sp=Path(ref['path'])
 if file_sha256(sp)!=ref['sha256']:raise ValueError('Retained real source support changed')
 source_cloud=defaultdict(list)
 for g in read_jsonl(sp):
  members=groups[g['source_group_id']]
  if not any(r.get('target_id')=='ice.C0:outer' for r in members):continue
  for s in g['actual_source_audits']:
   if s.get('current_uv') is not None and len(s.get('flank_uv') or [])==2:source_cloud[(g['shot_id'],g['native_pts'])].append(s['current_uv'])
 frames={f['native_pts']:f for f in read_json(native/'manifest.json')['frames']};guard=FrameRoleGuard(roles);data_root=next(x for x in native.parents if x.name=='RinkSpace_Data');mask_records={}
 for sid in {r['shot_id'] for r in targets}:
  nr=int(sid[1:]);base=data_root/'events/ankara_2026/programmes/okada_mei_sp/results'/f'shot{nr}_review_20261014';mp=base/'r2/frame_manifest.json' if nr==2 else base/'frame_manifest.json'
  for f in read_json(mp)['frames']:mask_records[(sid,int(f.get('native_pts',f.get('source_pts'))))]=f
 out.mkdir(parents=True);overlays=[];decisions=[];profiles=[];pindex=[];puv=[];pn=[];pch=[];counts=Counter();off=np.arange(-3,3.0001,.25);toff=np.arange(-2,3,dtype=float)
 for (sid,pts),rows in sorted(by.items(),key=lambda x:x[0][1]):
  guard.assert_training_pixels([pts],shot_id=sid);f=frames[pts];png=native/f['native_png']
  if file_sha256(png)!=f['native_png_sha256']:raise ValueError('Current native PNG changed')
  rgb=np.asarray(Image.open(png).convert('RGB'));channels=_channels(rgb);mr=mask_records[(sid,pts)];mp=Path(mr.get('mask_path') or mr.get('source',{}).get('mask_path'));ms=mr.get('mask_sha256') or mr.get('source',{}).get('mask_sha256')
  if ms and file_sha256(mp)!=ms:raise ValueError('Current source mask changed')
  mask=np.asarray(Image.open(mp).convert('L')).copy();mask[:170,:620]=255;mask[900:,1600:]=255;cloud=source_cloud[(sid,pts)];cloud+= [r['current_uv'] for r in named[(sid,pts)]];ellipse=None
  if len(cloud)>=8:
   c=np.asarray(cloud,np.float32);ellipse=cv2.fitEllipseDirect(c);dist=_ellipse_distance(c,ellipse)
   if np.quantile(dist,.9)>3.:ellipse=None
  for r in rows:
   m=r['source_measurement_audit'];n=m.get('normal');fit={'valid':False,'reason':'same_frame_C0_actual_source_curve_insufficient'}
   if ellipse is not None and n is not None and r.get('current_uv') is not None:fit=remeasure_circular_line(channels,r['current_uv'],n,ellipse,mask)
   valid=bool(fit.get('valid'));decision={'source_row_id':r['source_row_id'],'shot_id':sid,'native_pts':pts,'source_uv':r['source_uv'],'old_single_inner_edge_not_midline':True,'result':fit,'native_png_sha256':f['native_png_sha256'],'native_mask_sha256':file_sha256(mp),'world_or_camera_used':False,'solver_eligible':False,'existing_C0_reference_source_rows':[x['source_row_id'] for x in named[(sid,pts)]],'existing_C0_reference_measurement_hashes':[object_sha256({'source_row_id':x['source_row_id'],'current_uv':x['current_uv']}) for x in named[(sid,pts)]],'statistical_group_must_include_all_same_frame_C0_source_families':True};decisions.append(decision)
   if not valid:counts[fit['reason']]+=1;continue
   known=named[(sid,pts)][0];overlays.append({'source_row_id':r['source_row_id'],'source_uv':r['source_uv'],'current_uv':fit['current_uv'],'source_measurement_audit':fit,'target_id':known['target_id'],'joint_material_id':known['joint_material_id'],'measurement_operator_resolved':'paint_center','measurement_definition':known['measurement_definition'],'boundary_role':known['boundary_role'],'physical_surface':'ice','physical_identity_passed':True,'same_material_identity_evidence':'actual_same_frame_C0_source_curve_plus_two_current_paint_interfaces','original_binding_retained_in_source_history':True,'numeric_statistics_ready':False,'solver_eligible':False});counts['new_actual_two_flank_midline_rows']+=1
   n=np.asarray(fit['normal']);t=np.asarray([-n[1],n[0]])
   for side,point in enumerate(fit['flank_uv']):
    coords=np.asarray(point)+toff[:,None,None]*t+off[None,:,None]*n;allc=_sample(channels,coords.reshape(-1,2)).reshape(5,25,3);ch=int(np.max(abs(np.gradient(allc[2],.25,axis=0)),axis=0).argmax());pindex.append({'profile_index':len(profiles),'source_row_id':r['source_row_id'],'source_family_id':r['source_family_id'],'shot_id':sid,'native_pts':pts,'side':side,'native_png_sha256':f['native_png_sha256']});profiles.append(allc[:,:,ch]);puv.append(point);pn.append(n);pch.append(ch)
  print(json.dumps({'native_pts':pts,'rows_complete':len(decisions),'counts':dict(counts)}),flush=True)
 write_jsonl(out/'rows_overlay.jsonl.gz',overlays);write_jsonl(out/'source_receipts.jsonl.gz',decisions);write_json(out/'profile_index.json',pindex);np.savez_compressed(out/'profiles.npz',intensity=np.asarray(profiles,np.float32).reshape(-1,5,25),uv=np.asarray(puv,float).reshape(-1,2),normal=np.asarray(pn,float).reshape(-1,2),channel=np.asarray(pch,np.uint8),normal_offsets=off,tangent_offsets=toff);result={'schema':'rinkfit.joint9.C0_two_flank_source_recovery.v1','input_inner_edge_rows':len(targets),'frames':len(by),'counts':dict(counts),'all_original_inner_edge_records_accounted':len(decisions)==len(targets),'source_uv_unchanged':True,'old_inner_edge_promoted_as_midline':False,'new_observer':'two_current_material_flanks_width_center','formal_statistics_ready':False,'shared_source_covariance_with_existing_C0_required':True};write_json(out/'manifest.json',result);return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--audit',required=True);p.add_argument('--native',required=True);p.add_argument('--roles',required=True);p.add_argument('--out',required=True);a=p.parse_args();print(json.dumps(recover_c0(a.audit,a.native,a.roles,a.out)))
