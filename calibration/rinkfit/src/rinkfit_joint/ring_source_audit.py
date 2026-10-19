"""Actual neutral target layers, using the existing signed source reader.

Free image ellipses only orient pixel searches and test observed nesting. No
world radius, common radius ratio, spacing or rink symmetry enters this audit.
"""
from pathlib import Path
from collections import defaultdict,Counter
from itertools import groupby
import argparse,json,gzip,hashlib,sys
import cv2,numpy as np
from .io_utils import read_json,read_jsonl,write_json,file_sha256
from .measurement_contract import EventMeasurementContract
from .role_guard import FrameRoleGuard

def _plain(x):
 if isinstance(x,np.ndarray):return _plain(x.tolist())
 if isinstance(x,np.generic):return _plain(x.item())
 if isinstance(x,float) and not np.isfinite(x):return None
 if isinstance(x,dict):return {k:_plain(v) for k,v in x.items()}
 if isinstance(x,(tuple,list)):return [_plain(v) for v in x]
 return x

def ring_audit(project,run,source,outdir):
 project=Path(project);run=Path(run);source=Path(source);out=Path(outdir)
 if out.exists():raise ValueError('New immutable current target-layer generation required')
 sys.path[:0]=[str(project),str(project/'src')]
 from rinkspace.vision_source_shapes import _MaterialProfileFrame,read_source_material_transition_profiles
 native=run/'native_selected_v2';frames={int(f['native_pts']):f for f in read_json(native/'manifest.json')['frames']};guard=FrameRoleGuard(run/'roles/frame_roles.json');contract=EventMeasurementContract(project,'ankara_2026');data=next(x for x in run.parents if x.name=='RinkSpace_Data');masks={}
 for sid in sorted({f['shot_id'] for f in frames.values()}):
  n=int(sid[1:]);d=data/'events/ankara_2026/programmes/okada_mei_sp/results'/f'shot{n}_review_20261014';mp=d/'r2/frame_manifest.json' if n==2 else d/'frame_manifest.json'
  for f in read_json(mp)['frames']:masks[(sid,int(f.get('native_pts',f.get('source_pts'))))]=f
 out.mkdir(parents=True);(out/'layer_contexts').mkdir();counts=Counter();frame_reports=[];public=project/'src/rinkspace/vision_source_shapes.py';code_sha=file_sha256(public);write_json(out/'policy.json',dict(source_reader=str(public),sha256=code_sha,source_signal='actual native neutral-gray-to-white BT601 luminance, signed, never blue proxy',search_interval_px=[-8,8],context_interval_px=[-16,16],image_ellipses_used_only_for_search_normals_and_actual_nesting=True,world_radius_or_ratio_spacing_symmetry_used=False,contract_sha256=contract.digest,source_clock='official current native programme PTS'))
 with gzip.open(out/'rows.jsonl.gz','wt',encoding='utf8') as writer:
  for (sid,pts),allrows in groupby(read_jsonl(source),key=lambda r:(r['shot_id'],int(r['native_pts']))):
   groups=defaultdict(list)
   for r in allrows:
    t=str(r.get('target_id') or '');feature=t[4:].split(':')[0] if t.startswith('ice.') else ''
    if feature in contract.target_rings:groups[t].append(r)
   if not groups:continue
   guard.assert_training_pixels([pts],shot_id=sid);fr=frames[pts];png=native/fr['native_png']
   if file_sha256(png)!=fr['native_png_sha256']:raise ValueError('Actual target PNG changed')
   rgb=cv2.cvtColor(cv2.imread(str(png)),cv2.COLOR_BGR2RGB);mr=masks[(sid,pts)];mp=Path(mr.get('mask_path') or mr.get('source',{}).get('mask_path'));mask=cv2.imread(str(mp),0)>0;mask[:170,:620]=True;mask[900:,1600:]=True;binding=dict(source_sha256='e8b53966d56e72f3842d92c3ecbe6e9cc7e4670b7d0c9bd12f58fd2cff5ad57d',source_pts=pts,time_base='1/12800',shot_id=sid,native_hw=[1080,1920],decoded_native_rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest());frame=_MaterialProfileFrame(rgb,binding);layer_info={};pending=[]
   for target,members in groups.items():
    nav=np.asarray([r.get('current_uv') or r['source_uv'] for r in members],float);center=None;ellipse=None;normals=[]
    if len(np.unique(np.round(nav,4),axis=0))>=5:
     try:
      ellipse=cv2.fitEllipse(nav.astype(np.float32));center=np.asarray(ellipse[0]);angle=np.radians(ellipse[2]);R=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]]);axis=np.maximum(np.asarray(ellipse[1])*.5,1e-3);M=R@np.diag(axis**-2)@R.T
      if not np.isfinite(M).all() or axis.max()/axis.min()>100:ellipse=None
     except cv2.error:ellipse=None
    for r,q in zip(members,nav):
     n=M@(q-center) if ellipse is not None else np.asarray((r.get('source_measurement_audit') or {}).get('normal') or [np.nan,np.nan])
     if np.isfinite(n).all() and np.linalg.norm(n)>1e-8:n=n/np.linalg.norm(n)
     normals.append(n)
    normals=np.asarray(normals);delta=[];usable=[]
    for i,(q,n) in enumerate(zip(nav,normals)):
     if not np.isfinite(n).all():continue
     tangent=np.array([-n[1],n[0]]);off=np.arange(-18,19);xy=q+np.arange(-2,3)[:,None,None]*tangent+off[None,:,None]*n;ii=np.rint(xy).astype(int)
     if ii[:,:,0].min()<0 or ii[:,:,1].min()<0 or ii[:,:,0].max()>=1920 or ii[:,:,1].max()>=1080 or mask[ii[:,:,1],ii[:,:,0]].any():continue
     left=q-8*n;right=q+8*n;a=cv2.getRectSubPix(frame.luminance.astype(np.float32),(3,3),tuple(left.astype(float))).mean();b=cv2.getRectSubPix(frame.luminance.astype(np.float32),(3,3),tuple(right.astype(float))).mean();delta.append(float(b-a));usable.append(i)
    sign=1 if delta and np.median(delta)>0 else -1;consistent=bool(delta and np.mean(np.sign(delta)==sign)>=.8 and np.median(np.abs(delta))>=2.)
    transition='dark_to_light' if sign==1 else 'light_to_dark';readings={}
    if consistent and usable:
     result=read_source_material_transition_profiles(frame.rgb,binding=frame.binding,search_points_uv=nav[usable],normals=normals[usable],search_intervals_px=[[-8.,8.]]*len(usable),context_search_intervals_px=[[-16.,16.]]*len(usable),transition=transition,expected_RGB_sha256=frame.digest,keep_profiles=True,_validated_frame=frame)
     readings={i:q for i,q in zip(usable,result['readings'])}
    measured=[np.asarray(v['actual_source_uv']) for v in readings.values() if np.asarray(v['actual_source_uv']).shape==(2,)];actual_ellipse=None
    if len(measured)>=5:
     try:actual_ellipse=cv2.fitEllipse(np.asarray(measured,np.float32))
     except cv2.error:pass
    layer_info[target]=dict(target=target,navigation_ellipse=ellipse,actual_material_ellipse=actual_ellipse,actual_outward_luminance_deltas=delta,material_transition=transition,source_material_sign_consistent=consistent,current_source_mean_count=len(measured),readings=readings)
    for i,r in enumerate(members):
     reading=readings.get(i);q=np.asarray(reading['actual_source_uv']) if reading else np.empty((0,2));valid=q.shape==(2,) and np.isfinite(q).all();manual=bool(r.get('manual_reference_preserved'));d=dict(source_row_id=r['source_row_id'],source_family_id=r.get('source_family_id'),shot_id=sid,native_pts=pts,target_id=target,source_uv=r['source_uv'],current_uv=r['source_uv'] if manual else (q.tolist() if valid else None),source_operator='signed_actual_gray_target_material_transition_q50',material_transition=transition,normal=normals[i].tolist() if np.isfinite(normals[i]).all() else None,actual_source_reading=_plain(reading),manual_click_preserved=manual,local_material_interface_measured=valid,source_layer_nesting_verified=False,finite_physical_house_identity_proved=False,measurement_contract_sha256=contract.digest,native_png_sha256=fr['native_png_sha256'],solver_eligible=False,old_W_reused=False,source_uv_changed=False);pending.append(d);counts['actual_signed_material_candidates' if valid and not manual else 'actual_manual_preserved' if manual else 'actual_material_pending']+=1;counts['rows']+=1
   # Test nesting of separately measured source ellipses, without forcing any
   # concentricity, world-radius relation, or guessed completion of a short arc.
   nesting={}
   for feature in sorted({t.split(':')[0] for t in layer_info}):
    keys=[feature+':'+v for v in ('outer','middle','inner')];available=[k for k in keys if k in layer_info and layer_info[k]['actual_material_ellipse'] is not None];ok=len(available)==3
    if ok:
     polygons=[]
     for k in keys:
      e=layer_info[k]['actual_material_ellipse'];polygons.append(cv2.ellipse2Poly(tuple(np.rint(e[0]).astype(int)),tuple(np.maximum(1,np.rint(np.asarray(e[1])*.5)).astype(int)),int(round(e[2])),0,360,3))
     ok=all(all(cv2.pointPolygonTest(a.astype(np.float32),tuple(v.astype(float)),False)>=0 for v in b) for a,b in zip(polygons[:-1],polygons[1:]))
    nesting[feature]=dict(actual_three_layer_source_nesting_verified=bool(ok),available_actual_layer_ellipses=available,world_geometry_used=False,concentric_centers_or_radius_ratios_forced=False)
   path=out/'layer_contexts'/f'{pts}.json';write_json(path,_plain(dict(source_binding=binding,native_png=str(png),native_png_sha256=fr['native_png_sha256'],actual_layers=layer_info,actual_nesting=nesting,contract_sha256=contract.digest,source_identity_not_inferred_from_names=True)))
   for d in pending:
    feature=d['target_id'].split(':')[0];d['source_layer_nesting_verified']=nesting[feature]['actual_three_layer_source_nesting_verified'];d['actual_layer_context_artifact']=dict(path=str(path),sha256=file_sha256(path));writer.write(json.dumps(_plain(d),ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n')
   counts['frames']+=1;frame_reports.append(dict(native_pts=pts,shot_id=sid,layer_context=str(path),sha256=file_sha256(path),nesting=nesting));print(json.dumps(dict(native_pts=pts,counts=dict(counts))),flush=True)
 contract.verify_unchanged()
 if file_sha256(public)!=code_sha:raise ValueError('Public target source reader changed within generation')
 report=dict(schema='rinkfit.actual_target_material_layer_audit.v1',counts=dict(counts),frames=frame_reports,all_target_rows_accounted=counts['rows']==4361,source_curve_source_identity_still_separate=True,world_radius_ratio_spacing_or_symmetry_used=False,solver_admitted_rows=0);write_json(out/'manifest.json',report);return report

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--project',required=True);p.add_argument('--run',required=True);p.add_argument('--source',required=True);p.add_argument('--out',required=True);a=p.parse_args();print(json.dumps(ring_audit(a.project,a.run,a.source,a.out)))
