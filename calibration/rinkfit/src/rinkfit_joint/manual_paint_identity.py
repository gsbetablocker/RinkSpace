"""Direct human finite-curve anchors + actual unbroken current paint support."""
from pathlib import Path
from collections import defaultdict,Counter
import argparse,json,hashlib
import cv2,numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import spline_filter
from .io_utils import read_json,read_jsonl,write_json,write_jsonl,file_sha256,object_sha256
from .role_guard import FrameRoleGuard
from .physical_source_operator import measure_paint_midline
from .source_worker import _channels
from .measurement_contract import EventMeasurementContract

def plain(x):
 if isinstance(x,np.ndarray):return plain(x.tolist())
 if isinstance(x,np.generic):return x.item()
 if isinstance(x,float) and not np.isfinite(x):return None
 if isinstance(x,dict):return {k:plain(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [plain(v) for v in x]
 return x


def identify(project,run,statistics,outdir):
 project=Path(project);run=Path(run);out=Path(outdir)
 import sys
 sys.path[:0]=[str(project/'src'),str(project)]
 from rinkspace.vision_source_shapes import _MaterialProfileFrame,read_source_stroke_midlines
 if out.exists():raise ValueError('New immutable manual-anchored paint identity generation required')
 pairs=read_json(Path(statistics)/'source_pairs.json');manual=list(read_jsonl(run/'manual_source_geometry_v2/geometry_rows.jsonl.gz'));groups=defaultdict(list)
 for r in manual:
  if r['measurement_operator_resolved']=='paint_center':groups[(r['native_pts'],r['target_id'])].append(r)
 native=run/'native_selected_v2';frames={int(f['native_pts']):f for f in read_json(native/'manifest.json')['frames']};ledgers={int(r['native_pts']):r for n in ('dense_s03_v1','dense_remaining_v1') for r in read_jsonl(run/n/'frame_ledger.jsonl.gz')};guard=FrameRoleGuard(run/'roles/frame_roles.json');contract=EventMeasurementContract(project,'ankara_2026');byframe=defaultdict(list)
 for pair in pairs:byframe[(pair['shot_id'],int(pair['native_pts']))].append(pair)
 out.mkdir(parents=True);(out/'frame_sources').mkdir();results=[];accepted=[];counts=Counter();duplicate={};family_records=[]
 for (sid,pts),current in sorted(byframe.items()):
  framegroups={target:r for (p,target),r in groups.items() if p==pts}
  if not framegroups:
   for pair in current:results.append(dict(source_row_id=pair['source_row_id'],source_pair_id='paint_pair/'+pair['source_row_id'],native_pts=pts,shot_id=sid,target_id=pair['target_id'],finite_identity_passed=False,reasons=['no_direct_manual_same_exposure_finite_material_anchor_yet'],raw_preserved=True));counts['no_direct_manual_same_exposure_finite_material_anchor_yet']+=1
   continue
  guard.assert_training_pixels([pts],shot_id=sid);fr=frames[pts];png=native/fr['native_png']
  if file_sha256(png)!=fr['native_png_sha256']:raise ValueError('Actual current manual anchor PNG changed')
  rgb=cv2.cvtColor(cv2.imread(str(png)),cv2.COLOR_BGR2RGB);sha=hashlib.sha256(rgb.tobytes()).hexdigest();mask=np.zeros((1080,1920),bool)
  for b in ledgers[pts].get('current_person_boxes',[]):
   binding=b['source_binding']
   if binding.get('source_pts')!=pts or binding.get('decoded_native_rgb_sha256')!=sha:raise ValueError('Person box binding differs')
   q=np.asarray(b['box_xyxy']);a=np.maximum([0,0],np.floor(q[:2]-4).astype(int));e=np.minimum([1920,1080],np.ceil(q[2:]+4).astype(int));mask[a[1]:e[1],a[0]:e[0]]=True
  source_manifest=read_json(next(p for p in run.parents if p.name=='RinkSpace_Data')/'events/ankara_2026/programmes/okada_mei_sp/results/shot2_review_20261014/r2/frame_manifest.json');mr=next(f for f in source_manifest['frames'] if int(f.get('native_pts',f.get('source_pts')))==pts)
  if mr['decoded_native_RGB_sha256']!=sha or file_sha256(mr['mask_path'])!=mr['mask_sha256']:raise ValueError('Current pixel mask differs from bound actual exposure')
  mask=cv2.imread(mr['mask_path'],0)>0;mask[:170,:620]=True;channels=_channels(rgb);coeff=np.asarray([spline_filter(c,order=3) for c in channels]);curves={};receipts=[];binding=dict(source_sha256=read_json(run/'programme_scope.json')['source']['sha256'],source_pts=pts,time_base='1/12800',shot_id=sid,native_hw=[1080,1920],decoded_native_rgb_sha256=sha);source_frame=_MaterialProfileFrame(rgb,binding)
  for target,anchors in framegroups.items():
   a=np.asarray([r['source_uv'] for r in anchors]);center=a.mean(0)
   if len(a)<2:continue
   _,sv,V=np.linalg.svd(a-center,full_matrices=False);axis=V[0];normal=np.array([-axis[1],axis[0]]);s=(a-center)@axis;y=(a-center)@normal;circle=target.startswith('ice.C0') or any(target.startswith('ice.A'+str(i)+':') for i in range(1,5))
   if circle:
    if len(a)<5:continue
    try:e=cv2.fitEllipse(a.astype(np.float32))
    except cv2.error:continue
    center=np.asarray(e[0]);radii=np.asarray(e[1])*.5;theta=np.radians(e[2]);R=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    if radii.min()<1 or radii.max()/radii.min()>100 or radii.max()>1e5:continue
    local=(a-center)@R;s=np.mod(np.arctan2(local[:,1]/radii[1],local[:,0]/radii[0]),2*np.pi);ordered=np.sort(s);gaps=np.diff(np.r_[ordered,ordered[0]+2*np.pi]);start=float(ordered[(int(np.argmax(gaps))+1)%len(ordered)]);s=np.where(s<start,s+2*np.pi,s)
    def point(v,center=center,radii=radii,R=R):return center+np.c_[np.cos(v)*radii[0],np.sin(v)*radii[1]]@R.T
    def derivative(v,radii=radii,R=R):return np.c_[-np.sin(v)*radii[0],np.cos(v)*radii[1]]@R.T
    station_scale=float(radii.max());fitcurve=point(s);error=np.linalg.norm(fitcurve-a,axis=1)
   else:
    scale=max(float(np.ptp(s)),1.);poly=np.polyfit(s/scale,y,1);error=abs(np.polyval(poly,s/scale)-y)
    def point(v,center=center,axis=axis,normal=normal,poly=poly,scale=scale):return center+np.asarray(v)[:,None]*axis+np.polyval(poly,np.asarray(v)/scale)[:,None]*normal
    def derivative(v,axis=axis,normal=normal,poly=poly,scale=scale):return axis+np.polyval(np.polyder(poly),np.asarray(v)/scale)[:,None]/scale*normal
    station_scale=1.
   if max(error)>8:continue
   dense_s=np.linspace(0.,2*np.pi,1024) if circle else np.linspace(s.min()-2000,s.max()+2000,1024);dense=point(dense_s);curves[target]=dict(anchors=anchors,manual=a,s=s,point=point,derivative=derivative,station_scale=station_scale,dense_s=dense_s,dense=dense,tree=cKDTree(dense),manual_fit_max_px=float(max(error)),image_curve_only=True,is_circle=circle)
  for pair in current:
   rid=pair['source_row_id'];target=pair['target_id'];key=object_sha256(dict(native_pts=pts,flanks=pair['actual_flank_uv'],target=target));alias=duplicate.get(key);duplicate.setdefault(key,rid);reasons=[];bridge=[];curve=curves.get(target);mid=np.mean(pair['actual_flank_uv'],axis=0);original_target=target
   associations=sorted((float(c['tree'].query(mid)[0]),name) for name,c in curves.items())
   if associations and associations[0][0]<=8 and (len(associations)==1 or associations[1][0]-associations[0][0]>=8):
    target=associations[0][1];curve=curves[target]
   elif curve is None:curve=None
   if alias is not None:reasons.append('exact_same_actual_source_readout_alias_of_first')
   if curve is None:reasons.append('same_target_manual_finite_image_curve_not_supported')
   else:
    distance,index=curve['tree'].query(mid);station=float(curve['dense_s'][index]);manual_dist=np.linalg.norm(curve['manual']-mid,axis=1);anchor_index=int(np.argmin(manual_dist));anchor_station=float(curve['s'][anchor_index])
    if curve.get('is_circle'):anchor_station+=float(np.round((station-anchor_station)/(2*np.pi))*2*np.pi)
    if distance>8:reasons.append('actual_pair_not_on_direct_manual_named_finite_curve')
    if not reasons:
     samples=np.linspace(anchor_station,station,max(3,int(abs(station-anchor_station)*curve['station_scale']/20)+1));points=curve['point'](samples);deriv=curve['derivative'](samples);normals=np.c_[-deriv[:,1],deriv[:,0]];normals/=np.linalg.norm(normals,axis=1,keepdims=True);signal=pair['source_material_receipt']['material_signal'];widths=[];same=True
     for q,n in zip(points,normals):
      m=measure_paint_midline(rgb,mask,q,n,confirmed_signal=signal,channels=channels,spline_channels=coeff);ok=m.get('current_uv') is not None
      if not ok and m.get('status')=='no_unique_same_paint_two_flank_material_pair':
       weights={'red':[1.,-1.,0.],'blue':[-.5,-.5,1.],'dark':None}[signal];rr=read_source_stroke_midlines(source_frame.rgb,binding=source_frame.binding,search_points_uv=[q],normals=[n],entering_search_intervals_px=[[-24.,0.]],leaving_search_intervals_px=[[0.,24.]],material_signal_rgb_weights=weights,paint_signal_increases=signal!='dark',expected_RGB_sha256=sha,keep_profiles=True,_validated_frame=source_frame);reading=rr['readings'][0];uv=np.asarray(reading['actual_source_uv']);ok=uv.shape==(2,) and np.isfinite(uv).all()
       m={**m,'status':'existing_public_actual_signed_stroke_identity_only' if ok else m['status'],'current_uv':uv.tolist() if ok else None,'public_source_signed_stroke':plain(reading),'not_an_additional_observation_or_new_noise_model':True,'width_px':float(np.diff(reading['observed_paint_side_q50_normal_offsets_px'])[0]) if ok else None}
      bridge.append(dict(navigation_seed_uv=q.tolist(),actual_current_material_reading=m,used_as_additional_fit_observation=False))
      if not ok:same=False;break
      widths.append(m['width_px'])
     if not same:reasons.append('current_material_path_to_manual_anchor_has_visible_gap_occlusion_or_ambiguous_stroke')
     elif np.max(np.abs(np.asarray(bridge[-1]['actual_current_material_reading']['current_uv'])-mid))>3:reasons.append('unbroken_actual_material_path_reaches_other_pair_not_current_mean')
   passed=not reasons;result=dict(source_row_id=rid,source_pair_id='paint_pair/'+rid,native_pts=pts,shot_id=sid,target_id=target,original_target_id=original_target,source_binding_corrected_from_direct_same_exposure_manual_material=target!=original_target,finite_identity_passed=passed,source_local_two_flank_operator_passed=True,actual_4D_source_mean_sha256=pair['source_mean_sha256'],source_pair_hash_kind='actual_flank_uv4D_not_old_midpoint2D',reasons=reasons,exact_same_actual_source_readout_alias_of_first=alias,raw_preserved=True,measurement_contract_sha256=contract.digest)
   counts.update(reasons or ['direct_manual_finite_curve_and_unbroken_actual_material_identity_passed']);results.append(result);receipts.append(dict(decision=result,actual_current_pair=pair,manual_source_anchor_row_ids=[r['source_row_id'] for r in curve['anchors']] if curve else [],manual_source_authority=run.as_posix()+'/manual_source_geometry_v2/manifest.json',actual_unbroken_current_RGB_material_path=bridge,manual_image_curve_fit_max_px=curve['manual_fit_max_px'] if curve else None,world_geometry_or_model_projection_used=False))
   if passed:accepted.append(dict(pair=pair,actual_manual_source_target_id=target,identity_decision=result,manual_source_anchors=[{k:r.get(k) for k in ('source_row_id','target_id','source_uv','receipt_sha256','measurement_contract_certificate')} for r in curve['anchors']]))
  path=out/'frame_sources'/f'{pts}.json';write_json(path,dict(native_pts=pts,shot_id=sid,native_png=str(png),native_png_sha256=fr['native_png_sha256'],actual_native_RGB_sha256=sha,measurement_contract_sha256=contract.digest,actual_paint_identity_receipts=receipts,source_pixels_from_model=False,manual_point_coordinates_changed=False));family_records.append(dict(path=str(path),sha256=file_sha256(path)))
 for q in accepted:q['actual_finite_source_proof']=next(ref for ref in family_records if Path(ref['path']).stem==str(q['pair']['native_pts']))
 write_jsonl(out/'identity_decisions.jsonl.gz',results);write_json(out/'qualified_pairs.json',accepted);report=dict(schema='rinkfit.direct_manual_finite_paint_identity.v1',source_pairs=len(pairs),finite_identity_passed_pairs=len(accepted),counts=dict(counts),all_pairs_accounted=len(results)==len(pairs),only_actual_user_manual_anchors_used=True,old_world_ideal_relations_used=False,source_pair_current_mean_changed=False,manual_UV_changed=False,statistical_eligibility_requires_typed4D_gate=True,frame_source_proofs=family_records,source_statistics=dict(path=str(Path(statistics)/'manifest.json'),sha256=file_sha256(Path(statistics)/'manifest.json')));write_json(out/'manifest.json',report);contract.verify_unchanged();print(json.dumps(report));return report
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--project',required=True);p.add_argument('--run',required=True);p.add_argument('--statistics',required=True);p.add_argument('--out',required=True);a=p.parse_args();identify(a.project,a.run,a.statistics,a.out)
