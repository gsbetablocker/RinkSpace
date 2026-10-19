"""Actual finite ink components: current topology + board scope + RGB identity.

Old Ad numbers identify navigation ROIs only. Output materials use independent
physical component UIDs and never share a panel merely because a sponsor repeats.
"""
from pathlib import Path
from collections import defaultdict,Counter
import argparse,json,gzip,hashlib,sys
import cv2,numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import map_coordinates
from .io_utils import read_json,read_jsonl,write_json,write_jsonl,file_sha256,object_sha256
from .role_guard import FrameRoleGuard
from .measurement_contract import EventMeasurementContract
from .board_outer_source import source_outer_boundary_rows
from .current_person_exclusion import current_person_exclusion_mask
from .outer_cubic_source import prepare_cubic_channels,replay_outer_cubic,CONFIG as REPLAY_CONFIG

def plain(x):
 if isinstance(x,np.ndarray):return plain(x.tolist())
 if isinstance(x,np.generic):return plain(x.item())
 if isinstance(x,float) and not np.isfinite(x):return None
 if isinstance(x,dict):return {k:plain(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [plain(q) for q in x]
 return x

def poly_distance(points,poly):
 a=np.asarray(poly);v=np.roll(a,-1,axis=0)-a;p=np.asarray(points);rel=p[:,None]-a;f=np.einsum('nki,ki->nk',rel,v)/np.maximum(np.einsum('ki,ki->k',v,v),1e-12);q=a+np.clip(f,0,1)[:,:,None]*v;return np.linalg.norm(p[:,None]-q,axis=2).min(1)

def board_scope(q,outer):
 out=[]
 for role in ('board.top','board.contact'):
  a=np.asarray([r['current_uv'] for r in outer if r['material_role']==role],float)
  if a.shape[0]<2:return False
  yy=[]
  for point in q:
   dist=np.abs(a[:,0]-point[0]);ids=np.flatnonzero(dist<=24)
   if not len(ids):return False
   yy.append(float(np.median(a[ids,1])))
  out.append(np.asarray(yy))
 return bool(np.all(q[:,1]>out[0]+2) and np.all(q[:,1]<out[1]-2))

def audit(project,run,outdir):
 project=Path(project);run=Path(run);out=Path(outdir)
 if out.exists():raise ValueError('New immutable print source generation required')
 sys.path[:0]=[str(project),str(project/'src')]
 from calibration.program_tracking.runtime.bootstrap import add_geometry_dependencies
 add_geometry_dependencies()
 import torch
 from rinkspace.vision_data import extract_advertisement_ink_contours
 torch.set_num_threads(2);cv2.setNumThreads(1)
 contract=EventMeasurementContract(project,'ankara_2026');guard=FrameRoleGuard(run/'roles/frame_roles.json');native=run/'native_selected_v2';frames={int(f['native_pts']):f for f in read_json(native/'manifest.json')['frames']};ledgers={int(r['native_pts']):r for n in ('dense_s03_v1','dense_remaining_v1') for r in read_jsonl(run/n/'frame_ledger.jsonl.gz')};components=[m for m in read_json(run/'component_identity_v2/shared_components.json')['materials'] if m['cross_time_identity_passed']];ids={rid:m for m in components for rid in m['source_row_ids']};original={r['source_row_id']:r for r in read_jsonl(run/'source_audit_v10/relocated_rows.jsonl.gz') if r['source_row_id'] in ids};nav={r['source_row_id']:r for r in read_jsonl(run/'source_audit_v5/relocated_rows.jsonl.gz') if r['source_row_id'] in ids};byframe=defaultdict(list)
 for rid,r in original.items():byframe[(r['shot_id'],int(r['native_pts']))].append(r)
 out.mkdir(parents=True);(out/'frame_sources').mkdir();code_refs=[dict(path=str(Path(__file__)),sha256=file_sha256(__file__)),dict(path=str(project/'src/rinkspace/vision_data.py'),sha256=file_sha256(project/'src/rinkspace/vision_data.py'))];counts=Counter();allrows=[];framefacts=[];familyframes=defaultdict(set)
 for (sid,pts),members in sorted(byframe.items()):
  guard.assert_training_pixels([pts],shot_id=sid);fr=frames[pts];png=native/fr['native_png']
  if file_sha256(png)!=fr['native_png_sha256']:raise ValueError('Actual original PNG changed')
  bgr=cv2.imread(str(png));rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);bgrsha=hashlib.sha256(bgr.tobytes()).hexdigest()
  if bgrsha!=ledgers[pts]['decoded_native_BGR_sha256']:raise ValueError('Exact source RGB binding changed')
  mask,maskreceipt=current_person_exclusion_mask(rgb,ledgers[pts].get('current_person_boxes',[]),native_pts=pts);outer,outercert=source_outer_boundary_rows(rgb,mask);coeff=prepare_cubic_channels(rgb);binding=dict(source_sha256=read_json(run/'programme_scope.json')['source']['sha256'],source_pts=pts,time_base='1/12800',shot_id=sid,native_hw=[1080,1920],image_domain='native_brown_pixels',decoded_native_rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest());tensor=torch.from_numpy(rgb.copy()).to('cuda').permute(2,0,1).unsqueeze(0).float()/255.;families=defaultdict(list);facts=[]
  for r in members:families[ids[r['source_row_id']]['joint_material_id']].append(r)
  for oldid,source in families.items():
   material=ids[source[0]['source_row_id']];uid=oldid.split('.')[-1];target='print_component.'+uid;cloud=np.asarray([nav[r['source_row_id']].get('current_uv') or r['source_uv'] for r in source],float);lo=np.maximum([0,0],np.floor(cloud.min(0)-12).astype(int));hi=np.minimum([1920,1080],np.ceil(cloud.max(0)+13).astype(int));roi=[int(x) for x in np.r_[lo,hi]];why=[];contours=[];selected=None
   if not board_scope(cloud,outer):why.append('actual_component_not_enclosed_by_visible_current_physical_board_outer_material')
   if min(hi-lo)<6:why.append('current_component_navigation_roi_insufficient')
   if not why:
    measured=extract_advertisement_ink_contours(tensor,roi,source_binding=binding,content_family=target,instance_candidates=[],minimum_contour_points=8);contours=measured.get('contours',[]) if isinstance(measured,dict) else measured;scores=[]
    for ci,c in enumerate(contours):
     poly=np.asarray(c['uv'],float)
     if not c.get('closed') or len(poly)<8:continue
     if np.any(poly<=np.asarray(lo)+1) or np.any(poly>=np.asarray(hi)-2):continue
     ix=np.rint(poly).astype(int)
     if mask[ix[:,1],ix[:,0]].any():continue
     d1=cKDTree(poly).query(cloud)[0];d2=poly_distance(poly,cloud);score=min(float(np.mean(d1<=3)),float(np.mean(d2<=3)));scores.append((score,float(np.median(np.r_[d1,d2])),ci))
    scores.sort(reverse=True)
    if not scores or scores[0][0]<.7 or scores[0][1]>2.5 or (len(scores)>1 and scores[0][0]-scores[1][0]<.15):why.append('no_unique_actual_closed_ink_component_matching_cross_time_source')
    else:selected=contours[scores[0][2]]
   local=[];actualpoly=None;signal=None
   if selected is not None:
    actualpoly=np.asarray(selected['uv'],float);tree=cKDTree(actualpoly);dist,index=tree.query(cloud);normals=np.asarray(selected['normal'],float)[index];length=np.linalg.norm(normals,axis=1);normals=normals/np.maximum(length[:,None],1e-12);seeds=actualpoly[index];inside=[];outside=[]
    for i,(q,n) in enumerate(zip(seeds,normals)):
     if cv2.pointPolygonTest(actualpoly.astype(np.float32),tuple((q+2*n).astype(float)),False)>0:normals[i]=-n
    xy=np.stack((seeds-2*normals,seeds+2*normals),axis=1);values=np.stack([map_coordinates(c,[xy[:,:,1].ravel(),xy[:,:,0].ravel()],order=3,mode='nearest',prefilter=False).reshape(len(seeds),2) for c in coeff],axis=2);delta=np.median(values[:,1]-values[:,0],axis=0);channel=int(np.argmax(abs(delta)));polarity=1 if delta[channel]>0 else -1;signal=dict(channel=channel,polarity=polarity,current_RGB_inside_outside_channel_deltas=delta.tolist(),selection_frozen_for_all_family_rows=True)
    if abs(delta[channel])<4:why.append('actual_component_material_pair_insufficient_current_colour_separation')
    else:
     for i,r in enumerate(source):
      seed=seeds[i];n=normals[i];receipt=dict(navigation_seed_uv=seed.tolist(),normal=n.tolist(),source_measurement_audit=dict(profile=dict(channel=channel,polarity=polarity)));a=replay_outer_cubic(rgb,receipt,coefficients=coeff);q=np.asarray(a['current_uv']);off=np.asarray(REPLAY_CONFIG['normal_offsets']);t=np.array([-n[1],n[0]]);sample=seed+np.arange(-2,3)[:,None,None]*t+off[None,:,None]*n;ii=np.rint(sample).astype(int);safe=bool((ii[:,:,0]>=0).all() and (ii[:,:,0]<1920).all() and (ii[:,:,1]>=0).all() and (ii[:,:,1]<1080).all());excluded=not safe or (mask[ii[:,:,1],ii[:,:,0]].any() if safe else True);valid=bool(length[i]>1e-8 and dist[i]<=3 and a['numeric_valid'] and not a['synthetic_window_boundary_count'] and not excluded and polarity*(values[i,1,channel]-values[i,0,channel])>=4)
      local.append((r,q,n,a,receipt,valid))
   supported=sum(a[-1] for a in local);componentpassed=bool(not why and supported>=3 and supported>=.5*len(source))
   if componentpassed:familyframes[uid].add(pts)
   fact=dict(component_uid=uid,old_candidate_ad_number_navigation_only=material['physical_advertisement_number'],cross_time_source_identity_reference=dict(path=str(run/'component_identity_v2/component_correspondences.jsonl.gz'),sha256=file_sha256(run/'component_identity_v2/component_correspondences.jsonl.gz')),physical_material_label=target,original_cross_time_groups=material['source_group_ids'],current_actual_board_scope_passed='actual_component_not_enclosed_by_visible_current_physical_board_outer_material' not in why,current_component_closed_source_path=plain(selected),material_signal=signal,qualified_source_rows=supported,source_rows=len(source),frame_source_passed=componentpassed,pending_reasons=why)
   facts.append(fact)
   byrid={a[0]['source_row_id']:a for a in local}
   for r in source:
    loc=byrid.get(r['source_row_id']);valid=bool(componentpassed and loc and loc[-1]);base={**r,'original_target_id':r.get('target_id'),'target_id':target,'category':'ad_print','physical_surface':'board','joint_material_id':target+'|board|ink_boundary|'+material['kind'],'measurement_operator_resolved':'ink_boundary','measurement_operator':'ink_boundary','current_uv':loc[1].tolist() if valid else None,'normal':loc[2].tolist() if loc else None,'source_uv':r['source_uv'],'source_row_id':r['source_row_id'],'finite_physical_identity_source_verified':False,'source_local_material_operator_verified':valid,'frame_actual_component_source_passed':componentpassed,'component_uid':uid,'candidate_Ad_number_is_navigation_only':True,'pixels_from_model':False,'current_pixels_measured':valid,'decoded_native_BGR_sha256':bgrsha,'native_png_sha256':fr['native_png_sha256'],'numeric_statistics_ready':False,'solver_eligible':False,'measurement_contract_sha256':contract.digest,'old_W_reused':False,'source_measurement_audit':dict(status='current_native_same_physical_print_component_continuous_cubic_signed_interface' if valid else 'actual_print_component_source_pending',current_uv=loc[1].tolist() if valid else None,normal=loc[2].tolist() if loc else None,source_material_receipt=dict(operator_version='actual_finite_print_component_signed_cubic_v1',frozen_point_operator=loc[4],actual_point_reading=loc[3],current_material_signal=signal,whole_exposure_RGB_replay_domain=True,actual_contour_component_uid=uid) if loc else None)};allrows.append(base);counts['rows']+=1
  framefacts.append(dict(native_pts=pts,shot_id=sid,source_binding=binding,native_png=str(png),native_png_sha256=fr['native_png_sha256'],actual_outer_material_certificates=outercert,actual_outer_rows=outer,current_exclusion_receipt=maskreceipt,actual_components=facts,code_references=code_refs,measurement_contract_sha256=contract.digest));counts['frames']+=1;print(json.dumps(dict(native_pts=pts,frames=counts['frames'],rows=counts['rows'],current_component_candidates=sum(len(v)>=2 for v in familyframes.values()))),flush=True)
 qualified=[r for r in allrows if r['source_local_material_operator_verified'] and len(familyframes[r['component_uid']])>=2]
 for r in qualified:r['finite_physical_identity_source_verified']=True;r['source_geometry_eligible']=True
 bypts=defaultdict(list)
 for r in qualified:bypts[r['native_pts']].append(r)
 for fact in framefacts:
  pts=fact['native_pts'];fact['qualified_source_rows']=bypts[pts];path=out/'frame_sources'/f'{pts}.json';write_json(path,plain(fact));ref=dict(path=str(path.resolve().relative_to(next(p for p in run.parents if p.name=='RinkSpace_Data'))),sha256=file_sha256(path),role='physical_source_definition_certificate')
  for r in bypts[pts]:r['measurement_contract_certificate']=dict(contract_sha256=contract.digest,source_row_id=r['source_row_id'],native_pts=pts,joint_material_id=r['joint_material_id'],operator='ink_boundary',physical_role=contract.expected(r['target_id'],'ad_print')['physical_role'],current_measurement_sha256=object_sha256(dict(native_pts=pts,current_uv=r['current_uv'],source_row_id=r['source_row_id'])),actual_RGB_material_identity_verified=True,operator_selected_correct_material_transition=True,pixels_from_model=False,source_evidence=[ref])
 write_jsonl(out/'source_rows.jsonl.gz',(plain(r) for r in allrows));write_jsonl(out/'geometry_rows.jsonl.gz',(plain(r) for r in qualified));write_json(out/'manifest.json',dict(schema='rinkfit.actual_finite_print_component_source_admission.v1',source_rows=len(allrows),geometry_rows=len(qualified),qualified_physical_components=len({r['component_uid'] for r in qualified}),original_strong_cross_time_components=len(components),all_strong_source_rows_accounted=len(allrows)==len(ids),physical_Ad_number_inferred=False,old_source_uv_changed=False,old_W_reused=False,numeric_statistics_ready=False,source_geometry_only_pending_same_mean_W=True,counts=dict(counts),measurement_contract_sha256=contract.digest,code_references=code_refs));contract.verify_unchanged();return read_json(out/'manifest.json')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--project',required=True);p.add_argument('--run',required=True);p.add_argument('--out',required=True);a=p.parse_args();print(json.dumps(audit(a.project,a.run,a.out)))
