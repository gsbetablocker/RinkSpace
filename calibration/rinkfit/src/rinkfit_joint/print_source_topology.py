"""Whole selected-scope actual ink topology through the existing public reader.

Source ROIs only locate searches. Connected components, holes and crop/person
cuts remain separate. An isophote is a diagnostic candidate, never FIT UV.
"""
from pathlib import Path
from collections import defaultdict,Counter
from itertools import groupby
import argparse,json,gzip,hashlib,sys,time
import cv2,numpy as np
from scipy.spatial import cKDTree
from .io_utils import read_json,read_jsonl,write_json,file_sha256
from .measurement_contract import EventMeasurementContract
from .role_guard import FrameRoleGuard

def audit_prints(project,run,source,outdir):
 project=Path(project);run=Path(run);source=Path(source);out=Path(outdir)
 if out.exists():raise ValueError('New immutable actual print source generation required')
 sys.path[:0]=[str(project),str(project/'src')]
 from calibration.program_tracking.runtime.bootstrap import add_geometry_dependencies
 add_geometry_dependencies()
 import torch
 from rinkspace.vision_data import extract_advertisement_ink_contours
 torch.set_num_threads(2);cv2.setNumThreads(1)
 native=run/'native_selected_v2';frames={int(f['native_pts']):f for f in read_json(native/'manifest.json')['frames']};guard=FrameRoleGuard(run/'roles/frame_roles.json');contract=EventMeasurementContract(project,'ankara_2026');data=next(x for x in run.parents if x.name=='RinkSpace_Data');masks={}
 for sid in sorted({f['shot_id'] for f in frames.values()}):
  n=int(sid[1:]);folder=data/'events/ankara_2026/programmes/okada_mei_sp/results'/f'shot{n}_review_20261014';mp=folder/'r2/frame_manifest.json' if n==2 else folder/'frame_manifest.json'
  for f in read_json(mp)['frames']:masks[(sid,int(f.get('native_pts',f.get('source_pts'))))]=f
 out.mkdir(parents=True);(out/'source_contours').mkdir();counts=Counter();reviews=[];start=time.perf_counter();public=project/'src/rinkspace';refs=[dict(path=str(public/n),sha256=file_sha256(public/n)) for n in ('vision_data.py','vision_geometry.py','vision_instances.py')];write_json(out/'policy.json',dict(public_reader='rinkspace.vision_data.extract_advertisement_ink_contours',public_code_references=refs,source_uv_used_only_for_search_ROI=True,no_box_or_missing_outer_stroke_fabricated=True,holes_and_disconnected_components_separate=True,no_morphology=True,isophote_not_qualified_material_edge=True,measurement_contract_sha256=contract.digest,authority_references=contract.references))
 with gzip.open(out/'source_row_topology_diagnostics.jsonl.gz','wt',encoding='utf8') as writer:
  for (sid,pts),items in groupby(read_jsonl(source),key=lambda r:(r['shot_id'],int(r['native_pts']))):
   rows=[r for r in items if str(r.get('target_id') or '').startswith('Ad')]
   if not rows:continue
   counts['requested_rows']+=len(rows);guard.assert_training_pixels([pts],shot_id=sid);fr=frames[pts];png=native/fr['native_png']
   if file_sha256(png)!=fr['native_png_sha256']:raise ValueError('Actual print source PNG changed')
   bgr=cv2.imread(str(png));rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);mr=masks[(sid,pts)];mp=Path(mr.get('mask_path') or mr.get('source',{}).get('mask_path'));mask=cv2.imread(str(mp),0)>0;mask[:170,:620]=True;mask[900:,1600:]=True;binding=dict(source_sha256='e8b53966d56e72f3842d92c3ecbe6e9cc7e4670b7d0c9bd12f58fd2cff5ad57d',source_pts=pts,time_base='1/12800',shot_id=sid,native_hw=[1080,1920],image_domain='native_brown_pixels',decoded_native_rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest());tensor=torch.from_numpy(rgb.copy()).to('cuda').permute(2,0,1).unsqueeze(0).float()/255.;ads=defaultdict(list);frame_receipt=[]
   for r in rows:ads[r['target_id'].split('.')[0]].append(r)
   for ad,members in ads.items():
    cloud=np.asarray([r.get('current_uv') or r['source_uv'] for r in members],float);lo=np.maximum([0,0],np.floor(cloud.min(0)-12).astype(int));hi=np.minimum([1920,1080],np.ceil(cloud.max(0)+13).astype(int));roi=[int(v) for v in np.r_[lo,hi]]
    if min(hi-lo)<4:
     for r in members:writer.write(json.dumps(dict(source_row_id=r['source_row_id'],shot_id=sid,native_pts=pts,target_id=r['target_id'],status='source_navigation_ROI_outside_native_image_or_too_small',solver_eligible=False,source_uv_changed=False))+'\n');counts['rows']+=1
     continue
    measured=extract_advertisement_ink_contours(tensor,roi,source_binding=binding,content_family=ad,instance_candidates=[int(ad[2:])],minimum_contour_points=8);contours=measured['contours'] if isinstance(measured,dict) and 'contours' in measured else measured
    if isinstance(contours,dict):raise ValueError('Unexpected public actual contour reader contract')
    visible=[];allxy=[];alllabel=[]
    for ci,c in enumerate(contours):
     q=np.asarray(c['uv'],float);ii=np.rint(q).astype(int);good=~mask[ii[:,1],ii[:,0]];indices=np.flatnonzero(good)
     for indices in np.split(indices,np.flatnonzero(np.diff(indices)>1)+1):
      if len(indices)<2:continue
      v=dict(c);v['uv']=q[indices].tolist();v['normal']=np.asarray(c['normal'])[indices].tolist();v['closed']=bool(c['closed'] and good.all());v['actual_contour_index']=ci;v['source_actor_or_graphics_cut']=not bool(good.all());v['physical_identity_assigned']=False;visible.append(v);allxy.extend(q[indices].tolist());alllabel.extend([ci]*len(indices))
    path=out/'source_contours'/f'{pts}_{ad}.json';write_json(path,dict(source_binding=binding,native_png=str(png),native_png_sha256=fr['native_png_sha256'],current_mask=str(mp),current_mask_sha256=file_sha256(mp),search_roi_only=roi,visible_actual_ink_contours=visible,material_isophote_is_not_FIT_UV=True,actual_panel_identity_not_inferred_from_ROI_or_name=True));tree=cKDTree(np.asarray(allxy)) if allxy else None
    for r in members:
     q=np.asarray(r.get('current_uv') or r['source_uv']);distance,label=(tree.query(q) if tree is not None else (float('inf'),None));d=dict(source_row_id=r['source_row_id'],shot_id=sid,native_pts=pts,target_id=r['target_id'],source_family_id=r.get('source_family_id'),native_source_uv=r['source_uv'],source_navigation_uv=q.tolist(),actual_ink_contour_distance_px=float(distance) if np.isfinite(distance) else None,actual_contour_index=alllabel[int(label)] if label is not None else None,source_visibility_excluded=bool(mask[int(np.clip(round(q[1]),0,1079)),int(np.clip(round(q[0]),0,1919))]),topology_artifact=dict(path=str(path),sha256=file_sha256(path)),status='actual_isophote_near_navigation_requires_finite_identity_and_signed_interface_review' if distance<=3 else 'actual_ink_isophote_not_supported_near_navigation',solver_eligible=False,source_uv_changed=False);writer.write(json.dumps(d,separators=(',',':'))+'\n');counts[d['status']]+=1;counts['rows']+=1
    counts['actual_contour_fragments']+=len(visible);counts['panel_source_ROIs']+=1;frame_receipt.append(dict(advertisement_candidate=ad,roi=roi,actual_contours=len(visible),path=str(path),sha256=file_sha256(path)))
   reviews.append(dict(shot_id=sid,native_pts=pts,panel_sources=frame_receipt));counts['frames']+=1;print(json.dumps(dict(native_pts=pts,counts=dict(counts))),flush=True)
 contract.verify_unchanged()
 for ref in refs:
  if file_sha256(ref['path'])!=ref['sha256']:raise ValueError('Public source reader changed within generation')
 result=dict(schema='rinkfit.all_selected_actual_print_topology_audit.v1',counts=dict(counts),frames=reviews,all_current_source_ad_rows_accounted=counts['rows']==counts['requested_rows'],model_or_camera_predictions_used=False,actual_source_contours_have_not_become_solver_UV=True,source_identity_and_signed_transition_qualification_pending=True,elapsed_seconds=time.perf_counter()-start);write_json(out/'manifest.json',result);return result

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--project',required=True);p.add_argument('--run',required=True);p.add_argument('--source',required=True);p.add_argument('--out',required=True);a=p.parse_args();print(json.dumps(audit_prints(a.project,a.run,a.source,a.out)))
