"""Real native-image matching of local print components across selected frames.

A homography only navigates correspondence.  All output measurements remain
source-native contour readings; protected windows/cuts are never traversed.
"""
from pathlib import Path
from collections import defaultdict,Counter
import argparse,json,re
import cv2
import numpy as np
from scipy.spatial import cKDTree
from .io_utils import read_json,read_jsonl,write_json,write_jsonl,file_sha256,object_sha256
from .role_guard import FrameRoleGuard


def match_components(audit_dir,native_dir,role_file,output_dir):
 audit=Path(audit_dir);native=Path(native_dir);out=Path(output_dir)
 if out.exists():raise ValueError('An immutable component matching generation is required')
 guard=FrameRoleGuard(role_file);rows=list(read_jsonl(audit/'relocated_rows.jsonl.gz'));groups=defaultdict(list);frames={int(f['native_pts']):f for f in read_json(native/'manifest.json')['frames']}
 for r in rows:
  if r.get('target_id'):continue
  m=re.search(r'Ad(\d+)\.',r['source_feature_id'])
  if m:groups[(r['shot_id'],int(m.group(1)),r['native_pts'],r['provenance_group_id'])].append(r)
 components={};panels=defaultdict(list)
 for (sid,ad,pts,gid),members in groups.items():
  a=np.asarray([r['current_uv'] for r in members if r.get('current_uv') is not None],float)
  if len(a)<6:continue
  typ='hole' if 'holes' in members[0]['source_feature_id'] else 'outer'
  components[gid]={'group_id':gid,'shot_id':sid,'ad_number':ad,'native_pts':pts,'uv':a,'kind':typ,'source_row_ids':[r['source_row_id'] for r in members]};panels[(sid,ad,pts)].append(gid)
 mask_records={}
 data_root=next(x for x in native.parents if x.name=='RinkSpace_Data')
 for sid in {r['shot_id'] for r in rows}:
  nr=int(sid[1:]);mb=data_root/'events/ankara_2026/programmes/okada_mei_sp/results'/f'shot{nr}_review_20261014';mp=mb/'r2/frame_manifest.json' if nr==2 else mb/'frame_manifest.json'
  for mf in read_json(mp)['frames']:mask_records[(sid,int(mf.get('native_pts',mf.get('source_pts'))))]=mf
 sift=cv2.SIFT_create(nfeatures=1500,contrastThreshold=.025,edgeThreshold=12);feature_cache={};frame_cache={};receipts=[];matches=[];parent={g:g for g in components}
 def find(x):
  while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
  return x
 def union(x,y):
  a,b=find(x),find(y)
  if a!=b:parent[max(a,b)]=min(a,b)
 def features(sid,ad,pts):
  key=(sid,ad,pts)
  if key in feature_cache:return feature_cache[key]
  guard.assert_training_pixels([pts],shot_id=sid);fr=frames[pts];png=native/fr['native_png']
  if file_sha256(png)!=fr['native_png_sha256']:raise ValueError('Native source changed')
  if pts not in frame_cache:
   if len(frame_cache)>=3:frame_cache.pop(next(iter(frame_cache)))
   frame_cache[pts]=cv2.imread(str(png),cv2.IMREAD_GRAYSCALE)
  gray=frame_cache[pts];alluv=np.concatenate([components[g]['uv'] for g in panels[key]]);lo=np.maximum(np.floor(alluv.min(0)).astype(int)-8,0);hi=np.minimum(np.ceil(alluv.max(0)).astype(int)+9,[1920,1080]);mask=np.zeros_like(gray);mask[lo[1]:hi[1],lo[0]:hi[0]]=255
  mr=mask_records[(sid,pts)];mp=Path(mr.get('mask_path') or mr.get('source',{}).get('mask_path'));expected=mr.get('mask_sha256') or mr.get('source',{}).get('mask_sha256')
  if expected and file_sha256(mp)!=expected:raise ValueError('Native actor mask changed')
  pm=cv2.imread(str(mp),cv2.IMREAD_GRAYSCALE)
  if pm is None or pm.shape!=gray.shape:raise ValueError('Native actor exclusion mask unavailable')
  mask[pm>0]=0;mask[:120,:600]=0;mask[900:,1600:]=0
  k,d=sift.detectAndCompute(gray,mask);a=np.asarray([x.pt for x in k],float);feature_cache[key]=(a,d,lo.tolist()+hi.tolist())
  return feature_cache[key]
 byad=defaultdict(list)
 for sid,ad,pts in panels:byad[(sid,ad)].append(pts)
 for (sid,ad),times in sorted(byad.items()):
  times=sorted(times)
  for first,last in zip(times[:-1],times[1:]):
   receipt={'shot_id':sid,'physical_advertisement_number':ad,'first_pts':first,'last_pts':last,'model_projection_used':False}
   try:guard.assert_training_track(first,last,shot_id=sid)
   except ValueError as e:receipt.update(status='blocked_by_preexisting_validation_guard',reason=str(e));receipts.append(receipt);continue
   a,da,ra=features(sid,ad,first);b,db,rb=features(sid,ad,last)
   if da is None or db is None or min(len(a),len(b))<8:receipt.update(status='insufficient_native_print_features');receipts.append(receipt);continue
   bf=cv2.BFMatcher(cv2.NORM_L2);ab=bf.knnMatch(da,db,k=2);ba=bf.knnMatch(db,da,k=2);reverse={m.queryIdx:m.trainIdx for m,n in ba if m.distance<.75*n.distance};pairs=[(m.queryIdx,m.trainIdx) for m,n in ab if m.distance<.75*n.distance and reverse.get(m.trainIdx)==m.queryIdx]
   if len(pairs)<8:receipt.update(status='insufficient_mutual_native_feature_matches',mutual_matches=len(pairs));receipts.append(receipt);continue
   ia=np.asarray([a[i] for i,j in pairs]);ib=np.asarray([b[j] for i,j in pairs]);H,inside=cv2.findHomography(ia,ib,cv2.RANSAC,2.,maxIters=3000,confidence=.999)
   if H is None:receipt.update(status='native_feature_navigation_failed');receipts.append(receipt);continue
   good=inside.ravel().astype(bool);trans=cv2.perspectiveTransform(ia.astype(np.float64).reshape(-1,1,2),H).reshape(-1,2);err=np.linalg.norm(trans-ib,axis=1);fraction=float(good.mean());med=float(np.median(err[good]))
   if good.sum()<8 or fraction<.45 or med>1.25:receipt.update(status='native_feature_navigation_unreliable',inliers=int(good.sum()),inlier_fraction=fraction,inlier_median_px=med);receipts.append(receipt);continue
   receipt.update(status='native_feature_navigation_supported',homography=H.tolist(),mutual_matches=len(pairs),inliers=int(good.sum()),inlier_fraction=fraction,inlier_median_px=med,source_rois=[ra,rb],independent_world_measurements_generated=False)
   count=0
   for gid in panels[(sid,ad,first)]:
    component=components[gid];uv=component['uv'];nav=cv2.perspectiveTransform(uv.astype(np.float64).reshape(-1,1,2),H).reshape(-1,2);scores=[]
    for target in panels[(sid,ad,last)]:
     other=components[target]
     if other['kind']!=component['kind']:continue
     q=other['uv'];d1=cKDTree(q).query(nav)[0];d2=cKDTree(nav).query(q)[0];score=min(float(np.mean(d1<=2.5)),float(np.mean(d2<=2.5)));scores.append((score,float(np.median(np.r_[d1,d2])),target))
    scores.sort(reverse=True)
    if scores and scores[0][0]>=.70 and scores[0][1]<=1.5 and (len(scores)==1 or scores[0][0]-scores[1][0]>=.25):
     target=scores[0][2];union(gid,target);count+=1;matches.append({'source_group_a':gid,'source_group_b':target,'first_pts':first,'last_pts':last,'shot_id':sid,'physical_advertisement_number':ad,'reciprocal_contour_support_fraction':scores[0][0],'median_native_contour_distance_px':scores[0][1],'runner_up_support_fraction':scores[1][0] if len(scores)>1 else 0.,'all_uv_from_each_frame_native_pixels':True,'model_geometry_used':False})
   receipt['actual_contour_correspondences']=count;receipts.append(receipt)
   print(json.dumps({'shot_id':sid,'ad':ad,'first_pts':first,'last_pts':last,'matches':count}),flush=True)
 clusters=defaultdict(list)
 for gid in components:clusters[find(gid)].append(gid)
 registry=[];counts=Counter()
 for anchor,ids in sorted(clusters.items()):
  items=[components[g] for g in ids];views=sorted({x['native_pts'] for x in items});confirmed=len(views)>=2
  if confirmed:
   # A same-frame one-to-many merge would confuse touching letters or holes.
   if len(views)!=len(items):confirmed=False
  cid=f"native.Ad{items[0]['ad_number']}.component."+object_sha256(sorted(ids))[:20]
  registry.append({'joint_material_id':cid,'physical_surface':'board','physical_advertisement_number':items[0]['ad_number'],'operator':'ink_boundary','kind':items[0]['kind'],'source_group_ids':ids,'native_pts':views,'source_row_ids':[rid for x in items for rid in x['source_row_ids']],'cross_time_identity_passed':confirmed,'status':'source_matched_shared_finite_contour_requires_unwrapped_geometry_initialization' if confirmed else 'unmatched_or_ambiguous_actual_source_component','geometry':'same_advertisement_shared_unwrapped_contour','geometry_frozen':False,'free_depth_allowed':False,'anchor_native_pts':items[0]['native_pts'],'anchor_native_uv':items[0]['uv'].tolist(),'world_geometry_is_observed':False,'solver_eligible':False})
  counts['matched_shared_components' if confirmed else 'pending_components']+=1;counts['matched_source_rows' if confirmed else 'pending_source_rows']+=sum(len(x['source_row_ids']) for x in items)
 out.mkdir(parents=True);write_json(out/'shared_components.json',{'schema':'rinkfit.joint9.native_shared_print_components.v1','materials':registry});write_jsonl(out/'native_match_receipts.jsonl.gz',receipts);write_jsonl(out/'component_correspondences.jsonl.gz',matches)
 summary={'schema':'rinkfit.joint9.actual_print_component_matching.v1','counts':dict(counts),'native_pair_receipts':len(receipts),'actual_component_correspondences':len(matches),'protected_pair_skips':sum(x['status']=='blocked_by_preexisting_validation_guard' for x in receipts),'source_native_manifest_sha256':file_sha256(native/'manifest.json'),'role_file_sha256':file_sha256(role_file),'homography_is_navigation_only':True,'actual_source_uv_changed':False,'native_actor_and_broadcast_regions_excluded':True,'all_components_solved':False};write_json(out/'summary.json',summary);return summary
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--audit',required=True);p.add_argument('--native',required=True);p.add_argument('--roles',required=True);p.add_argument('--out',required=True);a=p.parse_args();print(json.dumps(match_components(a.audit,a.native,a.roles,a.out)))
