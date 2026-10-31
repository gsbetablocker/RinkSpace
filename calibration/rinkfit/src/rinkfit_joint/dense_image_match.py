"""Direct native source-image matching for gaps left by sequential navigation.

Keypoint homographies only navigate actual pixel measurements. Every released
point is re-read from current RGB with a verified source-material patch.
"""

from rinkspace.storage import resolve_asset
from pathlib import Path
from collections import defaultdict,Counter
from fractions import Fraction
import argparse,json,hashlib,sys,time,re
import av,cv2,numpy as np
from .io_utils import read_json,read_jsonl,write_json,write_jsonl,file_sha256
from .role_guard import FrameRoleGuard
from .dense_native import _seed_rows,_sample_patches,_ncc,_channels
from .source_worker import _sample
from .physical_source_operator import prepare_current_source,remeasure_physical_source,_sample as _sample_current_cubic

def _panel(target):
 m=re.match(r'Ad[0-9]+',target or '')
 return m.group(0) if m else ('flat_ice' if (target or '').startswith('ice.') else 'board')

def _transform(H,points):
 q=np.c_[np.asarray(points,float),np.ones(len(points))]@H.T
 return q[:,:2]/q[:,2,None]

def recover(run,project,first_pass,reverse,source,decisions,material_map,out,previous=None):
 start=time.perf_counter();run=Path(run);project=Path(project);first=Path(first_pass);reverse=Path(reverse);out=Path(out)
 if out.exists():raise ValueError('New immutable source image-match generation required')
 scope=read_json(run/'programme_scope.json');video=Path(scope['source']['path'])
 if file_sha256(video)!=scope['source']['sha256']:raise ValueError('Wrong official video')
 ledger={r['native_pts']:r for r in read_jsonl(first/'frame_ledger.jsonl.gz')};rr=list(read_jsonl(reverse/'frame_receipts.jsonl.gz'));targets=[r for r in rr if r['status']!='actual_current_source_reacquired']
 if previous:
  recovered={r['native_pts'] for r in read_jsonl(Path(previous)/'frame_receipts.jsonl.gz') if r['source_rows']>0};targets=[r for r in targets if r['native_pts'] not in recovered]
 guard=FrameRoleGuard(run/'roles/frame_roles.json');seeds=_seed_rows(source,decisions,material_map,project=project,audit_file=out.with_name(out.name+'_seed_gate.json'));native=run/'native_selected_v2';selected={f['native_pts']:f for f in read_json(native/'manifest.json')['frames']};data_root=next(x for x in run.parents if x.name=='RinkSpace_Data');mask_sources={}
 for sid in {x['shot_id'] for x in targets}:
  nr=int(sid[1:]);base=data_root/'events/ankara_2026/programmes/okada_mei_sp/results'/f'shot{nr}_review_20261014';mp=base/'r2/frame_manifest.json' if nr==2 else base/'frame_manifest.json'
  for f in read_json(mp)['frames']:mask_sources[(sid,int(f.get('native_pts',f.get('source_pts'))))]=f
 out.mkdir(parents=True);(out/'native_cache').mkdir();(out/'mask_cache').mkdir();sift=cv2.SIFT_create(nfeatures=3200,contrastThreshold=.012,edgeThreshold=15);matcher=cv2.BFMatcher(cv2.NORM_L2);anchors={};imagecache={};person_provider=None;counts=Counter();result_rows=[];receipts=[];profiles=[];profile_index=[];puv=[];pn=[];pch=[];off=np.arange(-3,3.0001,.25);toff=np.arange(-2,3,dtype=float)
 def image_for(pts):
  if pts in imagecache:return imagecache[pts]
  paths=[reverse/'native_cache'/f'{pts}.png',out/'native_cache'/f'{pts}.png']
  if previous:paths.insert(0,Path(previous)/'native_cache'/f'{pts}.png')
  if pts in selected:paths.append(native/selected[pts]['native_png'])
  bgr=None;path=None
  for pp in paths:
   if pp.is_file():bgr=cv2.imread(str(pp));path=pp;break
  if bgr is None:
   with av.open(str(video)) as c:
    stream=c.streams.video[0]
    if Fraction(stream.time_base)!=Fraction(1,12800):raise ValueError('Non-native clock')
    c.seek(pts,backward=True,any_frame=False,stream=stream)
    for frame in c.decode(stream):
     if int(frame.pts)>pts:break
     if int(frame.pts)==pts:bgr=frame.to_ndarray(format='bgr24');break
   if bgr is None:raise ValueError('Native exposure missing')
   path=out/'native_cache'/f'{pts}.png';cv2.imwrite(str(path),bgr,[cv2.IMWRITE_PNG_COMPRESSION,2])
  sha=hashlib.sha256(bgr.tobytes()).hexdigest()
  if sha!=ledger[pts]['decoded_native_BGR_sha256']:raise ValueError('Current actual pixel identity mismatch')
  rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);gray=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY);value=(rgb,gray,sha,str(path),file_sha256(path))
  if len(imagecache)>8:imagecache.pop(next(iter(imagecache)))
  imagecache[pts]=value;return value
 def mask_for(sid,pts,rgb):
  nonlocal person_provider
  cached_masks=[reverse/'mask_cache'/f'{pts}.png',out/'mask_cache'/f'{pts}.png']
  if previous:cached_masks.insert(0,Path(previous)/'mask_cache'/f'{pts}.png')
  for file in cached_masks:
   if file.is_file():return cv2.imread(str(file),cv2.IMREAD_GRAYSCALE)>0
  if (sid,pts) in mask_sources:
   record=mask_sources[(sid,pts)];path=Path(record.get('mask_path') or record.get('source',{}).get('mask_path'));sha=record.get('mask_sha256') or record.get('source',{}).get('mask_sha256')
   if sha and file_sha256(path)!=sha:raise ValueError('Authoritative native mask changed')
   mask=(cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)>0).astype(np.uint8)
  else:
   mask=np.zeros((1080,1920),np.uint8);binding=dict(source_sha256=scope['source']['sha256'],source_pts=pts,time_base='1/12800',native_hw=[1080,1920],shot_id=sid,decoded_native_rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest());prompts=[]
   for i,box in enumerate(ledger[pts].get('current_person_boxes',[])):
    old=box['source_binding']
    if old.get('source_pts')!=pts or old.get('decoded_native_rgb_sha256')!=binding['decoded_native_rgb_sha256']:raise ValueError('Wrong current person source pixels')
    q=np.asarray(box['box_xyxy']);a=np.maximum([0,0],np.floor(q[:2]-12).astype(int));b=np.minimum([1920,1080],np.ceil(q[2:]+12).astype(int));mask[a[1]:b[1],a[0]:b[0]]=1
    if q[3]>500 and q[3]-q[1]>100:prompts.append(dict(object_id=i+1,kind='occluder',origin='original_rgb_candidate',source_binding=binding,box_xyxy=q.tolist(),exclusion_authority=None))
   if prompts:
    if person_provider is None:
     sys.path[:0]=[str(project),str(project/'src')]
     from calibration.program_tracking.runtime.bootstrap import add_geometry_dependencies
     add_geometry_dependencies()
     import torch
     from rinkspace.vision_instances import Sam2SourceInstances
     torch.set_num_threads(2);person_provider=Sam2SourceInstances(resolve_asset(project, project/'models/cache/sam2.1-hiera-tiny'),device='cuda',reread_rgb=False,contour_stride=16)
    current=person_provider.read(rgb,binding,prompts=prompts,reset=True);sam=current['occluder_region_candidate_mask'].detach().cpu().numpy().astype(np.uint8);mask|=cv2.dilate(sam,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(25,25)))
  mask[:170,:620]=1;mask[900:,1600:]=1;cv2.imwrite(str(out/'mask_cache'/f'{pts}.png'),mask.astype(np.uint8)*255);return mask>0
 def anchor_for(sid,pts):
  key=(sid,pts)
  if key in anchors:return anchors[key]
  rgb,gray,sha,path,pngsha=image_for(pts);mask=mask_for(sid,pts,rgb);roi=np.zeros(gray.shape,np.uint8)
  for x in seeds[key]:
   u,v=np.rint(x['uv']).astype(int)
   if 0<=u<1920 and 0<=v<1080:cv2.circle(roi,(u,v),48,255,-1)
  roi[mask]=0;k,d=sift.detectAndCompute(gray,roi);obj=dict(gray=gray,pts=pts,keypoints=k,descriptors=d,rows=seeds[key],source_BGR_sha256=sha,native_png=path,native_png_sha256=pngsha)
  anchors[key]=obj;return obj
 cv2.setNumThreads(2)
 for target in sorted(targets,key=lambda r:r['native_pts']):
  sid=target['shot_id'];pts=int(target['native_pts']);guard.assert_training_pixels([pts],shot_id=sid);candidates=[];denied=0
  for s,q in seeds:
   if s!=sid:continue
   try:guard.assert_training_track(min(q,pts),max(q,pts),shot_id=sid)
   except ValueError:denied+=1;continue
   candidates.append(q)
  candidates.sort(key=lambda q:abs(q-pts));rgb,gray,bgrsha,path,pngsha=image_for(pts);mask=mask_for(sid,pts,rgb);keypoints,desc=sift.detectAndCompute(gray,(~mask).astype(np.uint8)*255);source_context=prepare_current_source(rgb,mask,project);channels=source_context['channels'];frame_rows=[];attempts=[]
  for anchor_pts in candidates:
   anchor=anchor_for(sid,anchor_pts);ad=anchor['descriptors'];ak=anchor['keypoints']
   if ad is None or desc is None or len(ad)<8 or len(desc)<8:attempts.append({'anchor_pts':anchor_pts,'status':'insufficient_actual_material_keypoints'});continue
   fwd=matcher.knnMatch(ad,desc,k=2);back=matcher.knnMatch(desc,ad,k=2);reverse_matches={m.queryIdx:m.trainIdx for m,n in back if m.distance<.72*n.distance};matches=[m for m,n in fwd if m.distance<.72*n.distance and reverse_matches.get(m.trainIdx)==m.queryIdx]
   if len(matches)<6:attempts.append({'anchor_pts':anchor_pts,'status':'no_unique_actual_material_keypoint_correspondence','mutual_matches':len(matches)});continue
   a=np.asarray([ak[m.queryIdx].pt for m in matches],np.float32);b=np.asarray([keypoints[m.trainIdx].pt for m in matches],np.float32);H,inliers=cv2.findHomography(a,b,cv2.RANSAC,1.5);selectedrows=anchor['rows'];physical_panel='all_verified_materials'
   if H is None or inliers.sum()<10:
    groups=defaultdict(list);seedxy=np.asarray([x['uv'] for x in selectedrows]);labels=[_panel(x['target_id']) for x in selectedrows]
    for mi,point in enumerate(a):
     nearest=int(np.argmin(np.sum((seedxy-point)**2,axis=1)));groups[labels[nearest]].append(mi)
    local=[]
    for label,ii in groups.items():
     if len(ii)<6:continue
     hh,ss=cv2.findHomography(a[ii],b[ii],cv2.RANSAC,1.5)
     if hh is not None and ss.sum()>=6 and cv2.contourArea(cv2.convexHull(a[ii][ss[:,0]>0]))>=64:local.append((int(ss.sum()),label,hh,ss,ii))
    if not local:attempts.append({'anchor_pts':anchor_pts,'status':'no_consistent_current_native_navigation','mutual_matches':len(matches)});continue
    _,physical_panel,H,inliers,ii=max(local,key=lambda x:x[0]);a=a[ii];b=b[ii];selectedrows=[x for x in selectedrows if _panel(x['target_id'])==physical_panel]
   old=np.asarray([x['uv'] for x in selectedrows]);mapped=_transform(H,old);offsets=np.asarray([(x,y) for y in range(-6,7) for x in range(-6,7)],float);patchcoords=_transform(H,(old[:,None,:]+offsets).reshape(-1,2)).reshape(len(old),169,2);currentpatch=cv2.remap(gray,patchcoords[:,:,0].astype(np.float32),patchcoords[:,:,1].astype(np.float32),cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT).astype(np.float32);ncc=_ncc(_sample_patches(anchor['gray'],old),currentpatch);reject=Counter()
   for j,x in enumerate(selectedrows):
    q=mapped[j]
    if not np.isfinite(q).all() or not(9<q[0]<1910 and 9<q[1]<1070) or mask[int(round(q[1])),int(round(q[0]))]:reject['current_source_domain_or_actor_exclusion']+=1;continue
    if ncc[j]<.8:reject['homography_warped_actual_material_patch_not_supported']+=1;continue
    n=x.get('normal')
    if n is not None:
     n=np.asarray(n,float);ex=_transform(H,[old[j]-.25*np.array([1.,0.]),old[j]+.25*np.array([1.,0.])]);ey=_transform(H,[old[j]-.25*np.array([0.,1.]),old[j]+.25*np.array([0.,1.])]);J=np.column_stack(((ex[1]-ex[0])/.5,(ey[1]-ey[0])/.5))
     if abs(np.linalg.det(J))<1e-10:reject['singular_navigation_normal_transport']+=1;continue
     n=np.linalg.solve(J.T,n);n=n/max(np.linalg.norm(n),1e-20)
    sides=None if x.get('flank_uv') is None else _transform(H,x['flank_uv']).tolist()
    if x['entity_point']:
     cq=cv2.cornerSubPix(gray,q.astype(np.float32).reshape(1,1,2),(4,4),(-1,-1),(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.001))[0,0].astype(float)
     if np.linalg.norm(cq-q)>3.:reject['current_entity_source_ambiguous']+=1;continue
     q=cq
    rr=remeasure_physical_source(source_context,dict(target_id=x['target_id'],category='ad_corner' if x['entity_point'] else {'poster_boundary':'ad_edge','ink_boundary':'ad_print'}.get(x['operator'])),q,n,sides)
    if rr.get('current_uv') is None or (x['entity_point'] and rr.get('physical_entity_source_verified') is not True):reject['no_current_two_flank_or_corner_support']+=1;continue
    uv=np.asarray(rr['current_uv']);u,v=np.rint(uv).astype(int)
    if mask[v,u]:reject['new_actual_material_measurement_occluded']+=1;continue
    rid=f"dense_image_match/{sid}/{pts}/{x['source_row_id']}";row=dict(shot_id=sid,native_pts=pts,time_base='1/12800',source_row_id=rid,anchor_source_row_id=x['source_row_id'],anchor_pts=anchor_pts,source_family_id=f"{sid}/{pts}/{x['source_family_id']}",joint_material_id=x['joint_material_id'],material_index=x['material_index'],target_id=x['target_id'],measurement_operator=x['operator'],point_correspondence=x['entity_point'],source_uv=uv.tolist(),current_uv=uv.tolist(),navigation_seed_uv=q.tolist(),measurement_contract_sha256=rr.get('measurement_contract_sha256'),auxiliary_profiles_diagnostic_only=True,source_measurement_audit=rr,source_navigation='native_material_SIFT_mutual_correspondence_RANSAC_only',homography_warped_native_patch_ncc=float(ncc[j]),decoded_native_BGR_sha256=bgrsha,current_pixels_measured=True,pixels_from_model=False,numeric_statistics_ready=False,solver_eligible=False,role='development',pending_reason='source-family covariance and physical replay required');frame_rows.append(row)
    nn=rr.get('normal')
    if nn is not None:
     nn=np.asarray(nn);t=np.array([-nn[1],nn[0]])
     for side,point in enumerate(rr.get('flank_uv') or [uv]):
      coords=np.asarray(point)+toff[:,None,None]*t+off[None,:,None]*nn;allc=_sample_current_cubic(source_context['spline_channels'],coords.reshape(-1,2)).reshape(5,25,3);ch=int((rr.get('source_material_receipt') or {}).get('channel',(rr.get('profile') or {}).get('channel',0)));profile_index.append(dict(profile_index=len(profiles),source_row_id=rid,source_family_id=row['source_family_id'],shot_id=sid,native_pts=pts,side=side,decoded_native_BGR_sha256=bgrsha));profiles.append(allc[:,:,ch]);puv.append(point);pn.append(nn);pch.append(ch)
   attempts.append(dict(anchor_pts=anchor_pts,status='current_native_source_measured' if frame_rows else 'navigation_only_no_current_material_support',H_navigation_only=H.tolist(),actual_source_keypoint_uv=a[inliers[:,0]>0].tolist(),actual_current_keypoint_uv=b[inliers[:,0]>0].tolist(),inlier_count=int(inliers.sum()),source_material_panel_navigation=physical_panel,source_anchor_BGR_sha256=anchor['source_BGR_sha256'],current_BGR_sha256=bgrsha,rejections=dict(reject)))
   if frame_rows:break
  result_rows.extend(frame_rows);counts['actual_source_recovered_frames' if frame_rows else 'still_pending_frames']+=1;counts['current_actual_source_rows']+=len(frame_rows);receipts.append(dict(shot_id=sid,native_pts=pts,status='actual_current_material_source_recovered' if frame_rows else 'no_direct_native_material_identity_supported',source_rows=len(frame_rows),source_BGR_sha256=bgrsha,native_png=path,native_png_sha256=pngsha,current_mask_array_sha256=hashlib.sha256(mask.tobytes()).hexdigest(),role_guard_verified=True,denied_role_crossing_anchors=denied,attempts=attempts,numeric_statistics_ready=False));print(json.dumps(dict(native_pts=pts,completed=len(receipts),counts=dict(counts))),flush=True)
 write_jsonl(out/'rows.jsonl.gz',result_rows);write_jsonl(out/'frame_receipts.jsonl.gz',receipts);write_json(out/'profile_index.json',profile_index);np.savez_compressed(out/'profiles.npz',intensity=np.asarray(profiles,np.float32).reshape(-1,5,25),uv=np.asarray(puv,float).reshape(-1,2),normal=np.asarray(pn,float).reshape(-1,2),channel=np.asarray(pch,np.uint8),normal_offsets=off,tangent_offsets=toff);result=dict(schema='rinkfit.joint9.direct_native_material_reacquisition.v1',input_gap_frames=len(targets),counts=dict(counts),all_gap_frames_accounted=len(receipts)==len(targets),protected_pixels_consumed=False,source_video_sha256=scope['source']['sha256'],source_frame_ledger_sha256=file_sha256(first/'frame_ledger.jsonl.gz'),parent_reverse_sha256=file_sha256(reverse/'frame_receipts.jsonl.gz'),source_UV_generated_from_model=False,formal_statistics_ready=False,elapsed_seconds=time.perf_counter()-start);write_json(out/'manifest.json',result);return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--project',required=True);p.add_argument('--first-pass',required=True);p.add_argument('--reverse',required=True);p.add_argument('--source',required=True);p.add_argument('--decisions',required=True);p.add_argument('--material-map',required=True);p.add_argument('--out',required=True);p.add_argument('--previous');a=p.parse_args();print(json.dumps(recover(a.run,a.project,a.first_pass,a.reverse,a.source,a.decisions,a.material_map,a.out,a.previous)))
