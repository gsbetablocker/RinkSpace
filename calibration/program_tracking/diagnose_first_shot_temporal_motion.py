"""Read-only motion-compensated diagnostics for first-shot observations.

Nothing here emits fitting observations, changes a candidate or fits a camera.
Inherited camera transport ranks suspects. Current source-image registration
independently checks their actual motion before a continuity claim is made.
"""
from __future__ import annotations
import argparse, hashlib, json, sys
from functools import lru_cache
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from calibration.program_tracking.runtime.bootstrap import add_geometry_dependencies
add_geometry_dependencies()
import cv2
import numpy as np
from calibration.program_tracking.projection import ground_xy,project_fast
from calibration.program_tracking.extract_single_frame_ice_source import ellipse_distance
DATA=Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/programmes/okada_mei_sp/results/frame_by_frame_review_20261013/first_shot_remaining_50fps')
OUT=DATA/'repair_20261014/temporal_diagnostics'
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False,default=lambda x:x.item() if isinstance(x,np.generic) else str(x))+'\n',encoding='utf8')
def points(f):return np.vstack([s['uv'] for s in f['segments']])
def feature_list(packet):
 return packet['features'] if 'features' in packet else list(packet['features_by_curve'].values())
def line(f):
 z=f['source_line_fit'];o=np.array(z['point_origin']);v=np.array(z['unit_tangent']);n=np.array(z['unit_normal']);q=z['normal_offset_coefficients']
 v=v+n*q[0];v/=np.linalg.norm(v);return o+n*q[1],v,np.array([-v[1],v[0]])
def transport(uv,a,b):
 xy,good=ground_xy(a,uv);uv,depth=project_fast(b,np.c_[xy,np.zeros(len(xy))]);return uv,good&(depth>.05)&np.isfinite(uv).all(1)
def distance_to_feature(uv,f):
 if 'source_line_fit' in f:
  o,v,n=line(f);d=(uv-o)@n;t=(uv-o)@v;good=np.zeros(len(uv),bool)
  for s in f['segments']:
   st=(np.asarray(s['uv'])-o)@v;good|=(t>=st.min()-15)&(t<=st.max()+15)
  return np.abs(d),good
 z=f.get('geometric_interpretation',{})
 if 'center_uv' in z:
  e=(z['center_uv'],z['axes_px'],z['angle_degrees']);d=ellipse_distance(uv,e)
  # Compare the same currently measured angular portions, not nearby upper
  # and lower branches of an extremely flat ellipse.
  angle=np.deg2rad(e[2]);rotation=np.array([[np.cos(angle),np.sin(angle)],[-np.sin(angle),np.cos(angle)]])
  radii=np.maximum(np.array(e[1])/2,.5)
  q=(uv-np.array(e[0]))@rotation.T/radii;phi=np.arctan2(q[:,1],q[:,0]);good=np.zeros(len(uv),bool)
  for s in f['segments']:
   local=(np.asarray(s['uv'])-np.array(e[0]))@rotation.T/radii;theta=np.unwrap(np.arctan2(local[:,1],local[:,0]))
   for turn in (-2*np.pi,0.,2*np.pi):good|=(phi+turn>=theta.min()-.08)&(phi+turn<=theta.max()+.08)
  return d,good
 return np.zeros(len(uv)),np.zeros(len(uv),bool)
@lru_cache(maxsize=3)
def native_person_mask(index):return cv2.imread(str(DATA/f'masks_current/{index:06}.png'),0)>0
@lru_cache(maxsize=3)
def path_person_mask(path):return cv2.imread(path,0)>0
def current_visible_points(index,uv,mask_path=None):
 good=np.isfinite(uv).all(1)&(uv>=[0,0]).all(1)&(uv<=[1919,1079]).all(1)
 mask=path_person_mask(str(mask_path)) if mask_path else native_person_mask(index)
 ip=np.rint(uv[good]).astype(int);active=np.flatnonzero(good);good[active]&=~mask[ip[:,1],ip[:,0]]
 good&=~((uv[:,0]>=1688)&(uv[:,0]<=1868)&(uv[:,1]>=940)&(uv[:,1]<=1006))
 return good
def rank(folder):
 cameras=np.load(DATA/'camera_seeds.npz');cs={int(i):c for i,c in zip(cameras['programme_indices'],cameras['camera25'])}
 records=[];packets={};checks=[]
 for i in range(201,1249):
  path=Path(folder)/f'{i:06}.json';packet=read(path);packets[i]={f['target_id']:f for f in packet['features'] if f.get('point_count')}
  checks.append(dict(index=i,path=str(path),sha256=sha(path)))
  if i==201:continue
  for tid,f in packets[i].items():
   old=packets[i-1].get(tid)
   if old is None:continue
   previous=points(old);predicted,valid=transport(previous,cs[i-1],cs[i]);dist,common=distance_to_feature(predicted,f)
   good=valid&common&(predicted>=[0,0]).all(1)&(predicted<=[1919,1079]).all(1)
   if good.sum()<2:continue
   records.append(dict(previous_index=i-1,index=i,target_id=tid,kind='line' if 'source_line_fit' in f else 'circle',
    common_compared_points=int(good.sum()),camera_transport_median_normal_error_px=float(np.median(dist[good])),
    camera_transport_max_normal_error_px=float(dist[good].max()),raw_point_motion_median_px=float(np.median(np.linalg.norm(predicted[good]-previous[good],axis=1))),
    previous_point_count=old['point_count'],current_point_count=f['point_count'],previous_navigation_origin=old.get('navigation_origin'),current_navigation_origin=f.get('navigation_origin')))
  if i%100==0:print('ranked',i,flush=True)
 records.sort(key=lambda r:r['camera_transport_median_normal_error_px'],reverse=True)
 report=dict(schema='rinkspace.readonly-first-shot-camera-motion-compensated-rank.v1',original_camera_only_ranking_not_source_verification=True,
  candidate_folder=str(folder),frame_count=1048,source_clock='official50 nativePTS',pair_comparisons=len(records),records=records,packet_integrity=checks,
  source_registration_required_before_final_identity_claim=True,fit_observations_emitted=False)
 write(OUT/'camera_transport_rank.json',report)
 for tid in ['ice.B1:segment0','ice.B2:segment0','ice.C:segment0','ice.R3:outer','ice.R4:outer','ice.F3:outer','ice.F4:outer','ice.C0:outer']:
  print(tid,[(r['previous_index'],r['index'],round(r['camera_transport_median_normal_error_px'],2),r['common_compared_points']) for r in records if r['target_id']==tid][:8],flush=True)
 return report
def registration_inputs(a,b,region='ice'):
 accepted_200=ROOT/'events/ankara_2026/programmes/okada_mei_sp/source_datasets/frame_by_frame_review_20261013/continuation_101_200_50fps/frames/000200.json'
 packets=[read(accepted_200) if i==200 else read(DATA/f'board_candidates/{i:06}.json') for i in (a,b)]
 image_paths=[p['source']['image_path'] if i==200 else DATA/f'frames/{i:06}.png' for i,p in zip((a,b),packets)]
 images=[cv2.imread(str(path)) for path in image_paths]
 masks=[]
 for i,im,packet in zip((a,b),images,packets):
  person=cv2.imread(str(packet['source']['mask_path'] if i==200 else DATA/f'masks_current/{i:06}.png'),0)>0
  person=cv2.dilate(person.astype(np.uint8),np.ones((31,31),np.uint8))>0
  valid=np.zeros(im.shape[:2],np.uint8)
  # Candidate contact is a navigation boundary only, padded into the clear
  # ice. No board/stands/person/text pixels enter this diagnostic registration.
  contact=np.array([q for f in packet['features'] if f['target_id'].startswith('board.contact') for s in f['segments'] for q in s.get('source_native_readout_uv',s['uv'])])
  by=np.interp(np.arange(1920),*([contact[np.argsort(contact[:,0]),0],contact[np.argsort(contact[:,0]),1]]))
  if region=='ice':valid[np.arange(1080)[:,None]>by[None]+30]=255
  else:
   top=np.array([q for f in packet['features'] if f['target_id'].startswith('board.top') for s in f['segments'] for q in s.get('source_native_readout_uv',s['uv'])])
   ty=np.interp(np.arange(1920),top[np.argsort(top[:,0]),0],top[np.argsort(top[:,0]),1]);valid[(np.arange(1080)[:,None]>ty[None]+8)&(np.arange(1080)[:,None]<by[None]-8)]=255
  valid[person]=0;valid[:140,:600]=0;valid[930:1020,1660:1900]=0;masks.append(valid)
 gray=[cv2.cvtColor(im,cv2.COLOR_BGR2GRAY) for im in images]
 return images,masks,gray
def source_motion(a,b,region='ice',save_tracks=True):
 """Independent LK forward/backward registration within source rink material."""
 images,masks,gray=registration_inputs(a,b,region)
 p=cv2.goodFeaturesToTrack(gray[0],maxCorners=2500,qualityLevel=.003,minDistance=10,mask=masks[0],blockSize=7)
 if p is None:return dict(usable=False,reason='no_source_rink_corners'),None
 q,ok,e=cv2.calcOpticalFlowPyrLK(gray[0],gray[1],p,None,winSize=(35,35),maxLevel=4,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,40,.01))
 back,back_ok,_=cv2.calcOpticalFlowPyrLK(gray[1],gray[0],q,None,winSize=(35,35),maxLevel=4,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,40,.01))
 p=p[:,0];q=q[:,0];back=back[:,0];ip=np.rint(q).astype(int);inside=(ip>=[0,0]).all(1)&(ip<=[1919,1079]).all(1)
 good=(ok[:,0]>0)&(back_ok[:,0]>0)&(np.linalg.norm(back-p,axis=1)<1.)&inside
 good[inside]&=masks[1][ip[inside,1],ip[inside,0]]>0;p=p[good];q=q[good]
 if len(p)<8:return dict(usable=False,reason='too_few_current_static_rink_tracks',tracks=len(p)),None
 h,inliers=cv2.findHomography(p,q,cv2.RANSAC,2.0)
 if h is None:return dict(usable=False,reason='source_registration_failed',tracks=len(p)),None
 keep=inliers[:,0]>0;fit=cv2.perspectiveTransform(p[None],h)[0];error=np.linalg.norm(fit-q,axis=1);extent=np.ptp(p[keep],axis=0)
 usable=bool(int(keep.sum())>=15 and extent[0]>350 and extent[1]>55 and float(np.median(error[keep]))<1.)
 record=dict(usable=usable,region=region,actual_current_original_RGB_only=True,model_or_observation_coordinates_used_as_tracks=False,
  static_source_tracks=len(p),inliers=int(keep.sum()),spatial_extent_px=extent.tolist(),median_reprojection_px=float(np.median(error[keep])),
  p95_reprojection_px=float(np.quantile(error[keep],.95)),homography=h.tolist(),previous_index=a,index=b,
  fitting_points_or_camera_parameters_created=False,tracking_source_excludes_person_timer_stands_and_board_when_ice=True,
  image_motion_tracks_are_diagnostic_only_not_registered_model_landmarks=True,individual_texture_physical_identity_not_claimed=True,
  inlier_bbox_xyxy=np.r_[p[keep].min(0),p[keep].max(0)].tolist(),inlier_convex_hull_uv=cv2.convexHull(p[keep].astype(np.float32))[:,0].tolist(),
  source_motion_is_local_to_inlier_support_not_global_camera_proof=True)
 if save_tracks:np.savez_compressed(OUT/f'source_tracks_{region}_{a:06}_{b:06}.npz',previous_uv=p,current_uv=q,inliers=keep)
 return record,h if usable else None
def source_anchor_motion(a,b,region='ice'):
 """Direct SIFT registration for source-anchor closure, never a field fit.

 Identity, not just numerical RANSAC residual, is protected by reciprocal
 ratio-filtered descriptor matches in the currently visible rink domain.
 Return no transform if the actual common material cannot establish it.
 """
 images,masks,gray=registration_inputs(a,b,region);sift=cv2.SIFT_create(nfeatures=6000,contrastThreshold=.008)
 kp=[];desc=[]
 for image,mask in zip(gray,masks):
  key,d=sift.detectAndCompute(image,mask);kp.append(key);desc.append(d)
 if any(d is None for d in desc):return dict(usable=False,previous_index=a,index=b,reason='missing_current_source_descriptors'),None
 matcher=cv2.BFMatcher();forward=matcher.knnMatch(desc[0],desc[1],k=2);backward=matcher.knnMatch(desc[1],desc[0],k=2)
 reverse={x.queryIdx:x.trainIdx for x,y in backward if x.distance<.65*y.distance}
 pairs=[x for x,y in forward if x.distance<.65*y.distance and reverse.get(x.trainIdx)==x.queryIdx]
 if len(pairs)<15:return dict(usable=False,previous_index=a,index=b,reason='too_few_reciprocal_distinct_material_matches',matches=len(pairs)),None
 p=np.array([kp[0][m.queryIdx].pt for m in pairs]);q=np.array([kp[1][m.trainIdx].pt for m in pairs]);h,valid=cv2.findHomography(p,q,cv2.USAC_MAGSAC,2.)
 if h is None:return dict(usable=False,previous_index=a,index=b,reason='source_anchor_homography_not_established'),None
 keep=valid[:,0]>0;predicted=cv2.perspectiveTransform(p[None],h)[0];e=np.linalg.norm(predicted-q,axis=1);extent=np.ptp(p[keep],axis=0)
 usable=bool(keep.sum()>=25 and extent[0]>350 and extent[1]>65 and np.median(e[keep])<1.0)
 record=dict(previous_index=a,index=b,region=region,usable=usable,method='reciprocal_ratio_SIFT_current_source_rink_domain_RANSAC',
  reciprocal_matches=len(pairs),inliers=int(keep.sum()),median_reprojection_px=float(np.median(e[keep])),p95_reprojection_px=float(np.quantile(e[keep],.95)),
  spatial_extent_px=extent.tolist(),homography=h.tolist(),fitting_observations_or_camera_parameters_emitted=False,
  individual_descriptor_texture_identity_not_declared_model_landmark=True)
 np.savez_compressed(OUT/f'anchor_tracks_{region}_{a:06}_{b:06}.npz',previous_uv=p,current_uv=q,inliers=keep)
 return record,h if usable else None
def compare_motion(a,b,h,folder,support_hull=None,lines_only=False):
 packets=[read(Path(folder)/f'{i:06}.json') for i in (a,b)]
 features=[{f['target_id']:f for f in feature_list(packet) if f.get('point_count')} for packet in packets]
 comparisons=[]
 for tid,f in features[1].items():
  if tid not in features[0]:continue
  if lines_only and ('source_line_fit' not in f or 'source_line_fit' not in features[0][tid]):continue
  previous=points(features[0][tid]);p=cv2.perspectiveTransform(previous[None],h)[0];d,common=distance_to_feature(p,f);common&=current_visible_points(b,p,packets[1].get('source',{}).get('mask_path'))
  if support_hull is not None:
   hull=np.asarray(support_hull,np.float32).reshape(-1,1,2)
   common&=np.array([cv2.pointPolygonTest(hull,tuple(q),False)>=0 for q in previous],bool)
  if common.sum()<2:continue
  comparisons.append(dict(previous_index=a,index=b,target_id=tid,compared_points=int(common.sum()),source_motion_compensated_median_normal_error_px=float(np.median(d[common])),source_motion_compensated_max_normal_error_px=float(d[common].max())))
 comparisons.sort(key=lambda r:r['source_motion_compensated_median_normal_error_px'],reverse=True)
 return comparisons

def supported_motion(folder,label,lines_only=False):
 """Compare the same source motions only within their observed support.

 This is still an identity diagnostic, not a source-material correctness pass.
 Unknown motion and absent observations remain explicitly unscored.
 """
 archive=np.load(OUT/'source_ice_motion_all.npz');support=read(OUT/'all_source_motion_support.json')
 if support['motion_archive_sha256']!=sha(OUT/'source_ice_motion_all.npz'):raise ValueError('Frozen motion/support mismatch')
 quality={r['index']:r for r in support['records']};rows=[];unscored=[]
 for index,h in zip(archive['indices'],archive['homographies']):
  index=int(index);q=quality[index]
  if not np.isfinite(h).all() or not q.get('usable') or not q.get('inlier_convex_hull_uv'):
   unscored.append(index);continue
  rows.extend(compare_motion(index-1,index,h,folder,q['inlier_convex_hull_uv'],lines_only=lines_only))
 rows.sort(key=lambda r:r['source_motion_compensated_median_normal_error_px'],reverse=True)
 summary={}
 for tid in sorted({r['target_id'] for r in rows}):
  values=np.array([r['source_motion_compensated_median_normal_error_px'] for r in rows if r['target_id']==tid])
  summary[tid]=dict(pair_count=len(values),median_px=float(np.median(values)),p95_px=float(np.quantile(values,.95)),max_px=float(values.max()),above_3px=int((values>3).sum()),above_6px=int((values>6).sum()))
 write(OUT/f'{label}_supported_motion.json',dict(candidate_folder=str(folder),motion_archive_sha256=sha(OUT/'source_ice_motion_all.npz'),support_sha256=sha(OUT/'all_source_motion_support.json'),
  verified_inlier_hull_only=True,only_source_line_features=lines_only,circle_distance_not_evaluated=lines_only,unscored_pair_indices=unscored,comparison_count=len(rows),summary=summary,records=rows,
  missing_features_or_unsupported_motion_are_not_passes=True,small_motion_residual_does_not_establish_physical_material_identity=True))

def ring_nesting(folder,label):
 """Inner material boundary points must be inside enclosing projected rings."""
 rows=[];checks=0;available={}
 for index in range(201,1249):
  features={f['target_id']:f for f in feature_list(read(Path(folder)/f'{index:06}.json')) if f.get('point_count')}
  for house in [f'{side}{i}' for side in 'RF' for i in range(1,5)]:
   for outer,inner in [('outer','middle'),('outer','inner'),('middle','inner')]:
    a=features.get(f'ice.{house}:{outer}');b=features.get(f'ice.{house}:{inner}')
    if a is None or b is None:continue
    z=a.get('geometric_interpretation',{})
    if not all(k in z for k in ('center_uv','axes_px','angle_degrees')):continue
    checks+=1;available.setdefault(house,0);available[house]+=1
    center=np.array(z['center_uv']);radii=np.array(z['axes_px'])/2;theta=np.deg2rad(z['angle_degrees'])
    raw=b.get('source_circle_fit',{}).get('direct_readouts')
    rot=np.array([[np.cos(theta),np.sin(theta)],[-np.sin(theta),np.cos(theta)]]);p=np.asarray([r['uv'] for r in raw]) if raw else points(b);local=(p-center)@rot.T
    implicit=np.sum((local/radii)**2,axis=1)-1;gradient=2*np.linalg.norm(local/(radii*radii),axis=1)
    approximate_signed=implicit/np.maximum(gradient,1e-8);outside=approximate_signed>5.
    if outside.any():rows.append(dict(index=index,house=house,enclosing_boundary=outer,inside_boundary=inner,
     points_outside_declared_enclosing_ring=int(outside.sum()),max_approximate_first_order_signed_distance_px=float(approximate_signed.max()),
     median_approximate_first_order_signed_distance_px=float(np.median(approximate_signed)),maximum_ellipse_implicit_value=float(implicit.max())))
 rows.sort(key=lambda r:r['max_approximate_first_order_signed_distance_px'],reverse=True)
 write(OUT/f'{label}_ring_nesting.json',dict(candidate_folder=str(folder),physical_ring_pair_checks=checks,per_house_checks=available,issue_count=len(rows),bad_frame_count=len({r['index'] for r in rows}),records=rows,
  signed_distance_is_first_order_approximation_not_exact_nearest_pixel_distance=True,outside_reporting_threshold_approximate_px=5.,missing_ring_pairs_are_unscored_not_passed=True,
  physical_nested_ring_order_preserved_under_camera_projection=True))
def check_pairs(pairs,folder):
 results=[]
 for a,b in pairs:
  motion,h=source_motion(a,b);comparisons=compare_motion(a,b,h,folder) if h is not None else []
  results.append(dict(source_registration=motion,comparisons=comparisons));print(a,b,motion,comparisons[:6],flush=True)
 write(OUT/'current_source_motion_checks.json',dict(pair_count=len(results),results=results,read_only_no_original_artifacts_modified=True))
 return results
def full_source_scan(folder):
 records=[];registration=[];matrices=[]
 for index in range(202,1249):
  motion,h=source_motion(index-1,index,save_tracks=False);motion.update(previous_index=index-1,index=index);registration.append(motion);matrices.append(h if h is not None else np.full((3,3),np.nan))
  if h is not None:records.extend(compare_motion(index-1,index,h,folder))
  if index%50==0:print('source_motion_scanned',index,'usable',sum(r['usable'] for r in registration),flush=True)
 records.sort(key=lambda r:r['source_motion_compensated_median_normal_error_px'],reverse=True)
 np.savez_compressed(OUT/'source_ice_motion_all.npz',indices=np.arange(202,1249),homographies=np.asarray(matrices))
 write(OUT/'all_source_motion_baseline.json',dict(schema='rinkspace.full-first-shot-source-image-motion-diagnostics.v1',pair_count=len(registration),usable_registration_count=sum(r['usable'] for r in registration),
  current_actual_source_image_registration_for_transport_only=True,no_fitting_points_or_camera_fit=True,person_timer_and_stands_excluded=True,
  candidate_folder=str(folder),registration=registration,comparisons=records,unscored_registration_indices=[r.get('index') for r in registration if not r['usable']],
  usable_motion_archive=dict(path=str(OUT/'source_ice_motion_all.npz'),sha256=sha(OUT/'source_ice_motion_all.npz'))))
 print('full_source_motion_ready',len(records),flush=True)
def failure_evidence(folder):
 cases=[(672,673,'ice.lane4:edge2',[180,820,650,1010]),(706,707,'ice.house_connection_F3_R3:segment0',[900,890,1500,1080]),
  (778,779,'ice.lane3:edge0',[1350,550,1920,740]),(1218,1219,'ice.C:segment0',[1400,480,1850,1050]),
  (976,977,'ice.F4:outer',[1550,760,1920,950]),(1004,1005,'ice.A3:outer',[900,760,1550,1040])]
 records=[]
 for a,b,tid,roi in cases:
  motion,h=source_motion(a,b)
  if h is None:continue
  old=next(f for f in read(Path(folder)/f'{a:06}.json')['features'] if f['target_id']==tid)
  now=next(f for f in read(Path(folder)/f'{b:06}.json')['features'] if f['target_id']==tid)
  raw=cv2.imread(str(DATA/f'frames/{b:06}.png'));canvas=raw.copy()
  for feature,matrix,color in ((old,h,(255,230,0)),(now,np.eye(3),(0,0,255))):
   for s in feature['segments']:
    p=cv2.perspectiveTransform(np.array(s['uv'],float)[None],matrix)[0];p=np.rint(p).astype(int)
    if len(p)>1:cv2.polylines(canvas,[p],False,color,2,cv2.LINE_AA)
    for q in p:cv2.circle(canvas,tuple(q),4,color,-1,cv2.LINE_AA)
  x0,y0,x1,y1=roi;views=[]
  for title,im in [('CURRENT RAW '+str(b),raw),('CYAN=PREV SOURCE MOTION; RED=CURRENT',canvas)]:
   crop=im[y0:y1,x0:x1];tile=np.full((crop.shape[0]+54,crop.shape[1],3),25,np.uint8);tile[54:]=crop
   cv2.putText(tile,title,(8,18),0,.43,(255,255,255),1,cv2.LINE_AA);cv2.putText(tile,f'{a}->{b} {tid}',(8,39),0,.43,(255,255,255),1,cv2.LINE_AA);views.append(tile)
  path=OUT/f'failure_{a:06}_{b:06}_{tid.replace(":","_").replace(".","_")}.png';cv2.imwrite(str(path),np.hstack(views))
  compare=next(r for r in compare_motion(a,b,h,folder) if r['target_id']==tid)
  records.append(dict(**compare,source_registration=motion,native_crop_xyxy=roi,evidence_png=dict(path=str(path),sha256=sha(path)),
   previous_source=dict(path=str(DATA/f'frames/{a:06}.png'),sha256=sha(DATA/f'frames/{a:06}.png')),
   current_source=dict(path=str(DATA/f'frames/{b:06}.png'),sha256=sha(DATA/f'frames/{b:06}.png'))))
 write(OUT/'failure_examples.json',dict(native_pair_count=len(records),records=records,read_only_no_fitting_observations=True))
def coverage(folder):
 targets={};frames=[]
 for index in range(201,1249):
  p=read(Path(folder)/f'{index:06}.json');counts={f['target_id']:int(f['point_count']) for f in p['features'] if f.get('point_count')}
  frames.append(dict(index=index,point_count=sum(counts.values()),counts=counts,packet_sha256=sha(Path(folder)/f'{index:06}.json')))
  for tid,n in counts.items():targets.setdefault(tid,[]).append(dict(index=index,count=n))
 write(OUT/'baseline_feature_coverage.json',dict(candidate_folder=str(folder),frame_count=len(frames),per_target={tid:dict(covered_frame_count=len(rows),point_count=sum(r['count'] for r in rows),frames=rows) for tid,rows in targets.items()},frames=frames))
def compare_repaired(folder):
 archived=np.load(OUT/'source_ice_motion_all.npz');records=[]
 for index,h in zip(archived['indices'],archived['homographies']):
  index=int(index)
  if not np.isfinite(h).all():continue
  records.extend(compare_motion(index-1,index,h,folder))
  if index%100==0:print('compared_repaired_motion',index,flush=True)
 records.sort(key=lambda r:r['source_motion_compensated_median_normal_error_px'],reverse=True)
 baseline=read(OUT/'all_source_motion_baseline.json');old={(r['index'],r['target_id']):r for r in baseline['comparisons']};common=[]
 for r in records:
  previous=old.get((r['index'],r['target_id']))
  if previous:common.append(dict(index=r['index'],target_id=r['target_id'],old_median_px=previous['source_motion_compensated_median_normal_error_px'],new_median_px=r['source_motion_compensated_median_normal_error_px']))
 write(OUT/'repaired_motion_comparison.json',dict(candidate_folder=str(folder),same_independent_source_motion_archive_reused=True,motion_archive_sha256=sha(OUT/'source_ice_motion_all.npz'),records=records,old_new_common=common,
  caution='A disappearing feature is not a passed comparison; inspect repaired_coverage_comparison independently.'))
 oldcoverage=read(OUT/'baseline_feature_coverage.json');oldframes={r['index']:r['counts'] for r in oldcoverage['frames']};targets={};missing=[]
 for index in range(201,1249):
  p=read(Path(folder)/f'{index:06}.json');new={f['target_id']:f['point_count'] for f in p['features'] if f.get('point_count')}
  for tid in set(oldframes[index])|set(new):
   record=targets.setdefault(tid,dict(old_covered_frames=0,new_covered_frames=0,old_points=0,new_points=0,new_missing_frames=[],newly_found_frames=[]))
   record['old_covered_frames']+=int(tid in oldframes[index]);record['new_covered_frames']+=int(tid in new);record['old_points']+=oldframes[index].get(tid,0);record['new_points']+=new.get(tid,0)
   if tid in oldframes[index] and tid not in new:record['new_missing_frames'].append(index);missing.append(dict(index=index,target_id=tid,old_points=oldframes[index][tid]))
   if tid not in oldframes[index] and tid in new:record['newly_found_frames'].append(index)
 write(OUT/'repaired_coverage_comparison.json',dict(candidate_folder=str(folder),per_target=targets,new_missing=missing,all_missing_require_source_visibility_review=True))
def support_quality():
 """Attach native support without rewriting the motion archive already used."""
 archive=np.load(OUT/'source_ice_motion_all.npz');rows=[]
 for index,frozen in zip(archive['indices'],archive['homographies']):
  index=int(index);record,h=source_motion(index-1,index,save_tracks=False);record.update(previous_index=index-1,index=index)
  record['matches_frozen_motion_archive']=bool((h is None and not np.isfinite(frozen).all()) or (h is not None and np.allclose(h,frozen,rtol=1e-9,atol=1e-8)))
  if not record['matches_frozen_motion_archive']:raise ValueError('Source transport changed: '+str(index))
  rows.append(record)
  if index%100==0:
   write(OUT/'all_source_motion_support.json',dict(status='running',record_count=len(rows),motion_archive_sha256=sha(OUT/'source_ice_motion_all.npz'),records=rows));print('source_motion_support',index,flush=True)
 write(OUT/'all_source_motion_support.json',dict(status='complete',record_count=len(rows),motion_archive_sha256=sha(OUT/'source_ice_motion_all.npz'),records=rows,
  points_only_credible_within_verified_inlier_support=True,outside_support_does_not_establish_camera_motion=True))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--rank',action='store_true');p.add_argument('--pairs',nargs='*');p.add_argument('--anchor-pairs',nargs='*');p.add_argument('--full-source-scan',action='store_true');p.add_argument('--support-quality',action='store_true');p.add_argument('--evidence',action='store_true');p.add_argument('--coverage',action='store_true');p.add_argument('--compare-repaired',action='store_true');p.add_argument('--supported-motion',action='store_true');p.add_argument('--lines-only',action='store_true');p.add_argument('--ring-nesting',action='store_true');p.add_argument('--label',default='baseline');p.add_argument('--candidate-folder',default=str(DATA/'ice_candidates'));args=p.parse_args();OUT.mkdir(parents=True,exist_ok=True);cv2.setNumThreads(2)
 if args.rank:rank(args.candidate_folder)
 if args.pairs:check_pairs([tuple(map(int,s.split(':'))) for s in args.pairs],args.candidate_folder)
 if args.full_source_scan:full_source_scan(args.candidate_folder)
 if args.evidence:failure_evidence(args.candidate_folder)
 if args.coverage:coverage(args.candidate_folder)
 if args.compare_repaired:compare_repaired(args.candidate_folder)
 if args.support_quality:support_quality()
 if args.supported_motion:supported_motion(args.candidate_folder,args.label,args.lines_only)
 if args.ring_nesting:ring_nesting(args.candidate_folder,args.label)
 if args.anchor_pairs:
  rows=[]
  for value in args.anchor_pairs:
   a,b=map(int,value.split(':'));record,h=source_anchor_motion(a,b);rows.append(record);print(a,b,record,flush=True)
  write(OUT/'direct_anchor_registration.json',dict(records=rows))
