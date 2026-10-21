"""Streaming source-only physical board print tracks; no world/image projection fills UV."""
from pathlib import Path
from collections import defaultdict,Counter
import os
for _k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[_k]='1'
import json,gzip,hashlib
import av,cv2,numpy as np
from .io_utils import read_json,read_jsonl,write_json,file_sha256
from .source_texture_tracklets import _source_mask
from .native_entity_tracklets import NAVIGATION,_ncc
from .source_worker import _channels,_remeasure

def ref(p):
 p=Path(p).resolve();return dict(file=str(p),sha256=file_sha256(p))
def acquire(config,output):
 cfg=read_json(config);out=Path(output);out.mkdir(parents=True,exist_ok=True)
 source=cfg['source'];roles=read_json(cfg['roles'])['frame_roles'];first,last=int(cfg['first_pts']),int(cfg['last_pts']);sid=cfg['shot_id']
 if file_sha256(source['path'])!=source['sha256']:raise ValueError('Changed official native source')
 ledgers={r['native_pts']:r for p in cfg['actor_ledgers'] for r in read_jsonl(p) if first<=r['native_pts']<=last}
 seeds=defaultdict(list)
 for r in cfg['regions']:
  for name in ('source_feature_file','native_png','native_person_mask'): 
   if r.get(name) and file_sha256(r[name]['file'])!=r[name]['sha256']:raise ValueError('Changed exact source seed')
  seeds[r['native_pts']].append(r)
 cv2.setNumThreads(1)
 opts=dict(winSize=(31,31),maxLevel=3,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,40,.001))
 active={};regions={};tracks=[];terminal=[];frames=[];oldpts=None;previous=None;nextid=0;counts=Counter();seedstat=[];line=0
 def patch(g,u):return cv2.getRectSubPix(g.astype(np.float32),(33,33),tuple(np.asarray(u,float)))
 def hullmask(points):
  m=np.zeros((1080,1920),np.uint8)
  if len(points)>=3:
   h=np.asarray(points,np.float32).reshape(-1,1,2).astype(np.int32)
   if cv2.contourArea(h)>0:cv2.fillPoly(m,[h],255)
  return cv2.erode(m,np.ones((33,33),np.uint8))
 def flow(a,b,points):
  p=np.asarray(points,np.float32).reshape(-1,1,2);q,s,_=cv2.calcOpticalFlowPyrLK(a,b,p,None,**opts);back,t,_=cv2.calcOpticalFlowPyrLK(b,a,q,None,**opts)
  return q[:,0].astype(float),(s[:,0]&t[:,0]).astype(bool),np.linalg.norm(back[:,0]-p[:,0],axis=1)
 def finish(k,reason,p,d=None):
  t=active.pop(k);tracks.append(dict(track_id=k,board_instance_id=t['ad'],shot_id=sid,first_pts=t['first'],last_pts=t['last'],source_rows=t['n'],rows=t['n'],source_line_numbers_1based=t['lines'],seed_region_id=t['seed_region_id'],terminal_native_pts=p,terminal_reason=reason,terminal_source_diagnosis=d,known_world_corner_correspondence=False,shared_surface_coordinates='one unknown bounded Ad [u,v] per finite track'))
  terminal.append(dict(track_id=k,native_pts=p,reason=reason,source_evidence=d));counts['terminal_'+reason]+=1
 def emit(h,t,p,digest,fb,ncc,diag,is_seed):
  nonlocal line
  line+=1;t['n']+=1;t['last']=p;t['lines'].append(line)
  row=dict(track_id=t['id'],shot_id=sid,native_pts=p,time_base=[1,12800],source_uv=t['uv'].tolist(),current_uv=t['uv'].tolist(),current_rgb_sha256=digest,source_video_sha256=source['sha256'],surface_class='board',surface_instance_id=t['ad'],source_material='source-confirmed physical board printed texture',source_measurement='current_native_RGB_2D_corner' if is_seed else 'current_native_RGB_pyramidal_LK_2D_texture',forward_backward_px=fb,patch_ncc=ncc,two_direction_tensor_diagnostic=diag,seed_region_id=t['seed_region_id'],fit_role='development',source_geometry_eligible=False,solver_eligible=False,statistical_objective_eligible=False,numeric_statistics_ready=False,precision_status='engineering_source_texture_no_covariance',pixels_from_model=False,known_world_corner_correspondence=False,existing_world_corner_or_XY_used=False,ice_H_used=False,world_position_is_shared_track_latent=True,physical_position='one shared unknown continuous-board Ad [u,v] per finite track',actor_graphics_exclusion='exact RGB-bound existing per-exposure actor boxes plus graphic policy; selected native mask PNG only where present')
  h.write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n')
 with gzip.open(out/'candidate_rows.jsonl.gz','wt',encoding='utf8') as h,av.open(source['path']) as con:
  st=con.streams.video[0];con.seek(first,stream=st,backward=True)
  if str(st.time_base)!='1/12800':raise ValueError('Changed native clock')
  for fr in con.decode(st):
   p=int(fr.pts)
   if p<first:continue
   if p>last:break
   rgb=fr.to_ndarray(format='rgb24');g=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY);ld=ledgers.get(p)
   if ld is None:raise ValueError('Missing source actor ledger')
   digest=hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest();role=roles.get(str(p),{}).get('role');counts['decoded_native_frames']+=1
   if role!='development_candidate' or ld['role']!=role:
    for k in list(active):finish(k,'guard_heldout_or_missing_development_role',p,dict(current_rgb_sha256=digest))
    regions={};previous=None;oldpts=None;frames.append(dict(native_pts=p,RGB_sha256=digest,role=role,source_actor_ledger=('decoded_native_BGR_sha256' in ld),native_person_mask_PNG=False,observed_rows=0,region_status='guard_not_used_for_navigation_or_fit'));continue
   mask,digest=_source_mask(rgb,ld,source,p,sid)
   # Fixed guard corner supports stay excluded from texture navigation and fit.
   for box in cfg.get('guard_support_bbox_by_pts',{}).get(str(p),[]):
    x0,y0,x1,y1=box;cv2.rectangle(mask,(int(np.floor(x0-18)),int(np.floor(y0-18))),(int(np.ceil(x1+18)),int(np.ceil(y1+18))),255,-1)
   if previous is not None and p!=oldpts+256:raise ValueError('Native source step gap')
   # All source-reviewed keys receive their actual original PNG mask, never a propagated mask claim.
   selected=seeds.get(p,[])
   if selected and selected[0].get('native_person_mask'):
    mm=cv2.imread(selected[0]['native_person_mask']['file'],cv2.IMREAD_GRAYSCALE)
    if mm.shape!=(1080,1920):raise ValueError('Wrong selected native mask dimensions')
    mask=np.maximum(mask,cv2.dilate((mm>0).astype(np.uint8)*255,np.ones((33,33),np.uint8)))
   for ad,r in list(regions.items()):
    if previous is None or len(r['uv'])<4:regions.pop(ad);counts['ownership_navigation_lost']+=1;continue
    q,ok,fb=flow(previous,g,r['uv']);good=[]
    for j,u in enumerate(q):
     x,y=np.rint(u).astype(int)
     if ok[j] and fb[j]<=NAVIGATION['reciprocal_max_px'] and 0<=x<1920 and 0<=y<1080 and not mask[y,x] and _ncc(r['patches'][j],patch(g,u))>=NAVIGATION['patch_ncc_min']:good.append(j)
    if len(good)<4:regions.pop(ad);counts['ownership_navigation_lost']+=1;continue
    H,inlier=cv2.findHomography(np.asarray(r['uv'])[good],q[good],cv2.RANSAC,NAVIGATION['reciprocal_max_px'])
    if H is None or inlier is None or np.count_nonzero(inlier)<4:regions.pop(ad);counts['ownership_navigation_lost']+=1;continue
    poly=cv2.perspectiveTransform(np.asarray(r['poly'],np.float32)[None],H)[0]
    if not np.isfinite(poly).all() or not cv2.isContourConvex(poly.astype(np.float32)):regions.pop(ad);counts['ownership_navigation_lost']+=1;continue
    area=hullmask(poly);area[mask>0]=0
    nq=cv2.goodFeaturesToTrack(g,maxCorners=0,qualityLevel=.015,minDistance=31,mask=area,blockSize=5)
    nq=np.empty((0,2),float) if nq is None else nq[:,0].astype(float)
    r.update(poly=poly,uv=nq,patches=[patch(g,u) for u in nq],mask=area,navigation_H=H.tolist(),navigation_inliers=int(np.count_nonzero(inlier)))
   # An ownership refresh uses only the present, source-reviewed exact exposure.
   for s in selected:
    if s['current_RGB_sha256']!=digest:raise ValueError('Seed source RGB differs')
    area=hullmask(s['seed_region_native_polygon_uv']);area[mask>0]=0
    q=cv2.goodFeaturesToTrack(g,maxCorners=0,qualityLevel=.015,minDistance=31,mask=area,blockSize=5)
    q=np.empty((0,2),float) if q is None else q[:,0].astype(float)
    regions[s['board_instance_id']]=dict(poly=np.asarray(s['seed_region_native_polygon_uv'],float),uv=q,patches=[patch(g,u) for u in q],mask=area,seed_region_id=s['region_id'])
    seedstat.append(dict(native_pts=p,board_instance_id=s['board_instance_id'],source_region_id=s['region_id'],navigation_corners=len(q)))
   channels=_channels(rgb);before=line
   if active:
    keys=list(active);q,ok,fb=flow(previous,g,[active[k]['uv'] for k in keys])
    for j,k in enumerate(keys):
     t=active[k];u=q[j];x,y=np.rint(u).astype(int);reason=None;ncc=None;diag=None;rr=regions.get(t['ad'])
     if not ok[j] or fb[j]>NAVIGATION['reciprocal_max_px']:reason='actual_native_forward_backward_navigation_failed'
     elif not(0<=x<1920 and 0<=y<1080) or mask[y,x]:reason='current_texture_support_actor_overlay_image_edge_or_excluded'
     elif rr is None or not rr['mask'][y,x]:reason='source_owned_print_region_not_currently_supported'
     else:
      b=patch(g,u);ncc=_ncc(t['patch'],b)
      if ncc<NAVIGATION['patch_ncc_min']:reason='current_native_patch_navigation_below_existing075'
      else:
       diag=_remeasure(u,[0,0],channels,'texture',corner=True)
       if diag['status']!='entity_corner_two_direction_support':reason='current_texture_no_two_direction_information'
     if reason:finish(k,reason,p,dict(current_rgb_sha256=digest,navigation_uv=u.tolist(),forward_backward_px=float(fb[j]),patch_ncc=ncc,two_direction_tensor_diagnostic=diag));continue
     t.update(uv=u,patch=b);emit(h,t,p,digest,float(fb[j]),ncc,diag,False)
   # Replenish actual current 2D corners only inside the continuing seed-owned hull.
   # Existing active and parent source support suppresses near duplicate seeding.
   for ad,r in regions.items():
    area=r['mask'].copy();area[mask>0]=0
    for t in active.values():
     if t['ad']==ad:cv2.circle(area,tuple(np.rint(t['uv']).astype(int)),31,0,-1)
    for uv in cfg['parent_source_uv_by_pts'].get(str(p),[]):cv2.circle(area,tuple(np.rint(uv).astype(int)),31,0,-1)
    q=cv2.goodFeaturesToTrack(g,maxCorners=0,qualityLevel=.015,minDistance=31,mask=area,blockSize=5)
    for u in [] if q is None else q[:,0].astype(float):
     diag=_remeasure(u,[0,0],channels,'texture',corner=True)
     if diag['status']!='entity_corner_two_direction_support':counts['weak_seed_2D_not_started']+=1;continue
     k='board_continuous_'+sid+'/'+ad+'_'+str(nextid);nextid+=1
     t=dict(id=k,ad=ad,uv=u,patch=patch(g,u),first=p,last=p,n=0,lines=[],seed_region_id=r['seed_region_id']);active[k]=t;emit(h,t,p,digest,None,None,diag,True)
   frames.append(dict(native_pts=p,RGB_sha256=digest,role=role,source_actor_ledger=True,native_person_mask_PNG=bool(selected and selected[0].get('native_person_mask')),observed_rows=line-before,owned_instances=list(regions),source_navigation_regions_are_not_world_geometry=True))
   previous=g;oldpts=p
   if counts['decoded_native_frames']%50==0:print(json.dumps(dict(frame=counts['decoded_native_frames'],pts=p,rows=line,active=len(active))),flush=True)
  for k in list(active):finish(k,'shot_end',last+256)
 write_json(out/'tracks.json',tracks);write_json(out/'terminals.json',terminal);write_json(out/'frame_bindings.json',frames)
 write_json(out/'generation_counts.json',dict(counts,raw_candidate_rows=line,raw_candidate_tracks=len(tracks),source_seed_status=seedstat,all_native_frames=len(frames),source_time_range_seconds=[first/12800,last/12800],full_frame_cache_written=False))
 return dict(rows=line,tracks=len(tracks),frames=len(frames))
