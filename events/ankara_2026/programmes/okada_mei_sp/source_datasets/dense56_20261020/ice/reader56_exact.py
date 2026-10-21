import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import sys,json,gzip,hashlib,time,collections,argparse
from pathlib import Path
P=Path(r'D:/OneDrive/codex_projs/RinkSpace'); D=Path(r'D:/CodexData/RinkSpace_Data'); R=D/'runs/okada_rec_1019/recover_v2'; O=R/'dense56_ice_v1'; O.mkdir(exist_ok=True)
sys.path[:0]=[str(P),str(P/'src'),str(P/'calibration/rinkfit/src')]
from calibration.program_tracking.runtime.bootstrap import add_geometry_dependencies
add_geometry_dependencies()
import cv2,numpy as np,av
from rinkfit_joint.screen_recovery_state import ScreenRecoveryState
from rinkspace.vision_programme_ice_review import ice_curve_inventory,_source_queries,_batched_actual_profiles,_actual_profile_candidates
from rinkspace.vision_source_shapes import _MaterialProfileFrame
cv2.setNumThreads(1)
S=ScreenRecoveryState(R/'screen/state.json')
sp=P/'events/ankara_2026/programmes/okada_mei_sp/source_datasets'
scene=json.load(open(json.load(open(R.parent/'ice_grid_nav8a28_v1/crosswalk.json',encoding='utf8'))['old_scene']['file'],encoding='utf8'))
role=json.load(open(P/'events/ankara_2026/model_constraints/paint_roles_1018.json',encoding='utf8'))
paths={p['curve_id']:p for p in ice_curve_inventory(scene,role)}
# The confirmed user definition is authoritative over stale per-shot guides.
for key,p in paths.items():
 name=key.split(':')[0]
 if name in ['C0','A1','A2','A3','A4']:
  p.update(operator='paint_center',material_signal='dark' if name=='C0' else 'red',material_transition=None)
for key,p in paths.items():
 if p['feature_id'].startswith('goal_crease'):
  p.update(operator='paint_center',material_signal='blue',material_transition=None)
TH=hashlib.sha256; source_sha='e8b53966d56e72f3842d92c3ecbe6e9cc7e4670b7d0c9bd12f58fd2cff5ad57d'
scopes={'s05':(948992,1300992),'s06':(1301248,1425408)}
coverage=collections.defaultdict(dict)
guides=collections.defaultdict(list); keyframes={}; source_refs=[]; foot_guides=collections.defaultdict(list)
def warp(points,H):
 q=np.c_[points,np.ones(len(points))]@H.T
 return q[:,:2]/q[:,2,None]
def ellipse_points(g):
 t=np.linspace(0,2*np.pi,1025); a=np.deg2rad(g['angle_degrees']); Q=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
 return (np.c_[np.cos(t),np.sin(t)]*(np.asarray(g['axes_px'])/2))@Q.T+g['center_uv']
for sid,folder in [('s05','shot5_20261014'),('s06','shot6_20261014')]:
 for f in sorted((sp/folder/'frames').glob('*.json')):
  data=json.load(open(f,encoding='utf8'));pts=int(data['source_pts']);keyframes[pts]=data; source_refs.append(str(f)); grouped=collections.defaultdict(list)
  for fe in data['features']:
   key=fe.get('world_curve',fe.get('target_id','').removeprefix('ice.'))
   if key not in paths:continue
   vals=[x for s in fe.get('segments',[]) for x in s.get('source_native_readout_uv',s.get('uv',[]))]
   if len(vals):grouped[key].append((fe,np.asarray(vals,float)))
  foot=np.asarray([x for fe in data['features'] if fe.get('target_id','').startswith('board.contact') for s in fe.get('segments',[]) for x in s.get('source_native_readout_uv',s.get('uv',[]))],float)
  if len(foot)>=4:
   foot_guides[sid].append(dict(pts=pts,base=warp(foot,np.linalg.inv(S.matrix(pts,sid)))))
  for key,items in grouped.items():
   vals=np.concatenate([v for _,v in items]); p=paths[key]; fe=max(items,key=lambda z:len(z[1]))[0];g=fe.get('geometric_interpretation'); circle=('center_xyz' in p or p['feature_id'] in ['C0','A1','A2','A3','A4'])
   if circle:
    if g and 'axes_px' in g and min(g['axes_px'])>3 and max(g['axes_px'])<50000:
     uv=ellipse_points(g)
    elif len(vals)>=8:
     e=cv2.fitEllipse(vals.astype('f4'));g=dict(center_uv=e[0],axes_px=e[1],angle_degrees=e[2]);
     if min(e[1])<3 or max(e[1])>50000:continue
     uv=ellipse_points(g)
    else:continue
    center=np.asarray(g['center_uv'],float)
   else:
    if len(vals)<3:continue
    mean=vals.mean(0);_,_,V=np.linalg.svd(vals-mean,full_matrices=False);v=V[0]; dist=(vals-mean)@v
    if np.ptp(dist)<80:continue
    # Extend the observed straight material as navigation only; RGB still decides support.
    uv=mean+np.linspace(dist.min()-1600,dist.max()+1600,513)[:,None]*v;center=None
   inv=np.linalg.inv(S.matrix(pts,sid));base=warp(uv,inv);bc=warp(center[None],inv)[0] if center is not None else None
   guides[(sid,key)].append(dict(pts=pts,base=base,center_base=bc,source=str(f),source_pixels=len(vals)))
# Fresh R2/R3 source ellipse navigation from already inspected 600320 source frame is included above.
for k in guides:guides[k].sort(key=lambda g:g['pts'])
mask_paths={}
for pts,r in coverage.items():
 if r['shot_id'] not in scopes:continue
 a=r.get('actual_current_source')
 if a:
  ref=a.get('current_mask')
  if ref:mask_paths[pts]=ref.get('file',ref.get('path'))
for pts,data in keyframes.items():mask_paths[pts]=data['source']['mask_path']
# Direct observations in historical keyframes remain available separately; this generation rereads pixels.

def nearest_guide(sid,key,pts):
 gs=guides[(sid,key)]; return min(gs,key=lambda g:abs(g['pts']-pts))


mask_index={json.loads(l)['native_pts']:json.loads(l)for l in gzip.open(R/'dense56_masks_v1/index.jsonl.gz','rt',encoding='utf8')}
# Frozen old-8a native arrays are ROI navigation only, with their own exact crosswalk.
NAV=R.parent/'ice_grid_nav8a28_v1';nm=json.load(open(NAV/'manifest.json',encoding='utf8'));cw=json.load(open(NAV/'crosswalk.json',encoding='utf8'))
navpts=np.load(NAV/'frame_native_pts.npy',mmap_mode='r');navoff=np.load(NAV/'frame_offsets.npy',mmap_mode='r');navuv=np.memmap(NAV/'target_uv.bin',dtype='<f8',mode='r',shape=tuple(nm['fields']['target_uv']['shape']));navci=np.memmap(NAV/'crosswalk_index.bin',dtype='<i2',mode='r',shape=tuple(nm['fields']['crosswalk_index']['shape']));navmap={p['crosswalk_index']:p['curve_id']for p in cw['paired_paths']}
manual={};static_rois=collections.defaultdict(list)
for sid,folder in [('s05','shot5_20261014'),('s06','shot6_20261014')]:
 data=json.load(open(sp/folder/'ice_source_guides.json',encoding='utf8'));manual[sid]=data
 for rr in data.get('static_spot_rois',[]):static_rois[sid].append(rr)
# Stale C0/A entity outer-boundary guides never override the current midpoint definition.
for key,p in paths.items():
 if p['feature_id']in ['A1','A2','A3','A4','B1','B2','C0']:p.update(operator='paint_center',material_transition=None)

def curve_nav(sid,pts):
 H=S.matrix(pts,sid);out={}
 for (ss,key),gg in guides.items():
  if ss!=sid:continue
  g=nearest_guide(sid,key,pts);p=dict(paths[key]);uv=warp(g['base'],H)
  if g['center_base']is not None:p['_projected_center']=warp(g['center_base'][None],H)[0]
  out[key]=(uv,p,dict(source_file=g['source'],source_frame_pts=g['pts'],source_H_used_only_to_locate_search=True))
 i=int(np.searchsorted(navpts,pts))
 if i<len(navpts)and int(navpts[i])==pts:
  sl=slice(int(navoff[i]),int(navoff[i+1]));a=np.asarray(navuv[sl]);ix=np.asarray(navci[sl])
  for ci in np.unique(ix):
   key=navmap.get(int(ci));q=a[ix==ci]
   if key is None or key in out or key not in paths or len(q)<2:continue
   p=dict(paths[key])
   if 'center_xyz'in p or p['feature_id']in ['C0','A1','A2','A3','A4']:
    if len(q)>=5:
     try:
      e=cv2.fitEllipse(q.astype('f4'));g=dict(center_uv=e[0],axes_px=e[1],angle_degrees=e[2])
      if min(e[1])>3 and max(e[1])<50000:q=ellipse_points(g);p['_projected_center']=np.array(e[0])
     except cv2.error:pass
   else:
    m=q.mean(0);_,_,V=np.linalg.svd(q-m,full_matrices=False);t=(q-m)@V[0];q=m+np.linspace(t.min()-160,t.max()+160,160)[:,None]*V[0]
   out[key]=(q,p,dict(source_file=str(NAV/'target_uv.bin'),source_frame_pts=pts,native_array_crosswalk=str(NAV/'crosswalk.json'),old_8a_navigation_only=True))
 return out

# Actual-current material domains: graphic presence, true boardfoot, 8px person guard.
sys.path.insert(0,str(R/'dense234_v1'))
from source_surface_qualification import current_domains
HIST_FILE=R/'dense56_v1/historical_guidance_registered.json';HIST=json.load(open(HIST_FILE,encoding='utf8'));HIST_SHA=TH(HIST_FILE.read_bytes()).hexdigest();HIST_ENTRIES=[]
for hg in HIST['guides']:
 hp=int(hg['official_native_pts']);hs='s05'if hp<1301248 else's06';hn=hg['feature_id'];hk=hn+(':segment0'if hn in ['B1','C']else':edge'+str(hg['curve']['segment_index']));hu=np.asarray(hg['official_source_navigation_uv'],float);hm=hu.mean(0);_,_,HV=np.linalg.svd(hu-hm,full_matrices=False);ht=(hu-hm)@HV[0];hl=hm+np.linspace(ht.min()-100,ht.max()+100,513)[:,None]*HV[0];hr=12800 if hn in ['B1','C']else 32000
 HIST_ENTRIES.append(dict(key=hk,sid=hs,pts=hp,first=max(scopes[hs][0],hp-hr),last=min(scopes[hs][1],hp+hr),base=warp(hl,np.linalg.inv(S.matrix(hp,hs))),old_annotation_file=hg['old_annotation_file'],old_annotation_sha256=hg['old_annotation_sha256'],user_statement=hg['user_statement'],official_source_navigation_uv_count=len(hu)))
BASE_CURVE_NAV=curve_nav
def curve_nav(sid,pts):
 out=BASE_CURVE_NAV(sid,pts)
 for e in HIST_ENTRIES:
  if e['sid']!=sid or not(e['first']<=pts<=e['last']):continue
  key=e['key'];p=dict(paths[key]);uv=warp(e['base'],S.matrix(pts,sid));g=dict(source_file=str(HIST_FILE),source_file_sha256=HIST_SHA,source_frame_pts=e['pts'],old_annotation_file=e['old_annotation_file'],old_annotation_sha256=e['old_annotation_sha256'],historical_user_statement=e['user_statement'],historical_guidance_registered_on_actual_source_image=True,source_H_used_only_to_locate_search=True,not_new_observation_or_click=True)
  out[key]=(uv,p,g)
 return out
LAST_OLD_MASK=None

def mask_frame(rgb,pts,sid):
 global LAST_OLD_MASK
 mp=R/'dense56_masks_v1/masks'/f'{pts}.png';bp=R/'dense56_board_v1/shape_v3/anchored_frames'/f'{pts}.json';qp=R/'dense56_board_v1/shape_v3/qualified_frames'/f'{pts}.json'
 while not mp.exists()or not bp.exists()or not qp.exists():time.sleep(1)
 raw=cv2.imread(str(mp),0);assert raw is not None
 mi=mask_index[pts];assert mi['source_RGB_sha256']==TH(rgb.tobytes()).hexdigest();assert mi['mask_sha256']==TH(mp.read_bytes()).hexdigest()
 board=json.load(open(bp,encoding='utf8'));qualified=json.load(open(qp,encoding='utf8'));assert board['decoded_native_BGR_sha256']==TH(rgb[:,:,::-1].copy().tobytes()).hexdigest()
 graphics,border,receipt=current_domains(rgb[:,:,::-1].copy(),qualified)
 mask=cv2.dilate((raw>0).astype('u1'),np.ones((17,17),'u1'))>0;mask|=graphics;mask|=np.arange(1080)[:,None]<=border[None]
 # Current pink electronic reflection has resolved saturated RGB support in this source.
 f=rgb.astype('i2');pink=(f[:,:,0]-f[:,:,1]>38)&(f[:,:,2]-f[:,:,1]>20)&(f[:,:,0]>180)&(np.arange(1080)[:,None]>500);pink=cv2.dilate(pink.astype('u1'),np.ones((17,17),'u1'))>0;mask|=pink
 old=cv2.dilate((raw>0).astype('u1'),np.ones((11,11),'u1'))>0;old[15:275,10:565]=True;old[925:1020,1650:1919]=True
 n,lab,stats,cen=cv2.connectedComponentsWithStats(old.astype('u1'),8)
 for x,y,w,h,ar in stats[1:]:
  if ar>1000 and y>170 and y+h>600 and w<800:old[min(1080,int(y+h)-5):,max(0,int(x)-12):min(1920,int(x+w)+12)]=True
 oldpath=R/'dense56_board_v1/shape_v2/anchored_frames'/f'{pts}.json';oldboard=json.load(open(oldpath,encoding='utf8'))if oldpath.exists()else board;oldfoot=np.asarray(oldboard['roles']['board.contact']['full_native_y'],float);old|=np.arange(1080)[:,None]<(oldfoot[None]+10);old|=pink
 for ex in manual[sid].get('unresolved_source_regions',[]):
  if ex.get('native_pts')!=pts:continue
  e=ex.get('source_exclusion_evidence',{});poly=e.get('polygon_native_uv')
  if poly:
   cv2.fillPoly(mask.view('u1'),[np.rint(poly).astype('i4')],1);cv2.fillPoly(old.view('u1'),[np.rint(poly).astype('i4')],1)
  elif'source_roi_xyxy'in ex:
   x0,y0,x1,y1=ex['source_roi_xyxy'];mask[y0:y1,x0:x1]=True;old[y0:y1,x0:x1]=True
 LAST_OLD_MASK=old
 return mask,dict(mask_file=str(mp),mask_sha256=mi['mask_sha256'],person_guard_pixels=8,boardfoot_file=str(bp),boardfoot_file_sha256=TH(bp.read_bytes()).hexdigest(),qualified_board_file=str(qp),qualified_board_file_sha256=TH(qp.read_bytes()).hexdigest(),current_graphics_presence_receipt=receipt,old_board_v2_available=oldpath.exists(),newly_opened_native_pixels=int((old&~mask).sum()),new_domain_not_previous_mask_filter_only=True,source_domain='actual_current_RGB_graphics_presence_SAM2_8px_qualified_v3_boardfoot_current_electronic_reflection_and_specific_manual_exclusions')

def context_clear(mask,uv,sides,normal,qs):
 pieces=[np.asarray(uv)[None],np.asarray(sides),np.asarray(qs).reshape(-1,2)]
 tangent=np.array([-normal[1],normal[0]])
 for u in [uv,*sides]:pieces.append(np.asarray(u)+np.arange(-3,4)[:,None]*tangent)
 xy=np.rint(np.vstack(pieces)).astype('i4')
 return not((xy<4).any()or(xy>[1915,1075]).any()or mask[xy[:,1],xy[:,0]].any())

def candidate_options(frame,q,nn,ii,meta,mask):
 primary=np.array([p['material_signal']for key,p,g in meta]); batches={}
 batches['primary']=_batched_actual_profiles(frame,q,nn,ii,primary)
 need=np.array([p['feature_id']in ['A1','A2','A3','A4','B1','B2','C0','C']for key,p,g in meta]);ids=np.flatnonzero(need)
 if len(ids):batches['gray']=_batched_actual_profiles(frame,q[ids],nn[ids],ii[ids],np.array(['dark']*len(ids)))
 gi={int(v):j for j,v in enumerate(ids)};bycurve=collections.defaultdict(list);gates=collections.Counter()
 for i,(key,p,g)in enumerate(meta):
  x,y=np.rint(q[i]).astype(int)
  if mask[y,x]:gates['ROI_query_in_excluded_context']+=1;continue
  opts=[]
  for bk,bi,signal in [('primary',i,p['material_signal'])]+([('gray',gi[i],'dark')]if i in gi else[]):
   off,prof,sm,valid=batches[bk];count=int(valid[bi].sum());a=off[bi,:count];pr=prof[bi,:count];ss=sm[bi,:count]
   if len(a)<20:continue
   windows=([30.,64.,128.,190.]if p['feature_id']in ['B1','B2','C']else[24.,48.,64.])if p['operator']=='paint_center'else[None]
   for half in windows:
    ix=np.ones(len(a),bool)if half is None else abs(a)<=half
    if ix.sum()<20:continue
    poss=_actual_profile_candidates(q[i],nn[i],a[ix],pr[ix],ss[ix],operator=p['operator'],signal=signal,transition=p['material_transition'],with_quantiles=True)
    for u,edges,bad,qs in poss:
     if bad or not context_clear(mask,u,edges,nn[i],qs):continue
     if any(np.linalg.norm(u-z['uv'])<1.1 and abs(np.linalg.norm(edges[1]-edges[0])-z['width'])<2 for z in opts):continue
     width=float(np.linalg.norm(edges[1]-edges[0]));shift=float((u-q[i])@nn[i]);pos=(qs-q[i])@nn[i];contr=[abs(float(np.interp(v[2],a,ss)-np.interp(v[0],a,ss)))for v in pos];iqr=float(np.max(abs(pos[:,2]-pos[:,0])))
     if min(contr)<(1.5 if signal=='dark'else 2.5)or iqr>22:continue
     if p['operator']=='paint_center'and not(1.5<width<(300 if p['feature_id']in ['B1','B2','C']else 75)):continue
     if abs(shift)>(70 if p['feature_id']in ['B1','B2','C']else 38 if p['feature_id'].startswith('A')else 22):continue
     r,gc,b=frame.rgb[int(round(u[1])),int(round(u[0]))].astype(float)
     if p['feature_id']=='C0'and r-gc>4 and b-r<0:continue
     opts.append(dict(uv=u,edges=edges,qs=qs,contrast=contr,width=width,shift=shift,signal=signal,query=i,normal=nn[i],key=key,p=p,g=g,iqr=iqr))
  if opts:bycurve[key].extend(opts)
  else:gates['no_complete_current_material_support']+=1
 return bycurve,gates

def coherent_branch(opts,q,nn,key,pts):
 # Current source band controls establish one continuous branch before the second RGB read.
 name=key.split(':')[0];byq=collections.defaultdict(list)
 for o in opts:byq[o['query']].append(o)
 initial=[min(v,key=lambda z:abs(z['shift']))for v in byq.values()]
 if len(initial)<6:return [],None
 if name in ['B1','B2','C']:
  base=q[[o['query']for o in initial]];mean=base.mean(0);_,_,V=np.linalg.svd(base-mean,full_matrices=False);tangent=V[0];normal=V[1];alluv=np.array([o['uv']for o in opts]);t=(alluv-mean)@tangent;perp=(alluv-mean)@normal;tc=float(np.mean(t));scale=max(float(np.ptp(t))/2,1);tx=(t-tc)/scale
  rng=np.random.default_rng(pts);best=None;queryids=np.array([o['query']for o in opts]);widths=np.array([o['width']for o in opts]);shifts=np.array([abs(o['shift'])for o in opts])
  trials=[np.polyfit(tx[[opts.index(o)for o in initial]],np.array([(o['uv']-mean)@normal for o in initial]),2)]if False else[]
  for _ in range(64):
   ix=rng.choice(len(opts),3,replace=False)
   if np.ptp(tx[ix])<.7 or min(abs(np.diff(np.sort(tx[ix]))))<.12:continue
   coef=np.polyfit(tx[ix],perp[ix],2)
   if abs(coef[0])>max(14,np.ptp(t)*.025):continue
   res=abs(perp-np.polyval(coef,tx));ok=res<3.0
   if ok.sum()<6:continue
   medw=float(np.median(widths[ok]));ok&=(widths>max(1.5,medw*.4))&(widths<max(10,medw*2.5))
   uniq=np.unique(queryids[ok]);span=np.ptp(t[ok])if ok.any()else 0
   score=len(uniq)+span/50-np.median(shifts[ok])*.08 if len(uniq)else -1
   if best is None or score>best[0]:best=(score,coef,ok)
  if best is None:return [],None
  coef=best[1];good=best[2]
  for _ in range(3):
   coef=np.polyfit(tx[good],perp[good],2);res=abs(perp-np.polyval(coef,tx));good=res<3.0
  medw=float(np.median(widths[good]));kept=[]
  for qi,oo in byq.items():
   oo=[o for o in oo if abs((o['uv']-mean)@normal-np.polyval(coef,((o['uv']-mean)@tangent-tc)/scale))<3 and max(1.5,medw*.45)<o['width']<max(12,medw*2.2)]
   if oo:kept.append(min(oo,key=lambda z:abs((z['uv']-mean)@normal-np.polyval(coef,((z['uv']-mean)@tangent-tc)/scale))))
  if len(kept)<6 or np.ptp(np.array([o['uv']for o in kept])@tangent)<50:return [],None
  return kept,dict(mean=mean.tolist(),tangent=tangent.tolist(),normal=normal.tolist(),coef=coef.tolist(),t0=tc,scale=scale,width_px=medw,controls=len(kept),control_query_indices=[o['query']for o in kept],control_range_t=[float(np.min((np.array([o['uv']for o in kept])-mean)@tangent)),float(np.max((np.array([o['uv']for o in kept])-mean)@tangent))],scope='current_source_only_polynomial_for_complete_material_branch_navigation')
 # For circle/short straight identity, require ordered current samples on one source curve branch.
 seq=sorted(initial,key=lambda z:z['query']);keep=[]
 for o in seq:
  near=[z['shift']for z in seq if 0<abs(z['query']-o['query'])<=5]
  if len(near)>=2 and abs(o['shift']-np.median(near))>7:continue
  keep.append(o)
 if len(keep)<4:return [],None
 return keep,None

def remeasure_band(rgb,frame,pts,sid,key,m,mask):
 mean=np.array(m['mean']);v=np.array(m['tangent']);n=np.array(m['normal']);coef=np.array(m['coef']);scale=m['scale'];t0=m['t0']
 base=mean+np.arange(m['control_range_t'][0]-64,m['control_range_t'][1]+64,5.)[:,None]*v;tt=(base-mean)@v;base+=np.polyval(coef,(tt-t0)/scale)[:,None]*n
 q,nn,ii=_source_queries(dict(operator='paint_center'),base,np.ones(len(base)),rgb.shape[:2],8.,max(30.,min(100.,m['width_px']*2.5)))
 p=dict(paths[key]);p.update(operator='paint_center',material_transition=None);met=[(key,p,dict(source_frame_pts=pts,source_file=str(O/'source_band_controls.jsonl.gz'),whole_current_source_band=True))]*len(q)
 if not len(q):return []
 # Narrow current-band reread retains only interfaces close to the full branch.
 by,gates=candidate_options(frame,q,nn,ii,met,mask);out=[]
 for o in by.get(key,[]):
  if abs(o['shift'])>6 or not(max(1.5,m['width_px']*.5)<o['width']<max(14,m['width_px']*2)):continue
  out.append(o)
 byq=collections.defaultdict(list)
 for o in out:byq[o['query']].append(o)
 return [min(v,key=lambda o:abs(o['shift']))for v in byq.values()]

def read_frame(rgb,pts,sid):
 t0=time.monotonic();mask,gateinfo=mask_frame(rgb,pts,sid);nav=curve_nav(sid,pts);queries=[];normals=[];ints=[];meta=[];navsumm=[]
 for key,(uv,p,g)in nav.items():
  # Ambiguous old goal operator is not a blue fill-area boundary.
  if key.startswith('goal_crease'):continue
  if p['operator']=='unresolved':continue
  radius=190. if p['feature_id']in ['B1','B2','C']else 64. if p['feature_id']in ['A1','A2','A3','A4']else 36.
  qq,nn,ii=_source_queries(p,uv,np.ones(len(uv)),rgb.shape[:2],8.,radius)
  queries.extend(qq);normals.extend(nn);ints.extend(ii);meta.extend([(key,p,g)]*len(qq));navsumm.append(dict(curve_id=key,queries=len(qq),source_navigation=g))
 if not queries:return [],dict(shot_id=sid,native_pts=pts,point_candidates=0,reason='native RGB retained for actual source visibility review; no usable curve navigation',**gateinfo),[]
 q=np.asarray(queries);nn=np.asarray(normals);ii=np.asarray(ints);frame=_MaterialProfileFrame(rgb,dict(source_pts=pts,source_sha256=source_sha,time_base='1/12800',native_hw=[1080,1920],shot_id=sid));bycurve,gates=candidate_options(frame,q,nn,ii,meta,mask);rows=[];controls=[];featurecounts=collections.Counter()
 for key,opts in bycurve.items():
  selected,m=coherent_branch(opts,q,nn,key,pts)
  if m:
   m.update(shot_id=sid,native_pts=pts,feature_id='ice.'+key,installed_source_navigation=nav[key][2]);controls.append(m);selected=remeasure_band(rgb,frame,pts,sid,key,m,mask)
  name=key.split(':')[0]
  for j,o in enumerate(selected):
   u=o['uv'];p=o['p'];sides=o['edges']if p['operator']=='paint_center'else None
   fid='ice.'+key;binding='source_entity_annotation_identity_retained_not_physical_accuracy_accepted'
   if name in ['A3','A4']:fid='unbound.ice.paired_paint_band.'+key;binding='unbound_not_for_physical_fit'
   row=dict(shot_id=sid,native_pts=pts,feature_id=fid,curve_id=key,uv=u.tolist(),paired_edge_uv=sides.tolist()if sides is not None else None,source_kind='actual_current_native_RGB_complete_band_on_source_continuous_branch'if m else'actual_current_native_RGB_material_profile',source_row_id=f'dense56_ice/{sid}/{pts}/{key}/{j}',source_UV_is_direct_measurement=True,definition='width_midpoint_between_both_material_edges'if sides is not None else'declared_target_ring_material_boundary',surface='ice',source_uv_from_model=False,source_sha256=source_sha,native_time_base='1/12800',pixel_support=dict(query_index=o['query'],normal=o['normal'].tolist(),native_shift_px=o['shift'],side_interface_quartiles_uv=o['qs'].tolist(),signed_interface_IQR_contrast=o['contrast'],paint_width_px=o['width'],interface_IQR_px=o['iqr'],material_signal=o['signal'],material_transition=p['material_transition'],complete_both_material_sides_visible=sides is not None,entire_source_band_coherence_used_to_disambiguate_crossing=m is not None,source_RGB_center=rgb[int(round(u[1])),int(round(u[0]))].tolist()),navigation=dict(o['g']),occlusion=gateinfo['source_domain'],status='direct_current_pixel_supported',world_entity_binding=binding,visible_span_id=f'{sid}/{pts}/{key}/actual_current_material',canonical_generation='dense56_ice_source_exact_domain_v2')
   if sides is not None:assert np.linalg.norm(np.mean(sides,axis=0)-u)<1e-6
   if nav[key][2].get('historical_guidance_registered_on_actual_source_image'):row['navigation']['registered_historical_source_guidance']=dict(nav[key][2])
   support=np.vstack([u[None],o['qs'].reshape(-1,2),o['edges']]);ix=np.rint(support).astype(int);row['pixel_support']['old_conservative_domain_would_exclude_material_support']=bool(LAST_OLD_MASK[ix[:,1],ix[:,0]].any());row['pixel_support']['current_UV_in_newly_opened_source_domain']=bool(LAST_OLD_MASK[int(round(u[1])),int(round(u[0]))]);rows.append(row);featurecounts[fid]+=1
 cov=dict(shot_id=sid,native_pts=pts,time_base='1/12800',point_candidates=len(rows),feature_counts=dict(featurecounts),source_kind_counts=dict(collections.Counter(r['source_kind']for r in rows)),native_RGB_sha256=TH(rgb.tobytes()).hexdigest(),source_navigation_inventory=navsumm,omission_gates=dict(gates),all_visible_material_complete_claimed=False,newly_opened_center_UV_count=sum(r['pixel_support']['current_UV_in_newly_opened_source_domain']for r in rows),newly_opened_material_support_rows=sum(r['pixel_support']['old_conservative_domain_would_exclude_material_support']for r in rows),registered_historical_guidance_features=[k for k,v in nav.items()if v[2].get('historical_guidance_registered_on_actual_source_image')],seconds=time.monotonic()-t0,**gateinfo)
 return rows,cov,controls

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--shot');ap.add_argument('--preview',action='store_true');args=ap.parse_args();shots=[args.shot]if args.shot else list(scopes)
 preview={948992,967168,997632,1044992,1119232,1195520,1242368,1273856,1300992,1301248,1330944,1376000,1406720,1425408}
 for sid in shots:
  a,b=scopes[sid];wanted=set(range(a,b+1,256));wanted= wanted&preview if args.preview else wanted;out=O/('preview'if args.preview else'canonical')/sid;out.mkdir(parents=True,exist_ok=True);tot=0;nf=0;features=collections.Counter();kinds=collections.Counter();framecounts=[];start=time.monotonic()
  with gzip.open(out/'rows.jsonl.gz','wt',encoding='utf8')as wr,gzip.open(out/'coverage.jsonl.gz','wt',encoding='utf8')as wc,gzip.open(out/'source_band_controls.jsonl.gz','wt',encoding='utf8')as wctrl,av.open(str(D/'events/ankara_2026/programmes/okada_mei_sp/Mei OKADA (JPN) ｜ Junior Women Short Program ｜ Ankara 2026.mp4'))as con:
   st=con.streams.video[0];assert str(st.time_base)=='1/12800';con.seek(min(wanted),stream=st,backward=True)
   for fr in con.decode(st):
    pts=int(fr.pts)
    if pts>max(wanted):break
    if pts not in wanted:continue
    rgb=fr.to_ndarray(format='rgb24');rr,cov,ctrl=read_frame(rgb,pts,sid)
    for m in ctrl:wctrl.write(json.dumps(m,separators=(',',':'))+'\n')
    for r in rr:
     if r['navigation'].get('whole_current_source_band'):r['navigation']['source_file']=str(out/'source_band_controls.jsonl.gz')
     wr.write(json.dumps(r,separators=(',',':'))+'\n')
    wc.write(json.dumps(cov,separators=(',',':'))+'\n');wr.flush();wc.flush();wctrl.flush();tot+=len(rr);nf+=1;features.update(cov.get('feature_counts',{}));kinds.update(cov.get('source_kind_counts',{}));framecounts.append(len(rr))
    if args.preview or nf%64==1 or not rr:
     im=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
     for r in rr:
      cv2.circle(im,tuple(np.rint(r['uv']).astype(int)),2,(0,255,255)if r['paired_edge_uv']is not None else(40,255,70),-1)
     cv2.putText(im,f'{sid} PTS {pts} actual RGB material {len(rr)}',(16,1056),0,.7,(0,255,255),2);cv2.imwrite(str(out/f'{pts}.jpg'),im,[cv2.IMWRITE_JPEG_QUALITY,95])
    info=dict(shot_id=sid,frames=nf,rows=tot,native_pts=pts,frame_rows=len(rr),elapsed_seconds=time.monotonic()-start);(out/'progress.json').write_text(json.dumps(info),encoding='utf8');print(json.dumps(info),flush=True)
  assert nf==len(wanted)
  manifest=dict(schema='rinkspace.dense56_ice_source.v1',complete=True,shot_id=sid,native_first_pts=min(wanted),native_last_pts=max(wanted),frame_count=nf,rows=tot,feature_counts=dict(features),source_kind_counts=dict(kinds),frame_point_min=min(framecounts),frame_point_max=max(framecounts),rows_file=str(out/'rows.jsonl.gz'),coverage_file=str(out/'coverage.jsonl.gz'),source_band_controls_file=str(out/'source_band_controls.jsonl.gz'),source_video_sha256=source_sha,time_base='1/12800',native_step=256,all_native_RGB_read=True,all_frames_gate_current_true_SAM2_and_source_boardfoot=True,source_UV_from_model=False,parent_UV_packets_models_unchanged=True,width_midpoints_verified=True,all_visible_material_complete_claimed=False)
  for k in ['rows_file','coverage_file','source_band_controls_file']:manifest[k.replace('_file','_sha256')]=TH(Path(manifest[k]).read_bytes()).hexdigest()
  (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8');print(json.dumps(dict(complete=True,shot_id=sid,frames=nf,rows=tot)),flush=True)

def worker_read(task):
 rgb,pts,sid=task;return read_frame(rgb,pts,sid)
def main_pool():
 import multiprocessing as mp
 ap=argparse.ArgumentParser();ap.add_argument('--shot',required=True);ap.add_argument('--workers',type=int,default=4);ap.add_argument('--first',type=int);args=ap.parse_args();sid=args.shot;a,b=scopes[sid];a=args.first if args.first is not None else a;out=O/'canonical'/sid;out.mkdir(parents=True,exist_ok=True);start=time.monotonic();features=collections.Counter();kinds=collections.Counter();fc=[];tot=0;nf=0;expected=list(range(a,b+1,256));seen=[]
 with mp.get_context('spawn').Pool(args.workers)as pool,gzip.open(out/'rows.jsonl.gz','wt',encoding='utf8')as wr,gzip.open(out/'coverage.jsonl.gz','wt',encoding='utf8')as wc,gzip.open(out/'source_band_controls.jsonl.gz','wt',encoding='utf8')as wctrl,av.open(str(D/'events/ankara_2026/programmes/okada_mei_sp/Mei OKADA (JPN) ｜ Junior Women Short Program ｜ Ankara 2026.mp4'))as con:
  st=con.streams.video[0];con.seek(a,stream=st,backward=True);batch=[]
  def consume(batch):
   nonlocal nf,tot
   results=pool.map(worker_read,batch,chunksize=1)
   for (rgb,pts,ss),(rr,cov,ctrl)in zip(batch,results):
    for m in ctrl:wctrl.write(json.dumps(m,separators=(',',':'))+'\n')
    for r in rr:
     if r['navigation'].get('whole_current_source_band'):r['navigation']['source_file']=str(out/'source_band_controls.jsonl.gz')
     wr.write(json.dumps(r,separators=(',',':'))+'\n')
    wc.write(json.dumps(cov,separators=(',',':'))+'\n');tot+=len(rr);nf+=1;features.update(cov.get('feature_counts',{}));kinds.update(cov.get('source_kind_counts',{}));fc.append(len(rr));seen.append(pts)
    if nf%64==1 or not rr:
     im=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
     for r in rr:cv2.circle(im,tuple(np.rint(r['uv']).astype(int)),2,(0,255,255)if r['paired_edge_uv']is not None else(40,255,70),-1)
     cv2.putText(im,f'{sid} PTS {pts} actual RGB material {len(rr)}',(16,1056),0,.7,(0,255,255),2);cv2.imwrite(str(out/f'{pts}.jpg'),im,[cv2.IMWRITE_JPEG_QUALITY,95])
   wr.flush();wc.flush();wctrl.flush();info=dict(shot_id=sid,frames=nf,rows=tot,native_pts=pts,frame_rows=len(rr),elapsed_seconds=time.monotonic()-start);(out/'progress.json').write_text(json.dumps(info),encoding='utf8');print(json.dumps(info),flush=True)
  for fr in con.decode(st):
   pts=int(fr.pts)
   if pts>b:break
   if pts<a:continue
   batch.append((fr.to_ndarray(format='rgb24'),pts,sid))
   if len(batch)==16:consume(batch);batch=[]
  if batch:consume(batch)
 assert seen==expected
 manifest=dict(schema='rinkspace.dense56_ice_source.v1',complete=True,shot_id=sid,native_first_pts=a,native_last_pts=b,frame_count=nf,rows=tot,feature_counts=dict(features),source_kind_counts=dict(kinds),frame_point_min=min(fc),frame_point_max=max(fc),rows_file=str(out/'rows.jsonl.gz'),coverage_file=str(out/'coverage.jsonl.gz'),source_band_controls_file=str(out/'source_band_controls.jsonl.gz'),source_video_sha256=source_sha,time_base='1/12800',native_step=256,all_native_RGB_read=True,all_frames_gate_current_true_SAM2_and_source_boardfoot=True,source_UV_from_model=False,parent_UV_packets_models_unchanged=True,width_midpoints_verified=True,all_visible_material_complete_claimed=False,parallel_current_frame_pixel_workers=args.workers)
 for k in ['rows_file','coverage_file','source_band_controls_file']:manifest[k.replace('_file','_sha256')]=TH(Path(manifest[k]).read_bytes()).hexdigest()
 (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8');print(json.dumps(dict(complete=True,shot_id=sid,frames=nf,rows=tot)),flush=True)

if __name__=='__main__':main_pool()

