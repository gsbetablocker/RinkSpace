"""Source-only fixed-birth template drift audit; emits diagnostics, never changes source UV."""
from pathlib import Path
from collections import defaultdict, Counter
import os
for _k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[_k]='1'
import json,gzip,hashlib
import av,cv2,numpy as np
from .io_utils import read_json, read_jsonl, write_json, file_sha256
from .source_texture_tracklets import _source_mask
from .native_entity_tracklets import NAVIGATION,_ncc

def ref(p):
 p=Path(p).resolve();return dict(file=str(p),sha256=file_sha256(p),size_bytes=p.stat().st_size)
def patch(g,u,n):return cv2.getRectSubPix(g.astype(np.float32),(n,n),tuple(np.asarray(u,float)))
def rect_ok(u,r,shape,guards):
 x,y=np.asarray(u,float);h,w=shape
 if x-r<0 or y-r<0 or x+r>=w or y+r>=h:return 'image_edge'
 for a,b,c,d in guards:
  if x+r>=a and x-r<=c and y+r>=b and y-r<=d:return 'guard_support_overlap'
 return None

def direct(a,b):
 # a fixed 33x33 template, b current 65x65 search. No intermediate frame is used.
 z=cv2.matchTemplate(b,a,cv2.TM_CCOEFF_NORMED);_,peak,_,loc=cv2.minMaxLoc(z)
 second=z.copy();x,y=loc;second[max(0,y-3):y+4,max(0,x-3):x+4]=-2
 runner=float(second.max())
 w=np.array([[1,0,x],[0,1,y]],np.float32)
 try: cc,w=cv2.findTransformECC(a,b,w,cv2.MOTION_AFFINE,(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,40,.001),None,3)
 except cv2.error as e:return dict(status='affine_ECC_failed',translation_NCC=float(peak),runnerup_translation_NCC=runner,reason=str(e).split('\n')[0])
 centre=w@np.array([16.,16.,1.]);corners=np.array([[0,0,1],[32,0,1],[32,32,1],[0,32,1]],float)@w.T
 det=float(np.linalg.det(w[:,:2]));sv=np.linalg.svd(w[:,:2],compute_uv=False)
 warped=cv2.warpAffine(b,w,(33,33),flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
 ncc=_ncc(a,warped)
 status='affine_registration'
 if not np.isfinite(w).all() or det<=0:status='affine_fold_or_nonfinite'
 elif corners.min()<1 or corners.max()>63:status='affine_support_leaves_buffer'
 return dict(status=status,warp=w.tolist(),centre_in_search=centre.tolist(),corners_in_search=corners.tolist(),affine_ECC=float(cc),registered_NCC=ncc,zero_mean_unit_variance_photometric_RMS=float(np.sqrt(max(0.,2.-2.*ncc))),translation_NCC=float(peak),runnerup_translation_NCC=runner,translation_peak_gap=float(peak-runner),affine_singular_values=sv.tolist(),affine_determinant=det)

def audit(generation,out,all_tracks=False):
 gd=Path(generation);out=Path(out);out.mkdir(parents=True,exist_ok=True)
 cfg=read_json(gd/'source_config.json');qual=read_json(gd/'qualification.json');source=cfg['source'];sid=cfg.get('shot_id',gd.name.split('_')[0]);shot=next(x for x in read_json(cfg['scope']['file'])['shots'] if x['id']==sid);cfg.setdefault('first_pts',shot['first_pts']);cfg.setdefault('last_pts',shot['last_pts'])
 if file_sha256(source['path'])!=source['sha256']:raise ValueError('Wrong actual source')
 roles=read_json(cfg['roles'])['frame_roles'];tracks=read_json(gd/'tracks.json');qualified=set(qual['qualified_tracks']) if isinstance(qual['qualified_tracks'],list) else None
 # Qualification authority contains an explicit per-track ledger, never numeric fit residuals.
 if qualified is None:
  pt=qual['per_track'];qualified={x['track_id'] for x in (pt.values() if isinstance(pt,dict) else pt) if x.get('physical_static_surface_qualified',x.get('qualified_for_unknown_shared_board_surface_point_initialization',False))}
 if not qualified:
  allowed={x['candidate_line_1based'] for x in qual['qualified_source_line_index']};qualified={t['track_id'] for t in tracks if any(n in allowed for n in t['source_line_numbers_1based'])}
 candidates=[t for t in tracks if t['track_id'] in qualified and t['rows']>=2]
 selected=[]
 if all_tracks:selected=candidates
 else:
  for ad in sorted({t['board_instance_id'] for t in candidates}):
   ts=[t for t in candidates if t['board_instance_id']==ad];m=max(t['rows'] for t in ts);selected.extend(t for t in ts if t['rows']==m)
 ids={t['track_id'] for t in selected};allrows=[]
 with gzip.open(gd/'candidate_rows.jsonl.gz','rt',encoding='utf8') as f:
  for i,line in enumerate(f,1):
   q=json.loads(line)
   if q['track_id'] in ids:q['source_line_1based']=i;allrows.append(q)
 bypts=defaultdict(list);bytrack=defaultdict(list)
 for q in allrows:bypts[q['native_pts']].append(q);bytrack[q['track_id']].append(q)
 for tid,rr in bytrack.items():
  rr.sort(key=lambda q:q['native_pts'])
  if any(b['native_pts']!=a['native_pts']+256 for a,b in zip(rr,rr[1:])):raise ValueError('Noncontinuous input track')
 ledgers={q['native_pts']:q for p in cfg['actor_ledgers'] for q in read_jsonl(p) if cfg['first_pts']<=q['native_pts']<=cfg['last_pts']}
 seeds={};stats=Counter();frames=[];summaryrows=[];affected=[];cv2.setNumThreads(1)
 with (out/'direct_rows.jsonl').open('w',encoding='utf8') as h,av.open(source['path']) as con:
  st=con.streams.video[0];con.seek(cfg['first_pts'],stream=st,backward=True)
  for fr in con.decode(st):
   p=int(fr.pts)
   if p<cfg['first_pts']:continue
   if p>cfg['last_pts']:break
   rgb=fr.to_ndarray(format='rgb24');digest=hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest();role=roles.get(str(p),{}).get('role');stats['decoded_frames']+=1
   frames.append(dict(native_pts=p,current_RGB_sha256=digest,role=role,used_for_registration=bool(bypts.get(p))))
   if not bypts.get(p):continue
   if role!='development_candidate':raise ValueError('Protected frame requested for registration')
   mask,digest2=_source_mask(rgb,ledgers[p],source,p,sid)
   if digest!=digest2:raise ValueError('RGB digest mismatch')
   g=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY);guards=cfg.get('guard_support_bbox_by_pts',{}).get(str(p),[])
   selected_seed=[s for s in cfg['regions'] if s['native_pts']==p]
   for s in selected_seed:
    if s.get('native_person_mask'):
     mm=cv2.imread(s['native_person_mask']['file'],cv2.IMREAD_GRAYSCALE)
     if file_sha256(s['native_person_mask']['file'])!=s['native_person_mask']['sha256']:raise ValueError('Changed mask')
     mask=np.maximum(mask,cv2.dilate((mm>0).astype(np.uint8)*255,np.ones((33,33),np.uint8)))
   for q in bypts[p]:
    tid=q['track_id'];uv=np.asarray(q['source_uv'],float)
    if q['current_rgb_sha256']!=digest:raise ValueError('Changed bound source row')
    base=dict(track_id=tid,source_row_id='static_board_texture/'+tid+'/'+str(p),shot_id=sid,native_pts=p,source_line_1based=q['source_line_1based'],source_uv=q['source_uv'],current_RGB_sha256=digest,surface_instance_id=q['surface_instance_id'],source_UV_changed=False)
    bad=rect_ok(uv,33,g.shape,guards)
    # The existing dilated mask at all search centres ensures source actor/graphics support is untouched.
    x,y=np.rint(uv).astype(int)
    if bad is None and mask[max(0,y-17):y+18,max(0,x-17):x+18].any():bad='actor_overlay_or_native_mask_search_support'
    if tid not in seeds:
     seeds[tid]=dict(pts=p,uv=uv.copy(),rgb=digest,bad=bad,template=patch(g,uv,33) if bad is None else None,buffer=patch(g,uv,65) if bad is None else None)
     base.update(status='birth_template' if bad is None else 'birth_template_unavailable',unavailable_reason=bad,birth_pts=p,age_native_steps=0)
    else:
     seed=seeds[tid];base.update(birth_pts=seed['pts'],birth_RGB_sha256=seed['rgb'],birth_uv=seed['uv'].tolist(),age_native_steps=(p-seed['pts'])//256)
     if bad or seed['bad']:
      base.update(status='unknown_support_unavailable',unavailable_reason=bad or 'birth_'+seed['bad'])
     else:
      now=patch(g,uv,65);d=direct(seed['template'],now);base['direct']=d
      if d['status']!='affine_registration':base.update(status='unknown_direct_registration_failed',unavailable_reason=d['status'])
      else:
       centre=np.asarray(d['centre_in_search']);dUV=uv+centre-32;delta=dUV-uv;dbad=rect_ok(dUV,16,g.shape,guards)
       di=np.rint(dUV).astype(int)
       if dbad is None and mask[di[1],di[0]]:dbad='actor_overlay_or_native_mask_registered_support'
       rev=direct(patch(g,dUV,33),seed['buffer']);base['reverse']=rev
       base.update(direct_current_uv=dUV.tolist(),direct_minus_original_uv=delta.tolist(),direct_minus_original_px=float(np.linalg.norm(delta)))
       if rev['status']=='affine_registration':
        back=seed['uv']+np.asarray(rev['centre_in_search'])-32;fb=float(np.linalg.norm(back-seed['uv']));base.update(long_baseline_forward_backward_px=fb,reverse_birth_uv=back.tolist())
        if dbad:base.update(status='unknown_registered_support_unavailable',unavailable_reason=dbad)
        elif d['registered_NCC']<NAVIGATION['patch_ncc_min'] or rev['registered_NCC']<NAVIGATION['patch_ncc_min']:base.update(status='unknown_long_baseline_appearance_changed',unavailable_reason='existing_NCC_075_not_met')
        elif fb>NAVIGATION['reciprocal_max_px']:base.update(status='unknown_long_baseline_reciprocal_failed',unavailable_reason='existing_FB_1px_not_met')
        else:base.update(status='fixed_birth_direct_registration_supported',unavailable_reason=None)
       else:base.update(status='unknown_reverse_registration_failed',unavailable_reason=rev['status'])
    if base.get('direct',{}).get('status')=='affine_registration' and base.get('reverse',{}).get('status')=='affine_registration':
     A=np.asarray(base['direct']['warp'])[:,:2];B=np.asarray(base['reverse']['warp'])[:,:2];bias=np.asarray(base['reverse']['centre_in_search'])-32
     e=np.array([[-16,-16],[-16,16],[16,-16],[16,16]],float)@(B@A-np.eye(2)).T+bias
     base['affine_patch_cycle_corner_errors_px']=np.linalg.norm(e,axis=1).tolist();base['affine_patch_cycle_max_px']=float(np.linalg.norm(e,axis=1).max());base['affine_jacobian_reciprocal_error_fro']=float(np.linalg.norm(B@A-np.eye(2)))
     base['direct_and_inverse_patch_reciprocal_within_existing_1px_reference']=base['status']=='fixed_birth_direct_registration_supported' and base['affine_patch_cycle_max_px']<=1.
    else:base['direct_and_inverse_patch_reciprocal_within_existing_1px_reference']=False
    h.write(json.dumps(base,separators=(',',':'),allow_nan=False)+'\n');stats[base['status']]+=1;summaryrows.append(base)
   if stats['decoded_frames']%50==0:print(json.dumps(dict(shot=sid,frames=stats['decoded_frames'],audited_rows=len(summaryrows))),flush=True)
 write_json(out/'frame_receipt.json',frames)
 valid=[q for q in summaryrows if q['status']=='fixed_birth_direct_registration_supported'];dr=np.asarray([q['direct_minus_original_px'] for q in valid]);ages={q['track_id']:max(z['age_native_steps'] for z in valid if z['track_id']==q['track_id']) for q in valid}
 for q in valid:
  if q['direct_minus_original_px']>NAVIGATION['reciprocal_max_px']:affected.append({k:q[k] for k in ('track_id','source_row_id','native_pts','source_line_1based','source_uv','direct_current_uv','direct_minus_original_px','long_baseline_forward_backward_px','age_native_steps')})
 report=dict(schema='rinkfit.fixed_birth_source_drift_audit.v1',generation=ref(gd/'manifest.json'),qualification=ref(gd/'qualification.json'),source_video=source,source_read_only=True,model_or_projection_used=False,source_UV_changed=False,selection='all qualified continuous tracks' if all_tracks else 'all ties at maximum qualified continuous length per physical Ad instance; no model residual selection',selected_tracks=selected,selected_track_count=len(ids),selected_rows=len(allrows),counts=dict(stats),drift_supported_quantiles_min_median_p95_max=None if not len(dr) else np.quantile(dr,[0,.5,.95,1]).tolist(),over_existing_1px_reciprocal_reference_rows=len(affected),highest_supported_disagreement=sorted(valid,key=lambda q:q['direct_minus_original_px'],reverse=True)[:20],limitations=['Fixed birth 33x33 texture template, 65x65 current source search, local affine image model only; no global H or world/model fit.','Original current UV supplies search navigation only. Forward affine ECC starts at actual translation NCC maximum; reverse registration is independently optimized against the fixed birth buffer.','Existing per-step FB<=1px/NCC>=.75 are reused as diagnostic gates, not issued source covariance or accuracy certification. Periodic pattern/affine ambiguity is retained in raw peak-gap and warp diagnostics.','Failures remain unknown. Protected roles and actual guard support are never read for registration; decoded receipt alone is not fit/navigation.','No current reader/generation/packet is changed. A source-supported disagreement requires visual/source identity assessment before a new generation can replace any UV.'],outputs=dict(direct_rows=ref(out/'direct_rows.jsonl'),frame_receipt=ref(out/'frame_receipt.json')))
 strong=[q for q in summaryrows if q['direct_and_inverse_patch_reciprocal_within_existing_1px_reference']];sv=[q['direct_minus_original_px'] for q in strong]
 report.update(whole_patch_reciprocal_supported_rows=len(strong),whole_patch_reciprocal_drift_quantiles_min_median_p95_max=np.quantile(sv,[0,.5,.95,1]).tolist() if sv else None,whole_patch_reciprocal_over_1px_rows=sum(v>1 for v in sv),highest_whole_patch_reciprocal_disagreement=sorted(strong,key=lambda q:q['direct_minus_original_px'],reverse=True)[:20])
 write_json(out/'affected_rows.json',affected);report['outputs']['affected_rows']=ref(out/'affected_rows.json');write_json(out/'report.json',report);return report
