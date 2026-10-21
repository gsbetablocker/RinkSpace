"""Current-source qualification only. Never moves or fills a source UV.
Red colour is a navigation signal, never a proof of reflection or a global mask.
The exclusion concerns an existing blue/gray width measurement when red mixture
occupies its actual complete two-sided profile, and its source identity is unresolved.
"""
import av, cv2, numpy as np, gzip,json,hashlib,collections,time,os
from pathlib import Path
O=Path(r'D:/CodexData/RinkSpace_Data/runs/okada_rec_1019/recover_v2/dense56_v1'); OUT=O/'reflection_v1';R=O.parent/'dense56_ice_v1'/'canonical'/'s05'
CFG=json.load(open(O/'consumer_s05.json',encoding='utf8'));SOURCE=CFG['source'];FIRST=948992;LAST=1300992;STEP=256
for d in ['candidate_domains','proof_crops']: (OUT/d).mkdir(exist_ok=True)
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def jd(x):return json.dumps(x,ensure_ascii=False,separators=(',',':'))+'\n'
def rowgroups():
 with gzip.open(R/'rows.jsonl.gz','rt',encoding='utf8') as f:
  group=[];last=None;line=0
  for st in f:
   r=json.loads(st);pts=r['native_pts']
   if last is not None and pts!=last:yield last,group;group=[]
   r['_line_0based']=line;line+=1;group.append(r);last=pts
  if group:yield last,group
# Explicit reader role, not colour-based relabelling. True red paint rows are
# outside this width qualification, including red material measured via luma.
def is_bluegray_width(r):
 p=r.get('pixel_support',{});key=r['curve_id']
 return r.get('definition')=='width_midpoint_between_both_material_edges' and r.get('paired_edge_uv') is not None and p.get('material_signal') in ('blue','dark') and (key.startswith('lane') or key.startswith(('B1:','B2:','C0:','house_connection_')))
def footprint(r):
 u=np.array(r['uv']);p=r['pixel_support'];n=np.array(p['normal']);n/=np.linalg.norm(n);t=np.array([-n[1],n[0]])
 sides=np.array(r['paired_edge_uv']);qs=np.array(p['side_interface_quartiles_uv']).reshape(-1,2);ref=np.vstack([u,sides,qs]);offset=(ref-u)@n
 # Filled complete normal profile (all source pixels), including the native
 # signed-interface quantiles and 8 px outside each furthest observed support.
 lo=float(offset.min()-8);hi=float(offset.max()+8)
 offsets=np.linspace(lo,hi,int(np.ceil(hi-lo))+1)
 xy=u[None,None,:]+offsets[None,:,None]*n+np.arange(-3,4)[:,None,None]*t
 return u,n,t,sides,qs,lo,hi,xy
expected={x['native_pts']:x for x in (json.loads(s) for s in gzip.open(R/'coverage.jsonl.gz','rt',encoding='utf8'))}
cap=av.open(SOURCE['file']);st=cap.streams.video[0];st.thread_type='AUTO';cap.seek(FIRST,stream=st,backward=True)
frames=iter(cap.decode(st));current=next(frames)
ex=gzip.open(OUT/'exclusions.jsonl.gz','wt',encoding='utf8');cov=gzip.open(OUT/'coverage.jsonl.gz','wt',encoding='utf8');cand=gzip.open(OUT/'candidate_profiles.jsonl.gz','wt',encoding='utf8')
features=collections.Counter();stats=collections.Counter();t0=time.time();proofsamples=[]
for fi,(pts,rows) in enumerate(rowgroups()):
 while current.pts<pts:current=next(frames)
 assert current.pts==pts,(current.pts,pts)
 rgb=current.to_ndarray(format='rgb24');digest=hashlib.sha256(rgb.tobytes()).hexdigest();assert digest==expected[pts]['native_RGB_sha256'],pts
 a=rgb.astype(np.int16);rg=a[:,:,0]-a[:,:,1];rb=a[:,:,0]-a[:,:,2]
 seed=(rg>=18)&(rb>=8)&(a[:,:,0]>=90)
 # Weaker fringe must be spatially connected to an actual strong red seed.
 fringe=(rg>=10)&(rb>=4)&(a[:,:,0]>=90)
 domain=(fringe & cv2.dilate(seed.astype('uint8'),np.ones((5,5),np.uint8)).astype(bool)).astype('uint8')
 domain=cv2.morphologyEx(domain,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
 npix,nlab,bb,cent=cv2.connectedComponentsWithStats(domain,8)
 # Retain every actual candidate component, with no projected/model mask.
 domain_path=OUT/'candidate_domains'/f'{pts}.png';cv2.imwrite(str(domain_path),domain*255);domain_sha=sha(domain_path)
 integ=cv2.integral(domain);frame_ex=[];frame_cand=0;frame_scope=0
 for r in rows:
  if not is_bluegray_width(r):continue
  frame_scope+=1;u,n,t,sides,qs,lo,hi,xy=footprint(r);xyi=np.rint(xy).astype('i4');xx=xyi[:,:,0];yy=xyi[:,:,1]
  x0=max(0,int(xx.min()));x1=min(1919,int(xx.max()));y0=max(0,int(yy.min()));y1=min(1079,int(yy.max()))
  if x1<x0 or y1<y0:continue
  count=int(integ[y1+1,x1+1]-integ[y0,x1+1]-integ[y1+1,x0]+integ[y0,x0])
  if not count:continue
  good=(xx>=0)&(xx<1920)&(yy>=0)&(yy<1080);hits=np.zeros_like(good);hits[good]=domain[yy[good],xx[good]]>0
  if not hits.any():continue
  # An actual current red mixture in the measured source strip is uncertain
  # when it reaches either complete interface's IQR/context, or occupies
  # the purported bluegray band between its two measured sides.
  si=(sides-u)@n;qi=(qs-u)@n;offsets=(xy[3]-u)@n
  inner=(offsets>=min(si))&(offsets<=max(si))
  interior_fraction=float(hits[:,inner].mean()) if inner.any() else 0
  interface_fractions=[];interface_rg=[];interface_rb=[]
  for side in sides:
   edge=side+np.arange(-2,3)[None,:,None]*n+np.arange(-3,4)[:,None,None]*t;ee=np.rint(edge).astype('i4');exx=np.clip(ee[:,:,0],0,1919);eyy=np.clip(ee[:,:,1],0,1079)
   interface_fractions.append(float(domain[eyy,exx].mean()));interface_rg.append(float(np.median(rg[eyy,exx])));interface_rb.append(float(np.median(rb[eyy,exx])))
  # Small isolated red flecks outside both edges are retained for review;
  # unresolvable competing material is withheld rather than given a UV.
  uncertain=(max(interface_fractions)>=.12 or interior_fraction>=.12)
  rec=dict(shot_id='s05',native_pts=pts,source_row_id=r['source_row_id'],feature_id=r['feature_id'],curve_id=r['curve_id'],uv=r['uv'],canonical_line_0based=r['_line_0based'],status='source_uncertain_competing_red_bluegray_material' if uncertain else 'candidate_fringe_only_width_not_withheld',reason='actual_current_red_mixture_prevents_independent_bluegray_two_edge_qualification' if uncertain else 'red_candidate_only_outside_complete_two_edge_material',source_RGB_sha256=digest,source_video_sha256=SOURCE['sha256'],candidate_domain_ref=dict(file=str(domain_path),sha256=domain_sha,semantics='actual_red_colour_candidate_only_not_a_global_exclusion_not_reflection_ground_truth'),full_support=dict(normal=n.tolist(),tangent=t.tolist(),normal_offset_range_px=[lo,hi],tangent_range_px=[-3,3],paired_edge_uv=r['paired_edge_uv'],side_interface_quartiles_uv=r['pixel_support']['side_interface_quartiles_uv'],native_filled_footprint_xyxy=[x0,y0,x1,y1]),red_support=dict(native_support_red_count=int(hits.sum()),native_support_sample_count=int(hits.size),band_red_fraction=interior_fraction,interface_red_fractions=interface_fractions,interface_median_RminusG=interface_rg,interface_median_RminusB=interface_rb),true_red_paint_rows_not_globally_excluded=True,source_UV_preserved=True,model_or_residual_used=False,thresholds_are_candidate_navigation=True,independent_physical_identity_issued=False)
  cand.write(jd(rec));frame_cand+=1
  if uncertain:
   ex.write(jd(rec));frame_ex.append(rec);features[r['feature_id']]+=1
  if len(proofsamples)<700 and (uncertain and (fi%25==0 or pts in [1095936,1096192,1096448])):
   proofsamples.append(rec)
 stats.update(frames=1,rows=len(rows),bluegray_width_rows=frame_scope,candidate_profiles=frame_cand,excluded_rows=len(frame_ex))
 components=[dict(component_id=i,area=int(bb[i,4]),xyxy=[int(bb[i,0]),int(bb[i,1]),int(bb[i,0]+bb[i,2]-1),int(bb[i,1]+bb[i,3]-1)]) for i in range(1,len(bb)) if bb[i,4]>=8]
 coverage=dict(shot_id='s05',native_pts=pts,time_base='1/12800',source_RGB_sha256=digest,canonical_RGB_SHA_matches=True,rows_scanned=len(rows),bluegray_width_rows_evaluated=frame_scope,candidate_profiles=frame_cand,excluded_rows=len(frame_ex),retained_raw_rows=len(rows)-len(frame_ex),candidate_domain_ref=dict(file=str(domain_path),sha256=domain_sha),actual_red_component_count=len(bb)-1,component_bboxes_navigation_only=components)
 cov.write(jd(coverage))
 if fi%25==0:
  ex.flush();cand.flush();cov.flush();json.dump(dict(stats=stats,last_pts=pts,seconds=time.time()-t0),open(OUT/'progress.json','w'),indent=2);print(fi,pts,len(frame_ex),dict(stats),round(time.time()-t0,1),flush=True)
ex.close();cov.close();cand.close();cap.close()
assert stats['frames']==1376 and stats['rows']==3051096,stats
json.dump(proofsamples,open(OUT/'proof_candidates.json','w',encoding='utf8'),indent=2)
manifest=dict(schema='rinkspace.current_source_competing_material_qualification.v1',complete=True,review_status='candidate_generated_visual_QA_pending',scope='S05 blue/gray width measurements only; true red paint never globally excluded',stats=stats,exclusions_by_feature=features,source=SOURCE,input_manifest=dict(file=str(R/'manifest.json'),sha256=sha(R/'manifest.json')),input_rows=dict(file=str(R/'rows.jsonl.gz'),sha256=sha(R/'rows.jsonl.gz')),coverage=dict(file=str(OUT/'coverage.jsonl.gz'),sha256=sha(OUT/'coverage.jsonl.gz')),exclusions=dict(file=str(OUT/'exclusions.jsonl.gz'),sha256=sha(OUT/'exclusions.jsonl.gz')),candidate_profiles=dict(file=str(OUT/'candidate_profiles.jsonl.gz'),sha256=sha(OUT/'candidate_profiles.jsonl.gz')),read_qualify_script=dict(file=str(OUT/'qualify.py'),sha256=sha(OUT/'qualify.py')),all_1376_current_RGB_SHA_match=True,all_raw_source_rows_preserved=True,source_UV_moves=0,no_refill_to_500=True,no_source_model_or_residual_used=True,all_red_pixels_proven_reflection=False,physical_fit_started=False,all_visible_material_qualified_claimed=False)
json.dump(manifest,open(OUT/'manifest.json','w',encoding='utf8'),indent=2);print('DONE',manifest['stats'],flush=True)
