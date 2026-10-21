"""Join immutable actual-source consumers; retain raw rows and their origins."""
from pathlib import Path
from collections import Counter,defaultdict
import gzip,json,hashlib,base64,time,sys,os
import cv2,numpy as np,av
R=Path(r'D:/CodexData/RinkSpace_Data/runs/okada_rec_1019/recover_v2');O=R/'dense234_v1';sys.path.insert(0,str(O))
from source_row_selection import select_source_rows
from source_surface_qualification import current_domains,rejection_reason
SPECS=[('s02',575744,720640),('s03',720896,745728),('s04',745984,948736)]
SOURCE=Path(r'D:/CodexData/RinkSpace_Data/events/ankara_2026/programmes/okada_mei_sp/Mei OKADA (JPN) ｜ Junior Women Short Program ｜ Ankara 2026.mp4')
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb')as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf8')
def source_rows(p):
 with gzip.open(p,'rt',encoding='utf8')as f:
  for n,l in enumerate(f):
   if l.strip():yield json.loads(l),n
class Cursor:
 def __init__(self,p):self.file=Path(p);self.it=iter(source_rows(p));self.nxt=next(self.it,None)
 def frame(self,p):
  if self.nxt and self.nxt[0]['native_pts']<p:raise ValueError('Unconsumed source PTS '+str(self.file))
  a=[]
  while self.nxt and self.nxt[0]['native_pts']==p:a.append(self.nxt);self.nxt=next(self.it,None)
  return a
 def close(self):
  if self.nxt:raise ValueError('Source outside requested interval '+str(self.file))
def infer(r):return r.get('source_UV_is_direct_measurement')is False or r.get('pixels_measured_in_this_frame')is False or r.get('source_evidence_measured_in_this_frame')is False or 'inferred_'in r.get('source_kind','')
def display(r,part):
 if r.get('user_confirmed')or r.get('source_kind')=='preserved_user_manual_native_image_click':return 'manual'
 if part=='ice':return 'ice_inferred'if infer(r)else'ice'
 if part=='board':return ('board_top'if r['feature_id']=='board.top'else'board_contact')+('_inferred'if infer(r)else'')
 return 'print'
def run():
 target=O/'generation_v1'
 if target.exists():raise FileExistsError(target)
 boardm=R/'dense_board_v1/qualified_manifest.json';bm=read(boardm)
 if not bm.get('complete')or bm.get('source_quality_status','').startswith('pending'):raise ValueError('Board source self-review not frozen')
 admanifest=R/'dense_print_v1/final_manifest_anchor.json'
 if not admanifest.exists():raise ValueError('Final advert source index not frozen')
 am=read(admanifest)
 if not am.get('complete'):raise ValueError('Advert generation incomplete')
 pins={str(boardm):sha(boardm),str(admanifest):sha(admanifest)}
 for recipe in [Path(__file__),O/'source_row_selection.py',O/'source_surface_qualification.py',Path(r'D:/OneDrive/codex_projs/RinkSpace/calibration/program_tracking/extract_first_shot_ice.py'),Path(r'D:/OneDrive/codex_projs/RinkSpace/calibration/program_tracking/extract_continuation_ice_source.py')]:pins[str(recipe)]=sha(recipe)
 adindex=Path(am['perframe_index']['file'])
 if sha(adindex)!=am['perframe_index']['sha256']:raise ValueError('Final advert index changed')
 adpins={j['native_pts']:j for j,n in source_rows(adindex)}
 if len(adpins)!=1458:raise ValueError('Incomplete advert index')
 pins[str(adindex)]=sha(adindex)
 ih=R/'dense_ice_v1/handoff_v3.json';pins[str(ih)]=sha(ih)
 for sid,a,b in SPECS:
  m=R/'dense_ice_v1/canonical_v3'/sid/'manifest.json';j=read(m)
  if not j.get('complete')or j['frame_count']!=(b-a)//256+1:raise ValueError('Ice generation incomplete')
  p=Path(j['rows_file'])
  if sha(p)!=j['rows_sha256']:raise ValueError('Ice rows SHA changed')
  pins[str(m)]=sha(m);pins[str(p)]=j['rows_sha256']
 correction_config=O/'source_corrections.json';cj=read(correction_config)
 if not cj.get('complete'):raise ValueError('Source corrections pending')
 pins[str(correction_config)]=sha(correction_config);exclude_ids={};delta_rows=defaultdict(list);applied_exclusions=set()
 for cm in cj['manifests']:
  mp=Path(cm['file'])
  if sha(mp)!=cm['sha256']:raise ValueError('Correction manifest changed')
  j=read(mp)
  if 'exclusions_file'not in j and 'exclusions'in j:j=dict(j,exclusions_file=j['exclusions']['file'],exclusions_sha256=j['exclusions']['sha256'])
  if not j.get('complete'):raise ValueError('Correction incomplete')
  pins[str(mp)]=sha(mp)
  for key in('rows','exclusions'):
   if key+'_file'not in j:continue
   q=Path(j[key+'_file'])
   if sha(q)!=j[key+'_sha256']:raise ValueError('Correction data changed')
   pins[str(q)]=sha(q)
  for r,n in source_rows(j['exclusions_file']):exclude_ids[r['source_row_id']]=(r,dict(file=j['exclusions_file'],jsonl_line_0based=n))
  if 'rows_file'in j:
   for r,n in source_rows(j['rows_file']):delta_rows[r['native_pts']].append((r,dict(file=j['rows_file'],jsonl_line_0based=n)))
 maskindex=R/'dense_masks_v1/index.jsonl.gz';masks={j['native_pts']:j for j,n in source_rows(maskindex)}
 pins[str(maskindex)]=sha(maskindex)
 manuals=defaultdict(list)
 for mp in [O/'manual_ice_anchors.jsonl.gz',R/'dense_print_v1/manual_anchor_rows.jsonl.gz']:
  pins[str(mp)]=sha(mp)
  for r,n in source_rows(mp):manuals[r['native_pts']].append((r,dict(file=str(mp),jsonl_line_0based=n)))
 target.mkdir();(target/'graphics_masks').mkdir();rowsout=gzip.open(target/'selected_rows.jsonl.gz','wt',encoding='utf8',compresslevel=4);framesout=gzip.open(target/'selected_frames.jsonl.gz','wt',encoding='utf8',compresslevel=4);coverage=gzip.open(target/'coverage.jsonl.gz','wt',encoding='utf8',compresslevel=4);excluded=gzip.open(target/'excluded.jsonl.gz','wt',encoding='utf8',compresslevel=4);bind=gzip.open(target/'consumer_bindings.jsonl.gz','wt',encoding='utf8',compresslevel=4)
 totals=Counter();rejection_totals=Counter();byshot=defaultdict(Counter);features=defaultdict(Counter);kinds=Counter();unknown=Counter();cache={};started=time.perf_counter();framecount=0;selectedcount=0;pointcounts=[]
 original=av.open(str(SOURCE));st=original.streams.video[0];original.seek(SPECS[0][1],stream=st,backward=True);decoder=iter(original.decode(st))
 def native_frame(p):
  for f in decoder:
   if int(f.pts)<p:continue
   if int(f.pts)!=p:raise ValueError('Missing native source-domain exposure')
   return f.to_ndarray(format='bgr24')
  raise ValueError('Source-domain video ended')
 try:
  for sid,first,last in SPECS:
   previous=();ic=Cursor(R/'dense_ice_v1/canonical_v3'/sid/'rows.jsonl.gz');cc=Cursor(R/'dense_ice_v1/canonical_v3'/sid/'coverage.jsonl.gz')
   for p in range(first,last+1,256):
    mi=masks[p];maskpath=Path(mi['mask_file'])
    if sha(maskpath)!=mi['mask_sha256']:raise ValueError('Mask changed')
    mask=cv2.imread(str(maskpath),0)
    if mask is None or mask.shape!=(1080,1920):raise ValueError('Invalid native person mask')
    bf=R/'dense_board_v1/qualified_frames'/f'{p}.json';b=read(bf)
    af=R/'dense_print_v1/final_frames_anchor'/f'{p}.json';ad=read(af)
    if sha(af)!=adpins[p]['sha256']:raise ValueError('Frozen advert frame changed')
    anchored=R/'dense_board_v1/anchored_frames'/f'{p}.json';ah=sha(anchored)
    if Path(ad['final_board_ref']['file']).resolve()!=anchored.resolve()or ad['final_board_ref']['sha256']!=ah:raise ValueError('Ads use different board generation')
    if b.get('anchored_file_sha256',ah)!=ah:raise ValueError('Board qualification differs from advert band')
    if b['decoded_native_BGR_sha256']!=mi['source_BGR_sha256']or ad['decoded_native_BGR_sha256']!=mi['source_BGR_sha256']:raise ValueError('Consumer source image mismatch')
    if sha(ad['mask_ref']['file'])!=mi['mask_sha256']or ad['mask_ref']['sha256']!=mi['mask_sha256']:raise ValueError('Advert mask differs')
    raw=native_frame(p)
    if hashlib.sha256(raw.tobytes()).hexdigest()!=mi['source_BGR_sha256']:raise ValueError('Surface domains decoded from different pixels')
    graphics,ice_border,domain_receipt=current_domains(raw,b);gm=target/'graphics_masks'/f'{p}.png';cv2.imwrite(str(gm),graphics.astype('uint8')*255);gmsha=sha(gm)
    cov=cc.frame(p)
    if len(cov)!=1 or cov[0][0]['native_RGB_sha256']!=mi['source_RGB_sha256']:raise ValueError('Ice uses different source pixels')
    candidates=[];refs=[];parts=[];dropped=Counter();seen=set()
    def append(r,ref,part):
     nonlocal candidates,refs,parts
     if r['native_pts']!=p or r['shot_id']!=sid:raise ValueError('Row clock mismatch')
     if part=='ice'and r.get('source_row_id')in exclude_ids:
      er,eref=exclude_ids[r['source_row_id']];applied_exclusions.add(r['source_row_id']);dropped['ice/source_qualification_correction']+=1
      excluded.write(json.dumps(dict(native_pts=p,reason=er['reason'],source_row_id=r['source_row_id'],raw_ref=ref,correction_ref=eref),ensure_ascii=False,separators=(',',':'))+'\n');return
     uv=np.asarray(r['uv'],float);reason=None
     if uv.shape!=(2,)or not np.isfinite(uv).all()or(uv<0).any()or(uv>[1919,1079]).any():reason='outside_native_pixel_domain'
     elif mask[int(round(uv[1])),int(round(uv[0]))]>0:reason='current_person_pixel_mask'
     if reason is None:reason=rejection_reason(r,part,graphics,ice_border)
     if reason is None and part=='ice'and r.get('source_kind')=='actual_current_native_RGB_single_blue_material_interface':reason='single_visible_wide_band_edge_cannot_certify_midpoint_retained_raw'
     if reason:
      dropped[part+'/'+reason]+=1
      if True:excluded.write(json.dumps(dict(native_pts=p,reason=reason,uv=r['uv'],source_row_id=r.get('source_row_id',r.get('point_id')),source_kind=r.get('source_kind'),raw_ref=ref),ensure_ascii=False,separators=(',',':'))+'\n')
      return
     identity=r.get('point_id')or r.get('source_row_id')or part+'/'+str(len(candidates))
     key=(part,identity)
     if key in seen:dropped[part+'/same_source_ID_duplicate']+=1;return
     seen.add(key);candidates.append(r);refs.append(ref);parts.append(part)
    for i,r in enumerate(b['rows']):append(r,dict(file=str(bf),json_pointer='/rows/'+str(i)),'board')
    for r,n in ic.frame(p):append(r,dict(file=str(ic.file),jsonl_line_0based=n),'ice')
    allads=[]
    for mn,m in enumerate(ad['source_members']):
     mf=str(Path(m['file']))
     if mf not in cache:cache[mf]=open(mf,'rb')
     f=cache[mf];f.seek(m['offset_bytes']);raw=f.read(m['length_bytes'])
     if hashlib.sha256(raw).hexdigest()!=m['sha256']:raise ValueError('Advert gzip member changed')
     rr=[json.loads(l)for l in gzip.decompress(raw).splitlines()if l.strip()]
     if len(rr)!=m['row_count']:raise ValueError('Advert member rows missing')
     allads.extend((r,dict(file=mf,gzip_member_offset=m['offset_bytes'],gzip_member_bytes=m['length_bytes'],gzip_member_sha256=m['sha256'],member_jsonl_line_0based=i))for i,r in enumerate(rr))
    eligible=base64.b64decode(ad['eligible_uint8_base64']);statuses=base64.b64decode(ad['status_uint8_base64'])
    if len(eligible)!=len(allads)or len(statuses)!=len(allads)or sum(eligible)!=ad['eligible_rows']:raise ValueError('Advert eligibility does not bind exact input rows')
    for i,(r,ref)in enumerate(allads):
     if eligible[i]:append(r,dict(ref,qualification_file=str(af),qualification_row_0based=i),'print')
     else:dropped['print/qualification_status_'+str(statuses[i])]+=1
    for r,ref in delta_rows.get(p,[]):append(r,ref,'ice')
    for r,ref in manuals.get(p,[]):append(r,ref,'ice'if r.get('physical_surface')=='ice'else'print')
    indices,sel=select_source_rows(candidates,previous_point_ids=previous)
    required={r['feature_id']for r in candidates if r['feature_id'].startswith(('board.','ice.','unbound.ice.','Ad'))}
    represented={candidates[i]['feature_id']for i in indices}
    if required-represented:raise ValueError('Visible named source entities omitted by selection')
    framepoints=[];frameby=Counter();framekinds=Counter();framefeatures=Counter();selectedids=[]
    for rank,i in enumerate(indices):
     r=candidates[i];part=parts[i];ref=refs[i];rid=f'dense234_v1/{sid}/{p}/{rank}';origin=r.get('source_kind','unknown');feature=r.get('feature_id',r.get('role','unknown'));frameby[part]+=1;framekinds[origin]+=1;framefeatures[feature]+=1
     if feature.startswith('unbound.')or r.get('crosswalk_status')=='board_material_confirmed_component_unbound':unknown[part]+=1
     point=dict(selected_row_id=rid,shot_id=sid,native_pts=p,time_base='1/12800',feature_id=feature,uv=r['uv'],display_class=display(r,part),source_kind=origin,measurement_definition=r.get('definition',r.get('measurement_operator','retained_parent_material_or_manual_definition')),physical_surface='ice'if part=='ice'else'board',user_confirmed=r.get('user_confirmed',False),source_UV_is_direct_measurement=not infer(r),source_pixel_or_annotation_ref=ref,source_point_id=r.get('point_id',r.get('source_row_id')),independent_fit_weight_issued=False,physical_world_correspondence_issued=False)
     rowsout.write(json.dumps(point,ensure_ascii=False,separators=(',',':'))+'\n');framepoints.append({k:point[k]for k in('selected_row_id','uv','display_class')});selectedids.append(r.get('point_id',r.get('source_row_id')))
    framesout.write(json.dumps(dict(shot_id=sid,native_pts=p,decoded_native_BGR_sha256=mi['source_BGR_sha256'],points=framepoints),separators=(',',':'))+'\n')
    coverage.write(json.dumps(dict(shot_id=sid,native_pts=p,candidate_rows_by_part=dict(Counter(parts)),selected_rows_by_part=dict(frameby),selected_feature_counts=dict(framefeatures),selected_source_kind_counts=dict(framekinds),rejected_rows=dict(dropped),selection=sel,board_visible_spans=b['final_visible_span_coverage'],source_domain_qualification=domain_receipt,ice_source_coverage_ref=dict(file=str(cc.file),jsonl_line_0based=cov[0][1]),visible_feature_completeness_claimed=False),ensure_ascii=False,separators=(',',':'))+'\n')
    bind.write(json.dumps(dict(shot_id=sid,native_pts=p,source_BGR_sha256=mi['source_BGR_sha256'],source_RGB_sha256=mi['source_RGB_sha256'],mask=dict(file=str(maskpath),sha256=mi['mask_sha256']),board=dict(file=str(bf),sha256=sha(bf)),anchored_board=dict(file=str(anchored),sha256=ah),advert=dict(file=str(af),sha256=sha(af)),graphics_mask=dict(file=str(gm),sha256=gmsha),source_domain_recipe_sha256=pins[str(O/'source_surface_qualification.py')]),separators=(',',':'))+'\n')
    for part,n in frameby.items():totals[part]+=n;byshot[sid][part]+=n
    features[sid].update(framefeatures);kinds.update(framekinds);rejection_totals.update(dropped);previous=set(x for x in selectedids if x);framecount+=1;selectedcount+=len(indices);pointcounts.append(len(indices))
    if framecount%50==0:write(target/'progress.json',dict(frames=framecount,total=1458,native_pts=p,selected_rows=selectedcount,elapsed_seconds=time.perf_counter()-started))
   ic.close();cc.close()
 finally:
  for f in(rowsout,framesout,coverage,excluded,bind):f.close()
  for f in cache.values():f.close()
  original.close()
 if framecount!=1458:raise ValueError('Missing requested source exposures')
 if set(exclude_ids)-applied_exclusions:raise ValueError('Source correction IDs did not bind actual parent rows')
 for f,h in pins.items():
  if sha(f)!=h:raise ValueError('Parent changed during source assembly '+f)
 outs={str(f.name):dict(file=str(f),sha256=sha(f))for f in target.glob('*.gz')}
 manifest=dict(schema='rinkspace.source_only_dense_234_generation.v1',complete=True,frames=framecount,selected_point_rows=selectedcount,point_count_per_native_frame=dict(min=min(pointcounts),max=max(pointcounts)),by_part=dict(totals),by_shot={k:dict(v)for k,v in byshot.items()},features_by_shot={k:dict(v)for k,v in features.items()},source_kind_counts=dict(kinds),rejected_source_rows_by_actual_reason=dict(rejection_totals),unknown_world_or_named_crosswalk_rows=dict(unknown),parent_pins=pins,source_correction_rows_applied=len(applied_exclusions),source_delta_rows_added=sum(map(len,delta_rows.values())),files=outs,source_UV_moved=False,old_source_generations_unchanged=True,physical_solver_or_live_packet_changed=False,selected_rows_are_independent_measurements=False,all_visible_entities_claimed_perfect=False,elapsed_seconds=time.perf_counter()-started)
 write(target/'manifest.json',manifest)
 cfg=dict(purpose='user_requested_source_points_review',source=dict(file=str(SOURCE),sha256='e8b53966d56e72f3842d92c3ecbe6e9cc7e4670b7d0c9bd12f58fd2cff5ad57d',native_fps='50/1',time_base='1/12800',native_frame_step_pts=256,image_size_hw=[1080,1920]),selected_frame_file=outs['selected_frames.jsonl.gz'],shots=[dict(id=s,first_pts=a,last_pts=b)for s,a,b in SPECS])
 write(target/'video_config.json',cfg);print(json.dumps(dict(frames=framecount,rows=selectedcount,by_shot=manifest['by_shot'],files=outs,elapsed_seconds=manifest['elapsed_seconds']),ensure_ascii=False))
if __name__=='__main__':run()
