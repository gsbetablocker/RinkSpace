"""Requalify frozen source UV with accepted actual actor and visible-surface generations."""
from pathlib import Path
from collections import Counter,defaultdict
import gzip,json,hashlib,base64,time,sys
import cv2,numpy as np,av
O=Path(__file__).resolve().parent
sys.path.insert(0,str(O))
from source_row_selection import select_source_rows
from source_surface_qualification import current_domains,rejection_reason,source_support_positions
SPECS=[('s05',948992,1300992),('s06',1301248,1425408)]
EXPECTED=[p for s,a,b in SPECS for p in range(a,b+1,256)]
PARTITION=False
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb')as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf8')
def stream(p):
 with gzip.open(p,'rt',encoding='utf8')as f:
  for n,l in enumerate(f):
   if l.strip():yield json.loads(l),n
class Cursor:
 def __init__(self,p):self.file=Path(p);self.it=iter(stream(p));self.next=next(self.it,None)
 def frame(self,p):
  if PARTITION:
   while self.next and self.next[0]['native_pts']<p:self.next=next(self.it,None)
  elif self.next and self.next[0]['native_pts']<p:raise ValueError('Unconsumed PTS '+str(self.file))
  rows=[]
  while self.next and self.next[0]['native_pts']==p:rows.append(self.next);self.next=next(self.it,None)
  return rows
 def close(self):
  if self.next and not PARTITION:raise ValueError('Extra source rows '+str(self.file))
def infer(r):return r.get('source_UV_is_direct_measurement')is False or r.get('pixels_measured_in_this_frame')is False or r.get('source_evidence_measured_in_this_frame')is False or 'inferred_'in r.get('source_kind','')
def display(r,part):
 if r.get('user_confirmed')or r.get('source_kind')=='preserved_user_manual_native_image_click':return'manual'
 if part=='ice':return'ice_inferred'if infer(r)else'ice'
 if part=='board':return('board_top'if r['feature_id']=='board.top'else'board_contact')+('_inferred'if infer(r)else'')
 return'print_inferred'if infer(r)else'print'
def supports(r,part):
 q=source_support_positions(r).tolist();ps=r.get('pixel_support',{})
 extra=ps.get('position_interval_endpoints_uv')
 if extra is not None:q.extend(np.asarray(extra,float).reshape(-1,2).tolist())
 platform=ps.get('material_plateau_support_endpoints_uv')
 if platform is not None:
  endpoints=np.asarray(platform,float).reshape(2,2);count=max(2,int(np.ceil(np.linalg.norm(endpoints[1]-endpoints[0])*2))+1)
  q.extend(np.linspace(endpoints[0],endpoints[1],count).tolist())
 if part=='print'and ps.get('normal')is not None:
  n=np.asarray(ps['normal'],float);u=np.asarray(r['uv'],float)
  if n.shape==(2,)and np.isfinite(n).all()and np.linalg.norm(n)>0:q.extend((u[None,:]+np.arange(-3.,3.001,.25)[:,None]*n[None,:]).tolist())
 if part=='print'and r.get('_root_qualification_status')in(2,3):
  u=np.asarray(r['uv'],float);q.extend((u[None,:]+np.asarray([(x,y)for y in range(-3,4)for x in range(-3,4)],float)).tolist())
 return np.asarray(q,float)
def run(config):
 global SPECS,EXPECTED,PARTITION
 cfg=read(config);SPECS=[q for q in SPECS if q[0]in cfg.get('shot_ids',['s05','s06'])]
 PARTITION='native_interval'in cfg
 if PARTITION:
  if len(SPECS)!=1:raise ValueError('Independent exposure partition must belong to one actual shot')
  s,a,b=SPECS[0];lo=int(cfg['native_interval']['first_pts']);hi=int(cfg['native_interval']['last_pts'])
  if lo<a or hi>b or lo>hi or(lo-a)%256 or(hi-a)%256:raise ValueError('Partition outside actual native shot clock')
  SPECS=[(s,lo,hi)]
 EXPECTED=[p for s,a,b in SPECS for p in range(a,b+1,256)]
 wanted=set(EXPECTED)
 target=Path(cfg['output']);pins={str(Path(config).resolve()):sha(config)}
 def pin(ref):
  p=Path(ref['file']);h=sha(p)
  if h!=ref['sha256']:raise ValueError('Frozen input changed '+str(p))
  pins[str(p)]=h;return p
 if cfg.get('purpose')!='user_requested_source_points_review':raise ValueError('Wrong task scope')
 if target.exists():raise FileExistsError(target)
 source=cfg['source'];pin(source)
 boardmanifest=pin(cfg['board_manifest']);bm=read(boardmanifest)
 if not bm.get('complete')or not bm.get('quality_accepted_for_raw_point_overlay')or bm.get('source_quality_status','').startswith('pending'):raise ValueError('Board source self-review not accepted')
 coverage_ref=bm['files']['qualified_coverage.jsonl.gz'];bc={j['native_pts']:j for j,n in stream(pin(coverage_ref))if j['native_pts']in wanted}
 if sorted(bc)!=EXPECTED:raise ValueError('Board source missing exposures')
 am=read(pin(cfg['advert_manifest']))
 if not am.get('complete')or not am.get('accepted_for_final_source_qualification'):raise ValueError('Advert current source qualification not accepted')
 for ref in am.get('canonical_member_files',[]):pin(ref)
 ap={j['native_pts']:j for j,n in stream(pin(am['perframe_index']))if j['native_pts']in wanted}
 if sorted(ap)!=EXPECTED:raise ValueError('Advert source missing exposures')
 masks={j['native_pts']:j for j,n in stream(pin(cfg['mask_index']))if j['native_pts']in wanted}
 if sorted(masks)!=EXPECTED:raise ValueError('Source mask exposure mismatch')
 upstream_masks={j['native_pts']:j for j,n in stream(pin(cfg['upstream_mask_index']))if j['native_pts']in wanted}
 if sorted(upstream_masks)!=EXPECTED:raise ValueError('Upstream source-mask exposure mismatch')
 actor_manifest=read(pin(cfg['actor_manifest']))
 if not actor_manifest.get('accepted_for_final_source_qualification'):raise ValueError('Current actor generation not accepted')
 surface_manifest=read(pin(cfg['surface_manifest']))
 if not surface_manifest.get('accepted_for_final_source_qualification'):raise ValueError('Current near-board generation not accepted')
 surface_index={j['native_pts']:j for j,n in stream(pin(cfg['surface_index']))if j['native_pts']in wanted}
 if 's06'in cfg.get('shot_ids',[])and wanted-set(surface_index):raise ValueError('Near-board visibility missing S06 source exposures')
 def source_mask(ref):
  path=pin(ref);arr=cv2.imread(str(path),0)
  if arr is None or arr.shape!=(1080,1920):raise ValueError('Bad actual source surface mask')
  return arr>0
 requalification_stats=Counter()
 for ref in cfg.get('additional_parent_pins',[]):pin(ref)
 for f in [Path(__file__),O/'source_row_selection.py',O/'source_surface_qualification.py',Path(r'D:/OneDrive/codex_projs/RinkSpace/calibration/program_tracking/extract_first_shot_ice.py')]:pins[str(f)]=sha(f)
 ice={};icecov={}
 for sid,a,b in SPECS:
  j=cfg['ice_shots'][sid];im=read(pin(j['manifest']))
  if not im.get('complete')or not im.get('sealed'):raise ValueError('Ice not sealed '+sid)
  ice[sid]=[Cursor(pin(r))for r in j['rows']];icecov[sid]=Cursor(pin(j['coverage']))
 exclusions={};applied_exclusions=set()
 for ref in cfg.get('ice_exclusions',[]):
  p=pin(ref)
  for r,n in stream(p):
   rid=r['source_row_id']
   if r['native_pts']not in wanted:continue
   if rid in exclusions:raise ValueError('Duplicate ice exclusion ID')
   exclusions[rid]=(r,dict(file=str(p),jsonl_line_0based=n))
 target.mkdir(parents=True);(target/'graphics_masks').mkdir()
 outputs={k:gzip.open(target/(k+'.jsonl.gz'),'wt',encoding='utf8',compresslevel=4)for k in['selected_rows','selected_frames','coverage','excluded','consumer_bindings']}
 def emit(k,r):outputs[k].write(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n')
 cache={};tot=Counter();byshot=defaultdict(Counter);features=defaultdict(Counter);kinds=Counter();classes=Counter();unknown=Counter();rejects=Counter();counts=[];framecount=0;selectedcount=0;start=time.perf_counter();manual=0
 original=av.open(source['file']);st=original.streams.video[0]
 if str(st.time_base)!='1/12800':raise ValueError('Wrong official time base')
 original.seek(EXPECTED[0],stream=st,backward=True);decoder=iter(original.decode(st))
 def image(p):
  for f in decoder:
   if int(f.pts)<p:continue
   if int(f.pts)!=p:raise ValueError('Missing official exposure')
   return f.to_ndarray(format='bgr24')
  raise ValueError('Official source ended')
 try:
  for sid,first,last in SPECS:
   previous=()
   for p in range(first,last+1,256):
    mi=masks[p];mp=Path(mi['mask_file']);mh=sha(mp)
    if mh!=mi['mask_sha256']:raise ValueError('Mask bytes changed')
    mask=cv2.imread(str(mp),0)
    if mask is None or mask.shape!=(1080,1920):raise ValueError('Bad source pixel mask')
    upstream_mi=upstream_masks[p];oldmaskpath=Path(upstream_mi['mask_file']);oldmasksha=sha(oldmaskpath)
    if oldmasksha!=upstream_mi['mask_sha256']:raise ValueError('Upstream actual actor mask changed')
    oldmask=cv2.imread(str(oldmaskpath),0)
    if oldmask is None or oldmask.shape!=mask.shape:raise ValueError('Bad upstream actor mask')
    lost_old_person=(oldmask>0)&(mask==0)
    if lost_old_person.any():
     disposition_ref=mi.get('verified_old_nonperson_disposition_ref')
     if not disposition_ref:raise ValueError('Upstream person pixels removed without actual source disposition')
     disposition=read(pin(disposition_ref))
     if disposition['native_pts']!=p or disposition['source_BGR_sha256']!=mi['source_BGR_sha256']or disposition['base_mask_sha256']!=oldmasksha or not disposition['old_true_person_pixels_preserved']:raise ValueError('Old nonperson disposition source differs')
     allowed_nonperson=source_mask(disposition['verified_old_nonperson_mask_ref'])
     if np.any(lost_old_person&~allowed_nonperson):raise ValueError('Unverified upstream person removal')
     requalification_stats['source_verified_old_nonperson_pixels_removed']+=int(lost_old_person.sum())
    if any(mi[k]!=upstream_mi[k]for k in('source_BGR_sha256','source_RGB_sha256')):raise ValueError('Actor source exposure identity changed')
    requalification_stats['actor_pixels_added']+=int(((mask>0)&(oldmask==0)).sum())
    distance=cv2.distanceTransform((mask==0).astype('uint8'),cv2.DIST_L2,5)
    unknown_ref=mi.get('source_unknown_exclusion_mask_ref');unknown_distance=None
    if unknown_ref:
     if sha(unknown_ref['file'])!=unknown_ref['sha256']:raise ValueError('Unknown mixed-material source mask changed')
     um=cv2.imread(unknown_ref['file'],0)
     if um is None or um.shape!=(1080,1920):raise ValueError('Invalid unknown source mask')
     unknown_distance=cv2.distanceTransform((um==0).astype('uint8'),cv2.DIST_L2,5)
    bf=Path(bc[p]['qualified_file']);bh=sha(bf)
    if bh!=bc[p]['qualified_file_sha256']:raise ValueError('Board frame changed')
    b=read(bf);anchored=Path(b['anchored_file']);ah=sha(anchored)
    if ah!=b['anchored_file_sha256']or cfg['board_generation']not in str(anchored):raise ValueError('Wrong final board source generation')
    if b['mask_sha256']!=oldmasksha:raise ValueError('Upstream board source mask differs')
    af=Path(ap[p]['file']);afh=sha(af)
    if afh!=ap[p]['sha256']:raise ValueError('Advert source frame changed')
    ad=read(af)
    if Path(ad['final_board_ref']['file']).resolve()!=anchored.resolve()or ad['final_board_ref']['sha256']!=ah:raise ValueError('Source consumers use different board generations')
    if ad['mask_ref']['sha256']!=mh:raise ValueError('Advert mask differs')
    for owner,j,key in(('board',b,'source_unknown_exclusion_mask_ref'),('advert',ad,'unknown_exclusion_ref')):
     actual=j.get(key);expected_unknown=upstream_mi.get('source_unknown_exclusion_mask_ref')if owner=='board'else unknown_ref
     if bool(actual)!=bool(expected_unknown)or(actual and(actual['sha256']!=expected_unknown['sha256']or Path(actual['file']).resolve()!=Path(expected_unknown['file']).resolve())):raise ValueError(owner+' actual unknown mask differs from its source index')
    for proof in ('static_motion_ref','static_qualification_ref','body_material_ref','graphics_ref','semantic_normal_ref'):
     if proof not in ad or sha(ad[proof]['file'])!=ad[proof]['sha256']:raise ValueError('Advert actual source qualification proof missing or changed '+proof)
    for ref in ad['adjacent_static_motion_refs']:
     if sha(ref['file'])!=ref['sha256']:raise ValueError('Advert adjacent exposure static proof changed')
    for j in(b,ad):
     if j['decoded_native_BGR_sha256']!=mi['source_BGR_sha256']:raise ValueError('Source BGR exposure differs')
    if ad['decoded_native_RGB_sha256']!=mi['source_RGB_sha256']:raise ValueError('Advert RGB exposure differs')
    raw=image(p)
    if hashlib.sha256(raw.tobytes()).hexdigest()!=mi['source_BGR_sha256']:raise ValueError('Domain source decode differs')
    graphics,border,domain=current_domains(raw,b)
    contact_source=[];contact_rejected=0
    for cr in b['rows']:
     if cr['feature_id']!='board.contact':continue
     cq=supports(cr,'board');inside=np.isfinite(cq).all(1)&(cq[:,0]>=0)&(cq[:,0]<=1919)&(cq[:,1]>=0)&(cq[:,1]<=1079);cxy=np.rint(cq[inside]).astype(int)
     ok=len(cxy)>0 and not(distance[cxy[:,1],cxy[:,0]]<=8).any()and not graphics[cxy[:,1],cxy[:,0]].any()
     if ok and unknown_distance is not None:ok=not(unknown_distance[cxy[:,1],cxy[:,0]]<=8).any()
     if ok:contact_source.append(cr['uv'])
     else:contact_rejected+=1
    if len(contact_source)<2:raise ValueError('Current actual source contact lacks unoccluded material support')
    domain_board=dict(b,roles=dict(b['roles']));domain_board['roles']['board.contact']=dict(b['roles']['board.contact'],trace_uv=contact_source)
    graphics,border,domain=current_domains(raw,domain_board)
    domain['current_contact_actor_or_unknown_or_graphics_rejected_source_samples']=contact_rejected;domain['current_contact_valid_source_samples']=len(contact_source);domain['contact_gap_connection_is_exclusion_only_not_a_source_point']=True
    gm=target/'graphics_masks'/f'{p}.png';cv2.imwrite(str(gm),graphics.astype('uint8')*255)
    surface=None;outside=None;ice_visible=None;body_visible=None;surface_ref=None
    if p in surface_index:
     si=surface_index[p];surface_ref=dict(file=si['frame_file'],sha256=si['frame_sha256']);surface=read(pin(surface_ref))
     if surface['native_pts']!=p or surface['source_BGR_sha256']!=mi['source_BGR_sha256']or surface['source_RGB_sha256']!=mi['source_RGB_sha256']:raise ValueError('Near-board domain belongs to different native pixels')
     outside=source_mask(surface['near_outside_exclusion_mask_ref']);ice_visible=source_mask(surface['ice_visible_mask_ref']);body_visible=source_mask(surface['board_body_visible_mask_ref'])
     domain['near_board_actual_source_qualification']=surface_ref;domain['near_board_state']=surface['state']
    cv=icecov[sid].frame(p)
    if len(cv)!=1 or cv[0][0]['native_RGB_sha256']!=mi['source_RGB_sha256']:raise ValueError('Ice RGB coverage differs')
    candidates=[];refs=[];parts=[];dropped=Counter();seen=set()
    def append(r,ref,part):
     if r['native_pts']!=p or r['shot_id']!=sid:raise ValueError('Source row clock differs')
     uv=np.asarray(r['uv'],float);reason=None;rid=r.get('source_row_id',r.get('point_id'))
     if part=='ice'and rid in exclusions:
      er,eref=exclusions[rid];applied_exclusions.add(rid);reason=er['reason'];ref=dict(ref,source_correction_ref=eref)
     if reason is None:
      if uv.shape!=(2,)or not np.isfinite(uv).all()or(uv<0).any()or(uv>[1919,1079]).any():reason='outside_native_pixel_domain'
      else:
       q=supports(r,part);inside=np.isfinite(q).all(1)&(q[:,0]>=0)&(q[:,0]<=1919)&(q[:,1]>=0)&(q[:,1]<=1079)
       if not inside.all()and part=='ice':reason='material_support_outside_native_exposure'
       else:
        xy=np.rint(q[inside]).astype(int)
        if len(xy)==0:reason='material_support_unavailable'
        elif graphics[xy[:,1],xy[:,0]].any():reason='actual_current_broadcast_graphics_occludes_complete_source_profile'
        elif(distance[xy[:,1],xy[:,0]]<=8).any():reason='current_actor_silhouette_and_blur_support_guard'
        elif unknown_distance is not None and(unknown_distance[xy[:,1],xy[:,0]]<=8).any():reason='current_source_reflection_or_mixed_material_identity_unknown'
        elif outside is not None and not ref.get('new_near_top_source_branch')and outside[xy[:,1],xy[:,0]].any():reason='actual_near_board_top_or_external_non_surface_domain'
        elif part=='ice'and ice_visible is not None and not ice_visible[xy[:,1],xy[:,0]].all():reason='outside_actual_visible_ice_surface'
        elif part=='print'and body_visible is not None and not body_visible[xy[:,1],xy[:,0]].all():reason='outside_actual_visible_board_advert_body'
     if reason is None:reason=rejection_reason(r,part,graphics,border)
     if reason is None and part=='ice'and r.get('source_kind')=='actual_current_native_RGB_single_blue_material_interface':reason='single_wide_band_side_cannot_certify_midpoint_retained_raw'
     if reason:
      dropped[part+'/'+reason]+=1;emit('excluded',dict(shot_id=sid,native_pts=p,reason=reason,uv=r['uv'],source_row_id=rid,source_kind=r.get('source_kind'),raw_ref=ref));return
     identity=r.get('point_id')or rid or part+'/'+str(len(candidates));key=(part,identity)
     if key in seen:dropped[part+'/same_source_ID_duplicate']+=1;return
     if part=='ice'and r.get('paired_edge_uv')is not None and 'midpoint'in r.get('definition',''):
      if np.linalg.norm(np.asarray(r['paired_edge_uv'],float).mean(0)-uv)>1e-4:raise ValueError('Wide paint midpoint differs from actual side edges')
     seen.add(key);candidates.append(r);refs.append(ref);parts.append(part)
    for i,r in enumerate(b['rows']):append(r,dict(file=str(bf),json_pointer='/rows/'+str(i),upstream_actor_mask_sha256=oldmasksha,current_visibility_requalified=True),'board')
    if surface:
     for i,r in enumerate(surface['new_source_rows']):
      if r['feature_id']!='board.top':raise ValueError('Near source branch must preserve shared upper-boundary identity')
      append(r,dict(surface_ref,json_pointer='/new_source_rows/'+str(i),new_near_top_source_branch=True,normal_only=True,texture_2D_tracking=False),'board')
    for cursor in ice[sid]:
     for r,n in cursor.frame(p):append(r,dict(file=str(cursor.file),jsonl_line_0based=n),'ice')
    ads=[]
    for m in ad['source_members']:
     path=str(Path(m['file']))
     if path not in cache:cache[path]=open(path,'rb')
     h=cache[path];h.seek(m['offset_bytes']);blob=h.read(m['length_bytes'])
     if hashlib.sha256(blob).hexdigest()!=m['sha256']:raise ValueError('Advert gzip source member changed')
     rr=[json.loads(l)for l in gzip.decompress(blob).splitlines()if l.strip()]
     if len(rr)!=m['row_count']:raise ValueError('Advert source member row count differs')
     ads.extend((r,dict(file=path,gzip_member_offset=m['offset_bytes'],gzip_member_bytes=m['length_bytes'],gzip_member_sha256=m['sha256'],member_jsonl_line_0based=i))for i,r in enumerate(rr))
    eligible=base64.b64decode(ad['eligible_uint8_base64']);status=base64.b64decode(ad['status_uint8_base64'])
    if len(eligible)!=len(ads)or len(status)!=len(ads)or sum(eligible)!=ad['eligible_rows']:raise ValueError('Advert qualification input identity differs')
    adtypes={k:base64.b64decode(ad[k+'_uint8_base64'])for k in('normal_only','texture_2D_tracking','observation_type')}
    if any(len(v)!=len(ads)for v in adtypes.values()):raise ValueError('Advert constraint array identity differs')
    for i,(r,ref)in enumerate(ads):
     if eligible[i]:
      r=dict(r,_root_qualification_status=int(status[i]));append(r,dict(ref,qualification_file=str(af),qualification_row_0based=i,qualification_status_code=int(status[i]),qualification_status_definition=ad.get('status_code_definition',{}).get(str(status[i])),normal_only=bool(adtypes['normal_only'][i]),texture_2D_tracking=bool(adtypes['texture_2D_tracking'][i]),observation_type_code=int(adtypes['observation_type'][i]),observation_type_definition=ad['observation_type_definition'][str(adtypes['observation_type'][i])]),'print')
     else:dropped['print/qualification_status_'+str(status[i])]+=1
    selection_candidates=[dict(r,feature_id=r['feature_id']+'#near_lower_long_side')if ref.get('new_near_top_source_branch')else r for r,ref in zip(candidates,refs)]
    indices,selection=select_source_rows(selection_candidates,previous_point_ids=previous)
    required_branches={r['feature_id']for r in selection_candidates if r['feature_id'].startswith(('board.','ice.','unbound.ice.','Ad'))}
    represented_branches={selection_candidates[i]['feature_id']for i in indices}
    if required_branches-represented_branches:raise ValueError('Visible source branch omitted by selection')
    required={r['feature_id']for r in candidates if r['feature_id'].startswith(('board.','ice.','unbound.ice.','Ad'))};represented={candidates[i]['feature_id']for i in indices}
    if required-represented:raise ValueError('Visible named source entity omitted by selection')
    points=[];frameby=Counter();framekinds=Counter();framefeatures=Counter();ids=[]
    for rank,i in enumerate(indices):
     r=candidates[i];part=parts[i];fid=r['feature_id'];kind=r.get('source_kind','unknown');cls=display(r,part);rid=f'{cfg["generation_id"]}/{sid}/{p}/{rank}'
     point=dict(selected_row_id=rid,shot_id=sid,native_pts=p,time_base='1/12800',feature_id=fid,source_branch_id='near_lower_long_side'if refs[i].get('new_near_top_source_branch')else r.get('source_branch_id',r.get('branch_id')),uv=r['uv'],display_class=cls,source_kind=kind,measurement_definition=refs[i]['observation_type_definition']if part=='print'else r.get('definition',bm.get('definition',{}).get(fid,r.get('measurement_operator','retained_parent_material_definition'))),source_qualification_status_code=refs[i].get('qualification_status_code'),source_qualification_status_definition=refs[i].get('qualification_status_definition'),normal_only=refs[i].get('normal_only'),texture_2D_tracking=refs[i].get('texture_2D_tracking'),observation_type_code=refs[i].get('observation_type_code'),observation_type_definition=refs[i].get('observation_type_definition'),physical_surface='ice'if part=='ice'else'board',user_confirmed=r.get('user_confirmed',False),source_UV_is_direct_measurement=not infer(r),source_pixel_or_annotation_ref=refs[i],source_point_id=r.get('point_id',r.get('source_row_id')),independent_fit_weight_issued=False,physical_world_correspondence_issued=False)
     emit('selected_rows',point);points.append({k:point[k]for k in('selected_row_id','uv','display_class')});ids.append(point['source_point_id']);frameby[part]+=1;framekinds[kind]+=1;framefeatures[fid]+=1;classes[cls]+=1;manual+=cls=='manual'
     if fid.startswith('unbound.')or'unbound'in r.get('crosswalk_status','')or r.get('world_material_crosswalk')=='unknown_not_used':unknown[part]+=1
    emit('selected_frames',dict(shot_id=sid,native_pts=p,decoded_native_BGR_sha256=mi['source_BGR_sha256'],points=points))
    emit('coverage',dict(shot_id=sid,native_pts=p,candidate_rows_by_part=dict(Counter(parts)),selected_rows_by_part=dict(frameby),selected_feature_counts=dict(framefeatures),selected_source_kind_counts=dict(framekinds),rejected_rows=dict(dropped),selection=selection,qualified_board_rows_by_source_branch=dict(Counter(('near_lower_long_side'if ref.get('new_near_top_source_branch')else'upstream_actual_material')for part,ref in zip(parts,refs)if part=='board')),upstream_raw_board_spans={k:{q:v[q]for q in('spans','visible_columns','occluded_columns','blurred_columns')if q in v}for k,v in b['roles'].items()},source_domain_qualification=domain,actor_visibility_guard_native_px=8,guard_is_engineering_visibility_not_sigma=True,ice_source_coverage_ref=dict(file=str(icecov[sid].file),jsonl_line_0based=cv[0][1]),all_visible_features_perfect_claimed=False))
    emit('consumer_bindings',dict(shot_id=sid,native_pts=p,source_BGR_sha256=mi['source_BGR_sha256'],source_RGB_sha256=mi['source_RGB_sha256'],mask=dict(file=str(mp),sha256=mh),upstream_board_actor_mask=dict(file=str(oldmaskpath),sha256=oldmasksha),actual_surface_override=surface_ref,source_unknown_exclusion_mask_ref=unknown_ref,board=dict(file=str(bf),sha256=bh),anchored_board=dict(file=str(anchored),sha256=ah),advert=dict(file=str(af),sha256=afh),graphics_mask=dict(file=str(gm),sha256=sha(gm)),advert_actual_source_proofs={k:ad[k]for k in('static_motion_ref','static_qualification_ref','body_material_ref','graphics_ref','semantic_normal_ref','adjacent_static_motion_refs')},source_domain_recipe_sha256=pins[str(O/'source_surface_qualification.py')]))
    tot.update(frameby);byshot[sid].update(frameby);features[sid].update(framefeatures);kinds.update(framekinds);rejects.update(dropped);previous=set(x for x in ids if x);framecount+=1;selectedcount+=len(indices);counts.append(len(indices))
    if framecount%25==0:write(target/'progress.json',dict(frames=framecount,total=len(EXPECTED),native_pts=p,selected_rows=selectedcount,elapsed_seconds=time.perf_counter()-start))
   for c in ice[sid]:c.close()
   icecov[sid].close()
 finally:
  for f in outputs.values():f.close()
  for f in cache.values():f.close()
  original.close()
 if framecount!=len(EXPECTED)or sum(counts)!=selectedcount:raise ValueError('Incomplete native source generation')
 if set(exclusions)-applied_exclusions:raise ValueError('Correction IDs do not bind actual source rows')
 for p,h in pins.items():
  if sha(p)!=h:raise ValueError('Source input changed during assembly '+p)
 files={p.name:dict(file=str(p),sha256=sha(p))for p in target.glob('*.gz')}
 m=dict(schema='rinkspace.source_only_dense56_generation.v2',generation_id=cfg['generation_id'],native_intervals=[dict(shot_id=s,first_pts=a,last_pts=b)for s,a,b in SPECS],complete=True,shot_ids=[s for s,a,b in SPECS],complete_s05_s06_scope=len(EXPECTED)==1862,frames=framecount,point_rows=selectedcount,point_count_per_native_frame=dict(min=min(counts),max=max(counts)),by_part=dict(tot),by_shot={s:dict(v)for s,v in byshot.items()},features_by_shot={s:dict(v)for s,v in features.items()},source_kind_counts=dict(kinds),display_classes=dict(classes),manual_source_rows=manual,rejected_source_rows_by_actual_reason=dict(rejects),unknown_world_or_named_crosswalk_rows=dict(unknown),parent_pins=pins,source_correction_rows_applied=len(applied_exclusions),actual_source_requalification_counts=dict(requalification_stats),files=files,source_UV_moved=False,old_generations_unchanged=True,physical_solver_or_live_packet_changed=False,selected_rows_are_independent_measurements=False,all_visible_features_perfect_claimed=False,elapsed_seconds=time.perf_counter()-start)
 write(target/'manifest.json',m)
 video=dict(purpose='user_requested_source_points_review',source=source,selected_frame_file=files['selected_frames.jsonl.gz'],shots=[dict(id=s,first_pts=a,last_pts=b)for s,a,b in SPECS]);write(target/'video_config.json',video)
 print(json.dumps(dict(complete=True,frames=framecount,rows=selectedcount,by_shot=m['by_shot'],elapsed_seconds=m['elapsed_seconds']),ensure_ascii=False))
if __name__=='__main__':run(sys.argv[1])
