import json
from pathlib import Path
BINDING_FILE=Path(r'D:/CodexData/RinkSpace_Data/runs/okada_rec_1019/recover_v2/dense56_ice_v1/final_board_binding.json')
BINDING=json.load(open(BINDING_FILE,encoding='utf8'))
assert BINDING.get('accepted_for_source_overlay')is True,'Final board source generation has not passed review'
BOARD_NAME=BINDING['board_generation'];FINAL_BOARD_MANIFEST=Path(BINDING['board_manifest_file'])
exec(open(r'D:/CodexData/RinkSpace_Data/runs/okada_rec_1019/recover_v2/dense56_ice_v1/reader56_exact.py',encoding='utf8').read().split('def main():')[0].replace('shape_v3',BOARD_NAME).replace('SAM2_8px','SAM2_footwear_composite_8px').replace('qualified_v3_boardfoot','qualified_'+BOARD_NAME+'_boardfoot').replace("mp=R/'dense56_masks_v1/masks'/f'{pts}.png';","mp=Path(mask_index[pts]['mask_file']);").replace("old=cv2.dilate((raw>0).astype('u1'),np.ones((11,11),'u1'))>0", "old_raw=cv2.imread(str(mask_index[pts].get('parent_mask_file',mp)),0);old=cv2.dilate((old_raw>0).astype('u1'),np.ones((11,11),'u1'))>0"))
FINAL_PERSON_INDEX=Path(BINDING['final_person_mask_index_file'])
assert TH(FINAL_PERSON_INDEX.read_bytes()).hexdigest()==BINDING['final_person_mask_index_sha256'],'Composite person mask index changed'
mask_index={json.loads(l)['native_pts']:json.loads(l)for l in gzip.open(FINAL_PERSON_INDEX,'rt',encoding='utf8')}
FINAL_PERSON_MANIFEST=Path(BINDING['final_person_mask_manifest_file'])
assert TH(FINAL_PERSON_MANIFEST.read_bytes()).hexdigest()==BINDING['final_person_mask_manifest_sha256']
assert json.load(open(FINAL_PERSON_MANIFEST,encoding='utf8'))['accepted_for_final_source_qualification']is True

BASE_MASK_FRAME=mask_frame
LAST_PERSON_MASK_GUARD=None
LAST_SOURCE_UNKNOWN_GUARD=None
LAST_ORIGINAL_PARENT_MASK_GUARD=None
def mask_frame(rgb,pts,sid):
 global LAST_PERSON_MASK_GUARD,LAST_SOURCE_UNKNOWN_GUARD,LAST_ORIGINAL_PARENT_MASK_GUARD
 mask,info=BASE_MASK_FRAME(rgb,pts,sid);mi=mask_index[pts];original=cv2.imread(mi['parent_mask_file'],0)>0;LAST_ORIGINAL_PARENT_MASK_GUARD=cv2.dilate(original.astype('u1'),np.ones((17,17),'u1'))>0;person=cv2.imread(mi['mask_file'],0)>0;LAST_PERSON_MASK_GUARD=cv2.dilate(person.astype('u1'),np.ones((17,17),'u1'))>0;ref=mi.get('source_unknown_exclusion_mask_ref');unknown=np.zeros_like(mask)
 if ref:
  uf=Path(ref['file']);assert TH(uf.read_bytes()).hexdigest()==ref['sha256'];assert ref['source_RGB_sha256']==TH(rgb.tobytes()).hexdigest();unknown=cv2.imread(str(uf),0)>0;unknown=cv2.dilate(unknown.astype('u1'),np.ones((17,17),'u1'))>0;mask|=unknown
 LAST_SOURCE_UNKNOWN_GUARD=unknown;info.update(person_mask_semantics=mi['person_mask_semantics'],clean_person_base_mask_file=mi['clean_person_base_mask_file'],clean_person_base_mask_sha256=mi['clean_person_base_mask_sha256'],original_comparison_parent_mask_file=mi['parent_mask_file'],person_composite_index_file=str(FINAL_PERSON_INDEX),person_composite_index_sha256=BINDING['final_person_mask_index_sha256'],source_unknown_exclusion_mask_ref=ref,source_unknown_guard_pixels=8,person_and_source_unknown_masks_separately_typed=True,typed_unknown_reason_clarification_file=str(R/'dense56_footwear_v1/typed_reason_clarification.json'),source_unknown_semantics='actual current reflection or motion-blurred mixed static material unknown, never true person pixels');info['source_domain']+='_and_separately_typed_current_source_unknown_guard'
 return mask,info


import heapq,itertools

def sha(path):
 h=TH()
 with open(path,'rb')as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
def read_lines(path,lo=None,hi=None):
 try:
  with gzip.open(path,'rt',encoding='utf8')as f:
   for l in f:
    try:r=json.loads(l)
    except json.JSONDecodeError:break
    pts=r['native_pts']
    if hi is not None and pts>hi:break
    if lo is not None and pts<lo:continue
    yield r
 except EOFError:pass

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--shot',required=True);args=ap.parse_args();sid=args.shot;a,b=scopes[sid]
 assert sha(FINAL_BOARD_MANIFEST)==BINDING['board_manifest_sha256'],'Final board manifest changed after acceptance'
 parts=[]
 if sid=='s05':parts.append(dict(id='exact_prefix',path=O/'exact_domain_partial/s05',lo=948992,hi=1039360,kind='base'))
 parts.append(dict(id='exact_tail',path=O/'exact_domain_tail'/sid,lo=1039616 if sid=='s05'else a,hi=b,kind='base'))
 if sid=='s05':parts.append(dict(id='static_goal',path=O/'static_goal_exact',lo=a,hi=b,kind='supplement'))
 for spec in BINDING['delta_sources']:
  p=dict(spec);p['path']=Path(p['path']);p['lo']=max(a,int(p['lo']));p['hi']=min(b,int(p['hi']));p['kind']='delta';parts.append(p)
 parts=[p for p in parts if p['hi']>=p['lo']]
 basecov={};deltacov={};suppcov={};replacement_owner={}
 for p in parts:
  target=basecov if p['kind']=='base'else deltacov if p['kind']=='delta'else suppcov
  for c in read_lines(p['path']/'coverage.jsonl.gz',p['lo'],p['hi']):
   if c['shot_id']!=sid:continue
   target[c['native_pts']]=c
   if p['kind']=='delta':
    for fe in c.get('replaces_complete_feature_ids',[]):replacement_owner[(c['native_pts'],fe)]=p['id']
 assert sorted(basecov)==list(range(a,b+1,256)),(sid,len(basecov))
 def tagged(p):
  for r in read_lines(p['path']/'rows.jsonl.gz',p['lo'],p['hi']):
   if r['shot_id']!=sid:continue
   r['_input_part']=p['id'];r['_input_kind']=p['kind'];r['_control_file']=str(p['path']/'source_band_controls.jsonl.gz');yield r
 stream=heapq.merge(*[tagged(p)for p in parts],key=lambda r:r['native_pts']);groups=itertools.groupby(stream,key=lambda r:r['native_pts']);current=next(groups,None);out=O/'canonical'/sid;out.mkdir(parents=True,exist_ok=True);counts=collections.Counter();features=collections.Counter();omissions=collections.Counter();n=0;total=0;recovered_center=0;recovered_support=0;v4_new=0;histfeatures=collections.Counter();sourceeffects=collections.Counter();clean_recovered=0;start=time.monotonic();seen=[];minpts=10**9;maxpts=0
 with gzip.open(out/'rows.jsonl.gz','wt',encoding='utf8',compresslevel=3)as wr,gzip.open(out/'coverage.jsonl.gz','wt',encoding='utf8',compresslevel=3)as wc,gzip.open(out/'exclusions.jsonl.gz','wt',encoding='utf8',compresslevel=3)as we,av.open(str(D/'events/ankara_2026/programmes/okada_mei_sp/Mei OKADA (JPN) ｜ Junior Women Short Program ｜ Ankara 2026.mp4'))as con:
  st=con.streams.video[0];con.seek(a,stream=st,backward=True)
  for fr in con.decode(st):
   pts=int(fr.pts)
   if pts>b:break
   if pts<a:continue
   rgb=fr.to_ndarray(format='rgb24');digest=TH(rgb.tobytes()).hexdigest();assert digest==basecov[pts]['native_RGB_sha256'];mask,ginfo=mask_frame(rgb,pts,sid);raw=[]
   if current and current[0]==pts:raw=list(current[1]);current=next(groups,None)
   dc=deltacov.get(pts);replace={fe for(pp,fe)in replacement_owner if pp==pts};sc=suppcov.get(pts);red_count=sum(r['feature_id']=='unbound.ice.flat_red_static_spot'and(r['_input_part']==replacement_owner.get((pts,r['feature_id']),r['_input_part']))for r in raw);accepted=[];fc=collections.Counter();kc=collections.Counter();rc=collections.Counter();opened_uv=0;opened_support=0;newfoot=0;hfc=collections.Counter();input_counts=collections.Counter();frame_clean_recovered=0
   for r in raw:
    part=r.pop('_input_part');kind=r.pop('_input_kind');ctrl=r.pop('_control_file');reason=None
    if (pts,r['feature_id'])in replacement_owner and part!=replacement_owner[(pts,r['feature_id'])]:reason='same_current_frame_feature_replaced_by_complete_final_board_domain_RGB_reread'
    ps=r.get('pixel_support',{})
    if'side_interface_quartiles_uv'not in ps and'signed_interface_quartiles_uv'in ps:ps['side_interface_quartiles_uv']=[ps['signed_interface_quartiles_uv']]
    if'normal'not in ps and'profile_normal'in ps:ps['normal']=ps['profile_normal']
    u=np.asarray(r['uv'],float);edges=np.asarray(r['paired_edge_uv'],float).reshape(-1,2)if r.get('paired_edge_uv')is not None else np.empty((0,2));qs=np.asarray(ps.get('side_interface_quartiles_uv',[]),float).reshape(-1,2);normal=np.asarray(ps.get('normal',[0,1]),float)
    if reason is None and(not np.isfinite(u).all()or not context_clear(mask,u,edges,normal,qs)):
     if np.isfinite(u).all()and not context_clear(LAST_PERSON_MASK_GUARD,u,edges,normal,qs):reason='current_actual_person_or_qualified_true_footwear_support_occluded_8px_guard'
     elif np.isfinite(u).all()and not context_clear(LAST_SOURCE_UNKNOWN_GUARD,u,edges,normal,qs):reason='current_source_person_reflection_or_motion_blurred_mixed_material_unknown_support_8px_guard'
     else:reason='final_actual_'+BOARD_NAME+'_ice_surface_graphics_reflection_or_full_material_support_exclusion'
    if reason is None and r['feature_id']=='unbound.ice.flat_red_static_spot'and red_count<4:reason='unknown_current_red_entry_exit_fragment_signed_interface_material_identity_not_resolved'
    if edges.size:assert len(edges)==2 and np.linalg.norm(u-edges.mean(0))<1e-6
    if reason:
     we.write(json.dumps(dict(shot_id=sid,native_pts=pts,source_row_id=r['source_row_id'],input_part=part,qualification='unknown'if reason.startswith('unknown')else'exclude',reason=reason),separators=(',',':'))+'\n');rc[reason]+=1;continue
    support=np.vstack([u[None],edges,qs]);xy=np.rint(support).astype(int);newcenter=bool(LAST_OLD_MASK[int(round(u[1])),int(round(u[0]))]);newsupport=bool(LAST_OLD_MASK[xy[:,1],xy[:,0]].any());ps['current_UV_in_newly_opened_source_domain']=newcenter;ps['old_conservative_domain_would_exclude_material_support']=newsupport;r['source_UV_is_direct_measurement']=True;r['source_input_part']=part;r['source_parent_generation']=r.get('canonical_generation');r['canonical_generation']='dense56_ice_source_sealed_'+BOARD_NAME;r['source_surface_qualification']='current_static_ice_material_with_final_actual_surface_gate';r['occlusion']=ginfo['source_domain'];r['source_uv_from_model']=False
    if r.get('navigation',{}).get('whole_current_source_band'):r['navigation']['source_file']=ctrl
    if r['feature_id']=='unbound.ice.flat_red_static_spot':r['source_material_qualification']=dict(status='supported',basis='prior confirmed native red spot ROIs plus current signed material interface and dense complete-sequence visual review',exact_world_identity='unbound_not_for_physical_fit')
    if r.get('navigation',{}).get('registered_historical_source_guidance'):hfc[r['feature_id']]+=1
    if r['pixel_support'].get('newly_opened_by_final_boardfoot_support'):newfoot+=1
    oldparent=not context_clear(LAST_ORIGINAL_PARENT_MASK_GUARD,u,edges,normal,qs);ps['current_support_recovered_from_original_parent_scene_mask']=oldparent;ps['actual_RGB_reread_in_opened_clean52_person_mask_domain']=bool(oldparent and pts in BINDING['source_domain_force_reread_native_pts']and part=='accepted_source_domain_material_v5');frame_clean_recovered+=int(ps['actual_RGB_reread_in_opened_clean52_person_mask_domain']);sourceeffects[part+'/final_rows']+=1;sourceeffects[part+'/original_parent_mask_recovered_support']+=int(oldparent);sourceeffects[part+'/original_conservative_recovered_center']+=int(newcenter);opened_uv+=int(newcenter);opened_support+=int(newsupport);fc[r['feature_id']]+=1;kc[r['source_kind']]+=1;input_counts[part]+=1;accepted.append(r)
   for r in accepted:wr.write(json.dumps(r,separators=(',',':'))+'\n')
   cover=dict(shot_id=sid,native_pts=pts,time_base='1/12800',point_candidates=len(accepted),feature_counts=dict(fc),source_kind_counts=dict(kc),input_part_counts=dict(input_counts),omissions=dict(rc),native_RGB_sha256=digest,final_domain=BOARD_NAME,newly_opened_center_UV_count=opened_uv,newly_opened_material_support_rows=opened_support,final_board_opened_domain_actual_reread_support_rows=newfoot,clean52_opened_domain_actual_RGB_reread_support_rows=frame_clean_recovered,registered_user_guidance_actual_RGB_rows=dict(hfc),source_kind='all_current_native_RGB_independent_of_model_UV',parent_source_UV_or_packets_unchanged=True,all_visible_material_complete_claimed=False,**ginfo);wc.write(json.dumps(cover,separators=(',',':'))+'\n');n+=1;seen.append(pts);total+=len(accepted);minpts=min(minpts,len(accepted));maxpts=max(maxpts,len(accepted));counts.update(kc);features.update(fc);omissions.update(rc);recovered_center+=opened_uv;recovered_support+=opened_support;v4_new+=newfoot;clean_recovered+=frame_clean_recovered;histfeatures.update(hfc)
   if n%100==0:wr.flush();wc.flush();we.flush();info=dict(shot_id=sid,frames=n,rows=total,native_pts=pts,recovered_original_domain_UV=recovered_center,final_board_new_domain_support=v4_new,elapsed_seconds=time.monotonic()-start);print(json.dumps(info),flush=True);(out/'sealed_progress.json').write_text(json.dumps(info),encoding='utf8')
 assert seen==list(range(a,b+1,256))and current is None
 # Per-shot data may be finalized after its last native delta row; publish only once every source stream is sealed.
 for p in parts:
  if p['id']=='accepted_source_domain_material_v5':
   mf=p['path']/'manifest.json'
   while not mf.exists():time.sleep(1)
   assert json.load(open(mf,encoding='utf8'))['complete']is True
 refs=[]
 for p in parts:
  ref={k:str(v)if isinstance(v,Path)else v for k,v in p.items()}
  for name in ['rows.jsonl.gz','coverage.jsonl.gz']:
   f=p['path']/name;ref[name]=dict(file=str(f),sha256=sha(f))
  cf=p['path']/'source_band_controls.jsonl.gz'
  if cf.exists():ref['source_band_controls']=dict(file=str(cf),sha256=sha(cf))
  refs.append(ref)
 manifest=dict(schema='rinkspace.dense56_ice_final_sealed_source.v2',complete=True,sealed=True,shot_id=sid,frame_count=n,native_first_pts=a,native_last_pts=b,rows=total,frame_point_min=minpts,frame_point_max=maxpts,source_video_sha256=source_sha,time_base='1/12800',native_step=256,rows_file=str(out/'rows.jsonl.gz'),coverage_file=str(out/'coverage.jsonl.gz'),exclusions_file=str(out/'exclusions.jsonl.gz'),feature_counts=dict(features),source_kind_counts=dict(counts),omission_counts=dict(omissions),newly_opened_actual_UV_count=recovered_center,newly_opened_actual_material_support_rows=recovered_support,final_board_opened_domain_actual_reread_support_rows=v4_new,registered_historical_guidance_actual_current_RGB_rows=dict(histfeatures),registered_historical_guidance_file=str(HIST_FILE),registered_historical_guidance_sha256=HIST_SHA,registered_historical_guides=[{k:v for k,v in e.items()if k!='base'}for e in HIST_ENTRIES],final_person_mask_index=dict(file=str(FINAL_PERSON_INDEX),sha256=sha(FINAL_PERSON_INDEX)),final_person_mask_manifest=dict(file=str(FINAL_PERSON_MANIFEST),sha256=sha(FINAL_PERSON_MANIFEST)),final_board_generation=BOARD_NAME,final_board_manifest=dict(file=str(FINAL_BOARD_MANIFEST),sha256=sha(FINAL_BOARD_MANIFEST)),final_board_binding=dict(file=str(BINDING_FILE),sha256=sha(BINDING_FILE)),final_board_eligibility_applied_to_every_source_support=True,all_native_RGB_masks_and_board_hashes_match=True,all_width_midpoints_actual_two_edges_arithmetic_checked=True,all_source_rows_are_direct_current_pixel_supported=True,source_curve_controls_are_search_only=True,all_unknown_red_fragments_excluded_from_formal_source_rows=True,raw_source_parts=refs,parent_models_packets_historical_UV_or_source_images_modified=False,physical_solver_started=False,independent_noise_W_or_metric_truth_issued=False,all_visible_material_complete_claimed=False)
 for k in ['rows_file','coverage_file','exclusions_file']:manifest[k.replace('_file','_sha256')]=sha(manifest[k])
 navigation=dict(existing_05_to_1s_keyframes=dict(count=len(source_refs),use='existing source-native saved UV/geometry for curve identity and search navigation only; every emitted UV reread from current official RGB',files=[dict(file=f,sha256=sha(f),native_pts=int(json.load(open(f,encoding='utf8'))['source_pts']))for f in source_refs]),registered_user_explanation=dict(file=str(HIST_FILE),sha256=HIST_SHA,official_registered_anchor_pts=[1071616,1332736],use='four actual image-registered user traces installed in native reader search windows; not new manual clicks or copied source UV',actual_RGB_feature_rows=dict(histfeatures)),compact_8a_navigation=dict(manifest=dict(file=str(NAV/'manifest.json'),sha256=sha(NAV/'manifest.json')),crosswalk=dict(file=str(NAV/'crosswalk.json'),sha256=sha(NAV/'crosswalk.json')),arrays=[dict(file=str(NAV/f),sha256=sha(NAV/f),use='mmap native search navigation only')for f in ['frame_native_pts.npy','frame_offsets.npy','target_uv.bin','crosswalk_index.bin']],demo_scene_entity_inventory=dict(file=cw['old_scene']['file'],sha256=sha(cw['old_scene']['file']),use='entity/ROI identity and operator navigation; not emitted observation source'),two_templates_kept_distinct_by_crosswalk=True,no_large_navigation_cache_rebuilt=True))
 manifest['navigation_inputs_actually_consumed']=navigation
 manifest['actual_source_effect_counts_by_input_part']=dict(sourceeffects)
 manifest['clean52_opened_domain_actual_RGB_reread_support_rows']=clean_recovered
 manifest['final_source_reader_files']=[dict(file=str(O/f),sha256=sha(O/f))for f in ['reader56_exact.py','read_accepted_board_delta.py','finalize_sealed56.py']]
 manifest['old_conservative_comparison_recipe']='original parent scene mask + old fixed HUD/timer rectangles + old full-person-xbox-below exclusion + v2 board foot if available else current foot proxy; current formal eligibility uses clean actual person and separately unknown plus actual current graphics'
 clar=R/'dense56_footwear_v1/typed_reason_clarification.json'
 manifest['typed_unknown_reason_clarification']=dict(file=str(clar),sha256=sha(clar))
 (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(dict(sealed=True,shot_id=sid,frames=n,rows=total)),flush=True)
if __name__=='__main__':main()
