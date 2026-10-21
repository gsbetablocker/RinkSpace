import json,gzip,hashlib,base64,struct,collections,time
from pathlib import Path
O=Path(r'D:/CodexData/RinkSpace_Data/runs/okada_rec_1019/recover_v2/dense56_print_v1');R=O.parent;F=O/'final_frames_v5';expected=list(range(948992,1425408+1,256))
def sha(b):return hashlib.sha256(b).hexdigest()
def load(p):return json.loads(Path(p).read_text(encoding='utf8'))
def ref(p):return dict(file=str(p),sha256=sha(Path(p).read_bytes()))
def readrows(j):
 rows=[]
 for m in j['source_members']:
  with open(m['file'],'rb')as f:f.seek(m['offset_bytes']);b=f.read(m['length_bytes'])
  assert sha(b)==m['sha256'];rr=[json.loads(s)for s in gzip.decompress(b).decode('utf8').splitlines()];assert len(rr)==m['row_count'];rows.extend(rr)
 return rows
assert sorted(int(p.stem)for p in F.glob('*.json'))==expected
bindings=load(O/'final_bindings_v5.json');assert bindings['board_audit_released']is True and bindings['composite_mask_released']is True
for k in ('board_release_ref','composite_mask_manifest_ref'):
 rr=bindings[k];assert sha(Path(rr['file']).read_bytes())==rr['sha256']
maskmanifest=load(bindings['composite_mask_manifest_ref']['file']);assert maskmanifest['accepted_for_final_source_qualification']is True;assert sha(Path(bindings['composite_mask_index']).read_bytes())==maskmanifest['index_sha256']
maskidx={}
with gzip.open(Path(bindings['composite_mask_index']),'rt',encoding='utf8')as f:
 for s in f:j=json.loads(s);maskidx[j['native_pts']]=j
idx=[];per=collections.defaultdict(lambda:dict(frames=0,eligible_rows=0,source_rows=0,minimum_eligible=10**9,unknown=0,motion_conflicts=0,surface_excluded=0));unique={};boardshas=[]
for pts in expected:
 fp=F/f'{pts}.json';j=load(fp);raw=fp.read_bytes();sid='s05'if pts<=1300992 else's06';assert j['shot_id']==sid and j['native_pts']==pts and j['time_base']=='1/12800';assert Path(j['final_board_ref']['file']).parent==Path(bindings['board_directory'])
 for k in ('final_board_ref','mask_ref','static_motion_ref','body_material_ref','graphics_ref','semantic_normal_ref','composite_mask_index_ref','static_qualification_ref'):
  assert sha(Path(j[k]['file']).read_bytes())==j[k]['sha256']
 for rr in j['adjacent_static_motion_refs']:assert sha(Path(rr['file']).read_bytes())==rr['sha256']
 ur=j.get('unknown_exclusion_ref');assert ur==maskidx[pts].get('source_unknown_exclusion_mask_ref')
 if ur is not None:assert sha(Path(ur['file']).read_bytes())==ur['sha256']
 assert maskidx[pts]['source_BGR_sha256']==j['decoded_native_BGR_sha256'];assert maskidx[pts]['source_RGB_sha256']==j['decoded_native_RGB_sha256'];assert maskidx[pts]['mask_sha256']==j['mask_ref']['sha256']
 el=base64.b64decode(j['eligible_uint8_base64']);status=base64.b64decode(j['status_uint8_base64']);assert len(el)==len(status)==j['source_row_count'];assert sum(el)==j['eligible_rows'];assert all(x in (0,1)for x in el);assert sum(m['row_count']for m in j['source_members'])==len(el)
 normal=base64.b64decode(j['normal_only_uint8_base64']);texture=base64.b64decode(j['texture_2D_tracking_uint8_base64']);types=base64.b64decode(j['observation_type_uint8_base64']);assert len(normal)==len(texture)==len(types)==len(el);assert all(bool(e)==(s in (1,2,3,7))for e,s in zip(el,status));assert all(not(n and t)for n,t in zip(normal,texture));assert all((s!=7)or(n==1 and t==0)for s,n,t in zip(status,normal,texture))
 for k in ('body_material_ref','graphics_ref','semantic_normal_ref'):
  detail=load(Path(j[k]['file']));nested=detail['body_material_mask_ref']if k=='body_material_ref'else detail['mask_ref'];assert sha(Path(nested['file']).read_bytes())==nested['sha256']
 for m in j['source_members']:
  with open(m['file'],'rb')as f:f.seek(m['offset_bytes']);member=f.read(m['length_bytes'])
  assert len(member)==m['length_bytes']and sha(member)==m['sha256'];unique[m['file']]=None
 pp=per[sid];pp['frames']+=1;pp['eligible_rows']+=j['eligible_rows'];pp['source_rows']+=j['source_row_count'];pp['minimum_eligible']=min(pp['minimum_eligible'],j['eligible_rows']);ct=collections.Counter(status);pp['unknown']+=ct[0]+ct[5];pp['motion_conflicts']+=ct[4];pp['surface_excluded']+=ct[6];pp['normal_only_rows']=pp.get('normal_only_rows',0)+sum(normal);pp['texture_2D_tracking_rows']=pp.get('texture_2D_tracking_rows',0)+sum(texture);pp['semantic_normal_only_rows']=pp.get('semantic_normal_only_rows',0)+ct[7]
 idx.append(dict(native_pts=pts,shot_id=sid,file=str(fp),sha256=sha(raw),source_row_count=j['source_row_count'],eligible_rows=j['eligible_rows'],decoded_native_BGR_sha256=j['decoded_native_BGR_sha256'],decoded_native_RGB_sha256=j['decoded_native_RGB_sha256'],final_board_ref=j['final_board_ref'],mask_ref=j['mask_ref'],unknown_exclusion_ref=ur,body_material_ref=j['body_material_ref'],static_motion_ref=j['static_motion_ref'],source_parts_binding_sha256=j['source_parts_binding_sha256']))
file=O/'final_index.jsonl.gz'
with gzip.open(file,'wt',encoding='utf8',compresslevel=6)as f:
 for j in idx:f.write(json.dumps(j,separators=(',',':'))+'\n')
# All original source_native_readout_uv remain bitwise exact, including excluded candidates.
inv=load(O/'original_source_inventory.json');preserved=0;eligible_original=0;manual=0
for entry in inv['frames']:
 pts=entry['native_pts'];j=load(F/f'{pts}.json');rows=readrows(j);el=base64.b64decode(j['eligible_uint8_base64']);lookup={r.get('seed_source_pointer'):(r,el[i])for i,r in enumerate(rows)if r['source_kind']=='preserved_original_native_advertisement_readout'}
 origin=load(Path(entry['file']));assert sha(Path(entry['file']).read_bytes())==entry['sha256']
 for fi,feature in enumerate(origin.get('features',[])):
  if not feature.get('id','').startswith('Ad'):continue
  for si,s in enumerate(feature.get('segments',[])):
   xy=s.get('source_native_readout_uv');key='source_native_readout_uv'
   if xy is None and s.get('pixels_measured_in_this_frame'):xy=s.get('uv',[]);key='uv'
   for vi,uv in enumerate(xy or []):
    pointer=f'/features/{fi}/segments/{si}/{key}/{vi}';a,e=lookup[pointer];assert a['feature_id']==feature['id'];assert struct.pack('<2d',*a['uv'])==struct.pack('<2d',*uv);preserved+=1;eligible_original+=e
assert preserved==inv['source_readout_rows']
sources=[]
for path in unique:
 p=Path(path);h=hashlib.sha256()
 with p.open('rb')as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 sources.append(dict(file=path,sha256=h.hexdigest(),bytes=p.stat().st_size))
qa=[948992,950528,999424,1015552,1124864,1135104,1170944,1187584,1187840,1188096,1188352,1235200,1300992,1301248,1330944,1363456,1411328,1414912,1415168,1418240,1424896,1425408]
dependencies=[O/'patch_v5_native.py',O/'qualify_v5.py',O/'body_material_domain.py',O/'static_print_semantics.py',O/'measure_release.py',R/'dense_print_v1/dense_print_source.py',R/'dense56_v1/source_graphics.py',Path(r'D:/OneDrive/codex_projs/RinkSpace/calibration/program_tracking/extract_first_shot_ice.py'),O/'qa_v5_source.py',O/'seal_v5.py',O/'repair_direct_v5.py',O/'qualify_direct_v5.py',O/'body_qualified_direct_domain.py',O/'qualify_current_print_semantics.py',O/'scan_direct_body_delta.py']
manifest=dict(executed_small_source_dependencies=[ref(p)for p in dependencies],executed_review_source_dependencies=[ref(O/n)for n in ['probe_semantic_interval.py','review_arm_occlusion.py','grid_arm.py','diagnose_arm_rows.py','probe_v5_counterexamples.py','audit_final_counterexamples.py','make_direct_final_review.py']],schema='rinkspace.final_dense_advertisement_native_source_release.v1',complete=True,frames=1862,first_pts=948992,last_pts=1425408,native_step_pts=256,source_time_base='1/12800',source_video_sha256='e8b53966d56e72f3842d92c3ecbe6e9cc7e4670b7d0c9bd12f58fd2cff5ad57d',per_shot=dict(per),consumer_frame_directory=str(F),perframe_index=ref(file),canonical_member_files=sources,source_generation=ref(O/'source_final_v5_manifest.json'),primary_source_v5_generation=ref(O/'source_v5_manifest.json'),qualified_direct_component_repair=ref(O/'source_direct_v5_manifest.json'),final_binding_receipt=ref(O/'final_bindings_v5.json'),raw_source_generation=ref(O/'source_release_manifest.json'),measurement_recipe=ref(O/'measure_release.py'),static_qualification=ref(O/'static_direct_v5_manifest.json'),actual_current_body_material_recipe=ref(O/'body_material_domain.py'),actual_qualified_direct_body_material_recipe=ref(O/'body_qualified_direct_domain.py'),actual_current_print_semantic_extension=ref(O/'qualify_current_print_semantics.py'),actual_22_extra_source_pattern_review=ref(O/'direct_component_source_review.json'),actual_22_final_overlay_review=ref(O/'direct22_final_qualified_QA.jpg'),final_review_counts=ref(O/'final_v5_review_counts.jsonl'),actual_semantic_normal_recipe=ref(O/'static_print_semantics.py'),actual_graphics_recipe=ref(R/'dense56_v1/source_graphics.py'),actual_delta_source_reader=ref(O/'repair_direct_v5.py'),actual_static_qualifier=ref(O/'qualify_direct_v5.py'),source_inventory=ref(O/'original_source_inventory.json'),source_RGB_mask_and_anchored_SHA_checks=1862,source_unknown_type_clarification=bindings['unknown_type_clarification_ref'],qualified_direct_contact_body_repair_frames=22,qualified_direct_contact_body_repair_actual_RGB_exposures=34,qualified_direct_contact_body_repair_new_raw_rows=397,original_advertisement_readouts_bitwise_preserved=preserved,original_readouts_static_eligible=eligible_original,original_manual_click_rows=inv['manual_rows'],deleted_source_points_not_restored=True,consumer_uses_frozen_board_generation=bindings['board_directory'],composite_mask_index=ref(Path(bindings['composite_mask_index'])),composite_mask_manifest=bindings['composite_mask_manifest_ref'],board_release_receipt=bindings['board_release_ref'],body_material_classification_excludes_unverified_blue_cap_highlights=True,unknown_cap_bounds_are_exclusion_only_no_measured_material_anchor=True,current_body_material_confirmed_by_real_cap_and_foot_connected_region_or_original_print_patch=True,actor_guard_native_px=8,unknown_source_exclusion_guard_native_px=8,unknown_exclusion_separate_from_person_mask=True,complete_plus_minus_3px_current_RGB_profile_checked=True,guard_is_engineering_exclusion_not_sigma_or_calibrated_blur=True,actual_shared_current_RGB_graphics_presence_and_full_profile_exclusion=True,static_support_requires_both_exposures_current_surface_qualified=True,legacy_raw_visibility_fields_are_historical_final_status_is_authoritative=True,onedrive_path_check=ref(O/'onedrive_path_check.json'),cross_shot_tracks_used=False,source_UV_or_ID_moved=False,source_sigma_unknown=True,world_material_crosswalk_not_issued=True,no_camera_model_or_solver=True,unknown_and_dynamic_candidates_preserved_but_excluded=True,source_candidate_count_is_not_final_delivered_500_count=True,normal_only_and_texture_2D_types_explicit_in_every_final_frame=True,world_correspondence_and_independent_weights_not_issued=True,actual_UI_and_cap_counterexample_native_pts=1124864,actual_UI_and_cap_current_v5_body_counterexample=ref(O/'body_v5_probe/1124864_counterexample.json'),final_bound_counterexample_audit=ref(O/'final_counterexample_audit.json'),actual_source_arm_occlusion_review=ref(O/'arm_occlusion_review/near_1188352_prior_eligibility.json'),actual_source_arm_interior_diagnostic=ref(O/'arm_occlusion_review/sleeve_interior_diagnostic.json'),actual_source_arm_final_independent_review=ref(O/'arm_occlusion_review/independent_arm_final_review.json'),actual_all15_semantic_source_review=ref(O/'semantic_interval_review/all15_source_receipt.json'),superseded_not_for_consumer=['final_frames','final8_frames','final_consumer_frames','source frames bound to rejected board arrays'],self_review=dict(current_original_RGB_frames_read=1862,visual_source_cases=qa,actual_current_source_position_and_adjacent_material_motion_checked=True,QA_files=[ref(O/f'{p}_final_v5_QA.jpg')for p in qa if(O/f'{p}_final_v5_QA.jpg').exists()]),unresolved=['Low-contrast or motion-blurred fragments without identifiable material and a reliable current half-transition remain unknown. Reviewed recognizable static source print without sufficient 2D LK support has explicit status7 normal-only, never fixed 2D/world correspondence.','Source-frame labels are preserved; unbound board-material components do not become world material identities.','The final <=500/frame selection across board, ice and fine print is performed by root.','The 52 current source frames contain no manual.* advertisement clicks; deleted historical clicks were not reconstructed.'])
(O/'final_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(dict(complete=True,frames=1862,index_sha256=manifest['perframe_index']['sha256'],manifest_sha256=sha((O/'final_manifest.json').read_bytes()),per_shot=dict(per),original_preserved=preserved)))
