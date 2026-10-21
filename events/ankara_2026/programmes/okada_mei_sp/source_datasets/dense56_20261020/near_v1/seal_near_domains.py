from pathlib import Path
import json,gzip,sys,hashlib,cv2,numpy as np
N=Path(sys.argv[1]);T=N/'final_v3'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,j):Path(p).write_text(json.dumps(j,ensure_ascii=False,separators=(',',':')),encoding='utf8')
idx=list(map(json.loads,gzip.open(T/'index.jsonl.gz','rt',encoding='utf8')));assert len(idx)==486
count=weak=0;ids=set();visible=[];unknown=[];counter=[];mask_pins=[]
for i,r in enumerate(idx):
 p=r['native_pts'];assert p==1301248+256*i;fp=Path(r['frame_file']);assert sha(fp)==r['frame_sha256'];f=json.load(open(fp,encoding='utf8'));assert f['source_BGR_sha256']==r['source_BGR_sha256']and f['source_RGB_sha256']==r['source_RGB_sha256'];assert f['source_UV_moved']is False
 masks={}
 for k in ('near_outside_exclusion','ice_visible','board_body_visible'):
  ref=f[k+'_mask_ref'];assert sha(ref['file'])==ref['sha256'];m=cv2.imread(ref['file'],0);assert m.shape==(1080,1920);masks[k]=m>0
 assert not np.any(masks['near_outside_exclusion']&masks['ice_visible']);assert not np.any(masks['near_outside_exclusion']&masks['board_body_visible'])
 old=cv2.imread(f['old_board_body_mask_ref']['file'],0)>0;assert sha(f['old_board_body_mask_ref']['file'])==f['old_board_body_mask_ref']['sha256'];assert not np.any(masks['board_body_visible']&~old)
 pref=f['current_person_mask_ref'];assert sha(pref['file'])==pref['sha256'];person=cv2.imread(pref['file'],0)>0;guard=cv2.dilate(person.astype('uint8'),np.ones((17,17),np.uint8))>0
 assert not np.any(guard&masks['ice_visible']);assert not np.any(guard&masks['board_body_visible'])
 rows=f['new_source_rows'];count+=len(rows)
 for q in rows:
  assert q['source_row_id']not in ids;ids.add(q['source_row_id']);q['source_branch_id']='near_lower_long_side';assert q['native_pts']==p and q['feature_id']=='board.top'and q['model_or_world_projection_used']is False and q['source_UV_is_direct_measurement']is True;weak+=int(q['weak_source'])
  uv=np.array(q['uv']);sup=q['pixel_support'];assert np.isfinite(uv).all()and 0<=uv[0]<1920 and 0<=uv[1]<1080;assert sup['normal_only'];assert not(sup['gray_plateaus'][0]>=145 and sup['material_contrast']<25)
  for ends in [sup['position_interval_endpoints_uv'],sup['material_plateau_support_endpoints_uv']]:
   a,z=np.array(ends);assert np.isfinite([a,z]).all();assert np.all([a[0]>=0,z[0]>=0,a[0]<1920,z[0]<1920,a[1]>=0,z[1]>=0,a[1]<1080,z[1]<1080]);qxy=np.round(a[None,:]+np.linspace(0,1,max(3,int(np.linalg.norm(z-a))*2+1))[:,None]*(z-a)[None,:]).astype(int);assert not np.any(guard[qxy[:,1],qxy[:,0]])
 # Explicit observed spans retain material gaps; the list is not a polyline drawn through unobserved UV.
 spans=[]
 for q in sorted(rows,key=lambda q:q['uv'][1]):
  if not spans or q['uv'][1]-spans[-1][-1]['uv'][1]>12:spans.append([])
  spans[-1].append(q)
 for br in f['near_top_branches']:
  br['polyline_is_sample_inventory_not_connection_across_unknown']=True;br['visible_source_spans']=[dict(polyline_uv=[q['uv']for q in ss],source_row_ids=[q['source_row_id']for q in ss])for ss in spans]
 f['source_definition_authority_ref']=dict(file=str(N/'historical_semantics_recovered.json'),sha256=sha(N/'historical_semantics_recovered.json'));f['physical_boundary_role']='same_user_defined_zero_thickness_board_top_near_and_far_visible_branches';f['occluded_or_no_gray_material_profile_regions_are_not_filled_with_source_points']=True
 if rows:visible.append(p)
 elif f['state'].startswith('border'):unknown.append(p)
 write(fp,f);r['frame_sha256']=sha(fp)
 if p in [1334016,1334272,1339648,1346048,1351424,1351680,1363456,1414912]:
  tests=[]
  if p==1339648:tests=[(6,564),(59,595),(0,625),(84,625)]
  if p==1346048:tests=[(12,611),(58,550)]
  for x,y in tests:assert masks['near_outside_exclusion'][y,x]and not masks['board_body_visible'][y,x]and not masks['ice_visible'][y,x]
  counter.append(dict(native_pts=p,state=f['state'],new_source_rows=len(rows),actual_wrong_external_candidates_excluded_xy=tests,near_exclusion_pixels=int(masks['near_outside_exclusion'].sum()),body_visible_pixels=int(masks['board_body_visible'].sum())))
 assert sha(f['old_far_board_source_ref']['file'])==f['old_far_board_source_ref']['sha256']
with gzip.open(T/'index.jsonl.gz','wt',encoding='utf8',compresslevel=6)as out:
 for r in idx:out.write(json.dumps(r,separators=(',',':'))+'\n')
assert visible==list(range(1334272,1351424+1,256));assert unknown==[1334016,1351680];assert count==4773
qa=json.load(open(T/'QA'/'index.json',encoding='utf8'));assert qa['frames']==list(range(1334016,1351680+1,256));pages=[dict(file=str(T/'QA'/f'page_{i:02d}.jpg'),sha256=sha(T/'QA'/f'page_{i:02d}.jpg'))for i in range(12)]
review=dict(schema='rinkspace.near_board_actual_source_qa.v1',reviewer='dense56_release_audit source reader',source_all486_exposures_decoded_BGR_RGB_hash_bound=True,all70_affected_current_source_crops_actually_visually_reviewed=True,reviewed_native_pts=qa['frames'],reviewed_pages=pages,material_definition_recovered_from_direct_user_authority=True,findings='Correct gray top ice-facing boundary identified as shared board.top. External camera/cables/top surface excluded from ice and advertisements. Real white bracket/material interruptions remain as source gaps. Two source mixed border frames remain unknown with no measured near rows. Initial shallow-ice false candidate at1336320(11.61,763.20) rejected after actual gray plateau inspection; failed candidates preserved.',new_source_rows=count,weak_direct_source_rows=weak,unknown_mixed_frames=unknown,all_visible_source_feature_perfect_or_physical_fit_claimed=False,source_sigma_calibrated=False,world_station_issued=False)
write(T/'source_review.json',review)
validation=dict(schema='rinkspace.near_board_consumer_validation.v1',frames=486,native_pts_complete=True,frame_files_and_all_three_mask_SHA_verified=True,all4773_source_rows_unique_finite_and_profile_supports_inside_source=True,current_person_guard_supports_clear=True,old_far_board_UV_files_unchanged=True,ice_and_board_body_do_not_intersect_near_outside=True,body_visibility_is_subset_of_original_actual_material_qualification=True,counterexamples=counter,new_source_rows=count,weak_direct_source_rows=weak,near_visible_frames=len(visible),unknown_mixed_frames=unknown,physical_fit_claimed=False)
write(T/'validation.json',validation)
m=json.load(open(T/'manifest.json',encoding='utf8'));m['accepted_for_final_source_qualification']=True;m['index']=dict(file=str(T/'index.jsonl.gz'),sha256=sha(T/'index.jsonl.gz'));m['source_review']=dict(file=str(T/'source_review.json'),sha256=sha(T/'source_review.json'));m['validation']=dict(file=str(T/'validation.json'),sha256=sha(T/'validation.json'));m['historical_user_semantics']=dict(file=str(N/'historical_semantics_recovered.json'),sha256=sha(N/'historical_semantics_recovered.json'));m['sealer']=dict(file=str(Path(__file__)),sha256=sha(__file__));m['new_source_rows']=count;m['weak_direct_source_rows']=weak;m['source_native_visible_pts']=[visible[0],visible[-1]];m['source_mixed_unknown_pts']=unknown;m['source_capture_range_chosen_from_current_RGB_not_hardcoded_detection']=True;m['source_branch_id']='near_lower_long_side';m['world_entity']='board.top';m['source_sigma_and_independence_not_issued']=True;m['all_visible_features_perfect_claimed']=False;m['original_v1_videos_and_frozen_readouts_unchanged']=True;m['failed_source_candidates_preserved']=['candidate_v3','final_v1','final_v2','detect_probe_v1.json'];write(T/'manifest.json',m);write(N/'manifest.json',m)
print(json.dumps({'manifest':str(N/'manifest.json'),'manifest_sha256':sha(N/'manifest.json'),'index':m['index'],'frames':486,'near_visible':len(visible),'unknown':unknown,'new_source_rows':count,'weak':weak,'validation_sha256':sha(T/'validation.json')}))
