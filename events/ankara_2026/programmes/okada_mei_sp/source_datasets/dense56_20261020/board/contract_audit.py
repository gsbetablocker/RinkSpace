import os,json,gzip,hashlib,time,numpy as np
from pathlib import Path
O=Path(r'D:/CodexData/RinkSpace_Data/runs/okada_rec_1019/recover_v2/dense56_board_v1/material_v5');PTS=list(range(948992,1425408+1,256))
def j(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def wr(p,r):p.write_text(json.dumps(r,ensure_ascii=False,separators=(',',':')),encoding='utf8')
while not(O/'qualified_manifest.json').exists()or not(O/'anchored_complete.json').exists():time.sleep(.2)
binding=j(O/'final_mask_binding.json');assert sha(Path(binding['file']))==binding['sha256'];mask_index={}
with gzip.open(binding['file'],'rt',encoding='utf8')as f:
 for line in f:
  v=json.loads(line);mask_index[v['native_pts']]=v
assert set(mask_index)==set(PTS)
frozen={}
with gzip.open(O/'frozen_native_uv.jsonl.gz','rt',encoding='utf8')as f:
 for line in f:
  v=json.loads(line);frozen[v['native_pts']]=v
assert set(frozen)==set(PTS)
def freeze_payload(q):return {ro:{k:z[k]for k in ['full_native_y','full_direct','full_gradient','full_chroma_before','full_chroma_after']}for ro,z in q['roles'].items()}|{'blue_cap_lower_y':q['roles']['board.top']['material_band']['full_blue_cap_lower_y']}
auth=j(O/'source_anchor_recipe.json');counts={};byrole={};n=0;changed=0;weak=0;combined=hashlib.sha256();firstlast=[];sourceHashes=0
for p in PTS:
 a=O/'anchored_frames'/(str(p)+'.json');qf=O/'qualified_frames'/(str(p)+'.json');ar=j(a);r=j(qf);hh=sha(a);assert hashlib.sha256(json.dumps(freeze_payload(ar),separators=(',',':'),allow_nan=False).encode()).hexdigest()==frozen[p]['frozen_native_material_arrays_sha256'];assert ar['decoded_native_BGR_sha256']==frozen[p]['decoded_native_BGR_sha256'];assert r['native_pts']==ar['native_pts']==p;assert r['anchored_file_sha256']==hh;assert r['decoded_native_BGR_sha256']==ar['decoded_native_BGR_sha256'];assert r['mask_sha256']==ar['mask_sha256']==sha(Path(r['mask_path']));combined.update((str(p)+':'+hh+'\n').encode());ids=set();assert r['mask_sha256']==mask_index[p]['mask_sha256'];assert Path(r['mask_path'])==Path(mask_index[p]['mask_file']);assert r['source_unknown_exclusion_mask_ref']==mask_index[p].get('source_unknown_exclusion_mask_ref');assert r['source_visibility_guard_px']==ar['source_visibility_guard_px']==8;assert r['source_unknown_exclusion_mask_ref']==r['unknown_exclusion_ref']==ar['source_unknown_exclusion_mask_ref']==ar['unknown_exclusion_ref'];ur=r['source_unknown_exclusion_mask_ref'];assert ur is None or sha(Path(ur['file']))==ur['sha256']
 for role,rr in r['roles'].items():
  assert len(ar['roles'][role]['full_native_y'])==1920
  sx=set();ex=set()
  for s in rr['spans']:
   lo,hi=s['x_range'];sx.update(range(lo,hi+1));assert s['endpoints_uv'][0]in rr['trace_uv']and s['endpoints_uv'][1]in rr['trace_uv']
  for lo,hi in rr['excluded_x_spans']:ex.update(range(lo,hi+1))
  assert not(sx&ex)and sx|ex==set(range(1920));assert len(sx)==rr['visible_columns'];assert len(rr['trace_uv'])==sum(z['role']==role for z in r['rows'])
  byrole[role]=byrole.get(role,0)+len(rr['trace_uv'])
 for row in r['rows']:
  assert row['source_row_id']not in ids;ids.add(row['source_row_id']);assert row['anchored_file_sha256']==hh;assert row['decoded_native_BGR_sha256']==r['decoded_native_BGR_sha256'];assert row['current_person_pixel_mask_sha256']==row['mask_sha256']==r['mask_sha256'];assert row['anchored_file']==r['anchored_file'];assert row['uv'][1]==ar['roles'][row['role']]['full_native_y'][int(row['uv'][0])];assert row['samples_independent']is False;assert row['model_or_world_projection_used']is False;assert row['pixel_support']['interval_is_not_sigma']is True;assert 'sigma'not in row['pixel_support'];assert 0<=row['uv'][0]<1920 and 0<=row['uv'][1]<1080
  if 'original_rejected_local_candidate_uv'in row:changed+=1
  if not row['pixels_measured_in_this_frame']:weak+=1
  n+=1
 counts[r['shot_id']]=counts.get(r['shot_id'],0)+len(r['rows']);sourceHashes+=1
for s,aa in auth['shots'].items():
 for a in aa:assert sha(Path(a['file']))==a['sha256']
r=dict(schema='rinkfit.complete_board_native_source_contract_audit.v1',complete=True,native_frames=1862,shots={'s05':1376,'s06':486},candidate_rows=n,rows_by_shot=counts,rows_by_role=byrole,all1862_native_BGR_and_current_mask_and_anchored_hash_bound=True,all_frame_source_row_IDs_unique=True,all_role_columns_partitioned_visible_or_excluded=True,all_visible_spans_have_actual_trace_endpoints=True,source_original_52_keyframe_sha256_unchanged=True,no_source_sigma_or_independent_sample_claim=True,raw_arrays_not_changed_by_local_requalification=True,all1862_frozen_parent_UV_and_material_support_arrays_identical_after_final_mask_qualification=True,all1862_person_and_unknown_refs_equal_unique_final_index=True,final_mask_index_sha256=binding['sha256'],ordered_frozen_native_material_arrays_sha256=j(O/'material_uv_freeze.json')['ordered_frozen_native_material_arrays_sha256'],locally_requalified_candidate_rows=changed,weak_current_source_rows=weak,ordered_anchored_files_sha256=combined.hexdigest(),source_is_not_physical_model_fit_acceptance=True)
wr(O/'contract_audit.json',r);print(json.dumps(r),flush=True)
