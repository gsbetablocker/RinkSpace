"""Pinned final v1 -> v4 source identity diff and critical-case metadata receipts.
No image, model, source UV, mask, or solver mutation.
"""
from pathlib import Path
from collections import Counter, defaultdict
import argparse, base64, gzip, hashlib, json, subprocess, sys

OLD_SHA='13572b4b1eb0fc96a41febb5be0acd4c85e8cb578a794b20966aa9eea56637cf'
COMPARE_SHA='568f0d7b1163b2023bb555f0e2ae4b0baa9209e8be48e6e28f9e55acb35ffe25'
S06_SHA='0db0b91dbb668c895e65cd5cb07563711e1241b9e1d94304a9bf3475061a181b'
V2_QA_SHA='c37ed281a075fb596492747ed0293c526dbeb0f33db6bfbaed25e1d1fdd77d1e'
CASES=(1038592,1057792,1068032,1071872,1078272,1093632,1096192,1342208)
EXPECTED=[(s,p) for s,a,b in [('s05',948992,1300992),('s06',1301248,1425408)] for p in range(a,b+1,256)]
VALID=set(EXPECTED)

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def ref(p):return dict(file=str(p),sha256=sha(p))
def pin(r):
 assert sha(r['file'])==r['sha256'], 'Pinned input changed: '+r['file']
 return r

def lines(p):
 with gzip.open(p,'rt',encoding='utf8') as f:
  for l in f:yield json.loads(l)
def ident(r):return (r['feature_id'],r.get('source_point_id') or tuple(r['uv']))
def clock(r):return r['shot_id'],r['native_pts']
def compact(r):
 return {k:r.get(k) for k in ('shot_id','native_pts','source_point_id','feature_id','uv','display_class','source_branch_id','observation_type_code')}
def selected_cases(m):
 cases=defaultdict(dict);order=[];prev=None;cnt=0;per=Counter()
 for r in lines(m['files']['selected_rows.jsonl.gz']['file']):
  k=clock(r);assert k in VALID
  if k!=prev:
   if prev is not None:assert cnt<=500
   assert prev is None or k>prev, 'Selected source frame order'
   order.append(k);prev=k;cnt=0
  cnt+=1;per[k[0]]+=1
  if k[1] in CASES:
   q=ident(r);assert q not in cases[k];cases[k][q]=r
 assert cnt<=500 and order==EXPECTED, 'Incomplete native source clock'
 return cases,dict(per)
def rejected_cases(m):
 out=defaultdict(lambda:defaultdict(list));prev=None;n=0
 for r in lines(m['files']['excluded.jsonl.gz']['file']):
  k=clock(r);assert k in VALID
  assert prev is None or k>=prev, 'Excluded rows must be sorted by shot/native PTS for FrameCursor'
  prev=k;n+=1
  if k[1] in CASES and r.get('source_row_id') is not None:out[k][str(r['source_row_id'])].append(r['reason'])
 return out,n


V3_GLOBAL_SHA='7ad095fc2aca1072337b13ceb27250a6dc55f9d22f8699d568218674af4cf86a'
V3_S05_SHA='1c1761bb80c9d74a85bf8e76148fa7e61d5f8832d89a3a4956a66725a34cc5b3'
LEG_SOURCE_ID='dense56_board/1071872/board.contact/1142'
LEG_UV=[1142,355.5]

def grouped_rows(p):
 batch=[];last=None
 for r in lines(p):
  k=clock(r)
  if last is not None and k!=last:
   assert k>last, 'Native frame order changed'
   yield last,batch;batch=[]
  last=k;batch.append(r)
 if last is not None:yield last,batch

def subset_v3_to_v4(root,new):
 from itertools import zip_longest
 vp=root/'generation_v3/manifest.json';assert sha(vp)==V3_GLOBAL_SHA
 prior=read(vp);assert prior['complete'] and prior['generation_id']=='dense56_v3'
 assert sha(root/'parts_v3/s05/manifest.json')==V3_S05_SHA
 for r in prior['files'].values():pin(r)
 # Rehash actual effective source input refs once, comparing bytes rather than path aliases.
 checked={}
 def signature(obj):
  if obj is None:return None
  if isinstance(obj,dict):
   if 'file' in obj and 'sha256' in obj:
    f=str(Path(obj['file']).resolve());s=obj['sha256']
    if f not in checked:pin(obj);checked[f]=s
    else:assert checked[f]==s, 'Conflicting source input SHA for same path'
    return s
   return {k:signature(v) for k,v in sorted(obj.items())}
  if isinstance(obj,list):return [signature(v) for v in obj]
  return obj
 relevant=('mask','source_unknown_exclusion_mask_ref','graphics_mask','board','anchored_board','actual_surface_override','upstream_board_actor_mask','advert','advert_actual_source_proofs')
 def bindings(m):
  out={};order=[]
  for r in lines(m['files']['consumer_bindings.jsonl.gz']['file']):
   k=clock(r);assert k not in out;order.append(k)
   assert r.get('source_RGB_sha256') and r.get('source_BGR_sha256')
   out[k]=dict(rgb=r['source_RGB_sha256'],bgr=r['source_BGR_sha256'],inputs={f:signature(r.get(f)) for f in relevant})
  assert order==EXPECTED
  return out
 pb=bindings(prior);nb=bindings(new)
 for k in EXPECTED:assert (pb[k]['rgb'],pb[k]['bgr'])==(nb[k]['rgb'],nb[k]['bgr']), 'Native RGB/BGR changed'
 # Selected-frame RGB hashes are verified independently from the consumer bindings.
 for m,b in ((prior,pb),(new,nb)):
  order=[]
  for r in lines(m['files']['selected_frames.jsonl.gz']['file']):
   k=clock(r);order.append(k);assert r['decoded_native_BGR_sha256']==b[k]['bgr']
  assert order==EXPECTED
 changed=[];unchanged=[];same_inputs=[];same_overlay=[];reuse=[];removed_records=[];retained=0;prior_total=0;new_total=0;byshot=defaultdict(Counter)
 leg_prior=[];leg_final=[];type_changes=Counter();seen=[]
 for oldgroup,newgroup in zip_longest(grouped_rows(prior['files']['selected_rows.jsonl.gz']['file']),grouped_rows(new['files']['selected_rows.jsonl.gz']['file'])):
  assert oldgroup is not None and newgroup is not None and oldgroup[0]==newgroup[0]
  k,a=oldgroup;b=newgroup[1];seen.append(k);aa={ident(r):r for r in a};bb={ident(r):r for r in b}
  assert len(aa)==len(a) and len(bb)==len(b)
  assert not (bb.keys()-aa.keys()), 'v4 introduced source IDs; no refill permitted'
  for q,r in bb.items():
   assert r['uv']==aa[q]['uv'], 'Retained v3 UV moved'
   ot=tuple(aa[q].get(z) for z in ('observation_type_code','normal_only','texture_2D_tracking'));nt=tuple(r.get(z) for z in ('observation_type_code','normal_only','texture_2D_tracking'))
   if ot!=nt:type_changes[str(ot)+' -> '+str(nt)]+=1
  rr=aa.keys()-bb.keys();retained+=len(bb);prior_total+=len(a);new_total+=len(b);byshot[k[0]].update(dict(prior_rows=len(a),final_rows=len(b),removed_rows=len(rr)))
  equal_overlay=a==b
  equal_inputs=pb[k]['inputs']==nb[k]['inputs']
  if equal_overlay:same_overlay.append(k)
  if equal_inputs:same_inputs.append(k)
  if equal_overlay and equal_inputs:reuse.append(k)
  if rr:
   changed.append(dict(shot_id=k[0],native_pts=k[1],prior_selected_rows=len(a),final_selected_rows=len(b),removed_rows=len(rr),same_source_RGB=True,same_source_BGR=True,actual_effective_source_inputs_equal=equal_inputs))
   for q in rr:removed_records.append(compact(aa[q]))
  else:unchanged.append(k)
  if k==('s05',1071872):
   leg_prior=[r for r in a if r.get('source_point_id')==LEG_SOURCE_ID];leg_final=[r for r in b if r.get('source_point_id')==LEG_SOURCE_ID]
 assert seen==EXPECTED and len(leg_prior)==1 and leg_prior[0]['uv']==LEG_UV and not leg_final
 lookup={(r['shot_id'],r['native_pts'],str(r['source_point_id'])):r for r in removed_records}
 for r in lines(new['files']['excluded.jsonl.gz']['file']):
  k=(r['shot_id'],r['native_pts'],str(r.get('source_row_id')))
  if k in lookup:lookup[k].setdefault('actual_exclusion_reasons',[]).append(r['reason'])
 for r in removed_records:assert r.get('actual_exclusion_reasons'), 'Removed v3 point lacks source-reason receipt'
 leg=[r for r in removed_records if r.get('source_point_id')==LEG_SOURCE_ID];assert len(leg)==1
 reasons=Counter()
 for r in removed_records:reasons[r['actual_exclusion_reasons'][-1]]+=1
 encode=lambda rows:[dict(shot_id=s,native_pts=p) for s,p in rows]
 return dict(schema='rinkspace.actual_v3_to_v4_source_subset.v1',complete=True,prior_generation=ref(vp),new_generation_id=new['generation_id'],all_native_frames=1862,all_frames_same_source_RGB_and_BGR=True,selected_frame_BGR_matches_actual_consumer_source_binding=True,retained_source_ids_and_UV_exactly_equal=True,new_source_ids=0,prior_selected_rows=prior_total,final_selected_rows=new_total,removed_rows=len(removed_records),retained_rows=retained,by_shot={s:dict(v) for s,v in byshot.items()},changed_native_frames=changed,unchanged_selected_source_ID_frames=encode(unchanged),exact_entire_overlay_record_equal_frames=encode(same_overlay),actual_effective_source_input_SHA_equal_frames=encode(same_inputs),source_evidence_reuse_frames=encode(reuse),source_evidence_reuse_requires_same_RGB_and_BGR_and_entire_overlay_records_and_effective_input_bytes=True,verified_effective_input_files=len(checked),removed_rows_by_source_reason=dict(reasons),removed_source_rows=removed_records,known_v3_leg_blocker=leg[0],retained_constraint_type_changes=dict(type_changes),source_image_QA_not_replaced=True,physical_solver_or_live_packet_changed=False)


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--new-sha256',required=True);a=ap.parse_args()
 root=a.root;oldp=root/'generation_v1/manifest.json';newp=root/'generation_v4/manifest.json';out=root/'source_diff_v4'
 assert not out.exists(), 'Do not overwrite prior diff'
 assert sha(oldp)==OLD_SHA and sha(newp)==a.new_sha256
 old,new=read(oldp),read(newp)
 assert new['complete'] and new['generation_id']=='dense56_v4' and new['frames']==1862
 assert new['schema']=='rinkspace.source_only_dense56_generation.v3'
 assert sha(root/'compare_source_generations.py')==COMPARE_SHA
 for m in (old,new):
  for r in m['files'].values():pin(r)
 parents=new['parent_shot_generations'];assert len(parents)==2
 for r in parents:pin(r)
 assert parents[1]['sha256']==S06_SHA, 'Accepted S06 must remain unchanged'
 qa_path=root/'qa_independent/v2_s05_image_review.json';assert sha(qa_path)==V2_QA_SHA
 qa=read(qa_path);v2p=root/'parts_v2/s05/manifest.json';v2=read(v2p)
 assert sha(v2p)==qa['generation_manifest']['sha256']
 pin(v2['files']['selected_rows.jsonl.gz'])
 subset_receipt=subset_v3_to_v4(root,new)
 # Parent v2 is S05-only; case load streams it without requiring both shots.
 v2cases=defaultdict(dict)
 for r in lines(v2['files']['selected_rows.jsonl.gz']['file']):
  if r['native_pts'] in CASES:v2cases[clock(r)][ident(r)]=r
 newcases,new_per=selected_cases(new)
 exclusions,excluded_count=rejected_cases(new)
 # Original comparator performs the complete v1/v3 source-ID intersection and exact UV check.
 subprocess.run([sys.executable,'-X','utf8',str(root/'compare_source_generations.py'),str(oldp),str(newp),str(out)],check=True)
 report=read(out/'manifest.json')
 subset_path=out/'v3_to_v4_subset.json';subset_path.write_text(json.dumps(subset_receipt,ensure_ascii=False,indent=2),encoding='utf8')
 assert report['counts']['frames']==1862 and report['retained_source_UV_exactly_equal']
 near=report['new_near_lower_long_side_upper_boundary']
 assert near==dict(visible_frames_with_selected_source_rows=68,selected_rows=1193,first_pts=1334272,last_pts=1351424)
 screenshot=[r for r in newcases[('s06',1342208)].values() if r.get('source_branch_id')=='near_lower_long_side']
 assert len(screenshot)==24, 'Actual displayed near top points, not raw candidates'
 known=[]
 for r in qa['confirmed_false_person_source_rows']+qa['reflection_unknown_source_rows']:
  k=clock(r);identity=ident(r);assert identity in v2cases[k]
  assert v2cases[k][identity]['uv']==r['uv']
  assert identity not in newcases[k], 'Known incorrect source point survived'
  reasons=exclusions[k].get(str(r['source_point_id']),[])
  assert reasons, 'Known incorrect point lacks actual source exclusion receipt'
  known.append(dict(**compact(r),final_selected=False,actual_exclusion_reasons=reasons,baseline_for_this_case='failed_v2_source_QA'))
 assert len(known)==8
 clear_id='dense56_ice/s05/1096192/lane1:edge0/60';clear=[r for r in newcases[('s05',1096192)].values() if r.get('source_point_id')==clear_id]
 assert len(clear)==1 and clear[0]['uv']==[1332.4743228478233,1026.4686834119811], 'Clear source anchor lost or moved'
 percase={}
 for p in CASES:
  s='s06' if p==1342208 else 's05';k=(s,p);aa=v2cases[k] if s=='s05' else {};bb=newcases[k]
  removed=aa.keys()-bb.keys();rr=Counter();unknown=0
  for q in removed:
   r=aa[q];rs=exclusions[k].get(str(r['source_point_id']),[])
   if rs:rr.update([rs[-1]])
   else:unknown+=1
  percase[str(p)]=dict(shot_id=s,source_seconds=p/12800,failed_v2_selected_rows=len(aa) if s=='s05' else None,final_selected_rows=len(bb),final_by_class=dict(Counter(r['display_class'] for r in bb.values())),removed_from_failed_v2=len(removed) if s=='s05' else None,removed_with_explicit_source_exclusion_by_reason=dict(rr),removed_without_explicit_exclusion=unknown,metadata_only_not_new_visual_acceptance=True)
 selection_reason='display_selection_changed_within_existing_500_ceiling'
 reasons=report['removed_overlay_rows_by_actual_reason'];removed=report['counts']['removed_old_display_rows'];display_only=reasons.get(selection_reason,0)
 assert sum(reasons.values())==removed
 case=dict(schema='rinkspace.final_source_critical_case_change.v4',new_generation=ref(newp),failed_v2_source_qa=ref(qa_path),checked_cases=percase,known_false_person_or_reflection_rows=known,clear_blue_material_anchor_retained=compact(clear[0]),s06_screenshot=dict(native_pts=1342208,source_seconds=104.86,clip_seconds=3.2,actual_selected_near_top_rows=len(screenshot)),six_shoe_pts=list(CASES[:6]),known_exact_rejected_rows=8,source_UV_moved=False,all_native_images_or_all_points_visually_accepted_by_this_audit=False,physical_solver_or_live_packet_changed=False)
 casep=out/'critical_cases.json';casep.write_text(json.dumps(case,ensure_ascii=False,indent=2),encoding='utf8')
 receipt=dict(schema='rinkspace.executed_final_source_diff_receipt.v4',complete=True,old_generation=ref(oldp),new_generation=ref(newp),comparator=ref(root/'compare_source_generations.py'),driver=ref(Path(__file__)),full_diff=ref(out/'manifest.json'),critical_cases=ref(casep),v3_to_v4_subset=ref(subset_path),excluded_stream_sorted_and_pinned=True,excluded_rows_scanned=excluded_count,counts=report['counts'],by_shot=report['by_shot'],actual_selected_rows_by_shot=new_per,removed_rows_total=removed,display_selection_only_removed_rows=display_only,removed_rows_with_actual_source_requalification_reason=removed-display_only,new_near_upper_boundary=near,screenshot_actual_near_upper_boundary_rows=24,retained_source_UV_exactly_equal=True,same_total_count_does_not_certify_accuracy=True,source_image_QA_is_separate=True,physical_solver_or_live_packet_changed=False)
 (out/'execution_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf8')
 for r in (receipt['old_generation'],receipt['new_generation'],receipt['comparator'],receipt['driver']):pin(r)
 print(json.dumps({k:receipt[k] for k in ('complete','counts','by_shot','display_selection_only_removed_rows','removed_rows_with_actual_source_requalification_reason','screenshot_actual_near_upper_boundary_rows')},ensure_ascii=False))
if __name__=='__main__':main()
