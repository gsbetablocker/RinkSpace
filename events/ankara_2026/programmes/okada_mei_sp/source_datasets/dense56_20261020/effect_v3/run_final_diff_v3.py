"""Pinned final v1 -> v3 source identity diff and critical-case metadata receipts.
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

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--new-sha256',required=True);a=ap.parse_args()
 root=a.root;oldp=root/'generation_v1/manifest.json';newp=root/'generation_v3/manifest.json';out=root/'source_diff_v3'
 assert not out.exists(), 'Do not overwrite prior diff'
 assert sha(oldp)==OLD_SHA and sha(newp)==a.new_sha256
 old,new=read(oldp),read(newp)
 assert new['complete'] and new['generation_id']=='dense56_v3' and new['frames']==1862
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
 # Parent v2 is S05-only; case load streams it without requiring both shots.
 v2cases=defaultdict(dict)
 for r in lines(v2['files']['selected_rows.jsonl.gz']['file']):
  if r['native_pts'] in CASES:v2cases[clock(r)][ident(r)]=r
 newcases,new_per=selected_cases(new)
 exclusions,excluded_count=rejected_cases(new)
 # Original comparator performs the complete v1/v3 source-ID intersection and exact UV check.
 subprocess.run([sys.executable,'-X','utf8',str(root/'compare_source_generations.py'),str(oldp),str(newp),str(out)],check=True)
 report=read(out/'manifest.json')
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
 case=dict(schema='rinkspace.final_source_critical_case_change.v3',new_generation=ref(newp),failed_v2_source_qa=ref(qa_path),checked_cases=percase,known_false_person_or_reflection_rows=known,clear_blue_material_anchor_retained=compact(clear[0]),s06_screenshot=dict(native_pts=1342208,source_seconds=104.86,clip_seconds=3.2,actual_selected_near_top_rows=len(screenshot)),six_shoe_pts=list(CASES[:6]),known_exact_rejected_rows=8,source_UV_moved=False,all_native_images_or_all_points_visually_accepted_by_this_audit=False,physical_solver_or_live_packet_changed=False)
 casep=out/'critical_cases.json';casep.write_text(json.dumps(case,ensure_ascii=False,indent=2),encoding='utf8')
 receipt=dict(schema='rinkspace.executed_final_source_diff_receipt.v3',complete=True,old_generation=ref(oldp),new_generation=ref(newp),comparator=ref(root/'compare_source_generations.py'),driver=ref(Path(__file__)),full_diff=ref(out/'manifest.json'),critical_cases=ref(casep),excluded_stream_sorted_and_pinned=True,excluded_rows_scanned=excluded_count,counts=report['counts'],by_shot=report['by_shot'],actual_selected_rows_by_shot=new_per,removed_rows_total=removed,display_selection_only_removed_rows=display_only,removed_rows_with_actual_source_requalification_reason=removed-display_only,new_near_upper_boundary=near,screenshot_actual_near_upper_boundary_rows=24,retained_source_UV_exactly_equal=True,same_total_count_does_not_certify_accuracy=True,source_image_QA_is_separate=True,physical_solver_or_live_packet_changed=False)
 (out/'execution_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf8')
 for r in (receipt['old_generation'],receipt['new_generation'],receipt['comparator'],receipt['driver']):pin(r)
 print(json.dumps({k:receipt[k] for k in ('complete','counts','by_shot','display_selection_only_removed_rows','removed_rows_with_actual_source_requalification_reason','screenshot_actual_near_upper_boundary_rows')},ensure_ascii=False))
if __name__=='__main__':main()
