"""Authenticate separated-circle exclusions and pack only nonempty source shots."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
BASE=Path('D:/CodexData/RinkSpace_Data/runs/native_programme_20261031/feature_refinement_1101')
RUN=BASE.parent
PROJECT=Path('D:/OneDrive/codex_projs/RinkSpace')
SOURCE=BASE/'circles_separated'
QUALIFIED=BASE/'qualified_separated_circles'

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def rowkey(sid,pt,row):return sid,int(pt),row['path_id'],*map(float,row['uv'])
def ref(path):
 path=Path(path).resolve(strict=True)
 with path.open('rb')as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
 return dict(path=str(path),sha256=digest)

def main():
 summary=read(QUALIFIED/'qualification_summary.json')
 reviews=read(BASE/'separated_circle_reviews.json')
 proof=read(SOURCE/'reconciliation_receipt.json')
 assert reviews['approved_components']=={} and reviews['include_inherited_seed_rows']is True
 assert proof['original_seed_inheritance']is False
 withheld={rowkey(r['shot'],r['pts'],r)for r in reviews['withheld_source_uv']}
 assert len(withheld)==3
 a4_source=read(SOURCE/'wide_full_3/s07/source_pixels.json')
 a4_bad={rowkey('s07',r['pts'],r)for collection in ('accepted','pending')for r in a4_source[collection]
         if r['pts']==1441024 and r['path_id']=='demo_initial/A4:outer' and r['candidate']in [25,27,29,31,32]}
 a4_good={rowkey('s07',r['pts'],r)for collection in ('accepted','pending')for r in a4_source[collection]
         if r['pts']==1441024 and r['path_id']=='demo_initial/A4:outer' and r['candidate']in [3,4,5,7]}
 assert len(a4_bad)==5 and len(a4_good)==4
 blocked=withheld|a4_bad
 pending_seen=set();qualified_seen=set();original_readout_count=0;nonempty=[];empty=[]
 for sid,counts in summary['shots'].items():
  (nonempty if counts['qualified_rows']else empty).append(sid)
  assert counts['eligible_original_rows']==counts['qualified_rows']+counts['sparse_omitted_rows']
  original_readout_count+=counts['eligible_original_rows']+counts['pending_rows']
  with (QUALIFIED/sid/'pending.jsonl').open(encoding='utf-8')as stream:
   for line in stream:
    item=json.loads(line);row=item['row'];pending_seen.add(rowkey(sid,row['pts'],row))
  with (QUALIFIED/sid/'source_rows.jsonl').open(encoding='utf-8')as stream:
   for line in stream:
    frame=json.loads(line)
    for row in frame['rows']:qualified_seen.add(rowkey(sid,frame['pts'],row))
 assert original_readout_count==proof['counts']['all_source_rows']==13309
 assert blocked<=pending_seen and not(blocked&qualified_seen)
 assert a4_good<=qualified_seen
 assert set(nonempty)=={'s02','s07'}
 assert summary['total_frames']==8329
 output=BASE/'separated_circle_qualification_checks.json'
 checks=dict(schema='rinkspace.separated_circle_qualification_checks.v1',
  source_graph_receipt=ref(SOURCE/'reconciliation_receipt.json'),inventory=ref(SOURCE/'all9_inventory.json'),
  reviews=ref(BASE/'separated_circle_reviews.json'),qualification_summary=ref(QUALIFIED/'qualification_summary.json'),
  all_original_readout_rows=original_readout_count,all_original_readout_fields_preserved_except_fragment=True,
  original_fragment_kept=True,all_source_exposures=summary['total_frames'],
  original_seeds_inherited=False,new_explicit_seed_fragments=proof['counts']['explicit_seed_fragments'],
  include_inherited_seed_rows_meaning='Compatibility switch consumes accepted rows of this newly rebuilt graph; all five seeds are explicit current source-review anchors. None of the 72 original old seeds was inherited.',
  explicit_withheld_UV_rows=3,withheld_UV_in_pending=len(withheld&pending_seen),withheld_UV_in_qualified=len(withheld&qualified_seen),
  A4_known_bad_upper_arc_rows=5,A4_bad_rows_in_pending=len(a4_bad&pending_seen),A4_bad_rows_in_qualified=len(a4_bad&qualified_seen),
  A4_explicit_good_arc_rows=4,A4_good_rows_in_qualified=len(a4_good&qualified_seen),
  A4_graph_separation=proof['shots']['s07']['A4_separation'],packet_shots=nonempty,no_packet_required_shots=empty,
  qualified_rows=summary['qualified_rows'],remaining_pending_rows=summary['remaining_pending_rows'],
  prepared_modified=False,prepare_executed=False,fit_executed=False,user_visual_acceptance=False,pixel_accuracy_accepted=False)
 with output.open('x',encoding='utf-8')as stream:stream.write(json.dumps(checks,ensure_ascii=False,indent=2)+'\n')
 print('QUALIFICATION_CHECKS '+json.dumps(checks,ensure_ascii=True),flush=True)
 subprocess.run([sys.executable,'-B',str(PROJECT/'events/ankara_2026/programmes/okada_mei_sp/build_registered_source_packets.py'),
  '--prepared',str(RUN/'prepared_source_recovered'),'--capture',str(QUALIFIED),
  '--output',str(BASE/'separated_circle_packets'),'--shots',','.join(nonempty)],cwd=str(PROJECT),check=True)

if __name__=='__main__':main()
