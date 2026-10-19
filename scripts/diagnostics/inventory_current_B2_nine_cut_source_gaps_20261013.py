"""Read-only exact B2 catalog/finite source support gap inventory."""
import hashlib
import json
from collections import Counter,defaultdict
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from calibration.program_tracking.compact_source_header_inventory import read_minimal_compact_source_headers
WORK=Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/diagnostics/current56_shared_active_set_filter_20261009/systematic_continuation_20261010/full_programme_strict_source_review_20261012')
DATASET=ROOT/'events/ankara_2026/programmes/okada_mei_sp/source_datasets/current_review_20261013/fit_dataset.json'
OUT=WORK/'current_B2_all_nine_source_gap_inventory_20261013_v1'

def ref(p):
 p=Path(p).resolve();return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def main():
 OUT.mkdir(exist_ok=False)
 dataset=json.loads(DATASET.read_text(encoding='utf-8'));dataset_ref=ref(DATASET)
 cachepath=WORK/'source_pixels/source_measurements.json'
 cache=json.loads(cachepath.read_text(encoding='utf-8'))
 ci=next(i for i,v in enumerate(cache['inventory']) if v['curve_id']=='B2:segment0')
 bycut=defaultdict(dict)
 for c in dataset['cuts'][:9]:
  p=c['source_pack']['path'];headers,offsets=read_minimal_compact_source_headers(p)
  with np.load(c['support_index']['path'],allow_pickle=False) as z:
   masks={k:z[k] for k in ('fit_point_indices','held_point_indices','pending_support_point_indices','pending_named_point_indices')}
  count=Counter(); ex=defaultdict(Counter)
  for ri,h in enumerate(headers):
   if h['physical_material_id']!='B2:segment0':continue
   lo,hi=map(int,offsets[ri:ri+2]);pts=h['actual_pts'];shot=h['shot_id']
   for k,ids in masks.items():
    n=int(np.searchsorted(ids,hi)-np.searchsorted(ids,lo));count[k]+=n;ex[pts][k]+=n
   record=bycut[shot].setdefault(pts,dict(actual_pts=pts,original_B2_rows=[],finite_FIT_points=0,HELD_points=0,pending_support_points=0,pending_named_points=0))
   record['original_B2_rows'].append(ri)
   for k,key in [('fit_point_indices','finite_FIT_points'),('held_point_indices','HELD_points'),('pending_support_point_indices','pending_support_points'),('pending_named_point_indices','pending_named_points')]:
    record[key]+=int(np.searchsorted(masks[k],hi)-np.searchsorted(masks[k],lo))
  print(json.dumps(dict(pack=Path(p).name,material='B2:segment0',point_counts=dict(count),exposures=len(ex))),flush=True)
 records=[]
 for chunk in cache['chunks']:
  with np.load(chunk['path'],allow_pickle=False) as z:
   pts,native,shots=z['actual_pts'],z['native_pts'],z['shot_id'];sha=z['BGR_sha256'];status=z['curve_status'];qo=z['query_offsets'];qc=z['query_curve'];qs=z['query_status']
   for fi,p in enumerate(pts):
    shot=str(shots[fi]);shot=shots[fi].decode() if isinstance(shots[fi],bytes) else shot
    lo,hi=map(int,qo[fi:fi+2]);ix=np.flatnonzero(qc[lo:hi]==ci)+lo
    r=dict(bycut[shot].get(int(p),dict(actual_pts=int(p),original_B2_rows=[],finite_FIT_points=0,HELD_points=0,pending_support_points=0,pending_named_points=0)))
    bgr=sha[fi].decode() if isinstance(sha[fi],bytes) else str(sha[fi])
    r.update(native_pts=int(native[fi]),shot_id=shot,decoded_BGR_sha256=bgr,
      cached_query_count=len(ix),cached_query_status_counts=dict(Counter(map(int,qs[ix]))),
      cached_curve_status=int(status[fi,ci]),source_cache_chunk_reference=chunk,
      profile_query_is_not_physical_visibility_proof=True)
    records.append(r)
 summaries=[]
 for shot in sorted({r['shot_id'] for r in records}):
  rr=[r for r in records if r['shot_id']==shot]
  query=[r for r in rr if r['cached_query_count']];fit=[r for r in rr if r['finite_FIT_points']]
  gaps=[r for r in query if not r['finite_FIT_points']]
  # Consecutive actual catalogue entries, not invented native presentations.
  runs=[];current=[]
  for r in rr:
   if r['cached_query_count'] and not r['finite_FIT_points']:current.append(r)
   elif current:runs.append(current);current=[]
  if current:runs.append(current)
  summaries.append(dict(shot_id=shot,actual_catalogue_exposures=len(rr),B2_cached_query_exposures=len(query),
    B2_exposures_with_finite_FIT=len(fit),B2_finite_FIT_points=sum(r['finite_FIT_points'] for r in rr),
    B2_query_exposures_without_finite_FIT=len(gaps),
    B2_HELD_exposures=sum(bool(r['HELD_points']) for r in rr),
    cached_curve_status_counts=dict(Counter(r['cached_curve_status'] for r in rr)),
    missing_query_runs=[dict(count=len(run),first_actual_pts=run[0]['actual_pts'],last_actual_pts=run[-1]['actual_pts'],
      example_actual_pts=[run[0]['actual_pts'],run[len(run)//2]['actual_pts'],run[-1]['actual_pts']]) for run in sorted(runs,key=len,reverse=True)],
    finite_FIT_actual_pts_range=[min(r['actual_pts'] for r in fit),max(r['actual_pts'] for r in fit)] if fit else None))
 result=dict(schema='rinkspace.B2-nine-cut-existing-source-gap-inventory.v1',dataset_reference=dataset_ref,
  source_cache_reference=ref(cachepath),implementation_reference=ref(__file__),expected_actual_exposures=8503,
  actual_catalogue_records=len(records),shot_summaries=summaries,records=records,
  original_source_visibility_not_inferred_from_query_count=True,source_FIT_HELD_or_dataset_changed=False,
  source_UV_or_labels_created=False,all_nine_cuts_complete_actual_catalogue_used=True)
 assert len(records)==8503
 p=OUT/'B2_all_nine_current_existing_source_gap_inventory.json'
 p.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 print(json.dumps(dict(reference=ref(p),summaries=summaries),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
