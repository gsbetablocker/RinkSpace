"""Decode selected existing-catalog B2 gap examples with the same full grid."""
import hashlib
import json
from pathlib import Path
import sys
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from calibration.solver_runtime import install
install()
from rinkspace.programme_source_stream import FrozenProgrammeSource
from calibration.program_tracking.render_result import SceneRenderer
WORK=Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/diagnostics/current56_shared_active_set_filter_20261009/systematic_continuation_20261010/full_programme_strict_source_review_20261012')
INVENTORY=WORK/'current_B2_all_nine_source_gap_inventory_20261013_v1/B2_all_nine_current_existing_source_gap_inventory.json'
OUT=WORK/'current_B2_all_nine_selected_gap_original_source_20261013_v1'
SOURCE=Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/media/originals/oxXu1avhrvs.mp4')
def ref(p):
 p=Path(p).resolve();return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def main():
 OUT.mkdir(exist_ok=False);cv2.setNumThreads(1)
 d=json.loads(INVENTORY.read_text(encoding='utf-8'))
 dataset=json.loads(Path(d['dataset_reference']['path']).read_text(encoding='utf-8'))
 catalogue={r['actual_pts']:r for r in d['records']}
 chosen=[]
 for s in d['shot_summaries']:
  rr=[r for r in d['records'] if r['shot_id']==s['shot_id']]
  pts=s['missing_query_runs'][0]['example_actual_pts'][1] if s['missing_query_runs'] else rr[len(rr)//2]['actual_pts']
  chosen.append(pts)
  if s['shot_id'].endswith('_004') and len(s['missing_query_runs'])>1:chosen.append(s['missing_query_runs'][1]['example_actual_pts'][1])
 programme=FrozenProgrammeSource(dataset['frozen_model_reference']['path'],SOURCE,ROOT/'calibration/program_tracking/source_shots.json')
 renderer=SceneRenderer(programme.scene,draw_grid=True,draw_advertisements=True)
 records=[]
 for pts in chosen:
  r=catalogue[pts];native=r['native_pts'];frame=next(programme.frames(start_pts=native,stop_pts=native+1,cpu_threads=1))
  assert frame['native_pts']==native and frame['actual_pts']==pts
  raw=frame['raw_bgr'];bgr=hashlib.sha256(raw.tobytes()).hexdigest();assert bgr==r['decoded_BGR_sha256']
  path=OUT/f'{pts}_EXACT_RAW_NATIVE.png';cv2.imwrite(str(path),raw)
  canvas,_=renderer.draw(raw,frame['camera25'],title=f'8a28 ALL GRID; existing B2 source gap navigation; actual {pts}; native {native}')
  grid=OUT/f'{pts}_SAME8a28_ALL_GRID.png';cv2.imwrite(str(grid),canvas)
  records.append(dict(existing_B2_inventory_record=r,source_binding=frame['source_binding'],
   decoded_native_BGR_sha256=bgr,original_raw_reference=ref(path),full_same_model_grid_reference=ref(grid),
   source_visibility_decision='awaiting_actual_original_image_review',source_q_or_point_created=False))
  print(json.dumps(dict(actual_pts=pts,native_pts=native,shot_id=r['shot_id'],queries=r['cached_query_count'])),flush=True)
 p=OUT/'B2_selected_original_source_gap_examples.json'
 p.write_text(json.dumps(dict(schema='rinkspace.B2-nine-cut-selected-gap-original-source-examples.v1',
  source_inventory_reference=ref(INVENTORY),model_reference=dataset['frozen_model_reference'],
  implementation_reference=ref(__file__),records=records,source_FIT_HELD_or_dataset_changed=False,
  full_same_8a28_model_navigation_used=True,source_profiles_remeasured=False),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 print(json.dumps(dict(reference=ref(p),selected_frames=len(records)),ensure_ascii=False))
if __name__=='__main__':main()
