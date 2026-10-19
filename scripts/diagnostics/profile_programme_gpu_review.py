"""Profile one unchanged native source frame, without a second bulk scan."""
import cProfile
import json
from pathlib import Path
import pstats
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'src'), str(ROOT/'calibration/program_tracking'), str(ROOT/'scripts')]
import cv2
import numpy as np
import torch
from review_programme_source_inputs import _initialize_worker, _measure_arrays

torch.set_num_threads(1)
cv2.setNumThreads(1)
data = Path('D:/CodexData/RinkSpace_Data/events/ankara_2026')
run = data/'calibration/runs/solver_acceleration_20261003/okada_q32_reviewed_source_pixels_current_points_ready_20261012'
state = run/'accepted_states/8a28afed8f9f896112e5b2f64be515b2bc02019e10e09c7e0eda2598579d4891'
identity = json.loads((state/'state_identity.json').read_text(encoding='utf-8-sig'))
scene = json.loads((state/'scene_UNACCEPTED.json').read_text(encoding='utf-8-sig'))
roles = json.loads((data/'observations/vision_training/paint_role_contract_current.json').read_text(encoding='utf-8-sig'))
pts = 229246976
with np.load(identity['camera_path'], allow_pickle=False) as saved:
    camera = saved['camera25'][np.flatnonzero(saved['pts'] == pts)[0]]
raw = cv2.imread(str(data/'evidence/user_video_review_20261009/a13_complete_programme/centre_C_04_08_44_933/original_presentation_native_229246976.png'))
binding = dict(source_sha256='f685d27882e78f2446c734ceee4321dcb275a5ad1ebf81ea90bca68c2fcec891',
    time_base='1/15360', source_pts=pts, actual_pts=pts, shot_id='223403520:R0402_034', native_hw=[1080,1920])
frame = dict(actual_pts=pts, native_pts=pts, camera25=camera, source_binding=binding)
mask = np.zeros(raw.shape[:2], bool)
_initialize_worker(scene, binding['source_sha256'], binding['time_base'], roles, 'cuda')
# Warm CUDA once; timings then reflect ongoing work rather than initialization.
_measure_arrays(raw, mask, frame, np.empty((0,2)))
profile = cProfile.Profile()
profile.enable()
result = _measure_arrays(raw, mask, frame, np.empty((0,2)))
profile.disable()
print('actual_queries', len(result['ice']['query_uv']), 'actual_candidates', len(result['ice']['candidate_uv']))
pstats.Stats(profile).strip_dirs().sort_stats('cumtime').print_stats(16)
