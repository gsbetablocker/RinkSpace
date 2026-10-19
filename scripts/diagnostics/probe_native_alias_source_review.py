"""Verify one original native alias pair; no FIT, source pack or image export."""
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'src'), str(ROOT/'calibration/program_tracking')]
import cv2
import numpy as np
import torch
from rinkspace.programme_source_stream import FrozenProgrammeSource
from rinkspace.programme_source_measurement_cache import ProgrammeSourceMeasurementCache
from rinkspace.vision_programme_ice_review import IceProgrammeSourceReader
from rinkspace.native_alias_source_review import review_native_alias

cv2.setNumThreads(1)
torch.set_num_threads(1)
base = Path('D:/CodexData/RinkSpace_Data/events/ankara_2026')
run = base/'calibration/runs/solver_acceleration_20261003/okada_q32_reviewed_source_pixels_current_points_ready_20261012'
state = run/'accepted_states/8a28afed8f9f896112e5b2f64be515b2bc02019e10e09c7e0eda2598579d4891'
work = base/'diagnostics/current56_shared_active_set_filter_20261009/systematic_continuation_20261010/full_programme_strict_source_review_20261012'
programme = FrozenProgrammeSource(state/'state_identity.json',
    base/'media/originals/oxXu1avhrvs.mp4', ROOT/'calibration/program_tracking/source_shots.json')
roles = json.loads((base/'observations/vision_training/paint_role_contract_current.json').read_text(encoding='utf-8-sig'))
reader = IceProgrammeSourceReader(programme.scene, source_sha256=programme.source_sha256,
    time_base=programme.time_base, role_contract=roles, profile_backend='cuda')
cache = ProgrammeSourceMeasurementCache(work/'source_pixels/source_measurements.json',
    source_sha256=programme.source_sha256, time_base=programme.time_base,
    expected_actual_pts=programme.actual_inventory)
aliases = np.flatnonzero((programme.native_pts != programme.actual_pts) & (programme.native_pts > 229246976))
index = int(aliases[0])
canonical_pts, native_pts = int(programme.actual_pts[index]), int(programme.native_pts[index])
raw = None
for frame in programme.frames(start_pts=canonical_pts, stop_pts=native_pts+1, cpu_threads=1):
    if frame['representative']:
        raw = frame['raw_bgr']
        continue
    start = time.monotonic()
    canonical = cache.frame(canonical_pts)
    mask = np.zeros(raw.shape[:2], bool)
    result = review_native_alias(frame=frame, canonical_measurement=canonical,
        exclusion_mask=mask, source_ice_exclusion_mask=mask, source_ice_scope_mask=~mask,
        fresh_board_measurement={}, navigation_reader=reader,
        canonical_old_uv=np.empty((0,2)), canonical_old_uv_start_index=0,
        canonical_raw_bgr=raw, canonical_board_rows=())
    status = result['canonical_candidate_status']
    supported = status == 1
    assert result['new_FIT_observations_created'] is False
    assert np.array_equal(result['measured_native_uv'][supported],
        result['measured_native_sides_uv'][supported].mean(axis=1))
    assert result['decoded_native_BGR_sha256'] == frame['source_binding']['decoded_native_BGR_sha256']
    print(json.dumps(dict(canonical_pts=canonical_pts, native_pts=native_pts,
        original_source_BGR_different=canonical['source_binding']['decoded_native_BGR_sha256'] != result['decoded_native_BGR_sha256'],
        actual_source_candidates_checked=len(status), status_counts=np.bincount(status,minlength=5).tolist(),
        elapsed_seconds=time.monotonic()-start, CPU_threads=1, profile_device='cuda',
        new_FIT_observations_created=False), separators=(',',':')), flush=True)
    break
