"""Source-only blur/displacement screen; stdout, no image output or fitting.

Quantile widths are mixed paint/optics/encoding/scan measurements, never a
calibrated shutter estimate. The optional rectangular-blur equivalent is a
dimensional diagnostic only.
"""
from pathlib import Path
import hashlib
import json
import sys

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src'), str(ROOT / 'scripts/diagnostics')]
from calibration.solver_runtime import install
install()
import replay_cut6_F3_existing_RF4_source_response_20261011 as original

CONFIG = ROOT / 'events/ankara_2026/programmes/okada_mei_sp/full_source_review86_q091f_photo_mount_range_ablation_20261011.json'
CLOCK = Path('D:/CodexData/RinkSpace_program_tracking_tmp/source_cadence_comparison/O6j8L8TmfKo/full_programme_alignment/native60_to_official50.jsonl')
CLOCK_SHA = 'a9be37308b9a39fb55519a0be69c741c53a8216a81a60a2066fff25bdcf54dcc'
PTS = (230506752, 230507776, 230508032)


def emit(stage, **value):
    print(json.dumps(dict(stage=stage, **value), ensure_ascii=False,
        default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x.item(),
        allow_nan=False), flush=True)


def quantile(profile, lo, sign, smooth):
    value = gaussian_filter1d(profile, smooth) if smooth else profile
    signed = sign * np.gradient(value)
    mass = np.maximum(signed, 0.)
    total = mass.sum()
    if total <= 0.: raise ValueError('No source edge gradient mass')
    q = np.interp([.1, .5, .9], np.cumsum(mass) / total,
        np.arange(lo, lo + len(value), dtype=float))
    return dict(q10_q50_q90=q, width=float(q[2] - q[0]),
        positive_gradient_mass=float(total),
        opposite_gradient_mass=float(np.maximum(-signed, 0.).sum()),
        net_endpoint_gray=float(sign * (value[-4:].mean() - value[:4].mean())))


def main():
    cv2.setNumThreads(1)
    options = original.read(CONFIG)['source_options']
    board = original.read(options['--additional-board-source'])['rows']
    ice = original.read(options['--additional-ice-source'])['rows']
    selected = {p: next(r for r in board if r['id'] in (
        'actual_tail_230506752_Ad7_left_vertical_source',
        'supplemental_native_230507776_Ad7_secondary_tail_left_external_visible_partial',
        'actual_tail_230508032_Ad7_left_vertical_source')
        and int(r.get('actual_pts', r['pts'])) == p) for p in PTS}
    if original.sha(CLOCK) != CLOCK_SHA: raise ValueError('Original clock changed')
    clock = {int(r['full_pts']): r for r in map(json.loads, CLOCK.open(encoding='utf8'))}
    for p in PTS:
        if clock[p]['match_status'] != 'unique_pixel_match':
            raise ValueError('Source exposure is not authenticated')
    emit('clock', source_pts=list(PTS), official_pts=[clock[p]['accepted_official_pts'] for p in PTS],
        official_time_base=[1, 12800], independent_exposure_interval_seconds=.02,
        shutter_duration_measured=False)
    gray = {}
    for p, row in selected.items():
        path = Path(row['source_raw_frame'])
        if original.sha(path) != row['source_file_sha256']:
            raise ValueError('Original source PNG changed')
        image = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
        if image.shape != (1080, 1920, 3): raise ValueError('Source dimensions changed')
        gray[p] = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(float)
    # Source-only windows around the already authenticated Ad7 external left
    # boundary. Fixed y=430 remains inside the same visible print on all frames.
    centers = {230506752: 1488, 230507776: 1717, 230508032: 1800}
    measurements = {}
    for p in PTS:
        lo, hi = centers[p] - 110, min(1919, centers[p] + 110)
        result = {}
        for half in (0, 2):
            profile = gray[p][430-half:431+half, lo:hi+1].mean(axis=0)
            for smooth in (0., 2.5):
                result[f'rows{2*half+1}_smooth{smooth}'] = quantile(profile, lo, -1., smooth)
        measurements[p] = result['rows5_smooth2.5']
        emit('same_Ad7_external_left_gray_edge', pts=p, source_scan_y=430,
            diagnostic_window=[lo, hi], measurement=result,
            TOP_CONTACT_not_used=True, physical_edge_paint_and_codec_not_removed=True)
    for pa, pb in zip(PTS[:-1], PTS[1:]):
        dt = (clock[pb]['accepted_official_pts'] - clock[pa]['accepted_official_pts']) / 12800.
        displacement = measurements[pb]['q10_q50_q90'][1] - measurements[pa]['q10_q50_q90'][1]
        velocity = displacement / dt
        emit('observed_same_edge_displacement', source_pair=[pa, pb], seconds=dt,
            displacement_px=float(displacement), velocity_px_per_second=float(velocity),
            equivalent_rectangular_shutter_seconds=[float(measurements[p]['width'] / (.8 * abs(velocity))) for p in (pa, pb)],
            equivalent_is_not_shutter_measurement=True)
    for i in (25, 28):
        row = ice[i]
        result = []
        for sample in row['source_search_support']:
            y, lo, hi = sample['manual_source_bracket']
            profile = np.asarray(sample['gray_profile'])
            original_q = quantile(profile, lo, 1., 2.5)
            np.testing.assert_array_equal(original_q['q10_q50_q90'], sample['source_q10_q50_q90_axis'])
            single = gray[int(row['pts'])][y, lo:hi+1]
            result.append(dict(original_fixed_y=y, diagnostic_window=[lo, hi],
                five_rows_smooth2p5=original_q,
                one_row_smooth2p5=quantile(single, lo, 1., 2.5),
                five_rows_unsmoothed=quantile(profile, lo, 1., 0.)))
        emit('original_F3_positive_gray_gradient_mass', row_index=i, id=row['id'],
            split=row['split'], samples=result, original_sigma=row['sigma_px'],
            q50_latent_edge_precision_certified=False)
    # Only source correspondences: no camera state or reprofiled depth needed.
    saved_path = original.DIRECTORY / 'complete_state_UNACCEPTED.npz'
    with np.load(saved_path, allow_pickle=False) as z: saved = {k: z[k].copy() for k in z.files}
    data = original.secondary_inputs(original.read(original.DIRECTORY.parent.parent / 'input_manifest.json'), saved)
    for pa, pb in zip(PTS[:-1], PTS[1:]):
        ids = (data['pa'] == pa) & (data['pb'] == pb)
        if not np.any(ids):
            emit('original_motion_pair_unavailable', source_pair=[pa, pb]); continue
        a, b = data['a'][ids], data['b'][ids]
        near = (a[:, 1] >= 330) & (a[:, 1] <= 550) & (abs(a[:, 0]-centers[pa]) < 150)
        d = b - a
        emit('original_observed_motion', source_pair=[pa, pb], points=len(a),
            near_same_Ad7_points=int(near.sum()),
            all_delta_xy_p10_p50_p90=np.quantile(d, [.1, .5, .9], axis=0),
            near_delta_xy_p10_p50_p90=None if not near.any() else np.quantile(d[near], [.1, .5, .9], axis=0),
            original_sigma_profile_roles_unchanged=True)


if __name__ == '__main__': main()
