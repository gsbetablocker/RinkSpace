"""Compare CPU/CUDA source samplers on unchanged, real native RGB queries."""
from __future__ import annotations
import json
import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'src'), str(ROOT/'calibration/program_tracking')]
import cv2
import numpy as np
from rinkspace.vision_source_shapes import _MaterialProfileFrame
from rinkspace.vision_programme_ice_review import _batched_actual_profiles, _actual_profile_candidates
from rinkspace.vision_source_cuda_profiles import batched_actual_profiles_cuda


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--write-cpu-reference', type=Path)
    parser.add_argument('--cpu-reference', type=Path)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    base = Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/diagnostics/current56_shared_active_set_filter_20261009/systematic_continuation_20261010/full_programme_strict_source_review_20261012/C_continuity_probe')
    descriptor = json.loads((base/'source_measurements.json').read_text(encoding='utf8'))
    with np.load(base/'source_pixels_0000.npz', allow_pickle=False) as saved:
        count = int(saved['query_offsets'][1])
        queries = {key: saved['query_'+key][:count] for key in ('uv', 'normal', 'interval', 'curve')}
        pts = int(saved['actual_pts'][0])
    raw = cv2.imread('D:/CodexData/RinkSpace_Data/events/ankara_2026/evidence/user_video_review_20261009/a13_complete_programme/centre_C_04_08_44_933/original_presentation_native_229246976.png')
    binding = dict(source_sha256=descriptor['source_sha256'], source_pts=pts,
        time_base='1/15360', shot_id='223403520:R0402_034', native_hw=[1080, 1920])
    frame = _MaterialProfileFrame(cv2.cvtColor(raw, cv2.COLOR_BGR2RGB), binding)
    inventory = descriptor['inventory']
    signals = np.array([inventory[int(ci)]['material_signal'] for ci in queries['curve']])
    params = (frame, queries['uv'], queries['normal'], queries['interval'], signals)
    began = time.perf_counter(); cpu = _batched_actual_profiles(*params); cpu_seconds = time.perf_counter()-began
    if args.write_cpu_reference:
        np.savez_compressed(args.write_cpu_reference, offsets=cpu[0], profiles=cpu[1],
            smoothed=cpu[2], valid=cpu[3], luminance=frame.luminance,
            query_uv=queries['uv'], query_normal=queries['normal'],
            opencv_version=cv2.__version__)
        print(json.dumps(dict(opencv_version=cv2.__version__, actual_pts=pts,
            reference=str(args.write_cpu_reference), queries=count)), flush=True)
        return
    import torch
    torch.set_num_threads(1)
    reference_version = cv2.__version__
    if args.cpu_reference:
        with np.load(args.cpu_reference, allow_pickle=False) as reference:
            if not np.array_equal(reference['query_uv'], queries['uv']):
                raise ValueError('Actual source queries changed')
            cpu = tuple(reference[k].copy() for k in ('offsets', 'profiles', 'smoothed', 'valid'))
            reference_version = str(reference['opencv_version'].item())
            if not np.array_equal(reference['luminance'], frame.luminance):
                raise ValueError('Native integer source luminance operator changed')
    gpu = batched_actual_profiles_cuda(*params)
    began = time.perf_counter(); gpu = batched_actual_profiles_cuda(*params); gpu_seconds = time.perf_counter()-began
    count_changes, differences = [], []
    candidate_began = time.perf_counter()
    for qi, ci in enumerate(queries['curve']):
        row = inventory[int(ci)]
        if row['operator'] == 'unresolved':
            continue
        values = []
        for reading in (cpu, gpu):
            offset, profile, smooth, valid = reading
            mask = valid[qi]
            values.append(_actual_profile_candidates(queries['uv'][qi], queries['normal'][qi],
                offset[qi, mask], profile[qi, mask], smooth[qi, mask],
                operator=row['operator'], signal=row['material_signal'], transition=row['material_transition']))
        if len(values[0]) != len(values[1]):
            count_changes.append(qi)
        else:
            differences.extend(float(np.max(np.abs(a[0]-b[0]))) for a, b in zip(*values))
    print(json.dumps(dict(device=torch.cuda.get_device_name(), real_source_pts=pts,
        cpu_reference_opencv=reference_version,
        candidate_comparison_seconds=time.perf_counter()-candidate_began,
        source_queries=count, cpu_sampler_seconds=cpu_seconds, cuda_sampler_seconds=gpu_seconds,
        profile_max_abs_difference=float(np.max(np.abs(cpu[1]-gpu[1]))),
        smoothed_max_abs_difference=float(np.max(np.abs(cpu[2]-gpu[2]))),
        candidate_count_changed_queries=count_changes,
        largest_matching_candidate_difference_native_px=max(differences, default=0.),
        cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated()), ensure_ascii=False), flush=True)
    if args.cpu_reference:
        if count_changes or max(differences, default=0.) > .001:
            raise ValueError('CUDA changed original actual source candidate readout')
        return
    qi, si = np.unravel_index(np.argmax(np.abs(cpu[1]-gpu[1])), cpu[1].shape)
    tangent = np.array([-queries['normal'][qi, 1], queries['normal'][qi, 0]])
    xy = (queries['uv'][qi]+queries['normal'][qi]*cpu[0][qi, si]
        + tangent[None]*np.arange(-2, 3)[:, None]).astype(np.float32)
    a, b = cv2.convertMaps(xy[:, 0:1], xy[:, 1:2], cv2.CV_16SC2)
    fixed = a[:, 0].astype(float)+np.c_[b[:, 0]%32, b[:, 0]//32]/32.
    gpu_debug = batched_actual_profiles_cuda(*params, _debug_query=(qi, si))[-1]
    profile_image = frame.luminance.astype(np.float32)
    direct = cv2.remap(profile_image, xy[:, 0:1], xy[:, 1:2], cv2.INTER_LINEAR)[:, 0]
    print(json.dumps(dict(max_query=int(qi), offset_index=int(si), signal=str(signals[qi]),
        CPU_profile_value=float(cpu[1][qi, si]), CUDA_profile_value=float(gpu[1][qi, si]),
        actual_xy=xy.tolist(), cv_fixed_xy=fixed.tolist(), nearest_table_xy=(np.rint(xy*32)/32).tolist(),
        GPU_debug=gpu_debug, CPU_direct_samples=direct.tolist())), flush=True)


if __name__ == '__main__':
    main()
