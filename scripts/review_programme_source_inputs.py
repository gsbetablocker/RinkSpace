"""One ordered original-video scan for actual ice/board pixel review.

One CUDA stream reads independent native RGB profiles. Source identities and
temporal admission are deliberately separate from candidate measurements.
No source image cache, solver, fitted residual, or video is produced.
"""
from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from multiprocessing import shared_memory
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src'), str(ROOT / 'calibration/program_tracking')]

import cv2
import numpy as np

from rinkspace.programme_source_stream import FrozenProgrammeSource
from calibration.program_tracking.render_result import PersonMasks, DEFAULT_MASKS, DEFAULT_PEOPLE

IMAGE_BYTES = 1080 * 1920 * 3
MASK_BYTES = 1080 * 1920
_READER = None


def _initialize_worker(scene, source_sha, time_base, roles, profile_backend='cpu'):
    global _READER
    cv2.setNumThreads(1)
    from rinkspace.vision_programme_ice_review import IceProgrammeSourceReader
    _READER = IceProgrammeSourceReader(scene, source_sha256=source_sha,
        time_base=time_base, role_contract=roles, profile_backend=profile_backend)


def _ice_exclusion(mask, bands):
    """Reject above measured board/ice contact; never output interpolated UV."""
    output = mask.copy()
    for band in bands:
        bottom = np.asarray(band['yellow_lower_uv'])
        for first, second in zip(bottom[:-1], bottom[1:]):
            if second[0] - first[0] > 29:
                continue
            xs = np.arange(int(np.ceil(first[0])), int(np.floor(second[0])) + 1)
            ys = np.interp(xs, [first[0], second[0]], [first[1], second[1]])
            for x, y in zip(xs, ys):
                output[:min(1080, int(np.ceil(y)) + 9), x] = True
    return output


def _observed_ice_region(bands):
    """Visible ice below same-frame measured board contact, with gaps unknown."""
    output = np.zeros((1080, 1920), bool)
    for band in bands:
        bottom = np.asarray(band['yellow_lower_uv'])
        for first, second in zip(bottom[:-1], bottom[1:]):
            if second[0] - first[0] > 29:
                continue
            xs = np.arange(int(np.ceil(first[0])), int(np.floor(second[0])) + 1)
            ys = np.interp(xs, [first[0], second[0]], [first[1], second[1]])
            for x, y in zip(xs, ys):
                output[min(1080, int(np.ceil(y)) + 9):, x] = True
    return output


def _measure_slot(name, frame, old_uv):
    shared = shared_memory.SharedMemory(name=name)
    try:
        image = np.ndarray((1080, 1920, 3), np.uint8, buffer=shared.buf, offset=0)
        exclusion = np.ndarray((1080, 1920), np.bool_, buffer=shared.buf, offset=IMAGE_BYTES)
        return _measure_arrays(image, exclusion, frame, old_uv)
    finally:
        shared.close()


def _measure_arrays(image, exclusion, frame, old_uv):
    from calibration.program_tracking.source_board_measurements import (
        measure_board_source_frame, board_bracket_candidates)
    from calibration.program_tracking.far_rail_source_components import source_exclusion
    if True:
        exclusion = source_exclusion(image, exclusion)
        board = measure_board_source_frame(image, exclusion, frame['source_binding'])
        bands = board['boundary_candidates']
        ice_mask = _ice_exclusion(exclusion, bands)
        ice = _READER.read(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), frame['actual_pts'],
            frame['camera25'], ice_mask, shot_id=frame['source_binding']['shot_id'])
        old_uv = np.asarray(old_uv, np.float64).reshape(-1, 2)
        # Category is a source audit disposition, never automatic admission.
        # 0 requires identity review, 1 person/overlay, 2 visibly above board,
        # 3 board patch requiring static material/temporal identity review.
        category = np.zeros(len(old_uv), np.uint8)
        pixel = np.rint(old_uv).astype(int)
        bounded = (pixel >= 0).all(1) & (pixel < [1920, 1080]).all(1)
        category[~bounded] = 1
        category[bounded & exclusion[np.clip(pixel[:, 1], 0, 1079), np.clip(pixel[:, 0], 0, 1919)]] = 1
        bracket = board_bracket_candidates(old_uv, bands, exclusion)
        category[(category == 0) & (bracket >= 0)] = 3
        for band in bands:
            top = np.asarray(band['blue_upper_uv'])
            inside = (old_uv[:, 0] >= top[0, 0]) & (old_uv[:, 0] <= top[-1, 0])
            y = np.interp(old_uv[:, 0], top[:, 0], top[:, 1])
            category[(category == 0) & inside & (old_uv[:, 1] < y - 8)] = 2
        return dict(ice=ice, bands=bands, old_category=category,
            actual_pts=frame['actual_pts'], native_pts=frame['native_pts'],
            source_binding=frame['source_binding'])


def _write_chunk(output, number, records):
    arrays = {}
    for prefix, keys in (
        ('query', ('curve', 'uv', 'normal', 'interval', 'status')),
        ('candidate', ('uv', 'sides', 'query', 'flags'))):
        counts = [len(r['ice'][prefix + '_' + keys[0]]) for r in records]
        arrays[prefix + '_offsets'] = np.r_[0, np.cumsum(counts)].astype(np.int64)
        for key in keys:
            name = prefix + '_' + key
            arrays[name] = np.concatenate([r['ice'][name] for r in records])
    arrays['curve_status'] = np.stack([r['ice']['curve_status'] for r in records])
    if all('candidate_side_quantiles_uv' in r['ice'] for r in records):
        arrays['candidate_side_quantiles_uv'] = np.concatenate([
            r['ice']['candidate_side_quantiles_uv'] for r in records])
    arrays['actual_pts'] = np.array([r['actual_pts'] for r in records], np.int64)
    arrays['native_pts'] = np.array([r['native_pts'] for r in records], np.int64)
    arrays['BGR_sha256'] = np.array([r['source_binding']['decoded_native_BGR_sha256'] for r in records])
    arrays['RGB_sha256'] = np.array([r['ice']['decoded_native_RGB_sha256'] for r in records])
    arrays['shot_id'] = np.array([r['source_binding']['shot_id'] for r in records])
    arrays['old_point_offsets'] = np.r_[0, np.cumsum([len(r['old_category']) for r in records])]
    arrays['old_point_category'] = np.concatenate([r['old_category'] for r in records])
    bands = [b for r in records for b in r['bands']]
    arrays['frame_band_offsets'] = np.r_[0, np.cumsum([len(r['bands']) for r in records])]
    arrays['band_point_offsets'] = np.r_[0, np.cumsum([len(b['blue_upper_uv']) for b in bands])]
    arrays['board_upper_uv'] = np.concatenate([b['blue_upper_uv'] for b in bands]) if bands else np.empty((0, 2))
    arrays['board_lower_uv'] = np.concatenate([b['yellow_lower_uv'] for b in bands]) if bands else np.empty((0, 2))
    if 'selected_ice' in records[0]:
        rows = [row for r in records for row in r['selected_ice']['rows']]
        arrays['frame_ice_row_offsets'] = np.r_[0, np.cumsum([len(r['selected_ice']['rows']) for r in records])]
        arrays['ice_row_point_offsets'] = np.r_[0, np.cumsum([len(row['uv']) for row in rows])]
        arrays['selected_ice_uv'] = np.concatenate([np.asarray(row['candidate_sides'], float).mean(1) if row['physical_edge_role'] == 'paint_centerline' else np.asarray(row['uv'], float) for row in rows]) if rows else np.empty((0, 2))
        arrays['selected_ice_sides'] = np.concatenate([np.asarray(row['candidate_sides'], float) for row in rows]) if rows else np.empty((0, 2, 2))
        # Keep compact identities/provenance; actual UV stays in numerical arrays.
        meta = [{k: v for k, v in row.items() if k not in ('uv', 'candidate_sides', 'source_seed_reference')} for row in rows]
        arrays['selected_ice_metadata_json'] = json.dumps(meta, ensure_ascii=False, separators=(',', ':'))
        arrays['ice_identity_decisions_json'] = json.dumps([r['selected_ice']['decisions'] for r in records], ensure_ascii=False, separators=(',', ':'))
    target = output / f'source_pixels_{number:04d}.npz'
    if target.exists():
        raise ValueError('Do not replace existing original source measurements')
    np.savez_compressed(target, **arrays)
    return dict(path=str(target), sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        actual_exposures=len(records), first_actual_pts=int(arrays['actual_pts'][0]),
        last_actual_pts=int(arrays['actual_pts'][-1]),
        measured_candidates=len(arrays['candidate_uv']), reviewed_old_pixels=len(arrays['old_point_category']))


def _resume_chunks(output, programme, old_offsets, old_lookup):
    """Validate and reuse a complete prefix, never replace measured pixels."""
    chunks, authenticated = [], {}
    observed = 0
    for number, path in enumerate(sorted(output.glob('source_pixels_*.npz'))):
        if path.name != f'source_pixels_{number:04d}.npz':
            raise ValueError('Existing source chunks are not a continuous prefix')
        with np.load(path, allow_pickle=False) as saved:
            # Decode every ZIP member so a interrupted/invalid chunk cannot be
            # silently counted as source coverage.
            arrays = {key: saved[key] for key in saved.files}
        actual = arrays['actual_pts']
        if not np.array_equal(actual, programme.actual_inventory[observed:observed+len(actual)]):
            raise ValueError('Saved actual exposure clock changed')
        if not np.array_equal(actual, arrays['native_pts']):
            raise ValueError('Saved pixels are not exact representative exposures')
        expected_counts = [int(old_offsets[old_lookup[int(p)]+1]-old_offsets[old_lookup[int(p)]]) for p in actual]
        if not np.array_equal(np.diff(arrays['old_point_offsets']), expected_counts):
            raise ValueError('Saved old source pixels do not match current frozen inventory')
        authenticated.update({int(p): str(sha) for p, sha in zip(actual, arrays['BGR_sha256'])})
        chunks.append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            actual_exposures=len(actual), first_actual_pts=int(actual[0]), last_actual_pts=int(actual[-1]),
            measured_candidates=len(arrays['candidate_uv']), reviewed_old_pixels=len(arrays['old_point_category'])))
        observed += len(actual)
    return chunks, authenticated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--identity', required=True, type=Path)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--roles', type=Path)
    parser.add_argument('--current-points', required=True, type=Path)
    parser.add_argument('--ice-seeds', type=Path)
    parser.add_argument('--registered-user-stroke', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--profile-backend', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--start-pts', type=int)
    parser.add_argument('--stop-pts', type=int)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    if args.profile_backend == 'cuda':
        import torch
        torch.set_num_threads(1)
        if args.workers != 1:
            raise ValueError('CUDA source review uses one image stream, not a CPU process pool')
    programme = FrozenProgrammeSource(args.identity, args.source,
        ROOT / 'calibration/program_tracking/source_shots.json')
    roles = json.loads(args.roles.read_text(encoding='utf-8-sig')) if args.roles else None
    masks = PersonMasks(DEFAULT_MASKS, DEFAULT_PEOPLE)
    # Historical masks retain the user's excluded 17 pre-programme frames.
    # Authenticate that archive without adding them to this source scan.
    masks.validate_coverage(np.array(sorted(masks.boxes), np.int64))
    if not set(map(int, programme.native_pts)).issubset(masks.boxes):
        raise ValueError('Original mask archive omits an authorized programme frame')
    with np.load(args.current_points, allow_pickle=False) as data:
        old_pts, old_offsets, old_uv = data['actual_pts'], data['offsets'], data['uv']
    if not np.array_equal(old_pts, programme.actual_inventory):
        raise ValueError('Current frozen source points do not cover the original exposure clock')
    old_lookup = {int(p): i for i, p in enumerate(old_pts)}
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    existing = list(output.glob('source_pixels_*.npz'))
    if existing and not args.resume:
        raise ValueError('Use the existing measurements; do not duplicate completed source scans')
    if args.resume and args.start_pts is not None:
        raise ValueError('Resume authenticates the full existing source prefix')
    slots = ([shared_memory.SharedMemory(create=True, size=IMAGE_BYTES + MASK_BYTES) for _ in range(args.workers)]
        if args.profile_backend == 'cpu' else [])
    waiting = deque()
    free = deque(range(len(slots)))
    records, native_clock, started = [], [], time.monotonic()
    _initialize_worker(programme.scene, programme.source_sha256, programme.time_base, roles, args.profile_backend)
    inventory = _READER.inventory
    if existing:
        previous = json.loads((output/'source_measurements.json').read_text(encoding='utf8'))
        if (previous['source_sha256'] != programme.source_sha256
                or previous['time_base'] != programme.time_base or previous['inventory'] != inventory):
            raise ValueError('Resume cannot change the actual source or physical readout inventory')
    chunks, reused = _resume_chunks(output, programme, old_offsets, old_lookup) if existing else ([], {})
    selector = None
    if args.ice_seeds or args.registered_user_stroke:
        if existing:
            raise ValueError('Temporal identities must replay the saved source prefix before resume admission')
        from rinkspace.vision_ice_temporal_identity import (
            TemporalIceIdentitySelector, registered_user_stroke_seed)
        seeds = json.loads(args.ice_seeds.read_text(encoding='utf8')) if args.ice_seeds else []
        if isinstance(seeds, dict):
            seeds = seeds['records']
        if args.registered_user_stroke:
            registered = json.loads(args.registered_user_stroke.read_text(encoding='utf-8-sig'))
            pts = int(registered['actual_exposure_pts'])
            shot = next(s for s in programme.shots if int(s['start_pts']) <= pts < int(s['end_pts']))
            seeds.append(registered_user_stroke_seed(args.registered_user_stroke,
                source_sha256=programme.source_sha256, time_base=programme.time_base,
                shot_id=shot['id'], curve_id=registered['feature_id']+':segment0'))
        selector = TemporalIceIdentitySelector(inventory, source_sha256=programme.source_sha256,
            time_base=programme.time_base, seed_registry=seeds)
    descriptor = dict(schema='rinkspace.complete-programme-actual-source-measurements.v1',
        source_sha256=programme.source_sha256, source_path=str(programme.source_path),
        time_base=programme.time_base, camera_identity=str(programme.identity_path),
        inventory=inventory, measurements_not_automatic_admission=True,
        solver_started=False, all_frames_visually_reviewed=False,
        actual_source_profiles_ready=False, source_review_complete=False,
        profile_backend=args.profile_backend, cpu_threads=1,
        native_RGB_interpolation_grid_px=1./32., chunks=chunks,
        reused_actual_exposures=len(reused))
    (output / 'source_measurements.json').write_text(json.dumps(descriptor, ensure_ascii=False), encoding='utf8')

    def record(result, image, mask):
        if selector is not None:
            result['selected_ice'] = selector.consume(cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                result['ice'], _ice_exclusion(mask, result['bands']),
                source_ice_scope_mask=_observed_ice_region(result['bands']))
        records.append(result)
        if len(records) == 128:
            chunks.append(_write_chunk(output, len(chunks), records))
            records.clear()
            descriptor.update(chunks=chunks, actual_exposures=sum(c['actual_exposures'] for c in chunks))
            (output/'source_measurements.json').write_text(json.dumps(descriptor, ensure_ascii=False), encoding='utf8')
            print(json.dumps(dict(actual_exposures=sum(c['actual_exposures'] for c in chunks),
                last_pts=chunks[-1]['last_actual_pts'], elapsed_s=round(time.monotonic()-started, 1))), flush=True)

    def consume():
        slot, future = waiting.popleft()
        result = future.result()
        image = np.ndarray((1080, 1920, 3), np.uint8, buffer=slots[slot].buf)
        mask = np.ndarray((1080, 1920), np.bool_, buffer=slots[slot].buf, offset=IMAGE_BYTES)
        record(result, image, mask)
        free.append(slot)

    try:
        if args.profile_backend == 'cuda':
            if reused:
                print(json.dumps(dict(reused_actual_exposures=len(reused),
                    backend='one_CUDA_stream', cpu_threads=1)), flush=True)
            for frame in programme.frames(start_pts=args.start_pts, stop_pts=args.stop_pts, cpu_threads=1):
                native_clock.append((frame['native_pts'], frame['actual_pts'],
                    frame['source_binding']['decoded_native_BGR_sha256']))
                if not frame['representative']:
                    continue
                if frame['actual_pts'] in reused:
                    if reused[frame['actual_pts']] != frame['source_binding']['decoded_native_BGR_sha256']:
                        raise ValueError('Existing measured exposure differs from the decoded original source')
                    continue
                image = frame.pop('raw_bgr')
                mask = masks.static_scene_exclusion(frame['native_pts'])[0]
                i = old_lookup[frame['actual_pts']]
                result = _measure_arrays(image, mask, frame, old_uv[old_offsets[i]:old_offsets[i+1]])
                record(result, image, mask)
        else:
          with ProcessPoolExecutor(max_workers=args.workers, initializer=_initialize_worker,
                initargs=(programme.scene, programme.source_sha256, programme.time_base, roles, 'cpu')) as pool:
            for frame in programme.frames(start_pts=args.start_pts, stop_pts=args.stop_pts):
                native_clock.append((frame['native_pts'], frame['actual_pts'],
                    frame['source_binding']['decoded_native_BGR_sha256']))
                if not frame['representative']:
                    continue
                if frame['actual_pts'] in reused:
                    if reused[frame['actual_pts']] != frame['source_binding']['decoded_native_BGR_sha256']:
                        raise ValueError('Saved source exposure pixels changed')
                    continue
                if not free:
                    consume()
                slot = free.popleft()
                image = np.ndarray((1080, 1920, 3), np.uint8, buffer=slots[slot].buf)
                mask = np.ndarray((1080, 1920), np.bool_, buffer=slots[slot].buf, offset=IMAGE_BYTES)
                image[:] = frame.pop('raw_bgr')
                mask[:] = masks.static_scene_exclusion(frame['native_pts'])[0]
                i = old_lookup[frame['actual_pts']]
                waiting.append((slot, pool.submit(_measure_slot, slots[slot].name, frame,
                    old_uv[old_offsets[i]:old_offsets[i+1]])))
            while waiting:
                consume()
        if records:
            chunks.append(_write_chunk(output, len(chunks), records))
        np.savez_compressed(output / 'exact_native_clock.npz',
            native_pts=np.array([r[0] for r in native_clock], np.int64),
            actual_pts=np.array([r[1] for r in native_clock], np.int64),
            BGR_sha256=np.array([r[2] for r in native_clock]))
        observed = sum(c['actual_exposures'] for c in chunks)
        complete = (len(native_clock) == len(programme.native_pts)
            and observed == len(programme.actual_inventory))
        descriptor.update(chunks=chunks, native_presentations=len(native_clock),
            actual_exposures=observed, all_native_frames_decoded=complete,
            actual_source_profiles_ready=complete, source_review_complete=False,
            elapsed_s=time.monotonic()-started)
        (output / 'source_measurements.json').write_text(json.dumps(descriptor, ensure_ascii=False), encoding='utf8')
        print(json.dumps(dict(native_presentations=len(native_clock), actual_exposures=observed,
            actual_source_profiles_ready=complete, source_review_complete=False)), flush=True)
    finally:
        for slot in slots:
            slot.close()
            slot.unlink()


if __name__ == '__main__':
    main()
