"""Show the saved programme root trajectory on its exact source and rink.

The renderer does no trajectory filtering, gap filling or body fitting. One
cosmetic display camera projects every layer of the same actual exposure;
the saved physical camera/body are retained. Native display aliases remain.
"""
from __future__ import annotations

import argparse
from collections import Counter
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing as mp
from multiprocessing import shared_memory
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src'), str(Path(__file__).parent)]
from runtime.bootstrap import add_geometry_dependencies
add_geometry_dependencies()
import av
import cv2
import numpy as np
from rinkspace.storage import AssetStore

STORAGE = AssetStore(ROOT)

from render_result import (SceneRenderer, PersonMasks, DEFAULT_MASKS, DEFAULT_PEOPLE,
                           DEFAULT_SHOTS, decode_selected, _ffmpeg_executable, _available_encoder)
from board_grid_overlay import make_board_grid_renderer
from display_camera_smoothing import display_model_probes, smooth_display_cameras
from projection import project_fast
from calibration.program_tracking.body_tracking.pelvis_proxy_programme import PlanView, source_time
from calibration.program_tracking.current_source_points import CurrentSourcePoints

SIZE = (2580, 1160)
GOLD = (25, 205, 250)
NATIVE_SHAPE = (1080, 1920, 3)
OUTPUT_SHAPE = (SIZE[1], SIZE[0], 3)


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key].copy() for key in z.files}


class ProgrammeRootView:
    def __init__(self, camera_path, body_path, scene_path, current_source_points_path=None, *, display_camera_smoothing=True):
        self.camera = load(camera_path)
        self.body = load(body_path)
        self.unit = float(self.body['metres_per_scene_unit'])
        if not np.isfinite(self.unit) or self.unit <= 0:
            raise ValueError('Body result must declare its positive uniform scene-to-metre scale')
        self.pts = self.camera['pts'].astype(np.int64)
        self.reps = self.camera['exposure_representative_pts'].astype(np.int64)
        self.current_source_points = (None if current_source_points_path is None else
                                      CurrentSourcePoints(current_source_points_path, camera_path))
        actual = self.body['actual_pts'].astype(np.int64)
        if np.any(np.diff(actual) <= 0) or np.any(np.diff(self.pts) <= 0):
            raise ValueError('Ordered actual-body and native-camera timestamps required')
        rows = np.searchsorted(actual, self.reps)
        if np.any(rows >= len(actual)) or not np.array_equal(actual[rows], self.reps):
            raise ValueError('Body must explicitly supply every camera exposure in this programme')
        if not np.array_equal(self.body['camera25'][rows], self.camera['camera25']):
            raise ValueError('Body was fitted against a different camera; rebind the fit first')
        self.P = self.body['P_world_m'][rows].astype(float)
        self.G = self.P.copy()
        self.G[:, 2] = 0.
        self.valid = np.asarray(self.body.get('valid', np.isfinite(self.body['P_world_m']).all(axis=1)), bool)[rows]
        self.valid &= np.isfinite(self.P).all(axis=1)
        self.inferred = np.asarray(self.body.get('inferred', np.zeros(len(actual), bool)), bool)[rows]
        actual_connected = np.asarray(self.body.get('connect_from_previous', np.r_[False, np.diff(actual) <= 1536]), bool)
        self.connect = actual_connected[rows].copy()
        self.connect[1:] |= self.reps[1:] == self.reps[:-1]
        self.connect[1:] &= self.valid[:-1] & self.valid[1:]
        self.connect[0] = False
        self.scene = json.loads(Path(scene_path).read_text(encoding='utf-8-sig'))
        BoardRenderer = make_board_grid_renderer(SceneRenderer, scene_unit_m=self.unit)
        self.renderer = BoardRenderer(self.scene,
            graphics_regions=((28,24,554,253),(1690,944,1865,1000)))
        self.display_camera_smoothing_enabled=bool(display_camera_smoothing)
        self.display_camera25=self.camera['camera25'].copy()
        self.display_camera_smoothing_stats=dict(enabled=False,
            raw_camera_world_and_trajectory_artifacts_unmodified=True,
            display_camera_is_cosmetic_not_new_solver_state=True)
        if self.display_camera_smoothing_enabled:
            shots_path=STORAGE.resolve(DEFAULT_SHOTS)
            shots=json.loads(shots_path.read_text(encoding='utf8'))
            metadata=json.loads(str(self.camera['metadata_json']))
            if (shots['source_sha256_from_timeline']!=metadata['source_sha256']
                    or shots['source_time_base']!=metadata['source_time_base']):
                raise ValueError('Display cuts must belong to the exact physical-camera source')
            probes=display_model_probes(self.renderer.model_paths,self.renderer.advertisement_paths)
            self.display_camera25,self.display_camera_smoothing_stats=smooth_display_cameras(
                self.pts,self.camera['camera25'],self.reps,
                [(s['start_pts'],s['end_pts'])for s in shots['shots']],probes,max_shift_px=1.)
            self.display_camera_smoothing_stats.update(
                source_cuts=dict(path=str(shots_path),sha256=hashlib.sha256(shots_path.read_bytes()).hexdigest()),
                all_grid_ads_board_and_P_G_use_same_display_camera=True,
                fixed_all_path_endpoint_midpoint_and_quarter_probes_used=True,
                all_model_paths_represented_without_fit_or_confidence_selection=True)
        self.plan = PlanView(self.scene, 0., board_paths=self.renderer.board_grid.paths)
        if self.renderer.path_counts != self.plan.path_counts:
            raise ValueError('Source and plan must draw the same complete scene')
        self.base = self.plan.base.copy()
        self.base[510:] = 248
        for y, value in [(540, '人物整体位置与连续轨迹'),
                         (588, 'P：估计骨盆中心'),
                         (632, 'G：P 的世界竖直投影'),
                         (690, '蓝线：骨盆垂直投影轨迹'),
                         (734, '橙线：前后画面推理的位置'),
                         (806, '当前修正候选，尚未全程验收')]:
            self.base = self.plan.text(self.base, (24, y), value, small=y != 540)
        self.trail = self.base.copy()
        self.trail_index = -1
        self.header = np.full((70, SIZE[0], 3), 20, np.uint8)
        self.header = self.plan.text(self.header, (18, 14),
            '全节目网格与人体轨迹（修正中）', color=(240,240,240))
        self.status_patches = {}
        for key, label in [('inferred', '前后画面推理'),
                           ('estimated', '画面与连续运动共同估计'),
                           ('unknown', '当前整体位置尚未恢复')]:
            patch = np.full((64, 660, 3), 248, np.uint8)
            self.status_patches[key] = self.plan.text(patch, (24, 4), label, small=True)
        self.masks = PersonMasks(DEFAULT_MASKS, DEFAULT_PEOPLE)
        self.lookup = {int(p): i for i, p in enumerate(self.pts)}
        self.input_paths = [Path(camera_path), Path(body_path), Path(scene_path)]
        if current_source_points_path is not None:
            self.input_paths.append(Path(current_source_points_path))

    def advance(self, end):
        if end < self.trail_index:
            self.trail = self.base.copy()
            self.trail_index = -1
        for i in range(max(1, self.trail_index + 1), end + 1):
            if not self.connect[i]:
                continue
            uv = self.plan.mapxy(self.G[i-1:i+1, :2] / self.unit)
            okay, a, b = cv2.clipLine((0, 0, 660, 510), tuple(uv[0]), tuple(uv[1]))
            if okay:
                inferred = self.inferred[i-1] or self.inferred[i]
                cv2.line(self.trail, a, b, (40, 150, 225) if inferred else (215, 120, 35), 2, cv2.LINE_AA)
        self.trail_index = end

    def draw(self, native, pts):
        i = self.lookup[int(pts)]
        cam = self.display_camera25[i]
        try:
            mask, _ = self.masks.get(int(pts), require_sam2=True)
        except ValueError:
            mask = None
        left, _ = self.renderer.draw(native, cam, mask, title=None)
        self.advance(i)
        right = self.trail.copy()
        if self.valid[i]:
            uv, _ = project_fast(cam, np.vstack((self.P[i], self.G[i])) / self.unit)
            if np.isfinite(uv).all() and np.max(np.abs(uv)) < 1e7:
                a, b = (tuple(np.rint(v).astype(int)) for v in uv)
                okay, aa, bb = cv2.clipLine((0, 0, 1920, 1080), a, b)
                if okay:
                    cv2.line(left, aa, bb, GOLD, 2, cv2.LINE_AA)
                for xy, label, marker in [(a, 'P', cv2.MARKER_SQUARE), (b, 'G', cv2.MARKER_DIAMOND)]:
                    if 0 <= xy[0] < 1920 and 0 <= xy[1] < 1080:
                        cv2.drawMarker(left, xy, GOLD, marker, 18, 2, cv2.LINE_AA)
                        cv2.putText(left, label, (xy[0]+14, xy[1]-8), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,0,0), 4, cv2.LINE_AA)
                        cv2.putText(left, label, (xy[0]+14, xy[1]-8), cv2.FONT_HERSHEY_SIMPLEX, .7, GOLD, 2, cv2.LINE_AA)
            q = self.plan.mapxy(self.G[i:i+1, :2] / self.unit)[0]
            if 0 <= q[0] < 660 and 0 <= q[1] < 510:
                cv2.circle(right, tuple(q), 6, GOLD, -1, cv2.LINE_AA)
                cv2.circle(right, tuple(q), 7, (30,30,30), 1, cv2.LINE_AA)
        status = ('inferred' if self.inferred[i] else 'estimated') if self.valid[i] else 'unknown'
        right[862:926] = self.status_patches[status]
        if self.current_source_points is not None:
            actual_pts = int(self.reps[i])
            self.current_source_points.draw(left, actual_pts)
            self.last_current_source_point_draw = (int(pts), actual_pts,
                len(self.current_source_points.points(actual_pts)))
        else:
            self.last_current_source_point_draw = None
        canvas = np.full((SIZE[1], SIZE[0], 3), 20, np.uint8)
        canvas[:70] = self.header
        canvas[70:1150, :1920] = left
        canvas[70:1150, 1920:] = right
        cv2.putText(canvas, f'|  source {source_time(int(pts))}', (760, 43),
                    cv2.FONT_HERSHEY_SIMPLEX, .85, (240,240,240), 2, cv2.LINE_AA)
        return canvas


# Each spawned process owns its renderer, source masks and incremental trail.
# Only slot numbers/PTS travel through multiprocessing pipes; pixels stay in
# the fixed-size shared ring.  A process receives increasing source times, and
# advance() draws any intervening trail segments before its assigned frame.
_WORKER_VIEW = None
_WORKER_NATIVE = None
_WORKER_OUTPUT = None
_WORKER_SHM = None


def _initialize_draw_worker(paths, native_name, output_name, slot_count, display_camera_smoothing=True):
    global _WORKER_VIEW, _WORKER_NATIVE, _WORKER_OUTPUT, _WORKER_SHM
    cv2.setNumThreads(1)
    _WORKER_SHM = (shared_memory.SharedMemory(name=native_name),
                   shared_memory.SharedMemory(name=output_name))
    _WORKER_NATIVE = np.ndarray((slot_count, *NATIVE_SHAPE), np.uint8,
                                buffer=_WORKER_SHM[0].buf)
    _WORKER_OUTPUT = np.ndarray((slot_count, *OUTPUT_SHAPE), np.uint8,
                                buffer=_WORKER_SHM[1].buf)
    _WORKER_VIEW = ProgrammeRootView(*paths,display_camera_smoothing=display_camera_smoothing)


def _draw_shared_chunk(items, verify_hash=False):
    started = time.perf_counter()
    hashes = []
    source_point_draws = []
    for slot, pts in items:
        canvas = _WORKER_VIEW.draw(_WORKER_NATIVE[slot], pts)
        if _WORKER_VIEW.last_current_source_point_draw is not None:
            source_point_draws.append(_WORKER_VIEW.last_current_source_point_draw)
        np.copyto(_WORKER_OUTPUT[slot], canvas)
        if verify_hash:
            hashes.append(hashlib.sha256(canvas).hexdigest())
    cache = getattr(_WORKER_VIEW.renderer, 'geometry_cache_info', None)
    return dict(pts=[int(p) for _, p in items], pid=os.getpid(),
                draw_seconds=time.perf_counter()-started,
                hashes=hashes, geometry_cache=cache() if cache else None,
                current_source_point_draws=source_point_draws)


def decode_programme_frames(source, selected):
    """Yield the exact ordered original native presentations, without aliases."""
    count = 0
    with av.open(str(STORAGE.source(source))) as container:
        stream = container.streams.video[0]
        container.seek(int(selected[0]), stream=stream, backward=True)
        for frame in container.decode(stream):
            if frame.pts is None or frame.pts < selected[0]:
                continue
            if frame.pts > selected[-1]:
                break
            if count >= len(selected) or int(frame.pts) != int(selected[count]):
                raise ValueError('Source native exposure clock changed')
            count += 1
            yield int(frame.pts), frame.to_ndarray(format='bgr24')
    if count != len(selected):
        raise ValueError(f'Source decode ended after {count}/{len(selected)} frames')


class OrderedSharedDraw:
    """Bounded Windows-safe drawing pool with ordered, zero-pickle pixels.

    A yielded canvas is valid until the caller resumes the iterator.  Thus the
    encoder consumes it before the same ring slot can receive a later source.
    Contiguous small tasks retain exact-camera alias cache locality.
    """
    def __init__(self, paths, workers=4, slots=None, chunk_size=4, verify_hash=False,
                 representative_by_pts=None, display_camera_smoothing=True):
        self.workers = int(workers)
        self.chunk_size = int(chunk_size)
        self.slots = int(slots or 2*self.workers*self.chunk_size)
        self.chunk_size = min(self.chunk_size, self.slots)
        if self.workers < 1 or self.slots < self.workers or self.chunk_size < 1:
            raise ValueError('Positive workers/chunks and at least one slot per worker required')
        self.paths = tuple(map(str, paths))
        self.verify_hash = verify_hash
        self.representative_by_pts = representative_by_pts
        self.display_camera_smoothing_enabled=bool(display_camera_smoothing)
        self.stats = dict(mode='ordered_process_pool_shared_memory', workers=self.workers,
                          slots=self.slots, chunk_size=self.chunk_size,
                          batch_slots=self.slots/self.chunk_size,
                          pixel_arrays_pickled=False, frames=0, worker_draw_seconds=0.,
                          confirmed_alias_families_kept_in_one_task=representative_by_pts is not None,
                          worker_latest_geometry_cache={}, worker_frame_counts={},
                          shared_memory_bytes=self.slots*(np.prod(NATIVE_SHAPE)+np.prod(OUTPUT_SHAPE)).item())

    def __enter__(self):
        self.native_memory = shared_memory.SharedMemory(create=True,
            size=self.slots*int(np.prod(NATIVE_SHAPE)))
        try:
            self.output_memory = shared_memory.SharedMemory(create=True,
                size=self.slots*int(np.prod(OUTPUT_SHAPE)))
        except BaseException:
            self.native_memory.close(); self.native_memory.unlink()
            raise
        self.native = np.ndarray((self.slots, *NATIVE_SHAPE), np.uint8,
                                 buffer=self.native_memory.buf)
        self.output = np.ndarray((self.slots, *OUTPUT_SHAPE), np.uint8,
                                 buffer=self.output_memory.buf)
        self.pool = ProcessPoolExecutor(max_workers=self.workers,
            mp_context=mp.get_context('spawn'), initializer=_initialize_draw_worker,
            initargs=(self.paths, self.native_memory.name, self.output_memory.name, self.slots,
                      self.display_camera_smoothing_enabled))
        return self

    def frames(self, source_frames):
        started = time.perf_counter()
        source = iter(source_frames)
        free = deque(range(self.slots)); pending = deque(); exhausted = False
        lookahead = None; next_group = None
        previous_submitted = None; previous_written = None
        point_frames = 0
        point_digest = hashlib.sha256()
        def source_group():
            nonlocal lookahead, exhausted
            try:
                first = lookahead if lookahead is not None else next(source)
            except StopIteration:
                exhausted = True
                return None
            lookahead = None
            group = [first]
            if self.representative_by_pts is not None:
                rep = self.representative_by_pts[int(first[0])]
                for item in source:
                    if self.representative_by_pts[int(item[0])] != rep:
                        lookahead = item
                        break
                    group.append(item)
                else:
                    exhausted = True
            return group
        def fill():
            nonlocal exhausted, previous_submitted, next_group
            while free and (not exhausted or next_group is not None):
                items = []
                while len(items) < self.chunk_size:
                    if next_group is None:
                        next_group = source_group()
                    if next_group is None:
                        break
                    if len(next_group) > self.slots:
                        raise ValueError('Confirmed alias family exceeds the bounded frame ring')
                    if len(next_group) > len(free):
                        break
                    for pts, native in next_group:
                        pts = int(pts)
                        if previous_submitted is not None and pts <= previous_submitted:
                            raise ValueError('Parallel drawing requires increasing native PTS')
                        if native.shape != NATIVE_SHAPE or native.dtype != np.uint8:
                            raise ValueError('Native shared slot requires exact BGR1920x1080 uint8')
                        slot = free.popleft(); np.copyto(self.native[slot], native)
                        items.append((slot, pts)); previous_submitted = pts
                    next_group = None
                if items:
                    pending.append((items, self.pool.submit(_draw_shared_chunk, items, self.verify_hash)))
                else:
                    break
        fill()
        while pending:
            items, future = pending.popleft(); result = future.result()
            if result['pts'] != [p for _, p in items]:
                raise ValueError('Worker returned a different source presentation sequence')
            point_draws = result['current_source_point_draws']
            if len(self.paths) == 4:
                if (len(point_draws) != len(items)
                        or [int(r[0]) for r in point_draws] != result['pts']
                        or (self.representative_by_pts is not None and
                            any(int(r[1]) != self.representative_by_pts[int(r[0])]
                                for r in point_draws))):
                    raise ValueError('Every drawn native frame requires its own current exposure points')
                for row in point_draws:
                    point_digest.update(np.asarray(row, np.int64).tobytes())
                point_frames += len(point_draws)
            key = str(result['pid'])
            self.stats['worker_draw_seconds'] += result['draw_seconds']
            self.stats['worker_latest_geometry_cache'][key] = result['geometry_cache']
            self.stats['worker_frame_counts'][key] = self.stats['worker_frame_counts'].get(key, 0)+len(items)
            for j, (slot, pts) in enumerate(items):
                if previous_written is not None and pts <= previous_written:
                    raise ValueError('Parallel output lost source order')
                previous_written = pts
                yield pts, self.output[slot], (result['hashes'][j] if self.verify_hash else None)
                self.stats['frames'] += 1
                free.append(slot)
            fill()
        self.stats['pipeline_seconds_including_decode_and_sink'] = time.perf_counter()-started
        if len(self.paths) == 4:
            self.stats['current_source_point_draw_frames'] = point_frames
            self.stats['current_source_point_draw_rows_sha256'] = point_digest.hexdigest()

    def __exit__(self, *exc):
        self.pool.shutdown(wait=True, cancel_futures=bool(exc[0]))
        self.native = self.output = None
        self.native_memory.close(); self.native_memory.unlink()
        self.output_memory.close(); self.output_memory.unlink()


def benchmark_draw(args, view, selected):
    """One real continuous replay: strict serial/process pixel equality."""
    selected = selected[:args.benchmark_frames]
    frames = list(decode_programme_frames(args.source, selected))
    serial = ProgrammeRootView(args.camera, args.body, args.scene, args.current_source_points,
                               display_camera_smoothing=view.display_camera_smoothing_enabled)
    expected = []
    started = time.perf_counter()
    for pts, native in frames:
        expected.append(hashlib.sha256(serial.draw(native, pts)).hexdigest())
    serial_seconds = time.perf_counter()-started
    actual = []
    started = time.perf_counter()
    with OrderedSharedDraw(view.input_paths, args.draw_workers,
                           args.draw_slots, args.draw_chunk_size, verify_hash=True,
                           representative_by_pts=dict(zip(map(int, view.pts), map(int, view.reps))),
                           display_camera_smoothing=view.display_camera_smoothing_enabled) as ring:
        for pts, canvas, digest in ring.frames(frames):
            actual.append((pts, digest))
        parallel_stats = ring.stats.copy()
    parallel_seconds = time.perf_counter()-started
    if [p for p, _ in actual] != selected.tolist() or [h for _, h in actual] != expected:
        raise ValueError('Real-source parallel replay differs from ordered serial pixels')
    _, encoder_name = _available_encoder(_ffmpeg_executable())
    return dict(native_frames=len(frames), start_pts=int(selected[0]), end_pts=int(selected[-1]),
                serial_draw_and_hash_seconds=serial_seconds,
                process_draw_and_hash_seconds_including_spawn_and_shutdown=parallel_seconds,
                speedup_including_spawn_and_shutdown=serial_seconds/parallel_seconds,
                exact_canvas_sha256_equality=True, exact_native_pts_order=True,
                selected_encoder_name=encoder_name, encoder_probe_only=True,
                no_source_pixels_pickled=True, no_frame_images_written=True,
                parallel=parallel_stats)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('camera', 'body', 'scene', 'source', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--select-pts', type=int, nargs='*', default=[])
    parser.add_argument('--video', action='store_true')
    parser.add_argument('--current-source-points', type=Path,
                        help='Exact saved-camera active source UV; current points only, without extra labels')
    parser.add_argument('--start-pts', type=int)
    parser.add_argument('--end-pts', type=int)
    parser.add_argument('--no-display-camera-smoothing',action='store_true',
                        help='Use raw saved cameras for a display-only comparison; default caps smoothing at 1 native pixel')
    parser.add_argument('--draw-workers', type=int, default=min(4, os.cpu_count() or 1),
                        help='Default up to 4 ordered drawing workers; 1 selects the serial diagnostic path')
    parser.add_argument('--draw-chunk-size', type=int, default=4)
    parser.add_argument('--draw-slots', type=int,
                        help='Shared native/output frame slots; default 2*workers*chunk-size')
    parser.add_argument('--benchmark-frames', type=int, default=0,
                        help='Compare one small continuous real-source serial/process replay and exit')
    args = parser.parse_args()
    if args.video and args.current_source_points is None:
        parser.error('Review video requires --current-source-points for its actual fitting inputs')
    if not args.output.resolve().is_relative_to(Path('D:/CodexData').resolve()):
        raise ValueError('Media output must remain in D:/CodexData')
    args.output.mkdir(parents=True, exist_ok=True)
    view = ProgrammeRootView(args.camera, args.body, args.scene, args.current_source_points,
                             display_camera_smoothing=not args.no_display_camera_smoothing)
    selected = view.pts[(view.pts >= (args.start_pts or view.pts[0])) &
                        (view.pts <= (args.end_pts or view.pts[-1]))]
    if not len(selected):
        raise ValueError('No programme native frames in requested scope')
    if args.benchmark_frames:
        if args.draw_workers <= 1:
            raise ValueError('Parallel benchmark requires --draw-workers >1')
        result = benchmark_draw(args, view, selected)
        (args.output/'drawing_pipeline_benchmark.json').write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return
    saved = []
    for p, native in decode_selected(args.source, sorted(set(args.select_pts))):
        name = args.output / f'{p}_grid_and_trajectory.png'
        if name.exists():
            raise FileExistsError(name)
        cv2.imwrite(str(name), view.draw(native, p))
        saved.append(str(name))
    count = 0
    video_stats = None
    if args.video:
        target = args.output / 'grid_and_trajectory_WORKING.mp4'
        if target.exists():
            raise FileExistsError(target)
        ffmpeg = _ffmpeg_executable()
        encoder, encoder_name = _available_encoder(ffmpeg)
        command = [ffmpeg, '-hide_banner', '-loglevel', 'error', '-f', 'rawvideo',
                   '-pixel_format', 'bgr24', '-video_size', f'{SIZE[0]}x{SIZE[1]}',
                   '-framerate', '60', '-i', 'pipe:0', '-an', *encoder,
                   '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(target)]
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        started = time.perf_counter()
        first_pts = last_pts = None
        pts_digest = hashlib.sha256()
        point_draw_digest = hashlib.sha256()
        point_actual_counts = {int(p): int(hi-lo) for p, lo, hi in zip(
            view.current_source_points.pts, view.current_source_points.offsets[:-1],
            view.current_source_points.offsets[1:])}
        point_rows = np.searchsorted(view.pts, selected)
        marker_counts = np.asarray([point_actual_counts[int(p)] for p in view.reps[point_rows]], np.int64)
        source_point_coverage = dict(native_presentations=len(selected),
            actual_exposures=len(np.unique(view.reps[point_rows])),
            native_presentations_with_points=int((marker_counts > 0).sum()),
            native_presentations_without_points=int((marker_counts == 0).sum()),
            first_native_pts=int(selected[0]), first_actual_pts=int(view.reps[point_rows[0]]),
            first_frame_points=int(marker_counts[0]),
            last_native_pts=int(selected[-1]), last_actual_pts=int(view.reps[point_rows[-1]]),
            last_frame_points=int(marker_counts[-1]),
            maximum_frame_points=int(marker_counts.max()),
            maximum_frame_points_native_pts=int(selected[np.argmax(marker_counts)]),
            only_current_exposure_points=True, missing_exposures_not_filled=True)
        ring_stats = None
        def write_frame(pts, canvas):
            nonlocal count, first_pts, last_pts
            if count >= len(selected) or pts != int(selected[count]):
                raise ValueError('Encoder input changed native source order')
            actual_pts = int(view.reps[point_rows[count]])
            point_draw_digest.update(np.asarray((pts, actual_pts, marker_counts[count]), np.int64).tobytes())
            # Buffered pipe accepts a contiguous buffer directly: no extra
            # full-frame bytes allocation and no interprocess pixel pickle.
            process.stdin.write(memoryview(canvas).cast('B'))
            pts_digest.update(np.int64(pts).tobytes())
            if first_pts is None:
                first_pts = pts
            last_pts = pts
            count += 1
            if count % 600 == 0:
                print(f'Rendered {count}/{len(selected)}', flush=True)
        try:
            decoded = decode_programme_frames(args.source, selected)
            if args.draw_workers > 1:
                with OrderedSharedDraw(view.input_paths, args.draw_workers,
                                       args.draw_slots, args.draw_chunk_size,
                                       representative_by_pts=dict(zip(map(int, view.pts), map(int, view.reps))),
                                       display_camera_smoothing=view.display_camera_smoothing_enabled) as ring:
                    for pts, canvas, _ in ring.frames(decoded):
                        write_frame(pts, canvas)
                    ring_stats = ring.stats.copy()
            else:
                for pts, native in decoded:
                    canvas = view.draw(native, pts)
                    if view.last_current_source_point_draw != (pts, int(view.reps[point_rows[count]]),
                                                              int(marker_counts[count])):
                        raise ValueError('Serial frame did not draw its current actual source points')
                    write_frame(pts, canvas)
            process.stdin.close()
            code = process.wait(timeout=60)
            if code or count != len(selected):
                raise RuntimeError(process.stderr.read().decode(errors='replace'))
            if ring_stats is not None and (ring_stats.get('current_source_point_draw_frames') != count
                    or ring_stats.get('current_source_point_draw_rows_sha256') != point_draw_digest.hexdigest()):
                raise ValueError('Parallel point overlay omitted or changed a current source frame')
        except BaseException:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=30)
            raise
        video_stats = dict(selected_encoder_name=encoder_name, selected_encoder_args=encoder,
                           ffmpeg_executable=ffmpeg,
                           drawing_pipeline=ring_stats or dict(mode='serial', workers=1),
                           source_pts_first=first_pts, source_pts_last=last_pts,
                           encoded_input_native_pts_sha256=pts_digest.hexdigest(),
                           encoded_input_exact_source_order=True,
                           current_source_point_draw_coverage=dict(source_point_coverage,
                               all_native_frames_points_draw_invoked=True,
                               native_actual_marker_count_rows_sha256=point_draw_digest.hexdigest()),
                           wall_seconds_including_decode_draw_encode=time.perf_counter()-started)
        saved.append(str(target))
    record = dict(files=saved, native_video_frames=count,
        source_and_plan_all_geometry=view.renderer.path_counts,
        advertisement_ids=view.plan.advertisement_ids,
        camera_and_body_exposures_equal=True, trajectory_filtered_in_renderer=False,
        display_camera_smoothing=view.display_camera_smoothing_stats,
        metres_per_scene_unit=view.unit,
        video_production=video_stats,
        current_source_points=None if view.current_source_points is None else view.current_source_points.reference,
        whole_programme_accuracy_accepted=False,
        inputs=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in view.input_paths])
    (args.output/'render.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(files=saved,native_video_frames=count),ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
