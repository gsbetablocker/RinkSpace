"""Lightweight complete-shot review workers, with the existing draw/encoder.

Call pack_review while the authenticated parent FirstShotReview is alive.
Workers load a frozen scene and camera-only state, never a source bundle or
geometry factory. Video remains an explicit call after a same-pack three-frame
pixel-exact transport check. No fitting is performed in this module.
"""
from __future__ import annotations

from collections import deque
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import hashlib
import importlib.util
import json
import multiprocessing
import os
from pathlib import Path
import pickle
import sys
import time
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import ImageFont

from ..solver_runtime import configure_native_runtime
from .first_shot_fit import _CameraState
from .first_shot_review import DATA, SIZE, FirstShotReview, render_video, review_source_uv
from .official_static_surface_dataset import reference
from .render_result import SceneRenderer
from .board_grid_overlay import make_board_grid_renderer
from .review_metric_grid import (FIELD as METRIC_GRID_FIELD, json_sha as grid_json_sha,
                                 validate_metric_review_scene)
from .body_tracking.pelvis_proxy_programme import PlanView
from . import row_scene_projection, row_scene_projection_fast

LEGACY_SCHEMA = 'rinkspace.lightweight-complete-first-shot-review-pack.v1'
SCHEMA = 'rinkspace.lightweight-complete-first-shot-review-pack.v2'
_WORKER = None


def _sha(array):
    return hashlib.sha256(np.asarray(array).tobytes()).hexdigest()


def _json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode('utf8')).hexdigest()


def _grid_identity(pack):
    metric = pack.get('review_grid_conversion')
    if metric is None:
        return dict(policy='legacy_working_grid_comparison_only_not_final_metres',
                    scene_unit_working_m=pack['scene_unit_working_m'],
                    verified_metre_scale=False)
    return dict(policy='uniform_closest60x30_after_geometry',
                conversion=deepcopy(metric), conversion_sha256=pack['review_grid_conversion_sha256'],
                scene_sha256=pack['scene_sha256'], verified_metre_scale=False)


def _data_path(path):
    path = Path(path).resolve()
    if not path.is_relative_to(DATA.resolve()):
        raise ValueError('Review packs, frames and diagnostics must stay in Data')
    return path


def _font_spec(font):
    path = Path(font.path).resolve()
    return dict(path=str(path), reference=reference(path), size=int(font.size),
                index=int(getattr(font, 'index', 0)),
                encoding=getattr(font, 'encoding', ''),
                layout_engine=int(font.layout_engine))


def _font(spec):
    if reference(spec['path']) != spec['reference']:
        raise ValueError('The parent review font changed')
    return ImageFont.truetype(spec['path'], spec['size'], index=spec['index'],
                              encoding=spec['encoding'], layout_engine=spec['layout_engine'])


def _producer_references(adapter):
    for ref in adapter.source_producers:
        if reference(ref['path']) != ref:
            raise ValueError('The authenticated parent producer changed before packing: '+ref['path'])
    paths = [r['path'] for r in adapter.source_producers]
    paths.extend([__file__, FirstShotReview.draw.__code__.co_filename,
                  SceneRenderer.draw.__code__.co_filename,
                  make_board_grid_renderer.__code__.co_filename,
                  PlanView.text.__code__.co_filename,
                  row_scene_projection.__file__, row_scene_projection_fast.__file__])
    from .current_source_points import draw_current_source_pixel_markers
    paths.append(draw_current_source_pixel_markers.__code__.co_filename)
    paths.append(validate_metric_review_scene.__code__.co_filename)
    from rinkspace.metric_scale import estimate_uniform_scale
    paths.append(estimate_uniform_scale.__code__.co_filename)
    return [reference(p) for p in dict.fromkeys(map(str, paths))]


def _frame_record(frame):
    uv = np.concatenate([np.asarray(row['uv']) for row in frame['rows']])
    if uv.shape != (frame['source_point_count'], 2) or not np.isfinite(uv).all():
        raise ValueError('Every current source point must be retained in its original order')
    result = dict(index=int(frame['index']), native_pts=int(frame['native_pts']),
                  source_packet=deepcopy(frame['source_packet']), rows=[dict(uv=uv)],
                  source_point_count=int(frame['source_point_count']))
    for name in ('inactive_source_correspondences', 'original_approved_source_point_count',
                 'source_revision_reference'):
        if name in frame:
            result[name] = deepcopy(frame[name])
    result['review_UV_sha256'] = _sha(review_source_uv(result))
    return result


def pack_review(view, adapter, prototype_dir, *, check_frames=(1, 842, 1248)):
    """Extract the already authenticated parent; do not assemble any dataset."""
    q = np.asarray(view.q, np.float64).copy()
    if q.shape != (7674,) or _sha(q) != view.q_sha256:
        raise ValueError('An authenticated complete7674 review is required')
    pts = np.asarray(adapter.state.pts, np.int64).copy()
    expected = np.arange(256256, 575489, 256, dtype=np.int64)
    if not np.array_equal(pts, expected) or len(adapter.bundle.frames) != 1248:
        raise ValueError('All 1248 official native50 exposures are required')
    frames = [_frame_record(frame) for frame in adapter.bundle.frames]
    if [f['native_pts'] for f in frames] != pts.tolist():
        raise ValueError('Frame packet order differs from the native PTS clock')
    sources = deepcopy(view.sources)
    for frame in frames:
        source = sources[frame['native_pts']]
        if (source['native_pts'] != frame['native_pts']
                or source['actual_pts'] != frame['native_pts']
                or source['time_base'] != '1/12800'
                or source['source_video_sha256'] != view.source['sha256']):
            raise ValueError('A source snapshot changed its authenticated clock or video')
    check_frames = tuple(map(int, check_frames))
    if len(set(check_frames)) != 3 or any(i < 1 or i > 1248 for i in check_frames):
        raise ValueError('Choose three distinct actual frames for the transport check')
    camera_state = _CameraState(adapter.bundle.state)
    if hasattr(camera_state, 'provider'):
        raise ValueError('A geometry factory must never enter the lightweight camera pack')
    cameras = np.asarray([adapter.state.camera25(q, i) for i in range(1, 1249)])
    row_proofs = []
    rows = np.linspace(0., 1079., 17)
    for index in check_frames:
        actual = adapter.model.prepare(adapter.state, q, index).camera25_at_rows(rows)
        rendered = row_scene_projection.prepare_render_projection(view.bundle.state, q, index)
        reference_rows = (np.repeat(cameras[index-1][None, :], len(rows), axis=0)
                          if rendered is None else rendered.context.camera25_at_rows(rows))
        if not np.array_equal(actual.view(np.uint64), reference_rows.view(np.uint64)):
            raise ValueError('Parent review row trajectory differs from the actual complete model')
        row_proofs.append(dict(frame=index, rows=rows.copy(), cameras=actual.copy()))
    revision = view.native_line_revision_proof
    grid_conversion = deepcopy(validate_metric_review_scene(view.scene))
    if view.renderer.board_grid.report['scene_unit_m'] != grid_conversion['metres_per_scene_unit']:
        raise ValueError('Ice and board review grids require the same uniform conversion')
    pack = dict(schema=SCHEMA, q=q, q_sha256=view.q_sha256, pts=pts,
                scene=deepcopy(view.scene), scene_sha256=_json_sha(view.scene),
                camera_state=camera_state, expected_camera25=cameras,
                model_reference=deepcopy(adapter.model_reference),
                prototype_dir=str(Path(prototype_dir).resolve()),
                frames=frames, sources=sources,
                source=deepcopy(view.source), video_reference=deepcopy(view.video_reference),
                checkpoint_receipt=deepcopy(view.checkpoint_receipt),
                native_line_revision_proof=(None if revision is None else
                                             dict(reference=deepcopy(revision['reference']))),
                right=view.right.copy(), right_sha256=_sha(view.right),
                font=_font_spec(view.plan.font), small_font=_font_spec(view.plan.small),
                path_counts=deepcopy(view.renderer.path_counts),
                advertisement_ids=deepcopy(view.plan.advertisement_ids),
                scene_unit_working_m=float(adapter.state.scene_unit_working_m),
                review_grid_conversion=grid_conversion,
                review_grid_conversion_sha256=grid_json_sha(grid_conversion),
                source_clock=deepcopy(adapter.state.source_clock),
                check_frames=list(check_frames), row_proofs=row_proofs,
                producers=_producer_references(adapter),
                source_bundle_included=False, world_factory_included=False,
                camera25_is_reference_only_row_trajectory_retained=True)
    pack['payload_summary'] = dict(
        active_UV_points=sum(f['source_point_count'] for f in frames),
        active_UV_array_bytes=sum(f['rows'][0]['uv'].nbytes for f in frames),
        core_array_bytes=sum(pack[name].nbytes for name in ('q', 'pts', 'right', 'expected_camera25')),
        source_packets=1248, fitting_rows_whitening_J_or_geometry_provider_included=False)
    _validate_pack(pack)
    return pack


def _validate_pack(pack):
    if pack.get('schema') not in (SCHEMA, LEGACY_SCHEMA) or pack.get('source_bundle_included') is not False:
        raise ValueError('Only a trusted lightweight first-shot review pack is supported')
    if (np.asarray(pack['q']).shape != (7674,) or _sha(pack['q']) != pack['q_sha256']
            or _json_sha(pack['scene']) != pack['scene_sha256']
            or _sha(pack['right']) != pack['right_sha256']):
        raise ValueError('Frozen state, complete geometry or plan backdrop changed')
    if pack['schema'] == SCHEMA:
        receipt = validate_metric_review_scene(pack['scene'])
        if (pack.get('review_grid_conversion') != receipt
                or pack.get('review_grid_conversion_sha256') != grid_json_sha(receipt)):
            raise ValueError('Frozen review pack metric conversion changed')
    elif METRIC_GRID_FIELD in pack['scene']:
        raise ValueError('A metric-grid scene requires its explicit v2 pack conversion receipt')
    if (not np.array_equal(pack['pts'], np.arange(256256, 575489, 256, dtype=np.int64))
            or len(pack['frames']) != 1248 or np.asarray(pack['expected_camera25']).shape != (1248, 25)
            or not np.isfinite(pack['expected_camera25']).all()
            or hasattr(pack['camera_state'], 'provider')):
        raise ValueError('Complete native clock and geometry-free camera chart required')
    for index, frame in enumerate(pack['frames'], 1):
        if frame['index'] != index or frame['native_pts'] != int(pack['pts'][index-1]):
            raise ValueError('Frozen source frame order changed')
        if _sha(review_source_uv(frame)) != frame['review_UV_sha256']:
            raise ValueError('Current source UV or preserved inactive facts changed')


def save_pack(pack, path):
    """Write a local trusted pickle once; workers authenticate its exact bytes."""
    _validate_pack(pack)
    path = _data_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        pickle.dump(pack, stream, protocol=pickle.HIGHEST_PROTOCOL)
    return reference(path)


def load_pack(pack_reference):
    path = _data_path(pack_reference['path'])
    if reference(path) != pack_reference:
        raise ValueError('Frozen review pack bytes changed')
    # This format is exclusively the trusted local pack produced above.
    with path.open('rb') as stream:
        pack = pickle.load(stream)
    _validate_pack(pack)
    return pack


def _load_model(pack):
    for ref in pack['producers']:
        if reference(ref['path']) != ref:
            raise ValueError('Frozen camera or review producer changed: ' + ref['path'])
    prototype = pack['prototype_dir']
    if prototype not in sys.path:
        sys.path.insert(0, prototype)
    ref = pack['model_reference']
    if reference(ref['path']) != ref:
        raise ValueError('Frozen complete model changed')
    name = 'rinkspace_parallel_review_model_' + ref['sha256'][:20]
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, ref['path'])
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    model = sys.modules[name]
    model.install_dispatch()
    return model


def _build_view(pack):
    model = _load_model(pack)
    state = model.TransverseNativePrincipalState(pack['camera_state'])
    if hasattr(state.original, 'provider') or state.count != 7674:
        raise ValueError('A render worker must contain only the complete camera chart')
    q = np.asarray(pack['q'], np.float64).copy()
    q.setflags(write=False)
    row_scene_projection_fast.install()
    for proof in pack['row_proofs']:
        index = proof['frame']
        camera = state.camera25(q, index)
        if not np.array_equal(camera.view(np.uint64), pack['expected_camera25'][index-1].view(np.uint64)):
            raise ValueError('Transferred camera25 differs from the authenticated parent')
        projected = row_scene_projection.prepare_render_projection(state, q, index)
        sampled = (np.repeat(camera[None, :], len(proof['rows']), axis=0)
                   if projected is None else projected.context.camera25_at_rows(proof['rows']))
        if not np.array_equal(sampled.view(np.uint64), proof['cameras'].view(np.uint64)):
            raise ValueError('Transferred row exposures differ from the authenticated parent')
    # Old packs retain their historical comparison grid without relabelling it.
    metric = pack.get('review_grid_conversion')
    grid_scale = metric['metres_per_scene_unit'] if metric else pack['scene_unit_working_m']
    renderer_type = make_board_grid_renderer(SceneRenderer, scene_unit_m=grid_scale)
    renderer = renderer_type(pack['scene'], draw_grid=True, draw_advertisements=True)
    ids = sorted({int(p['number']) for p in renderer.advertisement_paths})
    if renderer.path_counts != pack['path_counts'] or ids != pack['advertisement_ids']:
        raise ValueError('A worker omitted or changed the complete scene inventory')
    # Only text() is needed during draw. Reuse the parent's full plan backdrop;
    # do not rebuild or approximate any world geometry in each worker.
    plan = PlanView.__new__(PlanView)
    plan.font = _font(pack['font'])
    plan.small = _font(pack['small_font'])
    plan.path_counts = deepcopy(pack['path_counts'])
    plan.advertisement_ids = list(pack['advertisement_ids'])
    view = FirstShotReview.__new__(FirstShotReview)
    view.bundle = SimpleNamespace(state=state, frames=pack['frames'])
    view.q = q
    view.q_sha256 = pack['q_sha256']
    view.checkpoint_receipt = pack['checkpoint_receipt']
    view.scene = pack['scene']
    view.review_grid_conversion = deepcopy(metric) if metric else dict(
        policy='legacy_working_grid_comparison_only_not_final_metres',
        scene_unit_working_m=grid_scale, verified_metre_scale=False)
    view.source = pack['source']
    view.video_reference = pack['video_reference']
    view.renderer = renderer
    view.plan = plan
    view.right = pack['right']
    view.frames = {f['native_pts']: f for f in pack['frames']}
    view.sources = pack['sources']
    view.native_line_revision_proof = pack['native_line_revision_proof']
    view.last_metrics = None
    return view


def initialize_worker(pack_reference):
    global _WORKER
    configure_native_runtime(stage='lightweight-first-shot-render-worker', native_threads=1)
    cv2.setNumThreads(1)
    started = time.perf_counter()
    pack = load_pack(pack_reference)
    view = _build_view(pack)
    _WORKER = dict(pack=pack, view=view, initialized_seconds=time.perf_counter()-started,
                   pack_reference=pack_reference)


def render_worker_frame(index):
    if _WORKER is None:
        raise RuntimeError('Initialize the lightweight review worker first')
    pack, view = _WORKER['pack'], _WORKER['view']
    index = int(index)
    if not 1 <= index <= 1248:
        raise ValueError('Actual first-shot frame index required')
    pts = int(pack['pts'][index-1])
    source = pack['sources'][pts]
    started = time.perf_counter()
    native = cv2.imread(source['image_path'])
    if native is None:
        raise ValueError('The original native source image is missing')
    canvas = view.draw(native, pts)
    metrics = deepcopy(view.last_metrics)
    if (metrics['native_pts'] != pts or metrics['index'] != index
            or metrics['q_sha256'] != pack['q_sha256']
            or metrics['camera25_sha256'] != _sha(pack['expected_camera25'][index-1])
            or metrics['approved_UV_sha256'] != pack['frames'][index-1]['review_UV_sha256']):
        raise ValueError('A worker changed the state, camera, current source points or native exposure')
    if canvas.shape != (SIZE[1], SIZE[0], 3) or canvas.dtype != np.uint8:
        raise ValueError('Complete native-size review canvas required')
    return dict(index=index, native_pts=pts, canvas=canvas, metrics=metrics,
                canvas_sha256=_sha(canvas), seconds=time.perf_counter()-started,
                worker_pid=os.getpid(), worker_initialization_seconds=_WORKER['initialized_seconds'],
                source_bundle_loaded=False, geometry_factory_loaded=False)


def iter_parallel_frames(pack_reference, indices, *, workers=8, max_in_flight=None):
    """Bound outstanding canvases and yield the exact requested source order."""
    indices = tuple(map(int, indices))
    if not indices or len(set(indices)) != len(indices) or any(i < 1 or i > 1248 for i in indices):
        raise ValueError('Distinct actual source frame indices required')
    workers = int(workers)
    window = 2*workers if max_in_flight is None else int(max_in_flight)
    if workers < 1 or window < 1:
        raise ValueError('Positive render worker and queue sizes required')
    pending = deque()
    tasks = iter(indices)
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context('spawn'),
                             initializer=initialize_worker, initargs=(pack_reference,)) as pool:
        try:
            for _ in range(min(window, len(indices))):
                index = next(tasks)
                pending.append((index, pool.submit(render_worker_frame, index)))
            for expected in indices:
                index, future = pending.popleft()
                if index != expected:
                    raise ValueError('Parallel source order changed')
                result = future.result()
                if result['index'] != expected:
                    raise ValueError('A worker returned a different exposure')
                try:
                    following = next(tasks)
                except StopIteration:
                    following = None
                if following is not None:
                    pending.append((following, pool.submit(render_worker_frame, following)))
                yield result
        finally:
            for _, future in pending:
                future.cancel()


def validate_transport(view, pack_reference, output_dir, *, workers=1):
    """During the necessary final load, test three real complete PNG transfers."""
    pack = load_pack(pack_reference)
    if view.q_sha256 != pack['q_sha256'] or _json_sha(view.scene) != pack['scene_sha256']:
        raise ValueError('The single-process reference must use this exact frozen pack')
    output = _data_path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    row_scene_projection_fast.install()
    checks = []
    iterator = iter_parallel_frames(pack_reference, pack['check_frames'], workers=workers)
    try:
        for index, result in zip(pack['check_frames'], iterator, strict=True):
            pts = int(pack['pts'][index-1])
            native = cv2.imread(view.sources[pts]['image_path'])
            view.renderer._geometry_cache.clear()
            expected = view.draw(native, pts)
            parent_metrics = deepcopy(view.last_metrics)
            actual = result.pop('canvas')
            difference = np.abs(expected.astype(np.int16)-actual.astype(np.int16))
            same = bool(np.array_equal(expected, actual))
            original = output/f'frame_{index:04d}_single.png'
            transferred = output/f'frame_{index:04d}_worker.png'
            if not cv2.imwrite(str(original), expected) or not cv2.imwrite(str(transferred), actual):
                raise OSError('Unable to save exact transport check PNGs')
            decoded_equal = bool(np.array_equal(cv2.imread(str(original)), cv2.imread(str(transferred))))
            checks.append(dict(frame=index, single_image=reference(original), worker_image=reference(transferred),
                               identical=same and decoded_equal,
                               differing_pixels=int(np.any(difference != 0, axis=2).sum()),
                               maximum_channel_difference=int(difference.max(initial=0)),
                               camera25_identical=parent_metrics['camera25_sha256'] == result['metrics']['camera25_sha256'],
                               current_UV_identical=parent_metrics['approved_UV_sha256'] == result['metrics']['approved_UV_sha256'],
                               worker=result))
    finally:
        iterator.close()
    passed = len(checks) == 3 and all(c['identical'] and c['camera25_identical'] and c['current_UV_identical'] for c in checks)
    receipt = dict(schema='rinkspace.three-frame-lightweight-review-transport-check.v1',
                   pack=pack_reference, q_sha256=pack['q_sha256'], check_frames=pack['check_frames'],
                   review_grid_identity=_grid_identity(pack),
                   passed=passed, checks=checks, all_complete_geometry_and_source_checks_retained=True,
                   video_generated=False, accuracy_accepted=False)
    path = output/'transport_check.json'
    path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')
    if not passed:
        raise ValueError('Parallel drawing is not pixel identical; see '+str(path))
    return reference(path)


class _OrderedReviewStream:
    """Feed existing render_video without changing its encoder or clock checks."""
    def __init__(self, pack, iterator):
        self.bundle = SimpleNamespace(frames=pack['frames'], state=SimpleNamespace(pts=pack['pts']))
        self.sources = pack['sources']
        self.q_sha256 = pack['q_sha256']
        self.iterator = iterator
        self.expected_index = 1
        self.last_metrics = None

    def draw(self, native, pts):
        result = next(self.iterator)
        if result['index'] != self.expected_index or result['native_pts'] != int(pts):
            raise ValueError('A parallel video exposure was skipped, repeated or reordered')
        if not isinstance(native, np.ndarray) or _sha(native) != result['metrics']['native_BGR_sha256']:
            raise ValueError('Parent encoder input differs from the authenticated worker exposure')
        self.expected_index += 1
        self.last_metrics = result['metrics']
        return result['canvas']


def render_parallel_video(pack_reference, output, transport_receipt_reference, *, workers=8, max_in_flight=None):
    """Explicit video action after real same-pack single/worker PNG agreement."""
    pack = load_pack(pack_reference)
    if reference(transport_receipt_reference['path']) != transport_receipt_reference:
        raise ValueError('The transport validation receipt changed')
    receipt = json.loads(Path(transport_receipt_reference['path']).read_text(encoding='utf8'))
    checks = receipt.get('checks', [])
    if (receipt.get('schema') != 'rinkspace.three-frame-lightweight-review-transport-check.v1'
            or receipt.get('passed') is not True or receipt.get('pack') != pack_reference
            or receipt.get('q_sha256') != pack['q_sha256'] or len(checks) != 3
            or [c.get('frame') for c in checks] != pack['check_frames']
            or not all(c.get('identical') and c.get('camera25_identical') and c.get('current_UV_identical') for c in checks)):
        raise ValueError('Three real pixel-exact same-pack checks are required before video')
    if pack['schema'] == SCHEMA and receipt.get('review_grid_identity') != _grid_identity(pack):
        raise ValueError('Transport receipt uses a different scene-bound metric-grid conversion')
    iterator = iter_parallel_frames(pack_reference, range(1, 1249), workers=workers, max_in_flight=max_in_flight)
    stream = _OrderedReviewStream(pack, iterator)
    try:
        result = render_video(stream, output)
        if stream.expected_index != 1249:
            raise ValueError('Parallel video must cover all1248 native exposures')
    finally:
        iterator.close()
    return dict(**result, parallel_workers=int(workers), pack=pack_reference,
                review_grid_identity=_grid_identity(pack),
                transport_check=transport_receipt_reference,
                all1248_ordered_no_drops=True, source_bundle_loaded_per_worker=False,
                complete_geometry_fixed_from_authenticated_parent=True)
