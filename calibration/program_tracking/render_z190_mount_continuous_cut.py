"""Stream one complete native-PTS cut through the existing full scene renderer.

This is a visibly unaccepted camera review.  It replays the saved physical
camera values and confirmed exposure aliases without creating a body trajectory.
Large artifacts, including the encoded video and exact source manifest, remain
outside OneDrive; decoded source/rendered images are not cached per frame.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / 'src'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
from runtime.bootstrap import add_geometry_dependencies
add_geometry_dependencies()
import av
import cv2
import numpy as np
from rinkspace.storage import AssetStore

from render_z190_mount_source_keys import digest
from render_result import (SceneRenderer, PersonMasks, DEFAULT_MASKS,
                           DEFAULT_PEOPLE, _ffmpeg_executable, _available_encoder)
from z190_physical_camera import PhysicalCameraModel
from z190_continuous_source_bundle import HARD_SCENE_SHA, SOURCE_SHA, SCENE_UNIT_M
from three_frame_refit.program_tracking.body_tracking.pelvis_proxy_programme import PlanView, source_time

SIZE = (2580, 1180)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n', encoding='utf8')


def verify_video(path, expected_count):
    count = 0
    with av.open(str(path)) as container:
        if len(container.streams.video) != 1 or container.streams.audio:
            raise ValueError('Review must contain one silent video stream')
        stream = container.streams.video[0]
        if (stream.width, stream.height) != SIZE or float(stream.average_rate) != 60:
            raise ValueError('Encoded size or native presentation rate changed')
        first_time = last_time = None
        for frame in container.decode(stream):
            stamp = frame.pts * stream.time_base
            if first_time is None:
                first_time = stamp
            if stamp - first_time != count / 60:
                # Fractions must be compared without a binary-float timestamp.
                from fractions import Fraction
                if stamp - first_time != Fraction(count, 60):
                    raise ValueError('Encoded presentation cadence changed')
            last_time = stamp
            count += 1
    if count != expected_count:
        raise ValueError(f'Encoded {count} frames instead of {expected_count}')
    return dict(decoded_frames=count, video_size_wh=list(SIZE),
                video_rate='60/1', duration_seconds=count / 60,
                last_frame_seconds=float(last_time - first_time), audio_streams=0)


def render(camera, camera_sha, scene_path, source_path, native_scope, output):
    storage = AssetStore(ROOT)
    camera, scene_path, native_scope = map(storage.resolve, (camera, scene_path, native_scope))
    source_path = storage.source(source_path)
    output = Path(output).resolve()
    if not output.is_relative_to(Path('D:/CodexData').resolve()):
        raise ValueError('Continuous diagnostic artifacts must stay outside OneDrive')
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    if digest(camera) != camera_sha or digest(scene_path) != HARD_SCENE_SHA:
        raise ValueError('Camera or fixed hard-scene binding changed')
    if source_path != storage.source(ROOT / 'data/input/oxXu1avhrvs.mp4'):
        raise ValueError('Use the original source-bound programme video')
    model, camera_pts, aliases, views = PhysicalCameraModel.load_npz(camera)
    with np.load(camera, allow_pickle=False) as packet:
        camera_values = packet['camera25'].copy()
        metadata = json.loads(str(packet['metadata_json']))
    if metadata['source_sha256'] != SOURCE_SHA:
        raise ValueError('Camera belongs to another source')
    with np.load(native_scope, allow_pickle=False) as scope:
        expected = scope['pts'].astype(np.int64)
        representatives = scope['exposure_representative_pts'].astype(np.int64)
        if str(scope['source_sha256']) != SOURCE_SHA:
            raise ValueError('Native-cut inventory belongs to another source')
    if not len(expected) or np.any(np.diff(expected) != 256):
        raise ValueError('Scope must contain the whole consecutive 60-Hz native cut')
    if any(int(p) not in aliases for p in expected):
        raise ValueError('Current camera lacks an original native presentation')
    if not np.array_equal([aliases[int(p)] for p in expected], representatives):
        raise ValueError('Confirmed exposure mapping changed')
    index = {int(p): i for i, p in enumerate(camera_pts)}
    cameras = camera_values[[index[int(p)] for p in expected]]
    reconstructed = np.vstack([model.camera25(views[aliases[int(p)]]) for p in expected])
    reconstruction_delta = float(np.max(np.abs(cameras - reconstructed)))
    if reconstruction_delta > 1e-12 or not np.isfinite(cameras).all():
        raise ValueError('Saved and physical camera values disagree')
    alias_values = {}
    for p, c in zip(expected, cameras):
        representative = aliases[int(p)]
        if representative in alias_values and not np.array_equal(c, alias_values[representative]):
            raise ValueError('Same exposure acquired different native camera values')
        alias_values[representative] = c
    slots = {views[aliases[int(p)]].installation_id for p in expected}
    if len(slots) != 1:
        raise ValueError('Continuous-cut review crossed an installation cut')
    cal = model.calibrations[next(iter(slots))]
    scene = json.loads(Path(scene_path).read_text(encoding='utf8'))
    renderer, plan = SceneRenderer(scene), PlanView(scene, 0.)
    if renderer.path_counts != plan.path_counts:
        raise ValueError('Source and plan use different scene inventories')
    masks = PersonMasks(DEFAULT_MASKS, DEFAULT_PEOPLE)
    fixed_pivot = np.asarray(cal.pivot_xyz)
    lever_cm = np.asarray(cal.lever_xyz) * SCENE_UNIT_M * 100
    right_base = plan.base.copy()
    right_base[510:] = 248
    for y, label, color in (
        (535, '长边主机位：固定云台＋共享非零偏移', (40, 40, 40)),
        (579, '红十字：固定云台；蓝圈：当前光心', (40, 40, 40)),
        (621, '平面图采用实际尺度，二者可能接近重合', (40, 40, 40)),
        (669, f'固定云台高度：{fixed_pivot[2]*SCENE_UNIT_M:.3f} m', (40, 40, 40)),
        (751, '相机轴偏移 右／上／前（cm）', (40, 40, 40)),
        (790, f'{lever_cm[0]:.1f} ／ {-lever_cm[1]:.1f} ／ {lever_cm[2]:.1f}', (40, 40, 40)),
        (842, '高度相对冰面；摄影师地面高差未量', (40, 40, 40)),
        (889, '条件相机候选；尚未完成连续镜头验收', (160, 50, 40))):
        right_base = plan.text(right_base, (24, y), label, color=color, small=y != 535)
    header_base = np.full((80, SIZE[0], 3), 22, np.uint8)
    header_base = plan.text(header_base, (18, 45),
        '完整第 1 镜头；同一全部冰线、网格与实际广告边界持续显示',
        color=(220, 220, 220), small=True)
    manifest = dict(
        camera=dict(path=str(Path(camera).resolve()), sha256=camera_sha),
        scene=dict(path=str(Path(scene_path).resolve()), sha256=HARD_SCENE_SHA),
        source=dict(path=str(Path(source_path).resolve()), sha256=SOURCE_SHA,
                    time_base='1/15360', native_size_wh=[1920, 1080]),
        native_scope=dict(path=str(Path(native_scope).resolve()), sha256=digest(native_scope),
                          fields_used=['pts', 'exposure_representative_pts', 'source_sha256'],
                          old_scope_camera_used=False),
        installation_id=cal.installation_id, native_frames=len(expected),
        actual_exposures=len(set(map(int, representatives))),
        first_native_pts=int(expected[0]), last_native_pts=int(expected[-1]),
        first_source_time=source_time(int(expected[0])), last_source_time=source_time(int(expected[-1])),
        saved_physical_camera_max_abs_delta=reconstruction_delta,
        same_exposure_camera_bitexact=True, source_inventory=renderer.path_counts,
        plan_inventory=plan.path_counts, advertisement_ids=plan.advertisement_ids,
        model_inventory_always_enabled=True, confidence_gating=False,
        body_trajectory_generated=False, camera_accuracy_accepted=False,
        whole_programme_acceptance=False, complete_shot_acceptance=False,
        per_frame_image_cache_generated=False,
        layout=dict(video_size_wh=list(SIZE), left_source=[0, 80, 1920, 1160],
                    right_same_scene_plan=[1920, 80, 2580, 1160]))
    write_json(output / 'input_manifest.json', manifest)
    print('Verifying the original programme source digest', flush=True)
    if digest(source_path) != SOURCE_SHA:
        raise ValueError('Original video content binding changed')
    manifest['source_video_sha256_verified'] = True
    write_json(output / 'input_manifest.json', manifest)
    ffmpeg = _ffmpeg_executable()
    encoder_args, encoder = _available_encoder(ffmpeg)
    partial = output / 'WORKING_UNACCEPTED.partial.mp4'
    target = output / 'WORKING_UNACCEPTED.mp4'
    command = [ffmpeg, '-hide_banner', '-loglevel', 'error', '-y',
               '-f', 'rawvideo', '-pixel_format', 'bgr24', '-video_size', f'{SIZE[0]}x{SIZE[1]}',
               '-framerate', '60', '-i', 'pipe:0', '-an', *encoder_args,
               '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(partial)]
    count = 0
    mask_counts = Counter()
    refinement_max = Counter()
    render_inventory_verified = True
    facts_path = output / 'native_source_render_manifest.jsonl'
    print(f'Streaming {len(expected)} original native frames with {encoder}', flush=True)
    with (output / 'encoder_stderr.log').open('wb') as log:
        encoder_process = subprocess.Popen(command, stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL, stderr=log,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        try:
            with facts_path.open('w', encoding='utf8') as facts, av.open(str(source_path)) as source:
                stream = source.streams.video[0]
                if (stream.width, stream.height) != (1920, 1080) or str(stream.time_base) != '1/15360':
                    raise ValueError('Native source dimensions/time base changed')
                source.seek(max(0, int(expected[0]) - 15360), stream=stream, backward=True)
                for frame in source.decode(stream):
                    if frame.pts is None or frame.pts < expected[0]:
                        continue
                    if frame.pts > expected[-1]:
                        break
                    p = int(frame.pts)
                    if count >= len(expected) or p != int(expected[count]):
                        raise ValueError(f'Missing or reordered source frame at {p}, index {count}')
                    native = frame.to_ndarray(format='bgr24')
                    cam = cameras[count]
                    try:
                        person, mask_status = masks.get(p, require_sam2=True)
                    except ValueError:
                        person, mask_status = None, 'no confirmed source mask'
                    left, metrics = renderer.draw(native, cam, person)
                    if metrics['fixed_model_path_counts'] != manifest['source_inventory']:
                        render_inventory_verified = False
                        raise ValueError('A frame changed the fixed enabled inventory')
                    right = right_base.copy()
                    pivot_xy, optical_xy = plan.mapxy(np.vstack((fixed_pivot[:2], cam[:2])))
                    cv2.drawMarker(right, tuple(pivot_xy), (30, 30, 230), cv2.MARKER_CROSS, 13, 2)
                    cv2.circle(right, tuple(optical_xy), 4, (225, 90, 20), 1, cv2.LINE_AA)
                    right = plan.text(right, (24, 710),
                        f'当前光心高度：{cam[2]*SCENE_UNIT_M:.3f} m', small=True)
                    header = plan.text(header_base, (18, 10),
                        f'Z190 非零安装候选（未验收） | 原片 {source_time(p)}', color=(245, 245, 245))
                    canvas = np.full((SIZE[1], SIZE[0], 3), 22, np.uint8)
                    canvas[:80] = header
                    canvas[80:1160, :1920], canvas[80:1160, 1920:] = left, right
                    encoder_process.stdin.write(canvas.tobytes())
                    row = dict(frame_index=count, pts=p, source_time=source_time(p),
                        representative_pts=aliases[p], source_bgr_sha256=hashlib.sha256(native.tobytes()).hexdigest(),
                        saved_camera25_float64_sha256=hashlib.sha256(cam.tobytes()).hexdigest(),
                        fixed_pivot_xyz_scene=fixed_pivot.tolist(),
                        actual_optical_centre_scene=cam[:3].tolist(), mask_status=mask_status,
                        render_metrics=metrics)
                    facts.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
                    mask_counts[mask_status] += 1
                    for key, value in metrics.get('board_refinement', {}).items():
                        if isinstance(value, int):
                            refinement_max[key] = max(refinement_max[key], value)
                    count += 1
                    if count % 120 == 0 or count == len(expected):
                        print(f'{count}/{len(expected)} frames; source {source_time(p)}; '
                              f'elapsed {time.monotonic()-started:.1f}s', flush=True)
            if count != len(expected):
                raise ValueError(f'Source decode stopped after {count}/{len(expected)} frames')
            encoder_process.stdin.close()
            code = encoder_process.wait()
            if code:
                raise RuntimeError(f'Encoder exited with code {code}; see encoder_stderr.log')
        except BaseException as error:
            if encoder_process.stdin and not encoder_process.stdin.closed:
                encoder_process.stdin.close()
            if encoder_process.poll() is None:
                encoder_process.terminate()
            encoder_process.wait()
            write_json(output / 'production_failure.json', dict(
                error=str(error), rendered_frames=count, encoder_exit_code=encoder_process.returncode))
            raise
    verification = verify_video(partial, len(expected))
    partial.rename(target)
    report = dict(**manifest, video=dict(path=str(target), sha256=digest(target)),
        exact_native_render_manifest=dict(path=str(facts_path), sha256=digest(facts_path)),
        encoder=encoder, video_verification=verification,
        source_decode_exact_native_pts=True, alias_camera_values_preserved=True,
        every_frame_fixed_inventory_verified=render_inventory_verified,
        source_foreground_mask_status_counts=dict(mask_counts),
        renderer_board_refinement_maximums=dict(refinement_max),
        elapsed_seconds=time.monotonic()-started,
        production_complete=True, video_generated=True,
        visual_continuous_acceptance_performed=False,
        retained_failure_policy='all original frames and current camera regressions remain visible')
    write_json(output / 'review.json', report)
    print(json.dumps(dict(output=str(target), frames=count,
                         source_interval=[report['first_source_time'], report['last_source_time']],
                         sha256=report['video']['sha256']), ensure_ascii=False), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('camera', 'scene', 'source', 'native-scope', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--camera-sha', required=True)
    arguments = parser.parse_args()
    render(arguments.camera, arguments.camera_sha, arguments.scene, arguments.source,
           arguments.native_scope, arguments.output)
