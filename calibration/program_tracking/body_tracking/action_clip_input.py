"""Prepare real official-clock crops and measured pixels for short action clips.

This CPU-only adapter evaluates the selected frozen camera and decodes source
images. It does not infer bodies, fit a camera, or import historical world poses.
The optional ROI proposal only locates a crop: it never fills fitting pixels.
"""
from __future__ import annotations

from fractions import Fraction
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

TIME_BASE = 12800
NATIVE_STEP = 256
MODEL_STEP = 512
NATIVE_WH = (1920, 1080)


def _sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b''):
            result.update(chunk)
    return result.hexdigest()


def _load(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _summary(values):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    return dict(count=int(values.size), median=float(np.median(values)) if values.size else None,
                p95=float(np.percentile(values, 95)) if values.size else None,
                maximum=float(values.max()) if values.size else None)


def clip_pts(source_seconds):
    """Inclusive native interval; the 25 Hz phase starts at its first exposure."""
    seconds = np.asarray(source_seconds, float)
    if seconds.shape != (2,) or not np.isfinite(seconds).all() or seconds[1] <= seconds[0]:
        raise ValueError('source_seconds must be an increasing finite pair')
    pts = np.rint(seconds * TIME_BASE).astype(np.int64)
    if np.max(np.abs(seconds * TIME_BASE - pts)) > 1e-6 or np.any(pts % NATIVE_STEP):
        raise ValueError('Clip bounds must be real official 50 Hz exposure PTS')
    native = np.arange(pts[0], pts[1] + NATIVE_STEP, NATIVE_STEP, dtype=np.int64)
    return native, np.arange(0, len(native), 2, dtype=np.int64)


def map_observations(observations, ledger_rows, pts, shot_id):
    """Preserve real source pixels, and leave unmatched official rows unknown."""
    from .body_prior import original_observation_mask

    shot_index = int(shot_id.removeprefix('s'))
    old_index = {int(value): i for i, value in enumerate(observations['pts'])}
    wanted = set(map(int, pts))
    candidates = {}
    for row in ledger_rows:
        official = row.get('accepted_official_pts')
        old = int(row['full_pts'])
        if (official is None or int(row['shot_index']) != shot_index
                or int(official) not in wanted or old not in old_index):
            continue
        i = old_index[old]
        if int(observations['shot_index'][i]) != shot_index:
            raise ValueError('Observation and ledger shot identities differ')
        score = (not bool(observations['valid'][i]), float(row['best_roi_gray_mae']))
        if official not in candidates or score < candidates[official][0]:
            candidates[official] = (score, i, row)
    mask, clipped = original_observation_mask(observations)
    n = len(pts)
    result = dict(pts=pts.copy(), source_pts=pts.copy(),
                  body_uv=np.full((n, 17, 2), np.nan, np.float32),
                  body_score=np.zeros((n, 17), np.float32),
                  body_use=np.zeros((n, 17), bool),
                  bbox=np.full((n, 4), np.nan, np.float32),
                  source_legacy_pts=np.full(n, -1, np.int64),
                  source_observation_row=np.full(n, -1, np.int64),
                  source_mapping_missing=np.ones(n, bool),
                  source_observation_valid=np.zeros(n, bool),
                  source_clipped=np.zeros(n, bool),
                  source_mapping_mae=np.full(n, np.nan),
                  source_match_status=np.full(n, 'missing', dtype='<U64'))
    for k, stamp in enumerate(pts):
        item = candidates.get(int(stamp))
        if item is None:
            continue
        _, i, row = item
        result['body_uv'][k] = observations['native_coco17_xy'][i]
        result['body_score'][k] = observations['keypoint_scores'][i]
        result['body_use'][k] = mask[i] & bool(observations['valid'][i])
        result['bbox'][k] = observations['bbox_native_xyxy'][i]
        result['source_legacy_pts'][k] = observations['pts'][i]
        result['source_observation_row'][k] = i
        result['source_mapping_missing'][k] = False
        result['source_observation_valid'][k] = observations['valid'][i]
        result['source_clipped'][k] = clipped[i]
        result['source_mapping_mae'][k] = row['best_roi_gray_mae']
        result['source_match_status'][k] = row['match_status']
    return result


def roi_proposals(native, model_rows, *, missing_policy='error'):
    """Interpolate only crop boxes; measured bbox/UV/use arrays are untouched."""
    if missing_policy not in ('error', 'roi_proposal'):
        raise ValueError('missing_policy must be error or roi_proposal')
    boxes = np.asarray(native['bbox'], float)
    real = (~native['source_mapping_missing'] & np.isfinite(boxes).all(axis=1)
            & (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1]))
    roi = boxes[model_rows].copy()
    inferred = ~real[model_rows]
    anchors = np.repeat(native['pts'][model_rows, None], 2, axis=1)
    alpha = np.zeros(len(model_rows))
    if inferred.any() and missing_policy == 'error':
        raise ValueError('No measured mapped ROI at official PTS: '
                         + str(native['pts'][model_rows[inferred]].tolist()))
    known = np.flatnonzero(real)
    for slot in np.flatnonzero(inferred):
        i = int(model_rows[slot])
        left, right = known[known < i], known[known > i]
        if not len(left) or not len(right):
            raise ValueError('ROI proposal requires real boxes on both sides of PTS '
                             + str(int(native['pts'][i])))
        a, b = int(left[-1]), int(right[0])
        t = float((native['pts'][i] - native['pts'][a]) /
                  (native['pts'][b] - native['pts'][a]))
        roi[slot] = (1 - t) * boxes[a] + t * boxes[b]
        anchors[slot] = native['pts'][[a, b]]
        alpha[slot] = t
    return dict(roi_native_xyxy=roi, bbox_inferred=inferred,
                roi_source_official_pts=anchors, roi_interpolation_alpha=alpha)


def camera25_from_snapshot(pose):
    """Exact midpoint optics from NativeCore, including its evaluated wide hinge.

    Layout is documented in camera_bridge.py: C, Rodrigues(R_w2c), log(fxy),
    principal point, zero skew, OpenCV Brown14. The native forward implements
    only k1/k2/p1/p2, as specified by camera_state.hpp motion_camera_from_values.
    """
    import cv2
    n = len(pose['status'])
    if np.any(pose['status']):
        raise ValueError('Selected frozen camera has failed exposure snapshots')
    out = np.zeros((n, 25))
    out[:, :3] = pose['centre']
    rotation = np.asarray(pose['rotation'], float)
    if (rotation.shape != (n, 3, 3)
            or np.max(np.abs(rotation @ rotation.transpose(0, 2, 1) - np.eye(3))) > 1e-8
            or np.max(np.abs(np.linalg.det(rotation) - 1)) > 1e-8):
        raise ValueError('Snapshot world-to-camera rotation is invalid')
    out[:, 3:6] = np.stack([cv2.Rodrigues(r)[0].ravel() for r in rotation])
    focal = np.column_stack((pose['fx'], pose['fy']))
    if not np.isfinite(focal).all() or np.any(focal <= 0):
        raise ValueError('Snapshot focal lengths are invalid')
    out[:, 6:8] = np.log(focal)
    out[:, 8:10] = np.column_stack((pose['cx'], pose['cy']))
    out[:, 11:15] = np.column_stack([pose[key] for key in ('k1', 'k2', 'p1', 'p2')])
    if not np.isfinite(out).all() or np.any(out[:, 2] <= 0):
        raise ValueError('Nonfinite camera25 or optical centre below the rink')
    return out


def _official_frames(video, wanted):
    import av
    wanted = list(map(int, wanted))
    if not wanted or wanted != sorted(set(wanted)):
        raise ValueError('Official decode PTS must be sorted and unique')
    index = 0
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        if stream.time_base != Fraction(1, TIME_BASE):
            raise ValueError('Official source time base is not 1/12800')
        if (stream.width, stream.height) != NATIVE_WH:
            raise ValueError('Official source native pixel grid changed')
        container.seek(max(0, wanted[0] - 2 * TIME_BASE), stream=stream, backward=True)
        for frame in container.decode(stream):
            if frame.pts is None or frame.pts < wanted[index]:
                continue
            if frame.pts != wanted[index]:
                raise ValueError(f'Official exposure {wanted[index]} absent; next {frame.pts}')
            yield int(frame.pts), frame.to_ndarray(format='bgr24')
            index += 1
            if index == len(wanted):
                return
    raise ValueError(f'Official source ended before exposure {wanted[index]}')


def _native_camera_check(projector, cameras, boxes, rows):
    """Compare camera25 against the native instantaneous projection oracle."""
    from .camera_bridge import rays, project_world
    xyz, expected, times = [], [], []
    for row, box in zip(rows, boxes):
        x0, y0, x1, y1 = box
        pixels = np.array([[(x0 + x1) / 2, (y0 + y1) / 2], [x0, y0],
                           [x1, y0], [x0, y1], [x1, y1]])
        inside = ((pixels[:, 0] >= 0) & (pixels[:, 0] < NATIVE_WH[0])
                  & (pixels[:, 1] >= 0) & (pixels[:, 1] < NATIVE_WH[1]))
        direction, valid = rays(pixels, cameras[row])
        valid &= inside
        point = cameras[row, :3] + direction[valid]
        pixel, good = project_world(point, cameras[row])
        xyz.extend(point[good]); expected.extend(pixel[good])
        times.extend([projector.p['motion_times'][row]] * int(good.sum()))
    if not xyz:
        raise ValueError('No real source pixels support a native/Brown closure check')
    rig = projector.p['rig'].copy()
    rig[37] = 0.  # read-only instantaneous oracle, not a change to the saved camera
    result = projector.core.project_motion(
        rig, projector.p['controls'], float(projector.p['first_control_time']),
        float(projector.p['control_dt']), NATIVE_WH[1], times, np.asarray(xyz))
    if np.any(result['status']):
        raise ValueError('Native instantaneous camera rejected checked source rays')
    error = np.linalg.norm(result['uv'] - np.asarray(expected), axis=1)
    if error.max() > 1e-4:
        raise ValueError('camera25 and native midpoint projection differ by '
                         + str(float(error.max())) + ' px')
    return _summary(error)


def _resampling_check(bgr, camera, rgb, mapping):
    """Compare one-sample crop to a 2x supersampling diagnostic of the same rays."""
    import cv2
    from .camera_bridge import rectify_patch
    sigma = mapping['anti_alias_native_gaussian_sigma']
    image = cv2.GaussianBlur(bgr, (5, 5), sigma) if sigma > 0 else bgr
    # Half-pixel shift makes each 2x2 high-resolution cell share a low-res centre.
    affine = np.array([[2., 0., .5], [0., 2., .5], [0., 0., 1.]])
    high, valid, _ = rectify_patch(image, camera, (0, 0, 512, 512),
                                  K_rect=affine @ np.asarray(mapping['K_rect_crop']))
    reference = cv2.resize(high, (256, 256), interpolation=cv2.INTER_AREA)[:, :, ::-1]
    all_valid = valid.reshape(256, 2, 256, 2).all(axis=(1, 3))
    difference = np.abs(rgb.astype(float) - reference.astype(float))[all_valid]
    return dict(valid_pixels=int(all_valid.sum()), absolute_channel_difference_8bit=_summary(difference),
                semantics='Same source/rays/blur, direct bilinear256 versus 2x supersampling/area; '
                          'resampling sensitivity, not a measured image error or accuracy threshold')


def prepare_action_clip(project, output, *, shot_id, source_seconds, shared_betas,
                        body_scale, clip_id=None, data_root=None, source_video=None,
                        observations_path=None, mapping_path=None, camera_record=None,
                        native_library=None, compact_smplx=None, skate_binding=None,
                        solver_metres_per_scene_unit=60., missing_policy='error',
                        overwrite=False):
    """Return paths to 25Hz inputs, all 50Hz fitting pixels, and their contract.

    ``roi_proposal`` retains one uninterrupted sequence of real source images.
    Missing measured rows keep NaN UV/bbox and false body_use. Only a separate
    crop ROI is interpolated between real mapped boxes; its source PTS are saved.
    The caller owns HMR2/GVHMR inference and any action/contact interpretation.
    """
    from rinkspace.storage import AssetStore
    from .body_prior import _bbox_to_rect_xys, _rectified_crop
    from .camera_bridge import (native_to_rectified, rectified_to_native,
                                rectified_intrinsics)
    from .fullbody_state import camera_binding

    project, output = Path(project).resolve(), Path(output).resolve()
    store = AssetStore(project, data_root=data_root)
    scope_path = project / 'events/ankara_2026/programmes/okada_mei_sp/rebuild_scope.json'
    scope = json.loads(scope_path.read_text(encoding='utf8'))
    selected = [s for s in scope['shots'] if s['id'] == shot_id]
    if len(selected) != 1:
        raise ValueError('shot_id must identify one current programme shot')
    pts, model_rows = clip_pts(source_seconds)
    span = selected[0]['processing']
    if pts[0] < span['first_pts'] or pts[-1] > span['last_pts']:
        raise ValueError('A clip cannot cross the selected current shot bounds')
    betas = np.asarray(shared_betas, np.float32)
    if betas.shape != (10,) or not np.isfinite(betas).all() or not np.isfinite(body_scale) or body_scale <= 0:
        raise ValueError('Shared S01 beta[10] and positive body scale are required')
    clip_id = clip_id or f'{shot_id}_{int(pts[0])}_{int(pts[-1])}'
    if not clip_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in clip_id):
        raise ValueError('clip_id must be a single alphanumeric/underscore/hyphen name')
    cache = output / 'source_cache'
    clip_dir = output / clip_id
    native_path = cache / (clip_id + '.native50.npz')
    prepared_path, contract_path = clip_dir / 'prepared.npz', clip_dir / 'contract.json'
    if not overwrite and any(p.exists() for p in (native_path, prepared_path, contract_path)):
        raise FileExistsError('Clip inputs exist; choose a new clip_id or explicit overwrite')
    base = store.data / 'runs/body_s01_20261029'
    observations_path = Path(observations_path or base / 'source_cache/observations.npz')
    mapping_path = Path(mapping_path or store.data / (
        'events/ankara_2026/programmes/okada_mei_sp/history/tracking_runs/source_cadence_comparison/'
        'O6j8L8TmfKo/full_programme_alignment/native60_to_official50.jsonl'))
    observations = _load(observations_path)
    if str(observations['source_time_base']) != '1/15360':
        raise ValueError('Cached observations must retain their original source time base')
    ledger = [json.loads(line) for line in mapping_path.read_text(encoding='utf8').splitlines()]
    native = map_observations(observations, ledger, pts, shot_id)
    proposals = roi_proposals(native, model_rows, missing_policy=missing_policy)
    video = store.source(source_video or scope['source']['registered_path'])
    video_hash = _sha256(video)
    if video_hash != scope['source']['sha256']:
        raise ValueError('Decoded official video differs from the current source identity')
    reference = SimpleNamespace(p=dict(pts=pts, metres_per_scene_unit=np.array(solver_metres_per_scene_unit)))
    projector, camera_info, _, pose = camera_binding(
        project, reference, camera_record, native_library=native_library, shot_id=shot_id)
    cameras = camera25_from_snapshot(pose)
    camera_check = _native_camera_check(projector, cameras, proposals['roi_native_xyxy'], model_rows)
    native.update(camera25=cameras, R_w2c=pose['rotation'], camera_centre_scene=pose['centre'],
                  camera_centre_m=pose['centre'] * camera_info['metres_per_scene_unit'],
                  K_rect=np.stack([rectified_intrinsics(c) for c in cameras]),
                  motion_times=projector.p['motion_times'], source_seconds=pts / TIME_BASE,
                  source_time_base=np.array([1, TIME_BASE]), shared_betas=betas,
                  body_scale=np.array(body_scale),
                  metres_per_scene_unit=np.array(camera_info['metres_per_scene_unit']))
    crops, joints, boxes, affine, crop_k, sigmas, fractions, frame_hashes = [], [], [], [], [], [], [], []
    closure, crop_coords, diagnostics = [], [], []
    diagnostic_slots = {0, len(model_rows) // 2, len(model_rows) - 1}
    for slot, (stamp, bgr) in enumerate(_official_frames(video, pts[model_rows])):
        row = model_rows[slot]
        if stamp != int(pts[row]):
            raise ValueError('Decoded official PTS differs from the requested model slot')
        camera = cameras[row]
        box = _bbox_to_rect_xys(proposals['roi_native_xyxy'][slot], camera)
        if box is None:
            raise ValueError('ROI does not map into a real rectified crop at PTS ' + str(stamp))
        rgb, fraction, mapping = _rectified_crop(bgr, camera, box)
        if fraction < .5:
            raise ValueError('Less than half of the crop samples valid source pixels at PTS ' + str(stamp))
        mapped, good = native_to_rectified(native['body_uv'][row], camera)
        usable = good & native['body_use'][row]
        obs = np.zeros((17, 3), np.float32)
        obs[:, :2] = box[:2]
        obs[usable, :2] = mapped[usable]
        obs[usable, 2] = native['body_score'][row, usable]
        if native['source_mapping_missing'][row]:
            obs[:] = 0.  # no synthetic keypoint observation, only a real image ROI
        transform = np.asarray(mapping['full_rect_to_crop_affine'])
        crop_xy = np.full((17, 2), np.nan)
        crop_xy[usable] = (np.c_[mapped[usable], np.ones(usable.sum())] @ transform.T)[:, :2]
        if usable.any():
            inverse = np.linalg.inv(transform)
            restored = (np.c_[crop_xy[usable], np.ones(usable.sum())] @ inverse.T)[:, :2]
            pixels, ok = rectified_to_native(restored, camera)
            if not ok.all():
                raise ValueError('Measured pixels failed crop/native roundtrip at PTS ' + str(stamp))
            closure.extend(np.linalg.norm(pixels - native['body_uv'][row, usable], axis=1))
        if slot in diagnostic_slots:
            diagnostics.append(dict(model_index=slot, source_pts=stamp,
                                    **_resampling_check(bgr, camera, rgb, mapping)))
        crops.append(rgb); joints.append(obs); boxes.append(box); affine.append(transform)
        crop_k.append(mapping['K_rect_crop']); sigmas.append(mapping['anti_alias_native_gaussian_sigma'])
        fractions.append(fraction); crop_coords.append(crop_xy)
        frame_hashes.append(hashlib.sha256(bgr.tobytes()).hexdigest())
    closure_stats = _summary(closure)
    if closure_stats['count'] and closure_stats['maximum'] > 1e-4:
        raise ValueError('Measured native/crop pixel roundtrip exceeds numerical closure tolerance')
    model_pts = pts[model_rows]
    prepared = dict(source_pts=model_pts, official_pts=model_pts,
                    model_index=np.arange(len(model_rows), dtype=np.int64),
                    native50_row=model_rows, shot_slot=np.arange(len(model_rows), dtype=np.int32),
                    sequence_id=np.zeros(len(model_rows), np.int32),
                    source_seconds=model_pts / TIME_BASE, model_sample_time_s=model_pts / TIME_BASE,
                    source_time_base=np.array([1, TIME_BASE]),
                    rgb_crop256=np.stack(crops), rect_coco17=np.stack(joints),
                    bbox_rect_xys=np.asarray(boxes, np.float32),
                    K_rect=native['K_rect'][model_rows].astype(np.float32),
                    R_w2c=pose['rotation'][model_rows].astype(np.float32),
                    camera25=cameras[model_rows], camera_centre_scene=pose['centre'][model_rows],
                    camera_centre_m=native['camera_centre_m'][model_rows],
                    full_rect_to_crop_affine=np.asarray(affine), K_rect_crop=np.asarray(crop_k),
                    crop_joint_xy=np.asarray(crop_coords),
                    anti_alias_native_gaussian_sigma=np.asarray(sigmas),
                    crop_valid_fraction=np.asarray(fractions),
                    source_frame_bgr_sha256=np.asarray(frame_hashes),
                    source_mapping_missing=native['source_mapping_missing'][model_rows],
                    source_legacy_pts=native['source_legacy_pts'][model_rows],
                    shared_betas=betas, body_scale=np.array(body_scale), **proposals)
    references = {}
    for key, path in dict(compact_smplx=compact_smplx or base / 'compact_smplx.npz',
                          skate_binding=skate_binding or base / 'skate_s01_1103/binding.json').items():
        path = Path(path)
        references[key] = dict(path=str(path), sha256=_sha256(path))
    contract = dict(schema='rinkspace.action-clip-input.v1', clip_id=clip_id, shot_id=shot_id,
        source=dict(path=str(video), sha256=video_hash, native_size_wh=list(NATIVE_WH),
                    time_base=[1, TIME_BASE], pts_range=pts[[0, -1]].tolist()),
        model_time=dict(rate_hz=25, step_pts=MODEL_STEP, count=len(model_rows),
                        sequence_count=1, phase_first_pts=int(model_pts[0]),
                        timestamps_or_dt_passed_to_network=False,
                        checkpoint_training_rhythm_compatibility='unverified'),
        native50_count=len(pts), missing_policy=missing_policy,
        missing_mapping_pts=pts[native['source_mapping_missing']].tolist(),
        invalid_original_observation_pts=pts[~native['source_observation_valid']
                                             & ~native['source_mapping_missing']].tolist(),
        roi_proposal_pts=model_pts[proposals['bbox_inferred']].tolist(),
        roi_semantics='Only image crop location; linear native bbox interpolation between real mapped '
                      'boxes; not fitting observations, independent evidence, or measured source points',
        pixel_domains=dict(body_uv='original native Brown 1920x1080 pixels',
                           rect_coco17='full virtual rectified-image pixels, score zero for unavailable points',
                           rgb_crop256='RGB uint8; direct Brown remap into 256x256',
                           normalization='RGB/255, subtract ImageNet mean, divide ImageNet std',
                           mapping='native Brown -> virtual K_rect -> saved full_rect_to_crop_affine',
                           rolling_shutter='midpoint camera rectification only; original signed readout '
                                           'and native row-time equation remain necessary in joint projection'),
        camera=camera_info, camera25_layout='C3,world-to-camera Rodrigues3,log(fx,fy),cx,cy,skew0,Brown14',
        camera25_layout_sources=['camera_bridge.py:3-7', 'native/include/rinkspace/camera_state.hpp '
                                'motion_camera_from_values', 'src/rinkspace/core/native.py evaluate_motion_camera'],
        checks=dict(native_camera25_midpoint_reprojection_px=camera_check,
                    native_crop_roundtrip_px=closure_stats,
                    source_downsampling_diagnostics=diagnostics,
                    source_frames_decoded=len(crops), source_pts_not_rewritten=True,
                    minimum_valid_crop_fraction=float(min(fractions))),
        observation_source=dict(path=str(observations_path), sha256=_sha256(observations_path),
                                mapping_path=str(mapping_path), mapping_sha256=_sha256(mapping_path),
                                selection='accepted_official_pts only; prefer valid original row then lower ROI MAE',
                                mask='existing original_observation_mask AND original valid; no new source points'),
        shared_model=references, body_scale=float(body_scale), shared_betas=betas.tolist(),
        action_labels='candidate clip labels only; this helper assigns no action or contact truth',
        inference_performed=False, optimization_performed=False,
        prepared_npz=str(prepared_path), native50_npz=str(native_path))
    cache.mkdir(parents=True, exist_ok=True); clip_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(native_path, **native)
    np.savez_compressed(prepared_path, **prepared)
    contract['prepared_sha256'] = _sha256(prepared_path)
    contract['native50_sha256'] = _sha256(native_path)
    contract_path.write_text(json.dumps(contract, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf8')
    return dict(prepared=str(prepared_path), native50=str(native_path), contract=str(contract_path),
                native50_count=len(pts), model25_count=len(model_rows),
                roi_proposal_pts=contract['roi_proposal_pts'], checks=contract['checks'])
