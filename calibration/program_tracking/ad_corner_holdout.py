"""Measure advertisement corners on native frames excluded from joint fitting.

The fitted camera and ad geometry select a broad ROI.  Every observation is
read from the original video color/edge pixels, with SAM2 person exclusion.
No corner is inserted from a homography or from the model projection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

try:
    from .ad_corners import (_build_templates, _color_quad_from_proposal,
                             _color_rectangle, _edge_quad, _historic_quads,
                             _match_template, _proposal_near_scene, _root_sift)
    from .projection import project_fast
    from .render_result import (DEFAULT_MASKS, DEFAULT_PEOPLE, DEFAULT_SHOTS,
                                PersonMasks, decode_selected)
except ImportError:
    from ad_corners import (_build_templates, _color_quad_from_proposal,
                            _color_rectangle, _edge_quad, _historic_quads,
                            _match_template, _proposal_near_scene, _root_sift)
    from projection import project_fast
    from render_result import (DEFAULT_MASKS, DEFAULT_PEOPLE, DEFAULT_SHOTS,
                               PersonMasks, decode_selected)


ROOT = Path(__file__).resolve().parents[2]
TEMP = Path('D:/CodexData/RinkSpace_program_tracking_tmp')
ROLES = ('left_top', 'right_top', 'right_bottom', 'left_bottom')


def _midpoint_samples(native_pts: np.ndarray, fitted_pts: np.ndarray,
                      fitted_ads: dict[int, list[int]], interval_pts: int = 30720
                      ) -> list[int]:
    """Sample actual native PTS strictly between joint keyframes, ~2 s apart."""
    fitted_pts = np.asarray(sorted(set(map(int, fitted_pts))), np.int64)
    if len(fitted_pts) < 2 or not np.all(np.diff(fitted_pts) > 0):
        raise ValueError('At least two ordered joint keyframes are required')
    index = {int(p): i for i, p in enumerate(native_pts)}
    mids = []
    for a, b in zip(fitted_pts[:-1], fitted_pts[1:]):
        inside = native_pts[(native_pts > a) & (native_pts < b)]
        if len(inside):
            mids.append(int(inside[len(inside)//2]))
    if not mids:
        raise ValueError('No native frame between the fitted keyframes')
    marks = np.arange(int(native_pts[0])+interval_pts//2,
                      int(native_pts[-1])+1, interval_pts)
    if not len(marks):
        marks = np.asarray([int(native_pts[len(native_pts)//2])])
    chosen = {min(mids, key=lambda p: abs(p-int(mark))) for mark in marks}
    # Ensure a few independent test frames see each known physical print,
    # without selecting one of the fitted 373 timestamps.
    for ad_pts in fitted_ads.values():
        visible = [p for p in ad_pts if native_pts[0] <= p <= native_pts[-1]]
        if not visible:
            continue
        targets = np.asarray(visible)[np.rint(
            np.linspace(0, len(visible)-1, min(2, len(visible)))).astype(int)]
        chosen.update(min(mids, key=lambda p: abs(p-int(mark))) for mark in targets)
    chosen = sorted(chosen)
    if any(p in set(fitted_pts) or p not in index for p in chosen):
        raise AssertionError('Holdout selection includes a fitted or nonnative timestamp')
    return chosen


def run(tracks: list[Path], joint: Path, output: Path,
        source_shots: Path = DEFAULT_SHOTS,
        fitted_corners: Path = TEMP/'program_ad_corners/corners_annotated.json',
        masks_path: Path = DEFAULT_MASKS,
        people_path: Path = DEFAULT_PEOPLE) -> dict:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Use the already-configured CUDA SIFT matching runtime')
    if not output.resolve().is_relative_to(TEMP.resolve()):
        raise ValueError('Holdout cache must stay outside the OneDrive project')
    shots = json.loads(source_shots.read_text(encoding='utf-8'))
    source_hash = shots['source_sha256_from_timeline']
    scene_bytes = (ROOT/'calibration/deliverables/scene.json').read_bytes()
    scene = json.loads(scene_bytes)
    scene_hash = hashlib.sha256(scene_bytes).hexdigest()
    diagnostics_path = joint/'diagnostics.json'
    diagnostics = json.loads(diagnostics_path.read_text(encoding='utf-8'))
    geometry = {int(i): item['corners_xyz_scene_units'] for i, item
                in diagnostics['ad_geometry_candidates'].items()}
    fitted = json.loads(fitted_corners.read_text(encoding='utf-8'))
    if fitted['source_sha256'] != source_hash:
        raise ValueError('Fitted ad manifest refers to a different original video')
    fitted_ad_pts = defaultdict(list)
    for row in fitted['observations']:
        fitted_ad_pts[int(row['user_number'])].append(int(row['pts']))
    for key in fitted_ad_pts:
        fitted_ad_pts[key] = sorted(set(fitted_ad_pts[key]))
    all_joint_pts = set()
    for path in joint.glob('shot??_keyframes.npz'):
        with np.load(path, allow_pickle=False) as z:
            all_joint_pts.update(map(int, z['pts']))
    native = {}
    selections = {}
    for track in tracks:
        with np.load(track, allow_pickle=False) as z:
            shot_id = int(z['shot_index'])
            pts = z['pts'].astype(np.int64)
            camera = z['camera25'].astype(float)
            state = z['status'].astype(str)
            if str(z['source_sha256']) != source_hash or str(z['model_sha256']) != scene_hash:
                raise ValueError(f'Source or model mismatch: {track}')
        shot = shots['shots'][shot_id]
        expected = np.arange(shot['start_pts'], shot['end_pts'], 256, dtype=np.int64)
        if not np.array_equal(expected, pts) or not np.isfinite(camera).all():
            raise ValueError(f'Incomplete native track: {track}')
        with np.load(joint/f'shot{shot_id:02d}_keyframes.npz', allow_pickle=False) as z:
            key_pts = z['pts'].astype(np.int64)
        samples = _midpoint_samples(pts, key_pts, fitted_ad_pts)
        samples = [p for p in samples if p not in all_joint_pts]
        selections[shot_id] = samples
        index = {int(p): i for i, p in enumerate(pts)}
        for p in samples:
            native[p] = (shot_id, camera[index[p]], str(state[index[p]]))
    needed_refs = {p for p, _ in _historic_quads(scene)} | {229679360}
    refs = dict(decode_selected(Path(shots['source_path']), sorted(needed_refs)))
    reviewed_path = TEMP/'program_ad_corners/reviewed_seeds.json'
    reviewed = json.loads(reviewed_path.read_text(encoding='utf-8'))
    templates = _build_templates(scene, refs, reviewed)
    sift = cv2.SIFT_create(nfeatures=1900, contrastThreshold=.025)
    person_masks = PersonMasks(masks_path, people_path)
    observations = []
    frame_status = []
    for pts, image in decode_selected(Path(shots['source_path']), sorted(native)):
        shot_id, camera, track_status = native[pts]
        person, mask_status = person_masks.get(pts, require_sam2=True)
        small = cv2.resize(image, (960, 540), interpolation=cv2.INTER_AREA)
        keep = cv2.resize((~person).astype(np.uint8)*255, (960, 540),
                          interpolation=cv2.INTER_NEAREST)
        keys, descriptors = sift.detectAndCompute(
            cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), keep)
        feature_xy = np.asarray([key.pt for key in keys], np.float32).reshape(-1, 2)*2
        feature_desc = (_root_sift(descriptors) if descriptors is not None
                        else np.empty((0, 128), np.float32))
        found_ids = {}
        for ad_id, corners in geometry.items():
            world = np.asarray([corners[role] for role in ROLES], float)
            predicted, depth = project_fast(camera, world)
            if not np.all(depth > 0) or not np.isfinite(predicted).all():
                continue
            projected = dict(zip(ROLES, predicted))
            evidence = {}
            identity = None
            if ad_id == 2:
                evidence = _color_rectangle(image, person, projected, 'purple') or {}
                identity = {'kind': 'purple_source_print_camera_roi'}
            elif ad_id in templates:
                match = _match_template(templates[ad_id], feature_xy, feature_desc,
                                        min_inliers=11 if ad_id == 4 else 8)
                if match is None:
                    continue
                quad, inliers = match
                if not _proposal_near_scene(quad, projected):
                    continue
                color = {1: 'blue', 3: 'navy', 6: 'red', 7: 'blue'}.get(ad_id)
                if color:
                    evidence = _color_quad_from_proposal(image, person, quad, color) or {}
                    component = _color_rectangle(
                        image, person, dict(zip(ROLES, quad)), color,
                        max_left_shift=110.) or {}
                    for role, item in component.items():
                        if role not in evidence and np.linalg.norm(
                                np.asarray(item['uv'])-dict(zip(ROLES, quad))[role]) < 25:
                            evidence[role] = item
                elif ad_id == 5:
                    evidence = _edge_quad(image, person, quad, radius=16)
                    if len(evidence) < 2:
                        evidence = {}
                identity = {'kind': 'same_print_sift_cuda', 'inliers': inliers,
                            'reference_pts': int(templates[ad_id]['pts'])}
            for role, item in evidence.items():
                if role not in projected:
                    continue
                observed_uv = np.asarray(item['uv'], float)
                predicted_uv = projected[role]
                delta = observed_uv-predicted_uv
                observations.append({
                    'pts': pts, 'shot_index': shot_id,
                    'camera_id': shots['shots'][shot_id]['camera_id'],
                    'track_status': track_status, 'user_number': ad_id,
                    'role': role, 'observed_uv': observed_uv.tolist(),
                    'predicted_uv': predicted_uv.tolist(),
                    'error_xy_px': delta.tolist(),
                    'error_px': float(np.linalg.norm(delta)),
                    'measurement_origin': item['measurement_origin'],
                    'measurement_component': ('meiwa_color_bar_envelope'
                                              if ad_id == 5 else 'printed_outer_color_frame'),
                    'mask_status': mask_status,
                    'identity_evidence': identity,
                    'status': 'holdout_actual_source_pixel_not_joint_fit',
                })
                found_ids[ad_id] = found_ids.get(ad_id, 0)+1
        frame_status.append({'pts': pts, 'shot_index': shot_id,
                             'track_status': track_status,
                             'measured_ad_ids': found_ids})
    if len(frame_status) != len(native):
        raise ValueError('Original source decoder missed a holdout timestamp')
    groups = {}
    for name, key in (('by_ad', 'user_number'), ('by_camera', 'camera_id')):
        group = defaultdict(list)
        for row in observations:
            group[str(row[key])].append(row['error_px'])
        groups[name] = {i: {'corners': len(v), 'median_px': float(np.median(v)),
                            'p95_px': float(np.percentile(v, 95)),
                            'max_px': float(np.max(v))}
                        for i, v in group.items()}
    manifest = {
        'schema_version': 'rinkspace.ad-corner-native-holdout.v1',
        'source_sha256': source_hash,
        'model_sha256': scene_hash,
        'joint_diagnostics_sha256': hashlib.sha256(diagnostics_path.read_bytes()).hexdigest(),
        'fitted_ad_manifest_sha256': hashlib.sha256(fitted_corners.read_bytes()).hexdigest(),
        'track_paths': [str(p) for p in tracks],
        'sampling': 'native frame between consecutive fitted keyframes; ~2 s plus sparse ad coverage',
        'holdout_count': len(native), 'joint_keyframe_count': len(all_joint_pts),
        'not_used_in_joint_corner_fit': True,
        'selections': selections,
        'observations': observations, 'frame_status': frame_status,
        'summary': groups,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'output': str(output), 'frames': len(native),
                      'corners': len(observations), 'summary': groups}))
    return manifest


def rescore(tracks: list[Path], joint: Path, observations: Path, output: Path,
            source_shots: Path = DEFAULT_SHOTS,
            shot_indices: list[int] | None = None) -> dict:
    """Reuse frozen real corner pixels when evaluating a new camera solve.

    No video decoding, feature matching or pixel selection happens here. The
    Advertisement pixels must be absent from the joint corner observations.
    An all-native graph can still contain the frame's camera, LK and board
    observations; that is a withheld-feature check, not an unseen-frame test.
    """
    if not output.resolve().is_relative_to(TEMP.resolve()):
        raise ValueError('Holdout cache must stay outside the OneDrive project')
    source = json.loads(source_shots.read_text(encoding='utf-8'))
    measured = json.loads(observations.read_text(encoding='utf-8'))
    source_rows = measured['observations']
    if shot_indices is not None:
        selected = set(shot_indices)
        if not selected or not selected.issubset(range(len(source['shots']))):
            raise ValueError('Unknown or empty holdout shot selection')
        source_rows = [row for row in source_rows if any(
            int(source['shots'][index]['start_pts']) <= int(row['pts']) <
            int(source['shots'][index]['end_pts']) for index in selected)]
    scene_hash = hashlib.sha256((ROOT/'calibration/deliverables/scene.json').read_bytes()).hexdigest()
    source_hash = source['source_sha256_from_timeline']
    if measured.get('source_sha256') != source_hash or measured.get('model_sha256') != scene_hash:
        raise ValueError('Frozen observations refer to a different source or scene')
    diagnostic_path = joint/'diagnostics.json'
    diagnostic = json.loads(diagnostic_path.read_text(encoding='utf-8'))
    if (diagnostic['source_sha256'] != source_hash or
            diagnostic.get('base_model_sha256', diagnostic['model_sha256']) != scene_hash):
        raise ValueError('Joint geometry provenance differs')
    candidate_hash = diagnostic['model_sha256']
    if candidate_hash != scene_hash:
        scene_path = Path(diagnostic['effective_scene_file'])
        payload = scene_path.read_bytes()
        candidate = json.loads(payload)
        if (hashlib.sha256(payload).hexdigest() != candidate_hash or
                candidate['programme_calibration_candidate']['base_model_sha256'] != scene_hash):
            raise ValueError('Effective candidate scene does not match its base/provenance')
    fitted_pts = set()
    for path in joint.glob('shot??_keyframes.npz'):
        with np.load(path, allow_pickle=False) as data:
            fitted_pts.update(map(int, data['pts']))
    needed = {int(row['pts']) for row in source_rows}
    if 'ad_corner_errors' not in diagnostic:
        raise ValueError('Joint output must identify every fitted advertisement corner')
    fitted_ad_pts = {int(row['pts']) for row in diagnostic['ad_corner_errors']}
    if needed & fitted_ad_pts:
        raise ValueError('Frozen holdout timestamps supplied advertisement pixels to this joint fit')
    cameras = {}
    track_hashes = {}
    for path in tracks:
        with np.load(path, allow_pickle=False) as data:
            if str(data['source_sha256']) != source_hash or str(data['model_sha256']) != candidate_hash:
                raise ValueError(f'Native camera provenance differs: {path}')
            shot_id = int(data['shot_index'])
            shot = source['shots'][shot_id]
            expected = np.arange(shot['start_pts'], shot['end_pts'], 256, dtype=np.int64)
            if not np.array_equal(data['pts'], expected):
                raise ValueError(f'Incomplete native PTS: {path}')
            for i in np.flatnonzero(np.isin(data['pts'], list(needed))):
                stamp = int(data['pts'][i])
                if stamp in cameras:
                    raise ValueError(f'Duplicate native camera at {stamp}')
                cameras[stamp] = (data['camera25'][i].copy(), shot_id, shot['camera_id'])
        track_hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    if needed - cameras.keys():
        raise ValueError(f'Missing cameras for holdout PTS: {sorted(needed-cameras.keys())[:5]}')
    rows = []
    for row in source_rows:
        stamp, number, role = int(row['pts']), int(row['user_number']), row['role']
        camera, shot_id, camera_id = cameras[stamp]
        observed = np.asarray(row['observed_uv'] if 'observed_uv' in row else row['uv'], float)
        xyz = diagnostic['ad_geometry_candidates'][str(number)]['corners_xyz_scene_units'][role]
        predicted, depth = project_fast(camera, np.asarray([xyz], float))
        if not np.isfinite(predicted).all() or depth[0] <= 0:
            raise ValueError(f'Invalid holdout projection at {stamp}')
        delta = predicted[0]-observed
        rows.append({**row, 'observed_uv': observed.tolist(),
                     'predicted_uv': predicted[0].tolist(), 'shot_index': shot_id,
                     'camera_id': camera_id, 'error_xy_px': delta.tolist(),
                     'error_px': float(np.linalg.norm(delta))})
    summary = {}
    for group_name, key in (('by_ad', 'user_number'), ('by_camera', 'camera_id'), ('by_shot', 'shot_index')):
        groups = defaultdict(list)
        for row in rows:
            groups[str(row[key])].append(row['error_px'])
        summary[group_name] = {name: {'corners': len(values), 'median_px': float(np.median(values)),
                                     'p95_px': float(np.percentile(values, 95)), 'max_px': float(max(values))}
                               for name, values in groups.items()}
    result = {'schema_version': 'rinkspace.ad-corner-native-holdout.v1',
              'source_sha256': source_hash, 'model_sha256': candidate_hash,
              'observation_base_model_sha256': scene_hash,
              'observations_path': str(observations),
              'observations_sha256': hashlib.sha256(observations.read_bytes()).hexdigest(),
              'joint_diagnostics_sha256': hashlib.sha256(diagnostic_path.read_bytes()).hexdigest(),
              'track_sha256': track_hashes, 'observed_frames': len(needed),
              'selected_shots': sorted(set(shot_indices)) if shot_indices is not None else None,
              'not_used_in_joint_corner_fit': True, 'pixels_remeasured': False,
              'frames_present_as_joint_camera_states': len(needed & fitted_pts),
              'validation_scope': 'withheld_advertisement_pixels; camera states may use other source features',
              'observations': rows, 'summary': summary}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({'output': str(output), 'observed_frames': len(needed),
                      'corners': len(rows), 'summary': summary}))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--track', type=Path, action='append', required=True)
    parser.add_argument('--joint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fitted-corners', type=Path,
                        default=TEMP/'program_ad_corners/corners_annotated.json')
    parser.add_argument('--observations', type=Path,
                        help='Rescore frozen actual holdout pixels without measuring them again')
    parser.add_argument('--shot-index', type=int, action='append',
                        help='Explicit partial-shot rescore of frozen pixels; requires --observations')
    args = parser.parse_args()
    if args.observations is not None:
        rescore(args.track, args.joint, args.observations, args.output,
                shot_indices=args.shot_index)
    else:
        if args.shot_index is not None:
            parser.error('--shot-index requires frozen --observations')
        run(args.track, args.joint, args.output, fitted_corners=args.fitted_corners)
