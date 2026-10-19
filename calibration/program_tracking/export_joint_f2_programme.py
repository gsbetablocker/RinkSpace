"""Export a complete, explicitly unvalidated programme in one shared F2 scene.

Only the two supplied rig models are re-evaluated. The complete secondary
stream is retained numerically, with its old scene binding recorded. Official
source exposure identities and the newer main camera-pose groups stay distinct.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from runtime.bootstrap import add_geometry_dependencies
add_geometry_dependencies()
import numpy as np

from candidate_scene_geometry_audit import audit_exported_circle_geometry
from render_result import DEFAULT_SHOTS, expected_source_pts, load_tracked_poses

F2_KEY = 2001
TMP = Path('D:/CodexData/RinkSpace_program_tracking_tmp')
BASE = TMP/'joint_programme_atlas_provenance_full_v13'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False)+'\n', encoding='utf8')


def shared_scene(states):
    baseline = json.loads((BASE/'candidate_scene.json').read_text(encoding='utf8'))
    scene = copy.deepcopy(baseline)
    q = np.asarray(states[F2_KEY], np.float64)
    if q.shape != (11,) or not np.isfinite(q).all():
        raise ValueError('Shared F2 delta must be a finite Vector11')
    feature = next(row for row in scene['ice_features'] if row['id'] == 'F2')
    if 'center_xyz' in feature:
        feature['center_xyz'][:2] = (np.asarray(feature['center_xyz'][:2])+100*q[:2]).tolist()
    if 'radius' in feature:
        feature['radius'] += float(100*q[2])
    for ring in feature['rings']:
        ring['center_xyz'][:2] = (np.asarray(ring['center_xyz'][:2])+100*q[:2]).tolist()
        if ring['role'] == 'outer':
            ring['radius'] += float(100*q[2])
    feature['geometry_status'] = 'joint_main_low_shared_f2_unvalidated'
    scene['joint_shared_f2_revision'] = dict(
        base_scene_sha256=sha(BASE/'candidate_scene.json'), state11=q.tolist(),
        parameter_role='increment about candidate F2, not original absolute ring state',
        changed_geometry='F2 common center and outer radius only; inner radii retained',
        optimized_rigs=['main_long_side', 'low_end'],
        secondary_camera_parameters='unchanged; renders the same revised F2 world',
        scene_units='conditional scene units; metric scale unverified', accepted=False)
    scene['shared_geometry_checks'] = audit_exported_circle_geometry(
        scene, baseline, sha(BASE/'candidate_scene.json'))
    if not scene['shared_geometry_checks']['hard_geometry_valid']:
        raise ValueError('Exported F2 violates finite/nested circle geometry')
    return scene


def export(output, states, rigs, *, main_directory, low_directory,
           secondary_directory, joint_state_path, extra_reference_models=()):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Use a fresh joint programme export directory')
    joint_state_path = Path(joint_state_path).resolve(strict=True)
    with np.load(joint_state_path, allow_pickle=False) as saved:
        keys, values = saved['keys'], saved['states']
        if set(map(int, keys)) != set(states) or any(
                not np.array_equal(states[int(k)], value)
                for k, value in zip(keys, values)):
            raise ValueError('Joint export does not match its saved solver state')
    by_rig = {rig.camera_id: rig for rig in rigs}
    if set(by_rig) != {'main_long_side', 'low_end'}:
        raise ValueError('Export expects the complete main and low rig models')
    timeline = json.loads(DEFAULT_SHOTS.read_text(encoding='utf8'))
    expected = expected_source_pts(timeline)
    source_hash = timeline['source_sha256_from_timeline']
    base_hash = sha(BASE/'candidate_scene.json')
    directories = {'main_long_side': Path(main_directory),
                   'low_end': Path(low_directory),
                   'secondary_long_side': Path(secondary_directory)}
    cameras = np.full((len(expected), 25), np.nan)
    shot_ids = np.full(len(expected), -1, np.int16)
    camera_ids = np.full(len(expected), '', dtype='<U32')
    official, poses = expected.copy(), expected.copy()
    positions = {int(p): i for i, p in enumerate(expected)}
    components, group_packets = [], {}
    for group, directory in directories.items():
        path = directory/('initial_cameras.npz' if group in by_rig
                          else 'shot_cameras.npz')
        with np.load(path, allow_pickle=False) as z:
            packet = {key: z[key].copy() for key in z.files}
        if str(packet['source_sha256']) != source_hash or str(packet['model_sha256']) != base_hash:
            raise ValueError('Input rig source or base scene differs')
        pts = np.asarray(packet['pts'], np.int64)
        required = np.concatenate([np.arange(row['start_pts'], row['end_pts'], 256)
                                   for row in timeline['shots'] if row['camera_id'] == group])
        if not np.array_equal(pts, required):
            raise ValueError('Joint export cannot omit any source frame of a rig')
        indices = np.asarray([positions[int(p)] for p in pts])
        rig = by_rig.get(group)
        current = (np.asarray([rig.programme.camera25(int(p), states) for p in pts])
                   if rig is not None else packet['camera25'].copy())
        aliases = np.asarray(packet['exposure_representative_pts'], np.int64)
        pose_ids = np.asarray(packet.get('source_pose_representative_pts', aliases), np.int64)
        local = {int(p): i for i, p in enumerate(pts)}
        if (current.shape != (len(pts), 25) or not np.isfinite(current).all()
                or any(int(rep) not in local or not np.array_equal(current[i], current[local[int(rep)]])
                       for i, rep in enumerate(pose_ids))):
            raise ValueError('Camera stream or source pose alias identity is invalid')
        for row_id, row in enumerate(timeline['shots']):
            if row['camera_id'] == group:
                mask = (pts >= row['start_pts']) & (pts < row['end_pts'])
                if not np.all(packet['shot_index'][mask] == row_id):
                    raise ValueError('Input camera crossed a reviewed source shot')
        cameras[indices], shot_ids[indices], camera_ids[indices] = current, packet['shot_index'], group
        official[indices], poses[indices] = aliases, pose_ids
        group_packets[group] = dict(pts=pts, camera25=current, shot_index=packet['shot_index'],
            exposure_representative_pts=aliases, official_exposure_representative_pts=aliases,
            source_pose_representative_pts=pose_ids)
        components.append(dict(camera_id=group, input_camera_path=str(path.resolve()),
            input_camera_sha256=sha(path), input_scene_sha256=base_hash, frames=len(pts),
            joint_camera_optimized=rig is not None, secondary_numerically_unchanged=rig is None))
    if np.any(shot_ids < 0) or not np.isfinite(cameras).all():
        raise ValueError('Full programme export lacks a source frame')
    scene = shared_scene(states)
    output.mkdir(parents=True)
    write_json(output/'candidate_scene.json', scene)
    scene_hash = sha(output/'candidate_scene.json')
    release = json.loads((BASE/'candidate_release.json').read_text(encoding='utf8'))
    release.update(model_sha256=scene_hash, model_path=str(output/'candidate_scene.json'),
        candidate_scene=str(output/'candidate_scene.json'), base_model_sha256=base_hash,
        revision='joint-main-low-shared-F2-unvalidated',
        status='joint_source_camera_scene_candidate_unvalidated', registered_as_latest_model=False)
    write_json(output/'candidate_release.json', release)
    for group, packet in group_packets.items():
        folder = output/group
        folder.mkdir()
        target = folder/'shot_cameras.npz'
        np.savez_compressed(target, **packet, source_sha256=source_hash, model_sha256=scene_hash)
        write_json(folder/'summary.json', dict(camera_id=group, frames=len(packet['pts']),
            sha256=sha(target), source_sha256=source_hash, model_sha256=scene_hash, accepted=False))
    for ordinal, model in enumerate(extra_reference_models):
        pts = np.asarray(sorted(model.frames), np.int64)
        np.savez_compressed(output/f'extra_reference_cameras_{ordinal:02d}.npz', pts=pts,
            camera25=np.asarray([model.camera25(int(p), states) for p in pts]),
            source_sha256=source_hash, model_sha256=scene_hash)
    target = output/'candidate_poses.npz'
    np.savez_compressed(target, pts=expected, camera25=cameras, shot_index=shot_ids,
        camera_id=camera_ids, status=np.full(len(expected), 'temporal_candidate'),
        metrics=np.full((len(expected), 1), np.nan), source_sha256=source_hash,
        model_sha256=scene_hash, exposure_representative_pts=official,
        official_exposure_representative_pts=official, source_pose_representative_pts=poses,
        localization_acceptance='unvalidated_candidate')
    load_tracked_poses(target, timeline, scene_hash, candidate_review=True)
    manifest = dict(schema='rinkspace.joint-main-low-shared-f2-programme-export.v1',
        frames=len(expected), source_sha256=source_hash, model_sha256=scene_hash,
        poses_sha256=sha(target), joint_state_path=str(joint_state_path),
        joint_state_sha256=sha(joint_state_path), components=components,
        official_exposure_and_pose_roles_separate=True,
        complete_timeline=True, accepted=False,
        scope='All 10140 native PTS; common F2 scene; real-video accuracy unverified')
    write_json(output/'candidate_poses.json', manifest)
    return manifest
