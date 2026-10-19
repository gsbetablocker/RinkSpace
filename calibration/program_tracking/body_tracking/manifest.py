"""Freeze version-bound camera inputs and the active body/grid source scope."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from rinkspace.storage import AssetStore, resolve_asset

PROJECT = Path(__file__).resolve().parents[3]

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from body_tracking.scope import ProgrammeScope, SOURCE_SHOTS
    from body_tracking.camera_bridge import FrozenCameraStream
else:
    from .scope import ProgrammeScope, SOURCE_SHOTS
    from .camera_bridge import FrozenCameraStream

V15 = Path('D:/CodexData/RinkSpace_program_tracking_tmp/continuous_camera_model_v15_joint_f2_resume_20260929/programme_review')
PEOPLE = Path('D:/CodexData/RinkSpace_program_tracking_tmp/people')
EXPECTED_SCENE = '5e4faf7b525e4a16f660a81e45b97aafa6fac39531617927dde11776694cc63d'
EXPECTED_POSES = '5ac09663b320fa42e3722a2e31df93bcd200229248e781546baaccd0d1c5e8fb'


def sha(path):
    with resolve_asset(PROJECT, path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def freeze(output, source_shots=SOURCE_SHOTS, *, poses_path=None, scene_path=None,
           release_path=None, baseline=None, people_boxes_path=None,
           people_masks_path=None):
    """Freeze v15 by default, or an explicit poses/scene/release input set.

    New inputs are validated through the release profile; neither hardcoded
    v15 hashes nor source validation need to be disabled. Existing manifests
    remain immutable. People cache paths may be supplied for another source.
    """
    storage = AssetStore(PROJECT)
    output = storage.resolve(output)
    if storage.configured and not output.is_relative_to(storage.data):
        raise ValueError('Frozen run products must use the registered data root')
    source_shots = storage.resolve(source_shots)
    if (output / 'input_manifest.json').exists():
        raise FileExistsError('Existing frozen input manifest; inspect/reuse it explicitly')
    explicit = any(p is not None for p in (poses_path, scene_path, release_path))
    if explicit:
        if any(p is None for p in (poses_path, scene_path, release_path)):
            raise ValueError('Explicit camera inputs require poses, scene and release together')
        poses, scene, release = map(storage.resolve, (poses_path, scene_path, release_path))
        camera = FrozenCameraStream(poses, scene, source_shots, release_path=release)
    else:
        poses, scene, release = [storage.resolve(V15 / n) for n in (
            'candidate_poses.npz', 'candidate_scene.json', 'candidate_release.json')]
        camera = FrozenCameraStream(poses, scene, source_shots)
        registered = json.loads(release.read_text(encoding='utf-8-sig'))
        if registered.get('model_sha256') != camera.scene_sha256:
            raise ValueError('v15 release does not bind the selected camera scene')
    scope = ProgrammeScope(camera.source, camera.analysis_scope)
    archive = np.arange(*scope.source['window_pts'], camera.frame_step_pts, dtype=np.int64)
    boxes_path = storage.resolve(people_boxes_path if people_boxes_path is not None else PEOPLE / 'boxes.json')
    masks_path = storage.resolve(people_masks_path if people_masks_path is not None else PEOPLE / 'masks.json')
    boxes = json.loads(boxes_path.read_text(encoding='utf8'))
    masks = json.loads(masks_path.read_text(encoding='utf8'))
    if storage.source(boxes['source']) != storage.source(scope.source['source_path']):
        raise ValueError('People boxes belong to another source')
    if not np.array_equal([r['pts'] for r in boxes['frames']], archive):
        raise ValueError('People cache does not cover the exact original PTS')
    if masks['frame_count'] != len(archive) or boxes['sample_size_wh'] != [960, 540]:
        raise ValueError('Unexpected cache coverage/pixel domain')
    if 'source' in masks and storage.source(masks['source']) != storage.source(scope.source['source_path']):
        raise ValueError('People masks belong to another source')
    arrays = {k: getattr(camera, k) for k in (
        'pts', 'camera25', 'shot_index', 'camera_id', 'status',
        'official_exposure_representative_pts', 'source_pose_representative_pts')}
    scope.assert_pts(arrays['pts'])
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / 'source_camera_index.npz', **arrays,
        programme_time_s=scope.programme_seconds(arrays['pts']))
    version = camera.binding_metadata()
    manifest = dict(schema='rinkspace.body-trajectory-input.v1',
        baseline=baseline or ('explicit_versioned_camera' if explicit else 'A_frozen_final_v15'),
        source_path=scope.source['source_path'], source_sha256=scope.source['source_sha256_from_timeline'],
        source_path_original=scope.source.get('source_path_original', scope.source['source_path']),
        source_sha256_basis='verified source timeline and version-bound camera packet; original video not rehashed',
        source_time_base=scope.source['source_time_base'], archive_window_pts=scope.source['window_pts'],
        analysis_scope=scope.active, frame_count=len(scope.pts), shot_count=len(scope.shots),
        source_shots_path=str(Path(source_shots).resolve()), source_shots_sha256=sha(source_shots),
        poses_path=str(poses.resolve()), camera_stream_sha256=camera.poses_sha256,
        scene_path=str(scene.resolve()), scene_sha256=camera.scene_sha256,
        release_path=str(release), release_sha256=sha(release), camera_status='unvalidated_candidate',
        secondary_camera_status=('unchanged_v10_not_jointly_updated_in_v15' if not explicit
                                else version['camera_group_status'].get('secondary_long_side',
                                                                      version['camera_release_status'])),
        reference_point='smplx_native_pelvis', native_reference_index=0,
        metric_scale_status='unverified', units='conditional_scene_units',
        people_boxes_path=str(boxes_path), people_boxes_sha256=sha(boxes_path),
        people_masks_path=str(masks_path), people_masks_manifest_sha256=sha(masks_path),
        source_camera_index_sha256=sha(output / 'source_camera_index.npz'),
        body_observations_status='not_yet_generated', body_position_status='not_yet_generated',
        camera_optimization=False, additional_trajectory_smoothing=False,
        camera_binding=version)
    (output / 'input_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf8')
    return manifest


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--source-shots', type=Path, default=SOURCE_SHOTS)
    p.add_argument('--poses', type=Path)
    p.add_argument('--scene', type=Path)
    p.add_argument('--release', type=Path)
    p.add_argument('--baseline')
    p.add_argument('--people-boxes', type=Path)
    p.add_argument('--people-masks', type=Path)
    args = p.parse_args()
    print(json.dumps(freeze(args.output, args.source_shots, poses_path=args.poses,
        scene_path=args.scene, release_path=args.release, baseline=args.baseline,
        people_boxes_path=args.people_boxes, people_masks_path=args.people_masks),
        ensure_ascii=False, indent=2))
