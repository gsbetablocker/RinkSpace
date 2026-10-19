"""An initializer must not silently rewrite shared geometry or source state."""
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


MODULE = Path(__file__).parents[1] / 'calibration/solver_runtime/legacy/structured_programme_source_candidate/absolute_camera_pose_initializer.py'
spec = importlib.util.spec_from_file_location('camera_initializer_under_test', MODULE)
initializer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(initializer)


def packet(tmp_path, indices=(8, 9), pts=230419968):
    q = np.arange(24, dtype=np.float64) / 10
    source = tmp_path / 'bound_source.png'
    source.write_bytes(b'unchanged bound source evidence')
    arrays = tmp_path / 'pose_values.npz'
    indices = np.asarray(indices, dtype=np.int64)
    np.savez(arrays, parameter_indices=indices,
             original_values=q[indices], candidate_values=q[indices] + .02)
    metadata = dict(
        schema='rinkspace.source-qualified-camera-pose-initializer.v1',
        role='initializer_only', accuracy_accepted=False,
        source_video_sha256=initializer.SOURCE_SHA, time_base='1/15360',
        parent_q_sha256=initializer._q_sha(q), actual_pts=[pts],
        delta_npz=dict(path=str(arrays), sha256=initializer._sha(arrays)),
        source_evidence=[dict(path=str(source), sha256=initializer._sha(source), pts=pts)])
    path = tmp_path / 'initializer.json'
    path.write_text(json.dumps(metadata), encoding='utf8')
    block = SimpleNamespace(start=8, bundle=SimpleNamespace(
        reference_pts=230369024, index={pts: 0}))
    state = SimpleNamespace(bypts={pts: block}, original=SimpleNamespace(aliases={pts: pts}))
    return state, q, path, metadata


def test_exact_source_camera_estimate_preserves_every_other_parameter(tmp_path):
    state, q, path, _ = packet(tmp_path)
    original = q.tobytes()
    q.flags.writeable = False
    candidate, report = initializer.verified_camera_pose_estimate(state, q, path)
    assert q.tobytes() == original
    assert np.array_equal(np.flatnonzero(candidate != q), [8, 9])
    assert np.array_equal(candidate[np.r_[0:8, 10:24]], q[np.r_[0:8, 10:24]])
    assert report['fresh_complete_source_profiles_and_jacobian_required']


def test_shared_or_world_parameter_cannot_be_smuggled_into_camera_initializer(tmp_path):
    state, q, path, _ = packet(tmp_path, indices=(2, 8))
    with pytest.raises(ValueError, match='protected state'):
        initializer.verified_camera_pose_estimate(state, q, path)


def test_old_parent_and_changed_evidence_are_rejected(tmp_path):
    state, q, path, metadata = packet(tmp_path)
    wrong = q.copy()
    wrong[3] += 1
    with pytest.raises(ValueError, match='Exact parent'):
        initializer.verified_camera_pose_estimate(state, wrong, path)
    Path(metadata['source_evidence'][0]['path']).write_bytes(b'different source')
    with pytest.raises(ValueError, match='evidence has changed'):
        initializer.verified_camera_pose_estimate(state, q, path)


def test_reference_exposure_cannot_be_reinitialized(tmp_path):
    state, q, path, metadata = packet(tmp_path, pts=230369024)
    with pytest.raises(ValueError, match='fixed reference'):
        initializer.verified_camera_pose_estimate(state, q, path)


def physical_packet(tmp_path, indices=(3, 8, 9), pts=230369024):
    state, q, path, metadata = packet(tmp_path, indices=indices, pts=pts)
    block = state.bypts[pts]
    block.xyz_slots = np.array([1, 2, 3])
    block.bundle.calibration = SimpleNamespace(installation_id='same-tripod')
    metadata.update(schema=initializer.PHYSICAL_SCHEMA, installation_id='same-tripod',
        existing_shared_pivot_Z_parameter_index=3, anchor_actual_pts=[pts],
        derived_view_estimates_are_not_source_measurements=True,
        shared_world_Pxy_lever_and_optics_unchanged=True)
    path.write_text(json.dumps(metadata), encoding='utf8')
    return state, q, path, metadata


def test_explicit_shared_height_and_reference_estimate_preserves_world_and_other_mount_parameters(tmp_path):
    state, q, path, _ = physical_packet(tmp_path)
    original = q.copy()
    candidate, report = initializer.verified_camera_pose_estimate(state, q, path)
    assert np.array_equal(q, original)
    assert np.array_equal(np.flatnonzero(candidate != q), [3, 8, 9])
    assert report['shared_pivot_Z_initialized']
    assert report['derived_view_estimates_are_not_source_measurements']
    assert not report['shared_world_Pxy_lever_and_optics_changed']


def test_shared_height_estimate_cannot_change_Pxy_or_other_installation(tmp_path):
    state, q, path, _ = physical_packet(tmp_path, indices=(1, 3, 8))
    with pytest.raises(ValueError, match='protected state'):
        initializer.verified_camera_pose_estimate(state, q, path)
    state, q, path, _ = physical_packet(tmp_path, indices=(3, 8))
    state.bypts[230369024].bundle.calibration.installation_id = 'another-tripod'
    with pytest.raises(ValueError, match='cross installations'):
        initializer.verified_camera_pose_estimate(state, q, path)


def test_shared_height_estimate_requires_anchor_evidence_and_positive_height(tmp_path):
    state, q, path, metadata = physical_packet(tmp_path)
    metadata['anchor_actual_pts'] = [230369280]
    path.write_text(json.dumps(metadata), encoding='utf8')
    with pytest.raises(ValueError, match='source anchor'):
        initializer.verified_camera_pose_estimate(state, q, path)
    state, q, path, metadata = physical_packet(tmp_path)
    arrays = Path(metadata['delta_npz']['path'])
    np.savez(arrays, parameter_indices=np.array([3, 8]), original_values=q[[3, 8]],
             candidate_values=np.array([0., q[8] + .02]))
    metadata['delta_npz']['sha256'] = initializer._sha(arrays)
    path.write_text(json.dumps(metadata), encoding='utf8')
    with pytest.raises(ValueError, match='above ice'):
        initializer.verified_camera_pose_estimate(state, q, path)
