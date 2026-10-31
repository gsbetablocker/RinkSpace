"""Existing-shot packets enforce source scope without touching camera controls."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

PROGRAMME = Path(__file__).resolve().parents[1] / 'events/ankara_2026/programmes/okada_mei_sp'
sys.path.insert(0, str(PROGRAMME))
import build_registered_source_packets as packets


@pytest.fixture
def current_capture(tmp_path, monkeypatch):
    prepared, capture = tmp_path / 'prepared', tmp_path / 'capture'
    prepared.mkdir()
    folder = capture / 's01'
    folder.mkdir(parents=True)
    points = dict(pts=np.array([1000], dtype=np.int64), observation_ids=np.array(['old'], dtype='U128'),
        uv=np.array([[123.125, 456.75]]), kind=np.ones(1, dtype=np.int32), stations=np.full(1, .5),
        source_rows=np.array([9], dtype=np.int32), path_indices=np.array([0], dtype=np.int64))
    np.savez(prepared / 'point_observations.npz', **points)
    shot = dict(id='s01', first_pts=1000, last_pts=1256, stop_pts_exclusive=1512,
                time_base_num=1, time_base_den=12800)
    packets.write(prepared / 'joint_manifest.json', dict(shots=[shot], source_video_sha256='a' * 64,
        point_input_sha256=packets.digest(prepared / 'point_observations.npz')))
    packets.write(prepared / 'mapping.json', dict(path_ids=['ice.A'], source_video_sha256='a' * 64))
    scope = tmp_path / 'scope.json'
    packets.write(scope, dict(source=dict(sha256='a' * 64),
        source_clock=dict(time_base_num=1, time_base_den=12800, native_pts_step=256),
        shots=[dict(id='s01', processing=dict(shot, native_frame_count=2))]))
    monkeypatch.setattr(packets, 'SCOPE_PATH', scope)
    qualified = dict(pts=1000, fit_eligible=True, source_BGR_sha256='b' * 64,
        rows=[dict(path_id='ice.A', uv=[123.125, 456.75], fit_eligible=True,
                   measurement_operator='actual_RGB_boundary')])
    pending = dict(pts=1256, fit_eligible=False, rows=[], reason='Source material unresolved')
    path = folder / 'source_rows.jsonl'
    path.write_text(packets.json_line(qualified) + packets.json_line(pending), encoding='utf8')
    return prepared, capture, path, points, qualified, pending


def test_complete_registered_packet_authenticates_and_preserves_input_dtypes(current_capture, tmp_path):
    prepared, capture, path, points, qualified, pending = current_capture
    declaration = dict(policy='partial_line_spans', groups=[dict(pts=1000, path_id='ice.A',
        spans=[[[123.125, 456.75], [123.125, 456.75]]])])
    packets.write(path.parent / 'source_replacement.json', declaration)
    pending_row = b'{"fit_eligible":false,"reason":"existing source review"}\n'
    (path.parent / 'pending.jsonl').write_bytes(pending_row)
    inputs = {p: packets.digest(p) for p in [*prepared.iterdir(), *path.parent.iterdir()]}
    output = tmp_path / 'packets'
    result = packets.build(prepared, capture, output, 's01')
    packet = output / 's01'
    with np.load(packet / 'point_observations.npz', allow_pickle=False) as z:
        assert set(z.files) == set(points)
        assert all(z[key].dtype == points[key].dtype for key in points)
        np.testing.assert_array_equal(z['uv'], [qualified['rows'][0]['uv']])
        np.testing.assert_array_equal(z['pts'], [1000])
    mapping = packets.read(packet / 'mapping.json')
    assert mapping['source_replacement'] == declaration
    assert mapping['source_UV_synthesized'] is False
    assert (packet / 'pending_source_rows.jsonl').read_text() == packets.json_line(pending)
    assert (packet / 'pending.jsonl').read_bytes() == pending_row
    assert len(packets.read(packet / 'source_frames.json')) == 2
    assert not any((packet / name).exists() for name in ('shot_seed.json', 'controls.npy', 'poses.npz'))
    assert all(packets.digest(p) == sha for p, sha in inputs.items())
    assert result['rows'] == 1 and result['shots'][0]['authenticated_partial_span_groups'] == 1
    assert result['camera_pose_or_controls_initialized'] is False


@pytest.mark.parametrize('invalid', ['missing_exposure', 'empty_shot', 'unqualified_row', 'unknown_shot'])
def test_unusable_source_packet_fails_without_creating_output(current_capture, tmp_path, invalid):
    prepared, capture, path, _, qualified, pending = current_capture
    shots = 's01'
    if invalid == 'missing_exposure':
        path.write_text(packets.json_line(qualified), encoding='utf8')
    elif invalid == 'empty_shot':
        qualified['rows'] = []
        path.write_text(packets.json_line(qualified) + packets.json_line(pending), encoding='utf8')
    elif invalid == 'unqualified_row':
        del qualified['rows'][0]['fit_eligible']
        path.write_text(packets.json_line(qualified) + packets.json_line(pending), encoding='utf8')
    else:
        shots = 's09'
    output = tmp_path / 'rejected_packets'
    with pytest.raises(ValueError):
        packets.build(prepared, capture, output, shots)
    assert not output.exists()
