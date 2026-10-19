"""Feedback plumbing mechanics; no new-frame visual accuracy assertion."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'calibration/rinkfit/src'))
from rinkfit_joint.feedback_checkpoint import CheckpointObserver, collect_findings, consume_generation


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path, value):
    path.write_text(json.dumps(value), encoding='utf8')


@pytest.fixture
def checkpoint(tmp_path):
    generation = tmp_path / 'run/generation_0'
    generation.mkdir(parents=True)
    source = tmp_path / 'source'
    source.mkdir()
    arrays = {}
    values = dict(native_pts=np.array([256, 512, 768], '<i8'),
                  shot_index=np.zeros(3, '<u2'), material_index=np.zeros(3, '<u4'),
                  admission=np.ones(3, 'u1'), fit_role=np.ones(3, 'u1'),
                  E=np.ones(3, '<f8'), uv=np.zeros((3, 2), '<f8'),
                  profile_normal=np.tile([0., 1.], (3, 1)))
    for key, value in values.items():
        path = source / (key + '.bin')
        value.tofile(path)
        arrays[key] = dict(file=path.name, sha256=_sha(path), dtype=value.dtype.str, shape=value.shape)
    packet = source / 'packet.json'
    _json(packet, dict(row_count=3, arrays=arrays, shot_ids=['s01'],
                      source_row_ids=['a', 'b', 'c'], source_dataset_ids=['actual'] * 3,
                      queries=[dict(row_index=0)]))
    model = generation / 'model.json'
    _json(model, dict(materials=[dict(label='board.contact')], shots=dict(s01=dict(rig_id='main'))))
    np.array([[100., 0.], [3., 4.], [np.nan, np.nan]], '<f8').tofile(generation / 'raw_station_RS_error_native_uv.f64')
    np.array([1, 1, 0], 'u1').tofile(generation / 'raw_station_RS_valid.u8')
    files = {p.name: dict(sha256=_sha(p), size_bytes=p.stat().st_size) for p in generation.iterdir()}
    _json(generation / 'value_report.json', dict(files=files))
    _json(generation / 'formal_receipt.json', dict(source_packet_sha256=_sha(packet), state_token='complete-state'))
    context = dict(event_id='test_event', programme_id='test_programme',
                   source=dict(video_sha256='a' * 64, time_base='1/12800', coordinate_domain='native_pixels'))
    return generation, packet, context


def test_uses_native_operator_and_preserves_unknown(checkpoint):
    generation, packet, context = checkpoint
    findings, summary = collect_findings(generation, packet, context)
    assert summary['over_original_E_rows'] == 1
    assert summary['invalid_raw_RS_rows'] == 1
    assert {f['source_row_index'] for f in findings} == {1, 2}
    assert all(f['visual_resolution'] == 'unknown_until_current_native_source_review' for f in findings)
    assert findings[0]['source']['pts'] in (512, 768)


def test_exact_replay_is_idempotent_and_produces_pending_regression_jobs(checkpoint, tmp_path):
    generation, packet, context = checkpoint
    options = dict(context=context, database=tmp_path / 'feedback.sqlite', output=tmp_path / 'tasks.json')
    first = consume_generation(generation, packet, **options)
    second = consume_generation(generation, packet, **options)
    assert first['task_count'] == second['idempotent_findings'] == 2
    tasks = json.loads((tmp_path / 'tasks.json').read_text(encoding='utf8'))['tasks']
    assert all(t['regression_probe']['status'] == 'pending_current_source_review' for t in tasks)
    assert all(t['diagnostic']['is_truth_label'] is False for t in tasks)


def test_mutated_native_artifact_cannot_create_feedback(checkpoint):
    generation, packet, context = checkpoint
    (generation / 'raw_station_RS_valid.u8').write_bytes(b'\x01\x01\x01')
    with pytest.raises(ValueError, match='changed committed checkpoint artifact'):
        collect_findings(generation, packet, context)


def test_incomplete_live_checkpoint_is_reported_without_mutating_solver(checkpoint):
    generation, packet, context = checkpoint
    (generation / 'raw_station_RS_valid.u8').write_bytes(b'bad')
    observer = CheckpointObserver(generation.parent, packet, context=context)
    observer.scan(final=True)
    assert not observer.seen
    assert observer.errors[0]['feedback_not_consumed'] is True
    assert observer.errors[0]['type'] == 'ValueError'


def test_prepared_launch_preserves_config_and_stops_before_recalled_source(checkpoint, tmp_path, monkeypatch):
    from types import SimpleNamespace
    import rinkspace.storage
    from rinkfit_joint.feedback_launch import prepare, inspect, run
    generation, packet, _ = checkpoint
    monkeypatch.setattr(rinkspace.storage, 'AssetStore', lambda _: SimpleNamespace(data=tmp_path))
    root = generation.parent
    executable = root / 'joint_fit.exe'
    executable.write_bytes(b'not executed by this plumbing test')
    resume = root / 'complete_checkpoint'
    resume.write_text('complete checkpoint bytes', encoding='utf8')
    config = root / 'config.json'
    original_config = dict(max_inner_attempts_per_outer=2**64-1, initial_trust_region_radius=.03375,
                           resume_checkpoint='old', stop_file='old.stop', recapture_new_sources=False)
    _json(config, original_config)
    execution = root / 'execution.json'
    _json(execution, dict(command=[str(executable), '--formal', str(generation / 'model.json'),
                                  str(packet), str(root / 'previous-output'), str(config)]))
    _json(root / 'programme_scope.json', dict(source=dict(path='D:/events/fixture/programmes/example/video.mp4',
          sha256='a'*64, time_base='1/12800')))
    output = root / 'next-output'
    manifest = root / 'launch/candidate_launch.json'
    prepared = prepare(execution, resume, output, manifest)
    new_config = json.loads(Path(prepared['prepared_config']['path']).read_text(encoding='utf8'))
    assert new_config['max_inner_attempts_per_outer'] == 2**64-1
    assert {k:v for k,v in new_config.items() if k not in ('resume_checkpoint','stop_file')} == {
        k:v for k,v in original_config.items() if k not in ('resume_checkpoint','stop_file')}
    assert inspect(manifest)['native_executable_started'] is False
    _json(packet.parent / 'SOURCE_IDENTITY_CERTIFICATE_RECALLED.json', dict(reason='actual source recalled'))
    assert inspect(manifest)['current_source_qualification_blocks_launch'] is True
    with pytest.raises(ValueError, match='has been recalled'):
        run(manifest)
    assert not output.exists()


def test_engineering_checkpoint_complete_consumer_uses_engineering_reference(checkpoint, tmp_path):
    generation, packet, context = checkpoint
    formal = generation / "formal_receipt.json"
    receipt = json.loads(formal.read_text(encoding="utf8"))
    receipt.update(objective_id="engineering_recovery_native_pixel_v1", formal_acceptance_claimed=False)
    formal.unlink()
    engineering = generation / "engineering_checkpoint_receipt.json"
    _json(engineering, receipt)
    output = tmp_path / "engineering_tasks.json"
    result = consume_generation(generation, packet, context=context, database=tmp_path / "feedback.sqlite", output=output)
    assert result["task_count"] == 2
    task_document = json.loads(output.read_text(encoding="utf8"))
    assert task_document["generation_reference"]["sha256"] == _sha(engineering)
    assert Path(task_document["generation_reference"]["path"]) == engineering
