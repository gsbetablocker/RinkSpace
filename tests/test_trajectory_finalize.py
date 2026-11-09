"""Finalization destination/CLI regression; no real result, figure or GUI."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from calibration.program_tracking.body_tracking import trajectory_finalize as final


@pytest.fixture
def paths(tmp_path,monkeypatch):
    from rinkspace import storage
    project,data=tmp_path/'p',tmp_path/'d'
    monkeypatch.setattr(storage,'AssetStore',lambda project:SimpleNamespace(data=data))
    return project,data,data/'run'


def test_default_formal_id_and_custom_output_are_separate(paths):
    project,data,stage=paths
    old=final.finalization_paths(project,stage)
    assert old['output']==stage/'final'
    assert old['parameter'].name=='programme_trajectory_1106.npz'
    assert old['record'].name=='programme_trajectory_1106.json'
    assert old['viewer_config'].endswith('viewer3d_programme_trajectory_1106.json')
    assert old['variant_id']=='programme_trajectory'
    new=final.finalization_paths(project,stage,candidate_id='programme_source_1107',output=data/'final_v2')
    assert new['output']==data/'final_v2' and new['output']!=old['output']
    assert new['parameter'].name=='programme_source_1107.npz'
    assert new['viewer_config'].endswith('viewer3d_programme_source_1107.json')
    assert new['variant_id']=='programme_source_1107'
    assert not project.exists() and not data.exists()


@pytest.mark.parametrize('bad',['','../old','has-dash','非ASCII'])
def test_invalid_candidate_id_creates_nothing(paths,bad):
    project,data,stage=paths
    with pytest.raises(ValueError,match='candidate_id'):
        final.finalization_paths(project,stage,candidate_id=bad)
    assert not project.exists() and not data.exists()


@pytest.mark.parametrize('kind',['parameter','record'])
def test_any_formal_artifact_refuses_overwrite_before_final_output(paths,kind):
    project,data,stage=paths
    loc=final.finalization_paths(project,stage,candidate_id='new_1107')
    target=loc[kind];target.parent.mkdir(parents=True);target.write_bytes(b'preserve original')
    with pytest.raises(FileExistsError,match='formal candidate'):
        final.finalization_paths(project,stage,candidate_id='new_1107')
    assert target.read_bytes()==b'preserve original' and not loc['output'].exists()


def test_output_collision_and_project_output_refused(paths):
    project,data,stage=paths
    target=data/'failed_final';target.mkdir(parents=True)
    with pytest.raises(FileExistsError,match='new --output'):
        final.finalization_paths(project,stage,output=target)
    with pytest.raises(ValueError,match='Data root'):
        final.finalization_paths(project,stage,output=project/'final')


def test_source_clock_validation_precedes_any_output_creation(paths,monkeypatch):
    project,data,stage=paths;prepared=data/'prepared'
    scope=project/'events/ankara_2026/programmes/okada_mei_sp/rebuild_scope.json'
    scope.parent.mkdir(parents=True);scope.write_text('{}')
    pts=np.arange(8329,dtype=np.int64)*256+283136
    monkeypatch.setattr(final,'required_clock',lambda scope:(pts,np.resize(np.arange(9),len(pts)),(1,12800)))
    monkeypatch.setattr(final,'load',lambda path:dict(pts=pts+256 if Path(path)==stage/'fitted/result.npz' else pts,time_base=np.array([1,12800])))
    with pytest.raises(AssertionError):final.finalize(project,prepared,stage,candidate_id='new_1107')
    assert not (stage/'final').exists()


def test_cli_forwards_explicit_new_id_config_and_output(paths,monkeypatch,capsys):
    project,data,stage=paths;seen={}
    def fake(project,prepared,stage,**kw):seen.update(kw);return dict(no_artifacts=True)
    monkeypatch.setattr(final,'finalize',fake)
    config=project/'new_player.json'
    final.main(['--project',str(project),'--prepared',str(data/'prepared'),'--stage',str(stage),
        '--candidate-id','new_1107','--output',str(data/'final_v2'),'--viewer-config',str(config)])
    assert seen==dict(candidate_id='new_1107',output=data/'final_v2',viewer_config=config)
    assert 'no_artifacts' in capsys.readouterr().out
    assert not project.exists() and not data.exists()


@pytest.mark.parametrize('old_summary', [None, {'source_frames': 8329}])
def test_legacy_source_evidence_falls_back_to_prepared(paths, old_summary):
    project, data, stage = paths
    prepared = data/'prepared'
    prepared.mkdir(parents=True)
    evidence = prepared/'source_evidence.json'
    evidence.write_text('{"legacy": true}', encoding='utf-8')
    if old_summary is not None:
        stage.mkdir()
        (stage/'run_summary.json').write_text(json.dumps(old_summary), encoding='utf-8')
    document, identity = final._stage_source_document(project, prepared, stage)
    assert document == {'legacy': True}
    assert identity == {'path': str(evidence.resolve()), 'sha256': final.digest(evidence)}


@pytest.fixture
def model_com_stage(paths, monkeypatch, request):
    project, data, stage = paths
    prepared = data/'prepared'
    prepared.mkdir(parents=True)
    (stage/'solve').mkdir(parents=True)
    (stage/'fitted').mkdir()
    programme = project/'events/ankara_2026/programmes/okada_mei_sp'
    programme.mkdir(parents=True)
    n = 3
    pts = np.arange(n)*256+88*12800
    shots = np.full(n, 's01')
    time_base = [1, 12800]
    root = np.arange(n*3, dtype=float).reshape(n, 3)/10
    scope = dict(shots=[dict(id='s01', processing=dict(first_pts=int(pts[0]), last_pts=int(pts[-1])))])
    (programme/'rebuild_scope.json').write_text(json.dumps(scope), encoding='utf-8')
    monkeypatch.setattr(final, 'required_clock', lambda scope: (pts, shots, time_base))
    monkeypatch.setattr(final, 'boundary_receipt', lambda *a, **kw: {'source_frames': n})
    monkeypatch.setattr(final, 'ProgrammeContext',
                        SimpleNamespace(from_scope=lambda *a, **kw: SimpleNamespace(packet=lambda: {})))
    monkeypatch.setattr(final, '_figures', lambda *a: None)

    (prepared/'source_evidence.json').write_text('{"legacy": true}', encoding='utf-8')
    camera_record = project/'camera.json'
    camera_record.write_text('{"candidate": true}', encoding='utf-8')
    evidence = project/'source_revision.json'
    evidence.write_text(json.dumps(dict(
        trajectory_reference='model_com', programme_opening=dict(enabled=True),
        com_motion_policy=dict(acceleration_scale_mps2=12.),
        programme_camera_candidate=dict(record=str(camera_record), record_sha256=final.digest(camera_record)),
    )), encoding='utf-8')
    source_hash = final.digest(evidence)
    (stage/'run_summary.json').write_text(json.dumps(dict(
        source_evidence=evidence.name, source_evidence_sha256=source_hash,
    )), encoding='utf-8')

    class Camera:
        def __init__(self, project, packet, shots, *, record_path):
            assert Path(record_path) == camera_record
            self.info = dict(state_sha256='candidate-camera', shots={'s01': dict(state_sha256='shot-camera')})

        def project(self, world, indices):
            return world[..., :2]

    monkeypatch.setattr(final, 'ProgrammeCamera', Camera)
    binding = np.array(json.dumps(dict(state_sha256='candidate-camera')))
    kind = getattr(request, 'param', 'absolute_motion_v1')
    times = (pts-pts[0])/time_base[1]
    knots = np.repeat(times[[0, -1]], 4)
    controls = {}
    if kind == 'sequence_skates_v1':
        controls = dict(
            state_schema_version=np.array('rinkspace.absolute_motion_v1'),
            absolute_position_controls=np.zeros((4, 3)),
            absolute_rotation_controls=np.broadcast_to(np.eye(3), (4, 22, 3, 3)),
            relative_skate_schema=np.array('rinkspace.relative_skate_so3_v1'),
            relative_skate_knots=knots,
            relative_skate_rotation_controls=np.broadcast_to(np.eye(3), (4, 2, 3, 3)),
        )
    np.savez(stage/'solve/candidate.npz', state_kind=np.array(kind), absolute_knots=knots,
             camera_binding_json=binding, source_evidence_sha256=np.array(source_hash), **controls)
    np.savez(prepared/'input.npz', pts=pts, fit_time_seconds=times,
             shared_betas=np.zeros(10), body_scale=np.array(1.))
    np.savez(prepared/'observations.npz', uv=np.zeros((n, 17, 2)), use=np.ones((n, 17), dtype=bool))
    (prepared/'fixed_binding.npz').write_bytes(b'fixed binding identity')
    old = data/'old.npz'
    q = np.broadcast_to(np.eye(3), (n, 3, 3))
    np.savez(old, pts=pts[:1], P_world_mid=root[:1], Q_world_mid=q[:1])
    (prepared/'preparation.json').write_text(json.dumps(dict(
        old_S01_result=str(old), old_S01_result_sha256=final.digest(old),
    )), encoding='utf-8')
    momentum = root*.4
    momentum[[0, -1]] = np.nan
    diagnostics = dict(
        trajectory_reference=np.array('model_com'),
        com_world_mid=root+.1,
        com_velocity_world_mps=root+2.,
        com_acceleration_world_mps2=root-3.,
        com_specific_angular_momentum_world_m2ps=momentum,
        com_required_specific_external_force_world_mps2=root-3.+[0, 0, 9.81],
        com_derivative_interior=np.array([False, True, False]),
        com_mass_model=np.array('body volume proxy and two shoes'),
        com_derivative_method=np.array('native source-clock differences'),
        com_segment_mass_fractions=np.array([.96, .02, .02]),
    )
    raw = stage/'fitted/result.npz'
    np.savez(raw, **diagnostics,
             pts=pts, time_base=time_base, camera_binding_json=binding,
             source_action_ledger_sha256=np.array(source_hash), metres_per_scene_unit=np.array(1.),
             P_world_mid=root, P_velocity_world_mps=np.zeros_like(root),
             G_world_mid=root*[1, 1, 0], Q_world_mid=q, native22_world_mid=np.zeros((n, 22, 3)),
             source_identity_suspect=np.zeros(n, dtype=bool), skate_clearance_m_mid=np.zeros((n, 2)),
             skate_support_gap_m_mid=np.zeros((n, 2)), skate_edge_world_mid=np.zeros((n, 2, 1, 1, 3)),
             skate_reference_world_mid=np.zeros((n, 2, 3)), body_length_ratio=np.array(1.),
             world_at_row=np.zeros((n, 113, 3)), COCO_uv=np.zeros((n, 17, 2)))
    return SimpleNamespace(project=project, prepared=prepared, stage=stage, evidence=evidence,
                           source_hash=source_hash, diagnostics=diagnostics, raw=raw)


@pytest.mark.parametrize('model_com_stage', ['absolute_motion_v1', 'sequence_skates_v1'], indirect=True)
def test_finalization_uses_run_source_and_preserves_all_com_fields(model_com_stage):
    m = model_com_stage
    delivered = final.finalize(m.project, m.prepared, m.stage, candidate_id='model_com_test')
    with np.load(delivered['result'], allow_pickle=False) as result:
        for name, expected in m.diagnostics.items():
            np.testing.assert_array_equal(result[name], expected)
        embedded = json.loads(str(result['programme_context_json']))
    expected = dict(path=str(m.evidence.resolve()), sha256=m.source_hash)
    assert embedded['source_evidence'] == expected
    standalone = json.loads((m.stage/'final/programme_context.json').read_text(encoding='utf-8'))
    formal = json.loads(Path(delivered['parameter']).with_suffix('.json').read_text(encoding='utf-8'))
    assert standalone['source_evidence'] == formal['programme_context']['source_evidence'] == expected


def test_changed_run_source_refuses_finalization_before_creating_output(model_com_stage):
    m = model_com_stage
    m.evidence.write_text('{"changed": true}', encoding='utf-8')
    with pytest.raises(ValueError, match='source evidence missing or changed'):
        final.finalize(m.project, m.prepared, m.stage, candidate_id='model_com_test')
    assert not (m.stage/'final').exists()


def test_missing_com_field_refuses_finalization_before_creating_output(model_com_stage):
    m = model_com_stage
    raw = final.load(m.raw)
    del raw['com_required_specific_external_force_world_mps2']
    np.savez(m.raw, **raw)
    with pytest.raises(ValueError, match='Incomplete model COM export.*com_required_specific_external_force'):
        final.finalize(m.project, m.prepared, m.stage, candidate_id='model_com_test')
    assert not (m.stage/'final').exists()
