"""Finalization destination/CLI regression; no real result, figure or GUI."""
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
