from pathlib import Path

import pytest

from rinkspace.continuity_bundle import collect_continuity_dependencies
from rinkspace.io import dump_json, sha256


def run(root,name,**metadata):
    path=root/name;path.mkdir(parents=True)
    (path/'local_observations.npz').write_bytes(b'archived actual pairs '+name.encode())
    dump_json(path/'solver_context.json',{'frames':[]})
    dump_json(path/'localization.json',{'frames':[],**metadata})
    return path


def test_graph_closure_follows_nested_exact_inputs_and_renamed_code_snapshot(tmp_path):
    snapshot=tmp_path/'snapshots';snapshot.mkdir()
    graph=snapshot/'frozen_graph.py';graph.write_text('graph_version=2')
    merge=snapshot/'frozen_merge.py';merge.write_text('merge_version=3')
    raw=run(tmp_path,'raw')
    forward=run(tmp_path,'forward',joint_graph={'implementation_sha256':sha256(graph),'input_run':'raw',
        'input_localization_sha256':sha256(raw/'localization.json'),
        'input_observations_sha256':sha256(raw/'local_observations.npz')})
    final=run(tmp_path,'final',joint_graph={'implementation_sha256':sha256(graph),
        'merge_implementation_sha256':sha256(merge),'input_records':{'forward':{
            'path':'forward','localization_sha256':sha256(forward/'localization.json'),
            'observations_sha256':sha256(forward/'local_observations.npz')}}})
    result=collect_continuity_dependencies(tmp_path,[final],implementation_snapshots=[snapshot])
    assert result['run_count']==3 and not result['source_video_included']
    paths={row['path'] for row in result['files']}
    assert {'raw/local_observations.npz','forward/solver_context.json','snapshots/frozen_merge.py'}<=paths
    assert all(row['exact_input_sha256_checked'] for row in result['bindings'])
    (raw/'local_observations.npz').write_bytes(b'changed real pairs')
    with pytest.raises(ValueError,match='SHA mismatch'):
        collect_continuity_dependencies(tmp_path,[final],implementation_snapshots=[snapshot])


def test_reverse_closure_requires_exact_hash_resolved_raw_and_forward_dependencies(tmp_path):
    forward=run(tmp_path,'forward')
    graph=run(tmp_path,'graph')
    reverse=tmp_path/'reverse';reverse.mkdir()
    dump_json(reverse/'reverse_raw_observations.json',{'source_sha256':'source'})
    (reverse/'reverse_raw_observations.npz').write_bytes(b'reverse source pairs')
    dump_json(reverse/'camera_graph.json',{});dump_json(reverse/'merge_diagnostics.json',{})
    evidence={'forward_input_sha256':sha256(forward/'localization.json'),
              'forward_graph_input_sha256':sha256(graph/'localization.json')}
    run(tmp_path,'reverse/reverse_graph',source_sha256='source',reverse_pass_evidence=evidence)
    final=run(tmp_path,'reverse/merged',source_sha256='source',reverse_pass_evidence=evidence)
    with pytest.raises(ValueError,match='Missing reverse input'):
        collect_continuity_dependencies(tmp_path,[final],reverse_evidence=[reverse])
    result=collect_continuity_dependencies(tmp_path,[final],dependency_runs=[forward,graph],reverse_evidence=[reverse])
    assert any(row['kind']=='reverse_raw_observation_identity' for row in result['bindings'])
    assert any(row['path']=='reverse/reverse_raw_observations.npz' for row in result['files'])


def test_archive_input_cannot_leave_project(tmp_path):
    parent=tmp_path.parent/'outside_archive.json';dump_json(parent,{'frames':[]})
    with pytest.raises(ValueError,match='leaves project'):
        collect_continuity_dependencies(tmp_path,[parent])
