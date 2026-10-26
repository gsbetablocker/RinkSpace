"""A legacy delta receipt must not conceal an incomplete metadata write."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('audit_export_original_ad',
    ROOT / 'calibration/solve2/export_original_ad.py')
exporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exporter)


def json_file(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf8')


def auxiliary_files(root):
    for name in ('station.f64', 'path.u32', 'initial_row.f64', 'initial_alpha.f64'):
        (root/name).write_bytes(b'unchanged auxiliary bytes')
    json_file(root/'semantic/manifest.json', {'schema':'fixture'})
    json_file(root/'initializer_manifest.json', {'schema':'fixture'})
    json_file(root/'mapping.json', {'reset_similarity':[
        {'curve_id':'d78_material_012/path0', 'reset_world_vs_d78_exact_max':0.0}],
        'current_static_source_world_delta_max':0.0})


def geometry_input(tmp_path):
    root = tmp_path/'export'
    root.mkdir()
    model = json.loads((ROOT/'events/ankara_2026/programmes/okada_mei_sp/s01_original/original_model.json').read_text())
    for block in model['blocks']:
        block['constant'] = True
    material = next(deepcopy(m) for m in model['materials']
                    if any(p['id'] == 'w139/path0' for p in m['paths']))
    path = material['paths'][0]
    path['id'] = 'd78_material_012/path0'
    block = model['blocks'][path['geometry_block']]
    block['values'] = [1.,0.,0.,0.,1.,0.]
    block['constant'] = False
    model['materials'] = [material]
    ad = np.asarray(model['blocks'][path['advertisement_block']]['values'])
    points = np.asarray(path['points'])[:10] * [ad[1],ad[3]-ad[2]] + ad[[0,2]]
    points[:,0] += .0001
    json_file(root/'model.json', model)
    json_file(root/'packet/manifest.json', {'arrays':{},
        'event_model':{'sha256':exporter.sha(root/'model.json')}})
    auxiliary_files(root)
    evidence = tmp_path/'reviewed_geometry.json'
    json_file(evidence, {'current_w139_transformed_points':points.tolist(),
        'current_w139_is_same_10_vertex_I_entity':True, 'current_w139_closed':True})
    return root,evidence


def assert_completion(root, key, receipt):
    mapping = json.loads((root/'mapping.json').read_text())
    assert mapping[key]['sha256'] == exporter.sha(root/receipt)
    initializer = json.loads((root/'initializer_manifest.json').read_text())
    assert initializer['semantic_manifest']['sha256'] == exporter.sha(root/'semantic/manifest.json')
    for name, desc in initializer['files'].items():
        assert desc['sha256'] == exporter.sha(root/name)


def test_reviewed_geometry_finishes_current_initializer_and_mapping(tmp_path):
    root,evidence = geometry_input(tmp_path)
    original_sources = {p.name:p.read_bytes() for p in root.iterdir() if p.suffix in ('.f64','.u32')}
    exporter.apply_I_geometry_delta(root,evidence)
    assert_completion(root,'specific_I_geometry_delta','geometry_delta_012.json')
    for name,data in original_sources.items():
        assert (root/name).read_bytes() == data


def test_geometry_retry_finishes_metadata_after_write_failure(tmp_path,monkeypatch):
    root,evidence = geometry_input(tmp_path)
    save = exporter.save_json
    def fail_initializer(path,value):
        if Path(path).name == 'initializer_manifest.json':
            raise OSError('simulated interrupted metadata write')
        return save(path,value)
    monkeypatch.setattr(exporter,'save_json',fail_initializer)
    with pytest.raises(OSError,match='interrupted'):
        exporter.apply_I_geometry_delta(root,evidence)
    assert (root/'geometry_delta_012.json').is_file()
    committed = (root/'model.json').read_bytes()
    history = (root/'geometry_delta_previous_model.json').read_bytes()
    monkeypatch.setattr(exporter,'save_json',save)
    exporter.apply_I_geometry_delta(root,evidence)
    assert_completion(root,'specific_I_geometry_delta','geometry_delta_012.json')
    assert (root/'model.json').read_bytes() == committed
    assert (root/'geometry_delta_previous_model.json').read_bytes() == history


def test_missing_mapping_fails_before_changing_geometry(tmp_path):
    root,evidence = geometry_input(tmp_path)
    (root/'mapping.json').unlink()
    original = (root/'model.json').read_bytes()
    with pytest.raises(FileNotFoundError):
        exporter.apply_I_geometry_delta(root,evidence)
    assert (root/'model.json').read_bytes() == original
    assert not (root/'geometry_delta_012.json').exists()


@pytest.mark.parametrize('tampered', [False,True])
def test_binding_retry_repairs_metadata_and_authenticates_accepted_bytes(tmp_path,tampered):
    root = tmp_path/'export'
    root.mkdir()
    auxiliary_files(root)
    json_file(root/'model.json',{})
    json_file(root/'packet/manifest.json',{})
    evidence = tmp_path/'reviewed_binding.json'
    json_file(evidence,{'records':[]})
    result = {'reviewed_delta':{'sha256':exporter.sha(evidence)},'records':[],
        'current_packet_sha256':exporter.sha(root/'packet/manifest.json'),
        'current_semantic_sha256':exporter.sha(root/'semantic/manifest.json'),
        'current_station_sha256':exporter.sha(root/'station.f64'),
        'unchanged_array_sha256':{str(root/'initial_row.f64'):exporter.sha(root/'initial_row.f64')}}
    json_file(root/'binding_delta_01.json',result)
    if tampered:
        (root/'station.f64').write_bytes(b'changed after receipt')
        with pytest.raises(ValueError,match='differs from its recorded receipt'):
            exporter.apply_material_delta(root,evidence)
    else:
        exporter.apply_material_delta(root,evidence)
        assert_completion(root,'specific_binding_delta','binding_delta_01.json')


@pytest.mark.parametrize('unrelated_edit', [False,True])
def test_binding_retry_allows_only_the_verified_later_geometry_delta(tmp_path, unrelated_edit):
    root,geometry = geometry_input(tmp_path)
    binding = tmp_path/'reviewed_binding.json'
    json_file(binding, {'records':[]})
    result = {'reviewed_delta':{'sha256':exporter.sha(binding)}, 'records':[],
        'current_packet_sha256':exporter.sha(root/'packet/manifest.json'),
        'current_semantic_sha256':exporter.sha(root/'semantic/manifest.json'),
        'current_station_sha256':exporter.sha(root/'station.f64'),
        'unchanged_array_sha256':{str(root/'initial_row.f64'):exporter.sha(root/'initial_row.f64')}}
    json_file(root/'binding_delta_01.json', result)
    exporter.apply_material_delta(root,binding)
    exporter.apply_I_geometry_delta(root,geometry)
    if unrelated_edit:
        packet = json.loads((root/'packet/manifest.json').read_text())
        packet['unreviewed_change'] = True
        json_file(root/'packet/manifest.json',packet)
        with pytest.raises(ValueError,match='differs from its recorded receipt'):
            exporter.apply_material_delta(root,binding)
    else:
        exporter.apply_material_delta(root,binding)
        assert_completion(root,'specific_binding_delta','binding_delta_01.json')
        assert_completion(root,'specific_I_geometry_delta','geometry_delta_012.json')


def test_geometry_retry_cannot_authenticate_changed_auxiliary_bytes(tmp_path, monkeypatch):
    root,evidence = geometry_input(tmp_path)
    save = exporter.save_json
    def fail_initializer(path,value):
        if Path(path).name == 'initializer_manifest.json':
            raise OSError('simulated interrupted metadata write')
        return save(path,value)
    monkeypatch.setattr(exporter,'save_json',fail_initializer)
    with pytest.raises(OSError,match='interrupted'):
        exporter.apply_I_geometry_delta(root,evidence)
    (root/'station.f64').write_bytes(b'other station values!!!')
    monkeypatch.setattr(exporter,'save_json',save)
    with pytest.raises(ValueError,match='unchanged inputs differ'):
        exporter.apply_I_geometry_delta(root,evidence)
    assert 'files' not in json.loads((root/'initializer_manifest.json').read_text())


def test_geometry_retry_allows_a_verified_later_binding_without_reverting_it(tmp_path):
    root,evidence = geometry_input(tmp_path)
    exporter.apply_I_geometry_delta(root,evidence)
    old_packet = exporter.sha(root/'packet/manifest.json')
    old_semantic = exporter.sha(root/'semantic/manifest.json')
    old_station = exporter.sha(root/'station.f64')
    # Reproduce the committed outputs of a later reviewed binding, including
    # its before/after receipt, without invoking the unrelated native seed grid.
    untouched = {str(root/name):exporter.sha(root/name)
                 for name in ('path.u32','initial_row.f64','initial_alpha.f64')}
    (root/'station.f64').write_bytes(b'new reviewed station')
    json_file(root/'semantic/manifest.json', {'reviewed_binding':'different path'})
    packet = json.loads((root/'packet/manifest.json').read_text())
    packet['semantic_packet'] = {'sha256':exporter.sha(root/'semantic/manifest.json')}
    exporter.save_json(root/'packet/manifest.json',packet)
    result = {'previous_input':{'packet_sha256':old_packet,'semantic_sha256':old_semantic,'station_sha256':old_station},
        'current_packet_sha256':exporter.sha(root/'packet/manifest.json'),
        'current_semantic_sha256':exporter.sha(root/'semantic/manifest.json'),
        'current_station_sha256':exporter.sha(root/'station.f64'),
        'unchanged_array_sha256':untouched}
    json_file(root/'binding_delta_01.json',result)
    exporter.apply_I_geometry_delta(root,evidence)
    assert (root/'station.f64').read_bytes() == b'new reviewed station'
    assert_completion(root,'specific_I_geometry_delta','geometry_delta_012.json')
