"""Prepared revisions carry their current dependencies and bind their content."""
import json
from pathlib import Path
import shutil

import pytest

from rinkspace.domain.scene_assets import snapshot_scene, read_scene_assets, scene_content_identity
from rinkspace.session.dataset import read_json, write_json
from rinkspace.session.prepare import prepare
import test_prepare_source_binding as source_binding

ROOT = Path(__file__).resolve().parents[1]
FIELD = ROOT/'events/ankara_2026/scene/field/model.json'


def test_prepare_remains_self_contained_after_input_scene_is_removed(tmp_path):
    registered = read_json(FIELD)['advertisement_assets']
    expected_ids = {identity for ref in registered for identity in ref['instance_ids']}
    root = tmp_path/'input'
    source_binding.PrepareSourceBinding().make_input(root, [dict()])
    scene, _ = snapshot_scene(FIELD, root/'field')
    input_doc = read_json(root/'input.json')
    input_doc['scene_ref'] = 'field/scene.json'
    write_json(root/'input.json', input_doc)
    receipt = prepare(root/'input.json', root/'scope.json', 's01', tmp_path/'prepared')
    expected_identity = scene_content_identity(scene, root/'field/scene.json')
    shutil.rmtree(root)
    manifest = read_json(receipt.manifest_path)
    actual = read_json(tmp_path/'prepared/scene.json')
    loaded = read_scene_assets(actual, tmp_path/'prepared/scene.json')
    assert len(loaded) == len(registered)
    assert {ref['template_id'] for ref, _ in loaded} == {ref['template_id'] for ref in registered}
    assert {identity for ref, _ in loaded for identity in ref['instance_ids']} == expected_ids
    assert scene_content_identity(actual, tmp_path/'prepared/scene.json') == expected_identity
    assert manifest['dependency_revision']['scene'] == expected_identity
    dependencies = manifest['snapshot_dependencies']['scene']
    assert len(dependencies) == len(registered)
    assert {entry['sha256'] for entry in dependencies} == {ref['sha256'] for ref in registered}


@pytest.mark.parametrize('changed', ['scene', 'features', 'rigs', 'metadata', 'model_configuration'])
def test_same_input_label_does_not_hide_changed_current_dependency(tmp_path, changed):
    root = tmp_path/'input'
    source_binding.PrepareSourceBinding().make_input(root, [dict()])
    if changed == 'metadata':
        obs = read_json(root/'obs.json')
        obs['metadata_ref'] = 'observations.jsonl'
        write_json(root/'obs.json', obs)
        # A row sidecar is JSONL, not the indented document format.
        (root/'observations.jsonl').write_text(json.dumps({'feature_id':'ice.C',
            'measurement':'centreline','note':'before'})+'\n', encoding='utf-8')
    first = prepare(root/'input.json', root/'scope.json', 's01', tmp_path/'first')
    unchanged = prepare(root/'input.json', root/'scope.json', 's01', tmp_path/'unchanged')
    assert unchanged.revision == first.revision
    if changed == 'metadata':
        text = (root/'observations.jsonl').read_text(encoding='utf-8').replace('before', 'after')
        (root/'observations.jsonl').write_text(text, encoding='utf-8')
    elif changed == 'model_configuration':
        doc = read_json(root/'input.json')
        doc['model_configuration'] = {'active_parameters': ['camera']}
        write_json(root/'input.json', doc)
    else:
        path = root/(changed+'.json')
        doc = read_json(path)
        doc['current_test_value'] = 42
        write_json(path, doc)
    second = prepare(root/'input.json', root/'scope.json', 's01', tmp_path/'second')
    assert second.revision != first.revision
    assert read_json(root/'input.json')['revision'] == 'r0'
