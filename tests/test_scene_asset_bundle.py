"""Current scene snapshots must retain runnable numeric advertisements."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil

import pytest

from rinkspace.domain.scene_assets import (
    read_scene_assets, scene_content_identity, snapshot_scene,
)


ROOT = Path(__file__).resolve().parents[1]
FIELD = ROOT/'events/ankara_2026/scene/field/model.json'


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


@pytest.fixture
def local_scene(tmp_path):
    scene=json.loads(FIELD.read_text(encoding='utf-8'))
    # Resolve current numerical dependencies through the actual public scene.
    assets=read_scene_assets(scene,FIELD)
    source=tmp_path/'source'/'field'/'scene.json'
    for index,(ref,_) in enumerate(assets):
        raw=(FIELD.parent/ref['path']).read_bytes()
        target=source.parent.parent/f'ad{index}.json'
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(raw)
        scene['advertisement_assets'][index]['path']=f'../ad{index}.json'
    write(source,scene)
    return source,scene,assets


def test_current_field_resolves_all_registered_numeric_assets():
    scene=json.loads(FIELD.read_text(encoding='utf-8'))
    loaded=read_scene_assets(scene,FIELD)
    registered=scene['advertisement_assets']
    assert len(loaded)==len(registered)
    assert [ref for ref,_ in loaded]==registered
    assert {ref['template_id'] for ref,_ in loaded}=={ref['template_id'] for ref in registered}
    assert {i for ref,_ in loaded for i in ref['instance_ids']}=={
        i for ref in registered for i in ref['instance_ids']}
    for ref,assets in loaded:
        expected=json.loads((FIELD.parent/ref['path']).read_text(encoding='utf-8'))
        assert assets==expected
    baseline=[(ref,assets) for ref,assets in loaded if ref['advertisement_number'] in {1,2,3}]
    assert len(baseline)==3
    assert {ref['advertisement_number'] for ref,_ in baseline}=={1,2,3}
    assert {i for ref,_ in baseline for i in ref['instance_ids']}=={
        'print_instance/Ad1','print_instance/Ad14','print_instance/Ad2',
        'print_instance/Ad15','print_instance/Ad3','print_instance/Ad16'}


def test_snapshot_runs_without_original_scene_directory_and_preserves_every_asset_byte(local_scene,tmp_path):
    source,scene,original_assets=local_scene
    before=scene_content_identity(scene,source)
    original_bytes=[(source.parent/ref['path']).read_bytes() for ref,_ in original_assets]
    out=tmp_path/'prepared'
    snapshot,summary=snapshot_scene(source,out)
    restored=deepcopy(snapshot)
    for old,new in zip(scene['advertisement_assets'],restored['advertisement_assets']):
        new['path']=old['path']
    assert restored==scene  # No UV, parameters, claims or provenance rewritten.
    shutil.rmtree(source.parent.parent)
    assert not source.exists()
    assert scene_content_identity(snapshot,out/'scene.json')==before
    loaded=read_scene_assets(snapshot,out/'scene.json')
    registered=scene['advertisement_assets']
    assert len(loaded)==len(original_assets)==len(summary)==len(registered)
    assert {ref['template_id'] for ref,_ in loaded}=={ref['template_id'] for ref in registered}
    assert {i for ref,_ in loaded for i in ref['instance_ids']}=={
        i for ref in registered for i in ref['instance_ids']}
    for index,((ref,assets),(_,original)) in enumerate(zip(loaded,original_assets)):
        assert assets==original
        assert (out/ref['path']).read_bytes()==original_bytes[index]
        assert hashlib.sha256(original_bytes[index]).hexdigest()==summary[index]['sha256']
        assert assets['templates']==original['templates']
        assert assets['instances']==original['instances']
        assert assets['parameter_blocks']==original['parameter_blocks']


def test_invalid_sha_fails_before_snapshot_output_is_created(local_scene,tmp_path):
    source,scene,_=local_scene
    scene['advertisement_assets'][-1]['sha256']='0'*64
    write(source,scene)
    out=tmp_path/'prepared'
    with pytest.raises(ValueError,match='SHA256 mismatch'):
        snapshot_scene(source,out)
    assert not out.exists()


@pytest.mark.parametrize('defect',['duplicate_instance','absent_instance','wrong_template'])
def test_declared_instance_and_template_bindings_are_checked(local_scene,defect):
    source,scene,_=local_scene
    if defect=='duplicate_instance':
        scene['advertisement_assets'].append(deepcopy(scene['advertisement_assets'][0]))
        message='Duplicate scene advertisement instance_ids'
    elif defect=='absent_instance':
        scene['advertisement_assets'][0]['instance_ids'].append('print_instance/absent')
        message='Declared instance'
    else:
        scene['advertisement_assets'][0]['template_id']='print_template/other'
        message='template_id differ'
    with pytest.raises(ValueError,match=message):
        read_scene_assets(scene,source)


def test_scene_identity_ignores_only_asset_location(local_scene,tmp_path):
    source,scene,_=local_scene
    original=scene_content_identity(scene,source)
    snapshot,_=snapshot_scene(source,tmp_path/'prepared')
    assert scene_content_identity(snapshot,tmp_path/'prepared'/'scene.json')==original
    for field in ('shared_layout','geometry_review','source_model'):
        changed=deepcopy(scene)
        if field=='shared_layout':
            changed[field][0]+=.01
        elif field=='geometry_review':
            changed[field]['status']='different_declaration'
        else:
            changed[field]='different_provenance_path'
        assert scene_content_identity(changed,source)!=original
    first=scene['advertisement_assets'][0]
    asset_path=source.parent/first['path']
    asset=json.loads(asset_path.read_text(encoding='utf-8'))
    asset['fit_executed']=not asset['fit_executed']
    write(asset_path,asset)
    with pytest.raises(ValueError,match='SHA256 mismatch'):
        scene_content_identity(scene,source)
    first['sha256']=hashlib.sha256(asset_path.read_bytes()).hexdigest()
    assert scene_content_identity(scene,source)!=original


def test_snapshot_never_overwrites_existing_output(local_scene,tmp_path):
    source,_,_=local_scene
    out=tmp_path/'existing';out.mkdir()
    (out/'keep.txt').write_text('untouched',encoding='utf-8')
    with pytest.raises(FileExistsError):
        snapshot_scene(source,out)
    assert (out/'keep.txt').read_text(encoding='utf-8')=='untouched'


def test_scene_without_advertisements_is_valid(tmp_path):
    source=tmp_path/'input.json'
    scene={'schema':'example.scene','paths':[]}
    write(source,scene)
    snapshot,summary=snapshot_scene(source,tmp_path/'output')
    assert snapshot==scene and summary==[]
    assert read_scene_assets(snapshot,tmp_path/'output'/'scene.json')==[]
    assert scene_content_identity(snapshot,tmp_path/'output'/'scene.json')==scene_content_identity(scene,source)


def test_scene_archive_maps_only_retired_files_and_preserves_identity_catalog():
    from pathlib import PurePosixPath
    catalog = json.loads((ROOT/'events/ankara_2026/scene/archive.json').read_text(encoding='utf-8'))
    shared = json.loads((ROOT/'configs/storage.defaults.json').read_text(encoding='utf-8'))
    assert catalog['schema'] == 'rinkspace.scene-archive.v1'
    scene = json.loads(FIELD.read_text(encoding='utf-8'))
    current = {FIELD.resolve()}
    for ref in scene['advertisement_assets']:
        current.add((FIELD.parent/ref['path']).resolve())
        if 'source_index_ref' in ref:
            current.add((FIELD.parent/ref['source_index_ref']).resolve())
    seen, content_ids = set(), {}
    for entry in catalog['entries']:
        old, target = PurePosixPath(entry['old_path']), PurePosixPath(entry['data_path'])
        assert not old.is_absolute() and not target.is_absolute()
        assert '..' not in old.parts and '..' not in target.parts
        assert entry['old_path'] not in seen
        seen.add(entry['old_path'])
        assert (ROOT/entry['old_path']).resolve() not in current
        assert not (ROOT/entry['old_path']).exists()
        assert shared['legacy_paths'][entry['old_path']] == entry['data_path']
        assert len(entry['sha256']) == 64 and all(c in '0123456789abcdef' for c in entry['sha256'])
        assert entry['bytes'] > 0
        identity = (entry['sha256'], entry['bytes'])
        assert content_ids.setdefault(entry['data_path'], identity) == identity
    assert seen
    assert all(path.is_file() for path in current)
