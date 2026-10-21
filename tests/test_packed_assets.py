from __future__ import annotations
import hashlib
import json
from pathlib import Path
import zipfile
import pytest
from rinkspace.packed_assets import read_project_bytes, read_project_json, materialize_project_asset, iter_project_assets
from rinkspace.storage import AssetStore


def setup_bundle(tmp_path):
    root=tmp_path/'project';data=tmp_path/'data';(root/'configs').mkdir(parents=True);data.mkdir()
    value=b'{"source_point":[12.5,7.25],"native_pts":256256}\r\n'
    digest=hashlib.sha256(value).hexdigest();logical=root/'events/test/frames/1.json'
    with zipfile.ZipFile(root/'source.zip','w',zipfile.ZIP_DEFLATED) as z:z.writestr('files/original.json',value)
    (root/'source.json').write_text(json.dumps(dict(schema='rinkspace.packed-assets.v1',archive='source.zip',assets=[dict(path='events/test/frames/1.json',member='files/original.json',bytes=len(value),sha256=digest)])))
    (root/'configs/packed_assets.json').write_text(json.dumps(dict(schema='rinkspace.packed-assets-registry.v1',indexes=['source.json'])))
    (root/'configs/storage.json').write_text(json.dumps(dict(schema_version='rinkspace.storage.v1',data_root=str(data),event_root='events/test')))
    return root,data,logical,value


def test_original_bytes_clock_and_logical_enumeration_without_expansion(tmp_path):
    root,data,logical,value=setup_bundle(tmp_path)
    assert not logical.exists()
    assert read_project_bytes(logical,project_root=root)==value
    assert read_project_json(logical,project_root=root)['native_pts']==256256
    assert iter_project_assets(logical.parent,'*.json',project_root=root)==[logical]
    assert not logical.exists() and list(data.iterdir())==[]


def test_path_consumer_materializes_only_in_data_and_rejects_corruption(tmp_path):
    root,data,logical,value=setup_bundle(tmp_path)
    actual=AssetStore(root).resolve(logical,must_exist=True)
    assert actual.is_relative_to(data) and actual.read_bytes()==value and not logical.exists()
    actual.write_bytes(b'changed')
    with pytest.raises(ValueError,match='identity changed'):
        materialize_project_asset(logical,project_root=root)


def test_frozen_bundle_identity_is_enforced(tmp_path):
    root,data,logical,value=setup_bundle(tmp_path)
    with zipfile.ZipFile(root/'source.zip','w') as z:z.writestr('files/original.json',value.replace(b'12.5',b'99.5'))
    with pytest.raises(ValueError,match='identity changed'):
        read_project_bytes(logical,project_root=root)


def test_explicit_new_edit_overrides_without_modifying_frozen_bundle(tmp_path):
    root,data,logical,value=setup_bundle(tmp_path)
    logical.parent.mkdir(parents=True);logical.write_bytes(b'{"native_pts":256512}')
    assert read_project_json(logical,project_root=root)['native_pts']==256512
    logical.unlink()
    assert read_project_bytes(logical,project_root=root)==value


def test_registry_cannot_escape_project(tmp_path):
    root,data,logical,value=setup_bundle(tmp_path)
    (root/'configs/packed_assets.json').write_text(json.dumps(dict(schema='rinkspace.packed-assets-registry.v1',indexes=['../private.json'])))
    with pytest.raises(ValueError,match='stay inside'):
        read_project_bytes(logical,project_root=root)


def test_normal_io_keeps_cwd_semantics(tmp_path,monkeypatch):
    from rinkspace.io import load_json,sha256
    monkeypatch.chdir(tmp_path)
    value=b'{"ordinary":true}'
    Path('local.json').write_bytes(value)
    assert load_json('local.json')=={'ordinary':True}
    assert sha256('local.json')==hashlib.sha256(value).hexdigest()


def test_direct_materialization_rejects_relative_data_root(tmp_path):
    root,data,logical,value=setup_bundle(tmp_path)
    (root/'configs/storage.json').write_text(json.dumps(dict(schema_version='rinkspace.storage.v1',data_root='relative_data')))
    with pytest.raises(ValueError,match='absolute Data root'):
        materialize_project_asset(logical,project_root=root)
