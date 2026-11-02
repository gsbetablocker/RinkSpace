"""Frozen image readings can target a newer dataset without forged provenance."""
import hashlib,json
import pytest
from qualify_native_wide_lines import replacement_parent_binding

def prepared(tmp_path,name,payload,camera=1,scene='same_scene'):
    folder=tmp_path/name;folder.mkdir()
    (folder/'point_observations.npz').write_bytes(payload)
    digest=hashlib.sha256(payload).hexdigest()
    manifest=dict(point_input_sha256=digest,scene_file_sha256=scene,
        shots=[dict(id='s02',first_pts=100,last_pts=200,reference_pts=100,camera_index=camera)])
    (folder/'joint_manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
    return folder,digest

def test_preserves_capture_provenance_and_targets_new_dataset(tmp_path):
    old,old_sha=prepared(tmp_path,'old',b'old measurements')
    new,new_sha=prepared(tmp_path,'new',b'new dataset')
    target,manifest,binding=replacement_parent_binding(old,old_sha,new)
    assert target==new and manifest['point_input_sha256']==new_sha
    assert binding['source_capture_input_sha256']==old_sha
    assert binding['replacement_input_sha256']==new_sha

@pytest.mark.parametrize('kwargs',[dict(camera=2),dict(scene='other_scene')])
def test_cannot_reuse_across_different_scene_or_source_camera(tmp_path,kwargs):
    old,old_sha=prepared(tmp_path,'old',b'old')
    new,_=prepared(tmp_path,'new',b'new',**kwargs)
    with pytest.raises(ValueError):replacement_parent_binding(old,old_sha,new)
