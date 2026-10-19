from pathlib import Path
import importlib.util
import json
import zipfile
import pytest
ROOT=Path(__file__).resolve().parents[1]

def load_script(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/f'{name}.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod

def test_rink_dimensions_not_invented():
    cfg=json.loads((ROOT/'configs/project.example.json').read_text(encoding='utf-8'))
    assert cfg['rink']['length_m'] is None and cfg['rink']['width_m'] is None
    assert cfg['timing']['allow_frame_over_average_fps_fallback'] is False

def make_report(folder):
    folder.mkdir();(folder/'index.html').write_text('<img src="frame.jpg">')
    (folder/'frame.jpg').write_bytes(b'small-test-fixture-not-a-real-jpeg')
    (folder/'review_manifest.json').write_text(json.dumps({'entrypoint':'index.html','files':['index.html','frame.jpg']}))

def test_review_zip_complete(tmp_path):
    r=tmp_path/'run';make_report(r);out=tmp_path/'review.zip'
    result=load_script('make_review_bundle').make_bundle(r,out)
    assert result['size_bytes']>0
    with zipfile.ZipFile(out) as z:assert {'index.html','frame.jpg','review_manifest.json','BUNDLE_CHECKSUMS.json'}==set(z.namelist())

def test_review_zip_missing_asset(tmp_path):
    r=tmp_path/'run';make_report(r);(r/'index.html').write_text('<img src="not-listed.jpg">')
    with pytest.raises(ValueError,match='dependency'):load_script('make_review_bundle').make_bundle(r,tmp_path/'review.zip')

def test_review_zip_path_escape(tmp_path):
    r=tmp_path/'run';make_report(r)
    (r/'review_manifest.json').write_text(json.dumps({'entrypoint':'../secret.html','files':['../secret.html']}))
    with pytest.raises(ValueError,match='Unsafe'):load_script('make_review_bundle').make_bundle(r,tmp_path/'review.zip')

def test_review_zip_size_cap(tmp_path):
    r=tmp_path/'run';make_report(r)
    with pytest.raises(ValueError,match='exceeds'):load_script('make_review_bundle').make_bundle(r,tmp_path/'review.zip',.00001)
    assert not (tmp_path/'review.zip').exists()
