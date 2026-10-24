"""Portable public reference metadata never requires withheld source images."""
import json
from pathlib import Path

from PIL import Image
import pytest

from rinkspace.session import field_preview


class TextlessCanvas:
    """Keep source-image selection tests independent of machine font installs."""
    def __init__(self,size):
        self.image=Image.new('RGB',size,'white')

    def text(self,*args,**kwargs):
        pass


def reference_record(path):
    return {'path':str(path),'source_id':'official_source','pts':744704,
            'time_base_num':1,'time_base_den':12800,'seconds':58.18}


def scene_files(tmp_path):
    root=tmp_path/'relocated_checkout'/'events'/'event'/'scene'
    field=root/'field';field.mkdir(parents=True)
    scene_path=field/'model.json';scene_path.write_text('{}',encoding='utf8')
    out=tmp_path/'preview';out.mkdir()
    return root,scene_path,out


def test_relative_references_follow_scene_and_metadata_not_cwd(tmp_path,monkeypatch):
    root,scene_path,out=scene_files(tmp_path)
    frames=root/'frames';frames.mkdir()
    Image.new('RGB',(32,24),(11,72,133)).save(frames/'frame.png')
    (root/'end_sources.json').write_text(json.dumps({
        'official50_references':[reference_record('frames/frame.png')]}),encoding='utf8')
    unrelated=tmp_path/'other_checkout';unrelated.mkdir()
    (unrelated/'end_sources.json').write_text('invalid JSON in the wrong directory',encoding='utf8')
    monkeypatch.chdir(unrelated)
    monkeypatch.setattr(field_preview,'Canvas',TextlessCanvas)
    result=field_preview.source_reference(
        {'endpoint_source_references':{'path':'../end_sources.json'}},out,scene_path=scene_path)
    assert Path(result)==out/'source_reference.png'
    with Image.open(result) as rendered:
        assert rendered.getpixel((984,160))==(11,72,133)


def test_absolute_existing_local_index_and_image_remain_supported(tmp_path,monkeypatch):
    root,scene_path,out=scene_files(tmp_path)
    image=tmp_path/'private_source.png';Image.new('RGB',(32,24),(93,28,17)).save(image)
    index=root/'end_sources.json';index.write_text(json.dumps({
        'official50_references':[reference_record(image)]}),encoding='utf8')
    monkeypatch.setattr(field_preview,'Canvas',TextlessCanvas)
    result=field_preview.source_reference(
        {'endpoint_source_references':{'path':str(index)}},out,scene_path=scene_path)
    with Image.open(result) as rendered:
        assert rendered.getpixel((984,160))==(93,28,17)


def test_public_metadata_without_private_image_returns_no_reference(tmp_path):
    root,scene_path,out=scene_files(tmp_path)
    (root/'end_sources.json').write_text(json.dumps({
        'official50_references':[reference_record(tmp_path/'withheld'/'native.png')]}),encoding='utf8')
    result=field_preview.source_reference(
        {'endpoint_source_references':{'path':'../end_sources.json'}},out,scene_path=scene_path)
    assert result is None
    assert not (out/'source_reference.png').exists()


def test_corrupt_reference_json_is_not_silently_ignored(tmp_path):
    root,scene_path,out=scene_files(tmp_path)
    (root/'end_sources.json').write_text('{invalid',encoding='utf8')
    with pytest.raises(json.JSONDecodeError):
        field_preview.source_reference(
            {'endpoint_source_references':{'path':'../end_sources.json'}},out,scene_path=scene_path)


def test_invalid_reference_schema_is_not_treated_as_withheld_media(tmp_path):
    root,scene_path,out=scene_files(tmp_path)
    (root/'end_sources.json').write_text('{}',encoding='utf8')
    with pytest.raises(KeyError,match='official50_references'):
        field_preview.source_reference(
            {'endpoint_source_references':{'path':'../end_sources.json'}},out,scene_path=scene_path)
