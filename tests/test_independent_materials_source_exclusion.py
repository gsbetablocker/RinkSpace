"""Source-bound unresolved foreground must not silently become background."""
from pathlib import Path
import hashlib
import json
import sys

import cv2
import numpy as np
import pytest

PROGRAMME=Path(__file__).resolve().parents[1]/'events/ankara_2026/programmes/okada_mei_sp'
if str(PROGRAMME) not in sys.path:sys.path.insert(0,str(PROGRAMME))
import capture_independent_materials as materials


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def put_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value)+'\n',encoding='utf8')


def put_mask(path,mask):
    path.parent.mkdir(parents=True,exist_ok=True)
    assert cv2.imwrite(str(path),mask)


def original_exclusion(mask):
    if not mask.any():return np.ones((1080,1920),bool)
    result=cv2.dilate((mask>0).astype('uint8'),np.ones((7,7),'uint8')).astype(bool)
    result[:290,:570]=True;result[940:,1580:]=True
    return result


@pytest.fixture
def source(tmp_path):
    materials._SOURCE_MASK_DOCUMENTS.clear()
    root=tmp_path/'bootstrap';sid='s07';pts=12345
    base=np.zeros((1080,1920),'uint8');base[500:700,900:1000]=255
    refined=np.zeros_like(base);refined[500:550,900:1000]=255
    base_path=root/'person_masks'/sid/f'{pts}.png'
    refined_path=root/'person_masks_refined/full'/sid/f'{pts}.png'
    base_ledger=root/'person_masks/frames.jsonl'
    refined_ledger=root/'person_masks_refined/full/frames.jsonl'
    binding=dict(source_sha256='1'*64,source_pts=pts,time_base='1/12800',shot_id=sid,
        native_hw=[1080,1920],image_domain='native_brown_pixels')
    records=[]
    for path,mask,ledger in ((base_path,base,base_ledger),(refined_path,refined,refined_ledger)):
        put_mask(path,mask)
        record=dict(source_binding=binding,decoded_native_RGB_sha256='2'*64,
            mask_path=str(path),mask_sha256=sha(path),diagnostic=dict(unresolved_support=True))
        put_json(ledger,record);records.append(record)
    guard_path=root/'guard_candidates'/sid/f'{pts}.png'
    guard_ledger=root/'guard_candidates/frames.jsonl'
    put_mask(guard_path,base|refined)
    guard=dict(source_binding=binding,source_RGB_sha256='2'*64,unresolved_support=True,
        policy='base_or_refined_same_RGB',
        base_mask=dict(path=str(base_path),sha256=sha(base_path)),
        refined_mask=dict(path=str(refined_path),sha256=sha(refined_path)),
        guard_candidate=dict(path=f'{sid}/{pts}.png',sha256=sha(guard_path)))
    put_json(guard_ledger,guard)
    pointer=root/'person_mask_guard_binding.json'
    refs=dict(base_ledger=base_ledger,refined_ledger=refined_ledger,guard_ledger=guard_ledger)
    def bind():
        put_json(pointer,{name:dict(path=str(path.relative_to(root)),sha256=sha(path))for name,path in refs.items()})
    bind()
    return dict(root=root,sid=sid,pts=pts,base=base,refined=refined,
        base_path=base_path,refined_path=refined_path,base_ledger=base_ledger,
        refined_ledger=refined_ledger,base_record=records[0],refined_record=records[1],
        guard_path=guard_path,guard_ledger=guard_ledger,guard=guard,pointer=pointer,bind=bind)


def read(source):return materials.source_exclusion(source['root'],source['sid'],source['pts'])


def test_unresolved_nonempty_refined_returns_real_raw_union_path(source):
    originals={name:sha(source[name])for name in ('base_path','refined_path','base_ledger','refined_ledger')}
    actual,path=read(source)
    assert path==source['guard_path'].resolve()
    raw=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
    assert np.array_equal(raw,source['base']|source['refined'])
    assert np.array_equal(actual,original_exclusion(raw))
    assert originals=={name:sha(source[name])for name in originals}


def test_resolved_refined_keeps_original_values_and_path_without_guard(source):
    source['refined_record']['diagnostic']['unresolved_support']=False
    put_json(source['refined_ledger'],source['refined_record'])
    source['pointer'].unlink()
    actual,path=read(source)
    assert path==source['refined_path']
    assert np.array_equal(actual,original_exclusion(source['refined']))
    assert not actual[650,950]


def test_no_refined_keeps_existing_base_path_without_ledgers(source):
    source['refined_path'].unlink();source['base_ledger'].unlink()
    source['refined_ledger'].unlink();source['pointer'].unlink()
    actual,path=read(source)
    assert path==source['base_path']
    assert np.array_equal(actual,original_exclusion(source['base']))


def test_empty_resolved_refined_still_means_entire_frame_pending(source):
    put_mask(source['refined_path'],np.zeros_like(source['refined']))
    source['refined_record'].update(mask_sha256=sha(source['refined_path']),diagnostic=dict(unresolved_support=False))
    put_json(source['refined_ledger'],source['refined_record'])
    actual,path=read(source)
    assert actual.all() and path==source['refined_path']


def test_unresolved_base_from_different_RGB_fails_even_with_nonempty_refined(source):
    source['base_record']['decoded_native_RGB_sha256']='3'*64
    put_json(source['base_ledger'],source['base_record']);source['bind']()
    with pytest.raises(ValueError,match='base is not the same source RGB'):read(source)


def test_unresolved_without_same_exposure_base_record_fails(source):
    source['base_ledger'].write_text('',encoding='utf8');source['bind']()
    with pytest.raises(ValueError,match='Missing same-exposure source-mask provenance'):read(source)


def test_unresolved_without_guard_binding_fails(source):
    source['pointer'].unlink()
    with pytest.raises(FileNotFoundError,match='same-exposure guard binding missing'):read(source)


def test_unresolved_without_same_exposure_guard_record_fails(source):
    source['guard_ledger'].write_text('',encoding='utf8');source['bind']()
    with pytest.raises(ValueError,match='same-exposure guard missing'):read(source)


@pytest.mark.parametrize('name,description',[('base_path','base PNG'),('guard_path','guard PNG')])
def test_unresolved_missing_source_PNG_fails_explicitly(source,name,description):
    source[name].unlink()
    with pytest.raises(FileNotFoundError,match=f'same-exposure {description} missing'):read(source)


def test_ledger_cache_reused_and_invalidated_by_actual_file_change(source,monkeypatch):
    source['refined_record']['diagnostic']['unresolved_support']=False
    put_json(source['refined_ledger'],source['refined_record']);source['pointer'].unlink()
    original=Path.read_bytes;reads=[]
    def tracked(path):
        if path.resolve()==source['refined_ledger'].resolve():reads.append(path)
        return original(path)
    monkeypatch.setattr(Path,'read_bytes',tracked)
    read(source);read(source)
    assert len(reads)==1
    source['refined_record']['diagnostic']['unresolved_support']=True
    put_json(source['refined_ledger'],source['refined_record'])
    with pytest.raises(FileNotFoundError,match='guard binding missing'):read(source)
    assert len(reads)==2


def test_guard_ledger_change_invalidates_cached_SHA_binding(source):
    read(source)
    source['guard_ledger'].write_text(source['guard_ledger'].read_text('utf8')+'\n',encoding='utf8')
    with pytest.raises(ValueError,match='provenance SHA256 mismatch'):read(source)


def test_guard_with_valid_file_hash_but_wrong_union_fails(source):
    put_mask(source['guard_path'],source['refined'])
    source['guard']['guard_candidate']['sha256']=sha(source['guard_path'])
    put_json(source['guard_ledger'],source['guard']);source['bind']()
    with pytest.raises(ValueError,match='not the actual raw binary union'):read(source)


def test_nonempty_refined_cannot_bypass_unknown_support_status(source):
    source['refined_record']['diagnostic']={}
    put_json(source['refined_ledger'],source['refined_record'])
    with pytest.raises(ValueError,match='unresolved support unknown'):read(source)
