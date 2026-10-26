"""A saved solve must belong to the current native and exact source revision."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import numpy as np

from rinkspace.session.source_programme import (ProgrammeLedger,
    navigation_dependencies, navigation_result_target)
from rinkspace.vision_native import file_ref


def case(tmp_path):
    job=tmp_path/'job.json';job.write_text('{}')
    ledger=ProgrammeLedger(tmp_path/'run',job)
    frame=dict(source_binding=dict(source_pts=256,decoded_native_rgb_sha256='a'*64),
        candidates=[dict(candidate_id='retained',localization='measured',identity='confirmed',
            original_uv=[3.25,8.5],source_group_id='original-group')])
    ledger.commit('s02',256,frame)
    model=tmp_path/'native.json';model.write_text('{"current":1}')
    camera=tmp_path/'camera.json';camera.write_text('{"controls_ref":"seed.npy"}')
    (tmp_path/'seed.npy').write_bytes(b'original camera seed')
    scope=tmp_path/'scope.json';scope.write_text('{"pts":256}')
    geometry=SimpleNamespace(refs=[file_ref(model)])
    def unchanged():
        if geometry.refs!=[file_ref(model)]:raise ValueError('Current native geometry changed')
    geometry.assert_unchanged=unchanged
    dependencies=lambda:navigation_dependencies(ledger,'s02',geometry,camera,scope)
    return ledger,frame,model,geometry,dependencies


def saved(path,dependencies):
    path.parent.mkdir(parents=True,exist_ok=True)
    outputs={}
    for name in ('controls.npy','stations.npy','camera.json'):
        target=path.parent/name
        if name=='camera.json':target.write_text('{"controls_ref":"controls.npy"}')
        else:np.save(target,np.ones((4,3)) if name=='controls.npy' else np.array([.25,.75]))
        data=target.read_bytes()
        outputs[name]=hashlib.sha256(data).hexdigest()
    path.write_text(json.dumps(dict(source_dependencies=dependencies,output_sha256=outputs,
        camera_seed=dependencies['camera_seed']['path'])))


def test_geometry_change_retains_old_navigation_and_reuses_only_its_new_revision(tmp_path):
    ledger,frame,model,geometry,dependencies=case(tmp_path)
    first,used,_=navigation_result_target(ledger.root,'s02',dependencies())
    assert not used;saved(first,dependencies());old=first.read_bytes()
    model.write_text('{"current":2}')
    with pytest.raises(ValueError,match='native geometry changed'):dependencies()
    geometry.refs=[file_ref(model)]
    second,used,reason=navigation_result_target(ledger.root,'s02',dependencies())
    assert not used and second!=first and reason=='saved_navigation_dependencies_missing_or_changed'
    saved(second,dependencies())
    assert navigation_result_target(ledger.root,'s02',dependencies())[:2]==(second,True)
    assert first.read_bytes()==old and ledger.read('s02',256)==frame
    ledger.db.close()


def test_latest_measurement_revision_invalidates_navigation_without_changing_original_uv(tmp_path):
    ledger,frame,_,_,dependencies=case(tmp_path)
    first,_,_=navigation_result_target(ledger.root,'s02',dependencies());saved(first,dependencies())
    original=(ledger.root/'frames/s02/256.json.gz').read_bytes()
    frame['candidates'][0]['identity']='unresolved'
    ledger.revise('s02',256,frame,'new independent identity evidence')
    second,used,_=navigation_result_target(ledger.root,'s02',dependencies())
    assert not used and second!=first
    assert (ledger.root/'frames/s02/256.json.gz').read_bytes()==original
    row=ledger.read('s02',256)['candidates'][0]
    assert row['original_uv']==[3.25,8.5] and row['source_group_id']=='original-group'
    ledger.db.close()


@pytest.mark.parametrize('changed',['controls.npy','stations.npy','camera.json','seed.npy','scope.json'])
def test_changed_camera_scope_or_output_cannot_be_called_a_completed_current_solve(tmp_path,changed):
    ledger,_,_,_,dependencies=case(tmp_path)
    first,_,_=navigation_result_target(ledger.root,'s02',dependencies());saved(first,dependencies())
    target=first.parent/changed if changed in ('controls.npy','stations.npy','camera.json') else tmp_path/changed
    target.write_bytes(b'changed bytes')
    second,used,_=navigation_result_target(ledger.root,'s02',dependencies())
    assert not used and second!=first and target.read_bytes()==b'changed bytes'
    ledger.db.close()


def test_legacy_result_and_interrupted_output_are_preserved(tmp_path):
    ledger,_,_,_,dependencies=case(tmp_path)
    old=ledger.root/'navigation/s02/result.json';old.parent.mkdir(parents=True)
    old.write_text('{"actual_source_UV_entered_GPU":true}')
    target,used,_=navigation_result_target(ledger.root,'s02',dependencies())
    assert not used and target!=old
    target.parent.mkdir(parents=True);(target.parent/'controls.npy').write_bytes(b'partial')
    retry,used,_=navigation_result_target(ledger.root,'s02',dependencies())
    assert not used and retry!=target and (target.parent/'controls.npy').read_bytes()==b'partial'
    assert old.read_text()=='{"actual_source_UV_entered_GPU":true}'
    ledger.db.close()


def test_modified_committed_source_is_detected_even_if_database_pin_is_unchanged(tmp_path):
    ledger,_,_,_,dependencies=case(tmp_path)
    (ledger.root/'frames/s02/256.json.gz').write_bytes(b'not the committed source')
    with pytest.raises(ValueError,match='Committed source frame changed'):dependencies()
    ledger.db.close()


def test_review_consumes_selected_revision_and_rejects_stale_geometry_or_readouts(tmp_path):
    from rinkspace.review.source_programme import current_review_navigation
    ledger,frame,model,geometry,dependencies=case(tmp_path)
    job=dict(roots=dict(repo=str(tmp_path)),scope_ref=dict(path='scope.json'),sequences=[dict(shot_id='s02')])
    first,_,_=navigation_result_target(ledger.root,'s02',dependencies());saved(first,dependencies())
    model.write_text('{"current":2}');geometry.refs=[file_ref(model)]
    second,_,_=navigation_result_target(ledger.root,'s02',dependencies());saved(second,dependencies())
    cameras,refs=current_review_navigation(job,ledger,geometry,ledger.root/'navigation',{'s02':str(second)})
    np.testing.assert_array_equal(cameras['s02']['controls'],np.ones((4,3)))
    assert refs['s02']['path']==str(second.resolve())
    with pytest.raises(ValueError,match='current source/native revision'):
        current_review_navigation(job,ledger,geometry,ledger.root/'navigation',{'s02':str(first)})
    frame['candidates'][0]['identity']='unresolved';ledger.revise('s02',256,frame,'corrected identity')
    with pytest.raises(ValueError,match='current source/native revision'):
        current_review_navigation(job,ledger,geometry,ledger.root/'navigation',{'s02':str(second)})
    ledger.db.close()
