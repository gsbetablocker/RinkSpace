"""Exact association admission contracts; no synthetic visual-quality claim."""
from copy import deepcopy
import hashlib
import json

import numpy as np
import pytest

from rink_feedback.core import Store
from rinkspace.feedback_withdrawal import (WithdrawnSourceAssociations,
    check_candidate_withdrawals,enforce_packet_withdrawals,measurement_withdrawal_applicability)


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding='utf-8')


@pytest.fixture
def feedback(tmp_path):
    project=tmp_path/'project';data=tmp_path/'data'
    dump(project/'configs/storage.json',dict(schema_version='rinkspace.storage.v1',data_root=str(data)))
    dump(project/'events/test_event/programmes/test_programme/primary_source_selection.json',dict(event_id='test_event',programme_id='test_programme',primary_source=dict(sha256='a'*64,time_base='1/12800')))
    source=dict(video_sha256='a'*64,pts=256,time_base='1/12800',coordinate_domain='native_pixels')
    image=data/'native.png';image.parent.mkdir(parents=True,exist_ok=True);image.write_bytes(b'source-evidence-fixture')
    point=dict(point_id='source-point-1',target_id='ice.curve',uv=[10.,20.])
    case=dict(case_id='human_edit:real-schema-fixture',version=1,status='active',title='withdraw exact association',
        scope=dict(event_id='test_event',source_video_sha256='a'*64,camera_id=None,entity_ids=['ice.curve']),
        tags=['manual_edit'],assertions=dict(withdrawal=dict(source=source,points=[point],
            applicability='exact_source_frame_and_point_ids_only',authority='human_editor_transaction')),
        evidence_refs=[dict(path=str(image),sha256=hashlib.sha256(image.read_bytes()).hexdigest(),kind='native_image')],
        provenance=dict(authority='human_editor_transaction'))
    database=data/'events/test_event/feedback/feedback.sqlite'
    with Store(database) as store:store.upsert_case(case)
    return project,data,source,point,case


def check(feedback,source=None,candidates=None,**kwargs):
    project,data,binding,point,_=feedback
    selected=[point] if candidates is None else candidates
    snapshot=data/'events/test_event/programmes/test_programme/source_datasets/candidate.json'
    dump(snapshot,dict(source=source or binding,candidates=selected))
    reference=dict(path=str(snapshot),sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest())
    return check_candidate_withdrawals(source or binding,selected,
        project=project,event_id='test_event',packet_reference=reference,**kwargs)


def test_only_exact_association_is_withheld_and_consumption_receipt_is_real(feedback):
    project,data,source,point,_=feedback;before=deepcopy(point)
    result=check(feedback,candidates=[point,dict(point,point_id='different-id')])
    assert len(result['withheld'])==1 and result['withheld'][0]['point_id']==point['point_id']
    assert point==before and result['case_use_recorded']
    assert not result['withheld'][0]['background_negative']
    assert result['candidate_packet_preserved'] and not result['new_frame_visual_improvement_verified']
    with Store(data/'events/test_event/feedback/feedback.sqlite') as store:
        assert store.db.execute('SELECT COUNT(*) FROM uses').fetchone()[0]==1


@pytest.mark.parametrize('change',[
    dict(pts=512),dict(video_sha256='b'*64),dict(time_base='1/15360'),dict(coordinate_domain='crop_pixels')])
def test_other_frame_video_clock_domain_never_consumes_withdrawal(feedback,change):
    result=check(feedback,source=dict(feedback[2],**change))
    assert result['withheld']==[] and not result['case_use_recorded']


@pytest.mark.parametrize('change',[
    dict(point_id='new-identity'),dict(target_id='ice.other'),dict(uv=[10.,20.000001])])
def test_nearby_uv_and_similar_id_are_never_substituted(feedback,change):
    result=check(feedback,candidates=[dict(feedback[3],**change)])
    assert not result['withheld'] and not result['case_use_recorded']


def test_new_query_without_original_id_is_explicitly_inapplicable(feedback):
    result=check(feedback,candidates=[dict(query=0,target_id='ice.curve',uv=[10.,20.])])
    assert result['status']=='not_applicable_missing_stable_point_ids_and_targets'
    measurement=dict(candidate_uv=np.array([[10.,20.]]),candidate_query=np.array([0]),candidate_flags=np.array([0],np.uint8))
    before=measurement['candidate_flags'].copy()
    own=measurement_withdrawal_applicability(measurement)
    assert own['candidate_query_is_not_original_point_identity'] and not own['case_use_recorded']
    np.testing.assert_array_equal(measurement['candidate_flags'],before)


def test_latest_undo_retraction_allows_previously_withdrawn_association(feedback):
    _,data,_,_,case=feedback
    assert check(feedback)['withheld']
    retracted=deepcopy(case);retracted.update(version=2,status='retracted')
    with Store(data/'events/test_event/feedback/feedback.sqlite') as store:store.upsert_case(retracted)
    result=check(feedback)
    assert not result['withheld'] and not result['case_use_recorded']


def test_current_source_image_conflict_is_withheld_for_recheck_not_applied(feedback):
    _,data,source,_,case=feedback
    bound=deepcopy(case);bound['assertions']['withdrawal']['source']['decoded_RGB_sha256']='c'*64
    bound['version']=2
    with Store(data/'events/test_event/feedback/feedback.sqlite') as store:store.upsert_case(bound)
    result=check(feedback,source=dict(source,decoded_RGB_sha256='d'*64))
    assert result['withheld'][0]['status']=='withheld_source_version_recheck_required'
    assert not result['case_use_recorded'] and not result['background_negative_created']


def test_enforce_retains_input_packet_and_exposes_withheld_receipt(feedback):
    project,_,source,point,_=feedback
    packet=dict(source=dict(source,event_id='test_event'),features=[dict(id='curve',segments=[dict(
        uv=[point['uv']],point_ids=[point['point_id']])])])
    binding=dict(observations=[dict(source_feature_id='curve',source_segment_index=0,
        source_vertex_indices=[0],source_point_ids=[point['point_id']],target_id=point['target_id'])])
    before=deepcopy(packet)
    path=feedback[1]/'events/test_event/programmes/test_programme/source_datasets/candidate.json'
    dump(path,packet)
    with pytest.raises(WithdrawnSourceAssociations) as caught:
        enforce_packet_withdrawals(packet,binding,project=project,packet_path=path)
    assert packet==before and caught.value.receipt['status']=='withheld'
    assert caught.value.receipt['receipt']['path']


def test_diagnostic_tmp_candidate_never_consults_or_writes_production_feedback(feedback):
    project,data,source,point,_=feedback
    path=data/'tmp/test_candidate.json';dump(path,dict(source=source,candidates=[point]))
    reference=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    result=check_candidate_withdrawals(source,[point],project=project,event_id='test_event',packet_reference=reference)
    assert result['status']=='diagnostic_candidate_not_registered'
    assert not result['production_feedback_index_consulted'] and not result['case_use_recorded']
    with Store(data/'events/test_event/feedback/feedback.sqlite') as store:
        assert store.db.execute('SELECT COUNT(*) FROM uses').fetchone()[0]==0
