import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from rinkspace.feedback_runtime import (SourceFeedbackRead, feedback_store,
    native_source, learn_source_review)


def fixture_project(tmp_path):
    project = tmp_path / 'project'
    (project / 'configs').mkdir(parents=True)
    (project / 'events' / 'test_event' / 'feedback').mkdir(parents=True)
    (project / 'configs/storage.json').write_text(json.dumps(dict(
        schema_version='rinkspace.storage.v1', data_root=str(tmp_path / 'data'))))
    return project


def source(pts=256):
    return dict(source_sha256='a'*64, source_pts=pts, time_base='1/12800', native_hw=[40,60])


def add_case(project, evidence):
    case = dict(case_id='test_finite_width', version=1, status='active', kind='fact', title='test case',
        scope=dict(event_id='test_event', entity_ids=['C0']), tags=['measurement_operator'],
        assertions=dict(physical_material_roles=dict(finite_width_circular_lines=dict(
            ids=['C0'], measurement_operator='paint_center'))),
        provenance=dict(authority='test_fixture_only'), evidence_refs=[dict(path=str(evidence),
            sha256=hashlib.sha256(evidence.read_bytes()).hexdigest())])
    with feedback_store(project,'test_event') as store:store.upsert_case(case)


def test_complete_native_key_rejects_competing_hash_clock_or_boolean_pts():
    with pytest.raises(ValueError):native_source(dict(source(), video_sha256='b'*64))
    with pytest.raises(ValueError):native_source(dict(source(), native_pts=512))
    with pytest.raises(ValueError):native_source(dict(source(), source_pts=True))
    assert native_source(source())['pts'] == 256


def test_reader_actually_uses_operator_but_never_changes_old_uv(tmp_path):
    project=fixture_project(tmp_path);evidence=tmp_path/'fact.json';evidence.write_text('{}')
    add_case(project,evidence)
    reader=SourceFeedbackRead(source(), project=project, event_id='test_event',
        entity_ids=['C0'],tags=['measurement_operator'])
    inventory=[dict(feature_id='C0',curve_id='C0:outer',operator='material_interface',uv=[12.,14.])]
    effective=reader.apply_inventory(inventory)
    assert effective[0]['operator']=='paint_center' and inventory[0]['operator']=='material_interface'
    assert effective[0]['uv']==inventory[0]['uv']
    receipt=reader.finish(np.zeros((40,60,3),np.uint8),dict(candidates_remain_pending=True),tool='fixture.read')
    assert receipt['case_use_recorded'] and len(receipt['case_use_ids'])==1
    saved=json.loads(Path(receipt['receipt']['path']).read_text())
    assert saved['fit_mutated'] is False and saved['observation_transaction_applied'] is False
    # Another independent source/frame must be freshly queried and measured.
    next_read=SourceFeedbackRead(source(512),project=project,event_id='test_event',entity_ids=['C0'])
    assert next_read.ticket['context']['source']['pts']==512
    assert next_read.used=={}


def test_changed_fact_is_blocked_and_other_event_does_not_apply(tmp_path):
    project=fixture_project(tmp_path);evidence=tmp_path/'fact.json';evidence.write_text('{}');add_case(project,evidence)
    evidence.write_text('{"changed":true}')
    reader=SourceFeedbackRead(source(),project=project,event_id='test_event',entity_ids=['C0'])
    assert reader.blockers and not reader.applicable
    assert reader.apply_inventory([dict(feature_id='C0',curve_id='C0:outer',operator='unresolved')])[0]['operator']=='unresolved'
    other=SourceFeedbackRead(source(),project=project,event_id='other_event',entity_ids=['C0'])
    assert other.ticket['cases']==[]


def test_partial_lesson_has_no_position_truth_and_rebuilds_index(tmp_path):
    from rink_feedback.cycle import FeedbackCycle
    project=fixture_project(tmp_path);image=tmp_path/'native.png';image.write_bytes(b'fixture not visual evidence')
    src=native_source(source())
    native=dict(path=str(image),sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
        kind='native_pixels',source=src,current_frame=True,independent_of_model=True)
    with feedback_store(project,'test_event') as store:
        task=FeedbackCycle(store).checkpoint('fixture',[dict(source=src,event_id='test_event',
            signal_type='operator_conflict',entity_ids=['board.contact'],tags=['material_boundary'])])['tasks'][0]
    result=learn_source_review(project=project,event_id='test_event',finding_id=task['finding_id'],source=src,
        evidence_refs=[native],resolution_type='operator_error',
        checks=dict(measurement_operator=dict(status='verified',value='ordered_board_material_flanks',evidence_refs=[native['sha256']])),
        lesson=dict(reader_check='ordered_board_material_flanks',summary='fixture partial lesson'))
    assert result['resolution']['status']=='unknown'
    assert result['consumers']==dict(knowledge=True,regression=True,observation=False,
        observation_transaction=None,good_source_UV_retained=False)
    assert 'position' in result['resolution']['unresolved_dimensions']
    with feedback_store(project,'test_event') as store:
        learned=store.get_case(result['learned_case']['case_id'])
        assert learned['assertions']['transfer_pixel_coordinates_to_new_frame'] is False
    # Only derived DB is removed inside this isolated fixture; original case survives.
    db=tmp_path/'data/events/test_event/feedback/feedback.sqlite'
    db.unlink()
    with feedback_store(project,'test_event') as store:
        assert store.get_case(result['learned_case']['case_id']) is not None


def test_source_scoped_advisory_recipe_is_consumed_as_check_not_fact(tmp_path):
    project=fixture_project(tmp_path);evidence=tmp_path/'native.png';evidence.write_bytes(b'fixture')
    case=dict(case_id='recipe',version=1,status='active',kind='recipe',title='partial check',
        scope=dict(event_id='test_event',source_video_sha256='a'*64,entity_ids=['board.contact']),
        tags=['material_boundary'],assertions=dict(lesson=dict(reader_check='ordered_board_material_flanks')),
        provenance=dict(authority='tool_fixture_not_human'),evidence_refs=[dict(path=str(evidence),
            sha256=hashlib.sha256(evidence.read_bytes()).hexdigest())])
    with feedback_store(project,'test_event') as store:store.upsert_case(case)
    reader=SourceFeedbackRead(source(),project=project,event_id='test_event',entity_ids=['board.contact'],tags=['material_boundary'])
    assert reader.ticket['cases'][0]['match_status']=='advisory'
    authority=reader.material_authority({'source_sha256':'a'*64})
    assert authority['feedback_regression_checks']==['recipe']
    receipt=reader.finish(np.zeros((40,60,3),np.uint8),dict(status='unknown'),tool='fixture.read')
    assert receipt['used_case_refs'][0]['case_id']=='recipe'
    other=SourceFeedbackRead(dict(source(),source_sha256='b'*64),project=project,event_id='test_event',entity_ids=['board.contact'],tags=['material_boundary'])
    assert other.material_authority({})=={} and not other.used


def test_validation_lineage_and_explicit_paths_remain_guarded(tmp_path):
    project=fixture_project(tmp_path);evidence=tmp_path/'source.png';evidence.write_bytes(b'fixture')
    case=dict(case_id='learned',version=1,status='active',kind='recipe',title='source recipe',
        scope=dict(event_id='test_event',source_video_sha256='a'*64,entity_ids=['board.contact']),
        tags=['material_boundary'],assertions=dict(lesson=dict(reader_check='ordered_board_material_flanks')),
        provenance={},lineage=dict(source_video_sha256='a'*64),evidence_refs=[])
    with feedback_store(project,'test_event') as store:store.upsert_case(case)
    heldout=SourceFeedbackRead(source(),project=project,event_id='test_event',entity_ids=['board.contact'],
        split_context=dict(mode='independent_validation'))
    assert heldout.ticket['cases']==[]
    with pytest.raises(ValueError,match='bounded'):
        SourceFeedbackRead(source(),project=project,event_id='test_event',run_id='../escape')
