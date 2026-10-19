"""Delivery contract tests; synthetic UVs make no visual accuracy claim."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from rink_feedback.core import Store
from rinkspace.feedback_outbox import deliver_human_edit, apply_editor_role_labels


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')
    return hashlib.sha256(path.read_bytes()).hexdigest()


def packet(points):
    return dict(source=dict(video_sha256='a'*64, native_pts=256, time_base='1/12800',
        coordinate_domain='native_pixels', event_id='test_event', programme_id='test_programme'),
        source_pts=256, programme_frame_index_1based=1, point_count=len(points),
        features=[dict(id='curve', target_id='ice.curve',
            segments=[dict(uv=[uv for _, uv in points], point_ids=[pid for pid, _ in points])])])


@pytest.fixture
def journal(tmp_path):
    project=tmp_path/'project';data=tmp_path/'data'
    dump(project/'configs/storage.json',dict(schema_version='rinkspace.storage.v1', data_root=str(data)))
    dump(project/'events/test_event/programmes/test_programme/primary_source_selection.json',dict(event_id='test_event',programme_id='test_programme',primary_source=dict(sha256='a'*64,time_base='1/12800')))
    folder=data/'events/test_event/programmes/test_programme/media/point_editor_history/delete'
    before=packet([('p0',[10.,20.]),('p1',[20.,20.])]);after=packet([('p1',[20.,20.])])
    dump(folder/'before_packet.json',before);after_sha=dump(folder/'after_packet.json',after)
    binding=dict(source_pts=256, source_time_base='1/12800', point_count=1,
                 source_packet=dict(path='unused-live.json', sha256=after_sha))
    dump(folder/'after_binding.json',binding)
    tx=dict(schema='rinkspace.human-frame-point-edit.v1', id='delete', index=1,
        status='committed', source_pts=256, model_geometry_changed=False,
        action=dict(action='edit',operations=[dict(type='delete',point_id='p0')]),
        point_count_before=2, point_count_after=1, packet_after=dict(sha256=after_sha))
    dump(folder/'transaction.json',tx)
    return project,data,folder


def deliver(journal, **kwargs):
    project,data,folder=journal
    return deliver_human_edit(project,folder/'transaction.json',dataset=data/'events/test_event/programmes/test_programme/source_datasets/dataset',**kwargs)


def test_committed_journal_consumes_once_and_keeps_permanent_regression(journal):
    first=deliver(journal);second=deliver(journal)
    assert first==second and first['case_ref']['version']==1
    assert Path(first['durable_case']['path']).exists() and Path(first['regression']['path']).exists()
    project,data,_=journal
    with Store(data/'events/test_event/feedback/feedback.sqlite') as store:
        case=store.get_case('human_edit:delete')
        assert case['assertions']['source_edit']['deleted_points'][0]['point_id']=='p0'
        assert case['assertions']['source_edit']['reason']=='unknown'
        assert not case['assertions']['negative_supervision']['background_negative_created']
        assert store.db.execute('SELECT COUNT(*) FROM cases').fetchone()[0]==1


def test_matching_acceptance_advances_same_case_once_and_snapshots_receipt(journal):
    first=deliver(journal);project,data,folder=journal
    acceptance=data/'user_acceptance.json'
    dump(acceptance,dict(accepted=True,official_source=dict(sha256='a'*64),source_time_base='1/12800',
        frames=[dict(index=1,native_pts=256,point_count=1,
            packet_sha256=hashlib.sha256((folder/'after_packet.json').read_bytes()).hexdigest(),
            binding_sha256=hashlib.sha256((folder/'after_binding.json').read_bytes()).hexdigest())]))
    second=deliver(journal,acceptance_path=acceptance)
    assert second['case_ref']['version']==2 and second['source_observations_accepted']
    assert deliver(journal,acceptance_path=acceptance)==second
    with Store(data/'events/test_event/feedback/feedback.sqlite') as store:
        case=store.get_case('human_edit:delete')
        ref=next(r for r in case['evidence_refs'] if r['kind']=='acceptance')
        assert Path(ref['path']).parent==folder
        assert not case['assertions']['acceptance']['geometry_accuracy_inferred']
    assert first['case_ref']['case_id']==second['case_ref']['case_id']


def test_add_click_keeps_unknown_precision_and_undo_retracts_old_instruction(journal):
    project,data,folder=journal
    before=packet([('p0',[10.,20.])]);after=packet([('p0',[10.,20.]),('manual:new',[25.,30.])])
    dump(folder/'before_packet.json',before);after_sha=dump(folder/'after_packet.json',after)
    dump(folder/'after_binding.json',dict(source_pts=256,source_time_base='1/12800',point_count=2,source_packet=dict(sha256=after_sha)))
    tx=json.loads((folder/'transaction.json').read_text(encoding='utf-8'))
    tx.update(id='delete',point_count_before=1,point_count_after=2,packet_after=dict(sha256=after_sha),
              action=dict(action='edit',operations=[dict(type='add',target_id='ice.curve',uv=[25.,30.])]))
    dump(folder/'transaction.json',tx)
    deliver(journal)
    undo_folder=folder.parent/'undo';dump(undo_folder/'before_packet.json',after)
    undo_sha=dump(undo_folder/'after_packet.json',before)
    dump(undo_folder/'after_binding.json',dict(source_pts=256,source_time_base='1/12800',point_count=1,source_packet=dict(sha256=undo_sha)))
    undo_tx=dict(tx,id='undo',point_count_before=2,point_count_after=1,
                 action=dict(action='undo',undo_of='delete'),packet_after=dict(sha256=undo_sha))
    dump(undo_folder/'transaction.json',undo_tx)
    result=deliver_human_edit(project,undo_folder/'transaction.json',dataset=data/'events/test_event/programmes/test_programme/source_datasets/dataset')
    assert result['rollback_case_ref']['case_id']=='human_edit:delete'
    with Store(data/'events/test_event/feedback/feedback.sqlite') as store:
        addition=store.get_case('human_edit:delete',1)
        assert addition['assertions']['source_edit']['measurement_uncertainty']=='unknown'
        assert store.get_case('human_edit:delete')['status']=='retracted'
    assert deliver_human_edit(project,undo_folder/'transaction.json',dataset=data/'events/test_event/programmes/test_programme/source_datasets/dataset')==result


def test_uncommitted_and_unrecorded_delta_never_become_corrections(journal):
    _,data,folder=journal
    tx=json.loads((folder/'transaction.json').read_text(encoding='utf-8'))
    tx['status']='rolled_back';dump(folder/'transaction.json',tx)
    assert deliver(journal)['delivered'] is False
    assert not (data/'events/test_event/feedback/feedback.sqlite').exists()
    tx['status']='committed';tx['action']['operations']=[dict(type='move',point_id='p0',uv=[40,40])]
    dump(folder/'transaction.json',tx)
    with pytest.raises(ValueError,match='Unsupported'):deliver(journal)


def test_editor_label_uses_event_role_fact_without_rewriting_uv(tmp_path):
    root=tmp_path/'project';model=root/'events/test_event/models/model'
    dump(root/'events/test_event/model_constraints/roles.json',dict(
        schema='rinkspace.user-confirmed-ice-measurement-roles.v1',event_id='test_event',
        definitions=dict(lines=dict(ids=['C0'],measurement_operator='paint_center'))))
    registry=dict(targets={'ice.C0:outer':dict(kind='ice_curve',curve_id='C0:outer')})
    apply_editor_role_labels(registry,model,root)
    assert registry['targets']['ice.C0:outer']['effective_measurement_operator']=='paired_material_boundary_midline'


def test_deleted_derived_index_rebuilds_cases_and_real_snapshot_regression(journal):
    from rinkspace.feedback_outbox import verify_manual_regressions
    first=deliver(journal);project,data,_=journal
    database=data/'events/test_event/feedback/feedback.sqlite'
    database.unlink()
    assert deliver(journal)==first
    with Store(database) as store:
        assert store.get_case('human_edit:delete')['version']==1
    report=verify_manual_regressions(project,'test_event')
    assert report['checked_active_cases']==1
    assert report['results'][0]['exact_snapshot_delta_rechecked']
    assert not report['new_frame_visual_improvement_verified']


def test_diagnostic_dataset_cannot_publish_real_event_packet_as_human_fact(journal):
    project,data,folder=journal
    result=deliver_human_edit(project,folder/'transaction.json',dataset=data/'tmp/test_fixture/dataset')
    assert result['status']=='diagnostic_dataset_not_published' and not result['delivered']
    assert not (data/'events/test_event/feedback/feedback.sqlite').exists()
    assert not (project/'events/test_event/feedback/manual').exists()


def test_scope_uses_explicit_nested_project_roots_and_registered_source(tmp_path):
    from rinkspace.feedback_outbox import _scope,registered_source_dataset
    outer=tmp_path/'events/outer_event/feedback/tests'
    project=outer/'project';data=outer/'data'
    dump(project/'configs/storage.json',dict(schema_version='rinkspace.storage.v1',data_root=str(data)))
    dump(project/'events/inner_event/programmes/inner_programme/primary_source_selection.json',dict(
        event_id='inner_event',programme_id='inner_programme',primary_source=dict(sha256='a'*64,time_base='1/12800')))
    source=dict(video_sha256='a'*64,native_pts=256,time_base='1/12800',coordinate_domain='native_pixels',
        event_id='inner_event',programme_id='inner_programme')
    dataset=data/'events/inner_event/programmes/inner_programme/source_datasets/current'
    assert _scope(source,dataset,project)==('inner_event','inner_programme')
    assert registered_source_dataset(project,dataset,source)[0]
    with pytest.raises(ValueError,match='registered source'):
        _scope(dict(source,event_id='outer_event'),dataset,project)
    model=project/'events/inner_event/models/current'
    dump(project/'events/inner_event/model_constraints/roles.json',dict(
        schema='rinkspace.user-confirmed-ice-measurement-roles.v1',event_id='inner_event',
        definitions=dict(lines=dict(ids=['C0'],measurement_operator='paint_center'))))
    registry=dict(targets={'ice.C0:outer':dict(kind='ice_curve',curve_id='C0:outer')})
    apply_editor_role_labels(registry,model,project)
    assert registry['targets']['ice.C0:outer']['effective_measurement_operator']=='paired_material_boundary_midline'


def test_nested_project_scope_wins_over_its_outer_data_directory(tmp_path):
    from rinkspace.feedback_outbox import _scope
    data=tmp_path/'data';project=data/'events/outer_event/nested_project'
    dump(project/'configs/storage.json',dict(schema_version='rinkspace.storage.v1',data_root=str(data)))
    assert _scope({},project/'events/inner_event/models/current',project)==('inner_event',None)


def installation_contract(journal,path):
    _,data,folder=journal
    digest=dump(path,dict(schema='test_post_install_contract_input'))
    original=folder/'before_packet.json'
    image=data/'original_native.png';image.write_bytes(b'diagnostic-evidence-not-visual-truth')
    return dict(packet_sha256=digest,active_FIT_source_UV_and_optical_support_explicitly_revised=True,
        source_video_sha256='a'*64,time_base='1/12800',
        original_absolute_packet=dict(path=str(original),sha256=hashlib.sha256(original.read_bytes()).hexdigest()),
        source_evidence=[dict(path=str(image),sha256=hashlib.sha256(image.read_bytes()).hexdigest(),pts=256)],
        corrected_rows=[dict(locator=dict(block='test',group_actual_pts=256,group_order=0,row_order=0,collection_role='FIT'),
            point_indices=[0],physical_edge_role='diagnostic_material_boundary')])


def test_installation_inside_event_diagnostic_tree_never_publishes(journal):
    from rinkspace.feedback_outbox import record_source_installation
    project,data,_=journal
    path=data/'events/test_event/feedback/test_installer/correction.json'
    result=record_source_installation(project,path,installation_contract(journal,path))
    assert result['status']=='diagnostic_packet_not_published' and result['runtime_rows_installed']
    assert not result['production_feedback_index_consulted']
    assert not (data/'events/test_event/feedback/feedback.sqlite').exists()
    assert not (project/'events/test_event/feedback/installs').exists()


def test_installation_retraction_survives_replay_and_derived_index_rebuild(journal):
    from rinkspace.feedback_outbox import record_source_installation,replay_source_installation_outbox,_immutable
    project,data,_=journal
    path=data/'events/test_event/programmes/test_programme/source_datasets/correction.json'
    contract=installation_contract(journal,path)
    published=record_source_installation(project,path,contract)
    assert published['status']=='feedback_indexed'
    original=Path(published['durable_case']['path'])
    withdrawn=json.loads(original.read_text(encoding='utf-8'));withdrawn.update(version=2,status='retracted')
    withdrawn['provenance']['retracted_reason']='diagnostic_fixture_not_production_authority'
    _immutable(original.with_name(original.name.replace('.v1.case.json','.v2.case.json')),withdrawn,project)
    database=data/'events/test_event/feedback/feedback.sqlite';database.unlink()
    replayed=replay_source_installation_outbox(project,'test_event')
    assert [entry['status'] for entry in replayed]==['retracted_terminal_no_republish']
    with Store(database) as store:
        assert store.get_case(published['case_ref']['case_id'])['status']=='retracted'
    repeated=record_source_installation(project,path,contract)
    assert repeated['status']=='retracted_terminal_no_republish'
    assert repeated['runtime_rows_installed']
