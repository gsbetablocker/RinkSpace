import copy
import json
import numpy as np
import pytest
from rinkfit_joint.io_utils import file_sha256
from rinkfit_joint.joint_metrics import PacketArrays, metrics
from rinkfit_joint.report_identity import (actual_exclusion_ledger, frozen_target_binding,
    make_replay_binding, require_replay_binding, assert_frozen_source_prefix)


def write_json(path, value):
    path.write_text(json.dumps(value), encoding='utf8')
    return path


def fixture(tmp_path):
    n=3; arrays={}
    values=dict(uv=np.array([[10.,20.],[30.,40.],[50.,60.]],dtype='<f8'),
        native_pts=np.array([256256,256512,256768],dtype='<i8'),
        shot_index=np.zeros(n,dtype='<u2'), material_index=np.zeros(n,dtype='<u4'),
        kind=np.ones(n,dtype='u1'), admission=np.array([1,1,0],dtype='u1'),
        fit_role=np.array([1,1,0],dtype='u1'), E=np.full(n,2.,dtype='<f8'),
        station=np.full(n,.5,dtype='<f8'), row=np.full(n,20.,dtype='<f8'),
        alpha=np.zeros(n,dtype='<f8'), path=np.zeros(n,dtype='<u4'))
    for key,value in values.items():
        path=tmp_path/(key+'.bin'); value.tofile(path)
        arrays[key]=dict(file=str(path),sha256=file_sha256(path),dtype=value.dtype.str,shape=list(value.shape))
    W=np.eye(4,dtype='<f8'); wp=tmp_path/'W.bin'; W.tofile(wp)
    packet=dict(schema='rinkfit.joint_packet.v1', row_count=n, arrays=arrays,
        shot_ids=['s01'],source_row_ids=['r0','r1','r2'],source_dataset_ids=['original']*n,
        groups=[dict(id='actual_correlated_group',statistic_kind='original',rows=[0,1],rank=4,
            W=dict(file=str(wp),sha256=file_sha256(wp),dtype='<f8',shape=[4,4]))],
        queries=[dict(row_index=1,normal=[1.,0.],alpha=[-1.,1.],native_uv=[[29.,40.],[31.,40.]],alpha0=0.)],
        allowed_paths={'0':[0],'1':[0]})
    pp=write_json(tmp_path/'packet.json',packet)
    state=tmp_path/'state'; state.mkdir()
    model=dict(materials=[dict(label='Ad1.left_top')], shots={'s01':dict(rig_id='main')},
        rigs={'main':dict(optics_block=0)},blocks=[dict(values=[0.]*12,lower=[-1.]*12,upper=[1.]*12)])
    mp=write_json(state/'model.json',model)
    error=np.array([[5.,0.],[5.,0.],[np.nan,np.nan]],dtype='<f8'); pred=values['uv']+error
    data={'predicted_uv.f64':pred,'error_native_uv.f64':error,'valid.u8':np.array([1,1,0],dtype='u1')}
    for key in ('station','row','alpha','path'):
        data[key+('.u32' if key=='path' else '.f64')]=values[key]
    files={'model.json':dict(sha256=file_sha256(mp),size_bytes=mp.stat().st_size)}
    for key,value in data.items():
        path=state/key; value.tofile(path); files[key]=dict(sha256=file_sha256(path),size_bytes=path.stat().st_size)
    value=dict(model_sha256=file_sha256(mp),files=files,full_W_half_cost=25.,
        profiled_native_export_semantics='actual fixture replay locally profiles finite nuisance')
    ap=PacketArrays(pp)
    value['source_objective_binding']=make_replay_binding(pp,packet,value,actual_exclusion_ledger(packet,ap.load))
    write_json(state/'value_report.json',value)
    return pp,packet,mp,state,value


def test_original_packet_reports_original_E_and_actual_exclusion_even_when_small(tmp_path):
    pp,packet,mp,state,value=fixture(tmp_path)
    report=metrics(mp,pp,state,tmp_path/'report')
    assert report['original_target_verified']
    assert report['overall']['all_original_E_passed'] is False
    assert report['overall']['rows_above_original_E']==2
    assert report['CXX_full_W_recompute_difference']==0.
    assert [x['row_index'] for x in report['actual_exclusion_ledger']]==[2]
    assert report['s01_excluded_original_sources']==report['actual_exclusion_ledger']


@pytest.mark.parametrize('change',['E','role','material','Query','order','W','paths'])
def test_wrong_target_rejected_before_any_report_is_written(tmp_path,change):
    pp,packet,mp,state,value=fixture(tmp_path); bad=copy.deepcopy(packet)
    if change in {'E','role','material'}:
        key={'E':'E','role':'fit_role','material':'material_index'}[change]
        a=PacketArrays(pp).load(key).copy(); a[0]=20 if change=='E' else 0 if change=='role' else 1
        path=tmp_path/(change+'_changed.bin'); a.tofile(path)
        bad['arrays'][key].update(file=str(path),sha256=file_sha256(path))
    elif change=='Query':bad['queries'][0]['alpha']=[-300.,300.]
    elif change=='order':bad['source_row_ids']=['r1','r0','r2']
    elif change=='W':bad['groups'][0]['rows']=[1,0]
    else:bad['allowed_paths']['0']=[1]
    bp=write_json(tmp_path/'bad.json',bad); output=tmp_path/'must_not_exist'
    with pytest.raises(ValueError,match='packet differs'):
        metrics(mp,bp,state,output)
    assert not output.exists()
    # Even an accidental packet-only relabel cannot bypass the frozen target.
    forged=copy.deepcopy(value); forged['source_objective_binding']['source_packet_sha256']=file_sha256(bp)
    with pytest.raises(ValueError,match='frozen'):
        require_replay_binding(bp,bad,forged,state)


def test_changed_auxiliary_seed_does_not_redefine_the_source_operator(tmp_path):
    pp,packet,mp,state,value=fixture(tmp_path); modified=copy.deepcopy(packet)
    for key in ('station','row','alpha','path'):
        modified['arrays'][key]['sha256']='f'*64
    modified['queries'][0]['alpha0']=.2
    assert frozen_target_binding(packet)==frozen_target_binding(modified)


def test_legacy_inspection_is_explicit_and_has_no_original_pass_claim(tmp_path):
    pp,packet,mp,state,value=fixture(tmp_path); del value['source_objective_binding']
    write_json(state/'value_report.json',value)
    with pytest.raises(ValueError,match='lacks producer'):
        metrics(mp,pp,state,tmp_path/'must_not_exist')
    report=metrics(mp,pp,state,tmp_path/'diagnostic',legacy_unbound_diagnostic=True)
    assert report['original_target_verified'] is False
    assert report['source_objective_identity']['status']=='legacy_unbound_diagnostic'
    assert 'all_original_E_passed' not in report['overall']
    assert report['overall']['all_supplied_E_passed'] is False


def test_typed_append_preserves_entire_old_target_but_not_aux_seed(tmp_path):
    pp,packet,mp,state,value=fixture(tmp_path)
    current=copy.deepcopy(packet); current['row_count']=4
    for key in ('source_row_ids','source_dataset_ids'):current[key].append('new')
    old_values={k:PacketArrays(pp).load(k) for k in packet['arrays']}
    new_values={k:np.concatenate([v,v[-1:]]) for k,v in old_values.items()}
    assert_frozen_source_prefix(packet,current,old_values.__getitem__,new_values.__getitem__)
    for key in ('admission','fit_role','E','material_index'):
        changed={k:v.copy() for k,v in new_values.items()}; changed[key][0]+=1
        with pytest.raises(ValueError,match='frozen prefix array'):
            assert_frozen_source_prefix(packet,current,old_values.__getitem__,changed.__getitem__)
    for key in ('queries','groups','allowed_paths'):
        bad=copy.deepcopy(current)
        if key=='queries':bad[key][0]['normal']=[0.,1.]
        elif key=='groups':bad[key][0]['rows']=[1,0]
        else:bad[key]['0']=[1]
        with pytest.raises(ValueError):
            assert_frozen_source_prefix(packet,bad,old_values.__getitem__,new_values.__getitem__)


def test_same_length_production_aux_rebind_rejects_wrong_packet_and_legacy(tmp_path):
    from rinkfit_joint.joint_solver import rebind_current_auxiliaries
    pp,packet,mp,state,value=fixture(tmp_path)
    bad=copy.deepcopy(packet); bad['source_row_ids']=['r1','r0','r2']
    bp=write_json(tmp_path/'same_length_other.json',bad)
    with pytest.raises(ValueError,match='packet differs'):
        rebind_current_auxiliaries(bp,state,tmp_path/'must_not_exist.json')
    del value['source_objective_binding']; write_json(state/'value_report.json',value)
    with pytest.raises(ValueError,match='lacks producer'):
        rebind_current_auxiliaries(pp,state,tmp_path/'must_not_exist.json')


def test_native_report_identity_matches_python_producer(tmp_path):
    import os
    import subprocess
    executable=os.environ.get('RINKFIT_REPORT_IDENTITY_TEST_EXE')
    if not executable:
        pytest.skip('set RINKFIT_REPORT_IDENTITY_TEST_EXE to the compiled standalone native regression')
    pp,packet,mp,state,value=fixture(tmp_path)
    native_input=dict(packet=packet,source_packet_sha256=file_sha256(pp),report=value,
        exclusions=actual_exclusion_ledger(packet,PacketArrays(pp).load))
    ip=write_json(tmp_path/'native_input.json',native_input)
    result=subprocess.run([executable,str(ip)],capture_output=True,text=True,check=True)
    assert json.loads(result.stdout)==value['source_objective_binding']
    # The native loader retains compact ID prefixes; Python consumers hydrate
    # them. The SHA-bound suffix registry has the same objective either way.
    hydrated=copy.deepcopy(packet)
    hydrated['dense_source_binding']=dict(row_start=2,count=1,source_id_utf8=dict(sha256='a'*64))
    compact=copy.deepcopy(hydrated)
    for key in ('source_row_ids','source_dataset_ids'):compact[key]=compact[key][:2]
    native_input['packet']=compact; write_json(ip,native_input)
    result=subprocess.run([executable,str(ip)],capture_output=True,text=True,check=True)
    expected=make_replay_binding(pp,hydrated,value,native_input['exclusions'])
    assert json.loads(result.stdout)==expected


def test_legal_current_auxiliary_rebind_preserves_original_source_target(tmp_path):
    from rinkfit_joint.joint_solver import rebind_current_auxiliaries
    pp,packet,mp,state,value=fixture(tmp_path)
    result=rebind_current_auxiliaries(pp,state,tmp_path/'rebound.json')
    current=json.loads((tmp_path/'rebound.json').read_text(encoding='utf8'))
    assert result['row_count']==packet['row_count']
    assert frozen_target_binding(current)==frozen_target_binding(packet)
    assert current['current_full_state_auxiliary_binding']['source_UV_W_admission_roles_changed'] is False
    assert current['source_row_ids']==packet['source_row_ids']


def test_exact_dependency_location_change_is_not_a_new_E_W_objective(tmp_path):
    pp,packet,mp,state,value=fixture(tmp_path); moved=copy.deepcopy(packet)
    for d in moved['arrays'].values():d['file']='relocated/'+d['file'].split('/')[-1]
    moved['groups'][0]['W']['file']='elsewhere/W.bin'
    assert frozen_target_binding(moved)==frozen_target_binding(packet)


def test_final_model_serialization_refresh_and_explicit_snapshot_manifest_binding(tmp_path):
    pp,packet,mp,state,value=fixture(tmp_path)
    explicit=state/'explicit_nuisance'; explicit.mkdir()
    files={}
    for name in ('station.f64','row.f64','alpha.f64','path.u32'):
        path=explicit/name; path.write_bytes((state/name).read_bytes())
        files[name]=dict(sha256=file_sha256(path),size_bytes=path.stat().st_size)
    em=write_json(explicit/'manifest.json',dict(packet_sha256=file_sha256(pp),raw_rows=3,files=files))
    # The production wrapper can reserialize exact model blocks after replay.
    model=json.loads(mp.read_text(encoding='utf8')); model['exact_checkpoint_model_blocks']=True
    write_json(mp,model)
    value['model_sha256']=file_sha256(mp)
    value['files']['model.json']=dict(sha256=file_sha256(mp),size_bytes=mp.stat().st_size)
    value['complete_snapshot_attachment']=dict(schema='rinkfit.joint_replay_complete_snapshot_attachment.v1',
        state_token='actual_complete_snapshot_token',source_packet_sha256=file_sha256(pp),
        physical_model_sha256=file_sha256(mp),explicit_nuisance_manifest=dict(file='explicit_nuisance/manifest.json',sha256=file_sha256(em)),
        replay_arrays_locally_profile_auxiliaries=True,complete_snapshot_certification_claimed=False)
    with pytest.raises(ValueError,match='state identity changed'):
        require_replay_binding(pp,packet,value,state)
    ap=PacketArrays(pp)
    value['source_objective_binding']=make_replay_binding(pp,packet,value,actual_exclusion_ledger(packet,ap.load))
    assert require_replay_binding(pp,packet,value,state)['original_target_verified']
    (explicit/'row.f64').write_bytes(b'broken')
    with pytest.raises(ValueError,match='explicit auxiliary file changed'):
        require_replay_binding(pp,packet,value,state)


def test_changed_W_payload_is_rejected_before_writing_report(tmp_path):
    pp,packet,mp,state,value=fixture(tmp_path)
    (tmp_path/'W.bin').write_bytes(np.ones((4,4),dtype='<f8').tobytes())
    output=tmp_path/'must_not_exist'
    with pytest.raises(ValueError,match='Frozen target numerical dependency changed'):
        metrics(mp,pp,state,output)
    assert not output.exists()


def comparison6_fixture(tmp_path):
    pp,packet,mp,state,value=fixture(tmp_path)
    packet['shot_ids']=['s%02d'%i for i in range(1,7)]
    write_json(pp,packet)
    scopes=[(256256,575488),(575744,720640),(720896,745728),
        (745984,948736),(948992,1300992),(1301248,1425408)]
    model=json.loads(mp.read_text(encoding='utf8'))
    model['shots']={sid:dict(rig_id='main',scope=dict(first_pts=lo,last_pts=hi))
        for sid,(lo,hi) in zip(packet['shot_ids'],scopes)}
    model['materials'][0]['paths']=[dict(physical_point=True)]
    write_json(mp,model)
    for name in ('predicted_uv.f64','error_native_uv.f64','valid.u8'):
        target=state/('raw_station_RS_'+name); target.write_bytes((state/name).read_bytes())
        value['files'][target.name]=dict(sha256=file_sha256(target),size_bytes=target.stat().st_size)
    value['model_sha256']=file_sha256(mp)
    value['files']['model.json']=dict(sha256=file_sha256(mp),size_bytes=mp.stat().st_size)
    value['source_objective_binding']=make_replay_binding(pp,packet,value,actual_exclusion_ledger(packet,PacketArrays(pp).load))
    write_json(state/'value_report.json',value)
    return pp,packet,mp,state,value


def test_comparison_accepts_same_six_shot_scope_without_claiming_full_programme(tmp_path):
    from rinkfit_joint.recovery_metrics import compare
    pp,packet,mp,state,value=comparison6_fixture(tmp_path)
    report=compare(pp,[dict(label='baseline',state=str(state)),dict(label='candidate',state=str(state))],tmp_path/'six_report')
    assert report['native_exposures']==4568
    assert report['full_programme_scope'] is False
    assert report['scope']['coverage_scope']=='declared_shots_only'
    assert report['scope']['shot_ids']==packet['shot_ids']
    assert all(entry['native_exposures']==4568 for entry in report['states'].values())


@pytest.mark.parametrize('change',['shorter','longer','missing_shot','extra_shot'])
def test_comparison_rejects_different_candidate_native_scope_before_output(tmp_path,change):
    import shutil
    from rinkfit_joint.recovery_metrics import compare
    pp,packet,mp,state,value=comparison6_fixture(tmp_path)
    candidate=tmp_path/'other';shutil.copytree(state,candidate)
    model=json.loads((candidate/'model.json').read_text(encoding='utf8'))
    if change=='shorter':model['shots']['s06']['scope']['last_pts']-=256
    elif change=='longer':model['shots']['s06']['scope']['last_pts']+=256
    elif change=='missing_shot':del model['shots']['s06']
    else:model['shots']['s07']=dict(rig_id='main',scope=dict(first_pts=1425664,last_pts=1425664))
    mp=write_json(candidate/'model.json',model)
    value['model_sha256']=file_sha256(mp)
    value['files']['model.json']=dict(sha256=file_sha256(mp),size_bytes=mp.stat().st_size)
    value['source_objective_binding']=make_replay_binding(pp,packet,value,actual_exclusion_ledger(packet,PacketArrays(pp).load))
    write_json(candidate/'value_report.json',value)
    output=tmp_path/'must_not_exist'
    with pytest.raises(ValueError,match='scope|shot'):
        compare(pp,[dict(label='baseline',state=str(state)),dict(label='candidate',state=str(candidate))],output)
    assert not output.exists()


def test_native_unfinished_AL_receipt_is_distinct_from_failure_success_and_resume(tmp_path):
    from rinkfit_joint.joint_solver import _explicit_execution_status
    receipt=dict(run_status='inner_accuracy_unfinished',incomplete_inner=True,
        inner_budget_exhausted_yields_same_context=True)
    write_json(tmp_path/'engineering_receipt.json',receipt)
    result=_explicit_execution_status(tmp_path,3,True)
    assert result['native_execution_status']=='inner_accuracy_unfinished'
    assert result['inner_same_AL_context_preserved'] is True
    assert result['native_process_success'] is False
    assert result['fit_acceptance_claimed'] is False and result['automatic_continuation_started'] is False
    assert _explicit_execution_status(tmp_path,0,True)['native_execution_status']=='receipt_exit_status_mismatch'
    receipt['run_status']='stopped';write_json(tmp_path/'engineering_receipt.json',receipt)
    assert _explicit_execution_status(tmp_path,0,True)['native_execution_status']=='stopped'
