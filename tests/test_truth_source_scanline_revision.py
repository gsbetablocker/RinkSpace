from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from calibration.solver_runtime import truth_source_location_revision as common
from calibration.solver_runtime import truth_source_scanline_revision as api


def save(path, value):
    path = Path(path)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    return common.reference(path)


@pytest.fixture
def case(tmp_path):
    rgb = tmp_path / 'original.png'; rgb.write_bytes(b'original native RGB bytes')
    producer = tmp_path / 'original_producer.py'; producer.write_text('def edge(): pass\n')
    extraction = tmp_path / 'original_extraction.json'; extraction_ref = save(extraction, {'original': True})
    rows = [dict(id=f'ice{i}', split='holdout' if i == 1 else 'fit', pts=10+i, actual_pts=10+i,
        uv=[[2., 3.], [4., 5.]], sigma_px=[4., 6.],
        curve=dict(kind='house_ring', side='F', row=3, ring='outer'),
        source_raw_frame=str(rgb), source_file_sha256=common._sha(rgb), extraction_input=extraction_ref,
        source_search_support=[dict(source_scan_axis='x', uv=uv, sigma_px=sigma,
            manual_source_bracket=[uv[1], 0, 9], gray_profile=[1., 2.])
            for uv, sigma in zip([[2.,3.],[4.,5.]],[4.,6.])]) for i in range(3)]
    old_path = tmp_path / 'old_ice.json'; old_ref = save(old_path, {'schema':'ice-packet','rows':rows})
    receipt = dict(schema=api.RECEIPT_SCHEMA, original_packet_reference=old_ref,
        qualification_script_reference=common.reference(producer),
        source_only_RGB_replayed=True, source_images_written=0,
        model_or_camera_used_for_measurement=False, original_producer_reference=common.reference(producer),
        original_extraction_reference=extraction_ref, rows=[dict(original_row_index=i, id=rows[i]['id'],
            split=rows[i]['split'], source_row_reference=old_ref,
            source_row_canonical_json_sha256=common._row_hash(rows[i]), measured_source_row=rows[i],
            physical_branch='max_x', fixed_axis='y', replayed_original_UV_sigma_profiles_bitexact=True,
            actual_RGB_review={'basis':'existing_native_RGB_opened_without_model'}) for i in [0,1]])
    receipt_path = tmp_path / 'receipt.json'; receipt_ref=save(receipt_path,receipt)
    new = common._read(old_path)
    for i in [0,1]:
        new['rows'][i].update(curve_observation_kind='axis_scanline_branch', scanline_fixed_axis='y',
            scanline_branch='max_x', scanline_location_measurement=receipt_ref)
    new[api.REVISION_KEY] = dict(parent_packet_reference=old_ref, receipt_reference=receipt_ref,
        changed_row_indices=[0,1], changed_row_ids=['ice0','ice1'],
        all_original_source_rows_except_four_scanline_fields_unchanged=True)
    new_path = tmp_path / 'new_ice.json'; new_ref=save(new_path,new)
    board_ref = save(tmp_path / 'unchanged_board.json', {'board25': True})
    options = {api.ICE:str(old_path)}
    for key, index in ((common.NATIVE,2),(common.CONFIRMED,0)):
        path=tmp_path / (key.lstrip('-')+'.json')
        save(path,{'records':[dict(id=rows[index]['id'],family='native_ice_curve',packet_reference=old_ref,
            original_row_index=index,source_row_canonical_json_sha256=common._row_hash(rows[index]),
            uv=rows[index]['uv'],sigma_px=rows[index]['sigma_px'])]}); options[key]=str(path)
    arrays=tmp_path/'guard.npz';np.savez(arrays,uv=np.array(rows[2]['uv']))
    docs={
        common.GUARD:dict(primary_authority_reference=common.reference(options[common.NATIVE]),
            guard_arrays_reference=common.reference(arrays), records=[dict(id='ice2',
                authority_source_row_canonical_json_sha256=common._row_hash(rows[2]))]),
        common.CAP:dict(primary_authority_reference=common.reference(options[common.NATIVE]),
            full_trace_reference={},records=[dict(id='ice2',tangent_bounds=[-1.,1.])]),
        common.ENVELOPE:dict(authority_references={'native':common.reference(options[common.NATIVE])},
            records=[dict(id='ice2',normal_expansion_px=2.)]),
        common.INTEGRATION:dict(packets={'ice':old_ref,'board':board_ref},records=[dict(id='ice0',
            family='native_ice',source_row_reference=dict(old_ref,pointer='/rows/0'),packet_reference=old_ref,
            packet_row_index=0,original_row_canonical_json_sha256=common._row_hash(rows[0]),
            point_integration_weights=[.5,.5],stored_sigma_px=[4.,6.],working_sigma_px=[4.,6.])])}
    for key, value in docs.items():
        path=tmp_path/(key.lstrip('-')+'.json');save(path,value);options[key]=str(path)
    wide_ref=save(tmp_path/'wide.json',{'wide':True})
    ice_contract=dict(schema='rinkspace.additional-real-native-ice-curve-factor.v2',
        packet_path=old_ref['path'],packet_sha256=old_ref['sha256'],new_scalar_rows=8,
        observation_operator='original operator',trace_objective='original trace',held_rows_used_in_fit=False)
    nc=dict(authority=common.reference(options[common.NATIVE]),groups=1,
        full_guard_reference=common.reference(options[common.GUARD]),
        stroke_caps=common.reference(options[common.CAP]),envelope=common.reference(options[common.ENVELOPE]))
    cc=dict(authority=common.reference(options[common.CONFIRMED]),groups=1)
    ic=dict(policy_reference=common.reference(options[common.INTEGRATION]),
        source_packets={'native_ice':old_ref,'native_board':board_ref},groups=[{'budget':1.}])
    bc=dict(packet=board_ref,qualified_RGB_locations=25)
    manifest=dict(full_parameters=3,source_rows=12,additional_real_native_ice_curves=ice_contract,
        additional_real_native_board_edges=bc,highest_direct_human_hard_truth={'native':nc,'wide':{'authority':wide_ref}},
        highest_user_confirmed_native_sources=cc,native_base_curve_integration=ic,
        source_contract={'ice':ice_contract,'board':bc,'native':nc,'confirmed':cc,'integration':ic},
        source_jacobian_reuse={'enabled':False})
    manifest_path=tmp_path/'manifest.json';save(manifest_path,manifest)
    q=np.array([1.,2.,3.]);state_path=tmp_path/'state.npz';np.savez(state_path,q=q)
    packet=dict(schema='rinkspace.joint-source-truth-restoration-state.v1',
        complete_state_reference=common.reference(state_path),source_input_manifest_reference=common.reference(manifest_path),
        complete_q_sha256=hashlib.sha256(q.tobytes()).hexdigest(),source_target=common._target(manifest),
        families={k:dict(authority=ref,group_ids=[name],limits=[0. if k=='native' else .03],uncertainties=[.004])
            for k,ref,name in [('native',nc['authority'],'ice2'),('confirmed',cc['authority'],'ice0'),('wide',wide_ref,'wide0')]})
    truth=tmp_path/'truth.json';save(truth,packet)
    prepared=api.prepare_scanline_revision(options,new_path,tmp_path/'rebind',
        changed_row_indices=[0,1],truth_state_path=truth)
    derived=prepared['source_options']
    nc2=dict(nc,authority=common.reference(derived[common.NATIVE]),full_guard_reference=common.reference(derived[common.GUARD]),
        stroke_caps=common.reference(derived[common.CAP]),envelope=common.reference(derived[common.ENVELOPE]))
    cc2=dict(cc,authority=common.reference(derived[common.CONFIRMED]))
    ic2=dict(ic,policy_reference=common.reference(derived[common.INTEGRATION]),
        source_packets={'native_ice':new_ref,'native_board':board_ref})
    ice2=dict(ice_contract,packet_path=new_ref['path'],packet_sha256=new_ref['sha256'],
        observation_operator=api.OBSERVATION_OPERATOR,trace_objective=api.TRACE_OBJECTIVE,
        axis_scanline_location_objective=api.expected_scanline_contract([new['rows'][0],new['rows'][2],new['rows'][1]]))
    current=dict(manifest,additional_real_native_ice_curves=ice2,
        highest_direct_human_hard_truth={'native':nc2,'wide':{'authority':wide_ref}},
        highest_user_confirmed_native_sources=cc2,native_base_curve_integration=ic2,
        source_contract={'ice':ice2,'board':bc,'native':nc2,'confirmed':cc2,'integration':ic2})
    factors={k:SimpleNamespace(contract={'authority':rec['authority']},
        records=[{'record' if k=='wide' else 'row':{'id':rec['group_ids'][0]}}]) for k,rec in packet['families'].items()}
    factors['native'].contract['authority']=nc2['authority'];factors['confirmed'].contract['authority']=cc2['authority']
    return SimpleNamespace(old_path=old_path,new_path=new_path,receipt_path=receipt_path,
        truth=truth,prepared=prepared,manifest=current,factors=factors,q=q,packet=packet,options=options,board=bc)


def authenticate(case):
    return common.authenticate_location_revision(case.prepared['proof_reference']['path'],case.truth,
        case.q,case.manifest,case.factors)


def test_same_state_scanline_migration_preserves_all_truth_and_board(case):
    before=case.truth.read_bytes();result=authenticate(case)
    assert case.truth.read_bytes()==before
    assert result['migration']['protected_group_count']==3
    assert result['migration']['fresh_normal_and_J_required']
    assert case.manifest['additional_real_native_board_edges']==case.board
    assert common._read(case.old_path)['rows'][1]['split']=='holdout'
    assert common._read(case.new_path)['rows'][1]['split']=='holdout'
    for family,old in case.packet['families'].items():
        for key in ['group_ids','limits','uncertainties']:
            assert result['packet']['families'][family][key]==old[key]


def test_existing_driver_cli_loads_scanline_proof(case):
    from calibration.solver_runtime import install
    install()
    from run_current_named66_full_programme import load_truth_restoration_state
    result=load_truth_restoration_state(case.truth,case.q,case.manifest,case.factors,
        source_location_revision_proof=case.prepared['proof_reference']['path'])
    assert result['complete_q_sha256']==case.packet['complete_q_sha256']
    assert result['authenticated_source_location_revision']['fresh_normal_and_J_required']


@pytest.mark.parametrize('field,value',[('uv',[[2.1,3.],[4.,5.]]),('sigma_px',[3.,6.]),
    ('split','fit'),('pts',12),('source_search_support',[]),('scanline_branch','min_x')])
def test_scanline_revision_rejects_changed_original_support_or_role(case,field,value):
    new=common._read(case.new_path);new['rows'][1][field]=value;save(case.new_path,new)
    with pytest.raises(ValueError):api.validate_scanline_revision(case.old_path,case.new_path,[0,1])


@pytest.mark.parametrize('change',['q','reuse','board','caps','group','weight','qualification'])
def test_scanline_continuation_rejects_other_changes(case,change):
    if change=='q':case.q[0]+=.1
    elif change=='reuse':case.manifest['source_jacobian_reuse']['enabled']=True
    elif change=='board':case.manifest['source_contract']['board']['qualified_RGB_locations']=24
    elif change=='group':case.factors['confirmed'].records[0]['row']['id']='wrong'
    elif change=='qualification':case.manifest['source_contract']['ice']['source_only_review_qualification']={'removed':True}
    else:
        key=common.CAP if change=='caps' else common.INTEGRATION
        path=case.prepared['source_options'][key];value=common._read(path)
        value['records'][0]['tangent_bounds' if change=='caps' else 'point_integration_weights']=[0.,0.]
        save(path,value)
    with pytest.raises(ValueError):authenticate(case)


def test_mismatched_rgb_profile_receipt_rejected(case):
    receipt=common._read(case.receipt_path);receipt['rows'][0]['measured_source_row']['uv'][0][0]+=.1
    receipt_ref=save(case.receipt_path,receipt);new=common._read(case.new_path)
    new[api.REVISION_KEY]['receipt_reference']=receipt_ref
    for i in [0,1]:new['rows'][i]['scanline_location_measurement']=receipt_ref
    save(case.new_path,new)
    with pytest.raises(ValueError,match='original measured source row'):
        api.validate_scanline_revision(case.old_path,case.new_path,[0,1])


def test_original_packet_qualification_adapter_requires_proof(case):
    packet=common._read(case.new_path)
    assert api.original_qualified_packet(case.new_path,packet)==common._read(case.old_path)
    packet['rows'][1]['uv'][0][0]+=.1
    with pytest.raises(ValueError,match='Runtime scanline packet'):
        api.original_qualified_packet(case.new_path,packet)
