from copy import deepcopy
import hashlib
import json
from pathlib import Path
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from calibration.solver_runtime import original_measured_transition_locations as locations
from calibration.solver_runtime import truth_original_location_revision as proof_api
from calibration.solver_runtime import truth_source_location_revision as common


def save(path,value):
    path=Path(path);path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    return common.reference(path)


@pytest.fixture
def case(tmp_path):
    row=dict(id='F2.inner',pts=100,split='fit',origin='original-only source',
        uv=np.array([[10.,20.],[30.,40.]]),sigma=np.array([2.,3.]),
        normal=np.array([[0.,1.],[0.,1.]]),normal_interval=np.array([[-5.,5.],[-6.,6.]]),
        world=np.array([[0.,0.,0.],[1.,0.,0.]]),physical_curve_id='F2.inner',
        must_preserve_upper_branch=False,point_correspondence=False)
    row['samples']=[dict(source_uv=point.copy(),source_gradient_sign=1.) for point in row['uv']]
    item=dict(block='low',collection_role='FIT',group_actual_pts=100,group_order=0,row_order=0,source_row=row)
    parent=tmp_path/'source.pkl';parent.write_bytes(pickle.dumps({'programme_absolute_rows':[item]}))
    source=tmp_path/'original_measurements.json'
    source_ref=save(source,{'samples':[{'uv':[10.,20.],'q50':20.},{'uv':[30.,40.],'q50':40.}]})
    record=dict(original_packet_row_index=0,locator={k:item[k] for k in locations.LOCATOR_KEYS},
        source_row_id=row['id'],original_source_row_sha256=hashlib.sha256(locations.canonical(row)).hexdigest(),
        point_indices=[0,1],measured_uv=row['uv'].tolist(),estimator_kind='original_gray_gradient_mass_q50',
        source_reference=source_ref,measurement_bindings=[dict(point_index=i,measured_axis=1,
            location_path=['samples',i,'q50'],observed_uv_path=['samples',i,'uv']) for i in [0,1]])
    packet_path=tmp_path/'locations.json';save(packet_path,dict(schema=locations.SCHEMA,
        original_absolute_packet=common.reference(parent),records=[record]))
    _,_,contract=locations.validate_packet(packet_path)
    native_ref=save(tmp_path/'native_authority.json',{'original':True})
    wide_ref=save(tmp_path/'wide_authority.json',{'original':True})
    old=dict(full_parameters=3,source_rows=8,source_contract={'original_rows':8},
        highest_direct_human_hard_truth={'native':{'authority':native_ref,'full_stroke_guard':{'fixed':True}},
            'wide':{'authority':wide_ref,'practical_envelope':{'fixed':True}}},
        highest_user_confirmed_native_sources={'support':{'fixed':True}},
        native_base_curve_integration={'policy':{'fixed':True}},source_jacobian_reuse={'enabled':False})
    manifest_path=tmp_path/'manifest.json';save(manifest_path,old)
    q=np.array([1.,2.,3.]);state_path=tmp_path/'state.npz';np.savez(state_path,q=q)
    state=dict(complete_state_reference=common.reference(state_path),source_input_manifest_reference=common.reference(manifest_path),
        complete_q_sha256=hashlib.sha256(q.tobytes()).hexdigest(),source_target=common._target(old),
        families={'native':dict(authority=native_ref,group_ids=['fixed native'],limits=[0.],uncertainties=[.004]),
            'wide':dict(authority=wide_ref,group_ids=['fixed wide'],limits=[.03],uncertainties=[.004])})
    truth=tmp_path/'truth.json';save(truth,state)
    prepared=proof_api.prepare_original_location_revision(truth,{},packet_path,tmp_path/'proof')
    current=deepcopy(old);current[locations.CONTRACT_KEY]=contract
    current['source_contract'][locations.CONTRACT_KEY]=contract
    factors={k:SimpleNamespace(contract={'authority':r['authority']},
        records=[{'record' if k=='wide' else 'row':{'id':r['group_ids'][0]}}]) for k,r in state['families'].items()}
    runtime=deepcopy(row);block=SimpleNamespace(name='low',groups=[(100,[runtime])],bundle=SimpleNamespace())
    runtime_state=SimpleNamespace(blocks=[block],source_actors=None)
    return SimpleNamespace(packet_path=packet_path,prepared=prepared,truth=truth,q=q,manifest=current,
        factors=factors,runtime_state=runtime_state,row=row,old=old)


def authenticate(case):
    return proof_api.authenticate_original_location_revision(case.prepared['proof_reference']['path'],
        case.truth,case.q,case.manifest,case.factors)


def test_original_only_revision_preserves_truth_support(case):
    before=case.truth.read_bytes();result=authenticate(case)
    assert result['migration']['protected_group_count']==2
    assert case.truth.read_bytes()==before
    for family in result['packet']['families']:
        assert result['packet']['families'][family]['limits']==([0.] if family=='native' else [.03])


@pytest.mark.parametrize('key',['highest_direct_human_hard_truth',
    'highest_user_confirmed_native_sources','native_base_curve_integration'])
def test_original_only_proof_cannot_hide_changed_truth_support_or_loss(case,key):
    case.manifest[key]['silently_changed_support']=True
    with pytest.raises(ValueError,match='protected support or integration'):authenticate(case)


@pytest.mark.parametrize('field,value',[('world',np.array([[0.,0.,0.],[2.,0.,0.]])),
    ('physical_curve_id','F2.outer'),('must_preserve_upper_branch',True),('point_correspondence',True)])
def test_installer_cannot_bind_equal_pixels_to_changed_physical_or_raw_operator(case,field,value):
    block=case.runtime_state.blocks[0];old_groups=block.groups
    block.groups[0][1][0][field]=value
    with pytest.raises(ValueError,match='original saved measured observation'):
        locations.install_original_measured_transition_locations(case.runtime_state,case.packet_path)
    assert block.groups is old_groups
    assert 'original_boundary_location_observation' not in block.groups[0][1][0]


def test_installer_preserves_all_original_fields_and_both_group_routes(case):
    contract=locations.install_original_measured_transition_locations(case.runtime_state,case.packet_path)
    block=case.runtime_state.blocks[0];new=block.groups[0][1][0]
    assert block.bundle.abs_groups is block.groups
    for key,value in case.row.items():assert locations.canonical(new[key])==locations.canonical(value)
    assert contract==case.manifest[locations.CONTRACT_KEY]


def test_archived_array_sample_UV_accepts_unchanged_JSON_fixed_coordinate_ints(case):
    block=case.runtime_state.blocks[0]
    for sample in block.groups[0][1][0]['samples']:
        sample['source_uv']=sample['source_uv'].tolist();sample['source_uv'][1]=int(sample['source_uv'][1])
    before=case.packet_path.read_bytes()
    assert locations.runtime_source_differences(block.groups[0][1][0],case.row)==[]
    locations.install_original_measured_transition_locations(case.runtime_state,case.packet_path)
    assert case.packet_path.read_bytes()==before


@pytest.mark.parametrize('bad',['one_ulp','bool','string','shape','new_key'])
def test_archived_sample_representation_bridge_keeps_every_numeric_leaf_exact(case,bad):
    row=case.runtime_state.blocks[0].groups[0][1][0]
    sample=row['samples'][0]
    sample['source_uv']=sample['source_uv'].tolist()
    if bad=='one_ulp':sample['source_uv'][1]=np.nextafter(sample['source_uv'][1],np.inf)
    elif bad=='bool':sample['source_gradient_sign']=True
    elif bad=='string':sample['source_uv'][1]=str(sample['source_uv'][1])
    elif bad=='shape':sample['source_uv']=[sample['source_uv']]
    else:sample['new_metadata']=True
    with pytest.raises(ValueError,match='original saved measured observation'):
        locations.install_original_measured_transition_locations(case.runtime_state,case.packet_path)
