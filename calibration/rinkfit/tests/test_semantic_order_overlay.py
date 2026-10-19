import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
from rinkfit_start import semantic_order_overlay as order
from rinkfit_start.runtime import atomic_json


def descriptor(path):
    return {'file':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


@pytest.fixture
def overlay(tmp_path,monkeypatch):
    packet=tmp_path/'source.json'
    feature={'id':'print','current_source_only_readout':True,
        'model_or_template_generated_observation':False,
        'world_correspondence_status':'named_same_physical_print_component',
        'segments':[{'measurement_origin':'current_native_material_normal_half_transition',
                     'pixels_measured_in_this_frame':True,'uv':[[0,0],[1,0]]}]}
    atomic_json(packet,{'features':[feature]})
    raw=tmp_path/'raw.jsonl'
    raw.write_text(json.dumps({'source_packet':dict(path=str(packet),sha256=descriptor(packet)['sha256']),
        'source_feature_id':'print','source_segment_index':0,'source_vertex_indices':[0,1],
        'source_rows':[0,1]})+'\n',encoding='utf-8')
    query=tmp_path/'query.f64';np.zeros((2,2),dtype='<f8').tofile(query)
    weights=tmp_path/'weights.f64';np.ones(4,dtype='<f8').tofile(weights)
    pm={'observations':2,'arrays':{'query_uv':descriptor(query),'weights':descriptor(weights)},
        'assets':{'source_runs.jsonl':descriptor(raw)},'statistics':{'formal_residual_rows':4},
        'immutable_source_arrays':{'source_uv':'original-five-array-hash'}}
    parent=tmp_path/'parent.json';atomic_json(parent,pm)
    parent_hash=descriptor(parent)['sha256'];monkeypatch.setattr(order,'PARENT',parent_hash)
    fact=tmp_path/'fact';fact.write_bytes(b'reviewed producer and fact')
    d=dict(descriptor(fact),role='fixture')
    monkeypatch.setattr(order,'PRODUCERS',{'fixture':d['sha256']})
    monkeypatch.setattr(order,'FACTS',{'fixture':d['sha256']})
    classifications,rows,packets=order.classify_runs(raw,2)
    ledger=tmp_path/'classifications.jsonl';ledger.write_text(json.dumps(classifications[0])+'\n')
    hard=tmp_path/'hard.jsonl';hard.write_bytes(b'')
    proof={'schema':order.SCHEMA,'rule_id':order.RULE,'parent_semantic_sha256':parent_hash,
        'observations':2,'audit_source_runs':descriptor(raw),'classifications':descriptor(ledger),
        'active_hard_source_runs':descriptor(hard),'audit_run_count':1,'audit_row_count':2,
        'verified_source_packet_count':1,'active_hard_run_count':0,'active_hard_row_count':0,
        'producer_evidence':[d],'evidence':[d],'raw_arrays_W_E_and_authorities_unchanged':True}
    proof_path=tmp_path/'proof.json';atomic_json(proof_path,proof)
    m=copy.deepcopy(pm);m['assets']['source_runs.jsonl']=descriptor(hard)
    m.update(source_order_parent_manifest={'path':str(parent),'sha256':parent_hash},
             source_order_overlay=descriptor(proof_path))
    def parent_loader(path,n,expected_sha256):
        assert Path(path)==parent and n==2 and expected_sha256==parent_hash
        return pm,{},parent,{'enabled':True,'query_semantic_sha256':'query-original',
                             'source_exclusion':{'enabled':True,'active_observations':1}}
    return tmp_path,m,proof,proof_path,parent_loader


def load(fixture):
    root,m,_,_,loader=fixture
    return order.load_order_overlay(root/'manifest.json',m,2,loader)


def rewrite_proof(fixture):
    _,m,proof,path,_=fixture;atomic_json(path,proof);m['source_order_overlay']=descriptor(path)


def test_navigation_keeps_original_ledger_and_authority(overlay):
    m,arrays,path,info=load(overlay)
    assert info['enabled'] and info['query_semantic_sha256']=='query-original'
    assert info['source_exclusion']['active_observations']==1
    assert info['source_order_overlay']['audit_run_count']==1
    assert info['source_order_overlay']['active_hard_run_count']==0


@pytest.mark.parametrize('array',['query_uv','weights'])
def test_rejects_uv_or_W_identity_change(overlay,array):
    overlay[1]['arrays'][array]['sha256']='0'*64
    with pytest.raises(ValueError,match='parent semantic identity'):load(overlay)


def test_rejects_deleted_original_navigation_audit(overlay):
    overlay[2]['audit_source_runs']=overlay[2]['active_hard_source_runs']
    rewrite_proof(overlay)
    with pytest.raises(ValueError,match='audit mismatch'):load(overlay)


def test_rejects_wrong_parent(overlay):
    overlay[1]['source_order_parent_manifest']['sha256']='0'*64
    with pytest.raises(ValueError,match='wrong reviewed parent'):load(overlay)


def test_rejects_missing_run_classification(overlay):
    root,m,proof,_,_=overlay;path=root/'missing.jsonl';path.write_bytes(b'')
    proof['classifications']=descriptor(path);rewrite_proof(overlay)
    with pytest.raises(ValueError,match='classification'):load(overlay)


def test_rejects_fabricated_trace_rule(overlay):
    overlay[2]['rule_id']='all_points_are_real_boundary_traces';rewrite_proof(overlay)
    with pytest.raises(ValueError,match='authority'):load(overlay)


def test_actual_trace_evidence_cannot_be_reclassified(overlay):
    root=overlay[0];packet=root/'source.json';data=json.loads(packet.read_text())
    data['features'][0]['segments'][0]['connected_boundary_trace']={'verified':True}
    atomic_json(packet,data);raw=root/'raw.jsonl';run=json.loads(raw.read_text())
    run['source_packet']['sha256']=descriptor(packet)['sha256'];raw.write_text(json.dumps(run)+'\n')
    with pytest.raises(ValueError,match='trace obligations'):order.classify_runs(raw,2)


def test_zero_hard_seed_retains_navigation_audit_without_capture_comparisons(tmp_path):
    from rinkfit_start.semantic_capture_constraints import export_run_constraint_seed
    model=tmp_path/'model.json';atomic_json(model,{'materials':[]})
    np.zeros(2,dtype='<f8').tofile(tmp_path/'station.f64')
    np.zeros(2,dtype='<u4').tofile(tmp_path/'path.u32')
    atomic_json(tmp_path/'source_order_after.json',{
        'passed':True,'failed_runs':0,'records':[],'active_hard_source_runs':0,
        'semantic_manifest_sha256':'sem','source_runs_sha256':'empty',
        'source_order_overlay_sha256':'proof','audit_source_runs_retained':15419,
        'navigation_query_sequence_only_runs':15419,'navigation_query_sequence_only_rows':54948,
        'navigation_audit_registry':{'sha256':'original'},'navigation_classifications':{'sha256':'all'}})
    result=export_run_constraint_seed(model,tmp_path)
    assert result['runs']==0 and result['audit_source_runs_retained']==15419
    assert result['original_navigation_order_verified'] is False
    assert result['all_original_navigation_runs_are_real_traces'] is False


def test_relocation_must_really_have_original_bytes(overlay):
    root,m,_,_,_=overlay;changed=root/'changed.f64';np.ones((2,2),dtype='<f8').tofile(changed)
    m['arrays']['query_uv']['file']=str(changed)
    with pytest.raises(ValueError,match='checksum changed'):load(overlay)
