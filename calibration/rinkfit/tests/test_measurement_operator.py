"""Production Python/native guard regressions; these do not assert fit quality."""
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from rinkfit_joint.io_utils import file_sha256, object_sha256
from rinkfit_joint.measurement_operator import (
    canonical_tree_sha256, measurement_operator_binding,
    assert_measurement_operator_binding, assert_append_operator_prefix)
from rinkfit_joint.board_source_qualification_guard import (
    assert_board_material_source_qualification, BOUND_ARRAYS)
from rinkfit_joint.qualified_scope import QUALIFICATION_DIMENSIONS, SOURCE_SHA, DOMAIN


def ref(path):
    return dict(file=str(path), sha256=file_sha256(path))


def write(path, obj):
    path.write_text(json.dumps(obj), encoding='utf8')
    return ref(path)


def fixture(tmp_path):
    arrays = {}
    values = dict(uv=([[100.,100.]],'<f8'), native_pts=([621312],'<i8'),
        shot_index=([1],'<u2'), material_index=([0],'<u4'), kind=([0],'|u1'),
        admission=([1],'|u1'), fit_role=([1],'|u1'), profile_normal=([[0.,1.]],'<f8'),
        station=([.5],'<f8'), row=([100.],'<f8'), alpha=([0.],'<f8'))
    for key, (value, dtype) in values.items():
        a = np.asarray(value, dtype=dtype)
        path = tmp_path/(key+'.bin')
        a.tofile(path)
        arrays[key] = dict(**ref(path), dtype=a.dtype.str, shape=list(a.shape))
    ids, datasets = ['outer_native/s02/621312/0'], ['actual_geometry:test:s02']
    evidence = write(tmp_path/'evidence.json', {'fixture': True})
    record = dict(global_row=0, source_row_id=ids[0], source_dataset_id=datasets[0],
        native_pts=621312, current_uv=[100.,100.], status='pass', operator_status='pass',
        material_index=0, original_profile_normal=[0.,1.],
        qualification_dimensions={k:'pass' for k in QUALIFICATION_DIMENSIONS},
        actual_readout_alias_of=None, source_video_sha256=SOURCE_SHA,
        time_base='1/12800', coordinate_domain=DOMAIN,
        measurement_operator='normal_profile', evidence_refs=[evidence])
    record['current_measurement_sha256'] = object_sha256(dict(native_pts=621312,
        current_uv=[100.,100.], source_row_id=ids[0], measurement_operator='normal_profile'))
    ledger = tmp_path/'rows.jsonl'
    ledger.write_text(json.dumps(record)+'\n', encoding='utf8')
    index = tmp_path/'indices.bin'
    np.asarray([0],'<i8').tofile(index)
    recipe = write(tmp_path/'recipe.json', dict(version='ordered_same_board_material_roles_v1',
        public_function='rinkfit_joint.board_source_requalification.reevaluate_current_board_source_rows',
        frozen_module_refs=[evidence], counterexample_QA_ref=write(tmp_path/'qa.json',dict(passed=True)),
        historical_executed_recipe_ref=evidence))
    packet = dict(row_count=1, arrays=arrays, source_row_ids=ids, source_dataset_ids=datasets,
        allowed_paths={'0':[1]}, source_station_domains=dict(schema='rinkfit.source_row_station_domains.v1',
            source_refs=[evidence], rows=[dict(row_index=0,source_row_id=ids[0],path_index=1,allowed_station_intervals=[[.2,.7]])]),
        queries=[dict(row_index=0,alpha=[-3.,3.],native_uv=[[100.,97.],[100.,103.]],normal=[0.,1.],
            numerator_x=[100.,0.],numerator_y=[100.,1.],denominator=[1.],alpha_origin=0.,alpha_scale=1.)])
    cert = dict(schema='rinkfit.board_material_fit_qualification.v2_operator_bound',
        qualification_dimensions=list(QUALIFICATION_DIMENSIONS), source_video_sha256=SOURCE_SHA,
        time_base='1/12800', coordinate_domain=DOMAIN, exact_ID_scope_complete=True,
        whole_original_scope_qualified=False, consumer_module_ref=evidence, count=1,
        role_and_operator_pass_count=1, unresolved_or_wrong_rows_consumed=0,
        active_board_rows_sha256=file_sha256(index),
        membership_array_sha256={k:arrays[k]['sha256'] for k in BOUND_ARRAYS},
        recipe_ref=recipe, source_scope_ref=evidence, producer_module_is_historical_executed_archive=True,
        source_row_ids_sha256=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest(),
        source_dataset_ids_sha256=hashlib.sha256(json.dumps(datasets,separators=(',',':')).encode()).hexdigest(),
        qualified_active_board_rows_ref=ref(index), qualified_rows_ref=ref(ledger),
        qualified_rows_native_ref=ref(ledger), measurement_operator_binding=measurement_operator_binding(packet))
    packet['board_material_source_qualification'] = write(tmp_path/'certificate.json',cert)
    return packet, cert


def check(packet, path, expected):
    write(path, packet)
    if expected:
        assert_board_material_source_qualification(packet, path)
    else:
        with pytest.raises(ValueError):
            assert_board_material_source_qualification(packet, path)
    exe = os.environ.get('RINKFIT_OPERATOR_GUARD_EXE')
    if exe:
        run = subprocess.run([exe,'guard',str(path)],capture_output=True,text=True)
        assert (run.returncode == 0) == expected, run.stdout+run.stderr


def test_original_and_legal_auxiliary_continuation(tmp_path):
    packet, _ = fixture(tmp_path)
    check(packet,tmp_path/'packet.json',True)
    for name, value in [('station',.6),('row',101.),('alpha',.2)]:
        changed = copy.deepcopy(packet)
        a = np.asarray([value],'<f8'); path=tmp_path/('new_'+name+'.bin');a.tofile(path)
        changed['arrays'][name] = dict(**ref(path),dtype='<f8',shape=[1])
        changed['queries'][0]['alpha0'] = .2
        check(changed,tmp_path/(name+'.json'),True)


@pytest.mark.parametrize('change', ['window','samples','coefficients','query_normal','profile_normal',
    'query_removed','query_added','finite_paths','station_domain','source_id'])
def test_operator_changes_rejected_by_same_certificate(tmp_path, change):
    packet, _ = fixture(tmp_path)
    if change == 'window':packet['queries'][0]['alpha']=[-300.,300.]
    if change == 'samples':packet['queries'][0]['native_uv'][0][1]-=100.
    if change == 'coefficients':packet['queries'][0]['numerator_x'][1]=1.
    if change == 'query_normal':packet['queries'][0]['normal']=[1.,0.]
    if change == 'profile_normal':packet['arrays']['profile_normal']['sha256']='0'*64
    if change == 'query_removed':packet['queries']=[]
    if change == 'query_added':
        cert=json.loads(Path(packet['board_material_source_qualification']['file']).read_text())
        base=copy.deepcopy(packet);base['queries']=[]
        cert['measurement_operator_binding']=measurement_operator_binding(base)
        packet['board_material_source_qualification']=write(tmp_path/'no_query_certificate.json',cert)
    if change == 'finite_paths':packet['allowed_paths']['0']=[0,1]
    if change == 'station_domain':packet['source_station_domains']['rows'][0]['allowed_station_intervals']=[[0.,1.]]
    if change == 'source_id':packet['source_row_ids'][0]+='/different'
    check(packet,tmp_path/'changed.json',False)


def test_unbound_legacy_certificate_requires_explicit_revision(tmp_path):
    packet, cert = fixture(tmp_path)
    cert.pop('measurement_operator_binding')
    cert['schema']='rinkfit.board_material_fit_qualification.v2'
    packet['board_material_source_qualification']=write(tmp_path/'legacy.json',cert)
    check(packet,tmp_path/'legacy_packet.json',False)


def test_missing_path_authority_is_not_all_paths(tmp_path):
    packet, cert = fixture(tmp_path)
    packet['allowed_paths']={}
    cert['measurement_operator_binding']=measurement_operator_binding(packet)
    packet['board_material_source_qualification']=write(tmp_path/'empty_paths.json',cert)
    check(packet,tmp_path/'packet.json',False)


def test_canonical_tree_native_parity_and_relocation(tmp_path):
    packet, _ = fixture(tmp_path)
    moved=copy.deepcopy(packet)
    moved['arrays']['uv']['file']='moved/uv.bin'
    assert measurement_operator_binding(packet)==measurement_operator_binding(moved)
    exe = os.environ.get('RINKFIT_OPERATOR_GUARD_EXE')
    if not exe:pytest.skip('native guard executable supplied by focused build')
    trees=[None,True,-0.,1.23456789012345,2**54,-2**63,{'中文':[1,1.,False,None], 'x':{'z':[]}}]
    for i, value in enumerate(trees):
        path=tmp_path/('tree'+str(i)+'.json');write(path,value)
        run=subprocess.run([exe,'tree',str(path)],capture_output=True,text=True,check=True)
        assert run.stdout == canonical_tree_sha256(value)
    path=tmp_path/'packet.json';write(path,packet)
    run=subprocess.run([exe,'binding',str(path)],capture_output=True,text=True,check=True)
    assert json.loads(run.stdout)==measurement_operator_binding(packet)


def test_prefix_refuses_query_or_domain_rewrite(tmp_path):
    parent,_=fixture(tmp_path)
    result=copy.deepcopy(parent)
    assert_append_operator_prefix(parent,result,1)
    result['queries'][0]['alpha']=[-300.,300.]
    with pytest.raises(ValueError,match='Query'):assert_append_operator_prefix(parent,result,1)
    result=copy.deepcopy(parent);result['allowed_paths']={}
    with pytest.raises(ValueError,match='path'):assert_append_operator_prefix(parent,result,1)


def test_inexact_json_integer_is_rejected_in_both_languages(tmp_path):
    exe=os.environ.get('RINKFIT_OPERATOR_GUARD_EXE')
    for value in (2**53+1, 2**63-1, 2**64-1):
        with pytest.raises(ValueError):canonical_tree_sha256(value)
        if exe:
            path=tmp_path/'inexact.json';write(path,value)
            run=subprocess.run([exe,'tree',str(path)],capture_output=True,text=True)
            assert run.returncode != 0


def test_dense_prefix_hydration_and_raw_registry_have_same_identity(tmp_path):
    packet,_=fixture(tmp_path)
    packet['dense_source_binding']=dict(row_start=1,count=1,
        source_id_utf8=dict(file='ids.bin',sha256='b'*64,dtype='|u1',shape=[3]))
    packet['row_count']=2
    raw=copy.deepcopy(packet)
    packet['source_row_ids']+=['new']
    packet['source_dataset_ids']+=['dense_source']
    assert measurement_operator_binding(raw)==measurement_operator_binding(packet)
    exe=os.environ.get('RINKFIT_OPERATOR_GUARD_EXE')
    if exe:
        path=tmp_path/'raw.json';write(path,raw)
        run=subprocess.run([exe,'binding',str(path)],capture_output=True,text=True,check=True)
        assert json.loads(run.stdout)==measurement_operator_binding(packet)
