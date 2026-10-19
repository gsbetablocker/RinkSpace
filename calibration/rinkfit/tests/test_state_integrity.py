import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
from rinkfit_start.state_integrity import (STATE_FILES,ROW_FILES,StateIntegrityError,
    sha256_file,validate_report_files,validate_state_files,validate_prediction_arrays)


def write_json(path,value):
    path.write_text(json.dumps(value),encoding='utf8')


def ref(path):
    return dict(file=path.name,sha256=sha256_file(path))


@pytest.fixture
def bundle(tmp_path):
    n=3;active=np.array([True,True,False]);pts=np.array([100,100,200],'<i8')
    raw=np.array([[10.,20.],[30.,40.],[50.,60.]])
    error=np.array([[1.,0.],[0.,1.],[0.,0.]])
    z=dict(source_rows=np.arange(n),native_pts=pts.copy(),active_source_mask=active.copy(),
        uv=raw+error,stations=np.array([.2,.4,.6]),branches=np.zeros(n,'<u4'),
        root_rows=np.array([20.,41.,60.]),profile_alpha=np.zeros(n),error_native_uv=error,
        coverage_error_px=np.array([1.,1.,0.]),RS_valid=np.ones(n,bool))
    model=dict(blocks=[dict(name='g',values=[1.])],materials=[dict(label='m',paths=[dict(id='p')])])
    write_json(tmp_path/'model.json',model)
    for name,(dtype,key) in ROW_FILES.items():np.asarray(z[key],dtype).tofile(tmp_path/name)
    np.savez(tmp_path/'candidate_predictions.npz',**z)
    report=dict(files={name:sha256_file(tmp_path/name)for name in STATE_FILES})
    write_json(tmp_path/'report.json',report)
    kwargs=dict(n=n,native_pts=pts,active_mask=active,native_uv=raw,
        kind=np.array([0,2,0]),profile_normal=np.array([[0.,1.]]*n),
        material_index=np.zeros(n,dtype=int),model=model)
    return tmp_path,z,report,kwargs


def test_report_ledger_accepts_exact_six_files(bundle):
    state,_,report,_=bundle
    assert validate_report_files(state)==report


@pytest.mark.parametrize('name',STATE_FILES)
def test_report_ledger_rejects_changed_generated_file(bundle,name):
    state,_,_,_=bundle
    with (state/name).open('ab')as f:f.write(b'changed')
    with pytest.raises(StateIntegrityError,match='SHA'):validate_report_files(state)


@pytest.mark.parametrize('name',STATE_FILES)
def test_report_ledger_rejects_missing_producer_entry(bundle,name):
    state,_,report,_=bundle;del report['files'][name]
    with pytest.raises(StateIntegrityError,match='incomplete'):validate_report_files(state,report)


def test_ledger_descriptor_cannot_redirect_to_equal_bytes(bundle):
    state,_,report,_=bundle;other=state/'unrelated.json';other.write_bytes((state/'model.json').read_bytes())
    report['files']['model.json']=ref(other)
    with pytest.raises(StateIntegrityError,match='redirects'):validate_report_files(state,report)


def test_missing_report_is_not_implicitly_legacy(bundle):
    state,_,_,_=bundle;(state/'report.json').unlink()
    with pytest.raises(StateIntegrityError,match='legacy mode'):validate_state_files(state)
    with pytest.raises(StateIntegrityError,match='manifest'):validate_state_files(state,allow_legacy=True)


def legacy_manifest(state):
    keys={'station.f64':'station','path.u32':'branch','initial_row.f64':'row','initial_alpha.f64':'alpha'}
    m=dict(schema='rinkfit.complete_continuous_reduced_SLP_checkpoint.v1',
        model=ref(state/'model.json'),row_state_files={k:ref(state/name)for name,k in keys.items()},
        native_predictions=ref(state/'candidate_predictions.npz'))
    m['state_token']=hashlib.sha256(json.dumps(m,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return m


def test_explicit_legacy_requires_token_and_all_value_refs(bundle):
    state,_,_,_=bundle;(state/'report.json').unlink();m=legacy_manifest(state);write_json(state/'manifest.json',m)
    assert validate_state_files(state,allow_legacy=True)['authority']=='explicit_legacy_checkpoint'
    m['native_predictions']['sha256']='0'*64
    write_json(state/'manifest.json',m)
    with pytest.raises(StateIntegrityError,match='token'):validate_state_files(state,allow_legacy=True)


def test_resealed_legacy_manifest_still_checks_file_hash(bundle):
    state,_,_,_=bundle;(state/'report.json').unlink();m=legacy_manifest(state)
    m['model']['sha256']='0'*64;m.pop('state_token');m['state_token']=hashlib.sha256(json.dumps(m,sort_keys=True,separators=(',',':')).encode()).hexdigest();write_json(state/'manifest.json',m)
    with pytest.raises(StateIntegrityError,match='SHA'):validate_state_files(state,allow_legacy=True)


def test_prediction_arrays_accept_exact_saved_state(bundle):
    state,z,_,kwargs=bundle
    out=validate_prediction_arrays(state,**kwargs)
    for key in z:np.testing.assert_array_equal(out[key],z[key])


@pytest.mark.parametrize('case',['missing_uv','wrong_shape','row_order','PTS','scope','invalid_RS',
    'nonfinite_uv','residual','coverage','station_domain','noninteger_path','unknown_path'])
def test_prediction_arrays_reject_internal_or_source_mismatch(bundle,case):
    state,z,_,kwargs=bundle
    if case=='missing_uv':z.pop('uv')
    elif case=='wrong_shape':z['uv']=z['uv'].ravel()
    elif case=='row_order':z['source_rows']=np.array([1,0,2])
    elif case=='PTS':z['native_pts'][0]+=1
    elif case=='scope':z['active_source_mask'][0]=False
    elif case=='invalid_RS':z['RS_valid'][0]=False
    elif case=='nonfinite_uv':z['uv'][0,0]=np.nan
    elif case=='residual':z['error_native_uv'][0,0]+=1
    elif case=='coverage':z['coverage_error_px'][0]+=1
    elif case=='station_domain':z['stations'][0]=1.1;z['stations'].astype('<f8').tofile(state/'station.f64')
    elif case=='noninteger_path':z['branches']=np.zeros(3,float)
    elif case=='unknown_path':z['branches'][0]=1;z['branches'].astype('<u4').tofile(state/'path.u32')
    with pytest.raises(StateIntegrityError):validate_prediction_arrays(state,predictions=z,**kwargs)


@pytest.mark.parametrize('name',tuple(ROW_FILES))
@pytest.mark.parametrize('case',['truncated','wrong_values'])
def test_four_auxiliaries_require_exact_size_and_npz_values(bundle,name,case):
    state,z,_,kwargs=bundle
    if case=='truncated':(state/name).write_bytes(b'')
    else:
        dtype,key=ROW_FILES[name];value=z[key].copy();value[0]+=1;value.astype(dtype).tofile(state/name)
    with pytest.raises(StateIntegrityError,match='Auxiliary'):validate_prediction_arrays(state,**kwargs)


def test_loaded_mapping_can_be_reused_without_reopening_npz(bundle,monkeypatch):
    state,z,_,kwargs=bundle
    def unexpected(*a,**kw):raise AssertionError('NPZ was opened twice')
    monkeypatch.setattr(np,'load',unexpected)
    assert validate_prediction_arrays(state,predictions=z,**kwargs)['uv'] is z['uv']


def receipt_for_bundle(state,report):
    from rinkfit_start.state_integrity import validate_receipt_bindings
    packet=state/'packet.json';semantic=state/'semantic.json';write_json(packet,{});write_json(semantic,{})
    report.update(packet=ref(packet),semantic=ref(semantic),raw_rows=3,active_rows=2,frames=2,
        best=dict(locally_captured_ordinary_candidates=1,active_RS_valid_rows=2,query_numerical_roots=1),
        whole_static_scene=dict(passed=True))
    model=json.loads((state/'model.json').read_text());model['source_identity']='test-source'
    # validate_receipt_bindings reuses an already-validated mapping; these
    # miniature metadata stand in for that caller's full candidate validation.
    receipt=dict(schema='rinkfit.complete_recomputed_native_candidate.v1',packet=ref(packet),semantic=ref(semantic),
        model=ref(state/'model.json'),predictions=ref(state/'candidate_predictions.npz'),
        row_state_files={name:ref(state/name)for name in ROW_FILES},source_rows=3,active_rows=2,all_frames=2,
        all_ordinary_stations_recomputed=1,true_RS_valid_rows=2,profile_new_numerical_roots=1,
        metrics=report['best'].copy(),physical_guard=report['whole_static_scene'].copy(),
        source_identity='test-source',complete_state_for_fresh_relinearization=True,physical_fit_accepted=False)
    return receipt,dict(report=report,model=model)


def test_receipt_matches_already_validated_bundle(bundle):
    from rinkfit_start.state_integrity import validate_receipt_bindings
    state,_,report,_=bundle;receipt,validated=receipt_for_bundle(state,report)
    assert validate_receipt_bindings(receipt,state,validated=validated) is validated


@pytest.mark.parametrize('case',['model_ref','counts','metrics','scene','source','acceptance','missing_aux'])
def test_receipt_rejects_relabeling_validated_values(bundle,case):
    from rinkfit_start.state_integrity import validate_receipt_bindings
    state,_,report,_=bundle;receipt,validated=receipt_for_bundle(state,report)
    if case=='model_ref':receipt['model']['sha256']='0'*64
    elif case=='counts':receipt['all_ordinary_stations_recomputed']=3
    elif case=='metrics':receipt['metrics']['locally_captured_ordinary_candidates']=3
    elif case=='scene':receipt['physical_guard']['passed']=False
    elif case=='source':receipt['source_identity']='wrong-source'
    elif case=='acceptance':receipt['physical_fit_accepted']=True
    else:receipt['row_state_files'].pop('station.f64')
    with pytest.raises(StateIntegrityError):validate_receipt_bindings(receipt,state,validated=validated)
