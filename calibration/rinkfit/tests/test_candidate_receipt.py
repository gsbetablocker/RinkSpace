"""Reject inconsistent complete-state receipts even after a file is rehashed."""
import json
from pathlib import Path
import numpy as np
import pytest
from rinkfit_start.candidate_receipt import _validate_outputs,_validate_lineage,sha

@pytest.fixture
def bundle(tmp_path):
 d=tmp_path/'state';d.mkdir()
 def arr(name,value):
  value=np.asarray(value);value.tofile(d/name);return dict(file=str(d/name),sha256=sha(d/name),dtype=value.dtype.str,shape=list(value.shape))
 def put(name,value):
  p=d/name;p.write_text(json.dumps(value),encoding='utf-8');return p
 uv=np.array([[10.,20.],[30.,40.]]);pts=np.array([100,100],dtype='<i8');pred=uv+[[1.,.5],[0.,1.]];error=pred-uv;coverage=np.array([np.sqrt(1.25),1.]);active=np.ones(2,bool)
 packet=dict(observations=2,arrays=dict(native_uv=arr('uv.f64',uv),native_pts=arr('pts.i64',pts),source_rows=arr('rows.u64',np.arange(2,dtype='<u8')),engineering_budget_px=arr('budget.f64',np.full(2,2.))))
 table=put('string_tables.json',{});packet['string_tables']=dict(file=str(table),sha256=sha(table));PK=put('packet.json',packet)
 sem=dict(active_source_mask=arr('active.u8',np.ones(2,dtype='u1')),arrays=dict(kind=arr('kind.u8',np.array([0,2],dtype='u1')),profile_normal=arr('normal.f64',np.array([[0.,1.],[0.,1.]]))));SP=put('semantic.json',sem)
 a=dict(source_rows=np.arange(2),native_pts=pts,active_source_mask=active,uv=pred,stations=np.array([.3,.4]),branches=np.zeros(2,dtype='<u4'),root_rows=pred[:,1],profile_alpha=np.array([0.,.2]),error_native_uv=error,coverage_error_px=coverage,RS_valid=active)
 np.savez(d/'candidate_predictions.npz',**a)
 for file,key,dtype in [('station.f64','stations','<f8'),('path.u32','branches','<u4'),('initial_row.f64','root_rows','<f8'),('initial_alpha.f64','profile_alpha','<f8')]:np.asarray(a[key],dtype=dtype).tofile(d/file)
 put('model.json',{});report=dict(raw_rows=2,active_rows=2,source_W_E_unchanged=True,partition_exactly_once=True,whole_static_scene=dict(passed=True),best=dict(valid=True,active_RS_valid_rows=2,locally_captured_ordinary_candidates=1,inactive_under_current_worst_bound_rows=0,query_numerical_roots=1,max_q=float((coverage**2/4).max()),maximum_error_px=float(coverage.max()),quantiles_px=np.quantile(coverage,[.5,.95,.99,1]).tolist(),over2=0),files={name:sha(d/name)for name in ['model.json','candidate_predictions.npz','station.f64','path.u32','initial_row.f64','initial_alpha.f64']})
 return d,report,packet,sem,PK,SP,a

def check(b):return _validate_outputs(*b[:6])
def save_predictions(b,arrays):
 d,report,*_=b;np.savez(d/'candidate_predictions.npz',**arrays);report['files']['candidate_predictions.npz']=sha(d/'candidate_predictions.npz')

def test_complete_saved_arrays_pass(bundle):
 _,active,_,ordinary,profile=check(bundle)
 assert active.sum()==2 and ordinary==profile==1

def test_stale_report_hash_rejected(bundle):
 (bundle[0]/'station.f64').write_bytes(b'')
 with pytest.raises(ValueError,match='SHA mismatch'):check(bundle)

def test_missing_projection_even_rehashed_rejected(bundle):
 a=dict(bundle[-1]);del a['uv'];save_predictions(bundle,a)
 with pytest.raises(ValueError,match='key/shape'):check(bundle)

def test_zero_length_aux_even_rehashed_rejected(bundle):
 d,report,*_=bundle;(d/'path.u32').write_bytes(b'');report['files']['path.u32']=sha(d/'path.u32')
 with pytest.raises(ValueError,match='Auxiliary length'):check(bundle)

def test_aux_projection_mismatch_even_rehashed_rejected(bundle):
 d,report,*_=bundle;np.array([.7,.4],dtype='<f8').tofile(d/'station.f64');report['files']['station.f64']=sha(d/'station.f64')
 with pytest.raises(ValueError,match='Auxiliary contents'):check(bundle)

def test_incomplete_actual_station_recompute_rejected(bundle):
 bundle[1]['best']['locally_captured_ordinary_candidates']=0
 with pytest.raises(ValueError,match='actually recomputed'):check(bundle)

def test_invalid_active_rs_even_rehashed_rejected(bundle):
 a=dict(bundle[-1]);a['RS_valid']=np.array([False,True]);save_predictions(bundle,a)
 with pytest.raises(ValueError,match='validity'):check(bundle)

def test_false_scene_guard_rejected(bundle):
 bundle[1]['whole_static_scene']['passed']=False
 with pytest.raises(ValueError,match='scene guard'):check(bundle)

def test_unrelated_parent_or_model_rejected(bundle):
 d,report,packet,sem,PK,SP,_=bundle;p=d/'irrelevant.json';p.write_text('{}',encoding='utf8')
 with pytest.raises(ValueError,match='physical model structure'):_validate_lineage(d,p,p,PK,SP,report,packet)
