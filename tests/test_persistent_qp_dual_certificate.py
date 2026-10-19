import json

import numpy as np
import pytest

from calibration.solver_runtime import install
install()
from calibration.solver_runtime import persistent_qp_face_hint_runtime as runtime
import z190_global_schur_linear_step as qp


@pytest.mark.parametrize('provenance', ['generic', 'IPM'])
def test_persistent_consumer_requires_verified_complete_KKT_for_either_dual(tmp_path, monkeypatch, provenance):
    report=dict(complete_constraint_rows_retained=True,
        original_H_gradient_constraints_bounds_and_KKT_gate_retained=True,
        free_stationarity_max_absolute=0.,kkt_tolerance=1e-9,
        original_unit_complementarity_absolute_sum=0.,original_unit_complementarity_tolerance=1e-9,
        linear_constraint_count=2,complete_original_primal=dict(accepted=True,violated_original_rows=0,all_original_rows_checked=2))
    report['complete_original_nonnegative_'+('dual'if provenance=='generic'else'IPM_dual')+'_certificate_checked']=True
    def original(H,g,lo,hi,A,b):
        original._last_verified_separation_faces=dict(shape=(len(b),len(g)),scope='test_current_chart',rows=(0,1))
        return np.zeros(len(g)),0.,report
    monkeypatch.setitem(original.__globals__,'_linearly_constrained_shared_step',original)
    monkeypatch.setattr(qp,'_linearly_constrained_shared_step',original)
    monkeypatch.setattr(runtime,'_RECEIPT',None)
    monkeypatch.setenv('RINKSPACE_SOLVER_DATA_ROOT','test_current_chart')
    runtime.install_persistent_qp_face_hint_runtime(cache_directory=tmp_path)
    qp._linearly_constrained_shared_step(np.eye(2),np.zeros(2),np.full(2,-np.inf),np.full(2,np.inf),np.eye(2),np.zeros(2))
    files=list(tmp_path.glob('*.json'));assert len(files)==1
    saved=json.loads(files[0].read_text(encoding='utf8'))
    assert saved['authority']=='scheduling_only' and not saved['solution_or_duals_stored']
    snapshot=files[0].read_bytes()
    for failed in ('stationarity','complementarity','primal','missing dual'):
        if failed=='stationarity':report['free_stationarity_max_absolute']=1.
        elif failed=='complementarity':report['original_unit_complementarity_absolute_sum']=1.
        elif failed=='primal':report['complete_original_primal']['accepted']=False
        else:report.pop('complete_original_nonnegative_'+('dual'if provenance=='generic'else'IPM_dual')+'_certificate_checked')
        qp._linearly_constrained_shared_step(np.eye(2),np.zeros(2),np.full(2,-np.inf),np.full(2,np.inf),np.eye(2),np.zeros(2))
        assert files[0].read_bytes()==snapshot
        report['free_stationarity_max_absolute']=report['original_unit_complementarity_absolute_sum']=0.
        report['complete_original_primal']['accepted']=True


def test_changed_free_chart_dimension_cannot_load_previous_scheduling_hint(tmp_path, monkeypatch):
    called=[]
    def original(H,g,lo,hi,A,b):
        called.append(getattr(original,'_last_verified_separation_faces',None))
        return np.zeros(len(g)),0.,{}
    monkeypatch.setitem(original.__globals__,'_linearly_constrained_shared_step',original)
    monkeypatch.setattr(qp,'_linearly_constrained_shared_step',original)
    monkeypatch.setattr(runtime,'_RECEIPT',None)
    monkeypatch.setenv('RINKSPACE_SOLVER_DATA_ROOT','test_current_chart')
    # An authentic previous chart hint remains historical. Its dimension must
    # make the current free-chart namespace different, not relabel old indices.
    import hashlib
    oldshape=(2,3);scope='test_current_chart'
    key=hashlib.sha256(json.dumps(dict(scope=scope,shape=oldshape,
        primal_gate=runtime.PRIMAL_CERTIFICATE_SCHEMA),sort_keys=True).encode()).hexdigest()
    (tmp_path/(key+'.json')).write_text(json.dumps(dict(scope=scope,shape=list(oldshape),
        rows=[0],primal_gate=runtime.PRIMAL_CERTIFICATE_SCHEMA)),encoding='utf8')
    runtime.install_persistent_qp_face_hint_runtime(cache_directory=tmp_path)
    qp._linearly_constrained_shared_step(np.eye(2),np.zeros(2),np.full(2,-np.inf),np.full(2,np.inf),np.eye(2),np.zeros(2))
    assert called==[None]
