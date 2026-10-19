"""Actual small HiGHS LPs, full-row KKT and original-error fallback."""
from types import SimpleNamespace

import numpy as np
from scipy.optimize import linprog

from calibration.solver_runtime import separated_priority_LP_runtime as runtime


OPTIONS=dict(primal_feasibility_tolerance=1e-9,dual_feasibility_tolerance=1e-9)


def model():
    # Min t with a strict optimum .75. Lower-priority faces are inactive;
    # all four zero-expansion protected physical box rows remain active.
    protected=np.array([[1.,0.,0.],[-1.,0.,0.],[0.,1.,0.],[0.,-1.,0.]])
    curves=np.array([[1.,0.,-1.],[-1.,0.,-1.],[0.,1.,-1.],[0.,-1.,-1.]])
    A=np.vstack((protected,np.tile(curves,(100,1))))
    b=np.r_[np.ones(4),-.75,-.75,-.5,-.5,np.tile([-.2,-.2,-.1,-.1],99)]
    return np.array([0.,0.,1.]),dict(A_ub=A,b_ub=b,bounds=[(None,None),(None,None),(0.,1.)],method='highs-ipm',options=OPTIONS)


def test_actual_LP_same_optimum_and_every_original_KKT():
    c,kwargs=model()
    old=linprog(c,**kwargs)
    new=runtime.separated_original_linprog(linprog,c,**kwargs)
    assert old.success and new.success
    assert abs(new.fun-old.fun)<=1e-9
    trace=runtime._LATEST.get()
    assert trace['route']=='separated_original_LP_full_KKT'
    assert trace['complete_original_problem_KKT_certified']
    assert trace['protected_rows']==4
    assert len(new.ineqlin.marginals)==len(kwargs['b_ub'])
    assert np.min(kwargs['b_ub']-kwargs['A_ub']@new.x)>=-1e-9
    assert trace['batch_history'][-1]['active_rows']<len(kwargs['b_ub'])
    assert np.max(np.abs(c-kwargs['A_ub'].T@new.ineqlin.marginals-new.lower.marginals-new.upper.marginals))<=1e-9


def test_missing_dual_uses_original_full_call_once():
    c,kwargs=model();calls=[]
    def original(objective,**values):
        calls.append(len(values['b_ub']))
        result=linprog(objective,**values)
        if len(values['b_ub'])<len(kwargs['b_ub']):
            result.ineqlin=SimpleNamespace(residual=result.ineqlin.residual)
        return result
    new=runtime.separated_original_linprog(original,c,**kwargs)
    assert new.success and calls[-1]==len(kwargs['b_ub'])
    assert calls.count(len(kwargs['b_ub']))==1
    assert runtime._LATEST.get()['route']=='original_full_linprog'
    assert not runtime._LATEST.get()['complete_original_problem_KKT_certified']


def test_protected_faces_can_be_separated_only_with_exact_boxes_and_full_KKT():
    c,kwargs=model()
    # The original LP appends its physical boxes after all source faces.
    A=np.vstack((kwargs['A_ub'][4:],np.array([[0.,1.,0.],[0.,-1.,0.]])))
    b=np.r_[kwargs['b_ub'][4:],1.,1.]
    # Add source-protected faces which must be discovered by full separation,
    # rather than silently dropped with zero original expansion coefficient.
    A=np.vstack((np.array([[0.,-1.,0.],[1.,0.,0.]]),A))
    b=np.r_[0.,0.,b]
    values=dict(kwargs,A_ub=A,b_ub=b)
    old=linprog(c,**values)
    new=runtime.separated_original_linprog(linprog,c,original_box_row_count=2,
        separate_protected_faces=True,**values)
    assert old.success and new.success and abs(new.fun-old.fun)<=1e-9
    trace=runtime._LATEST.get()
    assert trace['complete_original_problem_KKT_certified']
    assert not trace['protected_rows_initially_active']
    assert trace['exact_original_box_rows_initially_active']==2
    assert len(trace['batch_history'])>1
    assert np.min(b-A@new.x)>=-1e-9


def test_full_original_error_not_repeated_or_masked():
    calls=[]
    def original(*args,**kwargs):
        calls.append(1)
        raise ValueError('original unchanged backend input error')
    import pytest
    with pytest.raises(ValueError,match='original unchanged backend input error'):
        runtime.separated_original_linprog(original,np.array([1.]))
    assert len(calls)==1


def test_original_contract_hook_keeps_targets_and_physical_rows(tmp_path):
    from calibration.solver_runtime import install
    install()
    from structured_programme_source_candidate import feasibility_priority_restoration as source
    old=source.contract_original_support_bounds
    lower=np.array([-1.,-1.]);upper=np.array([1.,1.])
    rows=np.array([[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]])
    bounds=np.array([.75,.75,.5,.5]);ids=np.array([0,0,1,1]);violations=np.ones(2)
    expected=old(lower,upper,rows,bounds,ids,violations)
    runtime.install_separated_priority_LP_runtime(report_directory=tmp_path)
    try:
        actual=source.contract_original_support_bounds(lower,upper,rows,bounds,ids,violations)
        assert np.array_equal(actual['matrix'],expected['matrix'])
        assert np.max(np.abs(actual['lower']-expected['lower']))<=1e-9
        assert abs(actual['common_fraction']-expected['common_fraction'])<=1e-9
        assert actual['report']['complete_original_LP_separation']['complete_original_problem_KKT_certified']
    finally:source.contract_original_support_bounds=old
