from types import SimpleNamespace
import numpy as np
import pytest
from calibration.solver_runtime import install
install()
from calibration.solver_runtime.original_halfspace_primal import original_halfspace_primal
from calibration.solver_runtime.original_halfspace_primal import original_full_qp_kkt_verified
from calibration.solver_runtime.optional_soc_runtime import (
    _previous_full_KKT_verified,restricted_original_KKT_report,
    _optional_soc,_optional_complete_feasibility_progress)
import z190_global_schur_linear_step as qp
from structured_programme_source_candidate.feasibility_priority_restoration import contract_original_support_bounds


def test_restricted_precision_failure_is_refined_then_complete_separation_continues(monkeypatch):
    import clarabel
    real=clarabel.DefaultSolver
    class PerturbedRestrictedSolver:
        def __init__(self,*args):self.solver=real(*args)
        def solve(self):
            raw=self.solver.solve()
            # Reproduce a backend candidate missing a selected face by more
            # than original-unit arithmetic precision, with valid duals.
            values={key:getattr(raw,key) for key in ('status','iterations','x','z')}
            values['x']=np.asarray(raw.x).copy();values['x'][0]-=1e-8
            return SimpleNamespace(**values)
    monkeypatch.setattr(clarabel,'DefaultSolver',PerturbedRestrictedSolver)
    qp._linearly_constrained_shared_step._last_verified_separation_faces=None
    A=np.array([[1.,0.],[0.,1.],[-1.,1.],[-1.,0.]])
    b=np.array([10.,1.,-.1,-1e14]);box=np.full(2,np.inf)
    x,_,report=qp._linearly_constrained_shared_step(np.eye(2),np.zeros(2),-box,box,A,b)
    np.testing.assert_allclose(x,[10.,9.9],rtol=0.,atol=1e-9)
    separation=report['verified_full_constraint_separation']
    assert separation['accepted'] and len(separation['rounds'])>=2
    first=separation['rounds'][0]
    assert first['restricted_original_KKT_refinement']['refined']
    assert not first['complete_original_primal']['accepted']
    assert first['unselected_violations']>0
    assert report['complete_original_primal']['all_original_rows_checked']==4
    assert report['complete_original_primal']['accepted']


def test_exact_constant_faces_keep_full_primal_certificate_and_infeasibility_is_rejected():
    A=np.array([[1.,0.],[0.,0.],[0.,1.],[-1.,1.]])
    b=np.array([10.,0.,1.,-.1]);lo=np.full(2,-np.inf);hi=np.full(2,np.inf)
    x,_,report=qp._linearly_constrained_shared_step(np.eye(2),np.zeros(2),lo,hi,A,b)
    np.testing.assert_allclose(x,[10.,9.9],atol=1e-8,rtol=0.)
    assert report['complete_original_primal']['all_original_rows_checked']==4
    assert report['exact_constant_original_halfspaces']==[1]
    b[1]=np.nextafter(0.,1.)
    with pytest.raises(ValueError,match='constant original QP'):qp._linearly_constrained_shared_step(np.eye(2),np.zeros(2),lo,hi,A,b)


def test_priority_LP_constant_faces_only_eliminate_mathematical_true_rows():
    lo=np.array([-2.]);hi=np.array([2.]);A=np.array([[1.],[-1.],[0.],[0.]])
    b=np.array([2.,0.,0.,-4.]);ids=np.array([0,0,-1,-1]);v=np.array([2.])
    result=contract_original_support_bounds(lo,hi,A,b,ids,v)
    assert result['common_fraction']<1.
    assert result['matrix'].shape==A.shape
    assert result['report']['exact_constant_original_faces_omitted_from_numerical_LP']==[2,3]
    np.testing.assert_array_equal(result['lower'][2:],b[2:])
    b[2]=np.nextafter(0.,1.)
    with pytest.raises(ValueError,match='constant original phase-I'):contract_original_support_bounds(lo,hi,A,b,ids,v)


@pytest.mark.parametrize('loose_row,loose_target', [([-1.,0.],-1e14),
    ([-1e-14,0.],-20.)])
def test_unrelated_loose_or_weak_row_cannot_hide_required_face(loose_row,loose_target):
    A=np.array([[1.,0.],[0.,1.],[-1.,1.],loose_row])
    b=np.array([10.,1.,-.1,loose_target])
    x,_,report=qp._linearly_constrained_shared_step(np.eye(2),np.zeros(2),
        np.full(2,-np.inf),np.full(2,np.inf),A,b)
    np.testing.assert_allclose(x,[10.,9.9],rtol=0.,atol=1e-8)
    assert report['complete_original_primal']['accepted']
    assert report['verified_full_constraint_separation']['accepted']
    assert report['complete_original_primal']['violated_original_rows']==0
    assert not original_halfspace_primal(A,b,np.array([10.,1.]))[3]['accepted']


def test_original_primal_threshold_ignores_all_other_row_rhs():
    A=np.eye(2);b=np.array([1.,-1e14]);x=np.array([1.-1e-6,0.])
    _,allowed,bad,report=original_halfspace_primal(A,b,x)
    assert allowed[0]<1e-9 and bad.tolist()==[True,False]
    assert not report['tolerance_uses_other_rows_or_unconstrained_center']


def test_restricted_selected_primal_cannot_use_legacy_huge_global_tolerance():
    A=np.eye(2);x=np.array([10.,1.]);b=np.array([10.,9.9]);dual=x.copy()
    result=restricted_original_KKT_report(np.eye(2),np.zeros(2),A,b,np.ones(2),
        A,np.arange(2),x,dual,A@x-b,1e8)
    assert not result['accepted']
    assert result['selected_original_primal']['violated_original_rows']==1


def test_optional_retreat_requires_explicit_complete_original_primal_certificate():
    report=dict(complete_constraint_rows_retained=True,
        complete_original_nonnegative_IPM_dual_certificate_checked=True,
        original_H_gradient_constraints_bounds_and_KKT_gate_retained=True,
        free_stationarity_max_absolute=0.,kkt_tolerance=1e-9,
        original_unit_complementarity_absolute_sum=0.,
        original_unit_complementarity_tolerance=1e-9,linear_constraint_count=2)
    state=SimpleNamespace(step=np.zeros(2),report={'shared_linear_domain_step':report})
    assert not _previous_full_KKT_verified(state)
    report['complete_original_primal']=dict(accepted=True,violated_original_rows=0,all_original_rows_checked=2)
    assert _previous_full_KKT_verified(state)
    report['complete_original_primal']['accepted']=False
    assert not _previous_full_KKT_verified(state)


@pytest.mark.parametrize('provenance',('generic','IPM'))
def test_hint_and_optional_generic_or_legacy_dual_require_same_full_KKT(provenance):
    field=('complete_original_nonnegative_dual_certificate_checked' if provenance=='generic'
        else 'complete_original_nonnegative_IPM_dual_certificate_checked')
    report=dict(complete_constraint_rows_retained=True,
        original_H_gradient_constraints_bounds_and_KKT_gate_retained=True,
        free_stationarity_max_absolute=0.,kkt_tolerance=1e-9,
        original_unit_complementarity_absolute_sum=0.,original_unit_complementarity_tolerance=1e-9,
        linear_constraint_count=2,complete_original_primal=dict(accepted=True,violated_original_rows=0,all_original_rows_checked=2))
    report[field]=True
    state=SimpleNamespace(step=np.zeros(2),report={'shared_linear_domain_step':report})
    assert original_full_qp_kkt_verified(report,state.step) and _previous_full_KKT_verified(state)
    for key,value in [('free_stationarity_max_absolute',1e-6),('original_unit_complementarity_absolute_sum',1e-6),
                      ('complete_constraint_rows_retained',False),(field,False)]:
        before=report[key];report[key]=value
        assert not original_full_qp_kkt_verified(report,state.step) and not _previous_full_KKT_verified(state)
        report[key]=before
    report['complete_original_primal']['violated_original_rows']=1
    assert not original_full_qp_kkt_verified(report,state.step)
    report['complete_original_primal']['violated_original_rows']=0
    report['complete_original_primal']['all_original_rows_checked']=1
    assert not original_full_qp_kkt_verified(report,state.step)
    report['complete_original_primal']['all_original_rows_checked']=2
    report.pop('linear_constraint_count')
    assert not original_full_qp_kkt_verified(report,state.step)


@pytest.mark.parametrize('candidate',[np.empty(0),np.ones((1,2)),np.array([np.nan,0.])])
def test_full_KKT_consumer_rejects_invalid_candidate_layout(candidate):
    report=dict(complete_original_nonnegative_dual_certificate_checked=True,complete_constraint_rows_retained=True,
        original_H_gradient_constraints_bounds_and_KKT_gate_retained=True,
        free_stationarity_max_absolute=0.,kkt_tolerance=1e-9,
        original_unit_complementarity_absolute_sum=0.,original_unit_complementarity_tolerance=1e-9,
        linear_constraint_count=2,complete_original_primal=dict(accepted=True,violated_original_rows=0,all_original_rows_checked=2))
    assert not original_full_qp_kkt_verified(report,candidate)


def test_optional_progress_uses_each_threshold_without_global_weak_row_allowance():
    token=_optional_soc.set({})
    try:
        first=_optional_complete_feasibility_progress(np.array([-1.,1e14]),np.array([1e-10,1e5]))
        assert first['current']['maximum_full_primal_excess']>.9
        second=_optional_complete_feasibility_progress(np.array([-2.,1e14]),np.array([1e-10,1e5]))
        assert not second['continue_optional_separation']
    finally:_optional_soc.reset(token)


@pytest.mark.parametrize('x',[np.empty(0),np.ones((1,2)),np.array([np.nan,0.])])
def test_invalid_candidate_is_rejected(x):
    with pytest.raises(ValueError):original_halfspace_primal(np.eye(2),np.zeros(2),x)


def test_exact_inactive_path_remains_original_solve_bytes():
    H=np.array([[4.,1.],[1.,3.]]);g=np.array([-1.,-.5])
    expected=np.linalg.solve(H,-g)
    actual,_,report=qp._linearly_constrained_shared_step(H,g,np.full(2,-10.),
        np.full(2,10.),np.eye(2),np.full(2,-10.))
    np.testing.assert_array_equal(actual,expected)
    assert report['linear_constraints_inactive']
