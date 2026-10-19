"""Backend success cannot replace the original complete primal witness."""
from types import SimpleNamespace
import numpy as np
import pytest
from calibration.solver_runtime import install
install()
from structured_programme_source_candidate import feasibility_priority_restoration as source


def result(x):return SimpleNamespace(success=True,x=np.asarray(x,float),nit=1,message='backend claims success')


def problem():return (np.array([-2.]),np.array([2.]),np.array([[1.]]),np.array([1.]),np.array([0]),np.array([1.]))


@pytest.mark.parametrize('invalid',[[0.,0.],[1.,-1e-14],[1.,1.+1e-14],[np.nan,0.]])
def test_false_primary_success_uses_existing_same_original_conic_witness(monkeypatch,invalid):
    calls=[]
    monkeypatch.setattr(source,'linprog',lambda *a,**k:result(invalid))
    def fallback(B,expansion,rhs,norm,*,original_failure):
        calls.append((B.copy(),expansion.copy(),rhs.copy(),norm.copy(),original_failure))
        r=result([1.,0.]);r.original_phaseI_report={};return r
    monkeypatch.setattr(source,'_verified_same_original_conic_phaseI',fallback)
    domain=source.contract_original_support_bounds(*problem())
    assert len(calls)==1
    assert not domain['report']['phase_I_primary_backend_complete_primal_check']['passed']
    assert domain['common_fraction']==.5
    np.testing.assert_array_equal(domain['matrix'],problem()[2])
    np.testing.assert_array_equal(domain['lower'],np.array([.5]))
    assert domain['report']['phase_I_primary_backend_complete_primal_check']['original_primal_tolerance']==1e-9


def test_valid_primary_primal_does_not_require_LP_dual_optimality(monkeypatch):
    monkeypatch.setattr(source,'linprog',lambda *a,**k:result([1.,0.]))
    def forbidden(*a,**k):raise AssertionError('A valid original primal witness should be retained')
    monkeypatch.setattr(source,'_verified_same_original_conic_phaseI',forbidden)
    domain=source.contract_original_support_bounds(*problem())
    assert domain['report']['phase_I_primary_backend_complete_primal_check']['passed']
    assert domain['common_fraction']==.5
    assert domain['report']['phase_I_feasible_common_fraction_witness']==0.
    assert domain['report']['original_midpoint_all_faces_and_physical_boxes_primal_verified']
    assert not domain['report']['phase_I_optimality_certificate_claimed']


@pytest.mark.parametrize('normalized_deficit,should_fallback',[(.9e-9,False),(1.1e-9,True)])
def test_primary_original_normalized_primal_threshold_unchanged(monkeypatch,normalized_deficit,should_fallback):
    monkeypatch.setattr(source,'linprog',lambda *a,**k:result([1.-np.sqrt(2.)*normalized_deficit,0.]))
    calls=[]
    def fallback(*a,**k):
        calls.append(True);r=result([1.,0.]);r.original_phaseI_report={};return r
    monkeypatch.setattr(source,'_verified_same_original_conic_phaseI',fallback)
    domain=source.contract_original_support_bounds(*problem())
    assert bool(calls)==should_fallback
    assert domain['report']['phase_I_primary_backend_complete_primal_check']['original_primal_tolerance']==1e-9
    assert domain['report']['original_midpoint_all_faces_and_physical_boxes_primal_verified']


def test_complete_check_includes_immutable_box_rows(monkeypatch):
    monkeypatch.setattr(source,'linprog',lambda *a,**k:result([3.,0.]))
    seen=[]
    def fallback(*a,**k):seen.append(True);r=result([1.,0.]);r.original_phaseI_report={};return r
    monkeypatch.setattr(source,'_verified_same_original_conic_phaseI',fallback)
    domain=source.contract_original_support_bounds(*problem())
    assert seen and not domain['report']['phase_I_primary_backend_complete_primal_check']['passed']
    assert domain['report']['phase_I_primary_backend_complete_primal_check']['original_all_face_normalized_minimum_slack']<0.
