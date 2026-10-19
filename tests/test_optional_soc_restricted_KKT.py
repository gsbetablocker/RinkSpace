import numpy as np
import pytest

from calibration.solver_runtime.optional_soc_runtime import (
    restricted_original_KKT_report, _optional_soc, _optional_complete_feasibility_progress)


def report(candidate, dual, *, rows=None, targets=None, selected=None):
    matrix = np.eye(2)
    gradient = np.array([-2., 0.])
    rows = np.array([[-1., 0.], [0., 1.]]) if rows is None else rows
    targets = np.array([-1., 1.]) if targets is None else targets
    selected = np.array([0]) if selected is None else selected
    scale = np.linalg.norm(rows, axis=1)
    return restricted_original_KKT_report(matrix, gradient, rows, targets, scale,
        rows / scale[:, None], selected, np.array(candidate), np.array(dual),
        rows @ candidate - targets, 1e-11)


def test_valid_restricted_witness_can_leave_unselected_faces_violated():
    result = report([1., 0.], [1., 0.])
    assert result['accepted']
    assert result['unselected_rows_still_checked_by_original_separation']


@pytest.mark.parametrize('candidate,dual', [([1.1, 0.], [.9, 0.]),
    ([1., 0.], [-1., 0.]), ([.9, 0.], [1.1, 0.]),
    ([1., 0.], [0., 0.]), ([float('nan'), 0.], [1., 0.])])
def test_invalid_selected_primal_dual_stationarity_or_complementarity_rejected(candidate, dual):
    assert not report(candidate, dual)['accepted']


def test_original_units_and_positive_row_scaling_preserve_gate():
    base = report([1., 0.], [1., 0.])
    changed = report([1., 0.], [1., 0.], rows=np.array([[-2000., 0.], [0., .003]]),
                     targets=np.array([-2000., .003]))
    assert base['accepted'] == changed['accepted'] == True
    assert base['maximum_original_stationarity'] == changed['maximum_original_stationarity'] == 0.
    assert not changed['arbitrary_iteration_or_time_limit_used']


def test_only_optional_complete_primal_nonprogress_retreats_without_caps():
    assert _optional_complete_feasibility_progress(np.array([-1.]), 1e-11) is None
    token = _optional_soc.set({})
    try:
        assert _optional_complete_feasibility_progress(np.array([-4.88, 0.]), 1e-11)['continue_optional_separation']
        report = _optional_complete_feasibility_progress(np.array([-682., -20.]), 1e-11)
        assert not report['continue_optional_separation']
        assert not report['mathematical_infeasibility_certified']
        assert not report['arbitrary_iteration_or_time_limit_used']
    finally:
        _optional_soc.reset(token)


def test_improvement_in_either_full_primal_merit_keeps_separation():
    token = _optional_soc.set({})
    try:
        _optional_complete_feasibility_progress(np.array([-10., -10.]), 1e-11)
        assert _optional_complete_feasibility_progress(np.array([-11., -1.]), 1e-11)['continue_optional_separation']
        assert _optional_complete_feasibility_progress(np.array([0., 0.]), 1e-11)['continue_optional_separation']
    finally:
        _optional_soc.reset(token)


def test_optional_selected_gate_runs_after_public_QR_polish_before_separation(monkeypatch):
    from calibration.solver_runtime import install
    install()
    import z190_global_schur_linear_step as qp
    import z190_joint_source_solver as joint
    from calibration.solver_runtime import optional_soc_runtime as runtime
    from types import SimpleNamespace
    original=qp._linearly_constrained_shared_step
    original_SOC=joint._sequential_measured_shared_source_curvature
    calls=[]
    def backend(solver,phase,rows,parameters,selected=None):
        value=solver.solve()
        y=np.asarray(value.x,float).copy()
        if not calls:
            # An inaccurate selected face is not a valid witness. Its existing
            # QR refinement is allowed first; all actual gates still decide.
            y[0]-=2e-8
        calls.append(len(selected))
        return SimpleNamespace(x=y,z=value.z,status=value.status,iterations=value.iterations)
    monkeypatch.setattr(runtime,'run_logged_QP_backend',backend)
    monkeypatch.setattr(qp,'_linearly_constrained_shared_step',original)
    monkeypatch.setattr(joint,'_sequential_measured_shared_source_curvature',original_SOC)
    monkeypatch.delattr(original,'_last_verified_separation_faces',raising=False)
    runtime.install_optional_SOC_numerical_failure_route()
    token=_optional_soc.set({})
    try:
        x,_,result=qp._linearly_constrained_shared_step(np.eye(2),np.zeros(2),
            np.full(2,-np.inf),np.full(2,np.inf),
            np.array([[1.,0.],[0.,1.],[-1.,1.],[-1.,0.]]),np.array([10.,1.,-.1,-1e14]))
        np.testing.assert_allclose(x,[10.,9.9],rtol=0.,atol=1e-9)
        rounds=result['verified_full_constraint_separation']['rounds']
        assert rounds[0]['restricted_original_KKT_refinement']['refined']
        assert rounds[0]['original_selected_KKT']['accepted']
        assert len(rounds)>=2 and result['complete_original_primal']['accepted']
        assert result['complete_original_nonnegative_dual_certificate_checked']
    finally:
        _optional_soc.reset(token)
        qp._linearly_constrained_shared_step=original
        joint._sequential_measured_shared_source_curvature=original_SOC


def test_tightened_face_seed_is_optional_exact_current_row_indices():
    from calibration.solver_runtime.optional_soc_runtime import _optional_strictly_tightened_face_seed
    np.testing.assert_array_equal(_optional_strictly_tightened_face_seed(4),[])
    token=_optional_soc.set({'strictly_tightened_original_faces':np.array([1,3])})
    try:
        np.testing.assert_array_equal(_optional_strictly_tightened_face_seed(4),[1,3])
        with pytest.raises(ValueError,match='current original row map'):
            _optional_strictly_tightened_face_seed(3)
    finally:_optional_soc.reset(token)
