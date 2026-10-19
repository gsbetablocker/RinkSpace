"""A failed full-scale SOC must not poison an independent shorter trial."""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csc_matrix,csr_matrix

from calibration.solver_runtime import install
from calibration.solver_runtime import optional_soc_runtime as runtime
from calibration.solver_runtime.trial_fraction_source_domain import attach_trial_fraction_domain


@pytest.fixture
def actual_runtime(monkeypatch):
    install()
    import z190_global_schur_linear_step as qp
    import z190_joint_source_solver as joint
    monkeypatch.setattr(qp,'_linearly_constrained_shared_step',qp._linearly_constrained_shared_step)
    monkeypatch.setattr(joint,'_sequential_measured_shared_source_curvature',joint._sequential_measured_shared_source_curvature)
    runtime.install_optional_SOC_numerical_failure_route()
    yield joint,qp
    runtime._retained_soc_face_model.set(None)


def source_example():
    from structured_programme_source_candidate import source_step_curvature_tightening as source
    A=np.array([[1.],[-1.]])
    original=np.array([1.,-10.]);rho=.2
    response=dict(row_start=0,row_stop=2,nominal_carrier=np.zeros(1),
        retained_face_mask=np.ones(2,bool),dynamic_group_limits=np.array([rho]),
        dynamic_row_group_index=np.array([0,0]),dynamic_guard_comparison_tolerance_px=1e-8)
    domain=attach_trial_fraction_domain(dict(matrix=A,lower=original-rho,
        curvature_responses={'native':response}),original_constraint_lower=original,
        row_group_index=np.array([0,0]),previous_group_violations=np.ones(1),common_fraction=rho)
    measurements=[]
    def actual(q,step,selected,domain):
        x=float(step[0]);carrier=np.array([x-.5*x*x])
        measurements.append(dict(x=x,target=float(domain['curvature_responses']['native']['dynamic_group_limits'][0])))
        return source.measured_outward_step_tightening(domain['matrix'],domain['lower'],step[selected],
            domain['curvature_responses'],{'native':carrier},
            actual_group_violations={'native':np.array([max(1.-carrier[0],0.)])})
    H=csc_matrix(np.eye(3));g=np.array([-.75,0.,0.]);diagonal=np.ones(3)
    options=dict(global_count=1,damping=.001,schur_view_lm_ratio=0.,
        global_step_lower_bounds=np.array([-1.]),global_step_upper_bounds=np.array([1.]),
        global_step_constraint_matrix=A,global_step_constraint_lower=domain['lower'].copy())
    return domain,actual,measurements,H,g,diagonal,options


def test_full_SOC_infeasible_but_short_scale_has_independent_verified_correction(actual_runtime):
    joint,_=actual_runtime
    domain,actual,measurements,H,g,diagonal,options=source_example()
    source_H=H.toarray().copy();source_g=g.copy();original_faces=domain['lower'].copy()
    initial=joint.schur_trial_linear_step(H,g,diagonal,**options)
    assert runtime._previous_full_KKT_verified(initial)
    full=joint._sequential_measured_shared_source_curvature(np.zeros(3),initial,np.array([0]),domain,
        options,H,g,diagonal,actual,joint.schur_trial_linear_step)
    report=full.report['sequential_measured_shared_source_curvature_tightening']
    assert report['stop_reason']=='tightened_QP_failed_keep_last_verified_direction'
    assert full is initial and joint._needs_trial_fraction_SOC(full,domain)
    failed=report['optional_SOC_restricted_numerical_failure']
    assert failed['status']=='PrimalInfeasible'
    ray=failed['original_restricted_infeasibility_ray']
    assert ray['accepted'] and failed['numerical_tightened_linear_infeasibility_ray_verified']
    assert not failed['mathematical_infeasibility_certified']
    assert ray['only_current_tightened_linear_subproblem_not_original_nonlinear_model']
    short=joint._trial_fraction_source_step(np.zeros(3),.5,np.array([0]),domain,options,
        H,g,diagonal,actual,joint.schur_trial_linear_step)
    assert runtime._previous_full_KKT_verified(short)
    assert short.step[0]>.5*full.step[0]
    assert 1.-short.step[0]+.5*short.step[0]**2<=.6+1e-8
    assert len([m for m in measurements if m['target']==.6])>1
    np.testing.assert_array_equal(H.toarray(),source_H)
    np.testing.assert_array_equal(g,source_g)
    np.testing.assert_array_equal(domain['lower'],original_faces)
    np.testing.assert_array_equal(options['global_step_constraint_lower'],original_faces)
    evidence=short.report['independent_trial_fraction_source_step']
    assert evidence['returned_step_is_actual_displacement_not_scaled_again']
    assert not evidence['accumulated_full_step_SOC_cuts_reused']
    assert evidence['actual_source_model_decrease']==pytest.approx(-g@short.step-.5*short.step@(H@short.step))


def test_actual_line_search_uses_short_returned_displacement_and_original_cost_prediction(actual_runtime,tmp_path):
    joint,_=actual_runtime
    domain,actual,measurements,H,g,diagonal,options=source_example()
    target=-g;acceptance=[];guards=[]
    def evaluate(q,jacobian):
        r=q-target
        return (r,csr_matrix(np.eye(3)))if jacobian else r
    def guard(q):
        guards.append(q.copy())
        if not -1.<=q[0]<=1.:raise ValueError('Original physical interval')
    def accept(parent,trial,*,before_cost,after_cost,default_source_cost_accepts,trial_context):
        alpha=trial_context['alpha'];x=trial[0];required=1.-alpha*.8
        decision=1.-x+.5*x*x<=required+1e-8
        acceptance.append(dict(alpha=alpha,step=trial_context['actual_step'].copy(),
            predicted=trial_context['predicted_source_reduction'],decision=decision))
        return dict(accept=bool(decision),mode='truth_restoration',report={},rejection_reason='Original nonlinear demand')
    result=joint.solve_joint_source(evaluate,np.zeros(3),np.ones(3,bool),np.ones(3),tmp_path,guard,
        global_schur_count=1,schur_backtracking=True,initial_damping=.001,
        shared_parameter_bounds={0:(-1.,1.)},shared_linear_constraints=lambda q,selected:domain,
        shared_step_curvature_tightening=actual,source_trial_acceptance=accept,
        checkpoint_callback=lambda q,event:({'stop_branch':True,'reason':'Synthetic accepted-step verification'}
            if event['checkpoint_kind']=='accepted'else None),verbose=False)
    assert result.history[0]['accepted'] and len(acceptance)==2
    assert acceptance[0]['alpha']==1. and not acceptance[0]['decision']
    assert acceptance[1]['alpha']==.5 and acceptance[1]['decision']
    np.testing.assert_array_equal(result.q,acceptance[1]['step'])
    assert result.q[0]>.5  # Scaling the already-short corrected step again fails this assertion.
    assert acceptance[1]['predicted']==pytest.approx(-g@result.q-.5*result.q@(H@result.q))
    trial=result.history[0]['global_schur_trial_records'][-1]
    assert trial['whole_source_step_fraction']is None
    assert trial['numerical_source_gradient_fraction']==.5
    assert trial['actual_step_basis']=='independently_solved_trial_fraction_source_model'
    assert result.history[0]['next_cost']==pytest.approx(.5*np.sum((result.q-target)**2))
    np.testing.assert_array_equal(guards[-1],result.q)


def test_no_fraction_callback_keeps_historical_shortening(actual_runtime):
    joint,_=actual_runtime
    assert not joint._needs_trial_fraction_SOC(SimpleNamespace(report={
        'sequential_measured_shared_source_curvature_tightening':{
            'stop_reason':'tightened_QP_failed_keep_last_verified_direction'}}),{'lower':np.zeros(1)})


def test_short_fraction_rejects_missing_complete_original_KKT(actual_runtime):
    joint,_=actual_runtime
    domain,actual,_,H,g,diagonal,options=source_example()
    def fake(*args,**kwargs):return SimpleNamespace(step=np.array([2.,0.,0.]),report={})
    with pytest.raises(ValueError,match='complete original-row primal/dual KKT'):
        joint._trial_fraction_source_step(np.zeros(3),.5,np.array([0]),domain,options,
            H,g,diagonal,actual,fake)
