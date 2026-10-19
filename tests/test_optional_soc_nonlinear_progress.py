"""The optional outer SOC must measure actual faces, not increasing cuts."""
from types import SimpleNamespace

import numpy as np

from calibration.solver_runtime import optional_soc_runtime as runtime


def test_complete_original_merit_includes_every_actual_axis_and_affine_row():
    matrix=np.array([[2.], [1.], [-1.], [1.], [-1.]])
    lower=np.array([0., .2, 0., 0., 0.])
    responses={'native':dict(row_start=1,row_stop=5,
        nominal_carrier=np.array([4., 8.]),retained_face_mask=np.ones(4,bool))}
    result=runtime.original_nonlinear_step_face_feasibility(matrix,lower,np.array([1.]),
        responses,{'native':np.array([3., 8.-1e-10])})
    assert result['complete_original_step_rows']==5
    assert result['actual_nonlinear_source_rows']==4
    assert result['original_affine_rows']==1
    assert result['maximum_violation_px']==1.2
    # The existing compensation cutoff is not used to discard small actual
    # violations from the nonlinear merit.
    assert result['total_violation_px']>1.2
    np.testing.assert_array_equal(lower,[0., .2, 0., 0., 0.])


def replay_actual_QP_and_nonlinear_source(*,curvature,original_target):
    from calibration.solver_runtime import install
    install()
    import z190_global_schur_linear_step as qp
    import z190_joint_source_solver as joint
    from structured_programme_source_candidate import source_step_curvature_tightening as source
    runtime._install_original_nonlinear_face_observer()
    producer=joint._sequential_measured_shared_source_curvature
    while hasattr(producer,'_original_SOC_producer'):producer=producer._original_SOC_producer
    target=np.atleast_1d(original_target).astype(float);n=len(target)
    H=np.eye(n);g=np.full(n,-.25);A=np.vstack((np.eye(n),-np.eye(n)))
    b=np.r_[target,np.full(n,-100.)];options={'global_step_constraint_lower':b.copy()}
    responses={'native':dict(row_start=0,row_stop=2*n,nominal_carrier=np.zeros(n),
                            retained_face_mask=np.ones(2*n,bool))}
    attempts=[]
    def solve(H,g,diagonal,**options):
        x,_,report=qp._linearly_constrained_shared_step(H,g,np.full(n,-np.inf),
            np.full(n,np.inf),A,options['global_step_constraint_lower'])
        result=SimpleNamespace(step=x,report={'shared_linear_domain_step':report})
        assert runtime._previous_full_KKT_verified(result)
        attempts.append(result)
        return result
    initial=solve(H,g,None,**options)
    def nonlinear(q,step,columns,domain):
        actual=step-curvature*step**2
        return source.measured_outward_step_tightening(A,b,step,responses,{'native':actual})
    final=runtime._with_outer_nonlinear_progress(producer,np.zeros(n),initial,
        np.arange(n),{'lower':b},options,H,g,np.ones(n),nonlinear,solve)
    return initial,final,attempts,b,options


def test_successful_KKT_correction_with_worse_true_faces_retains_previous_direction():
    initial,final,attempts,b,options=replay_actual_QP_and_nonlinear_source(
        curvature=.5,original_target=3.)
    assert len(attempts)==2
    assert final is initial
    assert runtime._previous_full_KKT_verified(attempts[-1])
    np.testing.assert_array_equal(options['global_step_constraint_lower'],b)
    outer=final.report['sequential_measured_shared_source_curvature_tightening']
    assert outer['stop_reason']=='optional_actual_original_nonlinear_feasibility_nonprogress'
    measurements=outer['actual_original_nonlinear_feasibility_progress']['measurements']
    assert (measurements[1]['actual_original_face_feasibility']['maximum_violation_px']>
        measurements[0]['actual_original_face_feasibility']['maximum_violation_px'])
    assert all(row['complete_original_KKT_verified']for row in measurements)
    assert not outer['actual_original_nonlinear_feasibility_progress']['mathematical_infeasibility_certified']


def test_actual_progress_can_continue_while_model_disagreement_holds_damping():
    initial,final,attempts,b,options=replay_actual_QP_and_nonlinear_source(
        curvature=.1,original_target=2.8)
    assert len(attempts)>2
    assert final is not initial
    outer=final.report['sequential_measured_shared_source_curvature_tightening']
    records=outer['actual_original_nonlinear_feasibility_progress']['measurements']
    merits=[row['actual_original_face_feasibility']['maximum_violation_px']for row in records]
    assert any(merits[i]<merits[i-1] and not records[i]['original_face_reduction_model_is_reliable']
        for i in range(1,len(records)))
    assert all(np.isfinite(row['original_quadratic_model_decrease'])for row in records)
    assert outer['actual_original_nonlinear_feasibility_progress']['no_iteration_or_time_limit']
    assert runtime._previous_full_KKT_verified(final)
    assert np.all(options['global_step_constraint_lower']>=b)
    assert outer['stop_reason']=='optional_actual_original_nonlinear_feasibility_insufficient_model_reduction'
    assert not records[-1]['current_candidate_discarded']
    assert final is attempts[-1]
    assert outer['retained_complete_KKT_direction_sha256']==runtime._digest(final.step)


def test_reliable_actual_face_reduction_continues_through_a_feasible_correction():
    initial,final,attempts,b,options=replay_actual_QP_and_nonlinear_source(
        curvature=.01,original_target=2.8)
    assert len(attempts)>2 and final is attempts[-1]
    outer=final.report['sequential_measured_shared_source_curvature_tightening']
    records=outer['actual_original_nonlinear_feasibility_progress']['measurements']
    assert all(row['original_face_reduction_model_is_reliable']for row in records)
    assert not records[-1]['another_measured_correction_requested']
    assert not outer['actual_original_nonlinear_feasibility_progress']['optional_correction_abandoned']
    assert runtime._previous_full_KKT_verified(final)


def test_lower_total_merit_cannot_replace_direction_with_worse_maximum_merit():
    initial,final,attempts,b,options=replay_actual_QP_and_nonlinear_source(
        curvature=np.array([.23,.1,.1]),original_target=[2.1,2.,2.])
    assert len(attempts)==2 and final is initial
    records=final.report['sequential_measured_shared_source_curvature_tightening'][
        'actual_original_nonlinear_feasibility_progress']['measurements']
    before,after=[row['actual_original_face_feasibility']for row in records]
    assert after['maximum_violation_px']>before['maximum_violation_px']
    assert after['total_violation_px']<before['total_violation_px']
    assert not records[-1]['maximum_and_total_actual_feasibility_nonworsening']
    assert records[-1]['current_candidate_discarded']
    np.testing.assert_array_equal(options['global_step_constraint_lower'],b)


def test_retained_better_SOC_model_disagreement_only_holds_restoration_damping():
    from calibration.solver_runtime import install
    install()
    import z190_joint_source_solver as joint
    runtime._install_retained_SOC_damping_trust(joint)
    replay_actual_QP_and_nonlinear_source(curvature=.1,original_target=2.8)
    trust=joint.restoration_source_model_agreement(10.,10.)
    assert trust['reliable_for_restoration_lambda_decrease']
    assert joint.accepted_schur_damping(.01,1.,truth_restoration=True,
        restoration_model_agreement=trust)==.01
    assert trust['source_model_reliable_before_SOC_face_check']
    assert not trust['reliable_for_restoration_lambda_decrease']
    assert trust['retained_SOC_original_face_model']['current_candidate_retained']
    # Starting a new SOC with a reliable model removes the old numerical flag.
    replay_actual_QP_and_nonlinear_source(curvature=.01,original_target=2.8)
    trust=joint.restoration_source_model_agreement(10.,10.)
    assert joint.accepted_schur_damping(.01,1.,truth_restoration=True,
        restoration_model_agreement=trust)==.003
    assert 'retained_SOC_original_face_model'not in trust
    runtime._retained_soc_face_model.set(None)
