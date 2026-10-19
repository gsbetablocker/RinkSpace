import numpy as np
import pytest
from scipy.sparse import csr_matrix

from calibration.solver_runtime import install
install()
import z190_joint_source_solver as solver
from z190_streaming_source_normal import SourceRowBlock


@pytest.mark.parametrize('conditional', [False, True])
def test_streaming_fixed_shared_uses_exact_tangent_and_full_view_backsolve(tmp_path, conditional):
    # Shared inactive yaw and a held view both retain complete-state slots.
    # The other view has a nonzero conditional response, rather than a dummy
    # diagonal toy that never exercises Schur elimination/backsubstitution.
    initial=np.array([0., .125, 0., -9.])
    jac=csr_matrix([[1.,0.,.5,0.],[.25,0.,1.,0.],[.5,0.,-.25,0.]])
    target=np.array([1.,-.4,.7]);mask=np.ones(3,bool);scales=np.ones(3)
    callbacks=[];recovery=[]
    def blocks(q, derivative):
        yield SourceRowBlock(jac@q-target,jac if derivative else None,mask,scales,
            dict(family='actual_fixed_declaration_test',original_start=0,original_stop=3))
    def halfspaces(q,selected):
        np.testing.assert_array_equal(selected,[0])
        callbacks.append(selected.copy())
        return dict(matrix=np.array([[1.]]),lower=np.array([.3-q[0]]),
            report={'complete_original_face_retained':True})
    def recover(parent,proposed):
        np.testing.assert_array_equal(proposed[[1,3]],parent[[1,3]])
        recovery.append(proposed.copy())
        return dict(q=proposed,report={'existing_operator_retained':True})
    def review(q,event):
        if event['checkpoint_kind']=='accepted':return {'stop_branch':True,'reason':'review exact completed trial'}
    def unused(*args):raise AssertionError('Streaming solve must use its original blocks')
    result=solver.solve_joint_source(unused,initial,mask,scales,tmp_path,lambda q:True,
        source_block_factory=blocks,cache_native_source_replay=True,
        global_schur_count=2,global_schur_indices=[0,1],fixed_parameter_indices=[1,3],
        shared_parameter_bounds={1:(.1,.2)},positive_parameter_indices=[1],positive_parameter_resolution=1e-8,
        shared_linear_constraints=halfspaces,trial_feasibility_recovery=recover,
        schur_backtracking=True,checkpoint_callback=review,
        conditional_view_backtracking=conditional,
        conditional_view_protected_parameter_indices=[0,1]if conditional else(),verbose=False)
    assert result.status=='stopped_by_source_review' and len(result.history)==1
    assert result.history[0]['accepted'] and callbacks and recovery
    np.testing.assert_array_equal(result.q[[1,3]],initial[[1,3]])
    assert result.q[0]>=.3 and abs(result.q[2])>.1
    assert np.linalg.norm(jac@result.q-target)<np.linalg.norm(jac@initial-target)
    elimination=result.history[0]['global_schur_linear_report']['declared_fixed_shared_elimination']
    assert elimination['requested_shared_parameter_indices']==[0,1]
    assert elimination['effective_shared_parameter_indices']==[0]
    assert elimination['algebraically_eliminated_fixed_shared_indices']==[1]
    assert elimination['all_declared_fixed_step_values_exact_zero']
    identity=result.history[0]['global_schur_trial_records'][-1]['trial_feasibility_recovery']['complete_state_recovery_identity']
    assert identity['recovery_called'] and identity['declared_fixed_values_preserved']
    assert identity['changed_complete_state_parameter_indices']==[]
    assert identity['proposed_q_sha256']==identity['recovered_q_sha256']


def test_declared_fixed_physical_value_is_checked_before_source_evaluation(tmp_path):
    def unused(*args):raise AssertionError('Illegal fixed bound must be rejected before source evaluation')
    with pytest.raises(ValueError,match='fixed source value is outside'):
        solver.solve_joint_source(unused,np.array([0.,.125]),np.ones(1,bool),np.ones(1),tmp_path,
            lambda q:True,global_schur_count=2,global_schur_indices=[0,1],fixed_parameter_indices=[1],
            shared_parameter_bounds={1:(.2,.3)},verbose=False)
