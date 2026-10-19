"""Complete source LM coordinates and coupled physical constraints."""
import numpy as np
from scipy.sparse import csc_matrix

from calibration.solver_runtime.full_free_physical import (
    original_trial_matrix, physical_trial_step, exact_separable_null_coordinates)


def test_complete_trial_retains_original_schur_lm_matrix_and_fixed_coordinates():
    H=np.array([[4.,1.,0.],[1.,3.,0.],[0.,0.,1.]])
    g=np.array([-2.,-3.,0.]);D=np.diag(H);lam=.1
    system=original_trial_matrix(csc_matrix(H),g,D,shared_indices=np.array([0]),
        fixed_indices=np.array([2]),damping=lam,schur_view_lm_ratio=1.)
    base=H[:2,:2]+np.diag([4e-8,3e-8+lam*3.])
    expected=base.copy();expected[0,0]+=lam*max(base[0,0]-base[0,1]**2/base[1,1],1.)
    np.testing.assert_array_equal(system['free'],[0,1])
    np.testing.assert_allclose(system['trial_normal'].toarray(),expected,rtol=0,atol=1e-15)
    # This halfspace couples the shared and temporal variable. A shared-only
    # restriction cannot produce the constrained KKT solution below.
    A=np.array([[1.,-2.,0.]])
    result=physical_trial_step(csc_matrix(H),g,D,shared_indices=np.array([0]),
        fixed_indices=np.array([2]),damping=lam,schur_view_lm_ratio=1.,matrix=A,lower=np.array([.2]))
    inverse=np.linalg.inv(expected);a=A[0,:2];unconstrained=-inverse@g[:2]
    multiplier=(.2-a@unconstrained)/(a@inverse@a)
    exact=unconstrained+inverse@a*multiplier
    np.testing.assert_allclose(result.step[:2],exact,rtol=0,atol=2e-10)
    assert result.step[2]==0.
    assert result.report['numerical_original_KKT_passed']
    assert result.report['mode_equalities']==0
    assert result.report['nonlinear_source_acceptance_and_all_original_guards_still_required']


def test_exact_null_elimination_retains_weak_and_physically_coupled_columns():
    H=csc_matrix(np.diag([0.,0.,1e-20,1.]))
    g=np.array([0.,0.,0.,-1.])
    # Column zero is independently zero-feasible. Column one is coupled to
    # the observed column and therefore must remain an ordinary QP variable.
    A=np.array([[1.,0.,0.,0.],[0.,1.,0.,1.]])
    null=exact_separable_null_coordinates(H,g,A,np.array([-1.,.1]))
    np.testing.assert_array_equal(null,[0])


def test_public_solver_uses_complete_current_q_callback_and_original_true_cost(tmp_path):
    import hashlib,inspect
    from types import SimpleNamespace
    from calibration.solver_runtime import install as install_runtime
    install_runtime()
    import z190_joint_source_solver as solver
    from calibration.program_tracking.first_shot_full_physical import install
    calls=[];cost_states=[]
    class Owner:
        shared=np.array([0]);fixed=np.empty(0,dtype=int)
        def __init__(self):
            self.source_producers=[];self.recovery_references=[];self.recover_calls=0
        def checkpoint_identity(self):return {'source':'unchanged coupled toy'}
        def guard(self,q):return np.bool_(q[0]-2*q[1]>=.2-1e-10)
        def recover(self,parent,proposed):
            self.recover_calls+=1
            return dict(q=np.array([.2,0.]),report={'kind':'original recovery'})
        def solver_kwargs(self):
            return dict(global_schur_count=1,global_schur_indices=self.shared,
                fixed_parameter_indices=self.fixed,schur_backtracking=True,
                initial_damping=.1,schur_view_lm_ratio=1.,checkpoint_identity=self.checkpoint_identity(),
                trial_feasibility_recovery=self.recover,
                shared_linear_constraints=lambda q,columns:dict(matrix=np.array([[1.]]),
                    lower=np.array([.2-q[0]+2*q[1]])))
    owner=Owner();driver=SimpleNamespace(solve_joint_source=solver.solve_joint_source)
    original=driver.solve_joint_source;original_helper=original.__globals__['schur_trial_linear_step']
    def domain(q):
        calls.append(q.copy())
        return dict(matrix=np.array([[1.,-2.]]),lower=np.array([.2-q[0]+2*q[1]]),
            q_sha256=hashlib.sha256(q.tobytes()).hexdigest())
    install(owner,driver=driver,entry_reference={'path':__file__,
        'sha256':hashlib.sha256(open(__file__,'rb').read()).hexdigest()},domain_builder=domain)
    assert inspect.signature(driver.solve_joint_source)==inspect.signature(original)
    assert original.__globals__['schur_trial_linear_step'] is original_helper
    def evaluate(q,derivative):
        r=q-np.ones(2)
        if not derivative:cost_states.append(q.copy())
        return (r,csc_matrix(np.eye(2)))if derivative else r
    def guard(q):return bool(q[0]-2*q[1]>=.2-1e-10)
    def review(q,event):
        if event['checkpoint_kind']=='accepted':return dict(stop_branch=True,reason='One verified trial')
    start=np.array([.2,0.])
    result=driver.solve_joint_source(evaluate,start,np.ones(2,bool),np.ones(2),tmp_path,
        guard,checkpoint_callback=review,verbose=False,**owner.solver_kwargs())
    assert result.status=='stopped_by_source_review'
    assert len(calls)==1 and np.array_equal(calls[0],start)
    assert cost_states and all(guard(q)for q in cost_states)
    assert result.history[0]['next_cost']<result.history[0]['cost']
    assert result.history[0]['global_schur_linear_report']['complete_free_physical_trial']
    assert abs(result.q[0]-2*result.q[1]-.2)<1e-9
    again=driver.solve_joint_source(evaluate,result.q,np.ones(2,bool),np.ones(2),tmp_path/'next',
        guard,checkpoint_callback=review,verbose=False,**owner.solver_kwargs())
    assert again.status=='stopped_by_source_review'
    assert len(calls)==2 and np.array_equal(calls[1],result.q)
    assert again.history[0]['next_cost']<again.history[0]['cost']
    assert owner.recover_calls==0  # The complete original guard already passed.
    recovery=owner.solver_kwargs()['trial_feasibility_recovery']
    invalid=np.array([0.,1.]);assert owner.guard(invalid) is np.False_
    repaired=recovery(start,invalid)
    assert owner.recover_calls==1 and repaired['report']['kind']=='original recovery'
    valid=np.array([1.,.2]);preserved=recovery(start,valid)
    assert np.array_equal(preserved['q'].view(np.uint64),valid.view(np.uint64))
    assert owner.recover_calls==1


def test_complete_physical_dispatch_composes_with_exact_source_LM_rejection_retry(tmp_path):
    import hashlib
    from types import SimpleNamespace
    from scipy.sparse import diags
    from calibration.solver_runtime import install as install_runtime
    install_runtime()
    import z190_joint_source_solver as solver
    from calibration.program_tracking.first_shot_prediction_refresh import compile_quality_solver
    from calibration.program_tracking.first_shot_full_physical import install
    class Owner:
        shared=np.array([0]);fixed=np.empty(0,dtype=int)
        def __init__(self):self.source_producers=[];self.recovery_references=[]
        def checkpoint_identity(self):return {'source':'unchanged degenerate source'}
        def solver_kwargs(self):
            return dict(global_schur_count=1,global_schur_indices=self.shared,
                fixed_parameter_indices=self.fixed,schur_backtracking=True,initial_damping=1e-12,
                checkpoint_identity=self.checkpoint_identity(),shared_linear_constraints=lambda q,columns:
                    dict(matrix=np.empty((0,1)),lower=np.empty(0)))
    calls=[]
    def domain(q):
        calls.append(q.copy())
        return dict(matrix=np.empty((0,2)),lower=np.empty(0),
            q_sha256=hashlib.sha256(q.tobytes()).hexdigest())
    driver=SimpleNamespace(solve_joint_source=compile_quality_solver(solver.solve_joint_source,
        retry_rejected_schur=True))
    owner=Owner();install(owner,driver=driver,entry_reference={'path':__file__,
        'sha256':hashlib.sha256(open(__file__,'rb').read()).hexdigest()},domain_builder=domain)
    def evaluate(q,derivative):
        r=np.array([q[0]**2-1.,q[1]])
        return (r,diags([2*q[0],1.],format='csr'))if derivative else r
    def review(q,event):
        if event['checkpoint_kind']=='accepted':return dict(stop_branch=True,reason='One verified step')
    start=np.array([5e-5,0.])
    result=driver.solve_joint_source(evaluate,start,np.array([False,True]),np.ones(2),tmp_path,
        lambda q:True,checkpoint_callback=review,verbose=False,**owner.solver_kwargs())
    assert result.status=='stopped_by_source_review'
    assert result.history[0]['next_cost']<result.history[0]['cost']
    assert result.history[0]['rejected_schur_direction_damping']['rounds']
    assert len(calls)>1 and all(np.array_equal(q,start)for q in calls)
