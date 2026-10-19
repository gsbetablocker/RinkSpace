import numpy as np
import pytest
from scipy.sparse import csc_matrix,csr_matrix
from calibration.solver_runtime.source_parameter_coordinates import conditional_coordinates,conditional_physical_trial_step
from calibration.solver_runtime.full_free_physical import physical_trial_step


def test_invertible_step_q_and_full_jacobian_preserve_model_and_guard():
    J=np.array([[1.,.5,2.,0.],[0.,2.,-.1,1.],[1.,1.,.2,-.3],[2.,-1.,.1,.4],[.3,.2,-1.,.2]])
    H=J.T@J;c=conditional_coordinates(H,anchor_indices=[0,1],coupled_indices=[2])
    z=np.array([.4,-.2,.07,.8]);delta=c.push_step(z);q0=np.array([3.,1.,.8,.2]);q=c.decode(z,q0)
    np.testing.assert_allclose(c.pull_step(delta),z,rtol=1e-14,atol=1e-14)
    np.testing.assert_allclose(c.encode(q,q0),z,rtol=1e-14,atol=1e-14)
    np.testing.assert_allclose(c.pull_jacobian(J)@z,J@delta,rtol=1e-14,atol=1e-14)
    np.testing.assert_allclose(c.pull_normal(H).toarray(),(J@c.matrix).T@(J@c.matrix),rtol=1e-14,atol=1e-14)
    assert abs(c.report['anchor_coupled_cross_max_absolute'])<1e-14
    # Source and nonlinear guard are still evaluated on the very same q.
    assert float(np.sin(q[0])+q[2]**2)==float(np.sin((q0+delta)[0])+(q0+delta)[2]**2)
    assert np.array_equal(q>0.,(q0+delta)>0.)


def test_singular_anchor_refused_and_weak_coupled_direction_not_truncated():
    with pytest.raises(np.linalg.LinAlgError):conditional_coordinates([[0.,0.],[0.,1.]],anchor_indices=[0],coupled_indices=[1])
    epsilon=1e-10;c=conditional_coordinates([[1.,1.],[1.,1.+epsilon]],anchor_indices=[0],coupled_indices=[1])
    assert c.matrix.shape==(2,2)
    assert c.pull_normal([[1.,1.],[1.,1.+epsilon]])[1,1]>0.
    assert c.report['all_coordinates_and_all_weak_directions_retained']


def _problem():
    rng=np.random.default_rng(1017);J=rng.normal(size=(16,5));J[:,2]=J[:,0]+.05*J[:,2];J[:,4]=0.
    H=csc_matrix(J.T@J);g=J.T@rng.normal(size=16);A=csr_matrix(np.r_[np.eye(5),-np.eye(5)]);b=np.full(10,-.4)
    return H,g,np.maximum(H.diagonal(),1.),A,b


def test_preserved_original_Q_direction_objective_constraints_and_fixed_match():
    H,g,d,A,b=_problem();kw=dict(shared_indices=[0,1,2,4],fixed_indices=[3],damping=1e-3,
        schur_view_lm_ratio=1.,matrix=A,lower=b)
    old=physical_trial_step(H,g,d,**kw)
    new=conditional_physical_trial_step(H,g,d,anchor_indices=[0],coupled_indices=[2],metric='preserve',**kw)
    np.testing.assert_allclose(new.step,old.step,rtol=2e-8,atol=2e-8)
    assert new.step[3]==new.step[4]==0.
    assert new.report['canonical_KKT_passed']
    assert new.report['source_null_free_coordinates_exactly_eliminated']==[3]
    np.testing.assert_allclose(A@new.step,A@old.step,rtol=2e-8,atol=2e-8)


def test_rebuilt_diagonal_LM_is_explicit_and_can_amplify_weak_directions():
    epsilon=1e-6;H=csc_matrix([[1e8,1e8],[1e8,1e8*(1+epsilon)]]);g=np.array([-1.,-101.]);d=H.diagonal()
    kw=dict(shared_indices=[0,1],damping=1e-3,matrix=csr_matrix((0,2)),lower=np.empty(0),anchor_indices=[0],coupled_indices=[1])
    old=conditional_physical_trial_step(H,g,d,metric='preserve',**kw)
    rebuilt=conditional_physical_trial_step(H,g,d,metric='rebuild_lm',**kw)
    assert abs(rebuilt.step[1])>100*abs(old.step[1])
    assert rebuilt.report['numerical_metric']=='rebuild_lm'
    assert rebuilt.report['metric_is_numerical_trust_strategy_not_physical_prior']


def test_constraint_coupling_prevents_false_source_null_elimination():
    H=csc_matrix([[4.,1.,0.],[1.,2.,0.],[0.,0.,0.]]);g=np.array([-2.,.2,0.]);A=csr_matrix([[1.,0.,1.],[0.,0.,-1.]])
    b=np.array([.3,0.]);d=np.maximum(H.diagonal(),1.)
    result=conditional_physical_trial_step(H,g,d,shared_indices=[0,1,2],damping=1e-3,matrix=A,lower=b,
        anchor_indices=[0],coupled_indices=[1])
    assert not result.report['source_null_free_coordinates_exactly_eliminated']
    assert np.min(A@result.step-b)>-1e-8


def test_rejects_fixed_or_overlapping_chart_columns():
    with pytest.raises(ValueError):conditional_coordinates(np.eye(2),anchor_indices=[0],coupled_indices=[0])
    with pytest.raises(ValueError):conditional_physical_trial_step(np.eye(2),np.ones(2),np.ones(2),
        shared_indices=[0],fixed_indices=[1],damping=1e-3,matrix=csr_matrix((0,2)),lower=np.empty(0),
        anchor_indices=[0],coupled_indices=[1])
