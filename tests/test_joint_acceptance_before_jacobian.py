"""Real SciPy TRF acceptance, with a two-variable objective and no body assets."""
import io
import json
import time
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from calibration.program_tracking.body_tracking.body_execution import BodyExecution, BodyExecutionConfig
from calibration.program_tracking.body_tracking.skate_integration import JointContactProblem
from calibration.program_tracking.body_tracking.skate_run_policy import RunPolicy


def run_problem(output, *, early, budget, initial=(0., 0.), nonlinear=False,bounded=False):
    problem=object.__new__(JointContactProblem)
    problem.output=output;problem.log=io.StringIO();problem.start=time.perf_counter()
    problem.outer=problem.accepted=0;problem.size=19;problem.state_kind='test'
    problem.optimize_shared=False;problem.record_acceptance_before_jacobian=early
    problem.binding=SimpleNamespace(x=np.zeros(17),base_parameters=None,collar_constraints_enabled=False)
    problem.body=SimpleNamespace(size=2,state=SimpleNamespace(active=np.array([0,3] if bounded else [0,1]),m=1,seconds=(0.,1.),use_pose_cost_guard=False))
    problem.body.execution=BodyExecution(BodyExecutionConfig(backend="cpu",workers=1))
    problem.ice_policy=SimpleNamespace();problem.al=SimpleNamespace(penalty=10.,multiplier=np.zeros(0))
    problem.physics_times=np.zeros(0)
    problem.cache=problem.cache_x=None;problem.jacobian_states=[];problem.fun_states=[]
    problem.checkpoint_fields=lambda x:dict(x=x.copy())

    def evaluate(x,jacobian=False):
        if problem.cache_x is not None and np.array_equal(x,problem.cache_x) and (not jacobian or problem.cache['jac'] is not None):
            return problem.cache
        u,v=x[17:]
        if nonlinear:
            residual=np.array([10*(v-u*u),1-u]);local=np.array([[-20*u,10.],[-1.,0.]])
        else:
            residual=x[17:]-np.array([3.,4.]);local=np.eye(2)
        cost=float(residual@residual/2)
        if jacobian:problem.jacobian_states.append(x[17:].copy())
        else:problem.fun_states.append(x[17:].copy())
        problem.cache_x=x.copy()
        problem.cache=dict(residual=residual,jac=csr_matrix(np.c_[np.zeros((2,17)),local]) if jacobian else None,
            parts=dict(data=cost),raw_data_parts=dict(body_pixels=cost),image_cost=cost,pose_cost=0.,
            gate=dict(feasible=True,max_violation_m=0.,rows_over_tolerance=0,tested_rows=0,collar_unknown_sections=0))
        return problem.cache

    problem.evaluate=evaluate
    result,report=problem.solve(initial=np.r_[np.zeros(17),initial],
        run_policy=RunPolicy(wall_seconds=1000,max_accepted=budget,stagnation_window=100))
    events=[json.loads(s) for s in problem.log.getvalue().splitlines()]
    assert report['selection']['retained_step']==problem.accepted
    assert len([e for e in events if e.get('event')=='accepted'])==problem.accepted
    with np.load(output/'candidate.npz') as saved:np.testing.assert_array_equal(saved['x'],result)
    return problem,result,report


@pytest.mark.parametrize('budget',[1,2,3])
@pytest.mark.parametrize('bounded',[False,True])
def test_budget_retains_same_accepted_state_without_terminal_jacobian(tmp_path,budget,bounded):
    legacy,x0,r0=run_problem(tmp_path/'legacy',early=False,budget=budget,bounded=bounded)
    early,x1,r1=run_problem(tmp_path/'early',early=True,budget=budget,bounded=bounded)
    assert legacy.accepted==early.accepted==budget
    np.testing.assert_allclose(x0,x1,rtol=0,atol=1e-14)
    assert len(legacy.jacobian_states)==budget+1
    assert len(early.jacobian_states)==budget
    assert r0['stop']['reasons'][0]['code']==r1['stop']['reasons'][0]['code']=='accepted_budget'


def test_initial_jacobian_is_not_an_accepted_step(tmp_path):
    problem,result,_=run_problem(tmp_path/'initial',early=True,budget=1)
    assert problem.accepted==1
    assert len(problem.jacobian_states)==1
    np.testing.assert_array_equal(problem.jacobian_states[0],np.zeros(2))
    assert np.linalg.norm(result[17:])>0


def test_rejected_trial_fun_states_do_not_count_as_accepted(tmp_path):
    legacy,x0,_=run_problem(tmp_path/'legacy',early=False,budget=5,initial=(-2.,1.),nonlinear=True)
    early,x1,_=run_problem(tmp_path/'early',early=True,budget=5,initial=(-2.,1.),nonlinear=True)
    assert legacy.accepted==early.accepted==5
    np.testing.assert_allclose(x0,x1,rtol=0,atol=1e-14)
    # Includes the initial and final reporting calls; an additional fun state
    # without a corresponding J is an actually rejected TRF candidate.
    linearized=legacy.jacobian_states
    rejected=[x for x in legacy.fun_states if not any(np.array_equal(x,j) for j in linearized)]
    assert rejected
