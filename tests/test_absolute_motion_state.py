"""Directional/clock regressions for the production absolute sparse chain."""
from types import SimpleNamespace
import numpy as np
from calibration.program_tracking.body_tracking.absolute_motion_curve import AbsoluteMotionCurve,clamped_cubic_knots,_exp
from calibration.program_tracking.body_tracking.absolute_motion_state import AbsoluteMotionState
from calibration.program_tracking.body_tracking.continuous_motion_factors import track_local

def fixture():
    rng=np.random.default_rng(1105);knots=clamped_cubic_knots(np.linspace(0,1,7));m=len(knots)-4
    curve=AbsoluteMotionCurve(knots,rng.normal(size=(m,3)),_exp(rng.normal(size=(m,22,3))*.2))
    state=AbsoluteMotionState(SimpleNamespace(absolute_curve=curve),(0,1))
    return rng,state,rng.normal(size=state.size)*.08,np.linspace(.02,.98,13)

def test_sparse_chain_direction_with_shuffled_repeated_sample_clock():
    rng,state,x,t=fixture();pick=np.array([12,2,2,0,8,6]);local=rng.normal(size=(len(pick),4,69));direction=rng.normal(size=state.size);direction/=np.linalg.norm(direction)
    delta=state.increments(x,t);J=delta.chain(local,sample_indices=pick)
    h=1e-6;numeric=np.einsum('nkd,nd->nk',local,(state.increments(x+h*direction,t)-state.increments(x-h*direction,t))[pick]/(2*h)).ravel()
    np.testing.assert_allclose(J@direction,numeric,rtol=2e-6,atol=2e-8)
    assert J.has_canonical_format

def test_rotation_blocks_equal_general_chain_after_other_state_evaluation():
    rng,state,x,t=fixture();delta=state.increments(x,t);pick=np.array([11,2,0,12,7]);blocks=rng.normal(size=(len(pick),22,3,3))
    state.increments(x+.1,t[::-1])
    block=delta.rotation_chain(blocks,sample_indices=pick);general=delta.chain(track_local(blocks),sample_indices=pick)
    np.testing.assert_allclose(block.toarray(),general.toarray(),rtol=0,atol=1e-14)
    np.testing.assert_array_equal(delta.parameters,x)

def test_retracted_state_does_not_reuse_controls_after_caller_mutates_x():
    rng,state,x,t=fixture();first=state.increments(x,t).copy();x[6:9]+=.2;changed=state.increments(x,t)
    assert np.linalg.norm(changed-first)>0

def test_exposure_chain_equals_five_node_sparse_sum():
    from calibration.program_tracking.body_tracking.absolute_motion_state import chain_exposure
    from scipy.sparse import csr_matrix
    rng,state,x,t=fixture();t=np.linspace(.03,.97,35);delta=state.increments(x,t)
    indices=np.arange(35).reshape(7,5)[[6,2,0,4]];local=rng.normal(size=(4,2,5,69))
    old=csr_matrix((8,state.size))
    for node in range(5):old+=delta.chain(local[:,:,node],sample_indices=indices[:,node])
    new=chain_exposure(local,delta,sample_indices=indices)
    np.testing.assert_allclose(new.toarray(),old.toarray(),rtol=1e-13,atol=1e-13)
    assert new.has_canonical_format
