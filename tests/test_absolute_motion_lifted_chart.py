"""A fitted control edge can cross pi without flipping its archived motion."""
from types import SimpleNamespace
import numpy as np
import pytest
from calibration.program_tracking.body_tracking.absolute_motion_curve import AbsoluteMotionCurve,clamped_cubic_knots,_exp,_log
from calibration.program_tracking.body_tracking.absolute_motion_state import (AbsoluteMotionState,evaluate_curve,curve_from_packet,with_warm_rotation_chart,LIFTED_SCHEMA)


def fixture():
    knots=clamped_cubic_knots(np.linspace(0,1,5));m=len(knots)-4
    rv=np.zeros((m,22,3));rv[3,15,2]=np.pi-2e-4
    seed=AbsoluteMotionCurve(knots,np.zeros((m,3)),_exp(rv))
    state=AbsoluteMotionState(SimpleNamespace(absolute_curve=seed),(0,1))
    x=np.zeros(state.size);saved=state.packet(x)
    lifted=with_warm_rotation_chart(seed,saved,lift=True)
    return seed,lifted,x


def test_exact_warm_replay_and_continuous_pi_crossing():
    seed,lifted,x=fixture();t=np.linspace(0,1,101)
    for key in ['P','Q','local_R']:
        np.testing.assert_allclose(evaluate_curve(seed,t)[key],evaluate_curve(lifted,t)[key],atol=3e-14,rtol=0)
    d=np.zeros((len(seed.position_controls),22,3));d[3,15,2]=3e-4
    before=lifted.retracted(np.zeros_like(seed.position_controls),-d)
    after=lifted.retracted(np.zeros_like(seed.position_controls),d)
    change=_log(evaluate_curve(before,t)['local_R'].swapaxes(-1,-2)@evaluate_curve(after,t)['local_R'])
    assert np.max(np.linalg.norm(change,axis=-1))<1e-3
    assert np.max(np.linalg.norm(after._check_log_chart(),axis=-1))>np.pi


def test_lifted_sparse_chain_and_checkpoint_replay():
    seed,lifted,x=fixture();state=AbsoluteMotionState(SimpleNamespace(absolute_curve=lifted),(0,1))
    x.reshape(state.m,69)[3,3+15*3+2]=4e-4
    rng=np.random.default_rng(110507);t=np.linspace(.1,.9,15);direction=rng.normal(size=len(x));direction/=np.linalg.norm(direction)
    local=rng.normal(size=(len(t),3,69));delta=state.increments(x,t);J=delta.chain(local)
    h=1e-6;numeric=np.einsum('nkd,nd->nk',local,(state.increments(x+h*direction,t)-state.increments(x-h*direction,t))/(2*h)).ravel()
    np.testing.assert_allclose(J@direction,numeric,rtol=4e-6,atol=3e-8)
    packet=state.packet(x);assert str(packet['state_schema_version'])==LIFTED_SCHEMA
    restored=curve_from_packet(packet)
    for key in ['P','Q','local_R']:
        np.testing.assert_array_equal(evaluate_curve(restored,t)[key],evaluate_curve(state.current_curve(x),t)[key])
    del packet['absolute_rotation_edge_reference_quaternion']
    with pytest.raises(ValueError,match='exact archived'):curve_from_packet(packet)


def test_old_principal_schema_is_not_silently_upgraded():
    seed,_,x=fixture();state=AbsoluteMotionState(SimpleNamespace(absolute_curve=seed),(0,1));packet=state.packet(x)
    restored=curve_from_packet(packet);assert restored.edge_reference_quaternion is None
    assert with_warm_rotation_chart(seed,packet).edge_reference_quaternion is None
