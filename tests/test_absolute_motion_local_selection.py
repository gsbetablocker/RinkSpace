"""A local repair keeps the same global curve and exact outside state."""
from types import SimpleNamespace
import numpy as np
from calibration.program_tracking.body_tracking.absolute_motion_curve import AbsoluteMotionCurve,clamped_cubic_knots,_exp
from calibration.program_tracking.body_tracking.absolute_motion_state import AbsoluteMotionState,evaluate_curve,with_warm_rotation_chart
from calibration.program_tracking.body_tracking.spin_continuous_continue import _configure_local_controls


def test_local_controls_leave_all_outside_pose_samples_bit_exact():
    rng=np.random.default_rng(281);knots=clamped_cubic_knots(np.linspace(0,1,17));m=len(knots)-4
    curve=AbsoluteMotionCurve(knots,np.zeros((m,3)),_exp(rng.normal(size=(m,22,3))*.1))
    state=AbsoluteMotionState(SimpleNamespace(absolute_curve=curve),(0,1))
    curve=with_warm_rotation_chart(curve,state.packet(np.zeros(state.size)),lift=True)
    state=AbsoluteMotionState(SimpleNamespace(absolute_curve=curve),(0,1))
    problem=SimpleNamespace(body=SimpleNamespace(state=state),reference=SimpleNamespace(p={'pts':np.array([76*12800])}),checkpoint_extra={})
    record=_configure_local_controls(problem,[76.2,76.8]);x=np.zeros(state.size)
    x[problem.active_parameter_indices-17]=rng.normal(size=len(problem.active_parameter_indices))*.03
    clock=np.linspace(0,1,501);old=evaluate_curve(curve,clock);new=evaluate_curve(state.current_curve(x),clock)
    outside=(clock<.2)|(clock>.8)
    for key in ['P','Q','local_R']:np.testing.assert_array_equal(old[key][outside],new[key][outside])
    assert np.linalg.norm(new['local_R'][~outside]-old['local_R'][~outside])>0
    assert record['active_variables']<state.size
    assert np.all(problem.active_parameter_indices>=17)
