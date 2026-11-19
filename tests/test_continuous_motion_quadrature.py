"""The actual curve, including motion hidden between exposures, enters priors."""
import numpy as np
import pytest
from scipy.linalg import null_space

from test_action_objective_context import _sequence_problem
from calibration.program_tracking.body_tracking.absolute_motion_curve import (
    AbsoluteMotionCurve, _exp, clamped_cubic_knots,
)
from calibration.program_tracking.body_tracking.absolute_motion_state import evaluate_curve
from calibration.program_tracking.body_tracking.continuous_motion_factors import ActualMotionFactors,ActualSkateAngularAcceleration,ActualSkateInclinationRate,ActualSkateAnkleAcceleration
from calibration.program_tracking.body_tracking.sequence_skate_state import SequenceState


@pytest.mark.parametrize('loss',['pseudo_huber','quadratic'])
def test_dense_motion_actual_state_directional_derivative(loss):
    problem, x = _sequence_problem()
    problem.reference.source_document = dict(generic_context=True)
    native_pts = problem.reference.p['pts'].copy()
    factor = ActualMotionFactors(problem, problem.reference, substeps=4,
                                 rotation_acceleration_loss=loss)
    value = factor.evaluate(x, True)
    rng = np.random.default_rng(29)
    direction = rng.normal(size=len(x))
    direction /= np.linalg.norm(direction)
    step = 1.e-6
    numeric = (factor.evaluate(x+step*direction)['residual']-
               factor.evaluate(x-step*direction)['residual'])/(2*step)
    analytic = value['jac'][:, 17:] @ direction
    np.testing.assert_allclose(analytic, numeric, atol=2.e-7, rtol=2.e-5)
    np.testing.assert_array_equal(problem.reference.p['pts'], native_pts)
    assert len(factor.times) == 4*(len(native_pts)-1)+1
    assert not factor.config['quadrature_is_source_observation']


def test_motion_between_identical_native_poses_is_not_invisible():
    problem, _ = _sequence_problem()
    ref = problem.reference
    ref.times = np.arange(5)*.02
    ref.p['pts'] = 450560 + np.arange(5, dtype=np.int64)*256
    knots = clamped_cubic_knots(np.linspace(0., .08, 17))
    count = len(knots)-4
    zero = AbsoluteMotionCurve(knots, np.zeros((count, 3)),
                               np.broadcast_to(np.eye(3), (count, 22, 3, 3)).copy())
    # A smooth nonzero curve in the nullspace of all native pose samples.
    angles = null_space(zero.basis(ref.times))[:, 3]
    angles *= .7/np.max(np.abs(angles))
    vectors = np.zeros((count, 22, 3))
    vectors[:, 7, 1] = angles
    curve = AbsoluteMotionCurve(knots, zero.position_controls, _exp(vectors))
    ref.absolute_curve = curve
    ref.at = lambda times: evaluate_curve(curve, times)
    ref.source_document = dict(generic_context=True)
    problem.body.state = SequenceState(ref, (35.2, 35.28))
    problem.body.size = problem.body.state.size
    x = np.zeros(problem.body.size)
    native = ActualMotionFactors(problem, ref).evaluate(x)
    dense = ActualMotionFactors(problem, ref, substeps=4).evaluate(x)
    quadratic = ActualMotionFactors(problem, ref, substeps=4,
        rotation_acceleration_loss='quadratic').evaluate(x)
    assert native['parts']['actual_so3_angular_acceleration'] < 1.e-15
    assert dense['parts']['actual_so3_angular_acceleration'] > 1.
    assert quadratic['parts']['actual_so3_angular_acceleration'] > dense['parts']['actual_so3_angular_acceleration']


@pytest.mark.parametrize('feet',[(0,1),(0,),(1,)])
def test_composed_world_skate_acceleration_derivative_including_relative_mount(feet):
    problem,x=_sequence_problem()
    times=np.linspace(problem.reference.times[0],problem.reference.times[-1],257)
    factor=ActualSkateAngularAcceleration(problem,times,feet=feet)
    value=factor.evaluate(x,True)
    rng=np.random.default_rng(714)
    direction=rng.normal(size=len(x));direction/=np.linalg.norm(direction)
    step=1e-6
    numeric=(factor.evaluate(x+step*direction)['residual']-
             factor.evaluate(x-step*direction)['residual'])/(2*step)
    np.testing.assert_allclose(value['jac'][:,17:]@direction,numeric,atol=3e-7,rtol=3e-5)
    assert value['parts'][factor.name]>0


@pytest.mark.parametrize('feet',[(0,1),(0,),(1,)])
def test_skate_inclination_derivative_with_articulated_and_mount_motion(feet):
    problem,x=_sequence_problem()
    times=np.linspace(problem.reference.times[0],problem.reference.times[-1],257)
    factor=ActualSkateInclinationRate(problem,times,feet=feet)
    value=factor.evaluate(x,True)
    rng=np.random.default_rng(719);direction=rng.normal(size=len(x))
    direction/=np.linalg.norm(direction);step=1e-6
    numeric=(factor.evaluate(x+step*direction)['residual']-
             factor.evaluate(x-step*direction)['residual'])/(2*step)
    np.testing.assert_allclose(value['jac'][:,17:]@direction,numeric,atol=2e-8,rtol=2e-5)


def test_skate_inclination_allows_fast_yaw_and_arbitrary_constant_lean():
    problem,_=_sequence_problem();ref=problem.reference
    base=ref.absolute_curve;count=len(base.position_controls)
    rotations=np.broadcast_to(base.rotation_controls[0],base.rotation_controls.shape).copy()
    yaw=np.c_[np.zeros(count),np.zeros(count),np.linspace(0.,4.,count)]
    rotations[:,0]=_exp(yaw)@rotations[:,0]
    curve=AbsoluteMotionCurve(base.knots,base.position_controls,rotations)
    ref.absolute_curve=curve
    shoe=ref.relative_skate_curve
    ref.relative_skate_curve=AbsoluteMotionCurve(shoe.knots,shoe.position_controls,
        np.broadcast_to(shoe.rotation_controls[0],shoe.rotation_controls.shape).copy())
    def at(times):
        s=evaluate_curve(curve,times)
        s['skate_relative_R']=evaluate_curve(ref.relative_skate_curve,times)['rotations']
        return s
    ref.at=at;problem.body.state=SequenceState(ref,(35.2,36.48))
    problem.body.size=problem.body.state.size
    x=np.zeros(problem.body.size)
    factor=ActualSkateInclinationRate(problem,np.linspace(0.,1.28,257))
    assert np.linalg.norm(factor.evaluate(x)['residual'])<1e-11
    # A heel/toe rocking motion remains visible even when the source is
    # foreshortened. It must not inherit the yaw exemption.
    controls=x[:problem.body.state.body_size].reshape(-1,69)
    controls[:,7]=.25*np.sin(np.linspace(0.,3*np.pi,count))
    assert np.linalg.norm(factor.evaluate(x)['residual'])>.01


@pytest.mark.parametrize('feet',[(0,1),(0,),(1,)])
def test_actual_ankle_acceleration_derivative(feet):
    problem,x=_sequence_problem()
    times=np.linspace(problem.reference.times[0],problem.reference.times[-1],257)
    factor=ActualSkateAnkleAcceleration(problem,times,feet=feet)
    value=factor.evaluate(x,True)
    rng=np.random.default_rng(720);direction=rng.normal(size=len(x))
    direction/=np.linalg.norm(direction);step=1e-6
    numeric=(factor.evaluate(x+step*direction)['residual']-
             factor.evaluate(x-step*direction)['residual'])/(2*step)
    np.testing.assert_allclose(value['jac'][:,17:]@direction,numeric,atol=2e-7,rtol=2e-5)
    # Rotating around the actual ankle pivot cannot translate that pivot.
    for foot in (0,1):
        columns=np.arange(problem.body.state.m)*69+3+3*(7+foot)
        for axis in range(3):
            np.testing.assert_array_equal(value['jac'][:,17+columns+axis].toarray(),0.)
