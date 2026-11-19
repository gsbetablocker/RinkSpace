from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from calibration.program_tracking.body_tracking.programme_source_step import SourceMargins, project_box_halfspaces
from calibration.program_tracking.body_tracking.scoped_motion_domains import local_surface_regressions
from calibration.program_tracking.body_tracking.scoped_motion_step import (
    metric_step, reduce_domains, relative_rotations, target_observations,
)


def test_constraint_is_solved_in_the_reprojection_metric():
    hessian = np.diag([1., 100.])
    gradient = -hessian @ np.array([2., 1.])
    lower, upper = np.full(2, -3.), np.full(2, 3.)
    matrix, slack = csr_matrix([[1., 1.]]), np.array([1.])
    legacy, _ = project_box_halfspaces(np.array([2., 1.]), lower, upper, matrix, slack)
    step, receipt = metric_step(hessian, gradient, lower, upper, matrix, slack, [])
    np.testing.assert_allclose(step, [2/101, 99/101], atol=2e-7)
    assert receipt['max_constraint_excess'] < 1.e-8
    cost = lambda x: x@hessian@x/2 + gradient@x
    assert cost(step) < cost(legacy) - 40.


def test_norm_constraint_includes_motion_tangent_to_the_initial_gradient():
    lower, upper = np.full(2, -3.), np.full(2, 3.)
    # x[1] is invisible to a first-order norm plane at the parent state.
    balls = [(np.eye(2), np.array([1., 0.]), 1., 'flight')]
    step, receipt = metric_step(np.eye(2), np.array([0., -2.]), lower, upper,
                                csr_matrix((0, 2)), np.empty(0), balls)
    np.testing.assert_allclose(step, [-1+1/np.sqrt(5), 2/np.sqrt(5)], atol=2.e-7)
    assert receipt['max_constraint_excess'] < 1.e-8


def test_relative_rotation_uses_both_spline_supports_in_the_saved_right_chart():
    from calibration.program_tracking.body_tracking.absolute_motion_curve import AbsoluteMotionCurve, clamped_cubic_knots, _exp
    from calibration.program_tracking.body_tracking.sequence_skate_state import SequenceState

    knots = clamped_cubic_knots([0., .02, .23, .28, .46, .5])
    count = len(knots)-4
    random = np.random.default_rng(19)
    ref = SimpleNamespace(absolute_curve=AbsoluteMotionCurve(
        knots, np.zeros((count, 3)), _exp(random.normal(scale=.32,size=(count,22,3)))))
    state = SequenceState(ref, (0., .5))
    x = np.r_[np.zeros(17),random.normal(scale=.08,size=state.size)]
    times = [[.21,.31],[.05,.115]]
    tracks = np.array([4,8])
    _, matrix = relative_rotations(state,x,times,tracks,jacobian=True)
    direction = random.normal(size=len(x))
    eps = 2.e-6
    numerical = (relative_rotations(state,x+eps*direction,times,tracks)[0]
                 -relative_rotations(state,x-eps*direction,times,tracks)[0])/(2*eps)
    np.testing.assert_allclose(matrix@direction,numerical.ravel(),rtol=3.e-6,atol=3.e-8)
    actual_tracks = ((matrix.indices-20)%69)//3
    assert set(actual_tracks) == set(tracks)


def test_cavity_norm_only_reduces_frozen_nonzero_residual_without_halfspaces():
    ball = dict(source_id='right_foot_cavity', labels=['right_foot_cavity'],
                offset=np.array([[3.,4.]]),radius=[5.],jacobian=csr_matrix([[1.],[0.]]))
    domains = SourceMargins(csr_matrix((0,1)),np.empty(0),[],[ball])
    J, b, _, balls = reduce_domains(domains,np.array([0]),np.array([1.]),np.array([-2.]),np.array([2.]))
    assert J.shape==(0,1) and b.shape==(0,)
    jac, residual, radius, label = balls[0]
    np.testing.assert_allclose(jac, [[1./3]])
    np.testing.assert_allclose(residual, [1.])
    np.testing.assert_allclose(radius, 1.)
    assert label=='right_foot_cavity'


def test_source_only_holdout_does_not_require_a_detector_derivative():
    body=SimpleNamespace(frames=np.array([0,1]),of=np.array([0]),model_oj=np.array([14]))
    problem=SimpleNamespace(body=body,reference=SimpleNamespace(p={'pts':np.array([1039104,1039360])}))
    fit=dict(pts=1039104,joint=14)
    holdout=dict(pts=1039360,joint=15,fit=False)
    frames,joints,points,rows=target_observations(problem,{'image_regions':[fit,holdout]})
    np.testing.assert_array_equal(frames,[0])
    np.testing.assert_array_equal(joints,[14])
    assert points==[0] and rows==[fit]
    with pytest.raises(ValueError,match='1039360'):
        target_observations(problem,{'image_regions':[fit,dict(holdout,protect_projection=True)]})


def test_collar_recheck_uses_each_reviewed_sample_and_semantic_pair():
    geometry=SimpleNamespace(part_names=['left_shank','left_boot','right_boot'],parts=np.array([0,1,2]))
    scan=dict(source_seconds=np.array([81.38,81.39,81.40]),sample=np.array([0,1,1,2]),
              face_a=np.array([0,1,0,1]),face_b=np.array([1,0,2,0]),initial_gap=np.array([-.007,-.005,-.04,-.03]))
    caps=[dict(parts=['left_shank','left_boot'],source_seconds=81.38,max_penetration_m=.009),
          dict(parts=['left_shank','left_boot'],source_seconds=81.39,max_penetration_m=0.)]
    assert local_surface_regressions(scan,geometry,caps)==[dict(**caps[1],measured_maximum_m=.005)]
