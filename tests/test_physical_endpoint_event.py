import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from calibration.solver_runtime import install
install()
import z190_joint_source_solver as solver
from z190_streaming_source_normal import SourceRowBlock
from structured_programme_source_candidate.source_step_curvature_tightening import bracketed_physical_endpoint_event
from structured_programme_source_candidate.finite_endpoint_restoration_step_constraints import authenticated_open_curve

PATH=Path(__file__).resolve().parents[1]/'calibration/solver_runtime/legacy/structured_programme_source_candidate/normal_scanline_human_query_acceleration.py'
spec=importlib.util.spec_from_file_location('endpoint_event_actual_query',PATH)
kernel=importlib.util.module_from_spec(spec);spec.loader.exec_module(kernel)


def test_actual_finite_rectangle_event_returns_endpoint_side_and_can_release():
    # The nearest point starts strictly inside a real open straight primitive.
    # Translating it across its first endpoint changes the actual finite query,
    # without inventing a source endpoint correspondence or expanding support.
    curve=np.array([[1.001,.999],[.001,1.999]])
    uv=np.array([[0.,0.]]);normal=np.array([[1.,0.]])
    bounds=np.zeros((1,2));tangent=bounds.copy()
    def queries(alpha):
        moved=curve+np.array([-alpha,alpha])
        return [kernel.finite_source_rectangle_projected_queries(moved,np.ones(2),
            np.ones(2,bool),np.ones(1,bool),uv,normal,bounds,tangent_bounds=tangent)]
    context=dict(group=0,id='physical_open_line',actual_pts=7,vertices=np.array([0]),
        first=np.array([True]),last=np.array([False]),last_segment=0)
    event=bracketed_physical_endpoint_event([context],queries)
    assert event is not None
    lo=event['report']['interior_fraction'];hi=event['fraction']
    assert np.nextafter(lo,np.inf)==hi
    assert queries(lo)[0]['nominal_finite_segment_categories'][0]==0
    assert queries(hi)[0]['nominal_finite_segment_categories'][0]==1
    assert hi==pytest.approx(.001,abs=1e-15)
    np.testing.assert_array_equal(bounds,0.)
    # Moving the endpoint towards its original source is still legal: no
    # permanent station halfspace was introduced by locating the event.
    clamped=queries(.002)[0]['source_support_distance_px'][0]
    released=queries(0.)[0]['source_support_distance_px'][0]
    assert released<=clamped


def test_only_declared_open_physical_primitives_are_eligible():
    world=np.array([[0.,0.,0.],[1.,0.,0.]])
    assert authenticated_open_curve({'kind':'current_ice_segment'},world)
    assert not authenticated_open_curve({'kind':'fixed_circle','layer':'outer'},world)
    assert not authenticated_open_curve({'kind':'circle'},world)
    assert not authenticated_open_curve({'kind':'current_ice_segment'},np.r_[world,world[:1]])
    assert bracketed_physical_endpoint_event([],lambda alpha:[])is None


def test_next_direction_is_fresh_after_event_and_does_not_repeat_tiny_seed(tmp_path):
    boundary=.01;target=.4;fresh=[];accepted=[]
    def position(x):return x if x<boundary else boundary+100.*(x-boundary)
    def blocks(q,derivative):
        if derivative:fresh.append(float(q[0]))
        J=csr_matrix([[1. if q[0]<boundary else 100.,0.],[0.,1.]])if derivative else None
        yield SourceRowBlock(np.array([position(q[0])-target,q[1]]),J,np.ones(2,bool),np.ones(2),
            dict(family='piecewise_physical_endpoint',original_start=0,original_stop=2))
    def event(q,step,maximum):
        if q[0]>=boundary or q[0]+maximum*step[0]<boundary:return None
        return dict(fraction=(boundary-q[0])/step[0],report={
            'actual_finite_endpoint_category_verified':True,'next_complete_source_linearization_must_be_fresh':True})
    def guard(q):
        if q[0]>.015:raise ValueError('Actual finite physical endpoint support exceeded')
    def review(q,event):
        if event['checkpoint_kind']=='accepted':
            accepted.append(float(q[0]))
            if len(accepted)==2:return {'stop_branch':True,'reason':'Two actual source accepted directions checked'}
    def unused(*args):raise AssertionError('Use the original streaming source')
    result=solver.solve_joint_source(unused,np.zeros(2),np.ones(2,bool),np.ones(2),tmp_path,guard,
        source_block_factory=blocks,cache_native_source_replay=True,reuse_source_jacobian=True,
        global_schur_count=1,schur_backtracking=True,schur_backtracking_aware_damping=True,
        adaptive_source_backtracking=True,source_physical_endpoint_event=event,
        checkpoint_callback=review,verbose=False)
    assert len(accepted)==2 and accepted[0]==boundary and accepted[1]>boundary+.002
    assert boundary in fresh
    first,second=result.history
    assert first['actual_physical_endpoint_transition']['complete_source_J_refresh_required_before_next_direction']
    assert second['source_jacobian_reuse']['exact_current_J']
    assert second['adaptive_source_backtracking']['initial_line_search_fraction']==1.
    assert 'actual_physical_endpoint_transition'not in second
    assert second['next_cost']<first['next_cost']<first['cost']
