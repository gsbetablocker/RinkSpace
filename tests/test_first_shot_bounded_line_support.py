from copy import deepcopy
import numpy as np
import pytest
from calibration.program_tracking.first_shot_bounded_line_support import bounded_lane_support


def _problem(q,edge='edge0'):
    a=np.r_[q[:2],0.];b=np.r_[q[2:4],0.];goal=np.array([[-2.+q[4],-4.,0.],[-2.+q[4]+q[5],4.,0.]])
    points=a+np.linspace(0,1,5)[:,None]*(b-a)
    J=np.zeros((5,3,6));t=np.linspace(0,1,5)
    J[:,0,0]=1-t;J[:,1,1]=1-t;J[:,0,2]=t;J[:,1,3]=t
    G=np.zeros((2,3,6));G[:,0,4]=1.;G[1,0,5]=1.
    feature=dict(id='lane1',polygon_left_vertices_role='unmeasured_display_proxy_not_physical_endpoints',
        left_endpoint_status='unresolved',left_endpoint_coordinate_roles=dict(upper='unmeasured_display_proxy',lower='unmeasured_display_proxy'),
        left_endpoint_x_intervals_scene_units=dict(upper=[-2.,0.],lower=[-2.,0.]),
        left_endpoint_x_interval_bound_sources=dict(lower_x=dict(feature_id='goal_line_left',geometry='segments_xyz[*][*][0]')))
    scene=dict(ice_features=[feature,dict(id='goal_line_left',segments_xyz=[goal])])
    return scene,dict(kind='ice_curve',curve_id='lane1:'+edge),points,J,G


def test_finite_allowed_support_preserves_real_other_end_without_changing_scene():
    scene,target,w,J,G=_problem(np.array([0.,.3,3.,.3,0.,0.]));before=deepcopy(scene)
    out,jac,r=bounded_lane_support(scene,target,w,J,goal_world_jac=G)
    np.testing.assert_array_equal(out[-1],w[-1]);np.testing.assert_allclose(out[0],[-2.,.3,0.])
    np.testing.assert_array_equal(jac[-1],J[-1]);assert r['support_finite']and not r['extended_to_rink_boundary']
    assert np.array_equal(scene['ice_features'][1]['segments_xyz'],before['ice_features'][1]['segments_xyz'])
    assert r['unknown_endpoint_profiled_not_measured']


@pytest.mark.parametrize('edge,q',[('edge0',[0.,.3,3.,.6,.1,.2]),('edge2',[3.,-.2,0.,-.4,.1,.2])])
def test_current_finite_cap_and_resampling_full_derivative(edge,q):
    q=np.array(q);scene,target,w,J,G=_problem(q,edge);out,jac,_=bounded_lane_support(scene,target,w,J,goal_world_jac=G)
    for k in range(len(q)):
        h=1e-6;dq=np.eye(len(q))[k]*h
        plus=_problem(q+dq,edge);minus=_problem(q-dq,edge)
        a=bounded_lane_support(plus[0],plus[1],plus[2])[0];b=bounded_lane_support(minus[0],minus[1],minus[2])[0]
        np.testing.assert_allclose(jac[:,:,k],(a-b)/(2*h),atol=8e-10,rtol=2e-8)


def test_no_authority_or_real_material_endpoint_never_silently_released():
    scene,target,w,J,G=_problem(np.array([0.,.3,3.,.3,0.,0.]));scene['ice_features'][0]['polygon_left_vertices_role']='measured_material_endpoint'
    with pytest.raises(ValueError):bounded_lane_support(scene,target,w)
    scene,target,w,J,G=_problem(np.array([0.,.3,3.,.3,0.,0.]));scene['ice_features'][0]['left_endpoint_x_interval_bound_sources']['lower_x']['feature_id']='rink_boundary'
    with pytest.raises(ValueError):bounded_lane_support(scene,target,w)


def test_moving_cap_derivative_must_be_explicit_and_shortening_is_rejected():
    scene,target,w,J,G=_problem(np.array([0.,.3,3.,.3,0.,0.]))
    with pytest.raises(ValueError):bounded_lane_support(scene,target,w,J)
    scene['ice_features'][1]['segments_xyz'][0][:,0]=1.
    with pytest.raises(ValueError):bounded_lane_support(scene,target,w)
