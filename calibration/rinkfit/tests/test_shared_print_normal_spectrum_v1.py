"""The conditional shape spectrum eliminates actual curve station directions."""
import numpy as np
from rinkfit_joint.shared_print_normal_spectrum_v1 import projected_normal_rows, rank_from_gram


def test_station_elimination_removes_tangent_but_retains_weighted_normal():
    j=np.broadcast_to(np.eye(2),(3,2,2)).copy()
    tangent=np.array([[1.,0.],[0.,2.],[0.,0.]])
    k,valid=projected_normal_rows(j,tangent,np.array([2.,3.,4.]))
    np.testing.assert_array_equal(valid,[True,True,False])
    np.testing.assert_allclose(k,[[0.,2.],[-3.,0.],[0.,0.]],atol=0,rtol=0)
    np.testing.assert_allclose(np.sum(k*tangent,axis=1),0,atol=0,rtol=0)


def test_shared_straight_edge_has_two_normal_coordinates_and_tangent_gauge():
    alpha=np.linspace(0.,1.,20)
    a=np.stack((np.zeros_like(alpha),1-alpha,np.zeros_like(alpha),alpha),axis=1)
    result=rank_from_gram(a.T@a,len(a))
    assert result['conditional_normal_rank']==2
    assert result['numerical_nullity']==2
    assert not result['coupled_camera_scene_observability_proven']
    assert not result['source_covariance_or_noise_precision_claimed']


def test_repeated_frames_do_not_create_independent_shape_directions():
    row=np.array([[0.,.7,0.,.3]])
    result=rank_from_gram(row.T@row*100,100)
    assert result['conditional_normal_rank']==1
    assert result['numerical_nullity']==3


def test_pixel_coordinate_change_does_not_reintroduce_station_tangent():
    j=np.array([[[4.,1.],[2.,3.]],[[1.,-2.],[2.,1.]]])
    tangent=np.array([[.4,.1],[-.3,.8]])
    k,valid=projected_normal_rows(j,tangent,np.ones(2))
    assert valid.all()
    np.testing.assert_allclose(np.sum(k*tangent,axis=1),0,atol=2e-16,rtol=0)
