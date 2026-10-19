import numpy as np

from rinkfit_start.native_query_global import certify_global_profiles,reduced_system
from test_query_certificate import _synthetic


def test_reduced_interval_encloses_analytic_map_across_clock_knot():
    scene,camera,curves=_synthetic()
    result=reduced_system(scene,camera,[0],[0],np.array([0.]),curves,
                          np.array([[0.,-20.]]),np.array([[1.,20.]]))
    assert result['smooth'][0] and result['cells_evaluated'][0]==2
    J=np.diag([100.,-1.])
    assert np.all(result['J_lower'][0]<=J) and np.all(J<=result['J_upper'][0])
    assert np.all(result['F_lower'][0]<=[-50.,-20.])
    assert np.all(result['F_upper'][0]>=[50.,20.])


def test_global_rectangle_has_exactly_the_one_transverse_root():
    scene,camera,curves=_synthetic()
    result=certify_global_profiles(scene,camera,[0],[0],[.5],[0.],curves,[0.],[540.],
                                   max_boxes_per_profile=256)
    assert result['global_unique_certified'].tolist()==[True]
    assert result['unresolved_boxes'].tolist()==[0]
    assert result['original_finite_lower'].tolist()==[[0.,-20.]]
    assert result['original_finite_upper'].tolist()==[[1.,20.]]
    assert all(node['result']!='pending'for node in result['proof_tree'])


def test_second_real_root_is_not_hidden_by_a_local_certificate():
    scene,camera,curves=_synthetic()
    curves=dict(curves,numerator_v=np.array([[539.,0.,1.,0.]]),bounds=np.array([[-2.,2.]]))
    result=certify_global_profiles(scene,camera,[0],[0],[.5],[0.],curves,[1.],[540.],
                                   max_boxes_per_profile=128)
    assert result['local_certificate']['certified'][0]
    assert not result['global_unique_certified'][0]
    assert result['unresolved_boxes'][0]>0


def test_no_local_existence_or_zero_box_budget_cannot_claim_global_uniqueness():
    scene,camera,curves=_synthetic()
    absent=certify_global_profiles(scene,camera,[0],[0],[.6],[0.],curves,[0.],[540.])
    assert not absent['global_unique_certified'][0]
    limited=certify_global_profiles(scene,camera,[0],[0],[.5],[0.],curves,[0.],[540.],
                                    max_boxes_per_profile=0)
    assert not limited['global_unique_certified'][0]
    assert limited['unresolved_boxes'][0]==1
