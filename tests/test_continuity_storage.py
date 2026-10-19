import copy
import numpy as np
from rinkspace.continuity_storage import save_observations, load_observations


def test_geometry_export_roundtrip_retains_full_actual_pairs_and_both_directions(tmp_path):
    rows=[{'sample_index':5,'pts':100,'time_base':'1/100','local_registration':{
        'H_keyframe_from_current':[[1,0,2],[0,1,-3],[0,0,1]],
        'H_reference_from_current':[[1,0,20],[0,1,-30],[0,0,1]],
        'edge_keyframe_sample_index':3,'fit_current_xy':[[.123456789123456,5.],[6.,7.]],
        'fit_keyframe_xy':[[2.123456789123456,2.],[8.,4.]],'fit_reference_xy':[[20.,-25.],[26.,-23.]]}}]
    original=copy.deepcopy(rows)
    info=save_observations(rows,tmp_path/'local_observations.npz')
    assert info['all_observed_inlier_pairs_retained']
    assert rows[0]['local_registration']['H_keyframe_from_current']==original[0]['local_registration']['H_keyframe_from_current']
    assert 'fit_current_xy' not in rows[0]['local_registration']
    load_observations(rows,tmp_path/'local_observations.npz')
    for name in ('fit_current_xy','fit_keyframe_xy','fit_reference_xy'):
        np.testing.assert_array_equal(rows[0]['local_registration'][name],original[0]['local_registration'][name])
    assert rows[0]['pts']==100
