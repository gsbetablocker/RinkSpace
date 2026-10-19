from copy import deepcopy

import numpy as np
import pytest

from calibration.program_tracking.gray_source_temporal_contour_review import review_gray_source_temporal_continuation
from calibration.program_tracking.test_gray_source_contour_review import _image_case


def _arguments():
    rgb,uv,sides,q,binding,paths,witness=_image_case()
    previous=dict(binding,source_sha256='c'*64,time_base='1/100',shot_id='cut1')
    current=dict(previous,actual_pts=200,native_pts=200)
    review=dict(source_binding=previous,model_coordinates_or_fit_errors_used=False,
        source_shape_verified_original_local_point_indices=list(range(9)))
    kwargs=dict(previous_binding=previous,current_binding=current,previous_verified_source_paths=paths,
        previous_source_review=review,previous_source_valid_mask=np.ones(rgb.shape[:2],bool),
        current_source_valid_mask=np.ones(rgb.shape[:2],bool),native_quantization_step_px=1.)
    args=(rgb,uv,sides,q,rgb.copy(),uv.copy(),sides.copy(),q.copy())
    return args,kwargs


def test_normal_LK_is_navigation_and_cannot_preserve_a_competing_current_branch(monkeypatch):
    import rinkspace.vision_ice_temporal_identity as source
    args,kwargs=_arguments();before=[a.tobytes()for a in args]
    def motion(previous,current,previous_mask,current_mask,track):
        assert track['actual_pts']==track['navigation_pts']==100
        assert len(track['actual_uv'])==9
        return track['actual_uv'].copy(),dict(source_RGB_tracking_confirmed=True,matched=27,coherent=27)
    monkeypatch.setattr(source,'_curve_local_navigation',motion)
    result=review_gray_source_temporal_continuation(*args,**kwargs)
    assert set(range(9,13)).issubset(result['pending_original_local_point_indices'])
    assert result['previous_source_anchor_point_count']==9
    assert result['LK_sample_count_is_not_independent_anchor_count']is True
    assert result['normal_navigation_is_not_two_dimensional_point_correspondence']is True
    assert result['source_FIT_admitted']is False
    assert before==[a.tobytes()for a in args]


def test_missing_native_RGB_motion_cannot_admit_current_candidates(monkeypatch):
    import rinkspace.vision_ice_temporal_identity as source
    args,kwargs=_arguments()
    monkeypatch.setattr(source,'_curve_local_navigation',lambda *a:(None,dict(source_RGB_tracking_confirmed=False)))
    result=review_gray_source_temporal_continuation(*args,**kwargs)
    assert result['source_shape_verified_original_local_point_indices']==[]
    assert result['pending_original_local_point_indices']==list(range(13))


def test_unverified_previous_points_or_cut_changes_cannot_supply_source_lineage():
    args,kwargs=_arguments();kwargs['previous_verified_source_paths']=[[0,1,9]]
    with pytest.raises(ValueError,match='previously source-verified'):
        review_gray_source_temporal_continuation(*args,**kwargs)
    args,kwargs=_arguments();kwargs['current_binding']['shot_id']='cut2'
    with pytest.raises(ValueError,match='previously source-verified'):
        review_gray_source_temporal_continuation(*args,**kwargs)


def test_singleton_does_not_supply_a_fabricated_two_point_tangent():
    args,kwargs=_arguments();kwargs['previous_verified_source_paths']=[[0]]
    result=review_gray_source_temporal_continuation(*args,**kwargs)
    assert result['source_shape_verified_original_local_point_indices']==[]
    assert result['previous_source_anchor_point_count']==1
