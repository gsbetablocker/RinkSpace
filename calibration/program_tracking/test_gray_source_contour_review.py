import hashlib

import cv2
import numpy as np
import pytest

from calibration.program_tracking.gray_source_contour_review import review_current_gray_source_contour,_all_sample_pixels_supported


def _image_case():
    rgb=np.full((140,240,3),240,np.uint8)
    cv2.line(rgb,(10,95),(220,53),(160,160,160),6,cv2.LINE_AA)
    cv2.line(rgb,(110,38),(200,74),(160,160,160),6,cv2.LINE_AA)
    uv=np.asarray([[20.,93.],[40.,89.],[60.,85.],[80.,81.],[100.,77.],
        [140.,69.],[160.,65.],[180.,61.],[200.,57.],
        [120.,42.],[140.,50.],[160.,58.],[180.,66.]])
    sides=np.stack([uv-[0.,3.],uv+[0.,3.]],axis=1)
    q=np.stack([sides-[0.,.1],sides,sides+[0.,.1]],axis=2)
    binding=dict(actual_pts=100,native_pts=100,decoded_native_BGR_sha256=hashlib.sha256(rgb[:,:,::-1].tobytes()).hexdigest())
    paths=[[0,1,2,3,4],[5,6,7,8]]
    witness=dict(source_binding=binding,actual_visible_named_material_reviewed=True,
        model_projection_used_as_source_truth=False,source_review_references=[dict(fixture='source-contour-unit-case')],
        trusted_current_stem_paths=paths)
    return rgb,uv,sides,q,binding,paths,witness


def _review(case,mask=None):
    rgb,uv,sides,q,binding,paths,witness=case
    if mask is None:mask=np.ones(rgb.shape[:2],bool)
    return review_current_gray_source_contour(rgb,uv,sides,q,source_binding=binding,
        trusted_current_stem_paths=paths,stem_witness=witness,source_valid_mask=mask,native_quantization_step_px=1.)


def test_source_crossing_branch_is_pending_and_witnessed_stems_remain_exact():
    case=_image_case();before=[a.tobytes() for a in case[:4]]
    result=_review(case)
    assert set(range(9,13)).issubset(result['pending_original_local_point_indices'])
    assert all(result['point_decisions'][i]['status']=='witnessed_current_named_source_stem_preserved' for i in range(5))
    assert result['original_source_points_removed']==0 and result['model_coordinates_or_fit_errors_used'] is False
    assert result['source_FIT_admitted'] is False
    assert before==[a.tobytes() for a in case[:4]]


def test_missing_quantile_support_and_scope_remain_pending_even_for_witnessed_pixel():
    case=_image_case();case[3][0]=np.nan
    mask=np.ones(case[0].shape[:2],bool);mask[81,80]=False
    result=_review(case,mask)
    assert {0,3}.issubset(result['pending_original_local_point_indices'])


def test_wrong_native_image_or_unwitnessed_stem_cannot_supply_named_shape_pass():
    case=_image_case();case[0][0,0]=0
    with pytest.raises(ValueError,match='own unchanged native image'):_review(case)
    case=_image_case();case[6]['actual_visible_named_material_reviewed']=False
    with pytest.raises(ValueError,match='Separately witnessed'):_review(case)


def test_scope_covers_tangent_samples_and_bilinear_pixel_contributors():
    mask=np.ones((20,20),bool)
    positions=np.asarray([[[8.25,8.25],[8.25,5.25],[8.25,11.25]]])
    normals=np.asarray([[0.,1.]])
    assert _all_sample_pixels_supported(mask,positions,normals).tolist()==[True]
    # This is a contributor at tangent offset +2, not one of three centers.
    mask[9,7]=False
    assert _all_sample_pixels_supported(mask,positions,normals).tolist()==[False]
    mask[:]=True
    positions[:,:,0]=.25
    assert _all_sample_pixels_supported(mask,positions,normals).tolist()==[False]


@pytest.mark.parametrize('gap',[np.nan,np.inf,0.,-1.])
def test_navigation_gap_must_be_finite_positive(gap):
    rgb,uv,sides,q,binding,paths,witness=_image_case()
    with pytest.raises(ValueError,match='Exact current'):
        review_current_gray_source_contour(rgb,uv,sides,q,source_binding=binding,
            trusted_current_stem_paths=paths,stem_witness=witness,
            source_valid_mask=np.ones(rgb.shape[:2],bool),native_quantization_step_px=1.,
            maximum_source_neighbor_gap_px=gap)
