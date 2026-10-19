import hashlib
from copy import deepcopy
import numpy as np
import pytest

from calibration.program_tracking.gray_source_original_clearance_scope import apply_exact_current_original_candidate_clearance
from rinkspace.vision_ice_local_clearance import _context_pixels


def inputs(tmp_path):
    authority=tmp_path/'authority.json';authority.write_bytes(b'{}')
    binding=dict(actual_pts=128,native_pts=128,shot_id='shot0',source_sha256='a'*64,
        time_base='1/15360',decoded_native_BGR_sha256='b'*64)
    point=np.array([60.,60.]);normal=np.array([0.,1.]);sides=np.array([[60.,57.],[60.,63.]])
    pixels=_context_pixels(point,normal,24.)
    proof=dict(kind='known_static_ice_material_visible_inside_primary_detector_box',original_candidate_index=7,
        **binding,source_identity_authority=dict(path=str(authority),sha256=hashlib.sha256(b'{}').hexdigest()),
        source_identity_confirmed=True,measured_person_dilated_contour_and_other_person_boxes_excluded=True,
        only_primary_box_context_exclusion_removed=True,actual_source_painted_ice_membership_confirmed=True,
        model_projection_or_fit_residual_used_for_identity=False,mask_probability_used_as_visibility_truth=False,
        current_complete_material_context=dict(source_center_uv=point.tolist(),source_normal=normal.tolist(),
            normal_interval_px=[-24.,24.],tangent_interval_px=[-3,3],actual_native_context_pixel_count=len(pixels),
            actual_native_context_pixel_indices_sha256=hashlib.sha256(pixels.tobytes()).hexdigest()))
    valid=np.ones((120,120),bool);primary=np.zeros_like(valid);primary[20:100,20:100]=True;valid[primary]=False
    return valid,dict(source_binding=binding,original_candidate_index=7,original_candidate_uv=point,
        original_query_normal=normal,original_candidate_sides_uv=sides,clearance_proof=proof,
        primary_person_box_mask=primary,hard_source_exclusion_mask=np.zeros_like(valid)),pixels


def test_only_exact_certified_context_changes_input_immutable(tmp_path):
    valid,kwargs,pixels=inputs(tmp_path);before=valid.copy()
    result=apply_exact_current_original_candidate_clearance(valid,**kwargs)
    expected=before.copy();expected[pixels[:,1],pixels[:,0]]=True
    assert np.array_equal(result['source_valid_mask'],expected)
    assert np.array_equal(valid,before)
    assert not result['source_valid_mask'][30,30]
    assert result['receipt']['source_named_identity_or_pixel_accuracy_admitted']is False


@pytest.mark.parametrize('mask_kind',['SAM2','other_person_box','known_negative','unrelated_scope'])
def test_any_current_hard_source_exclusion_preserves_context_pending(tmp_path,mask_kind):
    valid,kwargs,pixels=inputs(tmp_path);kwargs['hard_source_exclusion_mask'][60,60]=True
    result=apply_exact_current_original_candidate_clearance(valid,**kwargs)
    assert np.array_equal(result['source_valid_mask'],valid)
    assert result['receipt']['cleared_pixels']==0
    assert result['receipt']['status'].startswith('pending_')


@pytest.mark.parametrize('field,value',[('native_pts',129),('actual_pts',129),('decoded_native_BGR_sha256','c'*64),('shot_id','shot1')])
def test_another_presentation_cannot_borrow_clearance(tmp_path,field,value):
    valid,kwargs,_=inputs(tmp_path);kwargs['source_binding'][field]=value
    with pytest.raises(ValueError,match='exact current original image'):
        apply_exact_current_original_candidate_clearance(valid,**kwargs)


def test_context_hash_and_actual_candidate_normal_are_not_guessed(tmp_path):
    valid,kwargs,_=inputs(tmp_path);kwargs['clearance_proof']['current_complete_material_context']['actual_native_context_pixel_indices_sha256']='0'*64
    with pytest.raises(ValueError,match='pixel certificate'):
        apply_exact_current_original_candidate_clearance(valid,**kwargs)
    valid,kwargs,_=inputs(tmp_path);kwargs['original_query_normal']=np.array([.001,1.])
    with pytest.raises(ValueError,match='measured source fields'):
        apply_exact_current_original_candidate_clearance(valid,**kwargs)
