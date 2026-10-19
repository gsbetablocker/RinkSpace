import hashlib
import cv2
import numpy as np
import pytest
from rinkspace.vision_source_shapes import read_source_chromatic_stroke_centers

def bound(rgb):
    return dict(source_sha256='a'*64,source_pts=256,time_base='1/15360',shot_id='paint_test',
        native_hw=list(rgb.shape[:2]),decoded_native_rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest())

def reading(rgb,interval=(-80,80)):
    return read_source_chromatic_stroke_centers(rgb,binding=bound(rgb),search_points_uv=[[100,32]],
        normals=[[1,0]],search_intervals_px=[interval],keep_profiles=True)['readings'][0]

def test_real_band_mass_center_does_not_turn_paint_width_into_location_allowance():
    for width in (20,80):
        rgb=np.full((64,200,3),200,np.uint8)
        rgb[:,100-width//2:100+width//2]=[220,160,160]
        rgb=cv2.GaussianBlur(rgb,(0,0),2.)
        row=reading(rgb)['all_actual_source_chromatic_stroke_candidates'][0]
        assert row['disposition']=='measured_colored_stroke_center_candidate'
        assert abs(row['actual_source_uv'][0]-99.5)<1
        assert len(row['actual_RGB_band_interfaces'])==2
        assert row['centerline_zero_loss_interval_px'] is None
        assert row['paint_width_is_not_centerline_allowance']
        assert not row['solver_sigma_changed']

def test_clipped_band_and_competing_paint_remain_unqualified():
    rgb=np.full((64,200,3),200,np.uint8)
    rgb[:,70:130]=[220,160,160]
    clipped=reading(rgb,(-5,15))['all_actual_source_chromatic_stroke_candidates']
    assert all(c['disposition']=='source_material_role_unknown'for c in clipped)
    assert all(c['actual_source_uv'].size==0 for c in clipped)
    rgb[:,145:165]=[220,160,160]
    candidates=reading(rgb)['all_actual_source_chromatic_stroke_candidates']
    assert len(candidates)==2
    assert all(c['source_flags']['competing_disconnected_colored_paint']for c in candidates)
    assert all(c['actual_source_uv'].size==0 for c in candidates)

def test_actual_source_hash_and_paint_material_cannot_be_substituted():
    rgb=np.full((64,200,3),200,np.uint8);binding=bound(rgb);rgb[0,0]=0
    with pytest.raises(ValueError,match='bound original RGB differ'):
        read_source_chromatic_stroke_centers(rgb,binding=binding,search_points_uv=[[100,32]],normals=[[1,0]],search_intervals_px=[[-50,50]])
    with pytest.raises(ValueError,match='paint material signal'):
        read_source_chromatic_stroke_centers(rgb,binding=bound(rgb),search_points_uv=[[100,32]],normals=[[1,0]],search_intervals_px=[[-50,50]],material_signal='gray_house_boundary')
