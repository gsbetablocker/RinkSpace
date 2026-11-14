import numpy as np
import torch
from rinkspace.session.board_contacts import read_board_contacts


def source():
    image=np.full((1080,1920,3),185,np.uint8)
    image[:300]=210
    image[300:316]=[180,170,100]
    image[316:324]=[182,178,165]  # weaker reflection is not the board body
    binding=dict(source_sha256='a'*64,source_pts=256,shot_id='test',time_base='1/12800',native_hw=[1080,1920])
    authority=dict(kind='direct_user_confirmation',reference='test_fixture',source_sha256='a'*64,
        board_foot_material='warm_yellow_strip_belongs_to_board')
    return image,binding,authority


def test_actual_lower_transition_and_no_dense_fit_duplicates():
    torch.set_num_threads(2)
    image,binding,authority=source()
    old=np.array([[96,305],[960,305],[1800,305]],float)
    result=read_board_contacts(image,old,np.zeros(image.shape[:2],bool),binding=binding,authority=authority)
    assert result['old_query_supported'].all()
    assert result['old_outside_measured_transition'].all()
    assert np.all((result['old_query_fresh_uv'][:,1]>315)&(result['old_query_fresh_uv'][:,1]<316))
    assert len(result['queries'])>len(old)
    assert len(result['old_query_fresh_uv'])==len(old)
    assert result['dense_queries_are_fit_observations'] is False
    assert np.array_equal(result['candidate_source_uv'][result['old_query_indices']],result['old_query_fresh_uv'])


def test_sam2_exclusion_covers_material_support_not_only_contact_pixel():
    image,binding,authority=source();mask=np.zeros(image.shape[:2],bool)
    mask[302:308,950:970]=True  # contact pixel itself is outside this occluder
    result=read_board_contacts(image,[[96,305],[960,305],[1800,305]],mask,binding=binding,authority=authority)
    assert not result['old_query_supported'][1]
    assert result['source_mask_conflict'][result['old_query_indices'][1]]


def test_red_ad_above_true_yellow_strip_cannot_inherit_foot_role():
    image,binding,authority=source()
    image[260:280]=[220,100,70]  # stronger warm gamma, different 2D material
    result=read_board_contacts(image,[[96,280],[960,280],[1800,280]],
        np.zeros(image.shape[:2],bool),binding=binding,authority=authority)
    assert result['old_query_supported'].all()
    assert np.all((result['old_query_fresh_uv'][:,1]>315)&(result['old_query_fresh_uv'][:,1]<316))
    assert np.all(result['source_2D_material_selected_pair_support_fraction']>=.5)
    assert result['source_2D_material_component_evidence']['morphology_fill_pixels_are_material_support'] is False


def test_red_ad_without_long_yellow_body_stays_unresolved():
    image,binding,authority=source();image[300:324]=185
    image[260:280]=[220,100,70]
    result=read_board_contacts(image,[[96,280],[960,280],[1800,280]],
        np.zeros(image.shape[:2],bool),binding=binding,authority=authority)
    assert not result['old_query_supported'].any()


def test_explicit_scope_queries_do_not_replay_old_columns():
    image,binding,authority=source()
    requested=np.array([[981.,305.],[119.,305.],[555.,305.]])
    result=read_board_contacts(image,[[96,305],[1800,305]],
        np.zeros(image.shape[:2],bool),binding=binding,authority=authority,
        spacing=48.,source_query_points_uv=requested)
    assert np.array_equal(result['queries'],requested[[1,2,0]])
    assert result['explicit_source_query_points_only'] is True
    assert result['original_queries_reread'] is False
    assert len(result['old_query_indices'])==len(result['old_query_fresh_uv'])==0
    assert result['original_RGB_profiles']['sampled_native_uv'].shape[0]==3
    assert result['source_material_accepted'].all()
    assert np.all((result['candidate_source_uv'][:,1]>315)&(result['candidate_source_uv'][:,1]<316))
    assert result['dense_queries_are_fit_observations'] is False


def test_explicit_scope_queries_keep_actual_material_support_mask_gate():
    image,binding,authority=source();mask=np.zeros(image.shape[:2],bool)
    mask[302:308,950:970]=True
    result=read_board_contacts(image,[[96,305],[1800,305]],mask,
        binding=binding,authority=authority,spacing=48.,
        source_query_points_uv=np.array([[960.,305.],[1200.,305.]]))
    assert result['source_mask_conflict'].tolist()==[True,False]
    assert result['source_material_accepted'].tolist()==[False,True]
    assert result['original_queries_reread'] is False
