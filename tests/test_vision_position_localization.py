"""Human interval localization without same-kind competition or new labels."""
from copy import deepcopy
import numpy as np
import pytest

torch=pytest.importorskip("torch")
from rinkspace.vision_supervision import (
    confirmed_position_interval_localization_loss,training_supervision_consumption_record,
    confirmed_source_point_coverage,confirmed_source_crop_target,
)

CLASSES=["background","ice_contact","ice_paint"]

def annotation(name,y,*,visible=True,interval=(-.5,.5)):
    return dict(group_id=name,kind="ice_contact",primitive="free_curve",state="user_confirmed_visual_supervision",
        authority="actual_user",uv=[[14.,y],[16.,y],[18.,y]],normal=[[0.,1.]]*3,
        interval_px=list(interval),source_visible_mask=[visible]*3)

def sample(annotations):
    return dict(source_hash="source_A",source_pts=256,time_base="1/12800",frame_sha256="native_RGB",
        annotations=annotations)

def selected(original,indices=(1,),origin=(0,0)):
    interval=np.asarray(original["interval_px"])
    if interval.shape==(2,):interval=np.tile(interval,(len(original["uv"]),1))
    return {**deepcopy(original),"source_point_indices":np.asarray(indices),
        "uv":np.asarray(original["uv"])[list(indices)]-origin,
        "normal":np.asarray(original["normal"])[list(indices)],"interval_px":interval[list(indices)]}

def loss(logits,s,annotations,*,origin=(0,0),radius=12):
    return confirmed_position_interval_localization_loss(logits,annotations,CLASSES,
        original_sample=s,crop_origin_xy=origin,profile_radius_px=radius)

def logits():
    values=torch.zeros((1,3,33,33));values[:,1]=80.
    return values.requires_grad_()


def test_flat_class_probability_one_has_position_gradient_and_sharp_raw_odds_peak_is_better():
    original=annotation("lower",16.)
    flat=logits();sharp=logits().detach();sharp[:,1,16,16]=100.;sharp.requires_grad_()
    assert torch.all(flat.softmax(1)[:,1]==1.) and torch.all(sharp.softmax(1)[:,1]==1.)
    flat_loss,rows=loss(flat,sample([original]),[selected(original)])
    sharp_loss,_=loss(sharp,sample([original]),[selected(original)])
    assert flat_loss>3 and sharp_loss<1e-5
    flat_loss.backward()
    assert flat.grad[0,1,16,16]<0 and flat.grad[0,1,10,16]>0
    assert rows[0]["sample_count"]==1 and rows[0]["class_CE_replaced"] is False
    assert rows[0]["unlabelled_pixels_assigned_background_or_other_positive_class"] is False
    # The alternative roles remain symmetric; no background target was chosen.
    assert torch.equal(flat.grad[:,0],flat.grad[:,2])


def test_two_confirmed_circle_sides_share_positive_support_and_neither_is_penalized():
    upper=annotation("circle_upper",12.);lower=annotation("circle_lower",20.)
    upper["kind"]=lower["kind"]="ice_paint";upper["primitive"]=lower["primitive"]="circle"
    values=torch.zeros((1,3,33,33));values[:,2]=20.;values[0,2,12,16]=40.;values[0,2,20,16]=40.;values.requires_grad_()
    result,rows=loss(values,sample([upper,lower]),[selected(upper)])
    result.backward()
    assert rows[0]["added_same_kind_confirmed_support_sample_count"]>=1
    assert values.grad[0,2,20,16]<=0  # Would be positive if the other circle side competed as a negative.
    assert values.grad[0,2,12,16]<=0


def test_full_source_interval_support_survives_when_neighbour_anchor_is_outside_crop():
    upper=annotation("upper",8.);outside=annotation("outside_crop",16.,interval=(-2.,2.))
    values=torch.zeros((1,3,16,33));values[:,1]=20.;values[0,1,8,16]=35.;values[0,1,14,16]=35.;values.requires_grad_()
    result,rows=loss(values,sample([upper,outside]),[selected(upper)])
    result.backward()
    assert rows[0]["added_same_kind_confirmed_support_sample_count"]>=2
    assert values.grad[0,1,14,16]<=0


def test_retired_middle_does_not_bridge_a_manual_support_gap_or_consume_old_position():
    old=annotation("old",12.);old["source_active_supervision_mask"]=[True,False,True]
    new=dict(group_id="new",kind="ice_contact",primitive="free_curve",state="user_confirmed_visual_supervision",
        authority="new_actual_user",uv=[[16.,20.]],normal=[[0.,1.]],interval_px=[-.5,.5],source_visible_mask=[True])
    values=logits().detach();values[0,1,12,16]=95.;values.requires_grad_()
    result,rows=loss(values,sample([old,new]),[selected(new,(0,))])
    result.backward()
    assert values.grad[0,1,12,16]>0  # Retired point is not union-positive support.
    assert rows[0]["retired_source_positions_used_for_support"]==0
    old_support=next(s for s in rows[0]["full_source_confirmed_support"] if s["group_id"]=="old")
    assert old_support["active_original_indices"]==[0,2]
    with pytest.raises(ValueError,match="retired or source-invisible"):
        loss(logits(),sample([old,new]),[selected(old,(1,))])


def test_known_nonvisible_boundary_has_no_negative_or_positive_localization_gradient():
    current=annotation("current",12.);occluded=annotation("occluded",20.,visible=False)
    values=logits();result,rows=loss(values,sample([current,occluded]),[selected(current)])
    result.backward()
    assert values.grad[0,:,20,16].abs().max()==0
    assert rows[0]["known_nonvisible_support_samples_excluded"]>=1


def test_spatial_scores_cannot_cheat_using_a_shared_per_pixel_logit_offset():
    original=annotation("lower",16.);s=sample([original]);a=selected(original)
    values=logits();shift=torch.arange(33.)[None,None,:,None].expand_as(values)
    before,_=loss(values,s,[a]);after,_=loss(values+shift,s,[a])
    assert torch.allclose(before,after,atol=1e-6)
    assert s==sample([original])


def test_direct_log_softmax_preserves_interpolate_then_spatial_log_odds_even_at_probability_one():
    original=annotation("lower",16.);s=sample([original]);a=selected(original)
    raw=logits().detach();raw[0,1,16,16]=100.
    raw_loss,_=loss(raw,s,[a]);log_probability_loss,_=loss(raw.log_softmax(1),s,[a])
    assert torch.all(raw.softmax(1)[:,1]==1.)
    assert torch.allclose(raw_loss,log_probability_loss,atol=1e-6)


def test_changed_uncertainty_or_crop_recomputed_normal_cannot_become_human_authority():
    original=annotation("lower",16.);a=selected(original)
    a["interval_px"][0]=[-3.,3.]
    with pytest.raises(ValueError,match="unchanged original position intervals"):
        loss(logits(),sample([original]),[a])
    a=selected(original);a["normal"][0]=[1.,0.]
    with pytest.raises(ValueError,match="recompute normals"):
        loss(logits(),sample([original]),[a])


def test_no_original_position_interval_retains_ce_without_inventing_precision():
    original=annotation("lower",16.);a=selected(original);del original["interval_px"]
    result,rows=loss(logits(),sample([original]),[a])
    assert result==0 and rows[0]["sample_count"]==0
    assert "no original position" in rows[0]["skipped_reason"]


def test_explicit_segment_gap_has_no_interpolated_manual_positive_support():
    split=annotation("split",12.);split["segment_ids"]=[0,1,1]
    current=dict(group_id="current",kind="ice_contact",primitive="free_curve",state="user_confirmed_visual_supervision",
        authority="actual_user",uv=[[15.,20.]],normal=[[0.,1.]],interval_px=[-.5,.5])
    values=logits().detach();values[0,1,12,15]=95.;values.requires_grad_()
    result,_=confirmed_position_interval_localization_loss(values,[selected(current,(0,))],CLASSES,
        original_sample=sample([split,current]),crop_origin_xy=(0,0),profile_radius_px=12,tangent_tolerance_px=.25)
    result.backward()
    assert values.grad[0,1,12,15]>0


def test_localization_counts_are_separate_from_ce_and_reject_fake_retired_support():
    original=annotation("lower",16.);s=sample([original]);a=selected(original)
    _,diagnostics=loss(logits(),s,[a])
    args=dict(step=1,sample_index=0,sample=s,selected_annotations=[a],classes=CLASSES,
        source_loss_diagnostics=[dict(group_id="lower",kind="ice_contact",sample_count=1)],shot_key="source_A:0:512",
        crop_origin_xy=[0,0],crop_hw=[33,33],full_frame=True,previous_source_observation=dict(available=False),
        optimizer_step_succeeded=True,localization_diagnostics=diagnostics)
    row=training_supervision_consumption_record(**args)
    assert len(row["groups"])==len(row["actual_localization_groups"])==1
    assert row["actual_localization_groups"][0]["actual_localization_source_positions"]["count"]==1
    diagnostics[0]["retired_source_positions_used_for_support"]=1
    with pytest.raises(ValueError,match="retired source support"):
        training_supervision_consumption_record(**args)


def test_positive_rank_margin_breaks_flat_plateau_with_direct_ce_and_mass_preserved():
    original=annotation("lower",16.);s=sample([original]);a=selected(original);values=logits()
    result,rows=confirmed_position_interval_localization_loss(values,[a],CLASSES,original_sample=s,
        crop_origin_xy=(0,0),profile_radius_px=12,profile_step_px=.25,rank_weight=1.)
    assert rows[0]["rank_loss"]==1 and rows[0]["rank_sample_count"]==1
    assert rows[0]["mass_loss"]>0 and rows[0]["class_CE_replaced"] is False
    result.backward();updated=values.detach()-values.grad
    profile=updated[0,1,:,16]-updated[0,[0,2],:,16].logsumexp(0)
    assert 15.5<=profile.argmax().item()<=16.5
    assert rows[0]["rank_source_point_indices"]==[1]


def test_rank_negative_pool_does_not_include_a_stronger_confirmed_other_circle_side():
    upper=annotation("upper",12.);lower=annotation("lower",20.)
    upper["kind"]=lower["kind"]="ice_paint"
    values=torch.zeros(1,3,33,33);values[:,2]=20.;values[0,2,12,16]=40.;values[0,2,20,16]=60.;values.requires_grad_()
    result,rows=confirmed_position_interval_localization_loss(values,[selected(upper)],CLASSES,
        original_sample=sample([upper,lower]),crop_origin_xy=(0,0),profile_radius_px=12,rank_weight=1.)
    result.backward()
    assert rows[0]["rank_loss"]==0 and values.grad[0,2,20,16]<=0
    assert rows[0]["other_confirmed_same_kind_support_used_as_rank_negative"] is False


def test_zero_rank_weight_has_no_rank_consumption_and_zero_margin_is_rejected():
    original=annotation("lower",16.);s=sample([original]);a=selected(original)
    _,rows=loss(logits(),s,[a])
    assert rows[0]["rank_sample_count"]==0 and rows[0]["rank_source_point_indices"]==[]
    with pytest.raises(ValueError,match="positive log-odds margin"):
        confirmed_position_interval_localization_loss(logits(),[a],CLASSES,original_sample=s,
            crop_origin_xy=(0,0),rank_weight=1.,rank_margin_log_odds=0.)


def test_crop_coverage_prefers_never_consumed_human_endpoints_without_assistant_labels():
    import random
    first=annotation("human",16.);first["uv"]=[[float(i),16.] for i in range(4,29)]
    first["normal"]=[[0.,1.]]*25;first["source_visible_mask"]=[True]*25
    assistant=annotation("assistant",20.);assistant["state"]="source_reviewed_interval"
    s=sample([first,assistant]);s["frame_size_wh"]=[33,33]
    coverage=confirmed_source_point_coverage(s,[]);coverage["groups"]["human"]=[10]*24+[0]
    target=confirmed_source_crop_target(s,coverage,random.Random(4),crop_size=16,bin_size=8)
    assert target["group_id"]=="human" and target["original_source_point_index"]==24
    assert target["original_source_uv"]==[28.,16.] and target["crop_origin_native_xy"]==[17,8]
    assert target["no_new_supervision_created"] and target["shot_sampling_weight_modified"] is False


def test_crop_sampler_cannot_choose_retired_invisible_or_stale_revision_positions():
    import random
    original=annotation("human",16.);original["source_active_supervision_mask"]=[False,True,True]
    original["source_visible_mask"]=[True,False,True]
    s=sample([original]);s["frame_size_wh"]=[33,33];coverage=confirmed_source_point_coverage(s,[])
    target=confirmed_source_crop_target(s,coverage,random.Random(0),crop_size=16,bin_size=1)
    assert target["original_source_point_index"]==2
    stale=deepcopy(s);stale["annotations"][0]["uv"][2][1]+=1
    with pytest.raises(ValueError,match="another annotation revision"):
        confirmed_source_crop_target(stale,coverage,random.Random(0),crop_size=16)
    foreign=deepcopy(s);foreign["source_hash"]="different_video_same_PTS"
    with pytest.raises(ValueError,match="another original source"):
        confirmed_source_crop_target(foreign,coverage,random.Random(0),crop_size=16)


def actual_consumption_row(original, *, rank_weight=0.):
    s=sample([original]);a=selected(original)
    _,diagnostics=confirmed_position_interval_localization_loss(logits(),[a],CLASSES,
        original_sample=s,crop_origin_xy=(0,0),profile_radius_px=12,rank_weight=rank_weight)
    return s,training_supervision_consumption_record(step=1,sample_index=0,sample=s,
        selected_annotations=[a],classes=CLASSES,
        source_loss_diagnostics=[dict(group_id=original["group_id"],kind="ice_contact",sample_count=1)],
        shot_key="source_A:0:512",crop_origin_xy=[0,0],crop_hw=[33,33],full_frame=True,
        previous_source_observation=dict(available=False),optimizer_step_succeeded=True,
        localization_diagnostics=diagnostics)


def test_point_coverage_uses_actual_indices_and_rejects_corrupt_or_stale_consumption():
    s,row=actual_consumption_row(annotation("human",16.))
    foreign=deepcopy(row);foreign["source_binding"]["source_sha256"]="another_video_same_PTS"
    assert confirmed_source_point_coverage(s,[row,foreign])["groups"]["human"]==[0,1,0]
    corrupt=deepcopy(row)
    corrupt["groups"][0]["actual_semantic_consumed_source_positions"]["ordered_original_indices_int64_le_sha256"]="not_actual_indices"
    with pytest.raises(ValueError,match="count, range or SHA"):
        confirmed_source_point_coverage(s,[corrupt])
    stale=deepcopy(s);stale["annotations"][0]["uv"][1][1]+=1
    with pytest.raises(ValueError,match="another original annotation revision"):
        confirmed_source_point_coverage(stale,[row])
    duplicate=deepcopy(row);duplicate["groups"].append(deepcopy(row["groups"][0]))
    with pytest.raises(ValueError,match="twice in a step"):
        confirmed_source_point_coverage(s,[duplicate])


def test_rank_consumption_has_separate_original_index_encoding():
    _,row=actual_consumption_row(annotation("human",16.),rank_weight=1.)
    encoded=row["actual_localization_groups"][0]["actual_rank_source_positions"]
    assert encoded["count"]==1 and encoded["runs_start_length"]==[[1,1]]
    s=sample([annotation("human",16.)]);a=selected(s["annotations"][0])
    _,diagnostics=confirmed_position_interval_localization_loss(logits(),[a],CLASSES,
        original_sample=s,crop_origin_xy=(0,0),profile_radius_px=12,rank_weight=1.)
    diagnostics[0]["rank_source_point_indices"]=[2]
    with pytest.raises(ValueError,match="Rank consumption differs"):
        training_supervision_consumption_record(step=1,sample_index=0,sample=s,
            selected_annotations=[a],classes=CLASSES,
            source_loss_diagnostics=[dict(group_id="human",kind="ice_contact",sample_count=1)],
            shot_key="source_A:0:512",crop_origin_xy=[0,0],crop_hw=[33,33],full_frame=True,
            previous_source_observation=dict(available=False),optimizer_step_succeeded=True,
            localization_diagnostics=diagnostics)
