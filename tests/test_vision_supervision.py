"""Human authority, printed geometry roles and native-content split integrity."""
from copy import deepcopy

import pytest

from rinkspace.vision_supervision import (
    annotation_role, apply_supervision_roles, build_shot_plan, source_time, split_indices,
    validate_boundary_definition,
    native_pixel_sha256,
    measured_interval_annotation,
    shot_sampling_groups,
    attach_native_shot_scopes, native_shot_scope, validate_sequence_shot_scope,
    source_sample_key, source_updates_by_key,
    annotation_revision_sha256, apply_user_boundary_revisions,
    merge_user_boundary_samples, validate_source_visibility, timeline_with_native_boundary_overlay,
    merge_source_reviewed_regions,
    active_source_supervision_mask, source_supervision_point_counts,
    native_boundary_overlay_reference,
    training_supervision_consumption_record,
)


def consumption_fixture():
    import numpy as np
    original=dict(group_id="human_lower",kind="ice_contact",state="user_confirmed_visual_supervision",
        uv=[[11.,22.],[12.,22.],[13.,22.],[14.,22.],[15.,22.]],authority="actual_human",
        source_active_supervision_mask=[True,False,True,True,True],
        source_visible_mask=[True,True,True,False,True])
    sample=dict(source_hash="source_A",time_base="2/25600",source_pts=256,frame_sha256="original_RGB",
        annotations=[original],source_shot_scope=dict(source_sha256="source_A",time_base="1/12800",
        start_pts=0,end_pts_exclusive=512,shot_id="native_shot_one"))
    selected={**original,"source_point_indices":np.array([0,2,4]),"uv":np.array([[1.,2.],[3.,2.],[5.,2.]])}
    args=dict(step=11201,sample_index=4,sample=sample,selected_annotations=[selected],
        source_loss_diagnostics=[dict(group_id="human_lower",kind="ice_contact",sample_count=3,
        supervision="direct_user_confirmed_native_UV")],classes=["background","ice_contact"],
        shot_key="source_A:0:512",crop_origin_xy=[10,20],crop_hw=[16,16],full_frame=False,
        previous_source_observation=dict(available=True,reason="distinct native previous"),optimizer_step_succeeded=True)
    return args


def test_consumption_uses_actual_indices_not_original_manifest_count_and_is_json_safe():
    import hashlib
    import json
    import numpy as np
    args=consumption_fixture();untouched=deepcopy(args)
    row=training_supervision_consumption_record(**args)
    group=row["groups"][0]
    assert group["original_position_count"]==5
    assert group["selected_source_positions"]["count"]==group["actual_semantic_consumed_source_positions"]["count"]==3
    assert group["selected_source_positions"]["runs_start_length"]==[[0,1],[2,1],[4,1]]
    assert group["selected_source_positions"]["ordered_original_indices_int64_le_sha256"]==hashlib.sha256(np.array([0,2,4],dtype="<i8").tobytes()).hexdigest()
    assert row["source_binding"]["time_base"]=="1/12800"
    assert group["retired_source_positions_consumed"]==group["source_invisible_positions_consumed"]==0
    json.dumps(row,allow_nan=False)
    assert args["sample"]==untouched["sample"]
    assert np.array_equal(args["selected_annotations"][0]["uv"],untouched["selected_annotations"][0]["uv"])


@pytest.mark.parametrize("indices,uv",[([0,1,4],[[1,2],[2,2],[5,2]]),([0,3,4],[[1,2],[4,2],[5,2]])])
def test_consumption_rejects_retired_and_source_invisible_human_indices(indices,uv):
    import numpy as np
    args=consumption_fixture();args["selected_annotations"][0].update(source_point_indices=np.asarray(indices),uv=np.asarray(uv))
    with pytest.raises(ValueError,match="retired or source-invisible"):
        training_supervision_consumption_record(**args)


def test_consumption_rejects_loss_count_role_source_scope_and_displaced_uv_mismatches():
    for change,match in [(lambda a:a["source_loss_diagnostics"][0].update(sample_count=5),"actual loss diagnostic"),
                         (lambda a:a["source_loss_diagnostics"][0].update(kind="background"),"actual loss diagnostic"),
                         (lambda a:a.update(shot_key="other_source:0:512"),"shot/source scope"),
                         (lambda a:a["selected_annotations"][0]["uv"].__setitem__((0,1),3),"crop UVs")]:
        args=consumption_fixture();change(args)
        with pytest.raises(ValueError,match=match):training_supervision_consumption_record(**args)


def test_consumption_separates_interval_selected_from_actual_inbounds_semantic_consumers():
    import numpy as np
    args=consumption_fixture();original=args["sample"]["annotations"][0]
    original.update(state="source_reviewed_interval",uv=[[10.,22.],[11.,22.]],
                    source_active_supervision_mask=[True,True],source_visible_mask=[True,True])
    selected={**original,"source_point_indices":np.array([0,1]),"uv":np.array([[0.,2.],[1.,2.]]),
              "normal":np.array([[1.,0.],[1.,0.]]),"interval_px":np.array([[-2.,-1.],[-2.,-1.]])}
    args["selected_annotations"]=[selected];args["source_loss_diagnostics"][0]["sample_count"]=1
    row=training_supervision_consumption_record(**args);group=row["groups"][0]
    assert group["selected_source_positions"]["count"]==2
    assert group["actual_semantic_consumed_source_positions"]["count"]==1
    assert group["actual_semantic_consumed_source_positions"]["runs_start_length"]==[[1,1]]
    args["source_loss_diagnostics"][0]["sample_count"]=2
    with pytest.raises(ValueError,match="actual loss diagnostic"):
        training_supervision_consumption_record(**args)


def test_consumption_requires_actual_region_counts_and_keeps_exclusions_separate():
    args=consumption_fixture()
    args["sample"]["regions"]=[dict(region_id="reject_contact",excluded_kinds=["ice_contact"],rectangle_xyxy=[10,20,15,25]),
                                dict(region_id="positive_background",kind="background",rectangle_xyxy=[25,30,30,35])]
    with pytest.raises(ValueError,match="every actual reviewed-region"):
        training_supervision_consumption_record(**args)
    args["reviewed_region_counts"]=[dict(region_id="reject_contact",supervised_pixels=25,protected_human_pixels=0,
                                        human_protection_policy="fail_on_human_overlap"),
                                   dict(region_id="positive_background",supervised_pixels=0,protected_human_pixels=0,
                                        human_protection_policy="no_human_mask_subtraction")]
    row=training_supervision_consumption_record(**args)
    assert row["reviewed_regions"][0]["positive_kind"] is None
    assert row["reviewed_regions"][0]["excluded_kinds"]==["ice_contact"]
    assert row["reviewed_regions"][1]["actual_loss_counts"]["supervised_pixels"]==0
    args["reviewed_region_counts"][0]["protected_human_pixels"]=2
    with pytest.raises(ValueError,match="without actual mask subtraction"):
        training_supervision_consumption_record(**args)
    args["reviewed_region_counts"][0]["protected_human_pixels"]=0
    args["reviewed_region_counts"][0]["supervised_pixels"]=26
    with pytest.raises(ValueError,match="region/crop overlap"):
        training_supervision_consumption_record(**args)


def test_consumption_refuses_pre_optimizer_records_or_nonunique_indices():
    import numpy as np
    args=consumption_fixture();args["optimizer_step_succeeded"]=False
    with pytest.raises(ValueError,match="actually successful"):
        training_supervision_consumption_record(**args)
    args=consumption_fixture();args["selected_annotations"][0]["source_point_indices"]=np.array([0,2,2])
    with pytest.raises(ValueError,match="unique ordered"):
        training_supervision_consumption_record(**args)


def test_consumption_binds_legacy_regions_by_existing_source_index_without_inventing_ids():
    args=consumption_fixture()
    regions=[dict(kind="background",rectangle_xyxy=[10,20,15,25],authority="legacy_source_review_A"),
             dict(kind="background",rectangle_xyxy=[10,20,15,25],authority="legacy_source_review_B")]
    args["sample"]["regions"]=regions;untouched=deepcopy(regions)
    args["reviewed_region_counts"]=[dict(region_index=i,supervised_pixels=n,protected_human_pixels=0,
        human_protection_policy="no_human_mask_subtraction") for i,n in [(0,25),(1,12)]]
    row=training_supervision_consumption_record(**args)
    first,second=row["reviewed_regions"]
    assert first["region_id"] is second["region_id"] is None
    assert first["merged_source_manifest_pointer"]=="/samples/4/regions/0"
    assert second["merged_source_manifest_pointer"]=="/samples/4/regions/1"
    assert first["actual_source_region_sha256"]!=second["actual_source_region_sha256"]
    assert regions==untouched
    args["reviewed_region_counts"][1]["region_index"]=0
    with pytest.raises(ValueError,match="unique original source region"):
        training_supervision_consumption_record(**args)


def test_reviewed_update_same_pts_stays_on_its_source_and_preserves_all_evidence():
    annotation = dict(uv=[[12.5, 18.0]], interval_px=[-3, 4],
                      authority="original_user", target_role="external_printed_ink_component")
    samples = [dict(source_hash="source_A", time_base="1/12800", source_pts=256,
                    frame_sha256="frame_A", annotations=[annotation]),
               dict(source_hash="source_B", time_base="1/12800", source_pts=256,
                    frame_sha256="frame_B", annotations=[annotation])]
    update = dict(source_hash="source_B", time_base="1/12800", source_pts=256,
                  frame_sha256="frame_B", local_support_reviews=[annotation])
    untouched = deepcopy((samples, update))
    resolved = source_updates_by_key(samples, [update])
    assert source_sample_key(samples[0]) not in resolved
    assert resolved[source_sample_key(samples[1])] is update
    assert (samples, update) == untouched


def test_legacy_update_uses_unique_pts_and_frame_sha_without_rewriting_source_fields():
    samples = [dict(source_hash="source_A", time_base="1/15360", source_pts=512,
                    frame_sha256="different_frame"),
               dict(source_hash="source_B", time_base="1/12800", source_pts=512,
                    frame_sha256="reviewed_frame")]
    legacy = dict(source_pts=512, frame_sha256="reviewed_frame", regions=[dict(authority="source_review")])
    untouched = deepcopy(legacy)
    resolved = source_updates_by_key(samples, [legacy])
    assert list(resolved) == [source_sample_key(samples[1])]
    assert resolved[source_sample_key(samples[1])] is legacy
    assert legacy == untouched and "source_hash" not in legacy and "time_base" not in legacy


def test_legacy_update_fails_when_same_pts_and_image_cannot_disambiguate_sources():
    samples = [dict(source_hash=source, time_base="1/12800", source_pts=256,
                    frame_sha256="same_image") for source in ("source_A", "source_B")]
    with pytest.raises(ValueError, match="found 2"):
        source_updates_by_key(samples, [dict(source_pts=256, frame_sha256="same_image")])
    with pytest.raises(ValueError, match="found 0"):
        source_updates_by_key(samples, [dict(source_pts=256, frame_sha256="changed_image")])


def test_source_key_normalizes_clock_aliases_without_rewriting_original_clock():
    sample = dict(source_hash="source_A", time_base="1/15360", source_pts=512,
                  frame_sha256="original_frame")
    update = dict(source_sha256="source_A", time_base="2/30720", source_pts=512,
                  frame_sha256="original_frame")
    assert source_sample_key(sample) == source_sample_key(update)
    assert source_updates_by_key([sample], [update])[source_sample_key(sample)] is update
    assert update["time_base"] == "2/30720"
    with pytest.raises(ValueError, match="no original sample"):
        source_updates_by_key([sample], [{**update, "time_base": "1/12800"}])


def test_explicit_update_cannot_fall_back_from_partial_identity_or_wrong_frame():
    sample = dict(source_hash="source_A", time_base="1/12800", source_pts=256,
                  frame_sha256="original_frame")
    legacy = dict(source_pts=256, frame_sha256="original_frame")
    with pytest.raises(ValueError, match="both source hash and native clock"):
        source_updates_by_key([sample], [{**legacy, "source_hash": "source_A"}])
    explicit = {**legacy, "source_hash": "source_A", "time_base": "1/12800"}
    with pytest.raises(ValueError, match="frame SHA differs"):
        source_updates_by_key([sample], [{**explicit, "frame_sha256": "changed_image"}])
    with pytest.raises(ValueError, match="Duplicate reviewed update"):
        source_updates_by_key([sample], [explicit, legacy])


def boundary_revision_fixture():
    old=dict(group_id="older_user_contact",kind="ice_contact",primitive="free_curve",
             uv=[[2.0,5.0]],interval_px=[-2,2],state="user_confirmed_visual_supervision",authority="older_human_drawing")
    sample=dict(source_hash="source_A",time_base="1/12800",source_pts=256,frame_sha256="original_RGB",
                annotations=[old])
    corrected=dict(group_id="user_contact_lower_strip_v1",kind="ice_contact",primitive="free_curve",
                   uv=[[2.0,8.0]],interval_px=[-.5,.5],state="user_confirmed_visual_supervision",
                   boundary_role="lower_board_base_strip_to_ice",authority="new_user_red_line")
    revision=dict(schema="rinkspace.user-source-boundary-supervision-revision.v1",revision_id="user_lower_strip_v1",version=1,
                  authority=dict(kind="direct_user_confirmation",source="actual_user_red_line"),
                  source_binding=dict(source_sha256="source_A",time_base="1/12800",source_pts=256),
                  frame_sha256="original_RGB",annotations=[corrected],
                  supersedes=[dict(group_id=old["group_id"],original_annotation_sha256=annotation_revision_sha256(old))])
    return old,sample,corrected,revision


def test_ice_paint_user_revision_requires_physical_role_and_preserves_old_drawing():
    old,sample,corrected,revision=boundary_revision_fixture()
    corrected.update(kind="ice_paint",primitive="circle",boundary_role="paint_outer_boundary",feature_id="F2:outer")
    result=apply_user_boundary_revisions(dict(samples=[sample]),[revision])
    assert result["samples"][0]["annotations"][-1]["uv"]==corrected["uv"]
    assert result["boundary_supervision_revision_ledger"][0]["original_superseded_annotations"]==[old]
    corrected.pop("feature_id")
    with pytest.raises(ValueError,match="physical role"):
        apply_user_boundary_revisions(dict(samples=[sample]),[revision])


def test_authority_curriculum_keeps_human_and_does_not_rewrite_source_uv():
    from rinkspace.vision_supervision import select_training_authorities
    old,sample,_,_=boundary_revision_fixture()
    read={**deepcopy(old),"group_id":"RGB_propagated", "state":"source_reviewed_interval", "authority":"source_RGB"}
    sample["annotations"].append(read)
    original=deepcopy(sample)
    result=select_training_authorities(dict(samples=[sample]),["direct_user_confirmed_native_UV"])
    assert sample==original
    assert result["samples"][0]["annotations"][0]==old
    assert result["samples"][0]["annotations"][1]["uv"]==read["uv"]
    assert result["samples"][0]["annotations"][1]["state"]=="unreviewed"
    assert not result["training_authority_selection"]["excluded_readings_declared_wrong"]
    with pytest.raises(ValueError,match="preserve direct user"):
        select_training_authorities(dict(samples=[sample]),["assistant_original_source_interval"])


def test_balanced_source_cycles_preserve_uniform_policy_and_resume_exactly():
    import random
    from rinkspace.vision_supervision import balanced_source_selection
    groups=[dict(shot_key='one',sampling_pools=[[0],[1,2]]),dict(shot_key='two',sampling_pools=[[3,4]])]
    state={};rng=random.Random(5)
    for _ in range(40):balanced_source_selection(groups,state,rng)
    assert state['shots']=={'one':20,'two':20}
    assert state['frames']['one|1|1']==state['frames']['one|1|2']==5
    copied=deepcopy(state);next_rng=random.Random();next_rng.setstate(rng.getstate())
    assert balanced_source_selection(groups,state,rng)==balanced_source_selection(groups,copied,next_rng)
    assert state==copied
    with pytest.raises(ValueError,match='layout'):
        balanced_source_selection(groups[:1],state,rng)


def test_user_boundary_revision_appends_latest_role_and_keeps_old_failure_exactly():
    old,sample,corrected,revision=boundary_revision_fixture()
    other={**deepcopy(sample),"source_hash":"source_B","frame_sha256":"other_RGB"}
    original=dict(samples=[sample,other])
    untouched=deepcopy(original)
    revised=apply_user_boundary_revisions(original,[revision])
    assert original==untouched
    historical,new=revised["samples"][0]["annotations"]
    assert historical["uv"]==old["uv"] and historical["interval_px"]==old["interval_px"]
    assert historical["authority"]==old["authority"] and historical["state"]=="ignored"
    assert historical["superseded_by_user_boundary_revision"]==revision["revision_id"]
    assert revised["boundary_supervision_revision_ledger"][0]["original_superseded_annotations"]==[old]
    assert new["uv"]==corrected["uv"] and new["boundary_role"]=="lower_board_base_strip_to_ice"
    assert revised["samples"][1]==other
    roles=apply_supervision_roles(revised)["samples"][0]["annotations"]
    assert not roles[0]["vision_supervision"]["semantic_supervision_enabled"]
    assert roles[0]["vision_supervision"]["authority_role"]=="superseded_by_later_direct_user_boundary_revision"
    assert roles[1]["vision_supervision"]["semantic_supervision_enabled"]
    assert roles[1]["vision_supervision"]["ice_contact_role_consistent"]
    assert roles[1]["vision_supervision"]["base_strip_belongs_to_board"]


def test_user_boundary_revision_is_idempotent_and_cannot_change_existing_version():
    _,sample,_,revision=boundary_revision_fixture()
    revised=apply_user_boundary_revisions(dict(samples=[sample]),[revision])
    assert apply_user_boundary_revisions(revised,[revision])==revised
    changed=deepcopy(revision)
    changed["annotations"][0]["uv"]=[[2,10]]
    with pytest.raises(ValueError,match="same ID"):
        apply_user_boundary_revisions(revised,[changed])
    changed["revision_id"]="another_claim_same_version"
    with pytest.raises(ValueError,match="versions must increase"):
        apply_user_boundary_revisions(revised,[changed])


@pytest.mark.parametrize("mask",[None,True,[[True],[False]],[True,1],[True],[True,False,True]])
def test_active_source_mask_rejects_nonboolean_or_nonparallel_protocol(mask):
    with pytest.raises(ValueError,match="one explicit boolean"):
        active_source_supervision_mask(dict(uv=[[2.,5.],[3.,6.]],source_active_supervision_mask=mask))


def test_partial_user_revision_retires_only_named_source_positions_and_keeps_visibility():
    old,sample,_,revision=boundary_revision_fixture()
    old.update(uv=[[2.,5.],[3.,5.5],[4.,6.],[5.,6.5]],normal=[[0.,1.]]*4,
               interval_px=[[-2,2],[-3,3],[-4,4],[-5,5]],source_visible_mask=[True,True,False,True])
    original=deepcopy(old)
    revision["supersedes"]=[dict(group_id=old["group_id"],
        original_annotation_sha256=annotation_revision_sha256(old),source_point_indices=[1,3])]
    revised=apply_user_boundary_revisions(dict(samples=[sample]),[revision])
    historical=revised["samples"][0]["annotations"][0]
    assert sample["annotations"][0]==original
    for field in ("uv","normal","interval_px","state","authority","source_visible_mask"):
        assert historical[field]==original[field]
    assert active_source_supervision_mask(historical)==[True,False,True,False]
    assert historical["state"]=="user_confirmed_visual_supervision"
    assert "superseded_by_user_boundary_revision" not in historical
    assert revised["boundary_supervision_revision_ledger"][0]["original_superseded_annotations"]==[original]
    assert revised["boundary_supervision_revision_ledger"][0]["partial_superseded_source_positions"][0]["source_point_indices"]==[1,3]
    assert source_supervision_point_counts(historical)==dict(original_source_position_count=4,
        active_source_supervision_position_count=2,superseded_source_position_count=2,
        other_inactive_source_position_count=0)
    assert annotation_role(historical)["authority_role"]=="direct_user_confirmed_native_UV"
    assert apply_user_boundary_revisions(revised,[revision])==revised


@pytest.mark.parametrize("indices",[[],[0,0],[True],[-1],[1],[0.0],None,[[0]]])
def test_partial_user_revision_rejects_empty_duplicate_invalid_or_out_of_range_indices(indices):
    old,sample,_,revision=boundary_revision_fixture()
    revision["supersedes"][0]["source_point_indices"]=indices
    with pytest.raises(ValueError,match="nonempty unique in-range"):
        apply_user_boundary_revisions(dict(samples=[sample]),[revision])
    assert sample["annotations"][0]==old


def test_later_partial_revision_cannot_retire_a_source_position_again_even_with_current_SHA():
    old,sample,_,revision=boundary_revision_fixture()
    old["uv"]=[[2.,5.],[3.,6.],[4.,7.]]
    revision["supersedes"][0].update(original_annotation_sha256=annotation_revision_sha256(old),source_point_indices=[1])
    revised=apply_user_boundary_revisions(dict(samples=[sample]),[revision])
    current=revised["samples"][0]["annotations"][0]
    next_revision=deepcopy(revision)
    next_revision.update(revision_id="user_lower_strip_v2",version=2)
    next_revision["annotations"][0]["group_id"]="new_user_lower_strip_v2"
    next_revision["supersedes"][0]["original_annotation_sha256"]=annotation_revision_sha256(current)
    with pytest.raises(ValueError,match="retired twice"):
        apply_user_boundary_revisions(revised,[next_revision])
    next_revision["supersedes"][0]["source_point_indices"]=[2]
    twice=apply_user_boundary_revisions(revised,[next_revision])
    assert active_source_supervision_mask(twice["samples"][0]["annotations"][0])==[True,False,False]
    assert twice["boundary_supervision_revision_ledger"][1]["original_superseded_annotations"]==[current]
    assert revised["samples"][0]["annotations"][0]==current


def test_whole_group_revision_remains_compatible_after_partial_history():
    old,sample,_,revision=boundary_revision_fixture()
    old["uv"]=[[2.,5.],[3.,6.],[4.,7.]]
    revision["supersedes"][0].update(original_annotation_sha256=annotation_revision_sha256(old),source_point_indices=[1])
    partial=apply_user_boundary_revisions(dict(samples=[sample]),[revision])
    whole=deepcopy(revision)
    whole.update(revision_id="user_lower_strip_whole_v2",version=2)
    whole["annotations"][0]["group_id"]="new_user_lower_strip_whole_v2"
    whole["supersedes"][0].pop("source_point_indices")
    whole["supersedes"][0]["original_annotation_sha256"]=annotation_revision_sha256(partial["samples"][0]["annotations"][0])
    revised=apply_user_boundary_revisions(partial,[whole])
    historical=revised["samples"][0]["annotations"][0]
    assert historical["uv"]==old["uv"] and historical["state"]=="ignored"
    assert active_source_supervision_mask(historical)==[False,False,False]
    assert source_supervision_point_counts(historical)["superseded_source_position_count"]==3


def test_declared_native_RGB_for_legacy_sample_is_verified_before_visibility_review(tmp_path):
    import numpy as np
    from PIL import Image
    from rinkspace.vision_supervision import file_sha256
    _,sample,corrected,revision=boundary_revision_fixture()
    pixels=np.zeros((12,8,3),np.uint8)
    pixels[:,4:]=150
    frame=tmp_path/"source.png"
    Image.fromarray(pixels).save(frame)
    sample.update(frame_path=str(frame),frame_sha256=file_sha256(frame))
    revision["frame_sha256"]=sample["frame_sha256"]
    revision["decoded_native_RGB_sha256"]=native_pixel_sha256(frame)
    corrected.update(source_visible_mask=[True],source_visibility_evidence=dict(
        source_hash="source_A",time_base="1/12800",source_pts=256,frame_sha256=sample["frame_sha256"],
        decoded_native_RGB_sha256=revision["decoded_native_RGB_sha256"],
        method="assistant_original_source_visibility_review",model_confidence_used=False))
    revised=apply_user_boundary_revisions(dict(samples=[sample]),[revision])
    assert "decoded_native_RGB_sha256" not in sample
    assert revised["samples"][0]["decoded_native_RGB_sha256"]==revision["decoded_native_RGB_sha256"]
    wrong=deepcopy(revision);wrong["decoded_native_RGB_sha256"]="different_native_pixels"
    with pytest.raises(ValueError,match="actual native RGB"):
        apply_user_boundary_revisions(dict(samples=[sample]),[wrong])


def test_boundary_revision_requires_exact_source_clock_image_and_old_record():
    _,sample,_,revision=boundary_revision_fixture()
    original=dict(samples=[sample])
    for field,value in (("source_sha256","another_video"),("time_base","1/15360")):
        wrong=deepcopy(revision)
        wrong["source_binding"][field]=value
        with pytest.raises(ValueError,match="original source image"):
            apply_user_boundary_revisions(original,[wrong])
    wrong=deepcopy(revision)
    wrong["supersedes"][0]["original_annotation_sha256"]="another_original_group"
    with pytest.raises(ValueError,match="original group record"):
        apply_user_boundary_revisions(original,[wrong])
    assert original["samples"][0]["annotations"][0]["state"]=="user_confirmed_visual_supervision"


def test_upper_base_strip_is_board_internal_role_and_cannot_be_new_ice_contact():
    _,sample,_,revision=boundary_revision_fixture()
    wrong=deepcopy(revision)
    wrong["annotations"][0]["boundary_role"]="upper_base_strip_internal_board_edge"
    with pytest.raises(ValueError,match="lower board base-strip edge"):
        apply_user_boundary_revisions(dict(samples=[sample]),[wrong])
    wrong["annotations"][0]["kind"]="background"
    result=apply_user_boundary_revisions(dict(samples=[sample]),[wrong])
    role=annotation_role(result["samples"][0]["annotations"][-1])
    assert role["semantic_boundary_role"]=="upper_base_strip_internal_board_edge"
    assert role["base_strip_belongs_to_board"]
    assert role["semantic_kind"]!="board_top"  # Upper base-strip edge is not the top rail.


def test_new_human_drawing_keeps_occluded_positions_and_old_frame_labels():
    _,old_sample,corrected,revision=boundary_revision_fixture()
    new_sample={**deepcopy(old_sample),"source_pts":512,"frame_sha256":"new_RGB",
                "decoded_native_RGB_sha256":"native_new_RGB","frame_path":"actual_native.png","annotations":[]}
    revision.update(source_binding={key:new_sample[key] for key in ("source_hash","time_base","source_pts")},
                    frame_sha256="new_RGB",supersedes=[],authority=dict(kind="direct_user_drawing",human_authority=True))
    corrected.update(uv=[[2.,8.],[3.,9.],[4.,10.]],source_visible_mask=[True,False,True],
                     confidence=.001,source_visibility_evidence={
                         **revision["source_binding"],"frame_sha256":"new_RGB","decoded_native_RGB_sha256":"native_new_RGB",
                         "method":"assistant_original_source_visibility_review","model_confidence_used":False})
    revision["annotations"]=[corrected]
    packet=dict(schema="rinkspace.user-boundary-supervision-samples.v1",samples=[new_sample],revisions=[revision])
    original=dict(samples=[old_sample])
    result=merge_user_boundary_samples(original,packet)
    assert result["samples"][0]==old_sample and original==dict(samples=[old_sample])
    group=result["samples"][1]["annotations"][0]
    assert group["uv"]==corrected["uv"]  # The hidden position still belongs to the human drawing.
    assert validate_source_visibility(group,new_sample)["visible_source_position_count"]==2
    assert annotation_role(group)["authority_role"]=="direct_user_confirmed_native_UV"
    assert merge_user_boundary_samples(result,packet)==result
    wrong=deepcopy(packet)
    wrong["revisions"][0]["annotations"][0]["source_visible_mask"]=[True,False]
    with pytest.raises(ValueError,match="one explicit boolean"):
        merge_user_boundary_samples(original,wrong)
    wrong=deepcopy(packet)
    wrong["revisions"][0]["annotations"][0]["source_visibility_evidence"]["decoded_native_RGB_sha256"]="another_native"
    with pytest.raises(ValueError,match="actual native RGB"):
        merge_user_boundary_samples(original,wrong)
    wrong=deepcopy(packet)
    wrong["revisions"][0]["annotations"][0]["source_visibility_evidence"]["model_confidence_used"]=True
    with pytest.raises(ValueError,match="model confidence"):
        merge_user_boundary_samples(original,wrong)


@pytest.mark.parametrize("before_type",["shot","ignored"])
def test_native_overlay_places_first_new_view_in_new_shot_without_changing_cameras(tmp_path,before_type):
    import json
    from rinkspace.vision_supervision import file_sha256
    timeline=dict(source_sha256="native_video",time_base="1/100",intervals=[
        dict(id="main",start_pts=0,end_pts=21,camera_id="main_long_side",authority="user_original",interval_type=before_type),
        dict(id="low",start_pts=21,end_pts=30,camera_id="low_end",authority="assistant_existing")])
    path=tmp_path/"timeline.json"
    path.write_text(json.dumps(timeline),encoding="utf-8")
    overrides=[]
    for row,new_start,new_end in zip(timeline["intervals"],[0,20],[20,30]):
        corrected={**row,"start_pts":new_start,"end_pts":new_end}
        overrides.append(dict(shot_id=row["id"],original_start_pts=row["start_pts"],original_end_pts_exclusive=row["end_pts"],
            start_pts=new_start,end_pts_exclusive=new_end,original_row=row,corrected_row=corrected,
            original_row_json_sha256=annotation_revision_sha256(row),corrected_row_json_sha256=annotation_revision_sha256(corrected),
            camera_id=row["camera_id"],camera_authority=row["authority"],camera_review_status=None))
    overlay=dict(schema="rinkspace.native-shot-boundary-correction.v1",source_sha256="native_video",time_base="1/100",
                 original_timeline_sha256=file_sha256(path),shot_overrides=overrides)
    overlay_path=tmp_path/"overlay.json"
    overlay_path.write_text(json.dumps(overlay),encoding="utf-8")
    old_reference=tmp_path/"older_frozen_packet_overlay.json"
    old_reference.write_text('{"historical":true}',encoding="utf-8")
    selected=native_boundary_overlay_reference(dict(native_boundary_overlay=str(overlay_path),
        native_boundary_overlay_sha256=file_sha256(overlay_path)),
        dict(source_boundary_overlay_path=str(old_reference),source_boundary_overlay_sha256=file_sha256(old_reference)))
    assert selected["reference_origin"]=="configured_native_boundary_overlay"
    actual=timeline_with_native_boundary_overlay(path,"native_video",selected["path"])
    sample=dict(source_hash="native_video",time_base="1/100",source_pts=20)
    plan=build_shot_plan([sample],actual,evaluation_pts=[],development_pts=[10] if before_type=="shot" else [])
    assert split_indices([sample],plan)["train"]==[0]
    assert plan["shots"][-1]["id"]=="low" and plan["shots"][-1]["start_pts"]==20
    assert actual["intervals"][-1]["authority"]=="assistant_existing"
    assert actual["intervals"][0]["end_pts"]==20  # Ignored adjacent intervals must move atomically too.
    assert json.loads(path.read_text(encoding="utf-8"))==timeline
    overlay["shot_overrides"][-1]["corrected_row"]["camera_id"]="invented_camera"
    changed=overlay["shot_overrides"][-1]["corrected_row"]
    overlay["shot_overrides"][-1]["corrected_row_json_sha256"]=annotation_revision_sha256(changed)
    overlay_path.write_text(json.dumps(overlay),encoding="utf-8")
    with pytest.raises(ValueError,match="camera taxonomy"):
        timeline_with_native_boundary_overlay(path,"native_video",overlay_path)


def test_configured_native_overlay_digest_cannot_fall_back_to_old_packet(tmp_path):
    from rinkspace.vision_supervision import file_sha256
    old=tmp_path/"old.json";old.write_text('{"old":true}',encoding="utf-8")
    current=tmp_path/"new.json";current.write_text('{"new":true}',encoding="utf-8")
    packet=dict(source_boundary_overlay_path=str(old),source_boundary_overlay_sha256=file_sha256(old))
    original=deepcopy(packet)
    with pytest.raises(ValueError,match="declared SHA256"):
        native_boundary_overlay_reference(dict(native_boundary_overlay=str(current),
            native_boundary_overlay_sha256=file_sha256(old)),packet)
    assert packet==original


def test_legacy_packet_overlay_retains_its_own_digest_and_relative_current_path(tmp_path):
    from rinkspace.vision_supervision import file_sha256
    path=tmp_path/"native.json";path.write_text('{"native":true}',encoding="utf-8")
    packet=dict(source_boundary_overlay_path=str(path),source_boundary_overlay_sha256=file_sha256(path))
    reference=native_boundary_overlay_reference({},packet)
    assert reference["reference_origin"]=="legacy_user_boundary_packet_reference"
    assert reference["sha256"]==packet["source_boundary_overlay_sha256"]
    selected=native_boundary_overlay_reference(dict(native_boundary_overlay="native.json",
        native_boundary_overlay_sha256=file_sha256(path)),root=tmp_path)
    assert selected["path"]==str(path.resolve())
    packet["source_boundary_overlay_sha256"]="changed_historical_reference"
    with pytest.raises(ValueError,match="declared SHA256"):
        native_boundary_overlay_reference({},packet)


def test_board_internal_region_is_source_bound_and_preserves_human_contact_positions():
    _,sample,_,_=boundary_revision_fixture()
    sample.update(decoded_native_RGB_sha256="native_RGB",regions=[])
    region=dict(region_id="base_strip_interior",kind="background",rectangle_xyxy=[1,1,3,4],
                physical_role="board_bottom_strip_not_ice",state="source_reviewed_region",
                boundary_pixel_GT=False,model_projection_used=False,decoded_native_RGB_sha256="native_RGB")
    update={key:sample[key] for key in ("source_hash","time_base","source_pts","frame_sha256")}
    packet=dict(schema="rinkspace.source-reviewed-material-regions.v1",sample_updates=[{**update,"regions":[region]}])
    original=dict(samples=[sample]);untouched=deepcopy(original)
    result=merge_source_reviewed_regions(original,packet)
    assert original==untouched and result["samples"][0]["annotations"]==sample["annotations"]
    assert result["samples"][0]["regions"]==[region]
    assert merge_source_reviewed_regions(result,packet)==result
    wrong=deepcopy(packet);wrong["sample_updates"][0]["source_hash"]="different_source_same_PTS"
    with pytest.raises(ValueError,match="no original sample"):
        merge_source_reviewed_regions(original,wrong)


def test_reject_only_region_preserves_unknown_other_classes_and_human_authority():
    _,sample,_,_=boundary_revision_fixture()
    sample.update(decoded_native_RGB_sha256="native_RGB",regions=[])
    region=dict(region_id="actual_upper_board_noncontact",excluded_kinds=["ice_contact"],rectangle_xyxy=[1,1,3,4],
                state="source_reviewed_region",boundary_pixel_GT=False,model_projection_used=False,
                decoded_native_RGB_sha256="native_RGB",reject_only_authority="actual_visual_role_absence",
                confidence_used_as_rejection_evidence=False,model_probability_used_as_rejection_evidence=False)
    update={key:sample[key] for key in ("source_hash","time_base","source_pts","frame_sha256")}
    packet=dict(schema="rinkspace.source-reviewed-material-regions.v1",sample_updates=[{**update,"regions":[region]}])
    original=dict(samples=[sample]);untouched=deepcopy(original)
    result=merge_source_reviewed_regions(original,packet)
    assert original==untouched
    assert result["samples"][0]["annotations"]==sample["annotations"]
    added=result["samples"][0]["regions"][0]
    assert "kind" not in added and added["excluded_kinds"]==["ice_contact"]
    assert merge_source_reviewed_regions(result,packet)==result
    for change in ({"kind":"background"},{"excluded_kinds":[]},{"excluded_kinds":["ice_contact","ice_contact"]},
                   {"excluded_kinds":"ice_contact"}):
        wrong=deepcopy(packet);wrong["sample_updates"][0]["regions"][0].update(change)
        with pytest.raises(ValueError,match="no positive kind"):
            merge_source_reviewed_regions(original,wrong)
    for change in ({"reject_only_authority":"model_low_confidence"},{"confidence_used_as_rejection_evidence":True},
                   {"model_probability_used_as_rejection_evidence":True}):
        wrong=deepcopy(packet);wrong["sample_updates"][0]["regions"][0].update(change)
        with pytest.raises(ValueError,match="actual source-role absence"):
            merge_source_reviewed_regions(original,wrong)
    wrong=deepcopy(packet);wrong["sample_updates"][0]["regions"][0]["boundary_pixel_GT"]=True
    with pytest.raises(ValueError,match="separate from precise boundary labels"):
        merge_source_reviewed_regions(original,wrong)


def test_human_drawing_trains_directly_without_changing_source_positions():
    annotation = dict(state="user_trace_weak", kind="ice_paint", uv=[[12.5, 18.0]],
                      interval_px=[-18, 18], authority={"human_authority": True}, confidence=0.001)
    original = dict(samples=[dict(annotations=[annotation])])
    untouched = deepcopy(original)
    result = apply_supervision_roles(original)
    changed = result["samples"][0]["annotations"][0]
    assert original == untouched
    assert changed["uv"] == annotation["uv"]
    assert changed["interval_px"] == annotation["interval_px"]
    assert changed["original_supervision_state"] == "user_trace_weak"
    assert changed["state"] == "user_confirmed_visual_supervision"
    assert changed["vision_supervision"]["semantic_supervision_enabled"]
    assert not changed["vision_supervision"]["new_pixel_ground_truth_created"]


def test_confirmed_print_definition_retains_source_semantic_supervision():
    annotation = dict(state="source_reviewed_interval", kind="ad_boundary", uv=[[300, 480]],
                      boundary_role="external_printed_ink_component",
                      group_id="source_reviewed_Ad23_native_print_boundary_3",
                      authority="assistant_original_RGB_review", supervision_enabled=True)
    role = annotation_role(annotation)
    assert role["semantic_supervision_enabled"]
    assert role["semantic_boundary_role"] == "actual_printed_pattern_external_contour"
    assert role["printed_boundary_definition_authority"] == "direct_user_confirmation_20261006"
    assert not role["geometry_as_panel_border"]
    assert role["overall_occupied_extent_is_separate"]
    hole = annotation_role({**annotation, "boundary_role": "internal_ink_hole"})
    assert hole["semantic_boundary_role"] != role["semantic_boundary_role"]


def test_pixel_profile_does_not_create_supervision_or_override_user():
    candidate = dict(kind="ice_paint", state="RGB_edge_candidate", is_training_annotation=False)
    assert not annotation_role(candidate)["semantic_supervision_enabled"]
    human = {**candidate, "state": "user_confirmed_visual_supervision", "authority": "human"}
    assert annotation_role(human)["semantic_supervision_enabled"]
    occluded = {**human, "state": "occluded", "authority": {"human_authority": True}}
    assert not annotation_role(occluded)["semantic_supervision_enabled"]


def test_existing_outer_print_target_survives_without_promoting_free_curve_fragments():
    rim = dict(state="source_reviewed_weak_stroke", kind="ad_boundary", primitive="circle",
               group_id="spin_Ad24_outer_black_rim_native230231552_reviewed_training_weak",
               feature_id="Ad24", uv=[[637.2723250448169, 315.0]], interval_px=[-4, 4],
               target_role="outermost_visible_circular_print_stroke; not advertisement rectangular border",
               authority="two_agent_original_pixel_review_and_existing_physical_ad_identity")
    fragment = dict(state="source_reviewed_interval", kind="ad_boundary", primitive="free_curve",
                    group_id="source_reviewed_Ad24_native_print_boundary_0", feature_id="Ad24",
                    uv=[[2, 482]], interval_px=[-3, 3], authority="original_source_review",
                    boundary_role="visible_round_logo_ink_fragment_precise_ring_role_requires_edge_review")
    original = dict(samples=[dict(annotations=[rim, fragment])])
    untouched = deepcopy(original)
    result = apply_supervision_roles(original)
    assert original == untouched
    outer, ink = result["samples"][0]["annotations"]
    assert outer["uv"] == rim["uv"] and outer["interval_px"] == rim["interval_px"]
    assert outer["target_role"] == rim["target_role"]
    assert outer["vision_supervision"]["geometry_role"] == "outermost_visible_circular_print_stroke"
    assert outer["vision_supervision"]["original_target_role"] == rim["target_role"]
    assert outer["vision_supervision"]["authority_role"] == "assistant_original_source_interval"
    assert ink["vision_supervision"]["geometry_role"] == fragment["boundary_role"]
    assert ink["primitive"] == "free_curve" and ink["uv"] == fragment["uv"]
    # A real inner hole remains useful semantic evidence, without acquiring
    # the encompassing advertisement's outer-circle geometric target.
    hole = annotation_role({**fragment, "boundary_role": "internal_ink_hole", "target_role": rim["target_role"]})
    assert hole["geometry_role"] == "internal_ink_hole"
    for role in (outer["vision_supervision"], ink["vision_supervision"], hole):
        assert role["semantic_supervision_enabled"]
        assert not role["geometry_as_panel_border"]
        assert not role["new_pixel_ground_truth_created"]


def test_even_human_identity_cannot_make_projected_uv_source_labels():
    annotation = dict(state="user_confirmed_visual_supervision", kind="ice_paint",
                      model_projection_used_to_generate_or_select_uv=True)
    with pytest.raises(ValueError, match="model projection"):
        apply_supervision_roles(dict(samples=[dict(annotations=[annotation])]))


def test_whole_shot_split_and_native_rgb_alias_guard():
    timeline = dict(source_sha256="actual_content_SHA", time_base="1/100",
                    intervals=[dict(id="train", start_pts=0, end_pts=100),
                               dict(id="dev", start_pts=100, end_pts=200),
                               dict(id="held", start_pts=200, end_pts=300)])
    samples = [dict(source_hash="actual_content_SHA", source_pts=20, url="first_URL", native_RGB_sha256="train_RGB"),
               dict(source_hash="actual_content_SHA", source_pts=150, url="second_URL", native_RGB_sha256="dev_RGB"),
               dict(source_hash="actual_content_SHA", source_pts=299, native_RGB_sha256="held_RGB")]
    plan = build_shot_plan(samples, timeline, evaluation_pts=[210], development_pts=[120])
    assert split_indices(samples, plan) == dict(train=[0], development=[1], independent_evaluation=[2], pending_source_timeline=[])
    # A representative at 210 reserves 299 too, because the entire shot is held.
    assert plan["shots"][-1]["end_pts"] == 300
    assert not plan["input_url_used_for_split"]
    aliased = deepcopy(samples)
    aliased[-1]["native_RGB_sha256"] = "train_RGB"
    with pytest.raises(ValueError, match="native RGB content"):
        split_indices(aliased, plan)


def test_same_shot_cannot_be_development_and_independent():
    timeline = dict(source_sha256="same", time_base="1/100", intervals=[dict(start_pts=0, end_pts=100)])
    with pytest.raises(ValueError, match="both development and independent"):
        build_shot_plan([], timeline, evaluation_pts=[50], development_pts=[70])
    assert source_time(226160640, "1/15360") == "04:05:24.000"


def test_boundary_definition_never_confirms_extracted_pixel_precision():
    definition = dict(authority=dict(kind="direct_user_confirmation"),
                      definition=dict(primary_visual_boundary="actual_printed_graphic_outer_contours",
                                      visible_rectangle_required=False),
                      authority_scope=dict(assistant_extracted_pixel_precision_confirmed_by_this_reply=False))
    validate_boundary_definition(definition)
    definition["authority_scope"]["assistant_extracted_pixel_precision_confirmed_by_this_reply"] = True
    with pytest.raises(ValueError, match="pixel precision"):
        validate_boundary_definition(definition)


def test_native_content_sha_matches_png_and_bundle_member(tmp_path):
    import numpy as np
    from PIL import Image
    rgb = np.arange(12 * 15 * 3, dtype=np.uint8).reshape(12, 15, 3)
    png = tmp_path / "original.png"
    bundle = tmp_path / "source.npz"
    Image.fromarray(rgb).save(png)
    np.savez_compressed(bundle, native_frame=rgb, other_frame=rgb[::-1].copy())
    assert native_pixel_sha256(png) == native_pixel_sha256(bundle, "native_frame")
    assert native_pixel_sha256(png) != native_pixel_sha256(bundle, "other_frame")


def test_temporal_labels_use_actual_rgb_and_rebase_source_interval():
    guide = dict(source_group_id="human_lane", id="Lane1", kind="ice_paint", primitive="line",
                 physical_role="painted_stroke_center_axis", identity_authority="original_human")
    reading = dict(source_indices=[0, 1], actual_source_uv=[[5.25, 3.0], [3.0, 5.5]],
                   guide=dict(uv=[[0, 3], [3, 0]], normals=[[1, 0], [0, 1]]),
                   support=dict(normal_interval_px=[[5, 5.5], [5, 6]], method="actual_RGB_pair"))
    row = measured_interval_annotation(guide, reading, [True, False], source_pts=100, authority="RGB_source")
    assert row["uv"] == [[5.25, 3.0]]
    assert row["uv"] != reading["guide"]["uv"][:1]
    assert row["interval_px"] == [[-0.25, 0.25]]
    assert row["source_anchor_point_indices"] == [0]
    assert row["state"] == "source_reviewed_interval"
    assert not row["propagated_search_UV_used_as_labels"]
    assert not row["pixel_ground_truth"]
    assert measured_interval_annotation(guide, reading, [False, False], source_pts=100, authority="RGB_source") is None


def test_nearby_frames_do_not_change_shot_weights_or_remove_human_pool():
    timeline = dict(source_sha256="actual_SHA", time_base="1/100",
                    intervals=[dict(id="one", start_pts=0, end_pts=100),
                               dict(id="two", start_pts=100, end_pts=200)])
    human = dict(state="user_confirmed_visual_supervision", kind="ice_paint", authority="human")
    rgb = dict(state="source_reviewed_interval", kind="ice_paint", authority="actual_RGB")
    samples = [dict(source_hash="actual_SHA", source_pts=20, annotations=[human]),
               dict(source_hash="actual_SHA", source_pts=120, annotations=[human])]
    plan = build_shot_plan(samples, timeline, evaluation_pts=[], development_pts=[])
    before = shot_sampling_groups(samples, plan)
    expanded = samples + [dict(source_hash="actual_SHA", source_pts=pts, annotations=[rgb]) for pts in range(21, 90)]
    after = shot_sampling_groups(expanded, plan)
    assert [g["uniform_shot_weight"] for g in before] == [g["uniform_shot_weight"] for g in after] == [0.5, 0.5]
    assert after[0]["human_indices"] == [0]
    assert len(after[0]["source_RGB_indices"]) == 69
    assert after[0]["sampling_pools"] == [[0], list(range(2, 71))]
    assert after[1]["human_indices"] == [1]


def test_split_checks_the_native_clock():
    timeline = dict(source_sha256="same", time_base="1/100", intervals=[dict(start_pts=0, end_pts=100)])
    sample = dict(source_hash="same", source_pts=50, time_base="1/50")
    plan = build_shot_plan([sample], timeline, evaluation_pts=[], development_pts=[])
    with pytest.raises(ValueError, match="native clocks"):
        split_indices([sample], plan)


def test_explicit_native_neighbour_scope_rejects_cuts_and_wrong_source():
    timeline = dict(source_sha256="real_SHA", time_base="1/12800",
                    intervals=[dict(id="before", start_pts=0, end_pts=1024),
                               dict(id="after", start_pts=1024, end_pts=2048)])
    sample = dict(source_hash="real_SHA", source_pts=768, time_base="1/12800")
    plan = build_shot_plan([sample], timeline, evaluation_pts=[], development_pts=[])
    original = dict(samples=[sample])
    scoped = attach_native_shot_scopes(original, plan)["samples"][0]
    assert "source_shot_scope" not in sample
    assert scoped["source_shot_scope"]["end_pts_exclusive"] == 1024
    row = dict(source_video_sha256="real_SHA", time_base="1/12800", native_pts=[512, 768, 768])
    assert validate_sequence_shot_scope(scoped, row)["shot_id"] == "before"
    # The exact cut PTS belongs to the following shot, never this context.
    with pytest.raises(ValueError, match="across its native shot cut"):
        validate_sequence_shot_scope(scoped, {**row, "native_pts":[512, 768, 1024]})
    with pytest.raises(ValueError, match="source/clock"):
        validate_sequence_shot_scope(scoped, {**row, "source_video_sha256":"different_actual_SHA"})
    with pytest.raises(ValueError, match="middle native PTS"):
        validate_sequence_shot_scope(scoped, {**row, "native_pts":[512, 767, 768]})
    with pytest.raises(ValueError, match="verified native shot"):
        native_shot_scope({**sample,"source_hash":"different_actual_SHA"}, plan)
    with pytest.raises(ValueError, match="Cached neighbour shot scope"):
        validate_sequence_shot_scope({**scoped,"source_hash":"different_actual_SHA"}, row)
