"""Human source guides must retain boundary meaning, not infer it from shape."""
from copy import deepcopy

import numpy as np
import pytest

from rinkspace.vision import (confirmed_source_guides, _annotations_in_crop,
                             source_ignored_region_box, source_guide_visibility_override_crop)


def test_human_boundary_behind_person_keeps_geometry_and_skips_only_hidden_pixel_loss():
    boundary = annotation("ice_contact", "ice_contact", "free_curve")
    boundary["source_visible_mask"] = [True, False, True]
    original = sample(boundary)
    retained = deepcopy(original)
    selected, = _annotations_in_crop(original, (0, 0), (100, 100))
    assert original == retained
    assert selected["source_point_indices"].tolist() == [0, 2]
    assert selected["uv"].tolist() == [boundary["uv"][0], boundary["uv"][2]]
    assert selected["source_visible_mask"].tolist() == [True, True]
    assert selected["state"] == "user_confirmed_visual_supervision"
    # Compute normals on the entire original curve, retaining local directions
    # rather than inventing a tangent across the hidden source positions.
    tangent = np.gradient(np.asarray(boundary["uv"]), axis=0)
    normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    np.testing.assert_allclose(selected["normal"], normal[[0, 2]])


@pytest.mark.parametrize("mask", [[True, False], [1, 0, 1], [[True, False, True]]])
def test_visibility_is_explicit_boolean_per_original_position(mask):
    boundary = annotation("ice_contact", "ice_contact", "free_curve")
    boundary["source_visible_mask"] = mask
    with pytest.raises(ValueError, match="one explicit boolean"):
        _annotations_in_crop(sample(boundary), (0, 0), (100, 100))


def test_user_local_revision_changes_effective_points_without_joining_retired_gap():
    boundary = annotation("ice_contact", "ice_contact", "free_curve",
                          source_active_supervision_mask=[True, False, True], closed=True)
    original = sample(boundary)
    preserved = deepcopy(original)
    selected, = _annotations_in_crop(original, (0, 0), (100, 100))
    assert selected["source_point_indices"].tolist() == [0, 2]
    guide, = confirmed_source_guides(original)
    assert guide["points_uv"] == [boundary["uv"][0], boundary["uv"][2]]
    assert guide["segment_ids"] == [0, 1]
    assert guide["original_source_point_indices"] == [0, 2]
    assert not guide["closed"]
    assert guide["identity_confirmed"]
    # The two measured sides retain their original local, opposing normals;
    # replacing the missing middle with a chord would make both vertical.
    assert guide["normals"][0][0] < 0 < guide["normals"][1][0]
    assert original == preserved


def test_whole_superseded_curve_cannot_reenter_source_guides_through_human_authority():
    boundary = annotation("ice_contact", "ice_contact", "free_curve",
                          superseded_by_user_boundary_revision="new_direct_user_drawing")
    original = sample(boundary)
    assert _annotations_in_crop(original, (0, 0), (100, 100)) == []
    packet = confirmed_source_guides(original, return_packet=True)
    assert packet["guides"] == []
    assert packet["unresolved"][0]["reason"] == "superseded_by_later_direct_user_boundary_revision"


def test_historical_user_drawing_remains_human_source_guidance():
    boundary = annotation("ice_contact", "ice_contact", "free_curve")
    boundary["state"] = "user_trace_weak"
    guide, = confirmed_source_guides(sample(boundary))
    assert guide["origin"] == "user_confirmed_visual_supervision"
    assert guide["identity_confirmed"]
    assert guide["points_uv"] == boundary["uv"]


def test_legacy_native_x_range_retains_its_original_exclusion_scope():
    assert source_ignored_region_box({"x_range_source_px": [1375, 1500]}, (1080, 1920)) == (1375, 0, 1500, 1080)
    assert source_ignored_region_box({"rectangle_xyxy": [-1.5, 12.5, 30.25, 35.75]}, (40, 50)) == (0, 12, 31, 36)


def visibility_reviewed_boundary():
    evidence = dict(source_sha256="a"*64, source_pts=15360, time_base="1/15360",
                    source_crop_native_xyxy=[9, 18, 16, 24], model_confidence_used=False,
                    original_legacy_sample_ignored_x_ranges_are_historical_and_do_not_veto_this_new_source_review=True)
    return annotation("ice_contact", "ice_contact", "free_curve",
                      source_visible_mask=[True]*3, source_visibility_evidence=evidence)


def test_new_visible_review_is_feature_specific_and_expires_on_next_native_frame():
    guide, = confirmed_source_guides(sample(visibility_reviewed_boundary()))
    binding = guide["source_binding"]
    assert source_guide_visibility_override_crop(guide, binding, (1080, 1920)) == (9, 18, 16, 24)
    assert source_guide_visibility_override_crop(guide, {**binding, "source_pts":15616}, (1080, 1920)) is None
    hidden = {**guide, "source_visible_mask":[True,False,True]}
    assert source_guide_visibility_override_crop(hidden, binding, (1080, 1920)) is None
    incorrect = visibility_reviewed_boundary()
    incorrect["source_visibility_evidence"]["source_sha256"] = "b"*64
    with pytest.raises(ValueError, match="another native RGB frame"):
        confirmed_source_guides(sample(incorrect))


def test_reader_keeps_other_feature_legacy_exclusions_when_new_user_crop_is_visible(monkeypatch):
    # Exercise the real reader's exclusion assembly on CPU; a stub measures
    # nothing and captures only the masks passed into its CUDA source reader.
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace
    from rinkspace.vision import SourceRinkReader
    from rinkspace import vision_geometry
    from rinkspace.vision_geometry import source_annotations_to_guides
    revised = visibility_reviewed_boundary()
    other = annotation("board_top", "board_top", "free_curve")
    binding = dict(source_sha256="a"*64,source_pts=15360,time_base="1/15360",shot_id="shot",native_hw=[40,50])
    guides = source_annotations_to_guides([revised,other],source_binding=binding,
                                         annotation_source_binding=binding)["guides"]
    prediction = dict(source_binding=binding,prediction_source_binding=binding,memory_update="shot_reset",
                      model_sha256="b"*64,source_clock_evidence={},new_observation=True,motion={})
    reader = SourceRinkReader.__new__(SourceRinkReader)
    reader.vision = SimpleNamespace(device=torch.device("cpu"),current_image=torch.zeros((1,3,40,50)),
                                   predict=lambda *a,**k:prediction,
                                   model_usage=lambda:dict(sha256='b'*64,path='cpu_test_stub'))
    reader.active_guides=[];reader.active_binding=None;reader.active_global_exclusion=None
    monkeypatch.setattr(vision_geometry,"extract_source_primitives",lambda *a,**k:{"components":[]})
    observed = {}
    def capture(rgb,probabilities,source_guides,**kwargs):
        observed.update(guides=source_guides,global_mask=kwargs["source_exclusion_mask"])
        return dict(readings=[],metadata={})
    monkeypatch.setattr(vision_geometry,"guided_source_reading",capture)
    monkeypatch.setattr(vision_geometry,"coherent_named_source_reading",lambda value:value)
    prediction["class_probabilities"] = torch.zeros((1,6,40,50));reader.vision.classes=["background"]*6
    reader.read(np.zeros((40,50,3),np.uint8),binding,guides_2d=guides,
                ignored_regions=[{"x_range_source_px":[10,15]}],reset=True)
    first,second=observed["guides"]
    assert not first["source_exclusion_mask"][0,0,18:24,10:15].any()
    assert second["source_exclusion_mask"][0,0,18:24,10:15].all()
    assert first["source_exclusion_mask"][0,0,0:18,10:15].all()
    assert reader.active_global_exclusion[0,0,:,10:15].all()
    assert observed["global_mask"] is None


def sample(*annotations):
    return dict(source_hash="a" * 64, source_pts=15360, time_base="1/15360",
                frame_size_wh=[1920, 1080], annotations=list(annotations))


def annotation(feature, kind, primitive, **extra):
    return dict(feature_id=feature, group_id=f"user_{feature}", kind=kind,
                primitive=primitive, uv=[[10., 20.], [12., 21.], [14., 20.]],
                state="user_confirmed_visual_supervision",
                authority=dict(human_authority=True, user_text_alias=feature), **extra)


def test_round_advertisement_retains_explicit_print_rim_meaning():
    rim = annotation("Ad24", "ad_boundary", "circle",
                     target_role="outermost_visible_circular_print_stroke; not rectangular border")
    original = sample(rim)
    preserved = deepcopy(original)
    packet = confirmed_source_guides(original, return_packet=True)
    guide, = packet["guides"]
    assert original == preserved
    assert guide["physical_role"] == "printed_stroke_outer_boundary"
    assert guide["points_uv"] == rim["uv"]
    assert guide["identity_authority"] == rim["authority"]
    assert "rgb_transition_polarity" not in guide


def test_print_external_and_hole_roles_remain_separate():
    external = annotation("Ad21", "ad_boundary", "free_curve",
                          boundary_role="external_printed_ink_component")
    hole = annotation("Ad21", "ad_boundary", "free_curve",
                      boundary_role="internal_ink_hole")
    packet = confirmed_source_guides(sample(external, hole), return_packet=True)
    assert [g["physical_role"] for g in packet["guides"]] == [
        "printed_pattern_external_contour", "printed_pattern_internal_hole_contour"]
    assert all("signal" not in g for g in packet["guides"])


def test_circle_shape_alone_cannot_define_filled_disc_boundary():
    outer = annotation("R2", "ice_paint", "circle")
    packet = confirmed_source_guides(sample(outer), return_packet=True)
    assert packet["guides"] == []
    assert packet["unresolved"][0]["reason"] == "source_paint_boundary_vs_stroke_axis_role_unresolved"
    assert packet["metadata"]["human_source_visual_supervision_has_priority"]


def test_source_role_and_normals_define_actual_disc_transition():
    outer = annotation("R2", "ice_paint", "circle", ring_role="outer",
                       user_drawing_branch="upper", normal=[[0., -1.]] * 3)
    roles = {"R2:outer": dict(physical_role="filled_disc_boundary",
                              source_material_transition="gray_disc_vs_white_ice")}
    guide, = confirmed_source_guides(sample(outer), role_definitions=roles)
    assert guide["id"] == "R2:outer"
    assert guide["physical_role"] == "filled_disc_boundary"
    assert guide["normals"] == outer["normal"]
    assert guide["rgb_transition_polarity"] == "lightening_along_normal"
    assert guide["display_label"] == "R2"


def test_one_confirmed_position_is_not_discarded_for_lacking_a_tangent():
    point = annotation("B1", "ice_paint", "line")
    point["uv"] = [[10.5, 20.25]]
    original = deepcopy(point)
    selected, = _annotations_in_crop(sample(point), (0, 0), (40, 40))
    assert point == original
    np.testing.assert_array_equal(selected["uv"], point["uv"])
    np.testing.assert_array_equal(selected["normal"], [[0., 0.]])


def test_measured_single_point_interval_keeps_its_actual_normal():
    point = dict(group_id="actual_interval", feature_id="Lane1", kind="ice_paint",
                 primitive="line", uv=[[12., 18.]], normal=[[.6, .8]], interval_px=[[-1.25, 1.25]],
                 state="source_reviewed_interval", authority="actual_source_RGB")
    selected, = _annotations_in_crop(sample(point), (0, 0), (40, 40))
    np.testing.assert_allclose(selected["normal"], point["normal"])
    np.testing.assert_array_equal(selected["interval_px"], point["interval_px"])
    with pytest.raises(ValueError, match="measured source normal"):
        _annotations_in_crop(sample({k: v for k, v in point.items() if k != "normal"}), (0, 0), (40, 40))
