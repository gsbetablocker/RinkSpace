"""An exact programme run cannot be replaced by a completed local diagnostic."""
from copy import deepcopy

import pytest

from rinkspace.vision_supervision import (source_sequence_intervals, source_sequence_coverage_audit,
                                        crosscheck_source_observation_groups)


def fixture():
    digest, model, packet, proof = "a" * 64, "b" * 64, "c" * 64, "d" * 64
    intervals = [dict(id="one", start_pts=0, end_pts_exclusive=512, native_frame_count=2,
                      camera_id="low", camera_authority="actual_user_confirmation"),
                 dict(id="two", start_pts=512, end_pts_exclusive=1024, native_frame_count=2,
                      camera_id="main", camera_authority="assistant_original_source_review")]
    manifest = dict(source_sha256=digest, time_base="1/15360", start_pts=0,
        end_pts_exclusive=1024, native_step_pts=256, frame_count=4, source_intervals=intervals,
        primary_source_intervals=dict(sha256=packet), runtime_model=dict(sha256=model),
        source_clock_evidence=dict(exposure_proof=dict(sha256=proof)))
    summary = dict(source_sha256=digest, classification_sha256=packet, model_sha256=model,
        frame_count=4, source_pts_complete=True, all_decoded_original_frames_processed=True,
        status="source_range_complete", source_code_sha256_at_run_start={"reader.py": "actual"},
        source_code_sha256_at_run_end={"reader.py": "actual"}, prediction_is_pixel_truth=False,
        accuracy_accepted=False, source_board_foot_annotation_pixels_used_as_runtime_input=False,
        independent_mode_has_no_test_annotation_injection=True,
        shots=[dict(**interval, processed_frames=2, first_pts=interval["start_pts"],
                    last_pts=interval["end_pts_exclusive"]-256, source_pts_complete=True)
               for interval in intervals])
    groups=[dict(full_pts=[0,256]),dict(full_pts=[512]),dict(full_pts=[768])]
    rows=[]
    for pts in range(0,1024,256):
        shot=intervals[int(pts>=512)]
        first=pts==shot["start_pts"]
        rows.append(dict(source_binding=dict(source_sha256=digest,time_base="2/30720",source_pts=pts,
            shot_id=shot["id"],installation_id=shot["camera_id"],canonical_exposure_pts=0 if pts==256 else pts,
            exposure_proof_sha256=proof), model_sha256=model, RGB_sha256="e"*64,
            source_seed_injected=False,memory_update="shot_reset" if first else "continued",
            instance_source_shot_reset_requested=first,new_observation=pts!=256,issues=[]))
    return manifest,summary,rows,groups


def test_complete_exact_sequence_verifies_all_cuts_and_only_confirmed_exposure_reuse():
    manifest,summary,rows,groups=fixture()
    untouched=deepcopy((manifest,summary,rows,groups))
    result=source_sequence_coverage_audit(manifest,summary,observations=iter(rows),exposure_groups=groups)
    assert result["complete_native_source_sequence_verified"]
    assert result["actual_new_observation_counts"]=={"one":1,"two":2}
    assert result["cut_reset_count"]==2
    assert not result["accuracy_accepted"] and not result["independent_accuracy"]
    assert (manifest,summary,rows,groups)==untouched


def test_completed_two_frame_diagnostic_is_not_complete_programme():
    manifest,summary,rows,groups=fixture()
    summary["shots"]=summary["shots"][:1];summary["frame_count"]=2
    result=source_sequence_coverage_audit(manifest,summary,observations=rows[:2],exposure_groups=groups)
    assert not result["complete_native_source_sequence_verified"]
    assert result["failure_counts"]["actual_native_rows_do_not_cover_requested_sequence"]==1


def test_summary_only_cannot_verify_actual_source_rows():
    manifest,summary,_,groups=fixture()
    result=source_sequence_coverage_audit(manifest,summary,exposure_groups=groups)
    assert not result["source_rows_checked"]
    assert not result["complete_native_source_sequence_verified"]


def test_explicit_assisted_calibration_preserves_seed_truth_and_does_not_claim_independence():
    manifest, summary, rows, groups = fixture()
    policy = dict(schema="rinkspace.explicit-assisted-source-seed-policy.v1", mode="assisted",
        source_sha256=manifest["source_sha256"], time_base="2/30720", seed_packet_sha256="f"*64,
        human_coordinates_use="search_and_identity_only_not_output",
        assisted_calibration_not_independent_validation=True,
        allowed_source_bindings=[rows[2]["source_binding"]])
    summary.update(mode="assisted", source_seed_sha256="f"*64,
        independent_mode_has_no_test_annotation_injection=False, actual_RGB_is_only_output_geometry=True)
    rows[2]["source_seed_injected"] = True
    original = deepcopy((summary, rows, policy))
    result = source_sequence_coverage_audit(manifest, summary, observations=rows,
        exposure_groups=groups, source_seed_policy=policy)
    assert result["complete_native_source_sequence_verified"]
    assert result["runtime_input_policy"] == "explicit_assisted_calibration"
    assert not result["independent_accuracy"] and not result["accuracy_accepted"]
    assert (summary, rows, policy) == original
    rows[3]["source_seed_injected"] = True
    result = source_sequence_coverage_audit(manifest, summary, observations=rows,
        exposure_groups=groups, source_seed_policy=policy)
    assert result["failure_counts"]["native_row_source_seed_differs_from_explicit_calibration_policy"] == 1
    summary["source_seed_sha256"] = "e"*64
    result = source_sequence_coverage_audit(manifest, summary, observations=rows,
        exposure_groups=groups, source_seed_policy=policy)
    assert result["failure_counts"]["runtime_input_or_accuracy_contract_not_verified"] == 1


def test_foreign_or_duplicate_assisted_seed_policy_is_rejected_without_rebinding():
    manifest, summary, rows, groups = fixture()
    policy = dict(schema="rinkspace.explicit-assisted-source-seed-policy.v1", mode="assisted",
        source_sha256="e"*64, time_base="1/15360", seed_packet_sha256="f"*64,
        human_coordinates_use="search_and_identity_only_not_output",
        assisted_calibration_not_independent_validation=True,
        allowed_source_bindings=[rows[2]["source_binding"]])
    with pytest.raises(ValueError, match="same-source"):
        source_sequence_coverage_audit(manifest, summary, source_seed_policy=policy)
    policy["source_sha256"] = manifest["source_sha256"]
    policy["allowed_source_bindings"] *= 2
    with pytest.raises(ValueError, match="duplicate"):
        source_sequence_coverage_audit(manifest, summary, source_seed_policy=policy)


@pytest.mark.parametrize("change,reason",[
    (lambda rows: rows[1]["source_binding"].update(source_pts=0),"native_row_outside_exact_ordered_sequence"),
    (lambda rows: rows[2].update(memory_update="continued"),"native_first_frame_has_no_source_memory_reset"),
    (lambda rows: rows[1].update(new_observation=True),"source_observation_advance_differs_from_verified_exposure"),
    (lambda rows: rows[3].update(source_seed_injected=True),"native_row_human_seed_not_excluded"),
    (lambda rows: rows[3].update(model_sha256="other_weights"),"native_row_model_weights_differ"),
])
def test_actual_row_failure_is_retained_not_hidden_by_complete_summary(change,reason):
    manifest,summary,rows,groups=fixture();change(rows)
    result=source_sequence_coverage_audit(manifest,summary,observations=rows,exposure_groups=groups)
    assert not result["complete_native_source_sequence_verified"]
    assert result["failure_counts"][reason]>=1


def test_different_timeline_hash_cannot_replace_programme_native_cuts():
    manifest,summary,rows,groups=fixture();summary["classification_sha256"]="other_timeline_overlay"
    result=source_sequence_coverage_audit(manifest,summary,observations=rows,exposure_groups=groups)
    assert result["failure_counts"]["review_primary_source_intervals_SHA_differs"]==1


def test_explicit_source_actor_installation_alias_keeps_original_taxonomy_and_actual_rows():
    manifest,summary,rows,groups=fixture()
    for row in rows[:2]:
        row["source_binding"]["installation_id"]="source_owned_low_rig"
    aliases=dict(source_sha256=manifest["source_sha256"],time_base="2/30720",mappings=[
        dict(shot_id="one",camera_id="low",installation_id="source_owned_low_rig",
             authority_reference=dict(path="source_actor_inventory",sha256="f"*64))])
    original=deepcopy((manifest,summary,rows))
    result=source_sequence_coverage_audit(manifest,summary,observations=rows,
                                        exposure_groups=groups,installation_aliases=aliases)
    assert result["complete_native_source_sequence_verified"]
    assert result["explicit_source_installation_aliases_checked"]
    assert (manifest,summary,rows)==original
    aliases["mappings"][0]["installation_id"]="different_source_rig"
    result=source_sequence_coverage_audit(manifest,summary,observations=rows,
                                        exposure_groups=groups,installation_aliases=aliases)
    assert result["failure_counts"]["native_row_cut_or_camera_binding_differs"]==2


def test_foreign_source_installation_alias_cannot_hide_wrong_camera_binding():
    manifest,summary,rows,groups=fixture()
    aliases=dict(source_sha256="other_source",time_base="1/15360",mappings=[])
    with pytest.raises(ValueError,match="another source"):
        source_sequence_coverage_audit(manifest,summary,observations=rows,
                                     exposure_groups=groups,installation_aliases=aliases)


def test_cross_cut_exposure_proof_and_unclassified_native_gap_fail_explicitly():
    manifest,summary,rows,_=fixture()
    with pytest.raises(ValueError,match="crosses"):
        source_sequence_coverage_audit(manifest,summary,observations=rows,
                                      exposure_groups=[dict(full_pts=[0,256,512]),dict(full_pts=[768])])
    manifest["source_intervals"][1]["start_pts"]=768
    with pytest.raises(ValueError,match="partition"):
        source_sequence_intervals(manifest)


def crosscheck_fixture():
    binding=dict(source_sha256="a"*64,time_base="1/15360",source_pts=256)
    reference=dict(source_binding=binding,group_id="original_ring_outer",feature_id="F3:outer",
        geometry_role="outer_painted_fill_boundary",authority_role="direct_user_confirmed_native_UV",
        source_pointer="immutable_original#/annotations/0",uv=[[10.,20.],[12.,20.]],
        normal=[[0.,1.],[0.,1.]],normal_interval=[[-.5,.5],[-.5,.5]])
    measured=dict(source_binding=binding,feature_id="F3:outer",geometry_role="outer_painted_fill_boundary",
        uv=[[10.,20.25],[12.,21.]],actual_RGB_measurement=True,model_or_world_projection_used_as_UV=False)
    return binding,reference,measured


def test_legacy_crosscheck_preserves_highest_points_and_reports_original_interval_outside_only_post_output():
    binding,reference,measured=crosscheck_fixture();unchanged=deepcopy((binding,reference,measured))
    result=crosscheck_source_observation_groups([reference],[measured],source_binding=binding)
    row=result["comparisons"][0]
    assert row["nearest_Euclidean_native_pixel_distance"]==[.25,1.]
    assert row["original_normal_interval_outside_px"]==[0.,.5]
    assert row["reference_authority_role"]=="direct_user_confirmed_native_UV"
    assert not row["source_positions_used_to_query_network_or_reader"]
    assert not result["accuracy_accepted"]
    assert (binding,reference,measured)==unchanged


def test_nearby_other_identity_or_other_parallel_physical_role_cannot_satisfy_reference():
    binding,reference,measured=crosscheck_fixture()
    wrong_role={**measured,"geometry_role":"inner_painted_fill_boundary","uv":reference["uv"]}
    unknown={**measured,"feature_id":None,"uv":reference["uv"]}
    result=crosscheck_source_observation_groups([reference],[wrong_role,unknown],source_binding=binding)
    assert not result["comparisons"][0]["identity_and_physical_role_matched"]
    assert result["comparisons"][0]["status"].startswith("missing")
    assert result["unresolved_measured_groups"][0]["source_point_count"]==2


@pytest.mark.parametrize("change",[
    lambda group: group["source_binding"].update(source_sha256="b"*64),
    lambda group: group["source_binding"].update(time_base="1/12800"),
    lambda group: group.update(model_or_world_projection_used_as_UV=True),
])
def test_different_source_same_PTS_and_projected_UV_fail_without_implicit_aliases(change):
    binding,reference,measured=crosscheck_fixture();measured=deepcopy(measured);change(measured)
    with pytest.raises(ValueError):
        crosscheck_source_observation_groups([reference],[measured],source_binding=binding)
