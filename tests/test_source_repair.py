from rinkspace.session.source_repair import build_repair_plan


def frame(pts, *, shot="s02", action="reread_material", feature="ice.R3:outer"):
    return {"schema": "rinkspace.source_quality.v1", "source_sha256": "a"*64,
        "shot_id": shot, "native_pts": pts, "evaluation_role": "regression",
        "failures": [{"code": "localization", "feature_id": feature}],
        "repair_actions": [{"action": action, "feature_id": feature,
                            "reason_code": "localization"}]}


def test_repeated_error_is_one_machine_task_without_inventing_gap_failures():
    reports = [frame(575744), frame(576000), frame(576512), frame(575744)]
    result = build_repair_plan(reports, implementation_revision="reader-a")
    assert result["work_item_count"] == 1
    task, = result["work_items"]
    assert task["owner"] == "codex" and task["evidence_count"] == 3
    assert [r["evaluated_exposures"] for r in task["evaluated_ranges"]] == [2, 1]
    assert task["evaluated_ranges"][0]["stop_pts_exclusive"] == 576256
    assert not result["human_review_queue"] and not result["grants_human_approval"]


def test_identical_rerun_requires_actual_change_not_another_review_batch():
    initial = build_repair_plan([frame(575744)], implementation_revision="reader-a")
    repeat = build_repair_plan([frame(575744)], implementation_revision="reader-a", previous_plan=initial)
    changed = build_repair_plan([frame(575744)], implementation_revision="reader-b", previous_plan=initial)
    assert repeat["work_items"][0]["state"] == "change_implementation_or_inputs"
    assert changed["work_items"][0]["state"] == "repair_then_recheck"
    assert not repeat["work_items"][0]["requires_user_annotation"]


def test_independent_shots_and_actions_are_not_one_assumed_cause():
    reports = [frame(575744), frame(1425664, shot="s07"),
               frame(575744, action="reassociate_entity")]
    result = build_repair_plan(reports, implementation_revision="r1")
    assert result["work_item_count"] == 3


def test_empty_or_clean_comparison_does_not_admit_unseen_frames():
    unscored = {"evaluation_role": "unscored", "shot_id": "s09", "native_pts": 1838080}
    result = build_repair_plan([unscored], implementation_revision="r1")
    assert result["state"] == "machine_evidence_needed"
    assert not result["all_target_frames_accepted"] and not result["grants_source_admission"]
    clean = build_repair_plan([{**frame(575744), "failures": [], "repair_actions": []}],
                              implementation_revision="r1")
    assert clean["state"] == "continue_machine_checks"
    assert not clean["grants_accuracy_outside_evaluated_support"]


def test_missing_action_mapping_remains_a_software_repair():
    report = {**frame(575744), "repair_actions": []}
    result = build_repair_plan([report], implementation_revision="r1")
    assert result["work_item_count"] == 1
    assert result["work_items"][0]["action"] == "inspect_reader_contract"
    assert not result["requires_user_annotation"]


def test_partial_dispatch_does_not_hide_a_different_failure():
    report = frame(575744)
    report["failures"].append({"code": "missing_support", "feature_id": "ice.other"})
    result = build_repair_plan([report], implementation_revision="r1")
    assert result["work_item_count"] == 2
    missing = next(job for job in result["work_items"] if job["feature_id"] == "ice.other")
    assert missing["action"] == "inspect_reader_contract"
    assert missing["evidence"][0]["failures"] == [report["failures"][1]]


def test_missing_binding_or_no_reports_cannot_claim_scored_evidence():
    for reports in ([], [{"evaluation_role": "regression", "failures": [], "repair_actions": []}]):
        result = build_repair_plan(reports, implementation_revision="r1")
        assert result["state"] == "machine_evidence_needed"
        assert result["unique_evaluated_exposures"] == 0
        assert not result["grants_source_admission"]
