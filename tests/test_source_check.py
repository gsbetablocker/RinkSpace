"""Exercise the public CLI and its partial-input failure behavior."""
import json
from pathlib import Path

import pytest

from rinkspace.cli import main
from rinkspace.session.source_check import run_source_check
from rinkspace.session.source_quality import load_accepted_reference


def request_with_existing_reference(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    base = repo / "events/ankara_2026/programmes/okada_mei_sp/source_datasets/frame_by_frame_review_20261013/first_shot_remaining_50fps/repair_20261014"
    paths = {"packet": str(base / "frames/000210.json"),
             "binding": str(base / "bindings/000210.json"),
             "acceptance": str(base / "user_acceptance.json")}
    ref = load_accepted_reference(paths["packet"], paths["binding"], paths["acceptance"])
    # An empty reader result is deliberately not a visual success.
    reading = {"source_binding": {"source_sha256": ref.source_sha256,
        "source_pts": ref.native_pts, "shot_id": ref.shot_id,
        "time_base": ref.time_base, "native_hw": list(ref.native_hw)},
        "guided": {"readings": []}}
    (tmp_path / "reading.json").write_text(json.dumps(reading))
    request = {"schema": "rinkspace.source_check_request.v1",
        "implementation_revision": "test-implementation", "cases": [
            {**paths, "reading": "reading.json"}]}
    path = tmp_path / "request.json"
    path.write_text(json.dumps(request))
    return path, request


def test_public_command_detects_real_reference_missing_support_without_human_jobs(tmp_path, capsys):
    path, _ = request_with_existing_reference(tmp_path)
    out = tmp_path / "out"
    assert main(["observe", "check-source", "--request", str(path), "--out", str(out)]) == 0
    receipt = json.loads(capsys.readouterr().out)
    plan = json.loads((out / "repair.json").read_text())
    assert receipt["unique_evaluated_exposures"] == 1
    assert receipt["work_item_count"] == 14
    assert receipt["state"] == "machine_repair_required"
    assert plan["human_review_queue"] == []
    assert all(job["action"] == "discover_material" and job["owner"] == "codex" for job in plan["work_items"])
    assert not plan["grants_human_approval"]


def test_missing_input_does_not_hide_other_cases_or_create_false_pass(tmp_path):
    path, request = request_with_existing_reference(tmp_path)
    request["cases"].append({**request["cases"][0], "reading": "absent.json"})
    path.write_text(json.dumps(request))
    receipt = run_source_check(path, tmp_path / "out")
    check = json.loads(Path(receipt["check_path"]).read_text())
    plan = json.loads(Path(receipt["repair_path"]).read_text())
    assert receipt["case_count"] == 2 and receipt["input_error_count"] == 1
    assert receipt["exit_code"] == 2
    assert [report["evaluation_role"] for report in check["reports"]] == ["regression", "unscored"]
    assert any(job["action"] == "repair_input_binding" for job in plan["work_items"])
    assert not plan["all_target_frames_accepted"]


def test_empty_request_or_overwrite_cannot_erase_check_history(tmp_path):
    path, request = request_with_existing_reference(tmp_path)
    run_source_check(path, tmp_path / "out")
    with pytest.raises(ValueError, match="already exists"):
        run_source_check(path, tmp_path / "out")
    request["cases"] = []
    path.write_text(json.dumps(request))
    with pytest.raises(ValueError, match="at least one"):
        run_source_check(path, tmp_path / "empty")


def test_malformed_second_reading_does_not_discard_first_comparison(tmp_path):
    path, request = request_with_existing_reference(tmp_path)
    (tmp_path / "bad.json").write_text(json.dumps({"source_binding": []}))
    request["cases"].append({**request["cases"][0], "reading": "bad.json"})
    path.write_text(json.dumps(request))
    result = run_source_check(path, tmp_path / "out")
    check = json.loads(Path(result["check_path"]).read_text())
    assert result["input_error_count"] == 1
    assert len(check["reports"]) == 2
    assert check["reports"][0]["evaluation_role"] == "regression"
    assert check["reports"][1]["repair_actions"][0]["action"] == "repair_input_binding"
