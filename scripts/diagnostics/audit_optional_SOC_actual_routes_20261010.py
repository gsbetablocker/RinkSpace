"""Read saved QP stage/acceptance evidence without loading source graphs."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from calibration.solver_runtime.checkpoint_history import iter_solver_history


def read_json(path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8-sig"))


def file_evidence(path):
    if not path.exists():
        return None
    data = path.read_bytes()
    stat = path.stat()
    return dict(path=str(path), bytes=len(data), sha256=hashlib.sha256(data).hexdigest(),
                mtime_ns=stat.st_mtime_ns)


def kkt_summary(report):
    keys = ("complete_constraint_rows_retained",
            "complete_original_nonnegative_IPM_dual_certificate_checked",
            "original_H_gradient_constraints_bounds_and_KKT_gate_retained",
            "free_stationarity_max_absolute", "kkt_tolerance",
            "original_unit_complementarity_absolute_sum",
            "original_unit_complementarity_tolerance")
    result = {key: report.get(key) for key in keys}
    flags = all(result[key] is True for key in keys[:3])
    try:
        result["recorded_original_full_KKT_passed"] = bool(
            flags and result["free_stationarity_max_absolute"] <= result["kkt_tolerance"]
            and result["original_unit_complementarity_absolute_sum"]
            <= result["original_unit_complementarity_tolerance"])
    except TypeError:
        result["recorded_original_full_KKT_passed"] = False
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bootstrap = Path(str(args.run) + ".bootstrap")
    log_path = bootstrap / "stdout.log"
    receipt_path = bootstrap / "execution_acceleration_receipt.json"
    control_request_path = bootstrap / "actual_QP_stop_request.json"
    history_path = args.run / "solver" / "iterations.json"
    result_path = args.run / "solver" / "solver_result.json"
    records = []
    ignored_non_json = 0
    if log_path.exists():
        for number, raw in enumerate(log_path.read_text(encoding="utf-8-sig").splitlines(), 1):
            try:
                item = json.loads(raw)
            except json.JSONDecodeError:
                ignored_non_json += 1
                continue
            if isinstance(item, dict):
                item = dict(item, log_line=number)
                records.append(item)
    calls = {}
    sessions = []
    current = None
    failure_records = []
    issues = []
    initial_full_fallback = []
    for record in records:
        stage = record.get("stage")
        if stage == "optional_SOC_QP_begin":
            if current is not None:
                issues.append(dict(reason="overlapping_optional_SOC_log_scope", log_line=record["log_line"]))
            current = dict(begin=record, backend_calls=[], failure=None, end=None)
            sessions.append(current)
        elif stage == "QP_backend_begin":
            call = dict(begin=record, end=None)
            calls[record["QP_call"]] = call
            if current is not None:
                current["backend_calls"].append(record["QP_call"])
            if record.get("QP_phase") in ("full_diagonal", "full_affine"):
                if record.get("optional_measured_SOC") is True:
                    issues.append(dict(reason="full_retry_in_verified_optional_SOC", call=record["QP_call"]))
                else:
                    initial_full_fallback.append(record)
        elif stage == "QP_backend_end":
            call = calls.get(record["QP_call"])
            if call is None:
                issues.append(dict(reason="backend_end_without_begin", call=record["QP_call"]))
            else:
                call["end"] = record
        elif stage == "optional_SOC_restricted_numerical_failure":
            failure_records.append(record)
            if current is None:
                issues.append(dict(reason="optional_failure_without_SOC_scope", log_line=record["log_line"]))
                continue
            current["failure"] = record
            checks = {
                "eligible_SOC_scope": current["begin"].get("prior_full_KKT_direction_available") is True,
                "previous_full_KKT_direction_verified": record.get("previous_direction_full_original_KKT_verified") is True,
                "restricted_failed_original_gate": record.get("original_restricted_failure_report", {}).get("accepted") is False,
                "failure_reason_exactly_retained": record.get("restricted_failure_reason")
                == record.get("original_restricted_failure_report", {}).get("reason", "unreported_restricted_failure"),
                "not_claimed_mathematical_infeasibility": record.get("mathematical_infeasibility_certified") is False,
                "strictly_tighter_measured_faces": (record.get("strictly_tighter_measured_SOC_faces") or 0) > 0,
                "previous_direction_retained": record.get("previous_verified_direction_retained_for_original_backtracking") is True,
                "original_final_guards_retained": record.get("original_line_search_physical_guard_cost_and_acceptance_retained") is True,
            }
            current["failure_scope_checks"] = checks
            for key, okay in checks.items():
                if not okay:
                    issues.append(dict(reason=key, log_line=record["log_line"]))
        elif stage == "optional_SOC_QP_end":
            if current is None:
                issues.append(dict(reason="SOC_end_without_begin", log_line=record["log_line"]))
            else:
                current["end"] = record
                current = None
    history = iter_solver_history(history_path) if history_path.exists() else ()
    iteration_records = []
    logged_failures = {record.get("wall_time_ns"): record for record in failure_records}
    for entry in history:
        linear = entry.get("global_schur_linear_report", {})
        soc = linear.get("sequential_measured_shared_source_curvature_tightening", {})
        failure = soc.get("optional_SOC_restricted_numerical_failure")
        record = {key: entry.get(key) for key in ("iteration", "trial", "accepted", "cost", "next_cost", "failure_reason")}
        record["final_direction_KKT"] = kkt_summary(linear.get("shared_linear_domain_step", {}))
        record["SOC"] = {key: soc.get(key) for key in ("correction_QP_attempts", "solved_corrections", "stop_reason", "mathematical_infeasibility_certified")}
        record["SOC"]["steps"] = [
            {key: step.get(key) for key in ("attempt", "accepted_numerical_direction", "failure", "last_solved_direction_retained_for_original_backtracking")}
            for step in soc.get("steps", [])]
        record["explicit_source_trial_acceptance"] = {
            key: entry.get("explicit_source_trial_acceptance", {}).get(key)
            for key in ("accept", "mode", "rejection_reason")}
        if failure:
            record["optional_failure"] = failure
            record["optional_failure_exactly_matches_stdout"] = (
                {key: value for key, value in logged_failures.get(failure.get("wall_time_ns"), {}).items()
                 if key not in ("stage", "log_line")} == failure)
            if not record["optional_failure_exactly_matches_stdout"]:
                issues.append(dict(reason="saved_SOC_failure_stdout_mismatch", iteration=entry.get("iteration")))
        if entry.get("accepted") is True and not record["final_direction_KKT"]["recorded_original_full_KKT_passed"]:
            issues.append(dict(reason="accepted_entry_missing_recorded_full_KKT_pass", iteration=entry.get("iteration")))
        iteration_records.append(record)
    receipt = read_json(receipt_path) or {}
    qp_receipt = receipt.get("qp", {})
    snapshot = bootstrap / "execution_sources" / "optional_soc_runtime.py"
    snapshot_evidence = file_evidence(snapshot)
    installed_source_matches_snapshot = bool(snapshot_evidence and qp_receipt.get("helper_sha256") == snapshot_evidence["sha256"])
    if qp_receipt and not installed_source_matches_snapshot:
        issues.append(dict(reason="installed_QP_source_snapshot_mismatch"))
    phase_counts = Counter(call["begin"]["QP_phase"] for call in calls.values())
    solver_result = read_json(result_path)
    control_request = read_json(control_request_path)
    control_events = [record for record in records if record.get("stage") in (
        "source_control_observed_in_QP", "source_control_QP_returned_to_original_solver")]
    control_timing = None
    if control_request is not None:
        after_write = control_request["request_after_write_wall_time_ns"]
        control_timing = dict(request_after_write_wall_time_ns=after_write)
        for stage, name in (("source_control_observed_in_QP", "observed_in_IPM"),
                            ("source_control_QP_returned_to_original_solver", "returned_to_original_loop")):
            events = [event for event in control_events if event["stage"] == stage]
            if events:
                control_timing[name + "_wall_time_ns"] = events[0]["wall_time_ns"]
                control_timing[name + "_seconds_after_request"] = (events[0]["wall_time_ns"] - after_write) / 1e9
                if events[0]["wall_time_ns"] < control_request["request_before_write_wall_time_ns"]:
                    issues.append(dict(reason="control_event_predates_actual_request", stage=stage))
    result = dict(
        schema="rinkspace.actual-optional-SOC-restricted-failure-audit.v1",
        observed_utc=datetime.now(timezone.utc).isoformat(), run=str(args.run),
        read_only_runtime_observation=True, source_graph_loaded=False,
        solver_recomputed=False, control_file_written=False,
        source_evidence=dict(stdout=file_evidence(log_path), iterations=file_evidence(history_path),
                             installation_receipt=file_evidence(receipt_path), optional_SOC_snapshot=snapshot_evidence,
                             solver_result=file_evidence(result_path)),
        installed_QP_source_matches_snapshot=installed_source_matches_snapshot,
        QP_installation=qp_receipt,
        latest_logged_stage=(records[-1].get("stage") if records else None),
        ignored_non_JSON_lines=ignored_non_json,
        backend_call_count=len(calls), backend_phase_counts=dict(phase_counts),
        pending_backend_calls=[number for number, call in calls.items() if call["end"] is None],
        backend_calls=list(calls.values()), optional_SOC_sessions=sessions,
        explicit_control_cancelled_backend_calls=[
            call for call in calls.values()
            if (call.get("end") or {}).get("error_type") == "SourceControlStop"],
        actual_explicit_control_request=control_request,
        actual_explicit_control_events=control_events,
        actual_explicit_control_timing=control_timing,
        optional_restricted_failure_count=len(failure_records),
        initial_or_unverified_full_fallback_calls=initial_full_fallback,
        recorded_iteration_count=len(iteration_records),
        recorded_accepted_count=sum(entry.get("accepted") is True for entry in iteration_records),
        iterations=iteration_records, issues=issues,
        original_solver_result=solver_result,
        observed_completed_evidence_checks_passed=not issues,
        evidence_boundary="Runtime reports and paired stage logs; no independent re-solve of actual matrices or future direction bitexact claim.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({key: result[key] for key in (
        "latest_logged_stage", "backend_call_count", "backend_phase_counts", "pending_backend_calls",
        "optional_restricted_failure_count", "recorded_iteration_count", "recorded_accepted_count",
        "initial_or_unverified_full_fallback_calls", "issues")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
