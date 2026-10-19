"""Compare saved complete runs without recomputing either solver baseline."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from calibration.solver_runtime.checkpoint_history import iter_solver_history


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def identity(path):
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def stage_cpu_estimates(log_path, samples):
    """Align stage clocks and 2-second samples; mixed boundaries are estimates."""
    starts, intervals = {}, []
    for line in log_path.read_text(encoding="utf8").splitlines():
        if '"wall_time_ns"' not in line:
            continue
        event = json.loads(line)
        if 'wall_time_ns' not in event:
            continue
        key = (event.get("iteration"), event.get("operation"))
        timestamp = event["wall_time_ns"] / 1e9
        if event.get("stage") == "source_stage_begin":
            starts[key] = timestamp
        elif event.get("stage") == "source_stage_end" and key in starts:
            intervals.append((key, starts.pop(key), timestamp))
    totals = {}
    for (iteration, operation), begin, end in intervals:
        entry = totals.setdefault(operation, {"stage_seconds": 0., "estimated_cpu_seconds": 0., "calls": 0})
        entry["calls"] += 1
        entry["stage_seconds"] += end - begin
        for sample in samples:
            stop = datetime.fromisoformat(sample["timestamp_utc"].replace("Z", "+00:00")).timestamp()
            duration = sample["interval_seconds"]
            overlap = max(0., min(end, stop) - max(begin, stop - duration))
            if duration > 0.:
                entry["estimated_cpu_seconds"] += sample["cpu_seconds_delta"] * overlap / duration
    for entry in totals.values():
        entry["estimated_average_busy_cores"] = entry["estimated_cpu_seconds"] / entry["stage_seconds"]
    return {"clocked_stage_intervals": len(intervals), "stage_totals": totals,
            "boundary_overlap_uses_proportional_interpolation": True,
            "short_stages_are_not_precise_CPU_measurements": True}


def saved_run(path):
    performance_path = path / "actual_accepted_cycle_performance.json"
    data = read(performance_path)
    iteration_path = path / "solver" / "iterations.json"
    accepted = [item for item in iter_solver_history(iteration_path) if item.get("accepted")]
    cycles = data["cycles"]
    if len(accepted) < len(cycles):
        raise ValueError("Performance precedes the accepted iteration snapshot; retry after the atomic update")
    result = {"directory": str(path.resolve()), "performance_evidence": identity(performance_path),
              "iterations_evidence": identity(iteration_path), "cycles": [],
              "accepted_steps_awaiting_complete_review_record": len(accepted) - len(cycles)}
    for index, (cycle, iteration) in enumerate(zip(cycles, accepted)):
        result["cycles"].append({
            "index": index, "q_sha256": cycle["q_sha256"],
            "accepted_trial": iteration.get("trial"),
            "damping": iteration.get("damping"),
            "source_cost_before": iteration.get("cost"),
            "acceptance": iteration.get("explicit_source_trial_acceptance"),
            "solver_elapsed_cumulative_seconds": cycle["solver_elapsed_seconds"],
            "first_solver_seconds": cycle["solver_elapsed_seconds"] if index == 0 else None,
            "first_cold_to_accept_seconds": cycle["elapsed_from_run_start_seconds"] if index == 0 else None,
            "accepted_to_accepted_wall_seconds": cycle["accepted_to_accepted_wall_seconds"],
            "stages": cycle["source_stage_timing_seconds"],
        })
    cpu_path = Path(str(path) + ".bootstrap") / "process_tree_utilization.jsonl"
    if cpu_path.exists():
        cpu = [json.loads(line) for line in cpu_path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        elapsed = sum(item["interval_seconds"] for item in cpu)
        work = sum(item["cpu_seconds_delta"] for item in cpu)
        result["cpu"] = {"evidence": identity(cpu_path), "sample_count": len(cpu),
                         "sampled_wall_seconds": elapsed, "process_tree_cpu_seconds": work,
                         "average_busy_cores": work / elapsed if elapsed else None,
                         "maximum_sample_busy_cores": max((item["effective_busy_cores"] for item in cpu), default=None),
                         "summed_RSS_includes_shared_pages": True,
                         "maximum_summed_RSS_bytes": max((item["summed_working_set_bytes"] for item in cpu), default=None),
                         "includes_cold_preparation_and_incomplete_cycles": True}
        identity_guarded = all('creation_time_utc' in process for item in cpu for process in item['processes'])
        result["cpu"]["PID_creation_identity_recorded"] = identity_guarded
        if not identity_guarded:
            result["cpu"]["measurement_limit"] = (
                "Legacy monitor tracked descendant PIDs without creation identity; recycled PIDs can contaminate CPU totals. "
                "These CPU estimates are provisional and are not evidence of sustained solver-only CPU utilization. "
                "Solver stage and accepted-cycle wall times are independently recorded and unaffected.")
        log_path = cpu_path.parent / "stdout.log"
        if log_path.exists():
            result["cpu"]["aligned_stages"] = stage_cpu_estimates(log_path, cpu)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline, candidate = saved_run(args.baseline), saved_run(args.candidate)
    comparisons = []
    for before, after in zip(baseline["cycles"], candidate["cycles"]):
        same = before["q_sha256"] == after["q_sha256"]
        first = before["index"] == 0
        key = "first_solver_seconds" if first else "accepted_to_accepted_wall_seconds"
        old, new = before[key], after[key]
        comparisons.append({"index": before["index"], "accepted_state_byte_identity": same,
                            "same_accepted_trial": before["accepted_trial"] == after["accepted_trial"],
                            "timing_scope": key, "baseline_seconds": old, "candidate_seconds": new,
                            "speedup": old / new if same and old is not None and new else None,
                            "first_cold_speedup": before["first_cold_to_accept_seconds"] / after["first_cold_to_accept_seconds"] if first and same else None})
    result = {"schema": "rinkspace.saved-complete-execution-comparison.v1",
              "baseline_recomputed": False, "local_kernel_speedup_is_not_end_to_end": True,
              "accuracy_acceptance_implied": False, "baseline": baseline,
              "candidate": candidate, "comparisons": comparisons}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf8")
    print(json.dumps({"output": str(args.output), "comparisons": comparisons, "candidate_cpu": candidate.get("cpu")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
