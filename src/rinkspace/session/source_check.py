"""Execute source-bound comparisons and prepare Codex-owned repair work.

This command consumes existing readings.  It does not claim to have rerun the
vision network or to have repaired the source measurements.  Its output is the
input to the implementation/reread loop, not a queue of user annotation jobs.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from ..path_policy import guard_project_write
from ..vision_storage import hydrate_source_record_arrays
from .source_quality import EvaluationUsage, QualityProtocol, compare_source_frame, load_accepted_reference
from .source_repair import build_repair_plan


def _load(path):
    raw = Path(path).read_bytes()
    return json.loads(raw.decode("utf-8-sig")), hashlib.sha256(raw).hexdigest()


def _path(value, base, roots):
    if isinstance(value, str):
        path = Path(value)
        return path if path.is_absolute() else base / path
    if not isinstance(value, dict) or set(value) - {"root", "path", "sha256"}:
        raise ValueError("input paths are strings or explicit root/path/SHA references")
    root = value["root"]
    if root not in roots:
        raise ValueError(f"undeclared input root: {root}")
    relative = Path(value["path"])
    if relative.is_absolute():
        raise ValueError("root-bound input references require a relative path")
    parent = roots[root].resolve()
    path = (parent / relative).resolve()
    if not path.is_relative_to(parent):
        raise ValueError("input reference escapes its declared root")
    if value.get("sha256") and hashlib.sha256(path.read_bytes()).hexdigest() != value["sha256"]:
        raise ValueError("input file differs from its pinned SHA256")
    return path


def _usage(values):
    values = dict(values)
    values["human_seed_pts"] = tuple(values.get("human_seed_pts", ()))
    return EvaluationUsage(**values)


def _write(path, value):
    path = guard_project_write(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def run_source_check(request_path, output):
    """Compare every requested case; a bad input remains an explicit repair item.

    Required case keys: packet, binding, acceptance, reading. Optional keys:
    shot_id (default s01), arrays_root and usage. Paths can use declared roots.
    No case is silently dropped, and no absent input is counted as a pass.
    """
    request_path = Path(request_path).resolve()
    request, request_sha = _load(request_path)
    if request.get("schema") != "rinkspace.source_check_request.v1":
        raise ValueError("unknown source check request schema")
    cases = request.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("source check requires at least one concrete input case")
    protocol = QualityProtocol(**request.get("protocol", {}))
    default_usage = asdict(_usage(request.get("usage", {})))
    revision = request.get("implementation_revision")
    if not isinstance(revision, str) or not revision:
        raise ValueError("source check requires the current repair implementation revision")
    base = request_path.parent
    roots = {name: (Path(value) if Path(value).is_absolute() else base / value)
             for name, value in request.get("roots", {}).items()}
    output = guard_project_write(output)
    targets = [output / "check.json", output / "repair.json"]
    if any(path.exists() for path in targets):
        raise ValueError("check output already exists; use a new run directory to preserve evidence")
    reports = []
    input_errors = 0
    for ordinal, case in enumerate(cases):
        reading_ref = None
        ref = None
        try:
            if not isinstance(case, dict):
                raise ValueError("each source check case must be an object")
            paths = {key: _path(case[key], base, roots)
                     for key in ("packet", "binding", "acceptance", "reading")}
            reading, reading_sha = _load(paths["reading"])
            if (not isinstance(reading, dict) or not isinstance(reading.get("source_binding"), dict)
                    or not isinstance(reading.get("guided"), dict)
                    or not isinstance(reading["guided"].get("readings"), list)
                    or any(not isinstance(row, dict) for row in reading["guided"]["readings"])):
                raise ValueError("saved reading must contain source_binding and guided/readings objects")
            reading_ref = {"path": str(paths["reading"]), "sha256": reading_sha,
                           "producer": "saved source reading; not executed by this check"}
            ref = load_accepted_reference(paths["packet"], paths["binding"], paths["acceptance"],
                                          shot_id=case.get("shot_id", "s01"))
            arrays = _path(case["arrays_root"], base, roots) if "arrays_root" in case else paths["reading"].parent
            reading = hydrate_source_record_arrays(reading, arrays)
            usage = _usage({**default_usage, **case.get("usage", {})})
            report = compare_source_frame(ref, reading, protocol=protocol, usage=usage)
            report["reading_ref"] = reading_ref
            report["case_index"] = ordinal
        except (OSError, ValueError, KeyError, TypeError, AttributeError, IndexError) as error:
            input_errors += 1
            contract_error = isinstance(error, (AttributeError, IndexError))
            code = "input_or_reader_contract_error" if contract_error else "input_unavailable_or_invalid"
            report = {"schema": "rinkspace.source_check_input_failure.v1",
                "case_index": ordinal, "input_ref": {"request_sha256": request_sha, "case_index": ordinal},
                "source_sha256": ref.source_sha256 if ref is not None else None,
                "shot_id": ref.shot_id if ref is not None else case.get("shot_id", "s01") if isinstance(case, dict) else None,
                "native_pts": ref.native_pts if ref is not None else None,
                "evaluation_role": "unscored", "reading_ref": reading_ref,
                "failures": [{"code": code, "feature_id": None,
                              "details": {"error_type": type(error).__name__, "error": str(error)}}],
                "repair_actions": [{"action": "inspect_reader_contract" if contract_error else "repair_input_binding", "feature_id": None,
                                    "reason_code": code}],
                "accuracy_accepted": False}
        reports.append(report)
    previous = None
    if request.get("previous_plan"):
        previous, _ = _load(_path(request["previous_plan"], base, roots))
    repair = build_repair_plan(reports, implementation_revision=revision,
                              previous_plan=previous, pts_step=request.get("native_pts_step", 256))
    check = {"schema": "rinkspace.source_check_run.v1", "request_sha256": request_sha,
        "implementation_revision": revision, "case_count": len(cases), "input_error_count": input_errors,
        "state": repair["state"], "reports": reports,
        "vision_execution": "existing saved outputs only; no reader or GPU run claimed",
        "requires_user_annotation": False, "all_target_frames_accepted": False}
    _write(targets[0], check)
    _write(targets[1], repair)
    return {"schema": "rinkspace.source_check_result.v1", "check_path": str(targets[0]),
        "repair_path": str(targets[1]), "case_count": len(cases), "input_error_count": input_errors,
        "state": repair["state"], "work_item_count": repair["work_item_count"],
        "unique_evaluated_exposures": repair["unique_evaluated_exposures"],
        "human_review_queue": [], "all_target_frames_accepted": False,
        "exit_code": 2 if input_errors else 0,
        "exit_code_meaning": "execution/input status; zero does not mean visual acceptance"}
