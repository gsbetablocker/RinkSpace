"""Turn source-quality failures into bounded software repair work, never human labels.

This is a scheduler input, not a visual judge or an automatic source-admission
rule. It groups repeated failures while preserving the exact evaluated exposure
set. It never promotes a measurement, modifies UV, or grants human acceptance.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from typing import Iterable, Mapping


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _ranges(values, step):
    values = sorted(set(values))
    result = []
    for value in values:
        if result and value == result[-1]["last_pts"] + step:
            result[-1]["last_pts"] = value
            result[-1]["stop_pts_exclusive"] = value + step
            result[-1]["evaluated_exposures"] += 1
        else:
            result.append({"first_pts": value, "last_pts": value,
                           "stop_pts_exclusive": value + step, "evaluated_exposures": 1})
    return result


_TARGETS = {
    "resolve_source_support": {
        "code": "session/source_support.py::SourceSupportResolver.resolve",
        "next": "resolve the exact stored source support; retain unresolved input if unavailable",
    },
    "discover_material": {
        "code": "vision.py::SourceRinkReader.read and its unassigned material dispatch",
        "next": "produce a current-RGB query for the missing material, then rerun affected exposures",
    },
    "reassociate_entity": {
        "code": "vision_reidentify.py / vision_instances.py / common association dispatcher",
        "next": "repair instance or branch evidence while preserving measured source coordinates",
    },
    "reread_material": {
        "code": "vision_material_reader.py / current material profile readers",
        "next": "repair the material query and reread original pixels; do not move old UV to the model",
    },
    "repair_evaluation_setup": {
        "code": "session/source_read.py request and reference-dependency separation",
        "next": "keep this as a regression diagnostic; remove same-exposure assistance for a blind run",
    },
    "inspect_model_camera": {
        "code": "current native camera/geometry and held-out source factors",
        "next": "preserve reliable UV; inspect camera/model disagreement without relabelling source evidence",
    },
    "repair_input_binding": {
        "code": "session/source_inputs.py / source_support.py / exact source binding",
        "next": "repair the source path, hash or exposure mapping before consuming this input",
    },
    "inspect_reader_contract": {
        "code": "the producer of the reported source-quality failure",
        "next": "implement the missing machine repair handler; do not turn it into a user annotation task",
    },
}


def build_repair_plan(reports: Iterable[Mapping], *, implementation_revision: str,
                      previous_plan: Mapping | None = None, pts_step: int = 256) -> dict:
    """Group equal shot/entity/action failures; skip identical no-change replay.

    ``implementation_revision`` identifies the current repair implementation,
    not the producer of an old reading. Preserving that distinction allows old
    readings to expose a current regression without forging execution history.
    A clean comparison grants no accuracy claim outside its measured support.
    """
    if not isinstance(implementation_revision, str) or not implementation_revision:
        raise ValueError("an explicit current repair implementation revision is required")
    if type(pts_step) is not int or pts_step <= 0:
        raise ValueError("native PTS step must be a positive integer")
    previous = {}
    if previous_plan is not None:
        if previous_plan.get("schema") != "rinkspace.source_repair_plan.v1":
            raise ValueError("unknown previous repair plan")
        previous = {item["id"]: item for item in previous_plan["work_items"]}

    grouped = {}
    roles = defaultdict(int)
    evaluated = set()
    missing_reference = []
    report_count = 0
    for ordinal, report in enumerate(reports):
        report_count += 1
        if not isinstance(report, Mapping):
            raise ValueError("quality report must be a mapping")
        shot, pts = report.get("shot_id"), report.get("native_pts")
        source = report.get("source_sha256")
        if pts is not None and type(pts) is not int:
            raise ValueError("repair evidence uses a non-integer PTS")
        role = report.get("evaluation_role", "unscored")
        if role not in {"regression", "independent", "unscored"}:
            raise ValueError("unknown source evaluation role")
        source_bound = (isinstance(source, str) and len(source) == 64
                        and all(c in "0123456789abcdef" for c in source.lower())
                        and isinstance(shot, str) and bool(shot)
                        and type(pts) is int and pts >= 0)
        if role != "unscored" and not source_bound:
            role = "unscored"
            missing_reference.append({"report_index": ordinal, "shot_id": shot, "native_pts": pts,
                "reason": "claimed evaluation has no complete source/shot/native-PTS identity"})
        roles[str(role)] += 1
        if source_bound and role != "unscored":
            evaluated.add((source, shot, pts))
        actions = list(report.get("repair_actions", ()))
        failures = list(report.get("failures", ()))
        if any(not isinstance(item, Mapping) for item in actions + failures):
            raise ValueError("failure and repair entries must be mappings")
        # A partially implemented dispatcher cannot hide its unhandled errors.
        covered = {(a.get("feature_id"), a.get("reason_code", a.get("code", "unspecified")))
                   for a in actions}
        for failure in failures:
            identity, reason = failure.get("feature_id"), failure.get("code", "unmapped_failure")
            if (identity, reason) not in covered:
                actions.append({"action": "inspect_reader_contract", "feature_id": identity,
                                "reason_code": reason, "details": dict(failure)})
                covered.add((identity, reason))
        if role == "unscored" and not actions and not any(r["report_index"] == ordinal for r in missing_reference):
            missing_reference.append({"report_index": ordinal, "shot_id": shot, "native_pts": pts,
                                      "reason": "no scored source evidence; no pass inferred"})
        for action in actions:
            if not isinstance(action, Mapping):
                raise ValueError("repair action must be a mapping")
            name = action.get("action", "inspect_reader_contract")
            if not isinstance(name, str) or not name:
                raise ValueError("repair action needs a name")
            sid = action.get("shot_id", shot)
            if shot is not None and sid != shot:
                raise ValueError("repair action cannot move evidence to another shot")
            identity = action.get("feature_id")
            reason = action.get("reason_code", action.get("code", "unspecified"))
            # Unknown source failures cannot merge unrelated file inputs.
            source_key = source or report.get("input_ref") or f"unbound-report-{ordinal}"
            key = (source_key, sid, identity, name, str(reason))
            key_id = _hash(key)[:24]
            group = grouped.setdefault(key_id, {"id": key_id, "source_sha256": source,
                "shot_id": sid, "feature_id": identity, "action": name,
                "reason_code": str(reason), "input_ref": report.get("input_ref"),
                "evidence": [], "pts": set(), "roles": set()})
            if pts is not None:
                group["pts"].add(pts)
            group["roles"].add(str(role))
            related_failures = [failure for failure in failures
                                if failure.get("feature_id") == identity
                                and failure.get("code", "unmapped_failure") == reason]
            group["evidence"].append({"native_pts": pts, "action": dict(action),
                                      "failures": related_failures,
                                      "reference": report.get("reference"),
                                      "reading_ref": report.get("reading_ref")})

    work = []
    for key_id, group in sorted(grouped.items()):
        evidence = sorted(group.pop("evidence"), key=lambda item: json.dumps(item, sort_keys=True, ensure_ascii=False))
        # Repeated imports of the same report do not multiply the repair work.
        unique = {_hash(item): item for item in evidence}
        fingerprint = _hash(sorted(unique))
        old = previous.get(key_id, {})
        unchanged = (old.get("evidence_revision") == fingerprint
                     and old.get("implementation_revision") == implementation_revision)
        target = _TARGETS.get(group["action"], _TARGETS["inspect_reader_contract"])
        points = group.pop("pts")
        role_names = sorted(group.pop("roles"))
        work.append({**group, "owner": "codex", "requires_user_annotation": False,
            "requires_user_approval_to_repair": False,
            "evaluated_ranges": _ranges(points, pts_step),
            "range_semantics": "only these evaluated exposures are known to fail; gaps are not fabricated",
            "evidence_roles": role_names, "evidence_revision": fingerprint,
            "implementation_revision": implementation_revision,
            "evidence_count": len(unique), "evidence": list(unique.values()),
            "code_target": target["code"], "next_action": target["next"],
            "state": "change_implementation_or_inputs" if unchanged else "repair_then_recheck",
            "identical_replay_is_progress": False,
            "preserve_source_uv_ids_and_human_scopes": True})

    if not report_count:
        missing_reference.append({"report_index": None, "shot_id": None, "native_pts": None,
                                  "reason": "no source evaluation was supplied"})
    return {"schema": "rinkspace.source_repair_plan.v1",
        "implementation_revision": implementation_revision,
        "state": "machine_repair_required" if work else "machine_evidence_needed" if missing_reference else "continue_machine_checks",
        "work_items": work, "work_item_count": len(work),
        "unique_evaluated_exposures": len(evaluated), "evaluation_roles": dict(roles),
        "unscored_inputs": missing_reference,
        "requires_user_annotation": False, "human_review_queue": [],
        "grants_source_admission": False, "grants_human_approval": False,
        "grants_accuracy_outside_evaluated_support": False,
        "all_target_frames_accepted": False,
        "continuation_rule": "Codex fixes and rechecks machine work items; the user is not a per-frame or per-batch dependency"}
