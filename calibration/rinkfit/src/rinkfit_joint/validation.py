"""Source-clock schedules and honest validation ledgers for the nine-shot fit.

No detector, tracker, or renderer is implemented here. This module prepares
their exact PTS jobs and reviews their outputs without mistaking projected
curves, propagated pixels, or historical fitting data for independent truth.
All thresholds are explicit reporting thresholds, not inferred accuracy claims.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Mapping


def _row_id(row: Mapping[str, Any]) -> str:
    return str(row.get("row_id", row.get("source_row_id", "")))


def build_frame_schedule(
    shots: Iterable[Mapping[str, Any]],
    time_base: str = "1/12800",
    step_pts: int = 256,
) -> list[dict[str, Any]]:
    """Enumerate every included native PTS, without filling cut/excluded gaps.

    Each shot supplies id, rig_id, first_pts, last_pts (both inclusive), and
    optionally existing_pts, excluded_pts and exclusion_reason. Programme
    inclusion ranges must already be adjudicated in the source manifest.
    The output is a task ledger, not evidence that any frame was processed.
    """
    base = Fraction(time_base)
    if base <= 0 or step_pts <= 0:
        raise ValueError("Positive source time base and PTS step are required")
    result: list[dict[str, Any]] = []
    all_pts: set[int] = set()
    seen_shots: set[str] = set()
    for shot in shots:
        sid = str(shot["id"])
        if sid in seen_shots:
            raise ValueError(f"Duplicate shot id: {sid}")
        seen_shots.add(sid)
        first, last = int(shot["first_pts"]), int(shot["last_pts"])
        if last < first or (last - first) % step_pts:
            raise ValueError(f"Shot endpoints do not define a native PTS lattice: {sid}")
        existing = {int(p) for p in shot.get("existing_pts", ())}
        excluded = {int(p) for p in shot.get("excluded_pts", ())}
        domain = set(range(first, last + 1, step_pts))
        if not existing.issubset(domain) or not excluded.issubset(domain):
            raise ValueError(f"Known/excluded PTS outside shot inclusion interval: {sid}")
        if all_pts.intersection(domain):
            raise ValueError(f"Overlapping native frames across shots: {sid}")
        all_pts.update(domain)
        for pts in sorted(domain):
            relative = Fraction(pts - first) * base
            is_excluded = pts in excluded
            result.append({
                "shot_id": sid, "rig_id": str(shot["rig_id"]), "native_pts": pts,
                "time_base": str(base), "source_seconds": float(Fraction(pts) * base),
                "shot_seconds": float(relative), "shot_frame_1based": (pts - first) // step_pts + 1,
                "existing_sparse_evidence": pts in existing,
                "processing_status": "excluded_by_manifest" if is_excluded else "pending_native_evidence",
                "exclusion_reason": str(shot.get("exclusion_reason", "explicit_source_manifest")) if is_excluded else "",
                "task": "none" if is_excluded else "decode_mask_and_measure_static_source",
                "boundary_context": "shot_start" if pts == first else "shot_end" if pts == last else "interior",
                "cross_cut_tracking_allowed": False,
            })
    return result


class _UnionFind:
    def __init__(self, count: int) -> None:
        self.parent = list(range(count))

    def find(self, key: int) -> int:
        while self.parent[key] != key:
            self.parent[key] = self.parent[self.parent[key]]
            key = self.parent[key]
        return key

    def union(self, left: int, right: int) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def assign_locked_roles(
    rows: Iterable[Mapping[str, Any]],
    *,
    seed: str,
    validation_fraction: float = 0.2,
    block_ticks: int = 6400,
    guard_ticks: int = 2560,
) -> dict[str, Any]:
    """Assign fixed roles to whole source/time families before optimization.

    Inputs need row_id/source_row_id, shot_id, native_pts,
    provenance_group_id, previously_used_for_fit (bool), used_for_template
    (bool), and provenance_complete (bool). Optional lineage_group_ids link
    representatives, shared source fits, or propagated tracks. These identifiers
    denote the *measurement generator*, not the global physical material id.

    A whole time block and all linked generators receive one role. Any prior
    fitting/template use contaminates the whole family for independent testing.
    Missing lineage metadata is never silently promoted to sealed validation.
    A temporal guard is withheld from fitting near selected validation windows.
    This deterministic split does not by itself make correlated pixels or
    adjacent frames statistically independent.
    """
    if not 0 <= validation_fraction < 1 or block_ticks <= 0 or guard_ticks < 0 or not seed:
        raise ValueError("Invalid deterministic split configuration")
    output = [dict(row) for row in rows]
    ids = [_row_id(row) for row in output]
    if any(not key for key in ids) or len(ids) != len(set(ids)):
        raise ValueError("Stable, unique source row identities are required")
    uf = _UnionFind(len(output))
    key_first: dict[tuple[str, str], int] = {}
    for index, row in enumerate(output):
        sid, pts = str(row["shot_id"]), int(row["native_pts"])
        keys = [("time_block", f"{sid}/{pts // block_ticks}")]
        group = row.get("provenance_group_id")
        if group:
            keys.append(("provenance", str(group)))
        keys.extend(("lineage", str(key)) for key in row.get("lineage_group_ids", ()))
        for key in keys:
            if key in key_first:
                uf.union(index, key_first[key])
            else:
                key_first[key] = index
    components: dict[int, list[int]] = defaultdict(list)
    for index in range(len(output)):
        components[uf.find(index)].append(index)
    component_roles: dict[int, str] = {}
    component_ids: dict[int, str] = {}
    for root, indices in components.items():
        members = [output[i] for i in indices]
        key = min(ids[i] for i in indices)
        component_ids[root] = hashlib.sha256((seed + "\0" + key).encode()).hexdigest()
        complete = all(
            m.get("provenance_complete") is True
            and isinstance(m.get("previously_used_for_fit"), bool)
            and isinstance(m.get("used_for_template"), bool)
            and bool(m.get("provenance_group_id"))
            for m in members
        )
        if any(m.get("previously_used_for_fit") is True or m.get("used_for_template") is True for m in members):
            role = "historical_fit"
        elif not complete:
            role = "validation_eligibility_unknown"
        else:
            draw = int(component_ids[root][:16], 16) / float(2**64)
            role = "sealed_validation" if draw < validation_fraction else "train"
        component_roles[root] = role
    validation_intervals: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for root, indices in components.items():
        if component_roles[root] != "sealed_validation":
            continue
        shot_blocks = {(str(output[i]["shot_id"]), int(output[i]["native_pts"]) // block_ticks) for i in indices}
        for sid, block in shot_blocks:
            validation_intervals[sid].append((block * block_ticks - guard_ticks, (block + 1) * block_ticks - 1 + guard_ticks))
    for root, indices in components.items():
        if component_roles[root] != "train":
            continue
        touches_guard = any(
            any(lo <= int(output[i]["native_pts"]) <= hi for lo, hi in validation_intervals[str(output[i]["shot_id"])])
            for i in indices
        )
        if touches_guard:
            component_roles[root] = "validation_guard"
    for root, indices in components.items():
        for index in indices:
            output[index]["fit_role"] = component_roles[root]
            output[index]["validation_family_id"] = component_ids[root]
    by_shot: dict[str, Counter[str]] = defaultdict(Counter)
    for row in output:
        by_shot[str(row["shot_id"])][str(row["fit_role"])] += 1
    serialization = json.dumps(
        sorted((ids[i], output[i]["fit_role"], output[i]["validation_family_id"]) for i in range(len(output))),
        separators=(",", ":"), ensure_ascii=False,
    ).encode()
    return {
        "rows": output,
        "manifest": {
            "schema": "rinkfit.locked_validation_roles.v1", "seed": seed,
            "validation_fraction_requested": validation_fraction,
            "block_ticks": block_ticks, "guard_ticks": guard_ticks,
            "assignment_sha256": hashlib.sha256(serialization).hexdigest(),
            "role_counts": dict(Counter(row["fit_role"] for row in output)),
            "by_shot": {key: dict(value) for key, value in sorted(by_shot.items())},
            "source_families": len(components),
            "statistical_independence_certified": False,
            "sealed_validation_used_for_parameter_selection_allowed": False,
            "note": "After inspecting sealed results to change the model, retain this ledger as development evidence and reserve a fresh untouched test set.",
        },
    }


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    return values[lower] * (upper - position) + values[upper] * (position - lower) if lower != upper else values[lower]


def _summary(rows: list[Mapping[str, Any]], thresholds: tuple[float, ...]) -> dict[str, Any]:
    errors: list[float] = []
    invalid = 0
    pending = 0
    for row in rows:
        value = row.get("coverage_error_px")
        if value is None:
            pending += 1
        elif row.get("projection_valid") is not True or not math.isfinite(float(value)) or float(value) < 0:
            invalid += 1
        else:
            errors.append(float(value))
    return {
        "rows": len(rows), "finite_valid_rows": len(errors), "pending_rows": pending,
        "invalid_projection_or_error_rows": invalid,
        "median_px": _percentile(errors, 0.5), "p95_px": _percentile(errors, 0.95),
        "p99_px": _percentile(errors, 0.99), "maximum_px": max(errors) if errors else None,
        "rms_px": math.sqrt(math.fsum(e * e for e in errors) / len(errors)) if errors else None,
        "over_px": {str(limit): sum(e > limit for e in errors) for limit in thresholds},
        "all_rows_processed_and_valid": len(errors) == len(rows) and bool(rows),
    }


def summarize_residuals(
    rows: Iterable[Mapping[str, Any]],
    *,
    thresholds: tuple[float, ...] = (2.0, 5.0, 10.0, 20.0),
) -> dict[str, Any]:
    """Summarize exact engineering residuals without silently dropping failures.

    Caller provides exact semantic coverage_error_px from the physical model:
    Euclidean for an entity/ordinary curve point, specified normal metric for
    a genuine profile. Full W must be computed separately on native 2D errors.
    Unknown and excluded observations stay in their own admission/role ledger.
    """
    data = [dict(row) for row in rows]
    ids = [_row_id(row) for row in data]
    if any(not key for key in ids) or len(ids) != len(set(ids)):
        raise ValueError("Unique row identities are required for a residual ledger")
    by: dict[str, dict[str, Any]] = {}
    for field in ("shot_id", "rig_id", "material_id", "semantic_kind", "fit_role", "admission_status"):
        groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in data:
            groups[str(row.get(field, "unspecified"))].append(row)
        by[field] = {key: _summary(group, thresholds) for key, group in sorted(groups.items())}
    eligible = [row for row in data if row.get("admission_status") == "admitted"]
    fitted = [row for row in eligible if row.get("fit_role") in {"train", "historical_fit", "development"}]
    sealed = [row for row in eligible if row.get("fit_role") == "sealed_validation"]
    return {
        "schema": "rinkfit.joint_residual_summary.v1", "all_source_rows": _summary(data, thresholds),
        "admitted_fit_rows": _summary(fitted, thresholds),
        "admitted_sealed_validation_rows": _summary(sealed, thresholds),
        "by": by,
        "rows_missing_admission_status": sum(not row.get("admission_status") for row in data),
        "rows_missing_fit_role": sum(not row.get("fit_role") for row in data),
        "full_W_cost_computed_here": False,
        "metric_scale_or_world_accuracy_certified": False,
    }


def audit_dense_frame_receipts(
    schedule: Iterable[Mapping[str, Any]],
    receipts: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Demand evidence receipts for every included frame, including 'no support'.

    A receipt needs shot_id/native_pts and status in verified_static_evidence,
    no_visible_static_support, or evidence_failed. A frame with no visible
    support is an explicit gap, never a zero-error success. Verified receipts
    need source_pixel_evidence_sha256 and model_sha256, explicit static-source
    and visibility verification, and physical_projection_replayed=true.
    Projected/flow-only points cannot qualify as measured static evidence.
    """
    expected = {(str(row["shot_id"]), int(row["native_pts"])) for row in schedule if row.get("processing_status") != "excluded_by_manifest"}
    found: dict[tuple[str, int], Mapping[str, Any]] = {}
    errors: list[str] = []
    for receipt in receipts:
        key = (str(receipt["shot_id"]), int(receipt["native_pts"]))
        if key in found:
            errors.append(f"Duplicate receipt: {key}")
        found[key] = receipt
        if key not in expected:
            errors.append(f"Receipt outside processing scope: {key}")
        status = receipt.get("status")
        if status not in {"verified_static_evidence", "no_visible_static_support", "evidence_failed"}:
            errors.append(f"Unknown or unfinished receipt status: {key}")
        if status == "verified_static_evidence":
            if not receipt.get("source_pixel_evidence_sha256") or not receipt.get("model_sha256") or receipt.get("physical_projection_replayed") is not True:
                errors.append(f"Missing source/model replay evidence: {key}")
            if not receipt.get("observation_origin") or receipt.get("static_surface_identity_verified") is not True or receipt.get("visibility_verified") is not True:
                errors.append(f"Missing current-frame static-source/visibility verification: {key}")
            if receipt.get("observation_origin") in {"current_model_projection", "flow_only", "homography_only"} or receipt.get("measurement_generated_from_current_model") is not False:
                errors.append(f"Synthetic/propagated-only evidence mislabeled as measurement: {key}")
    missing = sorted(expected.difference(found))
    counts = Counter(str(row.get("status")) for key, row in found.items() if key in expected)
    return {
        "schema": "rinkfit.dense_frame_receipt_audit.v1",
        "expected_frames": len(expected), "received_in_scope": len(expected.intersection(found)),
        "missing_frames": len(missing), "missing_first_20": missing[:20],
        "status_counts": dict(counts), "errors": errors,
        "all_frames_accounted_for": not missing and not errors,
        "full_frame_geometric_validation_passed": bool(expected) and not missing and not errors and counts["verified_static_evidence"] == len(expected),
        "note": "Coverage accounting is separate from residual thresholds and world/metric accuracy. No-visible-support frames require explicit uncertainty, not synthetic landmarks.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("schedule", "split", "summary", "receipts"))
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", default="okada-20261018-joint-v1")
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    if args.mode == "schedule":
        result = build_frame_schedule(payload["shots"], payload.get("time_base", "1/12800"), int(payload.get("step_pts", 256)))
    elif args.mode == "split":
        result = assign_locked_roles(payload["rows"], seed=args.seed)
    elif args.mode == "receipts":
        result = audit_dense_frame_receipts(payload["schedule"], payload["receipts"])
    else:
        result = summarize_residuals(payload["rows"])
    text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
