"""Portable, source-preserving adapters. No project runner, model, or training import.

Only caller-supplied JSON objects/paths are consumed. Paths inside historical JSON
are evidence metadata, never executable or implicitly opened. See ADAPTER_CONTRACTS.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from fractions import Fraction
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping


class AdapterError(ValueError):
    """Input cannot be safely represented by this adapter contract."""


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path | str) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def checked_relative(root: Path | str, relative: str) -> Path:
    """Resolve only package-relative paths and reject traversal / foreign drives."""
    if not isinstance(relative, str) or not relative or "\\" in relative or ":" in relative:
        raise AdapterError("Expected a portable package-relative path")
    part = Path(relative)
    if part.is_absolute() or ".." in part.parts:
        raise AdapterError("Evidence path escapes the supplied root")
    root = Path(root).resolve()
    result = (root / part).resolve()
    if result != root and root not in result.parents:
        raise AdapterError("Evidence symlink escapes the supplied root")
    return result


def _clock(value: Any) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9]+/[0-9]+", value):
        raise AdapterError("time_base must be a native rational string")
    try:
        fraction = Fraction(value)
    except (ValueError, ZeroDivisionError) as exc:
        raise AdapterError("Invalid native time_base") from exc
    if fraction <= 0:
        raise AdapterError("Invalid native time_base")
    return f"{fraction.numerator}/{fraction.denominator}"


def normalize_source(source: Mapping[str, Any]) -> dict[str, Any]:
    sha = source.get("video_sha256", source.get("source_video_sha256"))
    if not isinstance(sha, str) or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
        raise AdapterError("source requires a lowercase SHA256")
    aliases = [source[k] for k in ("video_sha256", "source_video_sha256") if k in source]
    if any(value != sha for value in aliases):
        raise AdapterError("Conflicting source SHA aliases")
    pts = source.get("pts", source.get("native_pts", source.get("source_pts")))
    if isinstance(pts, bool) or not isinstance(pts, int):
        raise AdapterError("source PTS must be an integer, never rounded time")
    if any(source[k] != pts for k in ("pts", "native_pts", "source_pts", "actual_pts") if k in source):
        raise AdapterError("Conflicting native PTS aliases")
    raw_time_base = source.get("time_base", source.get("source_time_base"))
    time_base = _clock(raw_time_base)
    if any(_clock(source[k]) != time_base for k in ("time_base", "source_time_base") if k in source):
        raise AdapterError("Conflicting native time_base aliases")
    domain = source.get("coordinate_domain", "native_pixels" if source.get("native_hw") else None)
    if not isinstance(domain, str) or not domain:
        raise AdapterError("An explicit coordinate domain is required")
    result = {"video_sha256": sha, "pts": pts, "time_base": time_base,
              "coordinate_domain": domain}
    for key in ("native_hw", "decoded_RGB_sha256", "decoded_BGR_sha256", "png_sha256",
                "event_id", "programme_id", "source_id", "crop_transform"):
        if key in source:
            result[key] = deepcopy(source[key])
    if "native_hw" in result:
        hw = result["native_hw"]
        if not (isinstance(hw, list) and len(hw) == 2 and all(type(v) is int and v > 0 for v in hw)):
            raise AdapterError("native_hw must be [height, width]")
    return result


def source_key(source: Mapping[str, Any]) -> tuple[Any, ...]:
    item = normalize_source(source)
    # Equivalent rational notation is normalized; PTS and physical source never change.
    return item["video_sha256"], item["pts"], item["time_base"], item["coordinate_domain"]


def packet_points(packet: Mapping[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for feature in packet.get("features", []):
        fid = feature["id"]
        for si, segment in enumerate(feature.get("segments", [])):
            uvs = segment.get("uv", [])
            point_ids = segment.get("point_ids")
            if point_ids is not None and len(point_ids) != len(uvs):
                raise AdapterError("point_ids and UV length differ")
            bindings = segment.get("point_bindings", [])
            for vi, uv in enumerate(uvs):
                if not (isinstance(uv, list) and len(uv) == 2 and all(
                    isinstance(v, (float, int)) and not isinstance(v, bool) and math.isfinite(v) for v in uv
                )):
                    raise AdapterError("Nonfinite or malformed native UV")
                point_id = point_ids[vi] if point_ids is not None else f"{fid}/segment/{si}/vertex/{vi}"
                if point_id in seen:
                    raise AdapterError("Duplicate source point ID")
                seen.add(point_id)
                binding = bindings[vi] if vi < len(bindings) else {}
                result.append({"point_id": point_id, "feature_id": fid,
                               "target_id": binding.get("physical_target", feature.get("target_id", fid)),
                               "uv": deepcopy(uv), "segment_index": si, "vertex_index": vi})
    if packet.get("point_count") != len(result):
        raise AdapterError("Packet point_count does not match its actual UV records")
    return result


def _point_multiset(rows: list[dict[str, Any]]) -> Counter:
    # The editor can rebuild segment indices. UV + physical feature identity, not
    # regenerated row index, is the conservation invariant for untouched points.
    return Counter((r["feature_id"], r["target_id"], tuple(r["uv"])) for r in rows)


def adapt_frame_edit(
    *, transaction: Mapping[str, Any], before_packet: Mapping[str, Any],
    after_packet: Mapping[str, Any], binding: Mapping[str, Any],
    acceptance: Mapping[str, Any] | None, file_hashes: Mapping[str, str],
    evidence_refs: list[dict[str, Any]], earlier_packet: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert supported committed deletion transactions to an auditable event.

    This v1 handles the real 205/649/1247 deletion schema. Unsupported add/move
    schemas fail explicitly; they are never silently represented as deletions.
    file_hashes must be calculated from original bytes by the caller, with keys
    before_packet, after_packet, binding, transaction (and optionally acceptance).
    """
    if transaction.get("schema") != "rinkspace.human-frame-point-edit.v1" or transaction.get("status") != "committed":
        raise AdapterError("Only committed human-frame-point-edit.v1 is supported")
    if transaction.get("model_geometry_changed") is not False:
        raise AdapterError("This adapter requires a transaction that did not change geometry")
    for name in ("before_packet", "after_packet", "binding", "transaction"):
        value = file_hashes.get(name)
        if not isinstance(value, str) or len(value) != 64:
            raise AdapterError(f"Missing byte SHA256 for {name}")
    source = normalize_source(after_packet["source"])
    if source_key(before_packet["source"]) != source_key(source):
        raise AdapterError("Before/after source identity, native PTS, or domain differs")
    for packet in (before_packet, after_packet):
        if packet.get("source_pts") != source["pts"]:
            raise AdapterError("Packet outer source_pts differs from embedded source")
    if transaction.get("source_pts") != source["pts"] or binding.get("source_pts") != source["pts"]:
        raise AdapterError("Transaction/binding native PTS differs")
    if _clock(binding.get("source_time_base")) != source["time_base"]:
        raise AdapterError("Binding native time_base differs")
    index = transaction["index"]
    if index != after_packet.get("programme_frame_index_1based") or index != before_packet.get("programme_frame_index_1based"):
        raise AdapterError("Programme index mismatch")
    if transaction["packet_after"]["sha256"] != file_hashes["after_packet"]:
        raise AdapterError("Transaction does not bind the provided after packet bytes")
    if binding["source_packet"]["sha256"] != file_hashes["after_packet"]:
        raise AdapterError("Binding does not bind the provided after packet bytes")
    before = packet_points(before_packet)
    after = packet_points(after_packet)
    if transaction.get("point_count_before") != len(before) or transaction.get("point_count_after") != len(after):
        raise AdapterError("Human transaction point counts differ from actual before/after packets")
    if binding.get("point_count") != len(after):
        raise AdapterError("Binding point count differs")
    operations = transaction.get("action", {}).get("operations", [])
    if not operations or any(op.get("type") != "delete" for op in operations):
        raise AdapterError("This adapter version supports explicit delete operations only")
    deleted_ids = [op["point_id"] for op in operations]
    if len(set(deleted_ids)) != len(deleted_ids):
        raise AdapterError("Duplicate delete operation")
    lookup = {p["point_id"]: p for p in before}
    if any(pid not in lookup for pid in deleted_ids):
        raise AdapterError("A deleted point is absent from the true pre-transaction packet")
    deleted = [deepcopy(lookup[pid]) for pid in deleted_ids]
    untouched = [p for p in before if p["point_id"] not in set(deleted_ids)]
    if _point_multiset(untouched) != _point_multiset(after):
        raise AdapterError("Unrecorded UV/physical-target delta; cannot call it only these deletions")
    old_count = None
    if earlier_packet is not None:
        if source_key(earlier_packet["source"]) != source_key(source):
            raise AdapterError("Earlier assistant-repair packet has a different source")
        old_count = len(packet_points(earlier_packet))
    accepted = False
    acceptance_row = None
    if acceptance is not None:
        if acceptance.get("accepted") is not True:
            raise AdapterError("Supplied acceptance receipt is not accepted")
        if acceptance.get("official_source", {}).get("sha256") != source["video_sha256"]:
            raise AdapterError("Acceptance source SHA differs")
        if _clock(acceptance.get("source_time_base")) != source["time_base"]:
            raise AdapterError("Acceptance source time_base differs")
        rows = [r for r in acceptance.get("frames", []) if r.get("index") == index]
        if len(rows) != 1:
            raise AdapterError("Acceptance must bind exactly one row for this programme index")
        acceptance_row = rows[0]
        expected = {"packet_sha256": file_hashes["after_packet"], "binding_sha256": file_hashes["binding"],
                    "native_pts": source["pts"], "point_count": len(after)}
        if any(acceptance_row.get(k) != v for k, v in expected.items()):
            raise AdapterError("Acceptance packet/binding SHA or source PTS/count mismatch")
        accepted = True
    event = {
        "schema": "rink-feedback.correction-event.v1", "event_id": f"human_edit:{transaction['id']}",
        "kind": "withdraw_source_association", "source": source,
        "programme_frame_index_1based": index, "transaction_id": transaction["id"],
        "deleted_points": deleted,
        "counts": {"earlier_assistant_repair_input": old_count, "human_transaction_before": len(before),
                   "after": len(after), "human_deleted": len(deleted)},
        "reason": "unknown", "reason_authority": "not_recorded_in_human_transaction",
        "background_negative_created": False, "exact_source_frame_and_point_ids_only": True,
        "source_observations_accepted": accepted,
        "model_accuracy_accepted": acceptance.get("model_accuracy_accepted") if acceptance is not None else None,
        "geometry_accuracy_inferred_from_edit": False,
        "original_uv_changed_by_adapter": False, "historical_paths_executed": False,
        "file_hashes": dict(file_hashes), "acceptance_row": deepcopy(acceptance_row),
        "evidence_refs": deepcopy(evidence_refs),
        "implementation_status": "implemented_portable_adapter",
        "production_integration_status": "pending_live_runner_connection",
    }
    return event


def correction_event_to_case(event: Mapping[str, Any]) -> dict[str, Any]:
    source = normalize_source(event["source"])
    entities = sorted({p["target_id"] for p in event["deleted_points"]})
    return {
        "case_id": event["event_id"], "version": 1, "status": "active",
        "scope": {"event_id": source.get("event_id"), "source_video_sha256": source["video_sha256"],
                  "camera_id": None, "entity_ids": entities},
        "tags": ["manual_edit", "withdraw_association", "F4_R4", "reason_unknown", "native50"],
        "title": f"官方50 fps 第{event['programme_frame_index_1based']}帧：人工撤回{len(event['deleted_points'])}点关联，原因未记录",
        "assertions": {
            "withdrawal": {"source": source, "points": deepcopy(event["deleted_points"]),
                           "applicability": "exact_source_frame_and_point_ids_only", "authority": "human_editor_transaction"},
            "reason": {"value": "unknown", "authority": "not_recorded"},
            "position": {"pixel_ground_truth": False, "withdrawal_is_positive_position_truth": False},
            "negative_supervision": {"background_negative_created": False, "whole_entity_rejected": False},
            "acceptance": {"source_observations_accepted": event["source_observations_accepted"],
                           "model_accuracy_accepted": event["model_accuracy_accepted"],
                           "geometry_accuracy_inferred": False},
            "counts": deepcopy(event["counts"]),
            "counterexample": "The deleted association does not invalidate other visible portions or other frames.",
        },
        "evidence_refs": deepcopy(event["evidence_refs"]),
        "provenance": {"authority": "human_editor_transaction", "derived_by": "rink_feedback.adapters.correction_event_to_case",
                       "transaction_id": event["transaction_id"], "source": source,
                       "derivation_is_new_human_pixel_review": False},
        "split_role": "development", "lineage_ids": [f"capture:{source['video_sha256']}", event["event_id"]],
        "lineage": {"lineage_ids": [f"capture:{source['video_sha256']}", event["event_id"]]},
    }


def normalize_fit_finding(row: Mapping[str, Any], *, run_id: str, event_id: str) -> dict[str, Any]:
    """Convert a runner-emitted diagnostic signal; never conclude the UV is wrong.

    Required row fields: kind, source. Recommended: entity_ids, observation_ids,
    programme_id, shot_id, tags, metrics, evidence_refs, finding_id. The runner
    chooses its own uncertainty/needs-based triggers; no threshold is imposed here.
    """
    if not isinstance(run_id, str) or not run_id or not isinstance(event_id, str) or not event_id:
        raise AdapterError("run_id and event_id are required")
    source = normalize_source(row["source"])
    if source.get("event_id") not in (None, event_id):
        raise AdapterError("Finding source event differs from supplied event")
    kind = row.get("kind")
    allowed = {"residual", "coverage", "identity", "visibility", "tracking", "manual_edit", "operator_conflict"}
    if kind not in allowed:
        raise AdapterError(f"Unsupported finding kind: {kind!r}")
    entities = row.get("entity_ids", [row["entity_id"]] if row.get("entity_id") else [])
    if not isinstance(entities, list) or not all(isinstance(x, str) for x in entities):
        raise AdapterError("entity_ids must be strings")
    tags = list(dict.fromkeys([kind, *row.get("tags", [])]))
    metrics = deepcopy(row.get("metrics", {}))
    # Reject NaN/Infinity even in nested metrics before they enter persistent JSON.
    canonical_sha256(metrics)
    payload = {
        "run_id": run_id, "kind": kind, "source": source, "event_id": event_id,
        "programme_id": row.get("programme_id", source.get("programme_id", source["video_sha256"])),
        "shot_id": row.get("shot_id"), "entity_ids": entities,
        "scope": {"event_id": event_id, "source_video_sha256": source["video_sha256"],
                  "camera_id": row.get("camera_id"), "entity_ids": entities},
        "observation_ids": deepcopy(row.get("observation_ids", [])), "tags": tags,
        "metrics": metrics, "evidence_refs": deepcopy(row.get("evidence_refs", [])),
        "status": "open", "diagnostic_signal_is_ground_truth": False,
        "proposed_action": "retrieve_related_cases_and_reinspect_original_source",
        "provenance": {"authority": "runner_diagnostic_signal", "adapter": "normalize_fit_finding",
                       "original_row_sha256": canonical_sha256(row)},
    }
    payload["finding_id"] = row.get("finding_id") or "fit:" + canonical_sha256(payload)
    return payload


def record_findings_and_query(store: Any, rows: list[Mapping[str, Any]], *, run_id: str,
                              event_id: str) -> dict[str, Any]:
    """Executable callback for a connected runner; retrieval is not consumption.

    The caller must subsequently inspect native pixels and call record_use and
    record_resolution with that evidence. This adapter never fabricates either.
    """
    outputs = []
    for row in rows:
        finding = normalize_fit_finding(row, run_id=run_id, event_id=event_id)
        record = store.record_finding(finding)
        context = {**finding["scope"], "source": finding["source"], "tags": finding["tags"]}
        matches = store.query(context)
        outputs.append({"finding_id": finding["finding_id"], "record_result": record,
                        "query_context": context, "retrieval": matches,
                        "native_source_review_pending": True, "record_use_written": False})
    return {"schema": "rink-feedback.fit-adapter-receipt.v1", "run_id": run_id,
            "implementation_status": "implemented_portable_adapter",
            "production_integration_status": "caller_must_prove_live_runner_connection",
            "fit_or_training_executed": False, "items": outputs}


def import_seed_bundle(store: Any, package_root: Path | str) -> dict[str, Any]:
    """Hash-verify bundled input evidence and idempotently import its case records."""
    root = Path(package_root)
    manifest_path = checked_relative(root, "data/seed_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for evidence in manifest["evidence"]:
        path = checked_relative(root, evidence["path"])
        if path.stat().st_size != evidence["bytes"] or file_sha256(path) != evidence["sha256"]:
            raise AdapterError(f"Bundled evidence changed: {evidence['path']}")
    cases_path = checked_relative(root, manifest["cases"]["path"])
    if file_sha256(cases_path) != manifest["cases"]["sha256"]:
        raise AdapterError("Bundled seed cases changed")
    cases = [json.loads(line) for line in cases_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(cases) != manifest["cases"]["count"]:
        raise AdapterError("Seed case count differs")
    results = [store.upsert_case(case) for case in cases]
    return {"schema": "rink-feedback.seed-import-receipt.v1", "case_count": len(cases),
            "case_results": results, "verified_evidence_count": len(manifest["evidence"]),
            "training_or_fitting_executed": False, "new_visual_accuracy_claimed": False,
            "implementation_status": "implemented_portable_adapter",
            "production_integration_status": "pending_live_runner_connection"}


def import_project_seed_bundle(store: Any, manifest_path: Path | str) -> dict[str, Any]:
    """Import the project seed index while keeping heavy evidence outside OneDrive.

    The explicit local manifest is the sole resolver. Historical addresses in the
    package remain metadata. Every evidence byte and mapped case reference is
    checked before any case is appended to the ledger.
    """
    manifest_path = Path(manifest_path).resolve(strict=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "rink-feedback.project-seed-manifest.v1":
        raise AdapterError("Unsupported project seed manifest schema")
    evidence_root = Path(manifest["evidence_root"]).resolve(strict=True)
    if not evidence_root.is_dir():
        raise AdapterError("Evidence root must be an existing directory")
    verified = {}
    for evidence in manifest["evidence"]:
        path = checked_relative(evidence_root, evidence["path"])
        if path.stat().st_size != evidence["bytes"] or file_sha256(path) != evidence["sha256"]:
            raise AdapterError(f"Project evidence changed: {evidence['path']}")
        identity = (str(path), evidence["sha256"])
        if identity in verified:
            raise AdapterError("Duplicate evidence identity in project manifest")
        verified[identity] = evidence
    cases_path = checked_relative(manifest_path.parent, manifest["cases"]["path"])
    if file_sha256(cases_path) != manifest["cases"]["sha256"]:
        raise AdapterError("Project seed cases changed")
    cases = [json.loads(line) for line in cases_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(cases) != manifest["cases"]["count"]:
        raise AdapterError("Project seed case count differs")
    for case in cases:
        for ref in case.get("evidence_refs", []):
            identity = (str(Path(ref["path"]).resolve(strict=True)), ref["sha256"])
            if identity not in verified:
                raise AdapterError("Case evidence is not verified by project manifest")
            if ref.get("original_package_path") != verified[identity]["package_path"]:
                raise AdapterError("Case evidence package mapping differs")
    results = [store.upsert_case(case) for case in cases]
    return {
        "schema": "rink-feedback.project-seed-import-receipt.v1",
        "manifest_path": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "package_sha256": manifest["package_sha256"],
        "package_commit": manifest["package_commit"],
        "case_count": len(cases),
        "case_results": results,
        "verified_evidence_count": len(verified),
        "all_idempotent": all(row["idempotent"] for row in results),
        "training_or_fitting_executed": False,
        "new_visual_accuracy_claimed": False,
        "historical_seed_import_only": True,
    }
