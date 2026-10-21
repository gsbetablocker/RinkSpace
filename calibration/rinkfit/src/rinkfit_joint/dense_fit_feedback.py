"""Dense source feedback at immutable qualification/packet boundaries.

Source UV is immutable. Residuals select a subject for an actual source review;
they do not supply a replacement position or a rejection decision.
"""
from __future__ import annotations
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
from .io_utils import file_sha256, object_sha256, read_json, read_jsonl
from .source_record_store import SourceRecordStore
from .source_record_io import canonical_record

SCHEMA = "rinkfit.dense_source_feedback.v1"
ROOT = Path(__file__).resolve().parents[4]
CHECKS = ("semantic_class", "physical_surface", "physical_instance",
          "measurement_operator", "visibility", "position")


def reference(path):
    path = Path(path).resolve()
    return {"file": str(path), "sha256": file_sha256(path)}


def _checked(ref):
    path = Path(ref.get("file", ref.get("path"))).resolve()
    if file_sha256(path) != ref["sha256"]:
        raise ValueError("Changed actual source feedback input: " + str(path))
    return path


def _new_json(path, value):
    path = Path(path).resolve()
    if len(str(path).encode("utf-16-le")) // 2 >= 260 or any(len(p.encode("utf-16-le")) // 2 > 255 for p in path.parts):
        raise ValueError("Feedback output exceeds project path limits")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return reference(path)


def _one_frame(ref, shot, pts):
    """Read one compressed frame; metadata and every decompressed blob are checked."""
    if ref.get("store_content_addressed") is not True:
        raise ValueError("Dense feedback requires an explicit immutable logical stream")
    with SourceRecordStore(ref["store"], readonly=True) as store:
        store.db.execute("BEGIN")
        info = store.stream_info(ref["stream"])
        if not info["sealed"] or info["kind"] != "selected_rows" or info["semantic_sha256"] != ref["stream_semantic_sha256"]:
            raise ValueError("Dense source stream differs from its sealed reference")
        rows = list(store.iterate_records(ref["stream"], frame=(shot, int(pts))))
        store.db.execute("COMMIT")
    if not rows or any((r["shot_id"], r["native_pts"]) != (shot, pts) for r in rows):
        raise ValueError("Dense source frame missing or belongs to another shot")
    ids = [r["selected_row_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Ambiguous dense source selection IDs")
    digest = hashlib.sha256(b"".join(canonical_record(r) for r in rows)).hexdigest()
    return {r["selected_row_id"]: r for r in rows}, digest


def collect_dense_findings(rows_ref, triggers, source, output=None):
    """Bind real triggers to selected rows; no qualification/UV writes."""
    if not triggers:
        return []
    if source.get("time_base") != "1/12800" or not source.get("video_sha256"):
        raise ValueError("Explicit official source identity and native clock required")
    frames, result = {}, []
    for trigger in triggers:
        shot, pts = trigger["shot_id"], trigger["native_pts"]
        if isinstance(pts, bool) or not isinstance(pts, int) or pts % 256:
            raise ValueError("Trigger must use exact native PTS")
        if not trigger.get("signal") or not trigger.get("evidence_refs"):
            raise ValueError("A source trigger needs its actual diagnostic/evidence reference")
        for ref in trigger["evidence_refs"]:
            _checked(ref)
        key = shot, pts
        if key not in frames:
            frames[key] = _one_frame(rows_ref, *key)
        rows, frame_sha = frames[key]
        row = rows[trigger["selected_row_id"]]
        subject = {"shot_id": shot, "native_pts": pts,
            "selected_row_id": row["selected_row_id"], "source_point_id": row["source_point_id"],
            "feature_id": row["feature_id"], "source_uv": row["uv"],
            "time_base": row["time_base"], "physical_surface": row["physical_surface"],
            "measurement_definition": row["measurement_definition"]}
        binding = dict(source, pts=pts, coordinate_domain="native_pixels")
        finding = {"schema": SCHEMA + ".finding", "source": binding, "subject": subject,
            "source_rows": rows_ref, "source_frame_semantic_sha256": frame_sha,
            "qualification_snapshot": row,
            "signal": trigger["signal"], "diagnostic": trigger.get("diagnostic"),
            "evidence_refs": trigger["evidence_refs"],
            "source_review_required": True, "residual_is_truth_label": False,
            "replacement_UV_or_qualification_issued": False}
        finding["dense_finding_id"] = object_sha256(finding)
        result.append(finding)
    if output is not None:
        _new_json(output, {"schema": SCHEMA + ".findings", "findings": result,
            "source_rows": rows_ref, "source_UV_changed": False})
    return result


def dispatch_packet_reviews(task_documents, review_run, output, *, packet_path, interpreter=None, device="cuda"):
    """Use the existing batch-v4 actual native operators and exact reuse checks."""
    from .feedback_dispatch_batch_v4 import dispatch_checkpoint_reviews
    return dispatch_checkpoint_reviews(task_documents, review_run, output,
        packet_path=packet_path, interpreter=interpreter, device=device)


def bind_packet_findings(task_document, rows_ref, packet_bindings, output):
    """Add dense identities/dependencies to real batch findings without changing tasks.

    packet_bindings maps exact physical packet row index to shot/PTS/selected ID.
    Missing mappings remain explicit pending subjects and cannot mutate a source.
    """
    from .feedback_task_codec_v1 import read_task_document
    document = read_task_document(task_document)
    triggers, pending = [], []
    for task in document["tasks"]:
        index = str(task["source_row_index"])
        link = packet_bindings.get(index)
        if link is None:
            pending.append({"finding_id": task["finding_id"], "source_row_index": int(index),
                "reason": "no_exact_dense_source_selection_binding"})
            continue
        if int(link["native_pts"]) != int(task["context"]["source"]["pts"]):
            raise ValueError("Physical packet finding and dense native exposure differ")
        triggers.append(dict(link, signal=task["context"].get("signal_type", "physical_fit_source_recheck"),
            diagnostic=task.get("diagnostic"), evidence_refs=[reference(task_document)]))
    if triggers:
        source = document["tasks"][0]["context"]["source"]
        source = {k: v for k, v in source.items() if k not in {"pts", "coordinate_domain"}}
        findings = collect_dense_findings(rows_ref, triggers, source)
    else:
        findings = []
    return _new_json(output, {"schema": SCHEMA + ".packet_findings",
        "parent_tasks": reference(task_document), "source_rows": rows_ref,
        "findings": findings, "pending": pending,
        "batch_review_dependencies_preserved": True, "physical_packet_mutated": False})


def _record(ref, shot, pts):
    """An exact physical JSONL source index; do not cache unrelated frames."""
    path = _checked(ref)
    found = [r for r in read_jsonl(path) if r.get("native_pts") == pts and r.get("shot_id") == shot]
    if len(found) != 1:
        raise ValueError("Exact source frame has no unique accepted mask/proof binding")
    return found[0]


def review_existing_visibility(finding, *, mask_manifest_ref, mask_index_ref,
                               scope_ref, roles_ref, output, previous_review=None):
    """Actual one-frame mask read; only proven exclusion can change membership.

    Previously accepted material/source definitions remain unknown here. A clear
    pixel outside masks does not certify an ice/board material or 3D instance.
    """
    from .native_source import iter_scope_native_rgb
    from PIL import Image
    import cv2
    subject = finding["subject"]
    shot, pts = subject["shot_id"], subject["native_pts"]
    rows, frame_sha = _one_frame(finding["source_rows"], shot, pts)
    row = rows[subject["selected_row_id"]]
    if row != finding["qualification_snapshot"] or frame_sha != finding["source_frame_semantic_sha256"]:
        raise ValueError("Review finding source selection changed")
    manifest = read_json(_checked(mask_manifest_ref))
    if manifest.get("complete") is not True or manifest.get("accepted_for_final_source_qualification") is not True:
        raise ValueError("Mask generation has not been source-qualified")
    if manifest.get("source_video_sha256") != finding["source"]["video_sha256"]:
        raise ValueError("Mask source video identity differs")
    if manifest.get("index", {}).get("sha256") != mask_index_ref["sha256"]:
        raise ValueError("Mask index is not the accepted mask generation index")
    frame = _record(mask_index_ref, shot, pts)
    if frame["time_base"] != subject["time_base"] or frame["source_video_sha256"] != finding["source"]["video_sha256"]:
        raise ValueError("Mask native source clock differs")
    maskrefs = [dict(file=frame["mask_file"], sha256=frame["mask_sha256"])]
    unknown = frame.get("source_unknown_exclusion_mask_ref")
    if unknown:
        if (unknown.get("native_pts"), unknown.get("shot_id"), unknown.get("source_RGB_sha256")) != (pts, shot, frame["source_RGB_sha256"]):
            raise ValueError("Typed unknown mask is not bound to this exposure")
        qual = read_json(_checked(unknown["source_qualification_ref"]))
        if qual.get("accepted_for_source_exclusion") is not True or qual.get("source_video_sha256") != frame["source_video_sha256"]:
            raise ValueError("Unknown mask lacks an accepted source exclusion decision")
        maskrefs.append(unknown)
    code = [reference(__file__), reference(Path(__file__).with_name("native_source.py"))]
    evidence = [mask_manifest_ref, mask_index_ref, scope_ref, roles_ref] + maskrefs
    for ref in evidence:
        _checked(ref)
    dependencies = {"reader_code_sha256": object_sha256(code),
        "reader_config_sha256": object_sha256({"operator": "current_qualified_native_masks_at_original_UV",
            "nonzero_means": "excluded", "person_guard_px": manifest.get("person_guard_pixels", 0),
            "unknown_guard_px": manifest.get("source_unknown_guard_pixels", 0)}),
        "source_binding_sha256": object_sha256({"source": finding["source"],
            "source_RGB_sha256": frame["source_RGB_sha256"], "source_BGR_sha256": frame["source_BGR_sha256"],
            "masks": maskrefs}),
        "observation_snapshot_sha256": object_sha256(row),
        "effective_facts_sha256": object_sha256(evidence),
        "navigation_state_sha256": "not_used"}
    if previous_review is not None:
        prior = read_json(_checked(previous_review))
        if prior.get("review_dependencies") == dependencies and prior.get("subject") == subject and prior.get("source") == finding["source"]:
            for ref in prior["evidence_refs"]:
                _checked(ref)
            reused = dict(prior, review_reused_from=previous_review, original_review_record=reference(output) if Path(output).exists() else None,
                source_pixels_decoded_this_review=0, actual_current_dependency_and_same_subject_verified=True)
            reused.pop("original_review_record", None)
            _new_json(output, reused)
            return reused
    scope, roles = _checked(scope_ref), _checked(roles_ref)
    _, rgb, decode = next(iter_scope_native_rgb(scope, [pts], roles_path=roles))
    if decode["shot_id"] != shot or decode["decoded_native_RGB_sha256"] != frame["source_RGB_sha256"] or decode["decoded_native_BGR_sha256"] != frame["source_BGR_sha256"]:
        raise ValueError("Actual single native RGB differs from accepted mask exposure")
    masks = [np.asarray(Image.open(_checked(ref)).convert("L")) > 0 for ref in maskrefs]
    if any(m.shape != rgb.shape[:2] for m in masks):
        raise ValueError("Current native mask raster differs")
    x, y = np.rint(subject["source_uv"]).astype(int)
    if not (0 <= y < rgb.shape[0] and 0 <= x < rgb.shape[1]):
        raise ValueError("Original source UV outside native exposure")
    reads = []
    for i, mask in enumerate(masks):
        distance = cv2.distanceTransform((~mask).astype("uint8"), cv2.DIST_L2, 5)
        guard = int(manifest.get("person_guard_pixels" if i == 0 else "source_unknown_guard_pixels", 0))
        reads.append({"layer": "person" if i == 0 else "source_unknown",
            "pixel": [int(x), int(y)], "inside": bool(mask[y, x]),
            "distance_px": float(distance[y, x]), "guard_px": guard,
            "excluded_by_actual_source_mask": bool(distance[y, x] <= guard)})
    excluded = any(r["excluded_by_actual_source_mask"] for r in reads)
    checks = {name: {"status": "unknown", "reason": "visibility reader does not certify this dimension"} for name in CHECKS}
    checks["visibility"] = {"status": "verified" if excluded else "unknown", "visible": False if excluded else None,
        "reason": "original UV inside accepted source person/typed unknown mask or source guard" if excluded else "no mask exclusion; material and instance need their actual native reader"}
    review = {"schema": SCHEMA + ".visibility_review", "subject": subject, "source": finding["source"],
        "finding_id": finding["dense_finding_id"], "review_dependencies": dependencies,
        "evidence_refs": evidence + code, "actual_native_decode": decode, "mask_reads": reads,
        "checks": checks, "qualification_action": "exclude" if excluded else "pending",
        "qualification_reason": "current_native_actor_or_typed_source_uncertainty" if excluded else "unresolved_material_instance_or_position",
        "original_source_UV_retained": True, "source_pixels_decoded_this_review": 1,
        "model_residual_used_for_qualification": False, "new_observation_installed": False}
    _new_json(output, review)
    return review


def commit_qualification_reviews(*, base_rows, base_frames, review_refs,
                                 new_rows_stream, new_frames_stream, output,
                                 parent_packet=None):
    """Public source-store delta commit followed by an explicit packet-rebind manifest.

    This visibility-only adapter can remove a proven unavailable selected reading.
    It never creates precision, 3D correspondences, or source-position corrections.
    """
    out = Path(output).resolve()
    if out.exists():
        raise FileExistsError("Immutable dense feedback commit directory exists")
    changes, accepted, pending, affected, evidence = {}, [], [], [], []
    for ref in review_refs:
        review = read_json(_checked(ref))
        if review.get("schema") != SCHEMA + ".visibility_review" or review.get("model_residual_used_for_qualification") is not False:
            raise ValueError("Qualification needs an actual accepted native visibility review")
        for dependency in review["evidence_refs"]:
            _checked(dependency)
        s = review["subject"]
        if review.get("qualification_action") != "exclude":
            pending.append({"review": ref, "selected_row_id": s["selected_row_id"],
                "reason": review["qualification_reason"]})
            continue
        visibility = review["checks"]["visibility"]
        if visibility.get("status") != "verified" or visibility.get("visible") is not False or not any(r["excluded_by_actual_source_mask"] for r in review["mask_reads"]):
            raise ValueError("No verified current source evidence supports exclusion")
        rows, _ = _one_frame(base_rows, s["shot_id"], s["native_pts"])
        before = rows[s["selected_row_id"]]
        if before["source_point_id"] != s["source_point_id"] or before["uv"] != s["source_uv"]:
            raise ValueError("Review source identity/UV differs from commit parent")
        key = s["shot_id"], s["native_pts"]
        item = changes.setdefault(key, {"shot_id": key[0], "native_pts": key[1], "remove": []})
        if s["selected_row_id"] in item["remove"]:
            raise ValueError("Duplicate qualification removal")
        item["remove"].append(s["selected_row_id"])
        accepted.append({"review": ref, "reason": review["qualification_reason"], "original_qualification": before})
        affected.append(s)
        evidence.append({"file": str(_checked(ref)), "sha256": ref["sha256"], "role": "actual current source visibility read"})
    if not changes:
        return {"schema": SCHEMA + ".no_commit", "pending": pending, "new_packet_required": False}
    if Path(base_rows["store"]).resolve() != Path(base_frames["store"]).resolve():
        raise ValueError("Qualification rows and frame headers use different source stores")
    out.mkdir(parents=True)
    request = {"schema": "rinkspace.source_storage_commit_request.v1", "db": base_rows["store"],
        "base_rows": base_rows, "base_frames": base_frames,
        "new_rows_stream": new_rows_stream, "new_frames_stream": new_frames_stream,
        "changes": list(changes.values()), "source_evidence": evidence}
    request_ref = _new_json(out / "source_request.json", request)
    spec = importlib.util.spec_from_file_location("rinkspace_public_source_storage", ROOT / "scripts/source_storage.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.commit(request_ref["file"], out / "source_manifest.json")
    manifest = read_json(result["manifest"])
    rebind = {"schema": SCHEMA + ".next_packet_rebind",
        "source_commit": reference(result["manifest"]), "parent_source_rows": base_rows,
        "selected_rows": manifest["selected_rows"], "selected_frames": manifest["selected_frames"],
        "parent_packet": parent_packet, "affected_sources": affected, "accepted_reviews": accepted,
        "pending_candidates": pending, "required_rebind": ["source_membership", "W_connected_groups",
            "station", "row", "alpha", "Query", "RS", "path_support"],
        "old_packet_immutable": True, "old_original_UV_retained": True,
        "source_only_delta": True, "physical_world_correspondence_issued": False,
        "independent_fit_weight_issued": False, "next_packet_built": False,
        "next_fit_consumed": False, "actual_source_commit": result}
    _new_json(out / "next_packet_rebind.json", rebind)
    return rebind
