"""Freeze accepted official-source material observations without legacy pixels.

The selection is by authenticated source addresses, not model projection or
unknown depth. User acceptance receipts supersede stale packet workflow flags.
Shared line representatives and missing source scales stay available as evidence
but never silently enter the ordinary independent point residual interface.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from .framewise_model import build_row, feature_world
from .validate_opening_frame_batch import verify_physical_target

SCHEMA = "rinkspace.official-static-surface-dataset.v1"
KINDS = {"ice_curve": "ice", "board_edge": "board", "ad_edge": "board",
         "ad_corner": "board", "ad_print": "board"}


def reference(path):
    path = Path(path).resolve()
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path), "sha256": digest}


def read_reference(ref):
    if reference(ref["path"]) != ref:
        raise ValueError("Source reference changed: " + ref["path"])
    return json.loads(Path(ref["path"]).read_text(encoding="utf8"))


def source_scales(segment):
    """Preserve original per-vertex scales; never fill missing values with one."""
    n = len(segment["uv"])
    for key in ("sigma_px", "localization_sigma_px", "endpoint_evidence_localization_sigma_px"):
        if key in segment and np.asarray(segment[key]).shape == (n,):
            return np.asarray(segment[key], float), key
    result = np.full(n, np.nan)
    bindings = segment.get("point_bindings", [])
    if len(bindings) == n:
        for i, binding in enumerate(bindings):
            result[i] = binding.get("localization_sigma_px", np.nan)
    endpoints = np.asarray(segment.get("endpoint_evidence_localization_sigma_px", []), float)
    if n > 2 and endpoints.shape == (2,):
        result[[0, -1]] = endpoints
    return result, "point_bindings.localization_sigma_px_or_missing"


def source_point_id(feature_id, segment_index, segment, vertex_index):
    if "point_ids" in segment:
        if len(segment["point_ids"]) != len(segment["uv"]):
            raise ValueError("Original opaque point identities differ from coordinate inventory")
        return segment["point_ids"][vertex_index]
    return f"{feature_id}/segment/{segment_index}/vertex/{vertex_index}"


def classify_segment(feature, segment, descriptor, mask, *, reviewed_mask_vertices=(), timer=None):
    uv = np.asarray(segment["uv"], np.float64)
    scope = KINDS.get(descriptor.get("kind"))
    reason = None
    if scope is None or feature.get("physical_surface", scope) != scope:
        reason = "outside_or_unconfirmed_model_scope"
    elif (segment.get("hidden_span_filled") is True
          or feature.get("projected_uv_used_as_observations") is True
          or segment.get("projected_uv_used_as_observations") is True):
        reason = "inferred_or_projected_pixel"
    origin = str(segment.get("measurement_origin", ""))
    shared = (segment.get("representative_points_share_line_estimate") is True
              or segment.get("representative_points_share_circle_family_estimate") is True
              or segment.get("inferred_from_current_source_readouts") is True
              or "intersection" in origin or descriptor.get("kind") == "ad_corner")
    measured = (segment.get("pixels_measured_in_this_frame") is True
                or origin in ("two_source_measured_straight_edge_intersections",
                              "current_source_robust_line_intersection"))
    if shared:
        measured = (measured or segment.get("source_evidence_measured_in_this_frame") is True
                    or bool(feature.get("source_straight_line_estimate"))
                    or bool(feature.get("source_line_fit")) or bool(feature.get("source_circle_fit")))
    if reason is None and not measured:
        reason = "no_current_source_measurement_evidence"
    sigma, scale_field = source_scales(segment)
    decisions = []
    for i, point in enumerate(uv):
        if reason:
            decisions.append(reason)
        elif not np.isfinite(point).all() or (point < 0).any() or (point >= [1920, 1080]).any():
            decisions.append("invalid_native_pixel")
        elif timer and timer.get("actual_native_rectangle_xyxy_inclusive") and _inside_timer(point, timer):
            decisions.append("broadcast_graphic")
        elif (mask[tuple(np.clip(np.rint(point).astype(int), [0, 0], [1919, 1079])[::-1])]
              and i not in reviewed_mask_vertices):
            decisions.append("foreground_mask_conflict")
        elif shared:
            decisions.append("shared_line_evidence")
        elif not np.isfinite(sigma[i]) or sigma[i] <= 0:
            decisions.append("missing_source_scale")
        else:
            decisions.append("fit")
    return scope, decisions, sigma, scale_field


def _inside_timer(point, timer):
    x0, y0, x1, y1 = timer["actual_native_rectangle_xyxy_inclusive"]
    halo = timer.get("edge_halo_px", 0)
    x, y = np.rint(point)
    return x0-halo <= x <= x1+halo and y0-halo <= y <= y1+halo


def build_dataset(model_path, primary_path, selection_path, *, excluded_candidate_batches=(), scope_overrides=None,
                  accepted_frame_range=None):
    model_ref, primary_ref = reference(model_path), reference(primary_path)
    model, primary = read_reference(model_ref), read_reference(primary_ref)
    source = primary["primary_source"]
    if (primary["processing_policy"] != "ISU_official_single_programme_native_50fps_PTS"
            or source["native_fps"] != "50/1" or source["time_base"] != "1/12800"):
        raise ValueError("Official native 50fps source required")
    video_ref = {"path": source["path"], "sha256": source["sha256"]}
    if reference(video_ref["path"]) != video_ref:
        raise ValueError("Official source video bytes changed")
    identity = read_reference(model["parent"]["state_identity"])
    checkpoint = model["parent"]["complete_state"]
    if reference(checkpoint["path"]) != checkpoint:
        raise ValueError("Parent complete state bytes changed")
    with np.load(checkpoint["path"], allow_pickle=False) as saved:
        q = saved["q"]
        if hashlib.sha256(q.tobytes()).hexdigest() != model["parent"]["q_sha256"]:
            raise ValueError("Parent q identity differs")
    if identity["q_sha256"] != model["parent"]["q_sha256"] or len(q) != identity["parameter_count"]:
        raise ValueError("Parent parameter layout differs")
    scene, registry, templates = [read_reference(model[k])
                                  for k in ("scene", "feature_registry", "material_templates")]
    overrides = read_reference(scope_overrides)["reviewed_points"] if scope_overrides else []
    applied_overrides = []
    frames, counts, scope_counts, target_counts = [], Counter(), Counter(), Counter()
    exclusion_details = []
    authorities = []
    for batch_ref in model["accepted_batch_datasets"]:
        batch = read_reference({k: batch_ref[k] for k in ("path", "sha256")})
        acceptance_ref = batch_ref["user_acceptance"]
        acceptance = read_reference(acceptance_ref)
        if acceptance.get("accepted") is not True or acceptance["source_time_base"] != source["time_base"]:
            raise ValueError("Explicit native-source batch acceptance required")
        approved = {r["index"]: r for r in acceptance["frames"]}
        authorities.append(acceptance_ref)
        for record in batch["frames"]:
            index = record["index"]
            if accepted_frame_range is not None and not accepted_frame_range[0] <= index <= accepted_frame_range[1]:
                continue
            if (record["packet"]["sha256"] != approved[index]["packet_sha256"]
                    or record["model_binding"]["sha256"] != approved[index]["binding_sha256"]):
                raise ValueError("Acceptance receipt no longer binds current observations")
            packet, binding = read_reference(record["packet"]), read_reference(record["model_binding"])
            native = packet["source"]
            if (native["source_video_sha256"] != source["sha256"]
                    or native["time_base"] != source["time_base"]
                    or native["actual_pts"] != native["native_pts"]
                    or binding["source_packet"] != record["packet"]
                    or binding["source_pts"] != native["native_pts"]
                    or binding["source_time_base"] != source["time_base"]
                    or binding["model_id"] != model["model_id"]
                    or packet.get("observation_coordinates_from_projection") is not False):
                raise ValueError("Official source/native clock/model identity differs")
            image = cv2.imread(native["image_path"])
            if (image is None or image.shape != (1080, 1920, 3)
                    or hashlib.sha256(image.tobytes()).hexdigest() != native["decoded_BGR_sha256"]):
                raise ValueError("Exact native source image differs")
            mask_ref = {"path": native["mask_path"], "sha256": native["mask_sha256"]}
            if reference(mask_ref["path"]) != mask_ref:
                raise ValueError("Current source person exclusion changed")
            mask = cv2.imread(native["mask_path"], cv2.IMREAD_GRAYSCALE)
            if mask is None or mask.shape != image.shape[:2]:
                raise ValueError("Current native mask missing")
            features = {f["id"]: f for f in packet["features"]}
            expected = {(f["id"], si, vi) for f in packet["features"]
                        for si, s in enumerate(f["segments"]) for vi in range(len(s["uv"]))}
            seen, seen_ids, observations = set(), set(), []
            for observation in binding["observations"]:
                fid, si = observation["source_feature_id"], observation["source_segment_index"]
                feature, segment = features[fid], features[fid]["segments"][si]
                descriptor = registry["targets"].get(observation["target_id"], {})
                reviewed_vertices = []
                for override in overrides:
                    if (override["native_pts"] != native["native_pts"]
                            or override["source_feature_id"] != fid or override["source_segment_index"] != si):
                        continue
                    vi = override["source_vertex_index"]
                    if (override["source_packet"] != record["packet"]
                            or override["source_video_sha256"] != source["sha256"]
                            or override["time_base"] != native["time_base"]
                            or override["decoded_BGR_sha256"] != native["decoded_BGR_sha256"]
                            or override["uv"] != segment["uv"][vi]
                            or override["source_point_id"] != source_point_id(fid, si, segment, vi)
                            or override["scope"] != KINDS.get(descriptor.get("kind"))
                            or override.get("original_source_image_actually_reviewed") is not True
                            or override.get("static_material_confirmed") is not True
                            or not override.get("review_note") or not override.get("reviewer")):
                        raise ValueError("Exact source material review override differs")
                    reviewed_vertices.append(vi)
                    if override not in applied_overrides:
                        applied_overrides.append(override)
                scope, decisions, sigma, scale_field = classify_segment(feature, segment, descriptor, mask != 0,
                    reviewed_mask_vertices=reviewed_vertices, timer=native.get("broadcast_timer_exclusion"))
                if scope:
                    feature_world(scene, descriptor, templates)
                selected = {"fit": [], "shared_line_evidence": [], "missing_source_scale": []}
                for vi, point_id in zip(observation["source_vertex_indices"], observation["source_point_ids"], strict=True):
                    verify_physical_target(feature, segment, vi, descriptor)
                    address = (fid, si, vi)
                    if point_id != source_point_id(fid, si, segment, vi) or address in seen or point_id in seen_ids:
                        raise ValueError("Source coordinate address duplicated or changed")
                    seen.add(address)
                    seen_ids.add(point_id)
                    decision = decisions[vi]
                    counts[decision] += 1
                    if decision in selected:
                        selected[decision].append(vi)
                        scope_counts[scope] += 1
                        target_counts[observation["target_id"]] += 1
                    else:
                        exclusion_details.append(dict(frame=index, native_pts=native["native_pts"],
                            source_point_id=point_id, uv=segment["uv"][vi], reason=decision,
                            original_user_acceptance_preserved=True))
                observations.append(dict(binding_observation_id=observation["id"],
                    source_feature_id=fid, source_segment_index=si, target_id=observation["target_id"],
                    scope=scope, point_correspondence=observation["point_correspondence"],
                    dependence_group=observation["dependence_group"], scale_field=scale_field,
                    selected_vertex_indices=selected))
            if seen != expected or len(seen) != packet["point_count"]:
                raise ValueError("Original source inventory not completely partitioned")
            counts["top_level_inferred_vertices_excluded"] += sum(len(s.get("uv", [])) for s in packet.get("inferred_segments", []))
            frames.append(dict(index=index, native_pts=native["native_pts"],
                source_time=native["source_time"], packet=record["packet"], binding=record["model_binding"],
                mask=mask_ref, source_image=reference(native["image_path"]),
                user_acceptance=acceptance_ref, original_points=len(seen), observations=observations))
    if len(applied_overrides) != len(overrides):
        raise ValueError("Material review override was not consumed exactly")
    if len({f["native_pts"] for f in frames}) != len(frames):
        raise ValueError("Repeated official exposure")
    if accepted_frame_range is not None and [f["index"] for f in frames] != list(range(accepted_frame_range[0], accepted_frame_range[1]+1)):
        raise ValueError("Accepted frame range has missing, repeated or reordered frames")
    if any(f["native_pts"] % source["native_pts_step"] for f in frames):
        raise ValueError("Observation is not on the official native exposure clock")
    pending = []
    for ref in excluded_candidate_batches:
        batch = read_reference(ref)
        pending.append(dict(reference=ref, frame_count=batch["frame_count"],
            points=batch["density"]["total"], reason="unreviewed_candidate_not_admitted",
            candidate_user_acceptance_preserved=batch["user_accepted"]))
    selection = dict(schema=SCHEMA, model=model_ref, primary_source=primary_ref,
        video=video_ref, source_clock=dict(time_base=source["time_base"], fps=source["native_fps"]),
        parent_q_sha256=identity["q_sha256"], parent_complete_state=checkpoint,
        authorities=authorities, counts=dict(counts), scope_counts=dict(scope_counts),
        target_counts=dict(target_counts), frames=frames, excluded_points=exclusion_details,
        excluded_candidate_batches=pending, historical_pixels_imported=0,
        legacy_objectives_enabled=False, unknown_depth_points_admitted=0,
        inferred_navigation_is_observation=False, shared_line_points_are_independent=False,
        source_coordinates_modified=False, original_files_modified=False,
        mask_conflict_source_review=scope_overrides, applied_mask_conflict_reviews=applied_overrides,
        independent_HELD_created=False, accuracy_accepted=False, solver_started=False)
    if accepted_frame_range is not None:
        selection["accepted_frame_range"] = list(accepted_frame_range)
    selection_path = Path(selection_path)
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    with selection_path.open("x", encoding="utf8") as stream:
        json.dump(selection, stream, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return reference(selection_path), selection


def iter_observations(dataset_reference, *, role="fit"):
    """Yield exact official UV, original evidence and named finite model target."""
    if role not in ("fit", "shared_line_evidence", "missing_source_scale", "all_scope"):
        raise ValueError("Explicit supported dataset role required")
    data = read_reference(dataset_reference)
    if data["schema"] != SCHEMA or data["legacy_objectives_enabled"]:
        raise ValueError("Only scope-clean official observations may be consumed")
    model = read_reference(data["model"])
    registry = read_reference(model["feature_registry"])
    # The full source-choice document also contains live review progress.
    # Admission froze the source video and native clock; later progress edits
    # do not change those original bytes or invalidate accepted prior frames.
    source = dict(sha256=data["video"]["sha256"], time_base=data["source_clock"]["time_base"])
    if data["source_clock"] != dict(time_base="1/12800", fps="50/1"):
        raise ValueError("Only the frozen official native 50fps clock may be consumed")
    for ref in (data["video"], data["parent_complete_state"], model["scene"], model["material_templates"]):
        if reference(ref["path"]) != ref:
            raise ValueError("Frozen source/model bytes changed")
    overrides = (read_reference(data["mask_conflict_source_review"])["reviewed_points"]
                 if data.get("mask_conflict_source_review") else [])
    if overrides != data.get("applied_mask_conflict_reviews", []):
        raise ValueError("Original material review overrides changed")
    acceptances = {ref["path"]: read_reference(ref) for ref in data["authorities"]}
    for frame in data["frames"]:
        packet = read_reference(frame["packet"])
        binding = read_reference(frame["binding"])
        acceptance = acceptances[frame["user_acceptance"]["path"]]
        accepted_frame = next(r for r in acceptance["frames"] if r["index"] == frame["index"])
        if (acceptance.get("accepted") is not True
                or accepted_frame["packet_sha256"] != frame["packet"]["sha256"]
                or accepted_frame["binding_sha256"] != frame["binding"]["sha256"]):
            raise ValueError("Explicit source acceptance no longer matches")
        for ref in (frame["mask"], frame["source_image"]):
            if reference(ref["path"]) != ref:
                raise ValueError("Original image or exclusion mask changed")
        mask = cv2.imread(frame["mask"]["path"], cv2.IMREAD_GRAYSCALE) != 0
        original_observations = {r["id"]: r for r in binding["observations"]}
        native = packet["source"]
        if (native["source_video_sha256"] != source["sha256"] or native["time_base"] != source["time_base"]
                or native["native_pts"] != frame["native_pts"] or native["actual_pts"] != frame["native_pts"]
                or binding["source_packet"] != frame["packet"] or binding["source_pts"] != frame["native_pts"]
                or binding["source_time_base"] != native["time_base"]):
            raise ValueError("Dataset source clock differs")
        features = {f["id"]: f for f in packet["features"]}
        for observation in frame["observations"]:
            if role == "all_scope":
                allowed = set(v for values in observation["selected_vertex_indices"].values() for v in values)
                original = original_observations[observation["binding_observation_id"]]
                indices = [v for v in original["source_vertex_indices"] if v in allowed]
            else:
                indices = observation["selected_vertex_indices"][role]
            if not indices:
                continue
            descriptor = registry["targets"][observation["target_id"]]
            if observation["scope"] != KINDS.get(descriptor["kind"]):
                raise ValueError("Dataset contains an out-of-scope target")
            feature = features[observation["source_feature_id"]]
            segment = feature["segments"][observation["source_segment_index"]]
            original = original_observations[observation["binding_observation_id"]]
            if (original["target_id"] != observation["target_id"]
                    or original["point_correspondence"] != observation["point_correspondence"]
                    or original["source_feature_id"] != observation["source_feature_id"]
                    or original["source_segment_index"] != observation["source_segment_index"]
                    or not set(indices).issubset(original["source_vertex_indices"])):
                raise ValueError("Selected observation differs from original physical binding")
            reviewed = [r["source_vertex_index"] for r in overrides
                if r["source_packet"] == frame["packet"] and r["native_pts"] == frame["native_pts"]
                and r["source_feature_id"] == observation["source_feature_id"]
                and r["source_segment_index"] == observation["source_segment_index"]]
            _, decisions, _, _ = classify_segment(feature, segment, descriptor, mask,
                reviewed_mask_vertices=reviewed, timer=native.get("broadcast_timer_exclusion"))
            for vi in indices:
                verify_physical_target(feature, segment, vi, descriptor)
                if (decisions[vi] not in ("fit", "shared_line_evidence", "missing_source_scale")
                        if role == "all_scope" else decisions[vi] != role):
                    raise ValueError("Selected point no longer passes physical/evidence admission")
            sigma, _ = source_scales(segment)
            uv = np.asarray(segment["uv"], float)[indices]
            sigma = sigma[indices]
            if role == "fit" and (not np.isfinite(sigma).all() or (sigma <= 0).any()):
                raise ValueError("Missing source scales cannot enter ordinary FIT")
            yield dict(observation, uv=uv, sigma=sigma, model_target=descriptor,
                original_vertex_indices=indices,
                native_pts=frame["native_pts"], actual_pts=frame["native_pts"],
                time_base=native["time_base"], source_video_sha256=native["source_video_sha256"],
                source_packet=frame["packet"], source_segment=segment, source_feature=feature,
                source_point_ids=[source_point_id(observation["source_feature_id"], observation["source_segment_index"], segment, vi) for vi in indices])


def iter_fit_rows(dataset_reference):
    """Use the existing residual builder with actual retained source scales."""
    data = read_reference(dataset_reference)
    model = read_reference(data["model"])
    scene, templates = read_reference(model["scene"]), read_reference(model["material_templates"])
    for observation in iter_observations(dataset_reference, role="fit"):
        row = build_row(scene, observation["model_target"], observation["uv"], observation["sigma"], templates,
            row_id=observation["binding_observation_id"], point_correspondence=observation["point_correspondence"])
        row.update({key: observation[key] for key in ("native_pts", "actual_pts", "time_base",
            "source_video_sha256", "source_packet", "source_point_ids", "dependence_group")})
        yield row
