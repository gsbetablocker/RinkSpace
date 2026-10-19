"""CPU posterior audit of committed native programme evidence.

Original fitting arrays are read from their authenticated existing packets once.
They never enter a visual reader, and expected IDs never become measured IDs.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from fractions import Fraction
import gzip
import hashlib
import json
from pathlib import Path
import pickle
import re
import time

import numpy as np

from rinkspace.vision_storage import hydrate_source_record_arrays
from rinkspace.vision_supervision import (active_source_supervision_mask,
    crosscheck_source_observation_groups, source_sample_key,
    source_sequence_coverage_audit, source_sequence_intervals)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def authenticated(item):
    path = Path(item["path"])
    if sha(path) != item["sha256"]:
        raise ValueError(f"Original reference packet changed: {path}")
    return path


def binding(registry, pts):
    return dict(source_sha256=registry["source_sha256"],
                time_base=registry["time_base"], source_pts=int(pts))


def original_array(value, n=None, *, finite=True):
    array = np.asarray(value)
    if array.shape == (0,):
        array = array.reshape(0, 2)
    if array.ndim != 2 or array.shape[1] != 2 or (finite and not np.isfinite(array).all()):
        raise ValueError("Original source coordinates require finite Nx2; no transpose inference")
    if n is not None and len(array) != n:
        raise ValueError("Original source support fields have different point counts")
    return array


def array_metadata(array):
    return dict(dtype=array.dtype.str, shape=list(array.shape),
        raw_bytes_sha256=hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest())


def old_identity_role(row):
    """Normalize explicit original target semantics, never source position truth."""
    ident = row.get("feature_id") or row.get("id")
    curve = row.get("curve", {})
    if curve:
        if curve.get("advertisement_number", curve.get("advertisement_id")) is not None:
            ident = f"Ad{curve.get('advertisement_number', curve.get('advertisement_id'))}"
        if curve.get("kind") == "house_ring":
            ident = f"{curve['side']}{curve['row']}:{curve['ring']}"
        role = curve.get("edge") or row.get("role") or curve.get("kind")
        return ident, role
    kind = row.get("world_scene_binding", {}).get("kind")
    if ident in ("far_contact", "ice_board_contact") or kind == "finite_full_ice_board_contact_perimeter":
        return "ice_board_contact", "ice_board_contact"
    if ident in ("far_top", "board_top", "board_top_near_visible", "reviewed_lower_board_top") or kind == "finite_full_board_top_perimeter":
        return "board_top", "board_top_edge"
    match = re.fullmatch(r"([FR][1-4])_outer(?:_.*)?", str(ident))
    if match:
        return match[1] + ":outer", "outer_ring_outside_boundary"
    match = re.fullmatch(r"ad(\d+):(.+)", str(ident), re.I)
    if match:
        return f"Ad{match[1]}", match[2]
    return ident, row.get("physical_role") or row.get("geometry_role") or kind


def make_group(registry, row, locator, group_id, authority, split, *, active=True):
    uv = original_array(row["uv"])
    feature, role = old_identity_role(row)
    group = dict(source_binding=binding(registry, row["pts"]), group_id=group_id,
        feature_id=feature, geometry_role=role, authority_role=authority,
        source_pointer=locator, uv=uv, original_UV=array_metadata(uv),
        identity_as_stored=row.get("feature_id", row.get("id")),
        geometry_role_as_stored=row.get("physical_role", row.get("role")),
        curve_as_stored=row.get("curve"), original_split=split,
        active_reference=active, original_reference_is_not_automatically_pixel_truth=True,
        original_source_metadata={k: row[k] for k in (
            "origin", "method", "physical_ids", "source_raw_frame", "source_raw_image",
            "source_file_sha256", "physical_identity_status", "source_identity_authority",
            "source_review_authority", "world_scene_binding", "sigma_px") if k in row})
    # Preserve every original numerical support. Only paired valid normals can
    # participate in the interval comparison; never infer missing directions.
    for name in ("sigma", "normal", "normal_interval", "source_normal_uv", "normal_interval_px"):
        if name in row:
            group["original_" + name] = row[name]
    normal = row.get("normal", row.get("source_normal_uv"))
    interval = row.get("normal_interval", row.get("normal_interval_px"))
    if normal is not None and interval is not None:
        normal, interval = original_array(normal, len(uv)), original_array(interval, len(uv))
        if np.allclose(np.linalg.norm(normal, axis=1), 1., atol=1e-5, rtol=0) and np.all(interval[:, 0] <= interval[:, 1]):
            group.update(normal=normal, normal_interval=interval)
        else:
            group["original_normal_interval_comparison_unavailable"] = "original support is not a valid unit-normal pair; retained unchanged"
    return group


def point_role_groups(group):
    """Keep per-point printed corners/material identities distinct."""
    point_ids = group.get("original_point_physical_ids")
    roles = group.get("geometry_role")
    if point_ids is None and not isinstance(roles, list):
        return [group]
    if point_ids is not None and len(point_ids) != len(group["uv"]):
        raise ValueError("Original per-point physical identities have another UV domain")
    if isinstance(roles, list) and len(roles) != len(group["uv"]):
        raise ValueError("Original material roles have another UV domain")
    result = []
    for index in range(len(group["uv"])):
        row = dict(group, group_id=f"{group['group_id']}:point{index}",
                   uv=group["uv"][index:index + 1], original_point_indices=[index])
        if point_ids is not None:
            identity = point_ids[index]
            match = re.fullmatch(r"ad(\d+):(.+)", identity, re.I)
            row.update(feature_id=f"Ad{match[1]}" if match else identity,
                       geometry_role=f"corner:{match[2]}" if match else None,
                       original_point_physical_id=identity)
        elif isinstance(roles, list):
            row["geometry_role"] = roles[index]
        for field in ("normal", "normal_interval"):
            if field in group:
                row[field] = group[field][index:index + 1]
        result.append(row)
    return result


def load_references(registry):
    """Load unchanged typed UV once; return exact-native groups and retirements."""
    prefix = registry["original_absolute_packet"]
    raw = pickle.loads(authenticated(prefix).read_bytes())
    by_pts, counts = defaultdict(list), Counter()
    retired = set(registry["retired_absolute_record_indices"])
    for index, item in enumerate(raw["programme_absolute_rows"]):
        row = item["source_row"]
        pointer = f"{prefix['path']}#/programme_absolute_rows/{index}/source_row"
        authority = row.get("authority")
        explicit_human = (row.get("human_authority") is True or row.get("user_confirmed") is True or
            (isinstance(authority, dict) and authority.get("human_authority") is True) or
            row.get("state") == "user_confirmed_visual_supervision")
        group = make_group(registry, row, pointer, f"original_absolute:{index}",
            "direct_user_confirmed_native_UV" if explicit_human else "historical_assistant_source_measurement",
            item["collection_role"], active=index not in retired)
        group["actor_block_as_stored"] = item["block"]
        group["original_actual_exposure_pts"] = item["group_actual_pts"]
        # Known corners retain independent physical point identities; they are
        # never treated as a complete advertisement frame or rectangle.
        point_ids = row.get("physical_ids")
        if point_ids is not None and len(point_ids) == len(group["uv"]):
            group["original_point_physical_ids"] = point_ids
        by_pts[int(row["pts"])].extend(point_role_groups(group))
        counts[item["collection_role"]] += 1
    for packet in registry["additional_absolute_packets"]:
        content = read(authenticated(packet))
        if content.get("source_video_sha256", content.get("source_sha256")) != registry["source_sha256"]:
            raise ValueError("Additional reference belongs to another original video")
        for index, row in enumerate(content["rows"]):
            group = make_group(registry, row, f"{packet['path']}#/rows/{index}",
                f"{packet['dataset_key']}:{index}", "assistant_actual_native_source_review",
                row.get("split"))
            group["actual_input_manifest_selected_FIT"] = row.get("split") == "fit"
            group["old_packet_solver_admitted_as_stored"] = row.get("solver_admitted")
            by_pts[int(row["pts"])].extend(point_role_groups(group))
    for packet in registry.get("human_supervision_packets", []):
        content = read(authenticated(packet))
        for sample_index, sample in enumerate(content.get("samples", [])):
            key = source_sample_key(sample)
            if key[:2] != (registry["source_sha256"], str(Fraction(registry["time_base"]))) or not registry["start_pts"] <= key[2] < registry["end_pts_exclusive"]:
                continue
            for index, annotation in enumerate(sample.get("annotations", [])):
                if annotation.get("state") != "user_confirmed_visual_supervision":
                    continue
                active = np.asarray(active_source_supervision_mask(annotation), dtype=bool)
                visible = np.asarray(annotation.get("source_visible_mask", np.ones(len(active), dtype=bool)))
                if visible.dtype != np.bool_ or visible.shape != active.shape:
                    raise ValueError("Human visibility field has an invalid original domain")
                uv = original_array(annotation["uv"])
                row = dict(annotation, pts=key[2])
                group = make_group(registry, row,
                    f"{packet['path']}#/samples/{sample_index}/annotations/{index}",
                    annotation["group_id"], "direct_user_confirmed_native_UV", "training_reference")
                group.update(original_active_mask=active, original_visible_mask=visible,
                    original_source_point_indices=np.flatnonzero(active & visible),
                    uv=uv[active & visible], feature_id=annotation.get("feature_id"),
                    geometry_role="outer_ring_outside_boundary" if annotation.get("feature_id", "").endswith(":outer") else annotation.get("target_role"))
                # interval_px is an original scalar/per-point support; no
                # precision or normal is inferred for missing original fields.
                group["original_interval_px"] = annotation.get("interval_px")
                by_pts[key[2]].append(group)
    return dict(by_pts), dict(counts)


def measured_groups(row):
    groups, diagnostic = [], []
    for index, reading in enumerate(row.get("named_readings", [])):
        uv = original_array(reading.get("actual_source_uv", [])) if reading.get("actual_source_uv") else np.empty((0, 2))
        identity = reading.get("physical_id")
        groups.append(dict(source_binding=row["source_binding"],
            feature_id=identity, geometry_role=reading.get("physical_role"), uv=uv,
            actual_RGB_measurement=reading.get("hint_pixels_used_as_output") is False,
            model_or_world_projection_used_as_UV=reading.get("hint_pixels_used_as_output") is not False))
        diagnostic.append(dict(reader_index=index, expected_id=reading.get("expected_id"),
            physical_id=identity, identity_authority=reading.get("identity_authority"),
            geometry_role=reading.get("physical_role"), actual_point_count=len(uv),
            missing_count=reading.get("missing_search_sample_count"),
            flags=reading.get("source_flags", {}), qualified_for_fitting=False,
            qualification_pending="actual RGB role and identity verification; expected-ID is not confirmed"))
    return groups, diagnostic


def audit_frame(row, references, native_rgb=None):
    """Inspect all published source groups and providers without new inference."""
    key = source_sample_key(row["source_binding"])
    measured, names = measured_groups(row)
    active = [reference for reference in references if reference["active_reference"]]
    comparison = crosscheck_source_observation_groups(active, measured,
                                                     source_binding=row["source_binding"])
    failures, materials = [], []
    if native_rgb is not None and native_rgb.get("RGB_sha256", native_rgb.get("rgb_sha256")) != row.get("RGB_sha256"):
        failures.append("published_RGB_SHA_differs_from_independent_original_decode")
    provider = row.get("source_instance_provider_evidence") or {}
    provider_binding = provider.get("source_binding")
    if provider_binding is not None and source_sample_key(provider_binding) != key:
        failures.append("instance_provider_uses_other_native_source_frame")
    for index, reading in enumerate((row.get("source_board_foot_material_reading") or {}).get("readings", [])):
        candidate = original_array(reading["candidate_source_uv"], finite=False)
        usable = np.asarray(reading["candidate_usable"])
        primary = original_array(reading["primary_source_uv"])
        selected = np.asarray(reading["primary_role_samples"])
        if usable.dtype != np.bool_ or selected.dtype != np.bool_ or usable.shape != (len(candidate),) or selected.shape != usable.shape:
            failures.append(f"material_{index}_candidate_selector_domain_invalid")
        elif not np.array_equal(candidate[selected], primary, equal_nan=True):
            failures.append(f"material_{index}_primary_is_not_existing_actual_candidate_subset")
        if reading.get("mask_or_navigation_UV_used_as_output") is not False:
            failures.append(f"material_{index}_navigation_UV_output_not_excluded")
        flags = {name: int(np.asarray(value).sum()) for name, value in reading.get("flags", {}).items()
                 if isinstance(value, np.ndarray) and value.dtype == np.bool_}
        materials.append(dict(physical_role=reading.get("physical_role"), raw_candidate_count=len(candidate),
            nonfinite_raw_candidate_count=int((~np.isfinite(candidate).all(axis=1)).sum()),
            usable_count=int(usable.sum()), primary_count=len(primary),
            unresolved_count=len(reading.get("unresolved_source_uv", [])),
            all_optical_count=len(reading.get("all_observed_optical_source_uv", [])),
            source_conflict_counts=flags, primary_is_pixel_truth=False))
    retired = [r["group_id"] for r in references if not r["active_reference"]]
    statuses = Counter(item["status"] for item in comparison["comparisons"])
    return dict(schema="rinkspace.posterior-native-programme-source-check.v1",
        source_binding=row["source_binding"], RGB_sha256=row.get("RGB_sha256"),
        model_sha256=row.get("model_sha256"), source_seed_injected=row.get("source_seed_injected"),
        memory_update=row.get("memory_update"), new_observation=row.get("new_observation"),
        instance_source_shot_reset_requested=row.get("instance_source_shot_reset_requested"),
        source_time=row.get("source_time"), source_point_component_count=len(row.get("source_components", [])),
        source_role_status=row.get("source_role_reading_status"),
        named_source_readings=names, material_readings=materials,
        source_issues=row.get("issues", []), source_edge_class_overlap=row.get("source_edge_class_overlap"),
        primary_top=(row.get("source_board_top_role_assignment") or {}).get("primary_source_top", {}).get("source_pixel_count", 0),
        retired_original_reference_groups=retired,
        original_reference_group_count=len(references), source_crosscheck=comparison,
        original_reference_status_counts=dict(statuses),
        expected_named_ID_does_not_establish_physical_identity=True,
        failures=failures, all_frame_counts_are_accuracy=False,
        fitting_qualified_measurements=[], qualified_is_not_inferred_from_old_expected_ID=True)


def audit_compact_application_frame(row, references):
    """Check final application measurements without requiring a raw candidate pool.

    The immutable producer row remains the coordinate resource. This posterior
    record saves its group pointer, array dtype/hash and decisions, never copies
    source profiles, masks or another full set of measurement coordinates.
    """
    if row.get("schema") != "rinkspace.application-source-observations.v1":
        raise ValueError("Expected the explicit compact application source schema")
    key = source_sample_key(row["source_binding"])
    forbidden = {"source_components", "source_board_foot_material_reading", "source_board_top_role_assignment",
                 "profiles", "masks", "raw_candidates"}.intersection(row)
    if forbidden:
        raise ValueError(f"Compact application record contains a full raw resource: {sorted(forbidden)}")
    measured, diagnostics, failures = [], [], []
    for index, group in enumerate(row.get("measurement_groups", [])):
        current = group.get("source_binding", row["source_binding"])
        if source_sample_key(current) != key:
            raise ValueError("Compact measurement belongs to another native source frame")
        uv = original_array(group["actual_source_uv"])
        source_indices = group.get("source_indices")
        source_indices_missing = source_indices is None or (len(source_indices) == 0 and len(uv) > 0)
        if source_indices is not None:
            source_indices = np.asarray(source_indices)
            if source_indices.ndim != 1 or (len(source_indices) and (len(source_indices) != len(uv) or source_indices.dtype.kind not in "iu" or (source_indices < 0).any())):
                raise ValueError("Compact original point indices differ from the actual UV domain")
        exact_rgb = group.get("RGB_recheck_executed") is True
        authority = group.get("identity_authority")
        parent_identity = group.get("physical_id")
        identity = (group.get("physical_curve_id") or parent_identity) if parent_identity and authority else None
        role = group.get("geometry_role", group.get("physical_role"))
        role_unresolved = not role or any(token in str(role) for token in ("unresolved", "conflict", "hypothesis"))
        if parent_identity and not authority:
            failures.append(f"group_{index}_physical_identity_has_no_source_authority")
        if exact_rgb:
            measured.append(dict(source_binding=current, feature_id=identity if authority else None,
                geometry_role=role, uv=uv, actual_RGB_measurement=True,
                model_or_world_projection_used_as_UV=False))
        uncertainty = group.get("observed_uncertainty")
        uncertainty_missing = uncertainty is None or uncertainty.get("localization_interval_not_available") is True
        source_flags = group.get("source_flag_counts")
        if source_flags is None:
            source_flags = {name: int(value.sum()) if isinstance(value, np.ndarray) and value.dtype == np.bool_ else value
                for name, value in (group.get("source_flags") or {}).items()}
        diagnostics.append(dict(group_index=index,
            source_measurement_pointer=f"/measurement_groups/{index}",
            group_id=group.get("group_id", group.get("source_group_id")), expected_id=group.get("expected_id"),
            physical_id=group.get("physical_id"), physical_curve_id=group.get("physical_curve_id"),
            identity_authority_present=bool(authority),
            identity_authority_source_pointer=f"/measurement_groups/{index}/identity_authority" if authority else None,
            identity_authority_kind=authority.get("kind") if isinstance(authority, dict) else None,
            curve_label_not_confirmed_without_parent_identity=bool(group.get("physical_curve_id")) and not bool(parent_identity and authority),
            geometry_role=role, role_unresolved_or_conflicting=role_unresolved,
            actual_point_count=len(uv), original_array=array_metadata(uv),
            original_dtype_known=isinstance(group["actual_source_uv"], np.ndarray),
            legacy_inline_array_dtype_not_claimed_restored=not isinstance(group["actual_source_uv"], np.ndarray),
            RGB_recheck_executed=exact_rgb, measurement_method=group.get("measurement_method"),
            measurement_disposition=group.get("disposition"),
            original_source_indices_missing_not_fabricated=source_indices_missing,
            uncertainty_source_pointer=f"/measurement_groups/{index}/observed_uncertainty" if uncertainty is not None else None,
            uncertainty_missing_not_fabricated=uncertainty_missing,
            source_flags=source_flags, source_flags_pointer=f"/measurement_groups/{index}/source_flags", missing=group.get("missing"),
            source_coordinates_copied_to_posterior_record=False,
            named_fitting_qualification_pending=not (identity and authority and not role_unresolved and exact_rgb and not uncertainty_missing)))
    active = [item for item in references if item["active_reference"]]
    comparison = crosscheck_source_observation_groups(active, measured, source_binding=row["source_binding"])
    return dict(schema="rinkspace.posterior-compact-application-source-check.v1",
        source_binding=row["source_binding"], RGB_sha256=row.get("RGB_sha256"),
        model_sha256=row.get("model_sha256"), source_seed_injected=row.get("source_seed_injected"),
        memory_update=row.get("memory_update"), new_observation=row.get("new_observation"),
        instance_source_shot_reset_requested=row.get("instance_source_shot_reset_requested"),
        application_measurement_groups=diagnostics,
        final_actual_RGB_point_count=sum(len(item["uv"]) for item in measured),
        propagated_without_RGB_recheck_point_count=sum(item["actual_point_count"] for item in diagnostics if not item["RGB_recheck_executed"]),
        source_crosscheck=comparison, source_issues=row.get("issues", []),
        original_reference_status_counts=dict(Counter(item["status"] for item in comparison["comparisons"])),
        retired_original_reference_groups=[item["group_id"] for item in references if not item["active_reference"]],
        failures=failures, all_frame_counts_are_accuracy=False,
        expected_named_ID_does_not_establish_physical_identity=True,
        missing_or_flow_only_points_not_promoted_to_source_RGB_measurements=True,
        original_final_measurement_arrays_not_modified=True)


def aggregate_application_check(checked, entry):
    """One small native-frame QA row; detailed arrays stay with the producer."""
    groups = checked["application_measurement_groups"]
    identities = Counter(group["physical_id"] for group in groups if group["physical_id"])
    roles = Counter(group["geometry_role"] or "unresolved" for group in groups)
    return {key: value for key, value in checked.items() if key != "application_measurement_groups"} | dict(
        measurement_group_count=len(groups), distinct_confirmed_physical_instance_count=len(identities),
        confirmed_instance_fragment_counts=dict(identities), geometry_role_group_counts=dict(roles),
        uncertainty_missing_group_count=sum(group["uncertainty_missing_not_fabricated"] for group in groups),
        unresolved_or_conflicting_role_group_count=sum(group["role_unresolved_or_conflicting"] for group in groups),
        fitting_qualification_pending_group_count=sum(group["named_fitting_qualification_pending"] for group in groups),
        source_measurement_record=dict(committed_frame_ordinal=entry["frame_ordinal"],
            record_member_offset=entry["record_member_offset"],
            record_member_sha256=entry["record_member_sha256"],
            uncompressed_UTF8_sha256=entry["uncompressed_UTF8_sha256"]),
        posterior_duplicates_no_final_UV_or_group_arrays=True,
        original_array_integrity_checked_by_authenticated_source_storage=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--run-directory", required=True)
    parser.add_argument("--follow", action="store_true")
    parser.add_argument("--source-seed-policy", help="Explicit authenticated assisted-calibration policy; default remains independent")
    args = parser.parse_args()
    registry = read(args.registry)
    manifest = read(authenticated(registry["programme_manifest"]))
    source_sequence_intervals(manifest)
    seed_policy = read(args.source_seed_policy) if args.source_seed_policy else None
    if seed_policy is not None:
        if sha(seed_policy["seed_packet_path"]) != seed_policy["seed_packet_sha256"]:
            raise ValueError("Assisted source seed packet differs from its authenticated policy")
    references, _ = load_references(registry)
    run = Path(args.run_directory)
    progress = run / "readings_committed_frames.jsonl"
    output = run / "posterior_all_native_source_checks.jsonl.gz"
    completed = 0
    rows_for_coverage = []
    totals, crosscheck_totals = Counter(), Counter()
    last_pts = None

    def write_progress():
        payload = dict(schema="rinkspace.posterior-source-check-progress.v1",
            requested_native_frame_count=manifest["frame_count"], checked_native_frame_count=completed,
            last_checked_source_pts=last_pts, original_reference_context_count=len(references),
            failure_counts=dict(totals), original_reference_status_counts=dict(crosscheck_totals),
            all_requested_native_frames_checked=completed == manifest["frame_count"],
            all_frame_counts_are_accuracy=False, reliable_fitting_qualification_pending=True,
            checker_sha256=sha(__file__), registry_sha256=sha(args.registry))
        temporary = run / "posterior_source_check_progress.json.tmp"
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(run / "posterior_source_check_progress.json")
    # Append-only check output is a distinct posterior artifact. Resume must
    # use the preserved original output, never overwrite a partial audit.
    if output.exists():
        raise FileExistsError("Existing posterior evidence must not be overwritten")
    from rinkspace.vision_review import load_committed_source_record
    with gzip.open(output, "xt", encoding="utf-8") as target:
        while True:
            if progress.exists():
                with progress.open(encoding="utf-8") as index:
                    entries = [json.loads(line) for line in index if line.endswith("\n")]
                for entry in entries[completed:]:
                    row = load_committed_source_record(run / "readings.jsonl.gz", entry)
                    pts = row["source_binding"]["source_pts"]
                    checked = (audit_compact_application_frame(row, references.get(pts, []))
                        if row.get("schema") == "rinkspace.application-source-observations.v1"
                        else audit_frame(row, references.get(pts, [])))
                    if row.get("schema") == "rinkspace.application-source-observations.v1":
                        checked = aggregate_application_check(checked, entry)
                    target.write(json.dumps(checked, ensure_ascii=False, allow_nan=False) + "\n")
                    target.flush()
                    totals.update(checked["failures"])
                    crosscheck_totals.update(checked["original_reference_status_counts"])
                    rows_for_coverage.append({k: row[k] for k in (
                        "source_binding", "RGB_sha256", "model_sha256", "memory_update", "new_observation",
                        "source_seed_injected", "instance_source_shot_reset_requested", "issues") if k in row})
                    completed += 1
                    last_pts = pts
                    if completed % 100 == 0:
                        print(json.dumps(dict(checked_native_frames=completed, last_source_pts=pts,
                                              failure_counts=dict(totals))), flush=True)
                    del row, checked
                write_progress()
            summary_path = run / "review.json"
            summary = read(summary_path) if summary_path.exists() else {}
            finished = summary.get("status") in ("source_range_complete", "failed", "source_range_failed")
            if completed >= manifest["frame_count"] or finished or not args.follow:
                break
            time.sleep(5)
    proof = read(authenticated(registry["exposure_proof"]))
    coverage = source_sequence_coverage_audit(manifest, summary, observations=rows_for_coverage,
        exposure_groups=proof["groups"], installation_aliases=registry["installation_aliases"],
        source_seed_policy=seed_policy)
    result = dict(schema="rinkspace.posterior-complete-programme-source-audit.v1",
        registry_sha256=sha(args.registry), checker_sha256=sha(__file__),
        checked_native_frame_count=completed, all_requested_native_frames_checked=completed == manifest["frame_count"],
        original_reference_context_count=len(references), source_coverage=coverage,
        source_quality_failure_counts=dict(totals), checks_path=str(output), checks_sha256=sha(output),
        original_reference_status_counts=dict(crosscheck_totals),
        all_frame_counts_are_accuracy=False, fitting_or_new_training_started=False,
        physical_RGB_conflict_adjudication_and_reliable_source_qualification_pending=True)
    (run / "posterior_all_native_source_check_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
