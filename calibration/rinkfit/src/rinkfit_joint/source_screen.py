"""Transparent native-image support screening; never a fitting/admission rule.

Measurements stay unchanged.  Source support is different from correct material
identity, independent geometry, and camera-fit accuracy.  Thresholds below are
initial screening choices, not empirically calibrated false-positive rates.
"""
from __future__ import annotations

import gzip
import hashlib
import io
import itertools
import json
import math
from collections import Counter, defaultdict
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class ScreenConfig:
    smoothing_sigma_px: float = 0.8
    profile_base_radius_px: float = 6.0
    profile_max_radius_px: float = 32.0
    profile_step_px: float = 1.0
    minimum_channel_contrast_8bit: float = 4.0
    minimum_gradient_8bit_per_px: float = 0.75
    nearest_raw_warning_px: float = 6.0
    raw_warning_sigma_multiplier: float = 2.0
    corner_patch_radius_px: int = 3
    corner_second_direction_ratio: float = 0.025
    maximum_tangent_neighbor_gap_px: float = 64.0
    flank_minimum_offset_px: float = 1.0
    flank_local_radius_px: float = 2.0


def _load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _read_jsonl(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


def _frame_key(row):
    return row["shot_id"], int(row["native_pts"])


def _by_frame(path):
    for key, values in itertools.groupby(_read_jsonl(path), key=_frame_key):
        yield key, list(values)


def _writer(stack, path):
    raw = stack.enter_context(Path(path).open("wb"))
    zipped = stack.enter_context(gzip.GzipFile(filename="", fileobj=raw, mode="wb",
                                             compresslevel=3, mtime=0))
    return stack.enter_context(io.TextIOWrapper(zipped, encoding="utf-8"))


def _emit(handle, row):
    handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"),
                            allow_nan=False) + "\n")


def _points(value):
    if value is None:
        return np.empty((0, 2), dtype=np.float64)
    try:
        a = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return np.empty((0, 2), dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 2:
        return np.empty((0, 2), dtype=np.float64)
    return a[np.isfinite(a).all(axis=1)]


def _channels(rgb, config):
    rgb = np.asarray(rgb, dtype=np.float32)
    # Luminance plus two chromatic contrasts: a red/blue interface can have
    # little luminance gradient.  All channels retain explicit 8-bit units.
    channels = np.stack((0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2],
                         rgb[..., 0] - rgb[..., 1],
                         rgb[..., 2] - (rgb[..., 0] + rgb[..., 1]) * 0.5))
    channels = gaussian_filter(channels, sigma=(0, config.smoothing_sigma_px,
                                                config.smoothing_sigma_px))
    gy = np.gradient(channels, axis=1)
    gx = np.gradient(channels, axis=2)
    return channels, gx, gy


def _sample(channels, uv):
    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    return np.stack([map_coordinates(c, [uv[:, 1], uv[:, 0]], order=1,
                                     mode="nearest", prefilter=False)
                     for c in channels], axis=1)


def _circle_readouts(feature, target):
    fit = feature.get("source_circle_fit") or {}
    reads = fit.get("direct_readouts", [])
    if not isinstance(reads, list):
        return []
    role = target.rsplit(":", 1)[-1] if target and ":" in target else None
    if role and any(r.get("role") == role for r in reads):
        reads = [r for r in reads if r.get("role") == role]
    return [r for r in reads if _points([r.get("uv")]).shape == (1, 2)]


def _point_evidence(group, count):
    raw = group.get("point_source_evidence")
    if isinstance(raw, list) and len(raw) == count:
        return raw
    return [{} for _ in range(count)]


def _source_support(rows, group, feature, config):
    """Return original-read distances, candidate normals and real flank UVs."""
    uv = np.asarray([r["source_uv"] for r in rows], dtype=np.float64)
    raw = _points(group.get("source_native_readout_uv"))
    readouts = _circle_readouts(feature, rows[0].get("material_id"))
    raw_kind = "segment_source_native_readout_uv" if len(raw) else None
    source_evidence = None
    if not len(raw) and readouts:
        raw = np.asarray([r["uv"] for r in readouts], dtype=np.float64)
        source_evidence = [r.get("source_evidence", {}) for r in readouts]
        raw_kind = "feature_source_circle_fit_direct_readouts"
    normals = np.full((len(rows), 2), np.nan, dtype=np.float64)
    flank_uv = [None] * len(rows)
    normal_kind = [None] * len(rows)
    raw_dist = [None] * len(rows)
    if len(raw):
        distances, indices = cKDTree(raw).query(uv, k=1)
        raw_dist = distances.tolist()
        closed = bool(group.get("source_flags", {}).get("closed"))
        for j, index in enumerate(indices):
            if rows[j].get("point_correspondence"):
                continue
            evidence = source_evidence[int(index)] if source_evidence else {}
            normal = np.asarray(evidence.get("profile_normal", [np.nan, np.nan]), dtype=float)
            sides = _points(evidence.get("actual_source_sides_uv"))
            if len(sides) == 2:
                flank_uv[j] = sides.tolist()
            if normal.shape == (2,) and np.isfinite(normal).all() and np.linalg.norm(normal) > 1e-8:
                normals[j] = normal / np.linalg.norm(normal)
                normal_kind[j] = "original_source_profile_normal"
                continue
            if len(raw) >= 2:
                before = (int(index)-1) % len(raw) if closed else max(0, int(index)-1)
                after = (int(index)+1) % len(raw) if closed else min(len(raw)-1, int(index)+1)
                delta = raw[after] - raw[before]
                if (np.linalg.norm(delta) > 1e-8 and
                        max(np.linalg.norm(raw[int(index)]-raw[before]),
                            np.linalg.norm(raw[after]-raw[int(index)])) <= config.maximum_tangent_neighbor_gap_px):
                    normals[j] = [-delta[1], delta[0]] / np.linalg.norm(delta)
                    normal_kind[j] = "retained_source_polyline_tangent_candidate"
    # Individual representative evidence can contain direct flank/profile data
    # even when a shared curve also exists.  Preserve the direct record first.
    evidence = _point_evidence(group, len(rows))
    for j, e in enumerate(evidence):
        if rows[j].get("point_correspondence") or not isinstance(e, dict):
            continue
        normal = np.asarray(e.get("profile_normal", [np.nan, np.nan]), dtype=float)
        sides = _points(e.get("actual_source_sides_uv"))
        if len(sides) == 2:
            flank_uv[j] = sides.tolist()
        if normal.shape == (2,) and np.isfinite(normal).all() and np.linalg.norm(normal) > 1e-8:
            normals[j] = normal / np.linalg.norm(normal)
            normal_kind[j] = "representative_original_profile_normal"
    # Tangents are local to the original segment.  Never bridge its occlusions
    # or unrelated fragments; single entity corners never get a fake normal.
    closed = bool(group.get("source_flags", {}).get("closed"))
    for j, row in enumerate(rows):
        if row.get("point_correspondence") or np.isfinite(normals[j]).all() or len(uv) < 2:
            continue
        before = (j-1) % len(uv) if closed else max(0, j-1)
        after = (j+1) % len(uv) if closed else min(len(uv)-1, j+1)
        delta = uv[after]-uv[before]
        if np.linalg.norm(delta) > 1e-8:
            normals[j] = [-delta[1], delta[0]] / np.linalg.norm(delta)
            normal_kind[j] = "representative_segment_tangent_candidate"
    return raw_kind, raw_dist, normals, normal_kind, flank_uv


def _photometric(rows, normals, normal_kind, flank_uv, channels, gx, gy, config):
    uv = np.asarray([r["source_uv"] for r in rows], dtype=np.float64)
    patch_delta = np.array([(x, y) for y in range(-config.corner_patch_radius_px,
                                                 config.corner_patch_radius_px+1)
                           for x in range(-config.corner_patch_radius_px,
                                          config.corner_patch_radius_px+1)], dtype=float)
    patch_uv = uv[:, None, :] + patch_delta[None, :, :]
    patch = _sample(channels, patch_uv.reshape(-1, 2)).reshape(len(rows), len(patch_delta), 3)
    patch_gx = _sample(gx, patch_uv.reshape(-1, 2)).reshape(len(rows), len(patch_delta), 3)
    patch_gy = _sample(gy, patch_uv.reshape(-1, 2)).reshape(len(rows), len(patch_delta), 3)
    contrast_by_channel = np.percentile(patch, 90, axis=1) - np.percentile(patch, 10, axis=1)
    patch_contrast = contrast_by_channel.max(axis=1)
    gradient_norm = np.hypot(patch_gx, patch_gy)
    patch_gradient = gradient_norm.max(axis=(1, 2))
    strongest_channel = gradient_norm.max(axis=1).argmax(axis=1)
    ratios = []
    for j in range(len(rows)):
        channel = strongest_channel[j]
        xx, yy = patch_gx[j, :, channel], patch_gy[j, :, channel]
        eig = np.linalg.eigvalsh(np.array([[np.mean(xx*xx), np.mean(xx*yy)],
                                         [np.mean(xx*yy), np.mean(yy*yy)]]))
        ratios.append(float(max(eig[0], 0) / max(eig[1], 1e-12)))
        if not rows[j].get("point_correspondence") and not np.isfinite(normals[j]).all():
            k = gradient_norm[j, :, channel].argmax()
            normal = np.array([xx[k], yy[k]])
            if np.linalg.norm(normal) >= config.minimum_gradient_8bit_per_px:
                normals[j] = normal / np.linalg.norm(normal)
                normal_kind[j] = "image_neighborhood_gradient_candidate_only"
    result = []
    height, width = channels.shape[1:]
    for j, row in enumerate(rows):
        corner = bool(row.get("point_correspondence"))
        paint_center = row.get("measurement_operator") == "paint_center"
        value = {
            "patch_contrast_8bit": float(patch_contrast[j]),
            "patch_max_gradient_8bit_per_px": float(patch_gradient[j]),
            "corner_second_direction_ratio": ratios[j] if corner else None,
            "normal_candidate": None if corner or not np.isfinite(normals[j]).all() else normals[j].tolist(),
            "normal_candidate_source": None if corner else normal_kind[j],
            "support_mode": "corner_neighborhood" if corner else "local_interface_profile",
            "profile_contrast_8bit": None, "profile_peak_gradient_8bit_per_px": None,
            "profile_peak_offset_px": None, "profile_radius_px": None,
            "profile_outside_image": False, "actual_flank_uv": flank_uv[j] if paint_center else None,
            "left_flank_gradient_8bit_per_px": None, "right_flank_gradient_8bit_per_px": None,
            "basic_pixel_support": False, "two_flank_support": None,
        }
        if corner:
            value["basic_pixel_support"] = bool(patch_contrast[j] >= config.minimum_channel_contrast_8bit and
                                                 patch_gradient[j] >= config.minimum_gradient_8bit_per_px)
            result.append(value)
            continue
        if not np.isfinite(normals[j]).all():
            value["support_mode"] = "unoriented_neighborhood_only"
            value["basic_pixel_support"] = bool(patch_contrast[j] >= config.minimum_channel_contrast_8bit and
                                                 patch_gradient[j] >= config.minimum_gradient_8bit_per_px)
            result.append(value)
            continue
        sigma = row.get("declared_sigma_px") or 0.0
        radius = min(config.profile_max_radius_px, max(config.profile_base_radius_px, 2 * sigma))
        if paint_center and flank_uv[j]:
            actual_sides = np.asarray(flank_uv[j])
            radius = min(config.profile_max_radius_px, max(radius,
                         np.linalg.norm(actual_sides[1]-actual_sides[0]) * 0.65))
        offsets = np.arange(-radius, radius+config.profile_step_px*.5, config.profile_step_px)
        points = uv[j] + offsets[:, None] * normals[j]
        profiles = _sample(channels, points)
        grad = np.diff(profiles, axis=0) / config.profile_step_px
        contrast = np.percentile(profiles, 90, axis=0) - np.percentile(profiles, 10, axis=0)
        channel = int(np.argmax(contrast))
        gradient_abs = np.abs(grad[:, channel])
        k = int(np.argmax(gradient_abs))
        gradient_offsets = (offsets[:-1] + offsets[1:]) * .5
        value.update(profile_contrast_8bit=float(contrast[channel]),
                     profile_peak_gradient_8bit_per_px=float(gradient_abs[k]),
                     profile_peak_offset_px=float(gradient_offsets[k]), profile_radius_px=float(radius),
                     profile_outside_image=bool(np.any(points[:, 0] < 0) or np.any(points[:, 0] >= width)
                                               or np.any(points[:, 1] < 0) or np.any(points[:, 1] >= height)))
        if paint_center:
            value["support_mode"] = "paint_center_two_flanks"
            if flank_uv[j]:
                # Check measured flank locations themselves.  Do not demand a
                # gradient at the centre of a finite-width painted stripe.
                strengths = []
                local_offsets = np.arange(-config.flank_local_radius_px,
                                           config.flank_local_radius_px+0.5, 1.0)
                for side in flank_uv[j]:
                    probes = np.asarray(side) + local_offsets[:, None] * normals[j]
                    dx, dy = _sample(gx, probes), _sample(gy, probes)
                    directional = dx*normals[j, 0] + dy*normals[j, 1]
                    strengths.append(float(np.abs(directional).max()))
                value["flank_definition"] = "retained_actual_source_sides"
            else:
                left = gradient_offsets < -config.flank_minimum_offset_px
                right = gradient_offsets > config.flank_minimum_offset_px
                strengths = [float(gradient_abs[left].max(initial=0)),
                             float(gradient_abs[right].max(initial=0))]
                value["flank_definition"] = "two_sides_of_observation_profile_candidates_only"
            value["left_flank_gradient_8bit_per_px"], value["right_flank_gradient_8bit_per_px"] = strengths
            value["two_flank_support"] = bool(min(strengths) >= config.minimum_gradient_8bit_per_px)
            value["basic_pixel_support"] = bool(value["two_flank_support"] and
                                                 contrast[channel] >= config.minimum_channel_contrast_8bit)
        else:
            value["basic_pixel_support"] = bool(contrast[channel] >= config.minimum_channel_contrast_8bit and
                                                 gradient_abs[k] >= config.minimum_gradient_8bit_per_px)
        result.append(value)
    return result


def screen_sources(normalized_dir, native_frames_dir, output_dir,
                   *, config: ScreenConfig | dict | None = None):
    """Screen all supplied representatives against exact decoded native PNGs.

    Writes screen_rows.jsonl.gz, screen_summary.json, exception_jobs.jsonl.gz.
    This is an initial source-evidence layer; it has no model-fit residual input
    and never deletes or promotes any observation into a solver.
    """
    normalized = Path(normalized_dir).resolve()
    native = Path(native_frames_dir).resolve()
    output = Path(output_dir).resolve()
    if output == normalized or output.is_relative_to(normalized) or output == native or output.is_relative_to(native):
        raise ValueError("Screen output must be separate from immutable normalized/native inputs")
    output.mkdir(parents=True, exist_ok=True)
    config = ScreenConfig(**config) if isinstance(config, dict) else config or ScreenConfig()
    native_manifest = _load_json(native / "manifest.json")
    native_map = {_frame_key(f): f for f in native_manifest["frames"]}
    if len(native_map) != len(native_manifest["frames"]):
        raise ValueError("Duplicate native source frame identities")
    semantic_issues = set()
    for material in _load_json(normalized / "material_crosswalk_candidates.json"):
        values = set(material.get("measurement_operators", {})) - {"None"}
        if len(values) > 1:
            semantic_issues.add(material["target_id"])
    counts, flags_counter, statuses = Counter(), Counter(), Counter()
    by_shot, by_category = defaultdict(Counter), defaultdict(Counter)
    frame_results = []
    group_frames = iter(_by_frame(normalized / "groups.jsonl.gz"))
    with ExitStack() as stack:
        rows_out = _writer(stack, output / "screen_rows.jsonl.gz")
        jobs_out = _writer(stack, output / "exception_jobs.jsonl.gz")
        for key, rows in _by_frame(normalized / "observations.jsonl.gz"):
            group_key, group_rows = next(group_frames)
            if key != group_key:
                raise ValueError("Normalized rows and source groups are not in identical frame order")
            groups = {g["group_id"]: g for g in group_rows}
            record = native_map.get(key)
            if record is None:
                raise ValueError(f"Native PNG absent for {key}")
            image_path = Path(record["native_png"])
            if not image_path.is_absolute():
                image_path = native / image_path
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                if tuple(image.size) != (record["native_hw"][1], record["native_hw"][0]):
                    raise ValueError(f"Native PNG dimensions mismatch: {image_path}")
                channels, gx, gy = _channels(image, config)
            frame_flags = Counter()
            frame_counts = Counter()
            for group_id, row_iterator in itertools.groupby(rows, key=lambda r: r["provenance_group_id"]):
                segment_rows = list(row_iterator)
                group = groups[group_id]
                feature = groups.get(group.get("feature_source_group_id"), {})
                raw_kind, raw_dist, normals, normal_kind, flank_uv = _source_support(segment_rows, group, feature, config)
                photos = _photometric(segment_rows, normals, normal_kind, flank_uv, channels, gx, gy, config)
                exceptions = []
                reason_counter = Counter()
                for j, (row, photo) in enumerate(zip(segment_rows, photos)):
                    flags = list(row.get("qa_flags", []))
                    evidence_class = row.get("evidence_class")
                    if raw_dist[j] is None:
                        flags.append("no_retained_raw_readout_for_distance_check")
                        if evidence_class == "derived_source_curve":
                            flags.append("derived_curve_missing_retained_raw_readout")
                    else:
                        reference = max(config.nearest_raw_warning_px,
                                        config.raw_warning_sigma_multiplier * (row.get("declared_sigma_px") or 0))
                        if raw_dist[j] > reference:
                            flags.append("representative_raw_point_distance_large")
                    if not photo["basic_pixel_support"]:
                        flags.append("weak_local_photometric_support")
                        if evidence_class == "derived_source_curve":
                            flags.append("derived_curve_low_contrast")
                    if photo["support_mode"] == "paint_center_two_flanks" and not photo["two_flank_support"]:
                        flags.append("paint_flanks_not_both_supported_in_local_check")
                    if photo["profile_outside_image"]:
                        flags.append("local_profile_touches_image_boundary")
                    if row.get("point_correspondence") and photo["corner_second_direction_ratio"] < config.corner_second_direction_ratio:
                        flags.append("corner_second_direction_weak")
                    if row.get("material_id") in semantic_issues:
                        flags.append("target_operator_crosswalk_required")
                    if row.get("inside_supplied_person_mask") and row.get("fit_role") == "reference_manual":
                        flags.append("manual_point_mask_conflict_review_mask_first")
                    if row.get("manual_authoritative") and raw_dist[j] is not None:
                        flags.append("raw_evidence_requires_manual_deletion_visibility_intersection")
                    flags = sorted(set(flags))
                    named = row.get("material_id") is not None
                    if row.get("fit_role") == "reference_manual":
                        status = "manual_reference_retained"
                    elif "target_operator_crosswalk_required" in flags or "ice_measurement_operator_unspecified" in flags:
                        status = "measurement_operator_review_required"
                    elif not photo["basic_pixel_support"]:
                        status = "weak_source_support_requires_automated_followup"
                    elif not named:
                        status = "pixel_supported_candidate_identity_unbound"
                    else:
                        status = "pixel_supported_candidate_identity_unverified"
                    screened = {
                        "schema": "rinkfit.joint9.source-screen-row.v1",
                        "source_row_id": row["source_row_id"], "shot_id": row["shot_id"],
                        "rig_id": row["rig_id"], "native_pts": row["native_pts"],
                        "programme_frame_index_1based": row["programme_frame_index_1based"],
                        "source_uv": row["source_uv"], "material_id": row.get("material_id"),
                        "raw_target": row.get("raw_target"), "category": row.get("category"),
                        "provenance_group_id": group_id,
                        "source_packet_sha256": row["source_packet_sha256"],
                        "measurement_operator": row.get("measurement_operator"),
                        "fit_role": row.get("fit_role"), "evidence_class": evidence_class,
                        "native_png": record["native_png"],
                        "nearest_raw_point_distance_px": raw_dist[j], "raw_distance_source": raw_kind,
                        "raw_distance_is_not_measurement_accuracy": True,
                        "photometric": photo, "screen_flags": flags, "screen_status": status,
                        "source_coordinates_changed": False, "solver_admission_decided": False,
                        "material_identity_verified": False, "fit_accuracy_verified": False,
                        "manual_reference_preserved": row.get("fit_role") == "reference_manual",
                    }
                    _emit(rows_out, screened)
                    counts["rows_total"] += 1
                    counts["named_rows" if named else "unbound_rows"] += 1
                    counts["basic_pixel_support_true" if photo["basic_pixel_support"] else "basic_pixel_support_false"] += 1
                    if raw_dist[j] is not None:
                        counts["raw_distance_available"] += 1
                    statuses[status] += 1
                    by_shot[row["shot_id"]][status] += 1
                    by_category[row.get("category", "unknown")][status] += 1
                    frame_flags.update(flags)
                    flags_counter.update(flags)
                    frame_counts[status] += 1
                    if flags or not named:
                        exceptions.append(row["source_row_id"])
                        reason_counter.update(flags)
                        if not named:
                            reason_counter["material_identity_unbound"] += 1
                if exceptions:
                    counts["exception_group_jobs"] += 1
                    _emit(jobs_out, {
                        "schema": "rinkfit.joint9.source-exception-group.v1",
                        "job_id": "source-screen:" + group_id, "shot_id": key[0], "native_pts": key[1],
                        "provenance_group_id": group_id, "native_png": record["native_png"],
                        "material_ids": sorted({r["material_id"] for r in segment_rows if r.get("material_id")}),
                        "source_row_ids": exceptions, "reason_counts": dict(reason_counter),
                        "action": "automated_source_operator_identity_followup_preserve_original_records",
                        "human_per_frame_review_required": False,
                        "mask_conflict_with_manual_reference": bool(reason_counter.get("manual_point_mask_conflict_review_mask_first")),
                        "suggested_source_roi_xyxy": [max(0., min(r["source_uv"][0] for r in segment_rows)-12),
                                                       max(0., min(r["source_uv"][1] for r in segment_rows)-12),
                                                       min(float(record["native_hw"][1]), max(r["source_uv"][0] for r in segment_rows)+12),
                                                       min(float(record["native_hw"][0]), max(r["source_uv"][1] for r in segment_rows)+12)],
                    })
            counts["native_frames_screened"] += 1
            frame_results.append({"shot_id": key[0], "native_pts": key[1], "rows": len(rows),
                                  "status_counts": dict(frame_counts), "flag_counts": dict(frame_flags)})
    expected = _load_json(normalized / "summary.json")["counts"]["rows_total"]
    if counts["rows_total"] != expected:
        raise ValueError(f"Screened row count {counts['rows_total']} does not cover normalized count {expected}")
    try:
        next(group_frames)
    except StopIteration:
        pass
    else:
        raise ValueError("Unused source group frames")
    result = {"schema": "rinkfit.joint9.source-screen-summary.v1", "counts": dict(counts),
              "screen_status_counts": dict(statuses), "screen_flag_counts": dict(flags_counter),
              "by_shot": {k: dict(v) for k, v in by_shot.items()},
              "by_category": {k: dict(v) for k, v in by_category.items()},
              "config": asdict(config), "threshold_status": "initial_transparent_screening_choices_not_calibrated_acceptance",
              "source_video_sha256": native_manifest.get("source_video_sha256"),
              "native_manifest_sha256": hashlib.sha256((native/"manifest.json").read_bytes()).hexdigest(),
              "normalized_inventory_sha256": _load_json(normalized/"summary.json").get("source_inventory_sha256"),
              "fit_residuals_used": False, "source_coordinates_changed": False,
              "observations_deleted": 0, "solver_admission_decided": False,
              "material_identity_verified": False, "fit_accuracy_verified": False,
              "next_stage": "temporal_and_cross_view_identity_verification_operator_resolution_source_remeasurement",
              "limitations": ["Photometric structure can belong to the wrong physical material.",
                              "Nearest raw-vertex distance also depends on original sampling spacing.",
                              "Normal candidates are not material orientation certificates.",
                              "A two-flank profile can cross unrelated boundaries; identity remains pending.",
                              "Existing masks are flags, not authority to remove manual references.",
                              "Raw observations inside manual deleted regions must not be reactivated."]}
    (output/"screen_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    (output/"screen_frames.json").write_text(json.dumps(frame_results, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("normalized_dir", type=Path)
    parser.add_argument("native_frames_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    summary = screen_sources(args.normalized_dir, args.native_frames_dir, args.output_dir,
                             config=_load_json(args.config) if args.config else None)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
