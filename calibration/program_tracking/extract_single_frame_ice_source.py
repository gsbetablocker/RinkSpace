"""Sparse native-source ice observations for one explicitly requested frame.

The existing material reader provides every UV observation. Camera projection
only places signed profile searches and chooses a named material hypothesis.
This command does not fit a camera, alter geometry, or process other frames.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "src"))
from rinkspace.vision_programme_ice_review import (
    IceProgrammeSourceReader, _batched_actual_profiles, _source_queries,
    _actual_profile_candidates, _candidate_context_excluded,
)
from rinkspace.vision_source_shapes import _MaterialProfileFrame


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def line_distance(p, a, b):
    vector = b - a
    if float(vector @ vector) < 1e-12:
        return np.linalg.norm(p - a, axis=-1)
    t = np.clip(((p - a) @ vector) / (vector @ vector), 0, 1)
    return np.linalg.norm(p - (a + t[:, None] * vector), axis=-1)


def simplify(points, *, tolerance=2.5, maximum_gap=350., closed=False):
    """Retain measured vertices; no interpolation or synthetic circle points."""
    points = np.asarray(points, float)
    if len(points) < 3:
        return np.arange(len(points))
    anchors = {0, len(points) - 1}
    if closed:
        anchors.update(int(np.argmin(points[:, axis])) for axis in (0, 1))
        anchors.update(int(np.argmax(points[:, axis])) for axis in (0, 1))
    keep = set(anchors)

    def split(first, last):
        if last <= first + 1:
            return
        distances = line_distance(points[first + 1:last], points[first], points[last])
        k = first + 1 + int(np.argmax(distances))
        if float(np.max(distances)) > tolerance:
            keep.add(k)
            split(first, k)
            split(k, last)
        elif np.linalg.norm(points[last] - points[first]) > maximum_gap:
            lengths = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points[first:last + 1], axis=0), axis=1))]
            k = first + int(np.argmin(np.abs(lengths - lengths[-1] / 2)))
            if first < k < last:
                keep.add(k)
                split(first, k)
                split(k, last)

    ordered = sorted(anchors)
    for a, b in zip(ordered[:-1], ordered[1:]):
        split(a, b)
    return np.asarray(sorted(keep), int)


def candidate_support(path, qi, ci, raw, profiles):
    offsets, _, smooth, valid = profiles
    normal = raw["query_normal"][qi]
    query = raw["query_uv"][qi]
    quantiles = raw["candidate_side_quantiles_uv"][ci]
    sigma = 0.
    contrasts = []
    side_offsets = []
    for qs in quantiles:
        values = (qs - query) @ normal
        side_offsets.append(float(values[1]))
        sigma = max(sigma, float(abs(values[2] - values[0]) / 1.349))
        contrasts.append(abs(float(np.interp(values[2], offsets[qi, valid[qi]], smooth[qi, valid[qi]])
                                   - np.interp(values[0], offsets[qi, valid[qi]], smooth[qi, valid[qi]]))))
    shift = float((raw["candidate_uv"][ci] - query) @ normal)
    if path["operator"] == "paint_center":
        if abs(side_offsets[1] - side_offsets[0]) < 1.5:
            return None, "unresolved_paint_sides"
    if min(contrasts) < 3.0:
        return None, "insufficient_native_material_contrast"
    if sigma > 15.:
        return None, "material_transition_too_broad_or_merged"
    return dict(native_profile_shift_px=shift,
                signed_interface_iqr_contrast=contrasts,
                transition_sigma_px=max(1., sigma),
                query_index=int(qi), candidate_index=int(ci),
                candidate_flags=int(raw["candidate_flags"][ci]),
                query_uv=raw["query_uv"][qi].astype(float).tolist(),
                profile_normal=normal.astype(float).tolist(),
                actual_source_sides_uv=raw["candidate_sides"][ci].astype(float).tolist(),
                actual_source_side_quantiles_uv=quantiles.tolist()), None


def ellipse_distance(points, ellipse):
    center, axes, angle = ellipse
    theta = np.deg2rad(angle)
    rotation = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    local = (np.asarray(points) - center) @ rotation.T
    radii = np.maximum(np.asarray(axes) / 2., .5)
    implicit = (local ** 2 / radii ** 2).sum(axis=1) - 1.
    gradient = np.sqrt(((2 * local / radii ** 2) ** 2).sum(axis=1))
    return np.abs(implicit) / np.maximum(gradient, 1e-6)


def source_ellipse_guide(points):
    """Fit source readouts only, for branch selection; never emit fitted UVs."""
    points = np.asarray(points, np.float32)
    if len(points) < 8:
        return None
    active = np.ones(len(points), bool)
    ellipse = None
    for _ in range(5):
        if active.sum() < 8:
            break
        ellipse = cv2.fitEllipse(points[active])
        if not np.isfinite(np.r_[ellipse[0], ellipse[1], ellipse[2]]).all():
            return None
        distance = ellipse_distance(points, ellipse)
        median = float(np.median(distance[active]))
        mad = float(1.4826 * np.median(np.abs(distance[active] - median)))
        next_active = distance <= max(3., median + 2.5 * mad)
        if np.array_equal(next_active, active):
            break
        active = next_active
    return ellipse


def remove_reviewed_source_points(dataset, media, exclusions_file):
    """Apply explicit physical-identity vetoes without rereading other points."""
    packet_path = dataset / "ice_features.json"
    packet = read_json(packet_path)
    original = deepcopy(packet)
    rules = read_json(exclusions_file)
    snapshot = media / f"ice_features_before_reflection_review_{packet['point_count']}points.json"
    if not snapshot.exists():
        snapshot.write_bytes(packet_path.read_bytes())
    removed = []
    for rule in rules:
        feature = next(f for f in packet["features"] if f["target_id"] == rule["target_id"])
        wanted = np.asarray(rule["source_uv"], float)
        kept_segments, removed_uv = [], []
        for segment in feature["segments"]:
            groups, current = [], []
            for i, point in enumerate(segment["uv"]):
                reject = bool((np.linalg.norm(wanted - point, axis=1) < .001).any())
                if reject:
                    removed_uv.append(point)
                    if current:
                        groups.append(current)
                        current = []
                else:
                    current.append(i)
            if current:
                groups.append(current)
            for ids in groups:
                revised = deepcopy(segment)
                for key in ("uv", "sigma_px", "source_evidence"):
                    revised[key] = [segment[key][i] for i in ids]
                revised["vertex_count"] = len(ids)
                revised["closed"] = segment["closed"] and len(ids) == len(segment["uv"])
                kept_segments.append(revised)
        if len(removed_uv) != len(wanted):
            raise ValueError("Explicit source-identity veto did not match all requested observations")
        feature["segments"] = kept_segments
        feature["point_count"] = sum(len(s["uv"]) for s in kept_segments)
        feature["note_zh"] += " " + rule["note_zh"]
        feature["source_visible_material_exclusions"] = deepcopy(rule)
        if feature.get("current_source_curve_branch_guide"):
            feature["current_source_curve_branch_guide"]["initial_search_hypotheses_include_now_withheld_candidates"] = True
            feature["current_source_curve_branch_guide"]["not_fitting_admissible_geometry"] = True
        observed = np.asarray([uv for s in kept_segments for uv in s["uv"]], np.float32)
        sigmas = np.asarray([sg for s in kept_segments for sg in s["sigma_px"]])
        ellipse = cv2.fitEllipse(observed)
        residual = ellipse_distance(observed, ellipse)
        feature["geometric_interpretation"].update(center_uv=list(ellipse[0]), axes_px=list(ellipse[1]),
            angle_degrees=ellipse[2], maximum_source_readout_to_ellipse_distance_px=float(residual.max()),
            maximum_residual_divided_by_readout_sigma=float((residual / sigmas).max()),
            all_readouts_within_maximum_of_2point5_px_or_2point5_sigma=bool(
                (residual <= np.maximum(2.5, 2.5 * sigmas)).all()),
            fit_scope="remaining independently retained material observations only; four ambiguous skirt-left points excluded")
        report = next(r for r in packet["curve_readout_summary"] if r["target_id"] == rule["target_id"])
        report["final_points"] = feature["point_count"]
        report["note_zh"] += " " + rule["note_zh"]
        report["rejections"].append(dict(source_uv=removed_uv, note_zh=rule["note_zh"],
            reason="stationary_material_identity_unconfirmed_possible_body_reflection", independent_native_visual_review=True))
        record = dict(target_id=rule["target_id"], omitted_point_count=len(removed_uv),
            source_uv=removed_uv, note_zh=rule["note_zh"],
            native_candidate_evidence_retained_in_source_archive=True)
        packet["unresolved_source_regions"].append(record)
        removed.append(record)
    packet["point_count"] = sum(f["point_count"] for f in packet["features"])
    packet["source_visible_point_exclusions"] = removed
    packet["visual_review"]["independent_stationary_material_review_exclusions"] = removed
    assert packet["point_count"] == original["point_count"] - sum(r["omitted_point_count"] for r in removed)
    for old, new in zip(original["features"], packet["features"]):
        if old["target_id"] not in {r["target_id"] for r in rules}:
            assert old == new
    # Every remaining coordinate and its source readout are retained bytewise
    # as JSON values, without a new reader invocation or fitted replacement.
    original_rows = {tuple(uv): (sigma, evidence) for f in original["features"] for s in f["segments"]
        for uv, sigma, evidence in zip(s["uv"], s["sigma_px"], s["source_evidence"])}
    for f in packet["features"]:
        for s in f["segments"]:
            for uv, sigma, evidence in zip(s["uv"], s["sigma_px"], s["source_evidence"]):
                assert original_rows[tuple(uv)] == (sigma, evidence)
    data = json.dumps(packet, ensure_ascii=False, indent=2) + "\n"
    assert "\ufffd" not in data
    packet_path.write_text(data, encoding="utf-8")
    overlay = cv2.imread(str(media / "source.png"))
    for f in packet["features"]:
        color = (222, 85, 230) if "R4:" in f["id"] else (25, 210, 250) if "R3:" in f["id"] else (70, 235, 70)
        for s in f["segments"]:
            points = np.rint(np.asarray(s["uv"]) * 256).astype(np.int32)
            cv2.polylines(overlay, [points], s["closed"], color, 1, cv2.LINE_AA, shift=8)
            for p in points:
                cv2.circle(overlay, tuple(map(int, p)), 3 * 256, (15, 15, 15), -1, cv2.LINE_AA, shift=8)
                cv2.circle(overlay, tuple(map(int, p)), 2 * 256, color, -1, cv2.LINE_AA, shift=8)
    cv2.imwrite(str(media / "ice_points_review.png"), overlay)
    print(json.dumps(dict(point_count=packet["point_count"], removed=len(original_rows)-packet["point_count"],
        sha256=sha(packet_path), other_observations_unchanged=True, native_evidence_archive_unchanged=True)))


def reread_source_ring(frame, path, ellipse, exclusion):
    """Read signed interfaces around a source-derived curve search guide."""
    center, axes, angle = ellipse
    theta = np.deg2rad(angle)
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    angles = np.linspace(0., 2 * np.pi, 513)
    projected = np.c_[np.cos(angles), np.sin(angles)] * (np.asarray(axes) / 2.)
    projected = projected @ rotation.T + center
    queries, normals, intervals = _source_queries(dict(path, _projected_center=np.asarray(center)),
        projected, np.ones(len(projected)), frame.rgb.shape[:2], 24., 20.)
    profiles = _batched_actual_profiles(frame, queries, normals, intervals,
        np.full(len(queries), path["material_signal"]))
    uv, sides, quantiles, qi_list, flags = [], [], [], [], []
    for qi in range(len(queries)):
        off, profile, smooth, valid = (v[qi] for v in profiles)
        values = _actual_profile_candidates(queries[qi], normals[qi], off[valid], profile[valid], smooth[valid],
            operator=path["operator"], signal=path["material_signal"], transition=path["material_transition"],
            with_quantiles=True)
        for point, side, bad, qs in values:
            excluded = any(_candidate_context_excluded(exclusion, p, normals[qi]) for p in [point, *side])
            uv.append(point); sides.append(side); quantiles.append(qs); qi_list.append(qi)
            flags.append(int(bad) | (int(excluded) << 1) | (int(len(values) > 1) << 2) | 8)
    raw = dict(query_uv=queries, query_normal=normals, query_interval=intervals,
        candidate_uv=np.asarray(uv).reshape(-1, 2), candidate_sides=np.asarray(sides).reshape(-1, 2, 2),
        candidate_side_quantiles_uv=np.asarray(quantiles).reshape(-1, 2, 3, 2),
        candidate_query=np.asarray(qi_list), candidate_flags=np.asarray(flags, np.uint8))
    candidate_by_query = defaultdict(list)
    for ci, qi in enumerate(qi_list):
        candidate_by_query[qi].append(ci)
    readings, rejects = [], []
    for qi in range(len(queries)):
        eligible = []
        for ci in candidate_by_query[qi]:
            if raw["candidate_flags"][ci] & 3:
                continue
            support, reason = candidate_support(path, qi, ci, raw, profiles)
            if support is None:
                continue
            support["query_set"] = "current_source_curve_guided_refinement"
            shift = abs(support["native_profile_shift_px"])
            if shift < 9.:
                eligible.append((shift, raw["candidate_uv"][ci], support))
        eligible.sort(key=lambda x:x[0])
        if not eligible or (len(eligible) > 1 and eligible[1][0] - eligible[0][0] < 2.):
            readings.append(None)
            rejects.append(dict(query_index=qi, query_set="current_source_curve_guided_refinement",
                reason="no_unique_supported_current_source_ring_material_interface"))
        else:
            readings.append((eligible[0][1].copy(), eligible[0][2]))
    raw.update(profile_offsets=profiles[0], profile_values=profiles[1], profile_smoothed=profiles[2], profile_valid=profiles[3])
    vertex_index = np.array([int(np.argmin(np.linalg.norm(projected - q, axis=1))) for q in queries])
    return readings, rejects, vertex_index, projected, raw


def run(dataset: Path, media: Path, *, mask_path: Path, provisional=False, withheld_targets=None):
    source_binding = read_json(media / "source_binding.json")
    camera = read_json(dataset / "camera_seed.json")
    model_root = ROOT / "events/ankara_2026/models/framewise_8a28_20261013"
    scene = read_json(model_root / "scene.json")
    mapping = read_json(dataset / "navigation_mapping.json")
    confirmed = {r["curve_id"] for r in mapping["model_entry_status"]["projected_ice_curve_navigation"]}
    # The historical row3 proxy and user-confirmed house line are one source
    # material. Only the physical house line receives an observation row.
    confirmed.discard("curling_centerline_row3:segment0")
    bgr = cv2.imread(str(media / "source.png"))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) > 0
    if bgr.shape[:2] != (1080, 1920) or mask.shape != bgr.shape[:2]:
        raise ValueError("Native image and current-source mask must be 1920x1080")
    if hashlib.sha256(rgb.tobytes()).hexdigest() != source_binding["decoded_RGB_sha256"]:
        raise ValueError("Current source RGB hash differs from exact source binding")
    # Soft motion blur halo is excluded in addition to the source segmentation.
    exclusion = cv2.dilate(mask.astype(np.uint8), np.ones((13, 13), np.uint8)) > 0
    exclusion[25:105, 25:557] = True
    exclusion[943:997, 1685:1870] = True
    reader = IceProgrammeSourceReader(scene, source_sha256=source_binding["source_video_sha256"],
        time_base=source_binding["time_base"], sample_spacing_px=24., search_radius_px=64.)
    reader.paths = [p for p in reader.paths if p["curve_id"] in confirmed]
    raw = reader.read(rgb, source_binding["native_pts"], np.asarray(camera["camera25"]), exclusion, shot_id=1)
    frame = _MaterialProfileFrame(rgb, raw["source_binding"])
    profiles = _batched_actual_profiles(frame, raw["query_uv"], raw["query_normal"], raw["query_interval"],
        np.array([reader.paths[i]["material_signal"] for i in raw["query_curve"]]))
    candidates = defaultdict(list)
    for ci, qi in enumerate(raw["candidate_query"]):
        candidates[int(qi)].append(ci)
    features, curve_reports, refined_evidence, source_ellipses = [], [], {}, {}
    withheld_targets = dict(withheld_targets or {})
    selected_evidence = []
    for pi, path in enumerate(reader.paths):
        queries = np.flatnonzero(raw["query_curve"] == pi)
        readings, rejections, eligible_per_query = [], [], []
        is_ring = "center_xyz" in path
        for qi in queries:
            eligible = []
            for ci in candidates[int(qi)]:
                if raw["candidate_flags"][ci] & 3:
                    continue
                support, reason = candidate_support(path, qi, ci, raw, profiles)
                if support is not None:
                    eligible.append((abs(support["native_profile_shift_px"]), ci, support))
            eligible.sort(key=lambda x: x[0])
            eligible_per_query.append(eligible)
            if not eligible:
                readings.append(None)
                rejections.append(dict(query_index=int(qi), reason="no_clean_supported_native_material_candidate"))
                continue
            # Model search is a branch hypothesis only. All output UVs and
            # uncertainty below are the native measured signed interfaces.
            shift, ci, support = eligible[0]
            if shift > (48. if is_ring else 12.):
                readings.append(None)
                rejections.append(dict(query_index=int(qi), reason="source_candidate_identity_requires_additional_review"))
                continue
            if len(eligible) > 1 and eligible[1][0] - shift < 3.:
                readings.append(None)
                rejections.append(dict(query_index=int(qi), reason="competing_native_material_branches"))
                continue
            readings.append((raw["candidate_uv"][ci].astype(float), support))
        ellipse = None
        refined_navigation_index = None
        if is_ring:
            seed = [r[0] for r in readings if r is not None]
            ellipse = source_ellipse_guide(seed)
            if path["world_curve_role"] == "center":
                family = [source_ellipses.get(path["feature_id"] + ":" + role) for role in ("outer", "middle", "inner")]
                family = [f for f in family if f is not None]
                if family:
                    inner = source_ellipses.get(path["feature_id"] + ":inner") or family[-1]
                    inner_path = next(p for p in reader.paths if p["feature_id"] == path["feature_id"]
                        and p["world_curve_role"] == "inner")
                    radius = float(np.linalg.norm(path["xyz"][0] - path["center_xyz"]))
                    inner_radius = float(np.linalg.norm(inner_path["xyz"][0] - inner_path["center_xyz"]))
                    ratio = radius / inner_radius
                    # Shared material concentricity and size ratio only locate
                    # the weak centre-button search; UV remains a fresh read.
                    ellipse = (tuple(np.median([f[0] for f in family], axis=0)),
                               tuple(np.asarray(inner[1]) * ratio), inner[2])
            if ellipse is not None:
                source_ellipses[path["curve_id"]] = ellipse
                readings, refined_rejects, refined_navigation_index, refined_projected, archive = reread_source_ring(
                    frame, path, ellipse, exclusion)
                rejections.extend(refined_rejects)
                for key, value in archive.items():
                    refined_evidence[path["curve_id"].replace(":", "_") + "__" + key] = value
        else:
            # A source straight-paint identity must also remain coherent along
            # the current-frame material. A competing circle arc is withheld.
            active = [r for r in readings if r is not None]
            if len(active) >= 4:
                uv = np.asarray([r[0] for r in active])
                model_uv = raw["query_uv"][queries]
                model_slope = np.polyfit(model_uv[:, 0], model_uv[:, 1], 1)[0]
                line = np.polyfit(uv[:, 0], uv[:, 1], 1)
                if abs(line[0] - model_slope) > .015:
                    readings = [None] * len(readings)
                    rejections.append(dict(reason="current_source_band_not_same_named_straight_paint_direction"))
                else:
                    for ri, reading in enumerate(readings):
                        if reading is not None and abs(reading[0][1] - np.polyval(line, reading[0][0])) > 4.:
                            readings[ri] = None
                            rejections.append(dict(query_index=int(queries[ri]), reason="incoherent_current_source_straight_paint_branch"))
        from calibration.program_tracking.projection import project_fast
        projected, depth = project_fast(np.asarray(camera["camera25"]), path["xyz"])
        nearest_navigation_vertex = (refined_navigation_index if refined_navigation_index is not None else
            np.array([int(np.argmin(np.linalg.norm(projected - raw["query_uv"][qi], axis=1))) for qi in queries]))
        groups, current = [], []
        previous_ri = None
        for ri, reading in enumerate(readings):
            if reading is None:
                if current:
                    groups.append(current)
                    current = []
                continue
            navigation_gap = (is_ring and previous_ri is not None
                and abs(nearest_navigation_vertex[ri] - nearest_navigation_vertex[previous_ri]) >
                    (48 if refined_navigation_index is not None else 24))
            if current and (np.linalg.norm(reading[0] - current[-1][0]) > 72. or navigation_gap):
                groups.append(current)
                current = []
            current.append(reading)
            previous_ri = ri
        if current:
            groups.append(current)
        entirely_read = bool(len(readings)) and all(r is not None for r in readings)
        # A ring can close only if all of its actual visible material was read
        # and no part of its projection leaves the image or current mask.
        closure_projection = refined_projected if refined_navigation_index is not None else projected
        full_in_frame = bool((depth > .05).all() and np.isfinite(closure_projection).all()
            and (closure_projection >= [8, 8]).all() and (closure_projection <= [1911, 1071]).all())
        if is_ring and len(groups) > 1 and full_in_frame and readings[0] is not None and readings[-1] is not None:
            if np.linalg.norm(groups[-1][-1][0] - groups[0][0][0]) < 48.:
                groups = [groups[-1] + groups[0]] + groups[1:-1]
        segments = []
        for gi, group in enumerate(groups):
            if len(group) < 2:
                continue
            points = np.asarray([r[0] for r in group])
            closed = bool(is_ring and len(groups) == 1 and entirely_read and full_in_frame)
            keep = simplify(points, tolerance=2.5 if is_ring else 2.,
                            maximum_gap=350. if is_ring else 240., closed=closed)
            # Straight lines retain approximately consistent source sampling.
            if not is_ring:
                keep = [0]
                for k in range(1, len(points) - 1):
                    if np.linalg.norm(points[k] - points[keep[-1]]) >= 210.:
                        keep.append(k)
                if keep[-1] != len(points) - 1:
                    keep.append(len(points) - 1)
                keep = np.asarray(keep)
            evidence = [group[k][1] for k in keep]
            selected_evidence.extend(evidence)
            segments.append(dict(uv=np.round(points[keep], 4).tolist(), closed=closed,
                vertex_count=len(keep), sigma_px=[round(e["transition_sigma_px"], 4) for e in evidence],
                source_evidence=evidence, measurement_origin="current_native_signed_RGB_material_profiles",
                pixels_measured_in_this_frame=True, source_evidence_measured_in_this_frame=True,
                hidden_span_filled=False, user_confirmed=False))
        display_id = {"R4": "B4", "R3": "B3"}.get(path["feature_id"], path["feature_id"])
        display_role = {"outer": "外层材料界面", "middle": "中层材料界面", "inner": "内层材料界面", "center": "最内空心小圈"}.get(
            path["world_curve_role"], path["world_curve_role"])
        feature = dict(id="ice." + path["curve_id"], target_id="ice." + path["curve_id"],
            label_zh=display_id + " " + display_role, world_curve=path["curve_id"],
            measurement_operator=path["operator"], material_signal=path["material_signal"],
            material_transition=path["material_transition"], segments=segments,
            point_count=sum(s["vertex_count"] for s in segments),
            source_evidence_measured_in_this_frame=True,
            projected_uv_used_as_observations=False,
            physical_identity_evidence="same-frame source material independently inspected with named model navigation",
            current_source_curve_branch_guide=(dict(center_uv=list(ellipse[0]), axes_px=list(ellipse[1]),
                angle_degrees=ellipse[2], emitted_as_observations=False,
                source=("current_native_larger_ring_readouts_with_current_scene_center_button_size_ratio_search_prior"
                    if path["world_curve_role"] == "center" else "current_native_signed_interface_readouts_only"),
                measured_center_button_geometry_claimed=False,
                camera_projection_coordinates_emitted=False) if ellipse is not None else None),
            status="current_source_candidates_waiting_visual_and_user_review")
        observed = np.array([uv for segment in feature["segments"] for uv in segment["uv"]], float).reshape(-1, 2)
        sigmas = np.array([s for segment in feature["segments"] for s in segment["sigma_px"]])
        if is_ring and len(observed) >= 6:
            physical_ellipse = cv2.fitEllipse(observed.astype(np.float32))
            residual = ellipse_distance(observed, physical_ellipse)
            feature["geometric_interpretation"] = dict(kind="projected_physical_circle_ellipse",
                basis="one physical ice-plane circle; one ellipse for all visible observed segments",
                inference_only=True, inferred_uv_not_emitted_as_source_points=True,
                center_uv=list(physical_ellipse[0]), axes_px=list(physical_ellipse[1]), angle_degrees=physical_ellipse[2],
                maximum_source_readout_to_ellipse_distance_px=float(residual.max()),
                maximum_residual_divided_by_readout_sigma=float((residual / sigmas).max()),
                all_readouts_within_maximum_of_2point5_px_or_2point5_sigma=bool(
                    (residual <= np.maximum(2.5, 2.5 * sigmas)).all()),
                hidden_and_blurred_spans_are_inference_not_observations=True)
        elif not is_ring and len(observed) >= 2:
            design = np.c_[observed[:, 0], np.ones(len(observed))]
            line = np.linalg.lstsq(design / sigmas[:, None], observed[:, 1] / sigmas, rcond=None)[0]
            residual = observed[:, 1] - design @ line
            feature["geometric_interpretation"] = dict(kind="straight_ice_paint_centerline",
                inference_only=True, inferred_uv_not_emitted_as_source_points=True,
                current_source_line_y_equals_ax_plus_b=line.tolist(),
                maximum_source_readout_to_straight_line_distance_px=float(np.abs(residual).max()),
                maximum_residual_divided_by_readout_sigma=float(np.abs(residual / sigmas).max()),
                two_points_do_not_independently_establish_material_identity=len(observed) == 2)
        if feature["target_id"] in withheld_targets:
            feature["withheld_candidate_point_count"] = feature["point_count"]
            feature["withheld_reason"] = withheld_targets[feature["target_id"]]
            feature["segments"] = []
            feature["point_count"] = 0
            feature["status"] = "unresolved_native_material_not_admitted_as_source_points"
            rejections.append(dict(reason=feature["withheld_reason"], manual_native_source_and_geometry_review=True))
        note = feature.get("withheld_reason")
        if note is None and feature["point_count"] == 0:
            note = ("右上大弧虽有模型入口，本帧原图红色材料未得到唯一、完整的双边响应；不将灰阶条纹或倒影当作红色标线，未取点。"
                if path["curve_id"] == "A2:outer" else
                "该模型直线进入画面，但弱冰面条纹与邻圈界面、人物遮挡混淆，未找到原图可独立确认的一致直线材料读数，未取点。")
        elif note is None and is_ring:
            note = "同一材料层的全部可见段按一个物理圆的投影椭圆核对；只保留原图剖面读数，模糊尖端、出画和遮挡缺口不补源点。"
        elif note is None:
            note = "原图双边材料读数按同一条长直线核对；遮挡断段保留，未用直线拟合坐标补源点。"
        feature["note_zh"] = note
        features.append(feature)
        curve_reports.append(dict(target_id=feature["target_id"], query_count=len(readings),
            original_camera_navigation_query_count=len(queries), note_zh=note,
            actual_source_supported_queries=sum(x is not None for x in readings),
            final_points=feature["point_count"], rejections=rejections))
    source = dict(source_binding)
    source.update(mask_path=str(mask_path.resolve()), mask_sha256=sha(mask_path),
                  exclusion_halo_px=6, mask_provisional=bool(provisional))
    packet = dict(schema="rinkspace.single-frame-source-ice-features.v1", source=source,
        source_pts=int(source_binding["native_pts"]),
        status="provisional_mask_source_candidates" if provisional else "awaiting_user_review",
        user_accepted=False, frame_count=1, next_frame_processed=False,
        solver_inputs_modified=False, optimization_performed=False,
        navigation_camera_path=str((dataset / "camera_seed.json").resolve()),
        navigation_camera_sha256=sha(dataset / "camera_seed.json"),
        model_scene_sha256=sha(model_root / "scene.json"),
        physical_aliases={"B4": "R4", "B3": "R3"},
        duplicate_physical_material_suppressed={"ice.curling_centerline_row3:segment0": "ice.house_connection_F3_R3:segment0"},
        features=features, point_count=sum(f["point_count"] for f in features),
        unresolved_source_regions=[r for r in curve_reports if r["final_points"] == 0],
        curve_readout_summary=curve_reports,
        manual_source_qualifications=withheld_targets,
        visual_review=dict(status="native_overlay_inspected_by_extraction_agent_pending_main_and_user_review",
            original_and_overlay_inspected=True, camera_or_model_accuracy_accepted=False,
            unavailable_material="A2 red arc and lane4 upper/right edges did not yield unique clear native material readings",
            unresolved_curve_ends="blurred tangent tips and rejected ring profile intervals remain open; no hidden span filled"),
        projected_uv_used_as_observations=False,
        source_evidence_archive=str((media / "ice_native_profile_evidence.npz").resolve()),
        source_sampling_policy=dict(straight_native_spacing_px=[210, 240],
            curved_source_simplification_maximum_deviation_px=2.5,
            source_measurements_retained_without_interpolation=True,
            confirmed_actual_material_only=True,
            candidate_admission_requires_visual_review=True))
    media.mkdir(parents=True, exist_ok=True)
    dataset.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(media / "ice_native_profile_evidence.npz", **{
        k: v for k, v in raw.items() if isinstance(v, np.ndarray)}, **refined_evidence,
        profile_offsets=profiles[0], profile_values=profiles[1], profile_smoothed=profiles[2], profile_valid=profiles[3])
    packet["source_evidence_archive_sha256"] = sha(media / "ice_native_profile_evidence.npz")
    (dataset / "ice_features.json").write_text(json.dumps(packet, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    overlay = bgr.copy()
    palette = [(222, 85, 230), (25, 210, 250), (70, 235, 70)]
    for fi, feature in enumerate(features):
        color = palette[0] if "R4:" in feature["id"] else palette[1] if "R3:" in feature["id"] else palette[2]
        for segment in feature["segments"]:
            points = np.rint(np.asarray(segment["uv"]) * 256).astype(np.int32)
            cv2.polylines(overlay, [points], segment["closed"], color, 1, cv2.LINE_AA, shift=8)
            for p in points:
                cv2.circle(overlay, tuple(map(int, p)), 3 * 256, (15, 15, 15), -1, cv2.LINE_AA, shift=8)
                cv2.circle(overlay, tuple(map(int, p)), 2 * 256, color, -1, cv2.LINE_AA, shift=8)
    cv2.imwrite(str(media / "ice_points_review.png"), overlay)
    print(json.dumps(dict(point_count=packet["point_count"], by_feature={f["id"]:f["point_count"] for f in features},
        packet=str(dataset / "ice_features.json"), overlay=str(media / "ice_points_review.png")), ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--media", type=Path, required=True)
    parser.add_argument("--mask", type=Path)
    parser.add_argument("--provisional-mask", action="store_true")
    parser.add_argument("--withhold", action="append", default=[], metavar="TARGET=REASON",
        help="Native visual-review qualification: retain evidence but admit no points for this target")
    parser.add_argument("--withhold-file", type=Path, help="UTF-8 JSON target-to-reason native review qualifications")
    parser.add_argument("--apply-source-exclusions-only", type=Path,
        help="Apply explicit native source-point veto JSON to saved packet, without rereading other observations")
    args = parser.parse_args()
    if args.apply_source_exclusions_only:
        remove_reviewed_source_points(args.dataset, args.media, args.apply_source_exclusions_only)
        return
    withheld = read_json(args.withhold_file) if args.withhold_file else {}
    withheld.update(dict(item.split("=", 1) for item in args.withhold))
    run(args.dataset, args.media, mask_path=args.mask or args.media / "person_mask.png",
        provisional=args.provisional_mask, withheld_targets=withheld)


if __name__ == "__main__":
    main()
