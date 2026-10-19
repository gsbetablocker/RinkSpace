"""Source-only low-end reference rows for a movable-C, shared-optics graph.

This module reads frozen native-pixel observations. It creates no factor graph,
camera state, solve, or new measurement. World curves are resolved afresh from
the caller's current shared scene; historical camera25 values are optional
initializers supplied by the caller and never observation targets.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = Path(__file__).with_name("low_end_reference_source_set_20260928.json")
EXPECTED_SOURCE_SHA256 = "f685d27882e78f2446c734ceee4321dcb275a5ad1ebf81ea90bca68c2fcec891"
EXPECTED_PTS = (14177280, 205992960, 226160640)
EXPECTED_COUNTS = {14177280: (5, 41), 205992960: (11, 57), 226160640: (7, 50)}
RIGHT_CONTACT_COURSE = {"kind": "board_contact_course", "side": "right", "edge": "contact"}


def _require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def _path(value: str) -> Path:
    path = (ROOT / value).resolve()
    _require(path.is_relative_to(ROOT), f"Source path escapes repository: {value}")
    return path


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    _require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def _source_file(value: str, expected_sha256: str) -> tuple[Path, dict]:
    path = _path(value)
    _require(_digest(path) == expected_sha256, f"Source bytes changed: {path}")
    return path, _read(path)


def _world_curve(scene: dict, target: dict) -> np.ndarray:
    """Map only the primary set's finite, explicitly named physical curves."""
    kind = target.get("kind")
    if kind == "board_contact_course":
        _require(target == RIGHT_CONTACT_COURSE, "Unreviewed board contact course")
        boundary = np.asarray(scene["board_surface"]["bottom_edge_xyz"], np.float64)
        _require(boundary.ndim == 2 and boundary.shape[1] == 3
                 and len(boundary) >= 8 and np.isfinite(boundary).all(),
                 "Current scene lacks a finite board contact perimeter")
        # These are existing straight-to-corner tangencies of the current
        # scene, not guessed annotation transition points. Keep the whole
        # lower long side, lower-right corner, right end and upper-right
        # corner in one connected course for all six samples of the stroke.
        x, y = boundary[:, 0], boundary[:, 1]
        upper = np.flatnonzero(np.isclose(y, np.max(y), rtol=0, atol=1e-6))
        lower = np.flatnonzero(np.isclose(y, np.min(y), rtol=0, atol=1e-6))
        right = set(np.flatnonzero(np.isclose(x, np.max(x), rtol=0, atol=1e-6)))
        _require(len(upper) >= 2 and len(lower) >= 2 and len(right) >= 2,
                 "Current scene has no distinct straight board tangencies")
        start = int(lower[np.argmin(x[lower])])  # lower-long left tangent
        end = int(upper[np.argmax(x[upper])])   # upper-long right tangent
        def route(step):
            indices = [start]
            while indices[-1] != end and len(indices) <= len(boundary):
                indices.append((indices[-1] + step) % len(boundary))
            return indices
        candidates = [route(1), route(-1)]
        matches = [indices for indices in candidates
                   if len(indices) <= len(boundary) and right.issubset(indices)]
        _require(len(matches) == 1, "Right-side contact course is not unique")
        return boundary[matches[0]].copy()
    if kind == "board_perimeter":
        _require(target.get("edge") == "contact", "Unreviewed full board edge")
        boundary = np.asarray(scene["board_surface"]["bottom_edge_xyz"], np.float64)
        _require(boundary.ndim == 2 and boundary.shape[1] == 3
                 and len(boundary) >= 4 and np.isfinite(boundary).all(),
                 "Current scene lacks a finite contact perimeter")
        return np.vstack((boundary, boundary[0])).copy()
    if kind == "board_side":
        edge = target.get("edge")
        _require(edge in ("top", "contact"), "Unknown board edge")
        key = "top_edge_xyz" if edge == "top" else "bottom_edge_xyz"
        boundary = np.asarray(scene["board_surface"][key], np.float64)
        _require(boundary.ndim == 2 and boundary.shape[1] == 3,
                 "Current scene board perimeter has invalid shape")
        side = target.get("side")
        if side == "right_end":
            axis, value = 0, np.max(boundary[:, 0])
            order_axis = 1
        elif side == "lower_long_side":
            axis, value = 1, np.min(boundary[:, 1])
            order_axis = 0
        else:
            raise ValueError(f"Unreviewed low-end board side: {side}")
        path = boundary[np.isclose(boundary[:, axis], value, rtol=0, atol=1e-6)]
        _require(len(path) >= 2 and np.ptp(path[:, order_axis]) > 1,
                 f"Current scene has no finite {side} {edge} path")
        return path[np.argsort(path[:, order_axis])].copy()
    feature_id = target.get("feature_id")
    feature = next((f for f in scene["ice_features"] if f["id"] == feature_id), None)
    _require(feature is not None, f"Missing current-scene ice feature: {feature_id}")
    if kind == "circle_outline":
        _require(feature_id == "F1" and feature["kind"] == "circle"
                 and np.array_equal(np.asarray(feature["plane_normal_xyz"], np.float64),
                                    np.array([0., 0., 1.])),
                 "Optional user F1 requires a planar current-scene F1 circle")
        role = target.get("ring_role")
        _require(role in ("outer", "middle", "inner"), "Unreviewed F1 ring role")
        ring = next((r for r in feature["rings"] if r["role"] == role), None)
        _require(ring is not None, f"Missing current-scene F1 ring: {role}")
        center = np.asarray(ring["center_xyz"], np.float64)
        radius = float(ring["radius"])
        _require(center.shape == (3,) and np.isfinite(center).all()
                 and np.isfinite(radius) and radius > 0,
                 f"Invalid current-scene F1 ring: {role}")
        # Numeric representation of the same scene circle; the final point
        # closes the world path without creating a new source observation.
        angle = np.linspace(0., 2. * np.pi, 513)
        path = center[None, :] + radius * np.column_stack((
            np.cos(angle), np.sin(angle), np.zeros_like(angle)))
    elif kind == "line_segment":
        _require(feature["kind"] == "polyline"
                 and len(feature["segments_xyz"]) == 1,
                 f"Current-scene line is not one finite segment: {feature_id}")
        path = np.asarray(feature["segments_xyz"][0], np.float64)
    elif kind == "polygon_edge":
        _require(feature["kind"] == "polygon", "Current-scene lane is not a polygon")
        polygon = np.asarray(feature["polygon_xyz"], np.float64)
        index = int(target["index"])
        _require(index in (0, 2) and index not in feature.get("open_edges", []),
                 f"Unreviewed/open lane edge index: {feature_id}:{index}")
        path = polygon[[index, (index + 1) % len(polygon)]]
    else:
        raise ValueError(f"Unreviewed low-end target kind: {kind}")
    _require(path.ndim == 2 and path.shape[1] == 3 and len(path) >= 2
             and np.isfinite(path).all(), f"Nonfinite current-scene target: {target}")
    return path.copy()


def _optional_user_f1(scene: dict, manifest: dict, frame_entry: dict) -> list[dict]:
    """Expose Step02's valid F1 stroke samples and their *whole-group* budget.

    ``sigma_px=1`` below reproduces an old numerical pixel unit. It is not a
    claim of one-pixel physical annotation accuracy. A graph adapter must
    apply the stored point integration vector to the signed curve deltas.
    """
    spec = manifest["optional_user_F1"]
    _require(int(spec["source_pts"]) == 205992960
             and frame_entry["source_pts"] == 205992960,
             "Optional F1 source exposure changed")
    step_path, step = _source_file(spec["step02_observations"], spec["step02_sha256"])
    contour_path, contour = _source_file(
        spec["user_contour_replacements"], spec["user_contour_sha256"])
    _require(step["schema_version"] == "rinkspace.three_frame_refit.observations.v2"
             and contour["source_pts"] == 205992960
             and contour["source_frame_sha256"] == frame_entry["image_sha256"],
             "Optional F1 stroke/image source changed")
    frame_ids = {f["id"]: int(f["pts"]) for f in step["frames"]}
    by_id = {o["id"]: o for o in step["observations"]}
    curves = []
    for index, (role, observation_id) in enumerate(spec["ids_by_ring_role"].items()):
        _require(role in ("outer", "middle", "inner")
                 and observation_id in by_id, f"Optional F1 ID changed: {role}")
        observation = by_id[observation_id]
        original_ring = contour["F1_extraction"]["rings_outer_to_inner"][index]
        target = {"kind": "circle_outline", "feature_id": "F1", "ring_role": role}
        pointer = f"/F1_extraction/rings_outer_to_inner/{index}"
        _require(frame_ids[observation["frame_id"]] == 205992960
                 and observation["target"] == target
                 and observation["source"]["path"] == spec["user_contour_replacements"]
                 and observation["source"]["pointer"] == pointer
                 and observation["source"]["kind"] == "digitized_user_stroke"
                 and observation["use"] == "fit"
                 and observation["allow_fit"] is True
                 and observation["allow_check"] is False
                 and observation["physical_group_id"] == "F1"
                 and observation["correlation_group_id"] == "low3_205992960:F1",
                 f"Optional user F1 authority or role changed: {role}")
        uv = np.asarray(observation["pixels_uv"], np.float64)
        original_uv = np.asarray(observation["source_pixels_uv"], np.float64)
        recovered_uv = np.asarray(original_ring["upper_native_uv"]
                                  + original_ring["lower_native_uv"], np.float64)
        weight = np.asarray(observation["weight"]["point_integration_weights"],
                            np.float64)
        _require(uv.shape == (spec["historical_active_sample_counts"][index], 2)
                 and original_uv.shape
                 == (spec["historical_full_source_stroke_counts"][index], 2)
                 and np.array_equal(observation["active_indices"],
                                    np.arange(len(uv)))
                 and np.isfinite(uv).all() and np.isfinite(original_uv).all()
                 and recovered_uv.ndim == 2 and recovered_uv.shape[1] == 2
                 and set(map(tuple, original_uv)) <= set(map(tuple, recovered_uv))
                 and np.all((uv[:, 0] >= 0) & (uv[:, 0] < 1920))
                 and np.all((uv[:, 1] >= 0) & (uv[:, 1] < 1080))
                 and weight.shape == (len(uv),) and np.isfinite(weight).all()
                 and np.all(weight >= 0)
                 and abs(float(weight.sum()) - 1. / 3.) < 1e-7
                 and observation["weight"]["group_budget"] == 1
                 and abs(observation["weight"]["observation_share"] - 1. / 3.) < 1e-12
                 and observation["weight"]["point_count_multiplier"] is False
                 and observation["weight"]["solver_ready"] is True,
                 f"Optional user F1 points or correlation budget changed: {role}")
        _require(observation["uncertainty"]["sigma_px"] is None,
                 "F1 numeric pixel unit incorrectly became measurement sigma")
        curves.append(dict(
            id=observation_id, target=target, physical_group_id="F1",
            identity_status="user_confirmed_F1_ring_role; user_stroke_recovered_to_native_pixels",
            source_uv=uv.copy(), original_stroke_uv=original_uv.copy(),
            world_xyz=_world_curve(scene, target),
            sigma_px=1.0,
            sigma_role="historical_fit_joint_numerical_pixel_unit_not_accuracy_bound",
            correlated_curve_weight=1. / 3.,
            point_integration_weights=weight.copy(),
            original_uncertainty_status="not_a_calibrated_accuracy_bound",
            historical_residual_definition="sqrt(point_integration_weights_i) * signed_closest_projected_curve_delta_i / numerical_sigma_1px; linear_loss",
            source_file=str(step_path), source_file_sha256=spec["step02_sha256"],
            original_stroke_file=str(contour_path),
            original_stroke_sha256=spec["user_contour_sha256"],
            source_collection="observations",
            source_measurement_role="user_F1_stroke_arclength_quadrature",
            source_correlation_group="low3_205992960:F1",
            source_path_closed=list(observation["source_path_closed"]),
            source_path_indices=list(observation["path_indices"]),
            source_course_region=None,
        ))
    _require(len(curves) == 3 and sum(len(c["source_uv"]) for c in curves) == 796,
             "Optional F1 sample count changed")
    return curves


def _candidate_target(row: dict, selected: dict, frame_pts: int) -> dict:
    target = selected.get("target", row.get("target"))
    _require(isinstance(target, dict), f"No target for PTS {frame_pts}: {selected['id']}")
    original = row.get("target")
    if original is not None and target != original:
        # The 226160640 source schema calls the same straight physical line
        # board_surface/board_contact; the manifest normalizes that spelling.
        _require(frame_pts == 226160640
                 and original.get("kind") in ("board_surface", "board_contact")
                 and target.get("kind") == "board_side"
                 and (original.get("side"), original.get("edge"))
                 == (target.get("side"), target.get("edge")),
                 f"Manifest changed measured physical identity: {selected['id']}")
    return target


def load_inputs(scene: dict, *, camera25_initializer_by_pts: Mapping[int, np.ndarray] | None = None,
                source_sha256: str = EXPECTED_SOURCE_SHA256,
                include_user_f1: bool = False) -> dict:
    """Return frozen source UV plus current-scene world curves for three exposures.

    Each ``references`` row follows ``continuous_reference_calibration``'s
    source-row names. ``camera25_initializer`` is None unless a caller supplies
    an explicit 25-value start; it never enters any returned measurement.
    The optional three user F1 contours preserve Step02's pointwise correlated
    integration weights and require a graph adapter that applies those weights.
    """
    _require(source_sha256 == EXPECTED_SOURCE_SHA256, "Programme source video differs")
    _require(type(include_user_f1) is bool, "include_user_f1 must be an explicit boolean")
    manifest = _read(MANIFEST)
    _require(manifest["schema"] == "rinkspace.low_end_reference_source_set.v1"
             and manifest["source_sha256"] == source_sha256
             and manifest["time_base"] == "1/15360"
             and [int(f["source_pts"]) for f in manifest["frames"]] == list(EXPECTED_PTS),
             "Low-end primary-set manifest/source identity changed")
    revision = manifest.get("world_binding_revision", {})
    _require(revision.get("previous_manifest_sha256")
             == "43aae95d3710a56e393480e6bb3a3024677a66afa1ac9858904f0a9f7b99d8b8"
             and revision.get("source_uv_sigma_grouping_changed") is False,
             "Right-side course has lost its original manifest provenance")
    seeds = {int(k): np.asarray(v, np.float64) for k, v in
             (camera25_initializer_by_pts or {}).items()}
    _require(set(seeds) <= set(EXPECTED_PTS), "Initializer has unknown low-end PTS")
    for pts, camera in seeds.items():
        _require(camera.shape == (25,) and np.isfinite(camera).all(),
                 f"Invalid camera25 initializer for {pts}")
    assignments_path, assignments = _source_file(
        "evidence/low_end_reference_review_20260923/selected_frames.json",
        "eb87be2f74b35752ffe204448bed2b47a135441f9a18afb7dde0e0a9946565e8")
    assignment_by_pts = {int(x["source_pts"]): x for x in assignments["frames"]}
    rows = []
    for entry in manifest["frames"]:
        pts = int(entry["source_pts"])
        image = _path(entry["image_path"])
        _require(_digest(image) == entry["image_sha256"], f"Reference image changed: {pts}")
        if pts in (14177280, 205992960):
            assignment = assignment_by_pts[pts]
            authority = assignment["classification_authority"]
            _require(assignment["camera_id"] == "low_end"
                     and authority in ("user_confirmed_camera_and_continuity",
                                       "direct_user_confirmation_by_complete_shot_time")
                     and assignment["image_sha256"] == entry["image_sha256"]
                     and assignment["source_sha256"] == source_sha256,
                     f"Low-end physical-camera identity changed: {pts}")
        else:
            identity = _read(_path(entry["camera_identity_file"]))
            _require(identity["source_pts"] == pts
                     and identity["direct_user_confirmations"]["camera"] == "low_end"
                     and identity["source_video"] == manifest["source_video"]
                     and identity["native_frame"] == image.name,
                     "226160640 camera identity changed")
            authority = "direct_user_confirmation_by_source_frame"
        measurement_specs = (entry.get("measurement_files") or [dict(
            path=entry["measurement_file"], sha256=entry["measurement_sha256"])])
        measurements = [_source_file(x["path"], x["sha256"])
                        for x in measurement_specs]
        curves = []
        for selected in entry["groups"]:
            file_index = int(selected.get("file_index", 0))
            _require(0 <= file_index < len(measurements), "Invalid source-file index")
            source_path, source = measurements[file_index]
            _require(int(source["source_pts"]) == pts, "Source observation PTS changed")
            if "source_frame_sha256" in source:
                _require(source["source_frame_sha256"] == entry["image_sha256"],
                         "Observation/image bytes differ")
            collection = source[selected["collection"]]
            matches = [x for x in collection if x["id"] == selected["id"]]
            _require(len(matches) == 1, f"Missing/duplicate source group: {selected['id']}")
            observation = matches[0]
            _require(observation.get("use") in (None, "fit"),
                     f"A source check-only curve entered the fit set: {selected['id']}")
            is_user_trace = selected["collection"] == "groups"
            if is_user_trace:
                _require(pts == 205992960
                         and source["schema_version"] == "rinkspace.user_board_ice_contact_samples.v1"
                         and source["source_frame_sha256"] == entry["image_sha256"]
                         and selected["target"] == RIGHT_CONTACT_COURSE,
                         "Unexpected user-drawn source group")
            else:
                _require("user_drawn" not in observation.get("evidence_role", ""),
                         f"Drawn markup entered native-read source pixels: {selected['id']}")
            target = _candidate_target(observation, selected, pts)
            if selected["collection"] == "neutral_ice_lines":
                hypothesis = source["competing_identity_hypotheses"][0]
                _require(hypothesis["authority"] == "user_confirmed_feature_labels"
                         and hypothesis["mapping"][selected["id"]] == target["feature_id"],
                         "C/B1 neutral-line identity lost its user confirmation")
            uv = np.asarray(observation["samples_uv" if is_user_trace else "pixels_uv"], np.float64)
            _require(uv.ndim == 2 and uv.shape[1] == 2 and len(uv) >= (1 if is_user_trace else 2)
                     and np.isfinite(uv).all()
                     and np.all((uv[:, 0] >= 0) & (uv[:, 0] < 1920))
                     and np.all((uv[:, 1] >= 0) & (uv[:, 1] < 1080)),
                     f"Invalid native source UV: {selected['id']}")
            sigma = float(observation["sigma_px"])
            _require(np.isfinite(sigma) and sigma > 0, "Invalid original sigma")
            curves.append(dict(
                id=selected["id"], target=target,
                physical_group_id=("low3_user_contact_one_stroke" if is_user_trace
                                   else json.dumps(target, sort_keys=True)),
                identity_status=("user_confirmed_contact_course; exact_corner_transition_unmarked"
                                 if is_user_trace else
                                 "user_confirmed_feature_identity; provisional_native_pixel_read"),
                source_uv=uv.copy(), world_xyz=_world_curve(scene, target),
                sigma_px=sigma, source_file=str(source_path),
                source_file_sha256=measurement_specs[file_index]["sha256"],
                source_collection=selected["collection"],
                source_measurement_role=("user_corrected_board_contact_stroke_sample"
                                         if is_user_trace else observation.get(
                                             "evidence_role", "independent_unannotated_native_frame_read")),
                source_correlation_group=("low3_user_contact_one_stroke" if is_user_trace else None),
                source_course_region=observation.get("course_region") if is_user_trace else None,
                source_visibility_note=observation.get("visibility"),
            ))
        expected_groups, expected_pixels = EXPECTED_COUNTS[pts]
        _require(len(curves) == expected_groups
                 and sum(len(c["source_uv"]) for c in curves) == expected_pixels,
                 f"Primary-set curve/pixel count changed: {pts}")
        if include_user_f1 and pts == 205992960:
            curves.extend(_optional_user_f1(scene, manifest, entry))
        # 226160640 board contact is copied from an independently preserved
        # direct native trace; require exact support instead of trusting labels.
        if pts == 226160640:
            trace_path, trace = _source_file(entry["contact_trace"],
                "419fd2bbfa774436446edd306e09adb4b92cba462a21f07e4d43214b1711e1ce")
            _require(trace["status"] == "PROVISIONAL_DIRECT_PIXEL_OBSERVATION"
                     and trace["source_frame_sha256"] == entry["image_sha256"],
                     "Direct contact trace source changed")
            observed = {tuple(p["xy"]) for segment in trace["segments"]
                        for p in segment["points"]}
            for curve in curves:
                if curve["target"].get("edge") == "contact":
                    _require(all(tuple(p) in observed for p in curve["source_uv"]),
                             f"Contact UV not in direct source trace: {curve['id']}")
        rows.append(dict(
            id=f"low_end_{pts}", source_pts=pts, source_time_s=pts/15360.,
            camera_id="low_end", raw_camera_id="low_end",
            installation_id="user_confirmed_low_end_installation",
            camera_assignment_authority=authority,
            image_path=str(image), image_sha256=entry["image_sha256"],
            source_observation_file=str(MANIFEST.resolve()),
            source_observation_sha256=_digest(MANIFEST),
            source_measurement_files=[dict(path=str(p), sha256=spec["sha256"])
                                      for (p, _), spec in zip(measurements, measurement_specs)],
            camera25_initializer=(seeds[pts].copy() if pts in seeds else None),
            curves=curves, direct_ad3_uv=None, derived_ad3_excluded=False,
            ad3_world_xyz=None, ad3_sigma_px=None,
            deferred_same_frame=list(entry.get("deferred_same_frame", [])),
        ))
    return dict(
        schema="rinkspace.low_end_reference_source_inventory.v1",
        source_video_sha256=source_sha256,
        source_set_path=str(MANIFEST.resolve()), source_set_sha256=_digest(MANIFEST),
        scene_binding="current_caller_scene_only",
        world_binding_revision=revision,
        camera_id="low_end", references=rows,
        counts=dict(reference_frames=3, curve_groups=sum(len(r["curves"]) for r in rows),
                    source_uv=sum(len(c["source_uv"]) for r in rows for c in r["curves"]),
                    by_pts={str(r["source_pts"]): dict(
                        groups=len(r["curves"]),
                        source_uv=sum(len(c["source_uv"]) for c in r["curves"]))
                        for r in rows}),
        previous_fit_uses="historical source references/experiments exist; none is a fresh blind holdout",
        optional_user_f1_enabled=include_user_f1,
        optional_user_f1_historical_correlated_budget=(
            "three rings share F1 group budget 1; original per-point arc-length"
            " weights sum to 1/3 per ring; numeric sigma=1 was not measured"
            if include_user_f1 else None),
        source_uv_modified=False, derived_camera_xyz_used=False,
    )
