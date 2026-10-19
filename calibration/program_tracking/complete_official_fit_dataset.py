"""All accepted official pixels replace the old inputs, with correlated FIT.

Source-only whitening is fixed during camera/model evaluation. No optimizer,
old observation graph, default pixel uncertainty, or new source point enters.
"""
from __future__ import annotations

from collections import OrderedDict
import json
from pathlib import Path

import cv2
import numpy as np

from .framewise_model import build_row, evaluate_rows, feature_world
from .official_static_surface_dataset import iter_observations, read_reference, reference
from .single_frame_source_review import scan_curve
from .source_evidence_whitening import (SourceWhiteningOperator, line_evidence_from_feature,
    line_evidence_from_readouts, line_whitening_operator, joint_line_corner_operator)

SCHEMA = "rinkspace.complete-accepted-official-static-fit.v2"
BINDING_SCHEMA = "rinkspace.complete-official-fit-dataset-binding.v1"
EXTENSION_SCHEMA = "rinkspace.first-shot-accepted-fit-extension.v3"


def camera_fit_state(camera25):
    """Existing 15-column Jacobian chart, with explicit conditioning units."""
    camera = np.asarray(camera25, float)
    if camera.shape != (25,) or not np.isfinite(camera).all():
        raise ValueError("An inherited finite camera25 is required")
    return np.r_[camera[:3]/100., camera[3:8], camera[8:11]/1000., camera[11:15]]


def camera_from_fit_state(state15, template25):
    """Preserve every fixed inherited camera/Brown term outside this chart."""
    state = np.asarray(state15, float)
    camera = np.asarray(template25, float).copy()
    if state.shape != (15,) or camera.shape != (25,) or not np.isfinite(state).all() or not np.isfinite(camera).all():
        raise ValueError("Explicit finite state15 and template25 required")
    camera[:3], camera[3:8], camera[8:11], camera[11:15] = state[:3]*100., state[3:8], state[8:11]*1000., state[11:15]
    return camera


def _address(row, j):
    return (row["native_pts"], row["source_feature_id"], row["source_segment_index"],
            row["original_vertex_indices"][j])


def _source_lines(observations):
    result = {}
    for row in observations:
        segment = row["source_segment"]
        for name, coefficient in segment.get("source_corner_line_equations", {}).items():
            if name.startswith("Ad2."):
                result[name.split(".", 1)[1]] = np.asarray(coefficient, float)
        names = {"top": "top_edge", "bottom": "bottom_edge", "left": "left_vertical", "right": "right_vertical"}
        for name, coefficient in segment.get("source_line_coefficients", {}).items():
            if name in names:
                result[names[name]] = np.asarray(coefficient, float)
        for name, uv in segment.get("source_line_uv", {}).items():
            if name in names:
                p = np.asarray(uv, float); axis = 1 if "vertical" in names[name] else 0
                a = (p[-1, 1-axis]-p[0, 1-axis])/(p[-1, axis]-p[0, axis])
                result[names[name]] = np.array([a, p[0, 1-axis]-a*p[0, axis]])
    return result


def _corner_edges(role):
    side, level = role.split("_")
    return level + "_edge", side + "_vertical"


def _supplement_line(raw, mask, coefficient, role, bounds, readouts, frame):
    """One actual extra readout identifies slope when retained evidence has one.

    The supplementary readout only propagates uncertainty. It is never added
    to the accepted FIT membership, and never moves an accepted UV.
    """
    axis = 1 if "vertical" in role else 0
    a, b = coefficient
    lo, hi = bounds
    anchors = np.array([[lo, a*lo+b], [hi, a*hi+b]])
    if axis == 1:
        anchors = anchors[:, ::-1]
    segments = scan_curve(raw, mask, dict(navigation_xy=anchors.tolist(), axis=axis,
        normal_radius_px=24, signal="magenta"))
    candidates = [(p, w) for s in segments for p, w in zip(s["_native_uv"], s["_native_iqr"], strict=True)]
    if not candidates:
        raise ValueError("Current source cannot supply missing line uncertainty: " + role)
    original_stations = np.array([r["uv"][axis] for r in readouts])
    candidates.sort(key=lambda pair: min(abs(pair[0][axis]-original_stations)), reverse=True)
    p, width = candidates[0]
    if min(abs(p[axis]-original_stations)) <= 0 or width <= 0:
        raise ValueError("Supplementary current-source readout cannot identify slope")
    item = dict(readout_id=f'{frame["native_pts"]}/{role}/uncertainty_support', uv=p,
        sigma_px=float(width/1.349), transition_iqr_px=float(width),
        origin="current_official_RGB_magenta_half_transition_for_uncertainty_only",
        source_image=frame["source_image"], added_as_FIT_point=False,
        original_source_UV_modified=False)
    readouts.append(item)
    return item


def _ad_operator(observations, primitive, frame):
    coefficients = _source_lines(observations)
    members, readouts, bounds = [], {}, {}
    for item in primitive:
        row, j, pid, scale = item
        descriptor = row["model_target"]
        if descriptor.get("advertisement_id") != 2 or descriptor["kind"] not in ("ad_edge", "ad_corner"):
            continue
        uv = row["uv"][j]
        if descriptor["kind"] == "ad_corner":
            h, v = _corner_edges(descriptor["role"])
            members.append(dict(point_id=pid, uv=uv.tolist(), kind="derived_intersection",
                horizontal_line_id=h, vertical_line_id=v))
            for role in (h, v):
                axis = 1 if "vertical" in role else 0
                bounds.setdefault(role, []).append(float(uv[axis]))
        else:
            role = descriptor["role"]
            readouts.setdefault(role, []).append(dict(readout_id=pid, uv=uv.tolist(), sigma_px=scale,
                origin="unchanged_accepted_native_edge_readout"))
            axis = 1 if "vertical" in role else 0
            bounds.setdefault(role, []).append(float(uv[axis]))
            members.append(dict(point_id=pid, uv=uv.tolist(), kind="direct_readout", readout_id=pid, line_id=role))
    if not members:
        return None, [], [], {}
    used_roles = set(readouts)
    for member in members:
        if member["kind"] == "derived_intersection":
            used_roles.update((member["horizontal_line_id"], member["vertical_line_id"]))
    raw = cv2.imread(frame["source_image"]["path"])
    mask = cv2.imread(frame["mask"]["path"], cv2.IMREAD_GRAYSCALE) != 0
    lines, supplementary = {}, []
    for role in used_roles:
        if role not in coefficients or role not in readouts:
            raise ValueError("Accepted corner has no actual source edge evidence: " + role)
        values = readouts[role]
        axis = 1 if "vertical" in role else 0
        if len({r["uv"][axis] for r in values}) < 2:
            supplementary.append(_supplement_line(raw, mask, coefficients[role], role,
                [min(bounds[role]), max(bounds[role])], values, frame))
        line = line_evidence_from_readouts([r["uv"] for r in values], [r["sigma_px"] for r in values],
            coefficients[role], axis=axis, group_id=f'{frame["native_pts"]}/Ad2/{role}',
            estimator="unweighted_soft_l1", soft_l1_f_scale=3., readout_ids=[r["readout_id"] for r in values])
        lines[role] = line
    for member in members:
        if member["kind"] == "direct_readout":
            line = lines[member["line_id"]]
            member["displacement_xy_per_unit_noise"] = (line.normal*line.normal[1-line.axis]).tolist()
    operator = joint_line_corner_operator(members, lines, group_id=f'{frame["native_pts"]}/Ad2/native_edges_and_derived_corners')
    return operator, members, supplementary, {role: line.metadata() for role, line in lines.items()}


def prepare_complete_dataset(scope_reference, scale_reference, output_path):
    scope = read_reference(scope_reference)
    recovery = read_reference(scale_reference)
    scale_records = {(r["native_pts"], r["source_feature_id"], r["source_segment_index"], r["source_vertex_index"]): r
                     for r in recovery["records"]}
    by_frame = OrderedDict()
    for row in iter_observations(scope_reference, role="all_scope"):
        by_frame.setdefault(row["native_pts"], []).append(row)
    frames = []
    for source_frame in scope["frames"]:
        observations = by_frame[source_frame["native_pts"]]
        primitive, frame_scales, locations = [], {}, {}
        cursor = 0
        for row in observations:
            for j, pid in enumerate(row["source_point_ids"]):
                address = _address(row, j)
                value = float(row["sigma"][j])
                if address in scale_records:
                    recovered = scale_records[address]
                    if (recovered["source_packet"] != row["source_packet"]
                            or recovered["source_point_id"] != pid or recovered["uv"] != row["uv"][j].tolist()):
                        raise ValueError("Recovered uncertainty differs from accepted source address")
                    value = recovered["sigma_px"]
                primitive.append((row, j, pid, value))
                frame_scales[pid] = value
                locations[pid] = cursor
                cursor += 1
        groups, grouped = [], set()
        line_rows = OrderedDict()
        for row in observations:
            segment = row["source_segment"]
            if (segment.get("representative_points_share_line_estimate") is True
                    or segment.get("inferred_from_current_source_readouts") is True):
                line_rows.setdefault(row["source_feature_id"], []).append(row)
        for fid, values in line_rows.items():
            uv = np.concatenate([r["uv"] for r in values])
            ids = [pid for r in values for pid in r["source_point_ids"]]
            line = line_evidence_from_feature(values[0]["source_feature"], group_id=f'{source_frame["native_pts"]}/{fid}')
            operator = line_whitening_operator(uv, ids, line)
            groups.append(dict(operator=operator.to_record(), point_indices=[locations[p] for p in ids],
                line_evidence=line.metadata()))
            grouped.update(ids)
            for point, pid in zip(uv, ids, strict=True):
                design = np.array([point[line.axis], 1.])
                frame_scales[pid] = float(np.sqrt(design @ line.covariance_ab @ design))
        operator, members, supplementary, line_evidence = _ad_operator(observations, primitive, source_frame)
        if operator is not None:
            ids = list(operator.source_point_ids)
            if grouped.intersection(ids):
                raise ValueError("Source evidence entered two FIT groups")
            groups.append(dict(operator=operator.to_record(), point_indices=[locations[p] for p in ids],
                members=members, supplementary_uncertainty_readouts=supplementary, line_evidence=line_evidence))
            grouped.update(ids)
            # These positive source-propagated diagonal scales only bridge the
            # existing pixel residual API. Joint W, not these scalars, is FIT.
            gram = operator.matrix.T @ operator.matrix
            covariance = np.linalg.pinv(gram, hermitian=True)
            for i, pid in enumerate(ids):
                frame_scales[pid] = float(np.sqrt(np.trace(covariance[2*i:2*i+2, 2*i:2*i+2])))
        if any(not np.isfinite(s) or s <= 0 for s in frame_scales.values()):
            missing = [p for p, s in frame_scales.items() if not np.isfinite(s) or s <= 0]
            raise ValueError("Accepted FIT source uncertainty is still incomplete: " + str(missing))
        frames.append(dict(index=source_frame["index"], native_pts=source_frame["native_pts"],
            point_count=cursor, source_scales_by_point_id=frame_scales, whitening_groups=groups,
            source_packet=source_frame["packet"], source_binding=source_frame["binding"]))
    if [f["index"] for f in frames] != list(range(1, 201)):
        raise ValueError("The exact accepted first200 replacement is required")
    total = sum(f["point_count"] for f in frames)
    if total != sum(f["original_points"] for f in scope["frames"]):
        raise ValueError("An approved point was excluded or appended")
    result = dict(schema=SCHEMA, scope_selection=scope_reference, recovered_source_scales=scale_reference,
        model=scope["model"], source_clock=scope["source_clock"], parent_complete_state=scope["parent_complete_state"],
        parent_q_sha256=scope["parent_q_sha256"], frame_count=len(frames), fit_points=total, frames=frames,
        replacement=dict(mode="replace_all_previous_points_for_exact_official_exposures", frame_range=[1, 200],
            all_previous_active_points_removed=True, accepted_packet_only=True,
            historical_files_preserved=True, legacy_objectives_enabled=False, append_old_points=False),
        ordinary_points_and_correlated_points_all_in_FIT=True, missing_source_scales=0,
        supplementary_readouts_added_as_FIT_points=0, unknown_depth_points_admitted=0,
        all_uncertainty_operators_fixed_from_source_only=True,
        source_scales_are_localization_support_not_calibrated_noise=True,
        independent_HELD_created=False, source_UV_modified=False, solver_started=False, accuracy_accepted=False)
    with Path(output_path).open("x", encoding="utf8") as stream:
        json.dump(result, stream, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return reference(output_path)


def iter_fit_frames(dataset_reference):
    data = read_reference(dataset_reference)
    if data["schema"] == BINDING_SCHEMA:
        dataset_reference = data["complete_fit_input"]
        data = read_reference(dataset_reference)
    if data["schema"] == EXTENSION_SCHEMA:
        yield from _iter_extension_frames(data)
        return
    if data["schema"] != SCHEMA or data["replacement"]["append_old_points"]:
        raise ValueError("Only the accepted replacement FIT dataset may be loaded")
    scope = read_reference(data["scope_selection"])
    read_reference(data["recovered_source_scales"])
    model = read_reference(data["model"])
    scene, templates = read_reference(model["scene"]), read_reference(model["material_templates"])
    frames = {f["native_pts"]: f for f in data["frames"]}
    if (len(frames) != len(data["frames"]) or [f["index"] for f in data["frames"]] != list(range(1, 201))
            or len(data["frames"]) != data["frame_count"] or sum(f["point_count"] for f in data["frames"]) != data["fit_points"]):
        raise ValueError("Duplicate, missing or inconsistent accepted replacement frame inventory")
    expected_frames = {f["native_pts"]: f for f in scope["frames"]}
    if set(frames) != set(expected_frames):
        raise ValueError("Replacement source exposure clock changed")
    for pts, frame in frames.items():
        expected = expected_frames[pts]
        if (frame["source_packet"] != expected["packet"] or frame["source_binding"] != expected["binding"]
                or frame["point_count"] != expected["original_points"]):
            raise ValueError("Replacement admitted an old or foreign source inventory")
    current, rows = None, []
    for observation in iter_observations(data["scope_selection"], role="all_scope"):
        pts = observation["native_pts"]
        if current is not None and pts != current:
            yield _fit_frame(frames[current], rows, scene, templates)
            rows = []
        current = pts
        sigma = np.array([frames[pts]["source_scales_by_point_id"][pid] for pid in observation["source_point_ids"]])
        row = build_row(scene, observation["model_target"], observation["uv"], sigma, templates,
            row_id=observation["binding_observation_id"], point_correspondence=observation["point_correspondence"])
        row.update({k: observation[k] for k in ("native_pts", "actual_pts", "time_base", "source_video_sha256",
            "source_packet", "source_point_ids", "source_feature_id", "source_segment_index", "original_vertex_indices", "dependence_group")})
        rows.append(row)
    if current is not None:
        yield _fit_frame(frames[current], rows, scene, templates)


def _direct_dataset(reference_value):
    data = read_reference(reference_value)
    if data["schema"] == BINDING_SCHEMA:
        return data["complete_fit_input"], read_reference(data["complete_fit_input"])
    return reference_value, data


def _compatible_prefix_model(base_scope, appended_scope):
    """One common model; no prior FIT target changes with the extension."""
    base_model, new_model = read_reference(base_scope["model"]), read_reference(appended_scope["model"])
    old_scene, old_registry, old_templates = [read_reference(base_model[k]) for k in ("scene", "feature_registry", "material_templates")]
    scene, registry, templates = [read_reference(new_model[k]) for k in ("scene", "feature_registry", "material_templates")]
    used = {o["target_id"] for f in base_scope["frames"] for o in f["observations"]}
    for target_id in used:
        old, new = old_registry["targets"][target_id], registry["targets"].get(target_id)
        if old != new or not np.array_equal(feature_world(old_scene, old, old_templates), feature_world(scene, new, templates)):
            raise ValueError("The common model changed an existing accepted prefix target: " + target_id)
    return scene, templates


def _extension_contract(base, base_scope, appended_scope):
    if base["schema"] != SCHEMA or [f["index"] for f in base["frames"]] != list(range(1, 201)):
        raise ValueError("The unchanged accepted first200 FIT base is required")
    if (base["parent_q_sha256"] != appended_scope["parent_q_sha256"]
            or base["parent_complete_state"] != appended_scope["parent_complete_state"]
            or base["source_clock"] != appended_scope["source_clock"]
            or base_scope["video"] != appended_scope["video"]):
        raise ValueError("Extension changed the source clock, video or complete parent state")
    appended = appended_scope["frames"]
    if not appended or [f["index"] for f in appended] != list(range(201, appended[-1]["index"]+1)):
        raise ValueError("Missing, duplicate or reordered accepted extension frames")
    last_pts = base["frames"][-1]["native_pts"]
    if [f["native_pts"] for f in appended] != [last_pts+256*(i+1) for i in range(len(appended))]:
        raise ValueError("Extension is not contiguous on the official native exposure clock")


def extend_complete_dataset(base_reference, appended_scope_reference, output_path, *, frame_evidence_builder=None):
    """Reuse every prior FIT record verbatim; append only the approved range."""
    base_reference, base = _direct_dataset(base_reference)
    base_scope, appended_scope = read_reference(base["scope_selection"]), read_reference(appended_scope_reference)
    _extension_contract(base, base_scope, appended_scope)
    scene, templates = _compatible_prefix_model(base_scope, appended_scope)
    if frame_evidence_builder is None:
        from .first_shot_source_evidence import prepare_frame_evidence
        frame_evidence_builder = prepare_frame_evidence
    by_frame = OrderedDict()
    for observation in iter_observations(appended_scope_reference, role="all_scope"):
        by_frame.setdefault(observation["native_pts"], []).append(observation)
    appended_frames = []
    for source_frame in appended_scope["frames"]:
        observations = by_frame[source_frame["native_pts"]]
        ids = [pid for row in observations for pid in row["source_point_ids"]]
        if len(ids) != source_frame["original_points"] or len(set(ids)) != len(ids):
            raise ValueError("Approved extension omitted or duplicated source points")
        evidence = frame_evidence_builder(observations, source_frame)
        scales = evidence["source_scales_by_point_id"]
        if set(scales) != set(ids) or any(not np.isfinite(s) or s <= 0 for s in scales.values()):
            raise ValueError("Extension has missing, extra or invalid source scales")
        record = dict(index=source_frame["index"], native_pts=source_frame["native_pts"], point_count=len(ids),
            source_scales_by_point_id=scales, whitening_groups=evidence["whitening_groups"],
            source_packet=source_frame["packet"], source_binding=source_frame["binding"])
        if "uncertainty_evidence" in evidence:
            record["uncertainty_evidence"] = evidence["uncertainty_evidence"]
        rows = _assemble_observations(observations, record, scene, templates)
        _fit_frame(record, rows, scene, templates)
        appended_frames.append(record)
        if source_frame["index"] % 100 == 0:
            print("Prepared accepted FIT extension frame", source_frame["index"], flush=True)
    result = dict(schema=EXTENSION_SCHEMA, base_dataset=base_reference,
        appended_scope_selection=appended_scope_reference, appended_frames=appended_frames,
        model=appended_scope["model"], source_clock=base["source_clock"],
        parent_q_sha256=base["parent_q_sha256"], parent_complete_state=base["parent_complete_state"],
        frame_count=len(base["frames"])+len(appended_frames),
        frame_range=[1, appended_frames[-1]["index"]],
        fit_points=base["fit_points"]+sum(f["point_count"] for f in appended_frames),
        prefix_fit_points=base["fit_points"], added_fit_points=sum(f["point_count"] for f in appended_frames),
        prefix_frame_records_reused_unchanged=True, prefix_UV_scales_and_operators_changed=False,
        single_compatible_model_for_all_frames=True, supplementary_readouts_added_as_FIT_points=0,
        legacy_objectives_enabled=False, old_or_unaccepted_points_appended=0,
        source_UV_modified=False, missing_source_scales=0, solver_started=False, accuracy_accepted=False)
    with Path(output_path).open("x", encoding="utf8") as stream:
        json.dump(result, stream, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return reference(output_path)


def _assemble_observations(observations, record, scene, templates):
    ids = [pid for observation in observations for pid in observation["source_point_ids"]]
    if set(record["source_scales_by_point_id"]) != set(ids):
        raise ValueError("FIT scales do not exactly cover the accepted source inventory")
    rows = []
    for observation in observations:
        sigma = np.array([record["source_scales_by_point_id"][pid] for pid in observation["source_point_ids"]])
        row = build_row(scene, observation["model_target"], observation["uv"], sigma, templates,
            row_id=observation["binding_observation_id"], point_correspondence=observation["point_correspondence"])
        row.update({k: observation[k] for k in ("native_pts", "actual_pts", "time_base", "source_video_sha256",
            "source_packet", "source_point_ids", "source_feature_id", "source_segment_index", "original_vertex_indices", "dependence_group")})
        rows.append(row)
    return rows


def _iter_extension_frames(data):
    base_reference, base = _direct_dataset(data["base_dataset"])
    base_scope = read_reference(base["scope_selection"])
    appended_scope = read_reference(data["appended_scope_selection"])
    _extension_contract(base, base_scope, appended_scope)
    scene, templates = _compatible_prefix_model(base_scope, appended_scope)
    appended = data["appended_frames"]
    expected = appended_scope["frames"]
    if (len(appended) != len(expected) or data["frame_count"] != len(base["frames"])+len(appended)
            or data["fit_points"] != base["fit_points"]+sum(f["point_count"] for f in appended)
            or data["added_fit_points"] != sum(f["point_count"] for f in appended)
            or data["frame_range"] != [1, expected[-1]["index"]]
            or data["prefix_fit_points"] != base["fit_points"] or data["model"] != appended_scope["model"]
            or data["parent_q_sha256"] != base["parent_q_sha256"] or data["source_clock"] != base["source_clock"]
            or data["parent_complete_state"] != base["parent_complete_state"]
            or data["prefix_frame_records_reused_unchanged"] is not True
            or data["prefix_UV_scales_and_operators_changed"] is not False
            or data["source_UV_modified"] is not False
            or data["legacy_objectives_enabled"] is not False
            or data["old_or_unaccepted_points_appended"] != 0):
        raise ValueError("Extension manifest changed the accepted prefix or appended inventory")
    records = {}
    for record, original in zip(appended, expected, strict=True):
        if (record["index"] != original["index"] or record["native_pts"] != original["native_pts"]
                or record["source_packet"] != original["packet"] or record["source_binding"] != original["binding"]
                or record["point_count"] != original["original_points"] or record["native_pts"] in records):
            raise ValueError("Missing, duplicate or foreign accepted extension frame")
        records[record["native_pts"]] = record
    for frame in iter_fit_frames(base_reference):
        frame["scene"], frame["templates"] = scene, templates
        yield frame
    current, observations = None, []
    for observation in iter_observations(data["appended_scope_selection"], role="all_scope"):
        pts = observation["native_pts"]
        if current is not None and pts != current:
            yield _fit_frame(records[current], _assemble_observations(observations, records[current], scene, templates), scene, templates)
            observations = []
        current = pts
        observations.append(observation)
    if current is not None:
        yield _fit_frame(records[current], _assemble_observations(observations, records[current], scene, templates), scene, templates)


def _fit_frame(record, rows, scene, templates):
    ids = [pid for row in rows for pid in row["source_point_ids"]]
    if len(ids) != record["point_count"] or len(set(ids)) != len(ids):
        raise ValueError("The replacement FIT inventory lost or duplicated approved points")
    groups = []
    seen = set()
    for group in record["whitening_groups"]:
        operator = SourceWhiteningOperator.from_record(group["operator"])
        indices = group["point_indices"]
        if (any(type(i) is not int or not 0 <= i < len(ids) for i in indices)
                or len(set(indices)) != len(indices)):
            raise ValueError("Joint FIT point indices must be unique valid original addresses")
        if tuple(ids[i] for i in indices) != operator.source_point_ids or seen.intersection(indices):
            raise ValueError("Joint FIT group changed accepted point identities")
        seen.update(indices)
        groups.append(dict(operator=operator, point_indices=indices))
    if any(row["source_packet"] != record["source_packet"] for row in rows):
        raise ValueError("An old/foreign source packet entered the first200 replacement")
    return dict(native_pts=record["native_pts"], rows=rows, whitening_groups=groups,
        source_point_count=len(ids), scene=scene, templates=templates)


def evaluate_frame(camera25, frame, jacobian=True, *, scene=None, templates=None):
    """FIT r/J for all accepted points; J uses camera_fit_state's 15 columns."""
    rows = frame["rows"]
    result = evaluate_rows(camera25, rows, jacobian=jacobian,
        scene=scene if scene is not None else frame.get("scene"),
        templates=templates if templates is not None else frame.get("templates"))
    residual, derivative = result if jacobian else (result, None)
    scales = np.repeat(np.concatenate([row["sigma"] for row in rows]), 2)
    used, parts, jacobians = set(), [], []
    for group in frame["whitening_groups"]:
        indices = np.ravel([[2*i, 2*i+1] for i in group["point_indices"]])
        if used.intersection(indices):
            raise ValueError("A source scalar entered multiple FIT operators")
        used.update(indices)
        raw_r = residual[indices]*scales[indices]
        if jacobian:
            r, j = group["operator"].apply(raw_r, derivative[indices]*scales[indices, None])
            jacobians.append(j)
        else:
            r = group["operator"].apply(raw_r)
        parts.append(r)
    independent = np.array([i for i in range(len(residual)) if i not in used], int)
    parts.append(residual[independent])
    if jacobian:
        jacobians.append(derivative[independent])
    output = np.concatenate(parts)
    return (output, np.vstack(jacobians)) if jacobian else output
