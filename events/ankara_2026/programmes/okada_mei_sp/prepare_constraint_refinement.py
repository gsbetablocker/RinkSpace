"""Prepare a candidate projective-rank subset of current source rows; never fit.

The output root is the active prepared input. Its full_source directory links
the unchanged parent source input and numeric state for later comparison.
Projected geometry supplies only selection directions, never source pixels.
"""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import os
import shutil

import numpy as np

import prepare_native_multicamera as prepared
from prepare_spatial_refinement import _guard_path, _same_bytes, _write_prepared
from rinkspace.core.native import NativeCore
from rinkspace.domain.multi_shot_motion import ShotMotionTimeline
from rinkspace.domain.parameters import ParameterState, SceneParameterModel
from rinkspace.session import constraint_sampling
from rinkspace.session.gpu_geometry import NativeSceneGPU


def _physical_entities(native, path_ids):
    entities = {}
    for path_id in dict.fromkeys(path_ids):
        path = native.paths[path_id]
        if path["layer"] == "advertisement":
            entity = path_id.split("::", 1)[0]
        elif path["layer"] == "ice":
            entity = path["label"].removeprefix("ice.").split(":", 1)[0]
        elif path["layer"] == "board":
            entity = path_id
        else:
            raise ValueError("Unknown native physical layer: " + path["layer"])
        if not entity:
            raise ValueError("Native path has no physical entity: " + path_id)
        entities[path_id] = entity
    return np.asarray([entities[path_id] for path_id in path_ids])


def _camera_rows(manifest, count):
    camera_ids = np.empty(count, dtype=np.int32)
    begin = 0
    for shot in manifest["shots"]:
        if (shot["row_begin"] != begin or shot["row_stop"] < begin
                or shot["row_stop"] > count):
            raise ValueError("Prepared shots must cover contiguous original source rows")
        camera_index = shot["camera_index"]
        if not 0 <= camera_index < len(manifest["cameras"]):
            raise ValueError("Shot references an absent physical camera")
        camera_ids[begin:shot["row_stop"]] = camera_index
        begin = shot["row_stop"]
    if begin != count:
        raise ValueError("Prepared shots do not cover every original source row")
    return camera_ids


def _projected_normals(native, *, camera, camera_ids, motion_times, manifest,
                       state, observations, path_ids):
    stations = np.asarray(state["stations"])
    curve = np.asarray(observations["kind"]) == 1
    supports = {}
    for path_id in dict.fromkeys(path_ids):
        path = native.paths[path_id]
        if path["kind"] == 7:
            supports[path_id] = (-np.inf, np.inf)
            continue
        support = np.asarray(path.get("support", path.get("finite_support", (0., 1.))),
                             dtype=float)
        if (support.shape != (2,) or not np.isfinite(support).all()
                or support[0] > support[1]):
            raise ValueError("Invalid native path support: " + path_id)
        supports[path_id] = support
    low, high = np.asarray([supports[path_id] for path_id in path_ids]).T
    if (not np.isfinite(stations).all() or np.any(stations < low)
            or np.any(stations > high)):
        raise ValueError("Numeric station lies outside its current native path support")
    projected, timing = [], []
    for offset in (-1e-5, 1e-5):
        shifted = np.clip(stations + offset, low, high)
        with native.resident(camera=camera, rigs=state["rig"], camera_ids=camera_ids,
                controls=state["controls"], path_ids=path_ids, stations=shifted,
                pts=observations["pts"], observed=observations["uv"],
                motion_times=motion_times) as resident:
            resident.configure_projection_time_support(half_seconds=manifest["max_readout"] / 2)
            evaluated = resident.evaluate("cuda")
        values, status = np.asarray(evaluated["values"]), np.asarray(evaluated["status"])
        if (values.ndim != 2 or values.shape[0] != len(stations) or values.shape[1] < 5
                or status.shape != stations.shape):
            raise ValueError("Native projection did not preserve original row binding")
        if np.any(status[curve] != 0):
            raise ValueError("Native curve projection failed while deriving selection directions")
        projected.append(values[:, 3:5].copy())
        timing.append(evaluated.get("timings", {}))
    tangent = projected[1] - projected[0]
    length = np.hypot(tangent[:, 0], tangent[:, 1])
    if (not np.isfinite(tangent[curve]).all() or not np.isfinite(length[curve]).all()
            or np.any(length[curve] <= 0)):
        raise ValueError("Nonfinite or zero projected curve tangent; source rows cannot be excluded")
    normals = np.zeros_like(tangent)
    normals[curve] = np.c_[-tangent[curve, 1], tangent[curve, 0]] / length[curve, None]
    geometry_uv = .5*projected[0] + .5*projected[1]
    return normals, geometry_uv, timing


def _selected_shots(shots, indices):
    active = copy.deepcopy(shots)
    begin = 0
    for shot in active:
        count = int(np.count_nonzero((indices >= shot["row_begin"])
                                     & (indices < shot["row_stop"])))
        shot["row_begin"], shot["row_stop"] = begin, begin + count
        begin += count
    return active


def _entity_summary(groups):
    summary = {}
    for group in groups:
        key = str(group["physical_entity_id"])
        row = summary.setdefault(key, dict(exposures=0, available=0, unique_available=0,
                                           selected=0, rank_sum=0, rank_counts={}))
        row["exposures"] += 1
        for name in ("available", "unique_available", "selected"):
            row[name] += group[name]
        row["rank_sum"] += group["rank"]
        rank = str(group["rank"])
        row["rank_counts"][rank] = row["rank_counts"].get(rank, 0) + 1
    return summary


def _link_unchanged(source, target):
    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        shutil.copyfile(source, target)
        return "byte_copy"


def prepare(run_root, state_path, output):
    root, state_path = Path(run_root).resolve(), Path(state_path).resolve()
    output = _guard_path(output)
    if output.exists():
        raise FileExistsError("Preserve the completed or interrupted prepared output: " + str(output))
    if root == output or root.is_relative_to(output):
        raise ValueError("Output cannot replace or contain the parent prepared input")
    if output.is_relative_to(prepared.ROOT.resolve()):
        raise ValueError("Prepared observation outputs belong outside the OneDrive source tree")
    for directory in (output, output / "full_source"):
        for name in ("point_observations.npz", "initial_pixels.npz", "joint_manifest.json", "mapping.json"):
            _guard_path(directory / name)
    for name in ("selection.npz", "sampling_receipt.json"):
        _guard_path(output / name)

    manifest, mapping = prepared.read(root / "joint_manifest.json"), prepared.read(root / "mapping.json")
    point_path = root / "point_observations.npz"
    input_hash = prepared.digest(point_path)
    if input_hash != manifest["point_input_sha256"]:
        raise ValueError("Current source input digest differs")
    for path, expected in ((manifest["scene_path"], manifest["scene_file_sha256"]),
                           (manifest["native_library"], manifest["native_library_sha256"])):
        if prepared.digest(Path(path)) != expected:
            raise ValueError("Current model or native library digest differs")
    observations, state = prepared.npz(point_path), prepared.npz(state_path)
    if set(observations) != prepared.FIELDS:
        raise ValueError("Expected exactly seven unweighted original source fields")
    if not len(observations["pts"]):
        raise ValueError("Cannot prepare a projection selection without source rows")
    model = SceneParameterModel(prepared.read(manifest["scene_path"]),
                               Path(manifest["scene_path"]), optimize_scene=True)
    prepared.validate_snapshot(state, model.initial_state.layout, mapping, manifest, observations)
    protected = {str(path): prepared.digest(path) for path in
                 (root / "joint_manifest.json", root / "mapping.json", point_path, state_path)}
    timeline = ShotMotionTimeline(manifest["shots"], manifest["control_dt"], manifest["first_control_time"])
    motion_times = timeline.row_times(observations["pts"])
    timeline.assert_support_separation(observations["pts"], manifest["max_readout"])
    camera_ids = _camera_rows(manifest, len(observations["pts"]))
    cameras = [prepared.read(item["camera_definition"]) for item in manifest["cameras"]]
    image_size = [(item["image_width"], item["image_height"]) for item in cameras]
    if any(size != image_size[0] for size in image_size):
        raise ValueError("Physical cameras do not share the prepared source pixel domain")
    camera = {**cameras[0], "first_control_time": timeline.first, "control_dt": timeline.dt}
    native = NativeSceneGPU(manifest["scene_path"], NativeCore(manifest["native_library"]),
        parameter_state=ParameterState(model.initial_state.layout, state["scene_values"]))
    paths = [mapping["path_ids"][i] for i in observations["path_indices"]]
    entities = _physical_entities(native, paths)
    normals, geometry_uv, timing = _projected_normals(native, camera=camera, camera_ids=camera_ids,
        motion_times=motion_times, manifest=manifest, state=state,
        observations=observations, path_ids=paths)
    selection = constraint_sampling.sample_projective_constraint_rows(
        observations["uv"], normals, observations["pts"], entities,
        observations["observation_ids"], observations["kind"],
        image_width=image_size[0][0], image_height=image_size[0][1], geometry_uv=geometry_uv)
    indices, groups = selection["selected_indices"], selection["groups"]
    active = {name: values[indices] for name, values in observations.items()}
    numeric = state["stations"][indices]
    active_shots = _selected_shots(manifest["shots"], indices)
    for name in prepared.FIELDS:
        _same_bytes(active[name], observations[name][indices], "selected source " + name)
    _same_bytes(numeric, state["stations"][indices], "selected numeric stations")
    provenance = dict(schema="rinkspace.constraint_refinement_preparation.v1",
        parent_run_root=str(root), parent_input_sha256=input_hash,
        parent_numeric_state=str(state_path), parent_numeric_state_sha256=protected[str(state_path)],
        adapter_sha256=prepared.digest(Path(__file__)),
        sampler_sha256=prepared.digest(Path(constraint_sampling.__file__)),
        selection=str(output / "selection.npz"), sampling_receipt=str(output / "sampling_receipt.json"),
        rank_rtol=1e-6, geometry_used_only_for_selection=True, source_UV_changed=False,
        no_observation_weights=True, numeric_globals_changed=False,
        numeric_stations_changed=False, fit_executed=False)
    output.mkdir(parents=True)
    full_manifest, full_mapping = _write_prepared(output / "full_source", observations,
        state["stations"], manifest["shots"], state, manifest, mapping, model,
        dict(provenance, role="full_comparison_mother_set"), link_input=point_path)
    # _write_prepared validates and binds both forms. Replace only its newly
    # created full-state archive with the byte-identical supplied state link.
    full_state_path = output / "full_source" / "initial_pixels.npz"
    full_state_path.unlink()
    state_link = _link_unchanged(state_path, full_state_path)
    prepared.validate_snapshot(prepared.npz(full_state_path), model.initial_state.layout,
                               full_mapping, full_manifest, observations)
    if prepared.digest(full_state_path) != protected[str(state_path)]:
        raise ValueError("Full-source numeric state is not byte-identical to the supplied state")
    active_manifest, _ = _write_prepared(output, active, numeric, active_shots, state,
        manifest, mapping, model, dict(provenance, role="active_constraint_rank_subset",
        full_source_input_sha256=full_manifest["point_input_sha256"]))
    np.savez(output / "selection.npz", selected_indices=indices, normal_directions=normals,
        group_pts=np.asarray([group["pts"] for group in groups], dtype=observations["pts"].dtype),
        group_entity_ids=np.asarray([group["physical_entity_id"] for group in groups]),
        **{"group_" + name: np.asarray([group[name] for group in groups], dtype=np.int64)
           for name in ("rank", "available", "unique_available", "selected")})
    for path, expected in protected.items():
        if prepared.digest(Path(path)) != expected:
            raise ValueError("Parent input/state changed during preparation: " + path)
    receipt = dict(**provenance, parent_files_sha256=protected,
        parent_rows=len(observations["pts"]), active_rows=len(indices),
        full_source_input_sha256=full_manifest["point_input_sha256"],
        active_input_sha256=active_manifest["point_input_sha256"],
        full_source_state_link=state_link, original_fields_byte_preserved=True,
        selection_indices_refer_to="full_source/point_observations.npz",
        projection=dict(backend="cuda", evaluations=2, station_offsets=[-1e-5, 1e-5],
            clipped_to_native_finite_support=True, kind7_periodic_unclipped=True,
            rolling_shutter_half_seconds=manifest["max_readout"] / 2, timings=timing,
            flow_positions="mean_of_two_current_native_station_projections",
            geometry_uv_sha256=hashlib.sha256(geometry_uv.tobytes()).hexdigest(),
            normal_directions_sha256=hashlib.sha256(normals.tobytes()).hexdigest()),
        physical_entity_groups=_entity_summary(groups),
        source_points_created=False, optimizer_configured=False, production_sampler_replaced=False)
    (output / "sampling_receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(dict(output=str(output), parent_rows=len(observations["pts"]),
        active_rows=len(indices), fit_executed=False)), flush=True)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.run_root, args.state, args.output)
