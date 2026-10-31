"""Append qualified source pixels and prepare a spatial subset; never run a fit.

The full_source directory is the common comparison mother set. The output root
contains only rows selected from it by the shared source-UV coverage sampler.
No original observation, numeric global or motion chart is rewritten.
Optional shot strides change future fit columns without resampling controls.
Optional print-instance coverage shares one source-UV cover across an
advertisement's paths; it does not establish direction coverage or accuracy.
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
from apply_native_multicamera_review import revise_observations
from rinkspace.domain.motion_chart import UniformMotionChart
from rinkspace.domain.parameters import SceneParameterModel, ParameterState
from rinkspace.session.gpu_geometry import NativeSceneGPU, initialize_curve_stations
from rinkspace.session.spatial_sampling import SamplingReason, sample_source_pixels


def _write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                          encoding="utf-8")


def _jsonl(path, records):
    with Path(path).open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def _guard_path(path):
    """Check complete Windows UTF-16 paths before creating OneDrive files."""
    path = Path(path).resolve()
    if path.is_relative_to(prepared.ROOT.resolve()):
        if len(str(path).encode("utf-16-le")) // 2 >= 260:
            raise ValueError("OneDrive absolute path must be shorter than 260 UTF-16 units")
        if any(len(part.encode("utf-16-le")) // 2 > 255 for part in path.parts):
            raise ValueError("OneDrive name exceeds 255 UTF-16 units")
    return path


def _same_bytes(actual, expected, label):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if (actual.dtype != expected.dtype or actual.shape != expected.shape or
            actual.tobytes() != expected.tobytes()):
        raise ValueError("Protected array changed: " + label)


def _preserve_old_rows(new, original, numeric, old_numeric, origins):
    old_positions = np.flatnonzero(origins >= 0)
    if len(old_positions) != len(original["pts"]):
        raise ValueError("Append-only preparation lost an original row")
    if not np.array_equal(np.sort(origins[old_positions]), np.arange(len(old_positions))):
        raise ValueError("Original row identity is not one-to-one")
    for begin in range(0, len(old_positions), 32768):
        rows = old_positions[begin:begin + 32768]
        parent = origins[rows]
        for name in prepared.FIELDS:
            _same_bytes(new[name][rows], original[name][parent], name)
        _same_bytes(numeric[rows], old_numeric[parent], "original numeric stations")


def _pixel_key(pts, path_index, uv):
    return int(pts), int(path_index), float(uv[0]), float(uv[1])


def _measurement_id(shot, pts, path_id, uv):
    # Identity depends on the source location, never packet order or its filename.
    payload = [shot, int(pts), path_id,
               *(float(value if value != 0 else 0).hex() for value in uv)]
    digest = hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()
    return "src_" + digest


def _packet_path(value, packet):
    value = Path(value)
    return value.resolve() if value.is_absolute() else (packet.parent / value).resolve()


def collect_additions(packet_paths, original, manifest, path_ids):
    """Check only the newly supplied qualification dependencies; preserve aliases."""
    path_lookup = {value: i for i, value in enumerate(path_ids)}
    shots = {shot["id"]: shot for shot in manifest["shots"]}
    exposures = set(map(int, np.unique(original["pts"])))
    incoming, sources = [], []
    for packet in sorted({Path(path).resolve() for path in packet_paths}, key=str):
        data = prepared.read(packet)
        if data.get("fit_eligible") is not True:
            raise ValueError("Source packet is not explicitly fit_eligible: " + str(packet))
        if data.get("source_UV_synthesized") is True:
            raise ValueError("Synthetic source pixels are not eligible")
        if data.get("shot") not in shots or not isinstance(data.get("accepted"), list):
            raise ValueError("Source packet needs a current shot and accepted rows")
        dependencies = {}
        for name in ("source_packet", "source_links"):
            dependency = _packet_path(data[name], packet)
            digest = prepared.digest(dependency)
            if digest != data[name + "_sha256"]:
                raise ValueError("Source qualification dependency changed: " + str(dependency))
            dependencies[name] = dict(path=str(dependency), sha256=digest)
        packet_hash = prepared.digest(packet)
        sources.append(dict(path=str(packet), sha256=packet_hash, shot=data["shot"],
                            accepted_rows=len(data["accepted"]), dependencies=dependencies,
                            fit_eligible=True))
        for number, record in enumerate(data["accepted"]):
            row = copy.deepcopy(record)
            if row.get("fit_eligible") is False or row.get("model_UV_synthesized") is True:
                raise ValueError("An accepted row is explicitly ineligible or synthetic")
            pts = row["pts"]
            if isinstance(pts, bool) or not isinstance(pts, int) or pts not in exposures:
                raise ValueError("New source row must name an existing integer source PTS")
            shot = shots[data["shot"]]
            owners = [s["id"] for s in manifest["shots"]
                      if s["first_pts"] <= pts <= s["last_pts"]]
            if owners != [shot["id"]]:
                raise ValueError("New source row does not belong to its declared shot")
            path_id = row["path_id"]
            if path_id not in path_lookup:
                raise ValueError("Source path is absent from the current mapping: " + path_id)
            uv = np.asarray(row["uv"], dtype=np.float64)
            if (uv.shape != (2,) or not np.isfinite(uv).all() or
                    np.any(uv < 0) or np.any(uv > [1919, 1079])):
                raise ValueError("Invalid native source UV")
            if not np.array_equal(uv.astype(original["uv"].dtype), uv):
                raise ValueError("New UV would be changed by the current source dtype")
            evidence = _packet_path(row["evidence_image"], packet)
            if not evidence.is_file():
                raise ValueError("Source evidence image is missing: " + str(evidence))
            kind = row.get("kind", 1)
            if isinstance(kind, bool) or kind not in (0, 1):
                raise ValueError("Unknown source kind")
            if kind == 0 and "station" not in row:
                raise ValueError("A fixed point needs its explicitly identified station")
            if "observation_id" in row:
                row["packet_observation_id"] = row["observation_id"]
            row.update(observation_id=_measurement_id(shot["id"], pts, path_id, uv),
                       uv=uv.tolist(), kind=int(kind), shot=shot["id"],
                       evidence_image=str(evidence), source_verified_packet=str(packet),
                       source_verified_packet_sha256=packet_hash, packet_row=number,
                       reason=row.get("reason") or
                       "Qualified original source pixel; referenced material and source-link packet.")
            incoming.append(row)

    if incoming:
        ids_dtype = original["observation_ids"].dtype
        width = ids_dtype.itemsize // 4 if ids_dtype.kind == "U" else ids_dtype.itemsize
        if ids_dtype.kind not in "US" or width < 68:
            raise ValueError("Stable source IDs cannot preserve the original ID dtype")
    groups = {(row["pts"], path_lookup[row["path_id"]]) for row in incoming}
    parent_pixels = {}
    if groups:
        possible = np.flatnonzero(
            np.isin(original["pts"], list({key[0] for key in groups})) &
            np.isin(original["path_indices"], list({key[1] for key in groups})))
        for i in possible:
            key = _pixel_key(original["pts"][i], original["path_indices"][i], original["uv"][i])
            if key[:2] in groups:
                parent_pixels.setdefault(key, int(i))
    additions, aliases, measured = [], [], {}
    existing_ids = set(map(str, original["observation_ids"]))
    for row in incoming:
        key = _pixel_key(row["pts"], path_lookup[row["path_id"]], row["uv"])
        alias = dict(source_record=row)
        if key in parent_pixels:
            parent = parent_pixels[key]
            aliases.append(dict(alias, reason="exact_parent_PTS_path_UV",
                                kept_observation_id=str(original["observation_ids"][parent]),
                                parent_row=parent))
        elif key in measured:
            kept = measured[key]
            if row["kind"] != kept["kind"] or row.get("station", .5) != kept.get("station", .5):
                raise ValueError("Duplicate new source pixels have conflicting point semantics")
            aliases.append(dict(alias, reason="exact_new_PTS_path_UV",
                                kept_observation_id=kept["observation_id"]))
        else:
            if row["observation_id"] in existing_ids:
                raise ValueError("Stable new observation ID collides with a different source row")
            existing_ids.add(row["observation_id"])
            measured[key] = row
            additions.append(row)
    order = {shot["id"]: i for i, shot in enumerate(manifest["shots"])}
    additions.sort(key=lambda row: (order[row["shot"]], row["pts"], row["path_id"], *row["uv"]))
    return additions, aliases, sources


def _initialize_touched(new, numeric, shots, touched, state, manifest, mapping, model):
    notes = {}
    if not np.any(touched & (new["kind"] == 1)):
        return notes
    from rinkspace.core.native import NativeCore
    native = NativeSceneGPU(manifest["scene_path"], NativeCore(manifest["native_library"]),
                           parameter_state=ParameterState(model.initial_state.layout,
                                                          state["scene_values"]))
    for shot in shots:
        rows = np.arange(shot["row_begin"], shot["row_stop"])
        rows = rows[touched[rows] & (new["kind"][rows] == 1)]
        if not len(rows):
            continue
        camera = {**prepared.read(manifest["cameras"][shot["camera_index"]]["camera_definition"]),
                  "rig": state["rig"][shot["camera_index"]].tolist(),
                  "reference_pts": shot["reference_pts"],
                  "time_base_num": shot["time_base_num"], "time_base_den": shot["time_base_den"],
                  "control_dt": manifest["control_dt"],
                  "first_control_time": shot["first_control_time"]}
        seeded, note = initialize_curve_stations(
            native, camera=camera,
            controls=state["controls"][shot["control_begin"]:shot["control_stop"]],
            path_ids=[mapping["path_ids"][k] for k in new["path_indices"][rows]],
            stations=numeric[rows], pts=new["pts"][rows], observed=new["uv"][rows],
            kinds=new["kind"][rows], half_seconds=manifest["max_readout"] / 2)
        if note["unresolved"]:
            raise ValueError("New source rows have unresolved station initialization")
        numeric[rows] = seeded
        notes[shot["id"]] = dict(note, initialized_rows=len(rows), parent_rows_initialized=0)
    return notes


def _coverage(observations, shots, path_metadata):
    result = {}
    entities = np.asarray([item["entity"] for item in path_metadata])
    for shot in shots:
        rows = slice(shot["row_begin"], shot["row_stop"])
        pts, paths = observations["pts"][rows], observations["path_indices"][rows]
        physical = entities[paths]
        counts = {}
        for entity in np.unique(physical):
            member = physical == entity
            times = np.unique(pts[member])
            counts[str(entity)] = dict(rows=int(member.sum()), exposures=len(times),
                                       first_pts=int(times[0]), last_pts=int(times[-1]))
        result[shot["id"]] = dict(rows=len(pts), exposures=len(np.unique(pts)), entities=counts)
    return result


def _write_prepared(folder, observations, numeric, shots, parent_state, parent_manifest,
                    parent_mapping, model, provenance, link_input=None):
    folder.mkdir(parents=True, exist_ok=True)
    point_path = folder / "point_observations.npz"
    if link_input is None:
        np.savez(point_path, **observations)
    else:
        try:
            os.link(link_input, point_path)
        except OSError:
            shutil.copyfile(link_input, point_path)
    digest = prepared.digest(point_path)
    mapping = copy.deepcopy(parent_mapping)
    mapping["point_input_sha256"] = digest
    # Update count/hash fields when a parent supplied them, without stale metadata.
    for name in ("input_sha256",):
        if name in mapping:
            mapping[name] = digest
    for name in ("rows", "row_count", "observation_count"):
        if name in mapping:
            mapping[name] = len(observations["pts"])
    manifest = copy.deepcopy(parent_manifest)
    manifest.update(shots=copy.deepcopy(shots), point_input_sha256=digest,
                    current_resume_snapshot=str(folder / "initial_pixels.npz"),
                    spatial_refinement_preparation=provenance)
    if "point_input_path" in manifest:
        manifest["point_input_path"] = str(point_path)
    state = dict(parent_state)
    state["stations"], state["source_rows"] = numeric, observations["source_rows"]
    prepared.bind_snapshot(state, model.initial_state.layout, mapping, manifest)
    prepared.validate_snapshot(state, model.initial_state.layout, mapping, manifest, observations)
    mutable = {"stations", "source_rows", "joint_numeric_dependencies_json", "point_input_sha256"}
    for name in parent_state.keys() - mutable:
        _same_bytes(state[name], parent_state[name], "numeric global " + name)
    np.savez(folder / "initial_pixels.npz", **state)
    _write(folder / "joint_manifest.json", manifest)
    _write(folder / "mapping.json", mapping)
    return manifest, mapping


def _apply_motion_stride(shots, specifications, control_dt):
    """Change only future update columns, never the stored native trajectory.

    Native projection consumes the existing controls and storage control_dt;
    knot_stride is used only when ShotMotionTimeline constructs fit directions.
    No coarse-chart initializer or control interpolation belongs here.
    """
    lookup = {shot["id"]: shot for shot in shots}
    requested = {}
    for specification in specifications or ():
        if not isinstance(specification, str) or specification.count("=") != 1:
            raise ValueError("Motion stride must use SHOT=STRIDE")
        name, text = specification.split("=")
        try:
            stride = int(text)
        except ValueError as error:
            raise ValueError("Motion stride must be a positive integer") from error
        if name not in lookup:
            raise ValueError("Unknown motion-stride shot: " + name)
        if stride < 1 or name in requested:
            raise ValueError("Motion stride must be positive and each shot specified once")
        shot = lookup[name]
        UniformMotionChart.create(
            shot["control_stop"] - shot["control_begin"], shot["first_control_time"],
            control_dt, stride, shot["motion_reference_time"])
        requested[name] = stride
    changes = []
    for name, stride in requested.items():
        shot = lookup[name]
        old = shot["knot_stride"]
        if old != stride:
            changes.append(dict(shot=name, old_stride=old, new_stride=stride,
                                old_active_interval_seconds=old * control_dt,
                                new_active_interval_seconds=stride * control_dt))
            shot["knot_stride"] = stride
    return changes


def _sampling_group_map(path_ids, *, print_instance_coverage=False):
    """Map native paths to selection groups without changing source bindings."""
    if not isinstance(print_instance_coverage, (bool, np.bool_)):
        raise ValueError("print_instance_coverage must be an explicit boolean")
    lookup, groups = {}, []
    for path_index, path_id in enumerate(path_ids):
        key = ("native_path", path_index)
        if print_instance_coverage and "::" in path_id:
            instance = path_id.split("::", 1)[0]
            if instance.startswith("print_instance/Ad") and instance[len("print_instance/Ad"):].isdigit():
                key = ("print_instance", instance)
        if key not in lookup:
            lookup[key] = len(lookup)
        groups.append(lookup[key])
    groups = np.asarray(groups, dtype=np.int64)
    payload = dict(path_ids=list(path_ids), sampling_group_by_path_index=groups.tolist())
    fingerprint = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True,
        separators=(",", ":")).encode("utf-8")).hexdigest()
    return groups, dict(print_instance_coverage=bool(print_instance_coverage),
        sampling_grouping_strategy=("source_pts_print_instance_else_native_path" if print_instance_coverage
                                    else "source_pts_native_path"),
        sampling_group_map_sha256=fingerprint,
        sampling_group_map_location="selection.npz:sampling_group_by_path_index",
        sampling_establishes_direction_coverage=False, sampling_establishes_source_accuracy=False)


def prepare(run_root, state_path, source_packets, output, spacing_px=24., *, motion_stride=None,
            print_instance_coverage=False):
    root, state_path, output = Path(run_root).resolve(), Path(state_path).resolve(), _guard_path(output)
    spacing_px = float(spacing_px)
    if not np.isfinite(spacing_px) or spacing_px <= 0:
        raise ValueError("spacing_px must be finite and positive")
    if output.exists():
        raise FileExistsError("Preserve the completed or interrupted prepared output: " + str(output))
    if root == output or root.is_relative_to(output):
        raise ValueError("Output cannot replace or contain the parent prepared input")
    if output.is_relative_to(prepared.ROOT.resolve()):
        raise ValueError("Large prepared observation outputs belong outside the OneDrive source tree")
    manifest = prepared.read(root / "joint_manifest.json")
    mapping = prepared.read(root / "mapping.json")
    parent_hash = prepared.digest(root / "point_observations.npz")
    if parent_hash != manifest["point_input_sha256"]:
        raise ValueError("Parent input digest differs")
    for name, expected in ((manifest["scene_path"], manifest["scene_file_sha256"]),
                           (manifest["native_library"], manifest["native_library_sha256"])):
        if prepared.digest(Path(name)) != expected:
            raise ValueError("Current model or native library digest differs")
    original, state = prepared.npz(root / "point_observations.npz"), prepared.npz(state_path)
    if set(original) != prepared.FIELDS:
        raise ValueError("Expected exactly seven unweighted source fields")
    model = SceneParameterModel(prepared.read(manifest["scene_path"]),
                               Path(manifest["scene_path"]), optimize_scene=True)
    prepared.validate_snapshot(state, model.initial_state.layout, mapping, manifest, original)
    parent_files = {str(path): prepared.digest(path) for path in
                    (root / "joint_manifest.json", root / "mapping.json", state_path)}
    parent_files[str(root / "point_observations.npz")] = parent_hash
    additions, aliases, packets = collect_additions(source_packets, original, manifest,
                                                    mapping["path_ids"])
    report = dict(decisions=[], added_observations=additions)
    if additions:
        new, numeric, shots, touched, decisions, added, origins = revise_observations(
            original, state["stations"], manifest, mapping["path_ids"], report)
        if decisions:
            raise ValueError("Append-only preparation cannot revise existing observations")
    else:
        new, numeric = original, state["stations"].copy()
        shots = copy.deepcopy(manifest["shots"])
        touched = np.zeros(len(original["pts"]), dtype=bool)
        origins = np.arange(len(original["pts"]), dtype=np.int64)
        added = []
    notes = _initialize_touched(new, numeric, shots, touched, state, manifest, mapping, model)
    _preserve_old_rows(new, original, numeric, state["stations"], origins)
    motion_changes = _apply_motion_stride(shots, motion_stride, manifest["control_dt"])
    sampling_groups, sampling_policy = _sampling_group_map(
        mapping["path_ids"], print_instance_coverage=print_instance_coverage)
    selection = sample_source_pixels(new["uv"], new["pts"], sampling_groups[new["path_indices"]],
                                     new["observation_ids"], kind=new["kind"],
                                     spacing_px=spacing_px)
    indices = selection.selected_indices
    active = {name: values[indices] for name, values in new.items()}
    active_numeric = numeric[indices]
    active_shots, begin = copy.deepcopy(shots), 0
    for shot in active_shots:
        count = int(np.count_nonzero((indices >= shot["row_begin"]) & (indices < shot["row_stop"])))
        shot["row_begin"], shot["row_stop"] = begin, begin + count
        begin += count
    for name in prepared.FIELDS:
        _same_bytes(active[name], new[name][indices], "selected source " + name)
    _same_bytes(active_numeric, numeric[indices], "selected numeric stations")
    path_map = {path["id"]: path for path in model.scene["paths"]}
    metadata = [dict(entity=(path_map[pid].get("entity_id", pid) if pid in path_map
                             else pid.split("::", 1)[0]), path_id=pid)
                for pid in mapping["path_ids"]]
    coverage = dict(before=_coverage(original, manifest["shots"], metadata),
                    full_source=_coverage(new, shots, metadata),
                    active=_coverage(active, active_shots, metadata))
    provenance = dict(schema="rinkspace.spatial_refinement_preparation.v1",
                      parent_run_root=str(root), parent_input_sha256=parent_hash,
                      parent_numeric_state=str(state_path),
                      parent_numeric_state_sha256=parent_files[str(state_path)],
                      source_receipt=str(output / "source_receipt.json"),
                      selection=str(output / "selection.npz"), spacing_px=spacing_px,
                      no_observation_weights=True, source_UV_changed=False,
                      model_and_motion_chart_changed=False, fit_executed=False, **sampling_policy)
    if motion_changes:
        provenance.update(model_and_motion_chart_changed=True, model_changed=False,
                          motion_stride_changes=motion_changes,
                          motion_stride_semantics="future_active_update_columns_only",
                          motion_numeric_controls_changed=False,
                          motion_reparameterized=False, initial_projection_changed=False)
    output.mkdir(parents=True)
    full_manifest, _ = _write_prepared(
        output / "full_source", new, numeric, shots, state, manifest, mapping, model,
        dict(provenance, role="full_comparison_mother_set"),
        link_input=root / "point_observations.npz" if not additions else None)
    active_manifest, _ = _write_prepared(
        output, active, active_numeric, active_shots, state, manifest, mapping, model,
        dict(provenance, role="active_source_pixel_subset",
             full_source_input_sha256=full_manifest["point_input_sha256"]))
    np.savez(output / "selection.npz", selected_indices=indices,
             reason_codes=selection.reason_codes, parent_rows_for_full_source=origins,
             sampling_group_by_path_index=sampling_groups)
    _jsonl(output / "source_additions.jsonl", added)
    _jsonl(output / "source_aliases.jsonl", aliases)
    _write(output / "coverage_summary.json", coverage)
    for path, expected in parent_files.items():
        if prepared.digest(Path(path)) != expected:
            raise ValueError("Parent input/state was modified during preparation: " + path)
    receipt = dict(**provenance, parent_files_sha256=parent_files, source_packets=packets,
                   added_rows=len(added), exact_duplicate_aliases=len(aliases),
                   aliases=str(output / "source_aliases.jsonl"),
                   additions=str(output / "source_additions.jsonl"),
                   parent_rows=len(original["pts"]), full_source_rows=len(new["pts"]),
                   active_rows=len(indices), full_source_input_sha256=full_manifest["point_input_sha256"],
                   active_input_sha256=active_manifest["point_input_sha256"],
                   selection_reason_codes={reason.name: int(reason) for reason in SamplingReason},
                   original_fields_byte_preserved=True, numeric_globals_byte_preserved=True,
                   old_numeric_stations_byte_preserved=True,
                   selection_indices_refer_to="full_source/point_observations.npz",
                   station_initialization=notes, old_source_chain_replayed=False,
                   coverage_summary=str(output / "coverage_summary.json"))
    _write(output / "source_receipt.json", receipt)
    print(json.dumps(dict(output=str(output), parent_rows=receipt["parent_rows"],
                          added_rows=len(added), aliases=len(aliases), active_rows=len(indices),
                          full_source_rows=len(new["pts"]), fit_executed=False)), flush=True)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--source-packets", type=Path, nargs="*", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--spacing-px", type=float, default=24.)
    parser.add_argument("--print-instance-coverage", action="store_true",
                        help="Share source-UV coverage within each Ad instance; default keeps native-path groups")
    parser.add_argument("--motion-stride", nargs="+", metavar="SHOT=STRIDE",
                        help="Override future motion update-column stride without resampling controls")
    args = parser.parse_args()
    prepare(args.run_root, args.state, args.source_packets, args.output, args.spacing_px,
            motion_stride=args.motion_stride, print_instance_coverage=args.print_instance_coverage)
