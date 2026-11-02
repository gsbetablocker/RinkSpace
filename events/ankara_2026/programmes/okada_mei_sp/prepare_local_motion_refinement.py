"""Prepare local motion update directions without changing numeric inputs.

Only the selected shot's local_motion_refinement metadata and the new full
state's dependency binding change. Current source pixels, source qualification,
ROI metadata, clocks, camera, scene, controls and numeric stations are retained.
No historical source chain, native library load, CUDA build or fit is performed.
"""
from pathlib import Path
import argparse
import copy
import hashlib
import json

import numpy as np

import prepare_native_multicamera as prepared
from prepare_spatial_refinement import _guard_path, _same_bytes, _write, _write_prepared
from rinkspace.domain.parameters import SceneParameterModel


def _digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_digest(values):
    return hashlib.sha256(memoryview(np.ascontiguousarray(values)).cast("B")).hexdigest()


def _exact_array(actual, expected, label):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.dtype != expected.dtype or actual.shape != expected.shape:
        raise ValueError("Protected array dtype or shape changed: " + label)
    if actual.ndim:
        for begin in range(0, len(actual), 32768):
            _same_bytes(actual[begin:begin + 32768], expected[begin:begin + 32768], label)
    else:
        _same_bytes(actual, expected, label)
    return dict(dtype=expected.dtype.str, shape=list(expected.shape),
                parent_sha256=_array_digest(expected),
                output_sha256=_array_digest(actual), bitwise_unchanged=True)


def _requested_shots(manifest, observations, shot_id, first_pts, last_pts, target_stride,
                     channels=None):
    for label, value in (("first_pts", first_pts), ("last_pts", last_pts),
                         ("target_stride", target_stride)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(label + " must be an integer")
    if target_stride < 1 or first_pts > last_pts:
        raise ValueError("Local refinement needs ordered PTS and a positive stride")
    shots = copy.deepcopy(manifest["shots"])
    matches = [shot for shot in shots if shot["id"] == shot_id]
    if len(matches) != 1:
        raise ValueError("Expected one current shot: " + shot_id)
    shot = matches[0]
    if shot.get("local_motion_refinement"):
        raise ValueError("Selected shot already has local motion refinement")
    if target_stride >= shot["knot_stride"]:
        raise ValueError("Local target_stride must refine the existing knot_stride")
    if not (shot["first_pts"] <= first_pts <= last_pts <= shot["last_pts"]):
        raise ValueError("Local refinement is outside the selected shot")
    times = observations["pts"][shot["row_begin"]:shot["row_stop"]]
    if not np.any(times == first_pts) or not np.any(times == last_pts):
        raise ValueError("Local refinement endpoints must be current source exposures")
    exposure_fields = [[item["id"], key] for item in shots for key in item
                       if "exposure" in key.lower()]
    if exposure_fields:
        raise ValueError("This comparison requires shots without exposure mapping fields")
    interval = dict(first_pts=first_pts, last_pts=last_pts, target_stride=target_stride)
    if channels is not None:
        if (not isinstance(channels, (list, tuple)) or not channels or
                any(channel not in ("pan", "tilt", "log_f") for channel in channels) or
                len(set(channels)) != len(channels)):
            raise ValueError("channels must name unique pan, tilt or log_f channels")
        interval["channels"] = list(channels)
    shot["local_motion_refinement"] = [interval]
    return shots, interval


def prepare(prepared_root, state_path, shot_id, first_pts, last_pts, target_stride, output,
            *, channels=None):
    root, state_path = Path(prepared_root).resolve(), Path(state_path).resolve()
    output = _guard_path(output)
    if output.exists():
        raise FileExistsError("Preserve the completed or interrupted output: " + str(output))
    if (output == root or output.is_relative_to(root) or root.is_relative_to(output)):
        raise ValueError("Output must be separate from the parent prepared input")
    if output.is_relative_to(prepared.ROOT.resolve()):
        raise ValueError("Large prepared inputs belong outside the OneDrive source tree")
    for name in ("point_observations.npz", "initial_pixels.npz", "joint_manifest.json",
                 "mapping.json", "local_motion_refinement_receipt.json"):
        _guard_path(output / name)

    manifest = prepared.read(root / "joint_manifest.json")
    mapping = prepared.read(root / "mapping.json")
    point_path = root / "point_observations.npz"
    paths = [root / "joint_manifest.json", root / "mapping.json", point_path, state_path,
             Path(manifest["scene_path"]), Path(manifest["native_library"])]
    if (root / "initial_pixels.npz").is_file():
        paths.append(root / "initial_pixels.npz")
    before = {str(path): _digest(path) for path in paths}
    point_hash = before[str(point_path)]
    if point_hash != manifest["point_input_sha256"]:
        raise ValueError("Current point input checksum differs")
    for name in ("point_input_sha256", "input_sha256"):
        if name in mapping and mapping[name] != point_hash:
            raise ValueError("Current mapping point checksum differs: " + name)
    for path_key, hash_key in (("scene_path", "scene_file_sha256"),
                               ("native_library", "native_library_sha256")):
        if before[str(Path(manifest[path_key]))] != manifest[hash_key]:
            raise ValueError("Current dependency checksum differs: " + path_key)
    with np.load(point_path, allow_pickle=False) as archive:
        if set(archive.files) != prepared.FIELDS:
            raise ValueError("Expected exactly seven unweighted source fields")
        observations = dict(archive)
    with np.load(state_path, allow_pickle=False) as archive:
        state = dict(archive)
    model = SceneParameterModel(prepared.read(manifest["scene_path"]),
                               Path(manifest["scene_path"]), optimize_scene=True)
    layout = model.initial_state.layout
    prepared.validate_snapshot(state, layout, mapping, manifest, observations)
    shots, interval = _requested_shots(
        manifest, observations, shot_id, first_pts, last_pts, target_stride, channels=channels)
    provenance = dict(
        schema="rinkspace.local_motion_refinement_preparation.v1",
        parent_prepared=str(root), parent_numeric_state=str(state_path),
        parent_numeric_state_sha256=before[str(state_path)],
        parent_input_sha256=point_hash, shot=shot_id, interval=interval,
        base_knot_stride=next(s["knot_stride"] for s in manifest["shots"] if s["id"] == shot_id),
        control_dt=manifest["control_dt"],
        semantics="additional_local_future_update_directions_only",
        numeric_controls_changed=False, clock_changed=False, source_UV_changed=False,
        source_eligibility_changed=False, fit_executed=False,
        receipt=str(output / "local_motion_refinement_receipt.json"))
    result_manifest, result_mapping = _write_prepared(
        output, observations, state["stations"], shots, state, manifest, mapping,
        model, provenance, link_input=point_path)
    # The shared writer stamps spatial provenance; retain the parent's unrelated
    # spatial selection record and give this metadata change its own record.
    if "spatial_refinement_preparation" in manifest:
        result_manifest["spatial_refinement_preparation"] = copy.deepcopy(
            manifest["spatial_refinement_preparation"])
    else:
        result_manifest.pop("spatial_refinement_preparation", None)
    result_manifest["local_motion_refinement_preparation"] = provenance
    _write(output / "joint_manifest.json", result_manifest)
    expected_manifest = copy.deepcopy(manifest)
    expected_manifest.update(shots=shots, current_resume_snapshot=str(output / "initial_pixels.npz"),
                             local_motion_refinement_preparation=provenance)
    if "point_input_path" in expected_manifest:
        expected_manifest["point_input_path"] = str(output / "point_observations.npz")
    if result_manifest != expected_manifest or result_mapping != mapping:
        raise ValueError("Unexpected manifest or mapping modification")

    output_hash = _digest(output / "point_observations.npz")
    if output_hash != point_hash:
        raise ValueError("Output point archive differs from the parent")
    fields = {}
    with np.load(output / "point_observations.npz", allow_pickle=False) as archive:
        if set(archive.files) != prepared.FIELDS:
            raise ValueError("Output point field set changed")
        for name in sorted(prepared.FIELDS):
            fields[name] = _exact_array(archive[name], observations[name], "source " + name)
    with np.load(output / "initial_pixels.npz", allow_pickle=False) as archive:
        result_state = dict(archive)
    if result_state.keys() != state.keys():
        raise ValueError("State field set changed")
    prepared.validate_snapshot(result_state, layout, result_mapping, result_manifest, observations)
    numeric = {name: _exact_array(result_state[name], state[name], "state " + name)
               for name in sorted(state) if name != "joint_numeric_dependencies_json"}
    if np.array_equal(result_state["joint_numeric_dependencies_json"],
                      state["joint_numeric_dependencies_json"]):
        raise ValueError("Local refinement was not bound to the output full state")
    after = {path: _digest(path) for path in before}
    if before != after:
        raise ValueError("A parent input or current dependency changed")
    receipt = dict(
        **provenance, output_prepared=str(output),
        output_numeric_state=str(output / "initial_pixels.npz"),
        output_numeric_state_sha256=_digest(output / "initial_pixels.npz"),
        parent_files_sha256=before, protected_files_after_sha256=after,
        output_input_sha256=output_hash, rows=len(observations["pts"]),
        seven_field_comparisons=fields, state_array_comparisons=numeric,
        all_seven_source_fields_bitwise_unchanged=True,
        all_numeric_arrays_bitwise_unchanged=True,
        input_archive_byte_identical=True, parent_files_unchanged=True,
        mapping_exactly_unchanged=True,
        manifest_only_requested_shot_and_output_binding_changed=True,
        shot_clock_and_ROI_metadata_exactly_unchanged=True,
        shots_contain_exposure_mapping_fields=False,
        state_dependency_binding_validated=True,
        historical_source_chain_replayed=False, native_library_loaded=False,
        cuda_built=False, evaluation_executed=False,
        initial_projection_numeric_inputs_unchanged=True)
    _write(output / "local_motion_refinement_receipt.json", receipt)
    print(json.dumps(dict(
        output=str(output), state=receipt["output_numeric_state"],
        receipt=provenance["receipt"], parent_input_sha256=point_hash,
        output_input_sha256=output_hash, parent_state_sha256=before[str(state_path)],
        output_state_sha256=receipt["output_numeric_state_sha256"],
        rows=receipt["rows"], all_numeric_arrays_bitwise_unchanged=True,
        all_seven_source_fields_bitwise_unchanged=True, fit_executed=False)), flush=True)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--shot", required=True)
    parser.add_argument("--first-pts", type=int, required=True)
    parser.add_argument("--last-pts", type=int, required=True)
    parser.add_argument("--target-stride", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--channels", nargs="+", choices=("pan", "tilt", "log_f"),
                        help="Restrict added local directions; omitted keeps all three channels")
    args = parser.parse_args()
    prepare(args.prepared, args.state, args.shot, args.first_pts, args.last_pts,
            args.target_stride, args.output, channels=args.channels)
