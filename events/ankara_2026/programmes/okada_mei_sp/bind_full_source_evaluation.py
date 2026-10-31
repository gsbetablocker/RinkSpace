"""Bind an active accepted state to its unchanged full-source comparison set.

This only copies numeric values and rebinds dependencies. It does not evaluate,
initialize stations or run an optimizer. Consume the resulting state with
fit_native_multicamera.py --evaluate-only --profile-evaluation-stations.
"""
from pathlib import Path
import argparse
import json

import numpy as np

import prepare_native_multicamera as prepared
from prepare_spatial_refinement import _guard_path, _same_bytes
from rinkspace.domain.parameters import SceneParameterModel


def _compatible_manifests(active, full, active_mapping, full_mapping):
    for manifest, mapping in ((active, active_mapping), (full, full_mapping)):
        for key in ("point_input_sha256", "input_sha256"):
            if key in mapping and mapping[key] != manifest["point_input_sha256"]:
                raise ValueError("Mapping input hash differs from its manifest: " + key)
    if active_mapping["path_ids"] != full_mapping["path_ids"]:
        raise ValueError("Active and full source have different path mappings")
    if active_mapping.get("scene_revision") != full_mapping.get("scene_revision"):
        raise ValueError("Active and full source have different mapping scene revisions")
    for key in ("scene_path", "scene_file_sha256", "scene_revision",
                "native_library", "native_library_sha256", "cameras",
                "control_dt", "first_control_time", "max_readout"):
        if active[key] != full[key]:
            raise ValueError("Active/full numeric interpretation differs: " + key)
    if [s["id"] for s in active["shots"]] != [s["id"] for s in full["shots"]]:
        raise ValueError("Active and full source have different shot order")
    for active_shot, full_shot in zip(active["shots"], full["shots"]):
        if active_shot["knot_stride"] != full_shot["knot_stride"]:
            raise ValueError(
                "Active/full knot_stride differs for " + active_shot["id"] +
                "; this binder does not change motion charts")
        left = {k: v for k, v in active_shot.items() if k not in ("row_begin", "row_stop")}
        right = {k: v for k, v in full_shot.items() if k not in ("row_begin", "row_stop")}
        if left != right:
            raise ValueError("Active/full shot trajectory interpretation differs: " +
                             active_shot["id"])


def _validate_selection(active, full, indices, active_manifest, full_manifest):
    if (indices.ndim != 1 or indices.dtype.kind not in "iu" or
            len(indices) != len(active["pts"])):
        raise ValueError("Selection needs one original integer index per active row")
    if (np.any(indices >= len(full["pts"])) or np.any(indices < 0) or
            len(np.unique(indices)) != len(indices)):
        raise ValueError("Selection indices are out of range or repeated")
    for active_shot, full_shot in zip(active_manifest["shots"], full_manifest["shots"]):
        rows = indices[active_shot["row_begin"]:active_shot["row_stop"]]
        if np.any(rows < full_shot["row_begin"]) or np.any(rows >= full_shot["row_stop"]):
            raise ValueError("Selected row crosses its shot: " + active_shot["id"])
    for start in range(0, len(indices), 32768):
        stop = min(start + 32768, len(indices))
        for key in prepared.FIELDS:
            _same_bytes(active[key][start:stop], full[key][indices[start:stop]],
                        "active/full source correspondence " + key)


def bind(run_root, state_path, output):
    root, state_path = Path(run_root).resolve(), Path(state_path).resolve()
    output = _guard_path(output)
    if output.suffix.lower() != ".npz":
        raise ValueError("Output must name a new .npz state")
    receipt_path = _guard_path(output.with_suffix(".binding.json"))
    pending = _guard_path(output.with_name(output.stem + ".pending.npz"))
    if any(path.exists() for path in (output, receipt_path, pending)):
        raise FileExistsError("Preserve existing output, binding receipt or pending state")
    full_root = root / "full_source"
    active_manifest = prepared.read(root / "joint_manifest.json")
    full_manifest = prepared.read(full_root / "joint_manifest.json")
    active_mapping = prepared.read(root / "mapping.json")
    full_mapping = prepared.read(full_root / "mapping.json")
    _compatible_manifests(active_manifest, full_manifest, active_mapping, full_mapping)
    input_paths = [root / "joint_manifest.json", root / "mapping.json",
                   root / "point_observations.npz", root / "selection.npz", state_path,
                   full_root / "joint_manifest.json", full_root / "mapping.json",
                   full_root / "point_observations.npz", full_root / "initial_pixels.npz"]
    hashes = {str(path): prepared.digest(path) for path in input_paths}
    for folder, manifest in ((root, active_manifest), (full_root, full_manifest)):
        if hashes[str(folder / "point_observations.npz")] != manifest["point_input_sha256"]:
            raise ValueError("Current observation digest differs: " + str(folder))
    if prepared.digest(Path(full_manifest["scene_path"])) != full_manifest["scene_file_sha256"]:
        raise ValueError("Current full-source scene digest differs")
    with np.load(root / "selection.npz", allow_pickle=False) as selection:
        indices = selection["selected_indices"].copy()
    active_points = prepared.npz(root / "point_observations.npz")
    full_points = prepared.npz(full_root / "point_observations.npz")
    if set(active_points) != prepared.FIELDS or set(full_points) != prepared.FIELDS:
        raise ValueError("Expected exactly seven unweighted source fields")
    _validate_selection(active_points, full_points, indices, active_manifest, full_manifest)
    active_state = prepared.npz(state_path)
    full_initial = prepared.npz(full_root / "initial_pixels.npz")
    model = SceneParameterModel(prepared.read(full_manifest["scene_path"]),
                               Path(full_manifest["scene_path"]), optimize_scene=True)
    layout = model.initial_state.layout
    prepared.validate_snapshot(active_state, layout, active_mapping,
                               active_manifest, active_points)
    prepared.validate_snapshot(full_initial, layout, full_mapping,
                               full_manifest, full_points)
    if active_state["stations"].dtype != full_initial["stations"].dtype:
        raise ValueError("Active/full numeric station dtypes differ")
    result = dict(full_initial)
    result["stations"] = full_initial["stations"].copy()
    result["stations"][indices] = active_state["stations"]
    result["source_rows"] = full_initial["source_rows"].copy()
    for key in ("rig", "controls", "scene_values"):
        result[key] = active_state[key].copy()
        _same_bytes(result[key], active_state[key], "copied active " + key)
    _same_bytes(result["stations"][indices], active_state["stations"],
                "active stations in full source")
    unselected = np.ones(len(full_points["pts"]), dtype=bool)
    unselected[indices] = False
    _same_bytes(result["stations"][unselected], full_initial["stations"][unselected],
                "unselected full-source stations")
    _same_bytes(result["source_rows"], full_initial["source_rows"], "full-source row identity")
    prepared.bind_snapshot(result, layout, full_mapping, full_manifest)
    prepared.validate_snapshot(result, layout, full_mapping, full_manifest, full_points)
    changed_keys = {"rig", "controls", "scene_values", "stations",
                    "joint_numeric_dependencies_json", "point_input_sha256",
                    "native_library_provenance_sha256"}
    for key in full_initial.keys() - changed_keys:
        _same_bytes(result[key], full_initial[key], "full initial " + key)
    # All source packages remain read-only, including a supplied accepted state.
    for path, digest in hashes.items():
        if prepared.digest(Path(path)) != digest:
            raise ValueError("Source input changed during binding: " + path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(pending, **result)
    if output.exists():
        raise FileExistsError("Output appeared during binding")
    pending.replace(output)
    receipt = dict(
        schema="rinkspace.full_source_evaluation_binding.v1",
        active_run_root=str(root), active_state=str(state_path),
        full_source_run_root=str(full_root), output_state=str(output),
        output_state_sha256=prepared.digest(output), source_files_sha256=hashes,
        active_input_sha256=active_manifest["point_input_sha256"],
        full_input_sha256=full_manifest["point_input_sha256"],
        active_rows=len(indices), full_source_rows=len(full_points["pts"]),
        preserved_unselected_station_rows=int(unselected.sum()),
        all_seven_source_fields_match_selected_full_rows=True,
        active_globals_and_stations_copied_exactly=True,
        full_source_rows_and_other_stations_preserved=True,
        shot_motion_strides={s["id"]: s["knot_stride"] for s in full_manifest["shots"]},
        motion_charts_equal=True, source_packages_modified=False,
        fit_executed=False, evaluation_executed=False,
        intended_use="fit_native_multicamera.py --evaluate-only --profile-evaluation-stations")
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
    print(json.dumps(dict(output=str(output), active_rows=len(indices),
                          full_source_rows=len(full_points["pts"]), binding_receipt=str(receipt_path),
                          required_evaluation_flags=["--evaluate-only",
                                                     "--profile-evaluation-stations"],
                          fit_executed=False, evaluation_executed=False)), flush=True)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bind(args.run_root, args.state, args.output)
