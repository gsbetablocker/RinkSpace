"""One-time full d78 state migration, independent of historical runtime code.

The native kernel alone decodes the physical camera and evaluates projection.
The imported historical state is a frozen replay, never a fitted new scene.
"""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import numpy as np
from rinkspace.core.native import NativeCore
from .d78_entities import decode_entities


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def decode_camera(model, core):
    camera = model["camera"]
    known = {"R0_seed", "n_seed", "b_seed", "n_tangent1", "n_tangent2", "f_ref", "radius_max", "dt", "first_control_time", "n_controls", "control_block_begin", "dependent_control", "reference_indices", "reference_weights", "tau_bound", "brown_wide_extension"}
    unknown = set(camera)-known
    if unknown:
        raise ValueError(f"unsupported d78 camera fields: {sorted(unknown)}")
    if camera.get("brown_wide_extension") is not None:
        raise ValueError("unsupported d78 brown_wide_extension; coefficients are not silently discarded")
    blocks = [np.asarray(block["values"], dtype=np.float64) for block in model["blocks"]]
    if blocks[0].shape != (32,) or not np.isfinite(blocks[0]).all():
        raise ValueError("d78 global32 must be finite and complete")
    g = blocks[0]
    chart = [*g, *camera["R0_seed"], *camera["n_seed"], *camera["b_seed"], *camera["n_tangent1"], *camera["n_tangent2"], camera["f_ref"], camera["radius_max"]]
    rig = core.decode_d78_camera(chart)
    controls = np.asarray(blocks[camera["control_block_begin"]:camera["control_block_begin"]+camera["n_controls"]], dtype="<f8")
    if controls.shape != (camera["n_controls"], 3) or not np.isfinite(controls).all():
        raise ValueError("missing/nonfinite d78 motion controls")
    # This is elimination of a stored chart dependency, not a second evaluator.
    indices = np.asarray(camera["reference_indices"], dtype=np.int64)
    weights = np.asarray(camera["reference_weights"], dtype=np.float64)
    dependent = camera["dependent_control"]
    if indices.shape != (4,) or weights.shape != (4,) or np.any(indices<0) or np.any(indices>=len(controls)):
        raise ValueError("invalid d78 reference-control support")
    pivot = weights[indices == dependent]
    if pivot.shape != (1,) or abs(pivot[0]) < 1e-12:
        raise ValueError("d78 reference-control dependency is not unique")
    original = controls[dependent, :2].copy()
    keep = indices != dependent
    controls[dependent, :2] = -(weights[keep, None]*controls[indices[keep], :2]).sum(axis=0)/pivot[0]
    return {"schema": "rinkspace.camera.motion.v1", "rig": rig, "controls_ref": "controls.npy",
            "mount_family": "orthogonal_pan_tilt_pivot", "lens_family": "brown_zoom_linear",
            "pupil_model": "zoom_quadratic_camera_z", "readout_mode": "predicted_row_implicit_unique",
            "first_control_time": camera["first_control_time"], "control_dt": camera["dt"],
            "reference_dependency": {"index": dependent, "reference_indices": indices.tolist(), "reference_weights": weights.tolist(),
                                     "stored_inactive_angles": original.tolist(), "expanded_angles": controls[dependent, :2].tolist()},
            "original_camera_definition": deepcopy(camera), "original_global32": g.tolist()}, controls


def import_d78_state(model_path, out_dir, scope_path, library_path, complete_state_path=None, prepared_revision="unbound", asset_map_path=None, shot_id=None):
    model_path, out_dir, scope_path = Path(model_path).resolve(strict=True), Path(out_dir).resolve(), Path(scope_path).resolve(strict=True)
    if out_dir.exists():
        raise FileExistsError(f"state migration does not overwrite an existing candidate: {out_dir}")
    model, scope = _read(model_path), _read(scope_path)
    core = NativeCore(library_path)
    camera, controls = decode_camera(model, core)
    scene = decode_entities(model)
    # Native-time identity is inherited from the current scope, not frame/fps.
    selected_id = shot_id if shot_id is not None else scope["required_shot_ids"][0]
    matches = [shot for shot in scope["shots"] if shot["id"] == selected_id]
    if len(matches) != 1 or selected_id not in scope["required_shot_ids"]:
        raise ValueError(f"unknown or ambiguous scope shot ID: {selected_id}")
    selected = matches[0]
    clock = scope["source_clock"]
    camera.update(source_id=scope["source"]["id"], shot_id=selected["id"], rig_id=selected["rig_id"],
                  reference_pts=selected["processing"]["first_pts"], time_base_num=clock["time_base_num"],
                  time_base_den=clock["time_base_den"], image_height=scope["source"]["height"], image_width=scope["source"]["width"],
                  time_base={"num": clock["time_base_num"], "den": clock["time_base_den"]},
                  first_pts=selected["processing"]["first_pts"], stop_pts_exclusive=selected["processing"]["stop_pts_exclusive"])
    blocks, values, offset = [], [], 0
    for i, block in enumerate(model["blocks"]):
        raw = np.asarray(block["values"], dtype="<f8")
        if raw.ndim!=1 or not np.isfinite(raw).all():
            raise ValueError(f"invalid historical state block {i}: {block.get('name')}")
        blocks.append({"id": f"d78_block_{i:04d}", "name": block["name"], "offset": offset, "size": len(raw),
                       "unit": "original_chart_units", "parameterization": "frozen_d78_chart",
                       "owner": "frozen_replay", "active": False, "original_block": deepcopy(block)})
        values.extend(raw.tolist());offset+=len(raw)
    complete = _read(complete_state_path) if complete_state_path is not None else None
    local = np.empty(0, dtype="<f8")
    local_layout, local_arrays = {}, {}
    resolved_assets = {}
    if asset_map_path is not None:
        migration_map = _read(asset_map_path)
        matches = [row for row in migration_map.get("states", []) if row.get("model", {}).get("sha256") == _hash(model_path)]
        if len(matches) != 1:
            raise ValueError("current full-state asset map does not uniquely identify this d78 model")
        resolved_assets = matches[0].get("assets", {})
    if complete is not None:
        for name, record in complete.get("row_state_files", {}).items():
            source = Path(record["path"])
            if not source.is_file() and name in resolved_assets:
                migrated = resolved_assets[name]
                if migrated.get("sha256") != record.get("sha256"):
                    raise ValueError(f"current local-state mapping changes identity: {name}")
                source = Path(migrated["file"])
            if not source.is_file():
                raise FileNotFoundError(f"complete d78 local-state file is missing: {source}")
            dtype = "<u4" if name.endswith(".u32") else "<f8"
            # Existing original paths need the same identity check as relocated
            # paths. Hash exactly the bytes decoded, and reject truncated scalar
            # tails that np.fromfile would otherwise silently ignore.
            raw = source.read_bytes()
            if hashlib.sha256(raw).hexdigest() != record.get("sha256"):
                raise ValueError(f"complete d78 local-state content differs from its declared identity: {source}")
            if len(raw) % np.dtype(dtype).itemsize:
                raise ValueError(f"complete d78 local-state byte count is not aligned to {dtype}: {source}")
            array = np.frombuffer(raw, dtype=dtype).copy()
            if dtype == "<f8" and not np.isfinite(array).all():
                raise ValueError(f"nonfinite complete d78 local state: {source}")
            local_arrays[name.replace(".", "_")+".npy"] = array
            local_layout[name] = {"path": name.replace(".", "_")+".npy", "dtype": dtype, "count": len(array), "source_ref": deepcopy(record), "resolved_source_path": str(source)}
        if "station.f64" in local_layout:
            local = local_arrays[local_layout["station.f64"]["path"]].copy()
    # All writes are into a new Data directory; original models/states remain.
    out_dir.mkdir(parents=True)
    _write(out_dir/"scene.json", scene)
    _write(out_dir/"camera.json", camera)
    np.save(out_dir/"controls.npy", controls, allow_pickle=False)
    np.save(out_dir/"values.npy", np.asarray(values, dtype="<f8"), allow_pickle=False)
    np.save(out_dir/"local_values.npy", local, allow_pickle=False)
    for name, array in local_arrays.items():
        np.save(out_dir/name, array, allow_pickle=False)
    _write(out_dir/"original_complete_state.json", complete)
    _write(out_dir/"local_layout.json", local_layout)
    rigs = [{"rig_id": selected["rig_id"], "installation_id": "d78_original_installation", "coordinate_frame_id": "d78_L1", "camera_ref": "camera.json", "state_status": "available"}]
    for rig_id in sorted({shot["rig_id"] for shot in scope["shots"]}-{selected["rig_id"]}):
        rigs.append({"rig_id": rig_id, "state_status": "missing_original_parameters", "camera_ref": None})
    _write(out_dir/"rigs.json", {"schema": "rinkspace.rigs.v1", "rigs": rigs})
    shots = [{**deepcopy(shot), "source_id": scope["source"]["id"], "camera_ref": "camera.json" if shot["id"]==selected["id"] else None,
              "state_status": "available" if shot["id"]==selected["id"] else "not_imported_no_invented_state"} for shot in scope["shots"]]
    _write(out_dir/"shots.json", {"schema": "rinkspace.shots.v1", "shots": shots, "required_shot_ids": scope["required_shot_ids"]})
    state = {"schema": "rinkspace.state.v1", "state_id": "d78_frozen_replay_"+_hash(model_path)[:12],
             "scene_revision": scene["revision"], "prepared_revision": prepared_revision, "layout_revision": "d78_complete_frozen_v1",
             "parameter_blocks": blocks, "values_ref": "values.npy", "local_values_ref": "local_values.npy", "shots": shots,
             "camera_ref": "camera.json", "scene_ref": "scene.json", "replay_scene_ref": "scene.json", "rigs_ref": "rigs.json", "shots_ref": "shots.json", "local_layout_ref": "local_layout.json",
             "required_shot_ids": scope["required_shot_ids"], "selected_shot_ids": [selected["id"]], "scope_status": "subset",
             "origin": {"kind": "complete_d78_frozen_migration", "model_path": str(model_path), "model_sha256": _hash(model_path),
                        "complete_state_path": str(Path(complete_state_path).resolve()) if complete_state_path else None,
                        "complete_state_preserved_ref": "original_complete_state.json", "original_source_identity": model.get("source_identity"),
                        "imported_shot_id": selected_id, "shot_selection": "explicit" if shot_id is not None else "first_required_scope_shot",
                        "complete_local_state_imported": complete is not None},
             "status": "frozen_projection_replay", "fit_executed": False, "cuda_executed": False, "user_approval": "not_reviewed",
             "limitations": ["Frozen original geometry is a faithful replay, not fitted new common structure.", "Original d78 chart and full local state are preserved; historical LM process/duals are not a new optimizer checkpoint.", "Other shot camera values are not invented; all nine required shot IDs remain registered."]}
    _write(out_dir/"state.json", state)
    return out_dir/"state.json"


def load_replay_camera(state_path):
    state_path = Path(state_path).resolve(strict=True)
    state = _read(state_path)
    if state.get("status") != "frozen_projection_replay":
        raise ValueError("this loader requires a frozen replay state")
    def asset_ref(value):
        target = (state_path.parent/value).resolve(strict=True)
        target.relative_to(state_path.parent)
        return target
    camera = _read(asset_ref(state["camera_ref"]))
    if camera.get("mount_family") != "orthogonal_pan_tilt_pivot" or camera.get("lens_family") != "brown_zoom_linear" or camera.get("readout_mode") != "predicted_row_implicit_unique":
        raise ValueError("unsupported imported motion-camera family")
    controls = np.load(asset_ref(camera["controls_ref"]), allow_pickle=False)
    return state, camera, controls


def project_native_pts(core, camera, controls, pts, xyz):
    exposure = np.asarray(pts)
    if exposure.dtype.kind not in "iu":
        raise ValueError("projection exposure identity requires native integer PTS")
    if np.any(exposure<camera["first_pts"]) or np.any(exposure>=camera["stop_pts_exclusive"]):
        raise ValueError("PTS outside imported shot processing domain")
    times = (exposure.astype(np.int64)-int(camera["reference_pts"]))*camera["time_base_num"]/camera["time_base_den"]
    return core.project_motion(camera["rig"], controls, camera["first_control_time"], camera["control_dt"], camera["image_height"], times, xyz)
