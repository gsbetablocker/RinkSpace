"""Snapshot only current rigs/shots -> camera -> controls dependencies.

Relative references belong to the JSON file declaring them. Historical origin,
installation and observation references remain opaque metadata and are neither
opened nor replayed. No numeric camera value is changed by this module.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
from io import BytesIO
import json
import math
import os
from pathlib import Path

import numpy as np

from ..path_policy import guard_project_write
from .camera import complete_camera_rig


def _decode(raw, path):
    def invalid_constant(value):
        raise ValueError(f"Nonfinite JSON value {value} in {path}")
    value = json.loads(raw.decode("utf-8-sig"), parse_constant=invalid_constant)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _encode(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n").encode("utf-8")


def _dependency(owner, reference, field):
    if not isinstance(reference, str) or not reference:
        raise ValueError(f"{field} needs an explicit file path in {owner}")
    path = Path(reference)
    if not path.is_absolute():
        path = owner.parent/path
    path = path.resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"{field} is not a file: {path}")
    return path


def _finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _camera_data(path):
    raw = path.read_bytes()
    camera = _decode(raw, path)
    if camera.get("schema") != "rinkspace.camera.motion.v1":
        raise ValueError(f"Unsupported current camera schema: {path}")
    if camera.get("source_binding_verified") is False:
        raise ValueError(f"Unverified historical replay cannot become a prepared fit camera: {path}")
    rig = camera.get("rig")
    if not isinstance(rig, list) or len(rig) not in (38, 43) or not all(_finite_number(v) for v in rig):
        raise ValueError(f"Camera rig must contain 38 finite or 43 finite numbers: {path}")
    complete_camera_rig(camera)  # Validate the whole effective optics definition.
    if not _finite_number(camera.get("first_control_time")):
        raise ValueError(f"Camera first_control_time must be finite: {path}")
    if not _finite_number(camera.get("control_dt")) or camera["control_dt"] <= 0:
        raise ValueError(f"Camera control_dt must be finite and positive: {path}")
    controls_path = _dependency(path, camera.get("controls_ref"), "controls_ref")
    controls_raw = controls_path.read_bytes()
    controls = np.load(BytesIO(controls_raw), allow_pickle=False)
    if not isinstance(controls, np.ndarray):
        controls.close()
        raise ValueError(f"Camera controls must be a single NPY array: {controls_path}")
    if (controls.dtype.str != "<f8" or controls.ndim != 2 or controls.shape[1] != 3
            or controls.shape[0] < 4 or not controls.flags.c_contiguous
            or not np.isfinite(controls).all()):
        raise ValueError(f"Camera controls must be finite C-contiguous float64[N,3], N >= 4: {controls_path}")
    return camera, raw, controls_path, controls_raw, controls


def _snapshot(registries, out_dir):
    dependencies, summaries, cameras = [], [], {}
    snapshots = []
    for registry_path, table in registries:
        source = Path(registry_path).resolve(strict=True)
        registry = _decode(source.read_bytes(), source)
        schema = f"rinkspace.{table}.v1"
        if registry.get("schema", schema) != schema:
            raise ValueError(f"Unsupported current {table} schema: {source}")
        rows = registry.get(table)
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise ValueError(f"Expected {table} list of objects: {source}")
        snapshot = deepcopy(registry)
        identities = set()
        for row, rewritten in zip(rows, snapshot[table]):
            identity = row.get("rig_id", row.get("id")) if table == "rigs" else row.get("id")
            if identity is not None:
                if not isinstance(identity, str) or not identity or identity in identities:
                    raise ValueError(f"Invalid or duplicate {table} identity: {identity!r}")
                identities.add(identity)
            if row.get("camera_ref") is None:
                continue
            path = _dependency(source, row["camera_ref"], "camera_ref")
            if path not in cameras:
                camera, raw, controls_path, controls_raw, controls = _camera_data(path)
                index = len(cameras)
                camera_relative, controls_relative = f"cameras/c{index:03d}.json", f"cameras/k{index:03d}.npy"
                relocated = deepcopy(camera)
                relocated["controls_ref"] = Path(controls_relative).name
                camera_raw = _encode(relocated)
                # Identity is the explicitly referenced camera file, not rig_id:
                # different shots of one rig may have distinct motion controls.
                cameras[path] = (camera_relative, camera)
                for kind, original, relative, value in (
                    ("camera", path, camera_relative, camera_raw),
                    ("controls", controls_path, controls_relative, controls_raw),
                ):
                    dependencies.append((relative, value))
                    summaries.append({"kind": kind, "source_path": str(original),
                                      "snapshot_path": relative, "sha256": hashlib.sha256(value).hexdigest(),
                                      "size_bytes": len(value)})
                summaries[-2]["source_sha256"] = hashlib.sha256(raw).hexdigest()
                summaries[-1].update(shape=list(controls.shape), dtype=controls.dtype.str)
            relative, camera = cameras[path]
            rig_identity = identity if table == "rigs" else row.get("rig_id")
            if rig_identity is not None and camera.get("rig_id") is not None and rig_identity != camera["rig_id"]:
                raise ValueError(f"Rig and camera rig_id differ for {rig_identity}: {path}")
            if table == "shots" and identity is not None and camera.get("shot_id") is not None and identity != camera["shot_id"]:
                raise ValueError(f"Shot and camera shot_id differ for {identity}: {path}")
            rewritten["camera_ref"] = relative
        snapshots.append(snapshot)
    for (_, table), snapshot in zip(registries, snapshots):
        dependencies.append((f"{table}.json", _encode(snapshot)))

    requested = Path(out_dir).expanduser().absolute()
    if os.path.lexists(requested) and (not requested.is_dir() or requested.is_symlink()):
        raise ValueError(f"Prepared output must be a real directory: {requested}")
    out = guard_project_write(requested)
    for relative, _ in dependencies:
        guard_project_write(out/relative)
    targets = [out/f"{table}.json" for _, table in registries]
    if cameras:
        targets.append(out/"cameras")
    for target in targets:
        if os.path.lexists(target):
            raise FileExistsError(target)

    created_files, created_dirs = [], []
    try:
        if not out.exists():
            out.mkdir(parents=True, exist_ok=False)
            created_dirs.append(out)
        if cameras:
            (out/"cameras").mkdir(exist_ok=False)
            created_dirs.append(out/"cameras")
        for relative, raw in dependencies:
            target = out/relative
            with target.open("xb") as stream:
                created_files.append(target)
                stream.write(raw)
    except BaseException:
        for path in reversed(created_files):
            path.unlink()
        for path in reversed(created_dirs):
            path.rmdir()
        raise
    return snapshots, summaries


def snapshot_rigs(rigs_path, out_dir):
    """Write rigs.json and its explicitly referenced cameras into a prepared root.

    Return (rewritten_rigs, dependency_summaries). Each summary has a relative
    snapshot_path and the SHA256 of its actual output bytes. The output root may
    already contain scene/observation assets, but rigs.json and a needed cameras
    directory must not exist. All inputs are checked before any output mutation.
    Missing/None camera_ref entries retain their exact existing declarations.
    """
    snapshots, summaries = _snapshot([(rigs_path, "rigs")], out_dir)
    return snapshots[0], summaries


def snapshot_runtime(rigs_path, shots_path, out_dir):
    """Snapshot both current registries using one camera-file identity map.

    Return (rewritten_rigs, rewritten_shots, dependency_summaries), and write
    rigs.json, shots.json, plus their camera/controls files. Shared references
    reuse one camera snapshot; independent shots of the same rig keep distinct
    cameras and controls. No implicit rig-to-shot camera state is invented.
    """
    snapshots, summaries = _snapshot([(rigs_path, "rigs"), (shots_path, "shots")], out_dir)
    return snapshots[0], snapshots[1], summaries
