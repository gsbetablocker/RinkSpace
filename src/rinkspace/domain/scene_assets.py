"""Current numeric scene dependencies, independent of historical production.

Only advertisement_assets[].path is an executable scene dependency here.
Media, source-observation stores and installation/review history remain opaque
provenance inside the copied JSON; this module never opens or activates them.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import shutil

from ..path_policy import guard_project_write


def _decode(raw, path):
    def invalid_constant(value):
        raise ValueError(f"Nonfinite JSON value {value} in {path}")
    data = json.loads(raw.decode("utf-8-sig"), parse_constant=invalid_constant)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return data


def _named(records, name, path):
    if not isinstance(records, list):
        raise ValueError(f"Expected {name} list in {path}")
    result = {}
    for item in records:
        identity = item.get("id") if isinstance(item, dict) else None
        if not isinstance(identity, str) or not identity or identity in result:
            raise ValueError(f"Invalid or duplicate {name} identity in {path}: {identity!r}")
        result[identity] = item
    return result


def _asset_records(scene, scene_path):
    """Read each explicitly declared asset once and retain its verified bytes."""
    if not isinstance(scene, dict):
        raise ValueError("Scene must be a JSON object")
    refs = scene.get("advertisement_assets", [])
    if not isinstance(refs, list):
        raise ValueError("Scene advertisement_assets must be a list")
    root = Path(scene_path).resolve().parent
    records = []
    claimed_instances = set()
    for ref in refs:
        if not isinstance(ref, dict) or not isinstance(ref.get("path"), str) or not ref["path"]:
            raise ValueError("Advertisement reference needs an explicit path")
        expected = ref.get("sha256")
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError(f"Advertisement reference needs a SHA256: {ref['path']}")
        path = Path(ref["path"])
        if not path.is_absolute():
            path = root/path
        path = path.resolve(strict=True)
        raw = path.read_bytes()
        actual = hashlib.sha256(raw).hexdigest()
        if actual != expected:
            raise ValueError(f"Advertisement asset SHA256 mismatch: {path}; expected {expected}, got {actual}")
        assets = _decode(raw, path)
        instances = _named(assets.get("instances"), "instances", path)
        templates = _named(assets.get("templates"), "templates", path)
        selected = ref.get("instance_ids", [ref["instance_id"]] if "instance_id" in ref else None)
        if (not isinstance(selected, list) or not selected
                or any(not isinstance(identity, str) or not identity for identity in selected)
                or len(set(selected)) != len(selected)):
            raise ValueError(f"Advertisement reference needs unique instance_ids: {path}")
        if "instance_id" in ref and ref["instance_id"] not in selected:
            raise ValueError(f"Primary instance is absent from declared instance_ids: {path}")
        overlap = claimed_instances.intersection(selected)
        if overlap:
            raise ValueError(f"Duplicate scene advertisement instance_ids: {sorted(overlap)}")
        for identity in selected:
            if identity not in instances:
                raise ValueError(f"Declared instance {identity} is absent from {path}")
            template = instances[identity].get("template_id")
            if template not in templates:
                raise ValueError(f"Instance {identity} references absent template {template!r} in {path}")
            if "template_id" in ref and template != ref["template_id"]:
                raise ValueError(f"Scene and asset template_id differ for {identity}: {path}")
        claimed_instances.update(selected)
        records.append((ref, path, raw, assets))
    return records


def read_scene_assets(scene, scene_path):
    """Return [(original scene reference, verified current numeric assets)].

    Relative paths, including ../ad2, are anchored to the scene file. Only the
    explicitly referenced files are read; no directory search or history replay.
    """
    return [(ref, assets) for ref, _, _, assets in _asset_records(scene, scene_path)]


def _content_identity(scene, records):
    normalized = deepcopy(scene)
    for ref, (_, _, raw, _) in zip(normalized.get("advertisement_assets", []), records):
        # Relocation changes only the dependency address, never its identity.
        ref["path"] = "sha256:"+hashlib.sha256(raw).hexdigest()
    encoded = json.dumps(normalized, sort_keys=True, ensure_ascii=False,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def scene_content_identity(scene, scene_path):
    """Hash all scene declarations plus actual assets, ignoring only relocation.

    Every asset SHA is checked first. No dimensions, placements, observations,
    source claims, revision fields or other provenance are dropped from identity.
    """
    return _content_identity(scene, _asset_records(scene, scene_path))


def snapshot_scene(scene_path, out_dir):
    """Write scene.json and byte-identical prints/pNNN.json in a NEW directory.

    Return (rewritten_scene, dependency_summaries). All input assets are verified
    before creating output. Existing output is never overwritten. Other source
    paths remain provenance references; their data are neither copied nor used.
    """
    source = Path(scene_path).resolve(strict=True)
    requested = Path(out_dir).expanduser().absolute()
    if os.path.lexists(requested):
        raise FileExistsError(requested)
    out = guard_project_write(requested)
    scene = _decode(source.read_bytes(), source)
    records = _asset_records(scene, source)
    snapshot = deepcopy(scene)
    summaries = []
    for index, (ref, path, raw, _) in enumerate(records):
        relative = f"prints/p{index:03d}.json"
        guard_project_write(out/relative)
        snapshot["advertisement_assets"][index]["path"] = relative
        summaries.append({"source_path": str(path), "snapshot_path": relative,
                          "sha256": ref["sha256"], "size_bytes": len(raw)})
    scene_bytes = (json.dumps(snapshot, ensure_ascii=False, indent=2, allow_nan=False)+"\n").encode("utf-8")
    guard_project_write(out/"scene.json")
    out.mkdir(parents=True, exist_ok=False)
    try:
        if records:
            (out/"prints").mkdir()
        for summary, (_, _, raw, _) in zip(summaries, records):
            (out/summary["snapshot_path"]).write_bytes(raw)
        (out/"scene.json").write_bytes(scene_bytes)
    except BaseException:
        # Remove only the new directory created by this invocation.
        shutil.rmtree(out)
        raise
    return snapshot, summaries
