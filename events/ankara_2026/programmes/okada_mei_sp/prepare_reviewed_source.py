"""Prepare explicit source corrections in a new full-source package, without fitting.

Unchanged observations and numeric globals are byte-preserved. Every affected
original row is archived. Append qualified replacements with
prepare_spatial_refinement before selecting a new fit input.
"""
from pathlib import Path
import argparse
import json

import numpy as np

import prepare_native_multicamera as prepared
from apply_native_multicamera_review import revise_observations
from prepare_spatial_refinement import _guard_path, _initialize_touched, _same_bytes, _write_prepared
from rinkspace.domain.parameters import SceneParameterModel


def prepare(run_root, state_path, review_path, output):
    root, state_path, review_path = map(lambda p: Path(p).resolve(),
                                      (run_root, state_path, review_path))
    output = _guard_path(output)
    if output.exists() or root.is_relative_to(output):
        raise FileExistsError("Source correction output must be new and separate")
    if output.is_relative_to(prepared.ROOT.resolve()):
        raise ValueError("Large source packages belong outside OneDrive")
    manifest = prepared.read(root / "joint_manifest.json")
    mapping = prepared.read(root / "mapping.json")
    report = prepared.read(review_path)
    input_hash = prepared.digest(root / "point_observations.npz")
    state_hash = prepared.digest(state_path)
    if input_hash != manifest["point_input_sha256"] or input_hash != report["input_sha256"]:
        raise ValueError("Review input differs from the current full source")
    if state_hash != report["state_sha256"]:
        raise ValueError("Review numeric source changed")
    if report.get("added_observations"):
        raise ValueError("Append qualified packets in the subsequent preparation step")
    dependencies = {str(p): prepared.digest(p) for p in
                    (root / "joint_manifest.json", root / "mapping.json",
                     root / "point_observations.npz", state_path, review_path)}
    for field, hash_field in (("scene_path", "scene_file_sha256"),
                             ("native_library", "native_library_sha256")):
        if prepared.digest(Path(manifest[field])) != manifest[hash_field]:
            raise ValueError("Current model/native library differs: " + field)
    original = prepared.npz(root / "point_observations.npz")
    state = prepared.npz(state_path)
    model = SceneParameterModel(prepared.read(manifest["scene_path"]),
                               Path(manifest["scene_path"]), optimize_scene=True)
    prepared.validate_snapshot(state, model.initial_state.layout, mapping, manifest, original)
    new, numeric, shots, touched, decisions, added, origins = revise_observations(
        original, state["stations"], manifest, mapping["path_ids"], report)
    if added or np.any(origins < 0):
        raise ValueError("Source correction unexpectedly added observations")
    source_rows = {str(value): i for i, value in enumerate(original["observation_ids"])}
    reviewed = np.array([source_rows[d["observation_id"]] for d in decisions], dtype=np.int64)
    removed = np.setdiff1d(np.arange(len(original["pts"])), origins)
    allowed_removed = sorted(source_rows[d["observation_id"]] for d in decisions
                             if d["decision"] == "exclude_preserving_source")
    if not np.array_equal(removed, allowed_removed):
        raise ValueError("An unreviewed observation disappeared")
    modified_uv = {source_rows[d["observation_id"]] for d in decisions
                   if d["decision"] == "correct_uv"}
    modified_paths = {source_rows[d["observation_id"]] for d in decisions
                      if d["decision"] == "rebind_preserving_uv" or d.get("new_path_id")}
    for field in prepared.FIELDS:
        allowed = modified_uv if field == "uv" else modified_paths if field == "path_indices" else set()
        preserve = ~np.isin(origins, list(allowed))
        _same_bytes(new[field][preserve], original[field][origins[preserve]], "retained " + field)
    notes = _initialize_touched(new, numeric, shots, touched, state, manifest, mapping, model)
    _same_bytes(numeric[~touched], state["stations"][origins[~touched]], "untouched numeric stations")
    provenance = dict(schema="rinkspace.reviewed_source_preparation.v1",
        parent_root=str(root), parent_state=str(state_path), parent_input_sha256=input_hash,
        parent_numeric_state_sha256=state_hash, source_review=str(review_path),
        source_review_sha256=dependencies[str(review_path)], excluded_rows=len(removed),
        rebound_rows=sum(d["decision"] == "rebind_preserving_uv" for d in decisions),
        corrected_uv_rows=sum(d["decision"] == "correct_uv" for d in decisions),
        station_initialization=notes, seven_unweighted_fields=True, fit_executed=False,
        untouched_source_fields_byte_preserved=True, numeric_globals_byte_preserved=True)
    out_manifest, out_mapping = _write_prepared(output, new, numeric, shots, state,
                                               manifest, mapping, model, provenance)
    out_manifest["source_review_preparation"] = provenance
    prepared.write(output / "joint_manifest.json", out_manifest)
    saved = prepared.npz(output / "initial_pixels.npz")
    prepared.validate_snapshot(saved, model.initial_state.layout, out_mapping, out_manifest, new)
    for name in ("rig", "controls", "scene_values"):
        _same_bytes(saved[name], state[name], "numeric global " + name)
    np.savez(output / "original_reviewed_observations.npz",
             **{k: v[reviewed] for k, v in original.items()},
             mother_row_indices=reviewed, numeric_stations=state["stations"][reviewed])
    np.savez(output / "source_origins.npz", parent_row_indices=origins)
    for path, expected in dependencies.items():
        if prepared.digest(Path(path)) != expected:
            raise ValueError("Source dependency changed during preparation: " + path)
    receipt = dict(provenance, source_files_sha256=dependencies,
                   new_input_sha256=out_manifest["point_input_sha256"],
                   retained_rows=len(new["pts"]), decisions=decisions)
    prepared.write(output / "source_review_receipt.json", receipt)
    print(json.dumps({k: v for k, v in receipt.items() if k not in
                      ("decisions", "source_files_sha256", "station_initialization")}), flush=True)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run-root", "state", "review", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    prepare(args.run_root, args.state, args.review, args.output)
