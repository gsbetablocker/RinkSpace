"""One-time conversion of delivered d78 measurements, never source remeasurement.

This adapter reads static arrays and provenance only. It preserves rejected rows,
finite branches, Query constraints and complete whitening blocks. Normal prepare
and review consume the resulting current input without importing this module.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import numpy as np

from ..session.dataset import digest, read_json, require, write_array, write_json
from ..domain.measurement import POLICY_ID, required_measurement


def _legacy_array(desc):
    path = Path(desc.get("file", desc.get("path", ""))).resolve(strict=True)
    require(digest(path) == desc["sha256"], "Changed legacy adapter input", path=str(path))
    dtype = np.dtype(desc["dtype"])
    require(not dtype.hasobject and dtype.kind in "biuf", "Unsupported legacy array", path=str(path))
    array = np.fromfile(path, dtype=dtype)
    require(array.size == int(np.prod(desc["shape"])), "Legacy array byte count differs", path=str(path))
    return array.reshape(desc["shape"])


def _legacy_json(desc):
    path = Path(desc.get("file", desc.get("path", ""))).resolve(strict=True)
    require(digest(path) == desc["sha256"], "Changed legacy document", path=str(path))
    return read_json(path)


def _line_records(desc):
    path = Path(desc["file"]).resolve(strict=True)
    require(digest(path) == desc["sha256"], "Changed legacy row metadata", path=str(path))
    with path.open(encoding="utf-8-sig") as stream:
        yield from (json.loads(line) for line in stream if line.strip())


def _path_feature(mi, pi, mat, path, *, trusted):
    descriptor = json.loads(path.get("provenance", "{}"))
    source_descriptor = descriptor.get("descriptor", {})
    label = mat["label"]
    surface = "ice" if label.startswith("ice.") else "board"
    fixed = bool(path.get("physical_point"))
    measurement = required_measurement(label, "material_point" if fixed else "finite_curve")
    supplied = path.get("measurement", source_descriptor.get("measurement"))
    require(supplied == measurement, "Legacy data has no admissible canonical source measurement; edge fallback and pixel relabelling are forbidden",
            label=label, required_measurement=measurement)
    feature_id = path.get("current_curve_id", f"d78_material_{mi:03d}/path{pi}")
    return {"id": feature_id, "material_id": f"d78_material_{mi:03d}",
            "material_index": mi, "path_index": pi, "label": label,
            "kind": "material_point" if fixed else "finite_curve", "surface": surface,
            "identity": "confirmed" if trusted else "unresolved", "measurement": measurement,
            "source_measurement_evidence": {"measurement": supplied}, "measurement_policy_id": POLICY_ID,
            "original_path_id": path["id"], "original_variant_id": path["variant_id"],
            "finite_support": [0., 0.] if fixed else [0., 1.],
            "has_station_parameter": not fixed, "vertices_are_landmarks": False,
            "provenance": descriptor}


def import_d78_observations(input_path, model_path, scope_path, pts_index_path, out_dir, *,
                            path_indices_path=None, observation_ids_path=None,
                            scene_path=None, rigs_path=None, shots_path=None):
    """Return current input.json; all source UV/PTS/W remain bitwise unchanged.

    ``path_indices_path`` is the delivered finite branch state, not a nearest-
    model association. The optional stable row-ID prefix is accepted only when
    UV and native PTS agree with the frozen registered observations.
    """
    current_mode = Path(input_path).is_dir()
    input_document = Path(input_path) / "packet/manifest.json" if current_mode else Path(input_path)
    if current_mode:
        packet = read_json(input_document)
        sem = read_json(Path(input_path) / "semantic/manifest.json")
        current_query = read_json(Path(input_path) / "query/manifest.json")
        inp = {"raw_rows": packet["observations"], "source_arrays": packet["arrays"],
               "semantic_arrays": sem["arrays"], "resolved_material_index": sem["resolved_material_index"],
               "resolved_variant_index": sem["resolved_variant_index"], "resolved_variant_ids": sem["resolved_variant_ids"],
               "active_source_mask": sem["active_source_mask"], "query": current_query,
               "statistical_groups": {"groups": sem["assets"]["groups.jsonl"], "weights": sem["arrays"]["weights"],
                   "group_rows": sem["arrays"]["group_rows"], "independent_rows": sem["arrays"]["ordinary_rows"],
                   "source_sigma": sem["arrays"]["source_sigma"], "formal_rank": sem["statistics"]["formal_residual_rows"]}}
    else:
        inp = read_json(input_document)
        require(inp.get("schema") == "rinkspace.relocated_d78_native_input.v1", "Expected delivered frozen d78 input")
    model = read_json(model_path)
    scope = read_json(scope_path)
    out = Path(out_dir).resolve()
    require(not out.exists(), "Adapter output exists; preserve it and choose a new revision", path=str(out))
    n = inp["raw_rows"]
    source = {k: _legacy_array(v) for k, v in inp["source_arrays"].items()}
    semantic = {k: _legacy_array(v) for k, v in inp["semantic_arrays"].items()}
    if current_mode:
        tables = _legacy_json(packet["string_tables"])
        materials = {"materials": [{"label": m["label"], "source_identity_trusted": True} for m in model["materials"]]}
    else:
        materials = _legacy_json(inp["source_materials"])
        tables = _legacy_json(inp["source_id_tables"])
    resolved_material = _legacy_array(inp["resolved_material_index"]).astype(np.int64)
    resolved_variant = _legacy_array(inp["resolved_variant_index"]).astype(np.int64)
    active = _legacy_array(inp["active_source_mask"]).astype(np.uint8)
    require(source["native_uv"].shape == (n, 2) and source["native_pts"].shape == (n,), "Frozen row count differs")
    require(current_mode or len(model["materials"]) == 171, "Frozen d78 import requires its complete 171 material inventory")
    require(path_indices_path is not None, "Delivered finite path state is required; no nearest-curve reassociation")
    path_indices = np.fromfile(path_indices_path, dtype="<u4").astype(np.int64)
    require(path_indices.shape == (n,), "Finite path state row count differs")
    pts_index = np.load(pts_index_path, allow_pickle=False)
    require(pts_index.dtype.str == "<i8" and pts_index.ndim == 1 and np.all(np.diff(pts_index) > 0), "Expected actual decoded native PTS index")
    require(np.isin(source["native_pts"], pts_index).all(), "Frozen source PTS absent from registered decoded index")
    ids = np.asarray(tables["stable_observation_ids"], dtype="U") if current_mode else None
    if observation_ids_path and not current_mode:
        registered = Path(observation_ids_path).parent
        registered_uv = np.column_stack((np.load(registered / "s01_raw_u.npy", allow_pickle=False),
                                          np.load(registered / "s01_raw_v.npy", allow_pickle=False)))
        registered_pts = np.load(registered / "s01_raw_native_pts.npy", allow_pickle=False)
        require(np.array_equal(registered_uv[:n], source["native_uv"]) and np.array_equal(registered_pts[:n], source["native_pts"]),
                "Stable registered row IDs do not cover identical frozen UV/PTS")
        ids = np.load(observation_ids_path, allow_pickle=False)[:n]
    if ids is None:
        ids = np.asarray([f"{scope['source']['id']}:{int(pts)}:{tables['source_point_ids'][int(point)]}"
                          for pts, point in zip(source["native_pts"], source["point_id_index"])], dtype="U")
    require(len(np.unique(ids)) == n, "Frozen observation IDs are not unique")
    trust = {m["label"]: m.get("source_identity_trusted", False) for m in materials["materials"]}
    features, feature_lookup = [], {}
    for mi, mat in enumerate(model["materials"]):
        for pi, path in enumerate(mat["paths"]):
            feature_lookup[(mi, pi)] = len(features)
            features.append(_path_feature(mi, pi, mat, path, trusted=bool(trust.get(mat["label"]))))
    feature_index = np.empty(n, "<i4")
    for i, (mi, pi) in enumerate(zip(resolved_material, path_indices)):
        require((int(mi), int(pi)) in feature_lookup, "Resolved frozen entity branch is absent", row=i, material=int(mi), path=int(pi))
        feature_index[i] = feature_lookup[(int(mi), int(pi))]
        path = model["materials"][mi]["paths"][pi]
        require(path["variant_id"] == inp["resolved_variant_ids"][resolved_variant[i]], "Resolved finite variant differs from model", row=i)
    original_kind = semantic["kind"].astype(np.uint8)
    require(np.isin(original_kind, [0, 1, 2]).all(), "Unknown frozen measurement kind")
    kind = np.where(original_kind == 1, 0, 1).astype(np.uint8)
    for i in np.flatnonzero(kind == 0):
        require(features[feature_index[i]]["kind"] == "material_point", "Old fixed point is not a shared physical material point", row=int(i))
    require(all(features[int(f)]["identity"] == "confirmed" for f in np.unique(feature_index[active != 0])), "An active original entity is unconfirmed")
    noise = inp["statistical_groups"]
    old_members = _legacy_array(noise["group_rows"]).astype(np.int64)
    old_weights = _legacy_array(noise["weights"])
    groups, members, weights = [], [], []
    statistical_activity = (_legacy_array(sem["statistical_active_mask"]).astype(bool) if current_mode else active.astype(bool))
    group_index = np.full(n, -1, "<i4")
    replacements = {item["group_id"]: item for item in noise.get("replacements", [])}
    for original_ordinal, group in enumerate(_line_records(noise["groups"])):
        replacement = replacements.get(group["group_id"])
        if replacement:
            rr = np.asarray(replacement["source_rows"], np.int64)
            w = _legacy_array(replacement["W"])
            rank = replacement["rank"]
        else:
            start, count, rank = group["member_start"], group["member_count"], group["rank"]
            rr = old_members[start:start + count]
            w = old_weights[group["weight_start"]:group["weight_start"] + rank * count * 2].reshape(rank, 2 * count)
        require(w.shape == (rank, 2 * len(rr)) and np.isfinite(w).all(), "Changed original complete W layout", group_id=group["group_id"])
        require(np.all(group_index[rr] == -1), "Original statistical factors overlap", group_id=group["group_id"])
        require(np.all(active[rr] == 1), "Original group replacement did not remove a rejected member", group_id=group["group_id"])
        group_index[rr] = len(groups)
        groups.append({"id": group["group_id"], "rank": rank, "member_start": len(members), "member_count": len(rr),
                       "weight_start": len(weights), "matrix_shape": [rank, 2 * len(rr)],
                       "matrix_sha256": hashlib.sha256(np.asarray(w, dtype="<f8").tobytes()).hexdigest(),
                       "column_order": "u0,v0,u1,v1,...", "model": "square_root_information",
                       "covariance_origin": group.get("covariance_origin"), "source_group": group,
                       "equivalent_existing_marginal_replacement": replacement})
        members.extend(rr.tolist())
        weights.extend(w.ravel().tolist())
    independent = _legacy_array(noise["independent_rows"]).astype(np.int64)
    require(len(np.unique(independent)) == len(independent) and np.all(group_index[independent] == -1), "Original independent support overlaps groups")
    sigma = _legacy_array(noise["source_sigma"])
    require(np.all(np.isfinite(sigma[independent]) & (sigma[independent] > 0)), "Original independent sigma invalid")
    covariance = np.zeros((n, 3), "<f8")
    covariance[independent, 0] = sigma[independent] ** 2
    covariance[independent, 2] = sigma[independent] ** 2
    statistical_mask = (group_index >= 0)
    statistical_mask[independent] = True
    require(np.array_equal(statistical_mask, statistical_activity), "Original statistical source partition does not cover activity exactly once")
    constraint_rows = np.flatnonzero((active != 0) & ~statistical_activity)
    if len(constraint_rows):
        # These rows contribute measured engineering constraints only. A rank-zero
        # block preserves unknown noise; it does not invent statistical precision.
        group_index[constraint_rows] = len(groups)
        groups.append({"id": "current/constraint_only_unknown_noise", "rank": 0, "member_start": len(members),
                       "member_count": len(constraint_rows), "weight_start": len(weights), "matrix_shape": [0, 2 * len(constraint_rows)],
                       "matrix_sha256": hashlib.sha256(b"").hexdigest(), "column_order": "u0,v0,u1,v1,...",
                       "model": "constraint_only_unknown_noise", "covariance_origin": "original delivered zero statistical rank; engineering budgets retained"})
        members.extend(constraint_rows.tolist())
    support = np.tile([0., 1.], (n, 1))
    support[kind == 0] = 0
    local_index = np.full(n, -1, "<i8")
    local_index[kind == 1] = np.arange(np.sum(kind == 1), dtype=np.int64)
    out.mkdir(parents=True)
    arrays = {"uv": source["native_uv"], "pts": source["native_pts"].astype("<i8"),
              "source_index": np.zeros(n, "<i4"), "shot_index": np.zeros(n, "<i4"), "feature_index": feature_index,
              "kind": kind, "active": active, "split": np.zeros(n, "u1"), "noise_group_index": group_index,
              "covariance": covariance, "support": support, "local_index": local_index, "observation_ids": ids,
              "legacy_kind": original_kind, "legacy_source_rows": source["source_rows"],
              "legacy_material_index": source["material_index"], "legacy_resolved_material_index": resolved_material,
              "legacy_path_index": path_indices, "legacy_variant_index": resolved_variant,
              "engineering_pixel_budget": source["engineering_budget_px"],
              "source_profile_normal": semantic["profile_normal"], "source_query_uv": semantic["query_uv"],
              "source_alpha_bounds": semantic["alpha_bounds"], "statistical_active": statistical_activity.astype("u1")}
    for key, value in inp["query"]["arrays"].items():
        write_array(out, "query_" + key, _legacy_array(value))
    refs = {key: write_array(out, key, value) for key, value in arrays.items()}
    groups_doc = {"schema": "rinkspace.noise_groups.v1", "groups": groups,
                  "members_ref": write_array(out, "group_members", members, "<i8"),
                  "weights_ref": write_array(out, "group_weights", weights, "<f8"),
                  "formal_rank": sum(g["rank"] for g in groups) + 2 * len(independent)}
    require(groups_doc["formal_rank"] == noise["formal_rank"], "Whitening rank changed during equivalent conversion")
    source_record = {"id": scope["source"]["id"], "asset_path": scope["source"]["registered_path"],
                     "sha256": scope["source"]["sha256"], "width": scope["source"]["width"], "height": scope["source"]["height"],
                     "time_base_num": scope["source_clock"]["time_base_num"], "time_base_den": scope["source_clock"]["time_base_den"],
                     "pts_ref": write_array(out, "source_pts", pts_index, "<i8"),
                     "pts_registry_origin": str(Path(pts_index_path).resolve()), "external_asset_id": scope["source"]["id"]}
    shots = []
    for shot in scope["shots"]:
        item = dict(shot, source_id=source_record["id"], spline_reference_pts=scope["shots"][0]["processing"]["first_pts"])
        item["camera_domain"] = ({"status": "defined", "first_pts": shot["processing"]["first_pts"],
                                 "stop_pts_exclusive": shot["processing"]["stop_pts_exclusive"]}
                                 if shot["id"] == "s01" else {"status": "missing_state_not_fabricated"})
        shots.append(item)
    rigs = read_json(rigs_path) if rigs_path else {"rigs": [{"id": "main", "status": "frozen_d78_replay"}]}
    scene = read_json(scene_path) if scene_path else model
    write_json(out / "scene.json", scene)
    write_json(out / "rigs.json", rigs)
    write_json(out / "shots.json", {"schema": "rinkspace.shots.v1", "required_shot_ids": scope["required_shot_ids"], "shots": shots,
                                   "imported_shots_origin": str(shots_path) if shots_path else None})
    write_json(out / "sources.json", {"schema": "rinkspace.sources.v1", "sources": [source_record]})
    write_json(out / "features.json", {"schema": "rinkspace.features.v1", "features": features})
    write_json(out / "noise_groups.json", groups_doc)
    metadata_count = 0
    with (out / "observations.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        metadata_records = ({"source_full_row": i, "original_source_row": int(source["source_rows"][i]),
                             "current_packet": str(input_document.resolve()), "source_identity": packet.get("immutable_source_hashes", {})}
                            for i in range(n)) if current_mode else _line_records(inp["source_rows_metadata"])
        point_table = tables.get("source_point_ids", tables.get("point_ids", []))
        for row, original in enumerate(metadata_records):
            feature = features[feature_index[row]]
            sid = ids[row].decode("utf-8") if isinstance(ids[row], bytes) else str(ids[row])
            record = {"id": sid, "original_source_point_id": point_table[int(source["point_id_index"][row])] if point_table else sid,
                      "feature_id": feature["id"], "measurement": feature["measurement"],
                      "source_measurement_evidence": feature["source_measurement_evidence"], "quality": "existing_confirmed_source" if active[row] else "existing_rejected",
                      "origin": "legacy_import", "source_evidence": "measured", "provenance": original}
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")
            metadata_count += 1
    require(metadata_count == n, "Original provenance row count changed")
    query = {"schema": "rinkspace.query_constraints.v1", "arrays": {key: {"path": f"arrays/query_{key}.npy", "dtype": np.dtype(desc["dtype"]).str,
                      "shape": desc["shape"], "sha256": digest(out / f"arrays/query_{key}.npy")} for key, desc in inp["query"]["arrays"].items()},
             "definition": inp["query"]["definition"], "parameter_definition": inp["query"]["parameter_definition"],
             "certificates": _legacy_json(inp["query"]["certificates"])}
    write_json(out / "queries.json", query)
    write_json(out / "observations.json", {"schema": "rinkspace.observations.v1", "arrays": refs, "metadata_ref": "observations.jsonl",
                                           "query_constraints_ref": "queries.json"})
    provenance = {"schema": "rinkspace.equivalent_observation_import.v1", "original_input": str(input_document.resolve()),
                  "original_input_sha256": digest(input_document), "source_production_replayed": False,
                  "user_reannotation_required": False, "raw_rows": n, "active_rows": int(active.sum()),
                  "group_count": len(groups), "independent_rows": len(independent), "formal_rank": groups_doc["formal_rank"],
                  "raw_uv_identical": True, "native_pts_identical": True, "group_whitening_identical": True,
                  "existing_marginal_replacements_preserved": len(replacements), "queries": inp["query"]["arrays"]["source_rows"]["shape"][0],
                  "finite_branch_source": str(Path(path_indices_path).resolve()), "measurement_semantics": "original descriptor copied verbatim; no centreline or boundary reinterpretation",
                  "current_corrected_observations": "delivered current corrected binding retained" if current_mode else "kept at original registered current input; not overwritten by frozen replay import",
                  "constraint_only_rows": len(constraint_rows), "statistical_rows": int(statistical_activity.sum())}
    write_json(out / "provenance.json", provenance)
    revision = hashlib.sha256(json.dumps({"original": provenance["original_input_sha256"], "scope": digest(scope_path),
                "arrays": {k: v["sha256"] for k, v in refs.items()}, "groups": digest(out / "noise_groups.json"),
                "scene": digest(out / "scene.json"), "rigs": digest(out / "rigs.json"), "shots": digest(out / "shots.json")}, sort_keys=True).encode()).hexdigest()
    current = {"schema": "rinkspace.input.v1", "dataset_id": scope["event_id"] + "/" + scope["programme_id"] + ("/current_corrected" if current_mode else "/d78_replay"),
               "revision": revision, "scene_ref": "scene.json", "rigs_ref": "rigs.json", "shots_ref": "shots.json",
               "sources_ref": "sources.json", "observations_ref": "observations.json", "features_ref": "features.json",
               "noise_groups_ref": "noise_groups.json", "provenance_ref": "provenance.json",
               "model_configuration": {"measurement_policy_id": POLICY_ID, "state_kind": "current_corrected_static_import" if current_mode else "frozen_d78_replay", "fit_executed": False,
                                       "native_current_extended_entity_support": False if current_mode else True, "query_constraints_ref": "queries.json",
                                       "required_shot_ids": scope["required_shot_ids"], "available_camera_shot_ids": ["s01"]}}
    write_json(out / "input.json", current)
    return (out / "input.json").resolve()


def import_current_observations(current_dir, scope_path, pts_index_path, out_dir):
    """Migrate delivered current corrected observations through the same adapter.

    Native fitting of the extended current material inventory is a separate
    capability. This function preserves data and never claims it was fitted.
    """
    directory = Path(current_dir).resolve(strict=True)
    return import_d78_observations(directory, directory / "model.json", scope_path,
                                   pts_index_path, out_dir, path_indices_path=directory / "path.u32")
