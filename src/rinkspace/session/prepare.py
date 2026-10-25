"""Prepare a selected current input revision without replaying its production."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import numpy as np

from ..domain.scene_assets import snapshot_scene, scene_content_identity
from ..domain.runtime_rigs import snapshot_runtime
from .dataset import (DatasetError, PreparedReceipt, digest, load_array, read_json,
                      relative, require, write_array, write_json, iter_current_observations,
                      validate_current_noise_groups)

REQUIRED_ARRAYS = {
    "uv": ("<f8", (2,)), "pts": ("<i8", ()),
    "source_index": ("<i4", ()), "shot_index": ("<i4", ()),
    "feature_index": ("<i4", ()), "kind": ("|u1", ()),
    "active": ("|u1", ()), "split": ("|u1", ()),
    "noise_group_index": ("<i4", ()), "covariance": ("<f8", (3,)),
    "support": ("<f8", (2,)), "local_index": ("<i8", ()),
}


def select_shots(scope, shot_ids):
    required = scope["required_shot_ids"]
    require(required and len(set(required)) == len(required), "Required shot registry contains duplicates")
    chosen = list(required) if shot_ids == "all" else (shot_ids.split(",") if isinstance(shot_ids, str) else list(shot_ids))
    require(chosen and len(chosen) == len(set(chosen)), "Unknown or duplicate selected shot IDs", selected=chosen)
    require(all(s in required for s in chosen), "Unknown selected shot ID", selected=chosen, required=required)
    return chosen


def validate_current_feature_contracts(inp, scene, features, arrays, rows, *, scene_path=None, print_bindings=None, core=None):
    """Current changed model semantics only; frozen replay is left unchanged."""
    from ..domain.measurement import POLICY_ID, validate_feature_bindings, validate_scene_measurements
    from ..domain.board_station import is_boundary_curve, validate_boundary_definition
    from ..domain.print_assets import resolve_feature_print_bindings
    from ..domain.scene_assets import read_scene_assets
    if print_bindings is None:
        require(scene_path is not None or not scene.get('advertisement_assets'),
                'Current print features require the scene asset location')
        assets = read_scene_assets(scene, scene_path) if scene_path is not None else []
        active_rows = np.asarray(rows)[arrays['active'][rows] != 0]
        active_ids = {features[int(i)]['id'] for i in np.unique(arrays['feature_index'][active_rows])}
        print_bindings = resolve_feature_print_bindings(features, assets, active_feature_ids=active_ids)
    policy = inp.get("model_configuration", {}).get("measurement_policy_id") or scene.get("measurement_policy_id") or POLICY_ID
    report = None
    active_rows = np.asarray(rows)[arrays["active"][rows] != 0]
    if policy:
        active_ids = {features[int(i)]["id"] for i in np.unique(arrays["feature_index"][active_rows])}
        report = validate_feature_bindings(features, active_ids, policy, print_bindings=print_bindings)
    if len(rows):
        kinds = {'finite_curve': 1, 'material_point': 0, 'fixed_junction': 0}
        expected_kind = np.asarray([kinds.get(f.get('kind'), -1) for f in features], dtype='i1')
        expected_kind = expected_kind[arrays['feature_index'][rows]]
        require(np.all(expected_kind >= 0) and np.array_equal(arrays['kind'][rows], expected_kind),
                'Observation numerical kind differs from its current feature kind')
    scene_report = validate_scene_measurements(scene, policy)
    paths = {p['id']: p for p in scene.get('paths', [])}
    for record in paths.values():
        if is_boundary_curve(record):
            try:
                validate_boundary_definition(record)
            except ValueError as exc:
                raise DatasetError('invalid_input', str(exc), {'path_id':record['id']}) from exc
    for feature in features:
        path = paths.get(feature.get('path_id', feature['id']), {})
        if is_boundary_curve(feature) or (feature.get('kind') == 'finite_curve' and is_boundary_curve(path)):
            try:
                validate_boundary_definition(feature)
            except ValueError as exc:
                raise DatasetError('invalid_input', str(exc), {'feature_id':feature['id']}) from exc
    support_kind = arrays.get("support_kind")
    if support_kind is not None:
        require(support_kind.dtype == np.dtype("u1") and support_kind.shape == arrays["pts"].shape
                and np.isin(support_kind, [0, 1]).all(), "Invalid curve support-kind contract")
    else:
        support_kind = np.zeros(len(arrays["pts"]), dtype="u1")
    periodic_features = np.array([f.get("station_domain", {}).get("kind") == "unbounded_periodic" for f in features])
    expected = periodic_features[arrays["feature_index"]]
    require(np.all((support_kind[rows] == 1) == expected[rows]),
            "Continuous board station requires explicit support_kind=1; segment bounds cannot be reused")
    periodic = rows[support_kind[rows] == 1]
    if len(periodic):
        require(np.all(arrays["kind"][periodic] == 1) and np.all(arrays["local_index"][periodic] >= 0),
                "Periodic board station must own a curve local variable")
        require(np.all(arrays["support"][periodic] == 0), "Periodic support values must be non-binding zero placeholders")
        for feature_index in np.unique(arrays["feature_index"][periodic]):
            feature = features[int(feature_index)]
            require(feature.get("native_kind") == 7 and feature["station_domain"].get("hard_segment_bounds") is False,
                    "Continuous board feature cannot retain a segment-limited native path", feature_id=feature["id"])
    from ..domain.junctions import validate_junction_records
    junctions = validate_junction_records(scene, features, arrays, rows, core=core,
                                         print_bindings=print_bindings)
    return {"measurement": report, "scene_measurement": scene_report, "junctions": junctions,
            "periodic_station_rows": int(len(periodic)), "periodic_zero_support_is_not_bounds": True}


def prepare(input_path, scope_path, shot_ids, out_dir, *, core=None, local_charts=None):
    source_path = Path(input_path).resolve(strict=True)
    root = source_path.parent
    inp = read_json(source_path)
    scope = read_json(scope_path)
    require(inp.get("schema") == "rinkspace.input.v1", "Expected current rinkspace.input.v1")
    selected = select_shots(scope, shot_ids)
    for field in ("dataset_id", "revision", "scene_ref", "rigs_ref", "shots_ref", "sources_ref",
                  "observations_ref", "features_ref", "noise_groups_ref", "provenance_ref"):
        require(field in inp, "Missing required input field", field=field)
    sources_doc = read_json(relative(root, inp["sources_ref"]))
    shots_doc = read_json(relative(root, inp["shots_ref"]))
    features_doc = read_json(relative(root, inp["features_ref"]))
    groups_doc = read_json(relative(root, inp["noise_groups_ref"]))
    obs = read_json(relative(root, inp["observations_ref"]))
    sources, shots, features = sources_doc["sources"], shots_doc["shots"], features_doc["features"]
    for table, label in ((sources, "source"), (shots, "shot"), (features, "feature")):
        require(len({x["id"] for x in table}) == len(table), "Duplicate stable registry ID", table=label)
    shot_lookup = {s["id"]: i for i, s in enumerate(shots)}
    source_lookup = {s["id"]: i for i, s in enumerate(sources)}
    require(all(s in shot_lookup for s in scope["required_shot_ids"]), "Input must retain every required shot registration")
    arrays = {k: load_array(root, v, verify=True) for k, v in obs["arrays"].items()}
    n = len(arrays.get("pts", []))
    for key, (dtype, tail) in REQUIRED_ARRAYS.items():
        require(key in arrays and arrays[key].dtype.str == dtype and arrays[key].shape == (n,) + tail,
                "Observation array violates current layout", array=key)
    for key, size in (("source_index", len(sources)), ("shot_index", len(shots)), ("feature_index", len(features))):
        require(np.all((arrays[key] >= 0) & (arrays[key] < size)), "Registry index out of bounds", array=key)
    require(np.isin(arrays["kind"], [0, 1]).all() and np.isin(arrays["active"], [0, 1]).all()
            and np.isin(arrays["split"], [0, 1, 2]).all(), "Unknown kind, activity or split")
    require(np.isfinite(arrays["uv"]).all(), "Nonfinite source pixels")
    fixed = arrays["kind"] == 0
    require(np.all(arrays["local_index"][fixed] == -1) and np.all(arrays["support"][fixed] == 0),
            "Fixed material points cannot acquire per-observation stations")
    require(np.all(arrays["support"][~fixed, 0] <= arrays["support"][~fixed, 1]), "Reversed finite curve support")
    mask = np.isin(arrays["shot_index"], [shot_lookup[s] for s in selected])
    rows = np.flatnonzero(mask)
    new_rows = np.full(n, -1, np.int64)
    new_rows[rows] = np.arange(len(rows), dtype=np.int64)
    scene_doc = read_json(relative(root, inp["scene_ref"]))
    from ..domain.print_assets import resolve_feature_print_bindings
    from ..domain.scene_assets import read_scene_assets
    scene_path = relative(root, inp["scene_ref"])
    active_rows = rows[arrays['active'][rows] != 0]
    active_ids = {features[int(i)]['id'] for i in np.unique(arrays['feature_index'][active_rows])}
    print_bindings = resolve_feature_print_bindings(features, read_scene_assets(scene_doc, scene_path),
                                                   active_feature_ids=active_ids)
    current_contracts = validate_current_feature_contracts(inp, scene_doc, features, arrays, rows,
                                                          print_bindings=print_bindings, core=core)
    out = Path(out_dir).resolve()
    require(not out.exists(), "Prepared output already exists; use a new revision", path=str(out))
    # Resolve all selected exposure domains before any output mutation.
    source_pts = {}
    for source in sources:
        pts = load_array(root, source["pts_ref"], verify=True)
        require(pts.dtype.str == "<i8" and pts.ndim == 1 and np.all(np.diff(pts) > 0), "PTS registry must contain strictly ordered native integers")
        source_pts[source["id"]] = pts
    coverage = []
    for sid in selected:
        shot = shots[shot_lookup[sid]]
        require(shot.get("camera_domain", {}).get("status") == "defined", "Selected shot has no camera state domain", shot_id=sid)
        source_id = shot["source_id"]
        require(source_id in source_lookup, "Selected shot references an absent source", shot_id=sid, source_id=source_id)
        wanted = next(s for s in scope["shots"] if s["id"] == sid)["processing"]
        require(shot["processing"] == wanted, "Shot processing range differs from scope", shot_id=sid)
        # Source membership alone cannot establish shot membership: two videos
        # can contain the same integer PTS, and one video's index spans cuts.
        # Validate every retained row, including inactive rows, without changing
        # UV, PTS, association or activity to make inconsistent data fit.
        rr = rows[arrays["shot_index"][rows] == shot_lookup[sid]]
        wrong_source = rr[arrays["source_index"][rr] != source_lookup[source_id]]
        require(not len(wrong_source), "Observation source differs from its shot source",
                shot_id=sid, expected_source_id=source_id,
                source_row=int(wrong_source[0]) if len(wrong_source) else None)
        wrong_time = rr[(arrays["pts"][rr] < wanted["first_pts"]) |
                        (arrays["pts"][rr] >= wanted["stop_pts_exclusive"])]
        require(not len(wrong_time), "Observation exposure is outside its shot processing domain",
                shot_id=sid, first_pts=wanted["first_pts"], stop_pts_exclusive=wanted["stop_pts_exclusive"],
                source_row=int(wrong_time[0]) if len(wrong_time) else None,
                pts=int(arrays["pts"][wrong_time[0]]) if len(wrong_time) else None)
        pts = source_pts[source_id]
        pts = pts[(pts >= wanted["first_pts"]) & (pts < wanted["stop_pts_exclusive"])]
        require(len(pts) == wanted["native_frame_count"] and int(pts[0]) == wanted["first_pts"] and int(pts[-1]) == wanted["last_pts"],
                "Selected shot is missing registered native exposures", shot_id=sid)
        domain = shot["camera_domain"]
        require(domain["first_pts"] <= wanted["first_pts"] and domain["stop_pts_exclusive"] >= wanted["stop_pts_exclusive"], "Camera state does not cover selected exposure domain", shot_id=sid)
        coverage.append({"shot_id": sid, "first_pts": int(pts[0]), "last_pts": int(pts[-1]),
                         "stop_pts_exclusive": wanted["stop_pts_exclusive"], "frames": len(pts)})
    for source_number, source in enumerate(sources):
        rr = rows[arrays["source_index"][rows] == source_number]
        require(np.isin(arrays["pts"][rr], source_pts[source["id"]]).all(), "Observation exposure absent from actual source PTS index", source_id=source["id"])
        uv = arrays["uv"][rr]
        require(np.all((uv[:, 0] >= 0) & (uv[:, 0] < source["width"]) & (uv[:, 1] >= 0) & (uv[:, 1] < source["height"])), "UV outside native source image", source_id=source["id"])
    members = load_array(root, groups_doc["members_ref"], verify=True)
    weights = load_array(root, groups_doc["weights_ref"], verify=True)
    group_supports = validate_current_noise_groups(groups_doc['groups'], members, weights, arrays)
    group_rows, group_weights, groups, group_map = [], [], [], {}
    for ordinal, (group, (rr, w)) in enumerate(zip(groups_doc['groups'], group_supports)):
        take = mask[rr]
        require(not np.any(take) or np.all(take), "Selected shots truncate a correlation group", group_id=group["id"])
        if not np.any(take):
            continue
        group_map[ordinal] = len(groups)
        groups.append(dict(group, member_start=len(group_rows), weight_start=len(group_weights)))
        group_rows.extend(new_rows[rr].tolist())
        group_weights.extend(w.tolist())
    active = arrays["active"] != 0
    correlated = arrays["noise_group_index"] >= 0
    covariance = arrays["covariance"]
    require(np.all(covariance[correlated] == 0), "Correlated rows cannot also carry independent covariance")
    independent = active & ~correlated
    xx, xy, yy = covariance[independent].T
    require(np.all((xx > 0) & (yy > 0) & (xx * yy > xy * xy)), "Active independent row lacks positive covariance")
    admitted = np.array([f.get("identity") == "confirmed" and f.get("surface") in ("ice", "board") for f in features])
    require(np.all(admitted[arrays["feature_index"][active]]), "An active row has unconfirmed or out-of-model entity")
    # Preserve the complete current numerical scene, including its declared
    # print assets. Provenance references are metadata, not runtime producers.
    scene_snapshot, scene_dependencies = snapshot_scene(relative(root, inp['scene_ref']), out)
    rigs_snapshot, shots_snapshot, rig_dependencies = snapshot_runtime(
        relative(root, inp['rigs_ref']), relative(root, inp['shots_ref']), out)
    shots = shots_snapshot['shots']
    refs = {}
    for key, values in arrays.items():
        if values.shape[:1] == (n,):
            values = np.asarray(values[rows])
            if key == "noise_group_index":
                values = np.array([group_map.get(int(v), -1) for v in values], dtype="<i4")
            refs[key] = write_array(out, key, values)
    refs["noise_group_members"] = write_array(out, "noise_group_members", group_rows, "<i8")
    refs["noise_group_weights"] = write_array(out, "noise_group_weights", group_weights, "<f8")
    for source in sources:
        source["pts_ref"] = write_array(out, "pts_" + source["id"], source_pts[source["id"]], "<i8")
    # All static documents are copied once into the prepared asset root.
    for key in ("provenance_ref",):
        shutil.copy2(relative(root, inp[key]), out / (key.removesuffix("_ref") + ".json"))
    queries_ref = obs.get("query_constraints_ref")
    if queries_ref:
        queries = read_json(relative(root, queries_ref))
        compiled_query_arrays = {}
        for name, desc in queries.get("arrays", {}).items():
            values = load_array(root, desc, verify=True)
            compiled_query_arrays[name] = write_array(out, "query_" + name, values)
        queries["arrays"] = compiled_query_arrays
        queries["source_row_identity_preserved"] = True
        write_json(out / "queries.json", queries)
    write_json(out / "features.json", features_doc)
    write_json(out / "scope.json", scope)
    # Row metadata can be large; preserve exact records/order in a bounded sidecar.
    metadata = obs.get("metadata_ref")
    if metadata:
        with (out / "observations.jsonl").open("w", encoding="utf-8", newline="\n") as dst:
            count = 0
            for line in iter_current_observations(
                    relative(root, metadata), features, arrays['feature_index'],
                    mask=mask, print_bindings=print_bindings, arrays=arrays,
                    sources=sources, shots=shots, noise_groups=groups_doc['groups']):
                dst.write(line)
                count += 1
        require(count == len(rows), "Observation metadata row count differs from arrays")
    from ..domain.parameters import compile_scene_parameters, write_parameter_state, SceneParameterModel
    parameter_state = compile_scene_parameters(scene_snapshot, out/'scene.json', local_charts=local_charts)
    compiled_scene = SceneParameterModel(scene_snapshot, out/'scene.json', local_charts=local_charts)
    feature_dependencies = {}
    for feature in features:
        if feature.get('print_binding') is not None:
            path_id = feature.get('path_id', feature['id'])
            feature_dependencies[feature['id']] = compiled_scene.dependencies(path_id)
    write_json(out/'parameter_dependencies.json', {'schema':'rinkspace.parameter_dependencies.v1',
        'layout_revision':parameter_state.layout['layout_revision'], 'features':feature_dependencies,
        'source_rows_or_groups_changed':False})
    from ..domain.parameters import FactorPlan
    write_json(out/'factor_plan.json', FactorPlan(parameter_state, feature_dependencies,
        scope='prepared_print_geometry_dependencies_not_complete_camera_factors').to_dict())
    initial_state = write_parameter_state(out, parameter_state, shots=shots,
        required_shot_ids=scope['required_shot_ids'], selected_shot_ids=selected, name='initial_state.json')
    scope_status = "complete" if set(selected) == set(scope["required_shot_ids"]) else "subset"
    # A caller-supplied label is not a content identity. Bind exactly the current
    # dependencies consumed by this prepared revision, without walking history.
    dependency_revision = {
        "input": inp["revision"], "scope": digest(scope_path),
        "scene": scene_content_identity(scene_snapshot, out/'scene.json'),
        "rigs": digest(out/'rigs.json'), "shots": digest(out/'shots.json'),
        "rig_assets": {item['snapshot_path']: item['sha256'] for item in rig_dependencies},
        "features": digest(out/'features.json'), "provenance": digest(out/'provenance.json'),
        "observations": digest(out/'observations.jsonl') if metadata else None,
        "queries": digest(out/'queries.json') if queries_ref else None,
        "parameter_layout": digest(out/'parameter_layout.json'),
        "factor_plan": digest(out/'factor_plan.json'),
        "initial_parameters": initial_state['values_ref']['sha256'],
        "parameter_dependencies": digest(out/'parameter_dependencies.json'),
    }
    revision = hashlib.sha256(json.dumps({"input": inp["revision"], "selected": selected,
                 "scope": scope, "dependencies": dependency_revision,
                 "sources": sources, "shots": shots, "noise_groups": groups,
                 "model_configuration": inp.get('model_configuration', {}),
                 "arrays": {k: v["sha256"] for k, v in refs.items()}},
                 sort_keys=True, allow_nan=False).encode()).hexdigest()
    manifest = {"schema": "rinkspace.prepared.v1", "dataset_id": inp["dataset_id"], "revision": revision,
                "input_revision": inp["revision"], "scope_ref": "scope.json", "required_shot_ids": scope["required_shot_ids"],
                "selected_shot_ids": selected, "scope_status": scope_status, "coverage": coverage,
                "sources": sources, "shots": shots, "rigs": rigs_snapshot.get("rigs", []),
                "features": features, "noise_groups": groups, "arrays": refs, "scene_ref": "scene.json",
                "features_ref": "features.json", "provenance_ref": "provenance.json",
                "observations_ref": "observations.jsonl" if metadata else None,
                "model_configuration": inp.get("model_configuration", {}),
                "query_constraints_ref": "queries.json" if queries_ref else None,
                "dependency_revision": dependency_revision,
                "snapshot_dependencies": {"scene": scene_dependencies, "rigs": rig_dependencies},
                "parameter_layout_ref":"parameter_layout.json", "initial_state_ref":"initial_state.json",
                "factor_plan_ref":"factor_plan.json",
                "parameter_dependencies_ref":"parameter_dependencies.json",
                "current_contracts": current_contracts, "observations": len(rows), "active_observations": int(arrays["active"][rows].sum()),
                "user_approval": "not_reviewed", "fit_executed": False}
    initial_state['prepared_revision'] = revision
    write_json(out/'initial_state.json', initial_state)
    write_json(out / "manifest.json", manifest)
    return PreparedReceipt(str((out / "manifest.json").resolve()), inp["dataset_id"], revision, tuple(selected), scope_status)
