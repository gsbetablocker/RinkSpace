"""Compose current static artwork and accepted original-native placements.

This is a model-format adapter for complete review. It starts no optimizer,
adds no observations and preserves original camera, ice and board parameters.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "calibration" / "rinkfit" / "src"))
sys.path.insert(0, str(ROOT / "calibration" / "solve2"))
from rinkfit_start.native_scene import NativeScene
from rinksolve.model import load_model_bundle, plain

VERSION = "rinkfit.complete_current_native_artwork.v1"
IDENTITY = [1., 0., 0., 0., 1., 0.]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked(path):
    path = Path(path).resolve()
    units = lambda text: len(str(text).encode("utf-16-le")) // 2
    if units(path) >= 260 or any(units(part) > 255 for part in path.parts):
        raise ValueError("Windows path too long: " + str(path))
    return path


def reference(path):
    path = checked(path)
    return dict(path=str(path), sha256=sha(path))


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def write(path, value):
    path = checked(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf8")


def ownership(curve, definition):
    match = re.search(r"(?:^|\/)Ad(\d+)(?:[.\/]|$)", curve.label)
    if match:
        return int(match.group(1))
    match = re.fullmatch(r"artwork_group/Ad(\d+)/whole_material_object/similarity", definition.get("shared_layout_block_id", ""))
    return None if match is None else int(match.group(1))


def source_vertex_stations(definition):
    kind = definition["kind"]
    if kind == "board_material_polyline":
        points = definition["unwrapped_station_height"]
    elif kind == "installed_artwork_polyline":
        points = definition["normalized_points"]
    else:
        raise ValueError("Current advertisement is not a finite board material: " + kind)
    if not points:
        raise ValueError("Empty advertisement curve")
    return np.linspace(0., 1., len(points)) if len(points) > 1 else np.array([0.])


def current_paths(model, ad_index):
    result = {}
    for mi, material in enumerate(model["materials"]):
        if material.get("unused_legacy_witness", False):
            continue
        for pi, path in enumerate(material["paths"]):
            if path.get("advertisement_block") != ad_index:
                continue
            cid = path.get("id")
            if cid is None:
                try:
                    cid = json.loads(path.get("provenance", "{}"))["current_curve_id"]
                except (ValueError, KeyError):
                    raise ValueError("Accepted current path lacks stable source identity")
            if cid in result:
                raise ValueError("Duplicate accepted current curve: " + cid)
            result[cid] = (mi, pi, path)
    return result


def compose(source_directory, original_path, output, overlays=(), rejected_overlays=()):
    original_path = checked(original_path)
    source_directory = checked(source_directory)
    output = checked(output)
    bundle = load_model_bundle(source_directory)
    base = read(original_path)
    model = deepcopy(base)
    names = {b["name"]: i for i, b in enumerate(base["blocks"])}
    ad_indices = {ad: names[f"ad/{ad}"] for ad in range(2, 13)}
    for block in model["blocks"]:
        block["constant"] = True
    reset = bundle.state.updated({f"artwork_group/Ad{ad}/whole_material_object/similarity": np.zeros(4) for ad in range(2, 13)})
    board = bundle.scene.board(reset)
    materials = []
    mapping = {}
    inventories = []
    q_indices = {}
    for ad, ai in ad_indices.items():
        witness = np.asarray(base["blocks"][ai]["values"], float)
        origin = witness[[0, 2]]
        span = np.array([witness[1], witness[3] - witness[2]])
        if np.any(span <= 0):
            raise ValueError("Noninvertible original advertisement chart")
        qids = sorted({int(p["geometry_block"]) for m in base["materials"] for p in m["paths"]
                       if p.get("advertisement_block") == ai and int(p.get("geometry_block", -1)) >= 0})
        if not qids:
            raise ValueError(f"Missing original affine domain for Ad{ad}")
        qdef = base["blocks"][qids[0]]
        if any((base["blocks"][i]["lower"], base["blocks"][i]["upper"], base["blocks"][i]["scales"]) !=
               (qdef["lower"], qdef["upper"], qdef["scales"]) for i in qids):
            raise ValueError(f"Original Ad{ad} affine domains differ")
        qi = len(model["blocks"])
        q_indices[ad] = qi
        model["blocks"].append(dict(name=f"print/Ad{ad}_current_whole", values=IDENTITY.copy(),
                                    lower=qdef["lower"], upper=qdef["upper"], scales=qdef["scales"], constant=True))
        count = vertices = display_only = aliases = 0
        for curve in bundle.scene.curves:
            definition = bundle.scene.resolved_definition(curve, reset)
            if ownership(curve, definition) != ad:
                continue
            stations = source_vertex_stations(definition)
            unwrapped = bundle.scene.unwrapped_path(curve.id, stations, reset, board=board)
            local = (unwrapped - origin) / span
            if local.shape != (len(stations), 2) or not np.isfinite(local).all():
                raise ValueError("Invalid delivered current source: " + curve.id)
            is_outer = ad < 8 and "/installation" in str(definition.get("original_path_id", ""))
            path = dict(id=curve.id, variant_id=curve.id, kind=2, advertisement_block=ai,
                        points=local.tolist(), closed=bool(definition.get("closed", False)),
                        physical_point=len(local) == 1, current_curve_id=curve.id,
                        display_only_initializer=bool(definition.get("display_only_initializer", False)),
                        provenance=json.dumps(dict(current_curve_id=curve.id, current_definition=plain(definition),
                                                   original_curve_definition=plain(curve.definition),
                                                   new_stage_similarity_reset_to_zero=True), ensure_ascii=False))
            if not is_outer:
                path["geometry_block"] = qi
            material = dict(label=curve.label, paths=[path], current_source_curve=True,
                            observed_installation_boundary=is_outer,
                            display_only_initializer=path["display_only_initializer"])
            if curve.kind == "curve_alias":
                material["source_alias_target"] = curve.definition["target_curve_id"]
                aliases += 1
            mapping[curve.id] = (len(materials), 0, ad)
            materials.append(material)
            count += 1
            vertices += len(local)
            display_only += int(path["display_only_initializer"])
        if not count:
            raise ValueError(f"Current source catalogue lacks Ad{ad}")
        inventories.append(dict(advertisement_id=ad, curves=count, source_vertices=vertices,
                                display_only_initializer_curves=display_only, aliases=aliases,
                                placement="original_d78", geometry_block=qi, advertisement_block=ai))
    legacy = deepcopy(base["materials"])
    for material in legacy:
        if any(p.get("advertisement_block") in ad_indices.values() for p in material["paths"]):
            material["unused_legacy_witness"] = True
    model["materials"] = materials + legacy
    model["current_input_scope"] = dict(schema="rinkfit.current_advertisement_input.v1",
                                        scope_id="Ad2--12_complete_current_artwork_review", print_only_ad_ids=[])
    model["model_id"] = base["model_id"] + "/complete_current_artwork_review"
    model["print_only_ad_ids"] = []
    overlay_records = []
    overlay_models = {}
    for ad, path in overlays:
        ad = int(ad)
        if ad not in ad_indices or ad in overlay_models:
            raise ValueError("Invalid or duplicate accepted placement overlay")
        accepted = read(path)
        accepted_names = {b["name"]: i for i, b in enumerate(accepted["blocks"])}
        ai = ad_indices[ad]
        accepted_ai = accepted_names[f"ad/{ad}"]
        for i, block in enumerate(base["blocks"]):
            if i == ai:
                continue
            candidate = accepted["blocks"][accepted_names[block["name"]]]
            if not np.array_equal(block["values"], candidate["values"]):
                raise ValueError("Accepted per-ad overlay changed another original block: " + block["name"])
        accepted_paths = current_paths(accepted, accepted_ai)
        qids = {int(p.get("geometry_block", -1)) for _, _, p in accepted_paths.values()} - {-1}
        if len(qids) != 1:
            raise ValueError("Accepted whole artwork needs exactly one shared affine")
        q = accepted["blocks"][next(iter(qids))]["values"]
        model["blocks"][ai]["values"] = accepted["blocks"][accepted_ai]["values"]
        model["blocks"][q_indices[ad]]["values"] = q
        for item in inventories:
            if item["advertisement_id"] == ad:
                item["placement"] = "accepted_original_native_overlay"
        overlay_models[ad] = (accepted, accepted_paths)
        overlay_records.append(dict(advertisement_id=ad, model=reference(path),
                                    advertisement_values=model["blocks"][ai]["values"], affine_values=q))
    scene = NativeScene(model)
    counts = {"source": 0, "accepted": 0}
    maxima = {"source": 0., "accepted": 0.}
    accepted_scenes = {ad: NativeScene(item[0]) for ad, item in overlay_models.items()}
    source_curves = {curve.id: curve for curve in bundle.scene.curves}
    missing_accepted = []
    for curve in bundle.scene.curves:
        if curve.id not in mapping:
            continue
        mi, pi, ad = mapping[curve.id]
        definition = bundle.scene.resolved_definition(curve, reset)
        vertices = source_vertex_stations(definition)
        stations = np.unique(np.r_[vertices, (vertices[:-1] + vertices[1:]) / 2., np.linspace(0., 1., 33)])
        actual = scene.world(mi, pi, stations)
        if ad in overlay_models:
            paths = overlay_models[ad][1]
            comparison_id = curve.id
            visited = set()
            while comparison_id not in paths and source_curves[comparison_id].kind == "curve_alias":
                if comparison_id in visited:
                    raise ValueError("Cyclic source curve alias")
                visited.add(comparison_id)
                comparison_id = source_curves[comparison_id].definition["target_curve_id"]
            if comparison_id not in paths:
                missing_accepted.append(curve.id)
                continue
            ami, api, apath = paths[comparison_id]
            own = model["materials"][mi]["paths"][pi]
            own_points = np.asarray(own["points"], float)
            accepted_points = np.asarray(apath["points"], float)
            if own_points.shape != accepted_points.shape or not np.allclose(own_points, accepted_points, atol=1e-12, rtol=0.) or own["closed"] != apath.get("closed", False):
                raise ValueError("Accepted current source shape differs from complete source catalogue: " + curve.id)
            expected = accepted_scenes[ad].world(ami, api, stations)
            key = "accepted"
        else:
            expected = bundle.scene.world_path(curve.id, stations, reset, board=board)
            key = "source"
        maxima[key] = max(maxima[key], float(np.max(np.abs(actual - expected), initial=0.)))
        counts[key] += len(stations)
    if missing_accepted:
        raise ValueError("Accepted target model lacks delivered current curves: " + str(missing_accepted))
    if max(maxima.values()) > 1e-10:
        raise ValueError("Native full artwork changed current world mapping: " + str(maxima))
    changed_original = [b["name"] for i, b in enumerate(base["blocks"])
                        if not np.array_equal(b["values"], model["blocks"][i]["values"])]
    permitted = {f"ad/{ad}" for ad in overlay_models}
    if set(changed_original) - permitted:
        raise ValueError("Original camera/ice/board or other fixed parameters changed")
    rejected_records = [dict(advertisement_id=int(ad), model=reference(path),
                             reason="excluded_by_visual_review", parameters_applied=False)
                        for ad, path in rejected_overlays]
    if {item["advertisement_id"] for item in rejected_records} & set(overlay_models):
        raise ValueError("A visually rejected placement cannot be applied")
    model.setdefault("provenance", {})["complete_current_artwork"] = dict(
        version=VERSION, source_catalogue=str(source_directory), source_state_sha256=bundle.state.digest(),
        original_model=reference(original_path), accepted_native_placements=overlay_records,
        rejected_native_placements=rejected_records,
        camera_ice_board_original_values_bitexact=True, original_changed_blocks=changed_original,
        observations_created=0, source_vertices_are_new_observations=False, optimizer_started=False,
        source_detail_does_not_imply_native_frame_qualification=True, review_only=True)
    model_path = output / "model.json"
    write(model_path, model)
    manifest = dict(schema=VERSION, model=reference(model_path), source_catalogue=str(source_directory),
                    original_model=reference(original_path), accepted_native_placements=overlay_records,
                    rejected_native_placements=rejected_records,
                    advertisements=inventories, current_curves=len(mapping), legacy_materials=len(legacy),
                    enabled_legacy_nonadvertisement_materials=sum(not m.get("unused_legacy_witness", False) for m in legacy),
                    original_parameter_blocks=len(base["blocks"]), total_parameter_blocks=len(model["blocks"]),
                    original_changed_blocks=changed_original, camera_ice_board_original_values_bitexact=True,
                    world_mapping_max_absolute_error=maxima, verified_world_samples=counts,
                    no_new_observations_or_constraints=True, review_only=True, whole_shot_acceptance=False,
                    legacy_render_rule="skip unused_legacy_witness only when its advertisement has a current replacement")
    write(output / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model", required=True)
    parser.add_argument("--original-model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--overlay", action="append", default=[], help="Accepted native placement as AD_ID=MODEL_PATH")
    parser.add_argument("--rejected-overlay", action="append", default=[], help="Visually rejected native placement retained only as a reference: AD_ID=MODEL_PATH")
    args = parser.parse_args()
    overlays = []
    for item in args.overlay:
        ad, separator, path = item.partition("=")
        if not separator:
            parser.error("--overlay requires AD_ID=MODEL_PATH")
        overlays.append((int(ad), path))
    rejected = []
    for item in args.rejected_overlay:
        ad, separator, path = item.partition("=")
        if not separator:
            parser.error("--rejected-overlay requires AD_ID=MODEL_PATH")
        rejected.append((int(ad), path))
    result = compose(args.source_model, args.original_model, args.output, overlays, rejected)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
