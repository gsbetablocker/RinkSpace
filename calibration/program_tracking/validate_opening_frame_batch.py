"""Verify all 100 opening source packets against the actual model residual API.

This is an entry diagnostic with explicit sigma=1. It never optimizes, changes
measurements, selects formal weights, or accepts image/model accuracy.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
from rinkspace.packed_assets import iter_project_assets, materialize_project_asset, read_project_json
DEFAULT_DATASET = ROOT / "events/ankara_2026/programmes/okada_mei_sp/source_datasets/frame_by_frame_review_20261013/opening_100_50fps"
DEFAULT_MODEL = ROOT / "events/ankara_2026/models/framewise_8a28_20261013"
DEFAULT_OUTPUT = Path("D:/CodexData/RinkSpace_Data/events/ankara_2026/programmes/okada_mei_sp/results/frame_by_frame_review_20261013/opening_100_50fps/model_entry_validation.json")
APPROVED_SHA256 = "a00f829190cc531f13f1d09a91882b7a4f6dc09d99c203b2958b029e4b5db28a"
FIRST_PTS, STEP, COUNT = 256256, 256, 100
CORNER_ALIASES = {"top_left": "left_top", "bottom_left": "left_bottom", "top_right": "right_top", "bottom_right": "right_bottom"}
PRINT_LABELS = {"S": "SKATING_S", "left_S_visible_fragment": "ISU_S", "right_cropped_letter": "K", "A": "SKATING_A", "A_inner": "SKATING_A_inner", "outermost_visible_circular_print_stroke": "outer_circle_print_stroke"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return read_project_json(path, project_root=ROOT)


def sha(path):
    with materialize_project_asset(path, project_root=ROOT).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def reference(path):
    return {"path": str(Path(path).resolve()), "sha256": sha(path)}


def integer(value, label):
    require(isinstance(value, int) and not isinstance(value, bool), label + " must be an integer")
    return value


def source_vertices(packet):
    """Return coordinate addresses; point IDs remain opaque strings."""
    features, vertices = {}, {}
    for feature in packet["features"]:
        fid = feature["id"]
        require(isinstance(fid, str) and fid and fid not in features, "Duplicate/invalid source feature ID")
        features[fid] = feature
        for si, segment in enumerate(feature["segments"]):
            uv = np.asarray(segment["uv"], dtype=np.float64)
            require(uv.ndim == 2 and uv.shape[1:] == (2,) and len(uv) > 0 and np.isfinite(uv).all(), "Invalid source UV: " + fid)
            require(((uv[:, 0] >= 0) & (uv[:, 0] <= 1919) & (uv[:, 1] >= 0) & (uv[:, 1] <= 1079)).all(), "Source UV outside native image: " + fid)
            if "vertex_count" in segment:
                require(segment["vertex_count"] == len(uv), "Source vertex_count mismatch: " + fid)
            if "point_bindings" in segment:
                require(len(segment["point_bindings"]) == len(uv), "Per-vertex physical binding count mismatch: " + fid)
            if "point_ids" in segment:
                require(len(segment["point_ids"]) == len(uv), "Explicit source point ID count mismatch: " + fid)
            for vi, point in enumerate(uv):
                vertices[(fid, si, vi)] = point
    return features, vertices


def ordered_uv(packet):
    features, vertices = source_vertices(packet)
    return list(vertices), np.asarray(list(vertices.values()), dtype=np.float64)


def verify_physical_target(feature, segment, vi, descriptor):
    """Check semantic corners/edges and letter identity, not point-ID spelling."""
    bindings = segment.get("point_bindings")
    if feature.get('manual_source_observation'):
        declared = feature.get('physical_target_descriptor', {})
        for key in ('kind', 'advertisement_id', 'role', 'part', 'contour_key', 'curve_id'):
            require(declared.get(key) == descriptor.get(key), 'Manual point physical target changed: ' + key)
    if feature["id"].startswith("ice."):
        require(descriptor["kind"] == "ice_curve", "Ice source routed to a non-ice target")
        curve = feature.get("world_curve", feature["id"][4:])
        require(curve == descriptor.get("curve_id"), "Wrong physical ice circle/layer or line edge: " + feature["id"])
        require(feature["id"] == "ice." + curve, "Ice source label and physical curve disagree")
    if bindings is not None:
        binding = bindings[vi]
        physical = binding.get("physical_target", "")
        is_corner = binding.get("kind") == "corner"
        require(is_corner == (descriptor["kind"] == "ad_corner"), "Source corner routed as curve, or edge routed as corner")
        if physical.startswith("Ad") and "." in physical:
            name, role = physical.split(".", 1)
            require(name[2:].isdigit() and int(name[2:]) == descriptor.get("advertisement_id"), "Wrong advertisement identity")
            if descriptor['kind']=='ad_print':
                require(role=='print.'+descriptor['contour_key'], 'Wrong named print contour: '+physical)
            else:
                role = CORNER_ALIASES.get(role, role)
                require(descriptor.get("role") == role, "Wrong named advertisement corner/edge: " + physical)
        elif physical == "board_surface.top_edge_xyz":
            require(descriptor["kind"] == "board_edge" and descriptor["role"] == "top", "Wrong board top target")
        elif physical in ("ice_boundary.points_xyz", "board_surface.contact_edge_xyz"):
            require(descriptor["kind"] == "board_edge" and descriptor["role"] == "contact", "Wrong board contact target")
    if ".print." in feature["id"]:
        require(descriptor["kind"] == "ad_print", "Print source routed to a non-print target")
        ad_name = feature["id"].split(".", 1)[0]
        require(ad_name.startswith("Ad") and ad_name[2:].isdigit() and descriptor["advertisement_id"] == int(ad_name[2:]), "Wrong print advertisement")
        label = feature.get("label")
        if label:
            if label.startswith(feature["id"].split(".print.")[0] + ".print."):
                label = label.split(".print.", 1)[1]
            require(descriptor["contour_key"] in (label, PRINT_LABELS.get(label, label)), "Wrong print glyph/inner contour identity: "+feature['id']+" label="+label+" contour="+descriptor['contour_key'])
        if feature.get("source_contour_internal_hole"):
            require(descriptor.get("contour_role") == "white_print_internal_hole_boundary" or descriptor["contour_key"].endswith("_inner"), "Internal print hole routed to external contour")


def assemble(packet, binding, registry, scene, templates, *, apply_feedback_withdrawals=True):
    from calibration.program_tracking.framewise_model import build_row

    features, vertices = source_vertices(packet)
    count = integer(packet["point_count"], "point_count")
    require(count == len(vertices) and 0 < count <= 500, "Source point count inconsistent or exceeds500")
    require(binding["point_count"] == count, "Binding point_count mismatch")
    ids, addresses, observation_ids, rows, target_counts = set(), set(), set(), [], Counter()
    corners = []
    for observation in binding["observations"]:
        oid = observation["id"]
        require(isinstance(oid, str) and oid and oid not in observation_ids, "Duplicate/invalid observation ID")
        observation_ids.add(oid)
        fid = observation["source_feature_id"]
        require(fid in features, "Binding references missing source feature")
        si = integer(observation["source_segment_index"], "source_segment_index")
        require(0 <= si < len(features[fid]["segments"]), "Source segment index out of range")
        segment = features[fid]["segments"][si]
        indices, point_ids = observation["source_vertex_indices"], observation["source_point_ids"]
        require(len(indices) > 0 and len(indices) == len(point_ids), "Binding vertex/point-ID length mismatch")
        target = observation["target_id"]
        require(target in registry["targets"], "Unknown model target: " + target)
        descriptor = registry["targets"][target]
        point = observation["point_correspondence"]
        require(isinstance(point, bool) and point == (descriptor["kind"] == "ad_corner"), "Corner point_correspondence mismatch")
        require(not point or len(indices) == 1, "Named corner must bind exactly one source point")
        for vi, pid in zip(indices, point_ids):
            vi = integer(vi, "source_vertex_index")
            address = (fid, si, vi)
            require(address in vertices and address not in addresses, "Source point omitted/duplicated or vertex index invalid")
            require(isinstance(pid, str) and pid and pid not in ids, "Source point IDs must be unique nonempty opaque strings")
            if "point_ids" in segment:
                require(segment["point_ids"][vi] == pid, "Explicit source point identity mismatch")
            ids.add(pid)
            addresses.add(address)
            verify_physical_target(features[fid], segment, vi, descriptor)
            if point:
                corners.append({"source_point_id": pid, "source_address": list(address), "target_id": target, "physical_corner_role": descriptor["role"]})
        uv = np.asarray(segment["uv"], dtype=np.float64)[indices]
        row = build_row(scene, descriptor, uv, np.ones(len(uv)), templates, row_id=oid, point_correspondence=point, diagnostic_unit_weights=True)
        require(row["diagnostic_unit_weights"] and row["sigma_semantics"] == "diagnostic_unit_pixel_scale_not_formal_source_weight", "Diagnostic unit-weight declaration lost")
        rows.append(row)
        target_counts[target] += len(uv)
    require(addresses == set(vertices) and len(ids) == count, "All source coordinates must have exactly one unique target binding")
    if apply_feedback_withdrawals:
        from rinkspace.feedback_withdrawal import enforce_packet_withdrawals
        enforce_packet_withdrawals(packet,binding,project=ROOT,
            packet_path=binding.get('source_packet',{}).get('path'),gate='model_assembly')
    return rows, target_counts, corners


def validate(dataset=DEFAULT_DATASET, model=DEFAULT_MODEL, output=DEFAULT_OUTPUT):
    for path in (ROOT, ROOT / "src", ROOT / "calibration/program_tracking"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from calibration.program_tracking.framewise_model import evaluate_rows, feature_world

    dataset, model, output = map(Path, (dataset, model, output))
    report = {"schema": "rinkspace.opening100-model-entry-validation.v1", "status": "failed", "optimization_performed": False, "source_or_camera_or_model_modified": False, "diagnostic_weighting": "explicit_sigma_1_raw_pixel_residuals_only_not_formal_measurement_weights", "uncertainty_and_shared_line_covariance_replaced": False, "accuracy_or_user_acceptance_claimed": False, "frames": [], "errors": []}
    before = {}
    try:
        expected_names = {f"{i:06d}.json" for i in range(1, COUNT + 1)}
        for directory in (dataset / "frames", dataset / "bindings"):
            require({p.name for p in iter_project_assets(directory, "*.json", project_root=ROOT)} == expected_names, "Exactly100 complete frame/binding JSONs required: " + str(directory))
        approved_path = dataset.parent / "frame_000001_points.json"
        require(sha(approved_path) == APPROVED_SHA256, "Approved first-frame original SHA256 changed")
        approved = read(approved_path)
        require(approved["point_count"] == 99 and approved["user_accepted"], "Original first99 points are not the approved packet")
        nav = read(dataset / "navigation_mapping.json")
        seeds_path = Path(nav["model_seed"]["camera_seeds"]["path"])
        require(sha(seeds_path) == nav["model_seed"]["camera_seeds"]["sha256"], "Camera seed archive hash changed")
        manifest_path = seeds_path.parent / "frame_manifest.json"
        manifest = read(manifest_path)
        require(len(manifest["frames"]) == COUNT, "Native image manifest must contain100 frames")
        scene_path, registry_path, templates_path = (model / n for n in ("scene.json", "feature_registry.json", "ad2_material_template.json"))
        scene, registry, templates = map(read, (scene_path, registry_path, templates_path))
        require(scene["model_id"] == registry["model_id"], "Scene/registry model identities differ")
        tracked = [approved_path, dataset / "navigation_mapping.json", seeds_path, manifest_path, scene_path, registry_path, templates_path]
        tracked += [dataset / directory / name for directory in ("frames", "bindings") for name in sorted(expected_names)]
        before = {str(p.resolve()): sha(p) for p in tracked}
        for tid, descriptor in registry["targets"].items():
            require(np.isfinite(feature_world(scene, descriptor, templates)).all(), "Nonfinite registered model target: " + tid)
        target_totals = Counter()
        with np.load(seeds_path, allow_pickle=False) as seeds:
            expected_pts = np.arange(FIRST_PTS, FIRST_PTS + COUNT * STEP, STEP, dtype=np.int64)
            require(np.array_equal(seeds["official_pts"], expected_pts), "Camera seed official clock differs from exact100 frames")
            require(seeds["camera25"].shape == (COUNT, 25) and np.isfinite(seeds["camera25"]).all(), "Invalid100 camera25 seeds")
            for index, native in enumerate(manifest["frames"], 1):
                frame_result = {"index": index, "official_pts": int(expected_pts[index - 1]), "status": "failed"}
                try:
                    packet_path, binding_path = (dataset / d / f"{index:06d}.json" for d in ("frames", "bindings"))
                    packet, binding = map(read, (packet_path, binding_path))
                    pts, source = int(expected_pts[index - 1]), packet["source"]
                    require(native["index"] == index and native["native_pts"] == pts, "Image manifest native clock is not exact")
                    require(packet["batch_sequence_index_1based"] == index and packet["frame_count"] == 1, "Packet sequence/frame_count mismatch")
                    require(source["native_pts"] == source["actual_pts"] == binding["source_pts"] == pts, "Packet/binding source PTS mismatch")
                    require(source["time_base"] == binding["source_time_base"] == "1/12800", "Official source timebase mismatch")
                    require(source["source_video_sha256"] == manifest["source"]["sha256"] and source["decoded_BGR_sha256"] == native["decoded_native_BGR_sha256"], "Packet references another source/exposure")
                    require(Path(source["image_path"]).resolve() == Path(native["path"]).resolve(), "Packet image path differs from native manifest")
                    require(binding["model_id"] == registry["model_id"] and binding["frame_count"] == 1, "Binding model/frame_count mismatch")
                    require(binding["source_user_accepted"] == packet["user_accepted"], "Binding acceptance record differs from packet")
                    require(Path(binding["source_packet"]["path"]).resolve() == packet_path.resolve() and binding["source_packet"]["sha256"] == sha(packet_path), "Binding source packet path/hash mismatch")
                    if index == 1:
                        old_addresses, old_uv = ordered_uv(approved)
                        new_addresses, new_uv = ordered_uv(packet)
                        require(packet["point_count"] == 99 and old_addresses == new_addresses and np.array_equal(old_uv, new_uv), "Approved first99 source UVs changed or reordered")
                        frame_result["approved_99_uv_exactly_unchanged"] = True
                        frame_result["approved_uv_float64_sha256"] = hashlib.sha256(old_uv.tobytes()).hexdigest()
                    camera = seeds["camera25"][index - 1]
                    navrow = nav["frames"][index - 1]
                    require(navrow["official_pts"] == pts and navrow["legacy60_pts"] == int(seeds["legacy60_pts"][index - 1]), "Navigation/camera seed clock mismatch")
                    require(hashlib.sha256(camera.tobytes()).hexdigest() == navrow["camera25_float64_sha256"], "Camera25 seed bytes differ from navigation binding")
                    rows, targets, corners = assemble(packet, binding, registry, scene, templates)
                    residual, jacobian = evaluate_rows(camera, rows, jacobian=True)
                    count = packet["point_count"]
                    require(residual.shape == (2 * count,) and jacobian.shape == (2 * count, 15), "Actual residual/J15 does not cover every source point")
                    require(np.isfinite(residual).all() and np.isfinite(jacobian).all(), "Actual residual/J15 contains nonfinite values")
                    errors = np.linalg.norm(residual.reshape(-1, 2), axis=1)
                    frame_result.update(status="passed", source_packet=reference(packet_path), binding=reference(binding_path), point_count=count, source_point_ids_unique_and_complete=True, coordinate_addresses_covered_exactly_once=True, corner_bindings=corners, target_point_counts=dict(targets), actual_residual_shape=list(residual.shape), actual_jacobian_shape=list(jacobian.shape), finite_residual_and_jacobian=True, diagnostic_raw_pixel_error_median=float(np.median(errors)), diagnostic_raw_pixel_error_max=float(np.max(errors)), model_accuracy_accepted=False)
                    target_totals.update(targets)
                except Exception as exc:
                    frame_result["error"] = f"{type(exc).__name__}: {exc}"
                    report["errors"].append({"index": index, "error": frame_result["error"]})
                report["frames"].append(frame_result)
                if index % 10 == 0:
                    print(f"Validated model entries for {index}/100 frames; failures={len(report['errors'])}", flush=True)
        report.update(exact_official_clock={"first_pts": FIRST_PTS, "last_pts": FIRST_PTS + (COUNT - 1) * STEP, "end_pts_exclusive": FIRST_PTS + COUNT * STEP, "time_base": "1/12800", "frame_count": COUNT}, approved_original_packet={"path": str(approved_path.resolve()), "sha256": APPROVED_SHA256}, model_inputs=[reference(p) for p in (scene_path, registry_path, templates_path, seeds_path)], all_registered_targets_finite=True, aggregate_target_point_counts=dict(target_totals))
    except Exception as exc:
        report["errors"].append({"stage": "batch_inputs", "error": f"{type(exc).__name__}: {exc}"})
    unchanged = all(sha(path) == value for path, value in before.items())
    report["all_input_files_unchanged"] = unchanged
    if not unchanged:
        report["errors"].append({"stage": "input_immutability", "error": "A source/model/camera input changed during validation"})
    report["all_100_model_entries_passed"] = len(report["frames"]) == COUNT and all(f["status"] == "passed" for f in report["frames"]) and not report["errors"]
    report["status"] = "passed" if report["all_100_model_entries_passed"] else "failed"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = validate(args.dataset, args.model, args.output)
    print(json.dumps({"status": report["status"], "validated_frames": len(report["frames"]), "errors": report["errors"], "output": str(args.output)}, ensure_ascii=False))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
