"""Finalize two selected full-programme candidates and register their exact projections."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

import numpy as np

from calibration.program_tracking.body_tracking.programme_contract import required_clock
from calibration.program_tracking.body_tracking.programme_projection import (
    SOURCE_FIELDS,
    generate,
    merge_review_assets,
)
from calibration.program_tracking.body_tracking.rebuild_prepare import digest
from calibration.program_tracking.body_tracking.trajectory_finalize import finalize
from rinkspace.path_policy import guard_project_write
from rinkspace.storage import AssetStore
from rinkspace.viewer3d.blade_review import BladeReview
from rinkspace.viewer3d.data import ResultCatalog
from rinkspace.viewer3d.prepare import prepare_variant


PROJECT = Path(__file__).resolve().parents[1]
PROGRAMME = PROJECT / "events/ankara_2026/programmes/okada_mei_sp"
CONFIG = PROGRAMME / "viewer3d_programme_action_1107.json"
BASELINE_CAMERA = PROGRAMME / "fit_v11/camera_solver_refined_1102.json"
DEFAULT = "programme_action_physics_1107"
STRUCTURE = "programme_structure_1108"
JOINT = "programme_structure_camera_1108"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + chr(10)).encode("utf8")


def write_once(path, value):
    with Path(path).open("xb") as stream:
        stream.write(json_bytes(value))


def file_ref(path):
    return dict(path=str(path), sha256=digest(path))


def camera_identity(store, path, expected_sha=None):
    path = store.source(path).resolve(strict=True)
    record = file_ref(path)
    if expected_sha is not None and record["sha256"] != expected_sha:
        raise ValueError(f"Prepared candidate camera record changed: {path}")
    declaration = read(path)
    state = declaration["state"]
    compact = store.source(state["compact_path"]).resolve(strict=True)
    binding = file_ref(compact)
    if binding["sha256"] != state["compact_sha256"]:
        raise ValueError(f"Camera compact state changed: {compact}")
    return dict(record=record, state=binding)


def camera_projection_differences(first_path, second_path):
    """Compare numerics and the source-to-spline clock, independently of archive SHA."""
    if first_path == second_path:
        return []
    fields = ("rig", "controls", "board", "layout", "scene_values")
    shot_fields = (
        "id", "camera_index", "control_begin", "control_stop", "reference_pts",
        "time_base_num", "time_base_den", "first_control_time", "first_pts", "last_pts",
        "exposure_pts_offsets",
    )
    with np.load(first_path, allow_pickle=False) as first, np.load(second_path, allow_pickle=False) as second:
        differences = [key for key in fields if not np.array_equal(first[key], second[key])]
        a, b = [json.loads(packet["joint_numeric_dependencies_json"].item()) for packet in (first, second)]
    for key in ("control_dt", "first_control_time"):
        if a[key] != b[key]:
            differences.append(key)
    if [[s.get(key) for key in shot_fields] for s in a["shots"]] != [
        [s.get(key) for key in shot_fields] for s in b["shots"]
    ]:
        differences.append("shot motion clock/installation")
    return differences


def check_camera_binding(packet, expected_sha, shot_ids, origin):
    binding = json.loads(packet["camera_binding_json"].item())
    shot_hashes = {shot: record["state_sha256"] for shot, record in binding["shots"].items()}
    if (binding["state_sha256"] != expected_sha or set(shot_hashes) != set(shot_ids) or
            any(sha != expected_sha for sha in shot_hashes.values())):
        raise ValueError(f"Saved result or checkpoint uses a different programme camera: {origin}")
    return binding


def candidate_input(store, candidate, stage, prepared, camera_baseline, expected_pts, shot_ids, time_base):
    stage, prepared = (p.resolve(strict=True) for p in (stage, prepared))
    if not all(p.is_relative_to(store.data) for p in (stage, prepared)):
        raise ValueError("Prepared data and completed stages must be in the registered Data root")
    summary_path = stage / "run_summary.json"
    summary = read(summary_path)
    if (not summary["engineering_ready"] or not summary["trajectory_candidate"] or
            summary["source_frames"] != len(expected_pts)):
        raise ValueError(f"Selected stage is not a completed full-programme candidate: {stage}")
    raw, parameter = stage / "fitted/result.npz", stage / "solve/candidate.npz"
    source_evidence = prepared / "source_evidence.json"
    evidence = read(source_evidence)
    selection = evidence.get("programme_camera_candidate", {})
    explicit_camera = bool(selection.get("record"))
    if selection and not explicit_camera:
        raise ValueError(f"Prepared camera candidate lacks a record: {prepared}")
    camera = camera_identity(store, selection["record"], selection["record_sha256"]) if explicit_camera else camera_baseline
    sha = camera["state"]["sha256"]
    prepared_hashes = {
        "prepared_input_sha256": digest(prepared / "input.npz"),
        "prepared_observations_sha256": digest(prepared / "observations.npz"),
        "source_evidence_sha256": digest(source_evidence),
    }
    with np.load(raw, allow_pickle=False) as result, np.load(parameter, allow_pickle=False) as saved:
        if (not np.array_equal(result["source_pts"], expected_pts) or
                not np.array_equal(result["time_base"], time_base) or
                not np.array_equal(result["source_shot_ids"], shot_ids) or
                saved["state_kind"].item() != "absolute_motion_v1" or
                any(saved[key].item() != value for key, value in prepared_hashes.items())):
            raise ValueError(f"Completed stage and prepared programme inputs differ: {stage}")
        result_binding = check_camera_binding(result, sha, shot_ids, raw)
        parameter_binding = check_camera_binding(saved, sha, shot_ids, parameter)
        if result_binding["native_library_sha256"] != parameter_binding["native_library_sha256"]:
            raise ValueError(f"Saved camera runtimes differ: {stage}")
    runtime = store.source(result_binding["native_library"]).resolve(strict=True)
    native_library = file_ref(runtime)
    if native_library["sha256"] != result_binding["native_library_sha256"]:
        raise ValueError(f"Saved projection runtime changed: {runtime}")
    return dict(
        id=candidate, stage=str(stage), prepared=str(prepared), prepared_hashes=prepared_hashes,
        summary=file_ref(summary_path), raw=file_ref(raw), parameter=file_ref(parameter),
        camera=camera, explicit_camera=explicit_camera,
        projection_runtime=native_library,
    )


def verify_grid_identity(candidates, viewer_camera, baseline_camera, old_blade):
    if viewer_camera["sha256"] != baseline_camera["state"]["sha256"]:
        raise ValueError("The shared viewer scene snapshot differs from the frozen display camera")
    for candidate in candidates:
        differences = camera_projection_differences(viewer_camera["path"], candidate["camera"]["state"]["path"])
        if differences and old_blade.get("source_ice_grid"):
            raise ValueError(
                f"{candidate['id']} changes the projected camera ({', '.join(differences)}). "
                "The player still uses a global legacy source_ice_grid. Register camera-bound "
                "source_ice_grid_by_variant and source_plane_locations_by_variant in BladeReview "
                "before publishing a changed camera."
            )
        if differences:
            raise ValueError(f"{candidate['id']} requires an explicit per-variant source-plane projection binding")
        candidate["legacy_grid_reuse"] = dict(
            viewer_camera=viewer_camera, candidate_camera=candidate["camera"]["state"],
            identical_controls_rigs_scene_and_source_clock=True,
        )


def validate_navigation(navigation, pts, time_base, failure_events):
    document = copy.deepcopy(navigation)
    if tuple(document["time_base"]) != tuple(time_base):
        raise ValueError("Navigation uses another source clock")
    if failure_events is not None:
        failure_doc = read(failure_events)
        if "time_base" in failure_doc and tuple(failure_doc["time_base"]) != tuple(time_base):
            raise ValueError("Failure events use another source clock")
        document["events"] = failure_doc["events"]
        document["failure_events_source"] = file_ref(failure_events)
    events = document.get("events", [])
    lookup = set(pts.tolist())
    if len(events) > 8 or any(
        not isinstance(row.get("label"), str) or not row["label"].strip() or
        any(type(row.get(key)) is not int or row[key] not in lookup for key in ("start_pts", "end_pts")) or
        row["start_pts"] >= row["end_pts"] for row in events
    ):
        raise ValueError("Supply up to eight complete ordered failure intervals on exact source PTS")
    if failure_events is not None:
        document["events"] = sorted(events, key=lambda row: row["start_pts"])
    return document


def materialize(candidate, output, expected_pts, shot_ids, time_base):
    variable = "RINKSPACE_BODY_NATIVE_LIBRARY"
    previous = os.environ.get(variable)
    os.environ[variable] = candidate["projection_runtime"]["path"]
    try:
        return materialize_bound(candidate, output, expected_pts, shot_ids, time_base)
    finally:
        if previous is None:
            del os.environ[variable]
        else:
            os.environ[variable] = previous


def materialize_bound(candidate, output, expected_pts, shot_ids, time_base):
    variant = candidate["id"]
    stage, prepared = Path(candidate["stage"]), Path(candidate["prepared"])
    record_path = guard_project_write(PROGRAMME / (variant + ".json"))
    parameter_path = guard_project_write(PROGRAMME / (variant + ".npz"))
    if not record_path.is_file():
        finalize(PROJECT, prepared, stage, candidate_id=variant,
                 output=output / variant / "final", viewer_config=CONFIG.relative_to(PROJECT))
    formal = read(record_path)
    result_path = Path(formal["result"])
    if (formal["candidate_id"] != variant or Path(formal["parameter"]) != parameter_path or
            formal["parameter_sha256"] != candidate["parameter"]["sha256"] or
            digest(parameter_path) != formal["parameter_sha256"] or
            digest(result_path) != formal["result_sha256"]):
        raise ValueError(f"Final artifacts do not match the selected candidate: {variant}")
    sha = candidate["camera"]["state"]["sha256"]
    with np.load(result_path, allow_pickle=False) as result:
        check_camera_binding(result, sha, shot_ids, result_path)
        context = json.loads(result["programme_context_json"].item())
        provenance = json.loads(result["final_metadata_provenance_json"].item())
        if (not np.array_equal(result["source_pts"], expected_pts) or
                not np.array_equal(result["time_base"], time_base) or
                provenance["parent_result_sha256"] != candidate["raw"]["sha256"] or
                context["source_evidence"]["sha256"] != candidate["prepared_hashes"]["source_evidence_sha256"] or
                any(row["frozen_camera_state_sha256"] != sha for row in context["shots"]) or
                (candidate["explicit_camera"] and context["frozen_camera_sha256"] != sha)):
            raise ValueError(f"Finalized camera/source context differs from the selected candidate: {variant}")
    dest = output / variant / "projection"
    receipt_path = dest / "projection_registration.json"
    if not receipt_path.is_file():
        generate(PROJECT, prepared, Path(candidate["parameter"]["path"]), result_path, dest,
                 variant=variant, review_evidence=prepared / "source_evidence.json")
    receipt = read(receipt_path)
    projection = Path(receipt["output_asset"]["path"])
    projection_npz = Path(receipt["output_projection"]["path"])
    binding = read(projection)["prediction_binding"]
    if (receipt["frames"] != len(expected_pts) or tuple(receipt["time_base"]) != tuple(time_base) or
            not receipt["fixed_identity_verified"] or binding["result_sha256"] != formal["result_sha256"] or
            binding["parameter_sha256"] != candidate["parameter"]["sha256"] or
            digest(projection) != receipt["output_asset"]["sha256"] or
            digest(projection_npz) != receipt["output_projection"]["sha256"]):
        raise ValueError(f"Five-node replay does not bind the selected saved result: {variant}")
    return formal, record_path, projection


def combine_projections(old_path, old, candidates, output, store):
    originals = []
    for entry in old["independently_replayed_assets"]:
        path = store.source(entry["path"]).resolve(strict=True)
        if digest(path) != entry["sha256"]:
            raise ValueError(f"Preserved projection document changed: {path}")
        originals.append(path)
    destination = output / "blade_review"
    combined = destination / "programme_blade_review.json"
    if not combined.is_file():
        merge_review_assets([*originals, *candidates], destination, primary_variant=DEFAULT,
                            source_revision_variant=old["source_overlay_revision_variant"])
    merged = read(combined)
    if any(merged.get(key, []) != old.get(key, []) for key in SOURCE_FIELDS):
        raise ValueError("Combined viewer would change the original source overlays")
    for variant, prefix in old["variants"].items():
        if (merged["variants"][variant] != prefix or
                merged["prediction_bindings"][variant] != old["prediction_bindings"][variant] or
                merged.get("source_structure_fit_variants", {}).get(variant) !=
                old.get("source_structure_fit_variants", {}).get(variant)):
            raise ValueError(f"Combined viewer would change old prediction provenance: {variant}")
    old_npz = store.resolve(old["projection_npz"], base=old_path.parent, must_exist=True)
    new_npz = store.resolve(merged["projection_npz"], base=combined.parent, must_exist=True)
    with np.load(old_npz, allow_pickle=False) as a, np.load(new_npz, allow_pickle=False) as b:
        for key in a.files:
            left, right = a[key], b[key]
            identical = (np.array_equal(left, right, equal_nan=True) if left.dtype.kind in "fc"
                         else np.array_equal(left, right))
            if left.dtype != right.dtype or not identical:
                raise ValueError(f"Combined projection would change old values: {key}")
    for path in candidates:
        document = read(path)
        for variant in document["variants"]:
            if merged["prediction_bindings"][variant] != document["prediction_binding"]:
                raise ValueError(f"Combined projection does not bind the new saved candidate: {variant}")
    return combined


def main():
    store = AssetStore(PROJECT)
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("structure-stage", "joint-stage", "prepared-structure", "prepared-joint"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--out", type=Path,
                        default=store.data / "runs/body_s01_20261029/structure_camera_1108/player_install")
    parser.add_argument("--failure-events", type=Path,
                        help="JSON with up to eight complete exact-PTS events; otherwise preserve current events")
    parser.add_argument("--preflight-only", action="store_true", help="Check completed inputs without writing artifacts")
    args = parser.parse_args()
    out = guard_project_write(args.out)
    if not store.configured or out == store.data or not out.is_relative_to(store.data):
        raise ValueError("Player installation outputs must be under the registered Data root")

    initial_config_bytes = CONFIG.read_bytes()
    initial = json.loads(initial_config_bytes)
    if initial["default_variant"] != DEFAULT or any(v["id"] in (STRUCTURE, JOINT) for v in initial["variants"]):
        raise ValueError("Expected the preserved default and two unpublished candidate IDs")
    nav_path = store.resolve(initial["source_navigation"], base=CONFIG.parent, must_exist=True)
    old_blade_path = store.resolve(initial["blade_review_asset"], base=CONFIG.parent, must_exist=True)
    old_blade = read(old_blade_path)
    scope = read(PROGRAMME / "rebuild_scope.json")
    pts, shot_ids, time_base = required_clock(scope)
    if len(pts) != 8329 or tuple(time_base) != (1, 12800) or len(set(shot_ids)) != 9:
        raise ValueError("Player installation requires the full native nine-shot programme")
    nav = validate_navigation(read(nav_path), pts, time_base, args.failure_events)
    baseline_camera = camera_identity(store, BASELINE_CAMERA)
    snapshot = initial.get("camera_snapshot", str(Path(initial["scene_root"]) / "camera_snapshot.npz"))
    viewer_camera = file_ref((store.data / snapshot).resolve(strict=True))
    cases = [
        candidate_input(store, STRUCTURE, args.structure_stage, args.prepared_structure,
                        baseline_camera, pts, shot_ids, time_base),
        candidate_input(store, JOINT, args.joint_stage, args.prepared_joint,
                        baseline_camera, pts, shot_ids, time_base),
    ]
    if cases[0]["stage"] == cases[1]["stage"]:
        raise ValueError("Supply two separately selected completed stages")
    verify_grid_identity(cases, viewer_camera, baseline_camera, old_blade)
    inputs = dict(candidates=cases, source_frames=len(pts), time_base=list(time_base),
                  original_config=file_ref(CONFIG), original_navigation=file_ref(nav_path),
                  original_blade_review=file_ref(old_blade_path),
                  failure_events=None if args.failure_events is None else file_ref(args.failure_events.resolve()))
    if args.preflight_only:
        print(json_bytes(inputs).decode("utf8"), end="")
        return
    input_path = out / "installation_inputs.json"
    if out.exists():
        if read(input_path) != inputs:
            raise ValueError(f"An existing installation directory binds different inputs: {out}")
    else:
        out.mkdir(parents=True, exist_ok=False)
        write_once(input_path, inputs)

    results = [materialize(candidate, out, pts, shot_ids, time_base) for candidate in cases]
    combined = combine_projections(old_blade_path, old_blade, [row[2] for row in results], out, store)
    new_config = copy.deepcopy(initial)
    new_config["blade_review_asset"] = str(combined)
    new_config["latest_review_variant"] = JOINT
    if args.failure_events is not None:
        nav_path = out / "source_navigation.json"
        if nav_path.exists():
            if read(nav_path) != nav:
                raise ValueError("Existing failure event navigation differs")
        else:
            write_once(nav_path, nav)
        new_config["source_navigation"] = str(nav_path)
    labels = ("全节目结构候选（待复核）", "最新 · 联合检验候选（相机保留原值，待复核）")
    for candidate, (formal, record, _), label in zip(cases, results, labels, strict=True):
        new_config["variants"].append(dict(
            id=candidate["id"], label=label, kind="fullbody", result=formal["result"],
            shell_topology="saved", body_reference_label="programme_initial",
            body_refinement_intervals_seconds=[[22.12, 188.68]],
            user_accepted=False, accuracy_accepted=False, blade_review_available=True,
            source_projection_origin="own saved COCO and native five-node shoe/blade UV",
            camera_binding=candidate["camera"], projection_runtime=candidate["projection_runtime"],
            legacy_grid_reuse=candidate["legacy_grid_reuse"],
            diagnostic_record=str(record), diagnostic_record_sha256=digest(record),
            parameter=formal["parameter"], parameter_sha256=formal["parameter_sha256"],
        ))

    staged_path = out / "viewer_validation.json"
    staged_config = dict(new_config, source_navigation=str(nav_path))
    if staged_path.exists():
        if read(staged_path) != staged_config:
            raise ValueError("Existing staged viewer config differs")
    else:
        write_once(staged_path, staged_config)
    before = ResultCatalog(project=PROJECT, config=CONFIG)
    catalog = ResultCatalog(project=PROJECT, config=staged_path)
    np.testing.assert_array_equal(catalog.pts, pts)
    review = BladeReview(combined, assets=store, time_base=catalog.time_base)
    if (set(review.index) != set(map(int, pts)) or set(review.models) != {v["id"] for v in catalog.variants} or
            len(catalog.navigation_points("events")) != len(nav["events"]) or
            len(catalog.navigation_points("cuts")) != 8 or catalog.config["default_variant"] != DEFAULT):
        raise ValueError("Staged viewer did not preserve source coverage and navigation")
    for variant in initial["variants"]:
        identifier = variant["id"]
        if catalog.variant(identifier) != variant or catalog.cache_path(identifier)[0] != before.cache_path(identifier)[0]:
            raise ValueError(f"Viewer would change an existing variant or its cache: {identifier}")

    caches = {c["id"]: prepare_variant(catalog, c["id"]) for c in cases}
    pending = guard_project_write(CONFIG.with_name(CONFIG.stem + "_structure_pending.json"))
    watched = [inputs["original_navigation"], inputs["original_blade_review"]]
    for candidate in cases:
        watched.extend((candidate["raw"], candidate["parameter"], candidate["projection_runtime"],
                        candidate["camera"]["record"], candidate["camera"]["state"]))
    for ref in watched:
        if digest(ref["path"]) != ref["sha256"]:
            raise ValueError(f"Bound input changed during installation: {ref['path']}")
    with pending.open("xb") as stream:
        try:
            stream.write(json_bytes(new_config))
        except BaseException:
            stream.close()
            pending.unlink()
            raise
    try:
        if CONFIG.read_bytes() != initial_config_bytes:
            raise RuntimeError("Viewer config changed during installation")
        os.replace(pending, CONFIG)
    except BaseException:
        pending.unlink(missing_ok=True)
        raise
    report = dict(
        config=file_ref(CONFIG), source_exposures=len(pts), source_time_base=list(time_base),
        default_variant=DEFAULT, latest_variant=JOINT,
        launcher=str(PROJECT / "launchers/START_3D_RESULTS.ps1"),
        launcher_arguments=["-Config", str(CONFIG), "-Variant", JOINT],
        events=len(nav["events"]), navigation=file_ref(nav_path), projection=file_ref(combined),
        caches={name: dict(path=str(path), metadata=file_ref(path / "metadata.json")) for name, path in caches.items()},
        opened_viewer=False,
    )
    write_once(out / "player_validation.json", report)
    print(json_bytes(report).decode("utf8"), end="")


if __name__ == "__main__":
    main()
