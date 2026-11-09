"""Compare saved native reprojections and world motion on the same source clock."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from calibration.program_tracking.body_tracking.fullbody_fit import body_joint_assignment
from calibration.program_tracking.body_tracking.programme_contract import required_clock
from calibration.program_tracking.body_tracking.rebuild_prepare import digest
from calibration.program_tracking.body_tracking.skating_action_factors import (
    root_so3_diagnostics, validate_source_timeline,
)
from rinkspace.path_policy import guard_project_write
from rinkspace.storage import AssetStore


PROJECT = Path(__file__).resolve().parents[1]
GROUPS = {
    "torso": (5, 6, 11, 12),
    "lower_limbs": (13, 14, 15, 16),
    "ankles": (15, 16),
    "upper_arms": (5, 6, 7, 8),
    "face": (0, 1, 2, 3, 4),
}
POSITIONS = {"P": "P_world_mid", "G": "G_world_mid", "COM": "com_world_mid"}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def identity(path):
    path = path.resolve(strict=True)
    return dict(path=str(path), sha256=digest(path))


def candidate_arg(value):
    name, sep, path = value.partition("=")
    if not sep or not name.strip() or not path.strip():
        raise argparse.ArgumentTypeError("Use LABEL=RESULT.NPZ")
    return name, Path(path)


def measure(values, pts, tick, previous_pts=None):
    values = np.asarray(values)
    clocks = np.broadcast_to(np.asarray(pts).reshape((-1,) + (1,) * (values.ndim - 1)), values.shape)
    finite = np.isfinite(values)
    selected, stamps = values[finite], clocks[finite]
    report = dict(n=int(values.size), n_finite=len(selected), p50=None, p95=None, max=None, worst=None)
    if len(selected):
        p50, p95 = np.quantile(selected, [.5, .95])
        i = int(np.argmax(selected))
        stamp = int(stamps[i])
        worst = dict(source_pts=stamp, source_seconds=stamp * tick)
        if previous_pts is not None:
            previous = int(previous_pts[finite][i])
            worst.update(previous_source_pts=previous, previous_source_seconds=previous * tick)
        report.update(p50=float(p50), p95=float(p95), max=float(selected[i]), worst=worst)
    return report


def grouped_pixels(values, used, pts, tick):
    result = {}
    for name, ids in GROUPS.items():
        mask = used[:, ids]
        stamp_rows = np.nonzero(mask)[0]
        result[name] = measure(values[:, ids][mask], pts[stamp_rows], tick)
    return result


def camera_world(store, binding, shots, scale):
    if (binding.get("no_per_shot_alignment") is not True
            or binding["metres_per_scene_unit"] != scale
            or set(binding["shots"]) != set(shots)):
        raise ValueError("Candidate lacks a shared world and camera binding for the full programme")
    records, scene_hashes, states = {}, set(), {}
    for shot in shots:
        shot_binding = binding["shots"][shot]
        path = store.source(shot_binding["record"])
        if path not in records:
            records[path] = read_json(path)
        record = records[path]
        sha = shot_binding["state_sha256"]
        if record["state"]["compact_sha256"] != sha or not record.get("scene_sha256"):
            raise ValueError(f"Camera state or world scene is unbound in {shot}")
        states[shot] = sha
        scene_hashes.add(record["scene_sha256"])
    if len(scene_hashes) != 1:
        raise ValueError("Programme camera binds multiple world scenes")
    return scene_hashes.pop(), states


def read_case(path, pts, shots, time_base, observations, prepared_ids):
    with np.load(path, allow_pickle=False) as packet:
        for key, expected in (("pts", pts), ("source_pts", pts), ("source_shot_ids", shots),
                              ("time_base", time_base)):
            np.testing.assert_array_equal(packet[key], expected, err_msg=f"{path}: {key}")
        scale = float(packet["metres_per_scene_unit"])
        binding = json.loads(packet["camera_binding_json"].item())
        if "source_body_model_joint_ids" in packet:
            assigned = dict(observations, model_joint_ids=packet["source_body_model_joint_ids"])
            if "body_joint_assignment_provenance_json" in packet:
                assigned["body_joint_assignment_provenance_json"] = packet["body_joint_assignment_provenance_json"]
            ids = body_joint_assignment(assigned)
            mapping = "saved result source_body_model_joint_ids"
        else:
            ids, mapping = prepared_ids, "prepared body_joint_assignment"
        native_uv = packet["COCO_uv"]
        positions = {name: packet[key] for name, key in POSITIONS.items()}
        root, skates = packet["Q_world_mid"], packet["skate_R_world_mid"]
    if (native_uv.shape != (len(pts), 17, 2) or skates.shape != (len(pts), 2, 3, 3)
            or any(v.shape != (len(pts), 3) or not np.isfinite(v).all() for v in positions.values())):
        raise ValueError(f"Invalid saved native projections or world arrays in {path}")
    np.testing.assert_allclose(positions["G"][:, :2], positions["P"][:, :2], rtol=0., atol=1e-12)
    np.testing.assert_array_equal(positions["G"][:, 2], 0.)
    rotations = dict(Q=root, left_skate=skates[:, 0], right_skate=skates[:, 1])
    mapped_uv = native_uv[np.arange(len(pts))[:, None], ids]
    return positions, rotations, native_uv, mapped_uv, ids, mapping, scale, binding


def summarize_case(pts, shots, time_base, positions, rotations, native_uv, mapped_uv, baseline,
                   used, observed_uv):
    tick = time_base[0] / time_base[1]
    base_positions, base_steps, base_native, base_mapped = baseline
    residuals = np.linalg.norm(mapped_uv - observed_uv, axis=-1)
    position_shift = {name: np.linalg.norm(values - base_positions[name], axis=-1)
                      for name, values in positions.items()}
    native_shift = np.linalg.norm(native_uv - base_native, axis=-1)
    mapped_shift = np.linalg.norm(mapped_uv - base_mapped, axis=-1)
    scopes = {}
    for name, frame in [("programme", np.ones(len(pts), dtype=bool)),
                        *((shot, shots == shot) for shot in dict.fromkeys(shots))]:
        stamps = pts[frame]
        intervals = frame[:-1] & frame[1:] & (shots[:-1] == shots[1:])
        rotation = {}
        for part, (angle, ambiguous) in rotations.items():
            rotation[part] = dict(
                angle=measure(angle[intervals], pts[1:][intervals], tick, pts[:-1][intervals]),
                absolute_change_vs_baseline=measure(
                    np.abs(angle[intervals] - base_steps[part][0][intervals]),
                    pts[1:][intervals], tick, pts[:-1][intervals]),
                near_pi_ambiguous=int(ambiguous[intervals].sum()))
        scopes[str(name)] = dict(
            exposures=len(stamps), first_last_pts=stamps[[0, -1]].tolist(),
            source_reprojection_error_px=grouped_pixels(residuals[frame], used[frame], stamps, tick),
            position_shift_vs_baseline_m={part: measure(values[frame], stamps, tick)
                                          for part, values in position_shift.items()},
            native_COCO_prediction_shift_vs_baseline_px=measure(native_shift[frame], stamps, tick),
            mapped_used_prediction_shift_vs_baseline_px=grouped_pixels(mapped_shift[frame], used[frame], stamps, tick),
            adjacent_SO3_step_deg=rotation)
    return scopes


def report(prepared, paths, output):
    output = guard_project_write(output, project_root=PROJECT)
    if len(str(output).encode("utf-16-le")) // 2 >= 260:
        raise ValueError(f"Windows report path is too long: {output}")
    if output.exists():
        raise FileExistsError(output)
    store = AssetStore(PROJECT)
    document_path = prepared / "source_evidence.json"
    document = read_json(document_path)
    scope_path = store.source(document["programme_scope"]["path"])
    scope_id = identity(scope_path)
    if scope_id["sha256"] != document["programme_scope"]["sha256"]:
        raise ValueError("Prepared scope no longer matches its original source binding")
    pts, shots, time_base = required_clock(read_json(scope_path))
    if len(pts) != 8329:
        raise ValueError("Expected 8,329 native programme exposures")
    shot_order = list(dict.fromkeys(shots.tolist()))
    input_path = prepared / "input.npz"
    with np.load(input_path, allow_pickle=False) as packet:
        for key, expected in (("pts", pts), ("shot_ids", shots), ("time_base", time_base)):
            np.testing.assert_array_equal(packet[key], expected, err_msg=f"prepared {key}")
    obs_path = prepared / "observations.npz"
    with np.load(obs_path, allow_pickle=False) as packet:
        obs = {key: packet[key] for key in
               ("pts", "use", "uv", "model_joint_ids", "body_joint_assignment_provenance_json") if key in packet}
    np.testing.assert_array_equal(obs["pts"], pts)
    used, uv = obs["use"], obs["uv"]
    if (used.shape != (len(pts), 17) or used.dtype.kind != "b"
            or uv.shape != (len(pts), 17, 2) or not np.isfinite(uv[used]).all()):
        raise ValueError("Invalid immutable source UV/use population")
    prepared_ids = body_joint_assignment(obs)
    source_sha = document["source_sha256"]
    clock = validate_source_timeline(pts, time_base, source_id=source_sha)

    camera_context_path = prepared / "camera_context.json"
    prepared_camera = read_json(camera_context_path)
    scale = prepared_camera["metres_per_scene_unit"]
    scene_sha, _ = camera_world(store, prepared_camera, shot_order, scale)
    cases, baseline = {}, None
    for name, path in paths:
        positions, rotations, native_uv, mapped_uv, ids, mapping, length_scale, camera = read_case(
            path, pts, shots, time_base, obs, prepared_ids)
        if length_scale != scale:
            raise ValueError(f"{name} uses another conditional metric scale")
        bound_scene, camera_states = camera_world(store, camera, shot_order, scale)
        if bound_scene != scene_sha:
            raise ValueError(f"{name} uses another world scene")
        steps = {}
        for part, matrices in rotations.items():
            state = root_so3_diagnostics(clock, matrices)
            steps[part] = (np.rad2deg(state.principal_increment_rad), state.near_pi_ambiguous)
        if baseline is None:
            baseline = (positions, steps, native_uv, mapped_uv)
            baseline_ids = ids
        cases[name] = dict(
            result=identity(path), camera_state_sha256_by_shot=camera_states,
            source_joint_mapping=mapping,
            source_frames_with_reassignment=int(np.any(ids != np.arange(17), axis=1).sum()),
            used_rows_with_mapping_different_from_prepared=int(np.count_nonzero(used & (ids != prepared_ids))),
            used_rows_with_mapping_different_from_baseline=int(np.count_nonzero(used & (ids != baseline_ids))),
            **summarize_case(pts, shots, time_base, positions, steps, native_uv, mapped_uv,
                             baseline, used, uv))

    result = dict(
        schema="rinkspace.structure_reprojection.v1", baseline=paths[0][0],
        source=dict(sha256=source_sha, exposures=len(pts), time_base=list(time_base),
                    first_last_pts=pts[[0, -1]].tolist(), shot_order=shot_order),
        inputs=dict(prepared=identity(input_path), observations=identity(obs_path), scope=scope_id,
                    source_evidence=identity(document_path), camera_context=identity(camera_context_path)),
        pixel_groups_source_coco_ids=GROUPS, fixed_used_source_uv_rows=int(used.sum()),
        world_scene_sha256=scene_sha, conditional_metres_per_scene_unit=scale,
        independent_accuracy_accepted=False, cases=cases,
        interpretation=[
            "Pixel residuals use the original obs.use mask and source UV. Each result's source-to-model assignment selects its native COCO prediction.",
            "n counts admitted rows, and n_finite reports the rows supplying finite statistics.",
            "Native COCO pixel shifts compare the same saved model-joint IDs across all exposures. Mapped used shifts compare the predictions bound to the same admitted source labels.",
            "P is the saved pelvis. G is its Z=0 world projection. COM is the separate saved conditional body and equipment mass model.",
            "World positions use saved conditional metres in one common scene. Comparisons use no alignment or fitting.",
            "Principal SO(3) steps compare adjacent original exposures within each shot. The worst interval is labelled by both endpoints.",
            "Source reprojection residuals and saved-state changes are model diagnostics.",
        ],
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(str(output), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--candidate", type=candidate_arg, action="append", required=True,
                        help="LABEL=RESULT.NPZ, baseline first")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    names = [name for name, _ in args.candidate]
    if len(set(names)) != len(names):
        parser.error("Candidate labels must be unique")
    report(args.prepared, args.candidate, args.output)


if __name__ == "__main__":
    main()
