"""Initialization-only continuation of the unchanged exact shot03 graph.

Default invocation writes a reviewable phase_00 and 119 initial states only.
``--run`` requires that saved preflight, rebuilds the *same* conditioned graph,
and passes the override only to ``solve_one -> graph.update``. No observation,
prior target, factor weight, frozen scene node, or final holdout is changed.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import gtsam
import numpy as np

import conditioned_camera_comparison as driver
from projection import from_state
from rig_parameterization import camera25 as derived_camera25, physical_frame


TMP = Path("D:/CodexData/RinkSpace_program_tracking_tmp")
CONFIG = TMP / "joint_programme_atlas_provenance_full_v13_config.json"
CANDIDATE = TMP / "joint_programme_atlas_provenance_full_v13"
MANIFEST = TMP / "native_surface_model_comparison_v1/strict_board_manifest.json"
OLD = TMP / "conditioned_shot03_exact_shared_lens_v1"
OLD_CAMERA = OLD / "fixed/shot_cameras.npz"
OLD_STATES = OLD / "fixed/states.npz"
OLD_SUMMARY = OLD / "fixed/summary.json"
OLD_PREFLIGHT = OLD / "preflight.json"
NEW_CAMERA = TMP / "low_end_shared_lens_shape_v1/shared_lens_shape19/shot_cameras.npz"
NEW_REPORT = TMP / "low_end_shared_lens_shape_v1/report.json"
OUTPUT = TMP / "conditioned_shot03_lens19_seed_v1"

PINNED_SHA = {
    OLD_CAMERA: "37ec4fcb50deca59e0ac8652a65d7cd3ebeed540abec79a281edfb2ccecf1106",
    OLD_STATES: "dc30f2a558a95052033a3b68d50d885f92dc4c185dccdc97229a97c7ab2e0381",
    OLD_SUMMARY: "9c213d626aacc62aa11978fcddacd92c3d7c0aaa32980ebc7990728b5de62898",
    OLD_PREFLIGHT: "d7372be53dd6f92d3fc983e1cdac274791c35285a02893dbb3c137f36c4f2d60",
    NEW_CAMERA: "2419c4626055397463daa66eba49c6e2cbe590ddfa0c71fbd6a5a186dca2d822",
    NEW_REPORT: "7000bf5c6789583a3811832bc43b9723d35d7dfc6f551599a4a7d64ea1073fad",
}
ANCHORS = (229678080, 229679360, 229685760)


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def packets() -> dict:
    for path, expected in PINNED_SHA.items():
        require(driver.digest(path) == expected, f"Frozen input changed: {path}")
    old_summary = json.loads(OLD_SUMMARY.read_text(encoding="utf8"))
    old_preflight = json.loads(OLD_PREFLIGHT.read_text(encoding="utf8"))
    new_report = json.loads(NEW_REPORT.read_text(encoding="utf8"))
    with np.load(OLD_CAMERA, allow_pickle=False) as file:
        old_pts = np.asarray(file["pts"], np.int64).copy()
        old_camera = np.asarray(file["camera25"], float).copy()
        source = str(file["source_sha256"])
        scene = str(file["model_sha256"])
    with np.load(NEW_CAMERA, allow_pickle=False) as file:
        new_pts = np.asarray(file["pts"], np.int64).copy()
        new_camera = np.asarray(file["camera25"], float).copy()
        require(str(file["source_sha256"]) == source and
                str(file["model_sha256"]) == scene,
                "19D anchor source/scene differs from 119 packet")
    with np.load(OLD_STATES, allow_pickle=False) as file:
        keys = np.asarray(file["keys"], np.int64).copy()
        states = np.asarray(file["states"], float).copy()
    require(len(old_pts) == 119 and old_camera.shape == (119, 25) and
            np.array_equal(np.diff(old_pts), np.full(118, 256)) and
            len(set(old_pts)) == 119 and np.array_equal(new_pts, ANCHORS) and
            new_camera.shape == (3, 25) and np.isfinite(old_camera).all() and
            np.isfinite(new_camera).all(), "119/3 camera packet shape or PTS changed")
    require(keys.shape == (121,) and states.shape == (121, 11) and
            len(set(keys)) == 121 and np.isfinite(states).all(),
            "119 solved graph states changed")
    require(old_summary["frames"] == 119 and
            old_summary["rig_parameterization"] == "derived" and
            old_summary["mode"] == "fixed" and old_preflight["frames"] == 119 and
            old_preflight["native_counts"]["adjacent_pairs"] == 118 and
            new_report["scene_sha256"] == scene and
            new_report["source_video_sha256"] == source and
            old_summary["source_candidate_model_sha256"] == scene,
            "Saved source/scene/graph provenance changed")
    require(all(stamp in set(old_pts) for stamp in ANCHORS) and
            np.array_equal(old_camera[np.searchsorted(old_pts, ANCHORS), 11:25],
                           new_camera[:, 11:25]),
            "Anchor Brown templates changed")
    return dict(old_pts=old_pts, old_camera=old_camera,
                new_camera=new_camera, old_states=dict(zip(map(int, keys), states)),
                source_sha256=source, scene_sha256=scene,
                old_summary=old_summary, old_preflight=old_preflight)


def interpolation(packet: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    stamps = packet["old_pts"]
    old = packet["old_camera"]
    new = packet["new_camera"]
    anchor_old = old[np.searchsorted(stamps, ANCHORS)]
    delta_r = new[:, 3:6]-anchor_old[:, 3:6]
    delta_zoom = np.mean(new[:, 6:8], axis=1)-np.mean(anchor_old[:, 6:8], axis=1)
    ratio = new[:, 6]-new[:, 7]
    common = new[:, 8:11]
    require(np.max(np.abs(ratio-ratio[0])) < 1e-10 and
            np.max(np.abs(common-common[0])) < 1e-8 and
            np.max(np.abs(new[:, :3]-new[0, :3])) < 1e-8,
            "19D solved anchors do not share one C/lens shape")
    offsets_r = np.column_stack([np.interp(stamps, ANCHORS, delta_r[:, j])
                                 for j in range(3)])
    offsets_zoom = np.interp(stamps, ANCHORS, delta_zoom)
    cameras = old.copy()
    cameras[:, :3] = new[0, :3]
    cameras[:, 3:6] = old[:, 3:6]+offsets_r
    zoom = np.mean(old[:, 6:8], axis=1)+offsets_zoom
    cameras[:, 6] = zoom+ratio[0]/2.
    cameras[:, 7] = zoom-ratio[0]/2.
    cameras[:, 8:11] = common[0]
    require(np.max(np.abs(cameras[np.searchsorted(stamps, ANCHORS), :11]-
                          new[:, :11])) < 1e-8,
            "Interpolated anchor camera does not reproduce the 19D solution")
    policy = dict(baseline="all 119 saved exact-shot solved cameras",
        anchor_pts=list(ANCHORS),
        anchor_rvec_delta_rad=delta_r.tolist(),
        anchor_mean_log_f_delta=delta_zoom.tolist(),
        interpolation="piecewise linear in original source PTS; before first/after last anchor hold endpoint delta",
        C_scene=new[0, :3].tolist(),
        shared_lens_shape=dict(log_fx_over_fy=float(ratio[0]),
                               cx_cy_skew_px=common[0].tolist()),
        frame_zoom="old solved mean(logfx,logfy) plus interpolated anchor delta",
        no_new_observation_or_prior=True)
    return cameras, np.column_stack([offsets_r, offsets_zoom]), policy


def override(data: dict, packet: dict, cameras: np.ndarray) -> dict[int, np.ndarray]:
    active = set(data["active"])
    old_states = packet["old_states"]
    require(active == set(old_states) and len(data["pts"]) == 119 and
            data["camera_id"] == "low_end" and data["shot_indices"] == (3,),
            "Saved 119 graph states differ from original active graph")
    result = {key: old_states[key].copy() for key in active}
    mount, lens = data["mount"], data["lens"]
    result[mount][:3] = cameras[0, :3]/100.
    result[lens][6] = cameras[0, 6]-cameras[0, 7]
    result[lens][8:11] = cameras[0, 8:11]/1000.
    for stamp, camera in zip(packet["old_pts"], cameras):
        state = result[int(stamp)]
        state[:3] = 0.  # Exact-derived fixed-mount dummy slot.
        state[3:8] = camera[3:8]
        state[8:11] = camera[8:11]/1000.
        restored = derived_camera25(state, result[mount],
                                    data["captured"]["templates"][int(stamp)], "fixed")
        require(np.max(np.abs(restored-camera)) < 1e-7,
                f"Derived camera does not reproduce seed: {stamp}")
    require(all(np.max(np.abs(result[int(stamp)][:3])) == 0 for stamp in packet["old_pts"]),
            "Derived frame dummy C was not zeroed")
    return result


def source_hits(pairs: list, states: dict, mount_key: int) -> dict:
    invalid = []
    total = 0
    for a, b, pair in pairs:
        first = physical_frame(states[a], states[mount_key], "fixed")[0]
        second = physical_frame(states[b], states[mount_key], "fixed")[0]
        valid, reasons = pair.initial_validity([first, second])
        total += len(valid)
        for index in np.flatnonzero(~valid):
            invalid.append(dict(source_pts=int(a), target_pts=int(b),
                                point_index=int(index), reason=str(reasons[index])))
    return dict(actual_board_source_hits=total, valid_hits=total-len(invalid),
                invalid_count=len(invalid), invalid_rows=invalid)


def family_costs(graph, data: dict, pairs: list, saved: dict, seeded: dict) -> dict:
    labels = [("retained", spec["file"], spec["function"], spec["loop"] or "")
              for spec in data["retained"]]
    labels += [("dummy_center_prior", "", "", "")]*len(data["pts"])
    labels += [("native_board_source", "", "", "")]*len(pairs)
    require(len(labels) == len(graph._pending) == 633,
            "Factor count or captured factor order changed")
    values = []
    for states in (saved, seeded):
        v = gtsam.Values()
        for key, state in states.items():
            v.insert(graph._key(key), state)
        values.append(v)
    grouped = defaultdict(lambda: [0, 0., 0.])
    for label, factor in zip(labels, graph._pending):
        item = grouped[label]
        item[0] += 1
        item[1] += float(factor.error(values[0]))
        item[2] += float(factor.error(values[1]))
    rows = [dict(group=list(label), factors=int(count),
                 saved119_cost=old, interpolated_initial_cost=new,
                 initial_minus_saved=new-old)
            for label, (count, old, new) in sorted(grouped.items())]
    old_total = sum(row["saved119_cost"] for row in rows)
    new_total = sum(row["interpolated_initial_cost"] for row in rows)
    require(abs(old_total-data["old_batch_final_cost"]) < 1e-3,
            "Saved 119 cost differs from the same reconstructed objective")
    require(np.isfinite(new_total), "Interpolated initial cost is nonfinite")
    return dict(factor_count=len(labels), saved119_total=old_total,
                interpolated_initial_total=new_total,
                initial_minus_saved=new_total-old_total,
                by_captured_family=rows,
                note="Same original 633 factors and frozen external v13 states; costs describe the starting basin, not pixel accuracy")


def preflight(output: Path, *, run: bool) -> dict:
    require(output.resolve().is_relative_to(TMP.resolve()),
            "Output must remain in tracking TMP")
    packet = packets()
    cameras, offsets, policy = interpolation(packet)
    if run:
        require((output / "phase_00.json").is_file() and
                (output / "initial_states.npz").is_file(),
                "First run preflight without --run; review phase_00")
    else:
        require(not output.exists(), "Preflight output directory already exists")
        output.mkdir(parents=True)
    data = driver.prepare(CONFIG, CANDIDATE, MANIFEST, (3,), output,
                          complete_camera_group=False)
    require(data["captured"]["source_hash"] == packet["source_sha256"] and
            data["diag"]["model_sha256"] == packet["scene_sha256"],
            "Captured original source/scene changed")
    pairs, counts, counts_by_shot, components, old_validity = driver.native_pairs(data)
    require(counts["adjacent_pairs"] == 118 and counts["pixels"] == 8132 and
            old_validity["invalid_pair_count"] == 0,
            "Original native board evidence changed")
    initial = override(data, packet, cameras)
    hit_check = source_hits(pairs, initial, data["mount"])
    graph, _ = driver.build_graph(data, pairs, "fixed", "derived")
    data["old_batch_final_cost"] = float(packet["old_summary"]["batch"]["final_nonlinear_cost"])
    costs = (family_costs(graph, data, pairs, packet["old_states"], initial)
             if hit_check["invalid_count"] == 0 else None)
    initial_path = output / "initial_states.npz"
    keys = np.asarray(sorted(initial), np.int64)
    if run:
        with np.load(initial_path, allow_pickle=False) as file:
            require(np.array_equal(file["keys"], keys) and
                    np.array_equal(file["states"],
                                   np.asarray([initial[key] for key in keys])),
                    "Saved reviewed initial states changed")
    else:
        np.savez_compressed(initial_path, keys=keys,
                            states=np.asarray([initial[key] for key in keys]),
                            source_sha256=np.asarray(packet["source_sha256"]),
                            model_sha256=np.asarray(packet["scene_sha256"]))
    record = dict(schema="rinkspace.conditioned-shot03-lens19-initialization.v1",
        source_pts_count=119, anchor_pts=list(ANCHORS),
        source_sha256=packet["source_sha256"], scene_sha256=packet["scene_sha256"],
        original_config_sha256=driver.digest(CONFIG),
        original_manifest_sha256=driver.digest(MANIFEST),
        original_packet_sha256=driver.digest(data["packet_path"]),
        frozen_input_sha256={str(p):h for p,h in PINNED_SHA.items()},
        wrapper_script_sha256=driver.digest(__file__),
        driver_script_sha256=driver.digest(driver.__file__),
        initial_states_path=str(initial_path),
        initial_states_sha256=driver.digest(initial_path),
        interpolation=policy, anchor_camera_max_abs_error=float(np.max(np.abs(
            cameras[np.searchsorted(packet["old_pts"], ANCHORS), :11]-
            packet["new_camera"][:, :11]))),
        inactive_scene_reference_values="original saved v13 constants from captured graph",
        objective="unchanged original fixed/derived 119 graph; only graph.update initial values change",
        native_board_source_hits=hit_check, factor_family_costs=costs,
        old_native_coverage=dict(counts),
        no_new_observation_or_pose_prior=True, solver_started=False)
    phase_path = output / "phase_00.json"
    if run:
        saved = json.loads(phase_path.read_text(encoding="utf8"))
        require(saved == record, "Saved preflight phase differs; do not solve")
    else:
        phase_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf8")
    print(json.dumps(dict(phase="preflight", phase_path=str(phase_path),
        phase_sha256=driver.digest(phase_path),
        initial_states_sha256=driver.digest(initial_path),
        board_hit_valid=hit_check["invalid_count"] == 0,
        factor_costs=costs)), flush=True)
    require(hit_check["invalid_count"] == 0,
            "Interpolated initial state invalidates actual board source hits; no solve")
    if run:
        target = output / "fixed"
        require(not (target / "summary.json").exists() and
                not (target / "post_isam_states.npz").exists(),
                "A previous solve stage exists; inspect before restarting")
        provenance = dict(phase_00_path=str(phase_path),
            phase_00_sha256=driver.digest(phase_path),
            initial_states_sha256=driver.digest(initial_path),
            original_objective_unchanged=True)
        result = driver.solve_one(data, pairs, "fixed", target, "derived",
                                  initial_override=initial,
                                  initial_provenance=provenance)
        print(json.dumps(dict(phase="solve_complete", result=result)), flush=True)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--run", action="store_true",
                        help="Requires reviewed phase_00; run same 119 graph from saved init")
    args = parser.parse_args()
    preflight(args.output, run=args.run)


if __name__ == "__main__":
    main()
