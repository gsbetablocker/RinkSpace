"""Paired pivot check of the frozen 2219-frame source-edge interval objective.

The single extra physical form is C=P+R.T@d with the existing 0.6-scene-unit
offset prior. It starts once at the completed interval fixed solution with
d=0. All source UV, interval decisions, quality, Huber and prior targets stay
frozen. This conditional camera test is not programme acceptance.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import gtsam
import numpy as np

import conditioned_camera_comparison as driver
import low_end_board_edges_interval_diagnostic as fixed_interval
import low_end_calibration_transfer_profile as transfer
from rig_parameterization import camera25 as derived_camera25


TMP = transfer.TMP
FIXED_DIR = TMP / "low_end_board_edges_interval_diagnostic_v1"
OUTPUT = TMP / "low_end_board_edges_interval_pivot_v1"
FIXED_PHASE_SHA256 = "ca521565a77419d6dbb935a6da1e9e26ebc65eb7a674cd217b9c9f336ffa722d"
FIXED_SUMMARY_SHA256 = "3563f1b4672a0be3ddb759b3590f699fda124ebfa230687485630acb0b034b6a"
FIXED_STATES_SHA256 = "8b51f8f9d1139f5692eb7d9730be5ca2091343103ff34fa2ea0dde0513e55678"
FIXED_CAMERAS_SHA256 = "f687f1806d62bffc1f9127652c2278d144a4946e487e18ae780af6a677a3d057"


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def fixed_result(data: dict) -> tuple[dict, dict, dict]:
    paths = dict(phase=FIXED_DIR / "phase_00.json",
                 summary=FIXED_DIR / "fixed/summary.json",
                 states=FIXED_DIR / "fixed/states.npz",
                 cameras=FIXED_DIR / "fixed/shot_cameras.npz")
    hashes = dict(phase=FIXED_PHASE_SHA256, summary=FIXED_SUMMARY_SHA256,
                  states=FIXED_STATES_SHA256, cameras=FIXED_CAMERAS_SHA256)
    require(all(driver.digest(paths[name]) == value for name, value in hashes.items()),
            "Reviewed interval fixed phase/result SHA changed")
    phase = json.loads(paths["phase"].read_text(encoding="utf8"))
    summary = json.loads(paths["summary"].read_text(encoding="utf8"))
    require(phase["schema"] == "rinkspace.low-end-board-edge-interval-diagnostic.v1"
            and phase["mode"] == "interval"
            and phase["input_sha256"]["script"] == driver.digest(fixed_interval.__file__)
            and phase["source_sha256"] == data["captured"]["source_hash"]
            and phase["scene_sha256"] == data["diag"]["model_sha256"]
            and phase["original_factors"] == 11199
            and phase["replaceable_edge_factors"] == 4262
            and phase["untouched_factors"] == 6937
            and phase["prior_targets_and_original_quality_sigma_huber_unchanged"] is True
            and phase["original_uv_changed"] is False
            and phase["final_holdout_read"] is False
            and summary["mode"] == "fixed"
            and summary["rig_parameterization"] == "derived"
            and summary["frames"] == 2219
            and summary["factors"] == 11199
            and summary["shot_indices"] == [3, 5, 8]
            and summary["source_candidate_model_sha256"] == phase["scene_sha256"]
            and summary["initial_override_provenance"]["interval_phase_sha256"] == hashes["phase"]
            and summary["initial_override_provenance"]["source_support"] == phase["support"],
            "Fixed result is not the complete matching source-edge interval solve")
    with np.load(paths["states"], allow_pickle=False) as packet:
        keys = np.asarray(packet["keys"], np.int64)
        states = np.asarray(packet["states"], np.float64)
        require(keys.ndim == 1 and states.shape == (2221, 11)
                and len(keys) == len(set(map(int, keys)))
                and np.isfinite(states).all()
                and set(map(int, keys)) == set(data["active"]),
                "Reviewed fixed graph states do not cover the same 2221 keys")
        initial = {int(key): state.copy() for key, state in zip(keys, states)}
    mount = data["mount"]
    require(np.array_equal(initial[mount][3:6], np.zeros(3)),
            "Completed fixed result does not have the physical pivot seed d=0")
    pts = np.asarray(sorted(data["pts"]), np.int64)
    expected_camera25 = np.asarray([derived_camera25(initial[int(p)], initial[mount],
        data["captured"]["templates"][int(p)], "fixed") for p in pts])
    with np.load(paths["cameras"], allow_pickle=False) as packet:
        require(np.array_equal(packet["pts"], pts)
                and np.array_equal(packet["camera25"], expected_camera25)
                and str(packet["source_sha256"]) == phase["source_sha256"]
                and str(packet["model_sha256"]) == phase["scene_sha256"],
                "Fixed solved graph states do not reproduce its frozen camera25 output")
    return phase, summary, initial


def costs(graph, states: dict) -> np.ndarray:
    values = gtsam.Values()
    for key, vector in states.items():
        values.insert(graph._key(key), vector)
    return np.asarray([factor.error(values) for factor in graph._pending], np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    require(args.output.resolve().is_relative_to(TMP.resolve()),
            "Pivot output must remain in project tracking TMP")
    phase_path = args.output / "phase_00.json"
    if args.run:
        require(phase_path.is_file(), "Pivot solve requires matching reviewed preflight")
    else:
        require(not phase_path.exists(), "Pivot preflight exists; do not overwrite it")
    args.output.mkdir(parents=True, exist_ok=True)
    data, pairs, _, _, _ = transfer.load_inputs(args.output)
    fixed_phase, fixed_summary, initial = fixed_result(data)
    require(float(data["config"].get("offset_sigma_scene", .6)) == .6,
            "Historical pivot offset prior is no longer 0.6 scene units")
    edges, shot_by_pts = fixed_interval.source_edges(data)
    indices = fixed_interval.factor_indices(data, edges, shot_by_pts)
    packet_sha = {(number, edge_id): fixed_phase["support"][
        f"shot{number}_{fixed_interval.EDGE_NAMES[edge_id]}"]["sha256"]
        for number in (5, 8) for edge_id in range(2)}
    supports, manifest = fixed_interval.load_supports(
        data, edges, packet_sha, fixed_phase["support_manifest"]["sha256"])
    require(manifest == fixed_phase["support_manifest"],
            "Source support report changed since fixed interval solve")
    changed = fixed_interval.changed_data(data, edges, shot_by_pts, indices, supports)
    graph_fixed, _ = driver.build_graph(changed, pairs, "fixed", "derived")
    graph_pivot, _ = driver.build_graph(changed, pairs, "pivot", "derived")
    require(len(graph_fixed._pending) == len(graph_pivot._pending) == 11199,
            "Pivot changed the number of original source/prior/motion factors")
    fixed_cost = costs(graph_fixed, initial)
    pivot_cost = costs(graph_pivot, initial)
    require(np.isfinite(fixed_cost).all() and np.isfinite(pivot_cost).all(),
            "Completed fixed seed is outside the paired pivot objective")
    max_factor_difference = float(np.max(np.abs(fixed_cost-pivot_cost)))
    fixed_total, pivot_total = float(fixed_cost.sum()), float(pivot_cost.sum())
    require(abs(fixed_total-fixed_summary["batch"]["final_nonlinear_cost"]) < 1e-5
            and max_factor_difference < 1e-7,
            "At d=0, pivot data and prior costs must equal completed fixed solution")
    phase = dict(schema="rinkspace.low-end-source-interval-pivot-paired.v1",
        scope="Same full low_end 03/05/08 2219 frames, 11199 original factors",
        parameterization="derived physical C=P+R.T@d; one shared 3D d for low_end",
        historical_offset_prior_sigma_scene=float(data["config"].get("offset_sigma_scene", .6)),
        offset_jacobian_and_covariance_change_by_existing_pivot_model=True,
        original_uv_quality_huber_prior_targets_unchanged=True,
        source_support_manifest=manifest,
        fixed_phase_sha256=FIXED_PHASE_SHA256,
        fixed_summary_sha256=FIXED_SUMMARY_SHA256,
        fixed_states_sha256=FIXED_STATES_SHA256,
        fixed_cameras_sha256=FIXED_CAMERAS_SHA256,
        fixed_seed_offset_d=initial[data["mount"]][3:6].tolist(),
        fixed_seed_cost=fixed_total, pivot_at_d0_cost=pivot_total,
        max_per_factor_cost_difference_at_d0=max_factor_difference,
        factors=11199, source_edge_factors=len(indices),
        source_sha256=data["captured"]["source_hash"],
        scene_sha256=data["diag"]["model_sha256"],
        input_sha256=dict(driver=driver.digest(driver.__file__),
            fixed_interval=driver.digest(fixed_interval.__file__),
            transfer_loader=driver.digest(transfer.__file__),
            script=driver.digest(__file__)),
        conditional_diagnostic_only=True, final_holdout_read=False,
        solver_started=False)
    if args.run:
        require(json.loads(phase_path.read_text(encoding="utf8")) == phase,
                "Reviewed pivot preflight differs from current input or code")
    else:
        phase_path.write_text(json.dumps(phase, indent=2), encoding="utf8")
    print(json.dumps(dict(phase=str(phase_path), sha256=driver.digest(phase_path),
        factors=11199, fixed_seed_cost=fixed_total, pivot_at_d0_cost=pivot_total,
        max_per_factor_cost_difference_at_d0=max_factor_difference)), flush=True)
    if args.run:
        target = args.output / "pivot"
        require(not target.exists(), "Pivot solve output exists; review before retry")
        provenance = dict(paired_fixed_phase_sha256=FIXED_PHASE_SHA256,
            paired_fixed_states_sha256=FIXED_STATES_SHA256,
            pivot_phase_sha256=driver.digest(phase_path),
            seed_d=[0., 0., 0.], one_physical_offset_prior_sigma_scene=.6,
            no_parameter_or_sigma_scan=True)
        driver.solve_one(changed, pairs, "pivot", target, "derived",
                         initial_override=initial, initial_provenance=provenance)


if __name__ == "__main__":
    main()
