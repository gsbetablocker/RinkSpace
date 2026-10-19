"""Read-only complete low-end time-principal result audit.

Rebuilds physical15 programme/reference models from saved phase metadata and
final states. It does not construct a factor graph, decode images, rescore
source pixels, or start an optimizer. The report is evidence about the saved
camera model and its priors, not a source-accuracy certification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from runtime.bootstrap import add_geometry_dependencies
add_geometry_dependencies()
import cv2
import numpy as np

from continuous_camera_model import FrameBinding, ShotSpline
from continuous_reference_calibration import StaticReferenceBasis
from continuous_rig_camera import ContinuousRigCameraModel
from continuous_time_principal_optin import reconstruct_time_principal_rigs


SOURCE_SHA256 = "f685d27882e78f2446c734ceee4321dcb275a5ad1ebf81ea90bca68c2fcec891"
SCENE_SHA256 = "c7c59e2efb9782e6a3a45cf21ae7ad39958ad4ccdacb50437d4ad87056cd107a"
EXPECTED_SHOTS = (3, 5, 8)
EXPECTED_REFERENCES = (14177280, 205992960, 226160640)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf8"))


def _states(path: Path) -> dict[int, np.ndarray]:
    with np.load(path, allow_pickle=False) as packet:
        keys = np.asarray(packet["keys"], np.int64)
        values = np.asarray(packet["states"], np.float64)
    _require(values.shape == (len(keys), 11) and len(set(map(int, keys))) == len(keys)
             and np.isfinite(values).all(), f"Invalid Vector11 packet: {path}")
    return {int(key): value.copy() for key, value in zip(keys, values)}


def _cameras(path: Path, *, reference: bool) -> dict:
    with np.load(path, allow_pickle=False) as packet:
        result = {key: packet[key].copy() for key in packet.files}
    pts = np.asarray(result["pts"], np.int64)
    camera = np.asarray(result["camera25"], np.float64)
    _require(camera.shape == (len(pts), 25) and np.isfinite(camera).all()
             and len(set(map(int, pts))) == len(pts), f"Invalid camera packet: {path}")
    if not reference:
        shots = np.asarray(result["shot_index"], np.int64)
        aliases = np.asarray(result["exposure_representative_pts"], np.int64)
        _require(shots.shape == pts.shape and aliases.shape == pts.shape,
                 f"Incomplete complete-programme aliases: {path}")
    return result


def _shared_maps(phase: dict) -> dict:
    names = ("lens_base_keys", "lens_slope_keys", "mount_keys", "offset_keys",
             "offset_slope_keys", "brown_keys", "brown_slope_keys")
    return {name: {str(group): int(key) for group, key in phase[name].items()}
            for name in names}


def _rig(phase: dict, pack: dict, *, reference: bool) -> ContinuousRigCameraModel:
    maps = _shared_maps(phase)
    camera_id = "low_end"
    if reference:
        stamps = np.asarray(pack["pts"], np.int64)
        _require(tuple(map(int, stamps)) == EXPECTED_REFERENCES,
                 "Final reference source PTS changed")
        reference_phase = phase["reference_calibration"]
        _require(tuple(map(int, reference_phase["reference_pts"])) == EXPECTED_REFERENCES,
                 "Phase reference source PTS changed")
        shot_ids = phase["time_principal"]["reference_shot_ids"]
        _require(len(shot_ids) == len(stamps), "Reference shot ids changed")
        nodes = reference_phase["new_graph_nodes"]
        splines, frames = {}, {}
        for shot, pts, camera in zip(shot_ids, stamps, pack["camera25"]):
            row = nodes[str(int(pts))]
            key = int(row["c_residual_key"])
            splines[int(shot)] = StaticReferenceBasis(int(shot), int(pts), (key,))
            frames[int(pts)] = FrameBinding(
                int(pts), int(row["frame_key"]), int(shot), camera_id,
                np.asarray(camera, np.float64).copy())
    else:
        pts = np.asarray(pack["pts"], np.int64)
        shots = np.asarray(pack["shot_index"], np.int64)
        representatives = np.asarray(pack["exposure_representative_pts"], np.int64)
        _require(len(pts) == 2219 and set(map(int, shots)) == set(EXPECTED_SHOTS),
                 "Final complete low-end programme coverage changed")
        clocks = phase["frame_clock_virtual_pts"]
        splines = {int(shot): ShotSpline(int(shot), tuple(map(int, row["keys"])),
                                        tuple(map(float, row["pts"])))
                   for shot, row in phase["shot_knots"].items()}
        frames = {int(stamp): FrameBinding(
            int(stamp), int(rep), int(shot), camera_id,
            np.asarray(camera, np.float64).copy(), float(clocks[str(int(stamp))]))
            for stamp, rep, shot, camera in zip(
                pts, representatives, shots, pack["camera25"])}
    return ContinuousRigCameraModel(
        splines, frames, maps["lens_base_keys"], maps["lens_slope_keys"],
        phase["zoom_reference"], mount_keys=maps["mount_keys"],
        offset_keys=maps["offset_keys"],
        offset_slope_keys=maps["offset_slope_keys"],
        brown_keys=maps["brown_keys"],
        brown_slope_keys=maps["brown_slope_keys"])


def _stats(values) -> dict:
    a = np.asarray(values, np.float64).reshape(-1)
    _require(np.isfinite(a).all(), "Nonfinite result statistic")
    if not len(a):
        return {"count": 0}
    return dict(count=len(a), min=float(np.min(a)), p05=float(np.percentile(a, 5)),
                median=float(np.median(a)), p95=float(np.percentile(a, 95)),
                p99=float(np.percentile(a, 99)), max=float(np.max(a)),
                rms=float(np.sqrt(np.mean(a*a))))


def _vector_stats(rows: np.ndarray, names: tuple[str, ...]) -> dict:
    a = np.asarray(rows, np.float64).reshape(-1, len(names))
    result = {name: _stats(a[:, axis]) for axis, name in enumerate(names)}
    result["norm"] = _stats(np.linalg.norm(a, axis=1))
    return result


def _derivative_extrema(pts: np.ndarray, seconds: np.ndarray,
                        delta_px: np.ndarray) -> dict:
    if len(pts) < 3:
        return dict(velocity_count=max(len(pts)-1, 0), acceleration_count=0)
    dt = np.diff(seconds)
    _require(np.all(dt >= 0), "Corrected exposure clock reversed within one shot")
    zero = np.flatnonzero(dt == 0)
    valid_velocity = np.flatnonzero(dt > 0)
    valid_acceleration = np.flatnonzero((dt[:-1] > 0) & (dt[1:] > 0))
    _require(len(valid_velocity) > 0 and len(valid_acceleration) > 0,
             "No positive corrected-clock intervals to define derivatives")
    velocity = (delta_px[valid_velocity+1] - delta_px[valid_velocity]) / dt[valid_velocity, None]
    left = (delta_px[valid_acceleration+1] - delta_px[valid_acceleration]) / dt[valid_acceleration, None]
    right = (delta_px[valid_acceleration+2] - delta_px[valid_acceleration+1]) / dt[valid_acceleration+1, None]
    mid = .5 * (dt[valid_acceleration] + dt[valid_acceleration+1])
    acceleration = (right-left) / mid[:, None]
    vi = int(np.argmax(np.linalg.norm(velocity, axis=1)))
    ai = int(np.argmax(np.linalg.norm(acceleration, axis=1)))
    v_source = int(valid_velocity[vi])
    a_source = int(valid_acceleration[ai])
    return dict(
        positive_clock_velocity_intervals=len(valid_velocity),
        positive_clock_acceleration_triples=len(valid_acceleration),
        undefined_zero_clock_intervals=[
            [int(pts[index]), int(pts[index+1])] for index in zero],
        velocity_px_per_s=_vector_stats(velocity, ("cx", "cy")),
        acceleration_px_per_s2=_vector_stats(acceleration, ("cx", "cy")),
        max_velocity_source_pts=[int(pts[v_source]), int(pts[v_source+1])],
        max_velocity_dt_s=float(dt[v_source]),
        max_acceleration_source_pts=[int(pts[a_source]), int(pts[a_source+1]),
                                     int(pts[a_source+2])],
        max_acceleration_mid_s=float(mid[ai]))


def _rotation_changes_deg(baseline: np.ndarray, current: np.ndarray) -> np.ndarray:
    result = np.empty(len(current), np.float64)
    for i, (old, new) in enumerate(zip(baseline, current)):
        r0 = cv2.Rodrigues(old[3:6])[0]
        r1 = cv2.Rodrigues(new[3:6])[0]
        cosine = float(np.clip((np.trace(r1 @ r0.T)-1.)/2., -1., 1.))
        result[i] = float(np.degrees(np.arccos(cosine)))
    return result


def _correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    a = np.asarray(left, np.float64)
    b = np.asarray(right, np.float64)
    if len(a) < 3 or np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _prior_cost(states: dict[int, np.ndarray], metadata: dict,
                prior_metadata: dict) -> dict:
    programme = {int(k): tuple(map(int, v)) for k, v in
                 metadata["programme_principal_control_keys_by_shot"].items()}
    references = {int(k): tuple(map(int, v)) for k, v in
                  metadata["reference_principal_control_keys_by_shot"].items()}
    keys = [key for mapping in (programme, references)
            for controls in mapping.values() for key in controls]
    _require(keys == list(map(int, metadata["new_keys"]))
             and len(keys) == 50 and all(key in states for key in keys),
             "The frozen 50 principal controls changed")
    amplitude_sigma = float(prior_metadata["amplitude_sigma_px"]) / 1000.
    acceleration_sigma = float(prior_metadata["acceleration_sigma_px_s15"]) / 1000.
    _require(amplitude_sigma > 0 and acceleration_sigma > 0,
             "Principal prior scales changed")
    live = np.asarray([states[key][8:10] for key in keys])
    other_columns = [column for column in range(11) if column not in (8, 9)]
    dummy = np.asarray([states[key][other_columns] for key in keys])
    amplitude_cost = .5 * float(np.sum((live / amplitude_sigma)**2))
    dummy_cost = .5 * float(np.sum(dummy**2))
    acceleration_cost = 0.
    normalized_acceleration = []
    triple_count = 0
    for shot, controls in programme.items():
        times = np.asarray(metadata["programme_acceleration_times_seconds"][str(shot)],
                           np.float64)
        _require(len(times) == len(controls) and np.all(np.diff(times) > 0),
                 "Principal acceleration knot time changed")
        for index in range(1, len(controls)-1):
            dt0, dt1 = np.diff(times[index-1:index+2])
            mid = .5 * (dt0 + dt1)
            coefficients = np.array([1/dt0, -1/dt0-1/dt1, 1/dt1]) / mid
            values = np.asarray([states[key][8:10]
                                 for key in controls[index-1:index+2]])
            residual = coefficients @ values
            whitened = residual / (acceleration_sigma / np.sqrt(mid))
            normalized_acceleration.extend(whitened.tolist())
            acceleration_cost += .5 * float(whitened @ whitened)
            triple_count += 1
    _require((len(keys), triple_count) == (50, 41)
             and prior_metadata["amplitude_factors"] == 50
             and prior_metadata["dummy_factors"] == 50
             and prior_metadata["acceleration_factors"] == 41
             and prior_metadata["reference_temporal_factors"] == 0,
             "The 141 new prior factors changed")
    return dict(factor_count=141, amplitude_factor_count=50, dummy_factor_count=50,
                acceleration_factor_count=41, reference_temporal_factor_count=0,
                amplitude_sigma_px=float(prior_metadata["amplitude_sigma_px"]),
                acceleration_sigma_px_s15=float(prior_metadata["acceleration_sigma_px_s15"]),
                amplitude_cost=amplitude_cost, dummy_cost=dummy_cost,
                acceleration_cost=acceleration_cost,
                total_cost=amplitude_cost+dummy_cost+acceleration_cost,
                control_shift_px=_vector_stats(1000.*live, ("cx", "cy")),
                normalized_acceleration=_stats(normalized_acceleration),
                cost_convention="0.5*sum(whitened_residual^2); same nonuniform acceleration coefficients as fit_continuous_camera")


def _strip_f1_factor_keys(value: dict) -> dict:
    copy = {key: item for key, item in value.items() if key != "factor_rows"}
    copy["factor_rows"] = [
        {key: item for key, item in row.items() if key != "factor_keys"}
        for row in value["factor_rows"]]
    return copy


def _identity_gate(candidate: Path, baseline: Path, phase: dict,
                   previous: dict, programme: dict, old_programme: dict,
                   references: dict, old_references: dict,
                   states: dict[int, np.ndarray]) -> dict:
    _require(phase["camera_id"] == previous["camera_id"] == "low_end"
             and phase["source_sha256"] == previous["source_sha256"] == SOURCE_SHA256
             and phase["model_sha256"] == previous["model_sha256"] == SCENE_SHA256,
             "Original source, scene or camera identity changed")
    equal_keys = ("source_inputs", "aliases", "exposure_clock", "shot_knots",
                  "frame_clock_virtual_pts", "factor_counts",
                  "reference_calibration", "development_ad_source",
                  "development_ad_factors", "all_reference_source_curve_groups",
                  "all_reference_source_curve_pixels",
                  "excluded_unvalidated_ad4_absolute_pixel_factors")
    for name in equal_keys:
        _require(phase[name] == previous[name], f"Original source or prior configuration changed: {name}")
    _require(_strip_f1_factor_keys(phase["correlated_user_f1"])
             == _strip_f1_factor_keys(previous["correlated_user_f1"]),
             "Original F1 source pixels, correlation weights or identities changed")
    for name in ("pts", "shot_index", "exposure_representative_pts"):
        _require(np.array_equal(programme[name], old_programme[name]),
                 f"Original complete exposure identity changed: {name}")
    _require(np.array_equal(references["pts"], old_references["pts"]),
             "Original reference source PTS changed")
    for packet in (programme, old_programme, references, old_references):
        if "source_sha256" in packet:
            _require(str(packet["source_sha256"]) == SOURCE_SHA256,
                     "Camera packet source hash changed")
        if "model_sha256" in packet:
            _require(str(packet["model_sha256"]) == SCENE_SHA256,
                     "Camera packet scene hash changed")
    old_prior = _states(baseline / "prior_initializer_states.npz")
    new_prior = _states(candidate / "prior_initializer_states.npz")
    new_keys = set(map(int, phase["time_principal"]["new_keys"]))
    _require(set(new_prior) == set(old_prior) | new_keys
             and all(np.array_equal(value, new_prior[key])
                     for key, value in old_prior.items())
             and all(np.array_equal(new_prior[key], np.zeros(11)) for key in new_keys),
             "Original prior targets or zero new-node prior initialization changed")
    _require(set(new_prior) == set(states), "Final graph state key set changed")
    gate = _read(candidate / "zero_time_principal_equivalence.json")
    original_trace = _read(baseline / "solver_trace.json")
    _require(gate["original_complete_factors"] == previous["graph_factors_with_source_calibration"]
             and gate["new_optical_prior_factors"] == 141
             and gate["previous_final_cost"] == original_trace["final_cost"]
             and gate["zero_time_principal_cost"] == gate["previous_final_cost"]
             and gate["absolute_difference"] == 0.
             and gate["old_priors_bitwise_equal"]
             and gate["programme_and_reference_cameras_bitwise_equal"]
             and gate["baseline_camera_sha256"] == _sha(baseline / "shot_cameras.npz")
             and gate["baseline_states_sha256"] == _sha(baseline / "states.npz"),
             "Original source-cost equivalence gate failed")
    return dict(source_sha256=SOURCE_SHA256, scene_sha256=SCENE_SHA256,
                original_source_inputs_equal=True, exposure_aliases_equal=True,
                original_source_and_reference_identities_equal=True,
                original_f1_uv_identity_and_budget_equal=True,
                original_prior_targets_bitwise_equal=True,
                zero_new_node_priors=True,
                original_complete_factor_count=gate["original_complete_factors"],
                new_prior_factor_count=gate["new_optical_prior_factors"],
                old_final_cost=gate["previous_final_cost"],
                zero_new_node_full_cost=gate["zero_time_principal_cost"],
                original_equivalence_absolute_difference=gate["absolute_difference"])


def audit(candidate: Path, baseline: Path) -> dict:
    required = ("summary.json", "states.npz", "shot_cameras.npz",
                "reference_cameras.npz", "solver_trace.json")
    missing = [name for name in required if not (candidate / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Final time-principal export is pending: {missing}")
    phase = _read(candidate / "phase_00.json")
    previous = _read(baseline / "phase_00.json")
    meta = phase["time_principal"]
    _require(meta["schema"] == "rinkspace.optional-time-principal-physical15.v1"
             and meta["camera_id"] == "low_end"
             and not meta["source_uv_modified"],
             "Frozen time-principal model metadata changed")
    programme = _cameras(candidate / "shot_cameras.npz", reference=False)
    references = _cameras(candidate / "reference_cameras.npz", reference=True)
    old_programme = _cameras(baseline / "shot_cameras.npz", reference=False)
    old_references = _cameras(baseline / "reference_cameras.npz", reference=True)
    states = _states(candidate / "states.npz")
    identity = _identity_gate(candidate, baseline, phase, previous, programme,
                              old_programme, references, old_references, states)
    old_programme_rig = _rig(phase, programme, reference=False)
    old_reference_rig = _rig(phase, references, reference=True)
    programme_model, reference_model = reconstruct_time_principal_rigs(
        old_programme_rig, old_reference_rig, meta)

    def reconstruct(model, packet):
        computed = np.asarray([model.camera25(int(pts), states)
                               for pts in packet["pts"]])
        expected = np.asarray(packet["camera25"], np.float64)
        difference = np.abs(computed - expected)
        maximum = float(np.max(difference))
        _require(maximum <= 1e-9,
                 f"Final physical15 reconstruction differs by {maximum} camera units")
        return computed, dict(frames=len(computed), bitwise_equal=bool(np.array_equal(computed, expected)),
                              max_abs_camera25_difference=maximum)

    exported_programme, programme_check = reconstruct(programme_model, programme)
    exported_references, reference_check = reconstruct(reference_model, references)
    no_delta_programme = np.asarray([old_programme_rig.camera25(int(pts), states)
                                     for pts in programme["pts"]])
    no_delta_references = np.asarray([old_reference_rig.camera25(int(pts), states)
                                      for pts in references["pts"]])
    original_programme = np.asarray(old_programme["camera25"], np.float64)
    original_references = np.asarray(old_references["camera25"], np.float64)
    time_delta = exported_programme[:, 8:10] - no_delta_programme[:, 8:10]
    reference_delta = exported_references[:, 8:10] - no_delta_references[:, 8:10]
    _require(np.isfinite(time_delta).all() and np.isfinite(reference_delta).all(),
             "Nonfinite temporal principal-point state")
    zoom_by_pts = np.mean(exported_programme[:, 6:8], axis=1)
    old_zoom_by_pts = np.mean(original_programme[:, 6:8], axis=1)
    rotation_change = _rotation_changes_deg(original_programme, exported_programme)
    camera_centres = np.linalg.norm(exported_programme[:, :3]
                                    - original_programme[:, :3], axis=1)
    shared_K_delta = no_delta_programme[:, 8:11] - original_programme[:, 8:11]
    net_K_delta = exported_programme[:, 8:11] - original_programme[:, 8:11]

    shot_rows = {}
    pts = np.asarray(programme["pts"], np.int64)
    shots = np.asarray(programme["shot_index"], np.int64)
    reps = np.asarray(programme["exposure_representative_pts"], np.int64)
    clocks = phase["frame_clock_virtual_pts"]
    for shot in EXPECTED_SHOTS:
        members = np.flatnonzero(shots == shot)
        unique = np.asarray([index for index in members if pts[index] == reps[index]],
                            np.int64)
        unique = unique[np.argsort(pts[unique])]
        _require(len(set(map(int, reps[members]))) == len(unique),
                 f"Shot {shot} exposure alias coverage changed")
        seconds = np.asarray([float(clocks[str(int(pts[index]))]) / 15360.
                              for index in unique])
        source_pts = pts[unique]
        c = time_delta[unique]
        old_c = original_programme[unique]
        new_c = exported_programme[unique]
        shared_delta = shared_K_delta[unique]
        shot_rows[str(shot)] = dict(
            native_frames=len(members), unique_exposures=len(unique),
            source_pts_range=[int(source_pts[0]), int(source_pts[-1])],
            time_principal_delta_px=_vector_stats(c, ("cx", "cy")),
            derivative_extrema=_derivative_extrema(source_pts, seconds, c),
            relative_to_v9=dict(
                shared_zoom_only_K_change_px=_vector_stats(shared_delta, ("cx", "cy", "skew")),
                net_principal_and_skew_change_px=_vector_stats(net_K_delta[unique],
                                                               ("cx", "cy", "skew")),
                rotation_geodesic_change_deg=_stats(rotation_change[unique]),
                world_center_change_scene_units=_stats(camera_centres[unique]),
                mean_log_f_change=_stats(zoom_by_pts[unique]-old_zoom_by_pts[unique]),
                time_delta_vs_shared_zoom_K_correlation={
                    "cx": _correlation(c[:, 0], shared_delta[:, 0]),
                    "cy": _correlation(c[:, 1], shared_delta[:, 1])},
                v9_world_center_xyz_range=_vector_stats(old_c[:, :3], ("x", "y", "z")),
                final_world_center_xyz_range=_vector_stats(new_c[:, :3], ("x", "y", "z"))))

    ref_rows = {}
    reference_rotation = _rotation_changes_deg(original_references, exported_references)
    for index, stamp in enumerate(references["pts"]):
        ref_rows[str(int(stamp))] = dict(
            time_principal_delta_px=reference_delta[index].tolist(),
            shared_zoom_only_K_change_px=(no_delta_references[index, 8:11]
                                          - original_references[index, 8:11]).tolist(),
            net_principal_and_skew_change_px=(exported_references[index, 8:11]
                                              - original_references[index, 8:11]).tolist(),
            rotation_geodesic_change_deg=float(reference_rotation[index]),
            world_center_change_scene_units=float(np.linalg.norm(
                exported_references[index, :3]-original_references[index, :3])))
    priors = _prior_cost(states, meta, phase["time_principal_priors"])
    hashes = {name: _sha(candidate / name) for name in
              ("phase_00.json", "states.npz", "shot_cameras.npz",
               "reference_cameras.npz", "solver_trace.json",
               "prior_initializer_states.npz", "zero_time_principal_equivalence.json")}
    return dict(
        schema="rinkspace.low_end.time_principal_result_audit.v1",
        scope="Read-only reconstructed camera stream, original-source gate and new-prior cost; not source-pixel accuracy",
        candidate_directory=str(candidate.resolve()), baseline_directory=str(baseline.resolve()),
        candidate_sha256=hashes,
        baseline_sha256={name: _sha(baseline / name) for name in
                         ("phase_00.json", "states.npz", "shot_cameras.npz",
                          "reference_cameras.npz", "prior_initializer_states.npz")},
        audit_code_sha256=_sha(Path(__file__)),
        time_principal_implementation_sha256=meta["implementation_sha256"],
        original_source_equivalence_gate=identity,
        final_camera25_reconstruction={"programme": programme_check,
                                       "references": reference_check},
        fixed_exposure_aliases_equal=True,
        time_principal_node_counts=dict(programme=meta["programme_new_node_count"],
                                        references=meta["reference_new_node_count"],
                                        effective_parameters=meta["effective_new_parameter_count"]),
        new_141_prior_cost=priors,
        programme_by_shot=shot_rows,
        independent_references=ref_rows,
        interpretation_limits=[
            "The time-principal and shared-K terms can compensate under soft zero-mean priors; their individual fitted values are not direct OIS/EIS measurements.",
            "R and C changes are relative to the completed v9 model, not proof of a physical camera mechanism.",
            "No source pixels are rescored here; use the same complete-source and historical validation tools before any positioning accuracy claim.",
        ])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f"Refusing to overwrite existing result audit: {args.output}")
    result = audit(args.candidate, args.baseline)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf8")
    print(json.dumps(dict(output=str(args.output.resolve()),
                          camera_reconstruction=result["final_camera25_reconstruction"],
                          new_prior_cost=result["new_141_prior_cost"]["total_cost"]),
                     ensure_ascii=False))
