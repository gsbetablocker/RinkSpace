"""Existing rig/world derivatives of direct user axes and same-frame Ad7."""
from pathlib import Path
import os
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["RINKSPACE_PROJECTION_BACKEND"] = "numpy"
import argparse
from dataclasses import replace
import hashlib
import json
import sys
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from rinkspace.vision_camera_initialization import (current_complete_source_geometry,
    frozen_scene_curve, unit_source_normals, authenticated_view_slots,
    source_jacobian_in_camera25_units)
from calibration.program_tracking.z190_physical_camera import SharedCalibration, ViewState
from calibration.solver_runtime import install
install()
from z190_source_camera_diagnostic import absolute_value_jac15
from structured_programme_source_candidate.current_ice_material_scene import CurrentIceMaterialView

DATA = Path(r"D:/CodexData/RinkSpace_Data/events/ankara_2026")
RUNS = DATA / "calibration/runs/solver_acceleration_20261003"
DIRECTORY = DATA / "observations/vision_training/continuous_review/cut6_clear_source_constraints"
DEFAULT_STATE = RUNS / "source_corrected_clear_actual_complete_joint_20261007/accepted_states/0eb5c6d4d2b0a24ab201f65361ea0ed2de409b4bdfdddb1fc68c6b331ae54e63"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-dir", default=str(DEFAULT_STATE))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    directory = Path(args.state_dir)
    identity = read(directory / "state_identity.json")
    state_path = directory / "complete_state_UNACCEPTED.npz"
    with np.load(state_path, allow_pickle=False) as saved:
        keys = ["q", "aux__canonical_pts", "aux__block_lengths", "aux__block_start",
            "aux__installation_xyz_slots", "aux__current_shared_mount_parameter_indices",
            "aux__current_all_installation_optics_parameter_indices", "aux__current_named66_parameter_names",
            "aux__current_camera_mount_working_constraints_json"]
        state = {k: saved[k].copy() for k in keys}
    if hashlib.sha256(state["q"].tobytes()).hexdigest() != identity["q_sha256"]:
        raise ValueError("Frozen q changed")
    with np.load(identity["camera_path"], allow_pickle=False) as saved:
        camera = {k: saved[k].copy() for k in ("pts", "exposure_representative_pts", "camera25", "view_state4", "installation_id")}
        metadata = json.loads(saved["metadata_json"].item())
    view_slots, i, actual = authenticated_view_slots(state, camera, 230501120, allow_reference=True)
    installation = str(camera["installation_id"][i])
    cal = SharedCalibration.from_manifest(metadata["calibrations"][installation])
    view = camera["view_state4"][i].copy()
    physical_camera, shared_map, view_map = cal.camera25_and_jacobians(ViewState.from_state4(view))
    if not np.array_equal(physical_camera, camera["camera25"][i]):
        raise ValueError("Complete fixed-pivot/nonzero-lever rig does not reproduce camera25")
    mount = json.loads(state["aux__current_camera_mount_working_constraints_json"].item())
    unit = float(mount["scene_unit_working_m"])
    scene = read(directory / "scene_UNACCEPTED.json")
    provider, world, proof = current_complete_source_geometry(state_path,
        RUNS / "user_corrected_programme_candidate/latest9a8c_world66_initialization_ONLY.json")
    material = CurrentIceMaterialView(provider.current)
    features = material.features(world)
    oldfeatures = {f["id"]: f for f in scene["ice_features"]}
    for name in ("lane2", "lane3"):
        if not np.allclose(features[name]["paint_segments_xyz"], oldfeatures[name]["paint_segments_xyz"], rtol=0, atol=1e-10):
            raise ValueError("Current original Lane material does not reproduce frozen scene")
    user_packet = read(DIRECTORY / "complete_source_clear_anchor_append_20261007/ice_clear_anchors_user_lane2_lane3_20261007.json")
    user = [r for r in user_packet["rows"] if r.get("user_confirmed_source_axis")]
    ads = read(DIRECTORY / "late_230501120_qualified_Ad7_external_source_packet.json")["rows"]
    sources = user + ads
    working, ranges, normals, cursor = [], [], [], 0
    for source in sources:
        uv = np.asarray(source["uv"], float)
        n, _ = unit_source_normals(source["normal"], len(uv))
        working.append(dict(source, uv=uv, normal=n, sigma=np.ones(len(uv)),
            world=frozen_scene_curve(scene, source["curve"])))
        ranges.append((cursor, cursor + len(uv)))
        normals.append(n)
        cursor += len(uv)
    normals = np.concatenate(normals)

    def value(calibration=cal, current_view=view, current_world=None):
        c = calibration.camera25(ViewState.from_state4(current_view))
        ice = None if current_world is None else dict(ice_features=list(material.features(current_world).values()))
        pieces = []
        for r in working:
            row = dict(r)
            # Curling-only columns do not change any rink wall/printed material.
            # Ad7 stays the authenticated frozen same-frame material in this
            # derivative probe, even if concurrent print code repairs exist.
            if ice is not None and r.get("user_confirmed_source_axis"):
                row["world"] = frozen_scene_curve(ice, r["curve"])
            pieces.append(absolute_value_jac15(c, [row], False).reshape(-1, 2))
        return np.einsum("ni,ni->n", np.concatenate(pieces), normals)

    raw, camera_J = absolute_value_jac15(physical_camera, working, True)
    camera_J = source_jacobian_in_camera25_units(camera_J.toarray() if hasattr(camera_J, "toarray") else camera_J)
    raw = raw.reshape(-1, 2)
    baseline = np.einsum("ni,ni->n", raw, normals)
    normal_camera_J = np.einsum("ni,nij->nj", normals, camera_J.reshape(-1, 2, 15))
    normal_view_J = normal_camera_J @ view_map[:15]
    names = state["aux__current_named66_parameter_names"].tolist()
    columns = []
    for name in ("curling_tee_midpoint_x_m", "curling_row_midpoint_y_m", "curling_common_yaw_rad",
                 "curling_four_row_pitch_m", "four_Lane_paint_common_width_m"):
        k = names.index(name)
        h = 1e-6 if name.endswith("_rad") else 1e-4
        def probe(delta):
            q = world.copy(); q[k] += delta
            return value(current_world=q)
        d = (probe(h) - probe(-h)) / (2*h)
        repeated = (probe(h/2) - probe(-h/2)) / h
        columns.append(dict(name=name, family="shared_floor_geometry", units="rad" if name.endswith("_rad") else "working_m",
            global_q_indices=[int(proof["actual_world_parameter_indices"][k])], baseline_value=float(world[k]),
            derivative=d, method="actual current Lane material central FD", half_step_relative_L2=float(np.linalg.norm(d-repeated)/max(1., np.linalg.norm(repeated)))))
    # Analytic source-camera J is chained through the full existing rig map.
    # Pitch/roll/yaw below are local directions in the existing three Rodrigues
    # coordinates; no independent optical centre or new rotation DOF is added.
    for k, axis in enumerate(("pitch_camera_X", "yaw_camera_Y", "roll_camera_Z")):
        h = 1e-6
        R = cv2.Rodrigues(view[:3])[0]
        def direction(delta):
            rv = np.zeros(3); rv[k] = delta
            v = view.copy(); v[:3] = cv2.Rodrigues(cv2.Rodrigues(rv)[0] @ R)[0].reshape(-1)
            return v
        tangent = (direction(h)[:3] - direction(-h)[:3]) / (2*h)
        d = normal_view_J[:, :3] @ tangent
        fd = (value(current_view=direction(h)) - value(current_view=direction(-h))) / (2*h)
        columns.append(dict(name=axis, family="per_exposure_existing_rig_rotation", units="rad",
            global_q_indices=view_slots[:3].tolist(), existing_Rodrigues_q_direction=tangent.tolist(),
            derivative=d, method="analytic source J chained to existing full rig view J",
            FD_relative_L2=float(np.linalg.norm(d-fd)/max(1., np.linalg.norm(fd)))))
    d = normal_view_J[:, 3]
    h = 1e-5
    vp, vm = view.copy(), view.copy(); vp[3] += h; vm[3] -= h
    fd = (value(current_view=vp) - value(current_view=vm)) / (2*h)
    columns.append(dict(name="existing_log_focal", family="per_exposure_existing_focal", units="log_native_focal",
        global_q_indices=[int(view_slots[3])], derivative=d,
        method="analytic source J through full rig focal/PP/radial/axial map",
        FD_relative_L2=float(np.linalg.norm(d-fd)/max(1., np.linalg.norm(fd)))))
    for kind in ("pivot_Z", "lever_Z"):
        h = 1e-4
        field = "pivot_xyz" if kind == "pivot_Z" else "lever_xyz"
        def calibration(delta):
            vector = np.asarray(getattr(cal, field)).copy(); vector[2] += delta/unit
            return replace(cal, **{field: tuple(vector)})
        direction25 = np.zeros(15)
        if kind == "pivot_Z":
            direction25[2] = 1/unit
            index = int(state["aux__installation_xyz_slots"][1, 2])
        else:
            direction25[:3] = cv2.Rodrigues(view[:3])[0].T[:, 2]/unit
            index = int(mount["by_installation"][installation]["mount_parameter_indices"][2])
        d = normal_camera_J @ direction25
        fd = (value(calibration(h)) - value(calibration(-h))) / (2*h)
        columns.append(dict(name=kind, family="shared_installation_existing_mount", units="working_m",
            global_q_indices=[index], baseline_value=float(state["q"][index]), derivative=d,
            method="analytic source J; same fixed P plus R transpose nonzero shared lever",
            FD_relative_L2=float(np.linalg.norm(d-fd)/max(1., np.linalg.norm(fd)))))
    columns.append(dict(name="shared_principal_y_native_px", family="shared_existing_optics", units="native_px",
        global_q_indices=[int(state["aux__current_all_installation_optics_parameter_indices"][4])],
        derivative=normal_camera_J[:, 9], method="actual analytic native PPy source derivative"))
    # Per-column LS direction is a diagnostic of explanatory capacity only.
    # It is never written as an initializer/state or sent to the optimizer.
    summaries = []
    user_count = sum(len(r["uv"]) for r in user)
    for c in columns:
        d = c.pop("derivative")
        lane = d[:user_count]
        delta = float(-lane @ baseline[:user_count] / max(1e-30, lane @ lane))
        predicted = baseline + delta*d
        summary = dict(c, derivative_per_source_normal_native_px_per_unit=d.tolist(),
            linear_single_direction_lane_least_squares_delta=delta,
            linear_both_lane_mean_abs_before_px=float(np.abs(baseline[:user_count]).mean()),
            linear_both_lane_mean_abs_after_px=float(np.abs(predicted[:user_count]).mean()),
            linear_Ad7_mean_abs_before_px=float(np.abs(baseline[user_count:]).mean()),
            linear_Ad7_mean_abs_after_px=float(np.abs(predicted[user_count:]).mean()),
            conditional_Ad7_change_mean_abs_px=float(np.abs(delta*d[user_count:]).mean()),
            per_group_derivative_mean=[dict(id=r["id"], mean=float(d[a:b].mean())) for r, (a,b) in zip(sources,ranges)])
        summaries.append(summary)
    # Explain local height/rotation/focal ambiguity with the actual existing
    # source weights. These are diagnostics, never q updates or initializers.
    by_name = {c["name"]: c for c in summaries}
    source_sigma = np.concatenate([np.repeat(4., len(r["uv"])) if r.get("user_confirmed_source_axis")
        else np.asarray(r["sigma_px"], float) for r in sources])
    coupled = []
    for direction_names in (("pivot_Z", "pitch_camera_X"),
            ("pivot_Z", "pitch_camera_X", "existing_log_focal"),
            ("pitch_camera_X", "yaw_camera_Y", "roll_camera_Z", "existing_log_focal"),
            ("curling_row_midpoint_y_m", "four_Lane_paint_common_width_m")):
        J = np.column_stack([by_name[n]["derivative_per_source_normal_native_px_per_unit"] for n in direction_names])
        delta = np.linalg.lstsq(J/source_sigma[:, None], -baseline/source_sigma, rcond=None)[0]
        linear = baseline + J @ delta
        trial_cal, trial_view, trial_world = cal, view.copy(), world.copy()
        local_rotation = np.zeros(3)
        floor_changed = False
        for name, amount in zip(direction_names, delta):
            if name == "pivot_Z":
                pivot = np.asarray(cal.pivot_xyz).copy(); pivot[2] += amount/unit
                trial_cal = replace(cal, pivot_xyz=tuple(pivot))
            elif name == "existing_log_focal":
                trial_view[3] += amount
            elif name in ("pitch_camera_X", "yaw_camera_Y", "roll_camera_Z"):
                local_rotation[("pitch_camera_X", "yaw_camera_Y", "roll_camera_Z").index(name)] += amount
            else:
                trial_world[names.index(name)] += amount
                floor_changed = True
        if np.linalg.norm(local_rotation):
            trial_view[:3] = cv2.Rodrigues(cv2.Rodrigues(local_rotation)[0] @ cv2.Rodrigues(view[:3])[0])[0].reshape(-1)
        nonlinear = value(calibration=trial_cal, current_view=trial_view,
            current_world=trial_world if floor_changed else None)
        coupled.append(dict(parameter_names=list(direction_names), existing_parameter_deltas=delta.tolist(),
            linear_lane_mean_abs_px=float(np.abs(linear[:user_count]).mean()),
            linear_Ad7_mean_abs_px=float(np.abs(linear[user_count:]).mean()),
            actual_nonlinear_lane_mean_abs_px=float(np.abs(nonlinear[:user_count]).mean()),
            actual_nonlinear_lane_max_abs_px=float(np.abs(nonlinear[:user_count]).max()),
            actual_nonlinear_Ad7_mean_abs_px=float(np.abs(nonlinear[user_count:]).mean()),
            actual_nonlinear_Ad7_max_abs_px=float(np.abs(nonlinear[user_count:]).max()),
            diagnostic_only_not_written_to_state_or_solver=True,
            complete_motion_other_exposure_rink_house_or_mount_constraints_not_evaluated=True))
    report = dict(schema="rinkspace.user-lane-existing-rig-world-local-J-probe.v1",
        state_directory=str(directory), q_sha256=identity["q_sha256"], camera_path=identity["camera_path"],
        pts=230501120, actual_pts=actual, installation_id=installation,
        existing_camera25_reproduced_bitexact=True, current_Lane_factory_matches_frozen_export=True,
        physical_relation="C = shared fixed P + R.T @ shared nonzero camera-local lever; complete original focal/PP/Brown expression retained",
        source_groups=[dict(id=r["id"], curve=r["curve"], uv=r["uv"], signed_normal_error_px=baseline[a:b].tolist()) for r,(a,b) in zip(sources,ranges)],
        parameters=summaries, coupled_direction_diagnostics=coupled, source_points_or_existing_state_modified=False,
        diagnostic_linear_deltas_not_feasible_global_updates_or_initializers=True,
        complete_motion_other_exposure_rink_house_or_mount_constraints_not_evaluated=True,
        full_source_J_or_solver_not_started=True, localization_accuracy_accepted=False)
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(dict(output=str(output), parameters=[dict(name=r["name"],
        lane_group_J=[x["mean"] for x in r["per_group_derivative_mean"][:2]],
        single_delta=r["linear_single_direction_lane_least_squares_delta"],
        lane_after=r["linear_both_lane_mean_abs_after_px"],
        Ad7_change=r["conditional_Ad7_change_mean_abs_px"]) for r in summaries]), ensure_ascii=False))


if __name__ == "__main__":
    main()
