"""Export and compare a repaired native programme on its original source clock."""
from argparse import ArgumentParser
from pathlib import Path
import json
import shutil

import numpy as np

from calibration.program_tracking.body_tracking.absolute_motion_state import absolute_reference, curve_from_packet
from calibration.program_tracking.body_tracking.continuous_physics import ContinuousMassGeometry
from calibration.program_tracking.body_tracking.actual_observation_ledger import update_source_observation_export
from calibration.program_tracking.body_tracking.programme_com import export_com_diagnostics
from calibration.program_tracking.body_tracking.rebuild_prepare import load
from calibration.program_tracking.body_tracking.sequence_skate_state import shoe_curve_from_packet
from calibration.program_tracking.body_tracking.skate_checkpoint import validate_checkpoint_identity
from calibration.program_tracking.body_tracking.skate_delivery import export_candidate
from calibration.program_tracking.body_tracking.skate_physics import fk_state
from calibration.program_tracking.body_tracking.trajectory_backend import load_reference

from .body_skate_fit import Problem
from .deliver_body_skate_fit import render_video
from .inputs import PROJECT, atomic_json, digest, emit, read


VARIANT = "programme_opening_skate_direction_1113"
LABEL = "开场站姿与整鞋方向修正版"
PROOF_SECONDS = (
    22.12, 22.82, 23.42, 23.76, 24.32, 24.64, 25.12, 34.4, 47.3, 50.90,
    57.08, 69.02, 69.46, 81.02, 81.40, 97.74, 101.64, 101.66, 104.38, 107.16, 119.3,
    136.6, 140.88, 142.94, 143.10, 151.86, 157.26, 160.24, 172.88, 173.22,
)


def bind(prepared, evidence, parameter):
    ref, camera, info, document, _ = load_reference(PROJECT, prepared, evidence)
    controls = load(parameter)
    validate_checkpoint_identity(controls, ref.fixed_checkpoint_identity)
    ref = absolute_reference(ref, curve_from_packet(controls), root=prepared)
    ref.relative_skate_curve = shoe_curve_from_packet(controls)
    ref.binding.x = controls["x"][:17].copy()
    ref.binding._parameter_key = ref.binding._packet = None
    ref.binding.reference = ref
    return ref, camera, info, document, controls


def changed_support(previous, current, times):
    before, after = curve_from_packet(previous), curve_from_packet(current)
    np.testing.assert_array_equal(before.knots, after.knots)
    changed = np.any(before.position_controls != after.position_controls, axis=1)
    changed |= np.any(before.rotation_controls != after.rotation_controls, axis=(1, 2, 3))
    rows = np.flatnonzero(changed)
    return np.asarray(after.basis_sparse(times)[:, rows].getnnz(axis=1) > 0)


def intervals(mask, seconds):
    starts = np.flatnonzero(mask & ~np.r_[False, mask[:-1]])
    ends = np.flatnonzero(mask & ~np.r_[mask[1:], False])
    return np.column_stack((seconds[starts], seconds[ends])).tolist()


def export(args):
    output, previous = args.output, args.previous
    param = args.parameter
    prior = previous / "solve/candidate.npz"
    ref, camera, info, document, controls = bind(args.prepared, args.evidence, param)
    parent_controls = load(prior)
    for name in ("x", "relative_skate_rotation_controls", "relative_skate_knots"):
        fixed = slice(0, 17) if name == "x" else slice(None)
        np.testing.assert_array_equal(controls[name][fixed], parent_controls[name][fixed])
    manifest = read(output / "navigation.json")
    pts = ref.p["pts"]
    tb = np.asarray(info["source_time_base"], dtype=np.int64)
    np.testing.assert_array_equal(tb, manifest["time_base"])
    sec = pts * (tb[0] / tb[1])
    mask = changed_support(parent_controls, controls, ref.times)
    changed_intervals = intervals(mask, sec)
    emit("programme_export_start", parameter=str(param), pts=pts[[0, -1]].tolist(),
         exposures=len(pts), changed_frames=int(mask.sum()), camera_sha256=info["state_sha256"])

    result = export_candidate(args.prepared, output, ref, camera, info,
                              document.get("supports", []), write=False)
    native_state = ref.at(ref.times)
    world = fk_state(ref, native_state, np.zeros((len(pts), 69)))
    mass = ContinuousMassGeometry(ref)
    centres, rotations = mass.from_pose(world)
    result.update(export_com_diagnostics(mass, ref.times, centres, rotations))
    result["com_projection_world_mid"] = result["com_world_mid"].copy()
    result["com_projection_world_mid"][:, 2] = 0.
    result.update(native_body_local_R_mid=world["local_R"],
                  native_body_FK_world_R_mid=world["Q"][:, None] @ world["frames"],
                  P_velocity_world_mps=ref.root_derivative(ref.times, 1) * ref.binding.ratio,
                  shared_betas=ref.p["shared_betas"], shared_body_scale=ref.p["body_scale"],
                  ice_plane_known=np.array(ref.ice_plane_known),
                  source_action_ledger_sha256=np.array(digest(args.evidence)),
                  body_refinement_window_mask=mask, window_mask=mask)
    ledger = args.prepared / "observations.npz"
    observations = load(ledger)
    np.testing.assert_array_equal(observations["pts"], pts)
    # This export only replays the controls. A matching optimizer receipt is
    # required before claiming objective use for any admitted observation.
    update_source_observation_export(result, observations, ledger, np.empty(0, dtype=int))
    result["source_observation_objective_known"] = np.array(False)
    result["trajectory_evidence_status"] = np.where(
        observations["use"].sum(1) >= 4, "source_constrained_estimate", "temporal_inference")
    result["position_uncertainty_status"] = np.full(
        len(pts), "conditional_scale_and_depth; no independent accuracy acceptance")
    result["sequence_contact_modes"] = controls["sequence_contact_modes"]
    result["sequence_contact_status"] = controls["sequence_contact_status"]
    result["sequence_contact_provenance"] = np.array("historical mode hypotheses carried for comparison")
    result["opening_hidden_feet_status"] = np.where(
        sec <= 23.94, "pose_and_previous_shot_inference", "see source visibility")
    result["scoped_followup_json"] = np.array(json.dumps(dict(
        baseline_parameter_sha256=digest(prior), parameter_sha256=digest(param),
        source_evidence_sha256=digest(args.evidence),
        repaired_intervals=changed_intervals, control_support_mask=True,
        observations_and_ice_contact_are_distinct=True,
    ), sort_keys=True))
    np.testing.assert_array_equal(result["source_pts"], pts)
    np.testing.assert_allclose(result["native22_world_mid"], world["native_world"] * ref.binding.ratio,
                               rtol=0, atol=1.e-9)
    np.testing.assert_allclose(result["skate_R_world_mid"], world["R"], rtol=0, atol=1.e-9)
    np.testing.assert_allclose(result["skate_origin_world_mid"], world["O"] * ref.binding.ratio,
                               rtol=0, atol=1.e-9)
    assert np.isfinite(result["com_world_mid"]).all()

    fitted = output / "fitted"
    fitted.mkdir(exist_ok=True)
    target = fitted / "result.npz"
    pending = fitted / "result.pending.npz"
    if target.exists():
        raise FileExistsError(target)
    with pending.open("xb") as stream:
        np.savez_compressed(stream, **result)
    pending.replace(target)

    config = read(previous / "viewer3d.json")
    old = config["variants"][0]
    candidate = dict(old, id=VARIANT, candidate_id=VARIANT, label=LABEL,
                     result=str(target), result_sha256=digest(target),
                     parameter=str(param), parameter_sha256=digest(param),
                     body_reference_label=LABEL, body_refinement_intervals_seconds=changed_intervals,
                     required_saved_diagnostics=["com_world_mid"],
                     accuracy_accepted=False, user_accepted=False)
    config.update(variants=[candidate, *config["variants"]], default_variant=VARIANT, session_id=VARIANT,
                  latest_review_variant=VARIANT, comparison_variants=[VARIANT, old["id"]],
                  source_navigation=str(output / "navigation.json"), trajectory_reference="com",
                  body_window_scope="全节目连续轨迹，开场和所列整鞋事件局部重拟合",
                  body_scope_label="开场与整鞋方向复核",
                  display_outside_window_label="此处沿用上一轮同一人体状态。",
                  review_status="人体、整鞋、模型质心及冰面投影")
    atomic_json(output / "viewer3d.json", config)
    launcher = output / "START_REFITTED_3D.cmd"
    if not launcher.exists():
        shutil.copyfile(previous / launcher.name, launcher)
    atomic_json(output / "export_receipt.json", dict(
        parameter=str(param), parameter_sha256=candidate["parameter_sha256"],
        result=str(target), result_sha256=candidate["result_sha256"],
        camera_sha256=info["state_sha256"], source_pts=pts[[0, -1]].tolist(), source_time_base=tb.tolist(),
        exposures=len(pts), changed_exposures=int(mask.sum()), intervals=changed_intervals,
        exactly_replayed_fields=["native22_world_mid", "skate_R_world_mid", "skate_origin_world_mid"],
        com="same body and attached skate state; projected to world Z=0",
        accuracy_accepted=False))
    emit("programme_export_complete", result=str(target), sha256=candidate["result_sha256"])


def video(args):
    problem = Problem(args.visual, args.output)
    old = bind(args.prepared, args.evidence, args.previous / "solve/candidate.npz")
    new = bind(args.prepared, args.evidence, args.parameter)
    for ref, _, info, _, _ in (old, new):
        np.testing.assert_array_equal(ref.p["pts"], problem.i.pts)
        if info["state_sha256"] != problem.i.camera_sha:
            raise ValueError("Viewer and video must use the same native camera")
    render_video(problem, args.output, references=(old[0], new[0]), proof_seconds=PROOF_SECONDS,
                 source_window=args.window, parameter=args.parameter,
                 refit_label="Opening and skate direction repair")


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("export", "video"))
    for name in ("prepared", "evidence", "previous", "parameter", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--visual", type=Path)
    parser.add_argument("--window", type=float, nargs=2)
    args = parser.parse_args()
    if args.stage == "export":
        export(args)
    else:
        video(args)


if __name__ == "__main__":
    main()
