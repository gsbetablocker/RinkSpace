"""Camera-only diagnostic on explicit existing same-exposure FIT/HELD sources."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
import numpy as np
from scipy.optimize import least_squares
from scipy.optimize._numdiff import approx_derivative
from rinkspace.vision_camera_probe import FrozenCameraSourceProbe


def mixed_source_loss(quadratic):
    """Original native quadratic and original absolute Huber2, same scalars."""
    def loss(z):
        rho = np.zeros((3, len(z)))
        rho[0], rho[1] = z, 1.
        active = (~quadratic) & (z > 4.)
        root = np.sqrt(z[active])
        rho[0, active] = 4. * root - 4.
        rho[1, active] = 2. / root
        rho[2, active] = -1. / (z[active] * root)
        return rho
    return loss


def row_metrics(probe, rows, camera):
    result = []
    for item in rows:
        row = item["row"]
        record = dict(source_instance_key=item["source_instance_key"], id=row["id"],
            family=item["family"], collection_role=item["collection_role"],
            original_split=item["original_split"], primitive=probe.primitive_key(item),
            source_UV_context_count=len(row["uv"]), quadratic=item["quadratic"],
            point_correspondence=row.get("point_correspondence", False))
        try:
            residual = probe.residual(item, camera)
            pixels = residual.reshape(-1, 2) * np.asarray(row["sigma"]).reshape(-1, 1)
            distances = np.linalg.norm(pixels, axis=1)
            record.update(status="evaluated", original_operator_outside_support_native_px_p50_p90_max=
                          np.percentile(distances, [50, 90, 100]).tolist(),
                          original_operator_native_pixel_vectors=pixels.tolist())
        except Exception as error:
            record.update(status="failed", error_type=type(error).__name__, error=str(error))
        result.append(record)
    return result


def probe_exposure(probe, actual_pts):
    from z190_physical_camera import ViewState
    rows = probe.rows_at(actual_pts)
    fit = [item for item in rows if item["collection_role"] == "FIT"]
    held = [item for item in rows if item["collection_role"] == "HELD"]
    fit_primitives = {key for item in fit if (key := probe.primitive_key(item)) is not None}
    held_primitives = {key for item in held if (key := probe.primitive_key(item)) is not None}
    unresolved_fit = [item["source_instance_key"] for item in fit if probe.primitive_key(item) is None]
    unresolved_held = [item["source_instance_key"] for item in held if probe.primitive_key(item) is None]
    if not any(key.startswith("ice:") for key in fit_primitives):
        raise ValueError("No existing same-exposure ice-paint FIT carrier")
    if not any(key.startswith(("board:", "print:")) for key in fit_primitives):
        raise ValueError("No existing same-exposure off-ice FIT carrier")
    index = probe.actual_index[actual_pts]
    camera = probe.camera["camera25"][index].copy()
    identity = str(probe.camera["installation_id"][index])
    calibration = probe.calibrations[identity]
    view = probe.camera["view_state4"][index].copy()
    quadratic = np.concatenate([np.full(2 * len(item["row"]["uv"]), item["quadratic"], bool) for item in fit])
    loss = mixed_source_loss(quadratic)
    result = dict(actual_pts=actual_pts, physical_installation_id=identity,
        native_presentations_for_actual_exposure=probe.camera["pts"][
            probe.camera["exposure_representative_pts"] == actual_pts].tolist(),
        source_time_base=probe.metadata["source_time_base"], FIT_primitive_keys=sorted(fit_primitives),
        HELD_primitive_keys=sorted(held_primitives),
        HELD_primitives_not_fitted_this_exposure=sorted(held_primitives - fit_primitives) if not unresolved_fit else [],
        unresolved_FIT_material_instances=unresolved_fit, unresolved_HELD_material_instances=unresolved_held,
        unresolved_material_not_used_to_claim_HELD_independence=True,
        same_primitive_HELD_controls=sorted(held_primitives & fit_primitives),
        primitive_group_noise_independence_not_certified=True,
        geometry_previously_used_same_feature_other_exposures=True,
        frozen_parameters="same accepted scene, shared pivot/lever, PP/Brown function coefficients and focal aspect",
        principal_and_Brown_functions_evaluated_at_trial_focal=True,
        complete_motion_or_human_hard_objective_not_replaced=True,
        complete_motion_objective_evaluated=False, human_hard_guard_evaluated=False,
        small_absolute_source_probe_is_not_complete_production_acceptance=True,
        all_source_UV_sigma_support_roles_splits_unchanged=True,
        initial_camera25=camera.tolist(), B0=row_metrics(probe, rows, camera), candidates={})
    for mode in ("fixed_rig_RF4", "free_camera_R_C_logf7"):
        started = time.perf_counter()
        if mode == "fixed_rig_RF4":
            initial = view.copy()
            def to_camera(p):
                return calibration.camera25(ViewState.from_state4(p,
                    effective_pp_offset_px=probe.camera["effective_pp_offset_px"][index]))
        else:
            initial = np.r_[camera[:3] / 100., camera[3:6], float(np.mean(camera[6:8]))]
            def to_camera(p):
                candidate = camera.copy()
                candidate[:3], candidate[3:6] = p[:3] * 100., p[3:6]
                candidate[6:8] += p[6] - float(np.mean(camera[6:8]))
                # Re-evaluate the original PP/Brown functions at the trial f.
                # Only the diagnostic C is free; function coefficients stay frozen.
                trial_view = calibration.view_from_camera25(candidate)
                current = calibration.camera25(ViewState.from_state4(trial_view.state4(),
                    effective_pp_offset_px=probe.camera["effective_pp_offset_px"][index]))
                current[:3] = candidate[:3]
                return current
        def residual(p):
            candidate = to_camera(p)
            return np.concatenate([probe.residual(item, candidate) for item in fit])
        candidate_result = dict(actual_free_parameters=len(initial), held_used_to_select_seed_or_fit=False,
            original_loss_policy="original native quadratic; original absolute Huber2",
            generic_correspondence_or_operator_substitution_used=False)
        try:
            initial_residual = residual(initial)
            jacobian = approx_derivative(residual, initial, method="3-point")
            singular = np.linalg.svd(jacobian, compute_uv=False)
            scaled = jacobian / np.maximum(np.linalg.norm(jacobian, axis=0), np.finfo(float).tiny)
            column_singular = np.linalg.svd(scaled, compute_uv=False)
            candidate_result.update(initial_FD_singular_values=singular.tolist(),
                initial_FD_rank=int(np.sum(singular > singular[0] * max(jacobian.shape) * np.finfo(float).eps)),
                initial_chart_condition=float(singular[0] / singular[-1]) if singular[-1] else None,
                initial_unit_column_singular_values=column_singular.tolist(),
                initial_unit_column_condition=float(column_singular[0] / column_singular[-1]) if column_singular[-1] else None,
                initial_normalized_RMS=float(np.sqrt(np.mean(initial_residual ** 2))))
            solved = least_squares(residual, initial, jac="3-point", loss=loss, x_scale="jac",
                                   ftol=1e-9, xtol=1e-9, gtol=1e-9)
            candidate = to_camera(solved.x)
            fixed_rig = calibration.camera25(calibration.view_from_camera25(candidate))
            candidate_result.update(success=bool(solved.success), status=int(solved.status),
                termination=str(solved.message), function_evaluations=int(solved.nfev),
                jacobian_evaluations=int(solved.njev), source_cost=float(solved.cost),
                normalized_RMS_after=float(np.sqrt(np.mean(solved.fun ** 2))),
                candidate_camera25=candidate.tolist(), candidate_focal_px=np.exp(candidate[6:8]).tolist(),
                candidate_optical_C_minus_B0_scene_units=(candidate[:3] - camera[:3]).tolist(),
                candidate_optical_C_gap_from_same_R_f_current_fixed_rig_scene_units=float(np.linalg.norm(candidate[:3] - fixed_rig[:3])),
                all_original_rows_after=row_metrics(probe, rows, candidate),
                convergence_is_not_accuracy_or_full_scene_feasibility=True)
        except Exception as error:
            candidate_result.update(success=False, error_type=type(error).__name__, error=str(error),
                                    no_generic_fallback=True)
        candidate_result["elapsed_seconds"] = time.perf_counter() - started
        result["candidates"][mode] = candidate_result
    return result


def main():
    parser = argparse.ArgumentParser()
    for field in ("state", "camera", "auxiliary", "input-config", "output"):
        parser.add_argument("--" + field, type=Path, required=True)
    parser.add_argument("--actual-pts", type=int, nargs="+", required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve the existing diagnostic result; choose a new explicit output path")
    config = json.loads(args.input_config.read_text(encoding="utf-8-sig"))
    started = time.perf_counter()
    probe = FrozenCameraSourceProbe(args.state, args.camera, args.auxiliary, config["source_options"])
    report = dict(schema="rinkspace.current-mixed-source-frozen-camera-probe.v1", provenance=probe.provenance,
        loading_seconds=time.perf_counter() - started, requested_actual_exposures=args.actual_pts,
        poselib_installed=False, production_states_or_sources_written=False, accuracy_accepted=False,
        comparisons=[probe_exposure(probe, actual) for actual in args.actual_pts])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    print(args.output)
    for record in report["comparisons"]:
        print(record["actual_pts"], {mode: {key: candidate.get(key) for key in (
            "success", "initial_FD_rank", "initial_chart_condition", "initial_unit_column_condition",
            "initial_normalized_RMS", "normalized_RMS_after", "candidate_optical_C_gap_from_same_R_f_current_fixed_rig_scene_units", "error")}
            for mode, candidate in record["candidates"].items()})


if __name__ == "__main__":
    main()
