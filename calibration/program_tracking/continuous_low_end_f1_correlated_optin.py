"""Optional F1 user-stroke factors with the original Step02 curve budget.

This module only appends three factors after ordinary low-end reference nodes
and factors have been prepared. It does not edit the frozen source loader,
programme objective, shared curve engine, or solver. The numerical sigma of
one pixel reproduces the old Step02 calculation; it is not a measured one-pixel
annotation uncertainty. A physical F1 object receives one total weight budget
across its three recovered user-drawn rings.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from adaptive_curve_parallel import AdaptiveCurveJob
from continuous_adaptive_curve_implicit_optin import adaptive_implicit_optin_metadata
from continuous_brown_adapter import adapt_factor
from reference_board_pair import BoardIntersectionError
from secondary_nearboard_adaptive_curve import (
    AdaptiveProjectedPolyline, adaptive_curve_callback)


PTS = 205992960
ROLES = ("outer", "middle", "inner")
IDS = tuple(f"low3_{PTS}:user_F1_{role}_full" for role in ROLES)


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def _f1_rows(inventory: dict, prepared) -> tuple[dict, list[dict]]:
    _require(inventory.get("schema") == "rinkspace.low_end_reference_source_inventory.v1"
             and inventory.get("camera_id") == "low_end"
             and inventory.get("optional_user_f1_enabled") is True
             and inventory.get("counts", {}).get("reference_frames") == 3
             and inventory["counts"].get("curve_groups") == 26
             and inventory["counts"].get("source_uv") == 944,
             "Expected the reviewed explicit F1 opt-in inventory")
    _require(prepared.metadata["camera_id"] == "low_end"
             and prepared.metadata["source_video_sha256"]
             == inventory["source_video_sha256"],
             "F1 inventory differs from the prepared low-end rig")
    by_pts = {int(r["source_pts"]): r for r in inventory["references"]}
    prepared_by_pts = {int(r["source_pts"]): r for r in prepared.references}
    _require(set(by_pts) == set(prepared_by_pts) == {14177280, PTS, 226160640},
             "F1 inventory has a different reference exposure set")
    for pts, row in by_pts.items():
        old = prepared_by_pts[pts]
        _require(row["image_sha256"] == old["image_sha256"]
                 and row["camera_assignment_authority"]
                 == old["camera_assignment_authority"],
                 f"F1 inventory changed image or camera identity at {pts}")
        base_ids = [c["id"] for c in old["curves"]]
        _require([c["id"] for c in row["curves"][:len(base_ids)]] == base_ids
                 and (len(row["curves"]) == len(base_ids) + (3 if pts == PTS else 0)),
                 f"F1 inventory changed a primary source curve at {pts}")
        for original, opt in zip(old["curves"], row["curves"]):
            _require(np.array_equal(original["source_uv"], opt["source_uv"])
                     and np.array_equal(original["world_xyz"], opt["world_xyz"])
                     and original["source_file_sha256"] == opt["source_file_sha256"],
                     f"F1 opt-in changed primary observation {original['id']}")
    source = by_pts[PTS]
    additions = source["curves"][-3:]
    _require(tuple(c["id"] for c in additions) == IDS,
             "F1 rings lost their source order or identities")
    return prepared_by_pts[PTS], additions


def _curve_model(row: dict, *, tolerance_px: float) -> tuple[
        AdaptiveProjectedPolyline, np.ndarray, np.ndarray]:
    uv = np.asarray(row["source_uv"], np.float64)
    world = np.asarray(row["world_xyz"], np.float64)
    weight = np.asarray(row["point_integration_weights"], np.float64)
    _require(row["target"] == {"kind": "circle_outline", "feature_id": "F1",
                               "ring_role": row["target"].get("ring_role")}
             and row["target"]["ring_role"] in ROLES
             and row["physical_group_id"] == "F1"
             and row["source_correlation_group"] == f"low3_{PTS}:F1"
             and row["sigma_px"] == 1.
             and row["sigma_role"]
             == "historical_fit_joint_numerical_pixel_unit_not_accuracy_bound"
             and abs(float(row["correlated_curve_weight"]) - 1./3.) < 1e-12
             and weight.shape == (len(uv),)
             and np.isfinite(weight).all() and np.all(weight >= 0)
             and abs(float(weight.sum()) - 1./3.) < 1e-7
             and world.ndim == 2 and world.shape[1] == 3,
             f"F1 source correlation or physical ring changed: {row['id']}")
    measurements = [dict(x_interval_px=(float(x), float(x)), y=float(y))
                    for x, y in uv]
    curve = AdaptiveProjectedPolyline(
        world, measurements, tolerance_px=tolerance_px, closed=True,
        factor_family="original_low_end_user_F1_correlated",
        no_central_reason="user_F1_no_central_ring_segment")
    return curve, uv, weight


def correlated_f1_raw_callbacks(row: dict, template25: np.ndarray, *,
                                tolerance_px: float = .01):
    """Return raw physical15 residual/J plus the unweighted adaptive curve.

    The sign is the old Step02 source-minus-nearest convention; the shared
    rig adapter performs the full physical15-to-Vector11 chain afterward.
    """
    curve, uv, weight = _curve_model(row, tolerance_px=tolerance_px)
    _require(np.asarray(template25).shape == (25,), "F1 needs a Brown25 template")
    source_keys = (PTS,)
    base_raw, base_jac, prime = adaptive_curve_callback(
        curve, template25, source_keys, with_prime=True)
    scale = np.repeat(np.sqrt(weight), 2)

    def residual(states):
        value = np.asarray(base_raw(states), np.float64).reshape(-1)
        _require(value.shape == (2*len(uv),), "F1 adaptive residual changed size")
        return -scale * value

    def derivative(states):
        _require(len(states) == 1, "Correlated F1 expects one physical camera")
        physical = np.asarray(states[0], np.float64)
        _require(physical.shape == (15,),
                 "Correlated F1 requires the full physical15 rig camera")
        # The reviewed adaptive engine owns complete-factor implicit/J union-FD
        # selection. Its prime callback installs the unweighted exact-state
        # bundle computed by an existing independent-curve worker.
        selected = np.asarray(base_jac(states)[0], np.float64)
        _require(selected.shape == (2*len(uv), 15)
                 and np.isfinite(selected).all(), "F1 physical15 Jacobian invalid")
        return [-scale[:, None] * selected]

    return residual, derivative, curve, prime


def attach_correlated_f1_factors(graph, data: dict, prepared, inventory: dict,
                                 *, tolerance_px: float = .01) -> dict:
    """Append only the three optional F1 factors; no solver or primary edits."""
    _require(data["camera_id"] == "low_end"
             and np.isfinite(tolerance_px) and tolerance_px > 0,
             "F1 factors need a low-end rig and positive numerical tolerance")
    reference, additions = _f1_rows(inventory, prepared)
    model = prepared.model
    _require(PTS in model.frames
             and model.frames[PTS].camera_id == "low_end"
             and "low_end" in model.brown_keys
             and np.asarray(model.physical_state(PTS, prepared.initial)).shape == (15,),
             "F1 factor must use the movable-C, adjustable-K/Brown physical15 rig")
    active = set(prepared.initial)
    if not hasattr(graph, "_adaptive_curve_jobs"):
        graph._adaptive_curve_jobs = []
    registered = []
    for row in additions:
        template = np.asarray(model.frames[PTS].template25, np.float64)
        raw, jac, curve, prime = correlated_f1_raw_callbacks(
            row, template, tolerance_px=tolerance_px)
        adapted = adapt_factor(
            (PTS,), raw, jac,
            dict(residual_dim=2*len(row["source_uv"]), sigma=1., huber_k=None),
            active, data["states"], model)
        _require(adapted is not None, f"F1 graph factor became constant: {row['id']}")
        keys, residual, derivative, options = adapted

        def checked_residual(q, *, fn=residual):
            try:
                return fn(q)
            except BoardIntersectionError as exc:
                exc.evaluation_stage = "residual"
                exc.factor_family = "original_low_end_user_F1_correlated"
                exc.source_keys = (PTS,)
                raise

        def checked_derivative(q, *, fn=derivative):
            try:
                return fn(q)
            except BoardIntersectionError as exc:
                exc.evaluation_stage = "jacobian"
                exc.factor_family = "original_low_end_user_F1_correlated"
                exc.source_keys = (PTS,)
                raise

        graph.add_observation(keys, checked_residual,
                              jac_fn=checked_derivative, **options)
        graph._adaptive_curve_jobs.append(AdaptiveCurveJob(
            curve, template.copy(), model, PTS, (PTS,), prime))
        registered.append(dict(id=row["id"], ring_role=row["target"]["ring_role"],
                               factor_keys=list(keys), source_uv=len(row["source_uv"]),
                               full_original_stroke_uv=len(row["original_stroke_uv"]),
                               point_weight_sum=float(row["point_integration_weights"].sum()),
                               source_file_sha256=row["source_file_sha256"],
                               original_stroke_sha256=row["original_stroke_sha256"]))
    _require(len(registered) == 3
             and abs(sum(item["point_weight_sum"] for item in registered) - 1.) < 1e-7,
             "F1 physical group lost its total unit budget")
    return dict(
        schema="rinkspace.low_end.user_F1_correlated_optin.v1",
        source_pts=PTS, source_video_sha256=inventory["source_video_sha256"],
        source_set_sha256=inventory["source_set_sha256"],
        original_step02_sha256=additions[0]["source_file_sha256"],
        original_user_stroke_sha256=additions[0]["original_stroke_sha256"],
        numerical_sigma_px=1., numerical_sigma_is_measured_accuracy=False,
        objective="sqrt(original_step02_point_integration_weight)*"
                  "(source_uv-nearest_central_projected_current_scene_F1_ring)",
        source_uv_modified=False, ring_count=3,
        total_fitted_source_uv=sum(item["source_uv"] for item in registered),
        total_original_user_stroke_uv=sum(item["full_original_stroke_uv"]
                                          for item in registered),
        physical_group_budget=sum(item["point_weight_sum"] for item in registered),
        adaptive_curve_prime_jobs_registered=len(registered),
        physical15_jacobian_backend="existing_adaptive_curve_engine",
        implicit_policy="existing_global_complete_factor_guard_when_enabled",
        full_factor_fallback="existing_original_union_topology_physical15_FD",
        tolerance_px=tolerance_px,
        factor_rows=registered,
        module_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )


def runtime_counters(graph) -> dict:
    """Return global adaptive-engine counts; these are not F1-only counts."""
    _require(len(getattr(graph, "_adaptive_curve_jobs", ())) >= 3,
             "F1 prime jobs have not been registered")
    return dict(scope="all_adaptive_curves_in_process_not_F1_only",
                counts=adaptive_implicit_optin_metadata()["counts"])
