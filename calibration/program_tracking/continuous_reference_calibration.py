"""Prepare source-backed reference cameras for one complete v2 rig group.

This module does not run a solver. Fourteen same-event v0.25 reference images
provide 57 measured image curves and only two directly read ad-3 corner sets.
Their saved camera25 values initialize separate reference R/logf and one static
centre-residual state per exposure; no old camera value becomes a factor target.
All world curves/corners are rebound by physical target to the *current*
programme scene. Each reference model shares the programme group's P, K,
optical offsets and Brown4 nodes, and has no invented temporal links.
Curve UV samples use one conservative image-target group weight; this is an
explicit noise assumption rather than a measured pixel covariance. Invalid
trial projections raise the solver's domain error and never drop source rows.

Prepare the augmented initial state *before* calling ``base.build_graph`` so
its fixed-lag window is large enough. Build the programme graph with the base
model and its unchanged prior targets, then call ``attach_reference_factors``
with the separate reference model. Merge v2 warm states into the programme
keys only; ``seed_reference_initial`` may subsequently refresh the new nodes.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
import json
from typing import Mapping

import cv2
import numpy as np

try:
    from .continuous_camera_model import FrameBinding
    from .continuous_rig_camera import ContinuousRigCameraModel
    from .continuous_brown_adapter import adapt_factor
    from .projection import from_state, project_central_valid, require_central_projection
    from .robust_observations import pointwise_huber
    from .anchors.geometry import world_curve
    from .reference_calibration_source_loader import load_inputs
    from .reference_board_pair import BoardIntersectionError
    from .secondary_nearboard_adaptive_curve import (
        AdaptiveProjectedPolyline, adaptive_curve_callback)
    from .adaptive_curve_parallel import AdaptiveCurveJob
except ImportError:
    from continuous_camera_model import FrameBinding
    from continuous_rig_camera import ContinuousRigCameraModel
    from continuous_brown_adapter import adapt_factor
    from projection import from_state, project_central_valid, require_central_projection
    from robust_observations import pointwise_huber
    from reference_calibration_source_loader import load_inputs
    from three_frame_refit.program_tracking.anchors.geometry import world_curve
    from reference_board_pair import BoardIntersectionError
    from secondary_nearboard_adaptive_curve import (
        AdaptiveProjectedPolyline, adaptive_curve_callback)
    from adaptive_curve_parallel import AdaptiveCurveJob


CORNER_ORDER = ("left_top", "right_top", "right_bottom", "left_bottom")


@dataclass(frozen=True)
class StaticReferenceBasis:
    """One source exposure's C residual; exactly one basis weight, no time axis."""

    shot_id: int
    pts: int
    control_keys: tuple[int, ...]

    def basis_pairs(self, pts: float) -> tuple[tuple[int, float], ...]:
        if float(pts) != float(self.pts):
            raise ValueError("Reference centre residual has one supported source PTS")
        return ((self.control_keys[0], 1.0),)

    def weights(self, pts: float) -> dict[int, float]:
        return dict(self.basis_pairs(pts))


@dataclass
class ReferenceCalibration:
    model: ContinuousRigCameraModel
    initial: dict[int, np.ndarray]
    references: list[dict]
    metadata: dict


def _check(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def _native_camera(physical: np.ndarray, template25: np.ndarray) -> np.ndarray:
    physical = np.asarray(physical, np.float64)
    _check(physical.shape == (15,) and np.isfinite(physical).all(),
           "Reference callback needs the shared Brown4 physical state")
    return from_state(physical, template25)


def _curve_residual(camera25: np.ndarray, xyz: np.ndarray,
                    observed_uv: np.ndarray) -> np.ndarray:
    """Closest projected physical curve; no invented pointwise XYZ identity."""
    try:
        projected, _, valid = project_central_valid(camera25, xyz)
        _, _, midpoint_valid = project_central_valid(
            camera25, (xyz[:-1] + xyz[1:]) * .5)
    except cv2.error as exc:
        raise BoardIntersectionError("reference_curve_projection_cv2_domain",
                                     np.arange(len(xyz))) from exc
    segment_indices = np.flatnonzero(valid[:-1] & midpoint_valid & valid[1:])
    if not len(segment_indices):
        raise BoardIntersectionError("reference_curve_no_central_segment",
                                     np.arange(len(xyz)-1))
    a, b = projected[segment_indices], projected[segment_indices+1]
    segment = b-a
    denominator = np.maximum(np.sum(segment*segment, axis=1), 1e-9)
    delta = observed_uv[:, None, :]-a[None, :, :]
    fraction = np.clip(np.sum(delta*segment[None, :, :], axis=2)
                       / denominator[None, :], 0.0, 1.0)
    nearest = a[None, :, :]+fraction[:, :, None]*segment[None, :, :]
    squared = np.sum((nearest-observed_uv[:, None, :])**2, axis=2)
    winner = np.argmin(squared, axis=1)
    residual = (nearest[np.arange(len(observed_uv)), winner]-observed_uv).reshape(-1)
    if not np.isfinite(residual).all():
        raise BoardIntersectionError("reference_curve_residual_nonfinite",
                                     np.flatnonzero(~np.isfinite(residual.reshape(-1, 2)).all(axis=1)))
    return residual


def _physical_fd(raw, state: np.ndarray, step: float = 1e-6) -> np.ndarray:
    """Piecewise curve/corner derivative in physical15 before rig chaining."""
    state = np.asarray(state, np.float64)
    baseline = np.asarray(raw(state), np.float64).reshape(-1)
    jac = np.empty((len(baseline), 15), np.float64)
    for col in range(15):
        width = step*max(1.0, abs(float(state[col])))
        high, low = state.copy(), state.copy()
        high[col] += width
        low[col] -= width
        try:
            jac[:, col] = (raw(high)-raw(low))/(2*width)
        except BoardIntersectionError as exc:
            exc.evaluation_stage = "jacobian"
            raise
    _check(np.isfinite(jac).all(), "Nonfinite source-reference physical Jacobian")
    return jac


def _measurement_callback(kind: str, xyz: np.ndarray,
                          observed_uv: np.ndarray, template25: np.ndarray,
                          source_keys: tuple[int, ...] = ()):
    xyz = np.asarray(xyz, np.float64).reshape(-1, 3)
    observed_uv = np.asarray(observed_uv, np.float64).reshape(-1, 2)
    _check(len(xyz) >= (2 if kind == "curve" else 4)
           and len(observed_uv) > 0 and np.isfinite(xyz).all()
           and np.isfinite(observed_uv).all(), "Invalid reference source factor")

    def one(state):
        camera = _native_camera(state, template25)
        if kind == "curve":
            return _curve_residual(camera, xyz, observed_uv)
        uv, _ = require_central_projection(
            camera, xyz, stage="residual", family="original_reference_direct_ad3",
            source_keys=source_keys)
        return (uv-observed_uv).reshape(-1)

    return (lambda states: one(states[0]),
            lambda states: [_physical_fd(one, states[0])])


def _source_scene_rows(inventory: dict, current_scene: dict,
                       camera_id: str) -> list[dict]:
    ads = [ad for ad in current_scene["advertisements"]
           if ad["user_number"] == 3]
    _check(len(ads) == 1 and ads[0]["board_segment"] == "upper_long_side",
           "Current scene has no unique upper-long-side ad #3")
    corners = np.asarray([ads[0]["corners_xyz_scene_units"][role]
                          for role in CORNER_ORDER], np.float64)
    rows = []
    for ref in inventory["references"]:
        if ref["camera_id"] != camera_id:
            continue
        curves = []
        for curve in ref["curves"]:
            xyz = world_curve(current_scene, curve["target"])
            _check(xyz is not None and len(xyz) >= 2,
                   f"Current scene has no curve target: {curve['id']}")
            curves.append(dict(id=curve["id"], target=curve["target"],
                               physical_group_id=curve["physical_group_id"],
                               identity_status=curve["identity_status"],
                               source_uv=np.asarray(curve["source_uv"], np.float64),
                               world_xyz=np.asarray(xyz, np.float64),
                               sigma_px=float(curve["sigma_px"])))
        ad = ref["ad3"]
        rows.append(dict(id=ref["id"], source_pts=int(ref["source_pts"]),
                         camera_id=camera_id, image_path=ref["image_path"],
                         image_sha256=ref["image_sha256"],
                         raw_camera_id=ref["raw_camera_id"],
                         installation_id=ref["installation_id"],
                         camera_assignment_authority=ref["camera_assignment_authority"],
                         source_observation_file=ref["source_observation_file"],
                         source_observation_sha256=ref["source_observation_sha256"],
                         camera25_initializer=np.asarray(ref["initial_camera25"], np.float64),
                         curves=curves,
                         direct_ad3_uv=(np.asarray(ad["direct_source_uv"], np.float64)
                                        if ad["direct_source_uv"] is not None else None),
                         ad3_world_xyz=corners,
                         ad3_sigma_px=float(ad["sigma_px"]),
                         derived_ad3_excluded=ad["transferred_uv_derived"] is not None))
    return rows


def prepare_reference_calibration(data: dict, base_model: ContinuousRigCameraModel,
                                  programme_initial: Mapping[int, np.ndarray],
                                  *, inventory: dict | None = None,
                                  reserved_keys: set[int] | None = None
                                  ) -> ReferenceCalibration:
    """Create separate reference frame/C nodes sharing the v2 group optics."""
    inventory = load_inputs() if inventory is None else inventory
    camera_id = data["camera_id"]
    _check(camera_id in ("main_long_side", "secondary_long_side"),
           "The 14 source references contain no low_end reference group")
    _check(inventory["source_video_sha256"] == data["captured"]["source_hash"]
           == data["diag"]["source_sha256"], "Reference/source video differs")
    _check(inventory["scene_unit"] == data["scene"]["coordinate_system"]["unit"]
           == "conditional_scene_units"
           and data["scene"]["coordinate_system"]["verified_metre_scale"] is False,
           "Reference and programme coordinate-unit semantics differ")
    _check(set(base_model.lens_base_keys) == {camera_id}
           and set(base_model.mount_keys) == {camera_id}
           and set(base_model.brown_keys) == {camera_id},
           "Reference model requires one complete shared-Brown v2 camera group")
    rows = _source_scene_rows(inventory, data["scene"], camera_id)
    _check(len(rows) == (11 if camera_id == "main_long_side" else 3),
           "Reference bank does not cover this complete camera group")
    programme_pts = set(base_model.frames)
    _check(not programme_pts.intersection(r["source_pts"] for r in rows),
           "Reference source PTS collides with target programme PTS")
    reserved = set(map(int, reserved_keys or set())) | set(map(int, programme_initial))
    reserved |= set(map(int, data["captured"]["guesses"]))
    _check(not set(r["source_pts"] for r in rows).intersection(
        map(int, data["captured"]["guesses"])),
        "An added reference PTS already owns an original graph state")
    next_key = max(reserved | programme_pts |
                   {r["source_pts"] for r in rows} | {0})+1
    splines, bindings = {}, {}
    augmented = {int(k): np.asarray(v, np.float64).copy()
                 for k, v in programme_initial.items()}
    for ordinal, row in enumerate(rows):
        pts = row["source_pts"]
        frame_key, control_key = next_key, next_key+1
        next_key += 2
        shot_id = 1000+ordinal
        splines[shot_id] = StaticReferenceBasis(shot_id, pts, (control_key,))
        bindings[pts] = FrameBinding(pts, frame_key, shot_id, camera_id,
                                     row["camera25_initializer"].copy())
        row["frame_key"], row["control_key"] = frame_key, control_key
        augmented[frame_key] = np.zeros(11)
        augmented[control_key] = np.zeros(11)
    reference_model = ContinuousRigCameraModel(
        splines, bindings, base_model.lens_base_keys,
        base_model.lens_slope_keys, base_model.lens_zoom_reference,
        mount_keys=base_model.mount_keys, offset_keys=base_model.offset_keys,
        offset_slope_keys=base_model.offset_slope_keys,
        brown_keys=base_model.brown_keys,
        brown_slope_keys=base_model.brown_slope_keys)
    seed_reference_initial(reference_model, augmented, rows)
    metadata = dict(
        camera_id=camera_id, reference_frame_count=len(rows),
        source_curve_group_count=sum(len(r["curves"]) for r in rows),
        source_curve_pixel_count=sum(len(c["source_uv"])
                                     for r in rows for c in r["curves"]),
        direct_ad3_frame_count=sum(r["direct_ad3_uv"] is not None for r in rows),
        derived_ad3_frame_count_excluded=sum(r["derived_ad3_excluded"] for r in rows),
        source_video_sha256=inventory["source_video_sha256"],
        source_reference_scene_sha256=inventory["scene_sha256"],
        current_programme_scene_sha256=data["diag"]["model_sha256"],
        geometry_binding="source UV and target identities only; re-evaluate world curve/ad3 in data.scene",
        curve_noise_policy=(
            "sigma_factor=sigma_px*sqrt(n_uv*same_image_target_curve_count); "
            "Huber threshold=2*sigma_px per UV pair. Conservative group-weight "
            "assumption, not a calibrated covariance matrix"),
        same_camera_installation="historical camera classification; shared-P is a conditional mechanical hypothesis",
        initializer_role="old reference camera25 only sets initial R/logf/C residual; no target uses it",
        reference_pose_prior="none; no old camera25 component enters objective",
        reference_c_residual_prior_sigma_scene=.25,
        temporal_reference_links=0,
        old_atlas_xyz_used=False,
        reference_pts=[r["source_pts"] for r in rows],
        reference_image_sha256={str(r["source_pts"]): r["image_sha256"] for r in rows},
        reference_sources={str(r["source_pts"]): dict(
            image_path=r["image_path"], image_sha256=r["image_sha256"],
            raw_camera_id=r["raw_camera_id"],
            installation_id=r["installation_id"],
            camera_assignment_authority=r["camera_assignment_authority"],
            observation_file=r["source_observation_file"],
            observation_sha256=r["source_observation_sha256"],
            curve_targets=[c["target"] for c in r["curves"]],
            curve_identity_statuses=[c["identity_status"] for c in r["curves"]],
            direct_ad3=r["direct_ad3_uv"] is not None,
            derived_ad3_excluded=r["derived_ad3_excluded"])
            for r in rows},
        new_graph_nodes={str(r["source_pts"]): dict(frame_key=r["frame_key"],
                                                    c_residual_key=r["control_key"])
                         for r in rows},
    )
    return ReferenceCalibration(reference_model, augmented, rows, metadata)


def seed_reference_initial(model: ContinuousRigCameraModel,
                           states: dict[int, np.ndarray], rows: list[dict]) -> None:
    """Refresh new-node starts after warm programme states; no factor changes."""
    for row in rows:
        old = row["camera25_initializer"]
        frame_key, control_key = row["frame_key"], row["control_key"]
        frame = np.zeros(11)
        frame[3:6] = old[3:6]
        frame[6] = float(np.mean(old[6:8]))
        states[frame_key] = frame
        states[control_key] = np.zeros(11)
        without_residual = model.camera25(row["source_pts"], states)
        states[control_key][:3] = (old[:3]-without_residual[:3])/100.


def attach_reference_factors(graph, data: dict,
                             prepared: ReferenceCalibration, *,
                             adaptive_curves: bool = False,
                             adaptive_curve_tolerance_px: float = .01) -> dict:
    """Add original-image factors and explicit static-C model regularization.

    Call only after the base programme graph was built using the same augmented
    initial mapping; ``window_size`` must include every new node. No solve or
    camera export occurs in this helper.
    """
    if getattr(graph, "window_size", 0) < len(prepared.initial)+2:
        raise ValueError("Build base graph with augmented states before attaching references")
    model = prepared.model
    active = set(prepared.initial)
    if data["camera_id"] != prepared.metadata["camera_id"]:
        raise ValueError("Reference and programme camera groups differ")
    if adaptive_curves and not hasattr(graph, "_adaptive_curve_jobs"):
        graph._adaptive_curve_jobs = []
    factors = []
    repeated_target_groups = 0
    for row in prepared.references:
        frame_key, control_key, pts = (row["frame_key"], row["control_key"],
                                       row["source_pts"])
        # Dummy constraints anchor unused Vector11 coordinates, exactly as v2.
        for key, live in ((frame_key, (3, 4, 5, 6)),
                          (control_key, (0, 1, 2))):
            unused = [i for i in range(11) if i not in live]
            select = np.eye(11)[unused]
            graph.add_observation([key], lambda q, s=select: s @ q[0],
                                  jac_fn=lambda q, s=select: [s],
                                  residual_dim=len(unused), sigma=1.)
        # Zero residual is an explicit mechanical regularizer, not old-C data.
        select = np.eye(11)[:3]
        graph.add_observation([control_key], lambda q, s=select: s @ q[0],
                              jac_fn=lambda q, s=select: [s],
                              residual_dim=3, sigma=np.full(3, .25/100.))
        # Samples on one traced source curve, and repeated traces of the same
        # physical target in one image, are not independent measurements.
        target_counts = Counter(json.dumps(curve["target"], sort_keys=True)
                                for curve in row["curves"])
        repeated_target_groups += sum(n > 1 for n in target_counts.values())
        for curve in row["curves"]:
            target_count = target_counts[json.dumps(curve["target"], sort_keys=True)]
            factors.append((pts, "curve", curve["id"], curve["world_xyz"],
                            curve["source_uv"], curve["sigma_px"], target_count,
                            curve["target"]))
        if row["direct_ad3_uv"] is not None:
            factors.append((pts, "direct_ad3", "ad3_direct_corners",
                            row["ad3_world_xyz"], row["direct_ad3_uv"],
                            row["ad3_sigma_px"], 1, None))
    for pts, kind, name, xyz, uv, sigma, target_count, target in factors:
        template = model.frames[pts].template25
        if adaptive_curves and kind == "curve":
            measurements = [dict(x_interval_px=(float(x), float(x)), y=float(y))
                            for x, y in uv]
            curve_model = AdaptiveProjectedPolyline(
                xyz, measurements, tolerance_px=adaptive_curve_tolerance_px,
                closed=target["kind"] == "circle_outline",
                factor_family="original_reference_curve",
                no_central_reason="reference_curve_no_central_segment")
            raw, jac, prime = adaptive_curve_callback(
                curve_model, template, (pts,), with_prime=True)
        else:
            raw, jac = _measurement_callback(
                "curve" if kind == "curve" else "ad", xyz, uv, template, (pts,))
        raw, jac = pointwise_huber(raw, jac, group_size=2, delta=2.*sigma)
        factor_sigma = (sigma*np.sqrt(len(uv)*target_count)
                        if kind == "curve" else sigma)
        adapted = adapt_factor((pts,), raw, jac,
                               dict(residual_dim=2*len(uv), sigma=factor_sigma),
                               active, data["states"], model)
        _check(adapted is not None, f"Reference factor became constant: {pts}:{name}")
        keys, residual, derivative, options = adapted
        def checked_residual(q, *, fn=residual, stamp=pts, family=kind):
            try:
                return fn(q)
            except BoardIntersectionError as exc:
                exc.evaluation_stage = "residual"
                exc.factor_family = "original_reference_"+family
                exc.source_keys = (stamp,)
                raise
        def checked_derivative(q, *, fn=derivative, stamp=pts, family=kind):
            try:
                return fn(q)
            except BoardIntersectionError as exc:
                exc.evaluation_stage = "jacobian"
                exc.factor_family = "original_reference_"+family
                exc.source_keys = (stamp,)
                raise
        graph.add_observation(keys, checked_residual,
                              jac_fn=checked_derivative, **options)
        if adaptive_curves and kind == "curve":
            graph._adaptive_curve_jobs.append(AdaptiveCurveJob(
                curve_model, np.asarray(template, np.float64).copy(),
                model, int(pts), (int(pts),), prime))
    return dict(prepared.metadata,
                source_measurement_factors=len(factors),
                static_c_prior_factors=len(prepared.references),
                unused_coordinate_factors=2*len(prepared.references),
                repeated_same_image_target_groups=repeated_target_groups,
                adaptive_curve_factor_count=(sum(kind == "curve" for _, kind, *_ in factors)
                                             if adaptive_curves else 0),
                adaptive_curve_tolerance_px=(adaptive_curve_tolerance_px
                                             if adaptive_curves else None),
                adaptive_circle_closure="by circle_outline target identity" if adaptive_curves else None,
                source_uv_modified=False,
                transferred_ad3_factor_count=0,
                programme_model_factors_modified=False,
                solve_started=False)
