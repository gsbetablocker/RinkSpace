"""Opt-in source-normal intervals on the physical lower long-side board top.

Only source-reviewed rows may reach this adapter. The camera model, world
polyline and source measurements remain separate: pixels constrain distance
to a physical curve, never an invented along-board world point. No solver runs
here. The native/source graph is retained and the added curve jobs reuse the
audited adaptive projection and its parallel Jacobian path.
"""
from __future__ import annotations

from runtime.bootstrap import add_geometry_dependencies
add_geometry_dependencies()

import numpy as np

from adaptive_curve_parallel import AdaptiveCurveJob
from continuous_brown_adapter import adapt_factor
from robust_observations import pointwise_huber
from secondary_nearboard_adaptive_curve import (
    AdaptiveProjectedPolyline, adaptive_curve_callback)
from secondary_nearboard_top_source import lower_long_side_top
from source_edge_intervals import SourceNormalIntervals


EDGE_SIGMA_PX = 4.0  # Outside-support likelihood, not measured pixel covariance.
FAMILY = "source_reviewed_main_lower_board_top"


def _source_support(normals, bounds):
    normal = np.asarray(normals, np.float64)
    if normal.ndim != 2 or normal.shape[1] != 2:
        raise ValueError("Source normal must be an Nx2 array")
    lengths = np.linalg.norm(normal, axis=1)
    if np.any(~np.isfinite(lengths)) or np.any(np.abs(lengths-1.) > 1e-5):
        raise ValueError("Source normal is not unit length")
    # The source packet stores float32 normals. Restore unit length after
    # loading, correcting only storage rounding; source UV/bounds stay intact.
    return SourceNormalIntervals(normal/lengths[:, None], bounds,
                                 np.ones(len(normal), dtype=bool))


def source_interval_callback(raw, jac, normals, bounds):
    """Apply frozen source support after a physical nearest-curve evaluation."""
    support = _source_support(normals, bounds)

    def residual(states):
        return support.apply(raw(states))

    def derivative(states):
        return support.apply(raw(states), jac(states))[1]

    return residual, derivative


def attach_main_nearboard_normal_factors(graph, data, model, initial, rows):
    """Append reviewed same-exposure rows, with one correlated weight per frame.

    Each row binds representative_pts, reviewed_native_pts, source_sha256,
    observed_uv, outward_normal and normal_interval_px. Identity and source
    qualification must be explicitly reviewed before graph attachment.
    """
    if data["camera_id"] != "main_long_side":
        raise ValueError("This source family belongs to the complete main rig")
    world, geometry = lower_long_side_top(data["scene"])
    prepared = []
    seen = set()
    for row in rows:
        pts = int(row["representative_pts"])
        native = tuple(map(int, row["reviewed_native_pts"]))
        uv = np.asarray(row["observed_uv"], np.float64)
        normals = np.asarray(row["outward_normal"], np.float64)
        bounds = np.asarray(row["normal_interval_px"], np.float64)
        if (pts not in model.frames or pts in seen or not native
                or any(p not in model.frames or model.frames[p].frame_key != pts
                       for p in native)
                or row.get("source_review_status") != "reviewed_bluecap_ice_boundary"
                or row.get("physical_identity") != "lower_long_side_board_top"
                or row.get("source_sha256") != data["diag"]["source_sha256"]
                or uv.ndim != 2 or uv.shape[1] != 2 or len(uv) < 2
                or not np.isfinite(uv).all()):
            raise ValueError("Near-board row is unreviewed or source/alias mismatched")
        support = _source_support(normals, bounds)
        if len(support.eligible) != len(uv):
            raise ValueError("Source normals do not match original pixel rows")
        measurements = [dict(x_interval_px=[float(u), float(u)], y=float(v))
                        for u, v in uv]
        curve = AdaptiveProjectedPolyline(world, measurements,
                                          geometric_chord_error=True,
                                          independent_fd_topology=True,
                                          factor_family=FAMILY)
        prepared.append((pts, native, curve, support))
        seen.add(pts)

    active = set(map(int, initial))
    jobs = getattr(graph, "_adaptive_curve_jobs", None)
    if jobs is None:
        graph._adaptive_curve_jobs = jobs = []
    for pts, native, curve, support in prepared:
        template = model.frames[pts].template25
        raw, jac, prime = adaptive_curve_callback(curve, template, native,
                                                 with_prime=True)
        raw, jac = source_interval_callback(raw, jac, support.normals,
                                            support.bounds)
        raw, jac = pointwise_huber(raw, jac, group_size=2,
                                   delta=2.0 * EDGE_SIGMA_PX)
        keys, residual, derivative, options = adapt_factor(
            (pts,), raw, jac,
            dict(residual_dim=2 * len(curve.observations),
                 sigma=EDGE_SIGMA_PX * np.sqrt(len(curve.observations))),
            active, data["states"], model)
        graph.add_observation(keys, residual, jac_fn=derivative, **options)
        jobs.append(AdaptiveCurveJob(curve, np.asarray(template).copy(), model,
                                     pts, native, prime))
    return dict(factor_family=FAMILY, source_exposure_factors=len(prepared),
                physical_curve=geometry, source_uv_modified=False,
                scene_modified=False, existing_factors_modified=False,
                source_interval_role="measured blur support along source normal",
                likelihood_sigma_outside_support_px=EDGE_SIGMA_PX,
                correlation_policy="one total curve weight per confirmed exposure",
                solver_started=False)
