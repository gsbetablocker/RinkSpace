"""Eliminate a computational foot station from a local finite-curve model.

The caller authenticates the physical primitive and supplies its true station
bounds. Those bounds are never inferred from a drawing endpoint or a sampled
projection cell. The source rectangle and violation limit are used verbatim.
This module supplies a local step model; the original nonlinear query remains
the authority for a trial's physical/source feasibility.
"""
from __future__ import annotations

import numpy as np


def finite_rectangle_station_halfspaces(
    projected_foot_xy,
    fixed_foot_jacobian,
    station_screen_tangent,
    nominal_physical_station,
    physical_station_bounds,
    source_uv,
    source_normals,
    normal_bounds,
    tangent_bounds,
    violation_limits,
    *, station_tangent_cone=None,
):
    """Return ``matrix @ delta_q >= lower`` in the caller's original chart.

    For each source point, the local projected point is
    ``foot + J_fixed_foot @ delta_q + screen_tangent * delta_station``.
    The station is free within the supplied *physical* interval, including
    release from a currently clamped endpoint. Fourier--Motzkin elimination
    removes this computational auxiliary without adding chart parameters.

    Shapes are N x 2 for XY, normals and bounds; N x 2 x P for the Jacobian;
    and N for stations and limits. Inputs are not broadcast, normalized,
    clipped, weighted, or modified. The normal/tangent axes must be the
    original axes used to express the caller's source support.

    Zero rows and every eliminated face are retained with provenance. The
    caller may apply its existing, independently checked zero-row policy.
    No source sigma, tolerance, physical observation or authority is added.
    """
    foot = np.asarray(projected_foot_xy, dtype=float)
    if foot.ndim != 2 or foot.shape[1] != 2:
        raise ValueError("Explicit N x 2 projected feet required")
    count = len(foot)
    jacobian = np.asarray(fixed_foot_jacobian, dtype=float)
    if jacobian.ndim != 3 or jacobian.shape[:2] != (count, 2):
        raise ValueError("Explicit N x 2 x original-chart-column Jacobian required")
    columns = jacobian.shape[2]
    vector = np.asarray(station_screen_tangent, dtype=float)
    station = np.asarray(nominal_physical_station, dtype=float)
    domain = np.asarray(physical_station_bounds, dtype=float)
    uv = np.asarray(source_uv, dtype=float)
    normals = np.asarray(source_normals, dtype=float)
    nb = np.asarray(normal_bounds, dtype=float)
    tb = np.asarray(tangent_bounds, dtype=float)
    limits = np.asarray(violation_limits, dtype=float)
    shaped = (vector, domain, uv, normals, nb, tb)
    if any(value.shape != (count, 2) for value in shaped):
        raise ValueError("Original per-point XY, axes and two-sided support shapes required")
    if station.shape != (count,) or limits.shape != (count,):
        raise ValueError("Explicit per-point nominal stations and original limits required")
    if not all(np.isfinite(value).all() for value in (foot, jacobian, *shaped, station, limits)):
        raise ValueError("Finite original local-model inputs required")
    if np.any(domain[:, 0] >= domain[:, 1]):
        raise ValueError("Nondegenerate authenticated physical station interval required")
    if np.any(station < domain[:, 0]) or np.any(station > domain[:, 1]):
        raise ValueError("Nominal foot station must lie in the authenticated physical interval")
    if np.any(nb[:, 0] > nb[:, 1]) or np.any(tb[:, 0] > tb[:, 1]):
        raise ValueError("Ordered original normal/tangent source support required")
    if np.any(limits < 0) or np.any(~np.any(normals != 0, axis=1)):
        raise ValueError("Nonnegative original violation limits and nonzero source normals required")
    cone=None if station_tangent_cone is None else np.asarray(station_tangent_cone,float)
    if cone is not None:
        if (cone.shape!=(count,2)or np.isnan(cone).any()
            or not np.isin(cone[:,0],[-np.inf,0.]).all()
            or not np.isin(cone[:,1],[0.,np.inf]).all()
            or np.any((cone[:,0]==0.)&(cone[:,1]==0.))):
            raise ValueError('Original local station tangent cone must be free or one-sided at a true physical endpoint')
        if np.any((cone[:,0]==0.)&(station!=domain[:,0]))or np.any((cone[:,1]==0.)&(station!=domain[:,1])):
            raise ValueError('One-sided station tangent requires the nominal true global physical endpoint')

    tangents = np.c_[-normals[:, 1], normals[:, 0]]
    axes = np.stack((normals, tangents), axis=1)
    origin = np.einsum("nai,ni->na", axes, foot - uv)
    station_slopes = np.einsum("nai,ni->na", axes, vector)
    gradients = np.einsum("nai,nip->nap", axes, jacobian)
    support = np.stack((nb, tb), axis=1)
    matrices, targets, provenance = [], [], []

    def append(point, value, gradient, kind, parents, **details):
        # epigraph face: value + gradient @ delta_q <= original limit.
        matrices.append(-np.asarray(gradient, dtype=float))
        targets.append(float(value - limits[point]))
        provenance.append(dict(
            source_point_index=int(point), kind=kind,
            original_source_faces=parents,
            local_epigraph_face_at_nominal=float(value),
            original_violation_limit=float(limits[point]),
            nominal_physical_station=float(station[point]),
            physical_station_bounds=domain[point].tolist(),
            **details,
        ))

    for point in range(count):
        faces = []
        for axis, name in enumerate(("normal", "tangent")):
            m = station_slopes[point, axis]
            x = origin[point, axis]
            g = gradients[point, axis]
            faces.extend((
                (-m, support[point, axis, 0] - x, -g, dict(axis=name, side="lower")),
                (m, x - support[point, axis, 1], g, dict(axis=name, side="upper")),
            ))
        lower_station = domain[point, 0] - station[point] if cone is None else cone[point,0]
        upper_station = domain[point, 1] - station[point] if cone is None else cone[point,1]
        append(point, 0., np.zeros(columns), "nonnegative_epigraph", [])
        for m, c, g, parent in faces:
            if m == 0.:
                append(point, c, g, "station_independent_source_face", [parent])
            elif m > 0. and np.isfinite(lower_station):
                append(point, c + m * lower_station, g,
                       "physical_lower_station_bound"if cone is None else"physical_lower_station_tangent_cone", [parent])
            elif m < 0. and np.isfinite(upper_station):
                append(point, c + m * upper_station, g,
                       "physical_upper_station_bound"if cone is None else"physical_upper_station_tangent_cone", [parent])
        for mp, cp, gp, positive_parent in faces:
            if mp <= 0.:
                continue
            for mn, cn, gn, negative_parent in faces:
                if mn >= 0.:
                    continue
                # Convex weights avoid products of two potentially large
                # pixel-coordinate quantities; no weak face is dropped.
                scale = max(mp, -mn)
                positive_weight = (-mn / scale) / ((mp / scale) + (-mn / scale))
                negative_weight = (mp / scale) / ((mp / scale) + (-mn / scale))
                append(point,
                       positive_weight * cp + negative_weight * cn,
                       positive_weight * gp + negative_weight * gn,
                       "interior_station_feasibility",
                       [positive_parent, negative_parent],
                       source_face_weights=[float(positive_weight), float(negative_weight)])

    matrix = np.asarray(matrices, dtype=float).reshape(len(matrices), columns)
    lower = np.asarray(targets, dtype=float)
    if not np.isfinite(matrix).all() or not np.isfinite(lower).all():
        raise ValueError("Finite eliminated original-unit faces required")
    return dict(
        matrix=matrix, lower=lower, face_provenance=provenance,
        source_point_indices=np.asarray([item["source_point_index"] for item in provenance], dtype=int),
        contract=dict(
            schema="rinkspace.finite-rectangle-physical-station-local-step-faces.v1",
            inequalities="matrix @ delta_q >= lower",
            original_chart_column_count=int(columns),
            source_point_count=int(count),
            computational_station_eliminated=True,
            physical_station_bounds_supplied_by_caller=True,
            local_station_tangent_cone_used=cone is not None,
            interior_station_and_sampling_knots_have_no_distant_cap_faces=cone is not None,
            true_clamped_physical_endpoint_allows_only_inward_station_release=cone is not None,
            projection_sampling_knots_not_physical_caps=True,
            source_drawing_endpoints_not_physical_correspondences=True,
            endpoint_release_allowed=True,
            source_support_and_violation_limits_unchanged=True,
            new_physical_or_source_parameters=0,
            local_step_model_not_nonlinear_truth_replacement=True,
            original_nonlinear_guard_required=True,
            zero_and_weak_faces_not_threshold_filtered=True,
        ),
    )
