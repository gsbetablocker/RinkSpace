"""Exact finite human-query reductions using the existing strict nearest kernel.

Projection, positive central-domain checks and physical world production remain
with the caller. This helper preserves every query and support interval. It
avoids constructing scanline matrices for nearest-foot observations and uses
the existing reentrant nearest-segment implementation for the fallback foot.
"""
from __future__ import annotations

import numpy as np
from acceleration import nearest_segments


def nearest_finite_human_projected_queries(projected, depth, central, midpoint,
                                           uv, normals, bounds, *,
                                           tangent_bounds=None, world_closed=False):
    projected = np.asarray(projected, float)
    uv = np.asarray(uv, float)
    normals = np.asarray(normals, float)
    bounds = np.asarray(bounds, float)
    start, delta = projected[:-1], np.diff(projected, axis=0)
    length2 = np.einsum('ni,ni->n', delta, delta)
    valid = (central[:-1] & central[1:] & midpoint &
             np.isfinite(start).all(axis=1) &
             np.isfinite(delta).all(axis=1) &
             (depth[:-1] > 0) & (depth[1:] > 0) & (length2 > 1e-4))
    if not np.any(valid):
        raise ValueError('Human finite physical primitive has no positive central segment')
    ids = np.flatnonzero(valid)
    a, d, l2 = start[ids], delta[ids], length2[ids]
    closest, _, station_at_foot, nearest = nearest_segments(
        uv, a, d, l2, np.ones(len(ids), dtype=bool))
    vector = nearest - uv
    fallback_signed = np.einsum('ni,ni->n', vector, normals)
    tangent = np.c_[-normals[:, 1], normals[:, 0]]
    fallback_tangent = np.einsum('ni,ni->n', vector, tangent)
    segment = ids[closest]

    previous = np.zeros(len(valid), bool)
    following = previous.copy()
    previous[1:] = valid[:-1]
    following[:-1] = valid[1:]
    # Only actual WORLD closure connects first/last finite segments.
    # Projection coincidence alone cannot join physical endpoints.
    if world_closed:
        previous[0] = valid[-1]
        following[-1] = valid[0]
    coverage = (((station_at_foot > 0.) & (station_at_foot < 1.)) |
                ((station_at_foot == 0.) & previous[segment]) |
                ((station_at_foot == 1.) & following[segment]))
    selected_signed = fallback_signed
    selected_station = (segment + station_at_foot) / (len(projected) - 1)

    radius = np.max(np.abs(bounds), axis=1)
    tangent_bounds = (np.c_[-radius, radius] if tangent_bounds is None
                      else np.asarray(tangent_bounds, float))
    if (tangent_bounds.shape != bounds.shape or not np.isfinite(tangent_bounds).all()
            or np.any(tangent_bounds[:, 0] > tangent_bounds[:, 1])):
        raise ValueError('Finite authenticated drawing/context tangent envelope required')
    coverage_raw = np.where(coverage, 0., fallback_tangent)
    normal_violation = selected_signed - np.clip(
        selected_signed, bounds[:, 0], bounds[:, 1])
    coverage_violation = coverage_raw - np.clip(
        coverage_raw, tangent_bounds[:, 0], tangent_bounds[:, 1])
    violation = np.c_[normal_violation, coverage_violation].ravel()
    if not np.isfinite(violation).all():
        raise ValueError('Nonfinite direct human stroke query')
    return dict(residual=violation, signed_normal_px=selected_signed,
                coverage_tangent_px=coverage_raw, finite_scanline_intersection=coverage,
                finite_primitive_station=selected_station,
                normal_violation_px=normal_violation, coverage_violation_px=coverage_violation,
                nearest_finite_vector_px=vector, nearest_finite_segment_index=segment,
                nearest_finite_segment_station=station_at_foot,
                nearest_finite_signed_normal_px=fallback_signed,
                nearest_finite_tangent_px=fallback_tangent,
                finite_source_association='nearest_finite_segment',
                valid_projected_segment_count=len(ids), endpoint_coverage_allowance_px=radius)
