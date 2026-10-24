"""Source navigation for measured print polylines; no model projections.

A simplified polyline's vertices may be corners. Its edge interiors supply
one-sided normals; the interpolated locations only navigate a fresh RGB read.
Raw dense samples keep their existing local chord treatment.
"""
from __future__ import annotations

import numpy as np


def _endpoint_values(segment, key, count, endpoints):
    values = segment.get(key)
    if values is None:
        return [None, None]
    a = np.asarray(values, dtype=float)
    if a.shape != (count,):
        return [None, None]
    return [float(a[i]) if np.isfinite(a[i]) and a[i] >= 0 else None
            for i in endpoints]


def segment_navigation_seeds(segment, stride=4):
    """Return navigation seeds plus the unchanged source endpoint metadata.

    Sigma/IQR pairs describe the anchor's measured endpoints, not the midpoint
    or any tracked exposure. Missing precision stays unknown.
    """
    field = ('source_native_readout_uv' if segment.get('source_native_readout_uv')
             else 'uv')
    curve = np.asarray(segment.get(field, []), dtype=float)
    if curve.ndim != 2 or curve.shape[1] != 2 or len(curve) < 2 or not np.isfinite(curve).all():
        return []
    if stride < 1:
        raise ValueError('stride must be positive')
    count = len(curve)
    simplified = field == 'uv' and 'native_simplification_tolerance_px' in segment
    closed = bool(segment.get('closed', False))
    # A repeated closing sample is not a separate edge or seed.
    unique_count = count - int(closed and np.array_equal(curve[0], curve[-1]))
    if unique_count < 2:
        return []
    indices = sorted({*range(0, unique_count, stride), unique_count - 1})
    seeds, used_edges = [], set()
    for k in indices:
        if simplified:
            a = k if closed else min(k, unique_count - 2)
            b = (a + 1) % unique_count
            if (a, b) in used_edges:
                continue
            used_edges.add((a, b))
            tangent = curve[b] - curve[a]
            uv = (curve[a] + curve[b]) * .5
            endpoints = [a, b]
            location = dict(source_edge_indices=endpoints, source_edge_fraction=.5,
                            source_normal_basis='single_simplified_edge')
        else:
            a, b = max(0, k - 2), min(unique_count - 1, k + 2)
            if closed:
                a, b = (k - 2) % unique_count, (k + 2) % unique_count
            tangent = curve[b] - curve[a]
            uv, endpoints = curve[k], [k, k]
            location = dict(source_point_index=k, source_normal_basis='local_source_samples')
        length = float(np.linalg.norm(tangent))
        if length < 1e-6:
            continue
        normal = np.array([-tangent[1], tangent[0]]) / length
        seed = dict(uv=uv.tolist(), normal=normal.tolist(), source_field=field,
                    navigation_only=True, **location)
        if simplified:
            # Keep the three-point flow stencil inside this edge, away from its corners.
            seed['tangent_halfspan_px'] = min(2., length * .25)
        sigma_key = ('source_native_station_sigma_px' if field == 'source_native_readout_uv'
                     else 'localization_sigma_px')
        iqr_key = ('source_native_transition_iqr_px' if field == 'source_native_readout_uv'
                   else 'transition_iqr_px')
        seed['source_anchor_localization_sigma_px'] = _endpoint_values(segment, sigma_key, count, endpoints)
        seed['source_anchor_transition_iqr_px'] = _endpoint_values(segment, iqr_key, count, endpoints)
        seeds.append(seed)
    return seeds


def seed_geometry_columns(seeds):
    """Store uncertainty once per seed, joined by the existing seed_index column.

    No per-row duplication and no conversion into statistical fit weights.
    NaN means the original source did not provide this quantity.
    """
    count = len(seeds)
    sigma = np.full((count, 2), np.nan, dtype='<f8')
    iqr = np.full((count, 2), np.nan, dtype='<f8')
    edges = np.full((count, 2), -1, dtype='<i4')
    fraction = np.full(count, np.nan, dtype='<f8')
    for i, seed in enumerate(seeds):
        for target, key in ((sigma, 'source_anchor_localization_sigma_px'),
                            (iqr, 'source_anchor_transition_iqr_px')):
            values = seed.get(key)
            if values is not None:
                target[i] = [np.nan if value is None else value for value in values]
        if 'source_edge_indices' in seed:
            edges[i] = seed['source_edge_indices']
            fraction[i] = seed['source_edge_fraction']
    return dict(anchor_native_pts=np.asarray([s['native_pts'] for s in seeds], dtype='<i8'),
                anchor_endpoint_sigma_px=sigma, anchor_endpoint_transition_iqr_px=iqr,
                anchor_edge_indices=edges, anchor_edge_fraction=fraction)
