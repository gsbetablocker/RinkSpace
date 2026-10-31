"""Select an auditable subset of measured source pixels without changing them.

This is spatial coverage, not a residual filter or an information/weight model.
There is no station coordinate, projected pixel or fitted state in this API.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np


class SamplingReason(IntEnum):
    COVERED_BY_SUBSET = 0
    FIXED_POINT = 1
    EXTREME_SEED = 2
    FARTHEST_COVERAGE = 3
    DUPLICATE_UV = 4


@dataclass(frozen=True)
class SpatialSampleSelection:
    """Original row indices and one SamplingReason code per original row."""

    selected_indices: np.ndarray
    reason_codes: np.ndarray


def _vector(value, name, count):
    array = np.asarray(value)
    if array.shape != (count,):
        raise ValueError(f"{name} must have shape ({count},)")
    return array


def _identity_vector(value, name, count):
    array = _vector(value, name, count)
    if array.dtype.kind not in "iuUS":
        raise ValueError(f"{name} must contain integer or string identities")
    return array


def sample_source_pixels(
    uv,
    pts,
    path_ids,
    observation_ids,
    *,
    kind=None,
    spacing_px=24.0,
    stable_keys=None,
    fragment_ids=None,
) -> SpatialSampleSelection:
    """Return original rows covering each (PTS, physical path) in source UV.

    ``kind`` uses the existing observation convention: 0 is a fixed material
    point, 1 is a curve. All fixed rows are retained, including coincident
    fixed rows. Curve rows at the same UV are deduplicated by the smallest
    ``stable_keys`` (default: provenance ``observation_ids``), then ID. A
    coincident fixed row takes precedence over a curve duplicate.

    For each group, the four axis extrema seed a farthest-point cover. Ties
    between different UVs use lexicographic (u, v); duplicate ties use stable
    provenance. Iteration stops when every candidate UV lies within
    ``spacing_px`` of a selected source pixel. The spacing is one common
    coverage radius, not a minimum separation or a per-entity point quota.
    Optional existing ``fragment_ids`` seed each explicitly labeled visible
    fragment, without inventing fragments or interpolating across a gap.

    Duplicate multiplicity and row order do not affect selected identities.
    Adding new, distinct measured locations can change the discrete cover.
    The caller must isolate sources/pixel domains before invoking this API;
    PTS alone does not identify exposures across different source videos.
    Working distance storage is O(group size), not an all-pairs matrix.
    Inputs are never written; returned indices refer to their original order.
    """
    pixels = np.asarray(uv)
    if pixels.ndim != 2 or pixels.shape[1] != 2 or pixels.dtype.kind not in "iuf":
        raise ValueError("uv must be a finite numeric (N, 2) array")
    if not np.isfinite(pixels).all():
        raise ValueError("uv must be finite")
    pixels = np.asarray(pixels, dtype=np.float64)
    count = len(pixels)
    times = _vector(pts, "pts", count)
    if times.dtype.kind not in "iu":
        raise ValueError("pts must contain original integer source timestamps")
    paths = _identity_vector(path_ids, "path_ids", count)
    provenance = _identity_vector(observation_ids, "observation_ids", count)
    keys = provenance if stable_keys is None else _identity_vector(stable_keys, "stable_keys", count)
    kinds = np.ones(count, dtype=np.uint8) if kind is None else _vector(kind, "kind", count)
    if kinds.dtype.kind not in "iu" or not np.isin(kinds, [0, 1]).all():
        raise ValueError("kind must contain only 0 (fixed) or 1 (curve)")
    fragments = None if fragment_ids is None else _identity_vector(fragment_ids, "fragment_ids", count)
    spacing = np.asarray(spacing_px)
    if spacing.ndim or spacing.dtype.kind not in "iuf" or not np.isfinite(spacing) or spacing <= 0:
        raise ValueError("spacing_px must be finite and positive")
    spacing = float(spacing)
    reasons = np.full(count, SamplingReason.COVERED_BY_SUBSET, dtype=np.uint8)
    selected = kinds == 0
    reasons[selected] = SamplingReason.FIXED_POINT
    if count == 0:
        return SpatialSampleSelection(np.empty(0, dtype=np.int64), reasons)

    # Geometry sorts first inside a group, so argmax/argmin ties never depend
    # on row multiplicity or acquisition order. Fixed duplicates take priority.
    order = np.lexsort((provenance, keys, kinds, pixels[:, 1], pixels[:, 0], paths, times))
    ordered_pts, ordered_paths = times[order], paths[order]
    bounds = np.r_[0, np.flatnonzero(
        (ordered_pts[1:] != ordered_pts[:-1]) |
        (ordered_paths[1:] != ordered_paths[:-1])) + 1, count]
    for begin, stop in zip(bounds[:-1], bounds[1:]):
        rows = order[begin:stop]
        values = pixels[rows]
        first_at_uv = np.r_[True, np.any(values[1:] != values[:-1], axis=1)]
        duplicates = ~first_at_uv & (kinds[rows] == 1)
        reasons[rows[duplicates]] = SamplingReason.DUPLICATE_UV
        candidates = rows[first_at_uv]
        points = pixels[candidates]
        chosen = kinds[candidates] == 0
        distances = np.full(len(candidates), np.inf)

        def retain(local, reason):
            chosen[local] = True
            row = candidates[local]
            if kinds[row] != 0:
                selected[row] = True
                reasons[row] = reason
            # Only a vector and its temporary coordinate differences exist.
            np.minimum(distances, np.hypot(
                points[:, 0] - points[local, 0],
                points[:, 1] - points[local, 1]), out=distances)

        for local in np.flatnonzero(chosen):
            retain(local, SamplingReason.FIXED_POINT)
        if fragments is None:
            seed_sets = (np.arange(len(candidates)),)
        else:
            # A duplicate UV can have several real fragment labels; each
            # label seeds the same canonical UV rather than another row.
            uv_index = np.cumsum(first_at_uv) - 1
            fragment_order = np.argsort(fragments[rows], kind="stable")
            labels = fragments[rows][fragment_order]
            edges = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1, len(rows)]
            seed_sets = (np.unique(uv_index[fragment_order[a:b]]) for a, b in zip(edges[:-1], edges[1:]))
        for members in seed_sets:
            part = points[members]
            seeds = np.unique(members[[int(part[:, 0].argmin()), int(part[:, 0].argmax()),
                                       int(part[:, 1].argmin()), int(part[:, 1].argmax())]])
            for local in seeds:
                if not chosen[local]:
                    retain(local, SamplingReason.EXTREME_SEED)
        while True:
            farthest = int(distances.argmax())
            if distances[farthest] <= spacing:
                break
            retain(farthest, SamplingReason.FARTHEST_COVERAGE)

    return SpatialSampleSelection(np.flatnonzero(selected), reasons)
