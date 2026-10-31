"""Candidate rank representatives drawn only from original source observations.

This module does not replace the production sampler. A projective flow rank
is a local geometric diagnostic, not a solver weight or an accuracy claim.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import qr


def _vector(value, name, count, *, identities=False):
    array = np.asarray(value)
    if array.shape != (count,):
        raise ValueError(f"{name} must have shape ({count},)")
    if identities and array.dtype.kind not in "iuUS":
        raise ValueError(f"{name} must contain integer or string identities")
    return array


def _positive_scalar(value, name, *, less_than_one=False):
    array = np.asarray(value)
    if (array.ndim or array.dtype.kind not in "iuf" or not np.isfinite(array)
            or array <= 0 or (less_than_one and array >= 1)):
        bound = " and less than one" if less_than_one else ""
        raise ValueError(f"{name} must be finite, positive{bound}")
    return float(array)


def sample_projective_constraint_rows(
    uv,
    normals,
    pts,
    physical_entity_ids,
    observation_ids,
    kind,
    *,
    image_width,
    image_height,
    geometry_uv=None,
    rank_rtol=1e-6,
):
    """Select original observations spanning each (PTS, physical entity).

    Flow coordinates are normalized as ``x=u/image_width, y=v/image_height``.
    Optional ``geometry_uv`` supplies current model-projection positions for
    that matrix only; otherwise the source ``uv`` supplies its positions.
    Original source UV and observation IDs always determine canonical row
    ordering and duplicate representatives, and are never replaced or written.
    ``kind=0`` contributes both infinitesimal projective flow components;
    ``kind=1`` contributes the flow dotted with its known pixel-space normal.
    Normals transform as covectors to the normalized coordinate system and
    then become unit directions. Neither their length nor sampling density
    supplies observation weights.

    Pivoted QR of the constraint matrix transpose chooses independent rows.
    Numerical rank counts ``abs(diag(R)) > rank_rtol*abs(R[0, 0])``. If either
    fixed-point component is chosen, its original observation is retained.
    UV supplies canonical ordering so duplicate multiplicity cannot change
    QR tie order. At identical UV, a fixed observation takes precedence over
    a curve observation; otherwise the smallest stable observation ID wins.
    Known normals at one entity's identical UV must describe the same curve.

    Return ``selected_indices`` in original row order and sorted ``groups``
    containing ``pts``, ``physical_entity_id``, numerical ``rank``, original
    ``available`` count, unique-UV ``unique_available``, and ``selected`` count.
    Input arrays remain unchanged. Sources and pixel domains must be isolated
    by the caller: a timestamp alone does not identify a source exposure.
    This rank representative has no point quota or spatial coverage promise.
    """
    pixels = np.asarray(uv)
    if (pixels.ndim != 2 or pixels.shape[1] != 2 or pixels.dtype.kind not in "iuf"
            or not np.isfinite(pixels).all()):
        raise ValueError("uv must be a finite numeric (N, 2) array")
    pixels = np.asarray(pixels, dtype=np.float64)
    count = len(pixels)
    geometry = pixels if geometry_uv is None else np.asarray(geometry_uv)
    if (geometry.shape != (count, 2) or geometry.dtype.kind not in "iuf"
            or not np.isfinite(geometry).all()):
        raise ValueError("geometry_uv must be a finite numeric (N, 2) array")
    geometry = np.asarray(geometry, dtype=np.float64)
    directions = np.asarray(normals)
    if directions.shape != (count, 2) or directions.dtype.kind not in "iuf":
        raise ValueError("normals must be a numeric (N, 2) array")
    directions = np.asarray(directions, dtype=np.float64)
    times = _vector(pts, "pts", count)
    if times.dtype.kind not in "iu":
        raise ValueError("pts must contain original integer source timestamps")
    entities = _vector(physical_entity_ids, "physical_entity_ids", count, identities=True)
    identities = _vector(observation_ids, "observation_ids", count, identities=True)
    kinds = _vector(kind, "kind", count)
    if kinds.dtype.kind not in "iu" or not np.isin(kinds, [0, 1]).all():
        raise ValueError("kind must contain only 0 (fixed) or 1 (curve)")
    curve = kinds == 1
    if (not np.isfinite(directions[curve]).all()
            or (np.linalg.norm(directions[curve], axis=1) <= 0).any()):
        raise ValueError("curve normals must be finite and nonzero")
    scale = np.array([_positive_scalar(image_width, "image_width"),
                      _positive_scalar(image_height, "image_height")])
    tolerance = _positive_scalar(rank_rtol, "rank_rtol", less_than_one=True)
    if not count:
        return dict(selected_indices=np.empty(0, dtype=np.int64), groups=[])

    order = np.lexsort((identities, kinds, pixels[:, 1], pixels[:, 0], entities, times))
    ordered_times, ordered_entities = times[order], entities[order]
    bounds = np.r_[0, np.flatnonzero(
        (ordered_times[1:] != ordered_times[:-1]) |
        (ordered_entities[1:] != ordered_entities[:-1])) + 1, count]
    chosen = np.zeros(count, dtype=bool)
    reports = []
    for begin, stop in zip(bounds[:-1], bounds[1:]):
        rows = order[begin:stop]
        values = pixels[rows]
        distinct = np.r_[True, np.any(values[1:] != values[:-1], axis=1)]
        candidates = rows[distinct]
        x, y = (geometry[candidates] / scale).T
        zero, one = np.zeros(len(candidates)), np.ones(len(candidates))
        du = np.stack([x, y, one, zero, zero, zero, -x*x, -x*y], axis=1)
        dv = np.stack([zero, zero, zero, x, y, one, -x*y, -y*y], axis=1)
        matrix_rows, owners = [], []
        for local, original in enumerate(candidates):
            if kinds[original] == 0:
                matrix_rows.extend([du[local], dv[local]])
                owners.extend([original, original])
            else:
                normal = directions[original] * scale
                normal = normal / np.linalg.norm(normal)
                matrix_rows.append(normal[0]*du[local] + normal[1]*dv[local])
                owners.append(original)
        matrix = np.asarray(matrix_rows)
        _, triangular, pivots = qr(matrix.T, mode="economic", pivoting=True)
        diagonal = np.abs(np.diag(triangular))
        rank = int(np.count_nonzero(diagonal > tolerance*diagonal[0]))
        selected = np.unique(np.asarray(owners, dtype=np.int64)[pivots[:rank]])
        chosen[selected] = True
        reports.append(dict(pts=times[rows[0]].item(),
            physical_entity_id=entities[rows[0]].item(), rank=rank,
            available=len(rows), unique_available=len(candidates), selected=len(selected)))
    return dict(selected_indices=np.flatnonzero(chosen), groups=reports)
