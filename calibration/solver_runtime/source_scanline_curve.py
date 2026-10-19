"""Conditional x(y) observations on a complete finite projected curve.

The original horizontal source scan supplies y as context and measures x.
Left/right chooses the corresponding whole-curve intersection, independently
of the observed x, a source ROI, or a nearest-foot query. Raw scalar layout is
[dx, 0]; the caller retains the original source sigma and integration budget.
"""
import numpy as np


class SourceScanlineDomainError(ValueError):
    """The specified finite source branch has no usable intersection/chart."""


def _inputs(projected_uv, central_valid, observed_uv, side):
    points = np.asarray(projected_uv, dtype=np.float64)
    valid = np.asarray(central_valid)
    observed = np.asarray(observed_uv, dtype=np.float64)
    if (points.ndim != 2 or points.shape[1:] != (2,) or len(points) < 2
            or valid.shape != (len(points),) or valid.dtype.kind != 'b'
            or observed.ndim != 2 or observed.shape[1:] != (2,)
            or not np.isfinite(observed).all() or not np.isfinite(points[valid]).all()
            or side not in ('left', 'right')):
        raise ValueError('Finite projected polyline, bool central domain, source UV and explicit left/right side required')
    return points, valid, observed


def _project(camera, world):
    # Reuse the actual Brown14/central projector. Importing this module itself
    # needs neither the legacy runtime nor a source graph.
    from wide_anchor_candidate.full_brown_projection import project_central_valid_full_brown
    points, _, valid = project_central_valid_full_brown(camera, world)
    return np.asarray(points, float), np.asarray(valid, bool)


def projected_horizontal_scanline_value(projected_uv, central_valid, observed_uv, *,
                                       side, with_association=False):
    points, valid, observed = _inputs(projected_uv, central_valid, observed_uv, side)
    start, end = points[:-1], points[1:]
    delta = end-start
    admitted = valid[:-1] & valid[1:] & np.isfinite(delta).all(axis=1)
    # Duplicate closure/vertices are not additional intersections. A lone
    # zero-length point is not a finite source curve witness.
    admitted &= np.any(delta != 0., axis=1)
    segments, stations, regular, duplicates, xs = [], [], [], [], []
    for source_index, (_, y) in enumerate(observed):
        crossing = admitted & (np.minimum(start[:, 1], end[:, 1]) <= y) & (np.maximum(start[:, 1], end[:, 1]) >= y)
        candidates = []
        for segment in np.flatnonzero(crossing):
            dy = delta[segment, 1]
            if dy == 0.:
                # A coincident horizontal edge has an interval of intersections.
                # Its extreme is a value, but it has no regular x(y) derivative.
                for t in (0., 1.):
                    candidates.append((float(start[segment, 0] if t == 0. else end[segment, 0]), int(segment), t, True))
                continue
            t = float((y-start[segment, 1])/dy)
            # Preserve literal shared vertices; evaluating A+1*(B-A) may differ
            # from B by one ULP and otherwise create a spurious branch tie.
            x = (float(start[segment, 0]) if y == start[segment, 1] else
                 float(end[segment, 0]) if y == end[segment, 1] else
                 float(start[segment, 0]+t*delta[segment, 0]))
            candidates.append((x, int(segment), t, False))
        if not candidates:
            raise SourceScanlineDomainError(f'No positive-central finite {side} intersection at source vertex {source_index}, y={y!r}')
        selected_x = (min if side == 'left' else max)(c[0] for c in candidates)
        ties = [c for c in candidates if c[0] == selected_x]
        selected = next((c for c in ties if not c[3]), ties[0])
        _, segment, t, horizontal = selected
        is_regular = not any(c[3] for c in ties)
        if is_regular and t in (0., 1.):
            # At a duplicate vertex, an actual crossing has incident points
            # on both sides of y. Same-side incident edges merely touch y.
            signs = set()
            for _, s, st, h in ties:
                if h or st not in (0., 1.):
                    continue
                other_y = end[s, 1] if st == 0. else start[s, 1]
                signs.add(int(np.sign(other_y-y)))
            is_regular = -1 in signs and 1 in signs
        xs.append(selected_x); segments.append(segment); stations.append(t)
        regular.append(is_regular); duplicates.append(len(ties))
    residual = np.zeros_like(observed)
    residual[:, 0] = np.asarray(xs)-observed[:, 0]
    if not with_association:
        return residual.reshape(-1)
    selected = np.asarray(segments, dtype=np.int64)
    association = dict(segment_indices=selected, nominal_station=np.asarray(stations),
        observed_uv=observed.copy(), side=side, regular=np.asarray(regular, dtype=bool),
        nominal_dy_sign=np.sign(delta[selected, 1]), duplicate_witness_count=np.asarray(duplicates, dtype=np.int64),
        source_branch_selection='whole_finite_curve_min_x' if side == 'left' else 'whole_finite_curve_max_x',
        ROI_or_observed_x_used_to_select_branch=False,
        chart_is_nominal_segment_formula_not_dynamic_source_value=True)
    return residual.reshape(-1), association


def projected_horizontal_scanline_chart_response(projected_uv, central_valid,
                                                 observed_uv, association):
    """Fixed nominal segment formula for FD; never use as actual trial guard.

    An internal vertex uses the selected one-sided polyline chart. FD may
    extend that supporting segment infinitesimally across its nominal vertex;
    actual values always search the finite curve again. Tangencies and genuine
    physical endpoints do not acquire a fictitious regular derivative.
    """
    points, valid, observed = _inputs(projected_uv, central_valid, observed_uv, association['side'])
    selected = np.asarray(association['segment_indices'])
    regular = np.asarray(association['regular'])
    sign = np.asarray(association['nominal_dy_sign'])
    if (selected.dtype.kind not in 'iu' or selected.shape != (len(observed),)
            or np.any(selected < 0) or np.any(selected+1 >= len(points))
            or regular.shape != selected.shape or regular.dtype.kind != 'b'
            or sign.shape != selected.shape or not np.array_equal(observed, association['observed_uv'])):
        raise ValueError('Exact original scanline observation and nominal branch association required')
    if not np.all(regular):
        raise SourceScanlineDomainError('Source branch is tangent, coincident, or at a true finite endpoint; no regular x(y) chart')
    if not np.all(valid[selected] & valid[selected+1]):
        raise SourceScanlineDomainError('Nominal scanline branch left the positive-central domain')
    start, end = points[selected], points[selected+1]
    dy = end[:, 1]-start[:, 1]
    if np.any(dy == 0.) or not np.array_equal(np.sign(dy), sign):
        raise SourceScanlineDomainError('Nominal scanline segment slope changed branch')
    t = (observed[:, 1]-start[:, 1])/dy
    residual = np.zeros_like(observed)
    residual[:, 0] = start[:, 0]+t*(end[:, 0]-start[:, 0])-observed[:, 0]
    if not np.isfinite(residual).all():
        raise SourceScanlineDomainError('Nonfinite original scanline chart')
    return residual.reshape(-1)


def projected_horizontal_scanline_value_and_gradient(projected_uv, central_valid,
                                                     observed_uv, *, side):
    """Raw [dx,0], association and compact dx gradients of two endpoints.

    The final array has shape (queries, endpoint0/1, projected x/y). It is the
    nominal segment chart's derivative, with no sigma or loss weighting.
    """
    raw, association = projected_horizontal_scanline_value(projected_uv, central_valid,
        observed_uv, side=side, with_association=True)
    # Apply the same regularity/identity contract as coordinate FD.
    projected_horizontal_scanline_chart_response(projected_uv, central_valid, observed_uv, association)
    points = np.asarray(projected_uv, float)
    index = association['segment_indices']
    delta = points[index+1]-points[index]
    t = association['nominal_station']
    slope = delta[:, 0]/delta[:, 1]
    gradient = np.empty((len(index), 2, 2))
    gradient[:, 0, 0] = 1.-t
    gradient[:, 1, 0] = t
    gradient[:, 0, 1] = -(1.-t)*slope
    gradient[:, 1, 1] = -t*slope
    return raw, association, gradient


def horizontal_scanline_curve_value(camera, world, observed_uv, *, side,
                                    with_association=False):
    points, valid = _project(camera, world)
    return projected_horizontal_scanline_value(points, valid, observed_uv,
        side=side, with_association=with_association)


def horizontal_scanline_curve_chart_response(camera, world, observed_uv, association):
    points, valid = _project(camera, world)
    return projected_horizontal_scanline_chart_response(points, valid, observed_uv, association)
