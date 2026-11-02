"""Finite native chart selection for six removable ice nominal coordinates.

The state still stores board + layout146 in its original ambient slots. Chart
10 moves the eight actual house X coordinates; chart 11 moves the 32 actual
log-radii. Codes 12/13 mark the old nominal slots as witness outputs. Native
retraction must export a legal common nominal witness and its bounded legacy
deviations; this helper never freezes a nominal or rewrites stored values.

This selection applies only when all six nominal directions and all forty
deviation directions were active in the supported legacy charts, with no
external bound that would need an additional coupled witness feasibility
problem. Unsupported configurations return None for the caller's old chart.
"""

import numpy as np

ACTUAL_X_INDICES = np.arange(49, 57)
ACTUAL_RADIUS_INDICES = np.arange(61, 93)
NOMINAL_WITNESS_INDICES = np.array([6, 7, 10, 11, 12, 13])

_ACTUAL_INDICES = np.concatenate((ACTUAL_X_INDICES, ACTUAL_RADIUS_INDICES))
_REQUIRED_INDICES = np.concatenate((NOMINAL_WITNESS_INDICES, _ACTUAL_INDICES))
_LEGACY_NOMINAL_CHARTS = np.array([0, 0, 1, 3, 4, 5])


def actual_ice_chart_configuration(values, legacy_charts, active, lower=None, upper=None, *, relative_limit=.01):
    """Return copied charts/activity and forty ambient-index step overrides.

    Values and legacy_charts are full one-dimensional scene arrays; the first
    six scene rows hold the board. Active lists ambient indices, in the caller's
    column order. Bounds, when supplied, have the same shape as values and
    describe the old ambient state. The selected native relative_limit remains
    part of the physical model even when bounds are omitted (default +/-1%).

    The returned steps dictionary only overrides the forty actual rows.
    Its X steps have arena-length units and its radius steps have log-radius
    units. At the initial state they reproduce the geometric tangent scale of
    the old additive deviation step .001: dx=.001*pitch and dr=.001*nominal_r.
    No input array, bound, or state value is modified.
    """
    if not np.isfinite(relative_limit) or not 0 < relative_limit < 1:
        return None
    try:
        state = np.asarray(values, dtype=np.float64)
        charts = np.asarray(legacy_charts)
        columns = np.asarray(active)
    except (TypeError, ValueError, OverflowError):
        return None
    if (state.ndim != 1 or state.size < 152 or not np.isfinite(state).all()
            or state[9] <= 0 or charts.shape != state.shape or charts.dtype.kind not in 'iu'
            or columns.ndim != 1 or columns.dtype.kind not in 'iu'):
        return None
    if (np.any(charts > np.iinfo(np.int32).max) or np.any(charts < 0)
            or np.any(columns < 0) or np.any(columns >= state.size)
            or np.unique(columns).size != columns.size
            or not np.isin(_REQUIRED_INDICES, columns).all()):
        return None
    if (not np.array_equal(charts[NOMINAL_WITNESS_INDICES], _LEGACY_NOMINAL_CHARTS)
            or np.any(charts[_ACTUAL_INDICES] != 0)):
        return None

    for bound, is_lower in ((lower, True), (upper, False)):
        if bound is None:
            continue
        try:
            bounds = np.asarray(bound, dtype=np.float64)
        except (TypeError, ValueError, OverflowError):
            return None
        if bounds.shape != state.shape or np.isnan(bounds).any():
            return None
        unbounded = -np.inf if is_lower else np.inf
        if np.any(bounds[NOMINAL_WITNESS_INDICES] != unbounded):
            return None
        if is_lower:
            if np.any(bounds[_ACTUAL_INDICES] > -relative_limit):
                return None
        elif np.any(bounds[_ACTUAL_INDICES] < relative_limit):
            return None

    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        radius_steps = .001 / (1. + state[ACTUAL_RADIUS_INDICES])
    x_step = .001 * state[9]
    if (not np.isfinite(x_step) or x_step <= 0 or not np.isfinite(radius_steps).all()
            or np.any(radius_steps <= 0)):
        return None

    selected_charts = np.array(charts, dtype=np.int32, copy=True)
    selected_charts[ACTUAL_X_INDICES] = 10
    selected_charts[ACTUAL_RADIUS_INDICES] = 11
    selected_charts[NOMINAL_WITNESS_INDICES[:2]] = 12
    selected_charts[NOMINAL_WITNESS_INDICES[2:]] = 13
    selected_active = np.array(columns[~np.isin(columns, NOMINAL_WITNESS_INDICES)],
                               dtype=np.int64, copy=True)
    steps = {int(index): float(x_step) for index in ACTUAL_X_INDICES}
    steps.update((int(index), float(step))
                 for index, step in zip(ACTUAL_RADIUS_INDICES, radius_steps))
    return dict(charts=selected_charts, active=selected_active, steps=steps)
