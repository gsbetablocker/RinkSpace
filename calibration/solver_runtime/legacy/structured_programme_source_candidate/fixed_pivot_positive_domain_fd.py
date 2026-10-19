"""Finite differences inside the existing strict positive fixed-pivot domain.

No nominal coordinate, source value, scale or physical bound is changed.
The original symmetric step is retained whenever both samples are legal.
At a positive-height boundary, the same step uses a second-order inward
stencil instead of an illegal negative pivot or a cancellation-prone tiny h.
"""
from __future__ import annotations

import hashlib

import numpy as np


def positive_pivot_height_columns(blocks):
    """Return explicit global Pz slots for actual FixedLeverXYZ bundles."""
    from z190_fixed_lever_xyz_source_bundle import FixedLeverXYZSourceBundle
    columns = []
    for block in blocks:
        if not isinstance(block.bundle, FixedLeverXYZSourceBundle):
            raise TypeError('Actual fixed XYZ positive-pivot source bundle required')
        slots = np.asarray(block.xyz_slots, int)
        if slots.shape != (3,) or len(np.unique(slots)) != 3:
            raise ValueError('Explicit distinct fixed-pivot XYZ triplet required')
        columns.append(int(slots[2]))
    return np.unique(columns).astype(int)


def _domain(q, direction, h, height_columns):
    q = np.asarray(q, float)
    direction = np.asarray(direction, float)
    columns = np.unique(np.asarray(height_columns, int))
    h = float(h)
    if (q.ndim != 1 or direction.shape != q.shape or not np.isfinite(q).all()
            or not np.isfinite(direction).all() or not np.isfinite(h) or h <= 0.
            or np.any(columns < 0) or np.any(columns >= len(q))
            or np.any(q[columns] <= 0.)):
        raise ValueError('Finite unchanged state/direction and strictly positive pivots required')
    return q, direction, h, columns


def positive_pivot_fd_stencil(q, direction, h, height_columns):
    """Choose original-h central or legal second-order one-sided offsets.

    Offsets multiply the supplied direction; weights differentiate that exact
    direction at the unchanged nominal q.  No step reduction is attempted.
    If neither original-h inward stencil is legal, qualification stays refused.
    """
    q, direction, h, columns = _domain(q, direction, h, height_columns)

    def legal(offset):
        values = q[columns] + offset * direction[columns]
        return bool(np.isfinite(values).all() and np.all(values > 0.))

    common = dict(step=h, height_columns=columns.tolist(),
                  nominal_height_values=q[columns].tolist(),
                  nominal_state_changed=False, physical_bound_changed=False,
                  finite_difference_step_reduced=False)
    if legal(-h) and legal(h):
        return dict(common, method='original_symmetric',
                    offsets=[-h, h], weights=[-1. / (2. * h), 1. / (2. * h)])
    for sign in (1., -1.):
        if legal(sign * h) and legal(sign * 2. * h):
            return dict(common, method='inward_second_order', sign=sign,
                        offsets=[0., sign * h, sign * 2. * h],
                        weights=[-3. * sign / (2. * h), 2. * sign / h,
                                 -sign / (2. * h)])
    raise ValueError('Original h has no legal second-order stencil in the strict positive-pivot domain')


def positive_pivot_guard_direction(q, direction, h, height_columns):
    """Keep central qualification legal; identify columns needing extra proof.

    Only random guard-vector components whose +/-h samples violate an existing
    Pz bound are zeroed.  The source state and source Jacobian are untouched.
    Every reported column must be checked separately with real inward values.
    """
    q, direction, h, columns = _domain(q, direction, h, height_columns)
    safe = direction.copy()
    boundary = []
    for col in columns:
        col = int(col)
        if q[col] - h * abs(direction[col]) <= 0.:
            boundary.append(dict(column=col, nominal_Pz=float(q[col]),
                                 original_direction_component=float(direction[col]),
                                 guard_step=h, separate_real_column_check_required=True))
            safe[col] = 0.
    return safe, dict(boundary_columns=boundary, nominal_state_changed=False,
                      source_J_changed=False, symmetric_guard_step_changed=False)


def positive_pivot_coordinate_derivative(evaluate, q, col, h, height_columns,
                                        *, base=None):
    """Differentiate one source coordinate without leaving its positive domain."""
    q = np.asarray(q, float)
    col, h = int(col), float(h)
    columns = np.asarray(height_columns, int)
    # Preserve the old arithmetic and evaluation order on every legal column.
    if col not in columns or q[col] - h > 0.:
        plus, minus = q.copy(), q.copy()
        plus[col] += h
        minus[col] -= h
        try:
            return (evaluate(plus) - evaluate(minus)) / (2. * h)
        except ValueError as error:
            if not _print_domain_error(error):raise
            return print_domain_coordinate_derivative(evaluate,q,col,h,base=base)[0]
    direction = np.zeros_like(q)
    direction[col] = 1.
    stencil = positive_pivot_fd_stencil(q, direction, h, columns)
    if stencil['method'] != 'inward_second_order' or stencil['sign'] != 1.:
        raise ValueError('Positive Pz coordinate requires the original-h inward stencil')
    first, second = q.copy(), q.copy()
    first[col] += h
    second[col] += 2. * h
    nominal = evaluate(q) if base is None else np.asarray(base, float)
    one, two = evaluate(first), evaluate(second)
    # Subtract before multiplying to avoid -3*r0 + 4*r1 - r2 cancellation.
    return (4. * (one - nominal) - (two - nominal)) / (2. * h)


def _print_domain_error(error):
    text=str(error)
    return text.startswith('Ad') and text.endswith('current qualified print exceeds current board Z domain')


def print_domain_coordinate_derivative(evaluate,q,col,h,*,base=None):
    """Real source samples inside the unchanged qualified-print physical domain."""
    q=np.asarray(q,float);col=int(col);initial_h=float(h)
    nominal=np.asarray(evaluate(q)if base is None else base,float);trials=[]
    while True:
        values={};errors={}
        for sign in (1.,-1.):
            value=q.copy();value[col]+=sign*h
            if value[col]==q[col]:raise ValueError('No representable feasible print-domain source FD step')
            try:values[sign]=np.asarray(evaluate(value),float)
            except ValueError as error:
                if not _print_domain_error(error):raise
                errors[str(int(sign))]=str(error)
        trials.append(dict(step=float(h),positive_legal=1. in values,negative_legal=-1. in values,errors=errors))
        if len(values)==2:
            return (values[1.]-values[-1.])/(2*h),dict(column=col,method='central',initial_step=initial_h,step=float(h),trials=trials)
        for sign,first in values.items():
            value=q.copy();value[col]+=2*sign*h
            try:second=np.asarray(evaluate(value),float)
            except ValueError as error:
                if not _print_domain_error(error):raise
                trials[-1].setdefault('twice_step_errors',{})[str(int(sign))]=str(error);continue
            observed=sign*(4*(first-nominal)-(second-nominal))/(2*h)
            return observed,dict(column=col,method='feasible_print_second_order',side=int(sign),initial_step=initial_h,
                step=float(h),nominal_q_and_r_and_physical_guard_changed=False,trials=trials)
        h*=.5


def print_domain_directional_stencil(q,direction,h,validate_world):
    """Check actual physical world samples before a source directional gate."""
    q=np.asarray(q,float);direction=np.asarray(direction,float);initial_h=float(h)
    validate_world(q);trials=[]
    while True:
        legal={};errors={}
        for sign in (1.,-1.):
            value=q+sign*h*direction
            if not np.any(value!=q):raise ValueError('No representable physical-domain directional FD step')
            try:validate_world(value);legal[sign]=True
            except ValueError as error:
                if not _print_domain_error(error):raise
                errors[str(int(sign))]=str(error)
        trials.append(dict(step=float(h),positive_legal=1. in legal,negative_legal=-1. in legal,errors=errors))
        if len(legal)==2:
            return dict(method='original_symmetric',initial_step=initial_h,step=float(h),offsets=[-h,h],
                weights=[-1./(2*h),1./(2*h)],physical_domain_trials=trials)
        for sign in legal:
            try:validate_world(q+2*sign*h*direction)
            except ValueError as error:
                if not _print_domain_error(error):raise
                trials[-1].setdefault('twice_step_errors',{})[str(int(sign))]=str(error);continue
            return dict(method='feasible_print_second_order',side=int(sign),initial_step=initial_h,step=float(h),
                offsets=[0.,sign*h,2*sign*h],weights=[-3*sign/(2*h),2*sign/h,-sign/(2*h)],
                physical_domain_trials=trials,nominal_q_and_direction_and_J_and_guard_changed=False)
        h*=.5


def check_positive_pivot_boundary_column(evaluate, q, col, predicted, h,
                                         height_columns, *, base=None, allowed):
    """Check the actual real inward source values against one supplied J column.

    Returns (observed, JSON metadata).  ``allowed`` must be the caller's
    unchanged qualification threshold.  Failed checks are never replaced by
    a fixed-association residual or a relaxed physical domain.
    """
    q = np.asarray(q, float)
    col, h = int(col), float(h)
    direction = np.zeros_like(q)
    direction[col] = 1.
    stencil = positive_pivot_fd_stencil(q, direction, h, height_columns)
    if col not in np.asarray(height_columns, int):
        raise ValueError('Separate boundary proof requires an actual fixed-pivot Pz column')
    predicted = np.asarray(predicted, float)
    nominal = np.asarray(evaluate(q) if base is None else base, float)
    values = [nominal if offset == 0. else np.asarray(evaluate(q + offset * direction), float)
              for offset in stencil['offsets']]
    if stencil['method'] == 'original_symmetric':
        observed = (values[1] - values[0]) / (2. * h)
    else:
        observed = stencil['sign'] * (4. * (values[1] - values[0])
                                     - (values[2] - values[0])) / (2. * h)
    if (observed.shape != predicted.shape or not np.isfinite(observed).all()
            or not np.isfinite(predicted).all() or not np.isfinite(allowed) or allowed < 0.):
        raise ValueError('Finite matching real inward derivative/J column and original allowed required')
    difference = np.abs(observed - predicted)
    at = int(np.argmax(difference))
    report = dict(
        schema='rinkspace.actual-positive-pivot-boundary-column-qualification.v1',
        column=col, nominal_Pz=float(q[col]), stencil=stencil,
        q_sha256=hashlib.sha256(q.tobytes()).hexdigest(),
        scalar_rows=len(observed), max_absolute_error=float(difference[at]),
        argmax_scalar_row=at, observed_at_argmax=float(observed[at]),
        predicted_at_argmax=float(predicted[at]), original_allowed=float(allowed),
        passed=bool(difference.max() <= allowed),
        real_values_sha256=[hashlib.sha256(value.tobytes()).hexdigest() for value in values],
        source_values_scales_roles_and_physical_constraints_changed=False,
        original_h_retained=True, fixed_association_values_used=False,
        accuracy_accepted=False)
    return observed, report
