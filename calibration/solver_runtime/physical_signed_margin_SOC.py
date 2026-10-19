"""Measured curvature of existing signed-margin physical inequalities.

Physical margins retain their own original units and exact nonlinear domain.
This module adds neither a padding distance nor an observation.  Its cuts are
temporary numerical lower bounds on the same already declared QP rows.
"""
from __future__ import annotations

import numpy as np


def attach_original_signed_margin_rows(domain, *, row_indices,
                                       nominal_margins, row_ids):
    indices = np.asarray(row_indices)
    nominal = np.asarray(nominal_margins, float)
    matrix = np.asarray(domain['matrix'], float)
    lower = np.asarray(domain['lower'], float)
    if (indices.ndim != 1 or indices.dtype.kind not in 'iu'
            or len(np.unique(indices)) != len(indices)
            or np.any(indices < 0) or np.any(indices >= len(lower))
            or nominal.shape != indices.shape or len(row_ids) != len(indices)
            or len(set(row_ids)) != len(row_ids) or np.any(nominal < 0.)
            or not np.isfinite(nominal).all()
            or matrix.ndim != 2 or matrix.shape[0] != len(lower)
            or lower.ndim != 1 or not np.isfinite(matrix).all()
            or not np.isfinite(lower).all()):
        raise ValueError('Complete existing signed-margin physical row identities required')
    for response in domain.get('curvature_responses', {}).values():
        if np.any((indices >= response['row_start']) & (indices < response['row_stop'])):
            raise ValueError('Physical rows cannot replace any original source face')
    metadata = dict(row_indices=indices.astype(int, copy=True),
        nominal_margins=nominal.copy(), row_ids=tuple(row_ids),
        margin_units='original_working_m', outside_or_on_domain=True)
    result = dict(domain, original_signed_margin_rows=metadata)
    original_fraction = domain.get('at_trial_fraction')
    if callable(original_fraction):
        def fraction(alpha):
            # Every shortened solve inherits precisely these physical rows;
            # restoration interpolation still applies only to source rows.
            return dict(original_fraction(alpha), original_signed_margin_rows=metadata)
        result['at_trial_fraction'] = fraction
    return result


def measured_signed_margin_step_tightening(matrix, lower, shared_step,
                                           metadata, actual_margins, *, row_ids):
    matrix = np.asarray(matrix, float)
    lower = np.asarray(lower, float)
    step = np.asarray(shared_step, float)
    indices = np.asarray(metadata['row_indices'], int)
    nominal = np.asarray(metadata['nominal_margins'], float)
    actual = np.asarray(actual_margins, float)
    if (tuple(row_ids) != metadata['row_ids'] or actual.shape != nominal.shape
            or matrix.shape != (len(lower), len(step))
            or not np.isfinite(actual).all() or not np.isfinite(step).all()
            or not np.isfinite(matrix).all() or not np.isfinite(lower).all()):
        raise ValueError('Same-candidate original physical signed-margin identities required')
    linear = matrix[indices] @ step
    predicted_margin = nominal + linear
    excess = np.maximum(-actual, 0.)
    predicted_excess = np.maximum(-predicted_margin, 0.)
    measurement = dict(
        schema='rinkspace.original-physical-signed-margin-SOC-measurement.v1',
        valid=bool(np.all(actual >= 0.)), certified=True, row_ids=list(row_ids),
        original_row_indices=indices.tolist(), actual_signed_margins=actual.tolist(),
        predicted_signed_margins=predicted_margin.tolist(),
        maximum_violation_working_m=float(excess.max(initial=0.)),
        total_violation_working_m=float(excess.sum()),
        predicted_maximum_violation_working_m=float(predicted_excess.max(initial=0.)),
        predicted_total_violation_working_m=float(predicted_excess.sum()),
        original_physical_domain_and_boundary_roundoff_unchanged=True,
        physical_working_m_not_combined_with_source_pixel_merits=True,
        numerical_padding_added=False)
    tightened = lower.copy()
    changes = []
    for local in np.flatnonzero(actual < 0.):
        row = int(indices[local])
        # For the same row A*d >= b, the measured physical feasibility
        # correction is A*d >= A*d_current - margin(q+d_current).
        # One representable upward rounding adds no physical offset.
        endpoint = float(linear[local] - actual[local])
        target = float(np.nextafter(endpoint, np.inf))
        tightened[row] = max(tightened[row], target)
        if tightened[row] > lower[row]:
            changes.append(dict(original_row=row, row_id=row_ids[local],
                actual_signed_margin_working_m=float(actual[local]),
                numerical_lower_before=float(lower[row]),
                numerical_lower_after=float(tightened[row]),
                exact_measured_feasibility_endpoint=endpoint))
    correction = None if not changes else dict(lower=tightened, report=dict(
        schema='rinkspace.original-physical-signed-margin-SOC-cut.v1',
        corrected_original_physical_rows=changes,
        original_physical_domain_and_QP_rows_unchanged=True,
        original_signed_margin_measurement=measurement,
        numerical_padding_added=False))
    return correction, measurement


def require_identity_trial_recovery(proposed, recovered):
    """A proposal's KKT cannot certify a silently changed final candidate."""
    proposed = np.asarray(proposed, float)
    recovered = np.asarray(recovered, float)
    if (proposed.shape != recovered.shape or not np.isfinite(recovered).all()
            or not np.array_equal(proposed, recovered)):
        changed = (np.flatnonzero(proposed != recovered).tolist()
            if proposed.shape == recovered.shape else None)
        raise ValueError('Final trial recovery is not identity; proposal QP KKT cannot certify '
            'the changed candidate; changed parameters=' + str(changed))

