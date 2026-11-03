"""Finite ice penalties with explicit deadzones, scales and objective weights.

These rows remain in the optimization objective. They carry no multipliers,
feasibility decisions, contact classification or built-in acceptance thresholds.
"""
from dataclasses import dataclass, fields

import numpy as np


@dataclass(frozen=True)
class IceSoftPolicy:
    """Caller-selected engineering objective, in metres and finite weights.

    Deadzones are nonnegative; scales and objective weights are positive.
    No field has a default. These choices are not calibrated likelihoods or
    accuracy/acceptance thresholds, and do not change source observation weights.
    """
    penetration_deadzone_m: float
    penetration_scale_m: float
    penetration_weight: float
    support_hover_deadzone_m: float
    support_scale_m: float
    support_weight: float

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, (bool, np.bool_)) or np.ndim(value) != 0:
                raise ValueError(field.name+' must be a finite scalar')
            try:
                value = float(value)
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError(field.name+' must be a finite scalar') from error
            deadzone = field.name.endswith('deadzone_m')
            if not np.isfinite(value) or (value < 0 if deadzone else value <= 0):
                raise ValueError(field.name+' must be finite and '+('nonnegative' if deadzone else 'positive'))
            object.__setattr__(self, field.name, value)


def _vector(contact, key, length=None):
    value = np.asarray(contact[key], dtype=float)
    if value.ndim != 1 or not np.isfinite(value).all():
        raise ValueError(key+' must be a finite one-dimensional array')
    if length is not None and len(value) != length:
        raise ValueError(key+' must match margin_m')
    return value


def _raw_witness_height(contact, index):
    """Recover a raw geometric height only when its exact witness is exposed.

    Nonpenetration values/margins already include semantic allowances. The
    actual constraint packet also exposes the chosen local point and raw
    feature heights, so no allowance values or comparison tolerances are needed.
    Minimal packets without that metadata explicitly report a missing height.
    """
    features = contact.get('features')
    rows = contact.get('rows')
    if features is None or rows is None or 'height' not in features or 'local_point' not in features:
        return np.nan, 'raw_witness_metadata_unavailable'
    frame, foot, point, _ = rows[index]
    points = np.asarray(features['local_point'])[frame, foot]
    matches = np.flatnonzero(np.all(points == np.asarray(point), axis=-1))
    if not len(matches):
        return np.nan, 'exact_raw_witness_not_found'
    return float(np.asarray(features['height'])[frame, foot, matches[0]]), 'exact_feature_witness'


def ice_soft_rows(contact, policy):
    """Adapt constraints() ice rows to finite least-squares residuals.

    Nonpenetration margin_m is raw height + semantic allowance. Its residual
    uses max(-margin-deadzone, 0). Source support uses only each :upper row's
    raw height in values, with max(height-hover_deadzone, 0). The support lower
    row is omitted because universal nonpenetration supplies downward pressure.

    indices refer to original contact rows and include inactive/deadzone rows
    to preserve layout. derivative_per_margin_m is dr/d(margin in metres),
    negative for both active kinds: an upper margin decreases as height rises.
    A caller with dg/dx for g=margin/NORMALIZATION must multiply that Jacobian
    by NORMALIZATION before applying these derivatives. At the hinge, the
    selected derivative is zero. Unknown/airborne records cannot create support
    here: constraints() emits no support rows for those observations.
    """
    if not isinstance(policy, IceSoftPolicy):
        raise TypeError('policy must be an IceSoftPolicy')
    margins = _vector(contact, 'margin_m')
    weights = _vector(contact, 'weights', len(margins))
    values = _vector(contact, 'values', len(margins))
    labels = contact['labels']
    if len(labels) != len(margins) or np.any(weights <= 0):
        raise ValueError('labels and positive weights must match margin_m')
    indices = []
    kinds = []
    excess = []
    factors = []
    raw_heights = []
    raw_reasons = []
    reasons = []
    skipped = []
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        for index, label in enumerate(labels):
            if not isinstance(label, (tuple, list)) or len(label) != 2 or not isinstance(label[1], str):
                raise ValueError('labels must be (physical time id, row name) pairs')
            name = label[1]
            if name in ('left_nonpenetration', 'right_nonpenetration'):
                kind = 'ice_penetration'
                distance = -margins[index]-policy.penetration_deadzone_m
                factor = np.sqrt(weights[index])*np.sqrt(policy.penetration_weight)/policy.penetration_scale_m
                height, raw_reason = _raw_witness_height(contact, index)
                reason = 'penetration_soft_pressure' if distance > 0 else 'inside_penetration_deadzone'
            elif name.endswith(':upper'):
                kind = 'ice_support_hover'
                distance = values[index]-policy.support_hover_deadzone_m
                factor = np.sqrt(weights[index])*np.sqrt(policy.support_weight)/policy.support_scale_m
                height, raw_reason = float(values[index]), 'source_support_raw_height'
                reason = 'support_hover_soft_pressure' if distance > 0 else 'inside_support_hover_deadzone'
            else:
                reason = 'support_lower_covered_by_nonpenetration' if name.endswith(':lower') else 'not_an_ice_row'
                skipped.append(dict(index=index, label=label, reason=reason))
                continue
            indices.append(index)
            kinds.append(kind)
            excess.append(max(distance, 0.))
            factors.append(factor)
            raw_heights.append(height)
            raw_reasons.append(raw_reason)
            reasons.append(reason)
        indices = np.asarray(indices, dtype=int)
        excess = np.asarray(excess, dtype=float)
        factors = np.asarray(factors, dtype=float)
        residual = factors*excess
        derivative = np.where(excess > 0, -factors, 0.)
        kinds = np.asarray(kinds, dtype=str)
        parts = {kind: float(residual[kinds == kind]@residual[kinds == kind]/2)
                 for kind in ('ice_penetration', 'ice_support_hover')}
    return dict(residual=residual, indices=indices, derivative_per_margin_m=derivative, parts=parts,
        diagnostics=dict(labels=[labels[index] for index in indices], kinds=kinds.tolist(),
            margin_m=margins[indices].copy(), raw_height_m=np.asarray(raw_heights, dtype=float),
            raw_height_reasons=raw_reasons, excess_m=excess, reasons=reasons, skipped_rows=skipped,
            nonpenetration_margin_includes_semantic_allowance=True,
            interpretation='finite objective pressure; deadzones are not acceptance thresholds'))