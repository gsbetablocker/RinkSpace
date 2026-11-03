"""Small augmented-Lagrangian controller for contact INEQUALITIES.

Convention: every constraint g >= 0 is feasible. Supply fixed, normalized
rows during each inner solve. Examples: (clearance + ice_allowance)/scale,
(contact_tolerance - support_gap)/scale and its lower-band counterpart.
This module does not run a fit, move a root, classify contact or change a
camera. It augments the existing residual/Jacobian callback.
"""
from dataclasses import dataclass
import numpy as np


def _vector(value, name):
    value = np.asarray(value, dtype=float)
    if value.ndim != 1 or not np.isfinite(value).all():
        raise ValueError(f'{name} must be a finite one-dimensional array')
    return value


@dataclass
class ContactAL:
    """Algorithmic penalty/multipliers; they are not physical forces."""
    multiplier: np.ndarray
    penalty: float = 1.0
    previous_violation: float | None = None

    def __post_init__(self):
        self.multiplier = _vector(self.multiplier, 'multiplier').copy()
        if np.any(self.multiplier < 0):
            raise ValueError('Inequality multipliers must be nonnegative')
        if not np.isfinite(self.penalty) or self.penalty <= 0:
            raise ValueError('penalty must be finite and positive')

    @classmethod
    def zeros(cls, number_of_rows, penalty=1.0):
        if int(number_of_rows) != number_of_rows or number_of_rows < 0:
            raise ValueError('number_of_rows must be a nonnegative integer')
        return cls(np.zeros(int(number_of_rows)), penalty)

    def _checked(self, g):
        g = _vector(g, 'g')
        if g.shape != self.multiplier.shape:
            raise ValueError('Constraint rows changed; rebuild multipliers for the new row identities')
        return g

    def residual_and_jacobian(self, g, jacobian=None, *, row_weights=None):
        """Return AL least-squares rows and optional dg/dx chain.

        The omitted multiplier-only constant does not affect the inner
        minimizer. Compare data cost and physical feasibility separately
        across outer updates; AL costs have different offsets each round.
        At a hinge switch we select the inactive subgradient.
        Optional positive row_weights are fixed time-quadrature weights for
        the inner solve. Multipliers are densities under that measure, so
        update_after_inner_solve still receives unweighted g. Physical
        acceptance always uses unweighted metre margins.
        """
        g = self._checked(g)
        weights = np.ones_like(g) if row_weights is None else _vector(row_weights, 'row_weights')
        if weights.shape != g.shape or np.any(weights <= 0):
            raise ValueError('row_weights must match constraints and be strictly positive')
        root_weights = np.sqrt(weights)
        root = np.sqrt(self.penalty)
        shifted = self.multiplier / root - root * g
        residual = np.maximum(shifted, 0.0) * root_weights
        if jacobian is None:
            return residual, None
        if jacobian.shape[0] != len(g):
            raise ValueError('Jacobian row count differs from the constraint count')
        factors = np.where(shifted > 0.0, -root, 0.0) * root_weights
        if hasattr(jacobian, 'tocsr'):
            result = jacobian.multiply(factors[:, None]).tocsr()
        else:
            result = np.asarray(jacobian, float) * factors[:, None]
        return residual, result

    def update_after_inner_solve(self, g, *, required_reduction=.5, growth=4.0):
        """Update only after accepting an INNER solution, never per trial.

        No convergence or physical-validity claim is made here. A caller must
        inspect raw metre violations and the inner solver's stationarity.
        Reclassification or geometry-row changes require fresh row identities.
        """
        if not 0 < required_reduction < 1 or growth <= 1:
            raise ValueError('Invalid reduction/growth parameters')
        g = self._checked(g)
        violation = float(np.max(np.maximum(-g, 0.0), initial=0.0))
        self.multiplier = np.maximum(0.0, self.multiplier - self.penalty*g)
        increase = (self.previous_violation is not None and violation >
                    required_reduction*self.previous_violation)
        if increase:
            penalty = self.penalty*growth
            if not np.isfinite(penalty):
                raise FloatingPointError('Penalty overflow; report contact restoration failure')
            self.penalty = penalty
        self.previous_violation = violation
        return dict(max_normalized_violation=violation,
                    penalty=float(self.penalty), penalty_increased=bool(increase))


def physical_feasibility(inequality_margin_m, *, tolerance_m, row_labels=None):
    """Maximum-violation gate; averages cannot hide a bad exposure.

    inequality_margin_m already includes the configured ice allowance or
    support band. tolerance_m is only the numerical feasibility tolerance.
    Pass every tested native frame/exposure node, not only fitted landmarks.
    Temporal gaps between these nodes still require separate verification.
    """
    margin = _vector(inequality_margin_m, 'inequality_margin_m')
    if not np.isfinite(tolerance_m) or tolerance_m < 0:
        raise ValueError('tolerance_m must be finite and nonnegative')
    if row_labels is not None and len(row_labels) != len(margin):
        raise ValueError('row_labels length differs from constraints')
    violation = np.maximum(-margin, 0.0)
    worst = int(np.argmax(violation)) if len(violation) else None
    maximum = float(np.max(violation, initial=0.0))
    return dict(feasible=maximum <= tolerance_m,
                max_violation_m=maximum,
                rows_over_tolerance=int(np.sum(violation > tolerance_m)),
                tested_rows=len(margin), worst_row=worst,
                worst_label=None if worst is None or row_labels is None else row_labels[worst])


def contact_inequalities(contact_result, *, normalization_m, value_jacobian=None):
    """Adapt skate_contact.frame_residuals to g >= 0 without losing bounds.

    Use skate_contact.constraint_jacobian, NOT contact_jacobian, for the
    optional value_jacobian. A currently feasible row still needs a derivative
    after a nonzero AL multiplier is accumulated. Keep these label identities
    fixed within an inner solve; a change in the active geometric witness does
    not by itself change the row identity.
    """
    scale=float(normalization_m)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('normalization_m must be finite and positive')
    rows=contact_result['rows'];values=[];indices=[];signs=[];labels=[]
    for i,row in enumerate(rows):
        value=float(row['constraint_value_m'])
        lower=row['constraint_lower_m'];upper=row['constraint_upper_m']
        if lower is not None:
            values.append(value-float(lower));indices.append(i);signs.append(1.)
            labels.append(row['label']+':lower')
        if upper is not None:
            values.append(float(upper)-value);indices.append(i);signs.append(-1.)
            labels.append(row['label']+':upper')
    margin=_vector(values,'constraint margins')
    J=None
    if value_jacobian is not None:
        if value_jacobian.shape[0] != len(rows):
            raise ValueError('value_jacobian does not match original contact rows')
        J=value_jacobian[np.asarray(indices,dtype=int)]
        factors=np.asarray(signs)/scale
        J=(J.multiply(factors[:,None]).tocsr() if hasattr(J,'tocsr')
           else np.asarray(J)*factors[:,None])
    return dict(g=margin/scale,margin_m=margin,jacobian=J,labels=labels)
