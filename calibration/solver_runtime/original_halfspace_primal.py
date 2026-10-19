"""Per-original-row floating-point feasibility, independent of other faces."""
from __future__ import annotations
import numpy as np

PRIMAL_CERTIFICATE_SCHEMA='rinkspace.per-original-half-space-primal-certificate.v1'


def original_full_qp_kkt_verified(report, candidate):
    """Accept either backend or refined dual provenance under the same gates."""
    if not isinstance(report,dict):return False
    primal=report.get('complete_original_primal',{})
    if not isinstance(primal,dict):return False
    try:
        checked=int(primal['all_original_rows_checked']);required=int(report['linear_constraint_count'])
        if checked!=primal['all_original_rows_checked']or required!=report['linear_constraint_count']:
            return False
    except (KeyError,TypeError,ValueError,OverflowError):return False
    dual=(report.get('complete_original_nonnegative_dual_certificate_checked') is True
        or report.get('complete_original_nonnegative_IPM_dual_certificate_checked') is True)
    if not(dual and report.get('complete_constraint_rows_retained') is True
        and report.get('original_H_gradient_constraints_bounds_and_KKT_gate_retained') is True
        and primal.get('accepted') is True and primal.get('violated_original_rows')==0
        and 0<=required<=checked):
        return False
    try:
        value=np.asarray(candidate,float)
        numbers=np.asarray([report['free_stationarity_max_absolute'],report['kkt_tolerance'],
            report['original_unit_complementarity_absolute_sum'],report['original_unit_complementarity_tolerance']],float)
        return bool(value.ndim==1 and len(value) and np.isfinite(value).all()
            and np.isfinite(numbers).all() and np.all(numbers>=0.)
            and numbers[0]<=numbers[1] and numbers[2]<=numbers[3])
    except (KeyError,TypeError,ValueError):return False


def original_halfspace_primal(rows, targets, candidate, *, slack=None):
    """Certify A x >= b in the caller's original units.

    A remote loose face and an unconstrained quadratic center cannot enlarge
    another face's threshold. The absolute term is numerical precision, not
    a support expansion. The remaining allowance bounds this row's dot/subtract
    rounding; no source or metric scaling enters it.
    """
    A=np.asarray(rows,float);b=np.asarray(targets,float);x=np.asarray(candidate,float)
    if A.ndim!=2 or b.shape!=(len(A),) or x.shape!=(A.shape[1],):
        raise ValueError('Original ordered half-space matrix, targets and candidate required')
    if not np.isfinite(A).all() or not np.isfinite(b).all() or not np.isfinite(x).all():
        raise ValueError('Finite original half-spaces and candidate required')
    actual=A@x-b if slack is None else np.asarray(slack,float)
    if actual.shape!=b.shape or not np.isfinite(actual).all():
        raise ValueError('Finite original row slacks required')
    eps=np.finfo(float).eps
    operations=2*A.shape[1]+4
    gamma=operations*eps/(1.-operations*eps)
    magnitude=np.abs(A)@np.abs(x)+np.abs(b)
    tolerance=1e-10+gamma*magnitude
    violation=actual < -tolerance
    excess=np.maximum(-actual-tolerance,0.)
    worst=int(np.argmax(excess)) if violation.any() else None
    report=dict(schema=PRIMAL_CERTIFICATE_SCHEMA,
        accepted=bool(not violation.any()),all_original_rows_checked=int(len(b)),
        violated_original_rows=int(violation.sum()),
        minimum_original_slack=float(actual.min()) if len(b) else 0.,
        minimum_original_slack_row=int(np.argmin(actual)) if len(b) else None,
        maximum_original_primal_excess=float(excess.max()) if len(b) else 0.,
        worst_original_row=worst,
        worst_original_row_slack=None if worst is None else float(actual[worst]),
        worst_original_row_tolerance=None if worst is None else float(tolerance[worst]),
        minimum_original_row_tolerance=float(tolerance.min()) if len(b) else 0.,
        maximum_original_row_tolerance=float(tolerance.max()) if len(b) else 0.,
        absolute_numerical_precision=1e-10,
        dot_subtraction_roundoff_gamma=float(gamma),
        tolerance_uses_other_rows_or_unconstrained_center=False,
        original_supports_targets_and_coordinates_unchanged=True)
    return actual,tolerance,violation,report
