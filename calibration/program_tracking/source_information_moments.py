"""Read-only quadrature diagnostics for already normalized source r/J.

An exact finite set of camera moments does not authorize source admission or
certify world geometry, nonlinear trials outside that set, or solver speed.
"""
from __future__ import annotations

import numpy as np

from calibration.solver_runtime.source_loss import source_irls_weights


def production_point_moments(residual, jacobian, *, quadratic=False):
    """Per measured point cost, gradient and upper normal from production r/J.

    ``absolute_value_jac15`` already divides by the original working sigma.
    Sigma must not be applied a second time. Integration weights multiply these
    moments outside the unchanged component-wise production Huber2 loss.
    """
    residual = np.asarray(residual, float)
    jacobian = np.asarray(jacobian, float)
    if (residual.ndim != 1 or not len(residual) or len(residual) % 2
            or jacobian.ndim != 2 or len(jacobian) != len(residual)
            or not np.isfinite(residual).all() or not np.isfinite(jacobian).all()):
        raise ValueError('Finite already normalized original two-scalar source r/J required')
    mask = np.full(len(residual), quadratic, bool)
    robust = source_irls_weights(residual, mask)
    absolute = abs(residual)
    cost = np.where(absolute <= 2., .5 * residual**2, 2. * absolute - 2.)
    cost[mask] = .5 * residual[mask]**2
    n, parameters = len(residual) // 2, jacobian.shape[1]
    r = residual.reshape(n, 2)
    j = jacobian.reshape(n, 2, parameters)
    w = robust.reshape(n, 2)
    gradient = np.einsum('nc,ncp->np', w*r, j)
    normal = np.einsum('nc,ncp,ncq->npq', w, j, j)
    upper = np.triu_indices(parameters)
    return np.column_stack((cost.reshape(n, 2).sum(axis=1), gradient,
        normal[:, upper[0], upper[1]]))


def reduce_nonnegative_exact_moments(moment, original_weights, protected):
    """Eliminate numerical null directions without a requested count target.

    Protected source shape/extent stations retain their literal input weights.
    Every accepted step must preserve all supplied moments within floating
    point accumulation error; a numerical rank only proposes a removal.
    """
    moment = np.asarray(moment, float)
    original = np.asarray(original_weights, float)
    protected = np.asarray(protected, bool)
    if (moment.ndim != 2 or not len(moment) or original.shape != (len(moment),)
            or protected.shape != original.shape or not np.isfinite(moment).all()
            or not np.isfinite(original).all() or np.any(original <= 0)):
        raise ValueError('Finite moments, positive original weights and source protection required')
    weights = original.copy()
    scale = np.maximum(np.max(abs(moment), axis=0), np.finfo(float).tiny)
    scaled = moment / scale
    original_absolute_sum = np.sum(abs(original[:, None]*scaled), axis=0)
    accumulation_bound = 64*np.finfo(float).eps*len(weights)*original_absolute_sum
    active = np.ones(len(weights), bool)
    history = []
    termination = 'no_unprotected_source_stations'
    while True:
        available = np.flatnonzero(active & ~protected)
        if not len(available):
            break
        matrix = scaled[available].T
        _, singular, vt = np.linalg.svd(matrix,
            full_matrices=matrix.shape[0] < matrix.shape[1])
        rank_tolerance = np.finfo(float).eps*max(matrix.shape)*singular[0]
        rank = int(np.count_nonzero(singular > rank_tolerance))
        history.append(dict(unprotected_points=len(available), numerical_rank=rank,
            floating_point_rank_tolerance=float(rank_tolerance)))
        if rank == len(available):
            termination = 'no_numerical_null_direction'
            break
        direction = vt[-1]
        if not np.any(direction > 0):
            direction = -direction
        positive = direction > 0
        steps = np.where(positive, weights[available] / np.where(positive, direction, 1.), np.inf)
        victim_position = int(np.argmin(steps))
        candidate = weights.copy()
        candidate[available] -= steps[victim_position]*direction
        victim = available[victim_position]
        candidate[victim] = 0.
        if np.any(candidate < 0):
            termination = 'floating_point_positivity_check_failed'
            break
        if np.any(abs((candidate-original) @ scaled) > accumulation_bound):
            termination = 'floating_point_moment_preservation_check_failed'
            break
        weights = candidate
        active[victim] = False
    if not np.array_equal(weights[protected], original[protected]):
        raise ValueError('Protected source weights changed')
    return weights, dict(termination=termination, rank_history=history,
        protected_weights_bit_exact=True, no_point_count_target=True,
        maximal_scaled_absolute_moment_error=float(np.max(abs((weights-original) @ scaled))),
        maximal_scaled_floating_point_accumulation_bound=float(np.max(accumulation_bound)))


def moment_comparison(moment, original_weights, reduced_weights, parameter_count):
    moment = np.asarray(moment, float)
    original = np.asarray(original_weights, float)
    reduced = np.asarray(reduced_weights, float)
    full, compact = original @ moment, reduced @ moment
    absolute_sum = np.maximum(np.sum(abs(original[:, None]*moment), axis=0), np.finfo(float).tiny)
    gradient = slice(1, 1+parameter_count)
    normal = slice(1+parameter_count, None)
    return dict(cost_relative_difference=float(abs(compact[0]-full[0])/max(abs(full[0]),np.finfo(float).tiny)),
        gradient_relative_L2_difference=float(np.linalg.norm(compact[gradient]-full[gradient])/max(np.linalg.norm(full[gradient]),np.finfo(float).tiny)),
        normal_upper_relative_L2_difference=float(np.linalg.norm(compact[normal]-full[normal])/max(np.linalg.norm(full[normal]),np.finfo(float).tiny)),
        maximum_component_relative_absolute_sum_difference=float(np.max(abs(compact-full)/absolute_sum)))
