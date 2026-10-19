"""Two-stencil local column-space diagnostics; never compute a model update.

Inputs are original sigma-normalized source residuals/Jacobians and the same
nominal IRLS row weights.  Reported energies are this local quadratic metric,
not nonlinear Huber costs, pixel accuracy, or a feasible finite-change claim.
"""
from __future__ import annotations

import numpy as np
from scipy import linalg


def _norm2(a):
    return float(linalg.svdvals(a, check_finite=False)[0]) if min(a.shape) else 0.


def _project(u, x):
    return u @ (u.T @ x) if u.shape[1] else np.zeros_like(x)


def _energy(x):
    return float(np.sum(np.asarray(x, float) ** 2))


def _svd(a):
    if not min(a.shape):
        return np.empty((a.shape[0], 0)), np.empty(0)
    u, s, _ = linalg.svd(a, full_matrices=False, check_finite=False)
    return u, s


def _paired_space(first, second, label):
    u1, s1 = _svd(first)
    u2, s2 = _svd(second)
    eps = np.finfo(float).eps
    tau1 = eps * max(first.shape) * (float(s1[0]) if len(s1) else 0.)
    tau2 = eps * max(second.shape) * (float(s2[0]) if len(s2) else 0.)
    difference_norm = _norm2(first - second)
    # This is an observed change of the two numerical operators, not a
    # proven bound on either operator's error relative to an exact derivative.
    sensitivity_level = max(tau1, tau2, difference_norm)
    conservative_k1 = int(np.count_nonzero(s1 > sensitivity_level))
    conservative_k2 = int(np.count_nonzero(s2 > sensitivity_level))
    # Compare response and observed stencil change per parameter combination.
    # A global difference norm after column equilibration can otherwise let
    # one tiny noise-only column suppress unrelated, well-agreeing columns.
    mean = .5 * (first + second)
    change = first - second
    stack = np.vstack((mean, change))
    ust, sst = _svd(stack)
    stack_tau = eps * max(stack.shape) * (float(sst[0]) if len(sst) else 0.)
    stack_rank = int(np.count_nonzero(sst > stack_tau))
    # U blocks equal mean*T and change*T for one common right-whitening T.
    # We never construct that inverse transform or a parameter solution.
    mean_modes, change_modes = ust[:len(first), :stack_rank], ust[len(first):, :stack_rank]
    if stack_rank:
        agreement = mean_modes.T @ mean_modes - change_modes.T @ change_modes
        agreement = .5 * (agreement + agreement.T)
        eigenvalues, eigenvectors = linalg.eigh(agreement, check_finite=False)
        eigenvalues, eigenvectors = eigenvalues[::-1], eigenvectors[:, ::-1]
        agreement_tau = eps * max(stack.shape) * max(1., float(np.max(np.abs(eigenvalues))))
        keep = eigenvalues > agreement_tau
        first_modes = (mean_modes + .5 * change_modes) @ eigenvectors[:, keep]
        second_modes = (mean_modes - .5 * change_modes) @ eigenvectors[:, keep]
    else:
        eigenvalues = np.empty(0)
        agreement_tau = 0.
        keep = np.empty(0, dtype=bool)
        first_modes = second_modes = np.empty((len(first), 0))
    selected1, sm1 = _svd(first_modes)
    selected2, sm2 = _svd(second_modes)
    mt1 = eps * max(first_modes.shape) * (float(sm1[0]) if len(sm1) else 0.)
    mt2 = eps * max(second_modes.shape) * (float(sm2[0]) if len(sm2) else 0.)
    k1, k2 = int(np.count_nonzero(sm1 > mt1)), int(np.count_nonzero(sm2 > mt2))
    selected1, selected2 = selected1[:, :k1], selected2[:, :k2]
    overlap = linalg.svdvals(selected1.T @ selected2, check_finite=False)
    smallest_overlap = float(np.min(overlap)) if len(overlap) else 1.
    projector_change = (1. if k1 != k2 else
        float(np.sqrt(max(0., 1. - min(1., smallest_overlap) ** 2))))
    return dict(first=selected1, second=selected2,
        first_machine=u1[:, :int(np.count_nonzero(s1 > tau1))],
        second_machine=u2[:, :int(np.count_nonzero(s2 > tau2))],
        report=dict(label=label, shape=list(first.shape),
            singular_values_first=s1.tolist(), singular_values_second=s2.tolist(),
            machine_rank_first=int(np.count_nonzero(s1 > tau1)),
            machine_rank_second=int(np.count_nonzero(s2 > tau2)),
            machine_threshold_first=tau1, machine_threshold_second=tau2,
            two_stencil_operator_difference_spectral_norm=difference_norm,
            common_empirical_sensitivity_level=sensitivity_level,
            conservative_rank_above_global_operator_change_first=conservative_k1,
            conservative_rank_above_global_operator_change_second=conservative_k2,
            empirical_combination_agreement_rank_first=k1,
            empirical_combination_agreement_rank_second=k2,
            mean_and_change_stacked_singular_values=sst.tolist(),
            mean_and_change_stack_machine_threshold=stack_tau,
            mean_and_change_stack_numerical_rank=stack_rank,
            numerical_null_combinations_reported_not_removed_from_model=stack.shape[1] - stack_rank,
            combination_response_minus_change_energy_eigenvalues=eigenvalues.tolist(),
            combination_agreement_arithmetic_comparison_scale=agreement_tau,
            combination_agreement_arithmetically_ambiguous_modes=int(np.count_nonzero(np.abs(eigenvalues) <= agreement_tau)),
            selected_combination_criterion='Mean response energy exceeds observed two-stencil change energy; zero boundary is independent of residual',
            derivative_bias_shared_by_both_stencils_is_not_detectable_here=True,
            selected_projector_change_spectral_norm=projector_change,
            selected_principal_angle_cosines_between_stencils=overlap.tolist(),
            sensitivity_level_is_not_a_proven_exact_derivative_error_bound=True,
            no_rank_threshold_selected_from_residual_reduction=True))


def analyze_subspaces(r, J_camera_h, J_geometry_h, J_nuisance_h,
        J_camera_half, J_geometry_half, J_nuisance_half, row_weights,
        row_group_index=None, column_scales=None):
    """Return a compact summary and row arrays, without a parameter solution.

    ``column_scales`` may map camera/geometry/nuisance to positive arrays.
    Otherwise use inverse maximum column norm across both weighted stencils.
    The same invertible scaling is used for both stencils. Zero columns keep
    scale one; their original norms and two-stencil differences are reported.
    Dense local matrices only; callers retain canonical-column provenance.
    """
    residual = np.asarray(r, dtype=float)
    weights = np.asarray(row_weights, dtype=float)
    if (residual.ndim != 1 or weights.shape != residual.shape
            or not np.isfinite(residual).all() or not np.isfinite(weights).all()
            or np.any(weights < 0.)):
        raise ValueError('Finite original residual and nonnegative nominal IRLS weights required')
    sqrt_weights = np.sqrt(weights)
    weighted_r = residual * sqrt_weights
    supplied = dict(camera=(J_camera_h, J_camera_half),
                    geometry=(J_geometry_h, J_geometry_half),
                    nuisance=(J_nuisance_h, J_nuisance_half))
    matrices, scaling_report, scaling_arrays = {}, {}, {}
    for label, pair in supplied.items():
        first, second = (np.asarray(x, float) for x in pair)
        if (first.ndim != 2 or first.shape != second.shape or first.shape[0] != len(residual)
                or not np.isfinite(first).all() or not np.isfinite(second).all()):
            raise ValueError('Two finite same-chart local matrices required for ' + label)
        first, second = first * sqrt_weights[:, None], second * sqrt_weights[:, None]
        norm1, norm2 = np.linalg.norm(first, axis=0), np.linalg.norm(second, axis=0)
        maximum = np.maximum(norm1, norm2)
        if column_scales is None:
            scale = np.divide(1., maximum, out=np.ones_like(maximum), where=maximum != 0.)
        else:
            scale = np.asarray(column_scales[label], float)
        if scale.shape != maximum.shape or not np.isfinite(scale).all() or np.any(scale <= 0.):
            raise ValueError('One finite positive common scale per original column required')
        matrices[label] = (first * scale, second * scale)
        scaling_arrays[label + '_common_column_scale'] = scale
        scaling_arrays[label + '_weighted_column_norm_first'] = norm1
        scaling_arrays[label + '_weighted_column_norm_second'] = norm2
        scaling_arrays[label + '_weighted_column_difference_norm'] = np.linalg.norm(first - second, axis=0)
        scaling_report[label] = dict(columns=len(scale), zero_in_both=int(np.count_nonzero(maximum == 0.)),
            one_stencil_only_nonzero=int(np.count_nonzero((norm1 == 0.) != (norm2 == 0.))),
            weighted_original_coordinate_singular_values_first=linalg.svdvals(first, check_finite=False).tolist(),
            weighted_original_coordinate_singular_values_second=linalg.svdvals(second, check_finite=False).tolist(),
            common_invertible_scaling=True,
            tiny_columns_not_interpreted_as_information_by_normalization_alone=True)

    nuisance = _paired_space(*matrices['nuisance'], 'observation_nuisance')
    spaces = {'nuisance': nuisance}
    prepared, residuals = {}, {}
    for index, name in enumerate(('first', 'second')):
        un = nuisance[name]
        residuals[name] = weighted_r - _project(un, weighted_r)
        prepared[name] = {
            label: matrices[label][index] - _project(un, matrices[label][index])
            for label in ('camera', 'geometry')}
    for label in ('camera', 'geometry'):
        spaces[label] = _paired_space(prepared['first'][label], prepared['second'][label],
                                      label + '_after_nuisance')
    spaces['joint'] = _paired_space(
        np.column_stack((prepared['first']['camera'], prepared['first']['geometry'])),
        np.column_stack((prepared['second']['camera'], prepared['second']['geometry'])),
        'camera_and_geometry_after_nuisance')
    incremental = {}
    for name in ('first', 'second'):
        incremental[name] = prepared[name]['geometry'] - _project(spaces['camera'][name], prepared[name]['geometry'])
    spaces['geometry_after_camera'] = _paired_space(incremental['first'], incremental['second'],
                                                    'geometry_after_camera_and_nuisance')

    arrays = dict(original_residual=residual, nominal_IRLS_row_weights=weights,
                  weighted_residual=weighted_r, **scaling_arrays)
    outcomes = {}
    for name in ('first', 'second'):
        rn = residuals[name]
        camera_remaining = rn - _project(spaces['camera'][name], rn)
        geometry_remaining = rn - _project(spaces['geometry'][name], rn)
        joint_remaining = rn - _project(spaces['joint'][name], rn)
        geometry_after_camera_component = _project(spaces['geometry_after_camera'][name], camera_remaining)
        sequential_remaining = camera_remaining - geometry_after_camera_component
        arrays.update({name + '_nuisance_component': weighted_r - rn,
            name + '_camera_only_remaining': camera_remaining,
            name + '_geometry_only_remaining': geometry_remaining,
            name + '_joint_remaining': joint_remaining,
            name + '_geometry_after_camera_component': geometry_after_camera_component,
            name + '_sequential_remaining': sequential_remaining})
        # Machine-only projections are explicitly diagnostic; empirical
        # sensitivity selection above is used for the row attribution arrays.
        machine_remaining = rn - _project(spaces['joint'][name + '_machine'], rn)
        angles = linalg.svdvals(spaces['camera'][name].T @ spaces['geometry'][name], check_finite=False)
        outcomes[name] = dict(
            weighted_residual_energy=_energy(weighted_r),
            after_nuisance_energy=_energy(rn),
            camera_only_remaining_energy=_energy(camera_remaining),
            geometry_only_remaining_energy=_energy(geometry_remaining),
            joint_remaining_energy=_energy(joint_remaining),
            machine_only_joint_remaining_energy_not_trusted=_energy(machine_remaining),
            geometry_after_camera_component_energy=_energy(geometry_after_camera_component),
            sequential_remaining_energy=_energy(sequential_remaining),
            sequential_vs_direct_joint_remaining_difference_norm=float(np.linalg.norm(sequential_remaining - joint_remaining)),
            camera_geometry_principal_angle_cosines=angles.tolist(),
            maximum_joint_explained_remaining_inner_product=abs(float((rn - joint_remaining) @ joint_remaining)),
            joint_energy_partition_absolute_error=abs(_energy(rn) - _energy(rn - joint_remaining) - _energy(joint_remaining)))
    stability = {}
    for label in ('camera_only_remaining', 'geometry_only_remaining', 'joint_remaining', 'geometry_after_camera_component'):
        first, second = arrays['first_' + label], arrays['second_' + label]
        stability[label] = dict(two_stencil_difference_norm=float(np.linalg.norm(first - second)),
            two_stencil_maximum_absolute_row_difference=float(np.max(np.abs(first - second))) if len(first) else 0.)

    groups = []
    if row_group_index is not None:
        ownership = np.asarray(row_group_index)
        if ownership.shape != residual.shape:
            raise ValueError('Every original scalar row needs one provenance group')
        arrays['row_group_index'] = ownership
        for group in np.unique(ownership):
            mask = ownership == group
            record = dict(group=group.item() if isinstance(group, np.generic) else group,
                          scalar_rows=int(np.count_nonzero(mask)), weighted_original_energy=_energy(weighted_r[mask]))
            for name in ('first', 'second'):
                record[name] = {label + '_energy': _energy(arrays[name + '_' + label][mask]) for label in
                    ('camera_only_remaining', 'geometry_only_remaining', 'joint_remaining', 'geometry_after_camera_component')}
            record['joint_two_stencil_difference_norm'] = float(np.linalg.norm(
                arrays['first_joint_remaining'][mask] - arrays['second_joint_remaining'][mask]))
            groups.append(record)
    summary = dict(schema='rinkspace.two-stencil-local-absolute-column-spaces.v1',
        scalar_rows=len(residual), positive_weight_rows=int(np.count_nonzero(weights)),
        metric='Original sigma-normalized residual, once multiplied by sqrt of the same nominal IRLS weight',
        metrics_are_local_quadratic_energies_not_original_nonlinear_cost=True,
        scaling=scaling_report, spaces={label:space['report'] for label,space in spaces.items()},
        outcomes=outcomes, residual_attribution_two_stencil_stability=stability, groups=groups,
        limitations=[
            'Observed stencil differences measure sensitivity; they are not rigorous exact-derivative error bounds.',
            'Combination agreement cannot detect a derivative bias shared by both stencils or certify a true model rank.',
            'Machine precision rank alone does not establish a reliable explanatory direction.',
            'Column-space projection is unconstrained and local; no finite physically feasible update is computed.',
            'Shared camera/geometry subspaces do not identify the physical cause of their overlapping response.',
            'Sequential allocation depends on ordering and empirical sensitivity; compare the direct joint projection.',
            'Omitted observations/constraints can restrict updates. This is not a full-source identifiability certificate.'],
        parameter_solution_computed=False, candidate_model_created=False)
    return dict(summary=summary, arrays=arrays)
