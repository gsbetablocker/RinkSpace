"""Pure shadow domains for the existing truth-progress policy.

No source, support, historical guard cap, physical row, H/g, parameter or
acceptance callback is changed. These domains are for QP/value diagnostics,
never a replacement for the production common-fraction acceptance rule.
"""
from collections.abc import Mapping
import math

import numpy as np
from scipy.sparse import issparse


def _record_identity_memo(value, memo=None, visited=None):
    """Find canonical source records without traversing factor owner graphs."""
    memo = {} if memo is None else memo
    visited = set() if visited is None else visited
    if id(value) in visited:
        return memo
    visited.add(id(value))
    if isinstance(value, dict):
        for key, item in value.items():
            if key == 'record':
                memo[id(item)] = item
            else:
                _record_identity_memo(item, memo, visited)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _record_identity_memo(item, memo, visited)
    return memo


def _metadata_copy(value, memo=None):
    """Copy mutable metadata, preserving opaque original source-owner identity.

    The large QP matrix is passed separately by identity. Metadata arrays and
    sparse carrier maps own independent storage; source factors/callables and
    canonical context.record source objects are kept by identity. The native
    nominal-chart and station-projection public interfaces require that record
    identity. Source records remain caller-owned and read-only for this probe.
    """
    memo = _record_identity_memo(value) if memo is None else memo
    if id(value) in memo:
        return memo[id(value)]
    if isinstance(value, np.ndarray) or issparse(value):
        result = value.copy(); memo[id(value)] = result
        return result
    if isinstance(value, dict):
        result = {}; memo[id(value)] = result
        result.update((key, _metadata_copy(item, memo)) for key, item in value.items())
        return result
    if isinstance(value, list):
        result = []; memo[id(value)] = result
        result.extend(_metadata_copy(item, memo) for item in value)
        return result
    if isinstance(value, tuple):
        result = tuple(_metadata_copy(item, memo) for item in value)
        memo[id(value)] = result
        return result
    return value


def _alpha(value):
    alpha = float(value)
    if not math.isfinite(alpha) or not 0. < alpha <= 1.:
        raise ValueError('Diagnostic trial fraction must lie in (0, 1]')
    return alpha


def build_truth_progress_probe_domain(domain, *, current_group_violations,
        existing_group_limits=None, policy='preserve_current_violation',
        original_constraint_lower=None, row_group_index=None):
    """Remove only the common contraction from a fresh original driver domain.

    ``preserve_current_violation`` (alias ``current_nonworsening``) targets v.
    ``existing_caps`` targets the explicitly supplied original historical L.
    Source-zero groups and unowned/physical faces stay unchanged. Each trial
    fraction starts from this same pristine domain, without full-step SOC cuts.

    Prefer tapping b0 and ownership at contract_original_support_bounds in the
    runner: that path evaluates b0 - target*coefficient directly. Without the
    tap, the numerical rhs is reconstructed from the contracted rhs; the
    report explicitly does NOT claim bit-exact original-support recovery.
    All callbacks/guards that decide acceptance remain caller-owned.
    """
    aliases = {'current_nonworsening': 'preserve_current_violation'}
    policy = aliases.get(policy, policy)
    if policy not in ('preserve_current_violation', 'existing_caps'):
        raise ValueError('Explicit shadow policy required; use the original domain for common contraction')
    matrix_reference = domain['matrix']
    matrix = np.asarray(matrix_reference, float)
    lower = np.asarray(domain['lower'], float).copy()
    previous = np.asarray(current_group_violations, float).copy()
    if (matrix.ndim != 2 or lower.shape != (matrix.shape[0],) or previous.ndim != 1
            or not np.isfinite(matrix).all() or not np.isfinite(lower).all()
            or not np.isfinite(previous).all() or np.any(previous < 0.)):
        raise ValueError('Complete finite original driver domain and actual group violations required')
    priority = domain.get('report', {}).get('truth_feasibility_priority')
    if not isinstance(priority, Mapping) or priority.get('restoration_required') is not True:
        raise ValueError('A fresh original common-contraction domain is required')
    rho = float(priority['demanded_common_original_violation_fraction'])
    if not math.isfinite(rho) or not 0. <= rho < 1.:
        raise ValueError('Original strict common fraction must be authenticated, never fabricated')
    if not np.array_equal(previous, np.asarray(priority['current_original_group_violations_px'], float)):
        raise ValueError('Current violations differ from the original parent domain')
    positive = previous > 1e-8
    old_targets = previous.copy(); old_targets[positive] *= rho
    if not np.array_equal(old_targets, np.asarray(priority['demanded_group_violations_px'], float)):
        raise ValueError('Original priority target vector changed')
    owners = np.full(len(lower), -1, int)
    covered = np.zeros(len(lower), bool)
    spans = {}; offset = 0
    for family, response in domain['curvature_responses'].items():
        targets = np.asarray(response['dynamic_group_limits'], float)
        stop = offset + len(targets)
        begin, end = response['row_start'], response['row_stop']
        local = np.asarray(response['dynamic_row_group_index'])
        if (targets.ndim != 1 or not np.array_equal(targets, old_targets[offset:stop])
                or not isinstance(begin, (int, np.integer)) or not isinstance(end, (int, np.integer))
                or not 0 <= begin <= end <= len(lower) or np.any(covered[begin:end])
                or local.dtype.kind not in 'iu' or local.shape != (end-begin,)
                or np.any(local < -1) or np.any(local >= len(targets))):
            raise ValueError('Every original ordered source group and retained face must be owned exactly')
        owned = local >= 0
        owners[begin:end][owned] = offset + local[owned]
        covered[begin:end] = True
        spans[family] = (offset, stop)
        offset = stop
    if offset != len(previous):
        raise ValueError('All original truth groups must be present, without a filtered subset')
    supplied_limits = None
    if existing_group_limits is not None:
        supplied_limits = np.asarray(existing_group_limits, float).copy()
        if (supplied_limits.shape != previous.shape or not np.isfinite(supplied_limits).all()
                or np.any(supplied_limits < 0.) or np.any(previous > supplied_limits + 1e-8)):
            raise ValueError('The unchanged historical caps must admit the same parent by its original precision')
    if policy == 'existing_caps':
        if supplied_limits is None:
            raise ValueError('Existing caps must be explicitly supplied; never infer or enlarge them')
        if np.any(supplied_limits[~positive] != 0.):
            raise ValueError('Existing-caps probe would relax a currently protected source-zero group')
        new_targets = supplied_limits.copy()
    else:
        new_targets = previous.copy()
    coefficient = np.zeros(len(lower)); old_expansion = coefficient.copy(); new_expansion = coefficient.copy()
    owned_positive = owners >= 0
    owned_positive[owners >= 0] &= positive[owners[owners >= 0]]
    coefficient[owned_positive] = previous[owners[owned_positive]]
    old_expansion[owned_positive] = old_targets[owners[owned_positive]]
    new_expansion[owned_positive] = new_targets[owners[owned_positive]]
    tapped = original_constraint_lower is not None or row_group_index is not None
    if tapped:
        if original_constraint_lower is None or row_group_index is None:
            raise ValueError('Tap both the exact original b0 and row-group ownership together')
        original = np.asarray(original_constraint_lower, float).copy()
        tapped_owners = np.asarray(row_group_index)
        if (original.shape != lower.shape or not np.isfinite(original).all()
                or tapped_owners.dtype.kind not in 'iu' or not np.array_equal(tapped_owners, owners)
                or not np.array_equal(lower, original - rho * coefficient)):
            raise ValueError('Fresh complete contracted domain differs from tapped original b0/ownership')
        transformed = original - new_expansion
        new_lower = lower.copy()
        new_lower[owned_positive] = transformed[owned_positive]
        reconstruction = dict(exact_original_constraint_lower_tapped=True,
            original_support_reference_bitexact=True, numeric_reconstruction_used=False)
    else:
        original = lower + old_expansion
        transformed = lower - (new_expansion - old_expansion)
        new_lower = lower.copy()
        new_lower[owned_positive] = transformed[owned_positive]
        reconstruction = dict(exact_original_constraint_lower_tapped=False,
            original_support_reference_bitexact=False, numeric_reconstruction_used=True,
            original_b0_reconstruction_error_not_observable_without_tap=True,
            contracted_rhs_roundtrip_max=float(abs(original-old_expansion-lower).max(initial=0.)))
    protected_rows = ~owned_positive
    if new_lower[protected_rows].tobytes() != lower[protected_rows].tobytes():
        raise ValueError('Physical, unowned or original source-zero faces changed')
    original_fraction = domain.get('at_trial_fraction')
    if not callable(original_fraction):
        raise ValueError('Original pure trial-fraction domain is required to reject inherited SOC cuts')
    pristine = original_fraction(1.)
    if pristine['matrix'] is not matrix_reference or not np.array_equal(pristine['lower'], lower):
        raise ValueError('Only a pristine original domain is allowed, never accumulated full-step SOC cuts')
    signed = domain.get('original_signed_margin_rows')
    if signed is not None:
        indices = np.asarray(signed['row_indices'])
        if (indices.dtype.kind not in 'iu' or indices.ndim != 1
                or np.any(indices < 0) or np.any(indices >= len(lower))
                or np.any(owners[indices] >= 0)
                or new_lower[indices].tobytes() != lower[indices].tobytes()):
            raise ValueError('All original signed-margin physical rows must remain unowned and unchanged')
    # Do not copy the large A. Everything else numerical/mutable is isolated.
    base = {key: _metadata_copy(value) for key, value in domain.items()
            if key not in ('matrix', 'lower', 'at_trial_fraction', 'curvature_responses')}
    responses = _metadata_copy(domain['curvature_responses'])
    for family, response in responses.items():
        begin, end = spans[family]
        response['dynamic_group_limits'] = new_targets[begin:end].copy()
        response['dynamic_group_target_basis'] = 'diagnostic_' + policy + '_same_parent'
    base['report'] = dict(base.get('report', {}), truth_progress_policy_probe=dict(
        schema='rinkspace.diagnostic-truth-progress-domain-probe.v1', policy=policy,
        original_common_fraction=rho, original_parent_violations=previous.tolist(),
        probe_group_targets=new_targets.tolist(), historical_caps_changed=False,
        historical_caps_supplied=supplied_limits is not None,
        original_A_shared_by_identity=True, physical_unowned_and_zero_rows_bitexact=True,
        original_carrier_charts_sampling_and_signed_margins_unchanged=True,
        source_H_g_and_all_acceptance_callbacks_not_modified=True,
        strict_common_progress_acceptance_must_be_evaluated_separately=True,
        no_fake_common_fraction_used=True, **reconstruction))
    def at_trial_fraction(alpha):
        alpha = _alpha(alpha)
        child = _metadata_copy(base)
        child.update(matrix=matrix_reference, lower=new_lower.copy(),
            curvature_responses=_metadata_copy(responses))
        child['report']['trial_fraction_domain'] = dict(alpha=alpha, policy=policy,
            targets_constant_at_every_actual_trial_fraction=True,
            common_fraction_not_applied_or_fabricated=True,
            accumulated_full_step_SOC_cuts_reused=False,
            original_physical_and_zero_group_faces_unchanged=True)
        child['at_trial_fraction'] = at_trial_fraction
        return child
    return at_trial_fraction(1.)
