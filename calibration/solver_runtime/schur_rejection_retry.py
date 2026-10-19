"""Retry a rejected exact-source Schur direction with the existing LM rule.

This opt-in AST wrapper preserves the frozen solver. Only the numerical trial
search changes: the current source normal, profiles and guards stay unchanged.
"""
from __future__ import annotations

import ast
import numpy as np


def rejected_direction_damping(*, damping, before, q, step, normal, gradient):
    """Use the original lambda*10 rejection rule and acceptance precision.

    There is no iteration cap. A source-model change smaller than the existing
    source-acceptance resolution, an unrepresentable step, or finite lambda
    exhaustion terminates this numerical search without claiming convergence.
    """
    damping = float(damping)
    q, step, gradient = np.asarray(q), np.asarray(step), np.asarray(gradient)
    first = float(gradient @ step)
    quadratic = .5 * float(step @ (normal @ step))
    precision = 1e-10 * max(1., float(before))
    report = dict(previous_damping=damping,
        source_gradient_change=first, source_quadratic_change=quadratic,
        original_source_acceptance_precision=precision,
        same_current_q_r_J_H_and_frozen_profiles=True,
        original_objective_and_physical_guards_unchanged=True)
    if not np.isfinite(damping) or damping <= 0.:
        report.update(retry=False, reason='positive_finite_LM_damping_required')
    elif not np.isfinite(step).all() or not np.isfinite(first + quadratic):
        report.update(retry=False, reason='nonfinite_rejected_source_direction')
    elif np.array_equal(q + step, q):
        report.update(retry=False, reason='rejected_source_direction_not_representable')
    elif abs(first) + abs(quadratic) <= precision:
        report.update(retry=False, reason='rejected_source_model_below_original_acceptance_precision')
    elif damping > np.finfo(float).max / 10.:
        report.update(retry=False, reason='finite_numerical_LM_damping_exhausted')
    else:
        report.update(retry=True, next_damping=damping * 10.,
            reason='exact_source_direction_rejected_retry_existing_lambda_times_10')
    report['search_outcome'] = ('retry_with_larger_damping' if report['retry']
        else 'numerical_search_exhausted')
    report['convergence_established'] = False
    return report


def augment_rejected_schur_retry(tree):
    """Wrap exactly the original trial loop; preserve all trial statements."""
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if len(functions) != 1:
        raise ValueError('One original public solver definition required')
    function = functions[0]
    targets = []
    for parent in ast.walk(function):
        for field, items in ast.iter_fields(parent):
            if not isinstance(items, list):
                continue
            for index, node in enumerate(items):
                if (isinstance(node, ast.For) and isinstance(node.iter, ast.Call)
                        and ast.unparse(node.iter) == 'enumerate(trial_schedule)'):
                    targets.append((items, index, node))
    if len(targets) != 1:
        raise ValueError('Unique unchanged source trial schedule required')
    items, index, original_loop = targets[0]
    original_dump = ast.dump(original_loop, include_attributes=False)
    wrapper = ast.parse("""
_schur_rejection_rounds = []
while True:
    _trial_loop_placeholder()
    if (stop is not None or accepted or global_schur_count is None or not schur_backtracking
            or source_trial_acceptance is not None
            or (derivative_model is not None and use_lagged)):
        break
    _schur_retry = _rejected_direction_damping(damping=damping, before=before,
        q=q, step=schur_full_step, normal=normal, gradient=gradient)
    _schur_rejection_rounds.append(_schur_retry)
    if not _schur_retry['retry']:
        reason = _schur_retry['reason']
        break
    damping = _schur_retry['next_damping']
    schur_full_step = None
    conditional_step = None
    physical_endpoint_event = None
    fraction_SOC_needed = False
    line_search_seed = 1.
""").body
    wrapper[1].body[0] = original_loop
    if ast.dump(wrapper[1].body[0], include_attributes=False) != original_dump:
        raise ValueError('Original source trial statements changed')
    items[index:index + 1] = wrapper
    records = []
    for parent in ast.walk(function):
        for field, values in ast.iter_fields(parent):
            if not isinstance(values, list):
                continue
            for index, node in enumerate(values):
                if (isinstance(node, ast.Assign) and len(node.targets) == 1
                        and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'record'
                        and isinstance(node.value, ast.Call) and any(
                            keyword.arg == 'failure_reason' for keyword in node.value.keywords)):
                    records.append((values, index))
    if len(records) != 1:
        raise ValueError('Unique original completed-iteration record required')
    values, index = records[0]
    values[index + 1:index + 1] = ast.parse("""
if _schur_rejection_rounds:
    record['rejected_schur_direction_damping'] = dict(
        same_source_linearization_reused=True,
        fresh_source_direction_only=True,
        existing_lambda_times_10_rejection_rule=True,
        original_acceptance_precision_and_all_guards_unchanged=True,
        no_added_iteration_limit=True,
        rounds=_schur_rejection_rounds)
""").body
    schedules = []
    for node in ast.walk(function):
        if (isinstance(node, ast.If) and ast.unparse(node.test) == 'adaptive_source_backtracking'
                and any(isinstance(item, ast.Assign) and len(item.targets) == 1
                    and isinstance(item.targets[0], ast.Subscript)
                    and isinstance(item.targets[0].value, ast.Name)
                    and item.targets[0].value.id == 'record'
                    and isinstance(item.targets[0].slice, ast.Constant)
                    and item.targets[0].slice.value == 'adaptive_source_backtracking'
                    for item in node.body)):
            schedules.append(node)
    if len(schedules) != 1:
        raise ValueError('Unique original adaptive-search report required')
    schedules[0].body.extend(ast.parse("""
if _schur_rejection_rounds:
    record['adaptive_source_backtracking'].pop('real_source_cost_guard_and_trial_count_unchanged', None)
    record['adaptive_source_backtracking'].update(
        real_source_cost_and_physical_guards_unchanged=True,
        per_direction_trial_schedule_unchanged=True,
        total_trials_may_increase=True)
""").body)
    return tree
