"""Reject a numerical restoration step whose predicted descent reverses."""
from __future__ import annotations

from functools import wraps
import math


def with_restoration_trial_model_consistency(original):
    """Keep the original truth rule, then reject only a source-model reversal.

    A restoration direction can require source-cost growth. Such a direction
    retains the original truth decision. This check only rejects a resolvable
    predicted decrease followed by a resolvable actual increase, using the
    solver's existing frozen-profile source-cost comparison precision.
    """
    @wraps(original)
    def checked(parent, trial, *, before_cost, after_cost,
                default_source_cost_accepts, trial_context):
        decision = original(parent, trial, before_cost=before_cost,
            after_cost=after_cost, default_source_cost_accepts=default_source_cost_accepts,
            trial_context=trial_context)
        if (not isinstance(decision, dict) or decision.get('mode') != 'truth_restoration'
                or decision.get('accept') is not True):
            return decision
        try:
            before = float(before_cost)
            after = float(after_cost)
            predicted = float(trial_context['predicted_source_reduction'])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError('Original finite restoration source-model prediction required') from error
        if not all(map(math.isfinite, (before, after, predicted))):
            raise ValueError('Original finite restoration source-model prediction required')
        precision = 1e-10 * max(1., before)
        actual = before - after
        if predicted <= precision or actual >= -precision:
            return decision
        result = dict(decision)
        result['accept'] = False
        # Independent short-step directions are not shortened samples of the
        # original line. Do not feed them to its lagged-J slope detector.
        result['rejection_reason'] = 'restoration_source_model_descent_reversed'
        result['restoration_trial_model_consistency'] = dict(
            schema='rinkspace.restoration-trial-source-model-consistency.v1',
            original_truth_restoration_accepts=True,
            predicted_source_reduction=predicted,
            actual_source_reduction=actual,
            source_cost_numerical_precision=precision,
            numerical_precision_rule='1e-10 * max(1, before)',
            predicted_descent_and_actual_increase_resolvable=True,
            original_source_H_g_profile_weights_supports_and_caps_unchanged=True,
            rejection_only_numerical_check=True,
            modelled_source_growth_is_not_rejected=True,
            accuracy_accepted=False)
        return result

    return checked
