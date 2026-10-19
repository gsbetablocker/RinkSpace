"""Require the existing full source-descent gate during truth restoration.

The original truth callback runs first. Its complete source-group decision,
LP domain and nonlinear guards remain authoritative. This execution wrapper
can only reject a trial that passed truth restoration but failed the solver's
already computed, frozen-profile, complete-source descent condition.
"""
from __future__ import annotations

from functools import wraps
import hashlib
import inspect
from pathlib import Path


def with_restoration_source_cost_guard(original):
    @wraps(original)
    def guarded(parent, trial, *, before_cost, after_cost,
                default_source_cost_accepts, trial_context):
        decision = original(parent, trial, before_cost=before_cost,
            after_cost=after_cost, default_source_cost_accepts=default_source_cost_accepts,
            trial_context=trial_context)
        if (not isinstance(decision, dict) or decision.get('mode') != 'truth_restoration'
                or decision.get('accept') is not True):
            return decision
        if type(default_source_cost_accepts) is not bool:
            raise TypeError('Original complete-source boolean descent decision required')
        result = dict(decision)
        result['complete_source_descent_guard'] = dict(
            original_truth_restoration_accepts=True,
            original_complete_source_descent_accepts=default_source_cost_accepts,
            original_before_cost=float(before_cost), original_after_cost=float(after_cost),
            original_full_source_cost_change=float(after_cost-before_cost),
            original_rule='after < before - 1e-10 * max(1,before)',
            original_frozen_profiles_source_rows_weights_sigma_limits_and_LP_retained=True,
            rejection_only_execution_wrapper=True)
        if not default_source_cost_accepts:
            result['accept'] = False
            # Preserve the existing stale-J uphill trial detector and fresh-J
            # retry route. This is the original source-cost rejection reason.
            result['rejection_reason'] = 'objective_not_lower_at_profile_precision'
        return result

    guarded.execution_change = dict(
        schema='rinkspace.truth-restoration-existing-complete-source-descent-guard.v1',
        helper_path=str(Path(__file__).resolve()),
        helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        original_callback_sha256=hashlib.sha256(inspect.getsource(original).encode()).hexdigest(),
        original_truth_callback_called_with_identical_arguments=True,
        original_truth_progress_AND_existing_complete_source_descent_required=True,
        ordinary_source_objective_and_original_truth_rejections_unchanged=True,
        original_lagged_J_uphill_rejection_reason_and_fresh_J_retry_retained=True,
        original_LP_rows_supports_weights_sigma_profile_and_model_parameters_unchanged=True,
        no_new_observation_prior_or_cost_added=True,
        no_production_seed_modified=True)
    return guarded
