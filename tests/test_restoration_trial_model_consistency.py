"""Source-model reversals are numerical failures, not new human constraints."""
from copy import deepcopy
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from calibration.solver_runtime.restoration_trial_model_consistency import with_restoration_trial_model_consistency


REAL_ITERATIONS = Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/calibration/runs/'
    'solver_acceleration_20261003/okada_review86_q8_trial_fraction_SOC_fix_20261011/solver/iterations.json')


def original_callback(decision):
    def original(parent, trial, **kwargs):
        return decision
    return original


@pytest.mark.skipif(not REAL_ITERATIONS.exists(), reason='Frozen actual short-SOC iteration unavailable')
def test_actual_short_SOC_truth_acceptance_with_reversed_descent_is_rejected():
    row = json.loads(REAL_ITERATIONS.read_text(encoding='utf8'))[0]
    decision = deepcopy(row['explicit_source_trial_acceptance'])
    snapshot = deepcopy(decision)
    predicted = row['global_schur_linear_report']['independent_trial_fraction_source_steps'][0][
        'QP_and_SOC']['independent_trial_fraction_source_step']['actual_source_model_decrease']
    assert decision['accept'] is True and decision['mode'] == 'truth_restoration'
    assert row['cost'] == pytest.approx(7696.22317038495, rel=0., abs=1e-11)
    assert row['next_cost'] == pytest.approx(20178.56308333413, rel=0., abs=1e-11)
    assert predicted == pytest.approx(345.72061929245126, rel=0., abs=1e-11)
    result = with_restoration_trial_model_consistency(original_callback(decision))(
        np.zeros(1), np.ones(1), before_cost=row['cost'], after_cost=row['next_cost'],
        default_source_cost_accepts=False, trial_context={'predicted_source_reduction': predicted})
    assert result['accept'] is False
    assert result['rejection_reason'] == 'restoration_source_model_descent_reversed'
    assert result['report'] == snapshot['report']
    assert decision == snapshot
    assert result['restoration_trial_model_consistency']['actual_source_reduction'] < -12482.


@pytest.mark.parametrize('predicted, after', [(-2., 12.), (-2., 10000.), (2., 8.), (0., 12.)])
def test_necessary_growth_or_nonreversed_trial_retains_original_truth_object(predicted, after):
    decision = dict(accept=True, mode='truth_restoration', report={'original_supports': True})
    result = with_restoration_trial_model_consistency(original_callback(decision))(
        np.zeros(1), np.ones(1), before_cost=10., after_cost=after,
        default_source_cost_accepts=after < 10., trial_context={'predicted_source_reduction': predicted})
    assert result is decision


@pytest.mark.parametrize('predicted, after, rejected', [(1e-10, 2e-10, False),
                                                      (2e-10, 1e-10, False),
                                                      (2e-10, 2e-10, True)])
def test_both_prediction_and_actual_growth_must_exceed_existing_precision(predicted, after, rejected):
    decision = dict(accept=True, mode='truth_restoration')
    result = with_restoration_trial_model_consistency(original_callback(decision))(
        np.zeros(1), np.ones(1), before_cost=0., after_cost=after,
        default_source_cost_accepts=False, trial_context={'predicted_source_reduction': predicted})
    assert result['accept'] is (not rejected)


@pytest.mark.parametrize('mode, accept', [('source_objective', True), ('source_objective', False),
                                        ('truth_restoration', False)])
def test_ordinary_source_and_original_truth_rejection_are_unchanged(mode, accept):
    decision = dict(accept=accept, mode=mode, rejection_reason='original')
    # No model fields are needed on branches whose original decision is preserved.
    result = with_restoration_trial_model_consistency(original_callback(decision))(
        np.zeros(1), np.ones(1), before_cost=10., after_cost=20.,
        default_source_cost_accepts=False, trial_context={})
    assert result is decision


def test_original_arguments_and_unwrap_interface_are_preserved():
    parent, trial = np.zeros(1), np.ones(1)
    context = {'predicted_source_reduction': -1., 'actual_step': trial}
    decision = dict(accept=True, mode='truth_restoration')
    seen = []
    def original(p, t, **kwargs):
        seen.append((p, t, kwargs))
        return decision
    wrapper = with_restoration_trial_model_consistency(original)
    result = wrapper(parent, trial, before_cost=10., after_cost=11.,
                     default_source_cost_accepts=False, trial_context=context)
    assert result is decision and inspect.unwrap(wrapper) is original
    assert seen[0][0] is parent and seen[0][1] is trial
    assert seen[0][2]['trial_context'] is context
    assert seen[0][2]['default_source_cost_accepts'] is False
    np.testing.assert_array_equal(parent, [0.])
    np.testing.assert_array_equal(trial, [1.])


@pytest.mark.parametrize('context', [{}, {'predicted_source_reduction': np.nan},
                                    {'predicted_source_reduction': np.inf}])
def test_missing_or_nonfinite_original_model_cannot_authorize_truth_acceptance(context):
    wrapper = with_restoration_trial_model_consistency(original_callback(dict(accept=True, mode='truth_restoration')))
    with pytest.raises(ValueError, match='finite restoration source-model'):
        wrapper(np.zeros(1), np.ones(1), before_cost=10., after_cost=12.,
                default_source_cost_accepts=False, trial_context=context)
