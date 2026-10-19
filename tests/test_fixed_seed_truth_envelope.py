"""Source recovery can take a joint path without losing original protections."""
import numpy as np
import pytest

from calibration.solver_runtime.fixed_seed_truth_envelope import (
    make_fixed_seed_envelope, fixed_seed_trial_decision,
    attach_fixed_seed_trial_fraction_domain,
)
from calibration.solver_runtime.physical_signed_margin_SOC import attach_original_signed_margin_rows


def envelope():
    # Existing q11 cases: restored source, tiny Ad24 violation with four-pixel
    # working scale, and an unresolved positional stroke with zero expansion.
    return make_fixed_seed_envelope([0., .01801501778965431, 15.224261716092158],
        [4., 6., 0.], seed_q_sha256='1'*64,
        group_ids=['restored_rail', 'Ad24_outer', 'B2_positional_stroke'])


def test_joint_path_may_temporarily_worsen_an_unresolved_group_but_must_decrease_cost():
    limits, _ = envelope()
    decision = fixed_seed_trial_decision([0., .018, 15.], [0., 1.1, 14.], limits,
        before_cost=7403.8, after_cost=7350., default_source_cost_accepts=True)
    assert decision['accept']
    assert decision['report']['temporarily_increased_groups'] == 1
    assert not decision['report']['every_group_each_step_improvement_required']
    assert not decision['report']['historical_group_minimum_limits_preserved']


@pytest.mark.parametrize('trial', [[2e-8, 1., 14.], [0., 7., 14.], [0., 1., 16.]])
def test_cost_gain_cannot_overrule_source_zero_or_fixed_envelope(trial):
    limits, _ = envelope()
    result = fixed_seed_trial_decision([0., .018, 15.], trial, limits,
        before_cost=7403.8, after_cost=1., default_source_cost_accepts=True)
    assert not result['accept']
    assert result['rejection_reason'] == 'fixed_seed_truth_envelope_exceeded'


@pytest.mark.parametrize('after, default', [(7500., True), (7403.8, True), (7350., False)])
def test_truth_improvement_does_not_override_original_full_cost_rule(after, default):
    limits, _ = envelope()
    result = fixed_seed_trial_decision([0., .018, 15.], [0., 0., 10.], limits,
        before_cost=7403.8, after_cost=after, default_source_cost_accepts=default)
    assert not result['accept']


def test_continuation_does_not_add_uncertainty_again_or_latch_new_zero_group():
    limits, origin = envelope()
    continued, provenance = make_fixed_seed_envelope([0., 0., 10.], [4., 6., 0.],
        seed_q_sha256='2'*64, group_ids=origin['group_ids'], resumed=origin)
    np.testing.assert_array_equal(continued, limits)
    assert provenance['seed_q_sha256'] == '1'*64
    # A group which reached zero after the seed may move within its original
    # recovery room; the seed-feasible rail remains protected at zero.
    assert fixed_seed_trial_decision([0., 0., 10.], [0., .4, 9.], continued,
        before_cost=7000., after_cost=6900., default_source_cost_accepts=True)['accept']


@pytest.mark.parametrize('field,value', [
    ('limits_px', [0., 100., 100.]),
    ('declared_engineering_uncertainties_px', [4., 7., 0.]),
    ('seed_protected_zero_groups', [False, False, False]),
    ('historical_group_minimum_limits_preserved', True),
])
def test_resume_rejects_budget_or_protection_reinterpretation(field, value):
    _, record = envelope()
    record[field] = value
    with pytest.raises(ValueError):
        make_fixed_seed_envelope([0., 0., 10.], [4., 6., 0.], seed_q_sha256='2'*64,
            group_ids=record['group_ids'], resumed=record)


def test_trial_fraction_SOC_retains_physical_rows_and_starts_pristine_every_time():
    matrix = np.array([[1., 0.], [0., 1.], [0., -1.]])
    lower = np.array([-.2, -6.018, -1.])
    original = dict(matrix=matrix, lower=lower,
        curvature_responses=dict(native=dict(row_start=1, row_stop=3,
            dynamic_group_limits=np.array([6.018]), dynamic_row_group_index=np.array([0, 0]))))
    domain = attach_fixed_seed_trial_fraction_domain(original)
    domain = attach_original_signed_margin_rows(domain, row_indices=np.array([0]),
        nominal_margins=np.array([.2]), row_ids=['actual_exterior_pivot'])
    pristine = lower.copy()
    # Full-step SOC cuts are temporary; the next shorter QP starts anew.
    domain['lower'][1] = 100.
    domain['curvature_responses']['native']['dynamic_group_limits'][0] = 100.
    for alpha in (1., .5, .25, 1./2048.):
        child = domain['at_trial_fraction'](alpha)
        assert child['matrix'] is matrix
        np.testing.assert_array_equal(child['lower'], pristine)
        np.testing.assert_array_equal(child['curvature_responses']['native']['dynamic_group_limits'], [6.018])
        assert child['original_signed_margin_rows']['row_ids'] == ('actual_exterior_pivot',)
        assert not child['report']['trial_fraction_domain']['accumulated_full_step_SOC_cuts_reused']


@pytest.mark.parametrize('value', [np.nan, np.inf, -1.])
def test_invalid_violations_do_not_create_unbounded_recovery(value):
    with pytest.raises(ValueError):
        make_fixed_seed_envelope([value], [4.], seed_q_sha256='1'*64, group_ids=['source'])
