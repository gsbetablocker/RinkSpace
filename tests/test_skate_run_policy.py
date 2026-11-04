"""Budget and raw-quality decisions use supplied clocks and accepted states."""
from dataclasses import asdict, replace
import json

import numpy as np
import pytest

from calibration.program_tracking.body_tracking.skate_run_policy import CandidateRetention, Monitor, RunPolicy


def codes(decision):return {r['code'] for r in decision['reasons']}


def loose(**changes):
    return replace(RunPolicy(), max_accepted=100, max_outer=100, **changes)


def test_default_user_budget_is_explicit_and_penalty_growth_is_disabled():
    assert asdict(RunPolicy()) == dict(wall_seconds=900, max_accepted=8, max_outer=1,
        stagnation_window=4, min_violation_improvement_m=.0001, min_joint_relative_improvement=1e-7,
        max_image_cost_increase=.05, max_pose_cost_increase=.10, allow_penalty_growth=False)


def test_wall_budget_uses_supplied_elapsed_clock_without_sleeping():
    monitor = Monitor(now=100.)
    assert not monitor.observe(now=999.999)['stop']
    decision = monitor.observe(now=1000.)
    assert codes(decision) == {'wall_budget'}
    assert decision['reasons'][0]['elapsed_seconds'] == 900.
    assert decision['convergence_claim'] is False


def test_accepted_budget_counts_accepted_states_not_evaluations():
    monitor = Monitor(replace(RunPolicy(), max_accepted=2), now=0.)
    for i in range(10):assert not monitor.observe(now=i)['stop']
    assert not monitor.observe(now=10., accepted=True)['stop']
    assert codes(monitor.observe(now=11., accepted=True)) == {'accepted_budget'}
    assert monitor.record()['checks'] == 12
    assert monitor.record()['accepted'] == 2


def test_one_outer_is_allowed_until_it_has_completed():
    monitor = Monitor(now=0.)
    assert not monitor.observe(now=0., outer_completed=0)['stop']
    assert not monitor.observe(now=1., accepted=True)['stop']
    assert codes(monitor.observe(now=2., outer_completed=1)) == {'outer_budget'}


def test_stagnation_tracks_best_progress_over_four_accepted_updates():
    monitor = Monitor(loose(), now=0.)
    monitor.observe(now=0., violation_m=.004)
    values = [.0035, .003, .0025, .002, .0022, .0021, .0023, .0022]
    for step, value in enumerate(values, 1):
        decision = monitor.observe(now=step, violation_m=value, accepted=True)
        assert decision['stop'] == (step == 8)
    assert codes(decision) == {'violation_stagnation'}
    assert decision['reasons'][0]['improvement_m'] == 0.
    assert monitor.record()['best_violation_m'] == .002
    assert monitor.record()['best_at_accepted'] == 4
    assert monitor.record()['latest']['violation_m'] == .0022


def test_subthreshold_physical_progress_stops_without_claiming_convergence():
    monitor = Monitor(loose(), now=0.)
    monitor.observe(now=0., violation_m=.00166)
    for i in range(1, 5):decision = monitor.observe(now=i, violation_m=.00166-i*.00001, accepted=True)
    assert codes(decision) == {'violation_stagnation'}
    assert monitor.record()['best_violation_m'] > .001
    assert not monitor.record()['convergence_claim']


def test_missing_violation_breaks_the_comparable_stagnation_window():
    monitor = Monitor(loose(stagnation_window=3), now=0.)
    monitor.observe(now=0., violation_m=.004)
    for i, value in enumerate([.003, None, .003, .003, .003], 1):
        assert not monitor.observe(now=i, violation_m=value, accepted=True)['stop']
    assert codes(monitor.observe(now=6., violation_m=.003, accepted=True)) == {'violation_stagnation'}


@pytest.mark.parametrize('kind,baseline,boundary,failed', [('image', 100., 105., 105.01), ('pose', 20., 22., 22.01)])
def test_raw_component_degradation_is_relative_to_fixed_initial_state(kind, baseline, boundary, failed):
    monitor = Monitor(loose(), now=0.)
    monitor.observe(now=0., **{kind+'_cost': baseline})
    assert not monitor.observe(now=1., **{kind+'_cost': boundary})['stop']
    decision = monitor.observe(now=2., **{kind+'_cost': failed})
    assert codes(decision) == {kind+'_cost_increase'}
    reason = decision['reasons'][0]
    assert reason['baseline'] == baseline and reason['latest'] == failed
    assert monitor.record()['baseline'][kind+'_cost'] == baseline


@pytest.mark.parametrize('unavailable', [None, float('nan'), float('inf'), -1.])
def test_unavailable_initial_component_cannot_be_replaced_by_a_later_baseline(unavailable):
    monitor = Monitor(loose(), now=0.)
    monitor.observe(now=0., violation_m=.003, image_cost=unavailable, pose_cost=10.)
    assert not monitor.observe(now=1., image_cost=1e9, pose_cost=10.)['stop']
    assert monitor.record()['baseline']['image_cost'] is None
    json.dumps(monitor.record(), allow_nan=False)


def test_zero_initial_cost_does_not_fabricate_an_infinite_relative_change():
    monitor = Monitor(loose(), now=0.)
    monitor.observe(now=0., image_cost=0., pose_cost=0.)
    assert not monitor.observe(now=1., image_cost=1., pose_cost=1.)['stop']


def test_clock_only_polls_do_not_consume_the_quality_baseline():
    monitor = Monitor(loose(), now=0.)
    assert not monitor.observe(now=1.)['stop']
    assert monitor.baseline is None
    monitor.observe(now=2., image_cost=10., pose_cost=10.)
    assert codes(monitor.observe(now=3., image_cost=11.)) == {'image_cost_increase'}


def test_penalty_growth_request_is_stopped_unless_explicitly_allowed():
    monitor = Monitor(now=0.)
    assert codes(monitor.observe(now=0., penalty_growth_requested=True)) == {'penalty_growth_disabled'}
    allowed = Monitor(replace(RunPolicy(), allow_penalty_growth=True), now=0.)
    assert not allowed.observe(now=0., penalty_growth_requested=True)['stop']


def test_stop_is_latched_even_if_a_later_measurement_recovers():
    monitor = Monitor(loose(), now=0.)
    monitor.observe(now=0., image_cost=10.)
    decision = monitor.observe(now=1., image_cost=20.)
    assert monitor.observe(now=2., image_cost=5.) == decision
    assert monitor.record()['first_stop_now'] == 1.
    assert monitor.record()['latest']['image_cost'] == 5.


def test_nonmonotonic_clock_and_outer_count_are_rejected():
    monitor = Monitor(loose(), now=2.)
    with pytest.raises(ValueError):monitor.observe(now=1.)
    monitor.observe(now=3., outer_completed=2)
    with pytest.raises(ValueError):monitor.observe(now=4., outer_completed=1)


@pytest.mark.parametrize('changes', [dict(wall_seconds=float('inf')), dict(max_accepted=-1),
    dict(max_outer=.5), dict(stagnation_window=4.), dict(stagnation_window=0), dict(min_violation_improvement_m=-.001)])
def test_invalid_policy_cannot_silently_disable_its_limits(changes):
    with pytest.raises(ValueError):replace(RunPolicy(), **changes)


def candidate_value(image_cost, pose_cost=10.):
    return dict(image_cost=image_cost, pose_cost=pose_cost,
        parts=dict(data=image_cost), gate=dict(max_violation_m=.003),
        raw_data_parts=dict(body_pixels=image_cost), source_motion_metrics=dict(speed_m_s=[.1]))


def retain(retention, monitor, x, *, step, image_cost=100., pose_cost=10., **observations):
    value = candidate_value(image_cost, pose_cost)
    decision = monitor.observe(now=step, image_cost=image_cost, pose_cost=pose_cost, **observations)
    selected = retention.observe(x, value, step=step, decision=decision, metrics=monitor.latest)
    return selected, decision


def test_quality_failure_keeps_previous_coherent_candidate_and_attempt_for_diagnosis():
    monitor = Monitor(loose(), now=0.)
    retention = CandidateRetention()
    retain(retention, monitor, np.array([0.]), step=0)
    selected, decision = retain(retention, monitor, np.array([1.]), step=1,
        image_cost=106., accepted=True)
    assert decision['quality_failed']
    assert decision['quality_reasons'][0]['category'] == 'quality'
    assert retention.last_attempt.step == 1 and selected.step == 0
    np.testing.assert_array_equal(retention.last_attempt.x, [1.])
    np.testing.assert_array_equal(selected.x, [0.])
    assert selected.value['image_cost'] == selected.metrics['image_cost'] == 100.
    assert retention.last_attempt.value['image_cost'] == retention.last_attempt.metrics['image_cost'] == 106.
    record = retention.record()
    assert record['attempt_step'] == 1 and record['retained_step'] == 0
    assert record['attempt_metrics']['image_cost'] == 106.
    assert record['retained_metrics']['image_cost'] == 100.


@pytest.mark.parametrize('budget', ['wall', 'accepted', 'outer'])
def test_pure_budget_retains_latest_quality_normal_state(budget):
    policy = replace(loose(), wall_seconds=1. if budget == 'wall' else 900.,
        max_accepted=1 if budget == 'accepted' else 100, max_outer=1 if budget == 'outer' else 100)
    monitor = Monitor(policy, now=0.)
    retention = CandidateRetention()
    retain(retention, monitor, np.array([0.]), step=0)
    selected, decision = retain(retention, monitor, np.array([1.]), step=1,
        image_cost=101., accepted=True, outer_completed=1 if budget == 'outer' else 0)
    assert codes(decision) == {budget+'_budget'}
    assert not decision['quality_failed']
    assert decision['reasons'][0]['category'] == 'budget'
    assert selected.step == retention.last_attempt.step == 1
    assert selected.value['image_cost'] == selected.metrics['image_cost'] == 101.


def test_stagnation_retains_latest_quality_normal_state():
    monitor = Monitor(loose(stagnation_window=1), now=0.)
    retention = CandidateRetention()
    retain(retention, monitor, np.array([0.]), step=0, violation_m=.003)
    selected, decision = retain(retention, monitor, np.array([1.]), step=1,
        accepted=True, violation_m=.003)
    assert codes(decision) == {'violation_stagnation'}
    assert decision['reasons'][0]['category'] == 'stagnation'
    assert not decision['quality_failed']
    assert selected.step == 1
    np.testing.assert_array_equal(selected.x, [1.])


@pytest.mark.parametrize('final_cost,expected_x,failed', [(101., 1., False), (106., 0., True)])
def test_normal_solver_return_uses_same_quality_selection_even_without_an_accepted_callback(final_cost, expected_x, failed):
    monitor = Monitor(loose(), now=0.)
    retention = CandidateRetention()
    retain(retention, monitor, np.array([0.]), step=0)
    decision = monitor.observe(now=1., image_cost=final_cost, pose_cost=10.)
    selected = retention.observe(np.array([1.]), candidate_value(final_cost), step=0,
        decision=decision, metrics=monitor.latest)
    assert decision['quality_failed'] is failed
    assert monitor.accepted == 0
    np.testing.assert_array_equal(selected.x, [expected_x])
    assert selected.value['image_cost'] == selected.metrics['image_cost'] == (100. if failed else 101.)


def test_first_domain_failed_state_is_unavailable_instead_of_a_fabricated_fallback():
    monitor = Monitor(loose(), now=0.)
    retention = CandidateRetention()
    assert retention.selected is None
    selected, decision = retain(retention, monitor, np.array([0.]), step=0,
        quality_failures=[dict(code='collar_unknown_sections', category='domain', count=2)])
    assert decision['quality_failed']
    assert decision['quality_reasons'][0]['category'] == 'domain'
    assert selected is retention.last_good is None
    assert retention.last_attempt.step == 0
    assert retention.record()['selection'] == 'unavailable'
    assert retention.record()['retained_metrics'] is None


def test_candidate_snapshots_do_not_alias_inputs_each_other_or_exposed_snapshots():
    class LargeRuntimeObject:
        def __deepcopy__(self, memo):raise AssertionError('Runtime state must not be copied')
    retention = CandidateRetention()
    x = np.array([1., 2.])
    value = candidate_value(100.)
    value.update(jac=LargeRuntimeObject(), world=LargeRuntimeObject(), cache=LargeRuntimeObject(),
        residual=LargeRuntimeObject(), contact=LargeRuntimeObject())
    metrics = dict(image_cost=100., quality_metrics=dict(speed=dict(value=.1, unit='m/s')))
    selected = retention.observe(x, value, step=0, decision=dict(reasons=[]), metrics=metrics)
    x[:] = 9.;value['parts']['data'] = -1.;value['source_motion_metrics']['speed_m_s'][0] = 9.
    metrics['quality_metrics']['speed']['value'] = 9.
    selected.x[:] = 8.;selected.value['parts']['data'] = -2.
    attempt, good = retention.last_attempt, retention.last_good
    assert not np.shares_memory(attempt.x, good.x)
    np.testing.assert_array_equal(good.x, [1., 2.])
    assert good.value['parts']['data'] == 100.
    assert good.value['source_motion_metrics']['speed_m_s'] == [.1]
    assert good.metrics['quality_metrics']['speed']['value'] == .1
    assert not {'jac', 'world', 'cache', 'residual', 'contact'} & good.value.keys()
    attempt.x[:] = 7.;attempt.metrics['image_cost'] = -3.
    record = retention.record();record['retained_metrics']['image_cost'] = -4.
    np.testing.assert_array_equal(retention.last_attempt.x, [1., 2.])
    assert retention.last_good.metrics['image_cost'] == retention.last_attempt.metrics['image_cost'] == 100.


def test_zero_pose_baseline_and_unlimited_live_motion_are_explicit_diagnostics():
    monitor = Monitor(loose(), now=0.)
    monitor.observe(now=0., image_cost=100., pose_cost=0.)
    decision = monitor.observe(now=1., image_cost=100., pose_cost=1000.,
        quality_metrics={'source_motion.root_speed': dict(unit='m/s', value=1000.)})
    assert not decision['quality_failed'] and not decision['stop']
    diagnostics = monitor.record()['quality_diagnostics']
    assert any(d['code'] == 'relative_quality_unavailable' and d['metric'] == 'pose_cost'
        and d['reason'] == 'zero_initial_cost' for d in diagnostics)
    assert any(d['code'] == 'quality_metric_diagnostic_only' and d['unit'] == 'm/s' for d in diagnostics)
    assert monitor.record()['quality_limits'] == {}
    json.dumps(monitor.record(), allow_nan=False)


@pytest.mark.parametrize('measurement,expected_code', [
    (dict(unit='m/s', value=2.01), 'quality_metric_limit'),
    (dict(unit='mm/s', value=1.), 'quality_metric_unavailable'),
    (dict(unit='m/s', value=float('nan')), 'quality_metric_unavailable'),
    (None, 'quality_metric_unavailable')])
def test_explicit_unit_bearing_guard_can_reject_an_initial_state(measurement, expected_code):
    # Synthetic limit exercises the opt-in API; it is not a production motion threshold.
    monitor = Monitor(loose(), now=0., quality_limits={'source_motion.root_speed': dict(unit='m/s', maximum=2.)})
    assert not monitor.observe(now=0.)['stop']  # Clock-only polls are not incomplete state observations.
    metrics = {} if measurement is None else {'source_motion.root_speed': measurement}
    decision = monitor.observe(now=1., quality_metrics=metrics)
    retention = CandidateRetention()
    assert retention.observe(np.array([0.]), candidate_value(100.), step=0,
        decision=decision, metrics=monitor.latest) is None
    assert codes(decision) == {expected_code}
    assert decision['quality_failed']
    assert not retention.record()['available']
    json.dumps(monitor.record(), allow_nan=False)


def test_quality_failure_is_not_hidden_by_an_earlier_latched_budget_stop():
    monitor = Monitor(replace(loose(), wall_seconds=1.), now=0.)
    retention = CandidateRetention()
    retain(retention, monitor, np.array([0.]), step=0)
    assert codes(monitor.observe(now=1.)) == {'wall_budget'}
    selected, decision = retain(retention, monitor, np.array([1.]), step=2, image_cost=106.)
    assert codes(decision) == {'wall_budget'}  # Original first-stop latch remains intact.
    assert decision['quality_failed']
    assert {r['code'] for r in decision['quality_reasons']} == {'image_cost_increase'}
    assert selected.step == 0
