"""Budget and raw-quality decisions use supplied clocks and accepted states."""
from dataclasses import asdict, replace
import json

import pytest

from calibration.program_tracking.body_tracking.skate_run_policy import Monitor, RunPolicy


def codes(decision):return {r['code'] for r in decision['reasons']}


def loose(**changes):
    return replace(RunPolicy(), max_accepted=100, max_outer=100, **changes)


def test_default_user_budget_is_explicit_and_penalty_growth_is_disabled():
    assert asdict(RunPolicy()) == dict(wall_seconds=900, max_accepted=8, max_outer=1,
        stagnation_window=4, min_violation_improvement_m=.0001,
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
