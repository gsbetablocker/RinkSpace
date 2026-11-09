"""Source-clock sampling and flight-boundary tests without a fitted body model."""
from types import SimpleNamespace

import numpy as np
import pytest

from calibration.program_tracking.body_tracking.programme_com import (
    ModelCOMMotionFactor,
    acceleration_operator,
    com_quadrature_times,
)


DENSE = {'quadrature': 'native_exposure_and_curve_spans'}
TICK = 1/12800
ACCELERATION = np.array([2.3, -1.2, .7])


def problem_for(native, physical, knots):
    native = np.asarray(native)
    pts = 283136 + 256*np.arange(len(native))
    ref = SimpleNamespace(
        times=native, p={'pts': pts, 'source_time_base': [1, 12800]},
        source_document={'source_episode_review': {'episodes': []}},
        at=lambda t: np.asarray(t),
    )
    state = SimpleNamespace(knots=np.asarray(knots), increments=lambda x, t: None)
    return SimpleNamespace(reference=ref, physics_times=np.asarray(physical), body=SimpleNamespace(state=state))


def test_dense_clock_keeps_native_and_distinct_rs_times_without_rounding():
    native = np.array([0., .02, .04, .06, .08])
    # The near-centre copies come from existing exposure sampling. Distinct
    # physical probes are authoritative even when their gap is sub-tick.
    redundant = np.array([3e-10, .02-4e-10, .04+5e-10])
    exposure = np.array([-.007, -.003, .003, .007, .017, .023, .023+TICK/4,
                         .027, .037, .043, .047, .057, .063, .067, .077, .087])
    problem = problem_for(native, np.sort(np.r_[redundant, exposure, .06]), [-.01, .011, .089])
    times = com_quadrature_times(problem, DENSE)
    original = np.r_[native, exposure]
    np.testing.assert_array_equal(times[np.searchsorted(times, original)], original)
    assert not np.isin(redundant, times).any()
    assert np.all(np.diff(times) > 0)
    assert times[0] == exposure[0] and times[-1] == exposure[-1]
    np.testing.assert_array_equal(problem.reference.times, native)
    np.testing.assert_array_equal(problem.physics_times, np.sort(np.r_[redundant, exposure, .06]))


def test_short_spans_receive_probes_without_near_duplicate_knot_intervals():
    native = np.array([0., .02, .04, .06])
    exposure = np.array([.006, .014, .026, .034, .046, .054])
    near = .01 + 1e-12
    short_end = .0112
    problem = problem_for(native, exposure, [-.01, -.01, .01, near, short_end, .035, .07, .07])
    times = com_quadrature_times(problem, DENSE)
    existing = np.r_[native, exposure]
    added = times[~np.isin(times, existing)]
    assert np.any((added > near) & (added < short_end))
    assert np.min(np.diff(times)) >= TICK-1e-15
    assert np.min(np.abs(added[:, None]-existing)) >= TICK-1e-15


@pytest.mark.parametrize('config', [{}, {'quadrature': 'native_source_centres'}])
def test_legacy_mode_uses_only_an_exact_copy_of_native_centres(config):
    native = np.array([0., .02, .04, .06])
    problem = SimpleNamespace(reference=SimpleNamespace(times=native))
    result = com_quadrature_times(problem, config)
    np.testing.assert_array_equal(result, native)
    assert not np.shares_memory(result, native)


class QuadraticMass:
    def evaluate(self, times, unused_delta):
        centre = times[:, None]**2 * ACCELERATION/2 + times[:, None]*[.3, -.1, .4]
        return centre, None

    def com(self, centre):
        return centre


def test_dense_factor_recovers_native_flight_endpoints_and_weights_off_native_samples():
    # These clock representations would miss both inclusive PTS boundaries if
    # native rows were classified by arithmetic time-to-PTS alone.
    native = np.array([0., .02-1e-12, .04+1e-12, .06, .08])
    exposure = np.array([-.003, .017, .020+2e-10, .023, .037, .043, .063, .083])
    problem = problem_for(native, exposure, [-.01, .031, .09])
    reference = problem.reference
    pts = reference.p['pts']
    reference.source_document['source_episode_review']['episodes'] = [
        {'kind': 'flight_visible_candidate', 'start_pts': int(pts[1]), 'end_pts_inclusive': int(pts[2])},
    ]
    np.testing.assert_array_less(
        [pts[0] + native[1]/TICK, pts[2]],
        [pts[1], pts[0] + native[2]/TICK],
    )
    factor = ModelCOMMotionFactor(problem, QuadraticMass(), dict(DENSE, acceleration_scale_mps2=7.))
    native_rows = np.searchsorted(factor.times, native)
    evaluation = factor.evaluate(np.empty(0))
    block, = evaluation['weighted_blocks']
    metadata = block.provenance
    np.testing.assert_array_equal(metadata['source_pts'][native_rows-1], pts)
    np.testing.assert_array_equal(metadata['native_source_pts'], pts)
    assert metadata['quadrature'] == DENSE['quadrature']
    in_flight = (factor.times[1:-1] >= native[1]) & (factor.times[1:-1] <= native[2])
    np.testing.assert_array_equal(factor.w == 0, in_flight)
    # Every remaining centre sees the same acceleration, scaled by the
    # trapezoid measure of its unequal adjacent time steps.
    expected_weight = np.sqrt((factor.times[2:]-factor.times[:-2])/2)/7
    expected_weight[in_flight] = 0.
    np.testing.assert_allclose(factor.w, expected_weight, atol=1e-15, rtol=0)
    np.testing.assert_allclose(evaluation['residual'].reshape(-1, 3), expected_weight[:, None]*ACCELERATION,
                               atol=1e-12, rtol=0)
    metrics = evaluation['metrics'][block.name]
    assert metrics['quadrature'] == DENSE['quadrature']
    assert metrics['native_exposures'] == len(native)
    assert metrics['evaluation_samples'] == len(factor.times) > len(native)
    assert metrics['flight_excluded_samples'] == int(in_flight.sum())


def test_nonuniform_operator_recovers_constant_world_acceleration():
    times = np.array([0., .00013, .0017, .0183, .0239, .0403, .0712])
    path = .5*times[:, None]**2*ACCELERATION + times[:, None]*[-.6, .2, 1.3] + [.3, 1.5, -.4]
    actual = acceleration_operator(times)@path
    np.testing.assert_allclose(actual, np.broadcast_to(ACCELERATION, actual.shape), atol=3e-9, rtol=0)
