"""Independent pinhole oracles for native one-dimensional station searches.

These are synthetic arithmetic regressions, not source-footage validation.
The reference pixels and minima never come from the native projection/search.
"""
import os

import numpy as np
import pytest
from scipy.optimize import minimize_scalar

from rinkspace.core.gpu import ResidentResidualProblem
import test_g2_gpu as source_fixture


LIBRARY = os.environ.get("RINKSPACE_G4_LIBRARY") or os.environ.get("RINKSPACE_G3_LIBRARY")
pytestmark = pytest.mark.skipif(not LIBRARY, reason="explicit CUDA/cuDSS library required")


@pytest.fixture
def source(monkeypatch):
    # Reuse the common geometry/camera fixture, loading the explicitly selected
    # G3/G4 library even when a separate older G2 library is also configured.
    monkeypatch.setenv("RINKSPACE_G2_LIBRARY", LIBRARY)
    fixture = source_fixture.G2GPU()
    fixture.setUp()
    return fixture


def circle_pixels(stations):
    angle = 2 * np.pi * np.asarray(stations)
    # Identity camera, depth 3, focal length 100 and world radius .3.
    return 50 + 10 * np.stack((np.cos(angle), np.sin(angle)), axis=-1)


def corner_pixels(stations):
    stations = np.asarray(stations)
    # The world polyline (-.9,0,0) -> (0,0,0) -> (0,.9,0),
    # projected independently with u=50+100*x/3, v=50+100*y/3.
    return np.stack((np.interp(stations, [0, .5, 1], [20, 50, 50]),
                     np.interp(stations, [0, .5, 1], [50, 50, 80])), axis=-1)


def dense_scalar_oracle(project, observed, lower, upper):
    grid = np.linspace(lower, upper, 32769)
    costs = .5 * np.sum((project(grid) - observed) ** 2, axis=-1)
    index = int(np.argmin(costs))
    left, right = max(0, index - 1), min(len(grid) - 1, index + 1)

    def objective(station):
        return float(.5 * np.sum((project(station) - observed) ** 2))

    reference = minimize_scalar(objective, bounds=(grid[left], grid[right]),
                                method="bounded", options={"xatol": 1e-15})
    assert reference.success, reference
    # Include exact support endpoints instead of treating the scalar routine's
    # interior approximation as an endpoint oracle.
    candidates = [(float(grid[index]), float(costs[index])),
                  (reference.x, reference.fun), (lower, objective(lower)),
                  (upper, objective(upper))]
    return min(candidates, key=lambda item: item[1])


def problem(source, path, initial, observed, lower=0., upper=1.):
    rig = np.asarray(source.kw["rig"]).copy()
    rig[28:] = 0.  # No lens distortion or rolling readout.
    stations = np.array([initial, .23, .23, .64, .77])
    observed = np.vstack((observed, [[43., 38.], [47., 41.], [48., 51.], [53., 46.]]))
    kwargs = {**source.kw, "rig": rig, "controls": np.tile([0., 0., np.log(100.)], (8, 1)),
              "codes": np.zeros(5, np.int32), "stations": stations,
              "times": np.zeros(5), "observed": observed}
    resident = ResidentResidualProblem(source.core, **kwargs)
    try:
        resident.configure_geometry([path])
        resident.configure_pixel_linearization(
            directions=np.zeros((8, 3, 1)), local_ids=np.array([0, 1, 1, 2, -1], np.int32),
            local_scales=np.ones(3))
        resident.configure_solver(lower=[lower, 0., .64], upper=[upper, 1., .64],
                                  device_budget_bytes=resident.creation["free_after_alloc_bytes"],
                                  host_budget_bytes=2**30)
    except Exception:
        resident.close()
        raise
    return resident


def pixel_costs(evaluation):
    np.testing.assert_array_equal(evaluation["status"], 0)
    return .5 * np.sum(evaluation["values"][:, 9:11] ** 2, axis=1)


def assert_refinement(resident, mode, before, old_costs):
    info = resident.refine_curve_stations(mode=mode)
    after = resident.export_state()
    evaluation = resident.evaluate("cuda")
    new_costs = pixel_costs(evaluation)
    assert info["failed_rows"] == 0
    assert np.all(new_costs <= old_costs)
    assert info["pixel_cost_decrease"] == pytest.approx(
        float(np.sum(old_costs - new_costs)), rel=1e-12, abs=2e-10)
    # Shared station ID 1, equality-bounded ID 2, and the fixed row remain exact.
    np.testing.assert_array_equal(after["stations"][1:], before["stations"][1:])
    for key in before.keys() - {"stations"}:
        np.testing.assert_array_equal(after[key], before[key])
    return after, evaluation, new_costs, info


@pytest.mark.parametrize("mode", ["global", "local"])
def test_nonzero_circle_distance_matches_independent_scalar_minimum(source, mode):
    target = 67.37 / 256
    observed = 50 + 1.99 * (circle_pixels(target) - 50)
    expected_station, expected_cost = dense_scalar_oracle(circle_pixels, observed, 0., 1.)
    # The closed-form radial solution independently checks the scalar oracle.
    assert expected_station == pytest.approx(target, abs=2e-8)
    assert expected_cost == pytest.approx(.5 * 9.9**2, abs=2e-12)
    assert expected_cost > 40.  # A nonzero residual exposes omitted curvature.
    path = dict(kind=1, parameters=[0., 0., 0., .3, 0., 2 * np.pi])
    initial = target + (.11 if mode == "global" else .005)
    # Near this optimum, undamped Gauss-Newton has error multiplier -.99.
    # A fixed 32 tangent steps therefore leave visible station/cost error even
    # after good sampled navigation. The oracle tests the optimum, not a step.
    with problem(source, path, initial, observed) as resident:
        before = resident.export_state()
        old_costs = pixel_costs(resident.evaluate("cuda"))
        after, evaluation, costs, info = assert_refinement(resident, mode, before, old_costs)
        assert info["changed_rows"] == 1
        assert after["stations"][0] == pytest.approx(target, abs=5e-8)
        assert costs[0] == pytest.approx(expected_cost, rel=0., abs=2e-10)
        np.testing.assert_allclose(evaluation["values"][0, 3:5],
                                   circle_pixels(after["stations"][0]), atol=2e-11, rtol=0.)
        _, _, repeated_costs, repeated = assert_refinement(resident, mode, after, costs)
        assert repeated["pixel_cost_decrease"] < 2e-10
        assert float(np.max(costs - repeated_costs)) < 2e-10


@pytest.mark.parametrize("mode", ["global", "local"])
@pytest.mark.parametrize("observed,initial,lower,upper,expected", [
    ([10., 40.], .46, .23, .73, .23),       # Lower finite support endpoint.
    ([60., 95.], .46, .23, .73, .73),       # Upper finite support endpoint.
    ([62., 38.], .62, 0., 1., .5),         # Opposite one-sided slopes at the corner.
    ([45., 38.], .5, 0., 1., 5 / 12),      # Release from the corner onto its left side.
    ([45., 50.], .5, 0., 1., 5 / 12),      # Zero right tangent must still release left.
    ([62., 55.], .5, 0., 1., 7 / 12),      # Release from the corner onto its right side.
], ids=["lower", "upper", "corner", "corner_left", "corner_left_orthogonal", "corner_right"])
def test_finite_endpoints_and_corner_sides_keep_exact_pixel_minimum(
        source, mode, observed, initial, lower, upper, expected):
    observed = np.asarray(observed)
    path = dict(kind=4, parameters=[], vertices=[[-.9, 0., 0.], [0., 0., 0.], [0., .9, 0.]])
    expected_pixels = corner_pixels(expected)
    expected_cost = float(.5 * np.sum((expected_pixels - observed) ** 2))
    with problem(source, path, initial, observed, lower, upper) as resident:
        before = resident.export_state()
        old_costs = pixel_costs(resident.evaluate("cuda"))
        after, evaluation, costs, info = assert_refinement(resident, mode, before, old_costs)
        assert info["changed_rows"] == 1
        assert lower <= after["stations"][0] <= upper
        assert after["stations"][0] == pytest.approx(expected, abs=5e-8)
        assert costs[0] == pytest.approx(expected_cost, rel=0., abs=2e-9)
        np.testing.assert_allclose(evaluation["values"][0, 3:5], expected_pixels, atol=3e-6, rtol=0.)
        _, _, repeated_costs, repeated = assert_refinement(resident, mode, after, costs)
        assert repeated["pixel_cost_decrease"] < 2e-9
        assert float(np.max(costs - repeated_costs)) < 2e-9