"""Dense flight samples exercise the same COM and sparse body/shoe state."""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from calibration.program_tracking.body_tracking.absolute_motion_curve import (
    AbsoluteMotionCurve,
    clamped_cubic_knots,
)
from calibration.program_tracking.body_tracking.absolute_motion_state import evaluate_curve
from calibration.program_tracking.body_tracking.continuous_physics import ContinuousMassGeometry
from calibration.program_tracking.body_tracking.event_physics import ballistic_residual
from calibration.program_tracking.body_tracking.programme_com import ModelCOMMotionFactor
from calibration.program_tracking.body_tracking.sequence_skate_state import SequenceState


GRAVITY = np.array([0., 0., -9.81])
CURVATURE = "actual_model_com_acceleration"
DENSE_FAMILY = "flight_dense_com_acceleration"
SCALE_KEY = "confirmed_flight_acceleration_scale_mps2"
QUADRATURE = "native_exposure_and_curve_spans"
CONFIG = dict(quadrature=QUADRATURE, acceleration_scale_mps2=12.)
DENSE_CONFIG = {**CONFIG, SCALE_KEY: 4.}
NATIVE = np.arange(8) * .02
PTS = 1041408 + 256 * np.arange(len(NATIVE))
PHYSICAL = np.array([.003, .011, .027, .032, .057, .062, .079, .092, .105, .131, .137])
BREAKS = [0., .031, .082, .14]


def episode(name, lo, hi):
    return dict(episode_id=name, kind="flight_visible_candidate", start_pts=lo, end_pts_inclusive=hi)


def clocked_problem(episodes):
    reference = SimpleNamespace(
        times=NATIVE, p={"pts": PTS, "source_time_base": [1, 12800]},
        source_document={"source_episode_review": {"episodes": episodes}}, at=np.asarray,
    )
    state = SimpleNamespace(knots=np.asarray(BREAKS), increments=lambda x, t: None)
    return SimpleNamespace(reference=reference, physics_times=PHYSICAL, body=SimpleNamespace(state=state))


def two_flights():
    return [episode("entry", PTS[1], PTS[3]), episode("later", PTS[4], PTS[6])]


def ballistic_path(t):
    return .5 * t[:, None] ** 2 * GRAVITY + t[:, None] * [.4, .2, 1.3] + [.8, .7, 1.2]


class SampledMass:
    def __init__(self, path):
        self.path = path

    def evaluate(self, times, unused):
        return self.path(times), None

    def com(self, positions):
        return positions


def blocks(value):
    return {term.name: term for term in value["weighted_blocks"]}


def test_ballistic_acceleration_is_zero_on_nonuniform_flight_samples(tmp_path):
    problem = clocked_problem(two_flights())
    factor = ModelCOMMotionFactor(problem, SampledMass(ballistic_path), DENSE_CONFIG)

    value = factor.evaluate(np.empty(0))
    terms = blocks(value)
    assert set(terms) == {CURVATURE, f"episode/entry/{DENSE_FAMILY}", f"episode/later/{DENSE_FAMILY}"}
    assert np.unique(np.round(np.diff(factor.times), 7)).size > 3
    assert np.linalg.norm(terms[CURVATURE].residual) > 0.05
    for name in ("entry", "later"):
        term = terms[f"episode/{name}/{DENSE_FAMILY}"]
        assert len(term.residual) > 3
        np.testing.assert_allclose(term.residual, 0., atol=2e-11, rtol=0)
        np.testing.assert_array_equal(term.provenance["gravity_world_mps2"], GRAVITY)

    path = tmp_path / "quad.npz"
    factor.export(np.empty(0), path, parameter_sha256="fixture")
    with np.load(path, allow_pickle=False) as export:
        np.testing.assert_array_equal(export["confirmed_flight_episode_ids"], ["entry", "later"])
        times_as_pts = export["evaluation_source_pts"]
        groups = export["confirmed_flight_acceleration_group"]
        # Use the reviewed intervals, not the factor's row indices, as the oracle.
        # In particular a centred row at either native endpoint has a neighbour
        # outside that review. A row straddling the two reviewed intervals is unlabelled.
        triplets = np.lib.stride_tricks.sliding_window_view(times_as_pts, 3)
        for group, review in enumerate(two_flights()):
            lo, hi = review["start_pts"], review["end_pts_inclusive"]
            expected = (triplets >= lo).all(axis=1) & (triplets <= hi).all(axis=1)
            np.testing.assert_array_equal(groups == group, expected)
            np.testing.assert_array_equal(groups[np.isin(times_as_pts[1:-1], [lo, hi])], [-1, -1])
        np.testing.assert_array_equal(export["confirmed_flight_gravity_world_mps2"], GRAVITY)


def test_native_ballistic_residual_misses_exposure_interior_oscillation():
    problem = clocked_problem([episode("entry", PTS[1], PTS[6])])

    def oscillating_path(t):
        result = ballistic_path(t)
        result[:, 0] += .004 * np.sin(2 * np.pi * t / .02)
        return result

    # The same position curve visits the exact parabola on every exposure
    # centre. Its 4 mm variation between centres changes acceleration greatly.
    native_flight = NATIVE[1:7]
    old = ballistic_residual(oscillating_path(native_flight), native_flight, GRAVITY, sigma_position_m=.03)
    np.testing.assert_allclose(old["residual"], 0., atol=2e-13, rtol=0)
    factor = ModelCOMMotionFactor(problem, SampledMass(oscillating_path), DENSE_CONFIG)
    value = factor.evaluate(np.empty(0))
    assert np.linalg.norm(blocks(value)[f"episode/entry/{DENSE_FAMILY}"].residual) > 1.
    assert value["metrics"]["confirmed_flight_dense_com"]["episodes"][0]["max_gravity_error_mps2"] > 100.


@pytest.mark.parametrize("config, message", [
    (dict(acceleration_scale_mps2=12., **{SCALE_KEY: 4.}), "requires exposure and curve-span"),
    ({**CONFIG, SCALE_KEY: 0.}, "positive dense"),
    ({**CONFIG, SCALE_KEY: np.nan}, "positive dense"),
])
def test_invalid_dense_flight_scale_or_quadrature_is_rejected(config, message):
    with pytest.raises(ValueError, match=message):
        ModelCOMMotionFactor(clocked_problem(two_flights()), SampledMass(ballistic_path), config)


@pytest.mark.parametrize("episodes, message", [
    ([], "requires reviewed source intervals"),
    ([episode("short", PTS[2], PTS[2])], "Empty"),
    ([episode("reversed", PTS[3], PTS[2])], "Invalid reviewed"),
    ([episode("first", PTS[0], PTS[2]), episode("second", PTS[2], PTS[6])], "Overlapping reviewed"),
    ([episode("same", PTS[0], PTS[2]), episode("same", PTS[4], PTS[6])], "Duplicate dense"),
])
def test_invalid_reviewed_interval_population_is_rejected(episodes, message):
    with pytest.raises(ValueError, match=message):
        ModelCOMMotionFactor(clocked_problem(episodes), SampledMass(ballistic_path), DENSE_CONFIG)


def test_legacy_default_returns_only_native_curvature():
    problem = clocked_problem([episode("entry", PTS[1], PTS[3])])
    factor = ModelCOMMotionFactor(problem, SampledMass(ballistic_path), dict(acceleration_scale_mps2=12.))
    value = factor.evaluate(np.empty(0))

    np.testing.assert_array_equal(factor.times, NATIVE)
    assert set(value["parts"]) == {CURVATURE}
    assert set(blocks(value)) == {CURVATURE}
    assert "confirmed_flight_dense_com" not in value["metrics"]
    expected = np.zeros((len(NATIVE) - 2, 3))
    expected[3:] = np.sqrt(.02) * GRAVITY / 12.
    np.testing.assert_allclose(value["residual"].reshape(-1, 3), expected, atol=1e-13, rtol=0)


class AnklePivotBinding:
    ratio = .95

    def transforms(self, P, Q, native, frames, *, skate_relative_R):
        ankles = [7, 8]
        origins = P[:, None] + np.einsum("nij,nfj->nfi", Q, native[:, ankles])
        rotations = Q[:, None] @ frames[:, ankles] @ skate_relative_R
        return rotations, origins


def test_new_blocks_use_actual_sparse_body_and_relative_shoe_directional_derivatives():
    rng = np.random.default_rng(1203)
    problem = clocked_problem(two_flights())
    ref = problem.reference
    knots = clamped_cubic_knots(BREAKS)
    controls = len(knots) - 4
    rotation_vectors = rng.normal(scale=.08, size=(controls * 22, 3))
    rotations = Rotation.from_rotvec(rotation_vectors).as_matrix().reshape(controls, 22, 3, 3)
    ref.absolute_curve = AbsoluteMotionCurve(knots, rng.normal(scale=.03, size=(controls, 3)), rotations)
    ref.at = lambda t: evaluate_curve(ref.absolute_curve, t)
    ref.body = SimpleNamespace(
        J=rng.normal(scale=.16, size=(22, 3)), scale=.88,
        parents=np.array([-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]),
    )
    ref.binding = AnklePivotBinding()
    state = problem.body.state = SequenceState(ref, (0., .14))
    # Only mesh integration is skipped. Production FK, the proxy mass COM,
    # right-relative ankle rotation and both global sparse chains run normally.
    mass = ContinuousMassGeometry.__new__(ContinuousMassGeometry)
    mass.ref, mass.binding = ref, ref.binding
    mass.ids = np.array([0, 7, 8, 20])
    mass.count = len(mass.ids)
    mass.mass = np.array([.31, .14, .13, .32, .04, .06])
    mass.offsets = rng.normal(scale=.09, size=(len(mass.mass), 3))
    x = rng.normal(scale=.017, size=state.size)
    dense = ModelCOMMotionFactor(problem, mass, DENSE_CONFIG)
    legacy = ModelCOMMotionFactor(problem, mass, CONFIG)

    actual = dense.evaluate(x, jacobian=True)
    original = legacy.evaluate(x, jacobian=True)
    dense_terms = blocks(actual)
    ground = dense_terms[CURVATURE]
    np.testing.assert_array_equal(ground.residual, original["residual"])
    assert (ground.jacobian - original["jac"]).nnz == 0
    assert actual["parts"][CURVATURE] == original["parts"][CURVATURE]
    np.testing.assert_array_equal(
        actual["residual"], np.concatenate([term.residual for term in dense_terms.values()]),
    )
    assert actual["jac"].shape == (len(actual["residual"]), 17 + state.size)

    body_columns = np.arange(state.body_size)
    shoe_columns = np.arange(state.body_size, state.size)
    for columns in (body_columns, shoe_columns, np.arange(state.size)):
        direction = np.zeros(state.size)
        direction[columns] = rng.normal(size=len(columns))
        direction /= np.linalg.norm(direction)
        full_direction = np.r_[np.zeros(17), direction]
        eps = 1e-5
        plus = blocks(dense.evaluate(x + eps * direction))
        minus = blocks(dense.evaluate(x - eps * direction))
        for episode_id in ("entry", "later"):
            name = f"episode/{episode_id}/{DENSE_FAMILY}"
            matrix = dense_terms[name].jacobian
            assert matrix.format == "csr" and matrix[:, :17].nnz == 0
            predicted = matrix @ full_direction
            assert np.linalg.norm(predicted) > 1e-3
            numeric = (plus[name].residual - minus[name].residual) / (2 * eps)
            np.testing.assert_allclose(predicted, numeric, atol=1e-7, rtol=1e-6)
