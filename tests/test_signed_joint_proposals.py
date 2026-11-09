"""Regression checks for proposal-only signed anatomical boundaries."""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from scipy.spatial.transform import Rotation

from calibration.program_tracking.body_tracking.absolute_motion_curve import (
    AbsoluteMotionCurve, RotationChartDomainError, clamped_cubic_knots,
)
from calibration.program_tracking.body_tracking.anatomical_chart import JointAngles, JointChart, JointEnvelope
from calibration.program_tracking.body_tracking.programme_guard_proposals import ProgrammeGuardProposals
from calibration.program_tracking.body_tracking.programme_joint_limits import NAMES, ProgrammeJointLimits, authored_bases
from calibration.program_tracking.body_tracking.programme_source_step import SourceMargins
from calibration.program_tracking.body_tracking.sequence_skate_state import SequenceState
from calibration.program_tracking.body_tracking.weighted_blocks import block, full_columns


def test_signed_rows_restore_original_hinge_and_keep_inside_distances():
    angles = JointAngles(
        np.array([-.375, .25, 1.25, .75]),
        np.array([[.125, 0.], [0., .125], [.75, 0.], [0., .5]]),
        np.array([.125, .125, .75, .5]), np.zeros(4, bool), np.zeros(4, bool),
    )
    envelope = JointEnvelope(-.25, 1., [0., 1.], [[.25, .375], [.5, .625]], .25, .125, "test")
    expected_signed = [[.5, -5.5, -1.], [-2., -3., -2.5], [-6., 1., 2.], [-4., -1., -.5]]
    expected_hinge = [[.5, 0., 0.], [0., 0., 0.], [0., 1., 2.], [0., 0., 0.]]
    np.testing.assert_array_equal(envelope.signed_residual(angles), expected_signed)
    np.testing.assert_array_equal(envelope.residual(angles), expected_hinge)


@pytest.mark.parametrize(("rotvec", "message"), [([0., np.pi, 0.], "singular"), ([np.pi, 0., 0.], "branch seam")])
def test_signed_boundary_rejects_original_chart_domain_failures(rotvec, message):
    rest = np.eye(3)[None]
    chart = JointChart(rest, rest, [[0.]])
    rotation = Rotation.from_rotvec(rotvec).as_matrix()[None, None]
    angles = chart.evaluate(rotation)
    envelope = JointEnvelope(-.25, 1., [0., 1.], [.25, .5], .25, .125, "test")
    for evaluate in (envelope.residual, envelope.signed_residual):
        with pytest.raises(RotationChartDomainError, match=message):
            evaluate(angles)


def small_joint_problem():
    joints = np.zeros((22, 3))
    joints[[1, 2, 4, 5, 7, 8, 10, 11, 12, 18, 19, 20, 21]] = [
        [.1, 0., 0.], [-.1, 0., 0.], [.1, 0., -.4], [-.1, 0., -.4],
        [.1, 0., -.8], [-.1, 0., -.8], [.1, -.2, -.8], [-.1, -.2, -.8],
        [0., 0., .4], [.4, 0., .25], [-.4, 0., .25], [.65, 0., .25], [-.65, 0., .25],
    ]
    hip = NAMES.index("right_hip")
    hip_axis = authored_bases(joints)[hip, :, 0]
    local = np.broadcast_to(np.eye(3), (21, 3, 3)).copy()
    local[1] = Rotation.from_rotvec(.9 * hip_axis).as_matrix()
    times = np.array([0., .36, 1.])
    knots = clamped_cubic_knots([0., .53, 1.])
    n_controls = len(knots) - 4
    curve = AbsoluteMotionCurve(
        knots, np.zeros((n_controls, 3)),
        np.broadcast_to(np.concatenate((np.eye(3)[None], local)), (n_controls, 22, 3, 3)).copy(),
    )
    reference = SimpleNamespace(
        absolute_curve=curve, times=times, p={"pts": np.array([100, 460, 1100])},
        body=SimpleNamespace(J=joints),
        at=lambda t: {"local_R": np.broadcast_to(local, (len(t), *local.shape))},
    )
    state = SequenceState(reference, (0., 1.))
    body = SimpleNamespace(
        state=state, of=np.array([0]), oj=np.array([5]), frames=np.array([0]),
        cache={"uv": np.zeros((1, 17, 2))}, layout=SimpleNamespace(body_landmarks=np.arange(17)),
        obs={"uv": np.zeros((1, 17, 2)), "sigma_px": np.ones((1, 17))},
    )
    problem = SimpleNamespace(
        reference=reference, body=body, size=17 + state.size,
        cache_x=np.zeros(17 + state.size), image_time_weight=.2,
        objective_block_weights={"anatomical_joint_domain": 1., "surface_intersection": 1.},
    )
    envelope = dict(flex_degrees=np.rad2deg([-.25, 1.]), flex_grid_degrees=np.rad2deg([0., 1.]),
                    swing_axes_degrees=np.rad2deg([.3, .3]), scale_degrees=np.rad2deg([.25, .125]))
    config = dict(provenance="test", envelopes={name.rsplit("_", 1)[-1]: envelope for name in NAMES})
    factor = ProgrammeJointLimits(problem, config, problem.cache_x[17:])
    return problem, factor, hip_axis


def test_guard_proposals_see_inside_hip_boundary_with_original_caps_and_remote_columns():
    problem, joint, hip_axis = small_joint_problem()

    class ProgrammeSurfaceFactor:
        weight = 1.
        last_gap = np.array([.1])

    surface = ProgrammeSurfaceFactor()
    problem.action_factors = SimpleNamespace(existing=SimpleNamespace(factors=[joint, surface]))
    x = problem.cache_x
    hinge = joint.evaluate(x[17:], jacobian=True)
    signed, raw_jac = joint.proposal_linearization(x[17:])
    np.testing.assert_array_equal(np.maximum(signed, 0.) * joint.weights.repeat(30), hinge["residual"])
    value = {"weighted_blocks": hinge["weighted_blocks"] + [
        full_columns(block("surface_intersection", [0.], csr_matrix((1, problem.size - 17)))),
        full_columns(block("body_pixels", [0., 0.], csr_matrix((2, problem.size - 17)))),
    ]}
    source = SourceMargins(csr_matrix((0, problem.size)), np.empty(0), [])
    source.body_groups = [dict(indices=np.array([0]), name="torso", margin=.5)]
    joint_caps = {name: 0. for name in NAMES}
    joint_caps["right_knee"] = .17
    caps_before = joint_caps.copy()

    def evaluate(trial, _jacobian):
        surface.last_parameters = trial[17:].copy()
        return joint.evaluate(trial[17:])

    problem.evaluate = evaluate

    def original_guard(trial_value, _relative_limit):
        original = {"torso": {"rejected": False}}
        for row in trial_value["metrics"]["anatomical_joint_domain"]["joints"]:
            name = row["name"]
            original["joint_peak_" + name] = {
                "rejected": row["peak_dimensionless_violation"] > joint_caps[name] + 1e-5,
            }
        return original

    legacy = ProgrammeGuardProposals(problem, value, source, original_guard, joint_caps, .1)
    revised = ProgrammeGuardProposals(problem, value, source, original_guard, joint_caps, .1, signed_joint_domains=True)
    shape = (len(joint.times), len(NAMES), 3)
    hip = NAMES.index("right_hip")
    row = np.ravel_multi_index((1, hip, 1), shape)
    key = "__joint__/" + str(row)
    assert hinge["residual"][row] == 0. and hinge["jac"][row].nnz == 0
    assert key not in legacy.margins[2]
    assert signed[row] == pytest.approx(-.4)
    assert raw_jac[row].nnz > 0
    assert revised.joint_jac[:, :17].nnz == 0
    np.testing.assert_array_equal(revised.joint_caps, [caps_before[name] + 1e-5 for name in NAMES])
    assert joint_caps == caps_before

    A, slack, keys = revised.margins
    j = list(keys).index(key)
    assert slack[j] == pytest.approx(.8 * (.4 + 1e-5))

    controls = np.arange(joint.state.body_size).reshape(-1, 69) + 17
    hip_cols = controls[:, 9:12]
    P_cols = controls[:, :3]
    move = np.zeros_like(x)
    move[hip_cols] = .22 * hip_axis
    move[P_cols] = [.08, -.02, .04]
    assert (A[j] @ move).item() == pytest.approx(.22 / .25, abs=1e-8)
    assert (A[j] @ move).item() > slack[j]
    assert original_guard(evaluate(x + move, False), .05)["joint_peak_right_hip"]["rejected"]
    rejected = revised.check(x + move)[key]
    assert rejected["rejected"]
    assert rejected["before"] == pytest.approx(1e-5)
    assert rejected["after"] == pytest.approx((.9 + .22 - 1.) / .25)

    restricted = move.copy()
    restricted[hip_cols] *= slack[j] / (A[j] @ move).item()
    np.testing.assert_array_equal(restricted[P_cols], move[P_cols])
    assert np.max(A @ restricted - slack) < 1e-8
    assert not revised.check(x + restricted)[key]["rejected"]
    assert not original_guard(evaluate(x + restricted, False), .05)["joint_peak_right_hip"]["rejected"]
