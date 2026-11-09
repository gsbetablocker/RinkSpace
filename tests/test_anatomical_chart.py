import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from calibration.program_tracking.body_tracking.absolute_motion_curve import RotationChartDomainError
from calibration.program_tracking.body_tracking.anatomical_chart import JointChart, JointEnvelope


def _exp(rotvec):
    return Rotation.from_rotvec(rotvec).as_matrix()


def _pose(rest, basis, twist, swing):
    # Compose in the child rest frame, with a separately authored rest rotation.
    return rest @ basis @ _exp(np.r_[0., swing]) @ _exp([twist, 0., 0.]) @ basis.T


def test_anatomical_chart_rest_and_axial_mirror():
    rest_left = _exp([.3, -.4, .1])
    basis_left = _exp([-.3, .5, .2])
    mirror = np.diag([-1., 1., 1.])
    rest = np.stack((rest_left, mirror @ rest_left @ mirror))
    basis = np.stack((basis_left, -mirror @ basis_left))
    cases = [(0., [0., 0.]), (-.12, [0., 0.]), (1.7, [0., 0.]),
             (1.1, [.21, -.14]), (1.1, [1e-12, 0.])]
    left = np.stack([_pose(rest_left, basis_left, twist, swing) for twist, swing in cases])
    rotations = np.stack((left, mirror @ left @ mirror), axis=1)
    chart = JointChart.from_reference(rotations, rest, basis, initial_twist=[0., 0.])
    angles = chart.evaluate(rotations)
    twist = np.array([case[0] for case in cases])
    swing = np.array([case[1] for case in cases])
    np.testing.assert_allclose(angles.twist, np.repeat(twist[:, None], 2, axis=1), atol=1e-14)
    np.testing.assert_allclose(angles.swing_vector, np.repeat(swing[:, None], 2, axis=1), atol=1e-14)
    np.testing.assert_allclose(angles.swing, np.linalg.norm(angles.swing_vector, axis=-1), atol=1e-14)
    assert not angles.singular.any()
    assert not angles.outside_branch.any()


def test_reference_crosses_both_principal_cuts_and_restores_frozen_indices():
    identity = np.repeat(np.eye(3)[None], 2, axis=0)
    increasing = np.deg2rad([0., 80., 160., 179., 180., 181., 250., 330., 360.])
    twists = np.column_stack((increasing, -increasing))
    rotvecs = np.zeros((*twists.shape, 3))
    rotvecs[..., 0] = twists
    rotations = _exp(rotvecs.reshape(-1, 3)).reshape(*twists.shape, 3, 3)
    chart = JointChart.from_reference(rotations, identity, identity, initial_twist=0.)
    np.testing.assert_allclose(chart.reference_twist, twists, atol=1e-14)

    # Simulate serialization, then evaluate the same SO(3) rotation at two
    # anchors that differ by a turn. The late trial retains its winding.
    saved = chart.reference_twist.tolist()
    restored = JointChart(chart.rest, chart.basis, saved)
    samples = [8, 0, 8, 4]
    trial = rotations[samples].copy()
    trial[:3] = _exp([.03, 0., 0.])
    result = restored.evaluate(trial, sample_indices=samples)
    expected = np.array([[2*np.pi+.03, -2*np.pi+.03], [.03, .03],
                         [2*np.pi+.03, -2*np.pi+.03], [np.pi, -np.pi]])
    np.testing.assert_allclose(result.twist, expected, atol=1e-14)
    joints = [1, 0, 0, 1]
    paired = restored.evaluate(trial[np.arange(4), joints], sample_indices=samples, joint_indices=joints)
    np.testing.assert_allclose(paired.twist, expected[np.arange(4), joints], atol=1e-14)
    envelope = JointEnvelope(-.4, 3.1, [0., 3.1], [.2, .2], .5, .3, "synthetic test")
    assert envelope.residual(paired)[0, 0] > 10.
    assert envelope.residual(paired)[2, 1] > 6.
    assert np.all(envelope.residual(paired)[1] == 0.)
    assert not restored.reference_twist.flags.writeable


def test_swing_singularity_and_branch_seam_are_distinguished():
    rest = basis = np.eye(3)[None]
    chart = JointChart(rest, basis, [[0.], [0.]])
    rotations = _exp([[0., np.pi, 0.], [np.pi, 0., 0.]])[:, None]
    angles = chart.evaluate(rotations)
    np.testing.assert_array_equal(angles.singular[:, 0], [True, False])
    np.testing.assert_array_equal(angles.outside_branch[:, 0], [False, True])
    np.testing.assert_allclose(angles.swing[:, 0], [np.pi, 0.], atol=1e-14)
    assert np.isnan(angles.twist[0, 0])
    assert np.isnan(angles.swing_vector[0, 0]).all()
    envelope = JointEnvelope(-.4, 3.1, [0., 3.1], [.2, .2], .5, .3, "synthetic test")
    for sample, message in enumerate(("singular", "branch seam")):
        selected = chart.evaluate(rotations[[sample]], sample_indices=[sample])
        with pytest.raises(RotationChartDomainError, match=message):
            envelope.residual(selected)
    with pytest.raises(RotationChartDomainError, match="singular"):
        JointChart.from_reference(rotations[:1], rest, basis, initial_twist=0.)
    with pytest.raises(RotationChartDomainError, match="ambiguous"):
        JointChart.from_reference(rotations[1:], rest, basis, initial_twist=0.)


def test_elliptic_envelope_couples_twist_and_swing_direction():
    rest = basis = np.eye(3)[None]
    rotations = np.stack([_pose(rest[0], basis[0], twist, swing)
                          for twist, swing in [(0., [.4, 0.]), (0., [0., .4]), (1., [.4, 0.])]])[:, None]
    angles = JointChart.from_reference(rotations, rest, basis, initial_twist=0.).evaluate(rotations)
    envelope = JointEnvelope(-.2, 1.2, [0., 1.], [[.2, .5], [.6, .3]], .3, .2, "synthetic test")
    np.testing.assert_allclose(envelope.residual(angles)[:, 0], [[0., 0., 1.], [0., 0., 0.], [0., 0., 0.]],
                               atol=1e-14)


def test_joint_rows_directional_difference_uses_right_local_tangent():
    rest = _exp([.7, -.1, .3])
    basis = _exp([.3, -.4, .5])
    twist, swing = 1.35, .54
    flex_speed, swing_speed = -.71, .63
    swing_axis = np.array([0., .6, .8])
    rotation = _pose(rest, basis, twist, swing * swing_axis[1:])
    chart = JointChart.from_reference(rotation[None, None], rest[None], basis[None], initial_twist=twist)
    envelope = JointEnvelope(-.5, 1.15, [0., 2.], [[.2, .4], [.32, .32]], .6, .4, "synthetic test")

    # If D=S*T, the right tangent of D is T.T times the swing derivative
    # plus the axial twist derivative, converted out of the chart basis.
    T = _exp([twist, 0., 0.])
    chart_direction = T.T @ (swing_speed * swing_axis) + [flex_speed, 0., 0.]
    model_direction = basis @ chart_direction
    eps = 1e-6
    trial = rotation @ _exp(np.stack((-eps * model_direction, eps * model_direction)))
    angles = chart.evaluate(trial, sample_indices=[0, 0], joint_indices=[0, 0])
    np.testing.assert_allclose(np.diff(angles.twist)[0] / (2 * eps), flex_speed, atol=3e-10)
    np.testing.assert_allclose(np.diff(angles.swing_vector, axis=0)[0] / (2 * eps),
                               swing_speed * swing_axis[1:], atol=3e-10)

    axis_slopes = np.array([.06, -.04])
    limits = np.array([.2, .4]) + twist * axis_slopes
    allowed = np.linalg.norm(swing_axis[1:] / limits) ** -1
    allowed_slope = allowed ** 3 * np.sum(swing_axis[1:] ** 2 * axis_slopes / limits ** 3)
    expected = [0., flex_speed / envelope.flex_scale,
                (swing_speed - allowed_slope * flex_speed) / envelope.swing_scale]
    difference = np.diff(envelope.residual(angles), axis=0)[0] / (2 * eps)
    np.testing.assert_allclose(difference, expected, atol=5e-10)


def test_invalid_basis_and_envelope_rejected():
    identity = np.eye(3)[None]
    with pytest.raises(ValueError, match="proper SO"):
        JointChart(identity, -identity, [[0.]])
    with pytest.raises(ValueError, match="provenance"):
        JointEnvelope(-.5, 1.15, [0., 2.], [.2, .4], .6, .4, "")
    with pytest.raises(ValueError, match="coupled"):
        JointEnvelope(-.5, 1.15, [0., 2.], [[.2, .4], [0., .4]], .6, .4, "synthetic test")
