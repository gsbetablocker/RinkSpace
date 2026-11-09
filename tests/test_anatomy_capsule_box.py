"""Numerical regressions for the programme capsule-to-box separation."""

import numpy as np
from scipy.spatial.transform import Rotation

from calibration.program_tracking.body_tracking.programme_anatomy import capsule_box_separation


def test_interior_face_switch_in_rotated_box():
    # The active interior face changes from -X to +Y at the minimum.
    # The same line can start and end inside or pass fully through the box.
    x = np.array([0.9, 2.0])
    a = np.column_stack((-x, 0.7 - 2 * x / 9, np.zeros_like(x)))
    b = np.column_stack((x, 0.7 + 2 * x / 9, np.zeros_like(x)))
    rotation = Rotation.from_rotvec([0.3, -0.2, 0.5]).as_matrix()
    centres = np.array([[5., 8., 1.], [3., 1., 4.]])
    rotations = np.broadcast_to(rotation, (2, 3, 3))
    a = a @ rotation.T + centres
    b = b @ rotation.T + centres
    np.testing.assert_allclose(
        capsule_box_separation(a, b, centres, rotations, np.ones(3)), -47 / 110, rtol=0., atol=1e-14)
    np.testing.assert_allclose(
        capsule_box_separation(b, a, centres, rotations, np.ones(3)), -47 / 110, rtol=0., atol=1e-14)


def test_outside_stationary_point_and_degenerate_segments():
    a = np.array([[2., 3., 0.], [0., 0., 0.], [-2., 1., 0.], [2., 3., 0.]])
    b = np.array([[3., 1., 0.], [0., 0., 0.], [2., 1., 0.], [2., 3., 0.]])
    centres = np.zeros_like(a)
    rotations = np.broadcast_to(np.eye(3), (len(a), 3, 3))
    expected = [np.sqrt(3.2), -1., 0., np.sqrt(5.)]
    np.testing.assert_allclose(
        capsule_box_separation(a, b, centres, rotations, np.ones(3)), expected, rtol=0., atol=1e-14)


def test_batched_boxes_match_dense_sampling_bound():
    rng = np.random.default_rng(31)
    a, b = rng.uniform(-2, 2, (2, 48, 3))
    centres = np.zeros_like(a)
    rotations = np.broadcast_to(np.eye(3), (len(a), 3, 3))
    half = rng.uniform(0.6, 1.2, a.shape)
    result = capsule_box_separation(a, b, centres, rotations, half)

    sample_count = 10001
    times = np.linspace(0, 1, sample_count)
    points = a[:, None] + times[None, :, None] * (b - a)[:, None]
    excess = np.abs(points) - half[:, None]
    sdf = np.linalg.norm(np.maximum(excess, 0.), axis=-1) + np.minimum(excess.max(-1), 0.)
    sampled_minimum = sdf.min(axis=1)
    # The SDF is 1-Lipschitz and the nearest grid point is at most half a step away.
    sampling_error = np.linalg.norm(b - a, axis=1) / (2 * (sample_count - 1))
    assert np.all(result <= sampled_minimum + 1e-12)
    assert np.all(sampled_minimum - result <= sampling_error + 1e-12)
