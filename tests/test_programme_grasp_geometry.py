"""Analytic closest-point cases for the programme palm-to-shoe factor."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from calibration.program_tracking.body_tracking.programme_grasp import closest_triangle


TRIANGLE = np.array([[0., 0., 0.], [2., 0., 0.], [0., 2., 0.]])


@pytest.mark.parametrize(
    ("point", "expected", "expected_barycentric", "squared_distance"),
    [
        ([.5, .75, 3.], [.5, .75, 0.], [.375, .25, .375], 9.),
        ([1.5, 1.5, 2.], [1., 1., 0.], [0., .5, .5], 4.5),
        ([3., -1., 2.], [2., 0., 0.], [0., 1., 0.], 6.),
    ],
    ids=["interior-foot", "edge", "vertex"],
)
def test_closest_point_on_finite_triangle(point, expected, expected_barycentric, squared_distance):
    closest, barycentric, squared = closest_triangle(point, TRIANGLE[None])
    np.testing.assert_allclose(closest[0], expected, rtol=0., atol=1e-14)
    np.testing.assert_allclose(barycentric[0], expected_barycentric, rtol=0., atol=1e-14)
    np.testing.assert_allclose(squared, [squared_distance], rtol=0., atol=1e-14)


def test_degenerate_triangles_project_onto_segment_and_point():
    triangles = np.array([
        [[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]],
        [[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]],
    ])
    closest, barycentric, squared = closest_triangle([.5, 2., 1.], triangles)
    np.testing.assert_allclose(closest, [[.5, 0., 0.], [1., 1., 1.]], rtol=0., atol=1e-14)
    np.testing.assert_allclose(squared, [5., 1.25], rtol=0., atol=1e-14)
    np.testing.assert_allclose(barycentric.sum(axis=1), 1., rtol=0., atol=1e-14)
    assert np.all(barycentric >= 0.)
    np.testing.assert_allclose(
        np.einsum("ni,nij->nj", barycentric, triangles), closest, rtol=0., atol=1e-14)


def test_integer_inputs_match_floating_inputs():
    triangles = np.stack([TRIANGLE, TRIANGLE + [2., 2., 0.]])
    point = np.array([1, 1, 2])
    floating = closest_triangle(point.astype(float), triangles)
    integer = closest_triangle(point, triangles.astype(int))
    for actual, expected in zip(integer, floating, strict=True):
        np.testing.assert_array_equal(actual, expected)


def test_closest_point_commutes_with_rigid_transform():
    triangles = np.stack([TRIANGLE, TRIANGLE + [0., 1.5, -1.], TRIANGLE + [3., 3., 1.]])
    point = np.array([.5, .75, 3.])
    closest, barycentric, squared = closest_triangle(point, triangles)

    rotation = Rotation.from_rotvec([.4, -.7, .2]).as_matrix()
    translation = np.array([4., -3., 1.])
    transformed, moved_barycentric, moved_squared = closest_triangle(
        point @ rotation.T + translation, triangles @ rotation.T + translation)
    np.testing.assert_allclose(transformed, closest @ rotation.T + translation, rtol=0., atol=2e-14)
    np.testing.assert_allclose(moved_barycentric, barycentric, rtol=0., atol=2e-14)
    np.testing.assert_allclose(moved_squared, squared, rtol=0., atol=2e-14)
