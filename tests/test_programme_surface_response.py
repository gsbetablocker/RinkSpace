"""Physical response directions for the open-collar surface witnesses."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from calibration.program_tracking.body_tracking.programme_surface import (
    CollisionSurface,
    crossing_witnesses,
    validate_response_policy,
)


LEGACY = "legacy_external_v1"
CAVITY = "own_collar_cavity_side_v1"


def crossing_wall():
    # The +Z side is outside. Only the first skin edge crosses inside the
    # triangle of the wall, between a deep interior and a shallow exterior end.
    vertices = np.array([
        [0., 0., 0.], [4., 0., 0.], [0., 4., 0.],
        [1., 1., -5.], [1., 1., 1.], [7., 1., 1.],
    ])
    return vertices, np.arange(6).reshape(2, 3)


def skin_to_wall(witnesses):
    row, = [w for w in witnesses if (w[5], w[6]) == (1, 0)]
    return row


def hinge_at(vertices, row, dz):
    vertex, support, bary, normal = row[:4]
    return max(-normal @ (vertices[vertex] + [0., 0., dz] - bary @ vertices[support]), 0.)


@pytest.mark.parametrize(("pair", "signs"), [([0, 1], [1, -1]), ([1, 0], [-1, 1])])
def test_own_collar_response_moves_exterior_skin_endpoint_inward_without_changing_crossing(pair, signs):
    vertices, faces = crossing_wall()
    pairs = np.array([pair])
    legacy, legacy_crossed = crossing_witnesses(vertices, faces, pairs)
    cavity, cavity_crossed = crossing_witnesses(vertices, faces, pairs, response_signs=np.array([signs]))
    np.testing.assert_array_equal(legacy_crossed, pairs)
    np.testing.assert_array_equal(cavity_crossed, pairs)

    old = skin_to_wall(legacy)
    new = skin_to_wall(cavity)
    assert (old[0], old[4]) == pytest.approx((3, -5.))
    assert (new[0], new[4]) == pytest.approx((4, -1.))
    np.testing.assert_array_equal(old[3], [0., 0., 1.])
    np.testing.assert_array_equal(new[3], [0., 0., -1.])
    np.testing.assert_allclose(new[2] @ vertices[new[1]], [1., 1., 0.], atol=1e-14)
    assert hinge_at(vertices, old, +.2) == pytest.approx(4.8)
    assert hinge_at(vertices, new, +.2) == pytest.approx(1.2)
    assert hinge_at(vertices, new, -.2) == pytest.approx(.8)

    reverse_old = [w for w in legacy if (w[5], w[6]) == (0, 1)]
    reverse_new = [w for w in cavity if (w[5], w[6]) == (0, 1)]
    assert reverse_old
    assert len(reverse_new) == len(reverse_old)
    for before, after in zip(reverse_old, reverse_new, strict=True):
        for a, b in zip(before, after, strict=True):
            np.testing.assert_array_equal(a, b)


def test_cavity_witness_is_invariant_under_joint_rigid_motion():
    vertices, faces = crossing_wall()
    pairs = np.array([[0, 1]])
    signs = np.array([[1, -1]])
    rotation = Rotation.from_rotvec([.31, -.42, .5]).as_matrix()
    moved = vertices @ rotation.T + [4., -2., 3.]
    original, before = crossing_witnesses(vertices, faces, pairs, response_signs=signs)
    transformed, after = crossing_witnesses(moved, faces, pairs, response_signs=signs)
    np.testing.assert_array_equal(before, after)
    old, new = skin_to_wall(original), skin_to_wall(transformed)
    assert (new[0], new[5], new[6]) == (old[0], old[5], old[6])
    np.testing.assert_array_equal(new[1], old[1])
    np.testing.assert_allclose(new[2], old[2], atol=1e-14)
    np.testing.assert_allclose(new[3], rotation @ old[3], atol=1e-14)
    assert new[4] == pytest.approx(old[4], abs=1e-14)


def test_response_reverses_only_own_shank_to_collar_wall():
    surface = CollisionSurface.__new__(CollisionSurface)
    # Faces: left shank, left collar, right shank, right collar,
    # left non-collar shoe face, and a different left skin face.
    surface.face_shank_foot = np.array([0, -1, 1, -1, -1, -1])
    surface.collar_wall_foot = np.array([-1, 0, -1, 1, -1, -1])
    pairs = np.array([[0, 1], [1, 0], [2, 3], [3, 2],
                      [0, 3], [3, 0], [2, 1], [0, 4], [5, 1], [1, 3]])
    surface.response_policy = CAVITY
    np.testing.assert_array_equal(surface.response_signs(pairs), [
        [-1, 1], [1, -1], [-1, 1], [1, -1],
        [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1],
    ])
    surface.response_policy = LEGACY
    np.testing.assert_array_equal(surface.response_signs(pairs), np.ones((len(pairs), 2)))


@pytest.mark.parametrize("signs", [np.array([1, -1]), np.array([[1, 0]]), np.array([[1, np.nan]])])
def test_crossing_rejects_invalid_response_signs(signs):
    vertices, faces = crossing_wall()
    with pytest.raises(ValueError):
        crossing_witnesses(vertices, faces, np.array([[0, 1]]), response_signs=signs)


def test_witness_packet_policy_requires_explicit_matching_cavity_response():
    legacy = {"surface_response_policy": np.array(LEGACY)}
    cavity = {"surface_response_policy": np.array(CAVITY)}
    validate_response_policy({}, LEGACY)
    validate_response_policy(legacy, LEGACY)
    validate_response_policy(cavity, CAVITY)

    for packet, policy in (({}, CAVITY), (legacy, CAVITY), (cavity, LEGACY),
                           ({"surface_response_policy": np.array("unknown")}, "unknown"),
                           ({"surface_response_policy": np.array([CAVITY])}, CAVITY)):
        with pytest.raises(ValueError):
            validate_response_policy(packet, policy)
