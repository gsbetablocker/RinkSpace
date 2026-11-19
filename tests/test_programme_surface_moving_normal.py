"""Current target-face normals on fixed surface witnesses."""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from calibration.program_tracking.body_tracking.absolute_motion_curve import (
    AbsoluteMotionCurve,
    clamped_cubic_knots,
)
from calibration.program_tracking.body_tracking.absolute_motion_state import (
    AbsoluteMotionState,
    evaluate_curve,
)
from calibration.program_tracking.body_tracking.programme_surface import (
    CAVITY_RESPONSE_POLICY,
    FROZEN_NORMAL_MODE,
    MOVING_NORMAL_MODE,
    CollisionSurface,
    OrientedFaceWitness,
    ProgrammeSurfaceFactor,
    crossing_witnesses,
    witnesses_packet,
)
from calibration.program_tracking.body_tracking.skate_physics import fk_state
from test_compact_body_performance import synthetic_model
from test_programme_surface import _crossing_triangles, _surface


def geometry_batch():
    triangles = np.array([
        [[0., 0., 0.], [2., 0., .2], [.3, 1.8, .6]],
        [[.7, -.5, -.4], [1.5, -.2, .7], [-.5, 1., .4]],
    ])
    points = np.array([[.5, .4, -.7], [.3, .5, .9]])
    return points, triangles, np.array([[.25, .50, .25], [.2, .3, .5]]), np.array([1., -1.])


def test_current_gap_is_invariant_under_common_rigid_motion_and_tangent():
    point, triangle, bary, signs = geometry_batch()
    witness = OrientedFaceWitness.from_geometry(point, triangle, bary, signs)
    rotation = Rotation.from_rotvec([.31, -.71, .82]).as_matrix()
    translate = np.array([4.2, -2.3, .71])
    transformed = OrientedFaceWitness.from_geometry(
        point @ rotation.T + translate, triangle @ rotation.T + translate, bary, signs,
    )
    np.testing.assert_allclose(transformed.gap, witness.gap, atol=2e-15, rtol=0)
    np.testing.assert_allclose(transformed.normal, witness.normal @ rotation.T, atol=8e-16, rtol=0)
    angular_rate = np.array([-.18, .72, -.28])
    translation_rate = np.array([.74, -.14, 1.5])
    np.testing.assert_allclose(
        witness.derivative(np.broadcast_to(translation_rate, point.shape),
                           np.broadcast_to(translation_rate, triangle.shape)),
        0., atol=2e-15, rtol=0,
    )
    np.testing.assert_allclose(
        witness.derivative(np.cross(angular_rate, point), np.cross(angular_rate, triangle)),
        0., atol=2e-15, rtol=0,
    )


def test_random_deformation_has_normal_and_both_side_derivatives():
    point, triangle, bary, signs = geometry_batch()
    rng = np.random.default_rng(514)
    witness = OrientedFaceWitness.from_geometry(point, triangle, bary, signs)
    eps = 1e-6
    for _ in range(5):
        dp = rng.normal(size=point.shape)
        dt = rng.normal(size=triangle.shape)
        plus = OrientedFaceWitness.from_geometry(point + eps * dp, triangle + eps * dt, bary, signs)
        minus = OrientedFaceWitness.from_geometry(point - eps * dp, triangle - eps * dt, bary, signs)
        expected = (plus.gap - minus.gap) / (2 * eps)
        np.testing.assert_allclose(witness.derivative(dp, dt), expected, atol=3e-10, rtol=0)
    frozen_derivative = np.einsum("ni,ni->n", witness.normal, dp - np.einsum("ni,nij->nj", bary, dt))
    assert np.linalg.norm(expected - frozen_derivative) > .05


def test_cavity_sign_pushes_exterior_skin_back_inside_the_oriented_wall():
    surface = CollisionSurface.__new__(CollisionSurface)
    surface.response_policy = CAVITY_RESPONSE_POLICY
    surface.face_shank_foot = np.array([0, -1, 1])
    surface.collar_wall_foot = np.array([-1, 0, -1])
    # Same wall, first pair is its own shank and second is the other shank.
    signs = surface.response_signs(np.array([[0, 1], [2, 1]]))[:, 0]
    triangle = np.repeat(np.array([[[0., 0., 0.], [2., 0., 0.], [0., 2., 0.]]]), 2, axis=0)
    points = np.array([[.4, .5, .3], [.4, .5, .3]])
    bary = np.repeat([[.55, .2, .25]], 2, axis=0)
    witness = OrientedFaceWitness.from_geometry(points, triangle, bary, signs)
    np.testing.assert_array_equal(witness.gap, [-.3, .3])
    inward = np.array([[0., 0., -.1], [0., 0., -.1]])
    np.testing.assert_array_equal(witness.derivative(inward, np.zeros_like(triangle)), [.1, -.1])
    moved = OrientedFaceWitness.from_geometry(points + inward, triangle, bary, signs)
    assert np.maximum(-moved.gap[0], 0.) < np.maximum(-witness.gap[0], 0.)


@pytest.mark.parametrize("last_vertex", [[0., 0., 0.], [2., 0., 0.], [2., 1e-16, 0.]])
def test_degenerate_current_triangle_raises_explicitly(last_vertex):
    triangle = np.array([[[0., 0., 0.], [1., 0., 0.], last_vertex]])
    with pytest.raises(ValueError, match="target face is degenerate"):
        OrientedFaceWitness.from_geometry([[.2, .4, .9]], triangle, [[.2, .3, .5]], [1.])


@pytest.fixture
def surface_factor_fixture(tmp_path):
    body = synthetic_model(tmp_path / "moving-face.npz")
    vertices, faces = _crossing_triangles()
    # The below-plane endpoint has a tangential offset from the saved
    # intersection barycentrics. Turning the face changes its unit normal.
    vertices[4] = [2., 1.4, 1.]
    skin = _surface(body)
    skin.faces = faces
    skin.shaped[:6] = vertices / body.scale + body.J[0]
    skin.posedirs[:] = 0.
    skin.weights[:6] = 0.
    skin.weights[:3, 3] = 1.
    skin.weights[3:6, 4] = 1.
    assert body.parents[3] == body.parents[4] == 1

    template = dict(vertices=np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]),
                    faces=np.array([[0, 1, 2]]), face_parts=["outsole"])
    binding = SimpleNamespace(
        ratio=1., surface=skin, faces=np.empty((0, 3), int), region_ids=np.empty(0, int),
        packet=lambda: dict(template=template, upper=np.empty((0, 3)),
                            parameters=SimpleNamespace(sole_length_m=1.)),
        transforms=lambda P, Q, native, frames: (np.repeat(Q[:, None], 2, axis=1),
                                                  np.repeat(P[:, None], 2, axis=1)),
    )
    knots = clamped_cubic_knots([0., .26, .62, 1.])
    controls = len(knots) - 4
    curve = AbsoluteMotionCurve(knots, np.zeros((controls, 3)),
                                np.broadcast_to(np.eye(3), (controls, 22, 3, 3)).copy())
    times = np.array([.18, .41, .78])
    ref = SimpleNamespace(body=body, binding=binding, absolute_curve=curve, times=times,
                          at=lambda t: evaluate_curve(curve, t),
                          p=dict(pts=np.array([18, 41, 78]), time_base=[1, 100]))
    state = AbsoluteMotionState(ref, (0., 1.))
    problem = SimpleNamespace(reference=ref, body=SimpleNamespace(state=state))
    x = np.zeros(state.size)
    initial = fk_state(ref, ref.at(times), state.increments(x, times))
    points = skin.sample_vertices(initial["native"], initial["frames"], np.arange(6), double_precision=True)
    rows, _ = crossing_witnesses(points[0], faces, np.array([[0, 1]]))
    witness, = [row for row in rows if row[0] == 3 and row[6] == 0]
    packet = witnesses_packet([(i, witness) for i in range(len(times))])
    packet["times"] = times
    return problem, packet, x


def test_legacy_freezes_normal_but_moving_mode_cancels_common_joint(surface_factor_fixture):
    problem, packet, x = surface_factor_fixture
    default = ProgrammeSurfaceFactor(problem, {}, packet)
    explicit = ProgrammeSurfaceFactor(
        problem, {"programme_structure": {"surface_normal_mode": FROZEN_NORMAL_MODE}}, packet,
    )
    moving = ProgrammeSurfaceFactor(
        problem, {"programme_structure": {"surface_normal_mode": MOVING_NORMAL_MODE}}, packet,
    )
    assert default.association == explicit.association != moving.association
    rotation = x.copy().reshape(-1, 69)
    rotation[:, 6:9] = [.26, -.48, .31]  # Common ancestor of both surface faces.
    y = rotation.ravel()
    actual = default.evaluate(y, jacobian=True)
    explicit_value = explicit.evaluate(y, jacobian=True)
    np.testing.assert_array_equal(actual["residual"], explicit_value["residual"])
    assert (actual["jac"] - explicit_value["jac"]).nnz == 0
    weight = np.sqrt([(.41 - .18) / 2, (.78 - .18) / 2, (.78 - .41) / 2]) / .03
    rest_offset = np.array([-.5, -.2, -1.])
    rotated_offset = Rotation.from_rotvec([.26, -.48, .31]).apply(rest_offset)
    np.testing.assert_allclose(actual["residual"], -rotated_offset[2] * weight, atol=2e-6, rtol=0)
    assert np.linalg.norm(actual["residual"] - moving.evaluate(y)["residual"]) > .1

    moved = moving.evaluate(y, jacobian=True)
    np.testing.assert_allclose(moved["residual"], moving.evaluate(x)["residual"], atol=4e-14, rtol=0)
    common_joint_columns = (17 + 69 * np.arange(len(rotation))[:, None] + np.arange(6, 9)).ravel()
    np.testing.assert_allclose(moved["jac"][:, common_joint_columns].toarray(), 0., atol=2e-13, rtol=0)


def test_actual_surface_sparse_chain_matches_body_and_target_rotation_directions(surface_factor_fixture):
    problem, packet, x = surface_factor_fixture
    legacy = ProgrammeSurfaceFactor(problem, {}, packet)
    moving = ProgrammeSurfaceFactor(
        problem, {"programme_structure": {"surface_normal_mode": MOVING_NORMAL_MODE}}, packet,
    )
    rng = np.random.default_rng(8329)
    controls = x.reshape(-1, 69).copy()
    controls[:, 6:9] = rng.normal(scale=.10, size=(len(controls), 3))
    controls[:, 12:18] = rng.normal(scale=.09, size=(len(controls), 6))
    x = controls.ravel()
    result = moving.evaluate(x, jacobian=True)
    matrix = result["jac"]
    assert matrix.format == "csr" and matrix.shape == (3, 17 + len(x))
    assert matrix[:, :17].nnz == 0 and np.all(result["residual"] > 1.)
    eps = 2e-6

    for first, last in ((12, 15), (15, 18), (0, 69)):
        local_direction = np.zeros_like(controls)
        local_direction[:, first:last] = rng.normal(size=(len(controls), last - first))
        direction = local_direction.ravel() / np.linalg.norm(local_direction)
        derivative = matrix @ np.r_[np.zeros(17), direction]
        plus = moving.evaluate(x + eps * direction)["residual"]
        minus = moving.evaluate(x - eps * direction)["residual"]
        assert np.linalg.norm(derivative) > .1
        np.testing.assert_allclose(derivative, (plus - minus) / (2 * eps), atol=2e-8, rtol=0)
        if first == 12:
            frozen = legacy.evaluate(x, jacobian=True)["jac"] @ np.r_[np.zeros(17), direction]
            assert np.linalg.norm(frozen - derivative) > .1


def test_proposal_has_signed_derivative_before_a_refreshed_witness_penetrates(surface_factor_fixture):
    problem, packet, x = surface_factor_fixture
    factor = ProgrammeSurfaceFactor(problem, {"programme_structure": {"surface_normal_mode": MOVING_NORMAL_MODE}}, packet)
    direction = np.zeros_like(x).reshape(-1,69)
    direction[:,15] = 1.
    direction = direction.ravel()
    for angle in (-2.4,-1.6,-.8,.8,1.6,2.4):
        z = angle*direction
        factor.evaluate(z)
        if factor.last_gap.min() > .01:
            break
    else:
        pytest.fail('Synthetic vertex could not be moved clear of the face')
    clipped = factor.evaluate(z,jacobian=True)
    signed = factor.evaluate(z,jacobian=True,signed_proposal=True)
    np.testing.assert_array_equal(clipped['residual'],0.)
    assert clipped['jac'].nnz == 0 and np.all(signed['residual']<0.)
    slope = signed['jac']@np.r_[np.zeros(17),direction]
    assert np.linalg.norm(slope)>.1
    h=2.e-6
    numeric=(factor.evaluate(z+h*direction,signed_proposal=True)['residual']-
             factor.evaluate(z-h*direction,signed_proposal=True)['residual'])/(2*h)
    np.testing.assert_allclose(slope,numeric,rtol=3.e-6,atol=2.e-8)
