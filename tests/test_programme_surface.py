"""Synthetic checks for sampled programme surface intersections."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from scipy.spatial.transform import Rotation

from calibration.program_tracking.body_tracking.absolute_motion_state import MotionDelta
from calibration.program_tracking.body_tracking.compact_body_derivatives import CompactBodyLinearization
from calibration.program_tracking.body_tracking.fullbody_fit import right_jacobian
from calibration.program_tracking.body_tracking.programme_surface import (
    ProgrammeSurfaceFactor, crossing_witnesses, edge_triangle, triangle_pairs, witnesses_packet,
)
from calibration.program_tracking.body_tracking.selected_surface_linearization import SelectedSurfaceLinearization
from calibration.program_tracking.body_tracking.skate_physics import fk_state
from rinkspace.viewer3d.prepare import FullSurface
from test_compact_body_performance import synthetic_model


def _crossing_triangles():
    # The first face has a +Z normal. Edge (3, 4) passes through its interior.
    vertices = np.array([[0., 0., 0.], [4., 0., 0.], [0., 4., 0.],
                         [1., 1., -1.], [1., 1., 1.], [9., 1., 1.]])
    return vertices, np.arange(6).reshape(2, 3)


def _unordered_pairs(pairs):
    return {tuple(sorted(pair)) for pair in pairs.tolist()}


def test_triangle_candidates_keep_crossings_across_radius_buckets():
    vertices, faces = _crossing_triangles()
    # Face 6 shares a bounding-sphere candidate with face 0 but is above its AABB.
    vertices = np.concatenate((vertices, 0.04 * vertices + [30., 0., 0.],
                               2.0 * vertices + [60., 0., 0.], vertices[:3] + [0., 0., 1.5]))
    faces = np.concatenate((faces, faces + 6, faces + 12, [faces[0] + 18]))
    pairs = triangle_pairs(vertices, faces)
    assert _unordered_pairs(pairs) == {(0, 1), (2, 3), (4, 5)}
    assert len(pairs) == 3
    _, crossed = crossing_witnesses(vertices, faces, pairs)
    assert _unordered_pairs(crossed) == _unordered_pairs(pairs)


def test_triangle_candidates_filter_mesh_neighbours_but_keep_disconnected_crossing():
    vertices, faces = _crossing_triangles()
    vertices = np.concatenate((vertices, vertices + [30., 0., 0.]))
    faces = np.concatenate((faces, [[0, 4, 5]], faces + 6))
    assert _unordered_pairs(triangle_pairs(vertices, faces)) == {(0, 1), (0, 2), (1, 2), (3, 4)}

    # Faces 0 and 1 share no vertex but are within the supplied topology.
    # Face 2 directly shares vertices with both.
    topology = np.eye(len(vertices), dtype=bool)
    topology[0, 3] = topology[3, 0] = True
    result = triangle_pairs(vertices, faces, topology=csr_matrix(topology))
    assert _unordered_pairs(result) == {(3, 4)}


def test_native_pairs_keep_legacy_order_with_dense_and_mixed_radius_faces():
    from calibration.program_tracking.body_tracking.surface_broad_phase import SurfaceBroadPhase

    seed, _ = _crossing_triangles()
    rng = np.random.default_rng(4)
    vertices = np.concatenate([
        seed * scale + rng.uniform(-.6, .6, size=3)
        for scale in np.tile([.6, 1., 1.1, 2.], 28)
    ])
    faces = np.arange(len(vertices)).reshape(-1, 3)
    excluded = csr_matrix((np.ones(len(vertices), dtype=bool),
                           (np.arange(len(vertices)), np.arange(len(vertices)))),
                          shape=(len(vertices), len(vertices)))
    native = SurfaceBroadPhase.optional(faces, excluded)
    if native is None:
        pytest.skip('Native surface accelerator not configured')
    expected = triangle_pairs(vertices, faces, topology=excluded)
    assert len(expected) > 2 * len(faces)
    np.testing.assert_array_equal(native(vertices), expected)


def test_crossing_witness_uses_negative_endpoint_and_oriented_face_normal():
    vertices, faces = _crossing_triangles()
    hit, time, bary = edge_triangle(vertices[[3]], vertices[[4]], vertices[faces[[0]]])
    np.testing.assert_array_equal(hit, [True])
    np.testing.assert_allclose(time, [0.5], atol=1e-14)
    np.testing.assert_allclose(bary, [[0.5, 0.25, 0.25]], atol=1e-14)

    for face, expected_vertex, direction in ((faces[0], 3, 1.), (faces[0, ::-1], 4, -1.)):
        oriented_faces = np.array([face, faces[1]])
        witnesses, crossed = crossing_witnesses(vertices, oriented_faces, np.array([[0, 1]]))
        np.testing.assert_array_equal(crossed, [[0, 1]])
        row, = [witness for witness in witnesses if witness[6] == 0]
        vertex, support, weights, normal, gap, source_face, _ = row
        assert (vertex, source_face) == (expected_vertex, 1)
        np.testing.assert_array_equal(support, face)
        np.testing.assert_allclose(normal, [0., 0., direction], atol=1e-14)
        np.testing.assert_allclose(weights @ vertices[support], [1., 1., 0.], atol=1e-14)
        np.testing.assert_allclose(gap, -1., atol=1e-14)
        displaced_gap = normal @ (vertices[vertex] + 0.2 * normal - weights @ vertices[support])
        np.testing.assert_allclose(displaced_gap, -0.8, atol=1e-14)


def test_crossing_signed_gaps_are_rigid_transform_invariant():
    vertices, faces = _crossing_triangles()
    pairs = np.array([[0, 1]])
    rotation = Rotation.from_rotvec([0.34, -0.28, 0.49]).as_matrix()
    moved = vertices @ rotation.T + [2., -7., 4.]
    original, before = crossing_witnesses(vertices, faces, pairs)
    transformed, after = crossing_witnesses(moved, faces, pairs)
    np.testing.assert_array_equal(before, after)
    original_by_key = {(row[0], row[6]): row for row in original}
    transformed_by_key = {(row[0], row[6]): row for row in transformed}
    assert original_by_key
    assert original_by_key.keys() == transformed_by_key.keys()
    for key, row in original_by_key.items():
        other = transformed_by_key[key]
        np.testing.assert_allclose(other[3], rotation @ row[3], atol=1e-14)
        np.testing.assert_allclose(other[4], row[4], atol=1e-14)


def _surface(body):
    surface = FullSurface.__new__(FullSurface)
    surface.body = body
    surface.shaped = body.shaped.astype(np.float32)
    surface.posedirs = body.posedirs.astype(np.float32)
    surface.weights = body.W.astype(np.float32)
    surface._sample_double_cache = None
    return surface


def test_selected_skin_derivative_matches_finite_difference_on_rotvec_chart(tmp_path):
    body = synthetic_model(tmp_path / 'surface.npz')
    surface = _surface(body)
    rng = np.random.default_rng(431104)
    reference = Rotation.from_rotvec(rng.normal(0., .8, (84, 3))).as_matrix().reshape(4, 21, 3, 3)
    delta = rng.normal(0., .4, (4, 21, 3))
    delta[0] = 0.
    delta[1] *= 1e-9
    delta[2] *= 4
    kinematics = CompactBodyLinearization(body, reference, delta, include_vertices=False)
    ids = np.array([8, 1, 8, 3, 30, 0, 24, 5])
    linear = SelectedSurfaceLinearization(surface, kinematics.native, kinematics.frames, ids, chunk_size=2)
    step = 1e-5
    for column in range(6, 69):
        joint_index, axis = divmod(column - 6, 3)
        tangent = right_jacobian(delta[:, joint_index])[:, :, axis]
        actual = linear.derivative(joint_index + 1, tangent)
        states = kinematics.kinematic_pair(column, step)
        plus, minus = (surface.sample_vertices(s['native'], s['frames'], ids, double_precision=True)
                       for s in states)
        np.testing.assert_allclose(actual, (plus - minus) / (2 * step), atol=3e-9, rtol=2e-7,
                                   err_msg=f"body column {column}")


def test_surface_factor_includes_both_sides_of_one_mesh_and_cancels_root_motion(tmp_path):
    vertices, faces = _crossing_triangles()
    body = synthetic_model(tmp_path / 'crossing.npz')
    surface = _surface(body)
    surface.shaped[:6] = vertices / body.scale + body.J[0]
    surface.posedirs[:] = 0.
    surface.weights[:6] = 0.
    surface.weights[:3, 2] = 1.
    surface.weights[3:6, 1] = 1.
    base = dict(P=np.zeros((1, 3)), Q=np.eye(3)[None],
                local_R=np.broadcast_to(np.eye(3), (1, 21, 3, 3)))

    def shoe_transforms(P, Q, native, rotations):
        return np.repeat(Q[:, None], 2, axis=1), np.repeat(P[:, None], 2, axis=1)

    class LocalState:
        size = 69

        def increments(self, x, times):
            return MotionDelta(x[None], self, x, times)

        def chain(self, local, x, times, *, sample_indices):
            np.testing.assert_array_equal(sample_indices, [0])
            return csr_matrix(local[:, 0])

    reference = SimpleNamespace(body=body, binding=SimpleNamespace(transforms=shoe_transforms),
                                at=lambda times: base, p={'pts': np.array([0])})
    x = np.zeros(LocalState.size)
    pose = fk_state(reference, base, x[None])
    points = surface.sample_vertices(pose['native'], pose['frames'], np.arange(6), double_precision=True)[0]
    rows, _ = crossing_witnesses(points, faces, triangle_pairs(points, faces))
    witness, = [row for row in rows if row[0] == 3 and row[6] == 0]

    factor = object.__new__(ProgrammeSurfaceFactor)
    factor.ref = reference
    factor.state = LocalState()
    factor.geometry = SimpleNamespace(surface=surface, skin_count=len(surface.shaped), ratio=1., origin=0.)
    factor.packet = witnesses_packet([(0, witness)])
    factor.times = np.array([0.])
    factor.scale_m = 1.
    factor.weight = np.ones(1)
    factor.association = 'synthetic-crossing'
    result = factor.evaluate(x, jacobian=True)
    np.testing.assert_allclose(result['residual'], [1.], atol=2e-7)
    shared, jacobian = np.split(result['jac'].toarray()[0], [17])
    np.testing.assert_array_equal(shared, 0.)
    assert np.linalg.norm(jacobian[6:9]) > .5
    assert np.linalg.norm(jacobian[9:12]) > .5

    step = 1e-6
    numerical = [(factor.evaluate(x + bump)['residual'][0] - factor.evaluate(x - bump)['residual'][0]) / (2 * step)
                 for bump in step * np.eye(len(x))[:12]]
    np.testing.assert_allclose(jacobian[:12], numerical, atol=3e-9, rtol=2e-7)
    np.testing.assert_array_equal(jacobian[:6], 0.)
    moved = x.copy()
    moved[:6] = [1.2, -3.4, 5.6, 0.7, -0.8, 0.9]
    np.testing.assert_array_equal(factor.evaluate(moved)['residual'], result['residual'])
