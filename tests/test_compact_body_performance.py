"""Numerical regression checks for the exact compact CPU forward optimization.

The fixture is synthetic; no licensed SMPL-X assets are bundled. The independent
reference below is the unoptimized 78dba30 FK/LBS contraction order.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from calibration.program_tracking.body_tracking.rebuild_body_geometry import CompactBody, POSE_DOFS
from rinkspace.viewer3d.prepare import FullSurface


def original_forward(model, local):
    n = len(local)
    rotation = local.copy()
    position = np.broadcast_to(model.J, (n, 55, 3)).copy()
    for j in range(1, 55):
        parent = model.parents[j]
        rotation[:, j] = rotation[:, parent] @ local[:, j]
        position[:, j] = position[:, parent] + np.einsum(
            'nij,j->ni', rotation[:, parent], model.J[j] - model.J[parent])
    translation = position - np.einsum('njab,jb->nja', rotation, model.J)
    feature = (local[:, 1:] - np.eye(3)).reshape(n, 486)
    posed = model.shaped + (feature @ model.posedirs).reshape(n, -1, 3)
    blend_R = np.einsum('vj,njab->nvab', model.W, rotation, optimize=True)
    blend_t = np.einsum('vj,nja->nva', model.W, translation, optimize=True)
    vertices = np.einsum('nvab,nvb->nva', blend_R, posed) + blend_t
    coco = np.einsum('vj,nvc->njc', model.reg, vertices) - model.J[0]
    native = position[:, :22] - model.J[0]
    blade = model.scale * native[:, [7, 8], None] + np.einsum(
        'nfab,fkb->nfka', rotation[:, [7, 8]], model.blade_relative)
    return coco * model.scale, native * model.scale, blade, rotation[:, :22]


def synthetic_model(path):
    rng = np.random.default_rng(781103)
    v = 32
    weights = rng.random((v, 55)); weights /= weights.sum(1, keepdims=True)
    reg = rng.random((v, 17)); reg /= reg.sum(0, keepdims=True)
    np.savez(path,
        v_template=rng.normal(size=(v, 3)),
        shapedirs=rng.normal(scale=.01, size=(v, 3, 10)),
        posedirs=rng.normal(scale=.001, size=(486, v, 3)),
        lbs_weights=weights, smplx2coco17_interestd=reg,
        parents=np.r_[-1, (np.arange(1, 55) - 1) // 2],
        other_default_pose=rng.normal(scale=.3, size=99),
        J_template=rng.normal(scale=.1, size=(55, 3)),
        J_shapedirs=rng.normal(scale=.01, size=(55, 3, 10)),
        rest_patch_centroids=rng.normal(scale=.05, size=(2, 2, 3)),
        rest_patch_height=rng.random((2, 2)) * .05)
    return CompactBody(path, rng.normal(scale=.2, size=10), 1.13)


def original_surface(surface, joints, frames, vertex_ids=None, double_precision=False):
    """78dba30 skinning operations, including its display precision order."""
    body = surface.body; n = len(joints)
    rot = np.empty((n, 55, 3, 3)); position = np.empty((n, 55, 3))
    local = np.broadcast_to(np.eye(3), (n, 55, 3, 3)).copy()
    rot[:, :22] = frames; position[:, :22] = joints / body.scale + body.J[0]
    for j in range(1, 22):
        local[:, j] = frames[:, body.parents[j]].swapaxes(-1, -2) @ frames[:, j]
    local[:, 22:] = body.default
    for j in range(22, 55):
        parent = body.parents[j]; rot[:, j] = rot[:, parent] @ local[:, j]
        position[:, j] = position[:, parent] + np.einsum('nij,j->ni', rot[:, parent], body.J[j] - body.J[parent])
    translation = position - np.einsum('njab,jb->nja', rot, body.J)
    dtype = np.float64 if double_precision else np.float32
    feature = (local[:, 1:] - np.eye(3)).reshape(n, 486).astype(dtype)
    if vertex_ids is None:
        shaped, posedirs, weights = surface.shaped, surface.posedirs, surface.weights
    else:
        shaped = surface.shaped[vertex_ids]
        posedirs = surface.posedirs.reshape(486, -1, 3)[:, vertex_ids].reshape(486, -1)
        weights = surface.weights[vertex_ids]
    if double_precision:
        shaped = shaped.astype(np.float64); posedirs = posedirs.astype(np.float64); weights = weights.astype(np.float64)
    posed = shaped + (feature @ posedirs).reshape(n, -1, 3)
    r, t = rot.astype(dtype), translation.astype(dtype)
    blend_r = np.einsum('vj,njab->nvab', weights, r, optimize=True)
    blend_t = np.einsum('vj,nja->nva', weights, t, optimize=True)
    return (np.einsum('nvab,nvb->nva', blend_r, posed) + blend_t - body.J[0]) * body.scale


class CompactBodyOptimizedTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.model = synthetic_model(Path(self.temp.name) / 'synthetic.npz')
        self.rng = np.random.default_rng(301)
        self.pose = self.rng.normal(scale=.4, size=(9, 21, 3))

    def local(self, pose):
        n = len(pose)
        local = np.broadcast_to(np.eye(3), (n, 55, 3, 3)).copy()
        local[:, 1:22] = Rotation.from_rotvec(pose.reshape(-1, 3)).as_matrix().reshape(n, 21, 3, 3)
        local[:, 22:] = self.model.default
        return local

    def assert_outputs(self, expected, actual, atol=3e-13):
        self.assertEqual(len(expected), len(actual))
        for want, got in zip(expected, actual):
            self.assertEqual(got.dtype, np.float64)
            np.testing.assert_allclose(got, want, rtol=3e-13, atol=atol)

    def test_nonzero_pose_and_default_tail_match_original(self):
        local = self.local(self.pose)
        self.assert_outputs(original_forward(self.model, local),
                            self.model.forward_rotations(local[:, 1:22], with_frames=True))

    def test_explicit_tail_correction_keeps_general_lbs(self):
        delta = self.rng.normal(scale=.1, size=(len(self.pose), len(POSE_DOFS)))
        tail = self.rng.normal(scale=.2, size=(len(self.pose), 3))
        local = self.local(self.pose)
        for d, (j, axis) in enumerate(POSE_DOFS):
            value = np.zeros((len(self.pose), 3)); value[:, axis] = delta[:, d]
            local[:, j] = local[:, j] @ Rotation.from_rotvec(value).as_matrix()
        local[:, 45] = local[:, 45] @ Rotation.from_rotvec(tail).as_matrix()
        self.assert_outputs(original_forward(self.model, local),
            self.model.forward(self.pose, delta, additional_rotations={45: tail}, with_frames=True))

    def test_local_rotation_differences_are_preserved(self):
        direction = self.rng.normal(size=self.pose.shape)
        h = 1e-5
        expected = [original_forward(self.model, self.local(self.pose + sign * h * direction))
                    for sign in (1, -1)]
        actual = [self.model.forward_rotations(self.local(self.pose + sign * h * direction)[:, 1:22], with_frames=True)
                  for sign in (1, -1)]
        for old_plus, old_minus, new_plus, new_minus in zip(*expected, *actual):
            np.testing.assert_allclose((new_plus - new_minus) / (2 * h),
                                       (old_plus - old_minus) / (2 * h), rtol=2e-8, atol=2e-9)

    def test_one_model_is_reentrant_for_cpu_workers(self):
        rotations = [self.local(self.pose + index * .01)[:, 1:22] for index in range(4)]
        expected = [self.model.forward_rotations(r, with_frames=True) for r in rotations]
        with ThreadPoolExecutor(max_workers=4) as workers:
            actual = list(workers.map(lambda r: self.model.forward_rotations(r, with_frames=True), rotations))
        for want, got in zip(expected, actual):
            self.assert_outputs(want, got, atol=0)

    def surface(self):
        surface = FullSurface.__new__(FullSurface)
        surface.body = self.model
        surface.shaped = self.model.shaped.astype(np.float32)
        surface.posedirs = self.model.posedirs.astype(np.float32)
        surface.weights = self.model.W.astype(np.float32)
        surface._sample_double_cache = None
        return surface

    def test_surface_selected_cache_matches_original_and_keeps_index_order(self):
        surface = self.surface()
        _, joints, _, frames = original_forward(self.model, self.local(self.pose))
        for ids in (None, np.array([8, 1, 8, 3]), np.array([3, 8, 1, 8])):
            expected = original_surface(surface, joints, frames, ids, double_precision=True)
            actual = surface.sample_vertices(joints, frames, ids, double_precision=True)
            np.testing.assert_allclose(actual, expected, rtol=3e-13, atol=3e-13)
            cached = surface._sample_double_cache
            surface.sample_vertices(joints, frames, ids, double_precision=True)
            self.assertIs(surface._sample_double_cache, cached)
            for array in cached[1]:
                self.assertFalse(array.flags.writeable)

    def test_surface_display_path_is_bitwise_unchanged(self):
        surface = self.surface()
        _, joints, _, frames = original_forward(self.model, self.local(self.pose))
        for ids in (None, np.array([8, 1, 8, 3])):
            expected = original_surface(surface, joints, frames, ids)
            actual = surface.sample_vertices(joints, frames, ids)
            np.testing.assert_array_equal(actual, expected)

    def test_surface_concurrent_cache_misses_use_private_buffers(self):
        surface = self.surface()
        _, joints, _, frames = original_forward(self.model, self.local(self.pose))
        selections = [np.array([index, index + 1, index]) for index in range(4)]
        expected = [original_surface(surface, joints, frames, ids, double_precision=True) for ids in selections]
        with ThreadPoolExecutor(max_workers=4) as workers:
            actual = list(workers.map(lambda ids: surface.sample_vertices(joints, frames, ids, double_precision=True), selections))
        for want, got in zip(expected, actual):
            np.testing.assert_allclose(got, want, rtol=3e-13, atol=3e-13)

    def test_surface_cache_rebuilds_for_replaced_model_arrays_or_scale(self):
        surface = self.surface()
        _, joints, _, frames = original_forward(self.model, self.local(self.pose))
        ids = np.array([8, 1, 8, 3])
        surface.sample_vertices(joints, frames, ids, double_precision=True)
        first = surface._sample_double_cache
        surface.shaped = surface.shaped + np.float32(.001)
        actual = surface.sample_vertices(joints, frames, ids, double_precision=True)
        self.assertIsNot(surface._sample_double_cache, first)
        np.testing.assert_allclose(actual, original_surface(surface, joints, frames, ids, double_precision=True), rtol=3e-13, atol=3e-13)
        first = surface._sample_double_cache
        surface.body.scale *= 1.01
        actual = surface.sample_vertices(joints, frames, ids, double_precision=True)
        self.assertIsNot(surface._sample_double_cache, first)
        np.testing.assert_allclose(actual, original_surface(surface, joints, frames, ids, double_precision=True), rtol=3e-13, atol=3e-13)


if __name__ == '__main__':
    unittest.main()
