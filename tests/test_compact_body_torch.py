"""Numerical checks for the optional Torch compact-body adapter.

No model weights are embedded.  The synthetic fixture deliberately includes
nonzero pose blend shapes, dense skinning weights, nonidentity default hand
rotations, a displaced root and a nonunit body scale.  Real CUDA checks are
explicitly skipped when this interpreter has no compatible CUDA runtime.
"""
import importlib.util
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
from scipy.spatial.transform import Rotation
from calibration.program_tracking.body_tracking.compact_body_torch import TorchCompactBody


def make_fixture(vertices=79):
    rng = np.random.default_rng(718)
    parents = np.r_[-1, [0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19],
                    [15, 15, 15], [20] * 15, [21] * 15].astype(np.int64)
    W = rng.uniform(size=(vertices, 55)); W /= W.sum(axis=1, keepdims=True)
    reg = rng.uniform(size=(vertices, 17)); reg /= reg.sum(axis=0, keepdims=True)
    return SimpleNamespace(parents=parents, J=rng.normal(size=(55, 3)) * .3,
                           shaped=rng.normal(size=(vertices, 3)) * .2,
                           posedirs=rng.normal(size=(486, 3 * vertices)) * .002,
                           W=W, reg=reg, default=Rotation.from_rotvec(rng.normal(size=(33, 3)) * .15).as_matrix(),
                           blade_relative=rng.normal(size=(2, 5, 3)) * .07, scale=1.7)


def reference_forward(source, local, with_frames=True):
    """Frozen 78dba30 equation order, independent of the adapter's depth groups."""
    n = len(local)
    rotation = local.copy()
    position = np.broadcast_to(source.J, (n, 55, 3)).copy()
    for j in range(1, 55):
        parent = source.parents[j]
        rotation[:, j] = rotation[:, parent] @ local[:, j]
        position[:, j] = position[:, parent] + np.einsum('nij,j->ni', rotation[:, parent], source.J[j] - source.J[parent])
    translation = position - np.einsum('njab,jb->nja', rotation, source.J)
    feature = (local[:, 1:] - np.eye(3)).reshape(n, 486)
    posed = source.shaped + (feature @ source.posedirs).reshape(n, -1, 3)
    blend_R = np.einsum('vj,njab->nvab', source.W, rotation, optimize=True)
    blend_t = np.einsum('vj,nja->nva', source.W, translation, optimize=True)
    vertices = np.einsum('nvab,nvb->nva', blend_R, posed) + blend_t
    coco = np.einsum('vj,nvc->njc', source.reg, vertices) - source.J[0]
    native = position[:, :22] - source.J[0]
    blade = source.scale * native[:, [7, 8], None] + np.einsum('nfab,fkb->nfka', rotation[:, [7, 8]], source.blade_relative)
    result = (coco * source.scale, native * source.scale, blade)
    return (*result, rotation[:, :22]) if with_frames else result


class AdapterAvailabilityTests(unittest.TestCase):
    def test_missing_torch_is_explicit(self):
        with mock.patch.dict('sys.modules', {'torch': None}):
            with self.assertRaisesRegex(RuntimeError, 'requires PyTorch'):
                TorchCompactBody(make_fixture(), device='cuda')

    def test_batch_size_rejected_before_loading_device(self):
        for value in (0, -2, 1.5, True):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'positive integer'):
                TorchCompactBody(make_fixture(), batch_size=value)

    def test_bad_weights_and_nonfinite_model_rejected(self):
        source = make_fixture(); source.W = source.W[:, :-1]
        with self.assertRaisesRegex(ValueError, 'tensor W'):
            TorchCompactBody._validate_model(source)
        source = make_fixture(); source.posedirs[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, 'tensor posedirs'):
            TorchCompactBody._validate_model(source)


@unittest.skipUnless(importlib.util.find_spec('torch') is not None, 'PyTorch not installed; CPU and CUDA numerical checks pending')
class TorchNumericalTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch
        self.source = make_fixture()
        self.rng = np.random.default_rng(1103)
        self.local = Rotation.from_rotvec(self.rng.normal(size=(7 * 55, 3)) * .4).as_matrix().reshape(7, 55, 3, 3)

    def assert_forward_matches(self, device):
        adapter = TorchCompactBody(self.source, device=device, batch_size=3)
        actual = adapter._forward_local_rotations(self.local, with_frames=True)
        expected = reference_forward(self.source, self.local)
        for got, want in zip(actual, expected):
            self.assertEqual(got.dtype, np.dtype('float64'))
            np.testing.assert_allclose(got, want, atol=3e-12, rtol=3e-12)
        # Exercise the exact interface used by FrozenReference.world().
        body = self.local[:, 1:22]
        standard = np.broadcast_to(np.eye(3), self.local.shape).copy()
        standard[:, 1:22] = body; standard[:, 22:] = self.source.default
        for got, want in zip(adapter.forward_rotations(body, with_frames=True), reference_forward(self.source, standard)):
            np.testing.assert_allclose(got, want, atol=3e-12, rtol=3e-12)
        self.assertIs(adapter.J, self.source.J)
        self.assertEqual(adapter.backend_info()['dtype'], 'float64')

    def test_cpu_matches_original_equations(self):
        self.assert_forward_matches('cpu')

    def test_cuda_matches_original_equations(self):
        if not self.torch.cuda.is_available():
            self.skipTest('CUDA unavailable; this is not a GPU validation result')
        self.assert_forward_matches('cuda')

    def test_chunking_does_not_change_outputs_or_local_difference(self):
        a = TorchCompactBody(self.source, device='cpu', batch_size=1)
        b = TorchCompactBody(self.source, device='cpu', batch_size=19)
        for v, w in zip(a._forward_local_rotations(self.local, with_frames=True), b._forward_local_rotations(self.local, with_frames=True)):
            np.testing.assert_allclose(v, w, atol=3e-12, rtol=3e-12)
        # Compare the quantity actually consumed by the existing sparse chain.
        h = 1e-5
        direction = np.array([h, 0., 0.])
        plus, minus = self.local.copy(), self.local.copy()
        plus[:, 7] = plus[:, 7] @ Rotation.from_rotvec(direction).as_matrix()
        minus[:, 7] = minus[:, 7] @ Rotation.from_rotvec(-direction).as_matrix()
        batch = np.concatenate((plus, minus))
        device_result = b._forward_local_rotations(batch, with_frames=True)
        rp, rm = reference_forward(self.source, plus), reference_forward(self.source, minus)
        for got, p, m in zip(device_result, rp, rm):
            numeric = (got[:7] - got[7:]) / (2 * h)
            np.testing.assert_allclose(numeric, (p - m) / (2 * h), atol=4e-10, rtol=2e-8)

    def test_empty_inputs_and_invalid_rotations(self):
        a = TorchCompactBody(self.source, device='cpu', batch_size=3)
        empty = a.forward_rotations(np.empty((0, 21, 3, 3)), with_frames=True)
        self.assertEqual([x.shape for x in empty], [(0, 17, 3), (0, 22, 3), (0, 2, 5, 3), (0, 22, 3, 3)])
        with self.assertRaisesRegex(ValueError, 'Expected'):
            a.forward_rotations(np.zeros((1, 22, 3, 3)))
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            a.forward_rotations(np.full((1, 21, 3, 3), np.nan))

    def test_no_cuda_fallback(self):
        with mock.patch.object(self.torch.cuda, 'is_available', return_value=False):
            with self.assertRaisesRegex(RuntimeError, 'CUDA is unavailable'):
                TorchCompactBody(self.source, device='cuda')


if __name__ == '__main__':
    unittest.main()
