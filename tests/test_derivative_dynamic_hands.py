"""CPU body derivatives and exact selected-skin replay with sampled hands.

Synthetic dense model tensors exercise all pose directions and 55 weights;
the standard SMPL-X parent tree places fingers below each actual wrist.
These mathematical checks do not establish fitted-sequence performance.
"""
from pathlib import Path
import tempfile
import unittest

import numpy as np

from calibration.program_tracking.body_tracking.compact_body_derivatives import (
    CompactBodyLinearization, _matrices)
from calibration.program_tracking.body_tracking.selected_surface_linearization import (
    SelectedSurfaceLinearization, supports_surface)
from rinkspace.viewer3d.prepare import FullSurface
from test_compact_body_performance import synthetic_model, original_surface


class DynamicHandDerivativeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.body = synthetic_model(Path(self.temp.name) / 'dynamic_hand_model.npz')
        self.body.parents = np.array([
            -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19,
            15, 15, 15, 20, 25, 26, 20, 28, 29, 20, 31, 32, 20, 34, 35, 20, 37, 38,
            21, 40, 41, 21, 43, 44, 21, 46, 47, 21, 49, 50, 21, 52, 53])
        self.body._joint_offsets = self.body.J - self.body.J[self.body.parents]
        rng = np.random.default_rng(91104)
        self.reference = _matrices(rng.normal(0., .8, (5, 21, 3)))
        self.delta = rng.normal(0., .35, (5, 21, 3))
        self.delta[0] *= 1e-10
        self.hands = _matrices(rng.normal(.15, .65, (5, 2, 15, 3)))
        self.surface = FullSurface.__new__(FullSurface); self.surface.body = self.body
        self.surface.shaped = self.body.shaped.astype(np.float32)
        self.surface.posedirs = self.body.posedirs.astype(np.float32)
        self.surface.weights = self.body.W.astype(np.float32)
        self.surface._sample_double_cache = None
        self.ids = np.array([8, 1, 8, 3, 30, 0, 24, 5])

    def linearization(self, **kwargs):
        return CompactBodyLinearization(self.body, self.reference, self.delta,
            hand_local_R=self.hands, **kwargs)

    def forward(self, delta, hands=True):
        local = self.reference @ _matrices(delta)
        kwargs = dict(hand_local_R=self.hands) if hands else {}
        coco, native, _, frames = self.body.forward_rotations(local, with_frames=True, **kwargs)
        return dict(coco=coco, native=native, frames=frames, local_R=local)

    def full_fk(self, body_local):
        local = np.broadcast_to(np.eye(3), (len(body_local), 55, 3, 3)).copy()
        local[:, 1:22] = body_local; local[:, 22:] = self.body.default
        local[:, 25:55] = self.hands.reshape(len(body_local), 30, 3, 3)
        rotation = local.copy(); position = np.broadcast_to(self.body.J, (len(local), 55, 3)).copy()
        for j in range(1, 55):
            parent = self.body.parents[j]
            rotation[:, j] = rotation[:, parent] @ local[:, j]
            position[:, j] = position[:, parent] + np.einsum(
                'nij,j->ni', rotation[:, parent], self.body.J[j] - self.body.J[parent])
        return rotation, position

    def surface_context(self, lin):
        return SelectedSurfaceLinearization(self.surface, lin.native, lin.frames, self.ids,
            hand_local_R=self.hands, chunk_size=2)

    def test_all_63_body_columns_with_actual_nondefault_varying_hands(self):
        lin = self.linearization(chunk_size=2); expected = self.forward(self.delta)
        self.assertGreater(np.linalg.norm(self.hands[0] - self.hands[-1]), 1.)
        self.assertGreater(np.linalg.norm(lin.coco - self.forward(self.delta, hands=False)['coco']), 1e-4)
        for key in ('coco', 'native', 'frames', 'local_R'):
            np.testing.assert_allclose(getattr(lin, key), expected[key], atol=4e-14, rtol=4e-14)
        rotation, position = self.full_fk(lin.local_R)
        np.testing.assert_allclose(lin._rotation55, rotation, atol=4e-14, rtol=4e-14)
        np.testing.assert_allclose(lin._position55, position, atol=4e-14, rtol=4e-14)
        for column in range(6, 69):
            offset = (column - 6) // 3; axis = (column - 6) % 3; h = 2e-6
            changed = [self.delta.copy(), self.delta.copy()]
            changed[0][:, offset, axis] += h; changed[1][:, offset, axis] -= h
            plus, minus = (self.forward(d) for d in changed)
            got = lin.derivative(column)
            with self.subTest(column=column):
                for key in ('coco', 'native', 'frames', 'local_R'):
                    want = (plus[key] - minus[key]) / (2*h)
                    np.testing.assert_allclose(got[key], want, atol=7e-10, rtol=3e-7)

    def test_exact_pair_replays_entire_finger_subtree_without_changing_hands(self):
        lin = self.linearization(include_vertices=False)
        for column in range(6, 69):
            for state in lin.kinematic_pair(column, .13):
                rotation, position = self.full_fk(state['local_R'])
                with self.subTest(column=column):
                    np.testing.assert_allclose(state['rotation55'], rotation, atol=5e-14, rtol=5e-14)
                    np.testing.assert_allclose(state['position55'], position, atol=5e-14, rtol=5e-14)
                    np.testing.assert_array_equal(state['hand_local_R'], self.hands)
        pair = lin.kinematic_pair(64, .13)  # left wrist: joints 25..39 follow it.
        self.assertGreater(np.linalg.norm(pair[0]['rotation55'][:, 25:40] - pair[1]['rotation55'][:, 25:40]), .1)
        np.testing.assert_array_equal(pair[0]['rotation55'][:, 40:], pair[1]['rotation55'][:, 40:])

    def test_selected_skin_uses_full486_cache_and_exact_body_pairs(self):
        lin = self.linearization(include_vertices=False)
        # A previous default-hand evaluation must not poison sampled hands.
        self.surface.sample_vertices(lin.native, lin.frames, self.ids, double_precision=True)
        cache = self.surface._sample_double_cache
        context = self.surface_context(lin)
        self.assertTrue(supports_surface(self.surface))
        self.assertIs(self.surface._sample_double_cache, cache)
        self.assertIs(context._posedirs, cache[1][1]); self.assertEqual(context._posedirs.shape[0], 486)
        baseline = self.surface.sample_vertices(lin.native, lin.frames, self.ids,
            hand_local_R=self.hands, double_precision=True)
        np.testing.assert_allclose(context.samples, baseline, atol=5e-14, rtol=5e-14)
        for step in (1e-5, .13):
            # Legs, both shoulders, both wrists and independent axes.
            for column in (6, 10, 18, 28, 51, 55, 63, 64, 66, 68):
                states = lin.kinematic_pair(column, step)
                actual = context.pair(column, *states)
                expected = [self.surface.sample_vertices(s['native'], s['frames'], self.ids,
                    hand_local_R=self.hands, double_precision=True) for s in states]
                with self.subTest(step=step, column=column):
                    for got, want in zip(actual, expected):
                        np.testing.assert_allclose(got, want, atol=5e-13, rtol=5e-13)
                    np.testing.assert_allclose((actual[0]-actual[1])/(2*step),
                        (expected[0]-expected[1])/(2*step), atol=5e-9, rtol=3e-7)

    def test_pose_blend_only_body_column_is_retained_with_dynamic_hands(self):
        self.surface.weights[:] = 0.; self.surface.weights[:, 0] = 1.
        lin = self.linearization(include_vertices=False); context = self.surface_context(lin)
        states = lin.kinematic_pair(68, .13)
        actual = context.pair(68, *states)
        self.assertEqual(len(context._skin_indices[21]), 0)
        self.assertGreater(np.linalg.norm(actual[0]-actual[1]), 1e-5)
        for got, state in zip(actual, states):
            want = self.surface.sample_vertices(state['native'], state['frames'], self.ids,
                hand_local_R=self.hands, double_precision=True)
            np.testing.assert_allclose(got, want, atol=5e-13, rtol=5e-13)

    def test_none_preserves_original_fixed_tail_geometry(self):
        omitted = CompactBodyLinearization(self.body, self.reference, self.delta)
        explicit = CompactBodyLinearization(self.body, self.reference, self.delta, hand_local_R=None)
        old = self.forward(self.delta, hands=False)
        for key in ('coco', 'native', 'frames', 'local_R'):
            np.testing.assert_array_equal(getattr(omitted, key), getattr(explicit, key))
            np.testing.assert_allclose(getattr(explicit, key), old[key], atol=4e-14, rtol=4e-14)
        context = SelectedSurfaceLinearization(self.surface, omitted.native, omitted.frames, self.ids)
        none = SelectedSurfaceLinearization(self.surface, omitted.native, omitted.frames, self.ids, hand_local_R=None)
        np.testing.assert_array_equal(context.samples, none.samples)
        want = original_surface(self.surface, omitted.native, omitted.frames, self.ids, double_precision=True)
        np.testing.assert_allclose(context.samples, want, atol=5e-14, rtol=5e-14)

    def test_hand_inputs_are_copied_and_invalid_rotations_rejected(self):
        lin = self.linearization(); context = self.surface_context(lin)
        before = lin.hand_local_R.copy(); self.hands[:] = np.eye(3)
        np.testing.assert_array_equal(lin.hand_local_R, before)
        np.testing.assert_array_equal(context.hand_local_R, before)
        for hands in (np.empty((4, 2, 15, 3, 3)), np.zeros((5, 2, 15, 3, 3)),
                      np.full((5, 2, 15, 3, 3), np.nan)):
            with self.assertRaises(ValueError):
                CompactBodyLinearization(self.body, self.reference, self.delta, hand_local_R=hands)
            with self.assertRaises(ValueError):
                SelectedSurfaceLinearization(self.surface, lin.native, lin.frames, self.ids, hand_local_R=hands)


if __name__ == '__main__':
    unittest.main()
