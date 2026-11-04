"""Real compatibility methods with a tiny synthetic surface; no licensed mesh.

AST loading avoids unrelated video/native preparation imports in this focused
test. Both sides execute the production foot/collar geometry. The reference
retains the 3d6f51f call order (two implicit surface selections).
"""
import ast
import copy
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from calibration.program_tracking.body_tracking.footwear_layout import OBSERVATION_LAYOUT
from calibration.program_tracking.body_tracking.skate_template import (
    SkateParameters, build_skate, profile_height, _sole_bottom, _sole_half_width)

ROOT = Path(__file__).resolve().parents[1]
BODY = ROOT/'calibration/program_tracking/body_tracking'


def binding_class():
    namespace = dict(np=np, copy=copy, asdict=asdict, replace=replace,
        FootwearBinding=object, OBSERVATION_LAYOUT=OBSERVATION_LAYOUT,
        SkateParameters=SkateParameters, build_skate=build_skate,
        profile_height=profile_height, _sole_bottom=_sole_bottom,
        _sole_half_width=_sole_half_width, SHAPE_NAMES=(
            'sole_length_m', 'sole_width_m', 'front_mount_height_m',
            'rear_mount_height_m', 'heel_block_height_m'),
        __package__='calibration.program_tracking.body_tracking')
    namespace['matrices'] = lambda v: Rotation.from_rotvec(
        np.asarray(v).reshape(-1, 3)).as_matrix().reshape(*np.asarray(v).shape[:-1], 3, 3)
    tree = ast.parse((BODY/'skate_binding.py').read_text(encoding='utf8'))
    nodes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'SkateBinding']
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(BODY/'skate_binding.py'), 'exec'), namespace)
    return namespace['SkateBinding']


def old_compatibility(binding, joints, frames, samples=None, x=None):
    """Unchanged numerical expression and implicit-sampling order at 3d6f51f."""
    m = binding.foot_metrics(joints, frames, samples, x)
    n = len(joints)
    anchor = (m['anchors']-m['targets'][None, None])/(.012/binding.ratio)
    outside = np.stack([np.max(v, axis=1) for v in m['violations']], 1)/(.003/binding.ratio)
    rows = np.concatenate((anchor.reshape(n, -1), outside.reshape(n, -1),
        m['support'].reshape(n, -1)/(.005/binding.ratio)), axis=1)
    if getattr(binding, 'collar_constraints_enabled', False):
        collar = binding.collar_metrics(joints, frames, samples, x)
        rows = np.c_[rows, collar.get('residual_breach_solver', collar['breach_solver'])/(.003/binding.ratio)]
    return rows


class SkinFixture:
    def __init__(self):
        self.calls = []
        # A closed rectangular shank surface for each foot plus an extra
        # non-collar foot vertex. Varying selections exercises cache reuse.
        lower = np.array([[.01, -.02, .0], [.05, -.02, .0],
                          [.05, .02, .0], [.01, .02, .0]])
        self.vertices = np.r_[lower, lower+[0., 0., .3],
                              lower+[0., .3, 0.], lower+[0., .3, .3],
                              [[.09, 0., .01], [.09, .3, .01]]]
        side = np.array([f for i in range(4) for f in
            ((i, (i+1) % 4, 4+i), ((i+1) % 4, 4+(i+1) % 4, 4+i))])
        self.faces = np.r_[side, side+8]
        self.weights = np.zeros((len(self.vertices), 55))
        self.weights[:8, 7] = 1.; self.weights[8:16, 8] = 1.
        self.weights[16, 7] = 1.; self.weights[17, 8] = 1.

    def sample_vertices(self, joints, frames, ids, *, double_precision=False):
        self.calls.append(np.asarray(ids).copy())
        self.assert_precision = double_precision
        out = np.broadcast_to(self.vertices[ids], (len(joints), len(ids), 3)).copy()
        # A pose dependence, with the same result for union and subset calls.
        out[..., 0] += .001*frames[:, 7, 0, 1, None]
        return out


def fixture(collar=True):
    cls = binding_class(); b = cls.__new__(cls)
    b.layout = OBSERVATION_LAYOUT; b.ratio = .91
    b.base_parameters = SkateParameters(); b.x = np.zeros(17)
    b.R0 = np.broadcast_to(np.eye(3), (2, 3, 3)).copy()
    b.t0 = np.zeros((2, 3)); b._parameter_key = None; b._packet = None
    b.surface = SkinFixture()
    # Preserve a deliberately permuted foot selection, then append collar
    # vertices exactly as enable_collar_constraints does in production.
    b.sample_ids = np.array([16, 1, 0, 17, 9, 8])
    b.foot_indices = [np.array([0, 1, 2]), np.array([3, 4, 5])]
    b.patch_indices = [[np.array([2]), np.array([0])], [np.array([5]), np.array([3])]]
    b.neutral_native = np.zeros((1, 22, 3)); b.neutral_native[:, 8, 1] = .3
    b.neutral_frames = np.broadcast_to(np.eye(3), (1, 22, 3, 3)).copy()
    b.collar_constraints_enabled = False
    b.faces, b.region_ids = np.empty((0, 3), int), np.empty(0, int)
    # Use the production upper topology used by the real binding.
    tree = ast.parse((BODY/'skate_binding.py').read_text(encoding='utf8'))
    ns = dict(np=np)
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'upper_topology']
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(BODY/'skate_binding.py'), 'exec'), ns)
    packet = b.packet(); count = len(packet['xs'])
    end = int(np.searchsorted(packet['xs'], .46*packet['parameters'].sole_length_m))
    b.faces, b.region_ids = ns['upper_topology'](count, end)
    if collar: b.enable_collar_constraints()
    b.surface.calls.clear()
    return b


class CompatibilitySampleTests(unittest.TestCase):
    def test_complete_foot_and_collar_rows_match_two_sample_reference(self):
        b = fixture(); n = 7
        joints = np.repeat(b.neutral_native, n, axis=0)
        frames = np.repeat(b.neutral_frames, n, axis=0)
        frames[:, 7] = Rotation.from_rotvec(np.c_[np.zeros(n), np.linspace(-.2, .3, n), np.zeros(n)]).as_matrix()
        before = b.sample_ids.copy()
        for shift in (0., .01):
            b.x[0] = shift
            expected = old_compatibility(b, joints, frames)
            self.assertEqual(len(b.surface.calls), 2)
            b.surface.calls.clear()
            actual = b.compatibility_rows(joints, frames)
            self.assertEqual(len(b.surface.calls), 1)
            np.testing.assert_array_equal(b.surface.calls[0], b.sample_ids)
            np.testing.assert_allclose(actual, expected, rtol=0., atol=2e-13)
            b.surface.calls.clear()
        np.testing.assert_array_equal(before, b.sample_ids)
        self.assertEqual(actual.shape, (n, 34))

    def test_explicit_samples_and_registration_override_remain_live(self):
        b = fixture(); joints, frames = b.neutral_native, b.neutral_frames
        samples = b.surface.sample_vertices(joints, frames, b.sample_ids, double_precision=True)
        b.surface.calls.clear(); original = samples.copy(); x = b.x.copy(); x[0] = .008
        expected = old_compatibility(b, joints, frames, samples, x)
        actual = b.compatibility_rows(joints, frames, samples, x)
        self.assertFalse(b.surface.calls)
        np.testing.assert_array_equal(samples, original)
        np.testing.assert_allclose(actual, expected, rtol=0., atol=2e-13)
        self.assertEqual(b.x[0], 0.)

    def test_collar_disabled_does_not_change_original_single_sample_path(self):
        b = fixture(False); joints, frames = b.neutral_native, b.neutral_frames
        expected = old_compatibility(b, joints, frames); b.surface.calls.clear()
        actual = b.compatibility_rows(joints, frames)
        self.assertEqual(len(b.surface.calls), 1)
        self.assertEqual(actual.shape, (1, 32))
        np.testing.assert_array_equal(actual, expected)


if __name__ == '__main__':
    unittest.main()
