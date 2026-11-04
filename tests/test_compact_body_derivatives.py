"""Independent local derivatives and production world replay, without models.

Synthetic tensors have dense pose blend shapes and weights on all 55 joints.
World/binding methods below are AST-loaded unchanged from their production
files so these checks do not require video/inference preparation dependencies.
No synthetic result is reported as a real fitted-sequence benchmark.
"""
import ast
import copy
from dataclasses import asdict, replace
from functools import wraps
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from calibration.program_tracking.body_tracking.body_execution import BodyExecution, BodyExecutionConfig
from calibration.program_tracking.body_tracking.compact_body_derivatives import (
    CompactBodyLinearization, _matrices, supports_reference, resolve_reference)
from calibration.program_tracking.body_tracking.footwear_geometry import (
    INITIAL, equipment_world, local_vertices, boot_alignment, decode)
from calibration.program_tracking.body_tracking.footwear_layout import OBSERVATION_LAYOUT
from calibration.program_tracking.body_tracking.skate_template import (
    SkateParameters, build_skate, profile_height, _sole_bottom, _sole_half_width)
from rinkspace.viewer3d.prepare import FullSurface
from test_compact_body_performance import synthetic_model


BODY = Path(__file__).resolve().parents[1]/'calibration/program_tracking/body_tracking'
PACKAGE = 'calibration.program_tracking.body_tracking.'


def _class(path, name, namespace, methods=None):
    tree = ast.parse((BODY/path).read_text(encoding='utf8'))
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    if methods is not None:
        node.body = [n for n in node.body if isinstance(n, ast.FunctionDef) and n.name in methods]
    namespace = dict(namespace, __name__=PACKAGE+path.removesuffix('.py'),
                     __package__=PACKAGE.removesuffix('.'))
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(BODY/path), 'exec'), namespace)
    return namespace[name]


def world_types():
    shared = dict(np=np, matrices=_matrices, equipment_world=equipment_world)
    frozen = _class('fullbody_state.py', 'FrozenReference', shared, ['world'])
    shared['FrozenReference'] = frozen
    integrated = _class('footwear_binding.py', 'IntegratedBodyState', shared, ['world'])
    skate = _class('skate_integration.py', 'SkateReference', shared, ['world'])
    opening = _class('opening_repair.py', 'OpeningReference', shared, ['world', '__getattr__'])
    return frozen, integrated, skate, opening


def skate_binding():
    namespace = dict(np=np, copy=copy, asdict=asdict, replace=replace,
        FootwearBinding=object, OBSERVATION_LAYOUT=OBSERVATION_LAYOUT,
        SkateParameters=SkateParameters, build_skate=build_skate,
        profile_height=profile_height, _sole_bottom=_sole_bottom,
        _sole_half_width=_sole_half_width, matrices=_matrices,
        SHAPE_NAMES=('sole_length_m', 'sole_width_m', 'front_mount_height_m',
                     'rear_mount_height_m', 'heel_block_height_m'))
    namespace['transform'] = lambda R, t, p: t[..., None, :]+np.einsum('...ij,...kj->...ki', R, p)
    cls = _class('skate_binding.py', 'SkateBinding', namespace)
    value = cls.__new__(cls)
    value.layout = OBSERVATION_LAYOUT; value.ratio = .91
    value.base_parameters = SkateParameters()
    value.x = np.r_[np.array([.01, -.02, .03, .08, -.06, .13,
                             -.02, .01, .02, -.05, .09, -.04]),
                    np.log([1.04, .95, 1.08, .94, 1.06])]
    value.R0 = np.broadcast_to(np.array([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]]), (2, 3, 3)).copy()
    value.t0 = np.array([[.01, -.03, .02], [-.02, -.04, .01]])
    value._parameter_key = value._packet = None
    return value


def legacy_binding(geometry):
    namespace = dict(np=np, matrices=_matrices, local_vertices=local_vertices,
                     boot_alignment=boot_alignment, decode=decode)
    cls = _class('footwear_binding.py', 'FootwearBinding', namespace,
                 ['parameters', 'evaluate'])
    value = cls.__new__(cls)
    p = decode(geometry)
    value.c0 = np.array([0., -p['ankle_to_sole'], (p['shoe_front']-p['shoe_rear'])/2])
    value.template = local_vertices(geometry)[0]-value.c0
    value.R0 = np.stack([boot_alignment(geometry, f) for f in range(2)])
    value.t0 = np.einsum('fij,j->fi', value.R0, value.c0)
    value.R_BL = _matrices(np.array([p['blade_pitch'], 0., 0.]))
    value.t_BL = np.mean(value.template[14:], axis=0)
    value.x = np.r_[np.tile([.01, -.005, .003, .03, -.04, .05], 2), [.04, -.03]]
    value.deformation_enabled = True; value.shank_radius = .026
    value.cuff_clearance = .004
    value.mode_regularizer = np.array([.35, .5, .8]); value.maximum_modes = np.array([.07, .04, .025])
    return value


class AnalyticBodyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.body = synthetic_model(Path(self.temp.name)/'model.npz')
        self.rng = np.random.default_rng(61104)
        n = 5
        self.ref = dict(P=self.rng.normal(scale=10, size=(n, 3)),
            Q=_matrices(self.rng.normal(size=(n, 3))),
            local_R=_matrices(self.rng.normal(size=(n, 21, 3))))
        self.delta = self.rng.normal(scale=.4, size=(n, 69))
        self.delta[0, 6:] = 0.
        self.delta[1, 6:] *= 1e-10
        self.delta[2, 6:] *= 3.
        self.frozen, self.integrated, self.skate, self.proxy = world_types()

    def linearization(self, **kwargs):
        return CompactBodyLinearization(self.body, self.ref['local_R'],
            self.delta[:, 6:].reshape(-1, 21, 3), **kwargs)

    def source(self, kind='frozen'):
        cls = dict(frozen=self.frozen, integrated=self.integrated, skate=self.skate)[kind]
        obj = cls.__new__(cls); obj.body = self.body; obj.geometry = INITIAL.copy()
        if kind == 'integrated': obj.binding = legacy_binding(obj.geometry)
        if kind == 'skate': obj.binding = skate_binding(); obj.layout = OBSERVATION_LAYOUT
        return obj

    def finite_local(self, column, h=2e-6):
        values = []
        for sign in (1., -1.):
            delta = self.delta.copy(); delta[:, column] += sign*h
            local = self.ref['local_R'] @ _matrices(delta[:, 6:].reshape(-1, 21, 3))
            coco, native, _, frames = self.body.forward_rotations(local, with_frames=True)
            values.append(dict(coco=coco, native=native, frames=frames, local_R=local))
        return {key:(values[0][key]-values[1][key])/(2*h) for key in values[0]}, values

    def test_all_63_chart_derivatives_and_all_17_regressed_points(self):
        linearization = self.linearization(chunk_size=2)
        for column in range(6, 69):
            want, _ = self.finite_local(column)
            actual = linearization.derivative(column)
            with self.subTest(column=column):
                for key in want:
                    np.testing.assert_allclose(actual[key], want[key], rtol=2e-7, atol=6e-10)
        expected = self.body.forward_rotations(linearization.local_R, with_frames=True)
        for actual, want in zip((linearization.coco, linearization.native, linearization.frames),
                                (expected[0], expected[1], expected[3])):
            np.testing.assert_allclose(actual, want, rtol=2e-14, atol=3e-14)

    def test_fixed_33_joint_tail_is_live_in_skinning_derivative(self):
        actual = self.linearization().derivative(6)['coco']
        saved = self.body.default.copy()
        self.body.default = np.broadcast_to(np.eye(3), saved.shape).copy()
        try: without_tail = self.linearization().derivative(6)['coco']
        finally: self.body.default = saved
        self.assertGreater(np.linalg.norm(actual-without_tail), 1e-5)
        expected, _ = self.finite_local(6)
        np.testing.assert_allclose(actual, expected['coco'], atol=5e-10, rtol=2e-7)

    def test_pose_blend_dependency_survives_without_joint_skinning_weight(self):
        # No vertex is skinned to the perturbed limb: the only derivative is
        # its pose blend shape.  A rigid-joint-only shortcut fails this check.
        self.body.W[:] = 0.; self.body.W[:, 0] = 1.
        self.body._weights_T = np.ascontiguousarray(self.body.W.T)
        lin = self.linearization()
        got = lin.derivative(68)['coco']; expected, _ = self.finite_local(68)
        self.assertGreater(np.linalg.norm(got), 1e-5)
        np.testing.assert_allclose(got, expected['coco'], atol=5e-10, rtol=2e-7)

    def test_chunking_and_repeated_reordered_columns(self):
        cols = [68, 6, 17, 68, 49, 7, 6]
        first = self.linearization(chunk_size=1); second = self.linearization(chunk_size=30)
        for column in cols:
            for key, want in first.derivative(column).items():
                np.testing.assert_allclose(second.derivative(column)[key], want, rtol=3e-14, atol=3e-14)

    def test_exact_kinematic_pairs_match_original_forward_at_every_column(self):
        lin = self.linearization(include_vertices=False)
        self.assertIsNone(lin.coco); self.assertFalse(hasattr(lin, '_posed'))
        for column in range(6, 69):
            _, expected = self.finite_local(column)
            for got, want in zip(lin.kinematic_pair(column, 2e-6), expected):
                for key in ('native', 'frames', 'local_R'):
                    np.testing.assert_allclose(got[key], want[key], rtol=4e-14, atol=4e-14)

    def test_world_path_matches_old_fd_with_nonzero_root_and_every_binding(self):
        columns = list(range(6, 69))+[68, 6]
        for kind in ('frozen', 'integrated', 'skate'):
            reference = self.source(kind)
            self.assertTrue(supports_reference(reference))
            before = reference.binding.x.copy() if kind != 'frozen' else None
            old = BodyExecution(BodyExecutionConfig(local_derivatives='finite_difference'))
            new = BodyExecution(BodyExecutionConfig(workers=3))
            try:
                expected = list(old.world_derivatives(reference, self.ref, self.delta, columns))
                actual = list(new.world_derivatives(reference, self.ref, self.delta, columns))
            finally: old.close(); new.close()
            with self.subTest(kind=kind):
                self.assertEqual([c for c, _ in actual], columns)
                for (_, got), (_, want) in zip(actual, expected):
                    np.testing.assert_allclose(got, want, rtol=2e-6, atol=2e-8)
                self.assertEqual(new.record()['derivative_usage']['analytic_world_linearizations'], 1)
                self.assertEqual(new.record()['derivative_usage']['finite_difference_world_linearizations'], 0)
                if before is not None: np.testing.assert_array_equal(reference.binding.x, before)

    def test_native_replay_keeps_all_surface_pose_blend_columns(self):
        reference = self.source('skate')
        surface = FullSurface.__new__(FullSurface); surface.body = self.body
        surface.shaped = self.body.shaped.astype(np.float32)
        surface.posedirs = self.body.posedirs.astype(np.float32)
        surface.weights = self.body.W.astype(np.float32); surface._sample_double_cache = None
        ids = np.array([8, 1, 19, 8, 25])
        def consume(native, frames, samples=None):
            return (surface.sample_vertices(native, frames, ids, double_precision=True)
                    if samples is None else samples)
        def consume_world(state):
            native = np.einsum('nji,nkj->nki', state['Q'], state['native']-state['P'][:, None])
            return consume(native, state['frames'])
        execution = BodyExecution(BodyExecutionConfig(workers=3)); self.addCleanup(execution.close)
        columns = list(range(6, 69))
        expected = execution.state_differences(reference, self.ref, self.delta, columns, consume_world)
        actual = execution.native_state_differences(reference, self.ref, self.delta, columns, consume)
        self.assertIsNotNone(actual)
        for got, want in zip(actual, expected):
            np.testing.assert_allclose(got, want, rtol=2e-6, atol=5e-9)
        self.assertGreater(np.linalg.norm(actual[-1]), 1e-5)
        self.assertEqual(execution.record()['derivative_usage']['fk_replay_linearizations'], 1)
        fast_surface = execution.native_state_differences(reference, self.ref, self.delta, columns, consume,
            surface=surface, vertex_ids=ids)
        for got, want in zip(fast_surface, expected):
            np.testing.assert_allclose(got, want, rtol=2e-6, atol=5e-9)
        self.assertEqual(execution.record()['derivative_usage']['surface_replay_linearizations'], 1)
        self.assertEqual(execution.record()['last_surface_replay']['selected_vertices'], len(ids))

    def test_custom_surface_and_different_model_keep_original_sampler(self):
        reference = self.source('skate'); execution = BodyExecution()
        class CustomSurface:
            body = self.body
            def sample_vertices(self, native, frames, ids, double_precision=False):
                return native[:, ids]*1.1
        surface = CustomSurface(); seen = []
        def consumer(native, frames, samples=None):
            seen.append(samples)
            return surface.sample_vertices(native, frames, [2, 4], double_precision=True)
        value = execution.native_state_differences(reference, self.ref, self.delta, [6, 68], consumer,
            surface=surface, vertex_ids=np.array([2, 4]))
        self.assertEqual(len(value), 2); self.assertTrue(all(item is None for item in seen))
        self.assertEqual(execution.record()['derivative_usage']['surface_replay_linearizations'], 0)
        other = copy.copy(self.body); other.J = other.J+.01
        surface = FullSurface.__new__(FullSurface); surface.body = other
        # Do not access missing surface tensors when its skeleton is different.
        seen.clear()
        execution.native_state_differences(reference, self.ref, self.delta, [6],
            lambda n, f, samples=None: seen.append(samples) or n[:, [2, 4]],
            surface=surface, vertex_ids=np.array([2, 4]))
        self.assertEqual(seen, [None, None])
        self.assertEqual(execution.record()['derivative_usage']['surface_replay_linearizations'], 0)

    def test_native_replay_calls_no_compact_forward(self):
        reference = self.source('skate'); execution = BodyExecution()
        original = self.body.forward_rotations
        self.body.forward_rotations = lambda *a, **k: (_ for _ in ()).throw(AssertionError('Unneeded compact LBS'))
        try:
            value = execution.native_state_differences(reference, self.ref, self.delta, [6, 68], lambda j, f:j)
            self.assertEqual(len(value), 2)
        finally: self.body.forward_rotations = original

    def test_proxy_support_and_custom_world_fallback_are_explicit(self):
        reference = self.source('skate')
        proxy = self.proxy.__new__(self.proxy); proxy.parent = reference
        self.assertIs(resolve_reference(proxy), reference)
        class Custom(self.skate):
            def world(self, ref, delta=None):
                result = super().world(ref, delta)
                result['world'] = result['world'] * 1.01
                return result
        custom = Custom.__new__(Custom); custom.__dict__.update(reference.__dict__)
        self.assertFalse(supports_reference(custom))
        execution = BodyExecution()
        self.assertIsNone(execution.native_state_differences(custom, self.ref, self.delta, [6], lambda j, f:j))
        actual = list(execution.world_derivatives(custom, self.ref, self.delta, [6]))
        expected = execution.state_differences(custom, self.ref, self.delta, [6], lambda s:s['world'])
        np.testing.assert_array_equal(actual[0][1], expected[0])
        self.assertEqual(execution.record()['derivative_usage']['finite_difference_world_linearizations'], 1)

    def test_wrapped_world_and_binding_keep_their_custom_numerical_semantics(self):
        for target in ('world', 'binding'):
            with self.subTest(target=target):
                reference = self.source('skate')
                execution = BodyExecution()
                unmodified = execution.state_differences(reference, self.ref, self.delta, [6], lambda s:s['world'])[0]
                owner = reference if target == 'world' else reference.binding
                name = 'world' if target == 'world' else 'evaluate'
                field = 'world' if target == 'world' else 'gear'
                original = getattr(owner, name)
                @wraps(original)
                def changed(*args, **kwargs):
                    value = original(*args, **kwargs)
                    value[field] = value[field] * 1.01
                    return value
                setattr(owner, name, changed)
                self.assertFalse(supports_reference(reference))
                self.assertIsNone(execution.native_state_differences(reference, self.ref, self.delta, [6], lambda j, f:j))
                actual = list(execution.world_derivatives(reference, self.ref, self.delta, [6]))[0][1]
                expected = execution.state_differences(reference, self.ref, self.delta, [6], lambda s:s['world'])[0]
                np.testing.assert_array_equal(actual, expected)
                self.assertGreater(np.linalg.norm(actual-unmodified), 1e-8)
                counts = execution.record()['derivative_usage']
                self.assertEqual(counts['finite_difference_world_linearizations'], 1)
                self.assertEqual(counts['analytic_world_linearizations'], 0)

    def test_explicit_old_mode_and_cuda_do_not_enter_cpu_analytic_replay(self):
        reference = self.source('skate')
        for config in (BodyExecutionConfig(local_derivatives='finite_difference'),
                       BodyExecutionConfig(backend='cuda')):
            execution = BodyExecution(config)
            self.assertIsNone(execution.native_state_differences(reference, self.ref, self.delta, [6], lambda j, f:j))

    def test_invalid_inputs_and_empty_batch(self):
        for chunk in (0, -1, True):
            with self.assertRaises(ValueError): self.linearization(chunk_size=chunk)
        lin = self.linearization()
        for column in (5, 69, 6.2, True):
            with self.assertRaises(ValueError): lin.derivative(column)
        with self.assertRaises(ValueError): lin.kinematic_pair(6, 0)
        with self.assertRaises(ValueError): BodyExecutionConfig(local_derivatives='fast')
        empty = CompactBodyLinearization(self.body, np.empty((0, 21, 3, 3)), np.empty((0, 21, 3)))
        self.assertEqual(empty.derivative(6)['coco'].shape, (0, 17, 3))


if __name__ == '__main__':
    unittest.main()
