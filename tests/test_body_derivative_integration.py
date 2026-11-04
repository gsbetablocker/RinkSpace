"""The real BootWindow evaluate path uses the new derivatives and stays equal.

This bounded harness uses the production residual/CSR/exposure/boot methods,
a synthetic full SMPL-X tensor fixture and a pinhole camera. It proves wiring,
all 63 pose dependencies, cache upgrade and fallback, not real scene accuracy.
"""
import ast
import json
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, vstack

from calibration.program_tracking.body_tracking.body_execution import BodyExecution, BodyExecutionConfig
from calibration.program_tracking.body_tracking.compact_body_derivatives import _matrices
from calibration.program_tracking.body_tracking.footwear_layout import OBSERVATION_LAYOUT
from rinkspace.viewer3d.prepare import FullSurface
from test_compact_body_derivatives import BODY, PACKAGE, _class, world_types, legacy_binding
from test_compact_body_performance import synthetic_model


def _definitions(filename, names, namespace):
    tree = ast.parse((BODY/filename).read_text(encoding='utf8'))
    nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    if len(nodes) != len(names): raise AssertionError('Missing production definition')
    namespace['__name__'] = PACKAGE+filename.removesuffix('.py')
    namespace['__package__'] = PACKAGE.removesuffix('.')
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(BODY/filename), 'exec'), namespace)


def production_window():
    namespace = dict(np=np, time=time, json=json, Path=Path, WindowState=object,
                     coo_matrix=coo_matrix, csr_matrix=csr_matrix, vstack=vstack)
    _definitions('rebuild_exposure_body.py', {'row_weights', 'moving_projection'}, namespace)
    _definitions('fullbody_fit.py', {'skew', 'right_jacobian', 'FullBodyProblem'}, namespace)
    _definitions('boot_integration.py', {'BootWindowProblem'}, namespace)
    return namespace['BootWindowProblem']


class Pinhole:
    def project(self, world, frames):
        return np.stack((960.+400.*world[..., 0]/world[..., 2],
                         540.+400.*world[..., 1]/world[..., 2]), -1)

    def with_jacobian(self, world, frames):
        value = self.project(world, frames); J = np.zeros((*value.shape, 3))
        z = world[..., 2]; J[..., 0, 0] = 400./z; J[..., 1, 1] = 400./z
        J[..., 0, 2] = -400.*world[..., 0]/z**2
        J[..., 1, 2] = -400.*world[..., 1]/z**2
        return value, J


class Basis:
    active = np.arange(69, dtype=int)
    m = 2; d = 69; size = 138; mode = 'full'; seconds = (0., .04)

    def basis(self, t):
        t = np.asarray(t)
        return np.c_[1.-t, t]

    def increments(self, x, t):
        return self.basis(t) @ np.asarray(x).reshape(self.m, self.d)


def make_window(model, mode='analytic'):
    _, integrated, _, _ = world_types()
    reference = integrated.__new__(integrated); reference.body = model
    from calibration.program_tracking.body_tracking.footwear_geometry import INITIAL
    reference.geometry = INITIAL.copy(); reference.binding = legacy_binding(reference.geometry)
    binding = reference.binding
    # Use all real compatibility/cavity code; no fake residual consumer.
    namespace = dict(binding.evaluate.__globals__)
    full_binding = _class('footwear_binding.py', 'FootwearBinding', namespace)
    binding.__class__ = full_binding
    binding.lining = .003; binding.heel_bed = .022; binding.ball_bed = .007
    binding.surface = FullSurface.__new__(FullSurface); binding.surface.body = model
    binding.surface.shaped = model.shaped.astype(np.float32)
    binding.surface.posedirs = model.posedirs.astype(np.float32)
    binding.surface.weights = model.W.astype(np.float32); binding.surface._sample_double_cache = None
    binding.sample_ids = np.array([2, 4, 6, 9, 12, 15])
    binding.foot_indices = [np.arange(3), np.arange(3, 6)]
    binding.patch_indices = [[np.array([0]), np.array([2])], [np.array([3]), np.array([5])]]
    cls = production_window(); w = cls.__new__(cls)
    w.reference = reference; w.binding = binding; w.state = Basis()
    w.size = w.state.size; w.n = 3; w.frames = np.arange(w.n)
    w.projector = Pinhole(); w.layout = OBSERVATION_LAYOUT
    w.times = (np.arange(w.n)[:, None]*.02 + np.linspace(-.001, .001, 5)).ravel()
    w.mid_times = w.times.reshape(w.n, 5)[:, 2]

    def at(times):
        times = np.asarray(times); n = len(times)
        local = np.zeros((n, 21, 3)); local[..., 0] = .05+np.arange(21)*.011
        local[..., 1] = times[:, None]*.2; local[..., 2] = -.02
        return dict(P=np.tile([.1, -.2, 4.], (n, 1)),
            Q=_matrices(np.c_[.1+times*.05, -.2+times*.03, .08-times*.01]),
            local_R=_matrices(local), times=times)

    w.ref = at(w.times); w.mid_ref = at(w.mid_times)
    w.B = w.state.basis(w.times).reshape(w.n, 5, w.state.m)
    w.eval_basis = w.B.reshape(-1, w.state.m)
    w.execution = BodyExecution(BodyExecutionConfig(local_derivatives=mode))
    count = w.layout.point_count
    w.fi = np.repeat(np.arange(w.n), count); w.pi = np.tile(np.arange(count), w.n)
    w.uv_rows = (w.fi*count+w.pi)[:, None]*2+np.arange(2)
    w.spline_scatter = []
    for node in range(5):
        obs, knot = np.nonzero(w.B[w.fi, node]); shape = (len(obs), 2, w.state.d)
        rr = np.broadcast_to(w.uv_rows[obs, :, None], shape).ravel().copy()
        cc = np.broadcast_to(knot[:, None, None]*w.state.d+np.arange(w.state.d), shape).ravel().copy()
        weight = w.B[w.fi[obs], node, knot, None, None]
        w.spline_scatter.append((obs, rr, cc, weight))
    w.spline_rows = np.concatenate([v[1] for v in w.spline_scatter])
    w.spline_cols = np.concatenate([v[2] for v in w.spline_scatter])
    w.of, w.oj = np.nonzero(np.ones((w.n, 17), bool))
    w.obs = dict(sigma_px=np.full((w.n, 17), 24.), uv=np.zeros((w.n, 17, 2)))
    w.prior_matrices = [('fixture_prior', csr_matrix(np.eye(w.size)/.7))]
    w.cache = w.cache_x = w.binding_key = None
    w.start = time.perf_counter(); w.evaluations = 0; w.log = None
    value = reference.world(w.ref, np.zeros((len(w.times), 69)))
    uv = w.projector.project(value['world'].reshape(w.n, 5, count, 3)[:, 2], w.frames)
    w.obs['uv'] = uv[:, w.layout.body_landmarks]+.4
    w.gear = SimpleNamespace(visible_rows=[], blade_rows=[], landmark_rows=[])
    return w


class BodyDerivativeIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.model = synthetic_model(Path(self.temp.name)/'model.npz')

    def test_real_boot_evaluator_activates_fast_paths_and_matches_fresh_and_cached_jacobians(self):
        old = make_window(self.model, 'finite_difference')
        new = make_window(self.model); fresh = make_window(self.model)
        x = np.linspace(-.008, .009, new.size)
        # Count the unchanged production functions. Replacing the sampler with
        # even a functools.wraps wrapper is deliberately unsupported by the
        # fast path: a custom wrapper may change the model's numerical output.
        calls = dict(surface=0, compact=0)
        codes = {FullSurface.sample_vertices.__code__: 'surface',
                 self.model.forward_rotations.__func__.__code__: 'compact'}
        def count_original_calls(frame, event, arg):
            if event == 'call' and frame.f_code in codes:
                calls[codes[frame.f_code]] += 1
        previous_profile = sys.getprofile()
        sys.setprofile(count_original_calls)
        try:
            expected = old.evaluate(x, True)
            old_calls = calls['compact']; old_surface_calls = calls['surface']
            calls.update(surface=0, compact=0)
            first = new.evaluate(x, False); initial = first['residual'].copy()
            upgraded = new.evaluate(x, True)
            new_calls = calls['compact']; new_surface_calls = calls['surface']
        finally:
            sys.setprofile(previous_profile)
        direct = fresh.evaluate(x, True)
        self.assertEqual(old_calls, 254)
        # fun->jac reuses exposure forward but still checks the midpoint
        # residual again. Neither new derivative chain calls compact forward.
        self.assertEqual(new_calls, 3)
        self.assertEqual(old_surface_calls, 127)
        self.assertEqual(new_surface_calls, 2)
        self.assertEqual(upgraded['compatibility_execution'], 'exact_native_fk_replay')
        self.assertEqual(expected['compatibility_execution'], 'world_finite_difference')
        counts = new.execution.record()['derivative_usage']
        self.assertEqual(counts['analytic_world_linearizations'], 1)
        self.assertEqual(counts['fk_replay_linearizations'], 1)
        self.assertEqual(counts['surface_replay_linearizations'], 1)
        self.assertEqual(counts['finite_difference_world_linearizations'], 0)
        self.assertTrue(upgraded['forward_reused'])
        self.assertIsNot(upgraded, first)
        np.testing.assert_array_equal(upgraded['residual'], initial)
        np.testing.assert_array_equal(upgraded['residual'], expected['residual'])
        np.testing.assert_array_equal(upgraded['residual'], direct['residual'])
        np.testing.assert_allclose(upgraded['jac'].toarray(), expected['jac'].toarray(), rtol=2e-6, atol=2e-7)
        np.testing.assert_allclose(upgraded['jac'].toarray(), direct['jac'].toarray(), rtol=0., atol=1e-13)
        self.assertEqual(len(upgraded['residual']), len(expected['residual']))
        self.assertIs(new.evaluate(x, True), upgraded)

    def test_custom_compatibility_consumer_retains_original_world_differencing(self):
        def custom(window):
            base = type(window)
            class Extra(base):
                def compatibility_from_state(self, state):
                    return super().compatibility_from_state(state)*1.01
            window.__class__ = Extra
            return window
        old = custom(make_window(self.model, 'finite_difference'))
        new = custom(make_window(self.model))
        x = np.linspace(-.006, .005, new.size)
        expected = old.evaluate(x, True); actual = new.evaluate(x, True)
        self.assertEqual(actual['compatibility_execution'], 'world_finite_difference')
        self.assertEqual(new.execution.record()['derivative_usage']['fk_replay_linearizations'], 0)
        np.testing.assert_array_equal(actual['residual'], expected['residual'])
        np.testing.assert_allclose(actual['jac'].toarray(), expected['jac'].toarray(), rtol=2e-6, atol=2e-7)


if __name__ == '__main__':
    unittest.main()
