"""Asset-free tests for profile input identity, sparse comparison and caches."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from scipy.sparse import csr_matrix

spec = importlib.util.spec_from_file_location('profile_body_solver', Path(__file__).resolve().parents[1] / 'scripts/profile_body_solver.py')
profile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profile)


class ProfileArgumentTests(unittest.TestCase):
    def test_parse_without_native_or_torch_import(self):
        args = profile.parser().parse_args(['--kind', 'boot', '--root', 'data', '--output', 'profile',
                                           '--native-library', 'candidate.dll', '--native-threads', '8'])
        self.assertEqual(args.repeats, 3)
        self.assertEqual(args.backend, 'cpu')
        self.assertEqual(args.workers, 1)
        self.assertEqual(args.native_library, Path('candidate.dll'))
        self.assertEqual(args.native_threads, 8)

    def test_current_opening_kinds_accept_an_explicit_immutable_state(self):
        for kind in ('motion', 'timing'):
            args = profile.parser().parse_args(['--kind', kind, '--root', 'data',
                '--output', 'profile', '--state', 'accepted-copy.npz'])
            self.assertEqual(args.kind, kind)
            self.assertEqual(args.state, Path('accepted-copy.npz'))

    def test_original_baseline_uses_only_cpu_scope(self):
        with tempfile.TemporaryDirectory() as tmp:
            execution, config, kind = profile.execution_classes(tmp, 'cpu')
            self.assertEqual(kind, 'baseline_without_execution_hooks')
            owner = SimpleNamespace(body=object())
            original = owner.body
            policy = execution(config(workers=1))
            policy.attach(owner)
            self.assertIs(owner.body, original)
            policy.close()
            with self.assertRaisesRegex(ValueError, 'no body execution hooks'):
                profile.execution_classes(tmp, 'cuda')

    def test_invalid_parallel_cuda_rejected_before_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'new'
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                profile.main(['--kind', 'fullbody', '--root', tmp, '--output', str(output),
                              '--backend', 'cuda', '--workers', '2'])
            self.assertFalse(output.exists())

    def test_existing_output_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            marker = Path(tmp) / 'saved.txt'; marker.write_text('keep')
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                profile.main(['--kind', 'opening', '--root', tmp, '--output', tmp])
            self.assertEqual(marker.read_text(), 'keep')

    def test_result_cache_clear_preserves_prepared_structures(self):
        binding = SimpleNamespace(cache={'old': True}, cache_key=b'old', x=np.zeros(14))
        window = SimpleNamespace(cache={'_forward': 'old'}, cache_x=np.ones(3), binding_key=b'old', binding=binding,
                                 spline_scatter=['immutable prepared support'])
        problem = SimpleNamespace(cached={'jac': 'old'}, cached_x=np.ones(8), windows=[window], binding=binding)
        profile.clear_result_caches(problem)
        self.assertIsNone(problem.cached); self.assertIsNone(problem.cached_x)
        self.assertIsNone(window.cache); self.assertIsNone(window.cache_x); self.assertIsNone(window.binding_key)
        self.assertIsNone(binding.cache); self.assertIsNone(binding.cache_key)
        self.assertEqual(window.spline_scatter, ['immutable prepared support'])
        np.testing.assert_array_equal(binding.x, np.zeros(14))


class SnapshotTests(unittest.TestCase):
    def snapshot(self, identity='same', x=None, delta=0.):
        return profile.snapshot_arrays(dict(residual=np.array([1., 2.]) + delta,
                                            jac=csr_matrix([[1., 0., 2.], [0., 3. + delta, 0.]])),
                                       np.zeros(3) if x is None else x,
                                       dict(input_fingerprint=identity))

    def compare(self, current, path):
        return profile.compare_snapshot(current, path, residual_atol=1e-8, residual_rtol=1e-10,
                                        jacobian_atol=5e-5, jacobian_rtol=2e-4)

    def test_exact_csr_roundtrip_and_difference(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'baseline.npz'; snapshot = self.snapshot()
            np.savez_compressed(path, **snapshot)
            value = self.compare(snapshot, path)
            self.assertTrue(value['residual_within_tolerance']); self.assertTrue(value['jacobian_within_tolerance'])
            self.assertEqual(value['jacobian_max_abs'], 0.)
            changed = self.compare(self.snapshot(delta=.1), path)
            self.assertFalse(changed['residual_within_tolerance']); self.assertFalse(changed['jacobian_within_tolerance'])
            self.assertAlmostEqual(changed['jacobian_max_abs'], .1)
            self.assertFalse(changed['automatic_backend_selection'])

    def test_native_binary_is_implementation_but_camera_arrays_are_inputs(self):
        args = SimpleNamespace(kind='fullbody', mode='full', window=1)
        records = {'camera_state_path': {'sha256': 'same-camera'}, 'observation': {'sha256': 'same-observation'},
                   'camera_native_library': {'sha256': 'old-dll'}}
        before = profile.input_identity(args, records, np.zeros(3), 'saved x')
        records['camera_native_library']['sha256'] = 'new-dll'
        self.assertEqual(before, profile.input_identity(args, records, np.zeros(3), 'saved x'))
        records['camera_state_path']['sha256'] = 'changed-camera'
        self.assertNotEqual(before, profile.input_identity(args, records, np.zeros(3), 'saved x'))

    def test_refuse_different_inputs_or_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'baseline.npz'; np.savez_compressed(path, **self.snapshot())
            for current in (self.snapshot(identity='other'), self.snapshot(x=np.ones(3))):
                with self.assertRaisesRegex(ValueError, 'inputs, objective layout or x differ'):
                    self.compare(current, path)

    def test_saved_state_must_contain_finite_control_vector(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'state.npz'; np.savez(path, x=np.array([np.nan]))
            with self.assertRaisesRegex(ValueError, 'finite one-dimensional'):
                profile._load_x(path)


if __name__ == '__main__':
    unittest.main()
