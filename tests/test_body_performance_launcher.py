"""Check command forwarding and environment isolation without starting a fit."""
import importlib.util
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('run_body_performance', Path(__file__).resolve().parents[1]/'scripts/run_body_performance.py')
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class PerformanceLauncherTests(unittest.TestCase):
    def test_child_receives_args_and_settings_without_mutating_parent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'body.dll').touch()
            script = root/'probe.py'
            script.write_text("import json,os,sys; print(json.dumps(dict(args=sys.argv[1:], backend=os.environ['RINKSPACE_BODY_BACKEND'], workers=os.environ['RINKSPACE_BODY_WORKERS'])))")
            profile = dict(native_library='body.dll', cpu=dict(python_base='project', python=sys.executable, workers=4, native_threads=2))
            parent_before = dict(os.environ)
            args = ['--', '--root', 'path with spaces', '--warm-start', 'kept.npz']
            command, env = launcher.command_environment(root, root, profile, 'cpu', script, args)
            self.assertEqual(dict(os.environ), parent_before)
            result = subprocess.run(command, env=env, capture_output=True, text=True, check=True)
            output = json.loads(result.stdout)
            self.assertEqual(output['args'], args[1:])
            self.assertEqual((output['backend'], output['workers']), ('cpu', '4'))
            self.assertEqual(args[0], '--')
            self.assertEqual(dict(os.environ), parent_before)

    def test_cuda_rejects_multiple_body_owners(self):
        profile = dict(cuda=dict(python_base='data', python='python.exe', workers=1, native_threads=4))
        with self.assertRaisesRegex(ValueError, 'one body owner'):
            launcher.command_environment(Path('.'), Path('.'), profile, 'cuda', 'fit.py', [], workers=2)

    def test_modes_override_inherited_execution_settings_and_preserve_target_args(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'body.dll').touch()
            script = root/'probe.py'
            script.write_text('import json,os; print(json.dumps({k:v for k,v in os.environ.items() if k.startswith("RINKSPACE_BODY_")}))')
            cpu = dict(python_base='project', python=sys.executable, workers=12, native_threads=4)
            profile = dict(native_library='body.dll', cpu=cpu,
                cuda=dict(cpu, python_base='data', workers=1), hybrid=dict(cpu, python_base='data'))
            target_args = ['--', '--backend', 'kept_by_target', '--state', 'path with spaces.npz']
            inherited = dict(RINKSPACE_BODY_BACKEND='wrong', RINKSPACE_BODY_COMPATIBILITY_BACKEND='wrong',
                RINKSPACE_BODY_LOCAL_DERIVATIVES='wrong', RINKSPACE_BODY_WORKERS='99',
                RINKSPACE_BODY_NATIVE_THREADS='99', OPENBLAS_NUM_THREADS='99',
                MKL_NUM_THREADS='99', OMP_NUM_THREADS='99', PYTHONPATH='other checkout',
                LAUNCHER_TEST_UNRELATED='preserved')
            with patch.dict(os.environ, inherited):
                parent_before = dict(os.environ)
                for mode, body, compatibility, derivatives, workers in (
                    ('cpu', 'cpu', 'cpu', 'analytic', '12'),
                    ('cuda', 'cuda', 'cpu', 'finite_difference', '1'),
                    ('hybrid', 'cpu', 'cuda_resident', 'analytic', '12')):
                    with self.subTest(mode=mode):
                        command, env = launcher.command_environment(root, root, profile, mode, script, target_args)
                        self.assertEqual(command[5:], target_args[1:])
                        self.assertEqual(env['RINKSPACE_BODY_BACKEND'], body)
                        self.assertEqual(env['RINKSPACE_BODY_COMPATIBILITY_BACKEND'], compatibility)
                        self.assertEqual(env['RINKSPACE_BODY_LOCAL_DERIVATIVES'], derivatives)
                        self.assertEqual(env['RINKSPACE_BODY_WORKERS'], workers)
                        self.assertEqual(env['RINKSPACE_BODY_NATIVE_THREADS'], '4')
                        self.assertEqual(env['RINKSPACE_BODY_BLAS_THREADS'], '1')
                        self.assertEqual(env['PYTHONPATH'], os.pathsep.join((str(root/'src'), str(root))))
                        self.assertEqual(env['LAUNCHER_TEST_UNRELATED'], 'preserved')
                        for name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
                            self.assertEqual(env[name], '1')
                        self.assertEqual(dict(os.environ), parent_before)
                # Launch only this tiny environment reader, never a model or fit.
                child = subprocess.run(command, env=env, capture_output=True, text=True, check=True)
                settings = json.loads(child.stdout)
                self.assertEqual(settings['RINKSPACE_BODY_BACKEND'], 'cpu')
                self.assertEqual(settings['RINKSPACE_BODY_COMPATIBILITY_BACKEND'], 'cuda_resident')
                self.assertEqual(settings['RINKSPACE_BODY_WORKERS'], '12')
                self.assertEqual(dict(os.environ), parent_before)

    def test_profile_default_and_older_profile_default_are_explicitly_overridable(self):
        profile = dict(cpu={}, cuda={}, hybrid={}, default_backend='hybrid')
        self.assertEqual(launcher.select_backend(profile), 'hybrid')
        self.assertEqual(launcher.select_backend(profile, 'cpu'), 'cpu')
        del profile['default_backend']
        self.assertEqual(launcher.select_backend(profile), 'cpu')
        with self.assertRaisesRegex(ValueError, 'backend must be'):
            launcher.select_backend(dict(profile, default_backend='unknown'))
        with self.assertRaisesRegex(ValueError, 'no hybrid settings'):
            launcher.select_backend({'cpu': {}}, 'hybrid')

    def test_show_command_uses_hybrid_profile_default_without_running_child(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'body.dll').touch()
            script = root/'probe.py'; script.touch()
            profile = dict(default_backend='hybrid', native_library='body.dll',
                hybrid=dict(python_base='data', python=sys.executable, workers=12, native_threads=4))
            profile_path = root/'profile.json'
            profile_path.write_text(json.dumps(profile), encoding='utf8')
            storage = SimpleNamespace(AssetStore=lambda _: SimpleNamespace(data=root))
            output = io.StringIO()
            with patch.dict(sys.modules, {'rinkspace.storage': storage}), patch.object(sys, 'path', sys.path.copy()), \
                    patch.object(launcher.subprocess, 'run') as run, redirect_stdout(output):
                result = launcher.main(['--profile', str(profile_path), '--show-command', str(script), '--kept'])
            self.assertEqual(result, 0)
            run.assert_not_called()
            record = json.loads(output.getvalue())
            self.assertFalse(record['started'])
            self.assertEqual(record['backend'], 'hybrid')
            self.assertEqual(record['settings']['RINKSPACE_BODY_BACKEND'], 'cpu')
            self.assertEqual(record['settings']['RINKSPACE_BODY_COMPATIBILITY_BACKEND'], 'cuda_resident')
            self.assertEqual(record['command'][-1], '--kept')

    def test_missing_library_fails_before_any_target_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profile = dict(native_library='missing.dll', cpu=dict(python_base='project', python=sys.executable, workers=1, native_threads=1))
            with self.assertRaisesRegex(FileNotFoundError, 'native library'):
                launcher.command_environment(root, root, profile, 'cpu', 'fit.py', [])


if __name__ == '__main__':
    unittest.main()
