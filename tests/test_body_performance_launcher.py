"""Check command forwarding and environment isolation without starting a fit."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

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

    def test_missing_library_fails_before_any_target_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profile = dict(native_library='missing.dll', cpu=dict(python_base='project', python=sys.executable, workers=1, native_threads=1))
            with self.assertRaisesRegex(FileNotFoundError, 'native library'):
                launcher.command_environment(root, root, profile, 'cpu', 'fit.py', [])


if __name__ == '__main__':
    unittest.main()
