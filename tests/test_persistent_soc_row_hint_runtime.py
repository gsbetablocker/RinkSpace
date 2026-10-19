from contextvars import ContextVar
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from calibration.solver_runtime import persistent_soc_row_hint_runtime as helper


class PersistentSOCRowHintTest(unittest.TestCase):
    def test_initial_isolation_SOC_rows_only_load_save_and_original_error(self):
        with TemporaryDirectory() as temp, patch.dict(os.environ, {'RINKSPACE_SOLVER_DATA_ROOT': 'actual-event-fixture'}):
            directory = Path(temp)
            context_variable = ContextVar('SOC_hint_fixture', default=None)
            old_hint = dict(shape=(6, 2), scope='actual-event-fixture', rows=(0,))
            holder = SimpleNamespace(_last_verified_separation_faces=old_hint)
            initial_calls = []
            seen_SOC_hints = []
            original_error = ValueError('original InsufficientProgress fixture')
            mode = {'fail': False}

            def initial(*args):
                initial_calls.append(args)
                return 'initial-original-result'

            def correction(*args):
                context = context_variable.get()
                seen_SOC_hints.append(holder._last_verified_separation_faces['rows'])
                context['last_restricted_face_rows'] = np.array([2, 3] if mode['fail'] else [1, 2, 5])
                holder._last_verified_separation_faces = dict(shape=(6, 2),
                    scope='actual-event-fixture', rows=(4,))
                if mode['fail']:
                    raise original_error
                return 'SOC-original-full-KKT-result'

            wrapped = helper._wrap(initial, correction, holder, context_variable, directory)
            args = (np.eye(2), np.ones(2), np.array([-np.inf, 0.]),
                np.array([np.inf, 1.]), np.ones((4, 2)), np.zeros(4))
            self.assertEqual(wrapped(*args), 'initial-original-result')
            self.assertEqual(len(initial_calls), 1)
            self.assertEqual(list(directory.iterdir()), [])
            self.assertIs(holder._last_verified_separation_faces, old_hint)

            token = context_variable.set({})
            try:
                self.assertEqual(wrapped(*args), 'SOC-original-full-KKT-result')
                self.assertEqual(seen_SOC_hints[-1], (0,))
                self.assertIs(holder._last_verified_separation_faces, old_hint)
                path = helper._hint_path(directory, 'actual-event-fixture', (6, 2))
                payload = json.loads(path.read_text(encoding='utf8'))
                self.assertEqual(set(payload), {'schema', 'scope', 'shape', 'rows',
                    'solution_or_duals_stored', 'authority', 'primal_gate'})
                self.assertEqual(payload['primal_gate'],helper.PRIMAL_CERTIFICATE_SCHEMA)
                self.assertEqual(payload['rows'], [1, 2, 5])
                mode['fail'] = True
                with self.assertRaises(ValueError) as caught:
                    wrapped(*args)
                self.assertIs(caught.exception, original_error)
                self.assertEqual(seen_SOC_hints[-1], (1, 2, 5))
                self.assertIs(holder._last_verified_separation_faces, old_hint)
                self.assertEqual(json.loads(path.read_text())['rows'], [2, 3])
                mode['fail'] = False
                path.write_text('{invalid json', encoding='utf8')
                self.assertEqual(wrapped(*args), 'SOC-original-full-KKT-result')
                self.assertEqual(seen_SOC_hints[-1], (0,))
                self.assertIs(holder._last_verified_separation_faces, old_hint)
            finally:
                context_variable.reset(token)
            self.assertEqual(wrapped(*args), 'initial-original-result')
            self.assertEqual(len(initial_calls), 2)
            self.assertEqual(len(seen_SOC_hints), 3)


if __name__ == '__main__':
    unittest.main()
