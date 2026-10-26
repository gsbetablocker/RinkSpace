"""Regression cases for native stop boundaries and failed cached linearizations.

These require an explicit CUDA/cuDSS library; a CPU-only run must skip them.
They use planted numeric observations and make no footage-accuracy claim.
"""
import os
import unittest
from unittest.mock import patch

import numpy as np
import test_g3_gpu as fixture


@unittest.skipUnless(os.environ.get("RINKSPACE_G4_LIBRARY"), "explicit CUDA/cuDSS library required")
class NativeFailureBoundaries(unittest.TestCase):
    def setUp(self):
        library = os.environ["RINKSPACE_G4_LIBRARY"]
        with patch.dict(os.environ, RINKSPACE_G2_LIBRARY=library, RINKSPACE_G3_LIBRARY=library):
            previous = fixture.G3GPU()
            previous.setUp()
        for name in ("core", "kw", "directions", "W", "plan"):
            setattr(self, name, getattr(previous, name))

    run_problem = fixture.G3GPU.run_problem
    attach = fixture.G3GPU.attach

    def test_stop_reaches_candidate_controller_without_committing_a_trial(self):
        with self.run_problem() as problem:
            self.attach(problem)
            original = problem.export_state()
            problem.request_stop()
            # lm_step goes directly into G3, bypassing G4's outer stop check.
            with self.assertRaisesRegex(RuntimeError, "status 25:.*stopped"):
                problem.lm_step()
            for key, values in original.items():
                np.testing.assert_array_equal(problem.export_state()[key], values)
            stopped = problem.solve()
            self.assertEqual(stopped["stop_reason"], "stopped_by_caller")
            self.assertEqual(stopped["trials"], 0)
            self.assertTrue(problem.solve()["numerically_converged"])

    def test_overflowed_whitening_cannot_be_exported_as_a_ready_system(self):
        overflowing = {**self.plan, "weights": self.plan["weights"] * 1e200}
        for backend in ("cpu", "cuda"):
            with self.subTest(backend=backend), self.run_problem(overflowing) as problem:
                # Input doubles and raw source rows are finite; W*r overflows
                # only when its squared norm is accumulated.
                with self.assertRaisesRegex(RuntimeError, "invalid rows retained"):
                    problem.linearize(backend)
                with self.assertRaisesRegex(RuntimeError, "invalid G2 explicit block export"):
                    problem.export_blocks(backend)

    def test_async_stop_inside_real_factorization_retry_preserves_accepted_bytes(self):
        import threading
        directions=self.directions.copy();directions[:,:,1]=directions[:,:,0]
        with self.run_problem({**self.plan,"directions":directions}) as problem:
            n=int(problem.linearization_info["local_count"])
            problem.configure_solver(lower=np.zeros(n),upper=np.ones(n),initial_lambda=1e-300,
                device_budget_bytes=problem.creation["free_after_alloc_bytes"],host_budget_bytes=2**30)
            original={k:v.tobytes() for k,v in problem.export_state().items()}
            events=[]
            def on_retry(record):
                events.append(record)
                stopper=threading.Thread(target=problem.request_stop)
                stopper.start();stopper.join()
            stopped=problem.solve(numerical_progress=on_retry)
            self.assertGreaterEqual(len(events),1)
            self.assertEqual(stopped["stop_reason"],"stopped_by_caller")
            self.assertEqual(stopped["accepted"],0)
            self.assertEqual(original,{k:v.tobytes() for k,v in problem.export_state().items()})
            resumed=problem.solve()
            self.assertTrue(resumed["numerically_converged"])
