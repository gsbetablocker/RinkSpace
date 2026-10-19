from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.optimize import linprog

from calibration.solver_runtime import runtime
runtime.install()
from structured_programme_source_candidate import feasibility_priority_restoration as source
from calibration.solver_runtime import full_priority_clarabel_runtime as helper


class FullPriorityClarabelTest(unittest.TestCase):
    def test_complete_original_LP_full_dual_signs_zero_column_and_original_fallback_error(self):
        # Complete original problem: x+2t>=2, -x+2t>=0, -2<=x<=2,
        # 0<=t<=1. The second source coordinate is exactly free/zero.
        raw = np.array([[1., 0.], [-1., 0.], [1., 0.], [-1., 0.]])
        expansion = np.array([2., 2., 0., 0.])
        rhs = np.array([2., 0., -2., -2.])
        norm = np.sqrt(np.einsum('ij,ij->i', raw, raw) + expansion**2)
        c = np.array([0., 0., 1.])
        kwargs = dict(A_ub=-np.c_[raw, expansion]/norm[:, None], b_ub=-rhs/norm,
            bounds=[(None, None), (None, None), (0., 1.)], method='highs-ipm',
            options=dict(primal_feasibility_tolerance=1e-9, dual_feasibility_tolerance=1e-9))
        token = helper._LATEST.set(None)
        try:
            actual = helper.full_original_clarabel_linprog(linprog, c, **kwargs)
            trace = helper._LATEST.get()
            self.assertEqual(trace['route'], 'complete_original_Clarabel_full_KKT', trace)
            self.assertTrue(trace['complete_original_problem_KKT_certified'])
            self.assertTrue(trace['no_row_or_column_elimination'])
            self.assertEqual(actual.x.shape, (3,))
            self.assertAlmostEqual(actual.x[-1], .5, places=8)
            self.assertTrue(trace['complete_original_problem_KKT']['passed'])
            self.assertTrue(np.all(actual.ineqlin.marginals <= 1e-9))
            self.assertGreaterEqual(actual.lower.marginals[-1], -1e-9)
            self.assertLessEqual(actual.upper.marginals[-1], 1e-9)

            # Run the actual unchanged source LP definition before and after
            # this installer. No event/source graph is created.
            inputs = (np.array([-2., -np.inf]), np.array([2., np.inf]), raw[:2],
                rhs[:2], np.array([0, 0]), np.array([2.]))
            baseline = source.contract_original_support_bounds(*inputs)
            helper.install_full_priority_clarabel_runtime()
            candidate = source.contract_original_support_bounds(*inputs)
            self.assertEqual(candidate['report']['phase_I_backend'],
                'official Clarabel complete original LP (full primal/dual KKT)')
            np.testing.assert_array_equal(candidate['matrix'], baseline['matrix'])
            self.assertAlmostEqual(candidate['common_fraction'], baseline['common_fraction'], places=8)
            self.assertTrue(candidate['report']['complete_original_Clarabel_LP_backend']['complete_original_problem_KKT_certified'])

            original_error = ValueError('original HiGHS numeric exception')
            calls = []
            def original_failure(*args, **options):
                calls.append((args, options))
                raise original_error
            class BadPrimal:
                def solve(self):
                    return SimpleNamespace(status='Solved', iterations=1,
                        x=[0., 0., .2], z=np.zeros(6))
            with patch('clarabel.DefaultSolver', return_value=BadPrimal()):
                with self.assertRaises(ValueError) as caught:
                    helper.full_original_clarabel_linprog(original_failure, c, **kwargs)
            self.assertIs(caught.exception, original_error)
            self.assertEqual(len(calls), 1)
            self.assertIs(calls[0][0][0], c)
            self.assertIs(calls[0][1]['A_ub'], kwargs['A_ub'])
        finally:
            helper._LATEST.reset(token)


if __name__ == '__main__':
    unittest.main()
