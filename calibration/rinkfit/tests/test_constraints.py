import math
import unittest

import numpy as np

from rinkfit_start.constraints import (
    audit_native_budget, ordinary_budget_q, phase_one_budget_residual,
    phase_one_budget_stationarity, phr_equality_residual, phr_inequality_residual,
    phr_update, profile_budget_q, scaled_projected_gradient,
    verified_order_inequalities,
)


class PHRTests(unittest.TestCase):
    def test_half_square_gradient_and_old_multiplier_progress(self):
        for b in (0.0, 0.3, 100.0):
            self.assertAlmostEqual(0.5 * phase_one_budget_residual(b) ** 2, b + 0.5)
        g, lam, rho = np.array([-3., -.2, .4]), np.array([1., 1., 0.]), 2.
        np.testing.assert_allclose(phr_inequality_residual(g, lam, rho), [0., .6 / math.sqrt(2), .8 / math.sqrt(2)])
        update = phr_update(g, [.1], lam, [-.4], rho, previous_violation=.5)
        # V=max(abs(max(g,-lambda_old/rho)),abs(h))=.5, not max(g,0)=.4.
        self.assertAlmostEqual(update.violation, .5)
        self.assertEqual(update.rho, 10.)
        np.testing.assert_allclose(update.inequality_multipliers, [0., .6, .8])
        np.testing.assert_allclose(update.equality_multipliers, [-.2])
        np.testing.assert_allclose(phr_equality_residual([.1], [-.4], 2), [-.1 * math.sqrt(2)])
        self.assertFalse(update.inequality_multipliers.flags.writeable)

    def test_cap_is_a_failure_flag_and_bounds_have_correct_normal_cone(self):
        update = phr_update([1.], [], [0.], [], 1e6, previous_violation=1.)
        self.assertTrue(update.penalty_limit_reached)
        self.assertEqual(update.rho, 1e6)
        self.assertEqual(phase_one_budget_stationarity(0., [.3]), 0.)
        self.assertAlmostEqual(phase_one_budget_stationarity(2., [.3]), .7)
        np.testing.assert_allclose(scaled_projected_gradient([0., 1., .5], [2., -4., .2], [1., 1., 2.], [0., 0., 0.], [1., 1., 1.]), [0., 0., .25])
        np.testing.assert_allclose(verified_order_inequalities([.2, .2, .8], [[0, 1], [1, 2]]), [0., -.6])


class NativeBudgetTests(unittest.TestCase):
    def test_full_native_ellipse_error_is_not_robust_or_normal_only(self):
        # On a finite horizontal material the far-tangent error remains a failure.
        uv = np.zeros((3, 2))
        pred = np.array([[0., 1.], [5., .1], [np.nan, 0.]])
        audit = audit_native_budget(uv, pred, 4 * np.eye(2), expected_rows=3,
                                    projection_valid=np.array([True, True, False]),
                                    support_valid=np.ones(3, dtype=bool),
                                    root_valid=np.ones(3, dtype=bool), physical_valid=True)
        self.assertEqual(audit.accounted_rows, 3)
        self.assertAlmostEqual(audit.q[1], 25.01 / 4)
        self.assertTrue(math.isinf(audit.q[2]))
        self.assertTrue(math.isinf(audit.t_hat))
        self.assertFalse(audit.passed)
        np.testing.assert_array_equal(audit.over_budget_rows, [1, 2])
        with self.assertRaisesRegex(ValueError, "incomplete native replay"):
            audit_native_budget(uv, pred[:2], 4 * np.eye(2), expected_rows=3,
                                projection_valid=np.ones(3, dtype=bool),
                                support_valid=np.ones(3, dtype=bool),
                                root_valid=np.ones(3, dtype=bool), physical_valid=True)

    def test_genuine_profile_keeps_query_extent_and_native_units(self):
        flags = np.ones(2, dtype=bool)
        args = dict(expected_rows=2, projection_valid=flags, support_valid=flags,
                    root_valid=flags, physical_valid=True, profile_mask=flags,
                    profile_normal_uv=np.array([[0., 1.], [0., 1.]]),
                    profile_radius_px=np.array([2., 2.]),
                    profile_alpha_bounds_px=np.array([[-2., 2.], [-2., 2.]]))
        audit = audit_native_budget(np.zeros((2, 2)), [[0., 1.], [5., 1.]], 4 * np.eye(2), **args)
        np.testing.assert_allclose(audit.q, [.25, .25])
        self.assertTrue(audit.pixel_pass)
        self.assertFalse(audit.support_pass)
        self.assertFalse(audit.passed)
        self.assertEqual(audit.query_error_px[1], 5.)
        np.testing.assert_allclose(profile_budget_q([1.], [2.]), [.25])
        # A sigma=0.1 statistical residual would be 10; it has no budget input.
        np.testing.assert_allclose(ordinary_budget_q([[1., 2.]], [[4., 0.], [0., 16.]]), [.5])

    def test_support_and_root_failures_cannot_be_accepted_on_low_cost(self):
        audit = audit_native_budget([[0., 0.]], [[0., 0.]], 4 * np.eye(2), expected_rows=1,
                                    projection_valid=np.array([True]), support_valid=np.array([False]),
                                    root_valid=np.array([True]), physical_valid=True)
        self.assertEqual(audit.t_hat, 0.)
        self.assertFalse(audit.passed)
        audit = audit_native_budget([[0., 0.]], [[0., 0.]], 4 * np.eye(2), expected_rows=1,
                                    projection_valid=np.array([True]), support_valid=np.array([True]),
                                    root_valid=np.array([False]), physical_valid=True)
        self.assertTrue(math.isinf(audit.q[0]))
        self.assertFalse(audit.passed)


if __name__ == "__main__":
    unittest.main()
