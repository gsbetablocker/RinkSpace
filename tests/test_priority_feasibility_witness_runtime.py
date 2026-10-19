from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from calibration.solver_runtime import runtime
runtime.install()
from structured_programme_source_candidate import feasibility_priority_restoration as source
from calibration.solver_runtime import priority_feasibility_witness_runtime as helper


class PriorityFeasibilityWitnessTest(unittest.TestCase):
    def setUp(self):
        self.original = source._verified_same_original_conic_phaseI
        self.receipt = helper._RECEIPT
        helper._RECEIPT = None

    def tearDown(self):
        source._verified_same_original_conic_phaseI = self.original
        helper._RECEIPT = self.receipt

    @staticmethod
    def inputs():
        return (np.array([-2., -np.inf]), np.array([2., np.inf]),
                np.array([[1., 0.], [-1., 0.]]), np.array([2., 0.]),
                np.array([0, 0]), np.array([2.]))

    def test_primary_success_path_and_all_targets_unchanged(self):
        expected = source.contract_original_support_bounds(*self.inputs())
        helper.install_priority_feasibility_witness_runtime()
        actual = source.contract_original_support_bounds(*self.inputs())
        np.testing.assert_array_equal(actual['matrix'], expected['matrix'])
        np.testing.assert_array_equal(actual['lower'], expected['lower'])
        self.assertEqual(actual['common_fraction'], expected['common_fraction'])

    def test_nearly_solved_label_requires_complete_primal_witness(self):
        import clarabel
        real_solver = clarabel.DefaultSolver
        def almost(*args, **kwargs):
            solved = real_solver(*args, **kwargs).solve()
            result = SimpleNamespace(x=solved.x, z=solved.z,
                status='AlmostSolved', iterations=solved.iterations)
            return SimpleNamespace(solve=lambda: result)
        helper.install_priority_feasibility_witness_runtime()
        failed = SimpleNamespace(success=False, message='forced primary numerical failure')
        with patch.object(source, 'linprog', return_value=failed), patch('clarabel.DefaultSolver', side_effect=almost):
            actual = source.contract_original_support_bounds(*self.inputs())
        report = actual['report']['phase_I_failed_backend_conic_fallback']
        self.assertEqual(report['backend_terminal_status'], 'AlmostSolved')
        self.assertFalse(report['phase_I_optimality_certificate_claimed'])
        self.assertGreaterEqual(report['original_midpoint_all_face_normalized_minimum_slack'], -1e-9)
        self.assertAlmostEqual(actual['common_fraction'], .75, places=7)

    def test_false_success_and_no_strict_contraction_rejected(self):
        helper.install_priority_feasibility_witness_runtime()
        failed = SimpleNamespace(success=False, message='forced primary numerical failure')
        # There are two support rows, two finite coordinate-bound rows, and
        # two auxiliary-bound rows; the free source column is exactly absent.
        for candidate in ([0., .2], [0., 1.]):
            fake = SimpleNamespace(solve=lambda: SimpleNamespace(
                status='AlmostSolved', x=candidate, z=np.zeros(6), iterations=1))
            with patch.object(source, 'linprog', return_value=failed), patch('clarabel.DefaultSolver', return_value=fake):
                with self.assertRaises(ValueError):
                    source.contract_original_support_bounds(*self.inputs())

    def test_true_protected_domain_infeasibility_rejected(self):
        helper.install_priority_feasibility_witness_runtime()
        failed = SimpleNamespace(success=False, message='forced primary numerical failure')
        inputs = (np.array([0.]), np.array([1.]), np.array([[1.], [-1.]]),
                  np.array([2., -2.]), np.array([-1, 0]), np.array([1.]))
        with patch.object(source, 'linprog', return_value=failed):
            with self.assertRaises(ValueError):
                source.contract_original_support_bounds(*inputs)

    def test_empty_or_wrong_shape_candidate_rejected(self):
        helper.install_priority_feasibility_witness_runtime()
        failed = SimpleNamespace(success=False, message='forced primary numerical failure')
        for candidate in ([], [.5], [[0., .5]], [0., 0., .5]):
            fake = SimpleNamespace(solve=lambda: SimpleNamespace(
                status='AlmostSolved', x=candidate, z=np.zeros(6), iterations=1))
            with patch.object(source, 'linprog', return_value=failed), patch('clarabel.DefaultSolver', return_value=fake):
                with self.assertRaises(ValueError):
                    source.contract_original_support_bounds(*self.inputs())

    def test_valid_primal_with_invalid_LP_dual_optimality_remains_a_witness(self):
        helper.install_priority_feasibility_witness_runtime()
        failed = SimpleNamespace(success=False, message='forced primary numerical failure')
        # Original support/box rows require t >= .5. This feasible t=.6
        # is deliberately not an optimum, and zero dual fails stationarity.
        fake = SimpleNamespace(solve=lambda: SimpleNamespace(
            status='AlmostSolved', x=[1., .6], z=np.zeros(6), iterations=1))
        with patch.object(source, 'linprog', return_value=failed), patch('clarabel.DefaultSolver', return_value=fake):
            actual = source.contract_original_support_bounds(*self.inputs())
        report = actual['report']['phase_I_failed_backend_conic_fallback']
        self.assertEqual(report['stationarity_maximum_absolute'], 1.)
        self.assertAlmostEqual(report['primal_dual_gap'], .6)
        self.assertFalse(report['phase_I_optimality_certificate_claimed'])
        self.assertAlmostEqual(actual['report']['phase_I_feasible_common_fraction_witness'], .6)
        self.assertAlmostEqual(actual['common_fraction'], .8)
        self.assertTrue(actual['report']['original_midpoint_all_faces_and_physical_boxes_primal_verified'])

    def test_wrong_conic_dual_row_shape_rejected(self):
        helper.install_priority_feasibility_witness_runtime()
        failed = SimpleNamespace(success=False, message='forced primary numerical failure')
        fake = SimpleNamespace(solve=lambda: SimpleNamespace(
            status='AlmostSolved', x=[1., .6], z=np.zeros(1), iterations=1))
        with patch.object(source, 'linprog', return_value=failed), patch('clarabel.DefaultSolver', return_value=fake):
            with self.assertRaises(ValueError):
                source.contract_original_support_bounds(*self.inputs())

    def test_failed_conic_primal_repairs_only_auxiliary_t_at_exact_face(self):
        helper.install_priority_feasibility_witness_runtime()
        failed = SimpleNamespace(success=False, message='forced primary numerical failure')
        original = np.array([1., .5 - 2e-8])
        dual = np.zeros(6)
        fake = SimpleNamespace(solve=lambda: SimpleNamespace(
            status='AlmostSolved', x=original.copy(), z=dual, iterations=1))
        with patch.object(source, 'linprog', return_value=failed), patch('clarabel.DefaultSolver', return_value=fake):
            actual = source.contract_original_support_bounds(*self.inputs())
        report = actual['report']['phase_I_failed_backend_conic_fallback']
        repair = report['auxiliary_t_failed_primal_repair']
        self.assertEqual(repair['method'], 'original_inequality_interval_endpoint_upward_roundoff')
        self.assertEqual(repair['exact_original_auxiliary_interval_lower'], .5)
        self.assertEqual(repair['repaired_auxiliary_t'], np.nextafter(.5, np.inf))
        self.assertEqual(repair['original_failed_auxiliary_t'], original[-1])
        self.assertLess(repair['original_failed_normalized_minimum_slack'], -1e-9)
        self.assertGreaterEqual(repair['repaired_all_face_and_box_normalized_minimum_slack'], -1e-9)
        self.assertTrue(repair['model_coordinates_bit_exact_unchanged'])
        self.assertFalse(report['phase_I_optimality_certificate_claimed'])
        self.assertAlmostEqual(actual['common_fraction'], .75)
        np.testing.assert_array_equal(original, [1., .5 - 2e-8])

    def test_already_certified_conic_path_keeps_fraction_bit_exact(self):
        helper.install_priority_feasibility_witness_runtime()
        failed = SimpleNamespace(success=False, message='forced primary numerical failure')
        # The old complete-primal gate accepts this floating error; its
        # existing numerical path and midpoint must not be changed.
        fraction = .5 - 2e-10
        fake = SimpleNamespace(solve=lambda: SimpleNamespace(
            status='AlmostSolved', x=[1., fraction], z=np.zeros(6), iterations=1))
        with patch.object(source, 'linprog', return_value=failed), patch('clarabel.DefaultSolver', return_value=fake):
            actual = source.contract_original_support_bounds(*self.inputs())
        self.assertEqual(actual['report']['phase_I_feasible_common_fraction_witness'], fraction)
        self.assertEqual(actual['common_fraction'], .5 * (1. + fraction))
        self.assertNotIn('auxiliary_t_failed_primal_repair',
            actual['report']['phase_I_failed_backend_conic_fallback'])

    def test_failed_primal_invalid_dual_still_rejected_without_repair(self):
        helper.install_priority_feasibility_witness_runtime()
        failed = SimpleNamespace(success=False, message='forced primary numerical failure')
        for bad in (-1e-9, np.nan, np.inf):
            dual = np.zeros(6)
            dual[0] = bad
            fake = SimpleNamespace(solve=lambda: SimpleNamespace(
                status='AlmostSolved', x=[1., .5 - 2e-8], z=dual, iterations=1))
            with patch.object(source, 'linprog', return_value=failed), patch('clarabel.DefaultSolver', return_value=fake):
                with self.assertRaises(ValueError):
                    source.contract_original_support_bounds(*self.inputs())

    def test_repair_preserves_failed_vectors_and_checks_every_original_face(self):
        B = np.array([[1.], [-1.], [0.]])
        expansion = np.array([2., 0., 1.])
        rhs = np.array([2., -1.5, .2])
        norm = np.sqrt((B * B).sum(axis=1) + expansion * expansion)
        A = -np.c_[B, expansion] / norm[:, None]
        b = -rhs / norm
        x = np.array([1., .5 - 2e-8])
        slack = np.r_[b - A @ x, x[-1], 1. - x[-1]]
        repaired, checked, diagnostic = helper._repair_auxiliary_fraction(
            x, np.zeros(5), B, expansion, rhs, norm, A, b, slack, x[-1])
        np.testing.assert_array_equal(repaired[:-1], x[:-1])
        np.testing.assert_array_equal(diagnostic['failed_coordinate'], x)
        np.testing.assert_array_equal(diagnostic['failed_slack'], slack)
        np.testing.assert_array_equal(checked, np.r_[b - A @ repaired, repaired[-1], 1. - repaired[-1]])
        self.assertGreaterEqual(np.min(checked), -1e-9)
        self.assertEqual(repaired[-1], np.nextafter(.5, np.inf))

        # The immutable second face now genuinely fails.  Neither an
        # endpoint projection nor the midpoint may repair it by changing x.
        bad_rhs = rhs.copy()
        bad_rhs[1] = -.5
        bad_b = -bad_rhs / norm
        bad_slack = np.r_[bad_b - A @ x, x[-1], 1. - x[-1]]
        same, same_slack, rejected = helper._repair_auxiliary_fraction(
            x, np.zeros(5), B, expansion, bad_rhs, norm, A, bad_b, bad_slack, x[-1])
        self.assertIs(same, x)
        self.assertIs(same_slack, bad_slack)
        self.assertIsNone(rejected)

    def test_auxiliary_interval_needs_strict_contraction_and_exact_box(self):
        B = np.zeros((1, 1))
        expansion = norm = np.ones(1)
        A = -np.c_[B, expansion]
        for rhs, fraction in ((1., .9), (1.1, .9), (.5, 1.01), (.5, np.nan)):
            x = np.array([0., fraction])
            b = np.array([-rhs])
            slack = np.r_[b - A @ x, x[-1], 1. - x[-1]]
            same, _, diagnostic = helper._repair_auxiliary_fraction(
                x, np.zeros(3), B, expansion, np.array([rhs]), norm, A, b, slack, fraction)
            self.assertIs(same, x)
            self.assertIsNone(diagnostic)

    def test_midpoint_fallback_is_a_checked_candidate_not_an_assumption(self):
        # Normalized subtraction loses enough low bits at the exact t
        # endpoint to fail the old gate.  Its checked midpoint is feasible.
        B = expansion = np.ones((1, 1))
        expansion = expansion[:, 0]
        norm = np.sqrt(np.array([2.]))
        rhs = np.array([100000000.5])
        A = -np.c_[B, expansion] / norm[:, None]
        b = -rhs / norm
        x = np.array([100000000., .49])
        slack = np.r_[b - A @ x, x[-1], 1. - x[-1]]
        repaired, checked, diagnostic = helper._repair_auxiliary_fraction(
            x, np.zeros(3), B, expansion, rhs, norm, A, b, slack, x[-1])
        self.assertEqual(diagnostic['report']['method'], 'checked_original_auxiliary_midpoint')
        self.assertEqual(repaired[-1], .75)
        self.assertEqual(repaired[0], x[0])
        self.assertGreaterEqual(np.min(checked), -1e-9)
        np.testing.assert_array_equal(checked, np.r_[b - A @ repaired, repaired[-1], 1. - repaired[-1]])

        # An independent immutable row fails: no assumed midpoint can hide
        # that original-domain failure or change the model coordinate.
        B2 = np.r_[B, [[0.]]]
        e2 = np.r_[expansion, 0.]
        rhs2 = np.r_[rhs, 1.]
        norm2 = np.r_[norm, 1.]
        A2 = -np.c_[B2, e2] / norm2[:, None]
        b2 = -rhs2 / norm2
        s2 = np.r_[b2 - A2 @ x, x[-1], 1. - x[-1]]
        same, _, rejected = helper._repair_auxiliary_fraction(
            x, np.zeros(4), B2, e2, rhs2, norm2, A2, b2, s2, x[-1])
        self.assertIs(same, x)
        self.assertIsNone(rejected)

    def test_failed_repair_rejects_nonoriginal_negative_expansion(self):
        B = np.zeros((1, 1))
        expansion = np.array([-1.])
        rhs = norm = np.ones(1)
        A = -np.c_[B, expansion]
        b = -rhs
        x = np.array([0., .5])
        slack = np.r_[b - A @ x, x[-1], 1. - x[-1]]
        with self.assertRaisesRegex(ValueError, 'must be nonnegative'):
            helper._repair_auxiliary_fraction(
                x, np.zeros(3), B, expansion, rhs, norm, A, b, slack, x[-1])


if __name__ == '__main__':
    unittest.main()
