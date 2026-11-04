"""A root-only stage must not stall on invariant foot/shoe geometry.

No model, native camera, video, inference or optimization is required.
Run from a repository containing this patch:
    python -m unittest discover -s tests -p test_root_structural_stagnation.py -v
"""
import unittest

from calibration.program_tracking.body_tracking.skate_run_policy import (
    Monitor, RunPolicy, structural_progress_is_observable,
)


def reason_codes(decision):
    return {reason['code'] for reason in decision['reasons']}


class RootStructuralStagnationTests(unittest.TestCase):
    def monitor(self, *, columns=(0, 1, 2), optimize_shared=False, **policy):
        return Monitor(RunPolicy(wall_seconds=120., max_accepted=8, **policy), now=0.,
            violation_stagnation_enabled=structural_progress_is_observable(
                columns, optimize_shared=optimize_shared))

    def test_constant_collar_does_not_stop_improving_root_after_four_steps(self):
        # The invariant structural value is the one in the saved spin A root
        # run. Costs here are a synthetic descending source-image sequence.
        monitor = self.monitor()
        violation = .023319084072951135
        monitor.observe(now=0., violation_m=violation, image_cost=100., pose_cost=0.)
        for step in range(1, 6):
            decision = monitor.observe(now=float(step), accepted=True,
                violation_m=violation, image_cost=100.-step, pose_cost=float(step))
            self.assertFalse(decision['stop'])
        record = monitor.record()
        self.assertFalse(record['violation_stagnation_enabled'])
        self.assertEqual(record['latest']['violation_m'], violation)
        self.assertEqual(record['best_violation_m'], violation)
        self.assertFalse(record['convergence_claim'])

    def test_structural_stage_keeps_existing_stagnation_stop(self):
        for kwargs in ({'columns': (6, 7, 8)}, {'optimize_shared': True}):
            with self.subTest(**kwargs):
                monitor = self.monitor(**kwargs)
                monitor.observe(now=0., violation_m=.023, image_cost=100.)
                for step in range(1, 5):
                    decision = monitor.observe(now=float(step), accepted=True,
                        violation_m=.023, image_cost=100.-step)
                self.assertEqual(reason_codes(decision), {'violation_stagnation'})
                self.assertTrue(monitor.record()['violation_stagnation_enabled'])

    def test_root_still_stops_on_source_image_regression(self):
        monitor = self.monitor()
        monitor.observe(now=0., violation_m=.023, image_cost=100.)
        decision = monitor.observe(now=1., accepted=True, violation_m=.023, image_cost=106.)
        self.assertEqual(reason_codes(decision), {'image_cost_increase'})
        self.assertTrue(decision['quality_failed'])

    def test_root_still_stops_at_accepted_and_wall_budgets(self):
        monitor = self.monitor()
        monitor.observe(now=0., violation_m=.023, image_cost=100.)
        for step in range(1, 9):
            decision = monitor.observe(now=float(step), accepted=True,
                violation_m=.023, image_cost=100.-step)
        self.assertEqual(reason_codes(decision), {'accepted_budget'})
        self.assertFalse(decision['quality_failed'])
        monitor = self.monitor()
        decision = monitor.observe(now=120., violation_m=.023)
        self.assertEqual(reason_codes(decision), {'wall_budget'})

    def test_global_rigid_columns_are_not_internal_structure(self):
        self.assertFalse(structural_progress_is_observable(range(6), optimize_shared=False))
        self.assertTrue(structural_progress_is_observable(range(3, 69), optimize_shared=False))
        self.assertTrue(structural_progress_is_observable(range(3), optimize_shared=True))
        for columns in ((-1,), (69,), (True,), (1.5,)):
            with self.subTest(columns=columns), self.assertRaises(ValueError):
                structural_progress_is_observable(columns, optimize_shared=False)


if __name__ == '__main__':
    unittest.main(verbosity=2)
