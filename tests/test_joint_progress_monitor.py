"""Behavioral checks for the real 3d6f51f component-stagnation failure."""
import importlib.util
from pathlib import Path
import sys
import unittest

path=Path(__file__).resolve().parents[1]/'calibration/program_tracking/body_tracking/skate_run_policy.py'
spec=importlib.util.spec_from_file_location('_joint_progress_policy',path)
policy=importlib.util.module_from_spec(spec);sys.modules[spec.name]=policy;spec.loader.exec_module(policy)


class JointProgressMonitorTests(unittest.TestCase):
    def test_saved_stage2_joint_progress_is_not_structural_stagnation(self):
        # Derived from the ten accepted states, not fabricated physical truth.
        costs=[2861.973019809611,2827.7887002347316,2782.109230276084,
               2765.3799842930916,2741.9210336884207,2717.741429729498,
               2693.462867940395,2686.6819174856632,2673.9918017530763,
               2662.280411343362,2653.3881201474687]
        violations=[.0026665875934318898,.002558352644115911,.002311796381906174,
                    .006416810329535227,.0035148300483777402,.003388033328005783,
                    .0020786289946052475,.0020712639614948888,.002051049469029241,
                    .003862610995462429,.0021428978195171997]
        m=policy.Monitor(policy.RunPolicy(wall_seconds=1000,max_accepted=100),now=0.)
        for step,(cost,violation) in enumerate(zip(costs,violations)):
            status=m.observe(now=float(step),accepted=step>0,violation_m=violation,joint_cost=cost)
            self.assertFalse(status['stop'],(step,status))
        self.assertEqual(m.record()['progress_diagnostics'][0]['code'],
                         'structural_plateau_with_joint_progress')

    def test_joint_and_structure_plateau_remains_bounded(self):
        m=policy.Monitor(policy.RunPolicy(max_accepted=100),now=0.)
        m.observe(now=0.,violation_m=.01,joint_cost=100.)
        for step in range(1,5):
            result=m.observe(now=float(step),accepted=True,violation_m=.01,joint_cost=100.)
        self.assertEqual(result['reasons'][0]['code'],'violation_stagnation')

    def test_legacy_callers_without_joint_cost_retain_old_stop(self):
        m=policy.Monitor(policy.RunPolicy(max_accepted=100),now=0.)
        m.observe(now=0.,violation_m=.01)
        for step in range(1,5):
            result=m.observe(now=float(step),accepted=True,violation_m=.01)
        self.assertEqual(result['reasons'][0]['code'],'violation_stagnation')

    def test_budget_image_and_domain_guards_still_stop(self):
        cases=[('accepted_budget',dict(max_accepted=1),dict(image_cost=99.)),
               ('image_cost_increase',dict(max_accepted=10),dict(image_cost=110.)),
               ('collar_unknown_sections',dict(max_accepted=10),dict(quality_failures=[
                   dict(code='collar_unknown_sections',category='domain',count=1)]))]
        for expected,kwargs,sample in cases:
            with self.subTest(expected=expected):
                m=policy.Monitor(policy.RunPolicy(**kwargs),now=0.)
                m.observe(now=0.,violation_m=.01,image_cost=100.,joint_cost=100.)
                result=m.observe(now=1.,accepted=True,violation_m=.01,joint_cost=90.,**sample)
                self.assertTrue(result['stop'])
                self.assertIn(expected,[r['code'] for r in result['reasons']])


if __name__=='__main__':unittest.main()
