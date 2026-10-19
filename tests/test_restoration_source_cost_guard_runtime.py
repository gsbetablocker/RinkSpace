import copy
from pathlib import Path
import json
import tempfile
import unittest

import numpy as np
from scipy.sparse import csr_matrix

from calibration.solver_runtime.restoration_source_cost_guard_runtime import with_restoration_source_cost_guard
from calibration.solver_runtime import install
install()
import z190_joint_source_solver as solver


def source_callback(parent, trial, **kwargs):
    return dict(accept=True, mode='truth_restoration',
        report=dict(all_source_observations_and_final_supports_retained=True,
                    maximum_actual_original_violation_fraction=.99))


class RestorationSourceCostGuardTest(unittest.TestCase):
    def test_real_uphill_truth_decisions_are_rejected_without_mutating_truth(self):
        # Captured exact accepted decisions, not fabricated trial objectives.
        run=Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/calibration/runs/solver_acceleration_20261003/okada_full9_review82_speed_candidate_20261010_101450_850')
        history=json.loads((run/'solver/iterations.json').read_text(encoding='utf8'))
        for index, predicted, ratio in [(2,335.17173845653315,-9.049956614058232),
                                        (4,1266.4537426679624,-1.2462970647839489)]:
            row=history[index]
            decision=copy.deepcopy(row['explicit_source_trial_acceptance'])
            snapshot=copy.deepcopy(decision)
            def original(parent,trial,**kwargs):return decision
            guarded=with_restoration_source_cost_guard(original)
            before,after=row['cost'],row['next_cost']
            accepts=bool(after<before-1e-10*max(1,before))
            result=guarded(np.zeros(1),np.ones(1),before_cost=before,after_cost=after,
                default_source_cost_accepts=accepts,trial_context=dict(predicted_source_reduction=predicted))
            self.assertTrue(snapshot['accept']);self.assertFalse(result['accept'])
            self.assertEqual(result['report'],snapshot['report'])
            self.assertEqual(decision,snapshot)
            self.assertEqual(result['rejection_reason'],'objective_not_lower_at_profile_precision')
            self.assertAlmostEqual((before-after)/predicted,ratio,places=12)

    def test_descent_and_original_truth_rejection_retained_with_identical_arguments(self):
        parent=np.zeros(1);trial=np.ones(1);context={'actual_step':trial}
        seen=[]
        decision=dict(accept=True,mode='truth_restoration',report={'truth':True})
        def original(p,t,**kwargs):
            seen.append((p,t,kwargs));return decision
        wrapper=with_restoration_source_cost_guard(original)
        result=wrapper(parent,trial,before_cost=2.,after_cost=1.,default_source_cost_accepts=True,trial_context=context)
        self.assertTrue(result['accept']);self.assertIs(seen[0][0],parent);self.assertIs(seen[0][1],trial)
        self.assertIs(seen[0][2]['trial_context'],context)
        decision['accept']=False
        result=wrapper(parent,trial,before_cost=2.,after_cost=1.,default_source_cost_accepts=True,trial_context=context)
        self.assertIs(result,decision)
        decision['mode']='source_objective';decision['accept']=True
        result=wrapper(parent,trial,before_cost=2.,after_cost=3.,default_source_cost_accepts=False,trial_context=context)
        self.assertIs(result,decision)

    def test_uphill_rejection_reason_keeps_native_lagged_refresh_detector(self):
        before=100.
        trials=[dict(reason='objective_not_lower_at_profile_precision',whole_source_step_fraction=a,
            evaluated_trial_cost=before+2*a) for a in (1/16,1/32)]
        report=solver._uphill_lagged_source_trials(trials,before)
        self.assertIsNotNone(report)
        self.assertEqual(report['positive_directional_cost_slopes'],[2.,2.])

    def test_fresh_non_descent_stops_without_committing_or_lm_retry_loop(self):
        # Valid full H/g descent model deliberately disagrees with real value
        # evaluation. The original fresh-J search rejects all fractions and
        # returns the same saved parent, rather than looping or committing it.
        def evaluate(q,jacobian):
            residual=np.array([1.+q[0],q[1]])
            return (residual,csr_matrix([[-1.,0.],[0.,1.]])) if jacobian else residual
        with tempfile.TemporaryDirectory() as folder:
            result=solver.solve_joint_source(evaluate,np.zeros(2),np.array([True,True]),np.ones(2),
                folder,lambda q:True,global_schur_count=1,schur_backtracking=True,
                source_trial_acceptance=with_restoration_source_cost_guard(source_callback),verbose=False)
            self.assertEqual(result.status,'no_accepted_trial')
            np.testing.assert_array_equal(result.q,np.zeros(2))
            self.assertEqual(len(result.history),1)
            self.assertEqual(result.history[0]['trial'],12)
            self.assertFalse(result.history[0]['accepted'])
            self.assertEqual(result.damping,.001)


if __name__=='__main__':unittest.main()
