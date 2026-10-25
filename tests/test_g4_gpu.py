"""Native continuous LM, with planted native measurements rather than image claims."""
import os
import unittest
import numpy as np
import test_g3_gpu as fixture
from rinkspace.core.native import NativeCore
from rinkspace.core.gpu import ResidentResidualProblem

@unittest.skipUnless(os.environ.get("RINKSPACE_G4_LIBRARY"),"explicit G4 CUDA library required")
class G4GPU(unittest.TestCase):
    def setUp(self):
        old=fixture.G3GPU();old.setUp()
        for key in ("kw","directions","W","plan"):setattr(self,key,getattr(old,key))
        self.core=NativeCore(os.environ["RINKSPACE_G4_LIBRARY"])
    run_problem=fixture.G3GPU.run_problem
    attach=fixture.G3GPU.attach

    def test_continuous_acceptance_refreshes_and_converges(self):
        history=[]
        with self.run_problem() as p:
            self.attach(p)
            result=p.solve(progress=history.append)
            self.assertTrue(result["numerically_converged"])
            self.assertGreater(result["accepted"],1)
            self.assertEqual(result["symbolic_analyses"],1)
            self.assertEqual(result["linearizations"],result["accepted"]+1)
            self.assertEqual(len(history),result["trials"])
            previous=None
            for event in history:
                m=event["summary"]
                if previous is not None:self.assertAlmostEqual(m["old_cost"],previous,places=10)
                previous=m["trial_cost"] if m["accepted"] else m["old_cost"]
                if m["accepted"]:self.assertGreater(m["actual_decrease"],0)
            final=p.linearize("cpu")["summary"]["huber_cost"]
            self.assertAlmostEqual(final,result["final_cost"],places=10)
            self.assertLess(final,1e-10)
            np.testing.assert_allclose(p.evaluate("cpu")["values"],p.evaluate("cuda")["values"],atol=1e-10)

    def test_initial_zero_gradient_has_no_candidate(self):
        with ResidentResidualProblem(self.core,**self.kw) as p:
            self.kw["observed"]=p.evaluate("cuda")["values"][:,3:5]
        with self.run_problem() as p:
            self.attach(p);history=[];r=p.solve(progress=history.append)
            self.assertEqual(r["stop_reason"],"projected_gradient_small")
            self.assertEqual(r["trials"],0);self.assertEqual(history,[])
            np.testing.assert_array_equal(p.export_state()["controls"],self.kw["controls"])

    def test_fixed_stations_preserved_across_continuous_iterations(self):
        with self.run_problem() as p:
            self.attach(p,np.array([.1,.3]),np.array([.1,.3]))
            r=p.solve();self.assertTrue(r["numerically_converged"])
            np.testing.assert_array_equal(p.export_state()["stations"],self.kw["stations"])
            self.assertLess(p.linearize("cpu")["summary"]["huber_cost"],1e-10)

    def test_rank_deficient_local_factor_retries_without_dropping_group(self):
        self.kw["stations"]=np.full(4,.1)
        self.kw["times"]=np.zeros(4)
        with ResidentResidualProblem(self.core,**{**self.kw,"controls":self.kw["controls"]+self.directions[:,:,0]}) as target:
            self.kw["observed"]=target.evaluate("cpu")["values"][:,3:5]
        plan={**self.plan,"group_offsets":np.array([0,4]),"ranks":np.array([1]),
              "weight_offsets":np.array([0,8]),"weights":np.array([1.,0.,1.,0.,1.,0.,1.,0.])}
        with self.run_problem(plan) as p:
            p.configure_solver(lower=np.zeros(2),upper=np.ones(2),
                device_budget_bytes=p.creation["free_after_alloc_bytes"],host_budget_bytes=2**30,
                initial_lambda=1e-40)
            result=p.solve()
            self.assertGreater(result["numerical_factorization_rejections"],0)
            self.assertTrue(result["numerically_converged"])
            self.assertEqual(p.linearization_info["group_count"],1)
            self.assertLess(p.linearize("cpu")["summary"]["huber_cost"],1e-10)

    def test_invalid_tolerances_do_not_update(self):
        with self.run_problem() as p:
            self.attach(p)
            with self.assertRaisesRegex(RuntimeError,"invalid G4"):
                p.solve(step_tolerance=-1)
            np.testing.assert_array_equal(p.export_state()["controls"],self.kw["controls"])

    def test_stop_keeps_accepted_state_and_continues_same_damping(self):
        with self.run_problem() as p:
            self.attach(p);history=[]
            def once(record):history.append(record);return False
            stopped=p.solve(progress=once)
            self.assertEqual(stopped["stop_reason"],"stopped_by_caller")
            self.assertFalse(stopped["numerically_converged"])
            self.assertEqual(stopped["trials"],1)
            self.assertAlmostEqual(p.linearize("cpu")["summary"]["huber_cost"],stopped["final_cost"],places=10)
            continued=[];result=p.solve(progress=continued.append)
            self.assertTrue(result["numerically_converged"])
            self.assertEqual(continued[0]["summary"]["lambda_used"],stopped["lambda_next"])
            self.assertEqual(result["symbolic_analyses"],1)
            self.assertLess(result["final_cost"],1e-10)

    def test_callback_failure_stops_before_propagating(self):
        with self.run_problem() as p:
            self.attach(p)
            def broken(record):raise ValueError("telemetry unavailable")
            with self.assertRaisesRegex(ValueError,"telemetry unavailable"):p.solve(progress=broken)
            self.assertTrue(np.isfinite(p.export_state()["controls"]).all())
            self.assertTrue(p.solve()["numerically_converged"])

    def test_stop_before_first_candidate_keeps_original_state(self):
        with self.run_problem() as p:
            self.attach(p);p.request_stop();result=p.solve()
            self.assertEqual(result["stop_reason"],"stopped_by_caller")
            self.assertEqual(result["trials"],0)
            np.testing.assert_array_equal(p.export_state()["controls"],self.kw["controls"])

    def test_camera_only_problem_needs_no_local_station_block(self):
        with ResidentResidualProblem(self.core,**self.kw) as target:
            self.kw["observed"]=target.evaluate("cpu")["values"][:,3:5]
        self.kw["controls"]=self.kw["controls"]+self.directions[:,:,0]*.2
        plan={**self.plan,"local_ids":np.full(4,-1,np.int32),"local_scales":np.empty(0,np.float64)}
        original={key:np.array(self.kw[key],copy=True) for key in ("controls","stations","observed","times")}
        with self.run_problem(plan) as p:
            p.configure_solver(lower=np.empty(0),upper=np.empty(0),
                device_budget_bytes=p.creation["free_after_alloc_bytes"],host_budget_bytes=2**30)
            result=p.solve()
            self.assertTrue(result["numerically_converged"])
            self.assertLess(result["final_cost"],1e-10)
            self.assertEqual(p.linearization_info["local_count"],0)
            np.testing.assert_array_equal(p.export_state()["stations"],self.kw["stations"])
        for key,value in original.items():np.testing.assert_array_equal(self.kw[key],value)
