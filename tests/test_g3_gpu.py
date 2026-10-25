"""G3 CUDA tests: full-system reference, finite bounds, and rejected-state isolation."""
import os
import unittest
import numpy as np
from rinkspace.core.native import NativeCore
from rinkspace.core.gpu import ResidentResidualProblem
import test_g2_gpu as g2_fixture

@unittest.skipUnless(os.environ.get("RINKSPACE_G3_LIBRARY"),"explicit G3 CUDA library required")
class G3GPU(unittest.TestCase):
    def setUp(self):
        old=g2_fixture.G2GPU();old.setUp()
        for key in ("kw","directions","W","plan"):setattr(self,key,getattr(old,key))
        self.core=NativeCore(os.environ["RINKSPACE_G3_LIBRARY"])
        # Test-only planted source measurements from the same native projector.
        shifted=self.kw["controls"]+self.directions[:,:,0]
        with ResidentResidualProblem(self.core,**{**self.kw,"controls":shifted}) as target:
            self.kw["observed"]=target.evaluate("cpu")["values"][:,3:5]
    run_problem=g2_fixture.G2GPU.run_problem

    def attach(self,p,lo=None,hi=None):
        n=int(p.linearization_info["local_count"])
        return p.configure_solver(lower=np.zeros(n) if lo is None else lo,
            upper=np.ones(n) if hi is None else hi,device_budget_bytes=p.creation["free_after_alloc_bytes"],
            host_budget_bytes=2**30)

    def test_real_gpu_step_matches_independent_full_damped_system(self):
        with self.run_problem() as p:
            self.attach(p);before=p.linearize();blocks=p.export_blocks()
            old_controls=self.kw["controls"].copy()
            n0=p.diagnostic_download_bytes;result=p.lm_step()
            self.assertEqual(p.diagnostic_download_bytes,n0)
            self.assertTrue(result["summary"]["accepted"])
            system=p.export_step_system();lam=result["summary"]["lambda_used"]
            H=blocks["H"].reshape(4,4);g=blocks["g"]
            expected=np.linalg.solve(H+lam*np.eye(4),-g)
            np.testing.assert_allclose(system["step"],expected,rtol=1e-9,atol=1e-9)
            A=H[:2,:2]+lam*np.eye(2);C=H[2:,2:]+lam*np.eye(2);B=H[:2,2:]
            np.testing.assert_allclose(system["S"],A-B@np.linalg.solve(C,B.T),rtol=1e-10,atol=1e-12)
            applied=system["applied"]
            predicted=-g@applied-.5*applied@H@applied
            self.assertAlmostEqual(predicted,result["summary"]["predicted_decrease"],places=11)
            current=p.export_state();trial=p.export_state(trial=True)
            for key in current:np.testing.assert_array_equal(current[key],trial[key])
            self.assertFalse(np.array_equal(current["controls"],old_controls))
            # Explicit CPU snapshot must synchronize the newly accepted native state.
            np.testing.assert_allclose(p.evaluate("cpu")["values"],p.evaluate("cuda")["values"],rtol=1e-11,atol=1e-10)
            actual=p.linearize("cpu")["summary"]["huber_cost"]
            self.assertAlmostEqual(actual,result["summary"]["trial_cost"],places=10)
            self.assertLess(actual,before["summary"]["huber_cost"])

    def test_zero_width_original_support_removes_only_local_coordinates(self):
        with self.run_problem() as p:
            self.attach(p,np.array([.1,.3]),np.array([.1,.3]))
            p.linearize();b=p.export_blocks();r=p.lm_step();s=p.export_step_system()
            self.assertEqual(r["summary"]["blocked_station_count"],2)
            np.testing.assert_array_equal(s["step"][2:],0.)
            h=b["H"].reshape(4,4)
            np.testing.assert_allclose(s["step"][:2],np.linalg.solve(h[:2,:2]+r["summary"]["lambda_used"]*np.eye(2),-b["g"][:2]),rtol=1e-10)
            np.testing.assert_array_equal(p.export_state()["stations"],self.kw["stations"])

    def test_domain_rejection_keeps_accepted_state_and_reuses_symbolic_analysis(self):
        with ResidentResidualProblem(self.core,**self.kw) as base:
            uv=base.evaluate("cpu")["values"][:,3:5]
        self.kw["observed"]=50.+1e5*(uv-50.)
        dirs=np.zeros((8,3,1));dirs[:,2,0]=.001
        plan={**self.plan,"directions":dirs,"local_ids":np.full(4,-1,dtype=np.int32),"local_scales":np.zeros(0)}
        with self.run_problem(plan) as p:
            self.attach(p)
            first=p.lm_step();state=p.export_state()
            self.assertFalse(first["summary"]["accepted"])
            self.assertEqual(first["summary"]["reason"],"trial_domain_failure")
            np.testing.assert_array_equal(state["controls"],self.kw["controls"])
            np.testing.assert_array_equal(state["stations"],self.kw["stations"])
            second=p.lm_step()
            self.assertFalse(second["summary"]["accepted"])
            self.assertEqual(second["summary"]["analysis_count"],1)
            self.assertEqual(second["timings"]["linearization_wall_ms"],0.)
            self.assertEqual(second["summary"]["lambda_used"],first["summary"]["lambda_next"])

    def test_support_outside_current_native_domain_is_rejected(self):
        with self.run_problem() as p:
            with self.assertRaisesRegex(RuntimeError,"outside declared finite support"):
                self.attach(p,np.array([.2,0.]),np.ones(2))

if __name__=="__main__":unittest.main()
