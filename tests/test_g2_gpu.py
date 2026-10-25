"""Actual CUDA tests for group coupling, implicit RS and explicit diagnostics."""
import os
import unittest
import numpy as np
from rinkspace.core.native import NativeCore
from rinkspace.core.gpu import ResidentResidualProblem

@unittest.skipUnless(os.environ.get("RINKSPACE_G2_LIBRARY"),"explicit G2 CUDA library required")
class G2GPU(unittest.TestCase):
    def setUp(self):
        self.core=NativeCore(os.environ["RINKSPACE_G2_LIBRARY"])
        self.kw=dict(board=[.5,.25,.1,.1,2.,.02],
            layout=[-.3,.3,-.15,.1,.03,.1,.35,.67,.4,.07,-.4,-.4,-.4,-.4,.35,.35,.35,.35,
                    0.,.1,.07,-.25,.25,-.4,.4,.3,0.,.1,0.,.03,.03,.03,.03,.05,.07],options=0,
            rig=[1.,0.,0.,0.,1.,0.,0.,0.,1.,0.,0.,1.,1.,0.,0.,0.,0.,-3.,0.,0.,0.,0.,0.,0.,
                 100.,1.,50.,50.,0.,0.,0.,0.,0.,0.,0.,0.,0.,.02],
            controls=np.c_[np.linspace(-.01,.01,8),np.linspace(-.02,.02,8),np.full(8,np.log(100.))],
            first_control_time=-2.,control_dt=1.,image_height=101,codes=np.array([100]*4,np.int32),
            stations=np.array([.1,.3,.3,.7]),times=np.array([0.,0.,.1,.1]),
            observed=np.array([[12.,34.],[30.,40.],[32.,41.],[45.,44.]]))
        self.directions=np.zeros((8,3,2));self.directions[:,0,0]=.001;self.directions[:,2,1]=.001
        self.W=np.array([[1.,.2,.3,.4],[.2,1.,.4,-.3]])
        self.plan=dict(directions=self.directions,local_ids=np.array([0,1,1,-1]),local_scales=np.array([.001,.001]),
            group_offsets=np.array([0,2,4]),members=np.array([0,1,2,3]),ranks=np.array([2,2]),
            weight_offsets=np.array([0,8,16]),weights=np.r_[self.W.ravel(),self.W.ravel()],huber_delta=3.)

    def run_problem(self,plan=None):
        p=ResidentResidualProblem(self.core,**self.kw)
        p.configure_linearization(**(plan or self.plan))
        return p

    def test_shared_station_connects_groups_and_fixed_point_has_no_station(self):
        with self.run_problem() as p:
            self.assertEqual(p.linearization_info["component_count"],1)
            self.assertEqual(p.linearization_info["local_count"],2)
            gpu=p.linearize();cpu=p.linearize("cpu")
            self.assertEqual(gpu["timings"]["summary_d2h_bytes"],32)
            self.assertEqual(p.diagnostic_download_bytes,0)
            cb=p.export_blocks("cpu");gb=p.export_blocks()
            for key in cb:np.testing.assert_allclose(gb[key],cb[key],rtol=1e-11,atol=1e-11)
            jac=p.probe_rows(np.arange(4))
            np.testing.assert_array_equal(jac[3,:,-1],0.)
            # Explicit W mixes local coordinates; cross terms cannot be dropped.
            h=gb["H"].reshape(4,4)
            self.assertGreater(abs(h[2,3]),1e-10)
            np.testing.assert_allclose(h,h.T,atol=1e-14)

    def test_whitening_basis_rotation_preserves_group_huber_and_blocks(self):
        angle=.7;u=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        rotated={**self.plan,"weights":np.r_[(u@self.W).ravel(),(u@self.W).ravel()]}
        with self.run_problem() as a,self.run_problem(rotated) as b:
            ar=a.linearize();br=b.linearize();ab=a.export_blocks();bb=b.export_blocks()
            self.assertAlmostEqual(ar["summary"]["huber_cost"],br["summary"]["huber_cost"],places=11)
            for key in ab:np.testing.assert_allclose(ab[key],bb[key],rtol=1e-11,atol=1e-11)

    def test_dropped_rows_and_conflicting_shared_station_are_rejected(self):
        with ResidentResidualProblem(self.core,**self.kw) as p:
            with self.assertRaisesRegex(RuntimeError,"overlap"):
                p.configure_linearization(**{**self.plan,"members":np.array([0,1,1,3])})
        bad={**self.kw,"stations":np.array([.1,.3,.4,.7])}
        with ResidentResidualProblem(self.core,**bad) as p:
            with self.assertRaisesRegex(RuntimeError,"conflicting"):
                p.configure_linearization(**self.plan)

    def test_nonfinite_row_fails_whole_linearization_without_dropping_it(self):
        self.kw["times"][0]=np.nan
        with self.run_problem() as p:
            with self.assertRaisesRegex(RuntimeError,"invalid rows retained"):
                p.linearize()
            with self.assertRaises(RuntimeError):p.export_blocks()

if __name__=="__main__":unittest.main()
