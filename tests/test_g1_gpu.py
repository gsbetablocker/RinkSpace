"""GPU boundary regression: invalid inputs stay invalid on the actual device."""
import os
import unittest
import numpy as np
from rinkspace.core.native import NativeCore
from rinkspace.core.gpu import ResidentResidualProblem

@unittest.skipUnless(os.environ.get("RINKSPACE_G1_LIBRARY"),"explicit CUDA native library required")
class G1GPU(unittest.TestCase):
    def setUp(self):
        self.core=NativeCore(os.environ["RINKSPACE_G1_LIBRARY"])
        self.kw=dict(board=[.5,.25,.1,.1,2.,.02],
            layout=[-.3,.3,-.15,.1,.03,.1,.35,.67,
                    .4,.07,-.4,-.4,-.4,-.4,.35,.35,.35,.35,
                    0.,.1,.07,-.25,.25,-.4,.4,.3,0.,.1,0.,.03,.03,.03,.03,.05,.07],
            options=0,
            rig=[1.,0.,0.,0.,1.,0.,0.,0.,1.,0.,0.,1.,1.,0.,0.,
                 0.,0.,-3.,0.,0.,0.,0.,0.,0.,100.,1.,50.,50.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
            controls=np.tile([0.,0.,np.log(100.)],(8,1)),first_control_time=-2.,
            control_dt=1.,image_height=101,codes=np.array([100],np.int32),
            stations=np.array([.25]),times=np.array([0.]),observed=np.array([[12.,34.]]))

    def test_actual_gpu_finite_nonfinite_support_and_unsupported_rows(self):
        n=9
        self.kw.update(codes=np.array([100]*8+[999]),stations=np.array([.25,np.nan,np.inf,-.1,1.1,.25,.25,.25,.25]),
            times=np.array([0.,0.,0.,0.,0.,np.inf,100.,0.,0.]),
            observed=np.array([[12.,34.]]*7+[[np.nan,34.],[12.,34.]]))
        with ResidentResidualProblem(self.core,**self.kw) as p:
            cpu=p.evaluate("cpu");gpu=p.evaluate("cuda")
            np.testing.assert_array_equal(cpu["status"],[0,1,1,3,3,1,3,1,11])
            np.testing.assert_array_equal(gpu["status"],cpu["status"])
            self.assertTrue(np.isnan(gpu["values"][1:]).all())
            np.testing.assert_allclose(gpu["values"][0],cpu["values"][0],rtol=0,atol=1e-9)

    def test_real_residual_sign_and_immutable_resident_inputs(self):
        with ResidentResidualProblem(self.core,**self.kw) as p:
            first=p.evaluate("cuda")
            self.kw["observed"][:]=1000.;self.kw["stations"][:]=.75
            second=p.evaluate("cuda")
            np.testing.assert_array_equal(first["values"],second["values"])
            np.testing.assert_allclose(first["values"][0,9:11],first["values"][0,3:5]-[12.,34.],rtol=0,atol=1e-12)
            self.assertGreater(p.creation["owned_device_bytes"],0)
            self.assertGreater(first["timings"]["kernel_event_ms"],0)

    def test_wrong_row_binding_rejected_and_closed_handle_not_used(self):
        bad={**self.kw,"observed":np.zeros((2,2))}
        with self.assertRaisesRegex(ValueError,"binding"):ResidentResidualProblem(self.core,**bad)
        p=ResidentResidualProblem(self.core,**self.kw);p.close();p.close()
        with self.assertRaisesRegex(RuntimeError,"closed"):p.evaluate("cuda")

if __name__=="__main__":unittest.main()
