"""A lens-curve extremum in an unobserved focal gap remains a physical bound."""
import os,unittest
import numpy as np
from rinkspace.core.native import NativeCore
from rinkspace.core.gpu import ResidentResidualProblem
import test_g2_gpu as fixture

@unittest.skipUnless(os.environ.get('RINKSPACE_G3_LIBRARY'),'explicit native build required')
class JoinedZoomEnvelopeTests(unittest.TestCase):
    def setUp(self):
        self.core=NativeCore(os.environ['RINKSPACE_G3_LIBRARY'])
        if not hasattr(self.core._lib,'rs_camera_joined_zoom_domain_version'):self.skipTest('joined focal-domain native build required')
        self.rig=np.zeros(38);self.rig[:9]=[1,0,0,0,0,-1,0,1,0];self.rig[9:12]=[0,0,1];self.rig[12:15]=[1,0,0]
        self.rig[15:18]=[0,-3,1.5];self.rig[18:21]=[0,-.08,.1];self.rig[24:28]=[100,1,50,540]
        self.controls=np.tile([0,.15,np.log(100.)],(8,1));self.controls[4:,0]=1.;self.controls[4:,2]+=1.
        self.base=np.r_[[-.1,.3,-1,1],np.radians([3,5,-20,5,-3,3]),np.full(9,np.inf),[1.1,1.2,5.1,5.2]]
    def check(self,rig=None,joined=True,limits=None):
        return self.core.evaluate_camera_envelope(self.rig if rig is None else rig,self.controls,0,1,self.base if limits is None else limits,joined_zoom_domain=joined)
    def test_gap_vertex_is_checked_without_inventing_cross_cut_motion(self):
        rig=self.rig.copy();rig[22:24]=[-1.,1.]
        self.assertGreaterEqual(self.check(rig,joined=False)['margin'],0.)
        got=self.check(rig);self.assertEqual(got['kind'],6);self.assertAlmostEqual(got['margin'],-.05,places=12)
        limits=self.base.copy();limits[10:13]=.001
        rig[22:24]=0.
        self.assertGreaterEqual(self.check(rig,limits=limits)['margin'],0.)
    def test_vertex_face_derivatives_match_independent_polynomial(self):
        rig=self.rig.copy();rig[22:24]=[-1.,1.]
        for index,expected in ((20,1.),(22,.5),(23,.25)):
            hi=rig.copy();lo=rig.copy();hi[index]+=1e-6;lo[index]-=1e-6
            derivative=(self.check(hi)['margin']-self.check(lo)['margin'])/2e-6
            self.assertAlmostEqual(derivative,expected,places=8)
    def test_vertex_outside_joined_support_is_not_constrained(self):
        # Vertex at z=-1 is outside exposed [0,1], so only the local endpoint
        # bounds apply even though the polynomial has a lower value elsewhere.
        rig=self.rig.copy();rig[20]=0.;rig[22:24]=[1.,.5]
        limits=self.base.copy();limits[:4]=[-.1,2.,-1.,2.]
        self.assertGreaterEqual(self.check(rig,limits=limits)['margin'],0.)
    def test_gpu_rejects_the_same_gap_violation(self):
        f=fixture.G2GPU();f.setUp();rig=self.rig.copy();rig[22:24]=[-1.,1.];controls=self.controls.copy();controls[:,0]=0.
        kw={**f.kw,'rig':rig,'controls':controls,'image_height':1080,'first_control_time':0.,'control_dt':1.,'times':np.array([1.1,1.2,5.1,5.2])}
        with ResidentResidualProblem(self.core,**kw) as problem:
            with self.assertRaises(Exception):
                problem.configure_camera_envelope(forward_bounds=self.base[:2],pupil_bounds=self.base[2:4],max_pan_tilt=self.base[4],max_tilt_deviation=self.base[5],elevation_bounds=self.base[6:8],roll_bounds=self.base[8:10],frame_time_intervals=[[1.1,1.2],[5.1,5.2]],joined_zoom_domain=True)
    def test_gpu_joint_step_obeys_gap_face(self):
        f=fixture.G2GPU();f.setUp();controls=self.controls.copy();controls[:4,2]+= .2;controls[4:,2]-=.2;controls[:,0]=0.
        rig=self.rig.copy();rig[22:24]=[-.2,.2]
        kw={**f.kw,'rig':rig,'controls':controls,'image_height':1080,'first_control_time':0.,'control_dt':1.,'times':np.array([1.1,1.2,5.1,5.2])}
        target=rig.copy();target[22:24]=[-1.,1.];target[26]+=3.
        with ResidentResidualProblem(self.core,**{**kw,'rig':target}) as p:observed=p.evaluate('cuda')['values'][:,3:5]
        with ResidentResidualProblem(self.core,**{**kw,'observed':observed}) as p:
            rd=np.zeros((27,2));rd[13,0]=-1.;rd[14,0]=1.;rd[15,1]=1.
            p.configure_camera_parameters(directions=rd,max_readout=.1)
            p.configure_camera_envelope(forward_bounds=self.base[:2],pupil_bounds=self.base[2:4],max_pan_tilt=self.base[4],max_tilt_deviation=self.base[5],elevation_bounds=self.base[6:8],roll_bounds=self.base[8:10],frame_time_intervals=[[1.1,1.2],[5.1,5.2]],joined_zoom_domain=True)
            p.configure_pixel_linearization(directions=np.zeros((8,3,2)),local_ids=np.full(4,-1,np.int32),local_scales=np.zeros(0))
            p.configure_solver(lower=np.zeros(0),upper=np.zeros(0),initial_lambda=1e-4,device_budget_bytes=p.creation['free_after_alloc_bytes'],host_budget_bytes=2**30)
            result=p.lm_step()['summary'];self.assertTrue(result['accepted'],result)
            state=p.export_state();check=self.core.evaluate_camera_envelope(state['rig'],state['controls'],0,1,self.base,joined_zoom_domain=True)
            self.assertGreaterEqual(check['margin'],-4e-14,check);self.assertLessEqual(state['rig'][23],.8+1e-10)
            self.assertNotEqual(state['rig'][26],rig[26]);self.assertLess(result['trial_cost'],result['old_cost'])
