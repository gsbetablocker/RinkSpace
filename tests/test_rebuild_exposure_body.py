import os
import unittest
from pathlib import Path
import numpy as np
from calibration.program_tracking.body_tracking.rebuild_exposure_body import moving_projection,ExposureArticulatedSplineProblem


class LinearCamera:
    def project(self,world,frames):
        return world[:,:,:2]*20+np.array([960.,540.])
    def with_jacobian(self,world,frames):
        j=np.broadcast_to(np.array([[20.,0,0],[0,20.,0]]),(*world.shape[:2],2,3)).copy()
        return self.project(world,frames),j


class ExposureProjectionTests(unittest.TestCase):
    def test_linear_motion_has_the_closed_form_row_solution(self):
        tau=.02;y0=1.;vy=30.;nodes=np.zeros((1,3,1,3));nodes[0,:,0,1]=y0+vy*np.array([-tau/2,0,tau/2])
        result=moving_projection(LinearCamera(),nodes,np.array([0]),jacobian=True)
        expected=(540+20*(y0-vy*tau/2))/(1-20*vy*tau/1079)
        self.assertAlmostEqual(result['uv'][0,0,1],expected,places=7)
        for node in range(3):
            shifted=nodes.copy();shifted[0,node,0,1]+=1e-5
            a=moving_projection(LinearCamera(),shifted,np.array([0]))['uv']
            shifted[0,node,0,1]-=2e-5
            b=moving_projection(LinearCamera(),shifted,np.array([0]))['uv']
            np.testing.assert_allclose((a-b)[0,0]/2e-5,result['pixel_jac_nodes'][0,0,:,node,1],rtol=1e-6,atol=1e-6)
    def test_out_of_image_row_has_explicit_unknown_flag(self):
        nodes=np.zeros((1,3,1,3));nodes[:,:,:,1]=50
        result=moving_projection(LinearCamera(),nodes,np.array([0]))
        self.assertFalse(result['inside_exposure_rows'][0,0])


@unittest.skipUnless(os.environ.get('RINKSPACE_S01_BODY_ROOT'),'requires explicitly selected local body experiment')
class RealBodyDerivativeTests(unittest.TestCase):
    def test_full_native_body_contact_derivative_and_visibility(self):
        root=Path(os.environ['RINKSPACE_S01_BODY_ROOT']);p=ExposureArticulatedSplineProblem(root,'pose_contact')
        warm=root/'pose_contact/result.npz'
        x=np.load(warm)['x'] if warm.exists() else p.x0.copy()
        rng=np.random.default_rng(8);x=x+rng.normal(size=len(x))*.001
        a=p.evaluate(x,jacobian=True);direction=rng.normal(size=len(x));direction/=np.linalg.norm(direction)
        h=1e-5;fd=(p.fun(x+h*direction)-p.fun(x-h*direction))/(2*h);analytical=a['jac']@direction
        self.assertLess(np.linalg.norm(fd-analytical)/np.linalg.norm(fd),1e-5)
        reviewed=p.p['pts']<p.meta['visibility_review']['lower_body_unreliable_before_pts']
        self.assertTrue(p.p['lower_body_inferred'][reviewed].all())
        self.assertFalse(p.p['source_qualified'][reviewed,11:17].any())

if __name__=='__main__':unittest.main()
