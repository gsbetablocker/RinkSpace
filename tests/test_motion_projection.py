"""Native physical-camera semantics; synthetic optics are not fit evidence."""
import os
import unittest
import numpy as np
from rinkspace.core.native import NativeCore
from rinkspace.compat.d78_state import decode_camera, project_native_pts


@unittest.skipUnless(os.environ.get("RINKSPACE_CORE_LIBRARY"), "explicit native library required")
class MotionProjection(unittest.TestCase):
    def setUp(self):
        self.core = NativeCore(os.environ["RINKSPACE_CORE_LIBRARY"])
        # Identity head: pan Z, tilt X. Constant pinhole, no readout skew.
        self.rig = [1.,0.,0.,0.,1.,0.,0.,0.,1., 0.,0.,1., 1.,0.,0., 0.,0.,0., 0.,0.,1.,
                    0.,0.,0., 100.,1., 50.,50.,0., 0.,0., 0.,0.,0.,0.,0.,0.,0.]
        self.controls = np.tile([0.,0.,np.log(100.)], (8,1))

    def test_pinhole_and_nonzero_lever_are_preserved(self):
        a = self.core.project_motion(self.rig,self.controls,-2.,1.,101,0.,[[1.,0.,10.]])
        self.assertEqual(a["status"].tolist(),[0])
        self.assertAlmostEqual(a["uv"][0,0],50+100/9,places=10)
        self.assertAlmostEqual(a["depth"][0],9.,places=10)
        moved = list(self.rig);moved[18]=.7
        b = self.core.project_motion(moved,self.controls,-2.,1.,101,0.,[[1.,0.,10.]])
        self.assertAlmostEqual(b["uv"][0,0],50+100*.3/9,places=10)

    def test_native_pts_difference_and_predicted_row(self):
        self.rig[-1]=.02
        self.controls[:,1]=np.linspace(-.02,.02,8)
        camera={"rig":self.rig,"first_control_time":-2.,"control_dt":1.,"image_height":101,
                "reference_pts":10**12,"first_pts":10**12,"stop_pts_exclusive":10**12+1024,
                "time_base_num":1,"time_base_den":12800}
        a=project_native_pts(self.core,camera,self.controls,np.int64(10**12+256),[[0.,.5,10.]])
        self.assertEqual(a["status"].tolist(),[0])
        self.assertLess(abs(a["root_equality_px"][0]),1e-8)
        self.assertLess(a["contraction_bound"][0],1.)
        with self.assertRaises(ValueError):
            project_native_pts(self.core,camera,self.controls,float(10**12+256),[[0.,.5,10.]])

    def test_time_support_fails_but_image_boundary_does_not_clip_projection(self):
        a=self.core.project_motion(self.rig,self.controls,-2.,1.,101,[100.,0.],[[0.,0.,10.],[0.,8.,10.]])
        self.assertEqual(a["status"].tolist(),[3,0])
        self.assertTrue(np.isnan(a["uv"][0]).all())
        self.assertAlmostEqual(a['uv'][1,1],50+800/9,places=10)

    def test_off_image_roots_keep_predicted_row_equation_for_both_readout_signs(self):
        from scipy.optimize import brentq
        rate=.1
        self.controls[:,1]=rate*(-2+np.arange(8))
        for tau in (-.02,.02):
            for y in (-8.,8.):
                with self.subTest(tau=tau,y=y):
                    self.rig[-1]=tau
                    def project(row):
                        angle=rate*tau*(row/100-.5)
                        depth=np.sin(angle)*y+10*np.cos(angle)-1
                        return np.array([50+100/depth,50+100*(np.cos(angle)*y-10*np.sin(angle))/depth])
                    row=brentq(lambda v:v-project(v)[1],-100,200,xtol=1e-12)
                    result=self.core.project_motion(self.rig,self.controls,-2.,1.,101,0.,[[1.,y,10.]])
                    self.assertEqual(result['status'].tolist(),[0])
                    self.assertTrue(row<0 or row>100)
                    np.testing.assert_allclose(result['uv'][0],project(row),rtol=0,atol=1e-8)
                    self.assertLess(abs(result['root_equality_px'][0]),1e-8)
                    self.assertLess(result['contraction_bound'][0],1.)
                    slopes=np.gradient([project(v)[1] for v in np.linspace(min(0,row),max(100,row),4097)])
                    slopes/=np.linspace(min(0,row),max(100,row),4097)[1]-min(0,row)
                    self.assertLess(np.max(abs(slopes)),result['contraction_bound'][0])

    def test_unknown_camera_extensions_cannot_be_ignored(self):
        model={"camera":{"unsupported_lens_model":1},"blocks":[]}
        with self.assertRaisesRegex(ValueError,"unsupported d78 camera fields"):
            decode_camera(model,self.core)

    def test_instant_snapshot_preserves_nonzero_installation_and_zoom_optics(self):
        self.rig[15:24]=[.2,-.3,.4, .1,-.2,.5, .07,.03,.04]
        self.rig[26:38]=[51.,49.,.02, 2.,3., .01,.02,.003,.004,.001,.002,.02]
        self.controls[:,2]=np.log(100.)+.1
        result=self.core.evaluate_motion_camera(self.rig,self.controls,-2.,1.,0.)
        self.assertEqual(result['status'].tolist(),[0])
        self.assertEqual(result['values'].shape,(1,24))
        np.testing.assert_allclose(result['rotation'][0],np.eye(3),atol=1e-15)
        np.testing.assert_allclose(result['centre'][0],[.3,-.43,.9034],atol=1e-15)
        np.testing.assert_allclose(result['values'][0,16:],
            [100*np.exp(.11),100*np.exp(.09),51.2,49.3,.012,.0034,.001,.002],rtol=1e-14)
        self.assertAlmostEqual(result['zoom_log_ratio'][0],.1,places=14)

    def test_instant_at_solved_row_time_reproduces_complete_rs_projection(self):
        self.rig[-1]=.02
        self.rig[18:24]=[.02,-.03,.8,.01,.04,-.02]
        self.rig[26:37]=[50.,50.,.01,.3,-.2,.01,.02,.001,.002,.0003,-.0002]
        self.controls[:,0]=np.linspace(-.02,.02,8)
        self.controls[:,1]=np.linspace(-.03,.03,8)
        self.controls[:,2]+=np.linspace(-.05,.05,8)
        xyz=np.array([[.2,.4,10.]])
        moving=self.core.project_motion(self.rig,self.controls,-2.,1.,101,0.,xyz)
        self.assertEqual(moving['status'].tolist(),[0])
        row_time=self.rig[-1]*(moving['row'][0]/100.-.5)
        instant=self.core.evaluate_motion_camera(self.rig,self.controls,-2.,1.,row_time)
        self.assertEqual(instant['status'].tolist(),[0])
        v=instant['values'][0]
        static=[*v[4:16],*v[16:20],0.,*v[20:24],0.]
        uv,status=self.core.project(static,xyz)
        self.assertEqual(status,[0])
        np.testing.assert_allclose(uv,moving['uv'],rtol=0.,atol=1e-12)
        centre=self.core.evaluate_motion_camera(self.rig,self.controls,-2.,1.,0.)
        self.assertGreater(np.max(abs(instant['values']-centre['values'])),1e-7)

    def test_instant_failed_time_rows_retain_nan_and_status(self):
        result=self.core.evaluate_motion_camera(self.rig,self.controls,-2.,1.,[0.,100.,np.nan])
        self.assertEqual(result['status'].tolist(),[0,3,1])
        self.assertTrue(np.isfinite(result['values'][0]).all())
        self.assertTrue(np.isnan(result['values'][1:]).all())

    def test_old_library_snapshot_capability_is_explicit(self):
        old=NativeCore.__new__(NativeCore)
        old._lib=object()
        with self.assertRaisesRegex(ValueError,'lacks instantaneous motion-camera snapshot capability'):
            old.evaluate_motion_camera(self.rig,self.controls,-2.,1.,0.)

    def test_readout_partition_certifies_unique_root_rejected_by_coarse_bound(self):
        from scipy.optimize import brentq
        from rinkspace.core.gpu import ResidentResidualProblem
        import test_g2_gpu as fixture
        rate=6.;focal=10000.;tau=.02;height=1080;xyz=np.array([[.03,.01,0.]])
        rig=np.asarray(self.rig).copy();rig[15:24]=[0,0,-3,0,0,0,0,0,0]
        rig[24:28]=[focal,1.,960.,539.5];rig[37]=tau
        controls=np.c_[rate*(-2+np.arange(8)),np.zeros(8),np.full(8,np.log(focal))]
        speed3=rate*np.linalg.norm(xyz[0]-rig[15:18]);zmin=3-speed3*tau/2
        old_coarse_bound=tau*focal*speed3*(3+abs(xyz[0,1]))/(zmin*zmin)/(height-1)
        self.assertGreater(old_coarse_bound,1.)
        # Even infinitely small windows cannot fix a bound that assigns full
        # 3-D rotational speed to the vertical and depth components.
        self.assertGreater(tau*focal*speed3*(3+abs(xyz[0,1]))/9/(height-1),1.)
        def uv_at_row(row):
            a=rate*tau*(row/(height-1)-.5);x,y=xyz[0,:2]
            return np.array([960+focal*(x*np.cos(a)-y*np.sin(a))/3,
                539.5+focal*(x*np.sin(a)+y*np.cos(a))/3])
        row=brentq(lambda v:v-uv_at_row(v)[1],0.,height-1,xtol=1e-12)
        want=uv_at_row(row);cpu=self.core.project_motion(rig,controls,-2.,1.,height,0.,xyz)
        np.testing.assert_array_equal(cpu['status'],[0]);self.assertLess(cpu['contraction_bound'][0],1-1e-9)
        np.testing.assert_allclose(cpu['uv'][0],want,atol=1e-8,rtol=0)
        f=fixture.G2GPU();f.setUp()
        kw={**f.kw,'rig':rig,'controls':controls,'image_height':height,'codes':np.zeros(1,np.int32),
            'stations':np.zeros(1),'times':np.zeros(1),'observed':want[None,:]}
        with ResidentResidualProblem(self.core,**kw) as p:
            p.configure_geometry([dict(kind=4,parameters=[],vertices=xyz.tolist())])
            gpu=p.evaluate('cuda');self.assertFalse(gpu['status'].any())
            np.testing.assert_allclose(gpu['values'][:,3:5],cpu['uv'],atol=1e-8,rtol=0)
        # Subdivision does not waive the actual optical-domain requirement.
        rig[31]=-1e5
        invalid=self.core.project_motion(rig,controls,-2.,1.,height,0.,xyz)
        np.testing.assert_array_equal(invalid['status'],[8])

    def test_directional_certificate_with_spacing_pupil_and_zoom_optics(self):
        from scipy.optimize import brentq
        rig=np.asarray(self.rig).copy();theta=.2
        def rz(a):
            return np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1.]])
        def rx(a):
            return np.array([[1.,0,0],[0,np.cos(a),-np.sin(a)],[0,np.sin(a),np.cos(a)]])
        rig[:9]=rz(theta).ravel();rig[15:24]=[.1,-.2,-3.,.05,-.15,.2,.03,.12,.04]
        rig[24:38]=[10000.,1.,960.,539.5,.01,2.,3.,.03,.004,.001,.002,.0003,-.0002,.02]
        controls=np.c_[6*(-2+np.arange(8)),.05*(-2+np.arange(8)),np.log(10000.)+.1*(-2+np.arange(8))]
        def camera_at(t):
            R=rx(.05*t)@rz(theta)@rz(6*t);h=np.cross(rig[9:12],R.T@rig[12:15]);h/=np.linalg.norm(h)
            z=.1*t;lever=rig[18:21].copy();lever[2]+=rig[22]*z+rig[23]*z*z
            return R,rig[15:18]+rig[21]*h+R.T@lever,z
        R,C,_=camera_at(0.);xyz=np.array([C+R.T@ray for ray in [[.03,.01,3.],[-.04,-.05,3.]]])
        def uv_at(row,X):
            R,C,z=camera_at(rig[37]*(row/1079-.5));ray=R@(X-C);x,y=ray[:2]/ray[2];r2=x*x+y*y
            k1=rig[31]+rig[32]*z;k2=rig[33]+rig[34]*z;p1,p2=rig[35:37];L=1+k1*r2+k2*r2*r2
            return [np.exp(np.log(10000.)+z+rig[28]/2)*(x*L+2*p1*x*y+p2*(r2+2*x*x))+rig[26]+rig[29]*z,
                np.exp(np.log(10000.)+z-rig[28]/2)*(y*L+p1*(r2+2*y*y)+2*p2*x*y)+rig[27]+rig[30]*z]
        expected=[]
        for X in xyz:
            row=brentq(lambda v:v-uv_at(v,X)[1],0,1079,xtol=1e-11);expected.append(uv_at(row,X))
        result=self.core.project_motion(rig,controls,-2.,1.,1080,0.,xyz)
        np.testing.assert_array_equal(result['status'],[0,0]);np.testing.assert_allclose(result['uv'],expected,atol=1e-7,rtol=0)
        self.assertTrue(np.all(result['contraction_bound']<1-1e-9))
        rows=np.linspace(0,1079,4097)
        for i,X in enumerate(xyz):
            slope=np.gradient([uv_at(v,X)[1] for v in rows],rows)
            self.assertLess(np.max(np.abs(slope)),result['contraction_bound'][i])


if __name__=="__main__":
    unittest.main()
