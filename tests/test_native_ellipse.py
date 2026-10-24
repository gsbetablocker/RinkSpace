"""Native semi-ellipse/finite support regressions; synthetic, never fit evidence."""
import os
import unittest
import numpy as np
from rinkspace.core.native import NativeCore, NativeError


@unittest.skipUnless(os.environ.get('RINKSPACE_CORE_LIBRARY'), 'explicit native library required')
class NativeEllipse(unittest.TestCase):
    def setUp(self):
        self.core=NativeCore(os.environ['RINKSPACE_CORE_LIBRARY'])
        self.board=[.5,.25,.1,.1,2.,.02]
        self.prefix=[-.3,.2,-.15,.1,.03,.1,.35,.65, .31,.07,-.42,-.43,-.41,-.44,.28,.28,.28,.28,
                     .19,.12,.06,-.24,.24, -.42,.42,.08,.09,.1,.11]
        self.legacy=self.prefix+[.025,.03,.05,.07]
        self.extended=self.prefix+[.035,.025,.045,.03,.05,.07]

    def test_separate_axes_and_rotated_endpoints(self):
        yaw=self.prefix[18]
        for side,code in enumerate([410,411]):
            kind,p=self.core.shared_geometry(self.board,self.extended,code)
            self.assertEqual(kind,6)
            self.assertEqual(len(p),8)
            points=np.asarray(self.core.sample_path(self.board,kind,p,[],[0.,.5,1.]))
            rotation=np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]])
            local=(points[:,:2]-p[:2])@rotation
            expected=np.array([[0., -p[4]],[p[3],0.],[0.,p[4]]]) if side==0 else np.array([[0.,p[4]],[-p[3],0.],[0.,-p[4]]])
            np.testing.assert_allclose(local,expected,atol=1e-14)
            changed=list(self.extended);changed[29+side*2]*=1.25
            kind2,p2=self.core.shared_geometry(self.board,changed,code)
            altered=np.asarray(self.core.sample_path(self.board,kind2,p2,[],[0.,.5,1.]))
            np.testing.assert_allclose(altered[[0,2]],points[[0,2]],atol=1e-14)
            self.assertGreater(np.linalg.norm(altered[1]-points[1]),.001)

    def test_old_circle_and_other_shared_geometry_preserved(self):
        old_kind,old=self.core.shared_geometry(self.board,self.legacy,410)
        self.assertEqual(old_kind,1)
        self.assertEqual(len(old),6)
        for code in [100,131,200,203,300,311,400,405,409,420,421,422,423]:
            a=self.core.shared_geometry(self.board,self.legacy,code)
            b=self.core.shared_geometry(self.board,self.extended,code)
            self.assertEqual(a,b)

    def test_invalid_axes_and_finite_arc_support_fail(self):
        bad=list(self.extended);bad[30]=0.
        with self.assertRaises(NativeError):self.core.shared_geometry(self.board,bad,410)
        kind,p=self.core.shared_geometry(self.board,self.extended,410)
        with self.assertRaises(NativeError):self.core.sample_path(self.board,kind,p,[],[-.01])


if __name__=='__main__':
    unittest.main()