"""A print square is regular in metric (s,z), not in unequal UV axes."""
import ctypes as c
import os
import unittest
import numpy as np
from rinkspace.core.native import NativeCore, NativeError


@unittest.skipUnless(os.environ.get('RINKSPACE_CORE_LIBRARY'),'explicit native library required')
class NativePrintSquare(unittest.TestCase):
    def setUp(self):
        self.core=NativeCore(os.environ['RINKSPACE_CORE_LIBRARY'])

    def test_metric_square_survives_changed_panel_aspect_and_rotation(self):
        for width,height in [(.08,.012),(.16,.007),(.03,.025)]:
            for rotation in [0.,.37,3.141592653589793]:
                with self.subTest(width=width,height=height,rotation=rotation):
                    parameters=[1.28,.012,rotation,width,height]
                    uv=np.asarray(self.core.print_square_vertices(parameters,[.135,.5],.8))
                    self.assertEqual(uv.shape,(5,2))
                    np.testing.assert_array_equal(uv[0],uv[-1])
                    sz=np.asarray(self.core.sample_print_chart(parameters,uv,[0,.25,.5,.75,1.]))
                    sides=np.diff(sz,axis=0);lengths=np.linalg.norm(sides,axis=1)
                    np.testing.assert_allclose(lengths,.8*height,rtol=2e-13,atol=1e-16)
                    np.testing.assert_allclose(np.sum(sides*np.roll(sides,1,axis=0),axis=1),0.,atol=1e-16)

    def test_curved_course_uses_same_existing_kind5(self):
        board=[.5,.2443,.1085,.123,1.844,.0167]
        p=[1.355,.014,3.141592653589793,.0826,.0121]
        uv=self.core.print_square_vertices(p,[.135,.5],.8)
        t=np.linspace(0.,1.,401)
        sz=np.asarray(self.core.sample_print_chart(p,uv,t))
        xyz=np.asarray(self.core.sample_path(board,5,p,uv,t))
        np.testing.assert_array_equal(xyz,self.core.board_points(board,sz[:,0],sz[:,1]))
        # The top edge follows the real curved course, rather than its chord.
        edge=xyz[:101];delta=edge[-1]-edge[0]
        self.assertGreater(np.max(np.linalg.norm(np.cross(edge-edge[0],delta),axis=1)/np.linalg.norm(delta)),1e-7)

    def test_invalid_parameters_fail_and_leave_no_successful_vertices(self):
        for side in [0.,-1.,float('nan'),float('inf')]:
            with self.subTest(side=side):
                with self.assertRaises(NativeError):
                    self.core.print_square_vertices([0.,0.,0.,.08,.012],[.13,.5],side)
        for p,centre in [([0.,0.,0.,0.,.012],[.13,.5]),
                         ([0.,0.,0.,.08,-.012],[.13,.5]),
                         ([0.,0.,0.,.08,.012],[.13,float('nan')]),
                         ([0.,0.,0.,.08],[.13,.5]),
                         ([0.,0.,0.,.08,.012],[.13])]:
            with self.assertRaises(NativeError):self.core.print_square_vertices(p,centre,.8)
        p=(c.c_double*5)(0,0,0,.08,.012);centre=(c.c_double*2)(.13,.5);out=(c.c_double*10)()
        self.assertNotEqual(self.core._lib.rs_print_square_vertices(p,5,centre,2,0,out,10),0)
        self.assertTrue(np.isnan(list(out)).all())


class SquareCapability(unittest.TestCase):
    def test_old_library_cannot_silently_fall_back(self):
        core=object.__new__(NativeCore);core._lib=object()
        with self.assertRaisesRegex(ValueError,'lacks unfolded print-square capability'):
            core.print_square_vertices([],[],1.)
