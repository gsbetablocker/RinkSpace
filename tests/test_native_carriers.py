"""Native rink-aligned finite carriers; not source-fit evidence."""
import os
import unittest
import numpy as np
from rinkspace.core.native import NativeCore, NativeError


@unittest.skipUnless(os.environ.get('RINKSPACE_CORE_LIBRARY'), 'explicit native library required')
class NativeCarriers(unittest.TestCase):
    def setUp(self):
        self.core=NativeCore(os.environ['RINKSPACE_CORE_LIBRARY'])
        self.board=[.5,.25,.1,.12,1.85,.02]
        self.layout=[-.3,.2,-.18,.1,.03,.1,.35,.65,.31,.07,-.42,-.43,-.41,-.44,.28,.28,.28,.28,
                     .19,.12,.06,-.24,.24,-.43,.44,.08,.09,.1,.11,.035,.025,.045,.03,.05,.07]

    def membership(self,xyz):
        x,y,z=xyz;a,b,rx,ry,e,_=self.board
        u=max(0.,(abs(x)-(a-rx))/rx);v=max(0.,(abs(y)-(b-ry))/ry)
        return u**e+v**e

    def test_all_five_are_short_side_parallel_and_boundary_attached(self):
        expected=[0.,-.12,.12,-.43,.44]
        for code,x in zip(range(400,405),expected):
            kind,p=self.core.shared_geometry(self.board,self.layout,code,options=3)
            self.assertEqual(kind,0)
            xyz=np.asarray(p).reshape(2,3)
            self.assertEqual(xyz[0,0],x);self.assertEqual(xyz[1,0],x)
            self.assertGreater(xyz[0,1],0);self.assertLess(xyz[1,1],0)
            np.testing.assert_allclose(xyz[:,2],0,atol=0)
            for point in xyz:self.assertAlmostEqual(self.membership(point),1.,places=13)
            if abs(x)>.4:self.assertLess(xyz[0,1],self.board[1])

    def test_ignored_yaw_and_old_y_support_cannot_change_current_geometry(self):
        altered=list(self.layout);altered[18]=-.7;altered[21]=.8;altered[22]=-.9
        for code in range(400,405):
            a=self.core.shared_geometry(self.board,self.layout,code,options=3)
            b=self.core.shared_geometry(self.board,altered,code,options=3)
            self.assertEqual(a,b)
        for code in [406,407,408,409,410,411,420,421,422,423]:
            a=self.core.shared_geometry(self.board,self.layout,code,options=3)
            b=self.core.shared_geometry(self.board,altered,code,options=3)
            self.assertEqual(a,b)
        for code in [410,411]:
            kind,p=self.core.shared_geometry(self.board,self.layout,code,options=3)
            self.assertEqual(kind,6);self.assertEqual(p[5],0.)

    def test_near_end_wall_and_corner_join_use_the_same_continuous_shape(self):
        for x in [.399999999,.4,.400000001,.47,.499,.5-1e-10]:
            q=list(self.layout);q[24]=x
            _,p=self.core.shared_geometry(self.board,q,404,options=2)
            xyz=np.asarray(p).reshape(2,3)
            for point in xyz:self.assertAlmostEqual(self.membership(point),1.,places=12)
        for x in [.5,.50001]:
            q=list(self.layout);q[24]=x
            with self.assertRaises(NativeError):self.core.shared_geometry(self.board,q,404,options=2)

    def test_lane_left_support_cannot_cross_left_goal_line(self):
        q=list(self.layout);goal=q[23]
        q[10:14]=[goal,goal+.01,goal+.02,goal+.03]
        for row in range(4):
            _,p=self.core.shared_geometry(self.board,q,300+row*3,options=7)
            self.assertEqual(p[0],q[10+row])
        outside=list(q);outside[12]=goal-1e-10
        for code in range(300,312):
            with self.assertRaises(NativeError):self.core.shared_geometry(self.board,outside,code,options=7)
        q[0]=goal
        _,p=self.core.shared_geometry(self.board,q,200,options=7)
        self.assertEqual(p[0],goal)
        q[0]=goal-1e-10
        with self.assertRaises(NativeError):self.core.shared_geometry(self.board,q,200,options=7)
        with self.assertRaises(NativeError):self.core.shared_geometry(self.board,self.layout,300,options=4)

    def test_legacy_options_preserve_original_orientation_and_support(self):
        for option in [0,1]:
            _,p=self.core.shared_geometry(self.board,self.layout,400,options=option)
            p=np.asarray(p).reshape(2,3)
            yaw=self.layout[18];c,s=np.cos(yaw),np.sin(yaw)
            expected=np.array([[-s*self.layout[22],c*self.layout[22],0.],[-s*self.layout[21],c*self.layout[21],0.]])
            np.testing.assert_allclose(p,expected,atol=1e-15)
        with self.assertRaises(NativeError):self.core.shared_geometry(self.board,self.layout,400,options=16)

    def test_connection_must_pass_R_and_stop_before_N_in_native(self):
        for row in range(4):
            for endpoint in [self.layout[1]-.01,self.layout[1],self.layout[8],self.layout[8]+.01]:
                q=list(self.layout);q[14+row]=endpoint
                with self.assertRaises(NativeError) as exc:
                    self.core.shared_geometry(self.board,q,200+row,options=7)
                self.assertEqual(exc.exception.status,3)

    def test_valid_connection_supports_remain_independent_and_legacy_replay_is_preserved(self):
        q=list(self.layout);q[14:18]=[.21,.23,.26,.30]
        for row,end in enumerate(q[14:18]):
            _,p=self.core.shared_geometry(self.board,q,200+row,options=7)
            self.assertEqual(p[3],end)
            self.assertNotEqual(p[3],q[8])
        q[15]=q[8]+.03
        _,p=self.core.shared_geometry(self.board,q,201,options=0)
        self.assertEqual(p[3],q[15])
