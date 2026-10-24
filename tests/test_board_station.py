"""Real native boundary semantics: joins are not bounds on latent stations.

These tests concern numerical/contract correctness, not fitted source accuracy.
"""
import os
import unittest
import numpy as np
from rinkspace.core.native import NativeCore, NativeError


@unittest.skipUnless(os.environ.get('RINKSPACE_CORE_LIBRARY'), 'explicit native library required')
class BoardStation(unittest.TestCase):
    def setUp(self):
        self.core=NativeCore(os.environ['RINKSPACE_CORE_LIBRARY'])
        self.board=[.5,.25,.1,.12,1.85,.02]

    def test_every_straight_corner_join_and_closed_seam_can_be_crossed(self):
        breaks=self.core.board_course_chart(self.board)
        eps=1e-7
        for join in breaks:
            s=np.asarray([join-eps,join,join+eps])
            for h in (0.,1.):
                xyz=np.asarray(self.core.sample_path(self.board,7,[h],[],s))
                expected=np.asarray(self.core.board_points(self.board,s,np.full(3,h*self.board[5])))
                np.testing.assert_allclose(xyz,expected,atol=1e-13)
                self.assertLess(np.linalg.norm(xyz[2]-xyz[0]),2.01*eps)
                a=xyz[1]-xyz[0];b=xyz[2]-xyz[1]
                self.assertLess(np.linalg.norm(a/np.linalg.norm(a)-b/np.linalg.norm(b)),1e-3)

    def test_current_feature_import_preserves_identity_and_uv_metadata(self):
        from copy import deepcopy
        from rinkspace.domain.board_station import normalize_boundary_feature, sample_boundary
        feature={'id':'old/source_boundary_17','kind':'finite_curve','surface':'board','identity':'confirmed',
                 'measurement':'centreline','finite_support':[0.,1.], 'branch':'segment0',
                 'provenance':{'source_point_ids':['p1','p2'],'uv':[[10.,20.],[11.,21.]],'pts':[256,512],'noise_group_id':'W17'}}
        original=deepcopy(feature)
        values=np.array([.98,.98,1.]);indices=np.array([91,91,92],np.int64)
        imported=normalize_boundary_feature(feature,self.board,0,0.,values,self.core,local_indices=indices)
        self.assertEqual(feature,original)
        self.assertEqual(imported['feature']['id'],feature['id'])
        self.assertEqual(imported['feature']['provenance'],feature['provenance'])
        self.assertFalse(imported['feature']['station_domain']['hard_segment_bounds'])
        np.testing.assert_array_equal(imported['local_indices'],indices)
        np.testing.assert_array_equal(imported['support_kind'],[1,1,1])
        np.testing.assert_array_equal(imported['support'],np.zeros((3,2)))
        migrated=sample_boundary(imported['feature'],self.board,imported['global_station'],self.core)
        old=self.core.sample_path(self.board,3,[0.,0.],[],values)
        np.testing.assert_allclose(migrated,old,atol=1e-13)
        following=sample_boundary(imported['feature'],self.board,imported['global_station']+1e-3,self.core)
        self.assertTrue(np.isfinite(following).all())

    def test_unwrapped_station_does_not_clamp_or_wrap_its_state(self):
        P=self.core.board_perimeter(self.board)
        s=np.asarray([-.03,.03,P+.03,3*P+.03])
        original=s.copy()
        xyz=np.asarray(self.core.sample_path(self.board,7,[0.],[],s))
        np.testing.assert_array_equal(s,original)
        np.testing.assert_allclose(xyz[1:],np.tile(xyz[1],(3,1)),atol=1e-13)
        self.assertGreater(np.linalg.norm(xyz[0]-xyz[1]),.02)

    def test_legacy_segment_import_preserves_geometry_then_has_no_segment_bounds(self):
        for segment in (-2,-1,*range(8)):
            local=[0.,.17,.5,.83,1.]
            stations=self.core.board_stations_from_legacy(self.board,segment,local)
            old=np.asarray(self.core.sample_path(self.board,3,[segment,1.],[],local))
            new=np.asarray(self.core.sample_path(self.board,7,[1.],[],stations))
            np.testing.assert_allclose(new,old,atol=1e-12)
        old_end=self.core.board_stations_from_legacy(self.board,0,[1.])[0]
        following=self.core.sample_path(self.board,7,[0.],[],[old_end+1e-5])
        self.assertTrue(np.isfinite(following).all())
        with self.assertRaises(NativeError):
            self.core.sample_path(self.board,3,[0.,0.],[],[1.001])
        with self.assertRaises(NativeError):
            self.core.sample_path(self.board,7,[1.1],[],[old_end])


@unittest.skipUnless(os.environ.get('RINKSPACE_CORE_LIBRARY'), 'explicit native library required')
class CentredCurling(unittest.TestCase):
    def setUp(self):
        self.core=NativeCore(os.environ['RINKSPACE_CORE_LIBRARY'])
        self.board=[.5,.25,.1,.1,2.,.02]
        self.layout=[-.3,.2,-.18,.1,.03,.1,.35,.65,.31,.07,-.42,-.43,-.41,-.44,.28,.28,.28,.28,
                     .19,.12,.06,-.24,.24,-.42,.42,.08,.09,.1,.11,.025,.03,.05,.07]

    def test_all_circles_lanes_and_connections_share_derived_centred_rows(self):
        for pitch in (.1,.105):
            q=list(self.layout);q[3]=pitch
            for row in range(4):
                y=(-1.5+row)*pitch
                for side in range(2):
                    for ring in range(4):
                        _,p=self.core.shared_geometry(self.board,q,100+(row+4*side)*4+ring,options=1)
                        self.assertAlmostEqual(p[1],y,places=14)
                _,p=self.core.shared_geometry(self.board,q,200+row,options=1)
                self.assertAlmostEqual(p[1],y,places=14);self.assertAlmostEqual(p[4],y,places=14)
                _,right=self.core.shared_geometry(self.board,q,301+row*3,options=1)
                self.assertAlmostEqual((right[1]+right[4])/2,y,places=14)
            _,lo=self.core.shared_geometry(self.board,q,103,options=1)
            _,hi=self.core.shared_geometry(self.board,q,115,options=1)
            self.assertAlmostEqual(lo[1]+self.board[1],self.board[1]-hi[1],places=14)
            q[2]=.4
            _,ignored=self.core.shared_geometry(self.board,q,103,options=1)
            self.assertEqual(ignored,lo)

    def test_legacy_and_unknown_options_remain_explicit(self):
        _,old=self.core.shared_geometry(self.board,self.layout,103)
        self.assertEqual(old[1],self.layout[2])
        with self.assertRaises(NativeError):
            self.core.shared_geometry(self.board,self.layout,103,options=16)