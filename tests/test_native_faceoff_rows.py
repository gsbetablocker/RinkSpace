"""Native shared A1/A4 and A2/A3 rows; synthetic geometry only."""
import os
import unittest
import numpy as np
from rinkspace.core.native import NativeCore, NativeError


@unittest.skipUnless(os.environ.get('RINKSPACE_CORE_LIBRARY'), 'explicit native library required')
class NativeFaceoffRows(unittest.TestCase):
    def setUp(self):
        self.core=NativeCore(os.environ['RINKSPACE_CORE_LIBRARY'])
        self.board=[.5,.25,.1,.12,1.85,.02]
        self.q=[-.3,.2,-.18,.1,.03,.1,.35,.65,.31,.07,-.42,-.42,-.41,-.40,.28,.28,.28,.28,
                .19,.12,.06,-.24,.24,-.43,.44,.32,99.,.11,-55.,.035,.025,.045,.03,.05,.07]

    def centres(self,q=None):
        q=self.q if q is None else q
        result=[]
        for code in range(406,410):
            kind,p=self.core.shared_geometry(self.board,q,code,options=15)
            self.assertEqual(kind,1)
            self.assertEqual(p[3],q[20])
            result.append(p[:3])
        return np.asarray(result)

    def test_pair_rows_and_transverse_axis_mirror(self):
        for a,b in [(.32,.11),(.30,.125)]:
            q=list(self.q);q[25]=a;q[27]=b
            expected=[[a,-b,0.],[a,b,0.],[-a,b,0.],[-a,-b,0.]]
            c=self.centres(q)
            np.testing.assert_array_equal(c,expected)
            self.assertEqual(c[0,1],c[3,1]);self.assertEqual(c[1,1],c[2,1])
            self.assertEqual(c[0,0],-c[3,0]);self.assertEqual(c[1,0],-c[2,0])

    def test_unused_old_slots_cannot_move_current_circles(self):
        old=self.centres();q=list(self.q);q[26]=-123.;q[28]=666.
        np.testing.assert_array_equal(self.centres(q),old)
        # Independent goal positions do not silently become a new symmetry.
        q[23]=-.42;q[24]=.46
        np.testing.assert_array_equal(self.centres(q),old)
        _,left=self.core.shared_geometry(self.board,q,403,options=15)
        _,right=self.core.shared_geometry(self.board,q,404,options=15)
        self.assertEqual(left[0],-.42);self.assertEqual(right[0],.46)

    def test_invalid_current_half_spans_and_options_fail(self):
        for index,value in [(25,0.),(25,-.1),(25,.43),(27,0.),(27,-.1)]:
            q=list(self.q);q[index]=value
            with self.assertRaises(NativeError):self.centres(q)
        with self.assertRaises(NativeError):self.core.shared_geometry(self.board,self.q,406,options=8)
        with self.assertRaises(NativeError):self.core.shared_geometry(self.board,self.q,406,options=16)

    def test_legacy_chart_and_other_families_keep_their_meaning(self):
        legacy=list(self.q);legacy[25:29]=[.08,.09,.10,.11]
        expected=[[.35,-.11,0.],[.35,.11,0.],[-.35,.10,0.],[-.35,-.10,0.]]
        actual=[]
        for code in range(406,410):
            _,p=self.core.shared_geometry(self.board,legacy,code,options=7);actual.append(p[:3])
        np.testing.assert_allclose(actual,expected,atol=1e-15)
        for code in [100,131,200,203,300,311,400,404,405,410,411,420,421,422,423]:
            a=self.core.shared_geometry(self.board,self.q,code,options=7)
            b=self.core.shared_geometry(self.board,self.q,code,options=15)
            self.assertEqual(a,b)
