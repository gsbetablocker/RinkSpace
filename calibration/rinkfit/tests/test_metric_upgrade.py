"""Exactly two S3 initializer cases: shared multi-plane and single-plane."""
import unittest
import numpy as np
from rinkfit_start.metric_upgrade import (
    PlaneHomographyInput, MetricUpgradeStatus, common_projection_from_planes,
)


class MetricUpgradeTests(unittest.TestCase):
    @staticmethod
    def fixture():
        K=np.array([[900.,0.,959.5],[0.,920.,539.5],[0.,0.,1.]])
        C=np.array([.2,-.4,-3.])
        P=K @ np.c_[np.eye(3),-C]
        ice=np.array([[1.,0.,0.],[0.,1.,0.],[0.,0.,0.],[0.,0.,1.]])
        board=np.array([[1.,0.,0.],[0.,0.,.6],[0.,1.,0.],[0.,0.,1.]])
        Qs=[np.array([[1.2,.1,40.],[-.03,.9,10.],[.001,.002,1.]]),
            np.array([[.8,-.04,-20.],[.02,1.1,30.],[-.001,.001,1.]])]
        inputs=[]
        for i,(M,Q) in enumerate(zip((ice,board),Qs)):
            H=(P@M)@np.linalg.inv(Q)
            inputs.append(PlaneHomographyInput(f"plane_{i}","synthetic_clock/pts_0",M,H,Q,
                                               "known synthetic metric chart",True,"independent_metric_chart"))
        return P,inputs

    def test_multiplane_common_projection_with_explicit_metric_upgrade(self):
        expected,inputs=self.fixture()
        result=common_projection_from_planes(inputs)
        self.assertEqual(result.status,MetricUpgradeStatus.INITIALIZER_READY)
        self.assertEqual(result.numerical_nullity,1)
        self.assertEqual(result.numerical_rank,result.expected_projective_rank)
        self.assertTrue(result.initializer_only)
        self.assertFalse(result.final_physical_camera_recovered)
        P=result.require_projection()
        scale=np.sum(P*expected)/np.sum(expected*expected)
        np.testing.assert_allclose(P,scale*expected,atol=3e-12)
        self.assertLess(max(p.relative_matrix_residual for p in result.plane_replay),1e-11)
        # Projective initialization also predicts new finite 3D points, not
        # just the 3x3 input matrices, under this known synthetic metric chart.
        X=np.array([.1,.3,.2,1.])
        np.testing.assert_allclose((P@X)[:2]/(P@X)[2],(expected@X)[:2]/(expected@X)[2],atol=1e-7)

    def test_single_plane_is_degenerate_and_unknown_Q_is_not_identity(self):
        _,inputs=self.fixture()
        result=common_projection_from_planes(inputs[:1])
        self.assertEqual(result.status,MetricUpgradeStatus.DEGENERATE)
        self.assertGreater(result.numerical_nullity,1)
        self.assertIsNone(result.P)
        p=inputs[0]
        unknown=PlaneHomographyInput(p.plane_id,p.frame_key,p.M,p.H_atlas_to_native,None,"",True)
        missing=common_projection_from_planes([unknown,inputs[1]])
        self.assertEqual(missing.status,MetricUpgradeStatus.MISSING_METRIC_UPGRADE)
        self.assertIsNone(missing.P)


if __name__ == "__main__":
    unittest.main()

# IMPLEMENTED TEST COVERAGE: common same-frame P from known explicit Q and two
# physical planes; held-new-point projection; single-plane nullity; unknown-Q
# refusal.  NOT CLAIMED: actual atlas metric evidence, depth/head recovery,
# real-shot fit, final information rank or final physical acceptance.
