import io
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from calibration.program_tracking.body_tracking.stage_quality import StageQualityTrace,body_groups
from calibration.program_tracking.body_tracking.skate_integration import SupportJointWindow,state_from_packet


class SupportStageQualityTests(unittest.TestCase):
    def test_seed_does_not_replace_original_and_zero_steps_is_no_update(self):
        with tempfile.TemporaryDirectory() as d:
            original={'body':{'median_px':2.}}
            trace=StageQualityTrace(Path(d)/'quality.json',{'source':'fixed'},original,lambda x,v:v)
            original['body']['median_px']=999.
            trace.begin(None,{'body':{'median_px':12.}})
            trace.finish(None,{'body':{'median_px':12.}},accepted_steps=0)
            record=trace.record()
            self.assertEqual(record['original_A']['body']['median_px'],2.)
            self.assertEqual(record['stage_update'],'no_update')
            self.assertFalse(record['accuracy_accepted'])
            self.assertEqual(record['delivery_candidate']['status'],'requires_visual_review')

    def test_joint_correction_replays_and_has_C2_attachment(self):
        ref=type('Reference',(),{'p':{'pts':np.array([76*12800])}})()
        ws=SupportJointWindow(ref,(87.5,92.5),foot=1)
        self.assertTrue(np.array_equal(ws.active[:6],np.arange(6)))
        self.assertTrue(set([9,10,11,18,19,20,27,28,29]).issubset(ws.active))
        for order in (0,1,2):
            np.testing.assert_allclose(ws.basis([11.5,16.5],order),0.,atol=1e-12)
        packet=dict(state_kind=np.array(ws.state_kind),seconds=np.array(ws.seconds),knots=ws.knots,active=ws.active)
        restored=state_from_packet(ref,packet)
        np.testing.assert_array_equal(restored.active,ws.active)
        np.testing.assert_array_equal(restored.basis([12.,14.,16.]),ws.basis([12.,14.,16.]))

    def test_pooled_error_reports_worst_exposure_and_missing_coverage(self):
        predicted=np.zeros((2,17,2));observed=predicted.copy();observed[1,16]=[3.,4.]
        use=np.zeros((2,17),bool);use[:,16]=True
        result=body_groups(predicted,observed,use,np.array([1,2]))
        self.assertEqual(result['body']['worst_pts'],2)
        self.assertEqual(result['body_foot_0']['count'],0)
        self.assertEqual(result['body_foot_1']['worst_px'],5.)


if __name__=='__main__':unittest.main()
