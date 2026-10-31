import json,os,unittest,tempfile
from pathlib import Path
import numpy as np
from calibration.program_tracking.body_tracking.rebuild_prepare import native_metric_binding,prepare,load
from rinkspace.metric_scale import estimate_uniform_scale

@unittest.skipUnless(os.environ.get('RINKSPACE_S01_BODY_ROOT'),'requires frozen local S01 state')
class NativeBodyMetricTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root=Path(os.environ['RINKSPACE_S01_BODY_ROOT']);meta=json.loads((cls.root/'input.json').read_text(encoding='utf8'));cls.scene=meta['scene_path'];cls.snapshot=meta['snapshot'];cls.run_root=meta['run_root']
    def test_scale_uses_fitted_parameter_state_and_public_conversion(self):
        b=native_metric_binding(self.scene,self.snapshot);self.assertAlmostEqual(b['metres_per_scene_unit'],estimate_uniform_scale(b['source_span_xy_scene_units']),places=13);self.assertNotEqual(b['metres_per_scene_unit'],60.);self.assertEqual(b['policy'],'equal_weight_squared_dimension_error')
    def test_reject_inconsistent_snapshot_board(self):
        z=load(self.snapshot);z['board'][1]*=1.01
        with self.assertRaises(AssertionError):native_metric_binding(self.scene,self.snapshot,z)
    def test_reject_old_default_before_preparing_or_overwriting_data(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(ValueError,'Requested scale differs'):
                prepare(td,self.run_root,self.snapshot,'unused-ledger',metres_per_unit=60.)
            self.assertEqual(list(Path(td).iterdir()),[])

    def test_delivery_rejects_old_scale_before_exporting_parameters(self):
        from calibration.program_tracking.body_tracking.rebuild_body_deliver import deliver
        with tempfile.TemporaryDirectory() as td:
            folder=Path(td);np.savez(folder/'input.npz',metres_per_scene_unit=np.asarray(60.))
            (folder/'input.json').write_text(json.dumps(dict(scene_path=self.scene,snapshot=self.snapshot)),encoding='utf8')
            with self.assertRaisesRegex(ValueError,'obsolete metric conversion'):deliver(folder,folder/'unused-project')
            self.assertEqual(sorted(p.name for p in folder.iterdir()),['input.json','input.npz'])

if __name__=='__main__':unittest.main()
