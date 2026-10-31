"""Regression checks for discovered source-binding and empty-profile failures."""
import json,os,unittest
from pathlib import Path
import numpy as np
from calibration.program_tracking.body_tracking.footwear_joint import JointFootwearProblem

PROGRAMME=Path(__file__).resolve().parents[1]/'events/ankara_2026/programmes/okada_mei_sp'
REVIEW=PROGRAMME/'footwear_s01_joint_semantics_v2_20261029.json'

class SemanticBindingRegression(unittest.TestCase):
    def test_rebinding_preserves_every_source_pixel(self):
        old=json.loads((PROGRAMME/'footwear_s01_joint_20261029.json').read_text(encoding='utf8'));new=json.loads(REVIEW.read_text(encoding='utf8'));changed=[]
        self.assertEqual(old['blade_curves'],new['blade_curves'])
        for a,b in zip(old['boot_landmarks'],new['boot_landmarks']):
            self.assertEqual((a['pts'],a['foot']),(b['pts'],b['foot']))
            self.assertEqual(sorted([a['heel'],a['toe']]),sorted([b['heel'],b['toe']]))
            if a['heel']!=b['heel']:
                changed.append((b['pts'],b['foot']));self.assertEqual(a['heel'],b['toe']);self.assertEqual(a['toe'],b['heel']);self.assertFalse(b['human_reviewed'])
        self.assertEqual(changed,[(325120,1),(336128,1)])

@unittest.skipUnless(os.environ.get('RINKSPACE_S01_BODY_ROOT'),'requires selected local source experiment')
class LocalJointRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root=Path(os.environ['RINKSPACE_S01_BODY_ROOT']);cls.dataset=cls.root/'footwear_v2/semantic_transfer';cls.base=cls.root/'next_round_20261029/B5/result.npz'
    def test_no_blade_curve_local_profile_window(self):
        p=JointFootwearProblem(self.root,REVIEW,self.dataset,self.base,'J0',window=(round(32.36*12800),round(33.56*12800)))
        self.assertEqual(len(p.p.blade_rows),0);self.assertGreater(p.n,0);v=p.evaluate(p.x0,jacobian=True);self.assertTrue(np.isfinite(v['residual']).all());fast=p.p.state(np.r_[p.g,p.x0],fast=True);full=p.p.state(np.r_[p.g,p.x0]);mask=np.isfinite(fast['uv']);np.testing.assert_allclose(fast['uv'][mask],full['uv'][mask],atol=1e-8)
    def test_corrected_joint_all_active_blocks_have_consistent_derivatives(self):
        p=JointFootwearProblem(self.root,REVIEW,self.dataset,self.base,'J2',window=(round(25.3*12800),round(25.5*12800)));s=dict(np.load(self.root/'next_round_20261029/J2_s2/result.npz'));x=np.r_[s['leg_controls'].ravel(),s['root_delta_controls'].ravel(),s['shared_geometry'][p.geometry_ids]];J=p.jac(x);rng=np.random.default_rng(20261029)
        for block in [slice(0,p.nleg),slice(p.nleg,p.nleg+p.nroot),slice(p.nleg+p.nroot,None)]:
            d=np.zeros(len(x));d[block]=rng.normal(size=len(d[block]));d/=np.linalg.norm(d);h=1e-5;fd=(p.fun(x+h*d)-p.fun(x-h*d))/(2*h);self.assertLess(float(np.linalg.norm(fd-J@d)/max(np.linalg.norm(fd),1e-15)),1e-6)

if __name__=='__main__':unittest.main()
