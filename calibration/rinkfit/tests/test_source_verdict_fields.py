import unittest
from copy import deepcopy
from rinkfit_joint.source_identity_guard import require_verdicts,SourceIdentityError

class CurrentSourceFindingTests(unittest.TestCase):
    def fixture(self):
        r=dict(raw_row=7,source_row_id='raw/native/7',native_pts=256,shot_index=0,material_index=3,path=0,uv=[1.,2.])
        v=dict(r,decision='ambiguous',uses_fitted_model_projection=False,evidence=[{'file':'original.png','sha256':'a'*64}],finding='Native material identity is unresolved.')
        return {7:r},v
    def test_current_finding_and_historical_reason_keep_the_same_rejection(self):
        records,v=self.fixture()
        self.assertTrue(require_verdicts(records,[],[v])['passed'])
        with self.assertRaisesRegex(SourceIdentityError,'remain in fit'):require_verdicts(records,[7],[v])
        old=deepcopy(v);old['reason']=old.pop('finding')
        self.assertEqual(require_verdicts(records,[],[old]),require_verdicts(records,[],[v]))
    def test_empty_finding_or_changed_source_cannot_clear_a_verdict(self):
        records,v=self.fixture();v['finding']=' '
        with self.assertRaisesRegex(SourceIdentityError,'Missing source-image'):require_verdicts(records,[],[v])
        v['finding']='Native unresolved material';v['uv']=[3.,4.]
        with self.assertRaisesRegex(SourceIdentityError,'UV changed'):require_verdicts(records,[],[v])
if __name__=='__main__':unittest.main()
