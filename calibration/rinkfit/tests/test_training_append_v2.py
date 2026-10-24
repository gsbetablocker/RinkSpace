import unittest
import numpy as np
from rinkfit_joint.training_source_append_v2 import appended_representatives, declaration


class Reader:
    def __init__(self):
        self.data=dict(uv=np.array([[1.,1.],[2.,2.],[3.,3.],[80.,10.],[90.,20.]]),
            profile_normal=np.tile([1.,0.],(5,1)),shot_index=np.zeros(5,int),native_pts=np.array([256,256,256,512,512]),
            material_index=np.zeros(5,int),path=np.zeros(5,int),kind=np.zeros(5,int))
    def load(self,key):return self.data[key]


class NewSourceSelectionTests(unittest.TestCase):
    def test_same_grid_and_extrema_rule_selects_only_new_eligible_rows(self):
        reader=Reader();model=dict(materials=[dict(label='ice.C0:outer')])
        self.assertEqual(appended_representatives(reader,model,[0,1,2,3,4],np.array([0])).tolist(),[0,2,3,4])
        # An ineligible middle point never replaces a missing old representative.
        self.assertEqual(appended_representatives(reader,model,[3,4],np.array([0])).tolist(),[3,4])
        self.assertEqual(appended_representatives(reader,model,[],np.array([0])).tolist(),[])

    def test_explicit_declaration_keeps_original_state_and_current_parent_distinct(self):
        p=dict(parent_packet={'file':'f','sha256':'f'},original_state_packet={'file':'e','sha256':'e'},parent_raw_rows=10,current_raw_rows=12)
        rowref=dict(file='new.u64',sha256='a',dtype='<u8',shape=[2])
        result=declaration({'file':'proof','sha256':'p'},p,{'file':'old_selection','sha256':'o'},rowref)
        self.assertEqual(result['parent_packet']['file'],'f');self.assertEqual(result['original_state_packet']['file'],'e')
        self.assertTrue(result['old_domain_no_backfill']);self.assertEqual(result['maximum_added_fit_rows'],2)
        self.assertFalse(result['representative_policy']['selection_uses_fit_residual'])


if __name__=='__main__':unittest.main()
