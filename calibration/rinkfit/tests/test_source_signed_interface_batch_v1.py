import unittest
import numpy as np
from rinkfit_joint.source_signed_interface_batch_v1 import signed_candidates
from rinkfit_joint.source_worker import _channels

class SignedOriginalInterfaceTest(unittest.TestCase):
    def setUp(self):
        image=np.zeros((80,100,3),np.uint8);image[36:40]=255
        self.channels=_channels(image)
        self.paths=[dict(label='panel.top',path=0,source_native_uv=[[10,36],[90,36]]),dict(label='panel.bottom',path=0,source_native_uv=[[10,40],[90,40]])]
    def test_nearby_opposite_interfaces_are_distinct(self):
        # Equal geometric distance is resolved by recorded source polarity.
        candidates=signed_candidates(self.channels,self.paths,np.eye(3),np.array([[50.,38.],[50.,38.]]),np.array([[0.,1.],[0.,1.]]),np.array([[0,0,0,0,1],[0,0,0,0,-1]],float))
        self.assertEqual([[q['label']for q in x]for x in candidates],[['panel.top'],['panel.bottom']])
    def test_actual_normal_and_image_transform_preserve_identity(self):
        H=np.array([[1.2,0,3],[0,1.1,4],[0,0,1.]])
        candidates=signed_candidates(self.channels,self.paths,H,np.array([[63.,45.8]]),np.array([[0.,-1.]]),np.array([[0,0,0,0,-1]],float))
        self.assertEqual([q['label']for q in candidates[0]],['panel.top'])
    def test_weak_original_interface_is_unconfirmed(self):
        candidates=signed_candidates(np.zeros_like(self.channels),self.paths,np.eye(3),np.array([[50.,38.]]),np.array([[0.,1.]]),np.array([[0,0,0,0,1]],float))
        self.assertEqual(candidates,[[]])

if __name__=='__main__':unittest.main()
