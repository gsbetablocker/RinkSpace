"""Coordinate, visibility and real-image motion regressions for blade tracking."""
import os,unittest
from pathlib import Path
import cv2,numpy as np
from calibration.program_tracking.body_tracking.blade_temporal import image_flow,visible_segment
from calibration.program_tracking.body_tracking.blade_local_tracks import same_visible_blade
from calibration.program_tracking.body_tracking.blade_cotracker_review import source_to_model,model_to_source,overlap_error,longest_run

class BladeSourceGeometryTests(unittest.TestCase):
    def test_crop_transform_roundtrip_with_different_frame_origins_and_padding(self):
        q=np.array([[[750.2,801.7],[820.,840.]],[[1120.5,917.25],[1190.,928.]]]);tr=np.array([[700,650,1.4,1.399,7,0],[1040,770,1.22,1.219,0,39]])
        np.testing.assert_allclose(model_to_source(source_to_model(q,tr),tr),q,atol=1e-12)
    def test_partial_curve_overlap_is_separate_from_disjoint_fragments(self):
        a=np.c_[np.linspace(0,40,17),np.zeros(17)];b=np.c_[np.linspace(20,60,17),np.zeros(17)+3];self.assertAlmostEqual(overlap_error(a,b),3.,places=10);self.assertIsNone(overlap_error(a,b+[100,0]))
    def test_missing_edge_pixels_are_not_completed(self):
        q=np.c_[np.arange(17)*3.,np.zeros(17)];self.assertIsNone(visible_segment(q,np.zeros(17,bool)));mask=np.zeros(17,bool);mask[4:9]=True;part=visible_segment(q,mask);np.testing.assert_array_equal(part,q[4:9]);self.assertEqual((longest_run(mask).start,longest_run(mask).stop),(4,9))
    def test_two_nonoverlapping_fragments_of_one_blade_are_not_two_instances(self):
        a=np.c_[np.linspace(0,30,17),np.zeros(17)];b=np.c_[np.linspace(35,65,17),np.ones(17)];self.assertTrue(same_visible_blade(a,b));self.assertFalse(same_visible_blade(a,b+[0,20]))
    def test_navigation_can_flag_two_candidates_at_one_shoe(self):
        a=np.c_[np.linspace(0,15,9),np.zeros(9)];b=np.c_[np.linspace(25,40,9),np.ones(9)*5];self.assertTrue(same_visible_blade(a,b,[[20,0],[150,0]]));self.assertFalse(same_visible_blade(a,b+[100,0],[[20,0],[150,0]]))
    def test_motion_uses_native_coordinates_across_different_crops(self):
        rng=np.random.default_rng(21);im=np.full((200,240,3),150,np.uint8);patch=rng.integers(70,235,(65,100,3),dtype=np.uint8);im[38:103,35:135]=patch;cv2.line(im,(35,105),(135,105),(40,40,40),2);matrix=np.float32([[1,0,7],[0,1,4]]);moved=cv2.warpAffine(im,matrix,(240,200),borderValue=(150,150,150));prev=im[:170,:190];cur=moved[3:173,5:195];curve=np.c_[np.linspace(40,130,17),np.full(17,105.)];rows=[dict(crop_xyxy=[0,0,190,170]),dict(crop_xyxy=[5,3,195,173])];result=image_flow(prev,cur,*rows,[dict(curve=curve)],np.array([[85,55],[170,55]]))[0];self.assertTrue(result['image_motion_supported']);self.assertLess(float(np.median(np.linalg.norm(result['prediction']-(curve+[7,4]),axis=1))),1.)
        hidden=np.full_like(cur,150);lost=image_flow(prev,hidden,*rows,[dict(curve=curve)],np.array([[85,55],[170,55]]))[0];self.assertFalse(lost['image_motion_supported'])

@unittest.skipUnless(os.environ.get('RINKSPACE_S01_BODY_ROOT'),'requires actual source query packet')
class SavedSourceQueryTests(unittest.TestCase):
    def test_actual_queries_match_saved_native_to_model_transform(self):
        f=Path(os.environ['RINKSPACE_S01_BODY_ROOT'])/'blade_tracking_20261030/continuous/cotracker_input/queries.npz';z=dict(np.load(f));q=z['queries_native'];tr=z['crop_transforms'][q[:,0].astype(int)];np.testing.assert_allclose(source_to_model(q[:,1:],tr),z['queries_model'][:,1:],atol=1e-10);np.testing.assert_allclose(model_to_source(z['queries_model'][:,1:],tr),q[:,1:],atol=1e-10)

if __name__=='__main__':unittest.main()
