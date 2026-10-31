import unittest
import os
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from calibration.program_tracking.body_tracking.footwear_geometry import INITIAL,local_vertices,equipment_world,lowest_blade_world,decode
from calibration.program_tracking.body_tracking.footwear_fit import point_polyline_distance,bilinear


class FootwearGeometryTests(unittest.TestCase):
    def test_lowest_blade_matches_dense_curve_under_roll_and_pitch(self):
        rng=np.random.default_rng(123);R=Rotation.from_rotvec(rng.normal(size=(24,3))*.8).as_matrix();origin=rng.normal(size=(24,3));parameters=INITIAL.copy();parameters[-1]=.09
        result=lowest_blade_world(R,origin,parameters);p=decode(parameters);s=np.linspace(0,1,20001)
        zz=-p['shoe_rear']-p['blade_rear_extension']+(p['shoe_rear']+p['shoe_front']+p['blade_rear_extension']+p['blade_front_extension'])*s
        yy=-p['ankle_to_sole']-p['blade_drop']+p['rocker_rise']*(2*s-1)**2+(zz-(p['shoe_front']-p['shoe_rear'])/2)*np.tan(p['blade_pitch'])
        curves=np.concatenate([np.c_[np.full_like(s,side*.002),yy,zz] for side in [-1,1]])
        dense=origin[:,None]+np.einsum('nij,kj->nki',R,curves)
        np.testing.assert_allclose(result[:,2],dense[:,:,2].min(1),atol=1e-8)
    def test_equipment_is_rigid_in_each_ankle_frame(self):
        rng=np.random.default_rng(7);n=5;P=rng.normal(size=(n,3));Q=Rotation.random(n,random_state=rng).as_matrix();joints=rng.normal(size=(n,22,3));frames=Rotation.random(n*22,random_state=rng).as_matrix().reshape(n,22,3,3)
        vertices,parts=local_vertices(INITIAL);world,_,_,_=equipment_world(P,Q,joints,frames,INITIAL)
        expected=np.linalg.norm(vertices[0]-vertices[-1]);np.testing.assert_allclose(np.linalg.norm(world[:,:,0]-world[:,:,-1],axis=-1),expected,atol=1e-12)
    def test_source_distance_fields_do_not_hide_off_crop_predictions(self):
        image=np.zeros((20,30));values=bilinear(image,np.array([[10.,10.],[-5.,10.],[10.,25.]]))
        self.assertEqual(values[0],0);self.assertGreaterEqual(values[1],5);self.assertGreaterEqual(values[2],6)
        distance=point_polyline_distance(np.array([[.5,2.],[2.,1.]]),np.array([[0.,0.],[1.,0.]]))
        np.testing.assert_allclose(distance,[2,np.sqrt(2)])

@unittest.skipUnless(os.environ.get('RINKSPACE_S01_BODY_ROOT'),'requires selected local source-bound footwear fixture')
class NativeFootwearTests(unittest.TestCase):
    def test_reduced_projection_and_declared_sparse_support(self):
        from calibration.program_tracking.body_tracking.footwear_fit import FootwearProblem
        root=Path(os.environ['RINKSPACE_S01_BODY_ROOT'])
        review=root/'footwear_v2/semantic/articulated/source_review_at_run.json'
        p=FootwearProblem(root,review,'articulated',dataset=root/'footwear_v2/semantic')
        x=np.load(root/'footwear_v2/semantic/articulated/result.npz')['x']
        p.fast_projection=True;a=p.evaluate(x)['residual'].copy();p.cache_x=None;p.fast_projection=False;b=p.evaluate(x)['residual'].copy()
        np.testing.assert_allclose(a,b,atol=1e-8,rtol=1e-10)
        p.fast_projection=True
        for column in [0,11,28,451,780]:
            left=x.copy();right=x.copy();left[column]-=1e-5;right[column]+=1e-5
            derivative=(p.fun(right)-p.fun(left))/2e-5
            outside=~p.sparsity[:,column].toarray().ravel().astype(bool)
            self.assertLess(float(np.max(np.abs(derivative[outside]))),1e-7)

if __name__=='__main__':unittest.main()
