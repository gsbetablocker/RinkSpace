import copy
import cv2
import numpy as np

from rinkspace.geometry import project
from rinkspace.reference_camera_refit import (camera_matrix, decompose_ground_homography,
    ground_homography, refit_fixed_geometry)


def test_explicit_camera_and_homography_share_ground_projection():
    r=cv2.Rodrigues(np.array([2.7,.1,.05]))[0]
    camera={'K':camera_matrix(1200,[960,540]),'R':r,'t':-r@np.array([2.,-60.,8.])}
    world=np.array([[-20,30],[10,20],[15,-15],[0,0]],float)
    q=np.c_[world,np.zeros(len(world))]@r.T+camera['t']
    q=q@camera['K'].T
    assert np.allclose(project(ground_homography(camera),world),q[:,:2]/q[:,2:])


def test_homography_decomposition_recovers_camera_when_constraints_hold():
    r=cv2.Rodrigues(np.array([2.7,.1,.05]))[0]
    c=np.array([2.,-60.,8.])
    camera={'K':camera_matrix(1200,[960,540]),'R':r,'t':-r@c}
    h=ground_homography(camera)
    recovered=decompose_ground_homography(h,[960,540])
    assert np.isclose(recovered['focal_px'],1200,rtol=1e-8)
    assert np.allclose(ground_homography(recovered),h,atol=1e-8)
    assert np.allclose(-recovered['R'].T@recovered['t'],c,atol=1e-8)


def test_refit_does_not_modify_geometry_or_relabel_checks_independent():
    r=cv2.Rodrigues(np.array([2.7,.1,.05]))[0]
    c=np.array([2.,-60.,8.])
    camera={'K':camera_matrix(1200,[960,540]),'R':r,'t':-r@c}
    h=ground_homography(camera)
    points=[[-20,30],[10,20],[15,-15],[0,0],[-10,-10],[30,30]]
    features=[{'id':f'p{i}','kind':'point','points':[p]} for i,p in enumerate(points)]
    obs=[{'id':f'o{i}','view_id':'v','feature_id':f'p{i}','type':'point',
          'points_image':[q.tolist()],'sigma_image_px':1.,'role':'check' if i==0 else 'fit'}
         for i,q in enumerate(project(h,points))]
    seed=h.copy();seed[0,2]+=5
    model={'parameter_estimates':{},'features':features,'observations':obs,
      'view_poses':[{'id':'v','sample_size':[960,540],'source_size':[1920,1080],
       'source_id':'test','source_sha256':'a'*64,'pts':100,'source_time_s':1.,
       'time_base':'1/100','image':'fixture','H_image_from_model':seed.tolist()}]}
    before=copy.deepcopy(model)
    result=refit_fixed_geometry(model,'v')
    assert model==before
    assert result['observations'][0]['previous_role']=='check'
    assert result['observations'][0]['role']=='fit'
    assert result['status']=='development_camera_candidate_not_independently_validated'
    assert np.allclose(project(result['camera']['H_image_from_model'],points),project(h,points),atol=.01)
