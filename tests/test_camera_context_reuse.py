import numpy as np
from dataclasses import replace
from scipy.spatial.transform import Rotation

from rinkspace.multisurface_camera import Camera,world_rays,project_camera
from rinkspace.camera_context_reuse import fit_context_increment


def test_adjacent_increment_recovers_rotation_zoom_and_keeps_shared_installation():
    camera=Camera(np.eye(3),np.array([3.,-62.,6.]),1200.,np.array([480.,270.]),(960,540),.08)
    actual=replace(camera,rotation=Rotation.from_rotvec([.006,-.009,.001]).as_matrix(),focal=1224.)
    xx,yy=np.meshgrid(np.linspace(80,880,10),np.linspace(40,470,7));pixels=np.c_[xx.ravel(),yy.ravel()]
    rays,_=world_rays(camera,pixels);observed,_=project_camera(actual,rays,rays=True)
    recovered,evidence=fit_context_increment(camera,pixels,observed)
    assert recovered is not None
    assert np.array_equal(recovered.centre,camera.centre)
    assert recovered.radial_ray_k1==camera.radial_ray_k1
    assert evidence['heldout_count']>0
    assert evidence['heldout_p95_sample_px']<.01
    assert not evidence['geometry_validation']


def test_unrelated_frame_tracks_are_rejected_instead_of_smoothly_propagated():
    rng=np.random.default_rng(17)
    camera=Camera(np.eye(3),np.array([3.,-62.,6.]),1200.,np.array([480.,270.]),(960,540))
    pixels=rng.uniform([60,40],[880,470],(100,2));unrelated=pixels+rng.normal(0,30,(100,2))
    recovered,evidence=fit_context_increment(camera,pixels,unrelated)
    assert recovered is None
    assert evidence['reason']=='static_image_link_rejected'
    assert not evidence['image_link_accepted']
