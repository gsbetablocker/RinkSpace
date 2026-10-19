import cv2
import numpy as np

from rinkspace.camera_reuse_localization import (CameraReuseLocalizer, ReuseConfig,
    fit_evidence, camera_projection_chain, unused_ice_check)
from rinkspace.multisurface_camera import Camera, world_rays, project_camera


def cameras_and_points():
    a = Camera(np.eye(3), np.array([0.,0.,-10.]), 900., np.array([480.,270.]))
    b = Camera(cv2.Rodrigues(np.array([.035,.06,.01]))[0], a.centre.copy(),
               1030., a.principal.copy())
    x,y = np.meshgrid(np.linspace(100,820,9),np.linspace(130,420,5))
    left=np.c_[x.ravel(),y.ravel()]
    rays,_=world_rays(a,left)
    right,_=project_camera(b,rays,rays=True)
    return a,b,left,right


def test_withheld_ice_cannot_pull_background_camera_into_false_agreement():
    a,b,left,right=cameras_and_points()
    ice=np.zeros((540,960),np.uint8);ice[330:]=255
    changed=right.copy();changed[left[:,1]>=330] += [30.,-20.]
    fitted,report=fit_evidence(a,left,changed,ice,ReuseConfig())
    assert fitted is None
    assert report['check_kind']=='ice_points_excluded_from_rotation_zoom_fit'
    assert report['fit_median_px'] < .05
    assert report['check_p95_px'] > 25


def test_rotation_zoom_keeps_shared_installation_and_projection_direction():
    a,b,left,right=cameras_and_points()
    fitted,report=fit_evidence(a,left,right,np.zeros((540,960),np.uint8),ReuseConfig())
    assert fitted is not None
    np.testing.assert_allclose(fitted.centre,a.centre,atol=1e-10)
    np.testing.assert_allclose(fitted.H,b.H,atol=1e-4)
    assert report['independent_geometry_validation'] is False
    chain=camera_projection_chain(fitted)
    np.testing.assert_allclose(chain['H_current_sample_from_reference_raw'],np.eye(3))


def test_return_after_cut_must_retrieve_fresh_view():
    localizer=CameraReuseLocalizer([])
    localizer.active={'camera':'old_direction', 'key_time':1.,'anchor_time':1.}
    localizer.last_time=1.
    image=np.full((540,960,3),200,np.uint8)
    result=localizer.process(image,1.02,cut=True)
    assert result is None
    assert localizer.active is None
    assert localizer.sequence == 1


def test_unused_ice_is_checked_after_large_zoom_with_actual_texture():
    a = Camera(np.eye(3), np.array([0.,0.,-10.]), 650., np.array([480.,270.]))
    b = Camera(np.eye(3), a.centre.copy(), 1100., a.principal.copy())
    random=np.random.default_rng(16)
    gray=cv2.GaussianBlur(random.integers(50,220,(540,960),dtype=np.uint8),(5,5),.8)
    image=cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
    h=b.K @ np.linalg.inv(a.K)
    current=cv2.warpPerspective(image,h,(960,540))
    ice=np.zeros(gray.shape,np.uint8);ice[320:490,150:810]=255
    result=unused_ice_check({'camera':a,'image':image,'ice_mask':ice},current,b)
    assert result['passed']
    assert result['count']>=4 and result['p95_px']<1
    blank=unused_ice_check({'camera':a,'image':image,'ice_mask':ice},np.full_like(current,200),b)
    assert not blank['passed']
