import cv2
import numpy as np
import pytest

from rinkspace.multisurface_camera import (Camera, camera_from_homography,
    project_camera, world_rays, ground_points, fit_shared_camera, projection_chain)


def fixture():
    reference=Camera(np.diag([1.,-1.,-1.]),np.array([2.,3.,80.]),950.,np.array([480.,270.]))
    current=Camera(cv2.Rodrigues(np.array([.018,-.025,.007]))[0]@reference.rotation,
                   reference.centre.copy(),1030.,reference.principal.copy())
    xx,yy=np.meshgrid(np.linspace(-25,25,6),np.linspace(-18,18,4))
    ground=np.column_stack((xx.ravel(),yy.ravel(),np.zeros(xx.size)))
    board=ground.copy();board[:,1]=24.;board[:,2]=np.tile([5.,9.,13.,17.],6)
    return reference,current,ground,board


def test_elevated_points_constrain_the_same_camera_without_becoming_ground():
    reference,current,ground,board=fixture()
    pairs=[]
    for surface,points in [('ice_marking',ground),('board_surface',board)]:
        old,_=project_camera(reference,points);new,_=project_camera(current,points)
        pairs.append(dict(surface=surface,reference_camera=reference,reference_xy=old,current_xy=new))
    estimated,diagnostic=fit_shared_camera(reference,pairs)
    np.testing.assert_allclose(estimated.rotation,current.rotation,atol=1e-7)
    assert estimated.focal==pytest.approx(current.focal,abs=.001)
    assert diagnostic['board_surface']['p95_px']<.001
    assert diagnostic['ice_marking']['p95_px']<.001
    assert not diagnostic['board_used_as_ground']
    np.testing.assert_array_equal(estimated.centre,reference.centre)


def test_camera_and_ground_matrix_use_same_coordinate_direction():
    reference,_,ground,_=fixture()
    from rinkspace.geometry import project
    raw,_=project_camera(reference,ground)
    np.testing.assert_allclose(project(reference.H,ground[:,:2]),raw)
    back,valid=ground_points(reference,raw)
    assert valid.all()
    np.testing.assert_allclose(back,ground,atol=1e-9)
    chain=projection_chain(reference,source_size=(3840,2160))
    assert chain['H_source_image_from_sample'][0][0]==4


def test_lens_inverse_and_forward_preserve_raw_pixels_after_zoom():
    reference,current,ground,_=fixture()
    reference.radial_ray_k1=-.025
    raw,_=project_camera(reference,ground)
    ray,valid=world_rays(reference,raw)
    roundtrip,good=project_camera(reference,ray,rays=True)
    assert valid.all() and good.all()
    np.testing.assert_allclose(roundtrip,raw,atol=1e-7)
    current.radial_ray_k1=reference.radial_ray_k1
    assert current.distortion['k1']!=reference.distortion['k1']


def test_vertical_board_curve_cannot_be_accidentally_added_to_ground_route():
    reference,_,ground,board=fixture()
    old,_=project_camera(reference,ground)
    pair=dict(surface='ice_marking',reference_camera=reference,reference_xy=old,current_xy=old)
    with pytest.raises(ValueError,match='Elevated board'):
        fit_shared_camera(reference,[pair],[dict(surface='board_surface',world_polyline=board,observed_xy=old)])
    with pytest.raises(ValueError,match='Z=0'):
        fit_shared_camera(reference,[pair],[dict(surface='ice_board_contact',world_polyline=board,observed_xy=old)])


def test_no_correspondences_do_not_invent_a_camera():
    reference,*_=fixture()
    estimated,diagnostic=fit_shared_camera(reference,[])
    assert estimated is None and not diagnostic['success']


def test_board_xyz_preserves_height_and_does_not_take_the_ray_route():
    reference,current,_,board=fixture()
    pixels,_=project_camera(current,board)
    item=dict(surface='board_surface',representation='world_points',world_points=board,current_xy=pixels)
    estimate,report=fit_shared_camera(reference,[item])
    assert report['board_surface']['p95_px']<.001
    assert estimate.focal==pytest.approx(current.focal,abs=.001)
    item.pop('representation')
    with pytest.raises(ValueError,match='Explicit board representation'):
        fit_shared_camera(reference,[item])


def test_board_rays_cannot_silently_cross_camera_centres():
    reference,current,_,board=fixture()
    before,_=project_camera(reference,board);after,_=project_camera(current,board)
    current.centre[0]+=3.
    with pytest.raises(ValueError,match='same conditional optical centre'):
        fit_shared_camera(current,[dict(surface='board_surface',reference_camera=reference,
            reference_xy=before,current_xy=after)])


def test_folded_radial_and_backwards_origin_branches_are_explicit():
    reference,_,_,_=fixture();reference.radial_ray_k1=-1.
    p,ok=project_camera(reference,np.array([[1.,0.,1.]]),rays=True)
    assert not ok[0] and np.isnan(p).all()
    camera=Camera(np.eye(3),np.array([0.,0.,1.]),900.,np.array([480.,270.]))
    assert projection_chain(camera)['reference_visible_branch_sign']==-1


def test_physical_homography_decomposition_recovers_an_oblique_view():
    reference,current,ground,_=fixture()
    estimate,diagnostic=camera_from_homography(current.H)
    before,_=project_camera(current,ground)
    after,_=project_camera(estimate,ground)
    np.testing.assert_allclose(after,before,atol=1e-7)
    assert diagnostic['positive_camera_height']
    assert not diagnostic['independent_calibration']
