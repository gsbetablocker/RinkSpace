import cv2
import numpy as np
import pytest

from rinkspace.multisurface_camera import Camera, ground_points, project_camera
from rinkspace.vertical_board_geometry import (VerticalBoardSegment, intersect_reference_board,
    reproject_board, triangulation_observability)


def camera():
    return Camera(cv2.Rodrigues(np.array([1.4,0.,0.]))[0],np.array([0.,-60.,8.]),
                  1200.,np.array([480.,270.]))


def test_board_ray_plane_intersection_preserves_elevated_z_and_projection():
    cam=camera();points=np.array([[-20.,50.,2.],[10.,50.,4.]])
    xy,valid=project_camera(cam,points)
    assert valid.all()
    segment=VerticalBoardSegment('far',(-60.,50.),(60.,50.))
    result=intersect_reference_board(cam,xy,segment)
    got=np.array([o['world_xyz'] for o in result['observations']])
    assert np.allclose(got,points)
    assert np.allclose(np.array([p['image_xy'] for p in reproject_board(cam,result)]),xy)
    ground,good=ground_points(cam,xy)
    assert good.all()
    assert np.allclose(ground[:,2],0)
    assert np.max(np.linalg.norm(ground-points,axis=1))>5
    assert not result['pure_rotation_triangulation']


def test_corner_and_ground_pixels_cannot_become_straight_board_points():
    cam=camera();points=np.array([[70.,50.,2.],[0.,50.,-1.]])
    xy,_=project_camera(cam,points)
    result=intersect_reference_board(cam,xy,VerticalBoardSegment('far',(-60.,50.),(60.,50.)))
    assert [o['reason'] for o in result['observations']]==[
       'outside_known_straight_segment_corner_or_other_surface','nonpositive_height_not_board_surface']
    assert all(o['world_xyz'] is None for o in result['observations'])
    with pytest.raises(ValueError,match='Curved'):
        VerticalBoardSegment('corner',(0.,50.),(60.,0.),geometry_kind='arc')


def test_pure_pan_zoom_does_not_reveal_depth():
    a=camera();b=Camera(cv2.Rodrigues(np.array([0.,.08,0.]))[0]@a.rotation,
                         a.centre.copy(),1500.,a.principal.copy())
    point=np.array([[10.,50.,3.]])
    first,_=project_camera(a,point);second,_=project_camera(b,point)
    result=triangulation_observability(a,first,b,second)
    assert result['depth_observable'] is False
    assert result['returned_depth'] is None
    assert result['reason']=='pure_rotation_cannot_triangulate_depth'
    # Every positive distance on the same ray yields the same two observations.
    same_ray_point=a.centre+3*(point-a.centre)
    moved,_=project_camera(b,same_ray_point)
    assert np.allclose(moved,second)
