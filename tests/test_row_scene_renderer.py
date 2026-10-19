"""Disposable world/row projection fixtures, never rink measurements."""
from types import SimpleNamespace
import numpy as np
import pytest
from calibration.solver_runtime import install
install()
from calibration.program_tracking.row_scene_projection import RowSceneProjection,_batch_project
from calibration.program_tracking.projection import safe_project
from calibration.program_tracking import render_result as render

def camera():
    return np.r_[np.zeros(6),np.log([500.,500.]),960.,539.5,0.,np.zeros(14)]

class Trajectory:
    image_height=1080
    def __init__(self,readout=10.):
        self.readout_milliseconds=readout;self.cache_key=('test-native-row',readout)
    def camera25_at_rows(self,rows,*,geometry_navigation=False):
        cams=np.repeat(camera()[None],len(rows),axis=0)
        cams[:,1]=(np.asarray(rows)-539.5)*.0002*self.readout_milliseconds
        return cams
    def provenance(self):return dict(synthetic_fixture=True,signed_readout_milliseconds=self.readout_milliseconds)

def test_batch_projection_retains_same_central_guard_and_scalar_values():
    cams=np.repeat(camera()[None],5,axis=0);cams[:,11]=.02;cams[:,12]=-.001
    points=np.array([[1.,2.,10.],[-3.,1.,8.],[0.,0.,-2.],[1.,0.,.03],[4.,-1.,9.]])
    actual=_batch_project(cams,points)
    expected=[safe_project(c,p[None]) for c,p in zip(cams,points,strict=True)]
    np.testing.assert_allclose(actual[0],np.vstack([p[0] for p in expected]),atol=1e-10,equal_nan=True)
    np.testing.assert_allclose(actual[1],[p[1][0] for p in expected],atol=1e-12)
    np.testing.assert_array_equal(actual[2],[p[2][0] for p in expected])

def test_implicit_world_projection_has_exact_observed_row_physical_zero():
    context=Trajectory();projection=RowSceneProjection(context,camera())
    world=np.array([[0.,.4,10.],[2.,-2.,8.]])
    uv,depth,good=projection.project(camera(),world)
    assert good.all()
    np.testing.assert_allclose(uv[0,1],(559.5+53.95)/1.1,atol=1e-6)
    row_cameras=context.camera25_at_rows(uv[:,1])
    conditional=_batch_project(row_cameras,world)
    np.testing.assert_allclose(conditional[0],uv,atol=1e-6)
    # This is a changing 3D optical centre, not a post-projection image shear.
    assert abs(uv[0,1]-safe_project(camera(),world)[0][0,1])>1.

def test_zero_readout_projection_matches_original_values_byte_exact():
    world=np.array([[1.,2.,10.],[0.,0.,-2.]])
    result=RowSceneProjection(Trajectory(0.),camera()).project(camera(),world)
    expected=safe_project(camera(),world)
    for a,b in zip(result,expected,strict=True):
        assert a.tobytes()==b.tobytes()

def test_rs_near_plane_crossing_retains_visible_world_segment_and_wall():
    p=RowSceneProjection(Trajectory(),camera())
    lines,details=p.segments(camera(),np.array([[-4.,0.,-1.]]),np.array([[4.,0.,5.]]))
    assert len(lines)>0 and np.isfinite(lines).all()
    assert np.min(lines[:,:,0])<960 and np.max(lines[:,:,0])>960
    polygon=p.wall_polygon(camera(),np.array([[-4.,-2.,-1.],[4.,-2.,5.],[4.,2.,5.],[-4.,2.,-1.]]))
    assert len(polygon)>=3 and np.isfinite(polygon).all()
    assert (polygon[:,0]>=0).all() and (polygon[:,0]<=1919).all()
    assert (polygon[:,1]>=0).all() and (polygon[:,1]<=1079).all()

def test_complete_rs_draw_preserves_ads_foreground_graphics_and_cache_identity(monkeypatch):
    boundary=np.array([[-4.,-2.,5.],[4.,-2.,5.],[4.,2.,5.],[-4.,2.,5.],[-4.,-2.,5.]])
    top=boundary+np.array([0.,0.,1.])
    paths=(('contact',boundary),('board_top',top),('marking',np.array([[-2.,0.,10.],[2.,0.,10.]])))
    monkeypatch.setattr(render,'fixed_scene_paths',lambda scene,draw_grid:paths)
    ad=dict(number=1,extent='full_advertisement',xyz=np.array([[-2.,-2.,5.],[2.,-2.,5.]]),primitive='polyline')
    monkeypatch.setattr(render,'advertisement_review_paths',lambda scene:[ad])
    r=render.SceneRenderer({'ice_boundary':{'points_xyz':boundary.tolist()}},graphics_regions=((10,10,70,60),))
    native=np.full((1080,1920,3),90,np.uint8);foreground=np.zeros((1080,1920),bool);foreground[350:500,900:1030]=True
    normal,_=r.draw(native,camera(),foreground);same,_=r.draw(native,camera(),foreground,projection_context=None)
    assert normal.tobytes()==same.tobytes()
    for readout in [10.,12.]:
        image,metrics=r.draw(native,camera(),foreground,projection_context=RowSceneProjection(Trajectory(readout),camera()))
        np.testing.assert_array_equal(image[foreground],native[foreground])
        np.testing.assert_array_equal(image[10:60,10:70],native[10:60,10:70])
        assert metrics['advertisements']['modeled_ids']==[1]
        assert metrics['advertisements']['projected_ids']==[1]
        assert not metrics['row_projection']['screen_shear_used']
        assert metrics['occlusion']['projected_board_pixels']>0
    assert r.geometry_cache_info()['misses']==3
