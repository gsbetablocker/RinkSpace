import json
import numpy as np
import pytest
from rinkfit_start.native_camera import NativeCamera
from rinkfit_start.camera import FixedHead, Intrinsics, Brown, project


def simple_camera(tau=.01):
    return {"shared_head":{"R0":np.eye(3).tolist(),"n":[0,0,1],"b":[1,0,0],"P":[0,0,0],"d":[0,0,0],"a":0.},
      "camera_controls":{"basis":{"first_control_time":-.04,"dt":.02,"n_controls":8},
          "controls":[[0,0,np.log(1000.)] for _ in range(8)]},
      "optics":{"f_ref":1000.,"principal_point":[959.5,539.5],"eta":0.,"principal_linear":[0,0],
          "k1":[.01,0.],"k2":[0.,0.],"p1":.001,"p2":-.002,"radius_max":2.},
      "rolling_shutter":{"signed_readout_seconds":tau}}


def test_independent_forward_matches_pure_reference_and_true_root():
    s=simple_camera();c=NativeCamera(s);X=np.array([[.1,.2,2.],[-.3,-.2,1.]])
    out=c.at(X,np.array([0.,.02]));head=FixedHead(np.eye(3),[0,0,1],[1,0,0],[0,0,0],[0,0,0])
    expected=[project(x,head.pose(0,0,0),Intrinsics(1000,1000,959.5,539.5),Brown(.01,0,.001,-.002,radius_max=2)).uv for x in X]
    np.testing.assert_allclose(out['uv'],expected,atol=1e-10)
    rs=c.project_rs(X,[0.,.02]);assert rs['valid'].all()
    np.testing.assert_allclose(rs['uv'],expected,atol=1e-8)
    assert rs['certificate']['full_interval']


def test_out_of_view_and_behind_camera_never_disappear():
    c=NativeCamera(simple_camera());out=c.project_rs([[0,3,1],[0,0,-1]],[0.,0.])
    assert len(out['valid'])==2;assert not out['valid'].any()


def test_nonzero_rs_changes_prediction_and_certificate_bounds_velocity():
    s=simple_camera();controls=np.array(s['camera_controls']['controls']);controls[:,0]=np.arange(8)*.005
    # Set the function reference gauge by subtracting its actual value.
    controls[:,0]-=.01;s['camera_controls']['controls']=controls.tolist();c=NativeCamera(s)
    X=np.array([[.2,.4,2.]]);t=np.array([.02]);rs=c.project_rs(X,t)
    assert rs['valid'].all();assert np.linalg.norm(rs['uv']-c.at(X,t)['uv'])>1e-4
    grid=np.linspace(.015,.025,1001);Xg=np.repeat(X,len(grid),axis=0);v=c.at(Xg,grid)['uv'][:,1]
    measured=np.max(np.abs(np.diff(v)/np.diff(grid)))
    assert measured<rs['certificate']['speed_y_bound_px_s'][0]


def test_missing_time_extension_rejected():
    c=NativeCamera(simple_camera())
    with pytest.raises(ValueError):c.project_rs([[0,0,1]],[-.02])


def test_full_disk_and_zoom_hull_cannot_hide_a_fold_outside_observed_ray():
    source=simple_camera();source['optics'].update(radius_max=.32,k1=[.01,-10.],k2=[0.,0.],p1=0.,p2=0.)
    controls=np.asarray(source['camera_controls']['controls']);controls[-1,2]+=.5
    source['camera_controls']['controls']=controls.tolist();camera=NativeCamera(source)
    assert camera.project_rs([[.01,0,1]],[0.])['valid'].all()
    certificate=camera.optical_domain_certificate()
    assert not certificate['passed']
    assert certificate['Brown_Jacobian_eigenvalue_lower']<0
    assert certificate['covers_entire_frozen_ray_disk']
