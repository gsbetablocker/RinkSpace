import cv2
import numpy as np
import pytest
from calibration.solver_runtime import install
install()
from calibration.program_tracking.first_shot_source_kernel import _interfaces
from calibration.program_tracking.ice_motion_factor import evaluate_ice_motion


def case():
    a=np.zeros(25);a[:3]=[0.,-10.,8.]
    forward=-a[:3];forward/=np.linalg.norm(forward);right=np.cross(forward,[0.,0.,1.]);right/=np.linalg.norm(right);down=np.cross(forward,right)
    a[3:6]=cv2.Rodrigues(np.stack((right,down,forward)))[0].ravel();a[6:8]=np.log([950.,930.]);a[8:11]=[960.,540.,.7];a[11:15]=[-.08,.008,.0004,-.0003]
    b=a.copy();b[:3]+=[.02,.01,.03];b[3:6]+=[.003,-.006,.002];b[6:8]+=.004
    world=np.array([[-2.,-1.,0.],[1.,0.,0.],[2.,2.,0.]])
    project=_interfaces()[0];ua=project(a,world)[0];ub=project(b,world)[0]
    return a,b,ua,ub,world


def test_real_pair_zero_set_both_camera_derivatives_and_no_source_edits():
    a,b,ua,ub,world=case();before=[x.copy()for x in (a,b,ua,ub)]
    r,ja,jb,w=evaluate_ice_motion(a,b,ua,ub)
    np.testing.assert_allclose(r,0.,rtol=0,atol=1e-8);np.testing.assert_allclose(w,world,rtol=0,atol=1e-9)
    assert np.linalg.norm(ja)>1 and np.linalg.norm(jb)>1
    for x,y in zip((a,b,ua,ub),before):np.testing.assert_array_equal(x,y)
    raw=evaluate_ice_motion(a,b,ua,ub,jacobian=False)[0];np.testing.assert_array_equal(raw,r)


def test_both_camera15_derivatives_include_source_ray_intersection():
    a,b,ua,ub,world=case();ub=ub+[.3,-.2];r,ja,jb,w=evaluate_ice_motion(a,b,ua,ub)
    scale=np.r_[np.full(3,100.),np.ones(5),np.full(3,1000.),np.ones(4)];h=1e-6
    for endpoint,expected in [(0,ja),(1,jb)]:
        for col in range(15):
            plus=[a.copy(),b.copy()];minus=[a.copy(),b.copy()];plus[endpoint][col]+=h*scale[col];minus[endpoint][col]-=h*scale[col]
            rp=evaluate_ice_motion(*plus,ua,ub,jacobian=False)[0];rm=evaluate_ice_motion(*minus,ua,ub,jacobian=False)[0]
            np.testing.assert_allclose((rp-rm)/(2*h),expected[:,col],rtol=3e-5,atol=3e-4)


def test_plane_motion_does_not_claim_metric_scale_or_planar_origin_observed():
    a,b,ua,ub,world=case();base=evaluate_ice_motion(a,b,ua,ub,jacobian=False)[0]
    moved_a=a.copy();moved_b=b.copy();moved_a[:2]+=[3.,-2.];moved_b[:2]+=[3.,-2.]
    np.testing.assert_allclose(evaluate_ice_motion(moved_a,moved_b,ua,ub,jacobian=False)[0],base,rtol=0,atol=1e-8)
    moved_a=a.copy();moved_b=b.copy();moved_a[:3]*=1.7;moved_b[:3]*=1.7
    np.testing.assert_allclose(evaluate_ice_motion(moved_a,moved_b,ua,ub,jacobian=False)[0],base,rtol=0,atol=1e-8)


def test_individual_native_row_cameras_and_invalid_surface_domain_are_explicit():
    a,b,ua,ub,world=case();aa=np.repeat(a[None],3,axis=0);bb=np.repeat(b[None],3,axis=0)
    r=evaluate_ice_motion(aa,bb,ua,ub,jacobian=False)[0]
    np.testing.assert_allclose(r,0,rtol=0,atol=1e-8)
    wrong=a.copy();wrong[:3]=[0.,0.,-8.]
    with pytest.raises(ValueError,match='intersection|behind'):evaluate_ice_motion(wrong,b,ua,ub,jacobian=False)
