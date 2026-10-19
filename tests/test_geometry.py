import numpy as np
import pytest
from rinkspace.geometry import project, fit_homography, heldout_audit, PlaneMap, source_pixel_errors


def test_homography_round_trip():
    h = np.array([[1.1,.04,11],[.03,.9,4],[.0003,-.0001,1]])
    p = np.array([[3,4],[35,40],[140,300]],float)
    np.testing.assert_allclose(project(np.linalg.inv(h), project(h,p)), p, atol=1e-10)


def test_fit_known_planar_mapping():
    p = np.array([[0,0],[100,0],[0,100],[100,100],[50,30],[30,70]],float)
    h = np.array([[1.2,.1,2],[.03,1.1,3],[.0005,.0002,1]])
    q = project(h,p)
    fit, mask = fit_homography(p,q)
    assert mask.sum() == len(p)
    np.testing.assert_allclose(project(fit,p),q,atol=1e-4)


def test_collinear_points_rejected():
    p = np.array([[0,0],[1,0],[2,0],[3,0]])
    with pytest.raises(ValueError,match="Collinear"): fit_homography(p,p)


@pytest.mark.parametrize("h", [np.zeros((3,3)), np.ones((3,3)), np.eye(2)])
def test_invalid_h_rejected(h):
    with pytest.raises(ValueError): project(h, [[0,0]])


def test_horizon_rejected():
    with pytest.raises(ValueError,match="horizon"):
        project([[1,0,0],[0,1,0],[1,0,1]],[[-1,3]])


def test_metric_errors_still_source_pixels():
    h = np.diag([.01,.01,1])
    e = source_pixel_errors(h,[[100,100]],[[1.02,1]])
    assert e[0] == pytest.approx(2)


def test_holdout_physical_identity_leak_rejected():
    a=[[0,0],[100,0],[100,100],[0,100]]; c=[[20,20],[80,20],[80,80],[20,80]]
    with pytest.raises(ValueError,match="leaked"):
        heldout_audit(np.eye(3),a,["a","b","c","d"],c,c,["a","f","g","h"])


def test_holdout_neighborhood_leak_rejected():
    a=[[0,0],[100,0],[100,100],[0,100]]; c=[[1,1],[80,20],[80,80],[20,80]]
    with pytest.raises(ValueError,match="neighborhood"):
        heldout_audit(np.eye(3),a,["a","b","c","d"],c,c,["e","f","g","h"])


def test_holdout_is_not_scale_verification():
    a=[[0,0],[100,0],[100,100],[0,100]]; c=[[20,20],[80,20],[80,80],[20,80]]
    r=heldout_audit(np.eye(3),a,["a","b","c","d"],c,c,["e","f","g","h"])
    assert r['max_source_px']==pytest.approx(0)
    assert not r['metric_scale_verified_by_this_test']


@pytest.mark.parametrize("state",["airborne","unknown","occluded"])
def test_airborne_unknown_not_plane_contacts(state):
    with pytest.raises(ValueError,match="on-ice"):
        PlaneMap(np.eye(3)).map_contacts([[20,20]],[state])


def test_extrapolation_is_explicit():
    m=PlaneMap(np.eye(3),support_polygon=np.array([[0,0],[100,0],[100,100],[0,100]]))
    with pytest.raises(ValueError,match="outside"):m.map_contacts([[101,20]],["on_ice"])
    assert m.map_contacts([[101,20]],["on_ice"],allow_extrapolation=True)[0,0] == pytest.approx(101)


def test_projective_is_not_euclidean():
    with pytest.raises(ValueError,match="Projective"):PlaneMap(np.eye(3)).require_euclidean()


def test_geometry_level_unit_contract():
    with pytest.raises(ValueError):PlaneMap(np.eye(3),"metric_assumed","reference_px")
    PlaneMap(np.eye(3),"metric_assumed","m").require_euclidean()
