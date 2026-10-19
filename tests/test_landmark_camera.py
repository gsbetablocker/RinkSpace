import numpy as np
from scipy.spatial.transform import Rotation
from rinkspace.landmark_camera import GroundCurve,model_equation,curve_residual,curve_residual_h,fit_camera_curves
from rinkspace.multisurface_camera import Camera,project_camera


def setup(radial=0):
    c=Camera(np.diag([1.,-1.,-1.]),np.array([0.,0.,10.]),900.,np.array([480.,270.]),radial_ray_k1=radial)
    groups=[]
    for i,centre in enumerate([[-2.,0.],[2.,1.]]):
        theta=np.linspace(0,2*np.pi,40)
        xy=np.asarray(centre)+np.c_[np.cos(theta),np.sin(theta)]
        kind,equation=model_equation(dict(kind='circle',points_model=xy.tolist()))
        observed,_=project_camera(c,np.c_[xy,np.zeros(len(xy))])
        groups.append(GroundCurve(0,str(i),kind,equation,observed,1.))
    return c,groups


def test_visible_circle_samples_are_not_ellipse_centres():
    c,groups=setup()
    # Sampled bounds need cardinal angles for exact model-circle construction.
    for g in groups:
        assert np.max(np.abs(curve_residual(c,g)))<.5
    initial=Camera(Rotation.from_rotvec([.012,-.018,.009]).as_matrix()@c.rotation,c.centre.copy(),960.,c.principal)
    fitted,report=fit_camera_curves(initial,groups)
    assert report['solver_converged']
    assert max(g['p95_px'] for g in report['groups'])<.5
    assert abs(fitted.focal-c.focal)<2.


def test_radial_coordinate_chain_inverts_observed_points():
    c,groups=setup(.15)
    for g in groups:assert np.max(np.abs(curve_residual(c,g)))<.5


def test_duplicating_circle_samples_does_not_multiply_group_weight():
    c,groups=setup()
    initial=Camera(Rotation.from_rotvec([.012,-.018,.009]).as_matrix()@c.rotation,c.centre.copy(),960.,c.principal)
    a,_=fit_camera_curves(initial,groups)
    duplicated=[GroundCurve(g.frame,g.physical_id,g.kind,g.equation_model,np.repeat(g.observed_xy,5,axis=0),g.sigma_px,g.group_weight) for g in groups]
    b,_=fit_camera_curves(initial,duplicated)
    assert np.allclose(a.H,b.H,atol=.03)


def test_flat_ellipse_end_error_is_geometric_not_sampson_underestimate():
    h=np.diag([500.,20.,1.]);equation=np.diag([1.,1.,-1.])
    error=curve_residual_h(h,'circle',equation,np.array([[800.,0.],[0.,30.],[0.,0.]]))
    assert np.allclose(error,[300.,10.,-20.],atol=1e-6)
