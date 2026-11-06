"""Finite native contact candidates must not invent a filled planar footprint."""
import numpy as np
from calibration.program_tracking.body_tracking.source_contact_image import polyline_nearest
from calibration.program_tracking.body_tracking.spin_grasp_factors import region_terms

def test_open_steel_band_does_not_fill_the_convex_hull_between_branches():
    source=np.array([[0.,0.],[0.,10.],[10.,10.]])
    points=np.array([[5.,5.],[0.,5.],[12.,10.]])
    closest,D,segments,interior=polyline_nearest(points,source)
    np.testing.assert_allclose(np.linalg.norm(points-closest,axis=1),[5.,0.,2.])
    r,_,_=region_terms(points[0],dict(centre_uv=closest[0],region_shape_px2=np.eye(2),objective_weight=1.,reliability=1.))
    assert np.linalg.norm(r)>0 # The centre is inside the hull but outside the observed curve band.

def test_closest_source_segment_derivative_interior_and_endpoint():
    source=np.array([[0.,0.],[10.,0.]])
    for point in [np.array([4.,3.]),np.array([-2.,3.])]:
        closest,D,_,_=polyline_nearest(point[None],source);direction=np.array([.37,-.23]);h=1e-5
        def error(p):return p-polyline_nearest(p[None],source)[0][0]
        np.testing.assert_allclose((error(point+h*direction)-error(point-h*direction))/(2*h),D[0]@direction,atol=1e-9)
