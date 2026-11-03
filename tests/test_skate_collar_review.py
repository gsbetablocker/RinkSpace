"""Independent triangle-section topology and signed boundary-gap checks."""
import importlib.util
from pathlib import Path

import numpy as np


PATH=Path(__file__).resolve().parents[1]/'calibration/program_tracking/body_tracking/skate_collar_review.py'
SPEC=importlib.util.spec_from_file_location('_collar_review',PATH)
REVIEW=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(REVIEW)


def prism(polygon):
    p=np.asarray(polygon,float);n=len(p)
    vertices=np.r_[np.c_[p,np.zeros(n)],np.c_[p,np.ones(n)]]
    faces=np.array([face for i in range(n) for face in ((i,(i+1)%n,n+i),((i+1)%n,n+(i+1)%n,n+i))])
    return vertices,faces


def test_closed_triangle_slice_keeps_concave_body_and_exact_inside_gap():
    body=np.array([[-.4,-.4],[.4,-.4],[.4,.4],[0.,.2],[-.4,.4]])
    vertices,faces=prism(body)
    loop,reason=REVIEW._section_polygon(vertices,faces,.41,1e-10)
    assert reason is None and not REVIEW._polygon_is_convex(loop,1e-10)
    inner=np.array([[-1.,-1.],[1.,-1.],[1.,1.],[-1.,1.]])
    gap,_,_=REVIEW._signed_gap(loop,inner,1e-10)
    np.testing.assert_allclose(gap,.6,atol=1e-12)


def test_outside_gap_uses_actual_polygon_distance_and_retains_negative_sign():
    inner=np.array([[-1.,-1.],[1.,-1.],[1.,1.],[-1.,1.]])
    body=np.array([[-.4,-.4],[1.2,-.4],[1.2,.4],[-.4,.4]])
    gap,body_point,wall_point=REVIEW._signed_gap(body,inner,1e-10)
    np.testing.assert_allclose(gap,-.2,atol=1e-12)
    np.testing.assert_allclose(body_point-wall_point,[.2,0.],atol=1e-12)


def test_open_wall_is_unknown_instead_of_closed_by_a_hull():
    vertices,faces=prism([[-1.,-1.],[1.,-1.],[1.,1.],[-1.,1.]])
    loop,reason=REVIEW._section_polygon(vertices,faces[:-2],.41,1e-10)
    assert loop is None and reason=='open_section'


def test_multiple_body_loops_remain_unknown():
    one,faces=prism([[-1.,-1.],[1.,-1.],[1.,1.],[-1.,1.]])
    loop,reason=REVIEW._section_polygon(np.r_[one,one+[4.,0.,0.]],np.r_[faces,faces+len(one)],.41,1e-10)
    assert loop is None and reason=='multiple_section_loops'


def test_self_crossing_section_is_unknown_and_nonconvex_inner_is_not_convexified():
    vertices,faces=prism([[-1.,-1.],[1.,1.],[1.,-1.],[-1.,1.]])
    loop,reason=REVIEW._section_polygon(vertices,faces,.41,1e-10)
    assert loop is None and reason=='self_intersecting_section'
    concave=np.array([[-1.,-1.],[1.,-1.],[0.,0.],[1.,1.],[-1.,1.]])
    assert not REVIEW._polygon_is_convex(concave,1e-10)


def test_coplanar_triangles_do_not_invent_a_section_boundary():
    vertices=np.array([[0.,0.,.41],[1.,0.,.41],[0.,1.,.41]])
    loop,reason=REVIEW._section_polygon(vertices,np.array([[0,1,2]]),.41,1e-10)
    assert loop is None and reason=='coplanar_triangle_at_section'


def test_nonconvex_inner_checks_body_edge_between_inside_endpoints_with_a_certified_bound():
    inner=np.array([[-1.,-1.],[1.,-1.],[1.,1.],[.2,1.],[.2,0.],[-.2,0.],[-.2,1.],[-1.,1.]])
    body=np.array([[-.7,.7],[.7,.7],[0.,-.5]])
    assert REVIEW._inside_polygon(body,inner).all()
    gap,lower,body_point,wall_point=REVIEW._signed_gap_general(body,inner,1e-10,1e-6)
    assert lower<=-.2<=gap+1e-12
    assert gap-lower<=1e-6
    np.testing.assert_allclose(gap,-.2,atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(body_point-wall_point),.2,atol=1e-6)
