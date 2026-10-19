"""Independent affine-face oracle for the unchanged finite source rectangle."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

PATH = Path(__file__).resolve().parents[1]/'calibration/solver_runtime/legacy/structured_programme_source_candidate/normal_scanline_human_query_acceleration.py'
spec = importlib.util.spec_from_file_location('rectangle_test_kernel', PATH)
kernel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kernel)


def query(curve, uv, normal, bounds, tb, **kw):
    curve = np.asarray(curve, float)
    return kernel.finite_source_rectangle_projected_queries(curve, np.ones(len(curve)),
        np.ones(len(curve), bool), np.ones(len(curve)-1, bool),
        np.asarray(uv, float), np.asarray(normal, float), np.asarray(bounds, float),
        tangent_bounds=np.asarray(tb, float), **kw)


def oracle(curve, uv, normal, bounds, tb):
    tangent = [-normal[1], normal[0]]
    best = np.inf
    for a, b in zip(curve[:-1], curve[1:]):
        if np.sum((b-a)**2) <= 1e-4:
            continue
        n, t = np.dot(a-uv, normal), np.dot(a-uv, tangent)
        dn, dt = np.dot(b-a, normal), np.dot(b-a, tangent)
        lines = np.array([[0., 0.], [n-bounds[1], dn], [bounds[0]-n, -dn],
                          [t-tb[1], dt], [tb[0]-t, -dt]])
        stations = [0., 1.]
        for i in range(5):
            for j in range(i):
                slope = lines[i, 1]-lines[j, 1]
                if slope:
                    s = (lines[j, 0]-lines[i, 0])/slope
                    if 0 <= s <= 1:
                        stations.append(s)
        best = min(best, min(np.max(lines[:, 0]+s*lines[:, 1]) for s in stations))
    return best


def test_zero_deficit_normal_slab_uses_literal_bound_for_true_and_frozen_queries():
    uv=np.array([[1347.123456789,523.987654321]])
    n=np.array([[.6,.8]]);t=np.c_[-n[:,1],n[:,0]]
    a=uv[0]+4*n[0]-2*t[0];d=-10*n[0]-20*t[0]
    curve=np.stack((a,a+d));bounds=np.array([[-3.,3.]]);tb=np.array([[-10.,10.]])
    nominal=query(curve,uv,n,bounds,tb)
    assert nominal['source_support_branch_signature'].tolist()==[[0,0,0,2]]
    assert nominal['signed_normal_px'][0]==3. and nominal['normal_violation_px'][0]==0.
    frozen=dict(frozen_segments=nominal['nominal_finite_segments'],
        frozen_segment_categories=nominal['nominal_finite_segment_categories'],
        frozen_support_faces=nominal['nominal_source_support_faces'],
        frozen_station_branches=nominal['nominal_source_station_branches'])
    for h in (1e-3,1e-6,1e-8):
        plus=query(curve+h*n,uv,n,bounds,tb,**frozen)
        minus=query(curve-h*n,uv,n,bounds,tb,**frozen)
        assert plus['signed_normal_px'][0]==minus['signed_normal_px'][0]==3.
        assert (plus['signed_normal_px'][0]-minus['signed_normal_px'][0])/(2*h)==0.
        true=query(curve+h*n,uv,n,bounds,tb)
        assert true['signed_normal_px'][0]==3.


@pytest.mark.parametrize('branch,delta,expected',[(2,-1.,3.),(2,1.,-3.),(5,-1.,-3.),(5,1.,3.),
    (3,-1.,10.),(3,1.,-10.),(6,-1.,-10.),(6,1.,10.)])
def test_exact_slab_identity_respects_original_axis_and_delta_sign(branch,delta,expected):
    normals=np.array([[1.,0.]]);d=np.array([[delta,delta]])
    args=(normals,np.array([[-3.,3.]]),np.array([[-10.,10.]]),d,np.array([0]))
    before=(np.array([99.]),np.array([98.]))
    for category,face in ((0,0),(1,0),(0,5)):
        normal,tangent=kernel._rectangle_exact_constant_support_axes(*before,*args,
            np.array([category]),np.array([face]),np.array([branch]))
        if category==0 and face==0:
            assert (normal[0] if branch in (2,5) else tangent[0])==expected
            assert (tangent[0] if branch in (2,5) else normal[0])==(98. if branch in (2,5) else 99.)
        else:
            assert np.array_equal(normal,before[0]) and np.array_equal(tangent,before[1])


def test_exact_slab_kernel_matches_independent_face_intersection_oracle():
    assert kernel._rectangle_kernel is not None, kernel._error
    rng = np.random.default_rng(441122)
    curve = np.cumsum(rng.normal(size=(31, 2))*3., axis=0)
    uv = rng.normal(size=(59, 2))*10.
    theta = rng.uniform(-np.pi, np.pi, len(uv))
    normal = np.c_[np.cos(theta), np.sin(theta)]
    bounds = np.sort(rng.normal(size=(len(uv), 2))*2., axis=1)
    tb = np.sort(rng.normal(size=(len(uv), 2))*2., axis=1)
    bounds[0] = tb[0] = 0.
    normal[1:3] = [[1., 0.], [0., 1.]]
    expected = np.array([oracle(curve, u, n, b, t) for u, n, b, t in zip(uv, normal, bounds, tb)])
    native = query(curve, uv, normal, bounds, tb)
    reference = query(curve, uv, normal, bounds, tb, native=False)
    np.testing.assert_allclose(native['source_support_distance_px'], expected, atol=1e-13, rtol=1e-13)
    np.testing.assert_allclose(native['residual'], reference['residual'], atol=1e-13, rtol=1e-13)
    np.testing.assert_array_equal(native['nominal_finite_segments'], reference['nominal_finite_segments'])
    replay = query(curve, uv, normal, bounds, tb, frozen_segments=native['nominal_finite_segments'])
    np.testing.assert_array_equal(native['residual'], replay['residual'])


def test_remote_tangent_birth_cannot_replace_nearby_original_support():
    uv, normal, bounds, tb = [[0., 0.]], [[0., 1.]], [[-1., 1.]], [[-1., 1.]]
    # Nearby segment is outside the tangent slab, while a distant segment
    # acquires a ray crossing under an arbitrarily small displacement.
    def curve(e):
        return np.array([[2., -2.], [2., 2.], [50., 20.], [e, 20.], [-e, 21.]])
    values = [query(curve(e), uv, normal, bounds, tb)['source_support_distance_px'][0]
              for e in (-1e-6, 0., 1e-6)]
    np.testing.assert_array_equal(values, [1., 1., 1.])


def test_zero_and_asymmetric_support_and_finite_endpoints_stay_exact():
    value = query([[-2., 4.], [2., 4.]], [[0., 0.], [5., 0.]], [[0., 1.]]*2,
                  [[0., 0.], [-2., 3.]], [[0., 0.], [-1., 2.]])
    np.testing.assert_allclose(value['source_support_distance_px'], [4., 1.], atol=0., rtol=0.)
    assert value['nearest_finite_segment_station'][1] == 1.
    with pytest.raises(ValueError, match='Nominal'):
        query([[-2., 4.], [2., 4.]], [[0., 0.]], [[0., 1.]], [[0., 0.]], [[0., 0.]],
              frozen_segments=np.array([8]))


def test_fixed_nominal_segment_derivative_agrees_with_local_dynamic_value():
    uv, normal, bounds, tb = [[0., 0.]], [[1., 0.]], [[-1., 1.]], [[-.5, .5]]
    curve = np.array([[4., -2.], [6., 2.], [20., 4.]])
    nominal = query(curve, uv, normal, bounds, tb)['nominal_finite_segments']
    direction = np.array([[2., -.5], [-.5, 1.], [1., -2.]])
    def derivative(h, frozen):
        plus = query(curve+h*direction, uv, normal, bounds, tb, frozen_segments=frozen)
        minus = query(curve-h*direction, uv, normal, bounds, tb, frozen_segments=frozen)
        return (plus['residual']-minus['residual'])/(2*h)
    np.testing.assert_allclose(derivative(1e-6, nominal), derivative(1e-7, None), atol=1e-8, rtol=1e-8)


def test_coordinate_stencil_cannot_mix_in_computational_endpoint_clamp():
    uv, normal, bounds, tb = [[0., 0.]], [[1., 0.]], [[-1., 1.]], [[-1., 1.]]
    curve=np.array([[4., -4.0001], [6., -2.0001]])
    base=query(curve,uv,normal,bounds,tb)
    assert base['nominal_finite_segment_categories'][0]==0
    # Wide per-coordinate FD crosses t=1, while the coupled local stencil
    # remains interior; freeze the nominal category, never the source box.
    direction=np.array([[0., 1000.], [0., 1000.]])
    replay=dict(frozen_segments=base['nominal_finite_segments'],
                frozen_segment_categories=base['nominal_finite_segment_categories'])
    def fd(h,kw):
        return (query(curve+h*direction,uv,normal,bounds,tb,**kw)['residual']-
                query(curve-h*direction,uv,normal,bounds,tb,**kw)['residual'])/(2*h)
    np.testing.assert_allclose(fd(1e-6,replay),fd(1e-9,{}),atol=1e-6,rtol=1e-6)
    assert np.max(abs(fd(1e-6,{})-fd(1e-6,replay)))>100.


def test_exact_active_endpoint_face_classifies_one_ulp_interior_station():
    # Actual q757 vertex1048: near-normal-parallel segment forces its far
    # endpoint, while slab arithmetic rounds station just below one.
    curve=np.array([[-16.69221541669233,-2.283254537390855],
        [-16.68941819182995,-7.971479415899186]])
    uv,normal=[[0.,0.]],[[1.,0.]]
    bounds=[[-2.873291316528231,2.8732913165283045]]
    tb=[[-2.8732913165283045,2.8732913165283045]]
    base=query(curve,uv,normal,bounds,tb)
    assert base['nominal_source_support_faces'][0]==1
    assert base['nominal_finite_segment_categories'][0]==2
    replay=query(curve,uv,normal,bounds,tb,frozen_segments=base['nominal_finite_segments'],
        frozen_segment_categories=base['nominal_finite_segment_categories'],
        frozen_support_faces=base['nominal_source_support_faces'],
        frozen_station_branches=base['nominal_source_station_branches'])
    np.testing.assert_allclose(base['residual'],replay['residual'],atol=3e-12,rtol=0.)
    actual=query(curve+np.array([[0.,-1.],[0.,1.]])*1e-7,uv,normal,bounds,tb)
    np.testing.assert_array_equal(actual['source_support_branch_signature'][:,:3],
                                  base['source_support_branch_signature'][:,:3])


def test_frozen_positive_face_continues_smoothly_across_zero_deficit_coordinate_stencil():
    # Actual native35 has a positive 3.56e-5-pixel deficit. Individual raw
    # coordinate stencils can cross zero although its coupled direction does
    # not. Clamping the frozen face back to zero smears the tangent derivative.
    curve=np.array([[0.,2.00002],[2.00002,0.]])
    uv,normal,bounds,tb=[[0.,0.]],[[1.,0.]],[[-1.,1.]],[[-1.,1.]]
    base=query(curve,uv,normal,bounds,tb)
    assert base['nominal_source_support_faces'][0]==6
    replay=dict(frozen_segments=base['nominal_finite_segments'],
        frozen_segment_categories=base['nominal_finite_segment_categories'],
        frozen_support_faces=base['nominal_source_support_faces'],
        frozen_station_branches=base['nominal_source_station_branches'])
    direction=np.array([[0.,1.],[0.,1.]])
    def carriers(h,kw):
        q=query(curve+h*direction,uv,normal,bounds,tb,**kw)
        return np.c_[q['signed_normal_px'],q['coverage_tangent_px']]
    wide=(carriers(1e-3,replay)-carriers(-1e-3,replay))/(2e-3)
    local=(carriers(1e-7,{})-carriers(-1e-7,{}))/(2e-7)
    np.testing.assert_allclose(wide,[[.5,.5]],atol=2e-10,rtol=0.)
    np.testing.assert_allclose(wide,local,atol=1e-8,rtol=0.)
    actual=query(curve-1e-3*direction,uv,normal,bounds,tb)
    assert actual['source_support_distance_px'][0]==0.
