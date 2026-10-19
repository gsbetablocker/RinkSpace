"""Synthetic known-wall derivatives; no source-image or material qualification claim."""
from copy import deepcopy

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from calibration.program_tracking.board_motion_factor import evaluate_board_motion
from calibration.program_tracking.board_motion_geometry import evaluate_board_motion_geometry
from calibration.program_tracking.first_shot_source_kernel import _interfaces
from calibration.program_tracking.projection import project_fast


def problem():
    _interfaces()
    from calibration.program_tracking.reference_board_pair import ReferenceBoardPair
    bottom = np.array([[-10., -10., 0.], [10., -10., 0.],
                       [10., 10., 0.], [-10., 10., 0.]])
    angle = .17
    rotation = np.array([[np.cos(angle), -np.sin(angle)],
                         [np.sin(angle), np.cos(angle)]])
    bottom[:, :2] = bottom[:, :2]@rotation.T
    scene = {'ice_boundary': {'points_xyz': bottom},
             'board_surface': {'bottom_edge_xyz': bottom,
                               'top_edge_xyz': bottom+np.array([0., 0., 4.])}}
    a = np.zeros(25)
    a[:3] = [0., 0., 3.]
    a[3] = np.deg2rad(95.)
    a[6:8] = np.log([1200., 1210.])
    a[8:11] = [960., 540., .4]
    a[11:15] = [.01, -.005, .0001, -.0002]
    b = a.copy()
    b[:3] += [.8, -.4, .15]
    b[4] = .003
    b[6:8] += .001
    fraction = np.array([.3, .5, .7])
    world = bottom[2]+fraction[:, None]*(bottom[3]-bottom[2])
    world[:, 2] = [1.5, 2., 2.5]
    u, v = project_fast(a, world)[0], project_fast(b, world)[0]
    surface = ReferenceBoardPair(u, v, scene, np.array([2]), a, b)
    return surface, a, b, u, v, world


def moved_surface(surface, a, b, u, v, delta):
    from calibration.program_tracking.reference_board_pair import ReferenceBoardPair
    scene = deepcopy(surface.scene)
    bottom = np.asarray(surface.bottom).copy()
    bottom[:, :2] += np.asarray(delta).reshape(-1, 2)
    top = np.asarray(surface.top).copy()
    top[:, :2] = bottom[:, :2]
    scene['ice_boundary']['points_xyz'] = bottom
    scene['board_surface']['bottom_edge_xyz'] = bottom
    scene['board_surface']['top_edge_xyz'] = top
    return ReferenceBoardPair(u, v, scene, surface.allowed_panels, a, b)


def residual(surface, a, b, u, v):
    return evaluate_board_motion_geometry(surface, a, b, u, v, jacobian=False)[0]


def test_every_actual_panel_endpoint_derivative_and_legacy_camera_identity():
    surface, a, b, u, v, world = problem()
    v = v+np.array([.2, -.3])
    before = [x.copy() for x in (a, b, u, v, surface.bottom, surface.top)]
    r, ja, jb, geometry, hit = evaluate_board_motion_geometry(surface, a, b, u, v)
    legacy = evaluate_board_motion(surface, a, b, u, v)
    for actual, expected in zip((r, ja, jb, hit), legacy, strict=True):
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_allclose(hit, world, rtol=0, atol=1e-9)
    assert geometry.shape == (6, 8)
    expected = geometry.toarray()
    h = 1e-4
    for column in range(8):
        delta = np.zeros((4, 2))
        delta.flat[column] = h
        plus = moved_surface(surface, a, b, u, v, delta)
        minus = moved_surface(surface, a, b, u, v, -delta)
        # All rays remain strictly inside this same explicitly allowed panel.
        for candidate in (plus, minus):
            hits = candidate.source_hits(np.r_[a[:3]/100, a[3:8],
                a[8:11]/1000, a[11:15]], u)
            np.testing.assert_array_equal(hits.panel, 2)
        numerical = (residual(plus, a, b, u, v)-residual(minus, a, b, u, v))/(2*h)
        np.testing.assert_allclose(numerical, expected[:, column], rtol=3e-6, atol=3e-7)
    assert np.linalg.norm(expected[:, 4:8]) > 1.
    np.testing.assert_array_equal(expected[:, :4], 0.)
    for current, prior in zip((a, b, u, v, surface.bottom, surface.top), before, strict=True):
        np.testing.assert_array_equal(current, prior)


def test_low_dimensional_geometry_chain_dense_sparse_and_row_cameras():
    surface, a, b, u, v, _ = problem()
    aa = np.repeat(a[None, :], len(u), axis=0)
    bb = np.repeat(b[None, :], len(u), axis=0)
    aa[:, 4] += [-.0001, 0., .0001]
    bb[:, 4] += [-.0002, 0., .0002]
    chain = np.zeros((4, 2, 3))
    chain[:, 0, 0] = 1.       # common world-X translation
    chain[:, 1, 1] = 1.       # common world-Y translation
    chain[2, :, 2] = [.4, -.6] # one actual endpoint changes panel orientation
    chain[3, :, 2] = [-.2, .3]
    raw = evaluate_board_motion_geometry(surface, aa, bb, u, v)[3]
    dense = evaluate_board_motion_geometry(surface, aa, bb, u, v,
        boundary_xy_jacobian=chain)[3]
    sparse = evaluate_board_motion_geometry(surface, aa, bb, u, v,
        boundary_xy_jacobian=csr_matrix(chain.reshape(8, 3)))[3]
    np.testing.assert_allclose(dense.toarray(), raw@chain.reshape(8, 3), rtol=0, atol=1e-14)
    np.testing.assert_array_equal(dense.toarray(), sparse.toarray())
    h = 1e-4
    for column in range(3):
        delta = h*chain[:, :, column]
        plus = moved_surface(surface, a, b, u, v, delta)
        minus = moved_surface(surface, a, b, u, v, -delta)
        numerical = (residual(plus, aa, bb, u, v)-residual(minus, aa, bb, u, v))/(2*h)
        np.testing.assert_allclose(numerical, dense.toarray()[:, column], rtol=3e-6, atol=3e-7)


def test_missing_geometry_chain_is_explicit_and_original_visibility_guards_remain():
    surface, a, b, u, v, _ = problem()
    r, ja, jb, geometry, _ = evaluate_board_motion_geometry(surface, a, b, u, v,
        jacobian=False)
    assert ja.shape == jb.shape == geometry.shape == (len(r), 0)
    with pytest.raises(ValueError, match='2V'):
        evaluate_board_motion_geometry(surface, a, b, u, v,
            boundary_xy_jacobian=np.zeros((6, 3)))
    with pytest.raises(ValueError, match='Finite'):
        evaluate_board_motion_geometry(surface, a, b, u, v,
            boundary_xy_jacobian=np.full((8, 1), np.nan))
    surface._allowed[:] = False
    with pytest.raises(ValueError, match='first_hit_outside_verified_print_arc'):
        evaluate_board_motion_geometry(surface, a, b, u, v)
    surface._allowed[2] = True
    outside = b.copy()
    outside[:3] = [0., -20., 3.]
    with pytest.raises(ValueError, match='occluded|outside_verified'):
        evaluate_board_motion_geometry(surface, a, outside, u, v)
