"""Image agreement must not depend on vertex spacing along the same edge."""
from types import SimpleNamespace

import numpy as np

from calibration.program_tracking.body_tracking.programme_whole_skate import WholeSkateSourceFactor


def test_same_outline_matches_between_vertices_and_after_edge_subdivision():
    rectangle = np.array([[0., 0.], [100., 0.], [100., 40.], [0., 40.]])
    split = np.array([[0., 0.], [13., 0.], [63., 0.], [100., 0.],
                      [100., 40.], [81., 40.], [9., 40.], [0., 40.]])
    results = []
    for vertices in (rectangle, split):
        edges = np.column_stack((np.arange(len(vertices)), np.roll(np.arange(len(vertices)), -1)))
        # A dense, exact rectangle boundary stands in for the silhouette raster.
        boundary = np.r_[np.c_[np.arange(101), np.zeros(101)],
                         np.c_[np.full(41, 100), np.arange(41)],
                         np.c_[np.arange(100, -1, -1), np.full(101, 40)],
                         np.c_[np.zeros(41), np.arange(40, -1, -1)]]
        factor = object.__new__(WholeSkateSourceFactor)
        factor.assembly = SimpleNamespace(edges=edges, raster=lambda pixels: [boundary])
        pairs, fraction, target, _ = factor._segment_matches(vertices, rectangle)
        projected = vertices[pairs[:, 0]] + fraction[:, None] * (vertices[pairs[:, 1]]-vertices[pairs[:, 0]])
        np.testing.assert_allclose(projected, target, atol=2.e-12, rtol=0.)
        results.append(projected)
    np.testing.assert_allclose(results[0], results[1], atol=2.e-12, rtol=0.)


def test_partial_source_arc_does_not_collapse_unseen_half_of_boot():
    vertices=np.array([[0.,0.],[100.,0.],[100.,40.],[0.,40.]])
    edges=np.array([[0,1],[1,2],[2,3],[3,0]])
    boundary=np.r_[np.c_[np.arange(101),np.zeros(101)],
        np.c_[np.full(41,100),np.arange(41)],np.c_[np.arange(101),np.full(101,40)],
        np.c_[np.zeros(41),np.arange(41)]]
    factor=object.__new__(WholeSkateSourceFactor)
    factor.assembly=SimpleNamespace(edges=edges,raster=lambda pixels:[boundary])
    # Only the source top arc is visible; the other 60 samples are a bad mask
    # completion near the foreground shoe and must exert exactly zero force.
    observed=np.r_[np.c_[np.linspace(10,90,20),np.zeros(20)],np.full((60,2),150.)]
    valid=np.r_[np.ones(20,bool),np.zeros(60,bool)]
    pair,fraction,target,_=factor._segment_matches(vertices,observed,valid,partial=True)
    predicted=vertices[pair[:,0]]+fraction[:,None]*(vertices[pair[:,1]]-vertices[pair[:,0]])
    np.testing.assert_allclose(predicted,target,atol=1.e-12)
    assert np.all(target[:,1]==0.)
