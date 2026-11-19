"""An uncertain source foot label cannot pull the hidden shoe onto its partner."""
from types import SimpleNamespace

import numpy as np

from calibration.program_tracking.body_tracking.visible_skate_union import nearest_visible_edges,reviewed_arc_rows
from calibration.program_tracking.body_tracking.programme_whole_skate import ambiguous_shoe_frames


def test_visible_arc_uses_nearest_physical_boot_regardless_of_source_label():
    left=np.array([[0.,0.],[40.,0.],[40.,20.],[0.,20.]])
    right=left+[80.,0.]
    assembly=SimpleNamespace(edges=np.array([[0,1],[1,2],[2,3],[3,0]]),raster=lambda pixels:[pixels])
    # Sample exact edge midpoints, so the model raster surrogate keeps them.
    assembly.raster=lambda pixels:[np.r_[pixels,pixels[assembly.edges].mean(1)]]
    observed=np.c_[np.linspace(82,118,40),np.zeros(40)]
    errors=[nearest_visible_edges(assembly,p,observed)[2] for p in (left,right)]
    np.testing.assert_allclose(errors[1],0.,atol=1.e-12)
    assert np.all(errors[1]<errors[0])
    swapped=[nearest_visible_edges(assembly,p,observed)[2] for p in (right,left)]
    np.testing.assert_allclose(np.minimum(*errors),np.minimum(*swapped))


def test_source_identity_uncertainty_spreads_within_a_shot_only():
    role=np.ones((12,2),int);role[5,0]=0
    packet=dict(role=role,visible=np.ones((12,2),bool),
        partial_boundary=np.zeros((12,2),bool),shot_ids=np.r_[np.zeros(6),np.ones(6)])
    uncertain=ambiguous_shoe_frames(packet)
    assert uncertain[1:6].all()
    assert not uncertain[6:].any()


def test_reviewed_arcs_keep_gaps_unknowns_and_correlated_measure():
    a=dict(id='a',type='boot_outer_edge',points_xy=[[10,20],[20,20]],uncertainty_radius_px=8)
    b=dict(id='b',type='blade_outer_edge',points_xy=[[60,70],[70,70],[80,70]],uncertainty_radius_px=5)
    unknown=dict(a,type='unknown')
    rejected=dict(b,use_for_fitting=False)
    doc=dict(frames=[dict(source_pts=256,arcs=[a,unknown,b,rejected]),dict(source_pts=512,arcs=[a])])
    rows=reviewed_arc_rows(doc,np.array([0,256,512]),[1])
    assert len(rows)==1 and rows[0][0]==1
    _,points,radius,scale,material,groups=rows[0]
    np.testing.assert_array_equal(points,[[10,20],[20,20],[60,70],[70,70],[80,70]])
    np.testing.assert_array_equal(radius,[8,8,5,5,5])
    np.testing.assert_allclose(np.sum(scale[:2]**2),9)
    np.testing.assert_allclose(np.sum(scale[2:]**2),9)
    np.testing.assert_array_equal(material,[0,0,1,1,1])
    np.testing.assert_array_equal(groups,[0,0,1,1,1])


def test_blade_target_cannot_match_a_nearer_boot_edge():
    pixels=np.array([[0.,0.],[40.,0.],[40.,20.],[0.,20.]])
    assembly=SimpleNamespace(edges=np.array([[0,1],[2,3]]),raster=lambda p:[p])
    # The true blade edge is hidden in the initial raster: keep that material
    # available for correction instead of substituting the wrong upper.
    pair,fraction,error=nearest_visible_edges(assembly,pixels,np.array([[20.,1.]]),np.array([False,True]))
    np.testing.assert_array_equal(pair,[[2,3]])
    np.testing.assert_allclose(fraction,.5)
    np.testing.assert_allclose(error,19.**2)


def test_material_interface_can_remain_inside_whole_shoe_silhouette():
    pixels=np.array([[0.,0.],[40.,0.],[40.,30.],[0.,30.],
                     [10.,15.],[30.,15.],[30.,25.],[10.,25.]])
    pairs=np.array([[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4]])
    def raster(p,faces=None):
        take=pairs[:4] if faces is None else pairs[4:]
        return [np.r_[p[take].reshape(-1,2),p[take].mean(1)]]
    assembly=SimpleNamespace(edges=pairs,raster=raster)
    allowed=np.arange(8)>=4
    pair,fraction,error=nearest_visible_edges(assembly,pixels,np.array([[20.,15.]]),
        allowed,boundary_faces=np.array([[4,5,6],[4,6,7]]))
    np.testing.assert_array_equal(pair,[[4,5]])
    np.testing.assert_allclose(error,0.)
    doc=dict(frames=[dict(source_pts=256,arcs=[dict(id='gray',type='heel_block_edge',
        points_xy=[[10,15],[30,15]],uncertainty_radius_px=4)])])
    rows=reviewed_arc_rows(doc,np.array([0,256]),[1])
    np.testing.assert_array_equal(rows[0][4],[2,2])
