import numpy as np

from rinkspace.session.projected_curve_queries import projected_curve_queries


def test_line_subdivision_does_not_restart_spacing_or_add_vertex_queries():
    ends=np.array([[20.,50.],[180.,50.]])
    coarse=projected_curve_queries(ends,np.ones(2,bool),(120,220),24.,20.)
    dense_line=np.linspace(ends[0],ends[1],997)
    dense=projected_curve_queries(dense_line,np.ones(997,bool),(120,220),24.,20.)
    np.testing.assert_allclose(coarse['points_uv'],np.c_[20.+24*np.arange(7),np.full(7,50.)])
    np.testing.assert_allclose(dense['points_uv'],coarse['points_uv'],atol=1e-10)
    np.testing.assert_allclose(dense['normals'],coarse['normals'])
    assert not dense['is_axis_extremum'].any()


def test_flat_ellipse_keeps_horizontal_and_vertical_axis_extrema_exactly_outward():
    angle=np.linspace(0.,2*np.pi,257);center=np.array([400.,150.])
    ellipse=center+np.c_[350*np.cos(angle),3*np.sin(angle)]
    original=ellipse.copy()
    result=projected_curve_queries(ellipse,np.ones(len(ellipse),bool),(300,800),24.,40.,
                                  closed=True,center_uv=center)
    extrema=result['is_axis_extremum'];points=result['points_uv'][extrema];normals=result['normals'][extrema]
    for point,normal in [([50.,150.],[-1.,0.]),([750.,150.],[1.,0.]),
                         ([400.,147.],[0.,-1.]),([400.,153.],[0.,1.])]:
        index=np.argmin(np.linalg.norm(points-point,axis=1))
        np.testing.assert_allclose(points[index],point,atol=1e-12)
        np.testing.assert_array_equal(normals[index],normal)
    assert extrema.sum()==4
    assert np.all(np.diff(result['polyline_stations'])>=0.)
    # The regular seam query at max-x must use the exact extremum direction.
    right=np.isclose(result['points_uv'][:,0],750.,rtol=0.,atol=1e-10)
    assert right.sum()==1 and result['is_axis_extremum'][right].all()
    np.testing.assert_array_equal(ellipse,original)


def test_invalid_span_cannot_connect_or_invent_queries_across_missing_support():
    line=np.c_[10.+np.arange(0.,201.,20.),np.full(11,60.)]
    valid=np.ones(11,bool);valid[4:7]=False
    result=projected_curve_queries(line,valid,(140,260),24.,20.)
    np.testing.assert_allclose(result['points_uv'][:,0],[10.,34.,58.,150.,174.,198.])
    assert not np.any((result['points_uv'][:,0]>70.)&(result['points_uv'][:,0]<150.))
    assert np.all(np.diff(result['polyline_stations'])>0.)


def test_search_intervals_stop_at_the_source_four_pixel_halo_without_width_gate():
    line=np.array([[2.,10.],[2.,90.]])
    result=projected_curve_queries(line,np.ones(2,bool),(100,100),24.,7.)
    assert len(result['points_uv'])==4
    ends=result['points_uv'][:,None]+result['normals'][:,None]*result['intervals'][:,:,None]
    assert (ends>=4.-1e-12).all() and (ends<=95.+1e-12).all()
    assert np.all(result['intervals'][:,1]-result['intervals'][:,0]<20.)
    # A parallel search wholly outside the inset image has no real RGB support.
    horizontal=projected_curve_queries([[10.,2.],[90.,2.]],[True,True],(100,100),24.,1.)
    assert horizontal['points_uv'].shape==(0,2)


def test_segment_with_both_vertices_outside_still_reads_the_viewport_crossing():
    result=projected_curve_queries([[-20.,40.],[120.,40.]],[True,True],(100,100),24.,12.)
    np.testing.assert_allclose(result['points_uv'],np.c_[[24.,48.,72.],np.full(3,40.)])
    np.testing.assert_array_equal(result['normals'],np.tile([0.,1.],(3,1)))
    assert np.all((result['polyline_stations']>0.)&(result['polyline_stations']<1.))


def test_closed_curve_uses_finite_offscreen_center_and_retains_invalid_gap():
    angle=np.linspace(0.,2*np.pi,257);center=np.array([-30.,50.])
    ellipse=center+np.c_[70.*np.cos(angle),20.*np.sin(angle)]
    valid=np.ones(len(ellipse),bool);valid[10:20]=False
    result=projected_curve_queries(ellipse,valid,(100,100),8.,12.,closed=True,center_uv=center)
    assert len(result['points_uv'])>0
    assert not np.any((result['polyline_stations']>9.)&(result['polyline_stations']<20.))
    assert np.all(np.sum((result['points_uv']-center)*result['normals'],axis=1)>=0.)
    assert np.all((result['points_uv']>=[0.,0.])&(result['points_uv']<=[99.,99.]))


def test_zero_segments_preserve_runs_but_invalid_and_corner_contacts_split():
    from rinkspace.session.projected_curve_queries import _visible_runs
    vertices=np.array([[10.,10.],[30.,10.],[30.,10.],[50.,10.],[70.,10.],[90.,10.]])
    valid=np.array([True,True,True,True,False,True])
    runs=_visible_runs(vertices,valid,100,100)
    assert [[piece[0]for piece in run]for run in runs]==[[0,2]]
    corner=np.array([[10.,10.],[-10.,-10.],[0.,0.],[-10.,10.],[10.,10.]])
    runs=_visible_runs(corner,np.ones(len(corner),bool),100,100)
    assert [[piece[0]for piece in run]for run in runs]==[[0],[3]]
