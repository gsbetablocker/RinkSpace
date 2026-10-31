import numpy as np
from rinkspace.session.material_ring_bands import compatible_query_consensus


def test_missing_arc_does_not_veto_uncontested_measured_band():
    result=compatible_query_consensus([{'candidate_ids':[0,1,2]}, {'candidate_ids':[1,2,3]}],np.arange(4))
    assert result['selected']==[0,1,2,3]
    assert result['conflicting_queries']=={}


def test_competing_bands_at_same_query_are_not_averaged_or_selected():
    result=compatible_query_consensus([{'candidate_ids':[0,1,2]}, {'candidate_ids':[0,3,2]}],np.array([0,1,2,1]))
    assert result['selected']==[0,2]
    assert result['conflicting_queries']=={'1':[1,3]}


def test_radial_annulus_queries_do_not_cross_centre_or_swap_opposite_arc():
    from rinkspace.vision_programme_ice_review import _source_queries
    center=np.array([240.,220.]);angle=np.linspace(0.,2*np.pi,129)
    ellipse=center+np.c_[160*np.cos(angle),30*np.sin(angle)]
    points,directions,intervals=_source_queries(
        {'profile_direction':'outward_radial','_projected_center':center},ellipse,
        np.ones(len(ellipse)),(480,640),20.,160.)
    radial=points-center
    np.testing.assert_allclose(directions,radial/np.linalg.norm(radial,axis=1)[:,None])
    start=points+directions*intervals[:,:1]
    np.testing.assert_allclose(np.einsum('ij,ij->i',start-center,directions),4.)


def test_displaced_ring_recovers_both_arcs_from_paired_colour_boundaries():
    import cv2
    from rinkspace.session.material_ring_bands import read_coloured_ring_band
    rgb=np.full((480,800,3),210,np.uint8)
    cv2.ellipse(rgb,(280,260),(250,60),0,0,360,(180,140,140),-1)
    cv2.ellipse(rgb,(280,260),(175,42),0,0,360,(210,210,210),-1)
    rgb=cv2.GaussianBlur(rgb,(9,9),2.)
    angle=np.linspace(0.,2*np.pi,257);seed_center=np.array([315.,239.])
    guide=seed_center+np.c_[220*np.cos(angle),52*np.sin(angle)]
    measured,result=read_coloured_ring_band(rgb,np.zeros(rgb.shape[:2],bool),
        {'uv':guide,'valid':np.ones(len(guide),bool)},source_sha256='a'*64,pts=256,center_uv=seed_center)
    ids=result['selected'];assert len(ids)>=20
    sides=measured['candidate_side_quantiles_uv'][ids,:,1]
    # Both arcs and both real sides are needed; fitting a thin ellipse to
    # just the far arc must not pass this source-observation test.
    assert sides[:,0,1].min()<225 and sides[:,0,1].max()>295
    for side,axes in [(0,np.array([175.,42.])),(1,np.array([250.,60.]))]:
        level=np.linalg.norm((sides[:,side]-[280,260])/axes,axis=1)
        assert np.percentile(np.abs(level-1),95)<.07


def test_wide_low_chroma_annulus_reads_full_material_not_red_pigment_ripples():
    import cv2
    from rinkspace.session.material_ring_bands import read_coloured_ring_band
    rgb=np.full((360,1600,3),210,np.uint8)
    cv2.ellipse(rgb,(800,180),(650,80),0,0,360,(160,155,155),-1)
    cv2.ellipse(rgb,(800,180),(400,49),0,0,360,(210,210,210),-1)
    yy,xx=np.indices(rgb.shape[:2]);annulus=(rgb[:,:,1]==155)
    rgb[:,:,0][annulus]+=np.rint(3+3*np.sin(xx[annulus]/13.)).astype(np.uint8)
    rgb=cv2.GaussianBlur(rgb,(9,9),2.)
    angle=np.linspace(0.,2*np.pi,257);center=np.array([800.,180.])
    guide=center+np.c_[550*np.cos(angle),68*np.sin(angle)]
    measured,result=read_coloured_ring_band(rgb,np.zeros(rgb.shape[:2],bool),
        {'uv':guide,'valid':np.ones(len(guide),bool)},source_sha256='a'*64,pts=256,
        center_uv=center,material_signal='dark',refine_source_origin=True)
    sides=measured['candidate_side_quantiles_uv'][result['selected'],:,1]
    assert len(sides)>=30
    for side,axes in [(0,np.array([400.,49.])),(1,np.array([650.,80.]))]:
        level=np.linalg.norm((sides[:,side]-center)/axes,axis=1)
        assert np.percentile(np.abs(level-1),95)<.07
    assert result['source_origin_revision']['coordinates_used_as_observations'] is False


def test_observed_band_color_separates_red_pigment_from_neutral_dark_material():
    from rinkspace.session.material_ring_bands import observed_band_color
    rgb=np.full((70,120,3),210,np.uint8)
    rgb[20:41,10:41]=[180,100,100]
    rgb[20:41,70:101]=[80,80,80]
    centers=np.array([[25.,30.],[85.,30.]])
    sides=np.array([[[[x,19.],[x,20.],[x,21.]],
                     [[x,39.],[x,40.],[x,41.]]] for x in centers[:,0]])
    measured=dict(candidate_uv=centers,candidate_query=np.arange(2),
                  query_normal=np.tile([0.,1.],(2,1)),candidate_side_quantiles_uv=sides)
    red,red_receipt=observed_band_color(rgb,measured,'red')
    neutral,neutral_receipt=observed_band_color(rgb,measured,'neutral_dark')
    np.testing.assert_array_equal(red,[True,False])
    np.testing.assert_array_equal(neutral,[False,True])
    assert neutral_receipt['source_red_contrast'][1]==0.
    assert neutral_receipt['source_dark_contrast'][1]>100.
    assert red_receipt['eligible']==neutral_receipt['eligible']==1


def test_neutral_dark_band_hypothesis_scores_actual_dark_signal_without_red_contrast():
    import cv2
    from rinkspace.session.material_ring_bands import read_coloured_ring_band
    from rinkspace.session.material_rings import source_ring_hypotheses
    rgb=np.full((360,640,3),210,np.uint8)
    cv2.ellipse(rgb,(320,180),(230,75),0,0,360,(90,90,90),-1)
    cv2.ellipse(rgb,(320,180),(170,55),0,0,360,(210,210,210),-1)
    rgb=cv2.GaussianBlur(rgb,(7,7),1.5)
    angle=np.linspace(0.,2*np.pi,257);center=np.array([320.,180.])
    guide=center+np.c_[200*np.cos(angle),65*np.sin(angle)]
    measured,result=read_coloured_ring_band(rgb,np.zeros(rgb.shape[:2],bool),
        {'uv':guide,'valid':np.ones(len(guide),bool)},source_sha256='a'*64,pts=256,
        center_uv=center,material_signal='dark',band_color='neutral_dark')
    assert len(result['selected'])>=20
    assert result['color_receipt']['eligible']>0
    assert result['hypotheses'][0]['score']>0.
    assert result['hypotheses'][0]['median_source_contrast']>20.
    red=source_ring_hypotheses(rgb,measured,0,result['fragments'],band_centers=True,material_signal='red')
    assert red and all(h['score']==0. for h in red)
    # Exactly the same neutral source pixels must remain usable with the
    # declared dark signal; neutral RGB must not inherit a zero red score.
    dark=source_ring_hypotheses(rgb,measured,0,result['fragments'],band_centers=True,material_signal='dark')
    assert dark and dark[0]['score']>red[0]['score']
    assert dark[0]['median_source_contrast']>20.


def test_no_source_band_candidates_return_empty_without_color_or_ring_failure():
    from rinkspace.session.material_ring_bands import observed_band_color,read_coloured_ring_band
    empty,receipt=observed_band_color(np.full((20,20,3),210,np.uint8),
                                      {'candidate_uv':np.empty((0,2))},'neutral_dark')
    assert empty.shape==(0,) and empty.dtype==bool and receipt['eligible']==0
    angle=np.linspace(0.,2*np.pi,129);center=np.array([160.,120.])
    guide=center+np.c_[110*np.cos(angle),40*np.sin(angle)]
    measured,result=read_coloured_ring_band(np.full((240,320,3),210,np.uint8),
        np.zeros((240,320),bool),{'uv':guide,'valid':np.ones(len(guide),bool)},
        source_sha256='a'*64,pts=256,center_uv=center,
        material_signal='dark',band_color='neutral_dark')
    assert len(measured['candidate_uv'])==0
    assert result['status']=='no_source_ring_band'
    assert result['selected']==result['hypotheses']==result['fragments']==[]
    assert result['color_receipt']['eligible']==0


def test_partial_out_of_frame_paired_band_keeps_measured_fragments_without_selection():
    import cv2
    from rinkspace.session.material_ring_bands import read_coloured_ring_band
    rgb=np.full((360,600,3),210,np.uint8)
    # Both measured material sides enter the image, while most of the ring
    # and its centre are outside it. A short arc must not certify a full ring.
    cv2.ellipse(rgb,(-100,180),(250,65),0,0,360,(90,90,90),-1)
    cv2.ellipse(rgb,(-100,180),(175,45),0,0,360,(210,210,210),-1)
    rgb=cv2.GaussianBlur(rgb,(7,7),1.5)
    angle=np.linspace(0.,2*np.pi,257);center=np.array([-100.,180.])
    guide=center+np.c_[212.5*np.cos(angle),55*np.sin(angle)]
    measured,result=read_coloured_ring_band(rgb,np.zeros(rgb.shape[:2],bool),
        {'uv':guide,'valid':np.ones(len(guide),bool)},source_sha256='a'*64,pts=256,
        center_uv=center,search_radius_px=100.,spacing_px=12.,
        material_signal='dark',band_color='neutral_dark')
    assert result['status']=='partial_source_bands_without_complete_ring'
    assert result['hypotheses']==[] and result['selected']==[]
    assert result['partial_arcs_need_source_identity_and_correspondence'] is True
    fragments=result['fragments'];assert fragments
    ids=np.unique([i for f in fragments for i in f['candidate_ids']])
    assert len(ids)>=12 and ids.max()<len(measured['candidate_uv'])
    sides=measured['candidate_side_quantiles_uv'][ids,:,1]
    assert np.isfinite(sides).all() and sides[:,:,0].min()>=0.
    for side,axes in [(0,np.array([175.,45.])),(1,np.array([250.,65.]))]:
        level=np.linalg.norm((sides[:,side]-center)/axes,axis=1)
        assert np.percentile(np.abs(level-1.),95)<.07


def test_dark_ripple_inside_red_material_is_not_reclassified_as_neutral_dark():
    from rinkspace.session.material_ring_bands import observed_band_color
    # The central dark ink is less red than either red flank, yet its actual
    # pigment still has clear red chroma. Relative contrast cannot rename it.
    rgb=np.full((70,60,3),[200,120,120],np.uint8)
    rgb[20:41,10:41]=[110,80,80]
    measured=dict(candidate_uv=np.array([[25.,30.]]),candidate_query=np.array([0]),
        query_normal=np.array([[0.,1.]]),candidate_side_quantiles_uv=np.array([
            [[[25.,19.],[25.,20.],[25.,21.]],[[25.,39.],[25.,40.],[25.,41.]]]]))
    neutral,receipt=observed_band_color(rgb,measured,'neutral_dark')
    assert receipt['source_red_contrast'][0]<0.
    assert receipt['source_dark_contrast'][0]>20.
    assert receipt['measured_pigment_red_chroma'][0]>20.
    np.testing.assert_array_equal(neutral,[False])
    assert receipt['eligible']==0


def test_paired_boundary_continuity_requires_both_actual_sides_and_returns_original_ids():
    from rinkspace.session.material_ring_bands import continuous_paired_boundary_candidates
    good_ids=[19,2,41,7]
    inner_only_ids=[11,34,5,28]
    outer_only_ids=[45,13,30,9]
    isolated_id=52
    sides=np.zeros((60,2,3,2),float)
    for j in range(4):
        for cid,inner,outer in [
            (good_ids[j],[30.*j,0.],[30.*j,10.]),
            (inner_only_ids[j],[500.+30.*j,0.],[500.+300.*j,200.]),
            (outer_only_ids[j],[2000.+300.*j,0.],[2000.+30.*j,200.]),
        ]:
            sides[cid,0]=inner;sides[cid,1]=outer
    sides[isolated_id,0]=[4000.,0.];sides[isolated_id,1]=[4000.,10.]
    supplied=outer_only_ids+[isolated_id]+good_ids+inner_only_ids+[good_ids[0]]
    original=sides.copy()
    selected,receipt=continuous_paired_boundary_candidates(
        {'candidate_side_quantiles_uv':sides},supplied,max_gap_px=40.,minimum_span_px=70.)
    assert selected==sorted(good_ids)
    assert {cid for group in receipt for cid in group['candidate_ids']}==set(supplied)
    accepted=[group for group in receipt if group['continuous_source_arc']]
    assert len(accepted)==1 and sorted(accepted[0]['candidate_ids'])==sorted(good_ids)
    np.testing.assert_array_equal(sides,original)
