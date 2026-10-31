import cv2
import numpy as np
import pytest

from rinkspace.session.material_tracks import assemble_fragments, normal_profile_flow, propagate_identities, link_fragments, qualify_fragment_links, qualify_line_candidates, qualify_visible_line_fragments, source_line_fragments, _warp
from rinkspace.vision_ice_temporal_identity import _measured_material_descriptors


def test_line_identity_support_is_independent_of_tangent_sample_phase():
    from rinkspace.session.material_tracks import source_line_reference_support
    measured=np.c_[np.arange(5)*24.,np.full(5,50.)]
    reference=np.c_[[12.,36.,60.,84.],np.full(4,50.)]
    saved=measured.copy()
    result=source_line_reference_support(reference,measured,np.arange(5))
    assert result['qualified'] and result['count']==4 and result['span_px']==72.
    np.testing.assert_array_equal(measured,saved)


def test_line_identity_support_neither_bridges_gaps_nor_extends_endpoints():
    from rinkspace.session.material_tracks import source_line_reference_support
    measured=np.c_[[0.,24.,120.,144.],np.full(4,50.)]
    reference=np.c_[[48.,72.,96.,168.],np.full(4,50.)]
    result=source_line_reference_support(reference,measured,np.array([0,1,5,6]))
    assert not result['qualified'] and result['count']==0


def test_parallel_or_repeated_dense_points_do_not_seed_a_source_line():
    from rinkspace.session.material_tracks import source_line_reference_support
    measured=np.c_[np.arange(5)*24.,np.full(5,50.)]
    parallel=np.c_[[12.,36.,60.,84.],np.full(4,56.)]
    dense=np.c_[[12.,12.5,13.,13.5],np.full(4,50.)]
    assert not source_line_reference_support(parallel,measured,np.arange(5))['qualified']
    assert not source_line_reference_support(dense,measured,np.arange(5))['qualified']


def _band_nodes():
    y = np.arange(128)[:, None]
    value = 230.-100.*np.exp(-.5*((y-60.)/3.)**2)
    rgb = np.broadcast_to(value[:, :, None], (128, 160, 3)).astype(np.float32).copy()
    uv = np.array([[50., 60.], [80., 60.], [110., 60.]])
    normal = np.tile([0., 1.], (3, 1))
    sides = uv[:, None]+normal[:, None]*np.array([-3., 3.])[None, :, None]
    descriptor, strength = _measured_material_descriptors(rgb, uv, sides, normal)
    return rgb, dict(uv=uv, normal=normal, width=np.full(3, 6.), descriptor=descriptor, strength=strength)


def test_normal_registration_handles_aperture_without_inventing_tangent():
    rgb, nodes = _band_nodes()
    shifted = np.roll(rgb, 5, axis=0)
    prediction, valid, _, _ = normal_profile_flow(shifted, nodes, np.eye(3), np.zeros((128, 160), bool))
    assert valid.all()
    np.testing.assert_allclose(prediction[:, 1], 65., atol=.15)
    np.testing.assert_allclose(prediction[:, 0], nodes['uv'][:, 0], atol=1e-12)
    np.testing.assert_array_equal(nodes['uv'][:, 1], np.full(3, 60.))


def test_source_absence_and_competing_bands_do_not_make_unique_matches():
    rgb, nodes = _band_nodes()
    blank = np.full_like(rgb, 230.)
    assert not normal_profile_flow(blank, nodes, np.eye(3), np.zeros((128, 160), bool))[1].any()
    competing = np.minimum(np.roll(rgb, -10, axis=0), np.roll(rgb, 10, axis=0))
    assert not normal_profile_flow(competing, nodes, np.eye(3), np.zeros((128, 160), bool))[1].any()


def test_fragment_registration_measures_normals_in_the_current_image():
    y=np.arange(320)[:,None]
    rgb=np.broadcast_to((230.-100.*np.exp(-.5*((y-120.)/3.)**2))[:,:,None],(320,400,3)).astype('f').copy()
    theta=np.deg2rad(12.)
    h=np.array([[np.cos(theta),-np.sin(theta),40.],[np.sin(theta),np.cos(theta),10.],[0.,0.,1.]])
    moved=cv2.warpPerspective(rgb,h,(400,320),borderValue=(230.,)*3)
    uv=np.c_[[50.,115.,180.,245.],np.full(4,120.)]
    old_normal=np.tile([0.,1.],(4,1));new_uv=_warp(h,uv+[30.,0.])
    new_normal=np.tile([-np.sin(theta),np.cos(theta)],(4,1))
    def nodes(image,points,normals):
        sides=points[:,None]+normals[:,None]*np.array([-3.,3.])[None,:,None]
        descriptor,strength=_measured_material_descriptors(image,points,sides,normals)
        return dict(uv=points,normal=normals,width=np.full(4,6.),descriptor=descriptor,strength=strength,
                    family=np.zeros(4,int),fragment=np.zeros(4,int))
    edges=link_fragments(rgb,moved,nodes(rgb,uv,old_normal),nodes(moved,new_uv,new_normal),
        np.zeros((320,400),bool),np.zeros((320,400),bool),h)
    assert len(edges)==1 and edges[0]['unambiguous']
    assert edges[0]['support']==4


def test_blur_broadens_one_correlation_peak_without_creating_another_identity():
    y = np.arange(180)[:, None]
    rgb = np.broadcast_to((230.-80.*np.exp(-.5*((y-90.)/18.)**2))[:, :, None], (180, 160, 3)).astype(np.float32).copy()
    uv = np.array([[50., 90.], [80., 90.], [110., 90.]])
    normal = np.tile([0., 1.], (3, 1))
    sides = uv[:, None]+normal[:, None]*np.array([-25.,25.])[None,:,None]
    descriptor,strength = _measured_material_descriptors(rgb,uv,sides,normal)
    nodes = dict(uv=uv,normal=normal,width=np.full(3,50.),descriptor=descriptor,strength=strength)
    prediction,valid,_,_=normal_profile_flow(np.roll(rgb,4,axis=0),nodes,np.eye(3),np.zeros((180,160),bool))
    assert valid.all()
    np.testing.assert_allclose(prediction[:,1],94.,atol=.2)


def test_future_anchor_reaches_past_and_conflicting_identities_stay_visible():
    frames = [dict(pts=i) for i in range(3)]
    edges = [[dict(a=0, b=0, unambiguous=True)], [dict(a=0, b=0, unambiguous=True)]]
    labels = propagate_identities(frames, edges, [dict(pts=2, fragment=0, identity='C')])
    assert len(labels) == 3 and all(r['identities'] == ['C'] for r in labels)
    labels = propagate_identities(frames, edges, [dict(pts=2, fragment=0, identity='C'),
        dict(pts=0, fragment=0, identity='goal_line_right')])
    assert all(r['status'] == 'identity_conflict' for r in labels)
    edges[0][0]['unambiguous'] = False
    labels = propagate_identities(frames, edges, [dict(pts=2, fragment=0, identity='C')])
    assert [r['pts'] for r in labels] == [1, 2]


def test_fragment_can_cross_unreadable_profiles_without_inventing_readouts():
    query = np.c_[np.arange(10)*14., np.full(10, 40.)]
    ids = np.array([0, 4, 7, 9])
    uv = query[ids]+[0., 20.]
    measurement = dict(candidate_uv=uv, candidate_query=ids, query_uv=query,
        query_normal=np.tile([0., 1.], (10, 1)), query_curve=np.zeros(10, int),
        candidate_sides=uv[:, None]+np.array([[0., -2.], [0., 2.]]), candidate_flags=np.full(4, 8))
    fragments = assemble_fragments(measurement, [{}])
    assert len(fragments) == 1
    assert fragments[0]['candidate_ids'] == [0, 1, 2, 3]
    assert fragments[0]['physical_identity'] is None
    np.testing.assert_array_equal(measurement['candidate_uv'], uv)


def test_cross_view_width_changes_follow_the_image_map_scale():
    y = np.arange(240)[:, None]
    old_rgb = np.broadcast_to((230.-100.*np.exp(-.5*((y-90.)/4.)**2))[:, :, None],
                             (240, 400, 3)).astype(np.float32).copy()
    h = np.array([[1.,0.,0.],[0.,3.,20.],[0.,0.,1.]])
    new_rgb = cv2.warpPerspective(old_rgb,h,(400,740),borderValue=(230.,)*3)
    uv = np.c_[[50.,115.,180.,245.],np.full(4,90.)]
    normal = np.tile([0.,1.],(4,1))
    def nodes(image, points, width):
        sides = points[:,None]+normal[:,None]*np.array([-width/2,width/2])[None,:,None]
        descriptor,strength = _measured_material_descriptors(image,points,sides,normal)
        return dict(uv=points,normal=normal,width=np.full(4,width),descriptor=descriptor,
                    strength=strength,family=np.zeros(4,int),fragment=np.zeros(4,int))
    old = nodes(old_rgb,uv,8.)
    now = nodes(new_rgb,_warp(h,uv),24.)
    edges = link_fragments(old_rgb,new_rgb,old,now,np.zeros(old_rgb.shape[:2],bool),
                           np.zeros(new_rgb.shape[:2],bool),h)
    assert len(edges) == 1 and edges[0]['unambiguous']
    assert edges[0]['support'] == 4
    np.testing.assert_array_equal(now['width'],np.full(4,24.))


def test_disjoint_material_spans_keep_identity_through_a_split_and_merge():
    split = [dict(a=0,b=0,support=5,pairs=[[i,i] for i in range(5)]),
             dict(a=0,b=1,support=5,pairs=[[i,i] for i in range(5,10)])]
    merge = [dict(a=0,b=0,support=5,pairs=[[i,i] for i in range(5)]),
             dict(a=1,b=0,support=5,pairs=[[i,i] for i in range(5,10)])]
    edges = [qualify_fragment_links(split),qualify_fragment_links(merge)]
    labels = propagate_identities([dict(pts=i) for i in range(3)],edges,
                                  [dict(pts=0,fragment=0,identity='blue_line')])
    assert {(x['frame'],x['fragment']) for x in labels} == {(0,0),(1,0),(1,1),(2,0)}
    assert all(x['identities']==['blue_line'] for x in labels)


def test_shared_source_support_still_exposes_competing_material_branches():
    edges = [dict(a=0,b=0,support=6,pairs=[[i,i] for i in range(6)]),
             dict(a=0,b=1,support=6,pairs=[[i,i+10] for i in range(3,9)])]
    qualify_fragment_links(edges)
    assert not any(x['unambiguous'] for x in edges)


def test_shared_target_support_still_exposes_competing_parent_fragments():
    edges = [dict(a=0,b=0,support=10,pairs=[[i,i] for i in range(10)]),
             dict(a=1,b=0,support=6,pairs=[[i+20,i] for i in range(7,13)])]
    qualify_fragment_links(edges)
    assert not any(x['unambiguous'] for x in edges)


def test_whole_source_line_does_not_follow_a_local_shadow_branch():
    query=np.c_[np.arange(14)*28.,np.full(14,40.)]
    actual=query.copy();actual[5:9,1]+=9.
    measurement=dict(query_uv=query,query_normal=np.tile([0.,1.],(14,1)),
        query_curve=np.zeros(14,int),candidate_query=np.arange(14),
        candidate_flags=np.zeros(14,int),candidate_uv=actual.copy())
    ids,report=qualify_line_candidates(measurement,[0])
    np.testing.assert_array_equal(ids,np.r_[np.arange(5),np.arange(9,14)])
    np.testing.assert_array_equal(measurement['candidate_uv'],actual)
    assert report[0]['source_line_support_qualified']
    assert not report[0]['fitted_line_used_as_observation']
    grouped=source_line_fragments(measurement,report)
    assert len(grouped)==1
    assert grouped[0]['source_visible_candidate_pieces']==[list(range(5)),list(range(9,14))]
    assert grouped[0]['candidate_ids']==ids.tolist()
    assert grouped[0]['physical_identity'] is None


def _separated_line_measurement(offsets, slopes=(0., 0., 0., 0.)):
    query=np.c_[np.arange(36)*28.,np.full(36,200.)]
    candidate_query=np.concatenate([np.arange(begin,begin+4) for begin in (2,10,18,26)])
    uv=query[candidate_query].copy()
    uv[:,1]+=np.repeat(offsets,4)+np.repeat(slopes,4)*uv[:,0]
    sides=uv[:,None]+np.array([[0.,-4.],[0.,4.]])
    quantiles=sides[:,:,None]+np.array([[0.,-.5],[0.,0.],[0.,.5]])
    return dict(query_uv=query,query_normal=np.tile([0.,1.],(36,1)),
        query_curve=np.zeros(36,int),candidate_query=candidate_query,
        candidate_uv=uv,candidate_sides=sides,candidate_side_quantiles_uv=quantiles,
        candidate_flags=np.zeros(len(uv),int))


def _qualify_separated_line_fallback(measurement):
    original={key:value.copy() for key,value in measurement.items()}
    fragments,report=qualify_visible_line_fragments(measurement,[0])
    # Four 84px material spans are separated by 140px gaps. They cannot form
    # one local proposal or meet the whole 980px query's half-span threshold.
    assert not report['whole_line_reports'][0]['source_line_support_qualified']
    assert [row['query_interval'] for row in report['local_source_intervals']]==[
        [2,6],[10,14],[18,22],[26,30]]
    assert [row['selected_candidate_ids'] for row in report['local_source_intervals']]==[
        list(range(0,4)),list(range(4,8)),list(range(8,12)),list(range(12,16))]
    assert len(report['coherent_source_interval_reports'])==1
    assert report['coherent_source_interval_reports'][0]['source_support_restricted_to_proposed_intervals']
    assert report['missing_readouts_not_filled'] and report['source_uv_unchanged']
    for key,value in original.items():
        np.testing.assert_array_equal(measurement[key],value)
    return fragments,report


def test_separated_collinear_material_fragments_keep_readouts_and_gaps():
    m=_separated_line_measurement((20.,20.,20.,20.))
    fragments,report=_qualify_separated_line_fallback(m)
    assert len(fragments)==1
    fragment=fragments[0]
    assert fragment['candidate_ids']==list(range(16))
    assert fragment['source_visible_candidate_pieces']==[
        list(range(0,4)),list(range(4,8)),list(range(8,12)),list(range(12,16))]
    assert fragment['span_px']==pytest.approx(336.)
    assert fragment['physical_identity'] is None
    assert not fragment['unseen_source_pixels_created']
    assert report['coherent_source_interval_reports'][0]['source_line_support_qualified']
    # Every retained index still refers to an actual measured pixel, offset
    # from the navigation guide; no row exists at an unreadable query.
    np.testing.assert_array_equal(m['candidate_uv'][fragment['candidate_ids']],
        m['query_uv'][m['candidate_query']]+[0.,20.])


@pytest.mark.parametrize('offsets,slopes',[
    ((0.,80.,0.,80.),(0.,0.,0.,0.)),
    ((0.,110.,0.,110.),(.05,-.05,.05,-.05)),
],ids=['parallel-offsets','different-slopes'])
def test_separated_competing_line_families_cannot_merge_local_qualifications(offsets,slopes):
    # Each family has the same two 84px spans and the same paired-boundary
    # widths/quantiles. Local straightness supplies no reason to prefer one.
    m=_separated_line_measurement(offsets,slopes)
    fragments,report=_qualify_separated_line_fallback(m)
    assert fragments==[]
    coherent=report['coherent_source_interval_reports'][0]
    assert not coherent['source_line_support_qualified']
    assert coherent['selected_candidate_ids']==[]
    assert coherent['disposition']=='competing_source_line_families_unknown'


def test_separated_dominant_line_drops_switched_fragment_without_filling_gaps():
    m=_separated_line_measurement((0.,80.,0.,0.))
    fragments,report=_qualify_separated_line_fallback(m)
    assert len(fragments)==1
    fragment=fragments[0]
    expected=list(range(0,4))+list(range(8,16))
    assert fragment['candidate_ids']==expected
    assert fragment['source_visible_candidate_pieces']==[
        list(range(0,4)),list(range(8,12)),list(range(12,16))]
    assert fragment['span_px']==pytest.approx(252.)
    assert fragment['physical_identity'] is None
    assert not fragment['unseen_source_pixels_created']
    assert report['coherent_source_interval_reports'][0]['source_line_support_qualified']
    np.testing.assert_array_equal(m['candidate_uv'][expected],m['query_uv'][m['candidate_query'][expected]])


def test_missing_exposure_schedules_source_check_without_propagating_identity():
    from rinkspace.session.material_tracks import disconnected_source_boundaries
    frames=[dict(pts=i,fragments=([dict(curve=0)] if i!=2 else [])) for i in range(5)]
    edges=[[dict(a=0,b=0,unambiguous=True)],[],[],[dict(a=0,b=0,unambiguous=True)]]
    anchors=[dict(pts=4,fragment=0,identity='blue')]
    assert disconnected_source_boundaries(frames,edges)==[(1,3)]
    assert [x['pts'] for x in propagate_identities(frames,edges,anchors)]==[3,4]
    reference=[dict(reference=1,current=3,edges=[dict(a=0,b=0,unambiguous=False)])]
    assert disconnected_source_boundaries(frames,edges,reference)==[(1,3)]
    reference[0]['edges'][0]['unambiguous']=True
    assert disconnected_source_boundaries(frames,edges,reference)==[]
    labels=propagate_identities(frames,edges,anchors,reference)
    assert [x['pts'] for x in labels]==[0,1,3,4]
    assert labels[1]['proofs']['0']['edge_frame']=='reference:0'


def test_absent_material_does_not_shrink_visible_source_length():
    query=np.c_[10.+np.arange(20)*28.,np.full(20,40.)]
    ids=np.array([0,1,18,19])
    m=dict(query_uv=query,query_normal=np.tile([0.,1.],(20,1)),
           query_interval=np.tile([-4.,4.],(20,1)),query_curve=np.zeros(20,int),
           candidate_query=ids,candidate_flags=np.zeros(4,int),candidate_uv=query[ids].copy())
    selected,reports=qualify_line_candidates(m,[0])
    assert not len(selected)
    assert reports[0]['disposition']=='insufficient_distributed_source_line_span'
    assert reports[0]['visible_source_queries']==list(range(20))
    # The same lack of readouts is admissible only with actual exclusion
    # evidence covering the intervening profiles, not from absence alone.
    exclusion=np.zeros((100,600),bool);exclusion[:,60:505]=True
    selected,reports=qualify_line_candidates(m,[0],exclusion)
    np.testing.assert_array_equal(selected,np.arange(4))
    assert reports[0]['fully_excluded_source_queries']==list(range(2,18))
    np.testing.assert_array_equal(m['candidate_uv'],query[ids])


def test_empty_line_search_has_no_visible_support_with_source_mask():
    m=dict(query_uv=np.empty((0,2)),query_normal=np.empty((0,2)),
           query_interval=np.empty((0,2)),query_curve=np.empty(0,int),
           candidate_query=np.empty(0,int),candidate_flags=np.empty(0,int),candidate_uv=np.empty((0,2)))
    selected,reports=qualify_line_candidates(m,[0],np.zeros((10,10),bool))
    assert selected.size==0
    assert reports[0]['disposition']=='no_visible_source_queries'


def test_material_identity_does_not_leak_through_an_unverified_local_fragment():
    import pytest
    from rinkspace.session.material_tracks import propagate_readout_identities
    links=[dict(previous=0,current=1,pairs=[[0,0],[1,1]],unambiguous=True,source_only=True)]
    anchors=[dict(frame=0,candidate=0,identity='blue')]
    offsets,component,labels=propagate_readout_identities([2,2],links,anchors)
    assert component[0]==component[2]
    assert component[1]==component[3]
    assert component[0]!=component[1]
    assert len(labels)==1
    group=dict(frame=0,candidate_ids=[0,1],source_line_support_qualified=True)
    _,component,labels=propagate_readout_identities([2,2],links,anchors,[group])
    assert len(set(component))==1
    assert labels[0]['status']=='anchored'
    group['source_line_support_qualified']=False
    with pytest.raises(ValueError,match='Unqualified local fragments'):
        propagate_readout_identities([2,2],links,anchors,[group])


def test_material_node_anchor_conflict_remains_unresolved():
    from rinkspace.session.material_tracks import propagate_readout_identities
    links=[dict(previous=0,current=1,pairs=[[0,0]],unambiguous=True,source_only=True)]
    anchors=[dict(frame=0,candidate=0,identity='blue'),dict(frame=1,candidate=0,identity='red')]
    _,_,labels=propagate_readout_identities([1,1],links,anchors)
    assert labels[0]['status']=='identity_conflict'
    assert labels[0]['identities']==['blue','red']


def test_sparse_identity_matches_still_reject_a_local_off_line_readout():
    from rinkspace.session.material_tracks import check_matched_line_geometry
    quv=np.c_[10.+np.arange(40)*28.,np.full(40,40.)]
    queries=np.array([1,4,7,11,15,20,24,30,35]);uv=quv[queries].copy();uv[3,1]+=5.
    m=dict(query_uv=quv,query_normal=np.tile([0.,1.],(40,1)),query_curve=np.zeros(40,int),
        candidate_query=queries,candidate_uv=uv.copy())
    selected,report=check_matched_line_geometry(m,np.arange(len(queries)),
        [dict(curve=0,visible_source_queries=list(range(40)))])
    assert 3 not in selected and len(selected)==8
    assert not report[0]['source_line_support_qualified']
    assert report[0]['independent_node_identity_still_required']
    np.testing.assert_array_equal(m['candidate_uv'],uv)


def test_plausible_midpoints_cannot_hide_crossed_material_boundaries():
    query=np.c_[10.+np.arange(20)*28.,np.full(20,40.)]
    sides=np.stack([query-[0.,4.],query+[0.,4.]],axis=1)
    sides[6:10,0,1]-=6.;sides[6:10,1,1]+=6.
    m=dict(query_uv=query,query_normal=np.tile([0.,1.],(20,1)),
        query_curve=np.zeros(20,int),candidate_query=np.arange(20),
        candidate_flags=np.zeros(20,int),candidate_uv=query.copy(),candidate_sides=sides.copy())
    legacy,_=qualify_line_candidates(m,[0])
    assert len(legacy)==20
    selected,reports=qualify_line_candidates(m,[0],require_paired_boundaries=True)
    np.testing.assert_array_equal(selected,np.r_[np.arange(6),np.arange(10,20)])
    assert reports[0]['source_line_support_qualified']
    assert all(r['source_line_support_qualified'] for r in reports[0]['boundary_reports'])
    np.testing.assert_array_equal(m['candidate_uv'],query)
    np.testing.assert_array_equal(m['candidate_sides'],sides)


def test_perspective_band_width_change_retains_actual_source_boundaries():
    query=np.c_[10.+np.arange(20)*28.,np.full(20,40.)]
    half_width=np.linspace(3.,7.,20)
    sides=np.stack([query.copy(),query.copy()],axis=1)
    sides[:,0,1]-=half_width;sides[:,1,1]+=half_width
    m=dict(query_uv=query,query_normal=np.tile([0.,1.],(20,1)),
        query_curve=np.zeros(20,int),candidate_query=np.arange(20),
        candidate_flags=np.zeros(20,int),candidate_uv=query.copy(),candidate_sides=sides.copy())
    selected,reports=qualify_line_candidates(m,[0],require_paired_boundaries=True)
    np.testing.assert_array_equal(selected,np.arange(20))
    assert reports[0]['source_line_support_qualified']
    np.testing.assert_array_equal(m['candidate_sides'],sides)


def _curved_projected_line_measurement():
    # A physical straight band acquires source-image curvature through radial
    # lens distortion. Its projected sides are independent measured readouts.
    x=np.linspace(-.9,.9,25);y=.6;k=.35
    points=np.c_[x,np.full(len(x),y),np.ones(len(x))]
    camera=np.array([[500.,0.,600.],[0.,500.,200.],[0.,0.,1.]])
    def project(value):
        return cv2.projectPoints(value,np.zeros(3),np.zeros(3),camera,
            np.array([k,0.,0.,0.,0.]))[0][:,0]
    query=project(points)
    tangent=np.c_[1.+k*(3.*x*x+y*y),2.*k*x*y]
    tangent/=np.linalg.norm(tangent,axis=1)[:,None]
    normal=np.c_[-tangent[:,1],tangent[:,0]]
    sides=[]
    for offset in (-.008,.008):
        edge=points.copy();edge[:,1]+=offset;sides.append(project(edge))
    sides=np.stack(sides,axis=1)
    # Short alternative branches and scrambled storage expose accidental use
    # of organizer-local indices as original measurement indices.
    extra_queries=np.array([4,12,20])
    candidate_query=np.r_[np.arange(len(query)),extra_queries]
    candidate_sides=np.concatenate([sides,
        sides[extra_queries]+normal[extra_queries,None]*18.])
    uv=candidate_sides.mean(1)
    order=np.r_[np.arange(0,len(uv),2),np.arange(1,len(uv),2)][::-1]
    measurement=dict(query_uv=query,query_normal=normal,query_curve=np.zeros(len(query),int),
        candidate_query=candidate_query[order],candidate_flags=np.zeros(len(uv),int),
        candidate_uv=uv[order].copy(),candidate_sides=candidate_sides[order].copy())
    return measurement,np.flatnonzero(order<len(query))


def test_native_normal_strip_qualifies_curved_line_without_replacing_source_uv():
    m,expected=_curved_projected_line_measurement()
    original={key:value.copy() for key,value in m.items()}
    legacy,_=qualify_line_candidates(m,[0],require_paired_boundaries=True)
    assert legacy.size==0
    selected,reports=qualify_line_candidates(m,[0],require_paired_boundaries=True,
        native_shape_guided=True)
    np.testing.assert_array_equal(selected,expected)
    assert reports[0]['source_line_support_qualified']
    assert reports[0]['source_geometry_domain']=='native_normal_strip'
    assert reports[0]['source_uv_unchanged']
    assert not reports[0]['native_fit_residual_used']
    assert not reports[0]['fitted_line_used_as_observation']
    assert len(reports[0]['boundary_reports'])==2
    for boundary in reports[0]['boundary_reports']:
        assert boundary['source_line_support_qualified']
        assert boundary['selected_candidate_ids']==expected.tolist()
    fragments=source_line_fragments(m,reports)
    assert len(fragments)==1
    ids=np.asarray(fragments[0]['candidate_ids'])
    np.testing.assert_array_equal(m['candidate_query'][ids],np.arange(len(m['query_uv'])))
    np.testing.assert_array_equal(m['candidate_uv'][ids],original['candidate_uv'][ids])
    assert np.linalg.norm(m['candidate_uv'][ids]-m['query_uv'],axis=1).max()>1e-3
    assert fragments[0]['physical_identity'] is None
    for key,value in original.items():
        np.testing.assert_array_equal(m[key],value)


def test_native_normal_strip_does_not_qualify_one_boundary_as_a_band():
    m,_=_curved_projected_line_measurement()
    # Repeating one actual interface does not establish a second boundary.
    m['candidate_sides'][:,1]=m['candidate_sides'][:,0]
    m['candidate_uv']=m['candidate_sides'][:,0].copy()
    original_uv=m['candidate_uv'].copy();original_sides=m['candidate_sides'].copy()
    selected,reports=qualify_line_candidates(m,[0],require_paired_boundaries=True,
        native_shape_guided=True)
    assert selected.size==0
    assert not reports[0]['source_line_support_qualified']
    assert source_line_fragments(m,reports)==[]
    np.testing.assert_array_equal(m['candidate_uv'],original_uv)
    np.testing.assert_array_equal(m['candidate_sides'],original_sides)


def test_native_normal_strip_rejects_inconsistent_material_boundary_pairs():
    m,_=_curved_projected_line_measurement()
    query=m['candidate_query'];normal=m['query_normal'][query]
    # The actual midpoint supports the line, but the sides cycle among three
    # incompatible band widths over the full visible span.
    offset=np.array([24.,48.,72.])[query%3]
    m['candidate_sides']=m['candidate_uv'][:,None]+normal[:,None]*np.stack([-offset,offset],axis=1)[:,:,None]
    original_uv=m['candidate_uv'].copy();original_sides=m['candidate_sides'].copy()
    midpoint_only,_=qualify_line_candidates(m,[0],native_shape_guided=True)
    assert midpoint_only.size
    selected,reports=qualify_line_candidates(m,[0],require_paired_boundaries=True,
        native_shape_guided=True)
    assert selected.size==0
    assert not reports[0]['source_line_support_qualified']
    assert not reports[0]['boundary_reports'][1]['source_line_support_qualified']
    assert source_line_fragments(m,reports)==[]
    np.testing.assert_array_equal(m['candidate_uv'],original_uv)
    np.testing.assert_array_equal(m['candidate_sides'],original_sides)

@pytest.mark.parametrize('masked_y', [42, 45], ids=['band_interior', 'outer_transition'])
def test_paired_source_band_mask_checks_transitions_and_interior_without_changing_readouts(masked_y):
    query = np.c_[10. + np.arange(20)*28., np.full(20, 40.)]
    sides = np.stack([query-[0., 4.], query+[0., 4.]], axis=1)
    quantiles = sides[:, :, None, :] + np.array([[0., -1.], [0., 0.], [0., 1.]])
    measurement = dict(query_uv=query, query_normal=np.tile([0., 1.], (20, 1)),
        query_interval=np.tile([-12., 12.], (20, 1)), query_curve=np.zeros(20, int),
        candidate_query=np.arange(20), candidate_flags=np.zeros(20, int),
        candidate_uv=query.copy(), candidate_sides=sides.copy(),
        candidate_side_quantiles_uv=quantiles.copy())
    mask = np.zeros((100, 600), bool)
    blocked = 9
    mask[masked_y, int(query[blocked, 0])] = True
    # The midpoint and both q50 edges are clear. Only the actual band interior
    # or the outer q75 transition touches the mask; the query stays visible.
    assert not mask[40, int(query[blocked, 0])]
    assert not mask[36, int(query[blocked, 0])]
    assert not mask[44, int(query[blocked, 0])]
    original = {key: value.copy() for key, value in measurement.items()}
    original_mask = mask.copy()
    options = dict(require_paired_boundaries=True, native_shape_guided=True,
        use_source_quantiles=True)

    without_mask, _ = qualify_line_candidates(measurement, [0], **options)
    np.testing.assert_array_equal(without_mask, np.arange(20))
    selected, reports = qualify_line_candidates(measurement, [0], mask, **options)
    np.testing.assert_array_equal(selected, np.delete(np.arange(20), blocked))
    assert reports[0]['source_line_support_qualified']
    assert reports[0]['visible_source_queries'] == list(range(20))
    assert reports[0]['fully_excluded_source_queries'] == []
    for key, value in original.items():
        assert measurement[key].dtype == value.dtype
        assert measurement[key].tobytes() == value.tobytes()
    assert mask.tobytes() == original_mask.tobytes()



def _two_short_source_pieces_measurement():
    query=np.c_[np.arange(20)*24.,np.full(20,200.)]
    candidate_query=np.array([2,3,5,6])
    uv=query[candidate_query]+[0.,20.]
    sides=uv[:,None]+np.array([[0.,-4.],[0.,4.]])
    quantiles=sides[:,:,None]+np.array([[0.,-.5],[0.,0.],[0.,.5]])
    return dict(query_uv=query,query_normal=np.tile([0.,1.],(len(query),1)),
        query_curve=np.zeros(len(query),int),candidate_query=candidate_query,
        candidate_uv=uv,candidate_sides=sides,candidate_side_quantiles_uv=quantiles,
        candidate_flags=np.zeros(len(uv),int))


def test_coherent_short_pieces_keep_uniquely_qualified_actual_pixels_and_true_gap():
    measurement=_two_short_source_pieces_measurement()
    original={key:value.copy() for key,value in measurement.items()}
    fragments,report=qualify_visible_line_fragments(measurement,[0])
    assert not report['whole_line_reports'][0]['source_line_support_qualified']
    # Four actual readings propose a 96px distributed interval, with no
    # reading at its middle query. Both sides must qualify in real source code.
    assert [item['query_interval'] for item in report['local_source_intervals']]==[[2,7]]
    assert [item['selected_candidate_ids'] for item in report['local_source_intervals']]==[[0,1,2,3]]
    coherent,=report['coherent_source_interval_reports']
    assert coherent['source_line_support_qualified']
    assert coherent['disposition']=='source_line_family_supported'
    assert coherent['selected_candidate_ids']==[0,1,2,3]
    assert coherent['actual_boundary_quantiles_used_for_source_organization']
    assert all(edge['source_line_support_qualified'] for edge in coherent['boundary_reports'])
    assert len(coherent['boundary_reports'])==2
    assert len(fragments)==1
    fragment=fragments[0]
    assert fragment['candidate_ids']==[0,1,2,3]
    assert fragment['source_visible_candidate_pieces']==[[0,1],[2,3]]
    assert fragment['span_px']==pytest.approx(48.)
    assert fragment['organization']=='one_source_line_across_supported_intervals'
    assert not fragment['unseen_source_pixels_created']
    assert fragment['physical_identity'] is None
    kept=np.asarray(fragment['candidate_ids'])
    np.testing.assert_array_equal(measurement['candidate_query'][kept],[2,3,5,6])
    np.testing.assert_array_equal(measurement['candidate_uv'][kept],original['candidate_uv'])
    assert 4 not in measurement['candidate_query'][kept]
    for key,value in original.items():np.testing.assert_array_equal(measurement[key],value)


def test_short_disjoint_pieces_do_not_choose_between_equal_paired_source_families():
    measurement=_two_short_source_pieces_measurement()
    for key in ('candidate_query','candidate_flags'):
        measurement[key]=np.tile(measurement[key],2)
    for key in ('candidate_uv','candidate_sides','candidate_side_quantiles_uv'):
        measurement[key]=np.concatenate([measurement[key],measurement[key]+[0.,60.]])
    original={key:value.copy() for key,value in measurement.items()}
    fragments,report=qualify_visible_line_fragments(measurement,[0])
    assert fragments==[]
    assert not report['whole_line_reports'][0]['source_line_support_qualified']
    assert report['local_source_intervals']==[]
    for key,value in original.items():np.testing.assert_array_equal(measurement[key],value)


def test_short_source_proposals_still_need_seventy_pixels_before_qualification():
    measurement=_two_short_source_pieces_measurement()
    measurement['query_uv'][:,0]*=.5
    measurement['candidate_uv'][:,0]*=.5
    measurement['candidate_sides'][:,:,0]*=.5
    measurement['candidate_side_quantiles_uv'][:,:,:,0]*=.5
    fragments,report=qualify_visible_line_fragments(measurement,[0])
    assert fragments==[]
    assert report['local_source_intervals']==[]
    assert report['coherent_source_interval_reports']==[]


def test_visible_line_band_cache_keeps_masked_local_and_coherent_results(monkeypatch):
    from rinkspace.session import material_tracks as tracks
    from rinkspace.session import source_shape_context as context

    first=_two_short_source_pieces_measurement()
    measurement={}
    for key,value in first.items():
        if key=='query_curve':second=np.ones_like(value)
        elif key=='candidate_query':second=value+len(first['query_curve'])
        elif key in ('query_uv','candidate_uv','candidate_sides','candidate_side_quantiles_uv'):
            second=value+[0.,100.]
        else:second=value.copy()
        measurement[key]=np.concatenate([value,second])
    measurement['query_interval']=np.tile([-40.,40.],(len(measurement['query_curve']),1))
    mask=np.zeros((400,500),bool)
    # This only blocks a measured band of the second primitive. Its query
    # stays visible, and the first primitive's short pieces must still join.
    mask[321,120]=True
    original={key:value.copy() for key,value in measurement.items()}
    original_mask=mask.copy()
    qualify=tracks.qualify_line_candidates

    def uncached(*args,**kwargs):
        kwargs['source_band_unmasked']=None
        return qualify(*args,**kwargs)

    monkeypatch.setattr(tracks,'qualify_line_candidates',uncached)
    expected=tracks.qualify_visible_line_fragments(measurement,[0,1],mask)
    monkeypatch.setattr(tracks,'qualify_line_candidates',qualify)
    measured=context.measured_band_unmasked
    calls=[]

    def counted(quantiles,exclusion):
        calls.append(len(quantiles))
        return measured(quantiles,exclusion)

    monkeypatch.setattr(context,'measured_band_unmasked',counted)
    actual=tracks.qualify_visible_line_fragments(measurement,[0,1],mask)
    np.testing.assert_equal(actual,expected)
    assert calls==[len(measurement['candidate_uv'])]
    assert actual[0][0]['candidate_ids']==[0,1,2,3]
    assert actual[1]['local_source_intervals']
    assert actual[1]['coherent_source_interval_reports']
    proposals=tracks.assemble_fragments(measurement,[{},{}])

    def no_reassembly(*args,**kwargs):
        raise AssertionError('Explicit source fragments must be reused')

    monkeypatch.setattr(tracks,'assemble_fragments',no_reassembly)
    calls.clear()
    reused=tracks.qualify_visible_line_fragments(measurement,[0,1],mask,fragments=proposals)
    np.testing.assert_equal(reused,expected)
    assert calls==[len(measurement['candidate_uv'])]
    for key,value in original.items():np.testing.assert_array_equal(measurement[key],value)
    np.testing.assert_array_equal(mask,original_mask)


@pytest.mark.parametrize('cached',[np.ones(3,bool),np.ones(4,int)])
def test_cached_source_band_mask_requires_boolean_per_candidate(cached):
    measurement=_two_short_source_pieces_measurement()
    measurement['query_interval']=np.tile([-40.,40.],(len(measurement['query_curve']),1))
    with pytest.raises(ValueError,match='boolean per candidate'):
        qualify_line_candidates(measurement,[0],np.zeros((300,500),bool),
            require_paired_boundaries=True,native_shape_guided=True,
            use_source_quantiles=True,source_band_unmasked=cached)
