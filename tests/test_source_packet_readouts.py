from copy import deepcopy
from types import SimpleNamespace
import pytest

from rinkspace.session.source_inputs import SourceEnvelope
from rinkspace.session.source_observations import normalize_source


def _packet():
    return dict(features=[dict(id='ice.line',measurement_operator='paint_center',
        source_evidence_measured_in_this_frame=True,projected_uv_used_as_observations=False,
        physical_identity_evidence='historical source identity, still needs current verification',
        source_evidence_group_id='original-correlated-line',user_confirmed=False,
        source_line_fit=dict(current_frame_only=True,direct_readouts=[dict(uv=[10.125,20.25],
            source_evidence=dict(actual_source_sides_uv=[[10.125,18.25],[10.125,22.25]],
                profile_normal=[0.,1.],actual_source_support_outside_person_and_broadcast_masks=True))]),
        segments=[dict(uv=[[10.,20.]],representative_points_share_line_estimate=True,
            source_evidence=[dict(individual_native_pixel_readout=False)])])])


def _read(packet):
    envelope=SourceEnvelope({'packet_ref':{'path':'original.json','sha256':'a'*64}},'s08',1742848,{'packet':packet},())
    geometry=SimpleNamespace(lookup=lambda _:dict(id='current-native-line',layer='ice',measurement='centreline'))
    return normalize_source(envelope,None,geometry)


def test_direct_material_profiles_are_recovered_without_using_fitted_representatives():
    packet=_packet();before=deepcopy(packet);rows=_read(packet)
    assert packet==before and len(rows)==2
    direct,representative=rows
    assert direct.actual_uv==[10.125,20.25] and direct.identity=='named_source_candidate'
    assert direct.source_group_id==representative.source_group_id=='original-correlated-line'
    assert direct.source_ref['json_pointer']=='/features/0/source_line_fit/direct_readouts/0'
    assert representative.original_uv==[10.,20.] and representative.actual_uv is None
    assert direct.review['human_scope_extended'] is False


def test_wide_original_profiles_supply_full_query_extent_without_moving_observations():
    import numpy as np
    from rinkspace.session.source_observations import SourceCandidate,source_queries
    rows=[]
    for x in (20.,50.):
        rows.append(SourceCandidate(str(x),'s02',256,{},[x,50.],'ice.line',actual_uv=[x,50.],
            native_path_id='native/line',source_group_id='original-span',normal=[0.,1.],
            evidence=dict(current_semantics_match=True,pixel_support=dict(
                side_interface_quartiles_uv=[[[x,30.],[x,35.],[x,40.]],[[x,60.],[x,65.],[x,70.]]]))))
    before=deepcopy(rows)
    result=source_queries(rows,{'source_pts':256,'shot_id':'s02'},SimpleNamespace(bind=lambda g:(g,{})))[0]
    np.testing.assert_array_equal(result['points_uv'],[[20.,50.],[50.,50.]])
    assert result['material_profile_half_extent_px_per_point'].tolist()==[40.,40.]
    assert rows==before and result['identity_confirmed'] is False


@pytest.mark.parametrize('paired',[True,False])
@pytest.mark.parametrize('source_variant',['complete_frame','selected_definition','contradictory_support'])
def test_original_board_semantics_reach_query_without_becoming_human_acceptance(paired,source_variant):
    from rinkspace.session.source_observations import source_queries
    from rinkspace.vision_association import associate
    binding=dict(source_sha256='a'*64,source_pts=256,time_base='1/12800',shot_id='s02')
    context=dict(schema='rinkfit.dense_native_board_true_material_boundaries_source_frame.v1',
        native_pts=256,shot_id='s02',time_base='1/12800',decoded_native_BGR_sha256='d'*64,
        paired_blue_cap_and_white_board_body_semantics=paired,
        yellow_contact_from_current_global_band_not_accumulated_extrapolation=True,
        material_roles=['board.top','board.contact'])
    supports=[dict(uv=[x,40.],native_pts=256,shot_id='s02',time_base='1/12800',feature_id='board.top',
        segment_id='original-board-span',pixels_measured_in_this_frame=True,source_uv_from_model=False,
        model_or_world_projection_used=False,
        pixel_support=dict(current_pixel_profile_reread=True,normal=[0.,1.])) for x in (20.,60.)]
    if source_variant!='complete_frame':
        context['schema']='rinkfit.dense_native_board_source_frame.v1'
        context['local_material_requalification']=dict(actual_current_RGB_reread=paired,original_anchored_source_UV_preserved=True)
        if source_variant=='contradictory_support':
            for support in supports:support['definition']='lower edge of blue cap, internal interface'
    resolver=SimpleNamespace(resolve=lambda ref:dict(payload=supports[ref['index']],source_frame_context=context,
        provenance=dict(actual_file_sha256='e'*64,reference=dict(file='original-board.json',json_pointer=f"/rows/{ref['index']}"))))
    definition=dict(id='current/top',label='board.top',layer='board',measurement='contour')
    geometry=SimpleNamespace(lookup=lambda _:definition,bind=lambda values:(values,{}))
    candidates=[]
    for i,support in enumerate(supports):
        envelope=SourceEnvelope({},'s02',256,dict(uv=support['uv'],feature_id='board.top',source_point_id=str(i),
            measurement_definition='upper edge of blue rim at top of vertical board',
            source_UV_is_direct_measurement=True,user_confirmed=False,source_pixel_or_annotation_ref={'index':i}),())
        candidates.extend(normalize_source(envelope,resolver,geometry))
    original=deepcopy(supports)
    query=source_queries(candidates,binding,geometry)[0]
    result=associate({'binding':binding},[query],geometry)['queries'][0]
    assert supports==original and all(c.actual_uv==c.original_uv and c.review['human_scope_extended'] is False for c in candidates)
    confirmed=paired and source_variant!='contradictory_support'
    assert result['identity_confirmed'] is confirmed
    if confirmed:
        assert result['source_board_view']=='front_face'
        assert result['identity_authority']['human_pixel_acceptance'] is False
        assert result['source_board_view_evidence']['decoded_native_BGR_sha256']=='d'*64
    else:assert 'source_board_view' not in result


@pytest.mark.parametrize('invalid',[None,'unpaired','projected','wrong_pts','untyped','rejected'])
@pytest.mark.parametrize('context_kind',['paired_guard','exact_source_binding'])
def test_archived_packet_board_view_is_bound_to_paired_actual_material_runs(invalid,context_kind):
    from rinkspace.session.source_observations import source_queries
    definitions={'board.top':dict(id='current/top',label='board.top',layer='board',measurement='contour'),
        'board.bottom':dict(id='current/bottom',label='board.bottom',layer='board',measurement='contour')}
    geometry=SimpleNamespace(lookup=lambda name:definitions.get(name),bind=lambda rows:(rows,{}))
    features=[]
    for role,name,material,y in [('board_top','board.top','blue_cap_upper_body_edge',20.),
            ('board_contact','board.bottom','yellow_kickplate_body_lower_edge_reflection_excluded',80.)]:
        features.append(dict(id=name,physical_surface='board',review_layer=role,allowed_for_fit=False,
            segments=[dict(uv=[[30.,y],[60.,y]],source_native_readout_uv=[[30.,y+.2],[60.,y-.1]],
                pixels_measured_in_this_frame=False,representative_points_share_line_estimate=True,
                source_evidence_group_id='original/'+role,native_material_definition=material,
                measurement_origin='current_native_whole_material_joint_curve',current_interval_local_colour_support_count=2)]))
    packet=dict(schema='rinkspace.single-frame-source-review.v1',source_pts=256,
        source=dict(native_pts=256,time_base='1/12800',source_video_sha256='a'*64,decoded_BGR_sha256='b'*64,
            original_PTS_verified=True,PNG_roundtrip_lossless_verified=True),features=features,
        observation_coordinates_from_projection=False,board_material_ownership_guard=dict(
            source='current_measured_blue_cap_top_and_yellow_body_bottom',used_only_for_exclusion_not_observations=True))
    if context_kind=='exact_source_binding':
        packet.pop('board_material_ownership_guard')
        packet.update(current_exact_source_binding_and_mask_verified=True,current_source_only_readout=True,
            source_binding=dict(path='exact_binding.json',sha256='d'*64))
        packet['source']['source_binding']=deepcopy(packet['source_binding'])
    if invalid=='unpaired':features.pop()
    if invalid=='projected':packet['observation_coordinates_from_projection']=True
    if invalid=='wrong_pts':packet['source']['native_pts']=512
    if invalid=='untyped':features[0]['segments'][0]['native_material_definition']='gray_top_deck_outer_edge'
    if invalid=='rejected':features[0]['current_source_identity_review_rejected']=True
    before=deepcopy(packet)
    envelope=SourceEnvelope({'packet_ref':dict(path='archived.json',sha256='c'*64)},'s07',256,{'packet':packet},())
    rows=normalize_source(envelope,None,geometry)
    binding=dict(source_sha256='a'*64,source_pts=256,time_base='1/12800',shot_id='s07')
    queries=source_queries(rows,binding,geometry)
    assert packet==before
    assert all(c.review['human_scope_extended'] is False for c in rows)
    assert rows[0].original_uv==rows[0].actual_uv==[30.,20.2]
    assert rows[0].source_group_id=='original/board_top'
    assert all(bool(q.get('source_board_view'))==(invalid is None) for q in queries)
    if invalid is None:
        assert queries[0]['source_board_view_evidence']['decoded_native_BGR_sha256']=='b'*64
        unknown=deepcopy(rows[:2])
        for i,c in enumerate(unknown):
            c.candidate_id='unknown/'+str(i);c.original_uv=[c.original_uv[0],10.]
            c.actual_uv=list(c.original_uv);c.evidence.pop('source_board_material_frame')
        mixed=source_queries(rows+unknown,binding,geometry)
        top=[q for q in mixed if q['id']=='current/top']
        assert len(top)==2
        assert sum(q.get('source_board_view')=='front_face' for q in top)==1


@pytest.mark.parametrize('failure',['wrong_pixel','projection','previous_frame','masked','withdrawn'])
def test_direct_readout_preserves_failed_evidence_without_promoting_it(failure):
    packet=_packet();feature=packet['features'][0]
    if failure=='wrong_pixel':feature['source_line_fit']['direct_readouts'][0]['uv'][1]+=5
    elif failure=='projection':feature['projected_uv_used_as_observations']=True
    elif failure=='previous_frame':feature['source_line_fit']['current_frame_only']=False
    elif failure=='masked':feature['source_line_fit']['direct_readouts'][0]['source_evidence']['actual_source_support_outside_person_and_broadcast_masks']=False
    else:packet['unresolved_source_regions']=[dict(feature_id='ice.line',current_source_identity_review_rejected=True,reason='original correction')]
    before=deepcopy(packet);direct=_read(packet)[0]
    assert packet==before and direct.original_uv==feature['source_line_fit']['direct_readouts'][0]['uv']
    if failure=='withdrawn':assert direct.withdrawal['reason']=='original correction' and direct.identity=='rejected'
    else:assert direct.actual_uv is None and direct.identity=='unresolved'


def test_original_manual_segment_supplies_query_normals_without_inventing_observation_groups():
    import numpy as np
    from rinkspace.session.source_observations import SourceCandidate,source_queries
    rows=[]
    for index in (2,0,1):
        rows.append(SourceCandidate(f'original/{index}','s02',256,{},[10.+index*5.,20.],
            'ice.line',actual_uv=[10.+index*5.,20.],native_path_id='native/line',
            review=dict(human_source_ref=dict(reference=dict(file='original.json',sha256='a'*64,
                json_pointer=f'/features/7/segments/0/uv/{index}')))))
    original=deepcopy(rows)
    geometry=SimpleNamespace(bind=lambda g:(g,{}))
    query=source_queries(rows,dict(shot_id='s02',source_pts=256),geometry)[0]
    np.testing.assert_allclose(query['points_uv'],[[10.,20.],[15.,20.],[20.,20.]])
    np.testing.assert_allclose(query['normals'],[[0.,1.]]*3)
    assert query['segment_ids']==[0,0,0]
    assert query['original_dependence_groups_per_point']==[None,None,None]
    assert rows==original


def test_same_exposure_material_profile_definition_reaches_queries_without_native_color_guess():
    from rinkspace.session.source_observations import SourceCandidate,source_queries
    rows=[SourceCandidate(f'p/{i}','s02',256,{},[10.+i*5,20.],'ice.outer',
        actual_uv=[10.+i*5,20.],normal=[0.,1.],native_path_id='native/outer',source_group_id='original-ring',
        evidence=dict(current_semantics_match=True,pixel_support=dict(material_signal='dark',material_transition='dark_to_light')))
        for i in range(2)]
    geometry=SimpleNamespace(bind=lambda g:(g,{}));binding=dict(shot_id='s02',source_pts=256)
    query=source_queries(rows,binding,geometry)[0]
    assert query['signal']=='dark_band' and query['material_outward_transition']=='dark_to_light'
    assert query['source_material_definition_evidence']['definition_from_native_style'] is False
    rows[1].evidence['pixel_support']['material_transition']='light_to_dark'
    conflicted=source_queries(rows,binding,geometry)[0]
    assert 'material_outward_transition' not in conflicted
    assert conflicted['source_material_definition_evidence']['conflicting_transitions'] is True
