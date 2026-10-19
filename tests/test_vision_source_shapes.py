import hashlib
import cv2
import numpy as np
import pytest

from rinkspace.vision_source_shapes import (SourceMaterialShapeReader, partial_source_material_readings,
    source_circle_search_guides, source_paint_fragment_readings, source_primitive_material_readings)
from rinkspace.vision_source_shapes import bind_confirmed_source_instances, continue_confirmed_source_instances


def source_packet(rgb, *, straight=False):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 3, 8)
    yy, xx = np.nonzero(edges)
    components = []
    # Source edge islands arrive independently, as in a fragmented reader.
    for a, b in zip(np.linspace(0, rgb.shape[1], 7).astype(int)[:-1],
                    np.linspace(0, rgb.shape[1], 7).astype(int)[1:]):
        chosen = (xx >= a) & (xx < b)
        if chosen.sum() >= 6:
            components.append(dict(kind="ice_paint", source_support_uv=np.stack([xx[chosen], yy[chosen]], 1)))
    return dict(source_binding=dict(source_sha256="a"*64, source_pts=100, time_base="1/90000",
                                   shot_id="synthetic_native", native_hw=list(rgb.shape[:2])),
                RGB_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(), source_components=components,
                source_quality_rejections={"old_width_rejection_preserved": [[12, 10, 4]]})


def source_annulus():
    rgb = np.full((200, 520, 3), 220, np.uint8)
    cv2.ellipse(rgb, (260, 100), (175, 45), 0, 0, 360, (185, 185, 185), -1)
    cv2.ellipse(rgb, (260, 100), (112, 29), 0, 0, 360, (220, 220, 220), -1)
    cv2.ellipse(rgb, (260, 100), (45, 12), 0, 0, 360, (185, 185, 185), -1)
    return cv2.GaussianBlur(rgb, (0, 0), 2.3)


def test_source_annulus_keeps_three_rgb_loops_without_world_identity_or_disc_guess():
    rgb = source_annulus()
    actual = source_circle_search_guides(rgb, source_packet(rgb), hypothesis_count=80)
    assert len(actual["guides"]) == 3
    assert {row["material_contour_depth"] for row in actual["guides"]} == {0, 1, 2}
    assert actual["source_quality_rejections"]["old_width_rejection_preserved"] == [[12, 10, 4]]
    for row in actual["guides"]:
        assert row["physical_role"] == "source_paint_role_unresolved"
        assert row["physical_id"] is None and not row["identity_confirmed"]
        assert not row["fitted_points_used_as_measurements"]
        assert "actual_source_uv" not in row and "actual_boundary_uv" not in row
        profile = row["full_curve_source_RGB_profiles"]
        assert not profile["width_or_weakness_used_as_hard_veto"]
        assert profile["wide_source_peak"].any() or profile["weak_relative_to_old_extractor"].any()
        assert np.allclose(row["image_conic_center_uv"], [260, 100], atol=3)
    assert not actual["event_or_world_geometry_used"] and not actual["is_training_label"]
    assert actual["unresolved_source_components"]


def test_distributed_straight_shadow_bands_are_not_closed_material_circle_guides():
    rgb = np.full((200, 520, 3), 220, np.uint8)
    cv2.rectangle(rgb, (0, 78), (519, 96), (185, 185, 185), -1)
    cv2.rectangle(rgb, (0, 124), (519, 140), (185, 185, 185), -1)
    rgb = cv2.GaussianBlur(rgb, (0, 0), 2.)
    actual = source_circle_search_guides(rgb, source_packet(rgb), hypothesis_count=80)
    assert not actual["guides"]
    assert actual["original_source_components_not_erased"]
    assert actual["unresolved_source_components"]


def test_source_shape_organization_rejects_cross_frame_and_replaced_rgb():
    rgb = source_annulus()
    packet = source_packet(rgb)
    with pytest.raises(ValueError, match="source_pts"):
        source_circle_search_guides(rgb, packet, binding={**packet["source_binding"], "source_pts": 101})
    rgb[0, 0] = 0
    with pytest.raises(ValueError, match="original RGB differ"):
        source_circle_search_guides(rgb, packet)


def test_temporal_navigation_rereads_rgb_and_preserves_body_missing_runs():
    from types import SimpleNamespace
    rgb = source_annulus()
    previous = source_packet(rgb)
    reader = SourceMaterialShapeReader(hypothesis_count=40)
    first = reader.read(rgb, previous)
    changed = rgb.copy()
    cv2.rectangle(changed, (230, 45), (290, 156), (40, 40, 40), -1)
    packet = source_packet(changed)
    packet["source_binding"]["source_pts"] = 101
    height, width = rgb.shape[:2]
    pair = SimpleNamespace(forward=np.zeros((1, 2, height, width), np.float32),
        valid_forward=np.ones((1, 1, height, width), bool),
        metadata=dict(source_alignment_verified=True,
                      source_bindings=[previous["source_binding"], packet["source_binding"]]))
    actual = reader.read(changed, packet, flow_pair=pair)
    assert len(actual["source_shape_readings"]) == 3
    for row in actual["source_shape_readings"]:
        assert row["full_curve_missing_or_conflicting_profiles"].any()
        uv = row["actual_source_uv"]
        assert not ((uv[:, 0] > 232) & (uv[:, 0] < 288)).any()
        assert row["physical_id"] is None
        assert not row["fitted_or_flow_pixels_used_as_measurements"]
    wrong = SimpleNamespace(**{**pair.__dict__, "metadata": {**pair.metadata,
                            "source_bindings": [previous["source_binding"], previous["source_binding"]]}})
    with pytest.raises(ValueError, match="source_pts"):
        reader.read(changed, {**packet, "source_binding": {**packet["source_binding"], "source_pts": 102}}, flow_pair=wrong)


def test_partial_material_reader_keeps_clipped_rgb_arcs_without_frame_chord():
    rgb = np.full((200, 520, 3), 220, np.uint8)
    cv2.ellipse(rgb,(260,194),(230,65),0,0,360,(175,175,175),-1)
    cv2.ellipse(rgb,(260,194),(155,39),0,0,360,(220,220,220),-1)
    rgb=cv2.GaussianBlur(rgb,(0,0),2.)
    binding=source_packet(rgb)['source_binding']
    actual=partial_source_material_readings(rgb,binding=binding,
        source_ice_search_mask=np.ones(rgb.shape[:2],bool),source_ice_search_binding=binding)
    assert actual['source_shape_readings']
    for guide,reading in zip(actual['guides'],actual['source_shape_readings']):
        assert guide['primitive']=='free_curve' and not guide['closed']
        assert guide['physical_role']=='source_paint_role_unresolved'
        assert reading['physical_id'] is None
        points=reading['actual_source_uv']
        assert (points[:,1]<198).all()
        assert not reading['fitted_or_flow_pixels_used_as_measurements']
        for run in reading['measured_runs']:
            # Contiguous observed samples cannot draw a chord through a gap.
            assert np.linalg.norm(np.diff(run['actual_source_uv'],axis=0),axis=1).max(initial=0)<30


def test_partial_material_reader_rejects_straight_shadow_and_wrong_scope_source():
    rgb=np.full((200,520,3),220,np.uint8)
    cv2.rectangle(rgb,(0,150),(519,180),(175,175,175),-1)
    rgb=cv2.GaussianBlur(rgb,(0,0),2.)
    binding=source_packet(rgb)['source_binding']
    actual=partial_source_material_readings(rgb,binding=binding,
        source_ice_search_mask=np.ones(rgb.shape[:2],bool),source_ice_search_binding=binding)
    assert not actual['source_shape_readings']
    assert actual['rejected_source_material_candidates']
    with pytest.raises(ValueError,match='source_pts'):
        partial_source_material_readings(rgb,binding=binding,source_ice_search_mask=np.ones(rgb.shape[:2],bool),
                                        source_ice_search_binding={**binding,'source_pts':101})


def native_paint_search_packet(rgb):
    packet=source_packet(rgb)
    probabilities=np.zeros((1,6,*rgb.shape[:2]),np.float32)
    probabilities[:,0]=.1;probabilities[:,3]=.9
    packet['source']=dict(source_binding=packet['source_binding'],
        classes=['background','ice_contact','board_top','ice_paint','ad_boundary','shadow_or_reflection'],
        semantic_probabilities=probabilities,
        metadata=dict(probabilities_conditioned_on_geometry=False,world_camera_projection_input_used=False))
    return packet


def test_flat_perspective_paint_arcs_keep_actual_signed_rgb_without_curvature_veto():
    rgb=np.full((160,480,3),220,np.uint8)
    cv2.ellipse(rgb,(240,80),(190,6),0,0,360,(160,160,160),-1)
    cv2.ellipse(rgb,(240,80),(115,2),0,0,360,(220,220,220),-1)
    rgb=cv2.GaussianBlur(rgb,(0,0),1.1)
    binding=source_packet(rgb)['source_binding']
    result=partial_source_material_readings(rgb,binding=binding,
        source_ice_search_mask=np.ones(rgb.shape[:2],bool),source_ice_search_binding=binding,
        source_paint_probabilities=np.ones(rgb.shape[:2],np.float32),probability_source_binding=binding)
    points=np.concatenate([row['actual_source_uv'] for row in result['source_shape_readings']])
    assert (points[:,1]<80).any() and (points[:,1]>80).any()
    assert any(row['geometry_did_not_veto_typed_signed_RGB_fragment'] for row in result['guides'])
    assert all(row['physical_id'] is None and row['primitive']=='free_curve' for row in result['guides'])
    assert all(not row['fitted_or_flow_pixels_used_as_measurements'] for row in result['source_shape_readings'])
    with pytest.raises(ValueError,match='source_pts'):
        partial_source_material_readings(rgb,binding=binding,
            source_ice_search_mask=np.ones(rgb.shape[:2],bool),source_ice_search_binding=binding,
            source_paint_probabilities=np.ones(rgb.shape[:2],np.float32),
            probability_source_binding={**binding,'source_pts':101})


def test_wide_clipped_rgb_fragments_keep_measured_intervals_without_mask_or_size_veto():
    rgb=np.full((160,480,3),220,np.uint8)
    cv2.ellipse(rgb,(240,166),(220,62),0,0,360,(160,160,160),-1)
    rgb=cv2.GaussianBlur(rgb,(0,0),3.2)
    packet=native_paint_search_packet(rgb)
    result=source_paint_fragment_readings(rgb,packet,
        source_ice_search_mask=np.zeros(rgb.shape[:2],bool),source_ice_search_binding=packet['source_binding'])
    assert result['source_fragment_readings'] and not result['source_ice_mask_used_as_hard_veto']
    assert any(row['source_quality_flags']['wide_relative_to_previous_extractor'].any()
               for row in result['source_fragment_readings'])
    for row in result['source_fragment_readings']:
        assert np.allclose(row['actual_source_uv'],np.round(row['actual_source_uv']))
        assert row['physical_id'] is None and not row['width_or_small_component_used_as_hard_veto']
        assert row['measured_normal_interval_px'].shape==(len(row['actual_source_uv']),2)
        assert row['source_quality_flags']['outside_unverified_source_ice_search_mask'].all()
        for run in row['measured_runs']:
            assert np.linalg.norm(np.diff(run['actual_source_uv'],axis=0),axis=1).max(initial=0)<=np.sqrt(2)+1e-6
            assert not run['contains_interpolated_or_fitted_pixels']
    changed=rgb.copy();changed[0,0]=0
    with pytest.raises(ValueError,match='original RGB differ'):
        source_paint_fragment_readings(changed,packet)
    packet['source']['metadata']['probabilities_conditioned_on_geometry']=True
    with pytest.raises(ValueError,match='no-prior'):
        source_paint_fragment_readings(rgb,packet)


def test_two_layer_material_search_rereads_lower_line_like_pixels_without_world_identity():
    rgb=np.full((200,520,3),220,np.uint8)
    cv2.ellipse(rgb,(260,100),(175,45),0,0,360,(170,170,170),-1)
    cv2.ellipse(rgb,(260,100),(112,29),0,0,360,(220,220,220),-1)
    rgb=cv2.GaussianBlur(rgb,(0,0),1.8)
    packet=native_paint_search_packet(rgb);binding=packet['source_binding']
    primitive=np.zeros((3,*rgb.shape[:2]),np.float32);primitive[2]=1
    gray=rgb[:,:,0]<219;primitive[2,gray]=0;primitive[1,gray]=1
    # The same visible physical border is locally classified as a line.
    lower=np.zeros(rgb.shape[:2],bool);lower[139:150,205:315]=True;lower &= gray
    primitive[1,lower]=0;primitive[0,lower]=1
    result=source_primitive_material_readings(rgb,packet,primitive_probabilities=primitive,
        primitive_source_binding=binding,source_ice_search_mask=np.ones(rgb.shape[:2],bool),
        source_ice_search_binding=binding)
    assert len(result['source_shape_readings'])==2
    outer=next(row for row in result['source_shape_readings']
               if row['material_topological_role_candidate']=='source_dark_material_outer_border')
    assert ((outer['actual_source_uv'][:,1]>139)&(outer['actual_source_uv'][:,0]>210)&
            (outer['actual_source_uv'][:,0]<310)).any()
    for row in result['source_shape_readings']:
        assert row['physical_id'] is None and not row['fitted_or_flow_pixels_used_as_measurements']
    assert all(row['minimum_nested_material_layers_required']==2 for row in result['guides'])
    with pytest.raises(ValueError,match='source_pts'):
        source_primitive_material_readings(rgb,packet,primitive_probabilities=primitive,
            primitive_source_binding={**binding,'source_pts':101},
            source_ice_search_mask=np.ones(rgb.shape[:2],bool),source_ice_search_binding=binding)


def test_open_material_memory_rereads_source_rgb_without_fitting_or_closing_ellipse():
    from types import SimpleNamespace
    rgb=np.full((160,480,3),220,np.uint8)
    cv2.ellipse(rgb,(240,166),(220,62),0,0,360,(160,160,160),-1)
    cv2.ellipse(rgb,(240,166),(145,35),0,0,360,(220,220,220),-1)
    rgb=cv2.GaussianBlur(rgb,(0,0),2.)
    packet=native_paint_search_packet(rgb);binding=packet['source_binding']
    primitive=np.zeros((3,*rgb.shape[:2]),np.float32);primitive[2]=1
    gray=rgb[:,:,0]<219;primitive[2,gray]=0;primitive[1,gray]=1
    reader=SourceMaterialShapeReader()
    first=reader.read(rgb,packet,primitive_probabilities=primitive,primitive_source_binding=binding,
        source_ice_search_mask=np.ones(rgb.shape[:2],bool),source_ice_search_binding=binding)
    assert first['guides'] and all(row['primitive']=='free_curve' for row in first['guides'])
    changed=rgb.copy();cv2.rectangle(changed,(212,98),(260,158),(40,40,40),-1)
    current=native_paint_search_packet(changed);current['source_binding']['source_pts']=101
    h,w=rgb.shape[:2]
    pair=SimpleNamespace(forward=np.zeros((1,2,h,w),np.float32),valid_forward=np.ones((1,1,h,w),bool),
        metadata=dict(source_alignment_verified=True,source_bindings=[binding,current['source_binding']]))
    result=reader.read(changed,current,flow_pair=pair)
    assert result['guides'] and result['source_fragment_readings']
    assert all(not row['image_conic_used_as_temporal_search_model'] for row in result['guides'])
    for row in result['source_shape_readings']:
        uv=row['actual_source_uv']
        assert not ((uv[:,0]>214)&(uv[:,0]<258)).any()
        assert row['physical_id'] is None and not row['fitted_or_flow_pixels_used_as_measurements']
        assert row['full_curve_missing_or_conflicting_profiles'].any()


def confirmed_reference_packet(binding):
    from rinkspace.vision_geometry import source_annotations_to_guides
    phase=np.linspace(0,2*np.pi,48,endpoint=False)
    uv=np.stack((260+175*np.cos(phase),100+45*np.sin(phase)),1)
    annotations=[dict(feature_id='R1:outer',group_id='confirmed_R1_outer',kind='ice_paint',primitive='circle',
        uv=uv,state='user_confirmed_visual_supervision',authority={'origin':'direct_user_provided_drawing'},
        physical_role='filled_disc_boundary')]
    return source_annotations_to_guides(annotations,source_binding=binding,annotation_source_binding=binding)


def test_confirmed_parent_identity_attaches_source_layers_without_changing_rgb_measurements():
    rgb=source_annulus();packet=source_packet(rgb)
    result=source_circle_search_guides(rgb,packet,hypothesis_count=40)
    original={row['id']:row['actual_source_uv'].copy() for row in result['source_shape_readings']}
    bound=bind_confirmed_source_instances(result,confirmed_reference_packet(packet['source_binding']),
        role_definitions={'R1:middle':{'semantic_definition':'named_filled_disc_color_transition_boundary'},
                          'R1:inner':{'semantic_definition':'named_filled_disc_color_transition_boundary'}})
    assert [entry['physical_id'] for entry in bound['confirmed_source_instances']]==['R1']
    assert {row['physical_curve_id'] for row in bound['source_shape_readings']}=={'R1:outer','R1:middle','R1:inner'}
    for row in bound['source_shape_readings']:
        assert row['physical_id']=='R1' and row['physical_identity_verified']
        assert np.array_equal(row['actual_source_uv'],original[row['id']])
        assert not row['physical_paint_edge_verified']
    assert not bound['human_reference_UV_used_as_actual_boundary_output']
    wrong=confirmed_reference_packet(packet['source_binding']);wrong['source_binding']={**packet['source_binding'],'source_pts':101}
    with pytest.raises(ValueError,match='source_pts'):
        bind_confirmed_source_instances(result,wrong)
    weak=confirmed_reference_packet(packet['source_binding']);weak['guides'][0]['origin']='source_annotation_search_hint'
    with pytest.raises(ValueError,match='human source identity authority'):
        bind_confirmed_source_instances(result,weak)


def test_legacy_json_peak_vectors_bind_identity_without_changing_numeric_uv_or_claiming_dtype_restore():
    import json
    rgb=source_annulus();packet=source_packet(rgb)
    result=source_circle_search_guides(rgb,packet,hypothesis_count=40)
    original={row['id']:row['actual_source_uv'].copy() for row in result['source_shape_readings']}
    def plain(value):
        if isinstance(value,np.ndarray):return value.tolist()
        if isinstance(value,np.generic):return value.item()
        raise TypeError(type(value))
    legacy=json.loads(json.dumps(result,default=plain))
    bound=bind_confirmed_source_instances(legacy,confirmed_reference_packet(packet['source_binding']))
    assert bound['source_shape_readings']
    for reading in bound['source_shape_readings']:
        assert np.array_equal(reading['actual_source_uv'],original[reading['id']])
        assert reading['physical_id']=='R1'
        normalization=reading['source_array_input_normalization']
        assert normalization['legacy_JSON_vector_indexing_normalized']
        assert not normalization['original_dtype_known'] and not normalization['original_dtype_restored']
        assert normalization['numerical_source_UV_unchanged']
    # Typed in-memory/current min1 source arrays retain their original dtype.
    typed=bind_confirmed_source_instances(result,confirmed_reference_packet(packet['source_binding']))
    assert all(not row['source_array_input_normalization']['legacy_JSON_vector_indexing_normalized']
               for row in typed['source_shape_readings'])


def test_known_parent_missing_retains_authority_without_gray_fragment_name_or_copied_uv():
    rgb=source_annulus();binding=source_packet(rgb)['source_binding']
    empty=dict(source_binding=binding,guides=[],source_shape_readings=[])
    bound=bind_confirmed_source_instances(empty,confirmed_reference_packet(binding))
    assert bound['confirmed_source_instances'][0]['visibility_status']=='source_missing_identity_retained'
    assert not bound['source_shape_readings']
    current=dict(source_binding={**binding,'source_pts':101},guides=[],source_shape_readings=[],
                 previous_source_binding=binding,memory_update='missing_source_bound_temporal_correspondence')
    result=continue_confirmed_source_instances(current,bound)
    assert result['confirmed_source_instances'][0]['physical_id']=='R1'
    assert result['confirmed_source_instances'][0]['identity_confirmed']
    assert result['confirmed_source_instances'][0]['actual_RGB_point_count']==0
    assert not result['source_shape_readings'] and not result['human_reference_UV_reused_as_current_frame_guide']


def test_consecutive_missing_motion_does_not_forget_confirmed_physical_identity():
    rgb = source_annulus()
    packet = source_packet(rgb)
    reader = SourceMaterialShapeReader(hypothesis_count=40)
    initial = reader.read(rgb, packet, confirmed_source_guide_packet=confirmed_reference_packet(packet['source_binding']))
    assert initial['confirmed_source_instances'][0]['identity_confirmed']
    for pts in (101, 102):
        current = source_packet(rgb)
        current['source_binding']['source_pts'] = pts
        missing = reader.read(rgb, current)
        known = missing['confirmed_source_instances'][0]
        assert known['physical_id'] == 'R1' and known['identity_confirmed']
        assert known['identity_source_binding']['source_pts'] == 100
        assert known['current_source_binding']['source_pts'] == pts
        assert known['visibility_status'] == 'source_missing_identity_retained'
        assert not missing['source_shape_readings']
        assert not missing['guides']
        assert reader.previous is missing
