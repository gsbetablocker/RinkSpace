import copy
import hashlib
import json
import numpy as np
import pytest

from rinkspace.vision_instances import (source_binding, validate_prompt, region_contours, merge_prompt_identity,
                                       propose_source_region_prompts, transport_search_box,
                                       advertisement_identity_from_constraints, SourceAdvertisementMatcher,
                                       SceneGuideSearchAdapter, printed_pattern_candidates)
from rinkspace.vision_instances import (source_print_layout_evidence, verify_source_print_contours, reread_region_edges,
                                       source_print_boundary_contours,source_print_component_continuity)
from rinkspace.vision_instances import source_person_refresh_prompts, Sam2SourceInstances, source_colour_material_evidence
from rinkspace.vision_instances import advertisement_descriptor_search_mask
from rinkspace.vision_instances import source_print_material_search_proposals
from rinkspace.vision_instances import source_print_material_frame_evidence
from rinkspace.vision_instances import current_material_region_search_box
from rinkspace.vision_instances import group_source_material_prompt_search
from rinkspace.vision_instances import (source_RGB_tensors,material_prompts_with_shared_board_search,
                                        _checked_source_RGB_tensors)
from rinkspace.vision_instances import transfer_source_context_identities, SOURCE_CORRESPONDENCE_IDENTITY


def binding(pts=100):
    return dict(source_sha256="a"*64, source_pts=pts, time_base="1/15360",
                shot_id="actual_shot_1", native_hw=[80, 120])


def prompt(object_id=1, physical_id="Ad11", origin="model_projection_search_hint"):
    return dict(object_id=object_id, physical_id=physical_id, kind="advertisement_panel",
                origin=origin, source_binding=binding(), box_xyxy=[10, 10, 80, 50])


def test_model_hint_cannot_confirm_physical_number_or_pixels():
    value = prompt()
    before = copy.deepcopy(value)
    result = validate_prompt(value, binding())
    assert value == before
    assert result["expected_id"] == "Ad11"
    assert result["physical_id"] is None
    assert not result["prompt_is_boundary_measurement"]


def test_same_appearance_does_not_merge_two_human_advertisements():
    first, second = prompt(1, "Ad11"), prompt(2, "Ad24")
    for item in (first, second):
        item["identity_authority"] = dict(kind="human_confirmed_event_catalog", reference="event/source_catalog.json")
    a, b = validate_prompt(first, binding()), validate_prompt(second, binding())
    assert (a["physical_id"], b["physical_id"]) == ("Ad11", "Ad24")
    assert a["object_id"] != b["object_id"]


def test_prompt_from_different_source_frame_rejected():
    value = prompt()
    value["source_binding"]["source_pts"] = 101
    with pytest.raises(ValueError, match="source_pts"):
        validate_prompt(value, binding())


def test_human_positions_preserved_even_if_model_hint_origin():
    value = prompt()
    value["identity_authority"] = dict(kind="direct_user_confirmation", reference="source_time_100_human_fact")
    value.update(points_uv=[[12.25, 13.75]], point_labels=[1])
    result = validate_prompt(value, binding())
    assert result["physical_id"] == "Ad11"
    assert result["points_uv"] == [[12.25, 13.75]]
    assert not result["prompt_is_boundary_measurement"]


def test_model_refresh_retains_human_identity_and_new_user_can_correct_it():
    value = prompt()
    value["identity_authority"] = dict(kind="direct_user_confirmation", reference="human_fact")
    previous = validate_prompt(value, binding())
    refreshed = merge_prompt_identity(previous, validate_prompt(prompt(), binding()))
    assert refreshed["physical_id"] == "Ad11"
    correction = {**value, "physical_id": "Ad24"}
    corrected = merge_prompt_identity(previous, validate_prompt(correction, binding()))
    assert corrected["physical_id"] == "Ad24"


def test_region_touching_image_border_never_closes_an_unseen_side():
    mask = np.zeros((80, 120), np.uint8)
    mask[20:60, :50] = 1
    contours = region_contours(mask)
    assert contours and all(not item["closed"] for item in contours)
    assert all((item["uv"][:, 0] > 1).all() for item in contours)


def test_printed_pattern_holes_are_not_outer_contours_or_extent_edges():
    mask = np.zeros((80, 120), np.uint8)
    mask[20:60, 20:80] = 1
    mask[30:50, 35:60] = 0
    contours = region_contours(mask, role_prefix="printed_pattern")
    assert {item["role"] for item in contours} == {
        "printed_pattern_external_contour", "printed_pattern_internal_hole"}
    assert sum(item["internal_hole"] for item in contours) == 1


def test_source_clock_and_native_pixels_are_not_assumed():
    result = source_binding({**binding(), "time_base": "1/90000"}, [80, 120])
    assert result["time_base"] == "1/90000"
    assert "canonical_exposure_pts" not in result
    with pytest.raises(ValueError, match="dimensions"):
        source_binding(binding(), [81, 120])


def test_timer_and_letters_do_not_seed_an_ice_contact_or_rail():
    image = np.full((80, 120, 3), 230, np.uint8)
    image[10:20, 20:30] = 0
    probability = np.zeros((2, 80, 120), np.float32)
    probability[:, 5:25, 15:35] = .9
    result = propose_source_region_prompts(image, binding(), probability,
                 ["ice_contact", "board_top"], probabilities_conditioned_on_geometry=False)
    assert not result["prompts"]
    assert result["rejected_short_or_vertical_components"]


def test_prior_conditioned_semantics_cannot_become_original_source_proposals():
    image = np.full((80, 120, 3), 230, np.uint8)
    with pytest.raises(ValueError, match="no-geometry"):
        propose_source_region_prompts(image, binding(), np.zeros((2, 80, 120)),
             ["ice_contact", "board_top"], probabilities_conditioned_on_geometry=True)


def test_search_extent_transport_is_only_affine_region_navigation():
    assert transport_search_box([20, 20, 40, 40], [10, 10, 50, 50], [20, 30, 60, 70]) == [30, 40, 50, 60]


def test_original_context_expected_number_is_not_a_new_human_pixel_fact():
    value = {**prompt(), "physical_id": None, "expected_id": "Ad23",
             "identity_authority": dict(kind="original_reference_context_and_print_correspondence",
                                        reference="authenticated_original_source.png"),
             "candidate_physical_ids": ["Ad10", "Ad23"]}
    result = validate_prompt(value, binding())
    assert result["expected_id"] == "Ad23" and result["physical_id"] is None
    assert result["identity_authority"] is None
    assert result["expected_identity_authority"]["kind"] == "original_reference_context_and_print_correspondence"
    assert result["candidate_physical_ids"] == ["Ad10", "Ad23"]


def source_transfer_fixture(number=17):
    current=binding(200);reference=binding(100)
    row=dict(template_id=f'actual:Ad{number}:part',source_binding=current,reference_source_binding=reference,
        source_packet_sha256='b'*64,reference_physical_id=f'Ad{number}',physical_id=None,
        expected_id=None,content_id='event_name_group',candidate_numbers=[4,17],appearance_match=True,
        actual_source_keypoints_uv=[[11,31],[16,32],[21,33],[31,34],[41,35],[51,36]],
        event_identity_authority=dict(kind='human_confirmed_event_catalog',reference='event/catalog.json'))
    context=dict(reference_source_binding=reference,query_source_binding=current,source_packet_sha256='b'*64,
        distributed_distinctive_context_support=True,event_order_consistent=True,
        distinctive_descriptor_match_count=8,background_inlier_count=14,matched_reference_order=[number])
    query_prompt={**prompt(), 'source_binding':current,'physical_id':None,'expected_id':None,
        'appearance_correspondence_template_id':row['template_id']}
    return current,row,context,query_prompt


def test_actual_context_transfers_reference_copy_without_claiming_human_query_pixels():
    current,row,context,query_prompt=source_transfer_fixture()
    original_points=copy.deepcopy(row['actual_source_keypoints_uv'])
    rows,prompts=transfer_source_context_identities([row],[query_prompt],[context],current,
        event_id='event_a',query_RGB_sha256='c'*64)
    validated=validate_prompt(prompts[0],current)
    assert rows[0]['physical_id']=='Ad17' and rows[0]['candidate_numbers']==[4,17]
    assert validated['physical_id']=='Ad17' and validated['identity_confirmed_by_source_correspondence']
    assert not validated['identity_confirmed_by_source_human']
    assert not rows[0]['source_pixel_to_expected_identity_verified']
    assert rows[0]['actual_source_keypoints_uv']==original_points
    assert validated['identity_authority']['kind']==SOURCE_CORRESPONDENCE_IDENTITY
    assert not validated['identity_authority']['boundary_UV_verified_by_identity_transfer']
    assert not validated['prompt_is_boundary_measurement']


@pytest.mark.parametrize('failure', ['weak_context','wrong_packet','other_source','different_reference'])
def test_inconsistent_source_transfer_cannot_assign_a_repeated_physical_copy(failure):
    current,row,context,query_prompt=source_transfer_fixture()
    if failure=='weak_context':context['distributed_distinctive_context_support']=False
    elif failure=='wrong_packet':context['source_packet_sha256']='d'*64
    elif failure=='other_source':context['reference_source_binding']={**binding(100),'source_sha256':'e'*64}
    else:context['reference_source_binding']=binding(101)
    rows,prompts=transfer_source_context_identities([row],[query_prompt],[context],current,
        event_id='event_a',query_RGB_sha256='c'*64)
    assert rows[0]['physical_id'] is None and prompts[0]['physical_id'] is None


def test_two_authenticated_copies_matching_same_actual_print_remain_conflicting():
    current,row,context,query_prompt=source_transfer_fixture()
    _,other,other_context,other_prompt=source_transfer_fixture(4)
    rows,prompts=transfer_source_context_identities([row,other],[query_prompt,other_prompt],
        [context,other_context],current,event_id='event_a',query_RGB_sha256='c'*64)
    assert all(r['physical_id'] is None for r in rows)
    assert all(r['identity_status']=='conflicting_authenticated_reference_copy_correspondences' for r in rows)
    assert all(r['candidate_numbers']==[4,17] for r in rows)
    assert all(p['physical_id'] is None for p in prompts)


def test_exact_human_query_identity_survives_conflicting_machine_reference_copies():
    current,row,context,query_prompt=source_transfer_fixture()
    _,other,other_context,other_prompt=source_transfer_fixture(4)
    row.update(physical_id='Ad17',expected_id='Ad17',source_pixel_to_expected_identity_verified=True,
        identity_authority=dict(kind='direct_user_confirmation',reference='exact_source_user_identity'))
    rows,prompts=transfer_source_context_identities([row,other],[query_prompt,other_prompt],
        [context,other_context],current,event_id='event_a',query_RGB_sha256='c'*64)
    assert rows[0]['physical_id']=='Ad17' and rows[0]['source_pixel_to_expected_identity_verified']
    assert rows[0]['human_query_identity_retained_over_machine_context']
    assert rows[0]['identity_authority']['kind']=='direct_user_confirmation'
    assert prompts[0]['physical_id']=='Ad17' and prompts[0]['source_pixel_to_expected_identity_verified']
    assert rows[1]['physical_id'] is None


def test_machine_source_transfer_cannot_replace_human_identity_or_cross_current_RGB():
    current,row,context,query_prompt=source_transfer_fixture()
    _,prompts=transfer_source_context_identities([row],[query_prompt],[context],current,
        event_id='event_a',query_RGB_sha256='c'*64)
    with pytest.raises(ValueError,match='source_pts'):
        validate_prompt({**prompts[0],'source_binding':binding(201)},binding(201))
    with pytest.raises(ValueError,match='query RGB'):
        validate_prompt(prompts[0],{**current,'decoded_native_rgb_sha256':'d'*64})
    machine=validate_prompt(prompts[0],current)
    human=validate_prompt({**query_prompt,'physical_id':'Ad4',
        'identity_authority':dict(kind='direct_user_confirmation',reference='source_user_fact')},current)
    merged=merge_prompt_identity(human,machine)
    assert merged['physical_id']=='Ad4' and merged['identity_confirmed_by_source_human']
    assert not merged['identity_confirmed_by_source_correspondence']


def test_appearance_and_model_neighborhood_do_not_make_pixel_identity_ground_truth():
    b = binding()
    measured = [[12, 13], [16, 19]]
    ambiguous = advertisement_identity_from_constraints([10, 23], measured, b)
    assert ambiguous["candidate_numbers"] == [10, 23] and ambiguous["physical_id"] is None
    model = dict(status="exact_saved_camera_search_guides", source_binding=b,
                 scene_sha256="s", camera_sha256="c",
                 guides=[dict(advertisement_number=23, points_uv=[[10, 10], [20, 20]], search_radius_px=2)])
    resolved = advertisement_identity_from_constraints([10, 23], measured, b, model_search=model)
    assert resolved["expected_id"] == "Ad23" and resolved["physical_id"] is None
    assert not resolved["source_pixel_to_expected_identity_verified"]
    assert not resolved["constraint_evidence"][0]["is_pixel_identity_proof"]
    single = advertisement_identity_from_constraints([23], measured, b)
    assert single["physical_id"] is None
    with pytest.raises(ValueError, match="source_pts"):
        advertisement_identity_from_constraints([10, 23], measured, b,
              model_search={**model, "source_binding": binding(101)})


@pytest.fixture
def original_ad_packet(tmp_path):
    import cv2
    rgb = np.random.default_rng(17).integers(0, 256, (80, 120, 3), dtype=np.uint8)
    path = tmp_path/"original.png"
    assert cv2.imwrite(str(path), rgb[:, :, ::-1])
    catalog_path = tmp_path/"catalog.json"
    catalog_path.write_text(json.dumps(dict(model_id="event_a_shared", same_content_pairs=[
        dict(numbers=[4, 17], content_id="event_name_group")], positions=[dict(number=4), dict(number=17)])), encoding="utf-8")
    packet = dict(event_id="event_a", source={**binding(),
        "source_frame_path": str(path), "source_frame_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "decoded_native_rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest()},
        identity_catalog=dict(path=str(catalog_path), sha256=hashlib.sha256(catalog_path.read_bytes()).hexdigest()),
        image_left_to_right_physical_order=[17], advertisements=[dict(advertisement_number=17,
            physical_identity_human_confirmed=True, components=[
                dict(component_id="magenta_part", actual_source_navigation_roi=[10, 30, 55, 70]),
                dict(component_id="blue_part", actual_source_navigation_roi=[65, 30, 110, 70])])])
    packet_path = tmp_path/"source.json"
    packet_path.write_text(json.dumps(packet), encoding="utf-8")
    return packet_path, packet, rgb


def test_human_group_multiple_prints_keep_shared_instance_and_separate_source_components(original_ad_packet):
    path, packet, rgb = original_ad_packet
    matcher = SourceAdvertisementMatcher([path], event_id="event_a")
    assert len(matcher.templates) == 2
    assert {row["reference_physical_id"] for row in matcher.templates} == {"Ad17"}
    assert {row["source_component_id"] for row in matcher.templates} == {"magenta_part", "blue_part"}
    assert all(row["candidate_numbers"] == [4, 17] for row in matcher.templates)
    assert all(not row["navigation_roi_is_ad_boundary"] for row in matcher.templates)
    assert all(row["source_binding"]["source_pts"] == 100 for row in matcher.templates)
    assert len({row['observed_print_variant_id'] for row in matcher.templates}) == 2
    assert len({row['original_RGB_navigation_patch_sha256'] for row in matcher.templates}) == 2
    assert all(not row['same_content_pair_is_print_variant_verification'] for row in matcher.templates)
    query = {**binding(200), "decoded_native_rgb_sha256": "b"*64}
    with pytest.raises(ValueError, match="Query RGB differs"):
        matcher.read(rgb, query)
    with pytest.raises(ValueError, match="different event"):
        matcher.read(rgb, {**binding(200), "event_id": "event_b"})
    with pytest.raises(ValueError, match="different event"):
        SourceAdvertisementMatcher([path], event_id="event_b")


def test_changed_template_source_or_catalogue_cannot_retain_confirmation(original_ad_packet):
    path, packet, _ = original_ad_packet
    source_path = __import__("pathlib").Path(packet["source"]["source_frame_path"])
    source_path.write_bytes(source_path.read_bytes()+b"changed")
    with pytest.raises(ValueError, match="PNG changed"):
        SourceAdvertisementMatcher([path], event_id="event_a")


def test_changed_catalogue_cannot_transfer_old_instance_authority(original_ad_packet):
    from pathlib import Path
    path, packet, _ = original_ad_packet
    Path(packet["identity_catalog"]["path"]).write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="catalogue changed"):
        SourceAdvertisementMatcher([path], event_id="event_a")


def test_descriptor_navigation_excludes_ui_and_person_without_erasing_source_pixels():
    image = np.full((80,120,3),235,np.uint8)
    image[20:70,30:100] = [50,70,160]
    original=image.copy()
    region=np.zeros((80,120),bool);region[20:70,20:110]=True
    source_search=dict(source_binding=binding(),search_mask=region,origin='current_source_board_candidate')
    person=dict(source_binding=binding(),box_xyxy=[50,25,80,65])
    mask,evidence=advertisement_descriptor_search_mask(image,binding(),source_search=source_search,person_boxes=[person])
    assert mask[30,35] and not mask[30,60] and not mask[75,35] and not mask[5,5]
    assert np.array_equal(image,original)  # The original print/leg-gap RGB remains available for the independent reader.
    assert not evidence['RGB_measurements_erased'] and not evidence['physical_identity_confirmed_by_navigation']
    assert not evidence['source_box_navigation'][0]['box_used_as_RGB_veto']
    assert 'actual_source_uv' not in evidence
    with pytest.raises(ValueError,match='source_pts'):
        advertisement_descriptor_search_mask(image,binding(),source_search={**source_search,'source_binding':binding(101)})
    with pytest.raises(ValueError,match='RGB differs'):
        advertisement_descriptor_search_mask(image,binding(),source_search={**source_search,
            'source_binding':{**binding(),'decoded_native_rgb_sha256':'b'*64}})
    with pytest.raises(ValueError,match='source_pts'):
        advertisement_descriptor_search_mask(image,binding(),person_boxes=[{**person,'source_binding':binding(101)}])


def test_empty_descriptor_navigation_does_not_search_wall_or_promote_alias_identity(original_ad_packet):
    path,packet,rgb=original_ad_packet
    matcher=SourceAdvertisementMatcher([path],event_id='event_a')
    query=binding(200)
    reading=matcher.read(rgb,query,source_search=dict(source_binding=query,search_mask=np.zeros((80,120),bool)))
    assert not reading['prompts'] and reading['query_feature_count']==0
    assert reading['descriptor_search_evidence']['status']=='empty_source_descriptor_search_support'
    assert all(row['physical_id'] is None and row['candidate_numbers']==[4,17] for row in reading['matches'])
    assert all(not row['same_content_pair_is_print_variant_verification'] for row in reading['matches'])


def test_unmatched_visible_material_can_propose_print_search_without_borrowing_parent_identity():
    image=np.full((80,120,3),240,np.uint8)
    image[30:55,10:40]=[40,70,160]  # Existing matched parent.
    image[30:60,55:85]=[165,55,70]  # Clear new red print, no SIFT reference.
    image[30:60,96:118]=[140,55,150]  # Separate purple print, clipped native side.
    parent=dict(source_binding=binding(),object_id=2000,physical_id='Ad20',printed_pattern_extent=[10,30,40,55])
    result=source_print_material_search_proposals(image,binding(),print_parents=[parent])
    assert result['prompts']
    red=[x for x in result['material_candidates'] if x['source_material_candidate_mask'][40,65]]
    purple=[x for x in result['material_candidates'] if x['source_material_candidate_mask'][40,105]]
    assert red and purple and red[0]['object_id']!=purple[0]['object_id']
    assert all(x['physical_id'] is None and x['expected_id'] is None for x in result['prompts'])
    assert all(x['candidate_physical_ids']==[] and 'actual_source_uv' not in x for x in result['prompts'])
    assert all(not x['search_extent_is_measured_edge'] for x in result['material_candidates'])
    assert not result['sparse_ad_class_used_as_hard_domain']
    # No current board/print navigation is not permission to call the whole
    # image a board, even when the RGB contains saturated materials.
    empty=source_print_material_search_proposals(image,binding())
    assert not empty['prompts'] and empty['status']=='missing_current_source_print_or_board_navigation'
    with pytest.raises(ValueError,match='source_pts'):
        source_print_material_search_proposals(image,binding(),print_parents=[{**parent,'source_binding':binding(101)}])


def test_unmatched_chromatic_material_preserves_internal_colour_boundary_without_calling_it_print_hole():
    image=np.full((80,120,3),240,np.uint8)
    image[25:60,20:100]=[40,70,160];image[35:48,40:60]=240
    direction=np.array([40,70,160],float);direction-=direction.mean();direction/=np.linalg.norm(direction)
    appearance=dict(dominant_chromatic_print_background=True,RGB_chroma_direction=direction,
        source_material_authority_scope='current_RGB_material_candidate_not_reference_identity')
    pattern=printed_pattern_candidates(image,np.ones((80,120),bool),device='cpu',source_print_appearance=appearance,
                                       search_pattern_keypoints_uv=[[25,30]])
    contours=source_print_boundary_contours(pattern,appearance=appearance,stride=2)
    material=source_colour_material_evidence(image,binding(),pattern,appearance)
    evidence=dict(current_source_material_proposal_identity_unverified=True,
        _current_source_material_support=material['_current_source_material_support'])
    accepted,conflicts=verify_source_print_contours(contours,evidence)
    internal=[x for x in accepted if x['role']=='printed_pattern_internal_material_boundary']
    assert accepted and internal
    assert all(not x['whole_printed_pattern_hole_verified'] for x in internal)
    assert all(not x['solver_outer_boundary_eligible'] for x in internal)
    assert all(not x['source_print_role_supported_by_reference_correspondence'] for x in accepted)
    assert sum(len(x['uv']) for x in accepted+conflicts)==sum(len(x['uv']) for x in contours)


def test_unnamed_material_memory_can_refine_same_native_image_but_cannot_rebind_changed_rgb():
    image=np.full((80,120,3),240,np.uint8);image[25:60,20:100]=[40,70,160]
    memory=source_print_material_frame_evidence(image,binding())
    memory['_current_source_feature_uv_only']=np.array([[30.,35.]],np.float32)
    same=source_print_component_continuity(image,binding(),{},previous_evidence=memory,previous_rgb=image)
    assert same['status']=='same_source_frame_component_search_points_reused'
    assert not same['actual_source_boundary_UV_transported']
    following=source_print_component_continuity(image,binding(101),{},previous_evidence=memory,previous_rgb=image)
    assert following['status']=='missing_previous_linked_or_current_source_material_support'
    changed=image.copy();changed[0,0]=0
    with pytest.raises(ValueError,match='memory RGB changed'):
        source_print_component_continuity(image,binding(101),{},previous_evidence=memory,previous_rgb=changed)
    with pytest.raises(ValueError,match='current source binding'):
        source_print_material_frame_evidence(changed,{**binding(),'decoded_native_rgb_sha256':hashlib.sha256(image.tobytes()).hexdigest()})


def test_partial_material_seed_can_search_current_visible_object_beyond_initial_parent_extent():
    image=np.full((80,120,3),240,np.uint8);image[20:70,30:90]=[140,55,150]
    mask=np.zeros((80,120),bool);mask[20:70,30:90]=True
    direction=np.array([140,55,150],float);direction-=direction.mean();direction/=np.linalg.norm(direction)
    partial=[25,15,95,45]
    wider=current_material_region_search_box(dict(source_material_proposal_unverified=True),partial,[30,20,90,70])
    pattern=printed_pattern_candidates(image,mask,device='cpu',source_box=wider,
        source_print_appearance=dict(dominant_chromatic_print_background=True,RGB_chroma_direction=direction),
        search_pattern_keypoints_uv=[[50,30]])
    assert pattern['ink_mask'][60,50] and not pattern['extent_is_physical_edge']
    assert current_material_region_search_box({},partial,[30,20,90,70])==partial


def test_overlapping_materials_share_only_navigation_and_keep_independent_layers_and_no_physical_id():
    a=np.zeros((80,120),bool);a[20:60,20:75]=True
    b=np.zeros_like(a);b[20:60,35:90]=True
    c=np.zeros_like(a);c[20:60,100:119]=True
    prompts=[dict(object_id=i,source_binding=binding(),source_material_proposal_unverified=True,
        physical_id=None,expected_id=None,box_xyxy=box,points_uv=[point],point_labels=[1])
        for i,box,point in [(1,[20,20,75,60],[30,30]),(2,[35,20,90,60],[80,30]),(3,[100,20,119,60],[110,30])]]
    candidates=[dict(object_id=i,source_binding=binding(),source_material_candidate_mask=mask)
                for i,mask in [(1,a),(2,b),(3,c)]]
    grouped=group_source_material_prompt_search(prompts,candidates,binding())
    assert len(grouped['groups'])==2 and len(grouped['prompts'])==3
    assert grouped['prompts'][0]['sam_search_object_id']==grouped['prompts'][1]['sam_search_object_id']
    assert grouped['prompts'][2]['sam_search_object_id']!=grouped['prompts'][0]['sam_search_object_id']
    assert all(x['physical_id'] is None and x['independent_RGB_material_layer_preserved'] for x in grouped['prompts'])
    assert all(not x['content_or_logo_identity_used'] and not x['physical_instances_merged'] for x in grouped['groups'])
    assert 'sam_search_object_id' not in prompts[0]  # Input/failed-history source packet is untouched.
    with pytest.raises(ValueError,match='source_pts'):
        group_source_material_prompt_search(prompts,[{**candidates[0],'source_binding':binding(101)},*candidates[1:]],binding())


def test_board_search_gap_is_navigation_uncertainty_not_erased_visible_material():
    image=np.full((80,120,3),240,np.uint8);image[25:60,55:90]=[165,55,70]
    region=np.zeros((80,120),bool);region[20:65,:50]=True
    board=dict(source_binding=binding(),search_mask=region,missing_source_support_columns=list(range(50,120)))
    parent=dict(source_binding=binding(),object_id=2000,printed_pattern_extent=[10,25,40,60])
    result=source_print_material_search_proposals(image,binding(),board_search=board,print_parents=[parent])
    assert any(x['source_material_candidate_mask'][40,70] for x in result['material_candidates'])
    assert result['board_missing_or_uncertain_parent_fallback_columns']==list(range(50,120))
    assert not result['parent_fallback_is_measurement']
    assert not board['search_mask'][:,60].any()


def test_current_slanted_board_support_is_not_cropped_to_left_parent_mean_height():
    image=np.full((80,120,3),240,np.uint8)
    image[20:40,10:40]=[40,70,160];image[45:75,75:115]=[140,55,150]
    region=np.zeros((80,120),bool);region[18:42,:50]=True;region[42:78,65:]=True
    board=dict(source_binding=binding(),search_mask=region,missing_source_support_columns=list(range(50,65)))
    parent=dict(source_binding=binding(),object_id=2000,printed_pattern_extent=[10,20,40,40])
    result=source_print_material_search_proposals(image,binding(),board_search=board,print_parents=[parent])
    assert any(x['source_material_candidate_mask'][65,95] for x in result['material_candidates'])


def test_native_RGB_tensor_cache_cannot_cross_clock_frame_or_changed_bytes():
    image=np.full((80,120,3),240,np.uint8)
    cache=source_RGB_tensors(image,binding(),device='cpu')
    assert _checked_source_RGB_tensors(image,cache,binding=binding(),device='cpu') is cache
    with pytest.raises(ValueError,match='source_pts'):
        _checked_source_RGB_tensors(image,cache,binding=binding(101),device='cpu')
    changed=image.copy();changed[20,30]=0
    with pytest.raises(ValueError,match='image bytes'):
        _checked_source_RGB_tensors(changed,cache,binding=binding(),device='cpu')


def test_unindexed_CUDA_request_resolves_current_device_without_accepting_another_GPU(monkeypatch):
    import torch
    from types import SimpleNamespace
    image=np.full((80,120,3),240,np.uint8)
    cache=source_RGB_tensors(image,binding(),device='cpu')
    cache['image']=SimpleNamespace(device=torch.device('cuda:2'))
    monkeypatch.setattr(torch.cuda,'current_device',lambda:2)
    assert _checked_source_RGB_tensors(image,cache,binding=binding(),device='cuda') is cache
    with pytest.raises(ValueError,match='different device'):
        _checked_source_RGB_tensors(image,cache,binding=binding(),device='cuda:0')


def test_multiview_context_cannot_borrow_another_reference_frames_print_correspondence(monkeypatch):
    from rinkspace import camera_reuse_localization
    image=np.full((80,120,3),240,np.uint8)
    matcher=object.__new__(SourceAdvertisementMatcher);matcher.ransac_px=3.
    matcher.contexts=[dict(source_binding=binding(10),source_packet_sha256=digest,
                          points=np.empty((0,2),np.float32),descriptors=None)
                      for digest in ('1'*64,'2'*64)]
    matched_contexts=[]
    def no_static_match(reference,current,config):
        matched_contexts.append(reference)
        return None
    monkeypatch.setattr(camera_reuse_localization,'match_features',no_static_match)
    rows=[dict(appearance_match=True,source_packet_sha256='1'*64,reference_physical_id='Ad2',
               actual_source_keypoints_uv=[[60.,60.],[90.,60.]])]
    result=matcher._context_matches(image,binding(),rows)
    assert len(matched_contexts)==1
    assert result[1]['status']=='no_print_correspondence_to_this_reference_source'
    assert not result[1]['physical_wing_identity_verified']


def test_shared_source_search_preserves_exact_RGB_materials_not_palette_decoders():
    import torch
    image=np.full((80,120,3),240,np.uint8)
    image[25:65,20:50]=[40,70,160];image[25:65,70:110]=[165,55,70]
    region=np.zeros((80,120),bool);region[18:70]=True
    direction=np.array([40,70,160],float);direction-=direction.mean();direction/=np.linalg.norm(direction)
    appearance=dict(dominant_chromatic_print_background=True,RGB_chroma_direction=direction)
    ordinary=printed_pattern_candidates(image,region,device='cpu',source_print_appearance=appearance,
                                       search_pattern_keypoints_uv=[[30,40]])
    cache=source_RGB_tensors(image,binding(),device='cpu')
    support=torch.nn.functional.max_pool2d(torch.as_tensor(region).float()[None,None],25,1,12)[0,0]>.5
    shared=printed_pattern_candidates(image,region,device='cpu',source_print_appearance=appearance,
        search_pattern_keypoints_uv=[[30,40]],source_rgb_cache=cache,region_search_support=support)
    for key in ('ink_mask','search_support','source_chromatic_material_candidate_mask','unlinked_source_foreground_candidate_mask'):
        assert torch.equal(ordinary[key],shared[key])
    assert ordinary['extent']==shared['extent'] and ordinary['background_luminance']==shared['background_luminance']


def test_shared_board_layers_do_not_advance_same_exposure_or_inherit_physical_id(monkeypatch):
    import torch
    image=np.full((80,120,3),240,np.uint8);image[25:65,55:105]=[165,55,70]
    nav=dict(source_binding=binding(),search_mask=np.ones((80,120),bool),RGB_hard_veto_allowed=False)
    proposals=source_print_material_search_proposals(image,binding(),board_search=nav,first_object_id=20)
    board=dict(source_binding=binding(),object_id=10,kind='board_region',physical_id=None)
    shared=material_prompts_with_shared_board_search(image,binding(),proposals['prompts'],board_search=nav,board_instance=board)
    assert not shared['new_SAM_board_object_count'] and shared['material_layer_count']
    assert all(p['sam_search_object_id']==10 and p['physical_id'] is None for p in shared['prompts'])
    # A CPU-only same-exposure provider uses the already prepared region. A
    # model forward would fail this test, so it exercises the actual reuse path.
    provider=object.__new__(Sam2SourceInstances)
    provider.device=torch.device('cpu');provider.contour_stride=4;provider.search_radius_px=12.
    provider.printed_contour_search_radius_px=3.;provider.reread_rgb=False
    provider.objects={10:validate_prompt({**board,'origin':'original_rgb_candidate','box_xyxy':[0,0,120,80]},binding())}
    provider.previous_binding=binding();provider.previous_rgb_sha256=hashlib.sha256(image.tobytes()).hexdigest()
    provider.previous_masks={10:torch.ones((80,120),dtype=torch.bool)};provider.previous_rgb=image.copy()
    provider.frame_index=7;provider.print_layout_memory={};provider.current_RGB_tensor_cache=None;provider.model_sha256='c'*64
    result=provider.read(image,binding(),prompts=shared['prompts'],material_search_navigation=nav)
    assert provider.frame_index==7 and not result['SAM_forward_executed_this_call']
    assert result['memory_update']=='source_frame_prompt_refinement'
    layers=[r for r in result['instances'] if r['shared_SAM_board_search_only']]
    assert layers and all(r['physical_id'] is None for r in layers)
    assert any(r['printed_pattern_contours'] for r in layers)
    waiting=provider.read(image,binding())
    assert all(r['status']=='awaiting_current_source_material_navigation' and not r['boundary_readings']
               for r in waiting['instances'] if r['shared_SAM_board_search_only'])
    fallback=material_prompts_with_shared_board_search(image,binding(),proposals['prompts'],board_search=nav,board_object_id=11)
    assert fallback['new_SAM_board_object_count']==1
    assert sum(p['kind']=='board_region' for p in fallback['prompts'])==1
    with pytest.raises(ValueError,match='source_pts'):
        material_prompts_with_shared_board_search(image,binding(),proposals['prompts'],board_search={**nav,'source_binding':binding(101)})
    with pytest.raises(ValueError,match='named physical'):
        material_prompts_with_shared_board_search(image,binding(),[{**proposals['prompts'][0],'physical_id':'Ad21'}],board_search=nav)
    # Advance a native exposure before current rail/foot navigation is ready.
    # SAM is mocked here; actual RGB/material/memory operations are unchanged.
    # The deferred layer still owns its first-frame snapshot, while the global
    # previous RGB is now the second frame. Same-PTS refinement must pair each
    # layer with its own saved evidence rather than that global second image.
    from types import SimpleNamespace
    from contextlib import nullcontext
    monkeypatch.setattr(torch,'autocast',lambda *args,**kwargs:nullcontext())
    provider.session=SimpleNamespace(obj_with_new_inputs=[])
    provider.processor=SimpleNamespace(post_process_masks=lambda *args,**kwargs:[torch.ones((1,1,80,120),dtype=torch.bool)])
    class Processor:
        def __call__(self,*args,**kwargs):return {'pixel_values':torch.zeros((1,3,80,120))}
        post_process_masks=provider.processor.post_process_masks
    provider.processor=Processor()
    provider.model=lambda **kwargs:SimpleNamespace(pred_masks=None,object_ids=[10])
    provider._prune_architecture_memory=lambda:None
    next_image=image.copy();next_image[0,0]=239
    deferred=provider.read(next_image,binding(101))
    assert provider.frame_index==8 and deferred['SAM_forward_executed_this_call']
    layer_id=layers[0]['object_id']
    assert provider.print_layout_memory[layer_id]['source_binding']['source_pts']==100
    assert np.array_equal(provider.print_layout_RGB_memory[layer_id],image)
    refined=provider.read(next_image,binding(101),material_search_navigation={**nav,'source_binding':binding(101)})
    assert provider.frame_index==8 and not refined['SAM_forward_executed_this_call']
    assert provider.print_layout_memory[layer_id]['source_binding']['source_pts']==101
    assert np.array_equal(provider.print_layout_RGB_memory[layer_id],next_image)
    # A corrupted per-object snapshot must still fail the original byte check.
    provider.print_layout_RGB_memory[layer_id][0,0]=0
    with pytest.raises(ValueError,match='memory RGB changed'):
        provider.read(next_image,binding(101),material_search_navigation={**nav,'source_binding':binding(101)})


def test_bright_back_wall_does_not_outvote_a_rail_containing_the_actual_print_parent():
    image = np.full((80, 120, 3), 100, np.uint8)
    image[12:32] = 240
    image[32:50] = 170
    image[50:] = 250
    image[38:44, 20:100] = 60
    probabilities = np.zeros((3, 80, 120), np.float32)
    probabilities[0, 49:52] = .9
    probabilities[1, 10:14] = .8
    probabilities[1, 30:34] = .65
    probabilities[2, 36:46, 18:102] = .9
    proposal = propose_source_region_prompts(image, binding(), probabilities,
                 ["ice_contact", "board_top", "ad_boundary"], probabilities_conditioned_on_geometry=False)
    tops = [row for row in proposal["boundary_topology_hypotheses"] if row["kind"] == "board_top"]
    assert len(tops) == 2
    preferred = next(row for row in tops if row.get("preferred_region_search_hypothesis"))
    assert 30 <= preferred["mean_source_y"] <= 34
    assert preferred["source_gradient_pixel_runs_y_x_length"]
    assert not preferred["closed_grouping_pixels_used_as_source_evidence"]
    assert not proposal["same_board_structure_verified"]
    assert all(not pair["whiteness_or_pale_wall_used_for_ranking"] for pair in proposal["board_contact_pair_hypotheses"])


def test_navigation_box_cannot_replace_object_support_with_unrelated_foreground():
    image = np.full((80, 120, 3), 250, np.uint8)
    image[25:45, 20:60] = [30, 70, 150]
    image[15:65, 95:115] = [120, 60, 50]
    region = np.zeros((80, 120), bool)
    region[25:45, 20:60] = True
    result = printed_pattern_candidates(image, region, source_box=[0, 0, 120, 80], device="cpu")
    assert result["ink_mask"][30, 30]
    assert not result["ink_mask"][:, 95:115].any()
    assert not result["source_navigation_box_replaces_object_support"]


def test_same_blue_rail_and_skin_are_not_the_matched_blue_print_component():
    image = np.full((80, 120, 3), 250, np.uint8)
    image[30:55, 25:80] = [30, 70, 150]
    image[38:47, 43:55] = 250  # Separate internal hole in the actual source print.
    image[10:13, 10:100] = [30, 70, 150]
    image[24:50, 90:110] = [120, 60, 50]
    colour = np.array([30, 70, 150], float)
    direction = colour-colour.mean()
    direction /= np.linalg.norm(direction)
    result = printed_pattern_candidates(image, np.ones((80, 120), bool), source_box=[0, 0, 120, 80],
        source_print_appearance=dict(dominant_chromatic_print_background=True, RGB_chroma_direction=direction),
        search_pattern_keypoints_uv=[[35, 35]], device="cpu")
    assert result["ink_mask"][35, 35]
    assert not result["ink_mask"][10:13].any() and not result["ink_mask"][:, 90:110].any()
    assert result["unlinked_source_foreground_candidate_mask"][11, 30]
    assert result["unlinked_source_foreground_candidate_mask"][30, 95]
    assert not result["extent_is_physical_edge"] and not result["pattern_keypoints_used_as_boundary_measurements"]
    contours = region_contours(result["ink_mask"], role_prefix="printed_pattern")
    assert {row["role"] for row in contours} == {"printed_pattern_external_contour", "printed_pattern_internal_hole"}


def test_dark_print_and_white_glyph_hole_link_visible_fragments_without_linking_blue_rail():
    image=np.full((80,120,3),235,np.uint8)
    image[:,:57]=[172,170,170]  # Darker neutral board with a quantized warm tint.
    image[25:60,15:105]=[65,70,83]  # Weakly chromatic but coherent original print.
    image[35:50,30:46]=235
    image[25:60,57:67]=[165,110,90]  # Foreground separates two visible printed components.
    image[8:11,10:110]=[65,70,83]
    direction=np.array([65,70,83],float);direction-=direction.mean();direction/=np.linalg.norm(direction)
    pattern=printed_pattern_candidates(image,np.ones((80,120),bool),device='cpu',
        source_print_appearance=dict(dominant_chromatic_print_background=True,RGB_chroma_direction=direction),
        search_pattern_keypoints_uv=[[38,42],[80,35]])
    assert pattern['ink_mask'][30,25] and pattern['ink_mask'][30,80]
    assert not pattern['ink_mask'][8:11].any() and not pattern['ink_mask'][:,57:67].any()
    assert pattern['source_pattern_material_component_links'][0]['enclosing_source_hole_parent_labels']
    assert not pattern['pattern_keypoints_used_as_boundary_measurements']
    # A poor/missing template-edge H cannot veto a current print-to-board RGB
    # transition; the same source body interruption stays a role conflict.
    contours=region_contours(pattern['ink_mask'],stride=1,role_prefix='printed_pattern')
    evidence=source_colour_material_evidence(image,binding(),pattern,
        dict(RGB_chroma_direction=direction))
    visible,conflict=verify_source_print_contours(contours,evidence)
    assert visible and conflict
    uv=np.concatenate([r['uv'] for r in visible])
    assert (uv[:,0]<45).any() and (uv[:,0]>75).any()
    assert ((uv[:,0]==15)&(uv[:,1]>30)&(uv[:,1]<55)).any()
    # The visible bottom edge next to an occlusion corner is still print;
    # only the interior blue-to-foreground cut is the conflicting role.
    assert not ((uv[:,0]>=56)&(uv[:,0]<=68)&(uv[:,1]>30)&(uv[:,1]<55)).any()
    assert sum(len(r['uv']) for r in contours)==sum(len(r['uv']) for r in visible+conflict)


@pytest.fixture
def cpu_native_profile(monkeypatch):
    # Exercise the native sampling arithmetic on CPU. Production retains its
    # CUDA guard; the real source/SAM probe separately covers that GPU path.
    import torch
    import rinkspace.vision_geometry as geometry
    def tensor(value, *, dimensions=None):
        assert isinstance(value,torch.Tensor)
        if dimensions is not None:assert value.ndim==dimensions
        return value
    monkeypatch.setattr(geometry,'_cuda_tensor',tensor)


def test_white_glyph_in_coloured_print_is_material_boundary_not_whole_pattern_hole():
    mask=np.zeros((80,120),bool);mask[20:60,15:105]=True;mask[35:48,40:60]=False
    pattern=dict(ink_mask=mask)
    generic=region_contours(mask,stride=1,role_prefix='printed_pattern')
    layered=source_print_boundary_contours(pattern,stride=1,
        appearance=dict(dominant_chromatic_print_background=True))
    assert len(generic)==len(layered)
    for old,new in zip(generic,layered):assert np.array_equal(old['uv'],new['uv'])
    inner=next(r for r in layered if r['internal_hole'])
    assert inner['role']=='printed_pattern_internal_material_boundary'
    assert inner['selected_material_hole'] and not inner['whole_printed_pattern_hole_verified']
    assert not inner['solver_outer_boundary_eligible']
    outer=next(r for r in layered if not r['internal_hole'])
    assert outer['role']=='printed_pattern_external_contour' and outer['solver_outer_boundary_eligible']
    ink=source_print_boundary_contours(pattern,stride=1,appearance={})
    assert next(r for r in ink if r['internal_hole'])['role']=='printed_pattern_internal_hole'


def test_actual_material_tracks_keep_visible_component_after_its_glyph_is_occluded(monkeypatch):
    import cv2
    previous=np.full((120,260,3),235,np.uint8)
    previous[30:100,15:240]=[50,65,110]
    cv2.putText(previous,'ISU',(35,80),cv2.FONT_HERSHEY_SIMPLEX,.8,(235,235,235),2,cv2.LINE_AA)
    cv2.putText(previous,'TEST',(165,80),cv2.FONT_HERSHEY_SIMPLEX,.5,(235,235,235),1,cv2.LINE_AA)
    previous[20:110,125:145]=[170,120,90]
    direction=np.array([50,65,110],float);direction-=direction.mean();direction/=np.linalg.norm(direction)
    appearance=dict(dominant_chromatic_print_background=True,RGB_chroma_direction=direction)
    old={**binding(100),'native_hw':[120,260]};current={**old,'source_pts':101}
    initial=printed_pattern_candidates(previous,np.ones((120,260),bool),device='cpu',
        source_print_appearance=appearance,search_pattern_keypoints_uv=[[20,40],[230,40]])
    following=cv2.warpAffine(previous,np.array([[1,0,3],[0,1,1]],float),(260,120),borderValue=(235,235,235))
    # A wider foreground removes every left glyph while leaving a blue strip.
    following[20:111,48:145]=[170,120,90]
    person=np.zeros((120,260),bool);person[20:111,48:145]=True
    one_sided=printed_pattern_candidates(following,np.ones((120,260),bool),device='cpu',
        source_print_appearance=appearance,search_pattern_keypoints_uv=[[233,41]])
    assert not one_sided['ink_mask'][50,25] and one_sided['source_chromatic_material_candidate_mask'][50,25]
    memory=dict(source_binding=old,source_RGB_sha256=hashlib.sha256(previous.tobytes()).hexdigest(),
        _source_component_material_mask=initial['ink_mask'].numpy(),_source_person_search_mask=np.zeros((120,260),bool))
    tracks=source_print_component_continuity(following,current,one_sided,
        previous_evidence=memory,previous_rgb=previous,source_person_search_mask=person)
    assert tracks['status']=='actual_linked_source_material_component_tracks'
    assert (tracks['current_source_feature_uv'][:,0]<48).any()
    retained=printed_pattern_candidates(following,np.ones((120,260),bool),device='cpu',
        source_print_appearance=appearance,search_pattern_keypoints_uv=np.r_[[[233,41]],tracks['current_source_feature_uv']])
    assert retained['ink_mask'][50,25] and retained['ink_mask'][50,230]
    assert not retained['ink_mask'][person].any()
    assert not tracks['actual_source_boundary_UV_transported'] and not tracks['physical_instance_identity_reassigned']
    assert not tracks['source_component_link_is_pixel_ground_truth']
    # Same native prompt refinement reuses measured feature navigation; no
    # invented temporal step or H boundary coordinates are introduced.
    current_memory=dict(source_binding=current,source_RGB_sha256=hashlib.sha256(following.tobytes()).hexdigest(),
        _current_source_feature_uv_only=tracks['current_source_feature_uv'],source_material_instance_continuity=tracks,
        _current_RGB_component_overlap_search_points_uv=tracks['current_RGB_component_overlap_search_points_uv'])
    same=source_print_component_continuity(following,current,retained,previous_evidence=current_memory,previous_rgb=following)
    assert same['status']=='same_source_frame_component_search_points_reused'
    assert np.array_equal(same['current_source_feature_uv'],tracks['current_source_feature_uv'])
    assert np.array_equal(same['current_RGB_component_overlap_search_points_uv'],tracks['current_RGB_component_overlap_search_points_uv'])
    # Low-texture visible fragments need not retain any corner of their own.
    # Independently matched right-side features only navigate the previous
    # linked support; a current RGB component supplies a search pixel, not H UV.
    import rinkspace.camera_context_reuse as reuse
    feature_mask=np.zeros((120,260),np.uint8);feature_mask[35:95,165:235]=255
    points=cv2.goodFeaturesToTrack(cv2.cvtColor(previous,cv2.COLOR_RGB2GRAY),50,.003,5,mask=feature_mask,blockSize=5).reshape(-1,2)
    assert len(points)>=6 and (points[:,0]>160).all()
    def right_only(grays,valid_masks,start,end,*,min_points):
        following_points=points+np.array([3,1],np.float32)
        for p,q in zip(points,following_points):
            assert valid_masks[0][round(p[1]),round(p[0])]
            assert valid_masks[1][round(q[1]),round(q[0])]
        return points,following_points,dict(reason='synthetic_actual_right_features_only')
    monkeypatch.setattr(reuse,'temporal_correspondences',right_only)
    no_left_features=source_print_component_continuity(following,current,one_sided,
        previous_evidence=memory,previous_rgb=previous,source_person_search_mask=person)
    assert (no_left_features['current_source_feature_uv'][:,0]>160).all()
    search=no_left_features['current_RGB_component_overlap_search_points_uv']
    assert (search[:,0]<48).any() and no_left_features['prior_material_support_projected_for_navigation_only']
    linked=printed_pattern_candidates(following,np.ones((120,260),bool),device='cpu',
        source_print_appearance=appearance,search_pattern_keypoints_uv=np.r_[[[233,41]],search])
    assert linked['ink_mask'][50,25]
    assert no_left_features['overlap_search_points_are_current_material_pixels_not_boundary_UV']
    with pytest.raises(ValueError,match='source or shot'):
        source_print_component_continuity(following,{**current,'shot_id':'another_shot'},one_sided,
            previous_evidence=memory,previous_rgb=previous)
    changed=previous.copy();changed[0,0]=0
    with pytest.raises(ValueError,match='memory RGB changed'):
        source_print_component_continuity(following,current,one_sided,previous_evidence=memory,previous_rgb=changed)


def test_single_wide_RGB_print_transition_retains_measured_interval_without_hiding_blur(cpu_native_profile):
    import cv2
    image=np.full((80,120,3),225,np.uint8)
    image[:,30:95]=[65,70,100]
    image=cv2.GaussianBlur(image,(0,0),2.5)
    points=np.c_[np.full(31,30),np.arange(20,51)]
    contour=dict(uv=points,closed=False,internal_hole=False,role='printed_pattern_external_contour',
        current_source_material_boundary_evidence=dict(supported=np.ones(len(points),bool)))
    reading=reread_region_edges(image,[contour],binding(),device='cpu')[0]
    assert len(reading['actual_source_uv'])==len(points)
    assert np.allclose(reading['actual_source_uv'].numpy()[:,0],29.5,atol=1.)
    assert reading['flags']['blur_review'].all()
    assert reading['flags']['wide_source_material_RGB_interval_retained'].all()
    intervals=reading['observed_RGB_transition_intervals']
    assert (intervals['normal_interval_native_px'][:,1]-intervals['normal_interval_native_px'][:,0]>4).all()
    assert not intervals['interval_is_exact_physical_boundary']
    assert not reading['physical_boundary_verified']
    assert reading['observed_RGB_signed_peak_pool']['all_observed_local_signed_source_peaks_retained']
    # A search contour without the actual material evidence still retains its
    # source peaks/uncertainty, but does not acquire a measured wide role.
    unverified={k:v for k,v in contour.items() if k!='current_source_material_boundary_evidence'}
    raw=reread_region_edges(image,[unverified],binding(),device='cpu')[0]
    assert len(raw['actual_source_uv'])==0
    assert len(raw['observed_RGB_signed_peak_pool']['actual_source_peak_uv'])>0


@pytest.mark.parametrize('intermediate,outer',[(80,155),(130,80)])
def test_distinct_same_polarity_RGB_transitions_remain_competing(cpu_native_profile,intermediate,outer):
    import cv2
    image=np.full((80,120,3),60,np.uint8)
    image[:,:27]=180
    image[:,27:32]=intermediate
    image[:,32:37]=outer
    image=cv2.GaussianBlur(image,(0,0),.6)
    # A linked print parent alone cannot choose between two separated actual
    # outward-lightening peaks along this same source normal.
    points=np.c_[np.full(31,32),np.arange(20,51)]
    contour=dict(uv=points,closed=False,internal_hole=False,role='printed_pattern_external_contour',
        current_source_material_boundary_evidence=dict(supported=np.ones(len(points),bool)))
    reading=reread_region_edges(image,[contour],binding(),device='cpu')[0]
    assert reading['flags']['multi_edge_ambiguity'].all()
    assert len(reading['actual_source_uv'])==0
    assert len(reading['observed_RGB_signed_peak_pool']['actual_source_peak_uv'])>=2*len(points)


def test_actual_glyph_correspondence_keeps_visible_print_and_flags_connected_black_foreground(tmp_path):
    import cv2
    from calibration.program_tracking.runtime.bootstrap import add_geometry_dependencies
    add_geometry_dependencies()
    from rinkspace.camera_reuse_localization import descriptors
    original = np.full((180, 480, 3), 245, np.uint8)
    cv2.putText(original, "WYNDHAM ANKARA", (12, 105), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (35, 35, 35), 2, cv2.LINE_AA)
    path = tmp_path/"native_reference.png"
    assert cv2.imwrite(str(path), original[:, :, ::-1])
    points, features = descriptors(original[:, :, ::-1], np.full(original.shape[:2], 255, np.uint8))
    ref_binding = {**binding(), "native_hw": [180, 480]}
    reference = dict(source_binding=ref_binding, source_frame_path=str(path),
        source_frame_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        source_rgb_sha256=hashlib.sha256(original.tobytes()).hexdigest(), navigation_roi=[0, 60, 440, 125],
        feature_native_uv=points, descriptors=features)
    image_H = np.array([[1.2, 0, 20], [0, 1.2, 10], [0, 0, 1]], float)
    query = cv2.warpPerspective(original, image_H, (620, 220), borderValue=(245, 245, 245))
    cv2.ellipse(query, (250, 119), (39, 39), 0, 0, 360, (22, 22, 22), -1)
    current = {**binding(200), "native_hw": [220, 620]}
    evidence = source_print_layout_evidence(query, current, reference)
    assert evidence["status"] == "actual_query_glyph_correspondence_search_atlas"
    assert not evidence["atlas_pixels_are_output_measurements"]
    source_ink = printed_pattern_candidates(query, np.ones(query.shape[:2], bool), device="cpu")
    raw = region_contours(source_ink["ink_mask"], stride=2, role_prefix="printed_pattern")
    visible, conflicts = verify_source_print_contours(raw, evidence)
    assert visible and conflicts
    accepted = np.concatenate([row["uv"] for row in visible])
    rejected = np.concatenate([row["uv"] for row in conflicts])
    assert ((accepted[:, 0] < 180) | (accepted[:, 0] > 320)).any()  # Actual unobscured glyphs remain.
    assert ((rejected[:, 0] > 220) & (rejected[:, 0] < 280) & (rejected[:, 1] < 90)).any()
    assert not ((accepted[:, 0] > 220) & (accepted[:, 0] < 280) & (accepted[:, 1] < 90)).any()
    assert all(row["role"] == "printed_pattern_source_role_conflict" for row in conflicts)
    assert all(not row["source_contour_is_output_measurement"] for row in visible+conflicts)


def test_fresh_person_box_refreshes_missing_leg_search_without_excluding_leg_gap():
    current = binding(101)
    person = dict(source_binding=current, box_xyxy=[30, 10, 90, 75],
                  provenance=dict(kind="local_pretrained_person_box_prediction", model_sha256="c"*64))
    torso = np.zeros((80, 120), bool)
    torso[12:40, 40:80] = True
    objects = {1000: dict(kind="occluder", box_xyxy=[30, 10, 90, 75]), 2000: prompt(2000, "Ad20")}
    regions = [dict(object_id=1000, kind="occluder", region_mask=torso)]
    refresh = source_person_refresh_prompts(current, [person], regions, objects)
    assert len(refresh['prompts']) == 1 and refresh['prompts'][0]['object_id'] == 1000
    assert refresh['prompts'][0]['source_binding']['source_pts'] == 101
    assert not refresh['ledger'][0]['detector_box_used_as_RGB_veto']
    assert 'source_exclusion_mask' not in refresh and 'actual_source_uv' not in refresh['prompts'][0]
    assert not refresh['prompts'][0].get('exclusion_authority')
    wrong = {**person, 'source_binding':binding(100)}
    with pytest.raises(ValueError, match='source_pts'):
        source_person_refresh_prompts(current, [wrong], regions, objects)


def test_person_search_refinement_keeps_same_sam_frame_and_ad_memory(monkeypatch):
    reader = Sam2SourceInstances.__new__(Sam2SourceInstances)
    reader.previous_binding, reader.frame_index = binding(100), 8
    reader.objects = {1000:dict(kind='occluder',box_xyxy=[30,10,90,75]), 2000:prompt(2000,'Ad20')}
    mask = np.zeros((80,120), bool);mask[10:38,40:80] = True
    calls=[]
    def read(rgb, source, *, prompts=(), reset=False):
        calls.append((source.copy(),list(prompts),reset))
        if source['source_pts'] != reader.previous_binding['source_pts']:
            reader.frame_index += 1
        reader.previous_binding=source
        return dict(source_binding=source,instances=[dict(object_id=1000,kind='occluder',region_mask=mask)])
    monkeypatch.setattr(reader,'read',read)
    current=binding(101)
    result=reader.read_auto(np.zeros((80,120,3),np.uint8),current,
        person_boxes=[dict(source_binding=current,box_xyxy=[30,10,90,75])])
    assert reader.frame_index==9 and len(calls)==2
    assert not calls[1][2] and calls[1][1][0]['source_binding']['source_pts']==101
    assert reader.objects[2000]['physical_id']=='Ad20'
    assert result['source_person_reidentification']['source_boxes_are_search_only']


def test_colour_print_layout_separates_foreground_cutout_from_print_hole(tmp_path):
    import cv2
    from calibration.program_tracking.runtime.bootstrap import add_geometry_dependencies
    add_geometry_dependencies()
    from rinkspace.camera_reuse_localization import descriptors
    original=np.full((200,520,3),245,np.uint8)
    cv2.rectangle(original,(20,65),(495,155),(40,75,150),-1)
    cv2.putText(original,'ISU ANKARA',(35,125),cv2.FONT_HERSHEY_SIMPLEX,1.35,(245,245,245),3,cv2.LINE_AA)
    path=tmp_path/'native_colour_reference.png';assert cv2.imwrite(str(path),original[:,:,::-1])
    points,features=descriptors(original[:,:,::-1],np.full(original.shape[:2],255,np.uint8))
    reference=dict(source_binding={**binding(),'native_hw':[200,520]},source_frame_path=str(path),
        source_frame_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        source_rgb_sha256=hashlib.sha256(original.tobytes()).hexdigest(),navigation_roi=[10,55,505,165],
        feature_native_uv=points,descriptors=features,observed_kind='printed_colour_background')
    query=original.copy();cv2.ellipse(query,(265,92),(30,65),0,0,360,(170,125,90),-1)
    current={**binding(101),'native_hw':[200,520]}
    evidence=source_print_layout_evidence(query,current,reference)
    assert evidence['source_material_pair_evidence']['source_material_sides_reexamined']
    colour=np.array([40,75,150],float);direction=colour-colour.mean();direction/=np.linalg.norm(direction)
    ink=printed_pattern_candidates(query,np.ones(query.shape[:2],bool),source_box=[10,55,505,165],
        source_print_appearance=dict(dominant_chromatic_print_background=True,RGB_chroma_direction=direction),
        search_pattern_keypoints_uv=[[40,75]],device='cpu')
    raw=region_contours(ink['ink_mask'],stride=2,role_prefix='printed_pattern')
    current_material=source_colour_material_evidence(query,current,ink,dict(RGB_chroma_direction=direction))
    evidence['_current_source_material_support']=current_material['_current_source_material_support']
    visible,conflicts=verify_source_print_contours(raw,evidence)
    assert visible and conflicts
    rejected=np.concatenate([r['uv'] for r in conflicts])
    accepted=np.concatenate([r['uv'] for r in visible])
    assert ((rejected[:,0]>230)&(rejected[:,0]<300)&(rejected[:,1]<150)).any()
    assert not ((accepted[:,0]>235)&(accepted[:,0]<295)&(accepted[:,1]<150)).any()
    assert ((accepted[:,0]<200)&(accepted[:,1]<80)).any()  # True printed outer edge remains.
    # Every original navigation contour sample survives in one role ledger;
    # none of these atlas/search pixels is promoted to an output measurement.
    assert sum(len(r['uv']) for r in raw)==sum(len(r['uv']) for r in visible+conflicts)
    assert all(not r['source_contour_is_output_measurement'] for r in visible+conflicts)
    # Losing a new descriptor match must not discard validated same-shot
    # source correspondence. This second actual RGB has a measurable shift;
    # no model/camera UV or old boundary coordinates may become its output.
    moved=cv2.warpAffine(query,np.array([[1,0,3],[0,1,1]],float),(520,200),borderValue=(245,245,245))
    following={**current,'source_pts':102}
    temporal=source_print_layout_evidence(moved,following,reference,
        query_features=(np.empty((0,2),np.float32),None),previous_evidence=evidence,previous_rgb=query)
    assert temporal['status']=='source_temporal_RGB_feature_correspondence_search_atlas'
    assert temporal['source_temporal_correspondence']['supported_actual_feature_count']>=6
    assert not temporal['source_temporal_correspondence']['source_boundary_UV_transported']
    assert temporal['source_binding']['source_pts']==102 and temporal['H_is_search_only']
    changed=query.copy();changed[0,0]=0
    with pytest.raises(ValueError,match='memory RGB changed'):
        source_print_layout_evidence(moved,following,reference,
            query_features=(np.empty((0,2),np.float32),None),previous_evidence=evidence,previous_rgb=changed)


@pytest.fixture
def saved_search_camera(tmp_path, monkeypatch):
    import rinkspace.vision_instances as instances
    import rinkspace.vision as vision
    scene_path = tmp_path/"scene.json"
    scene_path.write_text("{}", encoding="utf-8")
    scene_sha = hashlib.sha256(scene_path.read_bytes()).hexdigest()
    roles = tmp_path/"roles.json"
    roles.write_text(json.dumps(dict(scene_sha256=scene_sha, curve_records=[])), encoding="utf-8")
    camera = np.r_[np.zeros(6), np.log([100., 100.]), [60., 40., 0.], np.zeros(14)]
    camera[2] = 1.
    camera_path = tmp_path/"camera.npz"
    np.savez(camera_path, pts=np.array([100, 101]), camera25=np.stack([camera, camera]),
             exposure_representative_pts=np.array([100, 100]), installation_id=np.array(["actual_a", "actual_a"]),
             metadata_json=json.dumps(dict(source_sha256="a"*64, source_time_base="1/15360")))
    monkeypatch.setattr(instances, "named_scene_search_paths", lambda scene: [dict(
        id="Ad23:source_print", feature_id="Ad23", advertisement_number=23, kind="ad_boundary",
        primitive="line", physical_role="printed_pattern_external_contour",
        world_xyz=np.array([[0., 0., 10.], [1., 0., 10.], [2., 0., 10.]]))])
    proof = dict(source_sha256="a"*64, proof_sha256="f"*64)
    groups = {stamp: dict(canonical=100, official_pts=10, official_time_base="1/12800") for stamp in (100, 101)}
    monkeypatch.setattr(vision, "confirmed_exposure_groups", lambda *args, **kwargs: (groups, proof))
    return SceneGuideSearchAdapter(scene_path, camera_path, role_contract_path=roles, event_id="event_a"), roles


def test_frozen_search_camera_uses_exact_bound_pts_and_keeps_projection_as_search(saved_search_camera):
    adapter, _ = saved_search_camera
    exact = adapter.guides_for(binding(101))
    assert exact["status"] == "exact_saved_camera_search_guides"
    assert exact["source_binding"]["canonical_exposure_pts"] == 100
    assert exact["source_binding"]["official_exposure_time_base"] == "1/12800"
    guide = exact["guides"][0]
    assert not guide["identity_confirmed"] and not guide["source_pixel_to_expected_identity_verified"]
    assert not guide["guide_pixels_are_source_measurements"]
    assert "actual_source_uv" not in guide and "actual_boundary_uv" not in guide
    assert adapter.guides_for(binding(102))["status"] == "missing_exact_camera_PTS"
    assert adapter.guides_for({**binding(), "source_sha256": "b"*64})["status"] == "missing_camera_source_binding"
    with pytest.raises(ValueError, match="exposure alias"):
        adapter.guides_for({**binding(101), "canonical_exposure_pts": 101})


def test_native_camera_needs_no_other_source_exposure_proof(saved_search_camera, monkeypatch):
    import rinkspace.vision as vision
    from rinkspace.vision_instances import SceneGuideSearchAdapter
    adapter, roles = saved_search_camera
    with np.load(adapter.camera_path, allow_pickle=False) as archive:
        values = {name: archive[name] for name in archive.files}
    values["exposure_representative_pts"] = values["pts"].copy()
    native_camera_path = adapter.camera_path.with_name("native_camera.npz")
    np.savez(native_camera_path, **values)
    calls = []
    def native(source_sha256, time_base, *, registry=None):
        calls.append((source_sha256, time_base, registry))
        return {}, dict(source_sha256=source_sha256, time_base=time_base, proof_sha256=None,
                        status="native_frames_without_verified_exposure_aliases")
    monkeypatch.setattr(vision, "confirmed_exposure_groups", native)
    native_adapter = SceneGuideSearchAdapter(adapter.scene_path, native_camera_path,
                                             role_contract_path=roles, event_id="other_event")
    actual = native_adapter.guides_for(binding(101))
    assert calls == [("a"*64, "1/15360", None)]
    assert actual["source_binding"]["canonical_exposure_pts"] == 101
    assert "official_exposure_pts" not in actual["source_binding"]
    assert actual["source_clock_evidence"]["proof_sha256"] is None


def test_saved_camera_non_native_alias_requires_bound_registry_proof(saved_search_camera, monkeypatch):
    import rinkspace.vision as vision
    from rinkspace.vision_instances import SceneGuideSearchAdapter
    adapter, roles = saved_search_camera
    monkeypatch.setattr(vision, "confirmed_exposure_groups", lambda source_sha256, time_base, **kwargs:
                        ({}, dict(source_sha256=source_sha256, time_base=time_base, proof_sha256=None)))
    with pytest.raises(ValueError, match="exposure alias"):
        SceneGuideSearchAdapter(adapter.scene_path, adapter.camera_path,
                                role_contract_path=roles, event_id="event_a")
