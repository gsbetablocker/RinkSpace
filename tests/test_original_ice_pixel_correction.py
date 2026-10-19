import hashlib
import importlib.util
import json
from pathlib import Path
import pickle
from types import SimpleNamespace

import numpy as np
import pytest


MODULE = Path(__file__).parents[1] / 'calibration/solver_runtime/legacy/structured_programme_source_candidate/original_ice_pixel_correction.py'
spec = importlib.util.spec_from_file_location('ice_correction_under_test', MODULE)
correction = importlib.util.module_from_spec(spec)
spec.loader.exec_module(correction)


def fixture(tmp_path, collection='FIT'):
    pts = 230365184
    row = dict(id='F2_inner:lower', pts=pts, split='fit' if collection == 'FIT' else 'held',
        origin='immutable_original_source', uv=np.array([[315., 896.], [340., 889.], [550., 879.]]),
        sigma=np.full(3, 5.), normal=np.tile([0., 1.], (3, 1)),
        normal_interval=np.tile([-15., 15.], (3, 1)), world=np.array([[0., 0., 0.], [1., 0., 0.]]))
    authority = tmp_path / 'original.pkl'
    authority.write_bytes(pickle.dumps(dict(programme_absolute_rows=[dict(block='low', group_actual_pts=pts,
        group_order=0, row_order=0, collection_role=collection, source_row=row)])))
    evidence = tmp_path / 'source.png'; evidence.write_bytes(b'original RGB evidence')
    packet = dict(schema=correction.PACKET_SCHEMA, source_video_sha256=correction.SOURCE_SHA,
        time_base='1/15360', model_projection_used_for_selection=False,
        original_absolute_packet=dict(path=str(authority), sha256=correction._sha(authority)),
        source_evidence=[dict(path=str(evidence), sha256=correction._sha(evidence), pts=pts)],
        records=[dict(original_packet_row_index=0, point_indices=[0, 1],
            corrected_uv=[[315., 859.], [340., 867.]], corrected_normal_interval_px=[[-3., 3.], [-4., 4.]],
            correction_reason='gray_disc_read_using_wrong_colour_signal',
            physical_edge_role='inner_gray_disc_to_white_hole', source_only_reading=True)])
    path = tmp_path / 'correction.json'; path.write_text(json.dumps(packet), encoding='utf8')
    block = SimpleNamespace(name='low', groups=[(pts, [row])])
    state = SimpleNamespace(blocks=[block], original=SimpleNamespace(aliases={pts: pts}))
    return state, path, packet, row, authority


def test_actual_active_row_changes_without_touching_historical_source_or_world(tmp_path):
    state, path, _, old, authority = fixture(tmp_path)
    digest = correction._sha(authority); original_uv = old['uv'].copy()
    result = correction.install_original_ice_pixel_correction(state, path)
    revised = state.blocks[0].groups[0][1][0]
    assert correction._sha(authority) == digest and np.array_equal(old['uv'], original_uv)
    np.testing.assert_array_equal(revised['uv'], [[315., 859.], [340., 867.], [550., 879.]])
    for field in ('world', 'sigma', 'normal'):
        np.testing.assert_array_equal(revised[field], old[field])
    assert result['original_HELD_not_fitted'] and result['fresh_complete_source_profiles_and_J_required']
    assert result['corrected_rows'][0]['original_source_row']['uv'] == original_uv.tolist()


def test_changed_source_bytes_and_original_runtime_pixels_reject_before_mutation(tmp_path):
    state, path, packet, old, _ = fixture(tmp_path)
    Path(packet['source_evidence'][0]['path']).write_bytes(b'changed')
    with pytest.raises(ValueError, match='evidence changed'):
        correction.install_original_ice_pixel_correction(state, path)
    assert state.blocks[0].groups[0][1][0] is old
    state, path, _, old, _ = fixture(tmp_path)
    old['uv'][0, 1] += 1
    with pytest.raises(ValueError, match='runtime source: uv'):
        correction.install_original_ice_pixel_correction(state, path)


def test_held_points_cannot_be_silently_promoted_or_model_selected_pixels_admitted(tmp_path):
    state, path, packet, old, _ = fixture(tmp_path, collection='HELD')
    with pytest.raises(ValueError, match='existing FIT'):
        correction.install_original_ice_pixel_correction(state, path)
    assert state.blocks[0].groups[0][1][0] is old
    state, path, packet, _, _ = fixture(tmp_path)
    packet['model_projection_used_for_selection'] = True
    path.write_text(json.dumps(packet), encoding='utf8')
    with pytest.raises(ValueError, match='source-only'):
        correction.install_original_ice_pixel_correction(state, path)


def test_later_bad_record_does_not_partially_correct_state(tmp_path):
    state, path, packet, old, _ = fixture(tmp_path)
    packet['records'].append(dict(packet['records'][0], original_packet_row_index=1))
    path.write_text(json.dumps(packet), encoding='utf8')
    with pytest.raises(ValueError, match='exact original packet row'):
        correction.install_original_ice_pixel_correction(state, path)
    assert state.blocks[0].groups[0][1][0] is old


def test_declared_lane_user_pixel_correction_preserves_material_and_without_inventing_support(tmp_path):
    state, path, packet, old, authority = fixture(tmp_path)
    old.update(id='lane3_edge2', curve_id='lane3_edge2',
               world_scene_binding=dict(kind='finite_known_lane_edge', feature='lane3', edge=2))
    old.pop('normal'); old.pop('normal_interval')
    payload = pickle.loads(authority.read_bytes())
    payload['programme_absolute_rows'][0]['source_row'] = old
    authority.write_bytes(pickle.dumps(payload))
    packet['original_absolute_packet']['sha256'] = correction._sha(authority)
    record = packet['records'][0]
    record.update(physical_curve_id='lane3_edge2', physical_identity_changed=False,
                  original_pts=old['pts'], original_id=old['id'], original_split=old['split'],
                  original_collection_role='FIT', original_uv=old['uv'][:2].tolist(),
                  original_sigma=[5., 5.],
                  original_full_uv_float64_sha256=hashlib.sha256(old['uv'].astype(np.float64).tobytes()).hexdigest(),
                  corrected_normal_interval_px=None)
    path.write_text(json.dumps(packet), encoding='utf8')
    correction.install_original_ice_pixel_correction(state, path)
    revised = state.blocks[0].groups[0][1][0]
    assert 'normal' not in revised and 'normal_interval' not in revised
    np.testing.assert_array_equal(revised['world'], old['world'])
    np.testing.assert_array_equal(revised['sigma'], old['sigma'])
    assert revised['active_source_pixel_correction']['original_named_lane_identity_preserved']
    # Even a source-confirmed pixel change must not silently rebind the curve.
    state.blocks[0].groups = [(old['pts'], [old])]
    record['physical_curve_id'] = 'lane2_edge2'
    path.write_text(json.dumps(packet), encoding='utf8')
    with pytest.raises(ValueError, match='exact existing physical identity'):
        correction.install_original_ice_pixel_correction(state, path)
    assert state.blocks[0].groups[0][1][0] is old


def test_authenticated_navigation_does_not_authorize_projected_uv(tmp_path):
    state, path, packet, _, authority = fixture(tmp_path)
    packet.update(model_projection_used_for_selection=True, model_guided_navigation=True,
                  observed_UV_generated_from_model=False, raw_source_pixels_read=True,
                  model_guidance_reference=dict(path=str(authority), sha256=correction._sha(authority)))
    path.write_text(json.dumps(packet), encoding='utf8')
    result = correction.install_original_ice_pixel_correction(state, path)
    assert result['model_guided_navigation'] and not result['observed_UV_generated_from_model']
    packet['observed_UV_generated_from_model'] = True
    path.write_text(json.dumps(packet), encoding='utf8')
    with pytest.raises(ValueError, match='never projected UV'):
        correction.install_original_ice_pixel_correction(state, path)


def painted_center_fixture(tmp_path):
    state, path, packet, old, authority = fixture(tmp_path)
    old.update(id='arbitrary_registered_painted_curve', point_correspondence=False,
               curve_observation_kind='axis_scanline_branch', scanline_fixed_axis='y', scanline_branch='unique_x',
               normal_interval_px=old['normal_interval'].copy(), source_integration_weights=np.array([.2, .3, .5]))
    payload = pickle.loads(authority.read_bytes()); payload['programme_absolute_rows'][0]['source_row'] = old
    authority.write_bytes(pickle.dumps(payload)); packet['original_absolute_packet']['sha256'] = correction._sha(authority)
    identity = ['line', 'independently_named_ice_line']
    state.geometry = SimpleNamespace(_binding=lambda row: ('source', dict(kind='line', contract=dict(physical_identity=identity))))
    record = packet['records'][0]
    record.update(point_indices=[0, 1, 2], corrected_uv=[[320., 896.], [345., 889.], [555., 879.]],
                  corrected_normal_interval_px=None, measurement_role='painted_line_center',
                  physical_edge_role='painted_line_center', original_geometry_identity=identity,
                  physical_identity_changed=False, paint_footprint_is_not_location_uncertainty=True,
                  remove_original_normal_interval=True, original_pts=old['pts'], original_id=old['id'],
                  original_split=old['split'], original_collection_role='FIT', original_actual_pts=old['pts'],
                  original_block='low', original_group_order=0, original_row_order=0,
                  original_uv=old['uv'].tolist(), original_sigma=old['sigma'].tolist(),
                  original_full_uv_float64_sha256=hashlib.sha256(old['uv'].astype(np.float64).tobytes()).hexdigest(),
                  paint_footprint_evidence=dict(role='historical_paint_band_for_review_only'))
    path.write_text(json.dumps(packet), encoding='utf8')
    return state, path, packet, old, authority


def test_painted_centre_correction_keeps_geometry_sampling_and_original_paint_history(tmp_path):
    state, path, packet, old, authority = painted_center_fixture(tmp_path)
    digest = correction._sha(authority)
    result = correction.install_original_ice_pixel_correction(state, path)
    revised = state.blocks[0].groups[0][1][0]
    assert correction._sha(authority) == digest
    assert 'normal_interval' not in revised and 'normal_interval_px' not in revised
    assert np.array_equal(old['normal_interval'], np.tile([-15., 15.], (3, 1)))
    for key in ('sigma', 'world', 'normal', 'source_integration_weights'):
        np.testing.assert_array_equal(revised[key], old[key])
    for key in ('curve_observation_kind', 'scanline_branch', 'scanline_fixed_axis', 'point_correspondence'):
        assert revised[key] == old[key]
    assert revised['source_paint_footprint_evidence']['original_source_fields']['normal_interval'] == old['normal_interval'].tolist()
    assert result['corrected_rows'][0]['original_normal_interval_removed_from_operator']
    assert result['fresh_complete_source_profiles_and_J_required']
    assert result['corrected_rows'][0]['original_source_row']['normal_interval'] == old['normal_interval'].tolist()


@pytest.mark.parametrize('change, message', [
    ({'original_geometry_identity': ['line', 'another_material']}, 'registered ordinary ice line'),
    ({'physical_identity_changed': True}, 'exact source/carrier identity'),
    ({'paint_footprint_is_not_location_uncertainty': False}, 'no paint-width location band'),
    ({'corrected_normal_interval_px': [[-15., 15.]] * 3}, 'no paint-width location band'),
    ({'point_indices': [0, 1], 'original_uv': [[315., 896.], [340., 889.]],
      'original_sigma': [5., 5.], 'corrected_uv': [[320., 896.], [345., 889.]]}, 'every original point'),
])
def test_painted_centre_binding_and_semantics_fail_closed_without_partial_mutation(tmp_path, change, message):
    state, path, packet, old, _ = painted_center_fixture(tmp_path)
    packet['records'][0].update(change); path.write_text(json.dumps(packet), encoding='utf8')
    with pytest.raises(ValueError, match=message):
        correction.install_original_ice_pixel_correction(state, path)
    assert state.blocks[0].groups[0][1][0] is old


@pytest.mark.parametrize('kind,identity', [
    ('circle', ['circle', 'F2', 'outer']),
    ('circle', ['fixed_circle', 'R1', 'inner']),
    ('circle', ['circle', 'C0', 'outer']),
    ('circle', ['fixed_circle', 'unseen_target', 'middle']),
    ('line', ['fixed_circle', 'unseen_target', 'inner']),
])
def test_ring_boundaries_cannot_be_reinterpreted_as_painted_midlines(tmp_path, kind, identity):
    state, path, packet, old, _ = painted_center_fixture(tmp_path)
    state.geometry = SimpleNamespace(_binding=lambda row: ('source', dict(kind=kind, contract=dict(physical_identity=identity))))
    packet['records'][0]['original_geometry_identity'] = identity
    path.write_text(json.dumps(packet), encoding='utf8')
    with pytest.raises(ValueError, match='ring boundaries retain outer/inner semantics'):
        correction.install_original_ice_pixel_correction(state, path)
    assert state.blocks[0].groups[0][1][0] is old
    assert np.array_equal(old['normal_interval'], np.tile([-15., 15.], (3, 1)))


@pytest.mark.parametrize('rings,fill,source_role,allowed', [
    (None, None, 'painted_line_center', True),
    ([], None, 'painted_line_center', True),
    ([{'role': 'outer'}, {'role': 'inner'}], None, 'painted_line_center', False),
    (None, [1., 1., 1.], 'painted_line_center', False),
    (None, None, 'caller_label_without_original_source_semantics', False),
])
def test_only_source_confirmed_single_unfilled_circular_stroke_can_use_centre_semantics(
        tmp_path, rings, fill, source_role, allowed):
    state, path, packet, old, authority = painted_center_fixture(tmp_path)
    name = 'arbitrary_single_circular_stroke'
    identity = ['fixed_circle', name, 'outer']
    feature = dict(id=name, fill_srgb=fill)
    if rings is not None:feature['rings'] = rings
    old['source_uv_role'] = source_role
    payload = pickle.loads(authority.read_bytes()); payload['programme_absolute_rows'][0]['source_row'] = old
    authority.write_bytes(pickle.dumps(payload)); packet['original_absolute_packet']['sha256'] = correction._sha(authority)
    packet['records'][0]['original_geometry_identity'] = identity
    state.geometry = SimpleNamespace(features={name: feature},
        _binding=lambda row: ('source', dict(kind='circle', name=name, contract=dict(physical_identity=identity))))
    path.write_text(json.dumps(packet), encoding='utf8')
    if allowed:
        correction.install_original_ice_pixel_correction(state, path)
        revised = state.blocks[0].groups[0][1][0]
        assert 'normal_interval' not in revised
        assert np.array_equal(revised['world'], old['world'])
    else:
        with pytest.raises(ValueError, match='ring boundaries retain outer/inner semantics'):
            correction.install_original_ice_pixel_correction(state, path)
        assert state.blocks[0].groups[0][1][0] is old


def painted_retired_fixture(tmp_path):
    from calibration.solver_runtime.source_point_qualification import SCHEMA, file_sha256, original_uv_sha256
    from calibration.solver_runtime.original_source_point_qualification import FAMILY
    state, path, packet, old, authority = painted_center_fixture(tmp_path)
    r = packet['records'][0]
    points = [0, 2]
    r.update(point_indices=points, corrected_uv=np.asarray(r['corrected_uv'])[points].tolist(),
        original_uv=old['uv'][points].tolist(), original_sigma=old['sigma'][points].tolist(),
        retired_original_point_indices=[1])
    review = tmp_path/'qualification_review.json';review.write_text('{"basis":"actual RGB ambiguity"}',encoding='utf8')
    record = dict(original_packet_row_index=0,
        locator=dict(block='low',collection_role='FIT',group_actual_pts=old['pts'],group_order=0,row_order=0),
        row_id=old['id'],source_pts=old['pts'],original_uv_float64_sha256=original_uv_sha256(old),
        excluded_point_indices=[1],review_basis='original_rgb_only',decision_is_not_from_model_residual_or_cost=True,
        reason='Actual RGB has two unresolved material branches',
        review_authority=dict(path=str(review),sha256=file_sha256(review)),
        native_rgb_evidence=[dict(packet['source_evidence'][0],source_pts=old['pts'],
            source_video_sha256=correction.SOURCE_SHA,time_base='1/15360')])
    manifest = dict(schema=SCHEMA,source_family=FAMILY,source_video_sha256=correction.SOURCE_SHA,time_base='1/15360',
        original_packet=dict(packet['original_absolute_packet'],format='original_absolute_compact_pickle'),records=[record])
    qualification=tmp_path/'qualification.json';qualification.write_text(json.dumps(manifest),encoding='utf8')
    r['source_point_qualification_reference']=dict(path=str(qualification),sha256=file_sha256(qualification))
    path.write_text(json.dumps(packet),encoding='utf8')
    state.blocks[0].quadratic=np.zeros(6,bool);state.blocks[0].bundle=SimpleNamespace(motion_rows=0)
    return state,path,packet,old,qualification,manifest


def test_paint_row_measured_and_retired_union_requires_actual_physical_removal(tmp_path):
    from calibration.solver_runtime.original_source_point_qualification import OriginalSourcePointQualification, SourceWithOriginalPointQualification
    from calibration.solver_runtime.original_absolute_input_compaction import slice_observed_points
    state,path,packet,old,qualification,_=painted_retired_fixture(tmp_path)
    correction.install_original_ice_pixel_correction(state,path)
    full=state.blocks[0].groups
    # The unknown point retains original UV in history. It is never relabelled
    # as a measurement, and zeroing its residual slots cannot open the stream.
    revised=full[0][1][0]
    np.testing.assert_array_equal(revised['uv'][1],old['uv'][1])
    with pytest.raises(ValueError,match='verified physical input compaction'):
        SourceWithOriginalPointQualification(None,state,qualification)
    pre=OriginalSourcePointQualification(qualification);pre.qualify_state(state)
    state.original_absolute_point_qualification_prequalification=pre
    state.original_absolute_input_compaction_contract=dict(removed_unqualified_absolute_points=1)
    state.blocks[0].original_absolute_groups=full
    with pytest.raises(ValueError,match='physical input removal, not zero slots'):
        OriginalSourcePointQualification(qualification).qualify_state(state)
    compact=slice_observed_points(revised,[0,2]);state.blocks[0].groups=[(old['pts'],[compact])]
    final=OriginalSourcePointQualification(qualification);final.qualify_state(state)
    assert final.contract['scalar_slots_removed']==2
    assert final.contract['painted_row_retirement_physical_removal_verified'][0]['retired_original_point_indices']==[1]
    np.testing.assert_array_equal(compact['sigma'],old['sigma'][[0,2]])
    np.testing.assert_array_equal(compact['world'],old['world'])
    assert 'normal_interval'not in compact
    compact['uv'][0,0]+=1
    with pytest.raises(ValueError,match='retained source bytes changed'):
        OriginalSourcePointQualification(qualification).qualify_state(state)


@pytest.mark.parametrize('change, message',[
    ({'retired_original_point_indices':[0]},'every original point'),
    ({'retired_original_point_indices':[]},'every original point'),
    ({'source_point_qualification_reference':None},'exact source point qualification reference'),
])
def test_paint_retirement_coverage_and_source_reference_fail_closed(tmp_path,change,message):
    state,path,packet,old,_,_=painted_retired_fixture(tmp_path)
    packet['records'][0].update(change);path.write_text(json.dumps(packet),encoding='utf8')
    with pytest.raises(ValueError,match=message):correction.install_original_ice_pixel_correction(state,path)
    assert state.blocks[0].groups[0][1][0]is old


def test_paint_retirement_changed_manifest_and_different_final_reference_reject(tmp_path):
    from calibration.solver_runtime.original_source_point_qualification import OriginalSourcePointQualification
    state,path,packet,old,qualification,manifest=painted_retired_fixture(tmp_path)
    qualification.write_text(json.dumps(dict(manifest,reason='changed')),encoding='utf8')
    with pytest.raises(ValueError,match='source evidence changed'):
        correction.install_original_ice_pixel_correction(state,path)
    packet['records'][0]['source_point_qualification_reference']['sha256']=correction._sha(qualification)
    path.write_text(json.dumps(packet),encoding='utf8');correction.install_original_ice_pixel_correction(state,path)
    other=tmp_path/'other_same_facts.json';other.write_bytes(qualification.read_bytes())
    with pytest.raises(ValueError,match='final exact physical qualification manifest'):
        OriginalSourcePointQualification(other).qualify_state(state)


def test_production_must_select_exact_qualification_and_physical_compactor(tmp_path):
    state,path,_,_,qualification,_=painted_retired_fixture(tmp_path)
    contract=correction.install_original_ice_pixel_correction(state,path)
    for selected,compactor in [(None,path),(qualification,None),(None,None)]:
        with pytest.raises(ValueError,match='selected qualification and physical input compaction'):
            correction.require_painted_retirement_compaction(contract,selected,compactor)
    correction.require_painted_retirement_compaction(contract,qualification,path)
    other=tmp_path/'same_facts.json';other.write_bytes(qualification.read_bytes())
    with pytest.raises(ValueError,match='exact row retirement authority'):
        correction.require_painted_retirement_compaction(contract,other,path)


@pytest.mark.parametrize('fill,allowed',[(None,True),('FFFFFF',False)])
def test_registered_single_circle_reads_explicit_material_style_fill(tmp_path,fill,allowed):
    state,path,packet,old,authority=painted_center_fixture(tmp_path)
    name='single_paint_circle_from_registered_scene';identity=['circle',name,'outer']
    old['source_uv_role']='painted_line_center'
    payload=pickle.loads(authority.read_bytes());payload['programme_absolute_rows'][0]['source_row']=old
    authority.write_bytes(pickle.dumps(payload));packet['original_absolute_packet']['sha256']=correction._sha(authority)
    packet['records'][0]['original_geometry_identity']=identity
    state.geometry=SimpleNamespace(features={name:dict(style=dict(fill_srgb=fill))},
        _binding=lambda row:('source',dict(kind='circle',name=name,contract=dict(physical_identity=identity))))
    path.write_text(json.dumps(packet),encoding='utf8')
    if allowed:
        correction.install_original_ice_pixel_correction(state,path)
        assert 'normal_interval'not in state.blocks[0].groups[0][1][0]
    else:
        with pytest.raises(ValueError,match='ring boundaries retain outer/inner semantics'):
            correction.install_original_ice_pixel_correction(state,path)


def original_course_fixture(tmp_path, policy='top'):
    # Use the public owned source declaration function, without constructing an
    # event owner, graph or optimizer. A changed registry must fail closed.
    from calibration.solver_runtime import install
    install()
    state, path, packet, old, authority = fixture(tmp_path)
    old.update(id='far_top' if policy == 'top' else 'far_contact',
               source_uv=old['uv'].copy(), source_integration_weights=np.array([.2, .3, .5]))
    payload = pickle.loads(authority.read_bytes())
    payload['programme_absolute_rows'][0]['source_row'] = old
    authority.write_bytes(pickle.dumps(payload))
    packet['original_absolute_packet']['sha256'] = correction._sha(authority)
    identity = 'original_named_board_top' if policy == 'top' else 'original_named_ice_board_contact'
    material = SimpleNamespace(old_world=old['world'].copy(), contract=dict(z_policy=policy))
    binding = dict(kind='course', material=material, contract=dict(physical_identity=identity))
    state.geometry = SimpleNamespace(_binding=lambda row: ('source', binding))
    state.q = np.array([.1, .2, .3])
    state.fixed_parameter_indices = np.array([2])
    record = packet['records'][0]
    record.update(corrected_normal_interval_px=None, physical_identity_changed=False,
                  original_pts=old['pts'], original_id=old['id'], original_split=old['split'],
                  original_collection_role='FIT', original_uv=old['uv'][:2].tolist(),
                  original_sigma=old['sigma'][:2].tolist(),
                  original_full_uv_float64_sha256=hashlib.sha256(old['uv'].astype(np.float64).tobytes()).hexdigest())
    if policy == 'top':
        record.update(measurement_role='board_top_boundary',
                      physical_edge_role='vertical_board_blue_upper_boundary',
                      original_geometry_identity=identity, original_course_z_policy='top',
                      original_actual_pts=old['pts'], original_block='low',
                      original_group_order=0, original_row_order=0)
    path.write_text(json.dumps(packet), encoding='utf8')
    return state, path, packet, old, authority, binding


def test_certified_top_pixels_preserve_finite_material_operator_sigma_support_and_fixed_q(tmp_path):
    state, path, _, old, authority, _ = original_course_fixture(tmp_path)
    original_digest = correction._sha(authority)
    original_q = state.q.copy()
    result = correction.install_original_ice_pixel_correction(state, path)
    revised = state.blocks[0].groups[0][1][0]
    assert correction._sha(authority) == original_digest
    np.testing.assert_array_equal(old['uv'], [[315., 896.], [340., 889.], [550., 879.]])
    np.testing.assert_array_equal(revised['uv'], [[315., 859.], [340., 867.], [550., 879.]])
    np.testing.assert_array_equal(revised['source_uv'], revised['uv'])
    for field in ('world', 'sigma', 'normal', 'normal_interval', 'source_integration_weights'):
        np.testing.assert_array_equal(revised[field], old[field])
    assert revised['id'] == old['id'] and revised['split'] == old['split']
    assert len(revised['uv']) == len(old['uv'])
    assert result['corrected_rows'][0]['original_top_material_preserved']
    assert not result['corrected_rows'][0]['original_contact_material_preserved']
    assert not result['corrected_rows'][0]['original_normal_interval_removed_from_operator']
    assert not result['corrected_rows'][0]['optical_support_updated']
    np.testing.assert_array_equal(state.q, original_q)
    assert result['original_world_sigma_roles_source_order_masks_and_scalar_slots_unchanged']


@pytest.mark.parametrize('typed_source', [False, True])
def test_top_original_semantic_alias_reuses_exact_certified_finite_material(tmp_path, typed_source):
    state, path, packet, old, authority, binding = original_course_fixture(tmp_path)
    declared = 'original_named_board_top'
    cached = 'full_board_top'
    if typed_source:
        old['world_scene_binding'] = dict(kind='finite_full_board_top_perimeter')
        declared, cached = cached, declared
        payload = pickle.loads(authority.read_bytes())
        payload['programme_absolute_rows'][0]['source_row'] = old
        authority.write_bytes(pickle.dumps(payload))
        packet['original_absolute_packet']['sha256'] = correction._sha(authority)
        packet['records'][0]['original_geometry_identity'] = declared
        path.write_text(json.dumps(packet), encoding='utf8')
    binding['contract']['physical_identity'] = cached
    binding['material'].contract.update(
        schema='rinkspace.original-declared-finite-absolute-course-material.v1', identity=cached)
    q = state.q.copy()
    result = correction.install_original_ice_pixel_correction(state, path)
    assert result['corrected_rows'][0]['original_top_material_preserved']
    np.testing.assert_array_equal(state.q, q)
    np.testing.assert_array_equal(state.blocks[0].groups[0][1][0]['world'], old['world'])


@pytest.mark.parametrize('cached, defect', [
    ('original_reviewed_lower_long_side_top', None),
    ('full_board_top', 'unqualified_schema'),
    ('full_board_top', 'certificate_identity_mismatch'),
])
def test_top_alias_does_not_admit_other_facets_or_uncertified_names(tmp_path, cached, defect):
    state, path, _, old, _, binding = original_course_fixture(tmp_path)
    binding['contract']['physical_identity'] = cached
    binding['material'].contract.update(
        schema='rinkspace.original-declared-finite-absolute-course-material.v1', identity=cached)
    if defect == 'unqualified_schema':
        binding['material'].contract['schema'] = 'arbitrary-metadata'
    elif defect == 'certificate_identity_mismatch':
        binding['material'].contract['identity'] = 'original_named_board_top'
    with pytest.raises(ValueError, match='finite current-course board-top certificate'):
        correction.install_original_ice_pixel_correction(state, path)
    assert state.blocks[0].groups[0][1][0] is old


@pytest.mark.parametrize('change, message', [
    ({'original_geometry_identity': 'another_course'}, 'exact original source identity'),
    ({'original_course_z_policy': 'contact'}, 'exact original source identity'),
    ({'physical_edge_role': 'advertisement_upper_print_edge'}, 'blue-upper boundary'),
    ({'physical_edge_role': 'blue_paint_band_center'}, 'blue-upper boundary'),
    ({'physical_identity_changed': True}, 'exact original source identity'),
    ({'original_group_order': 1}, 'original locator differs'),
    ({'original_sigma': [4., 4.]}, 'original point values differ'),
    ({'original_full_uv_float64_sha256': '0' * 64}, 'full UV identity differs'),
    ({'remove_original_normal_interval': True}, 'explicit painted-line centre'),
])
def test_top_source_identity_and_blue_upper_semantics_fail_before_any_mutation(tmp_path, change, message):
    state, path, packet, old, _, _ = original_course_fixture(tmp_path)
    packet['records'][0].update(change)
    path.write_text(json.dumps(packet), encoding='utf8')
    with pytest.raises(ValueError, match=message):
        correction.install_original_ice_pixel_correction(state, path)
    assert state.blocks[0].groups[0][1][0] is old


@pytest.mark.parametrize('defect', ['contact_material', 'ad_binding', 'unregistered_top', 'different_WORLD'])
def test_top_requires_the_preexisting_same_finite_top_certificate(tmp_path, defect):
    state, path, _, old, _, binding = original_course_fixture(tmp_path)
    if defect == 'contact_material':
        binding['material'].contract['z_policy'] = 'contact'
    elif defect == 'ad_binding':
        binding['kind'] = 'ad'
    elif defect == 'unregistered_top':
        binding['material'] = None
    else:
        binding['material'].old_world[0, 0] += 1.
    with pytest.raises(ValueError, match='finite current-course board-top certificate'):
        correction.install_original_ice_pixel_correction(state, path)
    assert state.blocks[0].groups[0][1][0] is old


@pytest.mark.parametrize('source_id', ['Ad4:top', 'unconfirmed_top_looking_curve', 'far_contact'])
def test_top_label_cannot_admit_an_ad_edge_or_contact_as_board_top(tmp_path, source_id):
    state, path, packet, old, authority, _ = original_course_fixture(tmp_path)
    old['id'] = source_id
    payload = pickle.loads(authority.read_bytes())
    payload['programme_absolute_rows'][0]['source_row'] = old
    authority.write_bytes(pickle.dumps(payload))
    packet['original_absolute_packet']['sha256'] = correction._sha(authority)
    packet['records'][0]['original_id'] = source_id
    path.write_text(json.dumps(packet), encoding='utf8')
    with pytest.raises(ValueError, match='board-top WORLD and declared physical identity'):
        correction.install_original_ice_pixel_correction(state, path)
    assert state.blocks[0].groups[0][1][0] is old


def test_existing_contact_packet_keeps_its_legacy_fields_and_operator(tmp_path):
    state, path, _, old, _, _ = original_course_fixture(tmp_path, policy='contact')
    result = correction.install_original_ice_pixel_correction(state, path)
    revised = state.blocks[0].groups[0][1][0]
    assert result['corrected_rows'][0]['original_contact_material_preserved']
    assert not result['corrected_rows'][0]['original_top_material_preserved']
    for field in ('world', 'sigma', 'normal', 'normal_interval', 'source_integration_weights'):
        np.testing.assert_array_equal(revised[field], old[field])


@pytest.mark.parametrize('change', [None, 'filled', 'multiring', 'identity', 'source_video', 'frame_pts', 'projected_UV'])
def test_source_only_single_stroke_role_authority_does_not_relax_material_or_source_identity(tmp_path, change):
    state, path, packet, old, _ = painted_center_fixture(tmp_path)
    name = 'C0'
    identity = ['fixed_circle', name, 'outer']
    record = packet['records'][0]
    record['original_geometry_identity'] = identity
    feature = dict(id=name, fill_srgb=None)
    state.geometry = SimpleNamespace(features={name: feature},
        _binding=lambda row: ('source', dict(kind='circle', name=name, contract=dict(physical_identity=identity))))
    role = dict(schema='rinkspace.original-single-stroke-center-role-revision.v1',
        source_video_sha256=correction.SOURCE_SHA, time_base='1/15360',
        previous_measurement_role='untyped_approximate_curve_points', revised_measurement_role='painted_line_center',
        single_unfilled_painted_stroke=True, authority_kind='explicit_user_observation_correction',
        user_message='原点未落在实际圆线中心，请按原片修正。', model_projection_used_for_measurement=False,
        source_frame=dict(packet['source_evidence'][0]))
    for key in ('original_packet_row_index', 'original_actual_pts', 'original_id',
                'original_full_uv_float64_sha256', 'original_geometry_identity'):
        role[key] = record[key]
    if change == 'filled':feature['fill_srgb'] = [1., 1., 1.]
    elif change == 'multiring':feature['rings'] = [dict(role='outer'), dict(role='inner')]
    elif change == 'identity':role['original_geometry_identity'] = ['fixed_circle', 'R4', 'outer']
    elif change == 'source_video':role['source_video_sha256'] = '0' * 64
    elif change == 'frame_pts':role['source_frame']['pts'] += 1
    elif change == 'projected_UV':role['model_projection_used_for_measurement'] = True
    authority = tmp_path / 'single_stroke_role_authority.json'
    authority.write_text(json.dumps(role), encoding='utf8')
    record['painted_center_role_revision_reference'] = dict(path=str(authority), sha256=correction._sha(authority))
    path.write_text(json.dumps(packet), encoding='utf8')
    if change is None:
        correction.install_original_ice_pixel_correction(state, path)
        revised = state.blocks[0].groups[0][1][0]
        assert revised['measurement_role'] == 'painted_line_center'
        np.testing.assert_array_equal(revised['world'], old['world'])
        np.testing.assert_array_equal(revised['sigma'], old['sigma'])
    else:
        with pytest.raises(ValueError):
            correction.install_original_ice_pixel_correction(state, path)
        assert state.blocks[0].groups[0][1][0] is old


def test_feedback_observer_failure_preserves_successfully_installed_source_rows(tmp_path,monkeypatch):
    from rinkspace import feedback_outbox
    state,path,_,old,_=fixture(tmp_path)
    def unavailable(*args,**kwargs):raise RuntimeError('derived feedback index unavailable')
    monkeypatch.setattr(feedback_outbox,'record_source_installation',unavailable)
    result=correction.install_original_ice_pixel_correction(state,path)
    assert state.original_ice_pixel_feedback['status']=='retry_pending'
    assert state.original_ice_pixel_feedback['runtime_rows_installed']
    np.testing.assert_array_equal(state.blocks[0].groups[0][1][0]['uv'],[[315.,859.],[340.,867.],[550.,879.]])
    np.testing.assert_array_equal(old['uv'],[[315.,896.],[340.,889.],[550.,879.]])
