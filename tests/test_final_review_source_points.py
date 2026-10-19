import hashlib
import json

import numpy as np
import pytest

from calibration.program_tracking import final_review_source_points as final


def write(path, value):
    path.write_text(json.dumps(value), encoding='utf8')


def ref(path):
    return dict(path=str(path), sha256=final.file_sha(path))


def fixture(tmp_path, monkeypatch):
    state = tmp_path / 'state'
    state.mkdir()
    source = dict(video_sha256='a' * 64, time_base='1/15360', pixel_domain='native1920x1080')
    dataset_path = tmp_path / 'dataset.json'
    write(dataset_path, dict(schema=final.SCHEMA, source=source, expected_actual_pts=[100, 200],
        fit_points=3, held_points=1, pending_support_points=2, pending_named_points=1))
    native_row = dict(id='manual', actual_pts=100, pts=100, uv=[[50.5, 51.], [60.5, 61.]],
        curve=dict(kind='board_top'), normal=[[0., 1.], [0., 1.]])
    extent_row = dict(id='extent', actual_pts=200, pts=200, uv=[[70.5, 71.]],
        curve=dict(kind='house_ring'), curve_observation_kind='horizontal_extent',
        extent_side='right', normal=[[1., 0.]], normal_interval_px=[[-2., 3.]],
        source_y_is_not_a_world_point_correspondence=True)
    packet = tmp_path / 'original.json'
    write(packet, dict(rows=[native_row, extent_row]))
    image = tmp_path / 'original_frame.bin'
    image.write_bytes(b'existing source bytes')

    def record(row, index):
        canonical = json.dumps(row, sort_keys=True, separators=(',', ':'),
                               ensure_ascii=False, allow_nan=False).encode('utf8')
        return dict(row, packet_reference=ref(packet), original_row_index=index,
            source_row_canonical_json_sha256=hashlib.sha256(canonical).hexdigest(),
            source_raw_frame=str(image), source_file_sha256=final.file_sha(image))

    manual = dict(record(native_row, 0), direct_human_GT=True,
                  model_or_RGB_transport_used_to_generate_UV=False)
    extent = dict(record(extent_row, 1), model_projection_used_to_generate_or_select_uv=False)
    companion = dict(extent, id='extent_companion', source_row_id='extent',
        raw_support_companion_of='extent', original_source_operator='original_complete_curve_horizontal_extent',
        original_source_support=dict(normal=[[1., 0.]], normal_interval_px=[[-2., 3.]]))
    native_path, confirmed_path = tmp_path / 'native.json', tmp_path / 'confirmed.json'
    write(native_path, dict(schema='rinkspace.registered-direct-human-hard-truth-authorities.v1',
        source_video_sha256=source['video_sha256'], time_base=source['time_base'], records=[manual]))
    write(confirmed_path, dict(schema='rinkspace.user-confirmed-native-source-priority-authorities.v1',
                              records=[extent, companion]))
    manifest_path = tmp_path / 'input_manifest.json'
    manifest = dict(legacy_pixel_objective_replaced=True, strict_source_dataset=dict(
        schema='rinkspace.strict-reviewed-finite-programme-source.v1',
        archived_pixel_factors_in_FIT=False, all_pixel_families_routed_to_strict_input=True,
        finite_dataset_contract=dict(dataset_reference=ref(dataset_path))),
        highest_direct_human_hard_truth=dict(native=dict(authority=ref(native_path),
            source_groups=1, source_UV_queries=2)),
        highest_user_confirmed_native_sources=dict(authority=ref(confirmed_path),
                                                   ids=['extent', 'extent_companion']))
    write(manifest_path, manifest)
    camera_path = tmp_path / 'camera.npz'
    np.savez(camera_path, pts=np.array([100, 101, 200], np.int64),
        exposure_representative_pts=np.array([100, 100, 200], np.int64),
        metadata_json=json.dumps(dict(source_sha256=source['video_sha256'], source_time_base=source['time_base'])))
    checkpoint = state / 'complete_state_UNACCEPTED.npz'
    q = np.array([1., 2.])
    np.savez(checkpoint, q=q)
    identity = dict(q_sha256=hashlib.sha256(q.tobytes()).hexdigest(), state_npz_sha256=final.file_sha(checkpoint),
        camera_path=str(camera_path), camera_sha256=final.file_sha(camera_path),
        input_manifest_sha256=final.file_sha(manifest_path))
    write(state / 'state_identity.json', identity)
    truth = dict(complete_q_sha256=identity['q_sha256'], complete_state_reference=ref(checkpoint),
        source_input_manifest_reference=ref(manifest_path), families=dict(
            native=dict(authority=ref(native_path), group_ids=['manual']),
            confirmed=dict(authority=ref(confirmed_path), group_ids=['extent', 'extent_companion'])))
    write(state / 'truth_restoration_state.json', truth)
    calls = []

    def once(reference):
        calls.append(reference)
        fit = [dict(split='fit', scope='ice', actual_pts=100, points=3,
            uv=np.array([[12., 13.], [12., 13.], [60.5, 61.]]), sigma=np.ones(3))]
        held = [dict(split='held', scope='board', actual_pts=200, points=1,
            uv=np.array([[30., 31.]]), sigma=np.array([np.nan]))]
        receipt = dict(full_source_pack_passes=1, fit_points=3, held_points=1,
            every_original_row_proof_UV_sigma_clock_and_role_checked=True,
            source_pixels_sigma_roles_locators_and_groups_unchanged=True)
        return fit, held, receipt

    monkeypatch.setattr(final, 'load_strict_dataset_rows_once', once)
    return state, manifest_path, calls


def test_complete_display_keeps_roles_duplicate_denominators_and_x_only_original_context(tmp_path, monkeypatch):
    state, manifest, calls = fixture(tmp_path, monkeypatch)
    shared, bound = tmp_path / 'all_points.npz', tmp_path / 'bound.npz'
    result = final.freeze_final_review_source_points(shared, bound, state, input_manifest=manifest)
    assert len(calls) == 1
    assert result['original_queries_by_role'] == dict(fit=3, held=1, native_guard_only=2, confirmed_guard_only=2)
    assert result['original_source_queries'] == 8 and result['points'] == 5
    identity = final.read(state / 'state_identity.json')
    view = final.CurrentSourcePoints(bound, identity['camera_path'], q_sha256=identity['q_sha256'])
    np.testing.assert_array_equal(view.points(100), [[12., 13.], [50.5, 51.], [60.5, 61.]])
    np.testing.assert_array_equal(view.points(200), [[30., 31.], [70.5, 71.]])
    summary, native, confirmed = view.metadata['family_receipts']
    assert summary['held_missing_sigma_displayed_points'] == 1
    assert summary['fitting_roles'] == ['fit'] and summary['guard_queries_used_as_new_FIT'] is False
    assert summary['all_original_guard_queries_displayed'] is True
    assert native['not_in_FIT'] and confirmed['not_in_FIT']
    for row in confirmed['original_queries']:
        assert row['original_uv'] == [[70.5, 71.]]
        assert row['active_pixel_axes'] == ['x'] and row['context_pixel_axes'] == ['y']
        assert row['new_2D_measurement_claimed'] is False and row['missing_axis_or_support_filled'] is False
    assert confirmed['original_queries'][1]['original_operator_fields']['original_source_support'] == dict(
        normal=[[1., 0.]], normal_interval_px=[[-2., 3.]])


def test_prepare_is_read_only_and_rejects_retained_group_inventory_changes(tmp_path, monkeypatch):
    state, manifest, calls = fixture(tmp_path, monkeypatch)
    before = sorted(str(p) for p in tmp_path.rglob('*'))
    prepared = final.prepare_final_display_inputs(state, input_manifest=manifest)
    assert len(prepared['guards']) == 3 and not calls
    assert sorted(str(p) for p in tmp_path.rglob('*')) == before
    truth_path = state / 'truth_restoration_state.json'
    truth = final.read(truth_path)
    truth['families']['confirmed']['group_ids'].reverse()
    write(truth_path, truth)
    with pytest.raises(ValueError, match='group must be enumerated'):
        final.prepare_final_display_inputs(state, input_manifest=manifest)


def test_final_state_bytes_must_match_before_any_display_write(tmp_path, monkeypatch):
    state, manifest, calls = fixture(tmp_path, monkeypatch)
    np.savez(state / 'complete_state_UNACCEPTED.npz', q=np.array([1., 3.]))
    shared, bound = tmp_path / 'all_points.npz', tmp_path / 'bound.npz'
    with pytest.raises(ValueError, match='complete-state file changed'):
        final.freeze_final_review_source_points(shared, bound, state, input_manifest=manifest)
    assert not shared.exists() and not bound.exists() and not calls


def test_guard_only_role_cannot_be_claimed_on_a_legacy_FIT_route(tmp_path, monkeypatch):
    state, manifest_path, calls = fixture(tmp_path, monkeypatch)
    manifest = final.read(manifest_path)
    manifest['strict_source_dataset']['archived_pixel_factors_in_FIT'] = True
    write(manifest_path, manifest)
    identity = final.read(state / 'state_identity.json')
    identity['input_manifest_sha256'] = final.file_sha(manifest_path)
    write(state / 'state_identity.json', identity)
    with pytest.raises(ValueError, match='actual strict dataset FIT route'):
        final.prepare_final_display_inputs(state, input_manifest=manifest_path)
    assert not calls


def test_model_derived_original_query_stays_missing_with_no_coordinate_invention(tmp_path, monkeypatch):
    state, manifest_path, calls = fixture(tmp_path, monkeypatch)
    manifest = final.read(manifest_path)
    truth = final.read(state / 'truth_restoration_state.json')
    authority = final.read(manifest['highest_user_confirmed_native_sources']['authority']['path'])
    authority['records'][0]['model_projection_used_to_generate_or_select_uv'] = True
    path = tmp_path / 'model_derived_authority.json'
    write(path, authority)
    manifest['highest_user_confirmed_native_sources']['authority'] = ref(path)
    truth['families']['confirmed']['authority'] = ref(path)
    rows, receipts = final.read_guard_query_inputs(manifest, truth, [100, 200], dict(
        video_sha256='a' * 64, time_base='1/15360'))
    assert len(rows) == 2
    assert receipts[1]['original_query_points'] == 2
    assert receipts[1]['displayed_original_query_points'] == 1
    assert receipts[1]['missing_original_queries'][0]['id'] == 'extent'


def test_original_source_row_cannot_be_changed_under_display_authority(tmp_path, monkeypatch):
    state, manifest_path, calls = fixture(tmp_path, monkeypatch)
    manifest = final.read(manifest_path)
    truth = final.read(state / 'truth_restoration_state.json')
    authority = final.read(manifest['highest_direct_human_hard_truth']['native']['authority']['path'])
    authority['records'][0]['uv'][0][0] += .25
    path = tmp_path / 'changed_authority.json'
    write(path, authority)
    manifest['highest_direct_human_hard_truth']['native']['authority'] = ref(path)
    truth['families']['native']['authority'] = ref(path)
    with pytest.raises(ValueError, match='exact original guard UV'):
        final.read_guard_query_inputs(manifest, truth, [100, 200], dict(video_sha256='a' * 64, time_base='1/15360'))
