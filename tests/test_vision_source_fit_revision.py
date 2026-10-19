import hashlib
import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from rinkspace.vision_source_fit import OriginalFITSourceReader, reread_original_fit_source_frame
from rinkspace.vision_source_shapes import read_source_material_transition_profiles, read_source_stroke_midlines
from rinkspace.vision import RinkVisionModule


def bound(rgb):
    return dict(source_sha256='a' * 64, source_pts=256, time_base='1/15360',
        native_hw=list(rgb.shape[:2]), shot_id='native',
        decoded_native_rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest())


def stroke(texture=False):
    rgb = np.full((96, 128, 3), 220, np.uint8)
    rgb[28:68] = 165
    if texture:rgb[38:49] = 85
    return cv2.GaussianBlur(rgb, (0, 0), 1.)


def midline(rgb):
    return read_source_stroke_midlines(rgb, binding=bound(rgb), search_points_uv=[[64, 48]],
        normals=[[0, 1]], entering_search_intervals_px=[[-32, -15]],
        leaving_search_intervals_px=[[10, 38]], keep_profiles=True)


def request(rgb, *, identity=('line', 'arbitrary_unlisted_ice_line')):
    definition = dict(source_binding=bound(rgb), physical_identity=list(identity),
        physical_role='painted_stroke_center_axis', identity_authority=dict(reference='confirmed_source_material'),
        measurement_operator='paired_material_boundary_midline', two_sides_belong_to_same_painted_stroke=True,
        source_search_normals=[[0, 1]], entering_search_intervals_px=[[-32, -15]],
        leaving_search_intervals_px=[[10, 38]])
    return dict(original_packet_row_index=17,
        locator=dict(block='main4', collection_role='FIT', group_actual_pts=256, group_order=3, row_order=2),
        source_row=dict(id='arbitrary_unlisted_ice_line', pts=256, split='fit', uv=[[64., 48.]],
            sigma=[10.], world=[[0., 0., 0.], [1., 0., 0.]]), definition=definition)


def geometry(identity=('line', 'arbitrary_unlisted_ice_line'), *, feature=None):
    return SimpleNamespace(features={} if feature is None else {identity[1]: feature},
        _binding=lambda row: ('row', dict(kind=identity[0], name=identity[1], contract=dict(physical_identity=identity))))


def test_same_paint_midline_uses_two_material_sides_despite_asymmetric_interior_texture():
    clear, textured = midline(stroke()), midline(stroke(True))
    np.testing.assert_allclose(clear['actual_source_uv'], textured['actual_source_uv'], atol=.01)
    assert abs(textured['actual_source_uv'][0, 1] - 47.5) < 1.
    reading = textured['readings'][0]
    assert reading['measurement_operator'] == 'geometric_midpoint_of_two_actual_material_interfaces'
    assert not reading['intensity_median_or_darkest_peak_used_as_center']
    assert np.ptp(reading['observed_center_normal_interval_px']) < 8.
    assert reading['paint_footprint_is_not_centerline_uncertainty']


def test_unlisted_ordinary_line_enters_explicit_original_fit_revision_with_immutable_world_sigma():
    rgb = stroke(True)
    original = request(rgb)
    result = reread_original_fit_source_frame(rgb, binding=bound(rgb), geometry=geometry(), requests=[original])
    edit, = result['row_edits']
    assert edit['corrected_original_point_indices'] == [0]
    assert edit['retired_original_point_indices'] == []
    assert edit['locator'] == original['locator']
    assert edit['original_sigma'] == [10.]
    assert original['source_row']['uv'] == [[64., 48.]]
    assert result['public_correction_qualification_and_physical_compaction_transaction_required']


def test_occluded_source_point_has_retirement_candidate_and_no_replacement_UV():
    rgb = stroke()
    original = request(rgb)
    original['definition']['source_visible_mask'] = [False]
    result = reread_original_fit_source_frame(rgb, binding=bound(rgb), geometry=geometry(), requests=[original])
    edit, = result['row_edits']
    assert edit['corrected_uv'] == []
    assert edit['retired_original_point_indices'] == [0]
    assert result['point_decisions'][0]['actual_source_uv'] is None
    assert result['point_decisions'][0]['source_flags']['known_source_occlusion']


def test_clipped_material_side_does_not_get_a_midline_from_the_other_side():
    rgb = stroke()
    result = read_source_stroke_midlines(rgb, binding=bound(rgb), search_points_uv=[[64, 48]],
        normals=[[0, 1]], entering_search_intervals_px=[[-32, -15]], leaving_search_intervals_px=[[16, 33]])
    assert result['actual_source_uv'].shape == (0, 2)
    assert result['readings'][0]['actual_RGB_leaving_interface']['source_flags']['search_interval_truncated']


@pytest.mark.parametrize('feature', [dict(fill_srgb=[1., 1., 1.]), dict(fill_srgb=None, rings=[dict(role='outer'), dict(role='inner')])])
def test_filled_and_layered_circles_cannot_be_reinterpreted_as_ordinary_stroke_midlines(feature):
    rgb = stroke()
    identity = ('circle', 'R4', 'outer')
    with pytest.raises(ValueError, match='never a layered or filled boundary'):
        reread_original_fit_source_frame(rgb, binding=bound(rgb), geometry=geometry(identity, feature=feature), requests=[request(rgb, identity=identity)])


def test_source_clock_and_physical_identity_are_not_relaxed_by_a_definition_label():
    rgb = stroke()
    original = request(rgb)
    original['definition']['source_binding']['source_pts'] += 1
    with pytest.raises(ValueError, match='source_pts'):
        reread_original_fit_source_frame(rgb, binding=bound(rgb), geometry=geometry(), requests=[original])
    original = request(rgb)
    original['definition']['physical_identity'] = ['line', 'another_material']
    with pytest.raises(ValueError, match='registered original material'):
        reread_original_fit_source_frame(rgb, binding=bound(rgb), geometry=geometry(), requests=[original])


@pytest.mark.parametrize('identity, weights, transition, colours', [
    (('course', 'top'), [-1., 0., 1.], 'dark_to_light', ([180, 180, 180], [100, 130, 220])),
    (('course', 'contact'), [.5, .5, -1.], 'light_to_dark', ([220, 200, 80], [180, 180, 180])),
    (('ad', 'unlisted_external_print_material'), [1., -1., 0.], 'dark_to_light', ([200, 200, 200], [240, 100, 150])),
])
def test_top_contact_and_arbitrary_print_boundaries_use_declared_signed_material_not_a_name_whitelist(identity, weights, transition, colours):
    rgb = np.empty((96, 128, 3), np.uint8)
    rgb[:48], rgb[48:] = colours
    rgb = cv2.GaussianBlur(rgb, (0, 0), 2.)
    original = request(rgb, identity=identity)
    original['definition'].update(measurement_operator='signed_material_transition',
        physical_role='confirmed_actual_material_boundary', material_boundary_role_confirmed=True,
        material_signal_rgb_weights=weights, transition=transition, search_intervals_px=[[-25, 25]])
    result = reread_original_fit_source_frame(rgb, binding=bound(rgb), geometry=geometry(identity), requests=[original])
    assert result['row_edits'][0]['corrected_original_point_indices'] == [0]
    assert abs(result['row_edits'][0]['corrected_uv'][0][1] - 47.5) < 1.


def test_explicit_candidate_weight_selection_retains_failed_training_status():
    module = RinkVisionModule.__new__(RinkVisionModule)
    module.model_path = Path('candidate.pt')
    module.model_sha256 = 'b' * 64
    module.training_scope = dict(latest_step=12600, status='candidate_trained_with_recorded_source_regressions',
        source_retention_guard=dict(failed=True), accuracy_accepted=False)
    receipt = module.model_usage()
    assert receipt['path'] == str(module.model_path)
    assert receipt['training_step'] == 12600 and receipt['source_retention_failed']
    assert not receipt['accuracy_accepted'] and not receipt['inference_selection_is_formal_model_promotion']


def test_narrow_interface_cannot_clear_original_wide_search_competitors():
    rgb = np.full((96, 128, 3), 140, np.uint8)
    rgb[40:] = 175
    rgb[70:] = 200
    rgb = cv2.GaussianBlur(rgb, (0, 0), 1.2)
    local = read_source_material_transition_profiles(rgb, binding=bound(rgb), search_points_uv=[[64, 48]],
        normals=[[0, 1]], search_intervals_px=[[-20, 4]], transition='dark_to_light')
    assert local['readings'][0]['disposition'] == 'measured_material_transition'
    contextual = read_source_material_transition_profiles(rgb, binding=bound(rgb), search_points_uv=[[64, 48]],
        normals=[[0, 1]], search_intervals_px=[[-20, 4]], context_search_intervals_px=[[-25, 38]],
        transition='dark_to_light')
    row = contextual['readings'][0]
    np.testing.assert_array_equal(row['actual_source_transition_candidate_uv'], local['actual_source_uv'][0])
    assert not row['actual_source_uv'].size and row['source_context_resolution_required']
    assert row['source_context_profile']['source_flags']['separated_same_sign_transitions']
    assert not row['local_source_flags']['separated_same_sign_transitions']


def test_prepared_multi_frame_reader_loads_packet_geometry_once_and_preserves_native_identity(tmp_path, monkeypatch):
    from rinkspace import vision_camera_initialization, vision_data
    calls = dict(packet=0, geometry=0)
    factory = geometry()
    factory.register_row = lambda row, **kwargs: None
    def get_geometry(*args):
        calls['geometry'] += 1
        return factory, None, dict(test_actual_material_producer=True)
    original_unpickler = vision_data._DataOnlyUnpickler
    class CountedUnpickler(original_unpickler):
        def load(self):
            calls['packet'] += 1
            return super().load()
    monkeypatch.setattr(vision_camera_initialization, 'current_complete_source_geometry', get_geometry)
    monkeypatch.setattr(vision_data, '_DataOnlyUnpickler', CountedUnpickler)
    rgb = stroke(True)
    frames, rows = [], []
    for stamp in (256, 512):
        item = request(rgb)
        item['source_row']['pts'] = stamp
        item['locator']['group_actual_pts'] = stamp
        item['definition']['source_binding'].update(source_pts=stamp, shot_id='main4')
        rows.append(dict(item['locator'], source_row=item['source_row']))
        frames.append(dict(source_binding=item['definition']['source_binding'],
            records=[dict(original_packet_row_index=len(rows)-1, definition=item['definition'])]))
    pkl = tmp_path/'original.pkl'
    pkl.write_bytes(pickle.dumps(dict(programme_absolute_rows=rows)))
    state = tmp_path/'state.npz'; state.write_bytes(b'authenticated_public_provider_fixture')
    world = tmp_path/'world.json'; world.write_text('{}', encoding='utf8')
    def ref(path):return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    requested = tmp_path/'request.json'
    requested.write_text(json.dumps(dict(schema='rinkspace.original-FIT-source-RGB-reread-request.v1',
        original_absolute_packet=ref(pkl), complete_state=ref(state), world_initializer=ref(world), frames=frames)), encoding='utf8')
    reader = OriginalFITSourceReader(requested)
    for stamp in (256, 512):
        binding = dict(bound(rgb), source_pts=stamp, shot_id=f'single_source_frame_{stamp}')
        result = reader.read(rgb, binding)
        assert result['corrected_point_count'] == 1
        assert result['source_binding']['shot_id'] == 'main4'
        assert result['runtime_source_binding']['shot_id'] == binding['shot_id']
    assert calls == dict(packet=1, geometry=1)
    missing = reader.read(rgb, dict(bound(rgb), source_pts=768))
    assert missing['frame_original_FIT_locator_status'] == 'no_original_FIT_locator_at_this_frame'
    assert not missing['row_edits']
    changed = rgb.copy(); changed[0, 0] = 0
    with pytest.raises(ValueError, match='bound original RGB differ'):
        reader.read(changed, dict(bound(rgb), shot_id='runtime'))


@pytest.mark.parametrize('command', ['read-frame', 'review-shot'])
def test_inference_cli_passes_explicit_weights_and_prepared_original_request(command, monkeypatch):
    from rinkspace import vision
    received = {}
    monkeypatch.setattr(vision, 'configure', lambda path: ({}, Path('.')))
    def capture(*args, **kwargs):received.update(kwargs)
    monkeypatch.setattr(vision, 'read_frame', capture)
    monkeypatch.setattr(vision, 'review_shot', capture)
    args = ['vision', command, '--model', 'candidate.pt', '--original-fit-request', 'source.json']
    args += ['--pts', '256'] if command == 'read-frame' else ['--video', 'source.mp4', '--start-pts', '256',
        '--end-pts', '512', '--time-base', '1/15360', '--shot-id', 'runtime', '--expected-sha256', 'a'*64]
    monkeypatch.setattr('sys.argv', args)
    vision.main()
    assert received['model_path'] == Path('candidate.pt')
    assert received['original_fit_request_path'] == Path('source.json')
