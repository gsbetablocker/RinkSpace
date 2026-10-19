import copy
from fractions import Fraction
from pathlib import Path

import av
import cv2
import numpy as np
import pytest

from rinkspace.io import dump_json, load_json, sha256
from rinkspace.whole_rink_localization import (WholeRinkConfig, WholeRinkLocalizer,
    projection_segments, render_frame, track_window, frame_masks, _ice_edge,
    cut_signature, cut_diagnostic)
from rinkspace.whole_rink_localization import reverse_fill
from rinkspace.whole_rink_localization import _similarity_fit
from rinkspace.whole_rink_localization import _chain_point
from rinkspace.whole_rink_localization import _source_shot_context
from rinkspace.local_registration import RegistrationConfig, _fit


def fixture(tmp_path, *, radial=False, blank=False):
    rng = np.random.default_rng(12)
    gray = np.clip(rng.normal(220, 11, (270, 480)), 175, 254).astype(np.uint8)
    gray = cv2.GaussianBlur(gray, (3, 3), .5)
    image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if blank: image[:] = 220
    cv2.imwrite(str(tmp_path/'reference.png'), image)
    model = {'model_id': 'synthetic-whole', 'revision': 'test1', 'geometry_level': 'euclidean_conditional',
        'unit': 'unverified_arbitrary_units', 'structure': {'bounds': [0, 0, 480, 270],
            'boundary': {'id': 'B', 'kind': 'boundary', 'points_model': [[10, 10], [470, 10], [470, 260], [10, 260]],
                         'status': 'assumed', 'label': '合成边界'},
            'primitives': [{'id': 'L', 'kind': 'line', 'points_model': [[20, 135], [460, 135]], 'status': 'observed'}]},
        'view_poses': [{'id': 'synthetic_reference', 'image': 'reference.png', 'image_sha256': sha256(tmp_path/'reference.png'),
            'source_id': 'synthetic', 'source_sha256': 'a'*64, 'pts': 100, 'time_base': '1/100',
            'sample_size': [480, 270], 'ice_polygon_sample': [[3, 3], [476, 3], [476, 266], [3, 266]],
            'person_polygons_sample': [], 'H_image_from_model': np.eye(3).tolist(),
            'radial_distortion': {'k1': -.08, 'centre_image': [240, 135], 'scale_image': 480} if radial else None}]}
    dump_json(tmp_path/'model.json', model)
    return image, model


def sample(index=0, pts=100, **extra):
    return {'sample_index': index, 'source_frame_index': index, 'source_id': 'synthetic',
        'source_sha256': 'a'*64, 'pts': pts, 'time_base': '1/100', 'source_time_s': pts/100,
        'width': 960, 'height': 540, 'shot_id': 'one', **extra}


def config(**extra):
    return WholeRinkConfig(sample_width=480, sample_height=270, continuity_mode='legacy', **extra)


def test_real_ice_correspondences_establish_model_bridge_and_bounded_propagation(tmp_path):
    image, _ = fixture(tmp_path)
    tracker = WholeRinkLocalizer(tmp_path/'model.json', tmp_path, config(reobserve_every=1000))
    first = tracker.push(image, sample())
    assert first['status'] == 'candidate'
    assert first['ice_reobservation']['train_inliers'] >= 14
    assert first['model_reference']['image_sha256'] == sha256(tmp_path/'reference.png')
    assert first['model_reference']['pts'] == 100
    np.testing.assert_allclose(first['H_image_from_model'], np.diag([2, 2, 1]), atol=.001)
    moved = cv2.warpAffine(image, np.float32([[1, 0, 2], [0, 1, 1]]), (480, 270), borderValue=(220, 220, 220))
    second = tracker.push(moved, sample(1, 102))
    assert second['status'] == 'predicted'
    assert second['anchor_age_s'] == pytest.approx(.02)
    np.testing.assert_allclose(np.asarray(second['H_image_from_model'])[:2, 2], [4, 2], atol=.5)
    assert not second['independent_validation']
    # Keep sampling gaps under .6 while age since genuine model match grows.
    tracker.push(moved, sample(2, 140))
    expired = tracker.push(moved, sample(3, 162))
    assert expired['status'] == 'unavailable' and expired['projection_chain'] is None


def test_cut_clears_global_bridge_even_if_local_component_restarts(tmp_path):
    image, _ = fixture(tmp_path)
    tracker = WholeRinkLocalizer(tmp_path/'model.json', tmp_path, config(reobserve_every=1000))
    assert tracker.push(image, sample())['status'] == 'candidate'
    unknown = np.full_like(image, 220)
    cut = tracker.push(unknown, sample(1, 102, shot_id='two'))
    assert cut['break_applied'] and cut['projection_chain'] is None
    assert cut['H_model_from_image'] is None and tracker.bridge is None


def test_fresh_model_observation_recovers_expired_local_component(tmp_path):
    image, _ = fixture(tmp_path)
    tracker = WholeRinkLocalizer(tmp_path/'model.json', tmp_path, config(reobserve_every=10))
    assert tracker.push(image, sample())['status'] == 'candidate'
    tracker.local.last_good_time = 0.  # Component has exhausted its recovery gap.
    recovered = tracker.push(image, sample(10, 102))
    assert recovered['status'] == 'candidate'
    assert recovered['local_restart_reason'].startswith('fresh_model_ice_reobservation')
    assert recovered['local_registration']['status'] == 'candidate'
    assert tracker.push(image, sample(11, 104))['status'] == 'predicted'


def test_identity_initial_guess_on_blank_ice_is_never_global_localization(tmp_path):
    image, _ = fixture(tmp_path, blank=True)
    row = WholeRinkLocalizer(tmp_path/'model.json', tmp_path, config()).push(image, sample())
    assert row['status'] == 'unavailable' and row['H_image_from_model'] is None
    assert row['reobservation_attempts'][0]['reason'] == 'reference_has_insufficient_ice_texture'


def test_radial_reference_retains_nonlinear_chain_and_never_fakes_raw_inverse(tmp_path):
    image, _ = fixture(tmp_path, radial=True)
    row = WholeRinkLocalizer(tmp_path/'model.json', tmp_path, config()).push(image, sample())
    assert row['status'] == 'candidate'
    assert row['H_model_from_image'] is None and row['H_image_from_model'] is None
    assert row['projection_chain']['reference_radial_distortion']['k1'] == -.08
    assert row['image_domain'] == 'raw_reference_transfer_approximation'


def test_render_full_ice_not_fit_hull_and_exclude_people(tmp_path):
    image, model = fixture(tmp_path)
    row = WholeRinkLocalizer(tmp_path/'model.json', tmp_path, config()).push(image, sample())
    row['fit_support_polygon_sample'] = [[1, 1], [2, 1], [2, 2]]
    row['foreground_polygons_sample'] = [[[210, 90], [280, 90], [280, 180], [210, 180]]]
    before = image.copy()
    output = render_frame(image, model, row, captions=False)
    np.testing.assert_array_equal(output[90:181, 210:281], image[90:181, 210:281])
    assert np.any(output[133:138, 35:180] != image[133:138, 35:180])
    np.testing.assert_array_equal(image, before)
    row.update(status='unavailable', projection_chain=None)
    np.testing.assert_array_equal(render_frame(image, model, row, captions=False), image)


def test_reject_negative_model_plane_branch_and_horizon_connection():
    chain = {'H_reference_undistorted_from_model': [[1, 0, 0], [0, 1, 100], [.01, 0, 1]],
        'reference_radial_distortion': None, 'H_current_sample_from_reference_raw': np.eye(3).tolist()}
    assert list(projection_segments([[-200, -300], [-110, -300]], chain, 640, 360)) == []
    parts = list(projection_segments([[-200, 20], [200, 20]], chain, 640, 360))
    assert all(np.isfinite(p).all() and np.all(p >= 0) and np.all(p <= [639, 359]) for p in parts)
    chain['H_reference_undistorted_from_model'] = np.eye(3).tolist()
    chain['reference_radial_distortion'] = {'k1': -.2, 'centre_image': [240, 135], 'scale_image': 480}
    parts = list(projection_segments([[10, 40], [470, 40]], chain, 480, 270))
    assert len(parts) > 1  # Candidate radial curvature is actually sampled.
    chain['reference_radial_distortion'] = {'k1': -1, 'centre_image': [0, 0], 'scale_image': 100}
    assert list(projection_segments([[80, 40], [110, 40]], chain, 480, 270)) == []
    chain['reference_radial_distortion'] = None
    expected = list(projection_segments([[10, 40], [470, 40]], chain, 480, 270))
    chain['H_reference_undistorted_from_model'] = (-np.eye(3)).tolist()
    chain['reference_visible_branch_sign'] = -1
    np.testing.assert_allclose(list(projection_segments([[10, 40], [470, 40]], chain, 480, 270)), expected)
    chain['H_current_sample_from_reference_raw'] = (-np.eye(3)).tolist()
    chain['raw_transfer_visible_branch_sign'] = -1
    np.testing.assert_allclose(list(projection_segments([[10, 40], [470, 40]], chain, 480, 270)), expected)
    chain['raw_transfer_visible_branch_sign'] = 1
    assert list(projection_segments([[10, 40], [470, 40]], chain, 480, 270)) == []


def test_reference_hash_and_bad_time_fail_closed(tmp_path):
    image, _ = fixture(tmp_path)
    tracker = WholeRinkLocalizer(tmp_path/'model.json', tmp_path, config())
    with pytest.raises(ValueError, match='PTS'):
        tracker.push(image, sample(source_time_s=1.1))
    (tmp_path/'reference.png').write_bytes(b'changed')
    with pytest.raises(ValueError, match='identity'):
        WholeRinkLocalizer(tmp_path/'model.json', tmp_path, config())


def test_scale_stable_ice_refinement_exposes_gates_and_isolates_mask(tmp_path):
    image, _ = fixture(tmp_path)
    tracker = WholeRinkLocalizer(tmp_path/'model.json', tmp_path, config())
    matrix = np.array([[1.18, 0, -28], [0, 1.18, -15], [0, 0, 1.]])
    current = cv2.warpPerspective(image, matrix, (480, 270), borderValue=(220, 220, 220))
    domain = np.zeros((270, 480), np.uint8); domain[20:250, 30:450] = 255
    masks = frame_masks(current, domain)
    fit = _ice_edge(tracker.refs[0], current, masks, matrix, prewarp=True)
    assert fit['status'] == 'candidate'
    stages = fit['stage_counts']
    assert stages['initial_visible_in_safe_current_ice'] >= stages['flow_both_directions']
    assert stages['flow_both_directions'] >= stages['forward_backward_pass']
    assert stages['forward_backward_pass'] >= stages['textured_patch_pass'] >= stages['photometric_patch_pass']
    assert stages['photometric_patch_pass'] >= stages['ransac_inliers'] >= 14
    # Changing players/background outside the ice fitting mask cannot change
    # the accepted image correspondences, even through prewarp interpolation.
    modified = current.copy(); modified[masks['usable'] == 0] = [0, 0, 255]
    again = _ice_edge(tracker.refs[0], modified, masks, matrix, prewarp=True)
    assert again['stage_counts'] == stages
    np.testing.assert_allclose(again['H_reference_from_current'], fit['H_reference_from_current'])


def test_full_vfr_run_preserves_all_source_pts_and_produces_no_html(tmp_path):
    image, model = fixture(tmp_path)
    source = tmp_path/'original.mp4'
    times = [100, 102, 105, 109]
    with av.open(str(source), 'w') as target:
        writer = target.add_stream('libx264', rate=50)
        writer.width, writer.height, writer.pix_fmt = 480, 270, 'yuv420p'
        writer.time_base = Fraction(1, 100)
        for pts in times:
            frame = av.VideoFrame.from_ndarray(image, format='bgr24')
            frame.pts, frame.time_base = pts, Fraction(1, 100)
            for packet in writer.encode(frame): target.mux(packet)
        for packet in writer.encode(): target.mux(packet)
    with av.open(str(source)) as source_container:
        rows = [{'source_frame_index': i, 'pts': int(f.pts), 'time_base': str(f.time_base),
                 'source_time_s': float(f.pts*f.time_base)} for i, f in enumerate(source_container.decode(video=0))]
    digest = sha256(source)
    model['view_poses'][0].update(source_sha256=digest, pts=rows[0]['pts'], time_base=rows[0]['time_base'])
    dump_json(tmp_path/'model.json', model)
    dump_json(tmp_path/'configs/local.json', {'sources': {'synthetic': {'path': str(source)}}})
    dump_json(tmp_path/'outputs/workspace/synthetic/frame_index.json', {'video_id': digest, 'frames': rows})
    out = tmp_path/'run'
    record = track_window(tmp_path, tmp_path/'model.json', 'synthetic', 1., 1.11, out,
        selection_basis='synthetic VFR fixture only', config=config())
    assert len(record['frames']) == 4 and record['all_selected_frames_in_denominator']
    assert record['selected_window']['last_pts'] == rows[-1]['pts']
    assert sha256(source) == digest
    assert len(list(out.glob('comparison_*.jpg'))) == 3
    assert not list(out.glob('*.html'))
    mapping = load_json(out/'source_pts_map.json')['frames']
    for original, row in zip(rows, mapping):
        expected = Fraction(original['pts'])*Fraction(original['time_base'])-Fraction(1)
        assert row['proxy_pts']*Fraction(row['proxy_time_base']) == expected
        assert row['derived_pts']*Fraction(row['derived_time_base']) == expected


def test_full_source_cadence_detects_cut_without_rewriting_source_shots():
    left = np.full((270, 480, 3), (235, 235, 235), np.uint8)
    right = np.full((270, 480, 3), (25, 30, 100), np.uint8)
    assert not cut_diagnostic(cut_signature(left), cut_signature(left))['candidate']
    cut = cut_diagnostic(cut_signature(left), cut_signature(right))
    assert cut['candidate'] and cut['status'] == 'automatic_cut_candidate_not_human_confirmed'


def test_reverse_uses_future_anchor_without_changing_source_pts_or_promoting_predictions(tmp_path, monkeypatch):
    image, _ = fixture(tmp_path)
    tracker = WholeRinkLocalizer(tmp_path/'model.json', tmp_path, config())
    anchor = tracker.push(image, sample(3, 100))
    assert anchor['status'] == 'candidate'
    rows = []
    for i, pts in enumerate([20, 60, 80]):
        row = copy.deepcopy(anchor)
        row.update(sample(i, pts), status='unavailable', reason='original_failed_ice_observation',
                   H_image_from_model=None, H_model_from_image=None, projection_chain=None)
        rows.append(row)
    rows.append(anchor)
    monkeypatch.setattr('rinkspace.whole_rink_localization._source_images',
        lambda root, source_id, selected, config: ((r, image) for r in selected))
    before = [(r['pts'], r['time_base'], r['source_time_s']) for r in rows]
    result = reverse_fill(tmp_path, 'synthetic', rows, config())
    assert result['filled_frames'] == 2 and not result['inferred_frames_become_anchors']
    assert rows[0]['status'] == 'unavailable'  # Beyond the 0.6 s bound.
    assert [r['status'] for r in rows[1:]] == ['predicted', 'predicted', 'candidate']
    assert [(r['pts'], r['time_base'], r['source_time_s']) for r in rows] == before
    for row in rows[1:3]:
        assert row['future_anchor_used'] and row['anchor_age_s'] <= .6
        assert row['future_anchor_observation']['pts'] == 100
        assert row['forward_reason_before_reverse'] == 'original_failed_ice_observation'
        assert row['reverse_local_registration']['tracking_clock']['kind'] == 'reverse_elapsed_from_future_anchor'


def test_relative_similarity_uses_original_support_gates_and_rejects_projective_overfit():
    x, y = np.meshgrid(np.linspace(45, 440, 12), np.linspace(30, 240, 9))
    current = np.c_[x.ravel(), y.ravel()].astype(np.float32)
    target = current*1.04+[12., -7.]
    fit = _similarity_fit(current, target, (270, 480), RegistrationConfig())
    assert fit['status'] == 'candidate' and fit['relative_degrees_of_freedom'] == 4
    np.testing.assert_allclose(fit['H_reference_from_current'], [[1.04, 0, 12], [0, 1.04, -7], [0, 0, 1]], atol=1e-4)
    h = np.array([[1., .2, 0.], [0., 1., 0.], [.001, .001, 1.]])
    homogeneous = np.c_[current, np.ones(len(current))] @ h.T
    target = homogeneous[:, :2]/homogeneous[:, 2:]
    assert _fit(current, target, (270, 480), RegistrationConfig())['status'] == 'candidate'
    assert _similarity_fit(current, target, (270, 480), RegistrationConfig())['status'] == 'unavailable'


def test_model_anchor_conflict_preserves_both_hypotheses_but_clears_projection_and_propagation(tmp_path, monkeypatch):
    image, model = fixture(tmp_path)
    tracker = WholeRinkLocalizer(tmp_path/'model.json', tmp_path, config(reobserve_every=2))
    first = tracker.push(image, sample())
    ref = tracker.last_anchor
    fit = copy.deepcopy(first['ice_reobservation'])
    contradictory = np.array([[1., 0., 24.], [0., 1., 0.], [0., 0., 1.]])
    monkeypatch.setattr(tracker, '_reobserve', lambda *args: ((ref, contradictory, fit), []))
    conflict = tracker.push(image, sample(2, 102))
    assert conflict['status'] == 'ambiguous' and conflict['reason'] == 'model_anchor_conflict'
    assert conflict['H_image_from_model'] is None and conflict['projection_chain'] is None
    assert len(conflict['model_anchor_hypotheses']) == 2
    assert conflict['model_anchor_consistency']['p95_sample_px'] == pytest.approx(24)
    assert not conflict['model_anchor_consistency']['independent_accuracy_measurement']
    assert tracker.last_anchor is None and tracker.bridge is None
    np.testing.assert_array_equal(render_frame(image, model, conflict, captions=False), image)
    next_row = tracker.push(image, sample(3, 104))
    assert next_row['status'] == 'unavailable'
    assert next_row['propagation_segment_id'] != conflict['propagation_segment_id']


def test_small_model_refresh_stays_candidate_and_reverse_never_crosses_conflict(tmp_path, monkeypatch):
    image, _ = fixture(tmp_path)
    tracker = WholeRinkLocalizer(tmp_path/'model.json', tmp_path, config(reobserve_every=2))
    first = tracker.push(image, sample())
    ref = tracker.last_anchor
    fit = copy.deepcopy(first['ice_reobservation'])
    h = np.array([[1., 0., 2.], [0., 1., 0.], [0., 0., 1.]])
    monkeypatch.setattr(tracker, '_reobserve', lambda *args: ((ref, h, fit), []))
    refreshed = tracker.push(image, sample(2, 102))
    assert refreshed['status'] == 'candidate'
    assert not refreshed['model_anchor_consistency']['conflict']
    earlier = copy.deepcopy(refreshed)
    earlier.update(sample(0, 70), status='unavailable', projection_chain=None, propagation_segment_id=-1)
    barrier = copy.deepcopy(earlier)
    barrier.update(sample(1, 80), status='ambiguous', reason='model_anchor_conflict', propagation_barrier=True)
    monkeypatch.setattr('rinkspace.whole_rink_localization._source_images',
        lambda *args: pytest.fail('Cannot decode a reverse sequence across a known conflict'))
    result = reverse_fill(tmp_path, 'synthetic', [earlier, barrier, refreshed], config())
    assert result['filled_frames'] == 0
    assert earlier['status'] == 'unavailable' and barrier['status'] == 'ambiguous'


def test_reverse_overlap_cannot_silently_join_conflicting_forward_and_future_anchors(tmp_path, monkeypatch):
    image, model = fixture(tmp_path)
    tracker = WholeRinkLocalizer(tmp_path/'model.json', tmp_path, config())
    forward = tracker.push(image, sample(0, 80))
    missing = copy.deepcopy(forward)
    missing.update(sample(1, 90), status='unavailable', reason='forward_bound_expired',
        projection_chain=None, H_image_from_model=None, H_model_from_image=None)
    future = copy.deepcopy(forward)
    future.update(sample(2, 100))
    future['projection_chain']['H_current_sample_from_reference_raw'][0][2] += 24
    rows = [forward, missing, future]
    monkeypatch.setattr('rinkspace.whole_rink_localization._source_images',
        lambda root, source_id, selected, config: ((r, image) for r in selected))
    result = reverse_fill(tmp_path, 'synthetic', rows, config(), model=model)
    assert result['filled_frames'] == 1 and result['overlap_conflict_frames'] == 1
    assert forward['status'] == 'ambiguous' and forward['propagation_barrier']
    assert forward['conflict_discovered_by'] == 'offline_reverse_same_frame_overlap'
    assert forward['forward_status_before_reverse_conflict'] == 'candidate'
    assert len(forward['model_anchor_hypotheses']) == 2
    assert forward['H_image_from_model'] is None and forward['projection_chain'] is None
    assert missing['status'] == 'predicted' and future['status'] == 'candidate'
    assert [r['pts'] for r in rows] == [80, 90, 100]


def test_changing_source_identity_resets_bridge_even_if_pixels_time_and_shot_label_match(tmp_path, monkeypatch):
    image, _ = fixture(tmp_path)
    tracker = WholeRinkLocalizer(tmp_path/'model.json', tmp_path, config(reobserve_every=1000))
    first = tracker.push(image, sample())
    assert first['status'] == 'candidate'
    monkeypatch.setattr(tracker, '_reobserve', lambda *args: (None, []))
    changed = tracker.push(image, sample(1, 102, source_sha256='b'*64))
    assert changed['source_identity_break_applied'] and changed['break_applied']
    assert changed['propagation_segment_id'] != first['propagation_segment_id']
    assert changed['status'] == 'unavailable' and changed['projection_chain'] is None
    assert tracker.bridge is None
    next_row = tracker.push(image, sample(2, 104, source_sha256='b'*64))
    assert not next_row['source_identity_break_applied']
    assert next_row['status'] == 'unavailable'


@pytest.mark.parametrize('extra', [dict(source_pts=101), dict(source_time_base='1/50'), dict(pts=100.5),
                                 dict(time_base='-1/100'), dict(source_time_s=float('nan'))])
def test_source_pts_aliases_cannot_change_ui_identity_or_round_ticks(tmp_path, extra):
    image, _ = fixture(tmp_path)
    tracker = WholeRinkLocalizer(tmp_path/'model.json', tmp_path, config())
    row = sample()
    row.update(extra)
    with pytest.raises(ValueError, match='PTS'):
        tracker.push(image, row)


@pytest.mark.parametrize('extra,match', [(dict(source_time_s=99), 'PTS'),
    (dict(image_domain='undistorted_px'), 'raw_px_distortion_unchecked'),
    (dict(sample_size=[0, 270]), 'sample_size'),
    (dict(H_image_from_model=np.zeros((3, 3)).tolist()), 'nonsingular')])
def test_inconsistent_reference_time_or_domain_is_rejected_before_registration(tmp_path, extra, match):
    _, model = fixture(tmp_path)
    model['view_poses'][0].update(extra)
    dump_json(tmp_path/'model.json', model)
    with pytest.raises(ValueError, match=match):
        WholeRinkLocalizer(tmp_path/'model.json', tmp_path, config())


def test_radial_reference_keeps_native_parameter_domain_when_processing_and_source_sizes_differ(tmp_path):
    image, _ = fixture(tmp_path, radial=True)
    tracker = WholeRinkLocalizer(tmp_path/'model.json', tmp_path, WholeRinkConfig(continuity_mode='legacy'))
    working = cv2.resize(image, (960, 540), interpolation=cv2.INTER_AREA)
    row = tracker.push(working, sample(width=1920, height=1080))
    assert row['status'] == 'candidate' and row['H_model_from_image'] is None
    chain = row['projection_chain']
    assert chain['reference_sample_size'] == [480, 270]
    assert chain['reference_geometry_domain'] == 'undistorted_reference_sample_px'
    p, centre = np.array([120., 90.]), np.array([240., 135.])
    raw_reference = centre+(p-centre)*(1-.08*np.sum(((p-centre)/480)**2))
    result = _chain_point(p, chain)[1]
    current = result[:2]/result[2]
    np.testing.assert_allclose(current, raw_reference*2, atol=.001)
    np.testing.assert_allclose(np.asarray(chain['H_source_image_from_sample']) @ np.r_[current, 1.],
                               np.r_[raw_reference*4, 1.], atol=.002)


def test_run_rejects_source_content_different_from_its_pts_index_before_decode(tmp_path):
    fixture(tmp_path)
    source = tmp_path/'changed.mp4'
    source.write_bytes(b'An indexed path can change even if metadata is absent')
    dump_json(tmp_path/'configs/local.json', {'sources': {'synthetic': {'path': str(source)}}})
    dump_json(tmp_path/'outputs/workspace/synthetic/frame_index.json', {'video_id': 'a'*64, 'frames': []})
    with pytest.raises(ValueError, match='SHA-256 differs'):
        track_window(tmp_path, tmp_path/'model.json', 'synthetic', 1, 2, tmp_path/'run',
            selection_basis='source identity rejection fixture', config=config())
    assert not list((tmp_path/'run').glob('*.mp4'))


def test_source_shot_roles_and_epochs_are_passthrough_not_live_contact_inferences():
    shots = [dict(shot_id='auto_1', start_s=1, end_s=2, role='programme_candidate', review_status='unreviewed'),
             dict(shot_id='review_2', start_s=2, end_s=3, video_role='replay', background_epoch_id='epoch_2')]
    first = _source_shot_context(shots, 1.98)
    assert first['role'] == 'programme_candidate' and 'video_role' not in first
    assert first['source_shot_metadata']['review_status'] == 'unreviewed'
    assert 'epoch' not in first and 'background_epoch_id' not in first
    replay = _source_shot_context(shots, 2)
    assert replay['video_role'] == 'replay' and replay['background_epoch_id'] == 'epoch_2'
    assert replay['source_shot_id'] == 'review_2' and 'shot_id' not in replay
    replay['source_shot_metadata']['video_role'] = 'modified_test_copy'
    assert shots[1]['video_role'] == 'replay'
    assert _source_shot_context(shots, 3) == {}
