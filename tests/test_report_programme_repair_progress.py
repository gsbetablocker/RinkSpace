"""Small checks of saved-ledger matching and the segment mass decomposition."""
import importlib
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def progress(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    return importlib.import_module('report_programme_repair_progress')


def test_crossing_transitions_exclude_unmatched_times_and_sample_kinds(progress):
    pair = ('palm', 'shoe')
    common = dict(parts=list(pair), summary={'topology': {'faces': 30}})

    def scan(times, kinds, events):
        counts = np.zeros(len(times), dtype=int)
        counts[events] = 3
        return dict(**common, keys=np.array(times), kinds=np.array(kinds),
                    counts=counts, events={pair: np.array(events)})

    before = scan([100, 200, 300, 400, 500], [0, 1, 2, 0, 1], [1, 2, 3, 4])
    after = scan([100, 200, 300, 400, 600], [0, 1, 1, 0, 2], [0, 1, 2, 4])
    compared = progress.compare_scans(before, after)
    assert compared['common_samples'] == 3
    assert compared['same_time_different_kind_samples'] == 1
    assert compared['before_only_samples'] == compared['after_only_samples'] == 2
    assert compared['semantic_pair_sample_transitions'] == dict(new=1, disappeared=1, retained=1)
    assert compared['changed_event_sets']['palm / shoe'] == dict(
        new=[[100, 100, 1]], disappeared=[[400, 400, 1]],
    )
    assert progress.compare_scans({}, after)['available'] is False


def test_segment_export_separates_body_and_shoe_terms_without_model_evaluation(progress):
    t = np.arange(4, dtype=float)
    root = t[:, None]*np.array([.7, -.4, .9])
    local_offsets = t[:, None, None]**2*np.array([[0., 1., 0.], [2., 0., 0.], [0., 0., 3.]])
    centres = root[:, None]+local_offsets
    weights = np.array([.8, .1, .1])
    result = dict(com_segment_centres_world_m=centres, com_segment_body_joint_ids=np.array([0]),
                  com_segment_mass_fractions=weights)
    positions = dict(P=root, COM=np.einsum('s,nsc->nc', weights, centres))
    pts = np.arange(len(t))
    operator = progress.acceleration_operator(t)
    parts, origin = progress.offset_accelerations(result, None, pts, (1, 1), positions, operator)
    np.testing.assert_allclose(parts['body'], [[0., 1.6, 0.]]*2, atol=1e-12)
    np.testing.assert_allclose(parts['shoe'], [[.4, 0., .6]]*2, atol=1e-12)
    assert 'saved segment' in origin['source']
    absent, origin = progress.offset_accelerations({}, None, pts, (1, 1), positions, operator)
    assert absent == {} and origin['available'] is False


def test_opening_hand_waist_full_window_max_does_not_supply_phase_statistics(progress):
    pts = np.arange(283136, 299520+256, 256)
    quiet = pts[pts <= 293120]
    saved = dict(opening_source_standing=dict(
        source_pts=quiet, minimum_steel_height_m=np.zeros(len(quiet)),
        hand_waist_max_distance_m=.346,
        held_hand_waist_max_distance_m=.12, release_hand_waist_max_distance_m=.346,
    ))
    report = progress.opening_report({}, pts, {'P': np.zeros((len(pts), 3))}, saved, {})
    assert report['hand_waist_max_distance_m'] == .346
    phases = report['hand_waist_distance_by_phase']
    assert phases['full_window'] == dict(
        source_pts_range=[283136, 299520], max_distance_field='hand_waist_max_distance_m',
        includes_release=True,
    )
    assert phases['held']['start_pts_inclusive'] == 283136
    assert phases['held']['end_pts_inclusive'] == phases['release']['start_pts_exclusive'] == 295680
    assert phases['release']['end_pts_inclusive'] == 299520
    assert phases['held']['source_frames'] == 50
    assert phases['release']['source_frames'] == 15
    for name in ('held', 'release'):
        assert phases[name]['available'] is False
        assert 'per-PTS' in phases[name]['reason']
        assert 'max_distance_m' not in phases[name]


@pytest.fixture
def saved_hand_distances():
    pts = np.arange(283136, 299520+256, 256)
    quiet = pts[pts <= 293120]
    distances = np.tile([.01, .02, .03], (len(pts), 1))
    distances[pts == 295680, 1] = .12
    distances[pts == 295936, 2] = .16
    distances[pts == 299520, 0] = .31
    return pts, dict(
        source_pts=quiet, minimum_steel_height_m=np.zeros(len(quiet)),
        hand_waist_max_distance_m=.31,
        hand_waist_source_pts=pts.tolist(), hand_waist_patch_distances_m=distances.tolist(),
        held_hand_waist_max_distance_m=.12, release_hand_waist_max_distance_m=.31,
    )


def test_opening_hand_waist_phase_stats_use_all_three_patches_and_native_boundaries(progress, saved_hand_distances):
    pts, opening = saved_hand_distances
    report = progress.opening_report({}, pts, {}, {'opening_source_standing': opening}, {})
    phases = report['hand_waist_distance_by_phase']
    held, release = (phases[name] for name in ('held', 'release'))
    for phase, frame_count, maximum, peak in ((held, 50, .12, 295680), (release, 15, .31, 299520)):
        assert phase['available'] is True
        assert phase['source_frames'] == frame_count
        assert phase['patch_distances_m']['samples'] == 3 * frame_count
        assert phase['patch_distances_m']['p50'] == pytest.approx(.02)
        assert phase['patch_distances_m']['max'] == phase['max_distance_m'] == maximum
        assert phase['max_distance_pts'] == peak
    assert report['hand_waist_max_distance_m'] == .31


@pytest.mark.parametrize('fault,expected_reason', [
    ('shifted_pts', 'source PTS'),
    ('float_pts', 'source PTS'),
    ('missing_distances', 'hand_waist_patch_distances_m'),
    ('two_patches', 'three finite'),
    ('nan_distance', 'three finite'),
])
def test_opening_hand_waist_phase_stats_reject_incomplete_or_misaligned_exports(
    progress, saved_hand_distances, fault, expected_reason,
):
    pts, opening = saved_hand_distances
    if fault == 'shifted_pts':
        opening['hand_waist_source_pts'] = (pts + 256).tolist()
    elif fault == 'float_pts':
        opening['hand_waist_source_pts'] = pts.astype(float).tolist()
    elif fault == 'missing_distances':
        del opening['hand_waist_patch_distances_m']
    elif fault == 'two_patches':
        opening['hand_waist_patch_distances_m'] = [[.01, .02]] * len(pts)
    elif fault == 'nan_distance':
        opening['hand_waist_patch_distances_m'][0][0] = np.nan
    report = progress.opening_report({}, pts, {}, {'opening_source_standing': opening}, {})
    for name in ('held', 'release'):
        phase = report['hand_waist_distance_by_phase'][name]
        assert phase['available'] is False
        assert expected_reason in phase['reason']
        assert 'max_distance_m' not in phase
    assert report['hand_waist_max_distance_m'] == .31
    assert report['exact_minimum_eligible_steel_height_m']['samples'] == 40
