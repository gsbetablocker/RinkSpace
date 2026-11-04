"""Exact source-label export; no assets, fitting or geometric contact inference."""
import json
from pathlib import Path

import numpy as np
import pytest

from calibration.program_tracking.body_tracking.source_support_export import apply_source_support


def packet():
    return dict(pts=np.array([100, 356, 612], np.int64), time_base=np.array([1, 12800]),
        skate_contact_mode_mid=np.full((3, 2), 'unknown'),
        skate_contact_confidence_mid=np.zeros((3, 2)),
        source_action_phase_mid=np.full((3, 2), 'spin'),
        skate_support_gap_m_mid=np.array([-.2, .5, 0.]))


def support(**changes):
    return dict(dict(id='source_356', pts=356, feet=[1], mode='glide', confidence=.65,
        source='fixed source-image review', source_time_base=[1, 12800], shot_id='s05'), **changes)


def test_exact_source_mode_is_independent_of_phase_and_bad_predicted_height():
    p = packet()
    phase, geometry = p['source_action_phase_mid'].copy(), p['skate_support_gap_m_mid'].copy()
    assert apply_source_support(p, [support()], shot_id='s05') is p
    assert p['skate_contact_mode_mid'].tolist() == [['unknown', 'unknown'], ['unknown', 'glide'], ['unknown', 'unknown']]
    assert p['skate_contact_confidence_mid'][1, 1] == .65
    assert p['source_support_mode_mid'][1, 1] == 'glide'
    assert json.loads(p['source_support_evidence_ids_mid'][1, 1]) == ['source_356']
    np.testing.assert_array_equal(p['source_action_phase_mid'], phase)
    np.testing.assert_array_equal(p['skate_support_gap_m_mid'], geometry)
    assert not json.loads(str(p['source_support_export_json']))['geometry_used_for_labels']


@pytest.mark.parametrize('changes', [dict(feet=[0, 1]), dict(mode='unknown'), dict(source=''),
                                    dict(confidence=None), dict(pts=357)])
def test_ambiguous_unknown_unsourced_and_neighbour_labels_are_not_invented(changes):
    p = packet()
    p['skate_contact_mode_mid'][0, 0] = 'pick'
    p['skate_contact_confidence_mid'][0, 0] = .4
    before = p['skate_contact_mode_mid'].copy(), p['skate_contact_confidence_mid'].copy()
    apply_source_support(p, [support(**changes)])
    np.testing.assert_array_equal(p['skate_contact_mode_mid'], before[0])
    np.testing.assert_array_equal(p['skate_contact_confidence_mid'], before[1])
    assert np.all(p['source_support_mode_mid'] == 'unknown')


@pytest.mark.parametrize('changes, match', [(dict(pts=356.), 'integer'),
    (dict(source_time_base=[1, 90000]), 'time base'), (dict(shot_id='s02'), 'shot identity')])
def test_identity_mismatches_fail(changes, match):
    with pytest.raises(ValueError, match=match):
        apply_source_support(packet(), [support(**changes)], shot_id='s05')


def test_conflicting_source_modes_do_not_select_a_geometric_winner():
    p = packet()
    apply_source_support(p, [support(), support(id='other', mode='pick')])
    assert p['skate_contact_mode_mid'][1, 1] == 'unknown'
    assert all(r['status'] == 'conflicting_source_modes'
               for r in json.loads(str(p['source_support_export_json']))['rows'])


def test_current_clip_labels_survive_export_and_repeated_stage_export():
    source = Path(__file__).resolve().parents[1]/'events/ankara_2026/programmes/okada_mei_sp/action_evidence_1103.json'
    clips = json.loads(source.read_text(encoding='utf8'))['clips']
    for clip, count in [('jump_s02', 4), ('step_s01', 4), ('spin_s05', 3)]:
        evidence = clips[clip]
        pts = np.array(sorted(r['pts'] for r in evidence['supports']), np.int64)
        p = dict(pts=pts, time_base=np.array([1, 12800]))
        apply_source_support(p, evidence['supports'], shot_id=evidence['shot_id'])
        assert np.count_nonzero(p['skate_contact_mode_mid'] != 'unknown') == count
        old = p['skate_contact_mode_mid'].copy(), p['skate_contact_confidence_mid'].copy()
        apply_source_support(p, evidence['supports'], shot_id=evidence['shot_id'])
        np.testing.assert_array_equal(p['skate_contact_mode_mid'], old[0])
        np.testing.assert_array_equal(p['skate_contact_confidence_mid'], old[1])
