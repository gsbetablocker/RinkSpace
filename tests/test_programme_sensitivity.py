"""Small CPU receipts only: no model, fit, network, real programme or GUI."""
import json
from pathlib import Path

import numpy as np
import pytest

from calibration.program_tracking.body_tracking import programme_sensitivity as sensitivity


def _json(path, value):
    path.write_text(json.dumps(value), encoding='utf-8')


def _case(root, *, shift=0., pts=None):
    stage, prepared = root / 'stage', root / 'prepared'
    (stage / 'fitted').mkdir(parents=True)
    (stage / 'solve').mkdir()
    prepared.mkdir(parents=True)
    pts = np.arange(6, dtype=np.int64) * 256 if pts is None else np.asarray(pts, dtype=np.int64)
    n = len(pts)
    tb = np.array([1, 12800])
    shots = np.array(['s01', 's01', 's02', 's02', 's03', 's03'])[:n]
    body = np.zeros((n, 3)) + shift
    state = dict(pts=pts, source_pts=pts, time_base=tb, source_shot_ids=shots,
                 P_world_mid=body, G_world_mid=body.copy(),
                 Q_world_mid=np.broadcast_to(np.eye(3), (n, 3, 3)),
                 skate_origin_world_mid=np.zeros((n, 2, 3)),
                 skate_R_world_mid=np.broadcast_to(np.eye(3), (n, 2, 3, 3)),
                 local_R_mid=np.broadcast_to(np.eye(3), (n, 21, 3, 3)),
                 COCO_uv=np.zeros((n, 17, 2)),
                 skate_clearance_m_mid=np.zeros((n, 2)),
                 skate_edge_world_mid=np.zeros((n, 2, 2, 3, 3)),
                 foot_cavity_max_violation_m_mid=np.zeros((n, 2)),
                 metres_per_scene_unit=np.array(60.), body_length_ratio=np.array(1.))
    np.savez(stage / 'fitted/result.npz', **state)
    np.savez(prepared / 'input.npz', pts=pts, time_base=tb, shot_ids=shots,
             shared_betas=np.zeros(10), body_scale=np.array(1.))
    np.savez(prepared / 'observations.npz', pts=pts, use=np.ones((n, 17), bool),
             uv=np.zeros((n, 17, 2)), sigma_px=np.ones((n, 17)))
    np.savez(stage / 'objective_block_weights.npz', body_pixels=np.ones(n * 17 * 2),
             finite_network_pose_prior=np.array(1.e12))
    np.savez(stage / 'solve/candidate.npz',
             prepared_input_sha256=np.array(sensitivity.file_identity(prepared / 'input.npz')['sha256']),
             prepared_observations_sha256=np.array(sensitivity.file_identity(prepared / 'observations.npz')['sha256']))
    _json(stage / 'run_summary.json', dict(source_frames=n, accepted_steps=2, accuracy_accepted=False))
    costs = dict(body_pixels=0., full_source_skate_structure=3., finite_network_pose_prior=8.,
                 finite_parent_root_prior=4., finite_ice=2., source_motion=1.,
                 source_named_prior=9., unknown_future_block=7.)
    initial = dict(weighted_parts=costs, structure=dict(feasible=False, max_violation_m=.01))
    final = dict(initial, primary_guards=dict(
        source_sample=dict(rejected=False, before=1., after=1.5, original_radius_px=4.),
        torso=dict(rejected=True, before=10., after=13.)))
    _json(stage / 'initial_metrics.json', initial)
    _json(stage / 'final_metrics.json', final)
    _json(stage / 'primary_quality_baseline.json', dict(source={'source_sample': [1., 4.]}, torso={'pixel_p95': 10.}))
    return stage, prepared


def test_source_costs_never_include_priors_or_unknown_blocks():
    original = dict(body_pixels=12., full_source_skate_structure=3.,
                    finite_parent_root_prior=4., source_named_prior=9.,
                    source_motion=1., finite_ice=2., unknown_source_looking=7.,
                    native_source_support_contact=5., source_support_location_xy=6.)
    receipt = sensitivity.classified_costs(dict(weighted_parts=original))
    assert receipt['weighted_parts'] == original
    assert receipt['categories']['source_image'] == {'full_source_skate_structure': 3.}
    assert receipt['categories']['priors']['source_named_prior'] == 9.
    assert receipt['categories']['priors']['source_motion'] == 1.
    assert receipt['categories']['source_derived_scene'] == {'native_source_support_contact': 5., 'source_support_location_xy': 6.}
    assert receipt['categories']['finite_physics'] == {'finite_ice': 2.}
    assert receipt['categories']['unknown'] == {'unknown_source_looking': 7.}
    assert receipt['total_checksum_only'] == sum(original.values())
    assert not receipt['used_for_candidate_selection']


def test_identical_saved_states_have_zero_delta_and_preserve_receipts(tmp_path):
    stage, prepared = _case(tmp_path / 'nominal')
    # The initializer is reached through recorded ancestry, not a sibling-name guess.
    parent = tmp_path / 'start'
    (parent / 'solve').mkdir(parents=True)
    initialization = dict(scale_factor=1., depth_profile={'requested_peak_m': .1})
    _json(parent / 'initialization.json', initialization)
    _json(stage / 'resume.json', dict(parameter=str(parent / 'solve/candidate.npz')))
    output = tmp_path / 'report'
    sensitivity.generate_report(stage, prepared, [('zero', stage, prepared)], output, data_root=tmp_path)
    report = json.loads((output / 'report.json').read_text(encoding='utf-8'))
    difference = report['comparisons']['zero']['differences']
    assert difference['whole']['P_distance_m']['maximum'] == 0.
    assert difference['whole']['Q_degrees']['maximum'] == 0.
    assert difference['cut_count'] == 2 and len(difference['by_shot']) == 3
    assert difference['each_source_guard']['source_sample']['nominal']['status'] == 'pass'
    assert difference['each_source_guard']['torso']['comparison']['status'] == 'fail'
    assert report['nominal']['initialization_ancestry'][-1]['saved'] == initialization
    assert report['nominal']['costs']['final']['weighted_parts'] == report['nominal']['saved_final_metrics']['weighted_parts']
    assert report['nominal']['files']['parameter']['sha256']
    assert report['nominal']['coverage']['frames'] == 6
    assert report['nominal']['raw_ice_and_structure']['arrays']['minimum_runner_edge_height_m']['whole']['maximum'] == 0.
    assert not report['candidate_selection_performed']
    assert report['common_body_evidence_review']['nominal_body_cost_equivalence']['status'] == 'match'
    assert report['nominal']['nominal_body_evidence_review']['exact_weighted_body_cost'] == 0.
    assert report['comparisons']['zero']['nominal_body_evidence_review']['review_only'] is True
    assert report['comparisons']['zero']['nominal_body_evidence_review']['exact_weighted_body_cost'] == 0.
    with np.load(output / 'comparison_zero.npz', allow_pickle=False) as packet:
        np.testing.assert_array_equal(packet['pts'], np.arange(6) * 256)
        for name in ('P_delta_world_m', 'G_delta_world_m', 'Q_degrees', 'skate_R_degrees', 'leg_local_R_degrees'):
            assert np.count_nonzero(packet[name]) == 0
    with pytest.raises(FileExistsError, match='Preserve existing report'):
        sensitivity.generate_report(stage, prepared, [('zero', stage, prepared)], output, data_root=tmp_path)


def test_different_pts_rejected_before_output_creation(tmp_path):
    stage, prepared = _case(tmp_path / 'nominal')
    other, other_prepared = _case(tmp_path / 'other', pts=np.arange(6) * 256 + 1)
    output = tmp_path / 'report'
    with pytest.raises(ValueError, match='Exact same PTS'):
        sensitivity.generate_report(stage, prepared, [('bad', other, other_prepared)], output, data_root=tmp_path)
    assert not output.exists()


def test_input_binding_and_result_clock_must_match(tmp_path):
    stage, prepared = _case(tmp_path / 'case')
    with np.load(prepared / 'input.npz', allow_pickle=False) as packet:
        p = dict(packet)
    p['body_scale'] = np.array(1.1)
    np.savez(prepared / 'input.npz', **p)
    with pytest.raises(ValueError, match='not bound to supplied prepared input'):
        sensitivity.read_stage(stage, prepared)


def test_rotation_geodesic_and_output_boundary(tmp_path):
    angle = np.deg2rad(30.)
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0.], [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
    assert sensitivity._rotation_degrees(np.eye(3), rotation) == pytest.approx(30.)
    with pytest.raises(ValueError, match='Data root'):
        sensitivity.generate_report('absent', 'absent', [('x', Path('absent'), Path('absent'))], tmp_path / 'outside', data_root=tmp_path / 'data')


def _body_review_inputs():
    pts = np.array([256, 512], dtype=np.int64)
    uv, predicted = np.zeros((2, 17, 2)), np.zeros((2, 17, 2))
    use = np.zeros((2, 17), bool)
    use[0, 13] = use[1, 16] = True
    uv[0, 13], uv[1, 16] = [3., 4.], [1., 2.]
    predicted[0, 13], predicted[0, 14] = [300., 400.], [6., 8.]
    predicted[1, 16] = [5., 10.]
    sigma = np.ones((2, 17))
    sigma[0, 13], sigma[1, 16] = 2., 4.
    mapping = np.broadcast_to(np.arange(17), (2, 17)).copy()
    mapping[0, 13:15] = [14, 13]
    observations = dict(pts=pts, uv=uv, use=use, sigma_px=sigma, model_joint_ids=mapping)
    # A saved candidate mapping must never override the nominal ledger.
    packet = dict(pts=pts, time_base=np.array([1, 12800]), COCO_uv=predicted,
                  source_body_model_joint_ids=np.zeros((2, 17), dtype=np.int64))
    return packet, observations, np.array([2., 3., .5, .25])


def test_nominal_source_mapping_sigma_and_explicit_weights_use_exact_body_formula():
    packet, observations, weights = _body_review_inputs()
    score = sensitivity.body_evidence_rescore(packet, observations, weights)
    # d=(1.5,2), (1,2); raw labels remain lower limbs after model mapping.
    expected = .5 * .02 * (16.5 * 2 / (np.sqrt(1 + 6.25 / 4) + 1)
                            + 1.5 * 2 / (np.sqrt(1 + 5 / 4) + 1))
    assert score['exact_weighted_body_cost'] == pytest.approx(expected, abs=1e-15)
    assert score['groups']['lower']['exact_weighted_body_cost'] == pytest.approx(expected, abs=1e-15)
    assert score['groups']['lower']['observations'] == 2
    assert score['groups']['face']['observations'] == 0
    assert score['groups']['lower']['pixel_median'] == pytest.approx(np.median([5., np.sqrt(80.)]))
    assert score['candidate_saved_mapping_ignored'] is True
    assert score['nominal_reassigned_source_labels'] == 2
    assert score['sample_period_seconds'] == .02
    without_mapping = {k: v for k, v in observations.items() if k != 'model_joint_ids'}
    wrong = sensitivity.body_evidence_rescore(packet, without_mapping, weights)
    assert wrong['exact_weighted_body_cost'] > score['exact_weighted_body_cost']
    assert score['review_only'] and not score['original_candidate_fit_cost']


def test_common_body_weights_and_nominal_clock_cannot_be_guessed():
    packet, observations, weights = _body_review_inputs()
    with pytest.raises(ValueError, match='explicit positive vector'):
        sensitivity.body_evidence_rescore(packet, observations, weights[:-1])
    with pytest.raises(ValueError, match='explicit positive vector'):
        sensitivity.body_evidence_rescore(packet, observations, np.array(1.))
    bad = dict(packet, pts=np.array([256, 512, 769], dtype=np.int64))
    with pytest.raises(ValueError, match='equal-step'):
        sensitivity.body_evidence_rescore(bad, observations, weights)
    with pytest.raises(ValueError, match='Exact same PTS'):
        sensitivity.body_evidence_rescore(packet, dict(observations, pts=observations['pts'] + 1), weights)


def test_missing_saved_uv_and_nominal_cost_mismatch_are_explicit():
    packet, observations, weights = _body_review_inputs()
    missing = sensitivity.body_evidence_rescore({k: v for k, v in packet.items() if k != 'COCO_uv'}, observations, weights)
    assert missing['status'] == 'missing' and 'COCO_uv' in missing['reason']
    assert sensitivity.body_cost_equivalence(missing, 12.)['status'] == 'missing'
    score = sensitivity.body_evidence_rescore(packet, observations, weights)
    own = score['exact_weighted_body_cost']
    assert sensitivity.body_cost_equivalence(score, own + 5e-11)['status'] == 'match'
    mismatch = sensitivity.body_cost_equivalence(score, own + 1e-5)
    assert mismatch['status'] == 'mismatch' and not mismatch['equivalent']
    assert mismatch['difference_recomputed_minus_saved'] == pytest.approx(-1e-5)
    assert mismatch['relative_tolerance'] == 0.
