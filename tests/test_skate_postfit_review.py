"""Audit physical units and source pairing without running a real candidate."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from calibration.program_tracking.body_tracking.skate_postfit_review import compare_packets, review_postfit, S01_PTS


def packet():
    n = len(S01_PTS)
    return dict(pts=S01_PTS.copy(), source_pts=S01_PTS.copy(), time_base=np.array([1, 12800]),
                P_world_mid=np.zeros((n, 3)), Q_world_mid=np.tile(np.eye(3), (n, 1, 1)),
                local_R_mid=np.tile(np.eye(3), (n, 21, 1, 1)),
                uv=np.array([[np.nan, -0.]]), use=np.array([[False]]))


def test_common_translation_and_rotation_are_visible_without_artificial_motion():
    old, new = packet(), packet()
    new['P_world_mid'][:] = [.3, -.4, .1]
    new['Q_world_mid'][:] = Rotation.from_euler('z', 30, degrees=True).as_matrix()
    report = compare_packets(old, new, policy_record={'baseline': {'pose_cost': 0.}})
    assert report['source_exposures'] == 1143
    assert report['all_compared_source_fields_byte_equal']
    assert report['same_exposure_difference']['pelvis_XY_distance_m']['max'] == pytest.approx(.5)
    assert report['same_exposure_difference']['root_Q_difference_deg']['max'] == pytest.approx(30.)
    assert report['single_step']['new']['pelvis_step_m']['max'] == 0.
    assert report['single_step']['new']['root_step_deg']['max'] == 0.
    assert report['opening_quiet']['new']['pelvis_distance_from_first_m']['max'] == 0.
    assert not report['policy_pose_guard']['relative_guard_comparable']


def test_old_boundary_step_and_joint_change_keep_exact_source_interval():
    old, new = packet(), packet()
    i = int(np.flatnonzero(S01_PTS == round(35.2*12800))[0])
    new['P_world_mid'][i:, 2] = .012
    new['local_R_mid'][i:, 19] = Rotation.from_euler('x', 8, degrees=True).as_matrix()
    report = compare_packets(old, new)
    assert report['single_step']['new']['pelvis_step_m']['max'] == pytest.approx(.012)
    assert report['single_step']['new']['pelvis_speed_m_per_s']['max'] == pytest.approx(.6)
    assert report['single_step']['new']['pelvis_step_m']['worst_pts'] == int(S01_PTS[i])
    assert report['single_step']['local_increment_change_deg']['20']['max'] == pytest.approx(8.)
    boundary = report['former_window_boundaries']['35.2']
    assert int(S01_PTS[i]) in boundary['new']['interval_end_pts']
    assert boundary['new']['velocity_after_minus_before_m_per_s'][2] == pytest.approx(-.6)
    assert report['opening_quiet']['new']['pelvis_distance_from_first_m']['max'] == 0.


def test_quiet_body_drift_is_reported_even_with_zero_pose_cost_baseline():
    old, new = packet(), packet()
    quiet = S01_PTS <= round(22.9*12800)
    new['P_world_mid'][quiet, 0] = np.arange(quiet.sum())*.001
    new['local_R_mid'][quiet, 0] = Rotation.from_euler('y', np.linspace(0., 4., quiet.sum())[:, None], degrees=True).as_matrix()
    report = compare_packets(old, new, policy_record={'baseline': {'pose_cost': 0.}})
    phase = report['opening_quiet']['new']
    assert phase['exposures'] == 40
    assert phase['pelvis_net_XYZ_m'][0] == pytest.approx(.039)
    assert phase['local_R_from_first_deg']['1']['max'] == pytest.approx(4.)
    assert phase['quiet_body_maximum_change_deg']['max'] == pytest.approx(4.)
    assert not report['user_accepted']


def test_source_bytes_are_audited_and_different_pts_are_not_positionally_paired():
    old, new = packet(), packet()
    new['uv'][0, 1] = 0.
    report = compare_packets(old, new)
    assert not report['source_fields']['uv']['byte_equal']
    assert not report['all_compared_source_fields_byte_equal']
    new['pts'][10] += 1
    with pytest.raises(ValueError, match='Different source PTS'):
        compare_packets(old, new)


def test_incomplete_source_denominator_and_nonfinite_body_cannot_be_hidden():
    old, new = packet(), packet()
    for data in (old, new):
        for key in ('pts', 'source_pts', 'P_world_mid', 'Q_world_mid', 'local_R_mid'):
            data[key] = data[key][:-1]
    with pytest.raises(ValueError, match='Incomplete'):
        compare_packets(old, new)
    old, new = packet(), packet()
    new['local_R_mid'][2, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match='nonfinite'):
        compare_packets(old, new)


def test_file_entry_preserves_old_snapshot_and_audits_the_external_ledger(tmp_path):
    root, out = tmp_path/'root', tmp_path/'root/new'
    old_path = root/'skate_contact_1103/fitted/result.npz'
    new_path = out/'fitted/result.npz'
    source_path = root/'boot_integration_1102/observations.npz'
    for path in (old_path, new_path, source_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(old_path, **packet())
    np.savez_compressed(new_path, **packet())
    ledger = dict(pts=S01_PTS, uv=np.array([[np.nan, -0.]]), use=np.array([[True]]))
    np.savez_compressed(source_path, **ledger)
    np.savez_compressed(old_path.parent.parent/'observations.npz', **ledger)
    changed = dict(ledger, use=np.array([[False]]))
    np.savez_compressed(out/'observations.npz', **changed)
    old_bytes = old_path.read_bytes()
    report = review_postfit(root, out)
    assert old_path.read_bytes() == old_bytes
    assert report['old_result']['bytes'] == len(old_bytes)
    assert report['external_observation_ledger']['comparisons']['old']['all_source_fields_byte_equal']
    assert not report['external_observation_ledger']['comparisons']['new']['fields_byte_equal']['use']
    assert (out/'postfit_audit.json').is_file()
